"""Model builder and numpyro model function for spectral line fitting.

The :class:`ModelBuilder` assembles a :class:`~unite.line.config.LineConfiguration`,
an optional :class:`~unite.continuum.config.ContinuumConfiguration`, and a
:class:`~unite.spectrum.spectrum.Spectra` collection into a numpyro model
function that can be passed to any numpyro inference algorithm (NUTS, SVI, etc.).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from astropy import constants

from unite.continuum.config import ContinuumConfiguration
from unite.line.config import LineConfiguration, _LineEntry

# Import all profile integration functions for dispatch
# Each function has signature: (low, high, center, lsf_fwhm, *profile_params) -> Array
# where lsf_fwhm is the instrumental line spread function FWHM in wavelength units
from unite.line.functions import (
    integrate_gaussHermite,
    integrate_gaussian,
    integrate_gaussianLaplace,
    integrate_split_normal,
    integrate_voigt,
)
from unite.prior import Fixed, Prior, topological_sort
from unite.spectrum.spectrum import Spectra, Spectrum

_C_KMS: float = constants.c.to('km/s').value
"""Speed of light in km/s."""


# ------------------------------------------------------------------
# Integration dispatch (vmapped over lines)
# ------------------------------------------------------------------


def _integrate_single_line(low, high, center, lsf_fwhm, p0, p1, p2, code):
    """Integrate one line profile over pixel bins, dispatched by *code*.

    All FWHM parameters are in wavelength units.  Shape parameters (h3, h4)
    are dimensionless.  ``lax.switch`` selects the profile at compile time per
    line, so every branch must have identical output shape.

    Parameters
    ----------
    low, high : jnp.ndarray, shape (n_pixels,)
        Pixel bin edges.
    center, lsf_fwhm, p0, p1, p2 : float
        Per-line scalars: observed center, LSF FWHM, and three profile
        parameter slots.  Slots unused by a given profile are zero.
    code : int
        Profile code (0=Gaussian, 1=Cauchy, 2=PseudoVoigt, 3=Laplace,
        4=SEMG, 5=GaussHermite, 6=SplitNormal).

    Returns
    -------
    jnp.ndarray, shape (n_pixels,)
        Integrated profile fraction per pixel bin.
    """
    # Profile dispatch functions - each corresponds to a profile code
    # Signature: (low, high, center, lsf_fwhm, p0, p1, p2) -> Array
    # where p0, p1, p2 are profile-specific parameters in wavelength units

    def _gaussian(lo, hi, c, lsf, p0, p1, p2):
        # p0 = fwhm_gauss (intrinsic), combined with LSF in quadrature.
        return integrate_gaussian(lo, hi, c, lsf, p0)

    def _cauchy(lo, hi, c, lsf, p0, p1, p2):
        # p0 = fwhm_lorentzian; implemented as PseudoVoigt with zero Gaussian width.
        return integrate_voigt(lo, hi, c, lsf, 0.0, p0)

    def _pseudovoigt(lo, hi, c, lsf, p0, p1, p2):
        # p0 = fwhm_gauss (intrinsic), p1 = fwhm_lorentz.
        return integrate_voigt(lo, hi, c, lsf, p0, p1)

    def _laplace(lo, hi, c, lsf, p0, p1, p2):
        # p0 = fwhm_exp; pure Laplace profile convolved with Gaussian LSF.
        return integrate_gaussianLaplace(lo, hi, c, lsf, 0.0, p0)

    def _semg(lo, hi, c, lsf, p0, p1, p2):
        # p0 = fwhm_gauss (intrinsic), p1 = fwhm_exp.
        return integrate_gaussianLaplace(lo, hi, c, lsf, p0, p1)

    def _gauss_hermite(lo, hi, c, lsf, p0, p1, p2):
        # p0 = fwhm_gauss (intrinsic), p1 = h3, p2 = h4.
        # integrate_gaussHermite handles LSF + fwhm_g convolution internally.
        return integrate_gaussHermite(lo, hi, c, lsf, p0, p1, p2)

    def _split_normal(lo, hi, c, lsf, p0, p1, p2):
        # p0 = fwhm_blue, p1 = fwhm_red.
        return integrate_split_normal(lo, hi, c, lsf, p0, p1)

    return jax.lax.switch(
        code,
        [
            _gaussian,
            _cauchy,
            _pseudovoigt,
            _laplace,
            _semg,
            _gauss_hermite,
            _split_normal,
        ],
        low,
        high,
        center,
        lsf_fwhm,
        p0,
        p1,
        p2,
    )


# vmap over lines: per-line scalars map to axis 0; pixel arrays are shared (None).
_integrate_lines = jax.vmap(
    _integrate_single_line, in_axes=(None, None, 0, 0, 0, 0, 0, 0)
)
"""Vectorised integration over all lines simultaneously.

Input shapes: ``low/high (n_pixels,)``, all others ``(n_lines,)``.
Output shape: ``(n_lines, n_pixels)``.
"""


# ------------------------------------------------------------------
# ConfigMatrices — precomputed parameter-to-line mappings
# ------------------------------------------------------------------


@dataclass
class ConfigMatrices:
    """Precomputed indicator matrices mapping unique parameters to per-line arrays.

    Each matrix has shape ``(n_unique_params, n_lines)``.  A ``1`` at
    position ``[i, j]`` means unique parameter ``i`` contributes to line
    ``j``.  Matrix-vector products turn a stacked vector of sampled values
    into a per-line array, replacing a Python loop at JIT trace time.

    Profile parameters are split into three *slots* matching the positional
    arguments of :func:`_integrate_single_line`:

    * **Slot 0** (``p0``): primary velocity FWHM (always ``fwhm_*``, converted
      to wavelength before vmap).
    * **Slot 1** (``p1``): secondary parameter — either a velocity FWHM
      (``fwhm_lorentz`` / ``fwhm_exp``, stored in ``p1v_*``) or a dimensionless
      shape parameter (``h3``, stored in ``p1d_*``).  The two sub-matrices are
      summed after appropriate unit conversion:
      ``p1 = p1v_kms * centers / C + p1d``.
    * **Slot 2** (``p2``): tertiary dimensionless parameter (``h4``), pass-through.

    Attributes
    ----------
    wavelengths : jnp.ndarray, shape (n_lines,)
        Rest-frame line wavelengths (raw floats; units must match spectra).
    strengths : jnp.ndarray, shape (n_lines,)
        Multiplet relative flux strengths.
    profile_codes : jnp.ndarray, shape (n_lines,)
        Integer profile codes for ``lax.switch`` dispatch.
    flux_names : list of str
        Numpyro site names for unique flux parameters (rows of ``flux_matrix``).
    flux_matrix : jnp.ndarray, shape (n_flux, n_lines)
    z_names : list of str
        Numpyro site names for unique redshift parameters.
    z_matrix : jnp.ndarray, shape (n_z, n_lines)
    p0_names : list of str
    p0_matrix : jnp.ndarray, shape (n_p0, n_lines)
    p1v_names : list of str
        Velocity FWHM parameters in slot 1.
    p1v_matrix : jnp.ndarray, shape (n_p1v, n_lines)
    p1d_names : list of str
        Dimensionless parameters in slot 1.
    p1d_matrix : jnp.ndarray, shape (n_p1d, n_lines)
    p2_names : list of str
    p2_matrix : jnp.ndarray, shape (n_p2, n_lines)
    priors : dict of str to Prior
        Priors for all unique line parameter tokens.
    """

    wavelengths: jnp.ndarray
    strengths: jnp.ndarray
    profile_codes: jnp.ndarray

    flux_names: list[str]
    flux_matrix: jnp.ndarray

    z_names: list[str]
    z_matrix: jnp.ndarray

    p0_names: list[str]
    p0_matrix: jnp.ndarray

    p1v_names: list[str]
    p1v_matrix: jnp.ndarray

    p1d_names: list[str]
    p1d_matrix: jnp.ndarray

    p2_names: list[str]
    p2_matrix: jnp.ndarray

    priors: dict[str, Prior]


# ------------------------------------------------------------------
# Matrix construction helpers
# ------------------------------------------------------------------


def _token_matrix(
    entries: list[_LineEntry], get_tok, n_lines: int
) -> tuple[list[str], jnp.ndarray]:
    """Build a ``(n_unique, n_lines)`` indicator matrix for a token accessor.

    Parameters
    ----------
    entries : list of _LineEntry
    get_tok : callable
        Extracts the token from a ``_LineEntry`` (e.g. ``lambda e: e.flux``).
    n_lines : int

    Returns
    -------
    names : list of str
        Numpyro site names in row order.
    matrix : jnp.ndarray, shape (n_unique, n_lines)
    """
    unique: list = []
    seen: dict[int, int] = {}
    for entry in entries:
        tok = get_tok(entry)
        if id(tok) not in seen:
            seen[id(tok)] = len(unique)
            unique.append(tok)
    mat = np.zeros((len(unique), n_lines), dtype=float)
    for j, entry in enumerate(entries):
        mat[seen[id(get_tok(entry))], j] = 1.0
    return [t.name for t in unique], jnp.array(mat)


def _slot_matrix(
    tok_pairs: list[tuple[int, object]], n_lines: int
) -> tuple[list[str], jnp.ndarray]:
    """Build an indicator matrix from ``(line_idx, token)`` pairs for one slot.

    Parameters
    ----------
    tok_pairs : list of (int, token)
        Each pair is ``(line_index, parameter_token)`` for lines that use this slot.
    n_lines : int

    Returns
    -------
    names : list of str
    matrix : jnp.ndarray, shape (n_unique, n_lines)
        Empty ``(0, n_lines)`` when *tok_pairs* is empty.
    """
    if not tok_pairs:
        return [], jnp.zeros((0, n_lines))
    unique: list = []
    seen: dict[int, int] = {}
    for _, tok in tok_pairs:
        if id(tok) not in seen:
            seen[id(tok)] = len(unique)
            unique.append(tok)
    mat = np.zeros((len(unique), n_lines), dtype=float)
    for j, tok in tok_pairs:
        mat[seen[id(tok)], j] = 1.0
    return [t.name for t in unique], jnp.array(mat)


def _build_matrices(entries: list[_LineEntry]) -> ConfigMatrices:
    """Construct :class:`ConfigMatrices` from a list of line entries.

    Parameters
    ----------
    entries : list of _LineEntry

    Returns
    -------
    ConfigMatrices
    """
    n = len(entries)

    wavelengths = jnp.array([float(e.wavelength.value) for e in entries])
    strengths = jnp.array([float(e.strength) for e in entries])
    profile_codes = jnp.array([e.profile.code for e in entries], dtype=int)

    flux_names, flux_matrix = _token_matrix(entries, lambda e: e.flux, n)
    z_names, z_matrix = _token_matrix(entries, lambda e: e.redshift, n)

    # Assign profile params to slots based on each profile's param_names() order.
    # Slot 0: first param (always a velocity FWHM, name starts with 'fwhm').
    # Slot 1: second param — velocity FWHM (→ p1v) or dimensionless (→ p1d).
    # Slot 2: third param (always dimensionless, only h4 currently).
    p0_pairs: list[tuple[int, object]] = []
    p1v_pairs: list[tuple[int, object]] = []
    p1d_pairs: list[tuple[int, object]] = []
    p2_pairs: list[tuple[int, object]] = []

    for j, entry in enumerate(entries):
        pnames = entry.profile.param_names()
        if len(pnames) >= 1:
            p0_pairs.append((j, entry.fwhms[pnames[0]]))
        if len(pnames) >= 2:
            pn1, tok1 = pnames[1], entry.fwhms[pnames[1]]
            (p1v_pairs if pn1.startswith('fwhm') else p1d_pairs).append((j, tok1))
        if len(pnames) >= 3:
            p2_pairs.append((j, entry.fwhms[pnames[2]]))

    p0_names, p0_matrix = _slot_matrix(p0_pairs, n)
    p1v_names, p1v_matrix = _slot_matrix(p1v_pairs, n)
    p1d_names, p1d_matrix = _slot_matrix(p1d_pairs, n)
    p2_names, p2_matrix = _slot_matrix(p2_pairs, n)

    # Collect priors from all unique tokens.
    priors: dict[str, Prior] = {}
    seen_ids: set[int] = set()
    for entry in entries:
        for tok in (entry.flux, entry.redshift, *entry.fwhms.values()):
            if id(tok) not in seen_ids:
                seen_ids.add(id(tok))
                priors[tok.name] = tok.prior

    return ConfigMatrices(
        wavelengths=wavelengths,
        strengths=strengths,
        profile_codes=profile_codes,
        flux_names=flux_names,
        flux_matrix=flux_matrix,
        z_names=z_names,
        z_matrix=z_matrix,
        p0_names=p0_names,
        p0_matrix=p0_matrix,
        p1v_names=p1v_names,
        p1v_matrix=p1v_matrix,
        p1d_names=p1d_names,
        p1d_matrix=p1d_matrix,
        p2_names=p2_names,
        p2_matrix=p2_matrix,
        priors=priors,
    )


# ------------------------------------------------------------------
# ModelArgs — data bundle for the numpyro model function
# ------------------------------------------------------------------


@dataclass
class ModelArgs:
    """Bundle of arguments passed to :func:`unite_model`.

    Created by :meth:`ModelBuilder.build`; not intended for direct
    construction by users.

    Attributes
    ----------
    matrices : ConfigMatrices
        Precomputed parameter matrices and line metadata.
    spectra : list of Spectrum
        Individual spectra.
    redshift : float
        Systemic redshift.
    cont_config : ContinuumConfiguration or None
        Continuum configuration.
    all_priors : dict of str to Prior
        All parameters with their priors (line, calibration, continuum).
    dependency_order : list of str
        Topological sampling order for all parameters.
    """

    matrices: ConfigMatrices
    spectra: list[Spectrum]
    redshift: float
    cont_config: ContinuumConfiguration | None
    all_priors: dict[str, Prior]
    dependency_order: list[str]
    name_to_token: dict[str, object]


# ------------------------------------------------------------------
# Numpyro model function
# ------------------------------------------------------------------


def unite_model(args: ModelArgs) -> None:
    """Numpyro model function for multi-spectrum emission-line fitting.

    All lines are integrated simultaneously via :func:`jax.vmap` with
    ``lax.switch`` dispatching to the correct profile kernel per line.
    Parameter broadcasting from unique tokens to per-line arrays is done
    with precomputed indicator matrices.

    Parameters
    ----------
    args : ModelArgs
        Pre-built data bundle from :meth:`ModelBuilder.build`.

    Notes
    -----
    All wavelengths (line centers and spectrum pixel edges) must be in
    consistent units.  No unit conversion is performed at runtime.
    """
    cm = args.matrices
    z_sys = args.redshift
    n_lines = cm.wavelengths.shape[0]

    # --- 1. Sample all parameters in dependency order ---
    # Two parallel dicts are maintained:
    #   context    — str → value, used for all downstream name-based lookups.
    #   obj_ctx    — token_object → value, passed to prior.to_dist() so that
    #                ParameterRef.resolve() can look up dependency values by
    #                token identity (the API ParameterRef expects).
    context: dict[str, jnp.ndarray] = {}
    obj_ctx: dict[object, jnp.ndarray] = {}
    for pname in args.dependency_order:
        prior = args.all_priors[pname]
        if isinstance(prior, Fixed):
            val = jnp.asarray(prior.value)
        else:
            val = numpyro.sample(pname, prior.to_dist(obj_ctx))
        context[pname] = val
        tok = args.name_to_token.get(pname)
        if tok is not None:
            obj_ctx[tok] = val

    # --- 2. Per-line parameter arrays via matrix products ---
    # Flux: include multiplet strengths.
    flux_vec = jnp.stack([context[n] for n in cm.flux_names])
    flux_per_line = flux_vec @ cm.flux_matrix * cm.strengths  # (n_lines,)

    # Redshift (delta from systemic).
    z_vec = jnp.stack([context[n] for n in cm.z_names])
    z_per_line = z_vec @ cm.z_matrix  # (n_lines,)

    # Observed-frame centers.
    centers = cm.wavelengths * (1.0 + z_sys + z_per_line)  # (n_lines,)

    # Slot 0: primary velocity FWHM → wavelength units.
    if cm.p0_names:
        p0_vec = jnp.stack([context[n] for n in cm.p0_names])
        p0_kms = p0_vec @ cm.p0_matrix  # (n_lines,)
    else:
        p0_kms = jnp.zeros(n_lines)
    p0 = centers * p0_kms / _C_KMS  # wavelength units

    # Slot 1: velocity FWHMs (p1v) + dimensionless params (p1d), summed.
    if cm.p1v_names:
        p1v_vec = jnp.stack([context[n] for n in cm.p1v_names])
        p1v_kms = p1v_vec @ cm.p1v_matrix  # (n_lines,)
    else:
        p1v_kms = jnp.zeros(n_lines)
    p1v = centers * p1v_kms / _C_KMS

    if cm.p1d_names:
        p1d_vec = jnp.stack([context[n] for n in cm.p1d_names])
        p1d = p1d_vec @ cm.p1d_matrix  # (n_lines,), dimensionless
    else:
        p1d = jnp.zeros(n_lines)
    p1 = p1v + p1d  # for any line, only one sub-matrix is nonzero

    # Slot 2: dimensionless params only.
    if cm.p2_names:
        p2_vec = jnp.stack([context[n] for n in cm.p2_names])
        p2 = p2_vec @ cm.p2_matrix  # (n_lines,)
    else:
        p2 = jnp.zeros(n_lines)

    # --- 3. Per-spectrum likelihood ---
    for i, spectrum in enumerate(args.spectra):
        disp = spectrum.disperser

        # Calibration values (fall back to identity when no token is attached).
        r_scale = context[disp.r_scale.name] if disp.r_scale is not None else 1.0
        flux_scale = (
            context[disp.flux_scale.name] if disp.flux_scale is not None else 1.0
        )
        pix_offset = (
            context[disp.pix_offset.name] if disp.pix_offset is not None else 0.0
        )

        # Apply pixel offset: shift edges by offset * dlam_dpix.
        low = spectrum.low
        high = spectrum.high
        if disp.pix_offset is not None:
            shift = pix_offset * disp.dlam_dpix((low + high) / 2.0)
            low = low + shift
            high = high + shift

        wavelength = (low + high) / 2.0

        # LSF FWHM at each line centre (n_lines,), in wavelength units.
        lsf_fwhm = centers / (disp.R(centers) * r_scale)

        # Integrate all lines simultaneously.  Shape: (n_lines, n_pixels).
        pixints = _integrate_lines(
            low, high, centers, lsf_fwhm, p0, p1, p2, cm.profile_codes
        )

        # Sum over lines weighted by flux.
        line_model = (flux_per_line[:, None] * pixints).sum(axis=0)  # (n_pixels,)

        # Continuum.
        continuum = jnp.zeros(spectrum.npix)
        if args.cont_config is not None:
            for k, region in enumerate(args.cont_config):
                obs_low = region.low * (1.0 + z_sys)
                obs_high = region.high * (1.0 + z_sys)
                obs_center = region.center * (1.0 + z_sys)
                in_region = (wavelength >= obs_low) & (wavelength <= obs_high)
                cont_params = {
                    pn: context[f'cont_{pn}_{k}'] for pn in region.form.param_names()
                }
                region_cont = region.form.evaluate(wavelength, obs_center, cont_params)
                continuum = continuum + jnp.where(in_region, region_cont, 0.0)

        # Likelihood.
        model = flux_scale * (line_model + continuum)
        obs_name = f'obs_{spectrum.name}' if spectrum.name else f'obs_{i}'
        numpyro.sample(obs_name, dist.Normal(model, spectrum.error), obs=spectrum.flux)


# ------------------------------------------------------------------
# ModelBuilder
# ------------------------------------------------------------------


class ModelBuilder:
    """Assemble configuration objects into a numpyro model.

    Collects all unique parameter tokens (line, calibration, continuum),
    builds precomputed indicator matrices, performs a topological sort for
    dependency resolution, and packages everything into a
    ``(model_fn, model_args)`` pair.

    Parameters
    ----------
    line_config : LineConfiguration
        Emission/absorption line configuration.
    continuum_config : ContinuumConfiguration or None
        Continuum configuration.  ``None`` for a lines-only model.
    spectra : Spectra
        Spectrum collection with systemic redshift.

    Examples
    --------
    >>> model_fn, args = ModelBuilder(line_config, cont, spectra).build()
    >>> kernel = numpyro.infer.NUTS(model_fn)
    >>> mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=1000)
    >>> mcmc.run(jax.random.PRNGKey(0), args)
    """

    def __init__(
        self,
        line_config: LineConfiguration,
        continuum_config: ContinuumConfiguration | None,
        spectra: Spectra,
    ) -> None:
        self._line_config = line_config
        self._cont_config = continuum_config
        self._spectra = spectra

        # Build precomputed matrices from line entries.
        self._matrices = _build_matrices(list(line_config._entries))

        # --- Collect all unique parameter tokens for prior / topo-sort ---
        all_priors: dict[str, Prior] = dict(self._matrices.priors)
        param_to_name: dict[object, str] = {
            # We can reconstruct token→name from the matrices' name lists and
            # the original entries since tokens carry their .name attribute.
            tok: tok.name
            for entry in line_config._entries
            for tok in (entry.flux, entry.redshift, *entry.fwhms.values())
        }

        # Calibration tokens from each unique disperser.
        seen_dispersers: set[int] = set()
        seen_tok_ids: set[int] = set(id(t) for t in param_to_name)
        for spectrum in spectra:
            disp = spectrum.disperser
            if id(disp) not in seen_dispersers:
                seen_dispersers.add(id(disp))
                for tok in (disp.r_scale, disp.flux_scale, disp.pix_offset):
                    if tok is not None and id(tok) not in seen_tok_ids:
                        seen_tok_ids.add(id(tok))
                        all_priors[tok.name] = tok.prior
                        param_to_name[tok] = tok.name

        # Continuum parameters (no token objects; keyed by site name directly).
        if continuum_config is not None:
            for k, region in enumerate(continuum_config):
                default_priors = region.form.default_priors()
                for pn in region.form.param_names():
                    all_priors[f'cont_{pn}_{k}'] = region.priors.get(
                        pn, default_priors[pn]
                    )

        self._all_priors = all_priors
        self._dep_order = (
            topological_sort(all_priors, param_to_name) if all_priors else []
        )
        # Reverse mapping: site name → token object, for obj_ctx in unite_model.
        # Continuum params have no token objects and are intentionally absent.
        self._name_to_token: dict[str, object] = {
            name: tok for tok, name in param_to_name.items()
        }

    @property
    def matrices(self) -> ConfigMatrices:
        """Precomputed matrices (after coverage filtering)."""
        return self._matrices

    def build(self) -> tuple[Callable, ModelArgs]:
        """Build the numpyro model function and its arguments.

        Returns
        -------
        model_fn : callable
            The numpyro model function (signature: ``model_fn(args)``).
        model_args : ModelArgs
            Pre-built data bundle to pass to the model function.
        """
        args = ModelArgs(
            matrices=self._matrices,
            spectra=list(self._spectra),
            redshift=self._spectra.redshift,
            cont_config=self._cont_config,
            all_priors=self._all_priors,
            dependency_order=self._dep_order,
            name_to_token=self._name_to_token,
        )
        return unite_model, args
