"""Model builder and numpyro model function for spectral line fitting.

The :class:`ModelBuilder` assembles a :class:`~unite.line.config.LineConfiguration`,
an optional :class:`~unite.continuum.config.ContinuumConfiguration`, and a
:class:`~unite.spectrum.spectrum.Spectra` collection into a numpyro model
function that can be passed to any numpyro inference algorithm (NUTS, SVI, etc.).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpyro
from astropy import constants
from numpyro import deterministic as determ, distributions as dist

from unite.continuum.config import ContinuumConfiguration
from unite.line.config import ConfigMatrices, LineConfiguration
from unite.line.profiles import integrate_lines
from unite.prior import Fixed, Prior, topological_sort
from unite.spectrum.spectrum import Spectra, Spectrum

_C_KMS: float = constants.c.to('km/s').value
"""Speed of light in km/s."""


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

        # Integrate all lines simultaneously and divide by pixel width to get average flux density in pixel.
        pixints = integrate_lines(
            low, high, centers, lsf_fwhm, p0, p1, p2, cm.profile_codes
        ) / (high - low)

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
        model = flux_scale * determ(f'{spectrum.name}_model', (line_model + continuum))
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
        self._matrices = line_config.build_matrices()

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
