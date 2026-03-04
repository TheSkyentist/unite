"""Model builder and numpyro model function for spectral line fitting.

The :class:`ModelBuilder` assembles a :class:`~unite.line.config.LineConfiguration`,
an optional :class:`~unite.continuum.config.ContinuumConfiguration`, and a
:class:`~unite.spectrum.spectrum.Spectra` collection into a numpyro model
function that can be passed to any numpyro inference algorithm (NUTS, SVI, etc.).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpyro
from numpyro import deterministic as determ, distributions as dist

from unite._utils import C_KMS, _wavelength_conversion_factor
from unite.continuum.config import ContinuumConfiguration
from unite.line.config import ConfigMatrices, LineConfiguration
from unite.line.profiles import integrate_lines
from unite.prior import Fixed, Parameter, Prior, topological_sort
from unite.spectrum.spectrum import Spectra, Spectrum

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
    cont_resolved_params : list of dict or None
        Resolved ``{param_name: ContinuumParam}`` mappings, one per region,
        as returned by :attr:`ContinuumConfiguration.resolved_params`.
    all_priors : dict of str to Prior
        All parameters with their priors (line, calibration, continuum).
    dependency_order : list of str
        Topological sampling order for all parameters.
    """

    matrices: ConfigMatrices
    spectra: list[Spectrum]
    redshift: float
    cont_config: ContinuumConfiguration | None
    cont_resolved_params: list[dict[str, Parameter]] | None
    all_priors: dict[str, Prior]
    dependency_order: list[str]
    name_to_token: dict[str, object]
    # --- Wavelength unit conversion ---
    spec_to_canonical: list[float]
    # --- Pre-converted continuum bounds (rest-frame, canonical unit) ---
    cont_low: list[float] | None
    cont_high: list[float] | None
    cont_center: list[float] | None
    # --- Unit conversion factors: region._unit → canonical unit, per region ---
    cont_nw_conv: list[float] | None
    # --- Flux normalization ---
    norm_factors: list[float]
    line_flux_scale: float


# ------------------------------------------------------------------
# Numpyro model function
# ------------------------------------------------------------------


def unite_model(args: ModelArgs) -> None:
    """Numpyro model function for multi-spectrum emission-line fitting.

    All lines are integrated simultaneously via :func:`jax.vmap` with
    ``lax.switch`` dispatching to the correct profile kernel per line.
    Parameter broadcasting from unique tokens to per-line arrays is done
    with precomputed indicator matrices.

    Wavelength unit conversion is handled via pre-computed scalar factors
    stored in ``args.spec_to_canonical`` (one per spectrum).  Flux is
    normalized per spectrum so that the likelihood operates on O(1) values.

    Parameters
    ----------
    args : ModelArgs
        Pre-built data bundle from :meth:`ModelBuilder.build`.
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

    # Observed-frame centers (in canonical wavelength unit).
    centers = cm.wavelengths * (1.0 + z_sys + z_per_line)  # (n_lines,)

    # Slot 0: primary velocity FWHM → canonical wavelength units.
    if cm.p0_names:
        p0_vec = jnp.stack([context[n] for n in cm.p0_names])
        p0_kms = p0_vec @ cm.p0_matrix  # (n_lines,)
    else:
        p0_kms = jnp.zeros(n_lines)
    p0 = centers * p0_kms / C_KMS  # canonical wavelength units

    # Slot 1: velocity FWHMs (p1v) + dimensionless params (p1d), summed.
    if cm.p1v_names:
        p1v_vec = jnp.stack([context[n] for n in cm.p1v_names])
        p1v_kms = p1v_vec @ cm.p1v_matrix  # (n_lines,)
    else:
        p1v_kms = jnp.zeros(n_lines)
    p1v = centers * p1v_kms / C_KMS

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

    # Expose normalization metadata as deterministics for back-conversion.
    determ('line_flux_scale', jnp.asarray(args.line_flux_scale))
    determ('norm_factors', jnp.asarray(args.norm_factors))

    # --- 3. Per-spectrum likelihood ---
    for i, spectrum in enumerate(args.spectra):
        disp = spectrum.disperser
        wl_scale = args.spec_to_canonical[i]
        inv_wl_scale = 1.0 / wl_scale
        norm = args.norm_factors[i]

        # Calibration values (fall back to identity when no token is attached).
        r_scale = context[disp.r_scale.name] if disp.r_scale is not None else 1.0
        flux_scale = (
            context[disp.flux_scale.name] if disp.flux_scale is not None else 1.0
        )
        pix_offset = (
            context[disp.pix_offset.name] if disp.pix_offset is not None else 0.0
        )

        # Pixel edges in canonical wavelength unit.
        low = spectrum.low * wl_scale
        high = spectrum.high * wl_scale
        if disp.pix_offset is not None:
            mid_disp = (spectrum.low + spectrum.high) / 2.0  # disperser unit
            shift = pix_offset * disp.dlam_dpix(mid_disp) * wl_scale
            low = low + shift
            high = high + shift

        wavelength = (low + high) / 2.0

        # LSF FWHM at each line centre (n_lines,).
        # R() expects disperser-unit wavelengths, result is in canonical unit.
        lsf_fwhm = centers / (disp.R(centers * inv_wl_scale) * r_scale)

        # Integrate all lines simultaneously and divide by pixel width
        # to get average flux density per pixel.
        pixints = integrate_lines(
            low, high, centers, lsf_fwhm, p0, p1, p2, cm.profile_codes
        ) / (high - low)

        # Sum over lines weighted by flux, scaled by line_flux_scale and
        # divided by norm to bring to the normalized O(1) space.
        line_model = (
            (flux_per_line * args.line_flux_scale / norm)[:, None] * pixints
        ).sum(axis=0)  # (n_pixels,)

        # Continuum (evaluated in canonical-unit wavelengths, then normalized).
        continuum = jnp.zeros(spectrum.npix)
        if args.cont_config is not None:
            for k, region in enumerate(args.cont_config):
                obs_low = args.cont_low[k] * (1.0 + z_sys)
                obs_high = args.cont_high[k] * (1.0 + z_sys)
                obs_center = args.cont_center[k] * (1.0 + z_sys)
                in_region = (wavelength >= obs_low) & (wavelength <= obs_high)
                cont_params = {
                    pn: (
                        context[tok.name] * args.cont_nw_conv[k] * (1.0 + z_sys)
                        if pn == 'normalization_wavelength'
                        else context[tok.name]
                    )
                    for pn, tok in args.cont_resolved_params[k].items()
                }
                region_cont = region.form.evaluate(wavelength, obs_center, cont_params)
                continuum = continuum + jnp.where(in_region, region_cont, 0.0)

        # Likelihood (normalized flux space).
        model = flux_scale * determ(
            f'{spectrum.name}_model', line_model + continuum / norm
        )
        determ(f'{spectrum.name}_norm', jnp.asarray(norm))
        obs_name = f'obs_{spectrum.name}' if spectrum.name else f'obs_{i}'
        numpyro.sample(
            obs_name,
            dist.Normal(model, spectrum.scaled_error / norm),
            obs=spectrum.flux / norm,
        )


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
        self._spectra = spectra

        # --- Auto-prepare if needed ---
        if not spectra.is_prepared:
            warnings.warn(
                'Spectra not prepared; calling spectra.prepare() with defaults. '
                'Call spectra.prepare(line_config, continuum_config) explicitly '
                'for full control.',
                UserWarning,
                stacklevel=2,
            )
            line_config, continuum_config = spectra.prepare(
                line_config, continuum_config
            )
        else:
            # Use the prepared configs.
            line_config = spectra.prepared_line_config
            continuum_config = spectra.prepared_cont_config

        # --- Auto-compute scales if needed ---
        if spectra.line_scale is None:
            warnings.warn(
                'Line scale not set; calling spectra.compute_scales() with '
                'defaults. Call spectra.compute_scales(line_config, '
                'continuum_config) explicitly for full control.',
                UserWarning,
                stacklevel=2,
            )
            spectra.compute_scales(line_config, continuum_config)

        self._line_config = line_config
        self._cont_config = continuum_config

        # --- Canonical wavelength unit: use the first spectrum's disperser unit ---
        self._canonical_unit = spectra[0].unit

        # Build precomputed matrices from line entries.
        self._matrices = line_config.build_matrices()

        # Convert line wavelengths to canonical unit.
        # Each line may have a different Quantity unit, so convert per-line.
        if len(line_config) > 0:
            canon_wls = jnp.array([
                float(e.wavelength.to(self._canonical_unit).value)
                for e in line_config._entries
            ])
            self._matrices.wavelengths = canon_wls

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

        # Continuum parameters: collect unique ContinuumParam tokens by identity.
        # Shared tokens (same object in multiple regions) produce one numpyro site.
        if continuum_config is not None:
            seen_cont_ids: set[int] = set()
            for resolved in continuum_config.resolved_params:
                for tok in resolved.values():
                    if id(tok) not in seen_cont_ids:
                        seen_cont_ids.add(id(tok))
                        all_priors[tok.name] = tok.prior
                        param_to_name[tok] = tok.name

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
        # Per-spectrum wavelength conversion factors.
        spec_to_canonical = [
            _wavelength_conversion_factor(s.unit, self._canonical_unit)
            for s in self._spectra
        ]

        # Per-spectrum flux normalization.
        norm_factors = [_compute_norm_factor(s) for s in self._spectra]

        # Line flux scale (from Spectra.compute_scales).
        line_flux_scale = self._spectra.line_scale

        # Pre-convert continuum region bounds to canonical unit.
        if self._cont_config is not None:
            cont_low = []
            cont_high = []
            cont_center = []
            cont_nw_conv = []
            for region in self._cont_config:
                conv = _wavelength_conversion_factor(region._unit, self._canonical_unit)
                cont_low.append(region.low * conv)
                cont_high.append(region.high * conv)
                cont_center.append(region.center * conv)
                cont_nw_conv.append(conv)
        else:
            cont_low = None
            cont_high = None
            cont_center = None
            cont_nw_conv = None

        args = ModelArgs(
            matrices=self._matrices,
            spectra=list(self._spectra),
            redshift=self._spectra.redshift,
            cont_config=self._cont_config,
            cont_resolved_params=(
                self._cont_config.resolved_params
                if self._cont_config is not None
                else None
            ),
            all_priors=self._all_priors,
            dependency_order=self._dep_order,
            name_to_token=self._name_to_token,
            spec_to_canonical=spec_to_canonical,
            cont_low=cont_low,
            cont_high=cont_high,
            cont_center=cont_center,
            cont_nw_conv=cont_nw_conv,
            norm_factors=norm_factors,
            line_flux_scale=line_flux_scale,
        )
        return unite_model, args


# ------------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------------


def _compute_norm_factor(spectrum: Spectrum) -> float:
    """Robust scale factor to bring a spectrum's flux to ~O(1).

    Uses the median of the absolute non-zero flux values.

    Parameters
    ----------
    spectrum : Spectrum

    Returns
    -------
    float
        Positive normalization factor.
    """
    absflux = jnp.abs(spectrum.flux)
    positive = absflux[absflux > 0]
    if positive.size == 0:
        fallback = float(jnp.max(spectrum.error))
        return fallback if fallback > 0 else 1.0
    return float(jnp.median(positive))


def _compute_line_flux_scale(
    spectra: list[Spectrum],
    matrices: ConfigMatrices,
    spec_to_canonical: list[float],
    norm_factors: list[float],
    z_sys: float,
) -> float:
    """Estimate a characteristic integrated line flux from the data.

    The estimate is ``peak_flux_density * typical_linewidth`` where
    the linewidth is derived from the disperser resolution at the
    median wavelength of each spectrum.

    Parameters
    ----------
    spectra : list of Spectrum
    matrices : ConfigMatrices
    spec_to_canonical : list of float
    norm_factors : list of float
    z_sys : float

    Returns
    -------
    float
        Positive scale factor for line fluxes, or 1.0 as fallback.
    """
    max_scale = 0.0
    for i, spectrum in enumerate(spectra):
        s2c = spec_to_canonical[i]
        norm = norm_factors[i]
        # Peak normalized flux density.
        peak_fd = float(jnp.max(jnp.abs(spectrum.flux))) / norm
        # Typical line width in canonical wavelength units:
        # use the median pixel centre and disperser R to get LSF FWHM.
        mid_wl = float(jnp.median(spectrum.wavelength))
        lsf_fwhm = mid_wl / spectrum.disperser.R(mid_wl) * s2c
        # Characteristic flux = peak_fd * ~3 * lsf_fwhm (a few resolution elements).
        flux_est = peak_fd * 3.0 * lsf_fwhm
        max_scale = max(max_scale, flux_est)
    return max_scale if max_scale > 0 else 1.0
