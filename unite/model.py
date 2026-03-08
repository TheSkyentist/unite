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
from numpyro import distributions as dist

from unite._utils import C_KMS, _wavelength_conversion_factor
from unite.continuum.config import ContinuumConfiguration
from unite.instrument.generic import GenericSpectrum
from unite.instrument.spectrum import Spectra
from unite.line.config import ConfigMatrices, LineConfiguration
from unite.line.profiles import integrate_lines
from unite.prior import Fixed, Parameter, Prior, topological_sort

# ------------------------------------------------------------------
# ModelArgs — data bundle for the numpyro model function
# ------------------------------------------------------------------


@dataclass
class ModelArgs:
    """Bundle of arguments passed to :func:`unite_model`.

    Created by :meth:`ModelBuilder.build`; not intended for direct
    construction by users.
    """

    #: Precomputed parameter matrices and line metadata.
    matrices: ConfigMatrices
    #: Individual spectra.
    spectra: list[GenericSpectrum]
    #: Systemic redshift.
    redshift: float
    #: Continuum configuration, or ``None`` if not used.
    cont_config: ContinuumConfiguration | None
    #: Resolved ``{param_name: ContinuumParam}`` mappings per region, from :attr:`ContinuumConfiguration.resolved_params`.
    cont_resolved_params: list[dict[str, Parameter]] | None
    #: All parameters with their priors (line, calibration, continuum).
    all_priors: dict[str, Prior]
    #: Topological sampling order for all parameters.
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
    #: Pre-converted continuum forms (static wavelength config in canonical unit).
    cont_forms: list | None
    # --- Flux normalization ---
    norm_factors: list[float]
    #: Per-spectrum line flux scale (in each spectrum's flux_unit * canonical_wl_unit).
    line_flux_scales: list[float]
    #: Per-spectrum continuum scale (in each spectrum's flux_unit).
    continuum_scales: list[float]
    #: Wavelength unit of canonical frame (first spectrum's disperser unit).
    canonical_unit: object
    #: Per-spectrum flux density units.
    flux_units: list
    #: The Quantity line_scale and continuum_scale from Spectra (for results output).
    line_scale_quantity: object
    continuum_scale_quantity: object
    #: Human-readable column labels for each line, parallel to ``matrices.wavelengths``.
    #: Derived from user-supplied line names and rest-frame wavelengths.
    line_labels: list[str]
    #: Human-readable column labels for each continuum region, parallel to ``cont_config``.
    #: Derived from form type and wavelength bounds.
    continuum_labels: list[str]


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

    # --- 3. Per-spectrum likelihood ---
    for i, spectrum in enumerate(args.spectra):
        disp = spectrum.disperser
        wl_scale = args.spec_to_canonical[i]
        inv_wl_scale = 1.0 / wl_scale
        norm = args.norm_factors[i]
        lfs = args.line_flux_scales[i]
        cs = args.continuum_scales[i]

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

        # Sum over lines weighted by flux, scaled by per-spectrum
        # line_flux_scale and divided by norm to bring to O(1) space.
        line_model = ((flux_per_line * lfs / norm)[:, None] * pixints).sum(
            axis=0
        )  # (n_pixels,)

        # Continuum (evaluated in canonical-unit wavelengths, then normalized).
        continuum = jnp.zeros(spectrum.npix)
        if args.cont_config is not None:
            for k in range(len(args.cont_config)):
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
                form = args.cont_forms[k]
                region_cont = form.evaluate(wavelength, obs_center, cont_params)
                continuum = continuum + jnp.where(in_region, region_cont, 0.0)

        # Likelihood (normalized flux space).
        # Continuum is scaled by per-spectrum continuum_scale so that
        # sampled 'scale' parameters are O(1).
        model = flux_scale * (line_model + continuum * cs / norm)
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

        self._cont_config = continuum_config

        # --- Canonical wavelength unit: use Spectra's canonical_unit ---
        self._canonical_unit = spectra.canonical_unit

        # Build precomputed matrices from line entries.
        self._matrices = line_config.build_matrices()

        # Convert line wavelengths to canonical unit.
        # Each line may have a different Quantity unit, so convert per-line.
        if len(line_config) > 0:
            canon_wls = jnp.array(
                [
                    float(e.wavelength.to(self._canonical_unit).value)
                    for e in line_config._entries
                ]
            )
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
        self._line_config = line_config
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
        # Trim spectra to union of continuum regions (observed frame).
        # Pixels outside all regions have model = 0 and would corrupt the
        # likelihood if observed flux is nonzero.  Trimming also reduces
        # array sizes passed to JAX.
        z = self._spectra.redshift
        if self._cont_config is not None:
            trimmed_spectra: list[GenericSpectrum] = []
            for s in self._spectra:
                mask = jnp.zeros(s.npix, dtype=bool)
                for region in self._cont_config:
                    conv = _wavelength_conversion_factor(region._unit, s.unit)
                    obs_low = region.low * conv * (1.0 + z)
                    obs_high = region.high * conv * (1.0 + z)
                    mask = mask | s.pixel_mask(obs_low, obs_high)
                trimmed_spectra.append(s._sliced(mask))
        else:
            trimmed_spectra = list(self._spectra)

        # Per-spectrum wavelength conversion factors.
        spec_to_canonical = [
            _wavelength_conversion_factor(s.unit, self._canonical_unit)
            for s in trimmed_spectra
        ]

        # Per-spectrum flux normalization.
        norm_factors = [_compute_norm_factor(s) for s in trimmed_spectra]

        # Line flux scale (Quantity from Spectra.compute_scales).
        line_scale_qty = self._spectra.line_scale
        # Continuum scale (Quantity from Spectra.compute_scales; fallback).
        cont_scale_qty = self._spectra.continuum_scale

        # Convert Quantity scales to per-spectrum float values.
        # line_flux_scale needs units of [flux_density * canonical_wl_unit]
        # for each spectrum.
        # continuum_scale needs units of [flux_density] for each spectrum.
        canonical_unit = self._canonical_unit
        line_flux_scales: list[float] = []
        continuum_scales: list[float] = []
        for s in trimmed_spectra:
            target_line_unit = s.flux_unit * canonical_unit
            lfs = float(line_scale_qty.to(target_line_unit).value)
            line_flux_scales.append(lfs)

            if cont_scale_qty is not None:
                cs = float(cont_scale_qty.to(s.flux_unit).value)
            else:
                cs = 1.0
            continuum_scales.append(cs)

        flux_units = [s.flux_unit for s in trimmed_spectra]

        # Pre-convert continuum region bounds to canonical unit.
        if self._cont_config is not None:
            cont_low = []
            cont_high = []
            cont_center = []
            cont_nw_conv = []
            cont_forms = []
            for region in self._cont_config:
                conv = _wavelength_conversion_factor(region._unit, self._canonical_unit)
                cont_low.append(region.low * conv)
                cont_high.append(region.high * conv)
                cont_center.append(region.center * conv)
                cont_nw_conv.append(conv)
                cont_forms.append(
                    region.form._prepare(self._canonical_unit, region._unit)
                )
        else:
            cont_low = None
            cont_high = None
            cont_center = None
            cont_nw_conv = None
            cont_forms = None

        line_labels = _make_line_labels(self._line_config)
        continuum_labels = (
            _make_continuum_labels(self._cont_config)
            if self._cont_config is not None
            else []
        )

        args = ModelArgs(
            matrices=self._matrices,
            spectra=trimmed_spectra,
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
            cont_forms=cont_forms,
            norm_factors=norm_factors,
            line_flux_scales=line_flux_scales,
            continuum_scales=continuum_scales,
            canonical_unit=canonical_unit,
            flux_units=flux_units,
            line_scale_quantity=line_scale_qty,
            continuum_scale_quantity=cont_scale_qty,
            line_labels=line_labels,
            continuum_labels=continuum_labels,
        )
        return unite_model, args


# ------------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------------


def _make_line_labels(line_config: LineConfiguration) -> list[str]:
    """Build a human-readable label for every line entry.

    For a name group with only one entry, the label is just the line name.

    For a name group with multiple entries, the label is built by appending
    the token names of every axis that varies *within the group*:

    * **Wavelength** — if rest-frame wavelengths differ, the rounded value is
      appended (e.g. ``'[NII]_6585'``).
    * **Redshift** — if redshift token names differ, the token name is
      appended (e.g. ``'Ha_z_nlr'``).
    * **FWHM(s)** — if any fwhm token names differ, every fwhm token name for
      that entry is appended in ``profile.param_names()`` order
      (e.g. ``'Ha_fwhm_narrow'``, ``'Ha_fwhm_gauss_narrow'``).
    * **Flux** — if flux token names differ, the token name is appended
      (e.g. ``'Ha_flux_broad'``).

    All varying axes are included simultaneously, so the label is always
    unambiguous and self-documenting regardless of how the tokens were
    named by the user.

    Examples
    --------
    Unique name::

        add_line('Ha', 6563 Å, ...)  →  'Ha'

    Multiplet — same kinematics, different wavelengths::

        add_line('[NII]', [6585, 6550] Å, z=z, fwhm_gauss=fwhm, flux=flux)
        →  '[NII]_6585', '[NII]_6550'

    Multiple components — same wavelength, different FWHM tokens::

        fwhm1 = FWHM('narrow');  fwhm2 = FWHM('broad')
        add_line('Ha', 6563 Å, fwhm_gauss=fwhm1, ...)
        add_line('Ha', 6563 Å, fwhm_gauss=fwhm2, ...)
        →  'Ha_narrow', 'Ha_broad'

    Three components sharing a flux token (the motivating bug case)::

        fwhm1 = FWHM('f1');  fwhm2 = FWHM('f2');  fwhm3 = FWHM('f3')
        flux  = Flux('flux')
        add_line('a', 6563 Å, fwhm_gauss=fwhm1, flux=flux, ...)
        add_line('a', 6563 Å, fwhm_gauss=fwhm2, flux=flux, ...)
        add_line('a', 6563 Å, fwhm_gauss=fwhm3, flux=flux, ...)
        →  'a_f1', 'a_f2', 'a_f3'

    Multiplet + multiple components::

        fwhm_n = FWHM('narrow');  fwhm_b = FWHM('broad')
        add_line('[NII]', [6585, 6550] Å, fwhm_gauss=fwhm_n, flux=flux1, ...)
        add_line('[NII]', [6585, 6550] Å, fwhm_gauss=fwhm_b, flux=flux2, ...)
        →  '[NII]_6585_narrow', '[NII]_6550_narrow',
           '[NII]_6585_broad',  '[NII]_6550_broad'
    """
    from collections import defaultdict

    entries = list(line_config._entries)
    labels: list[str] = [''] * len(entries)

    # Group entry indices by user-supplied line name.
    name_to_indices: dict[str, list[int]] = defaultdict(list)
    for j, entry in enumerate(entries):
        name_to_indices[entry.name].append(j)

    for name, indices in name_to_indices.items():
        if len(indices) == 1:
            labels[indices[0]] = name
            continue

        group = [entries[j] for j in indices]

        # Determine which axes vary within this name group.
        wl_strs = [f'{float(e.wavelength.value):.0f}' for e in group]
        z_names = [e.redshift.name for e in group]
        flux_names_list = [e.flux.name for e in group]
        # fwhm names: list-of-lists, one inner list per entry.
        fwhm_names_per_entry = [
            [e.fwhms[pn].name for pn in e.profile.param_names()] for e in group
        ]

        vary_wl = len(set(wl_strs)) > 1
        vary_z = len(set(z_names)) > 1
        vary_fwhm = len({tuple(fns) for fns in fwhm_names_per_entry}) > 1
        vary_flux = len(set(flux_names_list)) > 1

        for idx, j in enumerate(indices):
            parts = [name]
            if vary_wl:
                parts.append(wl_strs[idx])
            if vary_z:
                parts.append(z_names[idx])
            if vary_fwhm:
                parts.extend(fwhm_names_per_entry[idx])
            if vary_flux:
                parts.append(flux_names_list[idx])
            labels[j] = '_'.join(parts)

    return labels


def _make_continuum_labels(cont_config: ContinuumConfiguration) -> list[str]:
    """Build a human-readable label for every continuum region.

    Format: ``'{form_type}_{low:.4g}_{high:.4g}'`` where the wavelength
    values are in the region's native unit and ``form_type`` is the
    lower-cased class name of the functional form (e.g. ``'linear'``,
    ``'powerlaw'``, ``'polynomial'``).

    Examples: ``'linear_6400_6700'``, ``'powerlaw_0.95_2.5'``.
    """
    labels: list[str] = []
    for region in cont_config:
        form_type = type(region.form).__name__.lower()
        low_str = f'{region.low:.4g}'
        high_str = f'{region.high:.4g}'
        labels.append(f'{form_type}_{low_str}_{high_str}')
    return labels


def _compute_norm_factor(s: GenericSpectrum) -> float:
    """Robust scale factor to bring a spectrum's flux to ~O(1).

    Uses the median of the absolute non-zero flux values.

    Parameters
    ----------
    s : GenericSpectrum

    Returns
    -------
    float
        Positive normalization factor.
    """
    absflux = jnp.abs(s.flux)
    positive = absflux[absflux > 0]
    if positive.size == 0:
        fallback = float(jnp.max(s.error))
        return fallback if fallback > 0 else 1.0
    return float(jnp.median(positive))
