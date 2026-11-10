"""Model evaluator: decompose posterior predictions into per-line and per-region contributions.

Given posterior samples and :class:`~unite.model.ModelArgs`, this module
reconstructs the full model prediction for each spectrum, broken down into
individual line and continuum-region contributions in **original flux units**.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as u

from unite import _overlap
from unite._compose import compose_from_profiles, compose_leave_one_out
from unite._lsf import _FWHM_TO_SIGMA, _lsf_convolve
from unite.continuum.compute import (
    _group_sum,
    eval_continuum,
    eval_continuum_regions,
    integrate_continuum_regions,
    interval_lsf_fwhm,
    region_lsf_fwhm,
)
from unite.line.compute import (
    _build_cover_params,
    _build_line_params,
    _peak_to_area_tau,
    evaluate_lines,
    integrate_lines,
)
from unite.model import ModelArgs
from unite.prior import Fixed


@dataclass
class SpectrumPrediction:
    """Decomposed model prediction for a single spectrum.

    All arrays are in original (un-normalized) flux units.

    For **emission lines**, each entry in :attr:`lines` is the intrinsic
    (un-attenuated) flux profile: ``flux * profile``.  Summing all line
    contributions and continuum regions always reconstructs :attr:`total`
    regardless of zorder configuration.

    For **absorption lines**, each entry in :attr:`lines` is the flux
    *removed* by that absorber (negative): ``total - total_without_j``.

    :attr:`lines` and :attr:`continuum_regions` describe the first-order
    spectrum only.  When the disperser has higher-order overlap, the light of
    each order *m* (lines, absorbers and continuum together, a line's image
    included) is in :attr:`overlap`, so that
    ``sum(lines) + sum(continuum_regions) + sum(overlap.values()) == total``.
    All entries include ``FluxScale`` (and the overlap its ``OrderScale``).
    """

    #: Pixel-center wavelengths in the disperser's unit. Shape ``(n_pixels,)``.
    wavelength: np.ndarray
    #: Total model flux (lines + continuum). Shape ``(n_samples, n_pixels)``.
    total: np.ndarray
    #: Per-line contributions keyed by informative line labels (e.g. ``'Ha'``, ``'[NII]_6585'``).
    #: For emission lines: intrinsic (un-attenuated) flux profile (positive).
    #: For absorption lines: flux removed by the absorber (negative).
    #: Shape ``(n_samples, n_pixels)`` each.
    lines: dict[str, np.ndarray]
    #: Per-continuum-region contributions keyed by informative region labels
    #: (e.g. ``'linear_6400_6700'``, ``'powerlaw_0.95_2.5'``).
    #: Shape ``(n_samples, n_pixels)`` each.
    continuum_regions: dict[str, np.ndarray]
    #: LSF-convolved optical depth profiles ``tau_j * phi_j(λ)`` (before any
    #: covering fraction is applied) for absorption
    #: lines, evaluated at pixel midpoints.  Dimensionless and non-negative.
    #: Keyed by line label; empty dict for emission-only models.
    #: Shape ``(n_samples, n_pixels)`` each.
    tau_profiles: dict[str, np.ndarray]
    #: Higher-order overlap contribution per diffraction order *m*,
    #: ``FluxScale * OrderScale_m * r_m(λ) * F(λ / m)``.  Empty for spectra
    #: without overlap.  Shape ``(n_samples, n_pixels)`` each.
    overlap: dict[int, np.ndarray] = field(default_factory=dict)


def _evaluate_in_chunks(fn, context, n_samples: int, batch_size: int):
    """Apply a vmapped ``fn`` to ``context`` in fixed-size chunks of draws.

    Every chunk is padded to exactly ``batch_size`` rows (by repeating the last
    valid draw) so the jitted function compiles once.  Each chunk is moved to
    the host as soon as it is computed, and the padding is stripped before the
    chunks are concatenated along the sample axis.
    """
    outs = []
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        n_valid = stop - start
        chunk = {k: v[start:stop] for k, v in context.items()}
        pad = batch_size - n_valid
        if pad:
            chunk = {
                k: jnp.concatenate([v, jnp.repeat(v[-1:], pad, axis=0)])
                for k, v in chunk.items()
            }
        out = jax.device_get(fn(chunk))
        outs.append(jax.tree.map(lambda a, n=n_valid: np.asarray(a)[:n], out))
    return jax.tree.map(lambda *a: np.concatenate(a, axis=0), *outs)


def _build_context(
    samples: dict[str, np.ndarray], args: ModelArgs
) -> tuple[dict[str, jnp.ndarray], int]:
    """Build the parameter dict with a uniform ``(n_samples,)`` leading axis.

    Returns the dict and ``n_samples``.
    """
    context: dict[str, jnp.ndarray] = {}
    obj_ctx: dict[object, jnp.ndarray] = {}  # token -> value, for Fixed expr resolution
    n_samples = None
    for pname in args.dependency_order:
        prior = args.all_priors[pname]
        if isinstance(prior, Fixed):
            arr = jnp.asarray(prior.resolved_value(obj_ctx))
        else:
            arr = jnp.asarray(samples[pname])
            if n_samples is None and arr.ndim >= 1:
                n_samples = arr.shape[0]
        context[pname] = arr
        tok = args.name_to_token.get(pname)
        if tok is not None:
            obj_ctx[tok] = arr

    if n_samples is None:
        n_samples = 1

    # Broadcast Fixed (scalar) params to (n_samples,) so every leaf has the
    # same leading axis and vmap can map uniformly over the dict pytree.
    context = {
        k: (jnp.broadcast_to(v, (n_samples,)) if v.ndim == 0 else v)
        for k, v in context.items()
    }

    return context, n_samples


def evaluate_model(
    samples: dict[str, np.ndarray], args: ModelArgs, *, batch_size: int | None = None
) -> list[SpectrumPrediction]:
    """Evaluate the model for each posterior sample and decompose contributions.

    Uses :func:`jax.vmap` to evaluate all samples in a single vectorised XLA
    kernel launch rather than a Python loop, giving a large speed-up when the
    number of posterior samples is large.

    Parameters
    ----------
    samples : dict of str to ndarray
        Posterior samples as returned by ``mcmc.get_samples()`` or
        ``Predictive``.  Each value has shape ``(n_samples,)`` or
        ``(n_samples, ...)``.
    args : ModelArgs
        Pre-built data bundle from :meth:`ModelBuilder.build`.
    batch_size : int or None
        If ``None`` (default), all draws are evaluated in a single vmapped
        call.  If an integer, draws are evaluated in chunks of that size and
        the results are concatenated on the host, bounding peak device memory
        (useful in convolution mode, where per-draw intermediates are large).
        The output is identical to the unbatched result.  The last chunk is
        padded to ``batch_size`` so the evaluator compiles only once.

    Returns
    -------
    list of SpectrumPrediction
        One prediction per spectrum in ``args.spectra``.
    """
    if batch_size is not None and batch_size < 1:
        raise ValueError(f'batch_size must be a positive integer, got {batch_size}.')

    cm = args.matrices
    z_sys = args.redshift
    n_lines = cm.wavelengths.shape[0]
    has_tau = bool(cm.tau_names)

    # Resolve per-config dispatch (specialized to used profiles, or module-level fallback).
    _pcodes = (
        args._profile_codes_local
        if args._profile_codes_local is not None
        else cm.profile_codes
    )
    _int_fn = args._integrate_fn if args._integrate_fn is not None else integrate_lines
    _eval_fn = args._evaluate_fn if args._evaluate_fn is not None else evaluate_lines

    context, n_samples = _build_context(samples, args)

    # --- Per-spectrum evaluation (vectorised over samples) ---
    # Lazily build and JIT-compile per-spectrum vmapped evaluators on first call.
    # This eliminates Python-level JAX retracing on every subsequent evaluate_model
    # call with the same ModelArgs (e.g. make_spectra_tables + make_parameter_table).
    if args._evaluators is None:
        args._evaluators = [None] * len(args.spectra)

    results: list[SpectrumPrediction] = []

    for i, spectrum in enumerate(args.spectra):
        disp = spectrum.disperser
        wl_scale = args.spec_to_canonical[i]
        inv_wl_scale = 1.0 / wl_scale
        wl_out = np.asarray(spectrum.wavelength)

        # Static arrays: do not depend on sample values.
        edges_base = jnp.asarray(spectrum.edges * wl_scale)
        edges_disp_base = jnp.asarray(spectrum.edges)
        keep_mask = jnp.asarray(spectrum.keep_mask)
        dlam_edges = disp.dlam_dpix(edges_disp_base) * wl_scale  # (E,)
        # Static continuum-region membership of each kept pixel (same mask
        # as the model, from the unshifted grid).
        full_region_mask = args.region_masks[i]  # (n_regions, E - 1)
        pix_region_mask = full_region_mask[:, spectrum.keep_mask]

        line_scale = float(args.line_flux_scales[i])
        cont_scale = float(args.continuum_scales[i])
        windows = args.overlap_windows[i] if args.overlap_windows else []

        # Keyword-only defaults bind the per-iteration values at definition
        # time, avoiding late-binding closure issues (ruff B023) while
        # allowing jax.vmap to vmap only over the positional `params` arg.
        def _single(
            params,
            *,
            _edges=edges_base,
            _keep=keep_mask,
            _dlam_edges=dlam_edges,
            _disp=disp,
            _inv_wl_scale=inv_wl_scale,
            _line_scale=line_scale,
            _cont_scale=cont_scale,
            _ifn=_int_fn,
            _efn=_eval_fn,
            _pc=_pcodes,
            _prm=pix_region_mask,
            _full_rm=full_region_mask,
            _wins=windows,
        ):
            """Evaluate one posterior sample. ``params`` is a dict of 0-D arrays."""
            # --- Line parameters ---
            flux_per_line, tau_per_line, centers, p0, p1, p2 = _build_line_params(
                cm, params, n_lines, z_sys
            )
            cover_per_line = _build_cover_params(cm, params, n_lines)

            # --- Convert peak-tau to area-tau ---
            if cm.tau_names:
                tau_per_line = _peak_to_area_tau(
                    tau_per_line,
                    centers,
                    p0,
                    p1,
                    p2,
                    _pc,
                    cm.is_tau,
                    _eval_fn=args._evaluate_at_centers_fn,
                )

            # --- Calibration ---
            r_scale = params[_disp.r_scale.name]
            flux_scale_val = params[_disp.flux_scale.name]
            pix_offset = params[_disp.pix_offset.name]

            # --- Edge topology (apply sub-pixel offset) ---
            edges = _edges - pix_offset * _dlam_edges
            widths = jnp.diff(edges)
            # Per-pixel low/high for diagnostics and convolution mode.
            low = edges[:-1][_keep]
            high = edges[1:][_keep]
            wavelength = 0.5 * (low + high)

            # --- LSF ---
            # Per-line LSF (used for tau diagnostic at midpoints).
            lsf_fwhm = centers / (_disp.R(centers * _inv_wl_scale) * r_scale)
            # Per-edge LSF (shared across lines, used by analytic mode).
            edges_disp = edges * _inv_wl_scale
            lsf_at_edges = edges / (_disp.R(edges_disp) * r_scale)

            # Scaled line fluxes for this spectrum.
            scaled_flux = flux_per_line * _line_scale

            # --- Optical depth profiles at pixel midpoints (mode-independent) ---
            # For emission-only models has_tau is a Python False at trace time, so
            # the entire evaluate_lines call is compiled out of the XLA program.
            if has_tau:
                phi_mid = _efn(wavelength, centers, lsf_fwhm, p0, p1, p2, _pc)
                tau_profiles_arr = jnp.where(
                    cm.is_tau[:, None], tau_per_line[:, None] * phi_mid, 0.0
                )
            else:
                tau_profiles_arr = jnp.zeros((n_lines, wavelength.shape[0]))

            # --- Continuum (per-region for decomposition) ---
            # For analytic: the same pixel-averaged evaluation as the likelihood
            # (integrate_continuum_regions on the edges), so predictions are
            # exactly the fitted model.
            # For convolution: evaluated on the fine grid inside the branch below.
            cont_total_scaled: jnp.ndarray = jnp.zeros(
                (args.cont_group_matrix.shape[0], wavelength.shape[0])
            )
            cont_regions_scaled: list[jnp.ndarray] = []
            if args.integration_mode != 'convolution':
                cont_regions = [
                    r[_keep]
                    for r in integrate_continuum_regions(
                        edges,
                        args,
                        params,
                        z_sys,
                        region_lsf_fwhm(args, _disp, z_sys, _inv_wl_scale, r_scale),
                        _full_rm,
                        interval_lsf_fwhm(lsf_at_edges),
                    )
                ]
                cont_total_scaled = (
                    _group_sum(cont_regions, args.cont_group_matrix, wavelength.shape)
                    * _cont_scale
                )
                cont_regions_scaled = [
                    r * _cont_scale * flux_scale_val for r in cont_regions
                ]

            # --- Line decomposition ---
            if args.integration_mode == 'convolution':
                # Numerical LSF convolution: evaluate intrinsic model (lsf_fwhm=0)
                # on a fine sub-pixel grid, convolve with the wavelength-dependent
                # Gaussian LSF, then pixel-average.  Per-line decomposition uses
                # compose_leave_one_out on the fine grid; convolution is linear so
                # convolving deltas separately is exact.
                n_super = args.n_super
                half_width = args.conv_half_width
                assert n_super is not None
                assert half_width is not None
                n_pixels = low.shape[0]

                # Fine grid in wavelength order: all sub-bins of pixel 0, then pixel 1, etc.
                # Shape: (n_pixels * n_super,).
                offsets = (jnp.arange(n_super) + 0.5) / n_super
                x_fine = (
                    low[None, :] + offsets[:, None] * (high - low)[None, :]
                )  # (n_super, n_pixels)
                x_flat = x_fine.T.ravel()  # (n_pixels * n_super,) in wavelength order

                # Intrinsic profiles (lsf_fwhm=0) on fine grid.
                zero_lsf = jnp.zeros_like(centers)
                phi_fine = _efn(x_flat, centers, zero_lsf, p0, p1, p2, _pc)

                # LSF sigma at each fine-grid point.
                sigma_fine = (
                    x_flat
                    / (_disp.R(x_flat * _inv_wl_scale) * r_scale)
                    * _FWHM_TO_SIGMA
                )

                # Per-region continuum on fine grid (lsf_fwhm=0): convolve on fine
                # grid then pixel-average.
                cont_regions_fine = eval_continuum_regions(
                    x_flat, args, params, z_sys, 0.0, np.repeat(_prm, n_super, axis=1)
                )
                cont_regions_scaled = [
                    _lsf_convolve(x_flat, r * _cont_scale, sigma_fine, half_width)
                    .reshape(n_pixels, n_super)
                    .mean(axis=1)
                    * flux_scale_val
                    for r in cont_regions_fine
                ]

                # Total continuum on fine grid for model composition.
                cont_fine_scaled = (
                    _group_sum(cont_regions_fine, args.cont_group_matrix, x_flat.shape)
                    * _cont_scale
                )

                # Leave-one-out decomposition on fine grid.
                total_fine, deltas_fine = compose_leave_one_out(
                    phi_fine,
                    scaled_flux,
                    tau_per_line,
                    cm.is_tau,
                    cm.applies_matrix,
                    args.cont_applies,
                    cont_fine_scaled,
                    has_tau=has_tau,
                    cover_per_line=cover_per_line,
                )

                # Convolve on fine grid, then pixel-average.
                total_conv = _lsf_convolve(x_flat, total_fine, sigma_fine, half_width)
                total_pix = total_conv.reshape(n_pixels, n_super).mean(axis=1)

                deltas_conv = jax.vmap(
                    lambda d: _lsf_convolve(x_flat, d, sigma_fine, half_width)
                )(deltas_fine)  # (n_lines, n_pixels * n_super)
                deltas_pix = deltas_conv.reshape(-1, n_pixels, n_super).mean(axis=2)

                def _source_fine(wav, frm):
                    phi_m = _efn(wav, centers, zero_lsf, p0, p1, p2, _pc)
                    cont_m = eval_continuum(
                        wav, args, params, z_sys, 0.0, frm, args.cont_group_matrix
                    )
                    return compose_from_profiles(
                        phi_m,
                        scaled_flux,
                        tau_per_line,
                        cm.is_tau,
                        cm.applies_matrix,
                        args.cont_applies,
                        cont_m * _cont_scale,
                        has_tau=has_tau,
                        cover_per_line=cover_per_line,
                    )

                overlaps = []
                for win in _wins:
                    term = _overlap.convolved_term(
                        win,
                        x_flat,
                        n_super=n_super,
                        half_width=half_width,
                        disp=_disp,
                        r_scale=r_scale,
                        inv_wl_scale=_inv_wl_scale,
                        canonical_unit=args.canonical_unit,
                        source_fine=_source_fine,
                    )
                    overlaps.append(
                        jnp.zeros(n_pixels)
                        .at[win.q0 : win.q1]
                        .set(params[win.scale_name] * term)
                    )
                for ov in overlaps:
                    total_pix = total_pix + ov

                total = flux_scale_val * total_pix
                line_contribs = flux_scale_val * deltas_pix
                overlaps = [flux_scale_val * ov for ov in overlaps]

                return (
                    total,
                    line_contribs,
                    cont_regions_scaled,
                    tau_profiles_arr,
                    overlaps,
                )
            else:
                # Analytic: cumulative-at-edges per line, then diff + mask
                # → per-pixel-averaged profile.
                cum_per_line = _ifn(
                    edges, centers, lsf_at_edges, p0, p1, p2, _pc
                )  # (n_lines, E)
                per_interval = jnp.diff(cum_per_line, axis=1) / widths
                pixints = per_interval[:, _keep]
                total, line_contribs = compose_leave_one_out(
                    pixints,
                    scaled_flux,
                    tau_per_line,
                    cm.is_tau,
                    cm.applies_matrix,
                    args.cont_applies,
                    cont_total_scaled,
                    has_tau=has_tau,
                    cover_per_line=cover_per_line,
                )

                def _compose(pix, cont_g):
                    return compose_from_profiles(
                        pix,
                        scaled_flux,
                        tau_per_line,
                        cm.is_tau,
                        cm.applies_matrix,
                        args.cont_applies,
                        cont_g * _cont_scale,
                        has_tau=has_tau,
                        cover_per_line=cover_per_line,
                    )

                overlaps = []
                for win in _wins:
                    source = _overlap.analytic_source(
                        win,
                        edges,
                        disp=_disp,
                        r_scale=r_scale,
                        inv_wl_scale=_inv_wl_scale,
                        args=args,
                        context=params,
                        z_sys=z_sys,
                        integrate_lines_fn=_ifn,
                        line_params=(centers, p0, p1, p2, _pc),
                        compose=_compose,
                    )
                    ratio = _overlap.ratio_at_pixels(win, edges, args.canonical_unit)
                    overlaps.append(
                        jnp.zeros(wavelength.shape[0])
                        .at[win.wavelength_slice]
                        .set(params[win.scale_name] * ratio * source)
                    )
                for ov in overlaps:
                    total = total + ov

            # Apply flux_scale to total, per-line and overlap contributions.
            total = flux_scale_val * total
            line_contribs = flux_scale_val * line_contribs
            overlaps = [flux_scale_val * ov for ov in overlaps]

            return total, line_contribs, cont_regions_scaled, tau_profiles_arr, overlaps

        # Build JIT'd vmapped evaluator once per spectrum; reuse on subsequent calls.
        if args._evaluators[i] is None:
            args._evaluators[i] = jax.jit(jax.vmap(_single))
        if batch_size is None:
            total_arr, line_arr, cont_arr, tau_arr, ov_arr = args._evaluators[i](
                context
            )
        else:
            total_arr, line_arr, cont_arr, tau_arr, ov_arr = _evaluate_in_chunks(
                args._evaluators[i], context, n_samples, min(batch_size, n_samples)
            )
        # total_arr:  (n_samples, n_pix)
        # line_arr:   (n_samples, n_lines, n_pix)
        # cont_arr:   list of (n_samples, n_pix), one per continuum region
        # tau_arr:    (n_samples, n_lines, n_pix)
        # ov_arr:     list of (n_samples, n_pix), one per overlap window

        # Transfer line and tau arrays to host once, then use zero-copy NumPy
        # views for per-line slicing.  Individual np.asarray(jax_arr[:, j, :])
        # calls each dispatch a separate XLA slice → host copy.
        line_arr_np = np.asarray(line_arr)
        lines_dict: dict[str, np.ndarray] = {
            args.line_labels[j]: line_arr_np[:, j, :] for j in range(n_lines)
        }
        cont_dict: dict[str, np.ndarray] = {}
        if args.cont_config is not None:
            for k in range(len(args.cont_config)):
                cont_dict[args.continuum_labels[k]] = np.asarray(cont_arr[k])

        is_abs = np.asarray(cm.is_tau)
        if has_tau:
            tau_arr_np = np.asarray(tau_arr)
            tau_dict: dict[str, np.ndarray] = {
                args.line_labels[j]: tau_arr_np[:, j, :]
                for j in range(n_lines)
                if is_abs[j]
            }
        else:
            tau_dict = {}

        results.append(
            SpectrumPrediction(
                wavelength=wl_out,
                total=np.asarray(total_arr),
                lines=lines_dict,
                continuum_regions=cont_dict,
                tau_profiles=tau_dict,
                overlap={
                    w.order: np.asarray(a) for w, a in zip(windows, ov_arr, strict=True)
                },
            )
        )

    return results


def evaluate_intrinsic(
    samples: dict[str, np.ndarray],
    args: ModelArgs,
    wavelength: u.Quantity,
    flux_unit: u.UnitBase | str | None = None,
) -> SpectrumPrediction:
    """Evaluate the exact intrinsic model at arbitrary wavelengths.

    Returns the astrophysical model as the source emits it (at the systemic
    redshift) before any instrument touches it: no LSF broadening, no
    pixel integration, and no per-disperser calibration (``RScale``,
    ``PixOffset`` and ``FluxScale`` are ignored).  Line profiles and
    continuum forms are evaluated pointwise, and absorbers are applied as
    ``exp(-tau * phi)`` with the intrinsic profile, so the result is exact
    for emission, absorption and continuum alike.  It is independent of
    ``args.integration_mode``.

    The observed spectra were fit with their ``FluxScale`` applied, so this
    model differs from a spectrum's data by that factor when it is not 1.

    Parameters
    ----------
    samples : dict of str to ndarray
        Posterior samples as returned by ``mcmc.get_samples()``.  Sites for
        instrument calibration parameters are ignored.
    args : ModelArgs
        Pre-built data bundle from :meth:`ModelBuilder.build`.
    wavelength : astropy.units.Quantity
        Observed-frame wavelength grid, shape ``(n_wavelengths,)``, in any
        length unit.
    flux_unit : astropy.units.UnitBase or str, optional
        Output flux-density unit.  Defaults to the flux unit of the first
        spectrum in ``args.spectra``.

    Returns
    -------
    SpectrumPrediction
        ``wavelength`` holds the input grid in the unit it was given;
        all flux arrays have shape ``(n_samples, n_wavelengths)`` and are in
        ``flux_unit``.  ``lines``, ``continuum_regions`` and ``tau_profiles``
        follow the same conventions as :func:`evaluate_model`.
    """
    if not isinstance(wavelength, u.Quantity):
        raise TypeError('wavelength must be an astropy Quantity with length units.')
    if wavelength.ndim != 1:
        raise ValueError('wavelength must be one-dimensional.')

    cm = args.matrices
    z_sys = args.redshift
    n_lines = cm.wavelengths.shape[0]
    has_tau = bool(cm.tau_names)

    ref = args.spectra[0]
    base_unit = ref.flux_unit
    out_unit = base_unit if flux_unit is None else u.Unit(flux_unit)
    unit_factor = float((1.0 * base_unit).to(out_unit).value)
    line_scale = float(args.line_flux_scales[0]) * unit_factor
    cont_scale = float(args.continuum_scales[0]) * unit_factor

    wl = jnp.asarray(wavelength.to(args.canonical_unit).value)

    _pcodes = (
        args._profile_codes_local
        if args._profile_codes_local is not None
        else cm.profile_codes
    )
    _eval_fn = args._evaluate_fn if args._evaluate_fn is not None else evaluate_lines
    context, _ = _build_context(samples, args)

    def _single(params):
        flux_per_line, tau_per_line, centers, p0, p1, p2 = _build_line_params(
            cm, params, n_lines, z_sys
        )
        cover_per_line = _build_cover_params(cm, params, n_lines)
        if has_tau:
            tau_per_line = _peak_to_area_tau(
                tau_per_line,
                centers,
                p0,
                p1,
                p2,
                _pcodes,
                cm.is_tau,
                _eval_fn=args._evaluate_at_centers_fn,
            )
        zero_lsf = jnp.zeros_like(centers)
        phi = _eval_fn(wl, centers, zero_lsf, p0, p1, p2, _pcodes)

        cont_regions = eval_continuum_regions(wl, args, params, z_sys, 0.0)
        cont_total = _group_sum(cont_regions, args.cont_group_matrix, wl.shape)
        total, deltas = compose_leave_one_out(
            phi,
            flux_per_line * line_scale,
            tau_per_line,
            cm.is_tau,
            cm.applies_matrix,
            args.cont_applies,
            cont_total * cont_scale,
            has_tau=has_tau,
            cover_per_line=cover_per_line,
        )
        if has_tau:
            tau_profiles = jnp.where(
                cm.is_tau[:, None], tau_per_line[:, None] * phi, 0.0
            )
        else:
            tau_profiles = jnp.zeros((n_lines, wl.shape[0]))
        return (total, deltas, [r * cont_scale for r in cont_regions], tau_profiles)

    total_arr, line_arr, cont_arr, tau_arr = jax.jit(jax.vmap(_single))(context)

    line_np = np.asarray(line_arr)
    lines_dict = {args.line_labels[j]: line_np[:, j, :] for j in range(n_lines)}
    cont_dict: dict[str, np.ndarray] = {}
    if args.cont_config is not None:
        for k in range(len(args.cont_config)):
            cont_dict[args.continuum_labels[k]] = np.asarray(cont_arr[k])
    tau_dict: dict[str, np.ndarray] = {}
    if has_tau:
        tau_np = np.asarray(tau_arr)
        is_abs = np.asarray(cm.is_tau)
        tau_dict = {
            args.line_labels[j]: tau_np[:, j, :] for j in range(n_lines) if is_abs[j]
        }

    return SpectrumPrediction(
        wavelength=np.asarray(wavelength.value),
        total=np.asarray(total_arr),
        lines=lines_dict,
        continuum_regions=cont_dict,
        tau_profiles=tau_dict,
    )
