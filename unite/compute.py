"""Model evaluator: decompose posterior predictions into per-line and per-region contributions.

Given posterior samples and :class:`~unite.model.ModelArgs`, this module
reconstructs the full model prediction for each spectrum, broken down into
individual line and continuum-region contributions in **original flux units**.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from unite._compose import compose_leave_one_out
from unite._lsf import _FWHM_TO_SIGMA, _lsf_convolve
from unite.continuum.compute import eval_continuum_regions
from unite.line.compute import (
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
    #: LSF-convolved optical depth profiles ``tau_j * phi_j(λ)`` for absorption
    #: lines, evaluated at pixel midpoints.  Dimensionless and non-negative.
    #: Keyed by line label; empty dict for emission-only models.
    #: Shape ``(n_samples, n_pixels)`` each.
    tau_profiles: dict[str, np.ndarray]


def evaluate_model(
    samples: dict[str, np.ndarray], args: ModelArgs
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

    Returns
    -------
    list of SpectrumPrediction
        One prediction per spectrum in ``args.spectra``.
    """
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

    # --- Build parameter dict with a uniform (n_samples,) leading axis ---
    context: dict[str, jnp.ndarray] = {}
    n_samples = None
    for pname in args.dependency_order:
        prior = args.all_priors[pname]
        if isinstance(prior, Fixed):
            context[pname] = jnp.asarray(prior.value)
        else:
            arr = jnp.asarray(samples[pname])
            context[pname] = arr
            if n_samples is None and arr.ndim >= 1:
                n_samples = arr.shape[0]

    if n_samples is None:
        n_samples = 1

    # Broadcast Fixed (scalar) params to (n_samples,) so every leaf has the
    # same leading axis and vmap can map uniformly over the dict pytree.
    context = {
        k: (jnp.broadcast_to(v, (n_samples,)) if v.ndim == 0 else v)
        for k, v in context.items()
    }

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
        if disp.pix_offset is not None:
            dlam_edges = disp.dlam_dpix(edges_disp_base) * wl_scale  # (E,)
        else:
            dlam_edges = None

        line_scale = float(args.line_flux_scales[i])
        cont_scale = float(args.continuum_scales[i])

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
        ):
            """Evaluate one posterior sample. ``params`` is a dict of 0-D arrays."""
            # --- Line parameters ---
            flux_per_line, tau_per_line, centers, p0, p1, p2 = _build_line_params(
                cm, params, n_lines, z_sys
            )

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
            r_scale = params[_disp.r_scale.name] if _disp.r_scale is not None else 1.0
            flux_scale_val = (
                params[_disp.flux_scale.name] if _disp.flux_scale is not None else 1.0
            )
            pix_offset = (
                params[_disp.pix_offset.name] if _disp.pix_offset is not None else 0.0
            )

            # --- Edge topology (apply sub-pixel offset if present) ---
            edges = _edges
            if _dlam_edges is not None:
                edges = edges + pix_offset * _dlam_edges
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
            # LSF FWHM at pixel centres for continuum convolution mode.
            cont_lsf_fwhm = wavelength / (_disp.R(wavelength * _inv_wl_scale) * r_scale)

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
            # For analytic: evaluate at pixel centres with LSF.
            # For convolution: evaluated on the fine grid inside the branch below.
            cont_total_scaled: jnp.ndarray = jnp.zeros_like(wavelength)
            cont_regions_scaled: list[jnp.ndarray] = []
            if args.integration_mode != 'convolution':
                cont_regions = eval_continuum_regions(
                    wavelength, args, params, z_sys, cont_lsf_fwhm
                )
                cont_total = jnp.zeros_like(wavelength)
                for region in cont_regions:
                    cont_total = cont_total + region
                cont_total_scaled = cont_total * _cont_scale
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
                    x_flat, args, params, z_sys, 0.0
                )
                cont_regions_scaled = [
                    _lsf_convolve(x_flat, r * _cont_scale, sigma_fine, half_width)
                    .reshape(n_pixels, n_super)
                    .mean(axis=1)
                    * flux_scale_val
                    for r in cont_regions_fine
                ]

                # Total continuum on fine grid for model composition.
                cont_fine_total = jnp.zeros_like(x_flat)
                for r in cont_regions_fine:
                    cont_fine_total = cont_fine_total + r
                cont_fine_scaled = cont_fine_total * _cont_scale

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
                )

                # Convolve on fine grid, then pixel-average.
                total_conv = _lsf_convolve(x_flat, total_fine, sigma_fine, half_width)
                total_pix = total_conv.reshape(n_pixels, n_super).mean(axis=1)

                deltas_conv = jax.vmap(
                    lambda d: _lsf_convolve(x_flat, d, sigma_fine, half_width)
                )(deltas_fine)  # (n_lines, n_pixels * n_super)
                deltas_pix = deltas_conv.reshape(-1, n_pixels, n_super).mean(axis=2)

                total = flux_scale_val * total_pix
                line_contribs = flux_scale_val * deltas_pix

                return total, line_contribs, cont_regions_scaled, tau_profiles_arr
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
                )

            # Apply flux_scale to total and per-line contributions.
            total = flux_scale_val * total
            line_contribs = flux_scale_val * line_contribs

            return total, line_contribs, cont_regions_scaled, tau_profiles_arr

        # Build JIT'd vmapped evaluator once per spectrum; reuse on subsequent calls.
        if args._evaluators[i] is None:
            args._evaluators[i] = jax.jit(jax.vmap(_single))
        total_arr, line_arr, cont_arr, tau_arr = args._evaluators[i](context)
        # total_arr:  (n_samples, n_pix)
        # line_arr:   (n_samples, n_lines, n_pix)
        # cont_arr:   list of (n_samples, n_pix), one per continuum region
        # tau_arr:    (n_samples, n_lines, n_pix)

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
            )
        )

    return results
