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
from unite._utils import C_KMS
from unite.continuum.compute import eval_continuum, eval_continuum_regions
from unite.line.compute import evaluate_lines, integrate_lines
from unite.model import ModelArgs
from unite.prior import Fixed


@dataclass
class SpectrumPrediction:
    """Decomposed model prediction for a single spectrum.

    All arrays are in original (un-normalized) flux units.

    For **emission lines**, each entry in :attr:`lines` is the exact
    flux contribution of that line (positive, adds to the total).

    For **absorption lines**, each entry in :attr:`lines` is the flux
    *removed* by that absorber (negative): ``total - total_without_j``.
    """

    #: Pixel-center wavelengths in the disperser's unit. Shape ``(n_pixels,)``.
    wavelength: np.ndarray
    #: Total model flux (lines + continuum). Shape ``(n_samples, n_pixels)``.
    total: np.ndarray
    #: Per-line contributions keyed by informative line labels (e.g. ``'Ha'``, ``'[NII]_6585'``).
    #: For emission lines: flux-weighted profile (positive).
    #: For absorption lines: flux removed by the absorber (negative).
    #: Shape ``(n_samples, n_pixels)`` each.
    lines: dict[str, np.ndarray]
    #: Per-continuum-region contributions keyed by informative region labels
    #: (e.g. ``'linear_6400_6700'``, ``'powerlaw_0.95_2.5'``).
    #: Shape ``(n_samples, n_pixels)`` each.
    continuum_regions: dict[str, np.ndarray]


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
    results: list[SpectrumPrediction] = []

    for i, spectrum in enumerate(args.spectra):
        disp = spectrum.disperser
        wl_scale = args.spec_to_canonical[i]
        inv_wl_scale = 1.0 / wl_scale
        wl_out = np.asarray(spectrum.wavelength)

        # Static arrays: do not depend on sample values.
        low_base = jnp.asarray(spectrum.low * wl_scale)
        high_base = jnp.asarray(spectrum.high * wl_scale)
        if disp.pix_offset is not None:
            mid_disp = jnp.asarray((spectrum.low + spectrum.high) / 2.0)
            dlam = disp.dlam_dpix(mid_disp) * wl_scale  # (n_pix,)
        else:
            dlam = None

        line_scale = float(args.line_flux_scales[i])
        cont_scale = float(args.continuum_scales[i])

        # Keyword-only defaults bind the per-iteration values at definition
        # time, avoiding late-binding closure issues (ruff B023) while
        # allowing jax.vmap to vmap only over the positional `params` arg.
        def _single(
            params,
            *,
            _low=low_base,
            _high=high_base,
            _dlam=dlam,
            _disp=disp,
            _inv_wl_scale=inv_wl_scale,
            _line_scale=line_scale,
            _cont_scale=cont_scale,
        ):
            """Evaluate one posterior sample. ``params`` is a dict of 0-D arrays."""
            # --- Line parameters ---
            if cm.flux_names:
                flux_vec = jnp.stack([params[n] for n in cm.flux_names])
                flux_per_line = flux_vec @ cm.flux_matrix * cm.strengths
            else:
                flux_per_line = jnp.zeros(n_lines)

            if cm.tau_names:
                tau_vec = jnp.stack([params[n] for n in cm.tau_names])
                tau_per_line = tau_vec @ cm.tau_matrix
            else:
                tau_per_line = jnp.zeros(n_lines)

            z_vec = jnp.stack([params[n] for n in cm.z_names])
            z_per_line = z_vec @ cm.z_matrix
            centers = cm.wavelengths * (1.0 + z_sys + z_per_line)

            p0_kms = (
                jnp.stack([params[n] for n in cm.p0_names]) @ cm.p0_matrix
                if cm.p0_names
                else jnp.zeros(n_lines)
            )
            p0 = centers * p0_kms / C_KMS

            p1v_kms = (
                jnp.stack([params[n] for n in cm.p1v_names]) @ cm.p1v_matrix
                if cm.p1v_names
                else jnp.zeros(n_lines)
            )
            p1v = centers * p1v_kms / C_KMS

            p1d = (
                jnp.stack([params[n] for n in cm.p1d_names]) @ cm.p1d_matrix
                if cm.p1d_names
                else jnp.zeros(n_lines)
            )
            p1 = p1v + p1d

            p2 = (
                jnp.stack([params[n] for n in cm.p2_names]) @ cm.p2_matrix
                if cm.p2_names
                else jnp.zeros(n_lines)
            )

            # --- Calibration ---
            r_scale = params[_disp.r_scale.name] if _disp.r_scale is not None else 1.0
            flux_scale_val = (
                params[_disp.flux_scale.name] if _disp.flux_scale is not None else 1.0
            )
            pix_offset = (
                params[_disp.pix_offset.name] if _disp.pix_offset is not None else 0.0
            )

            # --- Pixel edges (apply sub-pixel offset if present) ---
            low = _low
            high = _high
            if _dlam is not None:
                low = low + pix_offset * _dlam
                high = high + pix_offset * _dlam
            wavelength = (low + high) / 2.0

            # --- LSF ---
            lsf_fwhm = centers / (_disp.R(centers * _inv_wl_scale) * r_scale)

            # LSF FWHM at pixel centres for continuum convolution.
            cont_lsf_fwhm = wavelength / (_disp.R(wavelength * _inv_wl_scale) * r_scale)

            # Scaled line fluxes for this spectrum.
            scaled_flux = flux_per_line * _line_scale

            # --- Continuum (per-region for decomposition) ---
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
            if args.integration_mode == 'quadrature':
                # GL quadrature: evaluate full composed model at sub-pixel
                # nodes and integrate.  Leave-one-out at each node gives
                # exact per-line contributions.
                mid = (low + high) / 2.0
                half_width = (high - low) / 2.0
                nodes = args.quadrature_nodes  # (n_nodes,)
                weights = args.quadrature_weights  # (n_nodes,)

                # Sub-pixel wavelengths: (n_nodes, n_pix)
                x = mid[None, :] + half_width[None, :] * nodes[:, None]

                # Evaluate all profiles and continuum at each node.
                def _at_node(wav):
                    phi = evaluate_lines(
                        wav, centers, lsf_fwhm, p0, p1, p2, cm.profile_codes
                    )
                    node_lsf = wav / (_disp.R(wav * _inv_wl_scale) * r_scale)
                    cont = (
                        eval_continuum(wav, args, params, z_sys, node_lsf) * _cont_scale
                    )
                    total, deltas = compose_leave_one_out(
                        phi,
                        scaled_flux,
                        tau_per_line,
                        cm.is_absorption,
                        cont,
                        args.absorber_position,
                    )
                    return total, deltas

                # (n_nodes, n_pix) and (n_nodes, n_lines, n_pix)
                node_totals, node_deltas = jax.vmap(_at_node)(x)

                # Pixel-average via GL weighted sum (factor 0.5 from
                # the [-1,1] → [low,high] change of variable).
                total = 0.5 * jnp.dot(weights, node_totals)
                line_contribs = 0.5 * jnp.einsum('n,nlp->lp', weights, node_deltas)
            else:
                # Analytic: CDF-based per-line integration, then compose.
                pixints = integrate_lines(
                    low, high, centers, lsf_fwhm, p0, p1, p2, cm.profile_codes
                ) / (high - low)
                total, line_contribs = compose_leave_one_out(
                    pixints,
                    scaled_flux,
                    tau_per_line,
                    cm.is_absorption,
                    cont_total_scaled,
                    args.absorber_position,
                )

            # Apply flux_scale to total and per-line contributions.
            total = flux_scale_val * total
            line_contribs = flux_scale_val * line_contribs

            return total, line_contribs, cont_regions_scaled

        # Vectorise over the leading sample axis of every parameter in context.
        # JAX treats the dict as a pytree and maps axis 0 of each leaf.
        total_arr, line_arr, cont_arr = jax.vmap(_single)(context)
        # total_arr:  (n_samples, n_pix)
        # line_arr:   (n_samples, n_lines, n_pix)
        # cont_arr:   list of (n_samples, n_pix), one per continuum region

        lines_dict: dict[str, np.ndarray] = {
            args.line_labels[j]: np.asarray(line_arr[:, j, :]) for j in range(n_lines)
        }
        cont_dict: dict[str, np.ndarray] = {}
        if args.cont_config is not None:
            for k in range(len(args.cont_config)):
                cont_dict[args.continuum_labels[k]] = np.asarray(cont_arr[k])

        results.append(
            SpectrumPrediction(
                wavelength=wl_out,
                total=np.asarray(total_arr),
                lines=lines_dict,
                continuum_regions=cont_dict,
            )
        )

    return results
