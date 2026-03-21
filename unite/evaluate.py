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

from unite._utils import C_KMS
from unite.line.profiles import integrate_lines
from unite.model import ModelArgs
from unite.prior import Fixed


@dataclass
class SpectrumPrediction:
    """Decomposed model prediction for a single spectrum.

    All arrays are in original (un-normalized) flux units.
    """

    #: Pixel-center wavelengths in the disperser's unit. Shape ``(n_pixels,)``.
    wavelength: np.ndarray
    #: Total model flux (lines + continuum). Shape ``(n_samples, n_pixels)``.
    total: np.ndarray
    #: Per-line contributions keyed by informative line labels (e.g. ``'Ha'``, ``'[NII]_6585'``).
    #: Shape ``(n_samples, n_pixels)`` each.
    lines: dict[str, np.ndarray]
    #: Per-continuum-region contributions keyed by informative region labels
    #: (e.g. ``'linear_6400_6700'``, ``'powerlaw_0.95_2.5'``).
    #: Shape ``(n_samples, n_pixels)`` each.
    continuum_regions: dict[str, np.ndarray]
    #: Combined transmission from all absorption lines. Shape ``(n_samples, n_pixels)``.
    #: ``None`` when no absorption lines are present.
    transmission: np.ndarray | None = None


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
        n_pix = spectrum.npix

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
            _n_pix=n_pix,
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

            # --- Line integration ---
            lsf_fwhm = centers / (_disp.R(centers * _inv_wl_scale) * r_scale)
            pixints = integrate_lines(
                low, high, centers, lsf_fwhm, p0, p1, p2, cm.profile_codes
            ) / (high - low)

            # Emission lines: flux-weighted profiles.
            emission_mask = ~cm.is_absorption
            emission_pixints = jnp.where(emission_mask[:, None], pixints, 0.0)
            line_contribs = (flux_per_line * _line_scale)[:, None] * emission_pixints
            line_contribs_scaled = line_contribs * flux_scale_val

            # Absorption: transmission = exp(-sum(tau * phi)).
            absorption_mask = cm.is_absorption
            absorption_pixints = jnp.where(absorption_mask[:, None], pixints, 0.0)
            total_tau = (tau_per_line[:, None] * absorption_pixints).sum(axis=0)
            transmission = jnp.exp(-total_tau)

            # --- Continuum ---
            continuum_total = jnp.zeros(_n_pix)
            cont_contributions = []
            if args.cont_config is not None:
                for k in range(len(args.cont_config)):
                    obs_low = args.cont_low[k] * (1.0 + z_sys)
                    obs_high = args.cont_high[k] * (1.0 + z_sys)
                    obs_center = args.cont_center[k] * (1.0 + z_sys)
                    in_region = (wavelength >= obs_low) & (wavelength <= obs_high)
                    cont_p = {
                        pn: (
                            params[tok.name] * args.cont_nw_conv[k] * (1.0 + z_sys)
                            if pn == 'norm_wav'
                            else params[tok.name]
                        )
                        for pn, tok in args.cont_resolved_params[k].items()
                    }
                    form = args.cont_forms[k]
                    region_cont = (
                        form.evaluate(wavelength, obs_center, cont_p, obs_low, obs_high)
                        * _cont_scale
                    )
                    region_cont = jnp.where(in_region, region_cont, 0.0)
                    cont_contributions.append(region_cont * flux_scale_val)
                    continuum_total = continuum_total + region_cont

            # Apply transmission based on absorber_position.
            emission_sum = line_contribs.sum(axis=0)
            absorber_position = args.absorber_position
            if absorber_position == 'foreground':
                total = flux_scale_val * transmission * (emission_sum + continuum_total)
            elif absorber_position == 'behind_lines':
                total = flux_scale_val * (emission_sum + transmission * continuum_total)
            else:  # behind_continuum (validated at build time)
                total = flux_scale_val * (transmission * emission_sum + continuum_total)

            return total, line_contribs_scaled, cont_contributions, transmission

        # Vectorise over the leading sample axis of every parameter in context.
        # JAX treats the dict as a pytree and maps axis 0 of each leaf.
        total_arr, line_arr, cont_arr, trans_arr = jax.vmap(_single)(context)
        # total_arr:  (n_samples, n_pix)
        # line_arr:   (n_samples, n_lines, n_pix)
        # cont_arr:   list of (n_samples, n_pix), one per continuum region
        # trans_arr:  (n_samples, n_pix)

        lines_dict: dict[str, np.ndarray] = {
            args.line_labels[j]: np.asarray(line_arr[:, j, :]) for j in range(n_lines)
        }
        cont_dict: dict[str, np.ndarray] = {}
        if args.cont_config is not None:
            for k in range(len(args.cont_config)):
                cont_dict[args.continuum_labels[k]] = np.asarray(cont_arr[k])

        # Only include transmission when there are absorption lines.
        has_absorption = bool(jnp.any(cm.is_absorption))
        transmission_out = np.asarray(trans_arr) if has_absorption else None

        results.append(
            SpectrumPrediction(
                wavelength=wl_out,
                total=np.asarray(total_arr),
                lines=lines_dict,
                continuum_regions=cont_dict,
                transmission=transmission_out,
            )
        )

    return results
