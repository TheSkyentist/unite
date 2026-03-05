"""Model evaluator: decompose posterior predictions into per-line and per-region contributions.

Given posterior samples and :class:`~unite.model.ModelArgs`, this module
reconstructs the full model prediction for each spectrum, broken down into
individual line and continuum-region contributions in **original flux units**.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    #: Pixel-centre wavelengths in the disperser's unit. Shape ``(n_pixels,)``.
    wavelength: np.ndarray
    #: Total model flux (lines + continuum). Shape ``(n_samples, n_pixels)``.
    total: np.ndarray
    #: Per-line contributions keyed by ``'line_{idx}'``. Shape ``(n_samples, n_pixels)`` each.
    lines: dict[str, np.ndarray]
    #: Per-continuum-region contributions keyed by ``'cont_{idx}'``. Shape ``(n_samples, n_pixels)`` each.
    continuum_regions: dict[str, np.ndarray]


def _get_scalar(context: dict, name: str, sample_idx: int) -> float:
    """Extract a scalar value from context for a given sample index."""
    val = context[name]
    if val.ndim == 0:
        return val
    return val[sample_idx]


def evaluate_model(
    samples: dict[str, np.ndarray], args: ModelArgs
) -> list[SpectrumPrediction]:
    """Evaluate the model for each posterior sample and decompose contributions.

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

    # --- Resolve all parameter values from samples ---
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

    # --- Per-spectrum evaluation, looping over samples ---
    results: list[SpectrumPrediction] = []

    for i, spectrum in enumerate(args.spectra):
        disp = spectrum.disperser
        wl_scale = args.spec_to_canonical[i]
        inv_wl_scale = 1.0 / wl_scale
        wl_out = np.asarray(spectrum.wavelength)

        all_totals = []
        all_line_contribs: dict[int, list] = {j: [] for j in range(n_lines)}
        all_cont_contribs: dict[str, list] = {}

        for s in range(n_samples):
            # --- Per-line parameters for this sample ---
            flux_vec = jnp.stack([_get_scalar(context, n, s) for n in cm.flux_names])
            flux_per_line = flux_vec @ cm.flux_matrix * cm.strengths

            z_vec = jnp.stack([_get_scalar(context, n, s) for n in cm.z_names])
            z_per_line = z_vec @ cm.z_matrix

            centers = cm.wavelengths * (1.0 + z_sys + z_per_line)

            if cm.p0_names:
                p0_vec = jnp.stack([_get_scalar(context, n, s) for n in cm.p0_names])
                p0_kms = p0_vec @ cm.p0_matrix
            else:
                p0_kms = jnp.zeros(n_lines)
            p0 = centers * p0_kms / C_KMS

            if cm.p1v_names:
                p1v_vec = jnp.stack([_get_scalar(context, n, s) for n in cm.p1v_names])
                p1v_kms = p1v_vec @ cm.p1v_matrix
            else:
                p1v_kms = jnp.zeros(n_lines)
            p1v = centers * p1v_kms / C_KMS

            if cm.p1d_names:
                p1d_vec = jnp.stack([_get_scalar(context, n, s) for n in cm.p1d_names])
                p1d = p1d_vec @ cm.p1d_matrix
            else:
                p1d = jnp.zeros(n_lines)
            p1 = p1v + p1d

            if cm.p2_names:
                p2_vec = jnp.stack([_get_scalar(context, n, s) for n in cm.p2_names])
                p2 = p2_vec @ cm.p2_matrix
            else:
                p2 = jnp.zeros(n_lines)

            # Calibration values.
            r_scale = (
                _get_scalar(context, disp.r_scale.name, s)
                if disp.r_scale is not None
                else 1.0
            )
            flux_scale_val = (
                _get_scalar(context, disp.flux_scale.name, s)
                if disp.flux_scale is not None
                else 1.0
            )
            pix_offset = (
                _get_scalar(context, disp.pix_offset.name, s)
                if disp.pix_offset is not None
                else 0.0
            )

            # Pixel edges in canonical wavelength unit.
            low = spectrum.low * wl_scale
            high = spectrum.high * wl_scale
            if disp.pix_offset is not None:
                mid_disp = (spectrum.low + spectrum.high) / 2.0
                shift = pix_offset * disp.dlam_dpix(mid_disp) * wl_scale
                low = low + shift
                high = high + shift

            wavelength = (low + high) / 2.0

            # LSF FWHM at each line centre.
            lsf_fwhm = centers / (disp.R(centers * inv_wl_scale) * r_scale)

            # Integrate all lines.
            pixints = integrate_lines(
                low, high, centers, lsf_fwhm, p0, p1, p2, cm.profile_codes
            ) / (high - low)

            # Per-line contributions in original flux units.
            # In the model: line_model = (flux * line_flux_scale / norm) * pixints
            # Back to original: multiply by norm, so contributions = flux * line_flux_scale * pixints
            line_contribs = (flux_per_line * args.line_flux_scale)[:, None] * pixints
            line_contribs_scaled = line_contribs * flux_scale_val

            # Continuum.
            cont_contribs_s: dict[str, np.ndarray] = {}
            continuum_total = jnp.zeros(spectrum.npix)
            if args.cont_config is not None:
                for k, region in enumerate(args.cont_config):
                    obs_low = args.cont_low[k] * (1.0 + z_sys)
                    obs_high = args.cont_high[k] * (1.0 + z_sys)
                    obs_center = args.cont_center[k] * (1.0 + z_sys)
                    in_region = (wavelength >= obs_low) & (wavelength <= obs_high)
                    cont_params = {}
                    for pn, tok in args.cont_resolved_params[k].items():
                        val = _get_scalar(context, tok.name, s)
                        if pn == 'normalization_wavelength':
                            val = val * args.cont_nw_conv[k] * (1.0 + z_sys)
                        cont_params[pn] = val
                    region_cont = (
                        region.form.evaluate(wavelength, obs_center, cont_params)
                        * args.continuum_scale
                    )
                    region_cont = jnp.where(in_region, region_cont, 0.0)
                    cont_contribs_s[f'cont_{k}'] = np.asarray(
                        region_cont * flux_scale_val
                    )
                    continuum_total = continuum_total + region_cont

            total = flux_scale_val * (line_contribs.sum(axis=0) + continuum_total)
            all_totals.append(np.asarray(total))
            for j in range(n_lines):
                all_line_contribs[j].append(np.asarray(line_contribs_scaled[j]))
            for key, val in cont_contribs_s.items():
                all_cont_contribs.setdefault(key, []).append(val)

        total_arr = np.stack(all_totals)
        lines_dict: dict[str, np.ndarray] = {
            f'line_{j}': np.stack(all_line_contribs[j]) for j in range(n_lines)
        }
        cont_dict: dict[str, np.ndarray] = {
            key: np.stack(vals) for key, vals in all_cont_contribs.items()
        }

        results.append(
            SpectrumPrediction(
                wavelength=wl_out,
                total=total_arr,
                lines=lines_dict,
                continuum_regions=cont_dict,
            )
        )

    return results
