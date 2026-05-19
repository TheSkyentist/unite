"""Continuum evaluation and integration.

Provides functions for evaluating continuum models at arbitrary wavelength
points:

- :func:`eval_continuum` — total continuum across all regions.
- :func:`eval_continuum_regions` — per-region continuum contributions.
- :func:`integrate_continuum` — pixel-integrated continuum via each form's
  :meth:`~unite.continuum.library.ContinuumForm.integrate` method.
"""

from __future__ import annotations

import jax.numpy as jnp


def _resolve_region_params(args, context, z_sys, k):
    """Extract observed-frame bounds and parameter dict for continuum region *k*.

    Shared between :func:`eval_continuum_regions` and
    :func:`integrate_continuum` to avoid duplicating the redshift transform
    and ``norm_wav`` unit-conversion logic.

    Returns
    -------
    obs_low, obs_high, obs_center : float
        Observed-frame region bounds and center in canonical wavelength units.
    cont_params : dict
        Parameter values for the region, with ``norm_wav`` converted to
        canonical wavelength units.
    """
    obs_low = args.cont_low[k] * (1.0 + z_sys)
    obs_high = args.cont_high[k] * (1.0 + z_sys)
    obs_center = args.cont_center[k] * (1.0 + z_sys)
    cont_params = {
        pn: (
            context[tok.name] * args.cont_nw_conv[k] * (1.0 + z_sys)
            if pn == 'norm_wav'
            else context[tok.name]
        )
        for pn, tok in args.cont_resolved_params[k].items()
    }
    return obs_low, obs_high, obs_center, cont_params


def eval_continuum(
    wavelength, args, context, z_sys, lsf_fwhm: float | jnp.ndarray = 0.0
):
    """Evaluate the total continuum at arbitrary wavelength points.

    Parameters
    ----------
    wavelength : jnp.ndarray
        Observed-frame wavelength array (canonical unit).
    args : ModelArgs
        Pre-built data bundle.
    context : dict
        Parameter values keyed by numpyro site name.
    z_sys : float
        Systemic redshift.
    lsf_fwhm : jnp.ndarray or float, optional
        LSF FWHM at each wavelength point (canonical unit).
        Default ``0.0`` means no LSF convolution.

    Returns
    -------
    jnp.ndarray
        Continuum flux density at each wavelength point (unscaled — before
        ``continuum_scale`` and normalisation are applied).
    """
    continuum = jnp.zeros_like(wavelength)
    if args.cont_config is not None:
        for region_cont in eval_continuum_regions(
            wavelength, args, context, z_sys, lsf_fwhm
        ):
            continuum = continuum + region_cont
    return continuum


def eval_continuum_regions(
    wavelength, args, context, z_sys, lsf_fwhm: float | jnp.ndarray = 0.0
):
    """Evaluate each continuum region independently at arbitrary wavelength points.

    Parameters
    ----------
    wavelength : jnp.ndarray
        Observed-frame wavelength array (canonical unit).
    args : ModelArgs
        Pre-built data bundle.
    context : dict
        Parameter values keyed by numpyro site name.
    z_sys : float
        Systemic redshift.
    lsf_fwhm : jnp.ndarray or float, optional
        LSF FWHM at each wavelength point (canonical unit).
        Default ``0.0`` means no LSF convolution.

    Returns
    -------
    list of jnp.ndarray
        One array per continuum region, each masked to zero outside its
        wavelength bounds.  Unscaled (before ``continuum_scale`` and
        normalisation).
    """
    regions = []
    if args.cont_config is not None:
        for k in range(len(args.cont_config)):
            obs_low, obs_high, obs_center, cont_params = _resolve_region_params(
                args, context, z_sys, k
            )
            in_region = (wavelength >= obs_low) & (wavelength <= obs_high)
            region_cont = args.cont_forms[k].evaluate(
                wavelength, obs_center, cont_params, obs_low, obs_high, lsf_fwhm, z_sys
            )
            regions.append(jnp.where(in_region, region_cont, 0.0))
    return regions


def integrate_continuum(
    edges, args, context, z_sys, lsf_fwhm_per_region: float | jnp.ndarray = 0.0
):
    """Pixel-averaged continuum via each form's edges-based integrate method.

    The returned array has length ``E - 1``; the caller applies the
    spectrum's ``keep_mask`` to drop entries that span inter-pixel gaps.

    Parameters
    ----------
    edges : jnp.ndarray, shape ``(E,)``
        Unique pixel edges (canonical unit).
    args : ModelArgs
        Pre-built data bundle.
    context : dict
        Parameter values keyed by numpyro site name.
    z_sys : float
        Systemic redshift.
    lsf_fwhm_per_region : float or jnp.ndarray, optional
        LSF FWHM scalar per continuum region (used only by polynomial-based
        forms via the analytic Gaussian-moment convolution; non-polynomial
        forms ignore it).  Scalar broadcasts to all regions; otherwise
        provide one entry per region.  Default ``0.0``.

    Returns
    -------
    jnp.ndarray, shape ``(E - 1,)``
        Pixel-averaged continuum (unscaled) across all ``E - 1`` edge
        intervals.  Entries that span inter-pixel gaps must be discarded
        by the caller via the spectrum's ``keep_mask``.
    """
    edges = jnp.asarray(edges)
    widths = jnp.diff(edges)
    mids = 0.5 * (edges[1:] + edges[:-1])
    continuum_per_interval = jnp.zeros_like(widths)
    if args.cont_config is not None:
        lsf_arr = jnp.atleast_1d(jnp.asarray(lsf_fwhm_per_region))
        if lsf_arr.shape[0] == 1:
            lsf_arr = jnp.broadcast_to(lsf_arr, (len(args.cont_config),))
        for k in range(len(args.cont_config)):
            obs_low, obs_high, obs_center, cont_params = _resolve_region_params(
                args, context, z_sys, k
            )
            in_region = (mids >= obs_low) & (mids <= obs_high)
            cum = args.cont_forms[k].integrate(
                edges, obs_center, cont_params, obs_low, obs_high, lsf_arr[k], z_sys
            )
            per_pixel_avg = jnp.diff(cum) / widths
            continuum_per_interval = continuum_per_interval + jnp.where(
                in_region, per_pixel_avg, 0.0
            )
    return continuum_per_interval
