"""Continuum evaluation and integration.

Provides functions for evaluating continuum models at arbitrary wavelength
points:

- :func:`eval_continuum` — total continuum across all regions.
- :func:`eval_continuum_regions` — per-region continuum contributions.
- :func:`integrate_continuum` — pixel-integrated continuum (evaluated at
  pixel centers, since continuum is assumed slowly varying).
"""

from __future__ import annotations

import jax.numpy as jnp


def eval_continuum(wavelength, args, context, z_sys):
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

    Returns
    -------
    jnp.ndarray
        Continuum flux density at each wavelength point (unscaled — before
        ``continuum_scale`` and normalisation are applied).
    """
    continuum = jnp.zeros_like(wavelength)
    if args.cont_config is not None:
        for region_cont in eval_continuum_regions(wavelength, args, context, z_sys):
            continuum = continuum + region_cont
    return continuum


def eval_continuum_regions(wavelength, args, context, z_sys):
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
            obs_low = args.cont_low[k] * (1.0 + z_sys)
            obs_high = args.cont_high[k] * (1.0 + z_sys)
            obs_center = args.cont_center[k] * (1.0 + z_sys)
            in_region = (wavelength >= obs_low) & (wavelength <= obs_high)
            cont_params = {
                pn: (
                    context[tok.name] * args.cont_nw_conv[k] * (1.0 + z_sys)
                    if pn == 'norm_wav'
                    else context[tok.name]
                )
                for pn, tok in args.cont_resolved_params[k].items()
            }
            form = args.cont_forms[k]
            region_cont = form.evaluate(
                wavelength, obs_center, cont_params, obs_low, obs_high
            )
            regions.append(jnp.where(in_region, region_cont, 0.0))
    return regions


def integrate_continuum(low, high, args, context, z_sys):
    """Pixel-integrated continuum evaluated at pixel centers.

    The continuum is assumed to vary slowly enough that pixel integration
    is unnecessary; this function evaluates at pixel centers and returns
    the result directly.

    Parameters
    ----------
    low, high : jnp.ndarray, shape ``(n_pixels,)``
        Pixel bin edges (canonical unit).
    args : ModelArgs
        Pre-built data bundle.
    context : dict
        Parameter values keyed by numpyro site name.
    z_sys : float
        Systemic redshift.

    Returns
    -------
    jnp.ndarray
        Total continuum at pixel centers (unscaled).
    """
    wavelength = (low + high) / 2.0
    return eval_continuum(wavelength, args, context, z_sys)
