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
import numpy as np


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


def static_region_masks(edges, cont_low, cont_high, z_sys):
    """Per-interval continuum-region membership from a fixed pixel grid.

    Membership must not depend on sampled parameters.  If it is computed on
    wavelengths shifted by a free ``PixOffset``, pixels cross the hard region
    boundaries as the offset varies and the likelihood becomes a step
    function of the offset.  :meth:`~unite.model.ModelBuilder.build` calls
    this once per spectrum with the *unshifted* edges and stores the result
    in :attr:`~unite.model.ModelArgs.region_masks`; it is then passed as
    ``region_mask`` to :func:`integrate_continuum` or (expanded to the fine
    grid) :func:`eval_continuum_regions`.

    Parameters
    ----------
    edges : array_like, shape ``(E,)``
        Unshifted unique pixel edges (canonical unit).
    cont_low, cont_high : list of float or None
        Rest-frame region bounds (canonical unit), one per region.  ``None``
        when there is no continuum configuration.
    z_sys : float
        Systemic redshift.

    Returns
    -------
    numpy.ndarray of bool, shape ``(n_regions, E - 1)``
        ``True`` where the midpoint of edge interval *i* lies inside region
        *k*.  Shape ``(0, E - 1)`` when there is no continuum configuration.
    """
    edges = np.asarray(edges, dtype=float)
    mids = 0.5 * (edges[1:] + edges[:-1])
    if not cont_low:
        return np.zeros((0, mids.shape[0]), dtype=bool)
    low = np.asarray(cont_low, dtype=float)[:, None] * (1.0 + z_sys)
    high = np.asarray(cont_high, dtype=float)[:, None] * (1.0 + z_sys)
    return (mids[None, :] >= low) & (mids[None, :] <= high)


def _group_sum(per_region, group_matrix, shape):
    """Sum per-region arrays within zorder groups.

    Returns shape ``(n_groups, *shape)``; zeros when there are no regions.
    """
    group_matrix = jnp.asarray(group_matrix, dtype=float)
    if not per_region:
        return jnp.zeros((group_matrix.shape[0], *shape))
    return jnp.tensordot(group_matrix, jnp.stack(per_region), axes=1)


def eval_continuum(
    wavelength,
    args,
    context,
    z_sys,
    lsf_fwhm: float | jnp.ndarray = 0.0,
    region_mask=None,
    group_matrix=None,
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
    region_mask : jnp.ndarray of bool, optional
        Static membership, shape ``(n_regions, n_points)``.  When given it
        replaces the wavelength-bound comparison (see
        :func:`static_region_masks`).
    group_matrix : array-like, optional
        Static ``(n_groups, n_regions)`` 0/1 matrix assigning regions to
        zorder groups (see ``ModelArgs.cont_group_matrix``).  When given, the
        regions are summed within each group instead of all together.

    Returns
    -------
    jnp.ndarray
        Continuum flux density at each wavelength point (unscaled — before
        ``continuum_scale`` and normalisation are applied).  Shape
        ``(n_points,)``, or ``(n_groups, n_points)`` when *group_matrix* is
        given.
    """
    regions = (
        eval_continuum_regions(wavelength, args, context, z_sys, lsf_fwhm, region_mask)
        if args.cont_config is not None
        else []
    )
    if group_matrix is None:
        continuum = jnp.zeros_like(wavelength)
        for region_cont in regions:
            continuum = continuum + region_cont
        return continuum
    return _group_sum(regions, group_matrix, wavelength.shape)


def eval_continuum_regions(
    wavelength,
    args,
    context,
    z_sys,
    lsf_fwhm: float | jnp.ndarray = 0.0,
    region_mask=None,
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
    region_mask : jnp.ndarray of bool, optional
        Static membership, shape ``(n_regions, n_points)``.  When given it
        replaces the wavelength-bound comparison (see
        :func:`static_region_masks`).

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
            if region_mask is None:
                in_region = (wavelength >= obs_low) & (wavelength <= obs_high)
            else:
                in_region = region_mask[k]
            region_cont = args.cont_forms[k].evaluate(
                wavelength, obs_center, cont_params, obs_low, obs_high, lsf_fwhm, z_sys
            )
            regions.append(jnp.where(in_region, region_cont, 0.0))
    return regions


def integrate_continuum_regions(
    edges,
    args,
    context,
    z_sys,
    lsf_fwhm_per_region: float | jnp.ndarray = 0.0,
    region_mask=None,
    lsf_fwhm_per_interval=None,
):
    """Per-region pixel-averaged continuum on the unique-edge topology.

    This is the single analytic-mode continuum evaluation shared by the
    likelihood (:func:`integrate_continuum`) and ``evaluate_model``, so a
    prediction is exactly the model that was fitted.

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
        LSF FWHM scalar per continuum region, used by forms that do not set
        ``supports_interval_lsf``.  Scalar broadcasts to all regions.
        Default ``0.0``.
    region_mask : jnp.ndarray of bool, optional
        Static membership, shape ``(n_regions, E - 1)``; replaces the
        comparison of interval midpoints with the region bounds (see
        :func:`static_region_masks`).
    lsf_fwhm_per_interval : jnp.ndarray, shape ``(E - 1,)``, optional
        LSF FWHM for each edge interval (pixel).  Forms with
        ``supports_interval_lsf`` (the polynomial family) use it in place of
        the per-region scalar.

    Returns
    -------
    list of jnp.ndarray
        One length-``(E - 1)`` array per region (unscaled), zero outside the
        region.  Entries spanning inter-pixel gaps must be discarded by the
        caller via the spectrum's ``keep_mask``.
    """
    edges = jnp.asarray(edges)
    widths = jnp.diff(edges)
    mids = 0.5 * (edges[1:] + edges[:-1])
    per_region: list[jnp.ndarray] = []
    if args.cont_config is not None:
        lsf_arr = jnp.atleast_1d(jnp.asarray(lsf_fwhm_per_region))
        if lsf_arr.shape[0] == 1:
            lsf_arr = jnp.broadcast_to(lsf_arr, (len(args.cont_config),))
        for k in range(len(args.cont_config)):
            obs_low, obs_high, obs_center, cont_params = _resolve_region_params(
                args, context, z_sys, k
            )
            if region_mask is None:
                in_region = (mids >= obs_low) & (mids <= obs_high)
            else:
                in_region = region_mask[k]
            form = args.cont_forms[k]
            lsf_k = (
                lsf_fwhm_per_interval
                if lsf_fwhm_per_interval is not None and form.supports_interval_lsf
                else lsf_arr[k]
            )
            cum = form.integrate(
                edges, obs_center, cont_params, obs_low, obs_high, lsf_k, z_sys
            )
            per_pixel_avg = jnp.diff(cum) / widths
            per_region.append(jnp.where(in_region, per_pixel_avg, 0.0))
    return per_region


def interval_lsf_fwhm(lsf_at_edges):
    """LSF FWHM of each edge interval (pixel), from the per-edge LSF.

    Parameters
    ----------
    lsf_at_edges : jnp.ndarray, shape ``(E,)``
        LSF FWHM at every unique pixel edge.

    Returns
    -------
    jnp.ndarray, shape ``(E - 1,)``
        Mean of the two bounding edges.
    """
    return 0.5 * (lsf_at_edges[1:] + lsf_at_edges[:-1])


def region_lsf_fwhm(args, disp, z_sys, inv_wl_scale, r_scale):
    """LSF FWHM at each continuum region centre (observed frame).

    Used for continuum forms that take one LSF per region.

    Parameters
    ----------
    args : ModelArgs
        Pre-built data bundle.
    disp : Disperser
        Disperser of the spectrum.
    z_sys : float
        Systemic redshift.
    inv_wl_scale : float
        Conversion from canonical wavelength to the disperser's unit.
    r_scale : float or jnp.ndarray
        Resolving-power scale.

    Returns
    -------
    jnp.ndarray, shape ``(n_regions,)``
        FWHM per region (canonical unit); an empty array without continuum.
    """
    if args.cont_config is None or args.cont_center is None:
        return jnp.zeros(0)
    centers_obs = jnp.array(args.cont_center) * (1.0 + z_sys)
    return centers_obs / (disp.R(centers_obs * inv_wl_scale) * r_scale)


def integrate_continuum(
    edges,
    args,
    context,
    z_sys,
    lsf_fwhm_per_region: float | jnp.ndarray = 0.0,
    region_mask=None,
    group_matrix=None,
    lsf_fwhm_per_interval=None,
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
    region_mask : jnp.ndarray of bool, optional
        Static membership, shape ``(n_regions, E - 1)``.  When given it
        replaces the comparison of interval midpoints with the region
        bounds (see :func:`static_region_masks`); required for a
        discontinuity-free dependence on ``PixOffset``.
    group_matrix : array-like, optional
        Static ``(n_groups, n_regions)`` 0/1 matrix assigning regions to
        zorder groups.  When given, regions are summed within each group.
    lsf_fwhm_per_interval : jnp.ndarray, shape ``(E - 1,)``, optional
        LSF FWHM for each edge interval (pixel).  Forms with
        ``supports_interval_lsf`` use it in place of the per-region scalar.
        Other forms still receive *lsf_fwhm_per_region*.

    Returns
    -------
    jnp.ndarray, shape ``(E - 1,)`` or ``(n_groups, E - 1)``
        Pixel-averaged continuum (unscaled) across all ``E - 1`` edge
        intervals (one row per zorder group when *group_matrix* is given).
        Entries that span inter-pixel gaps must be discarded by the caller
        via the spectrum's ``keep_mask``.
    """
    edges = jnp.asarray(edges)
    per_region = integrate_continuum_regions(
        edges,
        args,
        context,
        z_sys,
        lsf_fwhm_per_region,
        region_mask,
        lsf_fwhm_per_interval,
    )
    if group_matrix is None:
        total = jnp.zeros(edges.shape[0] - 1)
        for r in per_region:
            total = total + r
        return total
    return _group_sum(per_region, group_matrix, (edges.shape[0] - 1,))
