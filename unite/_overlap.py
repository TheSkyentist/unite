"""Higher-order grating overlap: static windows and per-order evaluation.

Shared by :func:`unite.model.unite_model` and
:func:`unite.compute.evaluate_model`, so a prediction is exactly the model
that was fitted.  For each order *m* of a spectrum's disperser the overlap
term on a pixel of (true, offset-corrected) observed wavelength ``λ`` is

.. code-block:: text

    OrderScale_m * r_m(λ) * F(λ / m)

where ``F`` is the full source model.  It is evaluated only over a static
*window* of pixels where ``r_m`` can be non-zero (:class:`OverlapWindow`),
built once by :meth:`~unite.model.ModelBuilder.build`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np

from unite._lsf import _FWHM_TO_SIGMA, _lsf_convolve
from unite.continuum.compute import (
    integrate_continuum,
    interval_lsf_fwhm,
    region_lsf_fwhm,
    static_region_masks,
)
from unite.instrument.overlap import OrderOverlap
from unite.prior import Fixed, Prior


@dataclass(frozen=True, eq=False)
class OverlapWindow:
    """Static evaluation window of one diffraction order on one spectrum.

    All indices refer to the (trimmed) spectrum the window was built for.
    Pixels are the spectrum's kept pixels; intervals are its unique-edge
    intervals (``E - 1``, including inter-pixel gaps).
    """

    #: Diffraction order *m*.
    order: int
    #: The tabulated ratio.
    overlap: OrderOverlap
    #: Numpyro site name of the order's :class:`OrderScale`.
    scale_name: str
    #: Kept-pixel range ``[pix0, pix0 + n_pix)`` of the window.
    pix0: int
    n_pix: int
    #: Edge-interval range ``[i0, i1)`` spanning the window's pixels; the
    #: window's edges are ``edges[i0 : i1 + 1]``.
    i0: int
    i1: int
    #: ``keep_mask[i0:i1]`` (numpy, static).
    keep: np.ndarray = field(repr=False)
    #: Static continuum-region membership of the order's photon intervals,
    #: from the *unshifted* edges divided by *m*, shape ``(n_regions, i1 - i0)``.
    region_mask: np.ndarray = field(repr=False)
    #: Convolution mode only: kept-pixel range ``[q0, q1)`` (the window widened
    #: by the kernel reach) and the matching fine-grid region membership,
    #: shape ``(n_regions, (q1 - q0) * n_super)``.
    q0: int = 0
    q1: int = 0
    fine_region_mask: np.ndarray | None = field(default=None, repr=False)

    @property
    def wavelength_slice(self) -> slice:
        """Slice of the kept pixels covered by the window."""
        return slice(self.pix0, self.pix0 + self.n_pix)


#: Errors meaning a prior's support has no finite numeric bounds.  (A named
#: tuple: ruff's py314 target would strip the parentheses of an inline one.)
_BOUND_ERRORS = (AttributeError, NotImplementedError, TypeError)


def pix_offset_margin(prior: Prior) -> int | None:
    """Window margin in pixels implied by the bounds of a ``PixOffset`` prior.

    Returns ``ceil(max |bound|) + 1`` (one extra pixel for the linear
    interpolation of ``r_m``), or ``None`` when the prior is unbounded or its
    bounds depend on other parameters, meaning the full spectrum must be
    evaluated.

    Parameters
    ----------
    prior : Prior
        Prior of the disperser's ``PixOffset`` token.

    Returns
    -------
    int or None
    """
    if prior.dependencies():
        return None
    if isinstance(prior, Fixed):
        bound = abs(float(prior.resolved_value({})))
    else:
        try:
            distribution = prior.to_dist({})
            if distribution is None:
                return None
            support = distribution.support
            lo = float(support.lower_bound)  # ty: ignore[unresolved-attribute]
            hi = float(support.upper_bound)  # ty: ignore[unresolved-attribute]
        except _BOUND_ERRORS:
            return None
        if not (math.isfinite(lo) and math.isfinite(hi)):
            return None
        bound = max(abs(lo), abs(hi))
    return math.ceil(bound) + 1


def build_windows(
    spectrum,
    canonical_unit,
    wl_scale: float,
    name_of_scale: Callable[[OrderOverlap], str],
    cont_low,
    cont_high,
    z_sys: float,
) -> list[OverlapWindow]:
    """Build the static analytic-mode windows of every order of a spectrum.

    Parameters
    ----------
    spectrum : Spectrum
        Trimmed spectrum.
    canonical_unit : astropy.units.UnitBase
        Canonical wavelength unit of the model.
    wl_scale : float
        Conversion from the spectrum's unit to *canonical_unit*.
    name_of_scale : callable
        Returns the site name of an overlap's ``OrderScale``.
    cont_low, cont_high : list of float or None
        Rest-frame region bounds (canonical unit).
    z_sys : float
        Systemic redshift.

    Returns
    -------
    list of OverlapWindow
        One per order with at least one active pixel, sorted by order.
        Orders with no active pixel are omitted.
    """
    disp = spectrum.disperser
    if not disp.order_overlap:
        return []
    edges = np.asarray(spectrum.edges, dtype=float) * wl_scale  # unshifted
    keep = np.asarray(spectrum.keep_mask, dtype=bool)
    mids = 0.5 * (edges[1:] + edges[:-1])
    kept_idx = np.flatnonzero(keep)
    pix_mids = mids[kept_idx]
    n_pixels = pix_mids.size
    margin = pix_offset_margin(disp.pix_offset.prior)

    windows: list[OverlapWindow] = []
    for ov in disp.order_overlap:
        m = ov.order
        bounds = ov.support_bounds(canonical_unit)
        if bounds is None:
            continue
        lo, hi = bounds
        active = np.flatnonzero((pix_mids >= lo) & (pix_mids <= hi))
        if margin is None:
            # Unbounded PixOffset: any pixel may become active.
            p0, p1 = 0, n_pixels
        elif active.size == 0:
            # The table may still reach the spectrum edge after a shift.
            edge = int(np.searchsorted(pix_mids, lo))
            p0, p1 = max(edge - margin, 0), min(edge + margin, n_pixels)
        else:
            p0 = max(int(active[0]) - margin, 0)
            p1 = min(int(active[-1]) + 1 + margin, n_pixels)
        if p1 <= p0:
            continue
        i0 = int(kept_idx[p0])
        i1 = int(kept_idx[p1 - 1]) + 1
        region_mask = static_region_masks(
            edges[i0 : i1 + 1] / m, cont_low, cont_high, z_sys
        )
        windows.append(
            OverlapWindow(
                order=m,
                overlap=ov,
                scale_name=name_of_scale(ov),
                pix0=p0,
                n_pix=p1 - p0,
                i0=i0,
                i1=i1,
                keep=keep[i0:i1].copy(),
                region_mask=region_mask,
            )
        )
    return windows


def with_convolution(
    window: OverlapWindow,
    spectrum,
    wl_scale,
    cont_low,
    cont_high,
    z_sys,
    n_super,
    half_width,
) -> OverlapWindow:
    """Return *window* widened by the convolution kernel reach (convolution mode).

    The order-*m* product ``r_m(x) F(x / m)`` is zero outside the window,
    so convolving over the window widened by the kernel half-width (in
    pixels) gives the exact result on every pixel it can reach.
    """
    from dataclasses import replace

    edges = np.asarray(spectrum.edges, dtype=float) * wl_scale
    keep = np.asarray(spectrum.keep_mask, dtype=bool)
    n_pixels = int(keep.sum())
    reach = math.ceil(half_width / n_super) + 1
    q0 = max(window.pix0 - reach, 0)
    q1 = min(window.pix0 + window.n_pix + reach, n_pixels)
    full_mask = static_region_masks(edges / window.order, cont_low, cont_high, z_sys)
    fine = np.repeat(full_mask[:, keep][:, q0:q1], n_super, axis=1)
    return replace(window, q0=q0, q1=q1, fine_region_mask=fine)


def order_lsf_fwhm(photon_wl, order: int, disp, r_scale, inv_wl_scale):
    """LSF FWHM of order-*m* light in photon wavelength.

    The order-*m* image has resolving power ``R_m(λ) = m R(λ)`` with ``R``
    (the first-order resolving power, scaled by ``r_scale``) evaluated at
    the photon wavelength ``λ``, so the FWHM is ``λ / (m R(λ))``.  On the
    observed axis it is *m* times larger.

    Parameters
    ----------
    photon_wl : array_like
        Photon wavelength (canonical unit).
    order : int
        Diffraction order *m*.
    disp : Disperser
        Disperser of the spectrum.
    r_scale : array_like
        Resolving-power scale (may be traced).
    inv_wl_scale : float
        Conversion from the canonical unit to the disperser's unit.

    Returns
    -------
    jnp.ndarray
        FWHM in photon wavelength (canonical unit).
    """
    photon_wl = jnp.asarray(photon_wl)
    return photon_wl / (order * disp.R(photon_wl * inv_wl_scale) * r_scale)


def analytic_source(
    window: OverlapWindow,
    edges_true,
    *,
    disp,
    r_scale,
    inv_wl_scale,
    args,
    context,
    z_sys,
    integrate_lines_fn,
    line_params,
    compose: Callable,
):
    """Pixel-averaged ``F(λ / m)`` on the window's pixels (analytic mode).

    Parameters
    ----------
    window : OverlapWindow
        The order's static window.
    edges_true : jnp.ndarray, shape ``(E,)``
        Offset-corrected unique edges (canonical unit) of the spectrum.
    disp, r_scale, inv_wl_scale
        Disperser, resolving-power scale and canonical-to-disperser unit
        factor.
    args : ModelArgs
        Pre-built data bundle.
    context : dict
        Parameter values by site name.
    z_sys : float
        Systemic redshift.
    integrate_lines_fn : callable
        Grouped line-integration function.
    line_params : tuple
        ``(centers, p0, p1, p2, profile_codes)``.
    compose : callable
        ``compose(pixints, cont) -> model`` combining line profiles
        ``(n_lines, n_pix)`` and grouped continuum ``(G, n_pix)``.

    Returns
    -------
    jnp.ndarray, shape ``(n_pix,)``
        The composed source model on the window (before ``r_m``,
        ``OrderScale`` and ``FluxScale``).
    """
    m = window.order
    centers, p0, p1, p2, pcodes = line_params
    e_m = edges_true[window.i0 : window.i1 + 1] / m  # photon-wavelength edges
    lsf_m = order_lsf_fwhm(e_m, m, disp, r_scale, inv_wl_scale)
    keep = window.keep
    cum = integrate_lines_fn(e_m, centers, lsf_m, p0, p1, p2, pcodes)
    pixints = (jnp.diff(cum, axis=1) / jnp.diff(e_m))[:, keep]
    cont = integrate_continuum(
        e_m,
        args,
        context,
        z_sys,
        region_lsf_fwhm(args, disp, z_sys, inv_wl_scale, r_scale, order=m),
        window.region_mask,
        args.cont_group_matrix,
        lsf_fwhm_per_interval=interval_lsf_fwhm(lsf_m),
    )[:, keep]
    return compose(pixints, cont)


def ratio_at_pixels(window: OverlapWindow, edges_true, canonical_unit):
    """``r_m`` at the window's offset-corrected pixel midpoints."""
    e = edges_true[window.i0 : window.i1 + 1]
    mids = (0.5 * (e[1:] + e[:-1]))[window.keep]
    return window.overlap.evaluate(mids, canonical_unit)


def convolved_term(
    window: OverlapWindow,
    x_flat,
    *,
    n_super: int,
    half_width: int,
    disp,
    r_scale,
    inv_wl_scale,
    canonical_unit,
    source_fine: Callable,
):
    """Convolved, pixel-averaged ``r_m(x) F(x / m)`` on pixels ``[q0, q1)``.

    Parameters
    ----------
    window : OverlapWindow
        The order's window, widened by :func:`with_convolution`.
    x_flat : jnp.ndarray
        Full fine grid of the spectrum (offset-corrected observed
        wavelength, canonical unit), pixel-major.
    n_super, half_width : int
        Fine-grid density and kernel half-width (fine indices).
    disp, r_scale, inv_wl_scale, canonical_unit
        As for :func:`analytic_source`.
    source_fine : callable
        ``source_fine(photon_wl, fine_region_mask) -> F`` evaluating the
        intrinsic (zero-LSF) composed source model.

    Returns
    -------
    jnp.ndarray, shape ``(q1 - q0,)``
        Pixel-averaged convolved term (before ``OrderScale`` and
        ``FluxScale``).
    """
    m = window.order
    x = x_flat[window.q0 * n_super : window.q1 * n_super]
    photon = x / m
    product = window.overlap.evaluate(x, canonical_unit) * source_fine(
        photon, window.fine_region_mask
    )
    # On the observed axis the order-m LSF is m times its photon-axis FWHM.
    sigma = m * order_lsf_fwhm(photon, m, disp, r_scale, inv_wl_scale) * _FWHM_TO_SIGMA
    conv = _lsf_convolve(x, product, sigma, half_width)
    return conv.reshape(window.q1 - window.q0, n_super).mean(axis=1)


def max_observed_lsf_fwhm(window: OverlapWindow, spectrum, wl_scale) -> float:
    """Largest observed-axis LSF FWHM of the order over the window (r_scale=1)."""
    m = window.order
    disp = spectrum.disperser
    mids = np.asarray((spectrum.low + spectrum.high) / 2.0)[window.wavelength_slice]
    photon_disp = mids / m  # disperser unit
    r = np.asarray(disp.R(jnp.asarray(photon_disp)))
    return float(np.max(photon_disp * wl_scale / r)) if mids.size else 0.0


def build_warnings(
    spectrum, windows: list[OverlapWindow], untrimmed, canonical_unit, wl_scale: float
) -> list[str]:
    """Diagnostic messages for the overlap of one trimmed spectrum.

    Checks, per order of the spectrum's disperser:

    * the order has no active pixel (``r_m > 0`` on no fitted pixel);
    * its source window (the photon wavelengths ``λ / m`` of the active
      pixels) has first-order data in some **untrimmed** spectrum but no
      continuum region, or is only partly covered by the continuum regions;
    * the photon wavelengths fall outside the disperser's calibrated range
      (only where the disperser exposes one).

    Parameters
    ----------
    spectrum : Spectrum
        Trimmed spectrum.
    windows : list of OverlapWindow
        Its windows from :func:`build_windows`.
    untrimmed : iterable of Spectrum
        All spectra before trimming (first-order data coverage).
    canonical_unit : astropy.units.UnitBase
        Canonical wavelength unit.
    wl_scale : float
        Conversion from the spectrum's unit to *canonical_unit*.

    Returns
    -------
    list of str
        One message per problem found.
    """
    from unite._utils import _get_conversion_factor

    disp = spectrum.disperser
    msgs: list[str] = []
    by_order = {w.order: w for w in windows}
    edges = np.asarray(spectrum.edges, dtype=float) * wl_scale
    mids_all = 0.5 * (edges[1:] + edges[:-1])
    label = f'Spectrum {spectrum.name!r}'
    for ov in disp.order_overlap:
        m = ov.order
        win = by_order.get(m)
        if win is not None:
            mids = mids_all[win.i0 : win.i1][win.keep]
            active = np.asarray(ov.evaluate(mids, canonical_unit)) > 0
        else:
            mids, active = np.zeros(0), np.zeros(0, dtype=bool)
        if not active.any():
            msg = (
                f'{label}: order {m} overlap has no active pixels '
                '(r_m is zero on every fitted pixel).'
            )
            if not isinstance(ov.scale.prior, Fixed):
                msg += (
                    f' Its OrderScale {ov.scale.name!r} is then constrained only '
                    'by its prior.'
                )
            msgs.append(msg)
            continue

        assert win is not None
        photon = mids[active] / m  # canonical unit
        lo, hi = float(photon.min()), float(photon.max())

        # First-order data in the source window, from the untrimmed spectra.
        has_data = False
        for s in untrimmed:
            conv = _get_conversion_factor(s.unit, canonical_unit)
            wl = np.asarray(s.wavelength) * conv
            if np.any((wl >= lo) & (wl <= hi)):
                has_data = True
                break
        if has_data:
            in_region = win.region_mask[:, win.keep][:, active].any(axis=0)
            if not in_region.any():
                msgs.append(
                    f'{label}: the order {m} source window '
                    f'[{lo:.6g}, {hi:.6g}] {canonical_unit} has first-order data '
                    'but no continuum region; only lines contribute to the '
                    'overlap there.'
                )
            elif not in_region.all():
                msgs.append(
                    f'{label}: a continuum region edge falls inside the order {m} '
                    f'source window [{lo:.6g}, {hi:.6g}] {canonical_unit}; the '
                    'overlap continuum switches off partway.'
                )

        # R evaluated at the photon wavelengths, outside its calibration.
        cal = disp.calibrated_range
        if cal is not None:
            inv = 1.0 / _get_conversion_factor(disp.unit, canonical_unit)
            p_lo, p_hi = lo * inv, hi * inv
            if p_lo < cal[0] or p_hi > cal[1]:
                msgs.append(
                    f'{label}: order {m} photon wavelengths [{p_lo:.6g}, '
                    f'{p_hi:.6g}] {disp.unit} extend outside the calibrated '
                    f'range [{cal[0]:.6g}, {cal[1]:.6g}] of {type(disp).__name__}; '
                    'its resolving power is extrapolated there.'
                )
    return msgs
