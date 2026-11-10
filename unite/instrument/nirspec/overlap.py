r"""Higher-order grating overlap for NIRSpec from msaexp sensitivity curves.

With the F070LP filter, G140M/H spectra contain second-order light redwards
of roughly 1.3 µm (twice the filter cut-on); with F170LP, G235M/H contain
second-order light redwards of roughly 3.3 µm.  Third-order light starts at
three times the cut-on and only reaches the detector in extended extractions.

The `msaexp <https://github.com/gbrammer/msaexp>`_ pipeline ships empirical
sensitivity curves (``msaexp/data/extended_sensitivity/
msaexp_sensitivity_{grating}_{filter}_001.fits``) with columns ``wavelength``
(µm), ``sensitivity`` (first order) and ``sensitivity_2``, ``sensitivity_3``
(higher orders), all in count rate per µJy.  This module turns such a table
into the per-order throughput-ratio callables expected by the
``order_overlap`` argument of :class:`~unite.instrument.base.Disperser`.

Convention
----------
msaexp tabulates ``sensitivity_m`` against the **intrinsic** (true)
wavelength ``λ_1`` of the photons; order-*m* light of wavelength ``λ_1``
lands on the pixel whose first-order wavelength is ``λ = m λ_1``.  The
pipeline divides the total count rate by the first-order sensitivity at the
observed wavelength, so in :math:`f_\nu` the calibrated spectrum is

.. math::

    F_\nu^{\rm obs}(\lambda) = F_\nu(\lambda)
      + \sum_{m \ge 2}
        \frac{S_m(\lambda / m)}{S_1(\lambda)}\, F_\nu(\lambda / m),

with the dispersion (pixel width) of order *m* already folded into the
empirical ``S_m``.  ``unite`` works in :math:`f_\lambda`; converting both
sides with :math:`f_\lambda = f_\nu c / \lambda^2` gives

.. math::

    F_\lambda^{\rm obs}(\lambda) = F_\lambda(\lambda)
      + \sum_{m \ge 2} r_m(\lambda)\, F_\lambda(\lambda / m),
    \qquad
    r_m(\lambda) = \frac{S_m(\lambda / m)}{m^2\, S_1(\lambda)} .

:func:`order_overlap_from_msaexp` returns exactly these :math:`r_m(\lambda)`
as functions of the observed wavelength in µm, zero wherever either
sensitivity is untabulated or non-positive.

Scope
-----
Only the parsing of the msaexp table is specific to NIRSpec: the factor
:math:`m^{-2}` follows from the :math:`f_\nu` to :math:`f_\lambda`
conversion alone, and :class:`SensitivityRatio` accepts any tabulated ratio.
For another instrument, derive :math:`S_m` from a blue spectrophotometric
standard observed in the same configuration.  NIRSpec needs no atmospheric
term; for ground-based spectra see the Notes of
:class:`~unite.instrument.base.Disperser`, which also list the instruments
to which this description of the overlap does and does not apply.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from astropy.table import Table
from jax.typing import ArrayLike

__all__ = ['SensitivityRatio', 'order_overlap_from_msaexp']


class SensitivityRatio:
    """Tabulated order-*m* to first-order throughput ratio, ``r_m(λ)``.

    A JAX-jittable callable that linearly interpolates a tabulated ratio
    curve in observed wavelength and returns zero outside the tabulated
    range.  Instances are produced by :func:`order_overlap_from_msaexp`; the
    class can also be constructed directly from arrays.

    Parameters
    ----------
    order : int
        Diffraction order *m* (``>= 2``).
    wavelength : array_like, shape ``(N,)``
        Observed (first-order-calibrated) wavelengths at which *ratio* is
        tabulated, strictly increasing.
    ratio : array_like, shape ``(N,)``
        ``r_m`` at each wavelength (``f_lambda`` convention).
    unit : str, optional
        Human-readable wavelength unit label, for ``repr`` only.
    """

    def __init__(
        self, order: int, wavelength: ArrayLike, ratio: ArrayLike, *, unit: str = 'um'
    ) -> None:
        wl = np.asarray(wavelength, dtype=float)
        r = np.asarray(ratio, dtype=float)
        if wl.ndim != 1 or wl.shape != r.shape:
            msg = 'wavelength and ratio must be 1-D arrays of equal length'
            raise ValueError(msg)
        if wl.size < 2 or np.any(np.diff(wl) <= 0):
            msg = 'wavelength must be strictly increasing with at least two entries'
            raise ValueError(msg)
        self.order: int = int(order)
        self.wavelength: jnp.ndarray = jnp.asarray(wl)
        self.ratio: jnp.ndarray = jnp.asarray(np.where(np.isfinite(r), r, 0.0))
        self.unit: str = unit

    def __call__(self, wavelength: ArrayLike) -> jnp.ndarray:
        """Interpolate ``r_m`` at *wavelength* (zero outside the table)."""
        return jnp.interp(
            jnp.asarray(wavelength), self.wavelength, self.ratio, left=0.0, right=0.0
        )

    @property
    def support(self) -> tuple[float, float]:
        """Observed-wavelength range ``(low, high)`` where ``r_m > 0``."""
        pos = np.asarray(self.ratio) > 0
        if not pos.any():
            return (float('nan'), float('nan'))
        wl = np.asarray(self.wavelength)[pos]
        return (float(wl.min()), float(wl.max()))

    def __repr__(self) -> str:
        lo, hi = self.support
        return (
            f'SensitivityRatio(order={self.order}, '
            f'support=({lo:.4g}, {hi:.4g}) {self.unit}, '
            f'max={float(jnp.max(self.ratio)):.3g})'
        )


def order_overlap_from_msaexp(
    table: str | Path | Table,
    orders: Sequence[int] = (2, 3),
    *,
    strict: bool = False,
    min_ratio: float = 0.0,
    wavelength_range: tuple[float, float] | None = None,
) -> dict[int, SensitivityRatio]:
    """Build ``order_overlap`` ratio callables from an msaexp sensitivity table.

    Parameters
    ----------
    table : str, Path or astropy.table.Table
        An msaexp extended-sensitivity table (e.g.
        ``msaexp_sensitivity_g140m_f070lp_001.fits``) or a path to one.
        Must contain the columns ``wavelength`` (µm), ``sensitivity`` and
        ``sensitivity_{m}`` for every requested order *m*.  Files are read
        with :meth:`astropy.table.Table.read`.
    orders : sequence of int, optional
        Diffraction orders to include (default ``(2, 3)``).  An order whose
        column is absent, or whose ratio is zero everywhere, is skipped
        unless *strict* is ``True``.
    strict : bool, optional
        Raise instead of skipping missing or all-zero orders.
        Default ``False``.
    min_ratio : float, optional
        Skip an order whose ratio never exceeds this value (within
        *wavelength_range*), as its contribution is negligible while each
        order adds a model evaluation.  Default ``0.0`` (keep every order with
        a positive ratio).
    wavelength_range : tuple of float, optional
        Observed wavelength range (µm) of the spectra over which *min_ratio*
        is assessed (the ratios themselves are not truncated).  The tables
        extend far beyond the detector, where the ratio can be large as the
        first-order sensitivity vanishes: the third order of G140M/F070LP, for
        instance, only exceeds 1% beyond 2.4 µm, outside the extracted
        spectra.  Default ``None`` (the full tabulated range).

    Returns
    -------
    dict of int to SensitivityRatio
        ``{m: r_m}`` suitable for the ``order_overlap`` argument of any
        NIRSpec disperser.  ``r_m`` is evaluated at the **observed**
        wavelength in µm and follows the ``f_lambda`` convention described
        in the module docstring, i.e. ``r_m(λ) = S_m(λ/m) / (m² S_1(λ))``.
        Empty when no requested order is present.

    Raises
    ------
    KeyError
        If the ``wavelength`` or ``sensitivity`` column is missing, or (with
        ``strict=True``) a requested ``sensitivity_{m}`` column is missing.
    ValueError
        If an order is smaller than 2, or (with ``strict=True``) a requested
        order has no positive ratio anywhere.

    Notes
    -----
    The tables are not bundled with ``unite``.  The returned ratios are
    fixed: the uncertainty on the msaexp sensitivity curves is not
    propagated into the model.

    Examples
    --------
    >>> from unite.instrument import nirspec
    >>> overlap = nirspec.order_overlap_from_msaexp(
    ...     'msaexp_sensitivity_g140m_f070lp_001.fits'
    ... )
    >>> g140m = nirspec.G140M(order_overlap=overlap)
    """
    if not isinstance(table, Table):
        table = Table.read(table)
    for col in ('wavelength', 'sensitivity'):
        if col not in table.colnames:
            msg = f'sensitivity table is missing the {col!r} column'
            raise KeyError(msg)

    wl = np.asarray(table['wavelength'], dtype=float)
    s1 = np.asarray(table['sensitivity'], dtype=float)
    s1 = np.where(np.isfinite(s1), s1, 0.0)
    # Keep a clean, sorted first-order grid (interp requires increasing x).
    order_idx = np.argsort(wl)
    wl, s1 = wl[order_idx], s1[order_idx]
    good = np.isfinite(wl)
    wl, s1 = wl[good], s1[good]

    out: dict[int, SensitivityRatio] = {}
    for m in orders:
        if isinstance(m, bool) or int(m) != m or m < 2:
            msg = f'orders must be integers >= 2, got {m!r}'
            raise ValueError(msg)
        m = int(m)
        col = f'sensitivity_{m}'
        if col not in table.colnames:
            if strict:
                msg = f'sensitivity table is missing the {col!r} column'
                raise KeyError(msg)
            continue
        sm = np.asarray(table[col], dtype=float)[order_idx][good]
        sm = np.where(np.isfinite(sm), sm, 0.0)
        # Order-m light of intrinsic wavelength wl lands at observed m * wl.
        wl_obs = m * wl
        s1_obs = np.interp(wl_obs, wl, s1, left=0.0, right=0.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where((s1_obs > 0) & (sm > 0), sm / (m**2 * s1_obs), 0.0)
        if not np.any(ratio > 0):
            if strict:
                msg = f'order {m} ratio is zero everywhere in the sensitivity table'
                raise ValueError(msg)
            continue
        in_range = (
            np.ones(wl_obs.shape, dtype=bool)
            if wavelength_range is None
            else (wl_obs >= wavelength_range[0]) & (wl_obs <= wavelength_range[1])
        )
        if not np.any(ratio[in_range] > min_ratio):
            continue
        out[m] = SensitivityRatio(m, wl_obs, ratio, unit='um')
    return out
