"""Higher-order grating overlap: order-*m* light of the same source.

A grating covering more than an octave behind its order-sorting filter
receives order-*m* light (``m >= 2``) on top of its first-order spectrum:
photons of wavelength ``λ / m`` land on the pixel whose first-order wavelength
is ``λ``.  :class:`OrderOverlap` describes one such order with a tabulated
throughput ratio ``r_m(λ)``; attach it to a disperser with
``order_overlap=[...]``.  The model then becomes

.. code-block:: text

    total(λ) = FluxScale * [ F(λ) + Σ_m OrderScale_m * r_m(λ) * F(λ / m) ]

where ``F`` is the full source model (lines, absorbers, continuum).
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from astropy import units as u
from jax.typing import ArrayLike

from unite.instrument.base import OrderScale

#: Default name of the throughput-ratio column.
RATIO_COLUMN = 'throughput_ratio'


class OrderOverlap:
    """Tabulated throughput ratio of one higher diffraction order.

    The file holds the order-*m* to first-order throughput ratio ``r_m`` in the
    **f_lambda convention**, indexed by **observed** wavelength (the
    first-order-calibrated wavelength of the pixel the light lands on, not the
    wavelength of the photons).  It is relative to the nominal first-order
    throughput, i.e. before any :class:`~unite.instrument.base.FluxScale`.
    Outside the tabulated range the contribution is exactly zero.

    Parameters
    ----------
    order : int
        Diffraction order *m* (``>= 2``).
    path : str or Path
        File readable by :meth:`astropy.table.Table.read` (FITS, ECSV, …).
        Stored as given, so a relative path resolves against the working
        directory whenever the file is read (as for
        :class:`~unite.continuum.Template`).  The file must contain:

        * one wavelength column, identified by its **length unit** and read
          as observed wavelength (strictly increasing);
        * a dimensionless ratio column, named ``throughput_ratio`` by
          default.
    column : str, optional
        Name of the ratio column.  When omitted, ``throughput_ratio`` is used
        if present; otherwise, if the file has exactly one column besides the
        wavelength, that column is used.
    scale : OrderScale, optional
        Multiplicative token on ``r_m`` (the calibration uncertainty of the
        curve).  ``None`` (default) creates a private token fixed at 1.0.
        Pass the same instance to several overlaps to share one parameter.

    Raises
    ------
    TypeError
        If *order* is not an integer or *scale* is not an
        :class:`~unite.instrument.base.OrderScale`.
    ValueError
        If ``order < 2``; the wavelength or ratio column is missing or
        ambiguous; the ratio column is not dimensionless (a flux-unit column
        is rejected so an absolute sensitivity cannot be passed by mistake);
        the wavelength is not strictly increasing; or any value is not finite.

    Notes
    -----
    msaexp-style sensitivity curves ``S_m`` (count rate per unit f_nu,
    tabulated against photon wavelength) convert to this convention as
    ``r_m(λ) = S_m(λ / m) / (m² S_1(λ))``, resampled onto observed wavelength
    ``λ``.  For ground-based data, fold the telluric/extinction factor
    ``T(λ / m) / T(λ)`` into the ratio.

    Examples
    --------
    >>> from unite.instrument import OrderOverlap, OrderScale
    >>> from unite.instrument.nirspec import G140M
    >>> from unite.prior import TruncatedNormal
    >>> s2 = OrderScale(prior=TruncatedNormal(1.0, 0.2, 0.5, 1.5))
    >>> g140m = G140M(order_overlap=[OrderOverlap(2, 'g140m_order2.fits', scale=s2)])
    """

    def __init__(
        self,
        order: int,
        path: str | Path,
        *,
        column: str | None = None,
        scale: OrderScale | None = None,
    ) -> None:
        if isinstance(order, bool) or not isinstance(order, int | np.integer):
            msg = f'order must be an integer, got {order!r}'
            raise TypeError(msg)
        if order < 2:
            msg = f'order must be >= 2, got {order}'
            raise ValueError(msg)
        if scale is not None and not isinstance(scale, OrderScale):
            msg = f'scale must be an OrderScale token, got {type(scale).__name__}'
            raise TypeError(msg)

        self.order: int = int(order)
        self.path: Path = Path(path)
        self.scale: OrderScale = scale if scale is not None else OrderScale()
        self._column_arg = column

        wl, ratio, col = _read_ratio_table(self.path, column)
        self.column: str = col
        #: Tabulated observed wavelengths (Quantity, strictly increasing).
        self.wavelength: u.Quantity = wl
        #: Tabulated ratio ``r_m`` (dimensionless).
        self.ratio: np.ndarray = ratio

    # -- evaluation ----------------------------------------------------------

    def evaluate(self, wavelength: ArrayLike, unit: u.UnitBase) -> jnp.ndarray:
        """Interpolate ``r_m`` at observed *wavelength* (zero outside the table).

        Parameters
        ----------
        wavelength : array_like
            Observed wavelengths, in *unit*.  May be traced.
        unit : astropy.units.UnitBase
            Unit of *wavelength*.

        Returns
        -------
        jnp.ndarray
            ``r_m`` at each wavelength (before ``OrderScale``).
        """
        grid = jnp.asarray(self.wavelength.to_value(unit))
        return jnp.interp(
            jnp.asarray(wavelength), grid, jnp.asarray(self.ratio), left=0.0, right=0.0
        )

    @property
    def support(self) -> u.Quantity | None:
        """Observed-wavelength range ``[low, high]`` where ``r_m > 0``.

        Returns a length-2 Quantity in the table's wavelength unit, or
        ``None`` when the ratio is nowhere positive.  Because ``r_m`` is
        linearly interpolated and zero outside the table, it is exactly zero
        outside this range only up to the neighbouring tabulated points; use
        :meth:`support_bounds` for a range guaranteed to contain every
        non-zero value.
        """
        pos = self.ratio > 0
        if not pos.any():
            return None
        wl = self.wavelength.value[pos]
        return u.Quantity([wl.min(), wl.max()], self.wavelength.unit)

    def support_bounds(self, unit: u.UnitBase) -> tuple[float, float] | None:
        """Conservative range, in *unit*, outside which ``r_m`` is exactly zero.

        Extends :attr:`support` to the neighbouring tabulated points (the
        interpolant is non-zero on the open intervals adjacent to a positive
        value) and to the table ends.  ``None`` when ``r_m`` is identically
        zero.

        Parameters
        ----------
        unit : astropy.units.UnitBase
            Output wavelength unit.

        Returns
        -------
        tuple of float or None
        """
        nz = np.flatnonzero(self.ratio != 0)
        if nz.size == 0:
            return None
        wl = self.wavelength.to_value(unit)
        lo = wl[max(int(nz[0]) - 1, 0)]
        hi = wl[min(int(nz[-1]) + 1, wl.size - 1)]
        return float(lo), float(hi)

    # -- comparison / display ------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Equal when ``order``, ``path`` and ``column`` agree."""
        if not isinstance(other, OrderOverlap):
            return NotImplemented
        return (self.order, self.path, self.column) == (
            other.order,
            other.path,
            other.column,
        )

    def __hash__(self) -> int:
        """Hash on ``order``, ``path`` and ``column``."""
        return hash((self.order, self.path, self.column))

    def __repr__(self) -> str:
        """Return a compact representation."""
        parts = [f'order={self.order}', f'path={str(self.path)!r}']
        if self.column != RATIO_COLUMN:
            parts.append(f'column={self.column!r}')
        parts.append(f'scale={self.scale.display_name}')
        return f'OrderOverlap({', '.join(parts)})'


def _read_ratio_table(
    path: Path, column: str | None
) -> tuple[u.Quantity, np.ndarray, str]:
    """Read and validate an order-overlap ratio file.

    Returns ``(wavelength, ratio, column_name)``.
    """
    from astropy.table import Table

    table = Table.read(path)

    length_cols = [
        c
        for c in table.colnames
        if getattr(table[c], 'unit', None) is not None
        and u.Unit(table[c].unit).is_equivalent(u.m)
    ]
    if not length_cols:
        msg = f'No wavelength column (with a length unit) found in {path}.'
        raise ValueError(msg)
    if len(length_cols) > 1:
        msg = f'Multiple length-unit columns in {path}: {length_cols}.'
        raise ValueError(msg)
    wl_col = length_cols[0]

    others = [c for c in table.colnames if c != wl_col]
    if column is not None:
        if column not in others:
            msg = (
                f'Ratio column {column!r} not found in {path}. '
                f'Available columns: {table.colnames}'
            )
            raise ValueError(msg)
        col = column
    elif RATIO_COLUMN in others:
        col = RATIO_COLUMN
    elif len(others) == 1:
        col = others[0]
    else:
        msg = (
            f'No {RATIO_COLUMN!r} column in {path} and the ratio column is '
            f'ambiguous ({others}); pass column= explicitly.'
        )
        raise ValueError(msg)

    col_unit = getattr(table[col], 'unit', None)
    if col_unit is not None and str(col_unit) not in ('', 'None'):
        unit = u.Unit(col_unit)
        if unit.physical_type != 'dimensionless':
            msg = (
                f'Ratio column {col!r} in {path} has unit {unit!s}; it must be a '
                'dimensionless throughput ratio (order m relative to first '
                'order), not an absolute sensitivity or flux.'
            )
            raise ValueError(msg)
        scale = float(unit.to(u.dimensionless_unscaled))
    else:
        scale = 1.0

    wl = u.Quantity(np.asarray(table[wl_col], dtype=float), table[wl_col].unit)
    ratio = np.asarray(table[col], dtype=float) * scale
    if wl.size < 2 or not np.all(np.diff(wl.value) > 0):
        msg = (
            f'Wavelength column {wl_col!r} in {path} must be strictly increasing '
            'with at least two entries.'
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(wl.value)) or not np.all(np.isfinite(ratio)):
        msg = f'Non-finite values in {path} (columns {wl_col!r}, {col!r}).'
        raise ValueError(msg)
    return wl, ratio, col
