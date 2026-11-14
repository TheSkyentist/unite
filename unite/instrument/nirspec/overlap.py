"""Helpers for building higher-order overlap ratio tables for NIRSpec."""

from __future__ import annotations

from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

__all__ = ['msaexp_order_ratio']

_MSAEXP_URL = (
    'https://raw.githubusercontent.com/gbrammer/msaexp/main/msaexp/data/'
    'extended_sensitivity/msaexp_sensitivity_{}_001.fits'
)


def msaexp_order_ratio(
    source: str | Path | Table, order: int, path: str | Path | None = None
) -> Table:
    r"""Build the order-*m* throughput ratio table from an msaexp sensitivity curve.

    Converts the msaexp extended-sensitivity curves :math:`S_m` (per unit
    :math:`f_\nu`, against photon wavelength) to the table read by
    :class:`~unite.instrument.OrderOverlap`:

    .. math::

        r_m(\lambda) = \frac{S_m(\lambda/m)}{m^2\, S_1(\lambda)},

    indexed by the **observed** wavelength :math:`\lambda = m\lambda_{\rm photon}`
    at which the order-*m* light lands.  The factor :math:`m^2` converts to the
    :math:`f_\lambda` convention.  The ratio is zero where either sensitivity is
    zero or undefined.

    Parameters
    ----------
    source : str, pathlib.Path or astropy.table.Table
        Either a grating/filter name such as ``'g140m_f070lp'`` (the matching
        table is downloaded from the msaexp repository and cached), a path or URL
        to an msaexp sensitivity FITS file, or an already-loaded table.  The table
        needs ``wavelength`` (µm, photon wavelength), ``sensitivity`` and
        ``sensitivity_{order}`` columns.
    order : int
        Diffraction order *m* (>= 2).
    path : str or pathlib.Path, optional
        If given, the table is also written there (overwriting), ready to pass to
        :class:`~unite.instrument.OrderOverlap`.

    Returns
    -------
    astropy.table.Table
        Columns ``wavelength`` (observed, µm) and ``throughput_ratio``
        (dimensionless).

    Raises
    ------
    ValueError
        If *order* is below 2 or a required column is missing.
    """
    if order < 2:
        raise ValueError(f'order must be >= 2, got {order}.')

    if isinstance(source, Table):
        table = source
    else:
        loc = str(source)
        if not (Path(loc).exists() or '/' in loc or loc.endswith('.fits')):
            from astropy.utils.data import download_file

            loc = download_file(_MSAEXP_URL.format(loc.lower()), cache=True)
        table = Table.read(loc)

    needed = ('wavelength', 'sensitivity', f'sensitivity_{order}')
    missing = [c for c in needed if c not in table.colnames]
    if missing:
        raise ValueError(
            f'Sensitivity table is missing column(s) {missing}; has {table.colnames}.'
        )

    wl = np.asarray(table['wavelength'], dtype=float)  # photon wavelength, µm
    s_1 = np.nan_to_num(np.asarray(table['sensitivity'], dtype=float))
    s_m = np.nan_to_num(np.asarray(table[f'sensitivity_{order}'], dtype=float))

    wl_obs = order * wl  # where the order-m photons land
    s1_obs = np.interp(wl_obs, wl, s_1, left=0.0, right=0.0)
    ratio = np.zeros_like(wl_obs)
    ok = (s1_obs > 0) & (s_m > 0)
    ratio[ok] = s_m[ok] / (order**2 * s1_obs[ok])

    out = Table({'wavelength': wl_obs * u.um, 'throughput_ratio': ratio})
    if path is not None:
        out.write(path, overwrite=True)
    return out
