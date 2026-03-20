"""Spectrum loader functions.

Convenience functions for constructing :class:`~unite.spectrum.Spectrum`
objects from pre-loaded arrays or instrument-native file formats.

All loaders are re-exported from :mod:`unite.spectrum`::

    from unite.spectrum import from_arrays, from_DJA, from_sdss_fits
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy import units as u

from unite.instrument.base import Disperser
from unite.instrument.sdss.disperser import SDSSDisperser
from unite.spectrum.spectrum import Spectrum


def from_arrays(
    low: u.Quantity,
    high: u.Quantity,
    flux: u.Quantity,
    error: u.Quantity,
    disperser: Disperser,
    *,
    name: str = '',
) -> Spectrum:
    """Construct a :class:`~unite.spectrum.Spectrum` from pre-loaded arrays.

    Parameters
    ----------
    low : astropy.units.Quantity
        Lower wavelength edges of each pixel.  Must be 1-D with wavelength
        (length) dimensions.
    high : astropy.units.Quantity
        Upper wavelength edges of each pixel.  Same shape and compatible
        units as *low*.
    flux : astropy.units.Quantity
        Flux density values per pixel (f_lambda).  Must be 1-D with the
        same length as *low*.
    error : astropy.units.Quantity
        Flux density uncertainty per pixel.  Must be 1-D with the same
        length as *low* and compatible units as *flux*.
    disperser : Disperser
        The disperser associated with this spectrum.  Carries any
        calibration tokens (``r_scale``, ``flux_scale``, ``pix_offset``).
    name : str, optional
        Human-readable label.  Defaults to ``disperser.name``.

    Returns
    -------
    Spectrum
    """
    return Spectrum(
        low=low, high=high, flux=flux, error=error, disperser=disperser, name=name
    )


def from_DJA(
    path: str | Path, disperser: Disperser, *, name: str = '', cache: bool = False
) -> Spectrum:
    """Construct a :class:`~unite.spectrum.Spectrum` from a DJA FITS file.

    Reads a NIRSpec x1d spectrum from the Dawn JWST Archive (DJA) format.
    Any :class:`~unite.instrument.base.Disperser` is accepted; a built-in
    NIRSpec disperser or a :class:`~unite.instrument.generic.GenericDisperser`
    are both valid.

    Parameters
    ----------
    path : str or Path
        Local path or URL to a NIRSpec x1d FITS file.
    disperser : Disperser
        The disperser associated with this spectrum.
    name : str, optional
        Human-readable label.  Defaults to ``disperser.name``.
    cache : bool, optional
        Whether to use astropy's caching when fetching a remote file.
        Defaults to ``False``.

    Returns
    -------
    Spectrum

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If the expected FITS extensions or columns are not found.
    """
    from astropy.table import Table

    spec = Table.read(path, hdu='SPEC1D', cache=cache)

    λ = u.Quantity(spec['wave']).to(u.um)
    equiv = u.equivalencies.spectral_density(λ)
    fλ_unit = 1e-16 * u.erg / (u.s * u.cm**2 * u.um)
    fλ = spec['flux'].to(fλ_unit, equivalencies=equiv)
    eλ = spec['err'].to(fλ_unit, equivalencies=equiv)

    δλ = np.diff(λ) / 2
    mid = λ[:-1] + δλ
    edges = np.concatenate([λ[0:1] - δλ[0:1], mid, λ[-2:-1] + δλ[-2:-1]])
    low = edges[:-1]
    high = edges[1:]

    mask = ~spec['flux'].mask
    low = low[mask]
    high = high[mask]
    fλ = fλ[mask]
    eλ = eλ[mask]

    return Spectrum(
        low=low, high=high, flux=fλ, error=eλ, disperser=disperser, name=name
    )


def from_sdss_fits(
    path: str | Path, disperser: SDSSDisperser, *, name: str = '', cache: bool = False
) -> Spectrum:
    """Construct a :class:`~unite.spectrum.Spectrum` from an SDSS FITS file.

    Reads a standard SDSS ``spec-*.fits`` file (DR7-DR17 format).  The coadded
    spectrum is read from the ``COADD`` HDU.  The disperser's internal wavelength
    and resolving-power grids are updated from the ``loglam`` and ``wdisp``
    columns in the file.

    Parameters
    ----------
    path : str or Path
        Path to an SDSS spec FITS file.
    disperser : SDSSDisperser
        The SDSS disperser for this spectrum.  Its internal wavelength grid and
        R(λ) curve will be updated from the FITS data.  Must be an
        :class:`~unite.instrument.sdss.SDSSDisperser` instance.
    name : str, optional
        Human-readable label.  Defaults to ``disperser.name``.
    cache : bool, optional
        Whether to use astropy's caching when reading the file.
        Defaults to ``False``.

    Returns
    -------
    Spectrum

    Raises
    ------
    TypeError
        If *disperser* is not an :class:`~unite.instrument.sdss.SDSSDisperser`.
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If the expected columns are not found.
    """
    if not isinstance(disperser, SDSSDisperser):
        msg = (
            f'from_sdss_fits requires an SDSSDisperser, got {type(disperser).__name__}.'
        )
        raise TypeError(msg)

    import jax.numpy as jnp
    from astropy.table import Table

    data = Table.read(path, hdu='COADD', cache=cache)

    loglam = np.asarray(data['loglam'], dtype=float)
    wavelength = 10.0**loglam  # Angstrom

    dlog = np.diff(loglam)
    log_edges = np.empty(len(loglam) + 1)
    log_edges[1:-1] = (loglam[:-1] + loglam[1:]) / 2.0
    log_edges[0] = loglam[0] - dlog[0] / 2.0
    log_edges[-1] = loglam[-1] + dlog[-1] / 2.0
    edges = 10.0**log_edges

    low = edges[:-1] * u.AA
    high = edges[1:] * u.AA

    flux_unit = 1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
    flux = np.asarray(data['flux'], dtype=float) * flux_unit

    ivar = np.asarray(data['ivar'], dtype=float)
    good = ivar > 0
    error_vals = np.zeros_like(ivar)
    error_vals[good] = 1.0 / np.sqrt(ivar[good])
    error_vals[~good] = np.median(error_vals[good]) * 100.0 if np.any(good) else 1.0
    error = error_vals * flux_unit

    if 'and_mask' in data.colnames:
        and_mask = np.asarray(data['and_mask'], dtype=int)
        good_pixels = (and_mask == 0) & (ivar > 0)
    else:
        good_pixels = ivar > 0

    low = low[good_pixels]
    high = high[good_pixels]
    flux = flux[good_pixels]
    error = error[good_pixels]

    wdisp = np.asarray(data['wdisp'], dtype=float)
    disperser._wavelength_grid = jnp.asarray(wavelength, dtype=float)
    disperser._R_grid = jnp.asarray(wavelength / (2.355 * wdisp), dtype=float)
    disperser._dlam_dpix_grid = jnp.gradient(disperser._wavelength_grid)

    return Spectrum(
        low=low, high=high, flux=flux, error=error, disperser=disperser, name=name
    )
