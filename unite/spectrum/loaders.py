"""Spectrum loader functions.

Convenience functions for constructing :class:`~unite.spectrum.Spectrum`
objects from pre-loaded arrays or instrument-native file formats.

All loaders are re-exported from :mod:`unite.spectrum`::

    from unite.spectrum import from_arrays, from_edges, from_centers
    from unite.spectrum import from_DJA, from_sdss_fits, from_desi_fits
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

import numpy as np
from astropy import units as u

from unite.instrument.base import Disperser
from unite.instrument.desi.disperser import DESIDisperser
from unite.instrument.sdss.disperser import SDSSDisperser
from unite.spectrum.spectrum import Spectrum


def _apply_mask(
    mask: np.ndarray | None,
    n: int,
    *arrays: u.Quantity | np.ndarray,
) -> tuple[u.Quantity | np.ndarray, ...]:
    """Validate and apply a bad-pixel mask to one or more arrays."""
    if mask is None:
        return arrays
    mask_arr = np.asarray(mask, dtype=bool)
    if mask_arr.ndim != 1:
        msg = f'mask must be 1-D, got shape {mask_arr.shape}.'
        raise ValueError(msg)
    if len(mask_arr) != n:
        msg = f'mask length ({len(mask_arr)}) does not match the number of pixels ({n}).'
        raise ValueError(msg)
    good = ~mask_arr
    return tuple(a[good] for a in arrays)


def from_arrays(
    *,
    low: u.Quantity | None = None,
    high: u.Quantity | None = None,
    center: u.Quantity | None = None,
    flux: u.Quantity,
    error: u.Quantity,
    disperser: Disperser,
    mask: np.ndarray | None = None,
    name: str = '',
) -> Spectrum:
    """Construct a :class:`~unite.spectrum.Spectrum` from pre-loaded arrays.

    All arguments are keyword-only.  Pixel boundaries may be supplied either
    as explicit bin edges (*low* + *high*) or as pixel centres (*center*), but
    not both.  Use :func:`from_edges` or :func:`from_centers` for a
    positional-argument interface.

    Parameters
    ----------
    low : astropy.units.Quantity, optional
        Lower wavelength edge of each pixel.  Required when *center* is not
        given.  Must be 1-D with wavelength (length) dimensions.
    high : astropy.units.Quantity, optional
        Upper wavelength edge of each pixel.  Required when *center* is not
        given.  Same shape and compatible units as *low*.
    center : astropy.units.Quantity, optional
        Pixel-centre wavelengths.  Required when *low* / *high* are not given.
        Pixel edges are derived via :func:`numpy.gradient`, so this path is
        most accurate for gapless, reasonably uniform grids.  For spectra with
        detector gaps or non-uniform spacing, supply *low* and *high* directly.
    flux : astropy.units.Quantity
        Flux density values per pixel (f_lambda).  Must be 1-D with the same
        length as the wavelength arrays.
    error : astropy.units.Quantity
        Flux density uncertainty per pixel.  Same length and compatible units
        as *flux*.
    disperser : Disperser
        The disperser associated with this spectrum.  Carries any calibration
        tokens (``r_scale``, ``flux_scale``, ``pix_offset``).
    mask : numpy.ndarray, optional
        Boolean bad-pixel mask of shape ``(npix,)``.  ``True`` marks a pixel
        as bad (excluded); ``False`` keeps it.  Follows the same convention as
        :class:`numpy.ma.MaskedArray`.  ``None`` (default) keeps all pixels.
    name : str, optional
        Human-readable label.  Defaults to ``disperser.name``.

    Returns
    -------
    Spectrum

    Raises
    ------
    ValueError
        If both *center* and *low* / *high* are provided, if only one of
        *low* / *high* is provided, if neither *center* nor *low* / *high* are
        provided, or if *mask* is not 1-D or its length does not match *npix*.
    """
    has_edges = low is not None or high is not None
    has_center = center is not None

    if has_center and has_edges:
        msg = 'provide either center or low/high, not both.'
        raise ValueError(msg)
    if not has_center and not has_edges:
        msg = 'must provide either center or both low and high.'
        raise ValueError(msg)
    if has_edges and (low is None or high is None):
        msg = 'must provide both low and high, not just one.'
        raise ValueError(msg)

    if has_center:
        n = len(center)  # type: ignore[arg-type]
        center, flux, error = _apply_mask(mask, n, center, flux, error)  # type: ignore[assignment]
        # Build edges as midpoints between adjacent centres so that high[i] == low[i+1]
        # exactly, preserving the shared-edge topology.  Boundary edges are extrapolated
        # using the local pixel spacing.
        cval = center.value  # type: ignore[union-attr]
        cunit = center.unit  # type: ignore[union-attr]
        mid = (cval[:-1] + cval[1:]) / 2.0
        low_edge = np.empty(len(cval))
        high_edge = np.empty(len(cval))
        low_edge[0] = cval[0] - (cval[1] - cval[0]) / 2.0
        low_edge[1:] = mid
        high_edge[:-1] = mid
        high_edge[-1] = cval[-1] + (cval[-1] - cval[-2]) / 2.0
        low = low_edge * cunit
        high = high_edge * cunit
    else:
        n = len(low)  # type: ignore[arg-type]
        low, high, flux, error = _apply_mask(mask, n, low, high, flux, error)  # type: ignore[assignment]

    return Spectrum(
        low=low, high=high, flux=flux, error=error, disperser=disperser, name=name
    )


def from_edges(
    low: u.Quantity,
    high: u.Quantity,
    flux: u.Quantity,
    error: u.Quantity,
    disperser: Disperser,
    *,
    mask: np.ndarray | None = None,
    name: str = '',
) -> Spectrum:
    """Construct a :class:`~unite.spectrum.Spectrum` from pixel bin edges.

    Positional-argument convenience wrapper around :func:`from_arrays` for the
    common case where pixel bin edges are already known.

    Parameters
    ----------
    low : astropy.units.Quantity
        Lower wavelength edge of each pixel.  Must be 1-D with wavelength
        (length) dimensions.
    high : astropy.units.Quantity
        Upper wavelength edge of each pixel.  Same shape and compatible units
        as *low*.
    flux : astropy.units.Quantity
        Flux density values per pixel (f_lambda).  Must be 1-D with the same
        length as *low*.
    error : astropy.units.Quantity
        Flux density uncertainty per pixel.  Same length and compatible units
        as *flux*.
    disperser : Disperser
        The disperser associated with this spectrum.  Carries any calibration
        tokens (``r_scale``, ``flux_scale``, ``pix_offset``).
    mask : numpy.ndarray, optional
        Boolean bad-pixel mask of shape ``(npix,)``.  ``True`` marks a pixel
        as bad (excluded); ``False`` keeps it.  ``None`` (default) keeps all
        pixels.
    name : str, optional
        Human-readable label.  Defaults to ``disperser.name``.

    Returns
    -------
    Spectrum
    """
    return from_arrays(
        low=low,
        high=high,
        flux=flux,
        error=error,
        disperser=disperser,
        mask=mask,
        name=name,
    )


def from_centers(
    center: u.Quantity,
    flux: u.Quantity,
    error: u.Quantity,
    disperser: Disperser,
    *,
    mask: np.ndarray | None = None,
    name: str = '',
) -> Spectrum:
    """Construct a :class:`~unite.spectrum.Spectrum` from pixel-centre wavelengths.

    Positional-argument convenience wrapper around :func:`from_arrays` for the
    common case where only pixel centres are available.  Pixel bin edges are
    derived via :func:`numpy.gradient`, so this path is most accurate for
    gapless, reasonably uniform grids.  For spectra with detector gaps or
    non-uniform spacing, supply *low* and *high* directly via :func:`from_edges`.

    Parameters
    ----------
    center : astropy.units.Quantity
        Pixel-centre wavelengths.  Must be 1-D with wavelength (length)
        dimensions.
    flux : astropy.units.Quantity
        Flux density values per pixel (f_lambda).  Must be 1-D with the same
        length as *center*.
    error : astropy.units.Quantity
        Flux density uncertainty per pixel.  Same length and compatible units
        as *flux*.
    disperser : Disperser
        The disperser associated with this spectrum.  Carries any calibration
        tokens (``r_scale``, ``flux_scale``, ``pix_offset``).
    mask : numpy.ndarray, optional
        Boolean bad-pixel mask of shape ``(npix,)``.  ``True`` marks a pixel
        as bad (excluded); ``False`` keeps it.  ``None`` (default) keeps all
        pixels.
    name : str, optional
        Human-readable label.  Defaults to ``disperser.name``.

    Returns
    -------
    Spectrum
    """
    return from_arrays(
        center=center,
        flux=flux,
        error=error,
        disperser=disperser,
        mask=mask,
        name=name,
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

    low = low[good_pixels]  # type: ignore[index]
    high = high[good_pixels]  # type: ignore[index]
    flux = flux[good_pixels]  # type: ignore[index]
    error = error[good_pixels]  # type: ignore[index]

    wdisp = np.asarray(data['wdisp'], dtype=float)
    disperser._wavelength_grid = jnp.asarray(wavelength, dtype=float)
    disperser._R_grid = jnp.asarray(wavelength / (2.355 * wdisp), dtype=float)
    disperser._dlam_dpix_grid = cast(
        jnp.ndarray, jnp.gradient(disperser._wavelength_grid)
    )

    return Spectrum(
        low=low, high=high, flux=flux, error=error, disperser=disperser, name=name
    )


def _desi_second_moment_R(res: np.ndarray, wavelength: np.ndarray) -> np.ndarray:
    """Derive *R(lambda)* from a DESI banded resolution matrix via second moment.

    The resolution matrix stores 11 diagonal bands (offsets -5..+5 pixels)
    per pixel.  Under the Gaussian LSF assumption the second moment of those
    weights gives sigma in pixels, which is converted to FWHM in Angstroms.

    Parameters
    ----------
    res : ndarray, shape (11, n_pix)
        Banded LSF weights for one spectrum (``{ARM}_RESOLUTION[index]``).
    wavelength : ndarray, shape (n_pix,)
        Pixel-centre wavelengths in Angstroms.

    Returns
    -------
    ndarray, shape (n_pix,)
        Resolving power *R = λ / FWHM* at each pixel.
    """
    offsets = np.arange(-5, 6, dtype=float)[:, None]  # (11, 1)
    norm = res.sum(axis=0)  # (n_pix,)
    mean = (offsets * res).sum(axis=0) / norm
    var = ((offsets - mean) ** 2 * res).sum(axis=0) / norm
    sigma_pix = np.sqrt(np.maximum(var, 0.0))
    fwhm_ang = 2.355 * sigma_pix * np.gradient(wavelength)
    return wavelength / fwhm_ang


def from_desi_fits(
    path: str | Path,
    *,
    dispersers: dict[str, DESIDisperser] | None = None,
    arms: Sequence[Literal['B', 'R', 'Z']] = ('B', 'R', 'Z'),
    index: int = 0,
    name: str = '',
    cache: bool = False,
) -> list[Spectrum]:
    """Construct :class:`~unite.spectrum.Spectrum` objects from a DESI FITS file.

    DESI spectra are split across three spectrograph arms (B, R, Z).  Each arm
    is returned as a separate :class:`~unite.spectrum.Spectrum` so that
    unite's coverage-filtering handles the ~40-100 Angstrom overlaps between arms
    naturally.  Wrap the result in a
    :class:`~unite.spectrum.Spectra` collection::

        spectra = Spectra(from_desi_fits('spectra.fits'), redshift=1.29)

    The resolving power *R(λ)* is derived from the banded LSF resolution
    matrix stored in the FITS file using the Gaussian second-moment method
    (assumes a Gaussian LSF, appropriate for DESI).

    Parameters
    ----------
    path : str or Path
        Path to a DESI ``spectra-*.fits`` file.
    dispersers : dict, optional
        Pre-constructed :class:`~unite.instrument.desi.DESIDisperser` instances
        keyed by upper-case arm letter (``'B'``, ``'R'``, ``'Z'``).  Useful
        when calibration tokens (``RScale``, ``FluxScale``, ``PixOffset``) need
        to be attached before loading.  Missing arms are auto-created.
    arms : sequence of {'B', 'R', 'Z'}, optional
        Which arms to load.  Defaults to all three.  Arms absent from the file
        emit a :class:`UserWarning` and are skipped.
    index : int, optional
        Index of the spectrum to extract from the file.  DESI coadd files may
        contain multiple spectra.  Defaults to 0.
    name : str, optional
        Base label for each arm's spectrum.  Each arm is named
        ``'{name}_{arm.lower()}'`` when *name* is non-empty, otherwise just
        the lower-case arm letter (``'b'``, ``'r'``, ``'z'``).
    cache : bool, optional
        Unused; kept for API consistency with other loaders.

    Returns
    -------
    list of Spectrum
        One entry per successfully loaded arm, in the order given by *arms*.

    Raises
    ------
    IndexError
        If *index* is out of range for the number of spectra in the file.
    ValueError
        If an element of *arms* is not one of ``'B'``, ``'R'``, ``'Z'``.
    """
    import jax.numpy as jnp
    from astropy.io import fits

    if dispersers is None:
        dispersers = {}

    flux_unit = 1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
    result: list[Spectrum] = []

    with fits.open(path) as hdul:
        arms_upper = [a.upper() for a in arms]
        for arm in arms_upper:
            if arm not in DESIDisperser.ARMS:
                msg = f"arm must be one of {DESIDisperser.ARMS}, got '{arm}'"
                raise ValueError(msg)

            wave_hdu = f'{arm}_WAVELENGTH'
            flux_hdu = f'{arm}_FLUX'
            ivar_hdu = f'{arm}_IVAR'
            mask_hdu = f'{arm}_MASK'
            res_hdu = f'{arm}_RESOLUTION'

            hdu_names = [h.name for h in hdul]
            if wave_hdu not in hdu_names:
                warnings.warn(
                    f'DESI arm {arm!r} not found in {path}; skipping.',
                    UserWarning,
                    stacklevel=2,
                )
                continue

            wavelength = np.asarray(hdul[wave_hdu].data, dtype=float)
            flux_data = np.asarray(hdul[flux_hdu].data, dtype=float)
            ivar_data = np.asarray(hdul[ivar_hdu].data, dtype=float)
            mask_data = np.asarray(hdul[mask_hdu].data, dtype=int)
            res_data = np.asarray(hdul[res_hdu].data, dtype=float)

            n_spectra = flux_data.shape[0]
            if index >= n_spectra or index < -n_spectra:
                msg = (
                    f'index {index} is out of range for {n_spectra} spectra '
                    f'in arm {arm!r} of {path}'
                )
                raise IndexError(msg)

            flux_1d = flux_data[index]
            ivar_1d = ivar_data[index]
            mask_1d = mask_data[index]
            res_1d = res_data[index]  # shape (11, n_pix)

            # Bad-pixel mask: mask bits set or zero inverse variance.
            good = (mask_1d == 0) & (ivar_1d > 0)

            error_vals = np.zeros_like(ivar_1d)
            error_vals[good] = 1.0 / np.sqrt(ivar_1d[good])

            # Pixel edges as midpoints between adjacent pixel centres so that
            # high[i] == low[i+1] exactly, enabling edge sharing in the topology.
            dlam = np.diff(wavelength)
            mid = (wavelength[:-1] + wavelength[1:]) / 2.0
            _edges = np.empty(len(wavelength) + 1)
            _edges[0] = wavelength[0] - dlam[0] / 2.0
            _edges[1:-1] = mid
            _edges[-1] = wavelength[-1] + dlam[-1] / 2.0
            low = _edges[:-1] * u.AA
            high = _edges[1:] * u.AA

            # Derive R(λ) from resolution matrix second moment.
            r_arr = _desi_second_moment_R(res_1d, wavelength)

            # Populate (or create) disperser and update its internal grids.
            disp = dispersers.get(arm, DESIDisperser(arm))
            disp._wavelength_grid = jnp.asarray(wavelength, dtype=float)
            disp._R_grid = jnp.asarray(r_arr, dtype=float)
            disp._dlam_dpix_grid = cast(
                jnp.ndarray, jnp.gradient(disp._wavelength_grid)
            )

            spec_name = f'{name}_{arm.lower()}' if name else arm.lower()
            result.append(
                Spectrum(
                    low=low[good],
                    high=high[good],
                    flux=flux_1d[good] * flux_unit,
                    error=error_vals[good] * flux_unit,
                    disperser=disp,
                    name=spec_name,
                )
            )

    return result
