"""SDSS spectrum class."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from unite.instrument.generic import GenericSpectrum

if TYPE_CHECKING:
    from unite.instrument.sdss.disperser import SDSSDisperser


class SDSSSpectrum(GenericSpectrum):
    """An SDSS optical spectrum.

    Extends :class:`~unite.instrument.generic.GenericSpectrum` with loaders for
    SDSS data formats.  Can be constructed directly via :meth:`from_arrays` or
    :meth:`from_fits`, or instantiated like any
    :class:`~unite.instrument.generic.GenericSpectrum` if you already have the
    array data.

    Parameters
    ----------
    low, high, flux, error, disperser, name
        See :class:`~unite.instrument.generic.GenericSpectrum`.

    Examples
    --------
    Load from a standard SDSS spec file:

    >>> from unite.instrument.sdss import SDSSDisperser, SDSSSpectrum
    >>> disperser = SDSSDisperser()
    >>> spec = SDSSSpectrum.from_fits('spec-1678-53433-0425.fits', disperser)
    """

    @classmethod
    def from_arrays(
        cls,
        low: u.Quantity,
        high: u.Quantity,
        flux: u.Quantity,
        error: u.Quantity,
        disperser: SDSSDisperser,
        *,
        name: str = '',
    ) -> SDSSSpectrum:
        """Create an :class:`SDSSSpectrum` from pre-loaded arrays.

        Parameters
        ----------
        low : astropy.units.Quantity
            Lower wavelength edges of each pixel (Angstrom).
        high : astropy.units.Quantity
            Upper wavelength edges of each pixel (Angstrom).
        flux : astropy.units.Quantity
            Flux density values per pixel (f_lambda).
        error : astropy.units.Quantity
            Flux density uncertainty per pixel.
        disperser : SDSSDisperser
            The SDSS disperser for this spectrum.
        name : str, optional
            Human-readable label.  Defaults to ``disperser.name``.

        Returns
        -------
        SDSSSpectrum
        """
        return cls(
            low=low,
            high=high,
            flux=flux,
            error=error,
            disperser=disperser,
            name=name or disperser.name,
        )

    @classmethod
    def from_fits(
        cls,
        path: str | Path,
        disperser: SDSSDisperser,
        *,
        name: str = '',
        cache: bool = False,
    ) -> SDSSSpectrum:
        """Create an :class:`SDSSSpectrum` from an SDSS FITS file.

        Reads a standard SDSS ``spec-*.fits`` file (DR7-DR17 format).
        The coadded spectrum is in HDU 1 with columns ``flux``,
        ``loglam``, ``ivar``, ``wdisp``, and ``and_mask``.

        The disperser's R(λ) curve and pixel grid are updated from the
        ``loglam`` and ``wdisp`` columns in the file.

        Parameters
        ----------
        path : str or Path
            Path to an SDSS spec FITS file.
        disperser : SDSSDisperser
            The SDSS disperser for this spectrum.  Its internal
            wavelength grid and R(λ) curve will be updated from the
            FITS data.
        name : str, optional
            Human-readable label.  Defaults to ``disperser.name``.
        cache : bool, optional
            Whether to use astropy's caching when reading the file. Defaults to False.

        Returns
        -------
        SDSSSpectrum

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        KeyError
            If the expected columns are not found.
        """
        from astropy.table import Table

        data = Table.read(path, hdu='COADD', cache=cache)

        # Log-linear wavelength grid (loglam is log10(λ/Å))
        loglam = np.asarray(data['loglam'], dtype=float)
        wavelength = 10.0**loglam  # Angstrom

        # Compute pixel edges from log-linear grid
        # In log-space, pixels are uniformly spaced
        dlog = np.diff(loglam)
        # Use half the spacing on each side for edges
        log_edges = np.empty(len(loglam) + 1)
        log_edges[1:-1] = (loglam[:-1] + loglam[1:]) / 2.0
        log_edges[0] = loglam[0] - dlog[0] / 2.0
        log_edges[-1] = loglam[-1] + dlog[-1] / 2.0
        edges = 10.0**log_edges

        low = edges[:-1] * u.AA
        high = edges[1:] * u.AA

        # Flux (1e-17 erg/s/cm²/Å) and inverse variance
        flux_unit = 1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
        flux = np.asarray(data['flux'], dtype=float) * flux_unit

        ivar = np.asarray(data['ivar'], dtype=float)
        # Convert ivar to error; mask pixels with ivar=0
        good = ivar > 0
        error_vals = np.zeros_like(ivar)
        error_vals[good] = 1.0 / np.sqrt(ivar[good])
        # Set bad pixels to a large error rather than zero
        error_vals[~good] = np.median(error_vals[good]) * 100.0 if np.any(good) else 1.0
        error = error_vals * flux_unit

        # Apply mask (and_mask: 0 = good)
        if 'and_mask' in data.colnames:
            and_mask = np.asarray(data['and_mask'], dtype=int)
            good_pixels = (and_mask == 0) & (ivar > 0)
        else:
            good_pixels = ivar > 0

        low = low[good_pixels]
        high = high[good_pixels]
        flux = flux[good_pixels]
        error = error[good_pixels]
        wavelength = wavelength[good_pixels]

        # Update the disperser with the actual R(λ) from wdisp
        wdisp = np.asarray(data['wdisp'], dtype=float)[good_pixels]

        # Rebuild disperser internals with data from the file
        import jax.numpy as jnp

        disperser._wavelength_grid = jnp.asarray(wavelength, dtype=float)
        disperser._R_grid = jnp.asarray(
            wavelength / (2.355 * np.maximum(wdisp, 1e-10)), dtype=float
        )
        disperser._dlam_dpix_grid = jnp.gradient(disperser._wavelength_grid)
        disperser._has_data = True

        return cls.from_arrays(low, high, flux, error, disperser, name=name)
