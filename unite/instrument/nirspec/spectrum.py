"""NIRSpec spectrum class."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

from unite.instrument.generic import GenericSpectrum

if TYPE_CHECKING:
    from unite.instrument.nirspec.disperser import NIRSpecDisperser


class NIRSpecSpectrum(GenericSpectrum):
    """A JWST NIRSpec spectrum.

    Extends :class:`~unite.instrument.generic.GenericSpectrum` with loaders for
    NIRSpec data formats.  Can be constructed directly via
    :meth:`from_arrays` or :meth:`from_DJA`, or instantiated like any
    :class:`~unite.instrument.generic.GenericSpectrum` if you already have the
    array data.

    Parameters
    ----------
    low, high, flux, error, disperser, name
        See :class:`~unite.instrument.generic.GenericSpectrum`.

    Examples
    --------
    Create from pre-loaded arrays:

    >>> from astropy import units as u
    >>> from unite.instrument.nirspec import G235H, NIRSpecSpectrum
    >>> import numpy as np
    >>> disperser = G235H()
    >>> n = 300
    >>> low = np.linspace(1.66, 3.17, n + 1)[:-1] * u.um
    >>> high = np.linspace(1.66, 3.17, n + 1)[1:] * u.um
    >>> flux_unit = 1e-20 * u.erg / (u.s * u.cm**2 * u.AA)
    >>> spec = NIRSpecSpectrum.from_arrays(low, high, np.ones(n) * flux_unit, np.full(n, 0.1) * flux_unit, disperser)
    """

    @classmethod
    def from_arrays(
        cls,
        low: u.Quantity,
        high: u.Quantity,
        flux: u.Quantity,
        error: u.Quantity,
        disperser: NIRSpecDisperser,
        *,
        name: str = '',
    ) -> NIRSpecSpectrum:
        """Create a :class:`NIRSpecSpectrum` from pre-loaded arrays.

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
        disperser : NIRSpecDisperser
            The NIRSpec disperser associated with this spectrum.  Carries any
            calibration tokens (``r_scale``, ``flux_scale``, ``pix_offset``).
        name : str, optional
            Human-readable label.  Defaults to ``disperser.name``.

        Returns
        -------
        NIRSpecSpectrum
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
    def from_DJA(
        cls,
        path: str | Path,
        disperser: NIRSpecDisperser,
        *,
        name: str = '',
        cache=False,
    ) -> NIRSpecSpectrum:
        """Create a :class:`NIRSpecSpectrum` from a DJA FITS file (or URL).

        Parameters
        ----------
        path : str or Path
            Path to a NIRSpec x1d FITS file.
        disperser : NIRSpecDisperser
            The NIRSpec disperser associated with this spectrum.
        name : str, optional
            Human-readable label.  Defaults to ``disperser.name``.
        cache : bool, optional
            Whether to use astropy's caching when reading the file. Defaults to False.

        Returns
        -------
        NIRSpecSpectrum

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        KeyError
            If the expected FITS extensions or columns are not found.
        """
        from astropy.table import Table

        # Load spectrum
        spec = Table.read(path, hdu='SPEC1D', cache=cache)

        # Convert to correct units
        λ = u.Quantity(spec['wave']).to(u.AA)
        equiv = u.equivalencies.spectral_density(λ)
        fλ_unit = 1e-20 * u.erg / (u.s * u.cm**2 * u.AA)
        fλ = spec['flux'].to(fλ_unit, equivalencies=equiv)
        eλ = spec['err'].to(fλ_unit, equivalencies=equiv)

        # Compute edges
        δλ = np.diff(λ) / 2
        mid = λ[:-1] + δλ
        edges = np.concatenate([λ[0:1] - δλ[0:1], mid, λ[-2:-1] + δλ[-2:-1]])
        low = edges[:-1]
        high = edges[1:]

        # Apply mask
        mask = ~spec['flux'].mask
        low = low[mask]
        high = high[mask]
        fλ = fλ[mask]
        eλ = eλ[mask]

        return cls.from_arrays(low, high, fλ, eλ, disperser, name=name)
