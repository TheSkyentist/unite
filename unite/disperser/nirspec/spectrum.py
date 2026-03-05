"""NIRSpec spectrum loader."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from astropy import units as u

if TYPE_CHECKING:
    from unite.disperser.nirspec.disperser import NIRSpecDisperser
    from unite.spectrum.spectrum import Spectrum


class NIRSpecSpectrum:
    """Loader for JWST NIRSpec spectra.

    Provides class methods to create :class:`~unite.spectrum.spectrum.Spectrum`
    objects from NIRSpec data, attaching a :class:`NIRSpecDisperser` that
    carries any calibration tokens the user has configured.

    This class is not instantiated directly — use :meth:`from_fits` or
    :meth:`from_arrays`.

    Parameters
    ----------
    None
        This class is not instantiated.

    Examples
    --------
    Create from pre-loaded arrays:

    >>> from astropy import units as u
    >>> from unite.disperser.nirspec import G235H, NIRSpecSpectrum
    >>> import numpy as np
    >>> disperser = G235H()
    >>> n = 300
    >>> low = np.linspace(1.66, 3.17, n + 1)[:-1] * u.um
    >>> high = np.linspace(1.66, 3.17, n + 1)[1:] * u.um
    >>> flux = np.ones(n)
    >>> error = np.full(n, 0.1)
    >>> spec = NIRSpecSpectrum.from_arrays(low, high, flux, error, disperser)
    """

    def __new__(cls, *args, **kwargs) -> None:
        msg = 'NIRSpecSpectrum is a loader class and cannot be instantiated.'
        raise TypeError(msg)

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
    ) -> Spectrum:
        """Create a :class:`~unite.spectrum.spectrum.Spectrum` from pre-loaded arrays.

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
        Spectrum
        """
        from unite.spectrum.spectrum import Spectrum

        return Spectrum(
            low=low,
            high=high,
            flux=flux,
            error=error,
            disperser=disperser,
            name=name or disperser.name,
        )

    @classmethod
    def from_DJA(
        cls, path: str | Path, disperser: NIRSpecDisperser, *, name: str = ''
    ) -> Spectrum:
        """
        Create a :class:`~unite.spectrum.spectrum.Spectrum` from a DJA FITS file (or URL).

        Parameters
        ----------
        path : str or Path
            Path to a NIRSpec x1d FITS file.
        disperser : NIRSpecDisperser
            The NIRSpec disperser associated with this spectrum.
        name : str, optional
            Human-readable label.  Defaults to ``disperser.name``.

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

        # Load spectrum
        spec = Table.read(path, hdu='SPEC1D')

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
