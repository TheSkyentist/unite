"""NIRSpec spectrum loader."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
from astropy import units as u
from jax.typing import ArrayLike

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
        flux: ArrayLike,
        error: ArrayLike,
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
        flux : ArrayLike
            Flux values per pixel.  Must be 1-D with the same length as *low*.
        error : ArrayLike
            Flux uncertainty per pixel.  Must be 1-D with the same length as *low*.
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
    def from_fits(
        cls,
        path: str | Path,
        disperser: NIRSpecDisperser,
        *,
        name: str = '',
    ) -> Spectrum:
        """Load a NIRSpec x1d FITS file and return a :class:`~unite.spectrum.spectrum.Spectrum`.

        Reads a standard NIRSpec x1d FITS file produced by the JWST pipeline.
        The ``EXTRACT1D`` extension is expected to contain columns
        ``WAVELENGTH``, ``FLUX``, and ``FLUX_ERROR`` (or ``ERROR``), all in
        the standard pipeline units.  If ``WAVELENGTH_BIN_LOW`` /
        ``WAVELENGTH_BIN_HIGH`` columns are present they are used directly for
        pixel edges; otherwise edges are derived from the wavelength centres
        via ``jnp.gradient``.

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
        from astropy.io import fits

        path = Path(path)
        with fits.open(path) as hdul:
            data = hdul['EXTRACT1D'].data
            wavelength = jnp.asarray(data['WAVELENGTH'], dtype=float)
            flux = jnp.asarray(data['FLUX'], dtype=float)

            # Try common error column names.
            if 'FLUX_ERROR' in data.names:
                error = jnp.asarray(data['FLUX_ERROR'], dtype=float)
            elif 'ERROR' in data.names:
                error = jnp.asarray(data['ERROR'], dtype=float)
            else:
                msg = 'FITS file has no FLUX_ERROR or ERROR column.'
                raise KeyError(msg)

            # Derive pixel edges.
            if 'WAVELENGTH_BIN_LOW' in data.names and 'WAVELENGTH_BIN_HIGH' in data.names:
                low_vals = jnp.asarray(data['WAVELENGTH_BIN_LOW'], dtype=float)
                high_vals = jnp.asarray(data['WAVELENGTH_BIN_HIGH'], dtype=float)
            else:
                # Approximate edges from centres via finite differences.
                half = jnp.gradient(wavelength) / 2.0
                low_vals = wavelength - half
                high_vals = wavelength + half

        low = low_vals * u.um
        high = high_vals * u.um

        return cls.from_arrays(low, high, flux, error, disperser, name=name)
