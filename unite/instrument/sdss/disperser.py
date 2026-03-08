"""SDSS spectrograph disperser implementation.

Provides :class:`SDSSDisperser`, a concrete
:class:`~unite.instrument.base.Disperser` for SDSS optical spectra.
The resolving power *R(λ)* is derived from the ``wdisp`` column
(wavelength dispersion per pixel, in Angstroms) of standard SDSS
``spec-*.fits`` files at load time via :meth:`SDSSSpectrum.from_fits`.

Before data is loaded, the disperser can be constructed with
default or placeholder R values, allowing calibration tokens
to be configured and the disperser configuration to be serialized.

Examples
--------
>>> from unite.instrument.sdss import SDSSDisperser
>>> d = SDSSDisperser()
>>> d.unit
Unit("Angstrom")
"""

from __future__ import annotations

import jax.numpy as jnp
from astropy import units as u
from jax.typing import ArrayLike

from unite.instrument.base import Disperser, FluxScale, PixOffset, RScale


class SDSSDisperser(Disperser):
    """Disperser for SDSS optical spectra.

    The SDSS spectrographs use a log-linear wavelength grid.  The
    resolving power and linear dispersion are set from data at spectrum
    load time (:meth:`SDSSSpectrum.from_fits`), but the disperser can
    also be pre-constructed with calibration tokens for serialization.

    Parameters
    ----------
    wavelength : ArrayLike, optional
        Pixel-centre wavelengths in Angstroms.  When not provided,
        :meth:`R` and :meth:`dlam_dpix` return placeholder values
        (constant R=2000, dlam_dpix=1.0).
    wdisp : ArrayLike, optional
        Wavelength dispersion per pixel (Angstroms/pixel) from the
        SDSS ``wdisp`` column.  Same length as *wavelength*.  The
        resolving power is computed as ``R(λ) = λ / (2.355 * wdisp)``
        (since ``wdisp`` is sigma in wavelength units per pixel, and
        FWHM = 2.355 * sigma).
    name : str, optional
        Human-readable label.  Defaults to ``'SDSS'``.
    r_scale : RScale, optional
        Token for the resolving-power scale.
    flux_scale : FluxScale, optional
        Token for the flux normalisation.
    pix_offset : PixOffset, optional
        Token for the pixel shift.
    """

    def __init__(
        self,
        wavelength: ArrayLike | None = None,
        wdisp: ArrayLike | None = None,
        *,
        name: str = '',
        r_scale: RScale | None = None,
        flux_scale: FluxScale | None = None,
        pix_offset: PixOffset | None = None,
    ) -> None:
        super().__init__(
            unit=u.AA,
            name=name or 'SDSS',
            r_scale=r_scale,
            flux_scale=flux_scale,
            pix_offset=pix_offset,
        )

        if wavelength is not None:
            self._wavelength_grid = jnp.asarray(wavelength, dtype=float)
            if wdisp is not None:
                wdisp_arr = jnp.asarray(wdisp, dtype=float)
                # R = λ / FWHM = λ / (2.355 * wdisp)
                self._R_grid = self._wavelength_grid / (2.355 * wdisp_arr)
            else:
                # Default: constant R = 2000
                self._R_grid = jnp.full_like(self._wavelength_grid, 2000.0)
            self._dlam_dpix_grid = jnp.gradient(self._wavelength_grid)
            self._has_data = True
        else:
            self._wavelength_grid = None
            self._R_grid = None
            self._dlam_dpix_grid = None
            self._has_data = False

    def R(self, wavelength: ArrayLike) -> ArrayLike:
        """Return the resolving power at the given wavelengths (Angstrom).

        Parameters
        ----------
        wavelength : ArrayLike
            Wavelength values in Angstroms.

        Returns
        -------
        ArrayLike
            Resolving power *R* at each wavelength.
        """
        if not self._has_data:
            return jnp.full_like(jnp.asarray(wavelength), 2000.0)
        return jnp.interp(wavelength, self._wavelength_grid, self._R_grid)

    def dlam_dpix(self, wavelength: ArrayLike) -> ArrayLike:
        """Return *dλ/dpix* in Angstrom/pixel.

        Parameters
        ----------
        wavelength : ArrayLike
            Wavelength values in Angstroms.

        Returns
        -------
        ArrayLike
            Linear dispersion at each wavelength.
        """
        if not self._has_data:
            return jnp.ones_like(jnp.asarray(wavelength))
        return jnp.interp(wavelength, self._wavelength_grid, self._dlam_dpix_grid)

    def __repr__(self) -> str:
        """Return a readable string representation."""
        data_str = f'{len(self._wavelength_grid)} px' if self._has_data else 'no data'
        return f'SDSSDisperser(name={self.name!r}, {data_str})'
