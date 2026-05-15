"""DESI spectrograph disperser implementation.

Provides :class:`DESIDisperser`, a concrete
:class:`~unite.instrument.base.Disperser` for DESI optical spectra.
The resolving power *R(λ)* is derived from the banded LSF resolution matrix
stored in native DESI ``spectra-*.fits`` files at load time via
:func:`~unite.spectrum.from_desi_fits`.

Before data is loaded, the disperser can be constructed with placeholder
values, allowing calibration tokens to be configured.

Examples
--------
>>> from unite.instrument.desi import DESIDisperser
>>> d = DESIDisperser('B')
>>> d.unit
Unit("Angstrom")
>>> d.arm
'B'
"""

from __future__ import annotations

import jax.numpy as jnp
from astropy import units as u
from jax.typing import ArrayLike

from unite.instrument.base import Disperser, FluxScale, PixOffset, RScale

_ARM_WAVE_RANGE: dict[str, tuple[float, float]] = {
    'B': (3600.0, 5800.0),
    'R': (5760.0, 7620.0),
    'Z': (7520.0, 9824.0),
}
_ARM_R_RANGE: dict[str, tuple[float, float]] = {
    'B': (2000.0, 4000.0),
    'R': (3000.0, 6000.0),
    'Z': (3000.0, 6500.0),
}


class DESIDisperser(Disperser):
    """Disperser for a single DESI spectrograph arm.

    The DESI focal-plane spectrograph splits each spectrum into three arms:
    B (blue, 3600-5800 Angstrom), R (red, 5760-7620 Angstrom), and Z (near-IR, 7520-9824 Angstrom).
    The resolving power and linear dispersion are set from data at spectrum
    load time (:func:`~unite.spectrum.from_desi_fits`), but the disperser
    can also be pre-constructed with calibration tokens for serialization.

    Parameters
    ----------
    arm : {'B', 'R', 'Z'}
        Spectrograph arm to model.
    name : str, optional
        Human-readable label.  Defaults to the lower-case arm letter.
    r_scale : RScale, optional
        Token for the resolving-power scale.
    flux_scale : FluxScale, optional
        Token for the flux normalisation.
    pix_offset : PixOffset, optional
        Token for the pixel shift.
    """

    ARMS: tuple[str, ...] = ('B', 'R', 'Z')

    def __init__(
        self,
        arm: str,
        name: str = '',
        r_scale: RScale | None = None,
        flux_scale: FluxScale | None = None,
        pix_offset: PixOffset | None = None,
    ) -> None:
        arm_upper = arm.upper()
        if arm_upper not in self.ARMS:
            msg = f"arm must be one of {self.ARMS}, got '{arm}'"
            raise ValueError(msg)
        self.arm: str = arm_upper

        wave_lo, wave_hi = _ARM_WAVE_RANGE[arm_upper]
        r_lo, r_hi = _ARM_R_RANGE[arm_upper]
        # Placeholder 2-point grids — overwritten by from_desi_fits() at load time.
        self._wavelength_grid: jnp.ndarray = jnp.array([wave_lo, wave_hi], dtype=float)
        self._R_grid: jnp.ndarray = jnp.array([r_lo, r_hi], dtype=float)
        self._dlam_dpix_grid: jnp.ndarray = jnp.array([0.8, 0.8], dtype=float)

        super().__init__(
            unit=u.AA,
            name=name or arm_upper.lower(),
            r_scale=r_scale,
            flux_scale=flux_scale,
            pix_offset=pix_offset,
        )

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
        return jnp.interp(wavelength, self._wavelength_grid, self._dlam_dpix_grid)

    def __repr__(self) -> str:
        """Return a readable string representation."""
        return f'DESIDisperser(arm={self.arm!r}, name={self.name!r})'
