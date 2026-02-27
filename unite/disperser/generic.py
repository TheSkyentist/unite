"""Generic disperser implementations for user-supplied callables and grids."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from astropy import constants, units as u
from jax.typing import ArrayLike

from unite.disperser.base import Disperser, FluxScale, PixOffset, RScale

_C_KMS: float = constants.c.to('km/s').value
"""Speed of light in km/s."""


class GenericDisperser(Disperser):
    """A disperser defined by user-supplied, JAX-jittable callables.

    This is the most flexible concrete disperser: you provide arbitrary
    functions for *R(λ)* and *dλ/dpix(λ)* and they are forwarded directly.

    Parameters
    ----------
    R_func : Callable[[ArrayLike], ArrayLike]
        A JAX-jittable function that returns the resolving power for a given
        array of wavelengths.
    dlam_dpix_func : Callable[[ArrayLike], ArrayLike]
        A JAX-jittable function that returns the linear dispersion for a
        given array of wavelengths.
    unit : astropy.units.UnitBase
        The wavelength unit the functions expect.
    name : str, optional
        Human-readable label.
    r_scale : RScale, optional
        Token for the resolving-power scale.
    flux_scale : FluxScale, optional
        Token for the flux normalisation.
    pix_offset : PixOffset, optional
        Token for the pixel shift.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from astropy import units as u
    >>> d = GenericDisperser(
    ...     R_func=lambda w: jnp.full_like(w, 2700.0),
    ...     dlam_dpix_func=lambda w: w / 2700.0,
    ...     unit=u.Angstrom,
    ... )
    >>> d.R(jnp.array([5000.0]))
    Array([2700.], dtype=float32)
    """

    def __init__(
        self,
        R_func: Callable[[ArrayLike], ArrayLike],  # noqa: N803
        dlam_dpix_func: Callable[[ArrayLike], ArrayLike],
        unit: u.UnitBase,
        *,
        name: str = '',
        r_scale: RScale | None = None,
        flux_scale: FluxScale | None = None,
        pix_offset: PixOffset | None = None,
    ) -> None:
        super().__init__(unit, name=name, r_scale=r_scale, flux_scale=flux_scale, pix_offset=pix_offset)
        self._R_func = R_func
        self._dlam_dpix_func = dlam_dpix_func

    def R(self, wavelength: ArrayLike) -> ArrayLike:
        """Return the resolving power by evaluating the stored callable."""
        return self._R_func(wavelength)

    def dlam_dpix(self, wavelength: ArrayLike) -> ArrayLike:
        """Return the linear dispersion by evaluating the stored callable."""
        return self._dlam_dpix_func(wavelength)


class SimpleDisperser(Disperser):
    """A disperser defined on a pixel-sampled wavelength grid.

    The wavelength array is interpreted as a sequence of pixel centres so that
    *dλ/dpix* is computed directly from the spacing of the array (via
    ``jnp.gradient``).

    The resolving power is derived from exactly **one** of three keyword
    arguments:

    * ``R`` — resolving power, scalar or array matching *wavelength*.
    * ``dlam`` — spectral resolution element Δλ (same unit as *wavelength*).
    * ``dvel`` — velocity fwhm in **km/s**.  Converted via *R = c / dvel*.

    A scalar value produces a **constant** R, Δλ, or Δv across the grid (and
    the corresponding resolving-power array is derived accordingly).  An array
    value must have the same length as *wavelength*.

    Parameters
    ----------
    wavelength : ArrayLike
        Pixel-centre wavelengths.  Must be 1-D.
    unit : astropy.units.UnitBase
        The wavelength unit of the grid.
    R : ArrayLike, optional
        Resolving power (scalar or per-pixel array).
    dlam : ArrayLike, optional
        Spectral resolution Δλ (scalar or per-pixel array, same unit as
        *wavelength*).
    dvel : ArrayLike, optional
        Velocity resolution in km/s (scalar or per-pixel array).
    name : str, optional
        Human-readable label.
    r_scale : RScale, optional
        Token for the resolving-power scale.
    flux_scale : FluxScale, optional
        Token for the flux normalisation.
    pix_offset : PixOffset, optional
        Token for the pixel shift.

    Raises
    ------
    ValueError
        If zero or more than one of ``R``, ``dlam``, ``dvel`` is provided,
        or if an array argument has the wrong length.

    Notes
    -----
    When :meth:`R` or :meth:`dlam_dpix` is called at wavelengths that differ
    from the stored grid, the values are linearly interpolated with
    ``jnp.interp``.
    """

    def __init__(
        self,
        wavelength: ArrayLike,
        unit: u.UnitBase,
        *,
        R: ArrayLike | None = None,  # noqa: N803
        dlam: ArrayLike | None = None,
        dvel: ArrayLike | None = None,
        name: str = '',
        r_scale: RScale | None = None,
        flux_scale: FluxScale | None = None,
        pix_offset: PixOffset | None = None,
    ) -> None:
        super().__init__(unit, name=name, r_scale=r_scale, flux_scale=flux_scale, pix_offset=pix_offset)

        n_specified = sum(x is not None for x in (R, dlam, dvel))
        if n_specified != 1:
            msg = (
                'Exactly one of R, dlam, or dvel must be provided, '
                f'but {n_specified} were given.'
            )
            raise ValueError(msg)

        self._wavelength = jnp.asarray(wavelength, dtype=float)

        # Compute dlam_dpix from the pixel grid.
        self._dlam_dpix_grid = jnp.gradient(self._wavelength)

        # Compute resolving power on the grid.
        if R is not None:
            self._R_grid = self._validated_input(R, 'R')
        elif dlam is not None:
            dlam_arr = self._validated_input(dlam, 'dlam')
            self._R_grid = self._wavelength / dlam_arr
        else:
            dvel_arr = self._validated_input(dvel, 'dvel')
            self._R_grid = _C_KMS / dvel_arr

    def _validated_input(self, value: ArrayLike, name: str) -> jnp.ndarray:
        """Convert *value* to a grid-shaped array.

        Scalars (0-d) are broadcast to the grid shape.  1-d arrays must match
        the grid length exactly; anything else raises `ValueError`.

        Parameters
        ----------
        value : ArrayLike
            Scalar or 1-d array supplied by the caller.
        name : str
            Parameter name used in error messages.

        Returns
        -------
        jnp.ndarray
            Array with the same shape as ``self._wavelength``.
        """
        arr = jnp.asarray(value, dtype=float)

        if arr.ndim == 0:
            return jnp.broadcast_to(arr, self._wavelength.shape)

        if arr.shape != self._wavelength.shape:
            msg = (
                f'{name} must be a scalar or have the same shape as '
                f'wavelength {self._wavelength.shape}, '
                f'got shape {arr.shape}.'
            )
            raise ValueError(msg)

        return arr

    def R(self, wavelength: ArrayLike) -> ArrayLike:
        """Return the resolving power, interpolated onto *wavelength*."""
        return jnp.interp(wavelength, self._wavelength, self._R_grid)

    def dlam_dpix(self, wavelength: ArrayLike) -> ArrayLike:
        """Return the linear dispersion, interpolated onto *wavelength*."""
        return jnp.interp(wavelength, self._wavelength, self._dlam_dpix_grid)
