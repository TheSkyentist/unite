"""Generic disperser and spectrum implementations.

These are the building blocks for custom instruments.  Instrument-specific
loaders (NIRSpecSpectrum, SDSSSpectrum) are subclasses of
:class:`GenericSpectrum`.

Import from this module directly::

    from unite.instrument.generic import GenericDisperser, SimpleDisperser, GenericSpectrum
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from astropy import units as u
from jax.typing import ArrayLike

from unite._utils import C_KMS, _ensure_flux_density_quantity, _ensure_wavelength
from unite.instrument.base import Disperser, FluxScale, PixOffset, RScale

# ---------------------------------------------------------------------------
# Generic dispersers
# ---------------------------------------------------------------------------


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
        super().__init__(
            unit,
            name=name,
            r_scale=r_scale,
            flux_scale=flux_scale,
            pix_offset=pix_offset,
        )
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
        super().__init__(
            unit,
            name=name,
            r_scale=r_scale,
            flux_scale=flux_scale,
            pix_offset=pix_offset,
        )

        n_specified = sum(x is not None for x in (R, dlam, dvel))
        if n_specified != 1:
            msg = f'Exactly one of R, dlam, or dvel must be provided, but {n_specified} were given.'
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
            self._R_grid = C_KMS / dvel_arr

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


# ---------------------------------------------------------------------------
# Generic spectrum
# ---------------------------------------------------------------------------


class GenericSpectrum:
    """A single observed spectrum.

    A spectrum is defined by pixel bin edges (*low*, *high*), flux and error
    arrays, and a :class:`~unite.instrument.base.Disperser`.  Calibration
    parameters live on the disperser as :class:`~unite.instrument.base.CalibParam`
    tokens (``disperser.r_scale``, ``disperser.flux_scale``,
    ``disperser.pix_offset``).

    Instrument-specific subclasses (e.g. :class:`~unite.instrument.nirspec.NIRSpecSpectrum`,
    :class:`~unite.instrument.sdss.SDSSSpectrum`) extend this class with
    loaders for instrument-native file formats.

    Parameters
    ----------
    low : astropy.units.Quantity
        Lower wavelength edges of each pixel.  Must be 1-D with wavelength
        (length) dimensions.
    high : astropy.units.Quantity
        Upper wavelength edges of each pixel.  Same shape and compatible
        units as *low*.
    flux : astropy.units.Quantity
        Flux density values per pixel.  Must be 1-D with the same length
        as *low* and carry spectral flux density per wavelength units
        (f_lambda, e.g. ``erg / s / cm^2 / Angstrom``).
    error : astropy.units.Quantity
        Flux density uncertainty per pixel.  Must be 1-D with the same
        length as *low* and carry units compatible with *flux*.
    disperser : Disperser
        Instrumental disperser associated with this spectrum.  Carries any
        calibration tokens (``r_scale``, ``flux_scale``, ``pix_offset``).
    name : str, optional
        Human-readable label (e.g. ``'G235H'``).  Used in repr and for
        constructing numpyro site names.  Defaults to ``disperser.name``.

    Raises
    ------
    TypeError
        If *low* / *high* are not Quantities with wavelength dimensions,
        if *flux* / *error* are not Quantities with f_lambda dimensions,
        or if *disperser* is not a :class:`Disperser` instance.
    ValueError
        If array shapes are inconsistent or *low* ≥ *high* for any pixel.
    """

    def __init__(
        self,
        low: u.Quantity,
        high: u.Quantity,
        flux: u.Quantity,
        error: u.Quantity,
        disperser: Disperser,
        *,
        name: str = '',
    ) -> None:
        # -- flux unit --------------------------------------------------------
        flux = _ensure_flux_density_quantity(flux, 'flux', ndim=1)
        error = _ensure_flux_density_quantity(error, 'error', ndim=1)
        if not flux.unit.is_equivalent(error.unit):
            msg = f'flux and error must have compatible units, got {flux.unit!r} and {error.unit!r}.'
            raise ValueError(msg)
        self._flux_unit: u.UnitBase = flux.unit

        # -- disperser --------------------------------------------------------
        if not isinstance(disperser, Disperser):
            msg = f'disperser must be a Disperser instance, got {type(disperser).__name__}.'
            raise TypeError(msg)
        self.disperser = disperser

        # -- wavelength edges -------------------------------------------------
        low = _ensure_wavelength(low, 'low', ndim=1)
        high = _ensure_wavelength(high, 'high', ndim=1)

        if low.shape != high.shape:
            msg = f'low and high must have the same shape, got {low.shape} and {high.shape}.'
            raise ValueError(msg)

        # Store in the disperser's wavelength unit as JAX arrays.
        self._low = jnp.asarray(low.to(disperser.unit).value, dtype=float)
        self._high = jnp.asarray(high.to(disperser.unit).value, dtype=float)

        # -- flux and error ---------------------------------------------------
        # Convert error to the same unit as flux, then store bare values.
        error_converted = error.to(self._flux_unit)
        flux_arr = jnp.asarray(flux.value, dtype=float)
        error_arr = jnp.asarray(error_converted.value, dtype=float)
        npix = self._low.shape[0]

        for arr, label in ((flux_arr, 'flux'), (error_arr, 'error')):
            if arr.shape[0] != npix:
                msg = f'{label} length ({arr.shape[0]}) does not match the number of pixels ({npix}).'
                raise ValueError(msg)

        self._flux = flux_arr
        self._error = error_arr
        self._error_scale: jnp.ndarray | float = 1.0

        # -- metadata ---------------------------------------------------------
        self.name = name or disperser.name

    # -- properties -----------------------------------------------------------

    @property
    def low(self) -> jnp.ndarray:
        """Lower pixel-edge wavelengths in the disperser's unit."""
        return self._low

    @property
    def high(self) -> jnp.ndarray:
        """Upper pixel-edge wavelengths in the disperser's unit."""
        return self._high

    @property
    def wavelength(self) -> jnp.ndarray:
        """Pixel-centre wavelengths (mean of low and high edges)."""
        return (self._low + self._high) / 2.0

    @property
    def flux(self) -> jnp.ndarray:
        """Observed flux values per pixel."""
        return self._flux

    @property
    def error(self) -> jnp.ndarray:
        """Flux uncertainty per pixel."""
        return self._error

    @property
    def npix(self) -> int:
        """Number of pixels."""
        return int(self._low.shape[0])

    @property
    def unit(self) -> u.UnitBase:
        """Wavelength unit inherited from the disperser."""
        return self.disperser.unit

    @property
    def flux_unit(self) -> u.UnitBase:
        """Flux density unit (f_lambda)."""
        return self._flux_unit

    @property
    def error_scale(self) -> jnp.ndarray | float:
        """Multiplicative scale factor applied to errors.

        Can be a scalar (applied uniformly) or a per-pixel array.
        """
        return self._error_scale

    @error_scale.setter
    def error_scale(self, value: float | jnp.ndarray) -> None:
        arr = jnp.asarray(value, dtype=float)
        if arr.ndim == 0:
            if float(arr) <= 0:
                msg = f'error_scale must be > 0, got {float(arr)}'
                raise ValueError(msg)
        else:
            if arr.shape != (self.npix,):
                msg = (
                    f'error_scale array must have shape ({self.npix},), got {arr.shape}'
                )
                raise ValueError(msg)
            if bool(jnp.any(arr <= 0)):
                msg = 'error_scale values must all be > 0'
                raise ValueError(msg)
        self._error_scale = arr if arr.ndim > 0 else float(arr)

    @property
    def scaled_error(self) -> jnp.ndarray:
        """Flux uncertainty scaled by :attr:`error_scale`."""
        return self._error * self._error_scale

    @property
    def wavelength_range(self) -> tuple[float, float]:
        """``(min, max)`` wavelength in the disperser's unit."""
        return float(self._low[0]), float(self._high[-1])

    # -- calibration ----------------------------------------------------------

    @property
    def has_calibration_priors(self) -> bool:
        """``True`` if any calibration token is set on the disperser."""
        return self.disperser.has_calibration_params

    # -- coverage -------------------------------------------------------------

    def covers(self, low: float, high: float) -> bool:
        """Return ``True`` if any pixel overlaps ``[low, high]``.

        Parameters
        ----------
        low : float
            Lower bound in the disperser's unit.
        high : float
            Upper bound in the disperser's unit.
        """
        return bool(jnp.any((self._high > low) & (self._low < high)))

    def pixel_mask(self, low: float, high: float) -> jnp.ndarray:
        """Return a boolean array selecting pixels that overlap ``[low, high]``.

        Parameters
        ----------
        low : float
            Lower bound in the disperser's unit.
        high : float
            Upper bound in the disperser's unit.

        Returns
        -------
        jnp.ndarray
            Boolean array of shape ``(npix,)``.
        """
        return (self._low > low) & (self._high < high)

    # -- slicing (internal) ---------------------------------------------------

    def _sliced(self, mask: jnp.ndarray) -> GenericSpectrum:
        """Return a new spectrum of the same type with arrays selected by a boolean mask.

        Bypasses ``__init__`` validation (arrays are already validated).
        Used internally by :class:`ModelBuilder` to trim spectra to
        continuum coverage before model evaluation.

        Parameters
        ----------
        mask : jnp.ndarray
            Boolean array of shape ``(npix,)``.
        """
        new = object.__new__(type(self))
        new._low = self._low[mask]
        new._high = self._high[mask]
        new._flux = self._flux[mask]
        new._error = self._error[mask]
        new._flux_unit = self._flux_unit
        new.disperser = self.disperser
        new.name = self.name
        if isinstance(self._error_scale, (int, float)):
            new._error_scale = self._error_scale
        else:
            new._error_scale = self._error_scale[mask]
        return new

    # -- repr -----------------------------------------------------------------

    def __repr__(self) -> str:
        lo, hi = self.wavelength_range
        unit_str = self.unit.to_string()
        cls_name = type(self).__name__
        label = f'{cls_name} {self.name!r}' if self.name else cls_name
        cal = ' [calibrated]' if self.has_calibration_priors else ''
        return f'{label}: {self.npix} px, λ ∈ [{lo:.4g}, {hi:.4g}] {unit_str}{cal}'
