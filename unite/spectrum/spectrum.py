"""The Spectrum class — a single observed spectrum."""

from __future__ import annotations

import jax.numpy as jnp
from astropy import units as u

from unite._utils import _ensure_flux_density, _ensure_wavelength
from unite.instrument.base import Disperser


class Spectrum:
    """A single observed spectrum.

    A spectrum is defined by pixel bin edges (*low*, *high*), flux and error
    arrays, and a :class:`~unite.instrument.base.Disperser`.  Calibration
    parameters live on the disperser as
    :class:`~unite.instrument.base.CalibParam` tokens (``disperser.r_scale``,
    ``disperser.flux_scale``, ``disperser.pix_offset``).

    Use :func:`~unite.spectrum.from_arrays`, :func:`~unite.spectrum.from_DJA`,
    or :func:`~unite.spectrum.from_sdss_fits` to construct spectra from arrays
    or instrument-native file formats.

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
        flux = _ensure_flux_density(flux, 'flux', ndim=1)
        error = _ensure_flux_density(error, 'error', ndim=1)
        _flux_unit = flux.unit
        if not _flux_unit.is_equivalent(error.unit):
            msg = f'flux and error must have compatible units, got {flux.unit!r} and {error.unit!r}.'
            raise ValueError(msg)
        self._flux_unit: u.UnitBase = _flux_unit

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
        self._scale_diagnostic: object = None

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
        """Pixel-center wavelengths (mean of low and high edges)."""
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
    def scale_diagnostic(self):
        """Continuum-fit diagnostics from the most recent :meth:`~unite.spectrum.Spectra.compute_scales` call.

        Returns a :class:`~unite.spectrum.SpectrumScaleDiagnostic`
        holding the line mask, the fitted continuum model array, and
        per-region fit details.  ``None`` if
        :meth:`~unite.spectrum.Spectra.compute_scales` has not been
        called yet.

        The spectrum's own :attr:`wavelength`, :attr:`flux`, :attr:`error`, and
        unit attributes provide the full picture alongside this diagnostic.

        Examples
        --------
        >>> diag = spectrum.scale_diagnostic
        >>> if diag is not None:
        ...     cont = diag.continuum_model   # NaN outside fitted regions
        ...     mask = diag.line_mask         # True where a pixel was excluded
        """
        return self._scale_diagnostic

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

    def _sliced(self, mask: jnp.ndarray) -> Spectrum:
        """Return a new spectrum with arrays selected by a boolean mask.

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


# Re-export ArrayLike for type hints in downstream code (keeps it importable).
__all__ = ['Spectrum']
