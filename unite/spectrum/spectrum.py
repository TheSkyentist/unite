"""Spectrum data class and multi-spectrum collection."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
from astropy import units as u
from jax.typing import ArrayLike

from unite.disperser.base import Disperser

if TYPE_CHECKING:
    from unite.continuum.config import ContinuumConfiguration, ContinuumRegion
    from unite.line.config import LineConfiguration

_C_KMS: float = 299_792.458
"""Speed of light in km/s."""


class Spectrum:
    """A single observed spectrum.

    A spectrum is defined by pixel bin edges (*low*, *high*), flux and error
    arrays, and a :class:`~unite.disperser.base.Disperser`.  Calibration
    parameters live on the disperser as :class:`~unite.disperser.base.CalibParam`
    tokens (``disperser.r_scale``, ``disperser.flux_scale``,
    ``disperser.pix_offset``).

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
        Flux uncertainty per pixel.  Must be 1-D with the same length as
        *low*.
    disperser : Disperser
        Instrumental disperser associated with this spectrum.  Carries any
        calibration tokens (``r_scale``, ``flux_scale``, ``pix_offset``).
    name : str, optional
        Human-readable label (e.g. ``'G235H'``).  Used in repr and for
        constructing numpyro site names.  Defaults to ``disperser.name``.

    Raises
    ------
    TypeError
        If *low* / *high* are not Quantities with wavelength dimensions, or
        if *disperser* is not a :class:`Disperser` instance.
    ValueError
        If array shapes are inconsistent or *low* ≥ *high* for any pixel.
    """

    def __init__(
        self,
        low: u.Quantity,
        high: u.Quantity,
        flux: ArrayLike,
        error: ArrayLike,
        disperser: Disperser,
        *,
        name: str = '',
    ) -> None:
        # -- disperser --------------------------------------------------------
        if not isinstance(disperser, Disperser):
            msg = (
                f'disperser must be a Disperser instance, '
                f'got {type(disperser).__name__}.'
            )
            raise TypeError(msg)
        self.disperser = disperser

        # -- wavelength edges -------------------------------------------------
        low = _validated_wavelength(low, 'low')
        high = _validated_wavelength(high, 'high')

        if low.shape != high.shape:
            msg = (
                f'low and high must have the same shape, '
                f'got {low.shape} and {high.shape}.'
            )
            raise ValueError(msg)

        # Store in the disperser's wavelength unit as JAX arrays.
        self._low = jnp.asarray(low.to(disperser.unit).value, dtype=float)
        self._high = jnp.asarray(high.to(disperser.unit).value, dtype=float)

        # -- flux and error ---------------------------------------------------
        flux = jnp.asarray(flux, dtype=float)
        error = jnp.asarray(error, dtype=float)
        npix = self._low.shape[0]

        for arr, label in ((flux, 'flux'), (error, 'error')):
            if arr.ndim != 1:
                msg = f'{label} must be 1-D, got {arr.ndim}-D array.'
                raise ValueError(msg)
            if arr.shape[0] != npix:
                msg = (
                    f'{label} length ({arr.shape[0]}) does not match the '
                    f'number of pixels ({npix}).'
                )
                raise ValueError(msg)

        self._flux = flux
        self._error = error

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
        return (self._high > low) & (self._low < high)

    # -- repr -----------------------------------------------------------------

    def __repr__(self) -> str:
        lo, hi = self.wavelength_range
        unit_str = self.unit.to_string()
        label = f'Spectrum {self.name!r}' if self.name else 'Spectrum'
        cal = ' [calibrated]' if self.has_calibration_priors else ''
        return f'{label}: {self.npix} px, λ ∈ [{lo:.4g}, {hi:.4g}] {unit_str}{cal}'


# ---------------------------------------------------------------------------
# Validation helper (module-private)
# ---------------------------------------------------------------------------


def _validated_wavelength(value: u.Quantity, name: str) -> u.Quantity:
    """Validate that *value* is a 1-D Quantity with wavelength dimensions."""
    if not isinstance(value, u.Quantity):
        msg = f'{name} must be an astropy Quantity, got {type(value).__name__}.'
        raise TypeError(msg)
    if value.unit.physical_type != 'length':
        msg = (
            f'{name} must have wavelength (length) dimensions, '
            f'got {value.unit.physical_type!r}.'
        )
        raise TypeError(msg)
    if value.ndim != 1:
        msg = f'{name} must be 1-D, got {value.ndim}-D array.'
        raise ValueError(msg)
    return value


# ---------------------------------------------------------------------------
# Spectra collection
# ---------------------------------------------------------------------------


class Spectra:
    """Collection of spectra with coverage filtering.

    Wraps one or more :class:`Spectrum` objects together with a systemic
    redshift estimate.  The main role of this class is :meth:`filter_config`,
    which drops lines and continuum regions not covered by any spectrum.

    Parameters
    ----------
    spectra : sequence of Spectrum
        Individual spectrum objects.  Must not be empty.
    redshift : float
        Systemic redshift estimate used for rest-frame → observed-frame
        conversion during coverage checks.  Default ``0.0``.

    Raises
    ------
    ValueError
        If *spectra* is empty or contains non-Spectrum objects.
    """

    def __init__(self, spectra: Sequence[Spectrum], redshift: float = 0.0) -> None:
        if not spectra:
            msg = 'Spectra collection must contain at least one spectrum.'
            raise ValueError(msg)
        for i, s in enumerate(spectra):
            if not isinstance(s, Spectrum):
                msg = (
                    f'All elements must be Spectrum instances; '
                    f'element {i} is {type(s).__name__}.'
                )
                raise TypeError(msg)
        self._spectra: list[Spectrum] = list(spectra)
        self._redshift: float = float(redshift)

    # -- properties -----------------------------------------------------------

    @property
    def redshift(self) -> float:
        """Systemic redshift estimate."""
        return self._redshift

    # -- coverage filtering ---------------------------------------------------

    def filter_config(
        self,
        line_config: LineConfiguration,
        continuum_config: ContinuumConfiguration | None = None,
        *,
        linedet: float = 1000.0,
    ) -> tuple[LineConfiguration, ContinuumConfiguration | None]:
        """Drop lines and continuum regions not covered by any spectrum.

        Each line's rest-frame wavelength is shifted to the observed frame
        using :attr:`redshift`, then padded by *linedet* km/s to form a
        detection window.  A line is kept if **any** spectrum partially
        overlaps that window.  Continuum regions are checked the same way.

        Parameters
        ----------
        line_config : LineConfiguration
            Line configuration to filter.
        continuum_config : ContinuumConfiguration, optional
            Continuum configuration to filter.  ``None`` is passed through.
        linedet : float
            Detection half-width in km/s.  Default ``1000`` km/s.

        Returns
        -------
        filtered_lines : LineConfiguration
        filtered_continuum : ContinuumConfiguration or None
        """
        from unite.continuum.config import ContinuumConfiguration

        eps = linedet / _C_KMS
        z = self._redshift

        # --- filter lines ---
        mask = []
        for wavelength in line_config.wavelengths:
            lam_obs = wavelength * (1.0 + z)
            covered = any(
                s.covers(lam_obs * (1.0 - eps), lam_obs * (1.0 + eps))
                for s in self._spectra
            )
            mask.append(covered)
        filtered_lines = line_config._filter(mask)

        # --- filter continuum regions ---
        if continuum_config is not None:
            kept: list[ContinuumRegion] = [
                region
                for region in continuum_config
                if any(
                    s.covers(region.low * (1.0 + z), region.high * (1.0 + z))
                    for s in self._spectra
                )
            ]
            filtered_cont: ContinuumConfiguration | None = ContinuumConfiguration(kept)
        else:
            filtered_cont = None

        return filtered_lines, filtered_cont

    # -- container interface --------------------------------------------------

    def __len__(self) -> int:
        return len(self._spectra)

    def __iter__(self) -> Iterator[Spectrum]:
        return iter(self._spectra)

    def __getitem__(self, idx: int) -> Spectrum:
        return self._spectra[idx]

    def __repr__(self) -> str:
        lines = [f'Spectra: {len(self._spectra)} spectrum/a, z={self._redshift:.4f}']
        for i, s in enumerate(self._spectra):
            lo, hi = s.wavelength_range
            unit_str = s.unit.to_string()
            label = s.name or f'#{i}'
            cal = ' [calibrated]' if s.has_calibration_priors else ''
            lines.append(
                f'  [{i}] {label}: {s.npix} px, '
                f'λ ∈ [{lo:.4g}, {hi:.4g}] {unit_str}{cal}'
            )
        return '\n'.join(lines)
