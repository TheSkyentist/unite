"""Spectrum data class and multi-spectrum collection."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import jax.numpy as jnp
from astropy import units as u

from unite._utils import (
    C_KMS,
    _ensure_flux_density_quantity,
    _ensure_velocity,
    _ensure_wavelength,
    _flux_density_conversion_factor,
)
from unite.disperser.base import Disperser

if TYPE_CHECKING:
    from unite.continuum.config import ContinuumConfiguration
    from unite.line.config import LineConfiguration


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
            msg = (
                f'flux and error must have compatible units, '
                f'got {flux.unit!r} and {error.unit!r}.'
            )
            raise ValueError(msg)
        self._flux_unit: u.UnitBase = flux.unit

        # -- disperser --------------------------------------------------------
        if not isinstance(disperser, Disperser):
            msg = (
                f'disperser must be a Disperser instance, '
                f'got {type(disperser).__name__}.'
            )
            raise TypeError(msg)
        self.disperser = disperser

        # -- wavelength edges -------------------------------------------------
        low = _ensure_wavelength(low, 'low', ndim=1)
        high = _ensure_wavelength(high, 'high', ndim=1)

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
        # Convert error to the same unit as flux, then store bare values.
        error_converted = error.to(self._flux_unit)
        flux_arr = jnp.asarray(flux.value, dtype=float)
        error_arr = jnp.asarray(error_converted.value, dtype=float)
        npix = self._low.shape[0]

        for arr, label in ((flux_arr, 'flux'), (error_arr, 'error')):
            if arr.shape[0] != npix:
                msg = (
                    f'{label} length ({arr.shape[0]}) does not match the '
                    f'number of pixels ({npix}).'
                )
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
        return (self._high > low) & (self._low < high)

    # -- slicing (internal) ---------------------------------------------------

    def _sliced(self, mask: jnp.ndarray) -> Spectrum:
        """Return a new Spectrum with arrays selected by a boolean mask.

        Bypasses ``__init__`` validation (arrays are already validated).
        Used internally by :class:`ModelBuilder` to trim spectra to
        continuum coverage before model evaluation.

        Parameters
        ----------
        mask : jnp.ndarray
            Boolean array of shape ``(npix,)``.
        """
        new = object.__new__(Spectrum)
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
        label = f'Spectrum {self.name!r}' if self.name else 'Spectrum'
        cal = ' [calibrated]' if self.has_calibration_priors else ''
        return f'{label}: {self.npix} px, λ ∈ [{lo:.4g}, {hi:.4g}] {unit_str}{cal}'


# ---------------------------------------------------------------------------
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
    canonical_unit : astropy.units.UnitBase, optional
        Wavelength unit used for all internal model computations (line
        centres, continuum bounds, pixel edges).  Defaults to the first
        spectrum's disperser unit.  Override this if you want a specific
        unit regardless of spectrum order.

    Raises
    ------
    ValueError
        If *spectra* is empty or contains non-Spectrum objects.
    """

    def __init__(
        self,
        spectra: Sequence[Spectrum],
        redshift: float = 0.0,
        canonical_unit: u.UnitBase | None = None,
    ) -> None:
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
        self._line_scale: u.Quantity | None = None
        self._continuum_scale: u.Quantity | None = None
        self._is_prepared: bool = False
        self._prepared_line_config: LineConfiguration | None = None
        self._prepared_cont_config: ContinuumConfiguration | None = None

        # Canonical wavelength unit: default to the first spectrum's unit.
        if canonical_unit is not None:
            canonical_unit = u.Unit(canonical_unit)
            if not canonical_unit.is_equivalent(u.m):
                msg = (
                    f'canonical_unit must have wavelength dimensions, '
                    f'got {canonical_unit!r}.'
                )
                raise u.UnitConversionError(msg)
            self._canonical_unit: u.UnitBase = canonical_unit
        else:
            self._canonical_unit = self._spectra[0].unit

    # -- properties -----------------------------------------------------------

    @property
    def redshift(self) -> float:
        """Systemic redshift estimate."""
        return self._redshift

    @property
    def canonical_unit(self) -> u.UnitBase:
        """Canonical wavelength unit used for all internal model computations.

        Defaults to the first spectrum's wavelength unit.  Can be overridden
        via the *canonical_unit* constructor parameter.
        """
        return self._canonical_unit

    @property
    def line_scale(self) -> u.Quantity | None:
        """Characteristic line flux scale (integrated flux), or ``None``."""
        return self._line_scale

    @line_scale.setter
    def line_scale(self, value: u.Quantity) -> None:
        if not isinstance(value, u.Quantity):
            msg = (
                f'line_scale must be an astropy Quantity with flux units '
                f'(flux_density * wavelength), got {type(value).__name__}.'
            )
            raise TypeError(msg)
        # Validate unit: must be flux_density * wavelength (integrated flux).
        ref = u.erg / u.s / u.cm**2
        if not value.unit.is_equivalent(ref):
            msg = (
                f'line_scale must have integrated flux units '
                f'(e.g. erg/s/cm^2), got {value.unit!r}.'
            )
            raise u.UnitConversionError(msg)
        if value.value <= 0:
            msg = f'line_scale must be > 0, got {value}'
            raise ValueError(msg)
        self._line_scale = value

    @property
    def continuum_scale(self) -> u.Quantity | None:
        """Characteristic continuum flux density scale, or ``None``."""
        return self._continuum_scale

    @continuum_scale.setter
    def continuum_scale(self, value: u.Quantity) -> None:
        if not isinstance(value, u.Quantity):
            msg = (
                f'continuum_scale must be an astropy Quantity with flux '
                f'density units, got {type(value).__name__}.'
            )
            raise TypeError(msg)
        ref = u.erg / u.s / u.cm**2 / u.AA
        if not value.unit.is_equivalent(ref):
            msg = (
                f'continuum_scale must have flux density units '
                f'(e.g. erg/s/cm^2/Angstrom), got {value.unit!r}.'
            )
            raise u.UnitConversionError(msg)
        if value.value <= 0:
            msg = f'continuum_scale must be > 0, got {value}'
            raise ValueError(msg)
        self._continuum_scale = value

    def compute_scales(
        self,
        line_config: LineConfiguration,
        continuum_config: ContinuumConfiguration | None = None,
        *,
        max_fwhm: u.Quantity = 1000.0 * u.km / u.s,
        line_mask_fwhm: u.Quantity = 1000.0 * u.km / u.s,
        error_scale: bool = False,
    ) -> None:
        """Estimate characteristic flux scales for lines and continuum.

        The algorithm:

        1. Mask pixels near emission lines (using *line_mask_fwhm*, a full
           FWHM convolved in quadrature with the local LSF).
        2. Fit a low-order polynomial to the unmasked pixels in each
           continuum region.
        3. Estimate the **line scale** as ``peak_above_continuum * line_width``
           where the peak is measured relative to the fitted continuum.
        4. Estimate the **continuum scale** as the maximum median ``|flux|``
           across continuum regions (after masking lines).
        5. Optionally compute per-region **error scale** factors
           (``sqrt(max(chi2_red, 1.0))``) and store them as per-pixel
           arrays on each :class:`Spectrum`.

        Both flux scales are stored as :class:`~astropy.units.Quantity`
        objects: integrated flux for ``line_scale`` and flux density for
        ``continuum_scale``.

        Parameters
        ----------
        line_config : LineConfiguration
            Line configuration.
        continuum_config : ContinuumConfiguration, optional
            Continuum configuration.  If ``None``, only line scale is
            computed (without continuum subtraction).
        max_fwhm : astropy.units.Quantity
            Maximum expected intrinsic line FWHM (full width).  Must have
            velocity units.  Default ``1000 km/s``.
        line_mask_fwhm : astropy.units.Quantity
            Full FWHM used for masking lines when fitting continuum.  Must
            have velocity units.  Default ``1000 km/s``.
        error_scale : bool
            If ``True``, compute per-region error scale factors from the
            reduced chi-squared of the continuum fit residuals and store
            them as per-pixel arrays on each :class:`Spectrum` via
            :attr:`Spectrum.error_scale`.  Default ``False``.
        """
        import numpy as np

        max_fwhm = _ensure_velocity(max_fwhm, 'max_fwhm')
        line_mask_fwhm = _ensure_velocity(line_mask_fwhm, 'line_mask_fwhm')
        max_fwhm_kms = float(max_fwhm.to(u.km / u.s).value)
        mask_fwhm_kms = float(line_mask_fwhm.to(u.km / u.s).value)
        from unite._utils import _wavelength_conversion_factor

        z = self._redshift
        ref_flux_unit = self._spectra[0].flux_unit
        ref_wl_unit = self._canonical_unit

        # --- Helper: build a per-pixel line exclusion mask for a spectrum ---
        def _build_line_mask(spectrum, fwhm_kms):
            wl = spectrum.wavelength
            mask = jnp.zeros(spectrum.npix, dtype=bool)
            for lam_rest in line_config.wavelengths:
                lam_obs = float(lam_rest.to(spectrum.unit).value) * (1.0 + z)
                if lam_obs <= 0:
                    continue
                lsf_fwhm = lam_obs / float(spectrum.disperser.R(lam_obs))
                user_hw = lam_obs * fwhm_kms / C_KMS / 2.0  # half of FWHM
                half_width = float(jnp.sqrt(user_hw**2 + (lsf_fwhm / 2.0) ** 2))
                mask = mask | (
                    (wl > lam_obs - half_width) & (wl < lam_obs + half_width)
                )
            return mask

        # --- Helper: fit continuum form in a region ---
        def _fit_continuum_region(wl, flux, error, obs_low, obs_high, line_mask, form):
            """Fit the region's continuum form to unmasked pixels.

            Returns (model_values_on_region, good_mask, chi2_red).
            model_values_on_region covers all pixels in_region.
            """
            from unite.continuum.fit import fit_continuum_form

            in_region = (wl >= obs_low) & (wl <= obs_high)
            good = in_region & ~line_mask
            n_good = int(jnp.sum(good))

            min_params = form.n_params  # includes normalization_wavelength
            if n_good < max(min_params + 1, 3):
                return None, in_region, None

            center = float((obs_low + obs_high) / 2.0)
            adapted_form = form._adapt_for_observed_region(obs_low, obs_high)
            result = fit_continuum_form(
                adapted_form, wl[good], flux[good], error[good], center,
            )

            # Evaluate on all in-region pixels.
            model_region = adapted_form.evaluate(
                wl[in_region], center, result.params,
            )

            return model_region, in_region, result.chi2_red

        # --- Line scale (with continuum subtraction when available) ---
        max_line_scale = 0.0
        for spectrum in self._spectra:
            wl = spectrum.wavelength
            flux_conv = _flux_density_conversion_factor(
                spectrum.flux_unit, ref_flux_unit
            )
            wl_conv = _wavelength_conversion_factor(spectrum.unit, ref_wl_unit)
            line_mask = _build_line_mask(spectrum, mask_fwhm_kms)

            # Build a full-spectrum continuum estimate for subtraction.
            continuum_est = jnp.zeros(spectrum.npix)
            if continuum_config is not None:
                for region in continuum_config:
                    conv = _wavelength_conversion_factor(region._unit, spectrum.unit)
                    obs_low = region.low * conv * (1.0 + z)
                    obs_high = region.high * conv * (1.0 + z)
                    result = _fit_continuum_region(
                        wl, spectrum.flux, spectrum.error, obs_low, obs_high,
                        line_mask, region.form,
                    )
                    model_region, in_region, _ = result
                    if model_region is not None:
                        continuum_est = continuum_est.at[in_region].set(model_region)

            # Measure peak height above continuum for each line.
            for lam_rest in line_config.wavelengths:
                lam_obs = float(lam_rest.to(spectrum.unit).value) * (1.0 + z)
                if lam_obs <= 0:
                    continue
                lsf_fwhm = lam_obs / float(spectrum.disperser.R(lam_obs))
                user_fwhm = lam_obs * max_fwhm_kms / C_KMS
                total_fwhm = float(jnp.sqrt(user_fwhm**2 + lsf_fwhm**2))
                in_window = (wl >= lam_obs - total_fwhm) & (wl <= lam_obs + total_fwhm)
                if not jnp.any(in_window):
                    continue
                peak_above = float(
                    jnp.max(
                        jnp.abs(spectrum.flux[in_window] - continuum_est[in_window])
                    )
                )
                flux_est = peak_above * flux_conv * total_fwhm * wl_conv
                max_line_scale = max(max_line_scale, flux_est)

        line_scale_val = max_line_scale if max_line_scale > 0 else 1.0
        self._line_scale = line_scale_val * ref_flux_unit * ref_wl_unit

        # --- Continuum scale and optional error scale ---
        if continuum_config is not None:
            max_cont_scale = 0.0
            for spectrum in self._spectra:
                wl = spectrum.wavelength
                flux_conv = _flux_density_conversion_factor(
                    spectrum.flux_unit, ref_flux_unit
                )
                line_mask = _build_line_mask(spectrum, mask_fwhm_kms)
                per_pixel_scale = jnp.ones(spectrum.npix)
                all_chi2_reds: list[float] = []

                for region in continuum_config:
                    conv = _wavelength_conversion_factor(region._unit, spectrum.unit)
                    obs_low = region.low * conv * (1.0 + z)
                    obs_high = region.high * conv * (1.0 + z)

                    in_region = (wl >= obs_low) & (wl <= obs_high)
                    if not jnp.any(in_region):
                        continue

                    good = in_region & ~line_mask
                    if jnp.any(good):
                        median_flux = float(jnp.median(jnp.abs(spectrum.flux[good])))
                        max_cont_scale = max(max_cont_scale, median_flux * flux_conv)

                    if error_scale:
                        result = _fit_continuum_region(
                            wl,
                            spectrum.flux,
                            spectrum.error,
                            obs_low,
                            obs_high,
                            line_mask,
                            region.form,
                        )
                        _, fit_region, chi2_red = result
                        if chi2_red is not None:
                            all_chi2_reds.append(chi2_red)
                            region_scale = float(jnp.sqrt(jnp.maximum(chi2_red, 1.0)))
                            per_pixel_scale = jnp.where(
                                fit_region, region_scale, per_pixel_scale
                            )

                if error_scale and all_chi2_reds:
                    # For pixels not in any continuum region, use the median
                    # scale across all regions.
                    median_chi2 = float(np.median(all_chi2_reds))
                    fallback_scale = float(jnp.sqrt(jnp.maximum(median_chi2, 1.0)))
                    no_region = per_pixel_scale == 1.0
                    per_pixel_scale = jnp.where(
                        no_region, fallback_scale, per_pixel_scale
                    )
                    spectrum.error_scale = per_pixel_scale

            cont_scale_val = max_cont_scale if max_cont_scale > 0 else 1.0
            self._continuum_scale = cont_scale_val * ref_flux_unit

    # -- preparation ----------------------------------------------------------

    def prepare(
        self,
        line_config: LineConfiguration,
        continuum_config: ContinuumConfiguration | None = None,
        *,
        linedet: u.Quantity = 1000.0 * u.km / u.s,
        drop_empty_regions: bool = True,
    ) -> tuple[LineConfiguration, ContinuumConfiguration | None]:
        """Filter configs for coverage and optionally drop empty continuum regions.

        Calls :meth:`filter_config` to remove lines and continuum regions not
        covered by any spectrum, then optionally drops continuum regions that
        contain no emission lines.  Stores the filtered configs on this
        object.

        Parameters
        ----------
        line_config : LineConfiguration
            Line configuration.
        continuum_config : ContinuumConfiguration, optional
            Continuum configuration.
        linedet : astropy.units.Quantity
            Detection half-width.  Must have velocity units.
            Default ``1000 km/s``.
        drop_empty_regions : bool
            If ``True`` (default), drop continuum regions that do not contain
            any filtered line (in rest frame).

        Returns
        -------
        filtered_lines : LineConfiguration
        filtered_continuum : ContinuumConfiguration or None
        """
        from unite.continuum.config import ContinuumConfiguration

        filtered_lines, filtered_cont = self.filter_config(
            line_config, continuum_config, linedet=linedet
        )

        if drop_empty_regions and filtered_cont is not None and len(filtered_lines) > 0:
            kept = []
            for region in filtered_cont:
                # Check if any filtered line falls in this region (rest frame).
                has_line = False
                for lam_rest in filtered_lines.wavelengths:
                    lam_val = float(lam_rest.to(region._unit).value)
                    if region.low <= lam_val <= region.high:
                        has_line = True
                        break
                if has_line:
                    kept.append(region)
            filtered_cont = ContinuumConfiguration(kept) if kept else None

        self._prepared_line_config = filtered_lines
        self._prepared_cont_config = filtered_cont
        self._is_prepared = True

        return filtered_lines, filtered_cont

    @property
    def is_prepared(self) -> bool:
        """``True`` if :meth:`prepare` has been called."""
        return self._is_prepared

    @property
    def prepared_line_config(self) -> LineConfiguration | None:
        """Filtered line configuration from :meth:`prepare`, or ``None``."""
        return self._prepared_line_config

    @property
    def prepared_cont_config(self) -> ContinuumConfiguration | None:
        """Filtered continuum configuration from :meth:`prepare`, or ``None``."""
        return self._prepared_cont_config

    # -- coverage filtering ---------------------------------------------------

    def filter_config(
        self,
        line_config: LineConfiguration,
        continuum_config: ContinuumConfiguration | None = None,
        *,
        linedet: u.Quantity = 1000.0 * u.km / u.s,
    ) -> tuple[LineConfiguration, ContinuumConfiguration | None]:
        """Drop lines and continuum regions not covered by any spectrum.

        Each line's rest-frame wavelength is shifted to the observed frame
        using :attr:`redshift`, then padded by *linedet* to form a
        detection window.  A line is kept if **any** spectrum partially
        overlaps that window.  Continuum regions are checked the same way.

        Parameters
        ----------
        line_config : LineConfiguration
            Line configuration to filter.
        continuum_config : ContinuumConfiguration, optional
            Continuum configuration to filter.  ``None`` is passed through.
        linedet : astropy.units.Quantity
            Detection half-width.  Must have velocity units.
            Default ``1000 km/s``.

        Returns
        -------
        filtered_lines : LineConfiguration
        filtered_continuum : ContinuumConfiguration or None
        """
        from unite._utils import _wavelength_conversion_factor
        from unite.continuum.config import ContinuumConfiguration

        linedet = _ensure_velocity(linedet, 'linedet')
        eps = float(linedet.to(u.km / u.s).value) / C_KMS
        z = self._redshift

        # --- filter lines ---
        mask = []
        for wavelength in line_config.wavelengths:
            # wavelength is a Quantity; convert to each spectrum's disperser unit
            lam_obs = wavelength * (1.0 + z)
            covered = any(
                s.covers(
                    float(lam_obs.to(s.unit).value) * (1.0 - eps),
                    float(lam_obs.to(s.unit).value) * (1.0 + eps),
                )
                for s in self._spectra
            )
            mask.append(covered)
        filtered_lines = line_config._filter(mask)

        # --- filter continuum regions ---
        if continuum_config is not None:
            kept = []
            for region in continuum_config:
                # Convert region bounds (stored as bare floats in region._unit)
                # to each spectrum's disperser unit before checking coverage.
                region_covered = False
                for s in self._spectra:
                    conv = _wavelength_conversion_factor(region._unit, s.unit)
                    obs_low = region.low * conv * (1.0 + z)
                    obs_high = region.high * conv * (1.0 + z)
                    if s.covers(obs_low, obs_high):
                        region_covered = True
                        break
                if region_covered:
                    kept.append(region)
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
