"""Spectrum collection and scale diagnostics."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax.numpy as jnp
from astropy import units as u

from unite._utils import C_KMS, _ensure_velocity, _get_conversion_factor
from unite.instrument.generic import GenericSpectrum

if TYPE_CHECKING:
    from unite.continuum.config import ContinuumConfiguration
    from unite.line.config import LineConfiguration


# ---------------------------------------------------------------------------
# Scale diagnostics dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RegionDiagnostic:
    """Diagnostics for a single continuum region fit.

    Attributes
    ----------
    obs_low : float
        Observed-frame lower bound of the region (disperser's unit).
    obs_high : float
        Observed-frame upper bound of the region (disperser's unit).
    in_region : jnp.ndarray
        Boolean mask of shape ``(npix,)`` selecting all pixels inside the
        region bounds.
    good_mask : jnp.ndarray
        Boolean mask of shape ``(npix,)`` selecting pixels used for
        fitting (``in_region & ~line_mask``).
    model_on_region : jnp.ndarray or None
        Best-fit continuum model evaluated at the ``in_region`` pixels.
        ``None`` if the fit failed (too few unmasked pixels).
    chi2_red : float or None
        Reduced chi-squared of the fit, or ``None`` if the fit failed.
    fit_params : dict of str to float
        Best-fit parameter dict returned by :func:`~unite.continuum.fit.fit_continuum_form`.
        Empty if the fit failed.
    """

    obs_low: float
    obs_high: float
    in_region: jnp.ndarray
    good_mask: jnp.ndarray
    model_on_region: jnp.ndarray | None
    chi2_red: float | None
    fit_params: dict = field(default_factory=dict)


class ScaleDiagnosticList(list):
    """A list of :class:`SpectrumScaleDiagnostic` objects that also supports lookup by spectrum name.

    Inherits all standard :class:`list` behaviour.  Additionally,
    ``diags["my_spectrum"]`` returns the first diagnostic whose
    :attr:`~SpectrumScaleDiagnostic.name` matches the given string,
    raising :exc:`KeyError` if no match is found.
    """

    def __getitem__(self, idx):
        if isinstance(idx, str):
            for diag in self:
                if diag.name == idx:
                    return diag
            msg = f'No scale diagnostic found for spectrum name {idx!r}.'
            raise KeyError(msg)
        result = super().__getitem__(idx)
        if isinstance(idx, slice):
            result = ScaleDiagnosticList(result)
        return result


@dataclass
class SpectrumScaleDiagnostic:
    """Diagnostics for one spectrum produced by :meth:`Spectra.compute_scales`.

    Attributes
    ----------
    name : str
        Spectrum name (from :attr:`GenericSpectrum.name`).
    wavelength : jnp.ndarray
        Pixel-centre wavelengths (disperser's unit), shape ``(npix,)``.
    flux : jnp.ndarray
        Observed flux values, shape ``(npix,)``.
    error : jnp.ndarray
        Flux uncertainty values, shape ``(npix,)``.
    line_mask : jnp.ndarray
        Boolean mask of shape ``(npix,)``; ``True`` where a pixel was
        excluded because it lies near an emission line.
    continuum_model : jnp.ndarray
        Full-spectrum continuum model array of shape ``(npix,)``.
        Pixels not covered by any continuum region are ``NaN``.
    regions : list of RegionDiagnostic
        Per-region fit diagnostics (one entry per region that overlaps
        this spectrum).
    flux_unit : astropy.units.UnitBase
        Flux density unit of *flux* and *error*.
    wavelength_unit : astropy.units.UnitBase
        Wavelength unit of *wavelength*.
    """

    name: str
    wavelength: jnp.ndarray
    flux: jnp.ndarray
    error: jnp.ndarray
    line_mask: jnp.ndarray
    continuum_model: jnp.ndarray
    regions: list
    flux_unit: object
    wavelength_unit: object


# ---------------------------------------------------------------------------
# Spectra collection
# ---------------------------------------------------------------------------


class Spectra:
    """Collection of spectra with coverage filtering.

    Wraps one or more :class:`~unite.instrument.generic.GenericSpectrum` objects
    together with a systemic redshift estimate.  The main role of this class is
    :meth:`filter_config`, which drops lines and continuum regions not covered
    by any spectrum.

    Parameters
    ----------
    spectra : sequence of GenericSpectrum
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
        If *spectra* is empty or contains non-GenericSpectrum objects.
    """

    def __init__(
        self,
        spectra: Sequence[GenericSpectrum],
        redshift: float = 0.0,
        canonical_unit: u.UnitBase | None = None,
    ) -> None:
        if not spectra:
            msg = 'Spectra collection must contain at least one spectrum.'
            raise ValueError(msg)
        for i, s in enumerate(spectra):
            if not isinstance(s, GenericSpectrum):
                msg = f'All elements must be GenericSpectrum instances; element {i} is {type(s).__name__}.'
                raise TypeError(msg)
        self._spectra: list[GenericSpectrum] = list(spectra)
        self._redshift: float = float(redshift)
        self._line_scale: u.Quantity | None = None
        self._continuum_scale: u.Quantity | None = None
        self._is_prepared: bool = False
        self._prepared_line_config: LineConfiguration | None = None
        self._prepared_cont_config: ContinuumConfiguration | None = None
        self._scale_diagnostics: ScaleDiagnosticList | None = None

        # Assign names to any anonymous calibration tokens on the dispersers.
        # Local import avoids circular dependency at module load time.
        from unite.instrument.config import InstrumentConfig

        seen_ids: dict[int, object] = {}
        for s in self._spectra:
            if id(s.disperser) not in seen_ids:
                seen_ids[id(s.disperser)] = s.disperser
        InstrumentConfig(list(seen_ids.values()))

        # Canonical wavelength unit: default to the first spectrum's unit.
        if canonical_unit is not None:
            canonical_unit = u.Unit(canonical_unit)
            if not canonical_unit.is_equivalent(u.m):
                msg = f'canonical_unit must have wavelength dimensions, got {canonical_unit!r}.'
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
            msg = f'line_scale must have integrated flux units (e.g. erg/s/cm^2), got {value.unit!r}.'
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
            msg = f'continuum_scale must be an astropy Quantity with flux density units, got {type(value).__name__}.'
            raise TypeError(msg)
        ref = u.erg / u.s / u.cm**2 / u.AA
        if not value.unit.is_equivalent(ref):
            msg = f'continuum_scale must have flux density units (e.g. erg/s/cm^2/Angstrom), got {value.unit!r}.'
            raise u.UnitConversionError(msg)
        if value.value <= 0:
            msg = f'continuum_scale must be > 0, got {value}'
            raise ValueError(msg)
        self._continuum_scale = value

    @property
    def scale_diagnostics(self) -> ScaleDiagnosticList | None:
        """Per-spectrum diagnostics from the most recent :meth:`compute_scales` call.

        Returns a :class:`ScaleDiagnosticList` of
        :class:`SpectrumScaleDiagnostic` objects (one per spectrum), each
        holding the line mask, the fitted continuum model array, and
        per-region fit details.  Supports both integer indexing and
        lookup by spectrum name (e.g. ``diags["my_spectrum"]``).
        ``None`` if :meth:`compute_scales` has not been called yet.
        """
        return self._scale_diagnostics

    def compute_scales(
        self,
        line_config: LineConfiguration,
        continuum_config: ContinuumConfiguration | None = None,
        *,
        line_mask_width: u.Quantity = 1000.0 * u.km / u.s,
        box_width: u.Quantity = 1000.0 * u.km / u.s,
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
           (``sqrt(chi2_red)``) and store them as per-pixel
           arrays on each :class:`~unite.instrument.generic.GenericSpectrum`.

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
            them as per-pixel arrays on each spectrum via
            :attr:`~unite.instrument.generic.GenericSpectrum.error_scale`.
            Default ``False``.
        """
        box_width = _ensure_velocity(box_width, 'max_fwhm', ndim=0)
        line_mask_width = _ensure_velocity(line_mask_width, 'line_mask_fwhm', ndim=0)
        box_width_kms = float(box_width.to(u.km / u.s).value)
        mask_width_kms = float(line_mask_width.to(u.km / u.s).value)

        z = self._redshift
        ref_flux_unit = self._spectra[0].flux_unit
        ref_wl_unit = self._canonical_unit

        # --- Helper: build a per-pixel line exclusion mask for a spectrum ---
        def _build_line_mask(spectrum, fwhm_kms):
            wl = spectrum.wavelength
            mask = jnp.zeros(spectrum.npix, dtype=bool)
            for lam_rest in line_config.wavelengths:
                lam_obs = float(lam_rest.to(spectrum.unit).value) * (1.0 + z)
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

            Returns (model_on_region, in_region, good, chi2_red, fit_params).
            model_on_region covers all pixels in_region; good is the
            unmasked pixel mask used for fitting.
            """
            from unite.continuum.fit import fit_continuum_form

            in_region = (wl >= obs_low) & (wl <= obs_high)
            good = in_region & ~line_mask
            n_good = int(jnp.sum(good))

            min_params = form.n_params  # includes normalization_wavelength
            if n_good < max(min_params + 1, 3):
                return None, in_region, good, None, {}

            center = float((obs_low + obs_high) / 2.0)
            result = fit_continuum_form(
                form, wl[good], flux[good], error[good], center, obs_low, obs_high
            )

            # Evaluate on all in-region pixels.
            model_region = form.evaluate(
                wl[in_region], center, result.params, obs_low, obs_high
            )

            return model_region, in_region, good, result.chi2_red, result.params

        # --- Single pass: fit continuum once per region per spectrum, then
        #     derive line scale, continuum scale, error scale, and diagnostics.
        max_line_scale = 0.0
        max_cont_scale = 0.0
        any_good_continuum_pixels = False
        any_successful_fit = False
        diag_list = []

        for spectrum in self._spectra:
            wl = spectrum.wavelength
            flux_conv = _get_conversion_factor(spectrum.flux_unit, ref_flux_unit)
            wl_conv = _get_conversion_factor(spectrum.unit, ref_wl_unit)
            line_mask = _build_line_mask(spectrum, mask_width_kms)

            continuum_model = jnp.full(spectrum.npix, jnp.nan)
            per_pixel_scale = jnp.ones(spectrum.npix)
            region_diags: list[RegionDiagnostic] = []

            if continuum_config is not None:
                for region in continuum_config:
                    conv = _get_conversion_factor(region._unit, spectrum.unit)
                    obs_low = region.low * conv * (1.0 + z)
                    obs_high = region.high * conv * (1.0 + z)

                    in_region = (wl >= obs_low) & (wl <= obs_high)
                    if not jnp.any(in_region):
                        continue

                    model_region, in_region, good, chi2_red, fit_params = (
                        _fit_continuum_region(
                            wl,
                            spectrum.flux,
                            spectrum.error,
                            obs_low,
                            obs_high,
                            line_mask,
                            region.form,
                        )
                    )

                    # Continuum estimate for line-scale peak subtraction.
                    if model_region is not None:
                        continuum_model = continuum_model.at[in_region].set(
                            model_region
                        )
                        any_successful_fit = True

                    # Continuum scale: median |flux| of unmasked pixels.
                    if jnp.any(good):
                        any_good_continuum_pixels = True
                        median_flux = float(jnp.median(jnp.abs(spectrum.flux[good])))
                        max_cont_scale = max(max_cont_scale, median_flux * flux_conv)

                    # Error scale: per-pixel sqrt(chi2_red) from fit residuals.
                    if error_scale and chi2_red is not None:
                        region_scale = float(jnp.sqrt(chi2_red))
                        per_pixel_scale = jnp.where(
                            in_region, region_scale, per_pixel_scale
                        )

                    region_diags.append(
                        RegionDiagnostic(
                            obs_low=float(obs_low),
                            obs_high=float(obs_high),
                            in_region=in_region,
                            good_mask=good,
                            model_on_region=model_region,
                            chi2_red=chi2_red,
                            fit_params=fit_params,
                        )
                    )

            if error_scale and (continuum_config is not None):
                spectrum.error_scale = per_pixel_scale

            # Line scale: peak above continuum within ±total_fwhm of each line.
            # Where continuum is not fitted (NaN), assume zero baseline.
            baseline = jnp.where(jnp.isnan(continuum_model), 0.0, continuum_model)
            for lam_rest in line_config.wavelengths:
                lam_obs = float(lam_rest.to(spectrum.unit).value) * (1.0 + z)
                lsf_fwhm = lam_obs / float(spectrum.disperser.R(lam_obs))
                mask_width_lam = lam_obs * mask_width_kms / C_KMS
                half_width = float(jnp.sqrt(mask_width_lam**2 + lsf_fwhm**2)) / 2
                in_window = (wl >= lam_obs - half_width) & (wl <= lam_obs + half_width)
                if not jnp.any(in_window):
                    continue
                peak_above = float(
                    jnp.max(spectrum.flux[in_window] - baseline[in_window])
                )
                box_width_lam = box_width_kms * lam_obs / C_KMS
                flux_est = peak_above * flux_conv * box_width_lam * wl_conv
                max_line_scale = max(max_line_scale, flux_est)

            diag_list.append(
                SpectrumScaleDiagnostic(
                    name=spectrum.name,
                    wavelength=wl,
                    flux=spectrum.flux,
                    error=spectrum.error,
                    line_mask=line_mask,
                    continuum_model=continuum_model,
                    regions=region_diags,
                    flux_unit=spectrum.flux_unit,
                    wavelength_unit=spectrum.unit,
                )
            )

        if max_line_scale <= 0:
            msg = (
                'Could not estimate line flux scale: no emission line peak '
                'was found above the continuum in any spectrum. This can '
                'happen if (a) the configured lines fall outside the spectral '
                'coverage, (b) the continuum model over-subtracts the flux, '
                'or (c) the lines are too faint relative to noise. '
                'Check that line_config wavelengths match the observed data, '
                'or set spectra.line_scale manually.'
            )
            raise ValueError(msg)
        self._line_scale = max_line_scale * ref_flux_unit * ref_wl_unit

        if continuum_config is not None:
            if not any_good_continuum_pixels:
                msg = (
                    'Could not estimate continuum scale: the line mask covers '
                    'all pixels in every continuum region, leaving no unmasked '
                    'pixels for fitting. Try reducing line_mask_width (currently '
                    f'{line_mask_width}) or widening the continuum regions.'
                )
                raise ValueError(msg)
            if not any_successful_fit:
                msg = (
                    'Could not estimate continuum scale: all continuum region '
                    'fits failed (too few unmasked pixels in every region after '
                    'line masking). Try reducing line_mask_width (currently '
                    f'{line_mask_width}) or widening the continuum regions.'
                )
                raise ValueError(msg)
            if max_cont_scale <= 0:
                msg = (
                    'Could not estimate continuum scale: the median |flux| in '
                    'all continuum regions is zero. Check that the input spectra '
                    'contain non-zero flux values, or set '
                    'spectra.continuum_scale manually.'
                )
                raise ValueError(msg)
            self._continuum_scale = max_cont_scale * ref_flux_unit

        self._scale_diagnostics = ScaleDiagnosticList(diag_list)

    # -- preparation ----------------------------------------------------------

    def prepare(
        self,
        line_config: LineConfiguration,
        continuum_config: ContinuumConfiguration | None = None,
        *,
        linedet_width: u.Quantity = 1000.0 * u.km / u.s,
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
        linedet_width : astropy.units.Quantity
            Detection width.  Must have velocity units.
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
            line_config, continuum_config, linedet_width=linedet_width
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
        linedet_width: u.Quantity = 1000.0 * u.km / u.s,
    ) -> tuple[LineConfiguration, ContinuumConfiguration | None]:
        """Drop lines and continuum regions not covered by any spectrum.

        Each line's rest-frame wavelength is shifted to the observed frame
        using :attr:`redshift`, then padded by *linedet_width* to form a
        detection window.  A line is kept if **any** spectrum partially
        overlaps that window.  Continuum regions are checked the same way.

        Parameters
        ----------
        line_config : LineConfiguration
            Line configuration to filter.
        continuum_config : ContinuumConfiguration, optional
            Continuum configuration to filter.  ``None`` is passed through.
        linedet_width : astropy.units.Quantity
            Detection width.  Must have velocity units.
            Default ``1000 km/s``.

        Returns
        -------
        filtered_lines : LineConfiguration
        filtered_continuum : ContinuumConfiguration or None
        """
        from unite.continuum.config import ContinuumConfiguration

        linedet_width = _ensure_velocity(linedet_width, 'linedet_width', ndim=0)
        eps = float(linedet_width.to(u.km / u.s).value) / (2 * C_KMS)
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
                    conv = _get_conversion_factor(region._unit, s.unit)
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

    def __iter__(self) -> Iterator[GenericSpectrum]:
        return iter(self._spectra)

    def __getitem__(self, idx: int | str) -> GenericSpectrum:
        """Return a spectrum by integer index or by name.

        Parameters
        ----------
        idx : int or str
            Integer index into the internal spectrum list, or a string
            matching :attr:`~unite.instrument.generic.GenericSpectrum.name`
            of the desired spectrum.

        Raises
        ------
        KeyError
            If *idx* is a string and no spectrum with that name exists.
        IndexError
            If *idx* is an integer and is out of range.
        """
        if isinstance(idx, str):
            for spectrum in self._spectra:
                if spectrum.name == idx:
                    return spectrum
            msg = f'No spectrum found with name {idx!r}.'
            raise KeyError(msg)
        return self._spectra[idx]

    def __repr__(self) -> str:
        lines = [f'Spectra: {len(self._spectra)} spectrum/a, z={self._redshift:.4f}']
        for i, s in enumerate(self._spectra):
            lo, hi = s.wavelength_range
            unit_str = s.unit.to_string()
            label = s.name or f'#{i}'
            cal = ' [calibrated]' if s.has_calibration_priors else ''
            lines.append(
                f'  [{i}] {label}: {s.npix} px, λ ∈ [{lo:.4g}, {hi:.4g}] {unit_str}{cal}'
            )
        return '\n'.join(lines)
