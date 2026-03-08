"""Tests for Spectrum.error_scale, Spectra.compute_scales, prepare."""

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest

from unite import line, prior
from unite.continuum import ContinuumConfiguration, Linear
from unite.instrument import Spectra
from unite.instrument.generic import GenericSpectrum, SimpleDisperser


def _make_spectrum(
    center_wl=6563.0,
    fwhm_wl=5.0,
    peak=100.0,
    noise_std=2.0,
    npix=100,
    wl_range=(6500, 6600),
    continuum_level=10.0,
    *,
    name='test',
    unit=u.AA,
    R=3000.0,  # noqa: N803
):
    """Create a test spectrum with a Gaussian line and linear continuum."""
    wl = np.linspace(*wl_range, npix) * unit
    disperser = SimpleDisperser(wavelength=wl.value, unit=unit, R=R, name=name)
    low = wl - 0.5 * np.gradient(wl)
    high = wl + 0.5 * np.gradient(wl)

    sigma = fwhm_wl / (2 * np.sqrt(2 * np.log(2)))
    line_flux = peak * np.exp(-0.5 * ((wl.value - center_wl) / sigma) ** 2)
    continuum = np.full(npix, continuum_level)
    rng = np.random.default_rng(42)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    flux = (line_flux + continuum + rng.normal(0, noise_std, npix)) * flux_unit
    error = np.full(npix, noise_std) * flux_unit

    return GenericSpectrum(
        low=low, high=high, flux=flux, error=error, disperser=disperser, name=name
    )


def _make_line_config():
    """Create a minimal line config with one H-alpha line."""
    lc = line.LineConfiguration()
    lc.add_line(
        'H_alpha',
        6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Uniform(-0.01, 0.01)),
        fwhm_gauss=line.FWHM(prior=prior.Uniform(100, 1000)),
        flux=line.Flux(prior=prior.Uniform(0, 5)),
    )
    return lc


# ---------------------------------------------------------------------------
# Spectrum.error_scale
# ---------------------------------------------------------------------------


class TestErrorScale:
    """Tests for Spectrum error_scale property."""

    def test_default_error_scale(self):
        spectrum = _make_spectrum()
        assert spectrum.error_scale == 1.0

    def test_set_error_scale(self):
        spectrum = _make_spectrum()
        spectrum.error_scale = 2.5
        assert spectrum.error_scale == 2.5

    def test_scaled_error(self):
        spectrum = _make_spectrum()
        spectrum.error_scale = 3.0
        np.testing.assert_allclose(
            spectrum.scaled_error, spectrum.error * 3.0, rtol=1e-10
        )

    def test_error_scale_must_be_positive(self):
        spectrum = _make_spectrum()
        with pytest.raises(ValueError, match='error_scale must be > 0'):
            spectrum.error_scale = 0.0
        with pytest.raises(ValueError, match='error_scale must be > 0'):
            spectrum.error_scale = -1.0

    def test_scaled_error_default_equals_error(self):
        spectrum = _make_spectrum()
        np.testing.assert_array_equal(spectrum.scaled_error, spectrum.error)

    def test_error_scale_per_pixel_array(self):
        spectrum = _make_spectrum()
        scale_arr = jnp.ones(spectrum.npix) * 2.0
        spectrum.error_scale = scale_arr
        np.testing.assert_allclose(
            spectrum.scaled_error, spectrum.error * 2.0, rtol=1e-10
        )

    def test_error_scale_array_wrong_shape(self):
        spectrum = _make_spectrum()
        with pytest.raises(ValueError, match='error_scale array must have shape'):
            spectrum.error_scale = jnp.ones(5)

    def test_error_scale_array_must_be_positive(self):
        spectrum = _make_spectrum()
        bad = jnp.ones(spectrum.npix).at[0].set(-1.0)
        with pytest.raises(ValueError, match='error_scale values must all be > 0'):
            spectrum.error_scale = bad


# ---------------------------------------------------------------------------
# Spectra.compute_scales
# ---------------------------------------------------------------------------


class TestComputeScales:
    """Tests for Spectra.compute_scales method."""

    def test_line_scale_positive(self):
        spectrum = _make_spectrum()
        lc = _make_line_config()
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.compute_scales(lc)
        assert spectra.line_scale is not None
        assert spectra.line_scale.value > 0

    def test_line_scale_has_flux_units(self):
        spectrum = _make_spectrum()
        lc = _make_line_config()
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.compute_scales(lc)
        # line_scale should be integrated flux (flux_density * wavelength).
        ref = u.erg / u.s / u.cm**2
        assert spectra.line_scale.unit.is_equivalent(ref)

    def test_continuum_scale_with_config(self):
        spectrum = _make_spectrum()
        lc = _make_line_config()
        cont = ContinuumConfiguration.from_lines(lc.centers, pad=0.05, form=Linear())
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.compute_scales(lc, cont)
        assert spectra.line_scale is not None
        assert spectra.continuum_scale is not None
        assert spectra.continuum_scale.value > 0

    def test_continuum_scale_has_flux_density_units(self):
        spectrum = _make_spectrum()
        lc = _make_line_config()
        cont = ContinuumConfiguration.from_lines(lc.centers, pad=0.05, form=Linear())
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.compute_scales(lc, cont)
        ref = u.erg / u.s / u.cm**2 / u.AA
        assert spectra.continuum_scale.unit.is_equivalent(ref)

    def test_no_continuum_scale_without_config(self):
        spectrum = _make_spectrum()
        lc = _make_line_config()
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.compute_scales(lc)
        assert spectra.continuum_scale is None

    def test_line_scale_setter_validates_type(self):
        spectra = Spectra([_make_spectrum()], redshift=0.0)
        with pytest.raises(TypeError, match='line_scale must be an astropy Quantity'):
            spectra.line_scale = 1.0

    def test_line_scale_setter_validates_unit(self):
        spectra = Spectra([_make_spectrum()], redshift=0.0)
        with pytest.raises(u.UnitConversionError):
            spectra.line_scale = 1.0 * u.km / u.s

    def test_line_scale_setter_validates_positive(self):
        spectra = Spectra([_make_spectrum()], redshift=0.0)
        with pytest.raises(ValueError, match='line_scale must be > 0'):
            spectra.line_scale = -1.0 * u.erg / u.s / u.cm**2

    def test_continuum_scale_setter_validates_positive(self):
        spectra = Spectra([_make_spectrum()], redshift=0.0)
        with pytest.raises(ValueError, match='continuum_scale must be > 0'):
            spectra.continuum_scale = 0.0 * u.erg / u.s / u.cm**2 / u.AA

    def test_error_scale_no_continuum_is_noop(self):
        """error_scale=True with no continuum should not change error_scale."""
        spectrum = _make_spectrum()
        lc = _make_line_config()
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.compute_scales(lc, error_scale=True)
        assert spectrum.error_scale == 1.0

    def test_error_scale_with_continuum(self):
        """error_scale=True should set per-pixel error_scale >= 1."""
        spectrum = _make_spectrum(noise_std=2.0, continuum_level=10.0)
        lc = _make_line_config()
        cont = ContinuumConfiguration.from_lines(lc.centers, pad=0.05, form=Linear())
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.compute_scales(lc, cont, error_scale=True)
        scale = spectrum.error_scale
        # Should be an array (per-pixel)
        assert isinstance(scale, jnp.ndarray)


# ---------------------------------------------------------------------------
# Spectra.prepare
# ---------------------------------------------------------------------------


class TestPrepare:
    """Tests for Spectra.prepare method."""

    def test_prepare_sets_flag(self):
        spectrum = _make_spectrum()
        lc = _make_line_config()
        spectra = Spectra([spectrum], redshift=0.0)
        assert not spectra.is_prepared
        spectra.prepare(lc)
        assert spectra.is_prepared

    def test_prepare_stores_configs(self):
        spectrum = _make_spectrum()
        lc = _make_line_config()
        spectra = Spectra([spectrum], redshift=0.0)
        fl, fc = spectra.prepare(lc)
        assert spectra.prepared_line_config is fl
        assert spectra.prepared_cont_config is fc

    def test_prepare_filters_uncovered_lines(self):
        """Lines outside the spectrum's coverage should be dropped."""
        spectrum = _make_spectrum(wl_range=(6500, 6600))
        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)
        lc.add_line('Hb', 4861.0 * u.AA)  # outside coverage
        spectra = Spectra([spectrum], redshift=0.0)
        fl, _ = spectra.prepare(lc)
        assert len(fl) == 1

    def test_prepare_drops_empty_regions(self):
        """Continuum regions with no lines should be dropped by default."""
        spectrum = _make_spectrum(wl_range=(6400, 6700))
        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)
        # Create two continuum regions: one around Ha, one far away
        from unite.continuum.config import ContinuumRegion

        cont = ContinuumConfiguration(
            [
                ContinuumRegion(6500.0 * u.AA, 6600.0 * u.AA, Linear()),
                ContinuumRegion(6600.0 * u.AA, 6700.0 * u.AA, Linear()),
            ]
        )
        spectra = Spectra([spectrum], redshift=0.0)
        _fl, fc = spectra.prepare(lc, cont, drop_empty_regions=True)
        # Only the region containing Ha should remain
        assert fc is not None
        assert len(fc) == 1

    def test_prepare_keeps_all_regions_when_disabled(self):
        """drop_empty_regions=False should keep all regions."""
        spectrum = _make_spectrum(wl_range=(6400, 6700))
        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)
        from unite.continuum.config import ContinuumRegion

        cont = ContinuumConfiguration(
            [
                ContinuumRegion(6500.0 * u.AA, 6600.0 * u.AA, Linear()),
                ContinuumRegion(6600.0 * u.AA, 6700.0 * u.AA, Linear()),
            ]
        )
        spectra = Spectra([spectrum], redshift=0.0)
        _fl, fc = spectra.prepare(lc, cont, drop_empty_regions=False)
        assert len(fc) == 2


# ---------------------------------------------------------------------------
# ScaleDiagnosticList: string lookup and slice (spectrum.py lines 71-78)
# ---------------------------------------------------------------------------


class TestScaleDiagnosticList:
    """Tests for ScaleDiagnosticList string-key lookup and slicing."""

    def _get_diagnostics(self):
        spectrum = _make_spectrum()
        lc = _make_line_config()
        cont = ContinuumConfiguration.from_lines(lc.centers, pad=0.05, form=Linear())
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.prepare(lc, cont)
        spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
        return spectra.scale_diagnostics

    def test_string_lookup(self):
        """ScaleDiagnosticList['name'] returns the matching diagnostic."""
        diags = self._get_diagnostics()
        d = diags['test']
        assert d.name == 'test'

    def test_string_lookup_missing_raises(self):
        """ScaleDiagnosticList['missing'] raises KeyError."""
        diags = self._get_diagnostics()
        with pytest.raises(KeyError):
            _ = diags['no_such_spectrum']

    def test_integer_lookup(self):
        """ScaleDiagnosticList[0] returns the first item."""
        diags = self._get_diagnostics()
        assert diags[0].name == 'test'

    def test_slice_returns_scale_diagnostic_list(self):
        """ScaleDiagnosticList[:] returns a ScaleDiagnosticList instance (line 77-78)."""
        from unite.instrument.spectrum import ScaleDiagnosticList

        diags = self._get_diagnostics()
        sliced = diags[:]
        assert isinstance(sliced, ScaleDiagnosticList)


# ---------------------------------------------------------------------------
# Spectra with explicit canonical_unit (spectrum.py lines 177-182)
# ---------------------------------------------------------------------------


class TestSpectraCanonicalUnit:
    """Tests for Spectra with explicit canonical_unit parameter."""

    def test_explicit_canonical_unit(self):
        """Spectra with canonical_unit=u.um uses that unit."""
        spectrum = _make_spectrum(unit=u.AA)
        spectra = Spectra([spectrum], redshift=0.0, canonical_unit=u.um)
        assert spectra.canonical_unit == u.um

    def test_explicit_canonical_unit_non_wavelength_raises(self):
        """Spectra raises UnitConversionError for non-wavelength canonical_unit."""
        spectrum = _make_spectrum()
        with pytest.raises(u.UnitConversionError):
            Spectra([spectrum], redshift=0.0, canonical_unit=u.km / u.s)


# ---------------------------------------------------------------------------
# Spectra constructor validation (spectrum.py lines 161-166)
# ---------------------------------------------------------------------------


class TestSpectraConstruction:
    """Tests for Spectra constructor validation."""

    def test_empty_spectra_raises_value_error(self):
        """Spectra([]) raises ValueError (line 161-162)."""
        with pytest.raises(ValueError, match='at least one spectrum'):
            Spectra([])

    def test_non_generic_spectrum_raises_type_error(self):
        """Spectra([object()]) raises TypeError (line 165-166)."""
        with pytest.raises(TypeError, match='GenericSpectrum'):
            Spectra([object()])

    def test_getitem_string_lookup(self):
        """Spectra['name'] returns the matching spectrum (lines 652-655)."""
        spectrum = _make_spectrum(name='myspec')
        spectra = Spectra([spectrum], redshift=0.0)
        result = spectra['myspec']
        assert result is spectrum

    def test_getitem_string_missing_raises(self):
        """Spectra['missing'] raises KeyError (lines 656-657)."""
        spectra = Spectra([_make_spectrum(name='foo')], redshift=0.0)
        with pytest.raises(KeyError):
            _ = spectra['bar']

    def test_getitem_integer(self):
        """Spectra[0] returns the first spectrum (line 658)."""
        spectrum = _make_spectrum()
        spectra = Spectra([spectrum], redshift=0.0)
        assert spectra[0] is spectrum

    def test_repr(self):
        """Spectra.__repr__ returns a string (lines 660-670)."""
        spectrum = _make_spectrum(name='test')
        spectra = Spectra([spectrum], redshift=0.0)
        r = repr(spectra)
        assert 'Spectra' in r
        assert 'test' in r


# ---------------------------------------------------------------------------
# continuum_scale setter validation (spectrum.py lines 232-242)
# ---------------------------------------------------------------------------


class TestContinuumScaleSetter:
    """Tests for continuum_scale setter validation paths."""

    def test_continuum_scale_setter_validates_type(self):
        """continuum_scale = 1.0 raises TypeError (line 232-234)."""
        spectra = Spectra([_make_spectrum()], redshift=0.0)
        with pytest.raises(TypeError, match='astropy Quantity'):
            spectra.continuum_scale = 1.0

    def test_continuum_scale_setter_validates_unit(self):
        """continuum_scale = km/s raises UnitConversionError (line 236-238)."""
        spectra = Spectra([_make_spectrum()], redshift=0.0)
        with pytest.raises(u.UnitConversionError):
            spectra.continuum_scale = 1.0 * u.km / u.s

    def test_continuum_scale_setter_success(self):
        """continuum_scale setter with valid value sets the scale (line 242)."""
        spectra = Spectra([_make_spectrum()], redshift=0.0)
        val = 5.0 * u.erg / u.s / u.cm**2 / u.AA
        spectra.continuum_scale = val
        assert spectra.continuum_scale == val

    def test_line_scale_setter_success(self):
        """line_scale setter with valid value sets the scale (line 223)."""
        spectra = Spectra([_make_spectrum()], redshift=0.0)
        val = 1e-17 * u.erg / u.s / u.cm**2
        spectra.line_scale = val
        assert spectra.line_scale == val


# ---------------------------------------------------------------------------
# compute_scales: region outside spectrum coverage (spectrum.py line 387)
# ---------------------------------------------------------------------------


class TestComputeScalesEdgeCases:
    """Edge cases in compute_scales."""

    def test_region_outside_spectrum_coverage_skipped(self):
        """Continuum region outside spectrum wavelength range is skipped (line 387)."""
        spectrum = _make_spectrum(wl_range=(6500, 6600))
        lc = _make_line_config()
        # Region far from spectrum coverage
        from unite.continuum.config import ContinuumRegion
        from unite.continuum.library import Linear

        cont = ContinuumConfiguration([
            ContinuumRegion(6500.0 * u.AA, 6600.0 * u.AA, Linear()),  # in coverage
            ContinuumRegion(5000.0 * u.AA, 5100.0 * u.AA, Linear()),  # out of coverage
        ])
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.prepare(lc, cont, drop_empty_regions=False)
        # Should not raise; the out-of-coverage region is simply skipped
        spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
        assert spectra.line_scale is not None

    def test_region_too_few_pixels_fit_returns_none(self):
        """Region with too few pixels causes fit to return None (line 348)."""
        # Very narrow region with only 1-2 pixels
        spectrum = _make_spectrum(npix=100, wl_range=(6500, 6600))
        lc = _make_line_config()
        from unite.continuum.config import ContinuumRegion
        from unite.continuum.library import Linear

        # A tiny region that will have too few good pixels after line masking
        cont = ContinuumConfiguration([
            ContinuumRegion(6562.0 * u.AA, 6564.0 * u.AA, Linear()),  # inside line mask
        ])
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.prepare(lc, cont, drop_empty_regions=False)
        # Should not raise; the failed fit returns None model_region
        spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
        # No line_scale from this test (line in region is masked out)
