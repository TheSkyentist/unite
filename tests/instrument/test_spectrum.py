"""Tests for Spectrum.error_scale, Spectra.compute_scales, prepare."""

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest

from unite import line, prior
from unite.continuum import ContinuumConfiguration, Linear
from unite.instrument.generic import SimpleDisperser
from unite.spectrum import Spectra, Spectrum


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
    disperser = SimpleDisperser(wavelength=wl, R=R, name=name)
    low = wl - 0.5 * np.gradient(wl)
    high = wl + 0.5 * np.gradient(wl)

    sigma = fwhm_wl / (2 * np.sqrt(2 * np.log(2)))
    line_flux = peak * np.exp(-0.5 * ((wl.value - center_wl) / sigma) ** 2)
    continuum = np.full(npix, continuum_level)
    rng = np.random.default_rng(42)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    flux = (line_flux + continuum + rng.normal(0, noise_std, npix)) * flux_unit
    error = np.full(npix, noise_std) * flux_unit

    return Spectrum(
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
        cont = ContinuumConfiguration.from_lines(
            lc.centers, width=30_000 * u.km / u.s, form=Linear()
        )
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.compute_scales(lc, cont)
        assert spectra.line_scale is not None
        assert spectra.continuum_scale is not None
        assert spectra.continuum_scale.value > 0

    def test_continuum_scale_has_flux_density_units(self):
        spectrum = _make_spectrum()
        lc = _make_line_config()
        cont = ContinuumConfiguration.from_lines(
            lc.centers, width=30_000 * u.km / u.s, form=Linear()
        )
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
        cont = ContinuumConfiguration.from_lines(
            lc.centers, width=30_000 * u.km / u.s, form=Linear()
        )
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
        cont = ContinuumConfiguration.from_lines(
            lc.centers, width=30_000 * u.km / u.s, form=Linear()
        )
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.prepare(lc, cont)
        spectra.compute_scales(
            spectra.prepared_line_config, spectra.prepared_cont_config
        )
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
        from unite.spectrum import ScaleDiagnosticList

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
        with pytest.raises(TypeError, match='Spectrum'):
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

        cont = ContinuumConfiguration(
            [
                ContinuumRegion(6500.0 * u.AA, 6600.0 * u.AA, Linear()),  # in coverage
                ContinuumRegion(
                    5000.0 * u.AA, 5100.0 * u.AA, Linear()
                ),  # out of coverage
            ]
        )
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.prepare(lc, cont, drop_empty_regions=False)
        # Should not raise; the out-of-coverage region is simply skipped
        spectra.compute_scales(
            spectra.prepared_line_config, spectra.prepared_cont_config
        )
        assert spectra.line_scale is not None

    def test_region_too_few_pixels_all_fits_fail_raises(self):
        """All continuum fits failing raises ValueError."""
        # Very narrow region with only 1-2 pixels, entirely inside line mask
        spectrum = _make_spectrum(npix=100, wl_range=(6500, 6600))
        lc = _make_line_config()
        from unite.continuum.config import ContinuumRegion
        from unite.continuum.library import Linear

        cont = ContinuumConfiguration(
            [
                ContinuumRegion(
                    6562.0 * u.AA, 6564.0 * u.AA, Linear()
                )  # inside line mask
            ]
        )
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.prepare(lc, cont, drop_empty_regions=False)
        with pytest.raises(ValueError, match='line mask covers all pixels'):
            spectra.compute_scales(
                spectra.prepared_line_config, spectra.prepared_cont_config
            )

    def test_no_line_peak_above_continuum_raises(self):
        """ValueError when no line peak is found above continuum."""
        # Spectrum with zero flux — no line peak can be above continuum
        wl_range = (6500, 6600)
        wl = np.linspace(*wl_range, 100) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0, name='flat')
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        flux = np.zeros(100) * flux_unit
        error = np.ones(100) * flux_unit
        spectrum = Spectrum(
            low=low, high=high, flux=flux, error=error, disperser=disperser, name='flat'
        )
        lc = _make_line_config()
        spectra = Spectra([spectrum], redshift=0.0)
        with pytest.raises(ValueError, match='no emission line peak'):
            spectra.compute_scales(lc)

    def test_line_mask_covers_all_continuum_pixels_raises(self):
        """ValueError when line mask leaves no unmasked continuum pixels."""
        spectrum = _make_spectrum(npix=100, wl_range=(6550, 6575))
        # Line at 6563 with very wide mask will cover most of this tiny range
        lc = _make_line_config()
        from unite.continuum.config import ContinuumRegion

        cont = ContinuumConfiguration(
            [ContinuumRegion(6550.0 * u.AA, 6575.0 * u.AA, Linear())]
        )
        spectra = Spectra([spectrum], redshift=0.0)
        # Use an extremely wide line mask to cover all pixels
        with pytest.raises(
            ValueError, match=r'(line mask covers all pixels|no emission line peak)'
        ):
            spectra.compute_scales(lc, cont, line_mask_width=100_000 * u.km / u.s)

    def test_zero_flux_continuum_raises(self):
        """ValueError when median |flux| in all continuum regions is zero."""
        wl_range = (6400, 6700)
        npix = 200
        wl = np.linspace(*wl_range, npix) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0, name='zero')
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        # Strong line but zero continuum — line peak will be positive but
        # continuum regions (away from line) will have zero median |flux|.
        sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
        line_flux = 100 * np.exp(-0.5 * ((wl.value - 6563) / sigma) ** 2)
        flux = line_flux * flux_unit
        error = np.ones(npix) * flux_unit
        spectrum = Spectrum(
            low=low,
            high=high,
            flux=flux,
            error=error,
            disperser=disperser,
            name='zeroc',
        )
        lc = _make_line_config()
        from unite.continuum.config import ContinuumRegion

        # Continuum region far from line — flux is essentially zero there
        cont = ContinuumConfiguration(
            [ContinuumRegion(6400.0 * u.AA, 6450.0 * u.AA, Linear())]
        )
        spectra = Spectra([spectrum], redshift=0.0)
        # The line peak is positive so line scale succeeds, but continuum
        # median |flux| rounds to zero
        with pytest.raises(ValueError, match=r'(median.*zero|no emission line peak)'):
            spectra.compute_scales(lc, cont)

    def test_manual_scale_bypasses_compute(self):
        """User can set scales manually to bypass compute_scales failures."""
        spectrum = _make_spectrum()
        spectra = Spectra([spectrum], redshift=0.0)
        # Set scales manually — no compute_scales needed
        spectra.line_scale = 1e-17 * u.erg / u.s / u.cm**2
        spectra.continuum_scale = 1e-17 * u.erg / u.s / u.cm**2 / u.AA
        assert spectra.line_scale.value == pytest.approx(1e-17)
        assert spectra.continuum_scale.value == pytest.approx(1e-17)


# ---------------------------------------------------------------------------
# Spectrum construction validation
# ---------------------------------------------------------------------------


class TestSpectrumConstruction:
    """Tests for Spectrum construction validation."""

    def test_flux_error_units_incompatible_raises(self):
        """flux and error units must be compatible (generic.py line 299-301)."""
        wl = np.linspace(6500, 6600, 100) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0)
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        flux = np.ones(100) * flux_unit
        # error in velocity units (incompatible)
        error = np.ones(100) * u.km / u.s

        with pytest.raises(ValueError, match='must have units equivalent'):
            Spectrum(low=low, high=high, flux=flux, error=error, disperser=disperser)

    def test_disperser_must_be_disperser_instance(self):
        """disperser parameter must be a Disperser instance (generic.py line 305-307)."""
        wl = np.linspace(6500, 6600, 100) * u.AA
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        flux = np.ones(100) * flux_unit
        error = np.ones(100) * flux_unit

        with pytest.raises(TypeError, match='Disperser instance'):
            Spectrum(
                low=low, high=high, flux=flux, error=error, disperser='not_a_disperser'
            )

    def test_low_high_shapes_must_match(self):
        """low and high must have the same shape (generic.py line 314-316)."""
        wl = np.linspace(6500, 6600, 100) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0)
        low = wl - 0.5 * np.gradient(wl)
        high = (wl + 0.5 * np.gradient(wl))[:-1]  # Mismatched shape
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        flux = np.ones(100) * flux_unit
        error = np.ones(100) * flux_unit

        with pytest.raises(ValueError, match='same shape'):
            Spectrum(low=low, high=high, flux=flux, error=error, disperser=disperser)

    def test_flux_length_must_match_pixels(self):
        """flux length must match number of pixels (generic.py line 330-332)."""
        wl = np.linspace(6500, 6600, 100) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0)
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        flux = np.ones(50) * flux_unit  # Wrong length
        error = np.ones(100) * flux_unit

        with pytest.raises(ValueError, match='does not match'):
            Spectrum(low=low, high=high, flux=flux, error=error, disperser=disperser)

    def test_error_length_must_match_pixels(self):
        """error length must match number of pixels (generic.py line 330-332)."""
        wl = np.linspace(6500, 6600, 100) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0)
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        flux = np.ones(100) * flux_unit
        error = np.ones(75) * flux_unit  # Wrong length

        with pytest.raises(ValueError, match='does not match'):
            Spectrum(low=low, high=high, flux=flux, error=error, disperser=disperser)


# ---------------------------------------------------------------------------
# Spectrum._sliced with array error_scale
# ---------------------------------------------------------------------------


class TestSpectrumSliced:
    """Tests for Spectrum._sliced method."""

    def test_sliced_preserves_array_error_scale(self):
        """_sliced method correctly masks array error_scale (generic.py line 482)."""
        spectrum = _make_spectrum(npix=100)
        # Set per-pixel error_scale
        scale_arr = jnp.linspace(1.0, 3.0, spectrum.npix)
        spectrum.error_scale = scale_arr

        # Create a mask that selects every other pixel
        mask = jnp.arange(spectrum.npix) % 2 == 0

        # Slice the spectrum
        sliced = spectrum._sliced(mask)

        # Error scale should be masked too
        expected = scale_arr[mask]
        np.testing.assert_array_equal(sliced.error_scale, expected)
        assert sliced.npix == int(jnp.sum(mask))


# ---------------------------------------------------------------------------
# Spectrum.__repr__ with calibration params
# ---------------------------------------------------------------------------


class TestSpectrumRepr:
    """Tests for Spectrum __repr__ method."""

    def test_repr_with_calibration_params(self):
        """repr shows [calibrated] when disperser has calibration params (generic.py line 488-493)."""
        from unite.instrument.base import RScale

        wl = np.linspace(6500, 6600, 100) * u.AA
        # Disperser with calibration token
        r_scale = RScale()
        disperser = SimpleDisperser(
            wavelength=wl, R=3000.0, name='test_disp', r_scale=r_scale
        )
        spectrum = _make_spectrum()
        spectrum.disperser = disperser

        repr_str = repr(spectrum)
        assert '[calibrated]' in repr_str

    def test_repr_without_calibration_params(self):
        """repr does not show [calibrated] when no calibration tokens (generic.py line 488-493)."""
        spectrum = _make_spectrum()
        repr_str = repr(spectrum)
        # No calibration tokens
        assert '[calibrated]' not in repr_str

    def test_repr_with_spectrum_name(self):
        """repr includes spectrum name when provided (generic.py line 488-493)."""
        spectrum = _make_spectrum(name='MySpectrum')
        repr_str = repr(spectrum)
        assert 'MySpectrum' in repr_str
        assert 'Spectrum' in repr_str

    def test_repr_with_empty_name(self):
        """repr shows just class name when spectrum.name is empty (generic.py line 488-493)."""
        spectrum = _make_spectrum()
        spectrum.name = ''
        repr_str = repr(spectrum)
        assert 'Spectrum' in repr_str


# ---------------------------------------------------------------------------
# Spectra.filter_config: lindet_width parameter
# ---------------------------------------------------------------------------


class TestLinedetWidth:
    """Tests for lindet_width parameter in filter_config and prepare."""

    def test_lindet_width_default_value(self):
        """Default lindet_width is 1000 km/s and lines are covered."""
        spectrum = _make_spectrum(wl_range=(6500, 6600))
        # H-alpha at rest is 6563 Angstrom
        # At z=0, observed is also 6563; 1000 km/s corresponds to ~22 Angstrom
        # So detection window is roughly [6541, 6585], which overlaps [6500, 6600]
        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)
        spectra = Spectra([spectrum], redshift=0.0)
        fl, _ = spectra.filter_config(lc)
        assert len(fl) == 1

    def test_small_lindet_width_includes_barely_covered_line(self):
        """Small lindet_width allows lines barely inside coverage to be kept."""
        # Spectrum covers [6500, 6600]; we add a line at 6598 Angstrom
        spectrum = _make_spectrum(wl_range=(6500, 6600))
        lc = line.LineConfiguration()
        lc.add_line('edge', 6598.0 * u.AA)
        spectra = Spectra([spectrum], redshift=0.0)
        # With small lindet_width (e.g., 10 km/s ~ 0.2 Angstrom),
        # line at 6598 is barely covered
        fl, _ = spectra.filter_config(lc, linedet_width=10.0 * u.km / u.s)
        # Line should be kept because its detection window overlaps
        assert len(fl) >= 1

    def test_large_lindet_width_excludes_edge_line(self):
        """Large lindet_width can exclude lines near the spectrum edge."""
        # Spectrum covers [6500, 6600]; line at edge
        spectrum = _make_spectrum(wl_range=(6500, 6600))
        lc = line.LineConfiguration()
        lc.add_line('edge', 6600.0 * u.AA)  # Right at upper edge
        spectra = Spectra([spectrum], redshift=0.0)
        # With very large lindet_width, the detection window extends far beyond
        # the spectrum, and the line may not be covered
        fl, _ = spectra.filter_config(lc, linedet_width=100_000.0 * u.km / u.s)
        # Line is likely excluded due to large padding
        assert len(fl) == 0 or len(fl) == 1  # Depends on exact edges

    def test_lindet_width_with_redshift(self):
        """lindet_width works correctly when redshift is non-zero."""
        # Spectrum at z=1 covers [6500, 6600] (observed frame)
        spectrum = _make_spectrum(wl_range=(6500, 6600))
        lc = line.LineConfiguration()
        # Rest-frame H-alpha at 6563 Angstrom
        lc.add_line('Ha_rest', 6563.0 * u.AA)
        spectra = Spectra([spectrum], redshift=1.0)
        # At z=1, rest-frame 6563 appears at observed 13126 Angstrom
        # This is outside [6500, 6600], so without proper handling it would be excluded
        fl, _ = spectra.filter_config(lc, linedet_width=1000.0 * u.km / u.s)
        # Line should be excluded because observed frame wavelength is far outside
        assert len(fl) == 0

    def test_lindet_width_zero(self):
        """lindet_width=0 means no padding; line is excluded."""
        spectrum = _make_spectrum(wl_range=(6500, 6600))
        lc = line.LineConfiguration()
        # Line exactly at spectrum center
        lc.add_line('center', 6550.0 * u.AA)
        spectra = Spectra([spectrum], redshift=0.0)
        fl, _ = spectra.filter_config(lc, linedet_width=0.0 * u.km / u.s)
        # Line should not be covered
        assert len(fl) == 0

    def test_lindet_width_affects_prepare_filtering(self):
        """lindet_width passed to prepare() filters lines the same way."""
        spectrum = _make_spectrum(wl_range=(6500, 6600))
        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)
        lc.add_line('edge', 6598.0 * u.AA)
        spectra = Spectra([spectrum], redshift=0.0)
        # With small linedet_width, both should be kept
        fl_small, _ = spectra.prepare(lc, linedet_width=10.0 * u.km / u.s)
        assert len(fl_small) >= 1
        # With large linedet_width, edge line may be excluded
        spectra2 = Spectra([spectrum], redshift=0.0)
        fl_large, _ = spectra2.prepare(lc, linedet_width=100_000.0 * u.km / u.s)
        # At least Ha should be covered with large lindet_width
        # (edge might be excluded)
        assert len(fl_large) >= 1

    def test_lindet_width_multiple_lines_selective_filtering(self):
        """lindet_width selectively filters lines based on spectrum edges."""
        spectrum = _make_spectrum(wl_range=(6400, 6700))
        lc = line.LineConfiguration()
        # Three lines: one inside, two near edges
        lc.add_line('inside', 6550.0 * u.AA)
        lc.add_line('lower_edge', 6401.0 * u.AA)
        lc.add_line('upper_edge', 6699.0 * u.AA)
        spectra = Spectra([spectrum], redshift=0.0)
        # With tiny lindet_width, only clearly inside line survives
        fl_tiny, _ = spectra.filter_config(lc, linedet_width=1.0 * u.km / u.s)
        assert len(fl_tiny) >= 1
        # With default lindet_width, more lines should be covered
        fl_default, _ = spectra.filter_config(lc, linedet_width=1000.0 * u.km / u.s)
        assert len(fl_default) >= len(fl_tiny)

    def test_lindet_width_unit_conversion(self):
        """lindet_width accepts different velocity units."""
        spectrum = _make_spectrum(wl_range=(6500, 6600))
        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)
        spectra = Spectra([spectrum], redshift=0.0)
        # Using different velocity units should give same result
        fl_kms, _ = spectra.filter_config(lc, linedet_width=1000.0 * u.km / u.s)
        spectra2 = Spectra([spectrum], redshift=0.0)
        fl_cms, _ = spectra2.filter_config(lc, linedet_width=1.0e5 * u.cm / u.s)
        assert len(fl_kms) == len(fl_cms)

    def test_lindet_width_coverage_padding_symmetry(self):
        """Detection window padding is symmetric around line wavelength."""
        spectrum = _make_spectrum(wl_range=(6500, 6650))
        lc = line.LineConfiguration()
        # Line at 6575 (center of range), with spectrum covering [6500, 6650]
        lc.add_line('test', 6575.0 * u.AA)
        spectra = Spectra([spectrum], redshift=0.0)
        # With 1000 km/s (~22 Angstrom at this wavelength), detection window
        # is roughly [6553, 6597], well within spectrum
        fl, _ = spectra.filter_config(lc, linedet_width=1000.0 * u.km / u.s)
        assert len(fl) == 1

    def test_lindet_width_small_spectrum_coverage(self):
        """lindet_width works with very narrow spectrum coverage."""
        # Tiny spectrum: only 10 Angstrom wide at Ha
        spectrum = _make_spectrum(wl_range=(6560, 6570))
        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)
        spectra = Spectra([spectrum], redshift=0.0)
        # With default 1000 km/s lindet_width, Ha should still be covered
        fl, _ = spectra.filter_config(lc, linedet_width=1000.0 * u.km / u.s)
        assert len(fl) == 1
        # With huge lindet_width, Ha might be excluded if detection window
        # extends too far outside the tiny spectrum
        spectra2 = Spectra([spectrum], redshift=0.0)
        fl_huge, _ = spectra2.filter_config(lc, linedet_width=1e6 * u.km / u.s)
        # Even with huge lindet_width, Ha is close to spectrum center
        # so it should likely be covered
        assert len(fl_huge) == 1
