"""Tests for Spectrum.error_scale, Spectra.compute_scales, prepare, resize_errors."""

import astropy.units as u
import numpy as np
import pytest

from unite import line, prior
from unite.continuum import ContinuumConfiguration, Linear
from unite.disperser.generic import SimpleDisperser
from unite.spectrum import Spectra, Spectrum


def _make_spectrum(center_wl=6563.0, fwhm_wl=5.0, peak=100.0, noise_std=2.0,
                   npix=100, wl_range=(6500, 6600), continuum_level=10.0,
                   *, name='test', unit=u.AA, R=3000.0):  # noqa: N803
    """Create a test spectrum with a Gaussian line and linear continuum."""
    wl = np.linspace(*wl_range, npix) * unit
    disperser = SimpleDisperser(
        wavelength=wl.value, unit=unit, R=R, name=name,
    )
    low = wl - 0.5 * np.gradient(wl)
    high = wl + 0.5 * np.gradient(wl)

    sigma = fwhm_wl / (2 * np.sqrt(2 * np.log(2)))
    line_flux = peak * np.exp(-0.5 * ((wl.value - center_wl) / sigma) ** 2)
    continuum = np.full(npix, continuum_level)
    rng = np.random.default_rng(42)
    flux = line_flux + continuum + rng.normal(0, noise_std, npix)
    error = np.full(npix, noise_std)

    return Spectrum(
        low=low, high=high, flux=flux, error=error,
        disperser=disperser, name=name,
    )


def _make_line_config():
    """Create a minimal line config with one H-alpha line."""
    lc = line.LineConfiguration()
    lc.add_line(
        'H_alpha', 6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Uniform(-0.01, 0.01)),
        fwhm_gauss=line.FWHM(prior=prior.Uniform(1.0, 10.0)),
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
        assert spectra.line_scale > 0

    def test_continuum_scale_with_config(self):
        spectrum = _make_spectrum()
        lc = _make_line_config()
        cont = ContinuumConfiguration.from_lines(
            lc.centers, pad=0.05, form=Linear()
        )
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.compute_scales(lc, cont)
        assert spectra.line_scale is not None
        assert spectra.continuum_scale is not None
        assert spectra.continuum_scale > 0

    def test_no_continuum_scale_without_config(self):
        spectrum = _make_spectrum()
        lc = _make_line_config()
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.compute_scales(lc)
        assert spectra.continuum_scale is None

    def test_line_scale_setter_validates(self):
        spectra = Spectra([_make_spectrum()], redshift=0.0)
        with pytest.raises(ValueError, match='line_scale must be > 0'):
            spectra.line_scale = -1.0

    def test_continuum_scale_setter_validates(self):
        spectra = Spectra([_make_spectrum()], redshift=0.0)
        with pytest.raises(ValueError, match='continuum_scale must be > 0'):
            spectra.continuum_scale = 0.0


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
        cont = ContinuumConfiguration([
            ContinuumRegion(6500.0 * u.AA, 6600.0 * u.AA, Linear()),
            ContinuumRegion(6600.0 * u.AA, 6700.0 * u.AA, Linear()),
        ])
        spectra = Spectra([spectrum], redshift=0.0)
        fl, fc = spectra.prepare(lc, cont, drop_empty_regions=True)
        # Only the region containing Ha should remain
        assert fc is not None
        assert len(fc) == 1

    def test_prepare_keeps_all_regions_when_disabled(self):
        """drop_empty_regions=False should keep all regions."""
        spectrum = _make_spectrum(wl_range=(6400, 6700))
        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)
        from unite.continuum.config import ContinuumRegion
        cont = ContinuumConfiguration([
            ContinuumRegion(6500.0 * u.AA, 6600.0 * u.AA, Linear()),
            ContinuumRegion(6600.0 * u.AA, 6700.0 * u.AA, Linear()),
        ])
        spectra = Spectra([spectrum], redshift=0.0)
        fl, fc = spectra.prepare(lc, cont, drop_empty_regions=False)
        assert len(fc) == 2


# ---------------------------------------------------------------------------
# Spectra.resize_errors
# ---------------------------------------------------------------------------


class TestResizeErrors:
    """Tests for Spectra.resize_errors method."""

    def test_resize_errors_no_continuum(self):
        """resize_errors with no continuum should be a no-op."""
        spectrum = _make_spectrum()
        lc = _make_line_config()
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.resize_errors(lc, None)
        assert spectrum.error_scale == 1.0

    def test_resize_errors_with_continuum(self):
        """resize_errors should set error_scale >= 1."""
        spectrum = _make_spectrum(noise_std=2.0, continuum_level=10.0)
        lc = _make_line_config()
        cont = ContinuumConfiguration.from_lines(
            lc.centers, pad=0.05, form=Linear()
        )
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.resize_errors(lc, cont)
        # error_scale should be >= 1.0 (clamped)
        assert spectrum.error_scale >= 1.0
