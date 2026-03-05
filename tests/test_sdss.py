"""Tests for SDSS disperser."""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u

from unite.disperser.sdss import SDSSDisperser, SDSSSpectrum


class TestSDSSDisperser:
    """Tests for SDSSDisperser."""

    def test_default_construction(self):
        d = SDSSDisperser()
        assert d.unit == u.AA
        assert d.name == 'SDSS'
        assert not d._has_data

    def test_placeholder_R(self):
        d = SDSSDisperser()
        wl = jnp.array([5000.0, 6000.0])
        r = d.R(wl)
        np.testing.assert_allclose(r, 2000.0)

    def test_placeholder_dlam_dpix(self):
        d = SDSSDisperser()
        wl = jnp.array([5000.0, 6000.0])
        dlam = d.dlam_dpix(wl)
        np.testing.assert_allclose(dlam, 1.0)

    def test_construction_with_data(self):
        wl = np.linspace(3800, 9200, 100)
        wdisp = np.full(100, 1.0)  # 1 Angstrom/pixel sigma
        d = SDSSDisperser(wavelength=wl, wdisp=wdisp)
        assert d._has_data
        # R = lambda / (2.355 * wdisp)
        expected_r = wl / (2.355 * wdisp)
        np.testing.assert_allclose(d.R(jnp.asarray(wl)), expected_r, rtol=1e-5)

    def test_construction_with_wavelength_only(self):
        wl = np.linspace(3800, 9200, 100)
        d = SDSSDisperser(wavelength=wl)
        assert d._has_data
        # Default R = 2000
        np.testing.assert_allclose(d.R(jnp.asarray(wl)), 2000.0)

    def test_calibration_tokens(self):
        from unite.disperser.base import FluxScale, RScale

        r = RScale()
        f = FluxScale()
        d = SDSSDisperser(r_scale=r, flux_scale=f)
        assert d.r_scale is r
        assert d.flux_scale is f
        assert d.has_calibration_params

    def test_repr(self):
        d = SDSSDisperser()
        assert 'SDSSDisperser' in repr(d)
        assert 'no data' in repr(d)

        wl = np.linspace(3800, 9200, 100)
        d2 = SDSSDisperser(wavelength=wl)
        assert '100 px' in repr(d2)


class TestSDSSSpectrum:
    """Tests for SDSSSpectrum loader."""

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError, match='loader class'):
            SDSSSpectrum()

    def test_from_arrays(self):
        npix = 50
        wl = np.linspace(4000, 8000, npix + 1) * u.AA
        low = wl[:-1]
        high = wl[1:]
        flux_unit = 1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
        flux = np.ones(npix) * flux_unit
        error = np.full(npix, 0.1) * flux_unit
        d = SDSSDisperser(wavelength=np.linspace(4000, 8000, npix))
        spec = SDSSSpectrum.from_arrays(low, high, flux, error, d)
        assert spec.npix == npix
        assert spec.name == 'SDSS'
