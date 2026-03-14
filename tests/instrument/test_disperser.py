"""Tests for GenericDisperser and SimpleDisperser."""

import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u

from unite.instrument.base import RScale
from unite.instrument.generic import GenericDisperser, SimpleDisperser

# ---------------------------------------------------------------------------
# GenericDisperser
# ---------------------------------------------------------------------------


class TestGenericDisperser:
    """Tests for GenericDisperser."""

    def test_construction(self):
        d = GenericDisperser(
            R_func=lambda w: jnp.full_like(w, 2700.0),
            dlam_dpix_func=lambda w: w / 2700.0,
            unit=u.AA,
        )
        assert d.unit == u.AA
        assert not d.has_calibration_params

    def test_R_returns_correct_values(self):
        d = GenericDisperser(
            R_func=lambda w: jnp.full_like(w, 2700.0),
            dlam_dpix_func=lambda w: w / 2700.0,
            unit=u.AA,
        )
        wl = jnp.array([5000.0, 6000.0, 7000.0])
        np.testing.assert_allclose(d.R(wl), 2700.0)

    def test_dlam_dpix_returns_correct_values(self):
        d = GenericDisperser(
            R_func=lambda w: jnp.full_like(w, 2700.0),
            dlam_dpix_func=lambda w: w / 2700.0,
            unit=u.AA,
        )
        wl = jnp.array([5400.0])
        np.testing.assert_allclose(d.dlam_dpix(wl), 2.0, rtol=1e-5)

    def test_wavelength_dependent_R(self):
        d = GenericDisperser(
            R_func=lambda w: w * 0.5,
            dlam_dpix_func=lambda w: jnp.ones_like(w),
            unit=u.um,
        )
        wl = jnp.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(d.R(wl), [0.5, 1.0, 1.5])

    def test_name(self):
        d = GenericDisperser(
            R_func=lambda w: w,
            dlam_dpix_func=lambda w: w,
            unit=u.AA,
            name='my_disperser',
        )
        assert d.name == 'my_disperser'

    def test_with_calibration_tokens(self):
        r = RScale()
        d = GenericDisperser(
            R_func=lambda w: w, dlam_dpix_func=lambda w: w, unit=u.AA, r_scale=r
        )
        assert d.has_calibration_params
        assert d.r_scale is r


# ---------------------------------------------------------------------------
# SimpleDisperser
# ---------------------------------------------------------------------------


class TestSimpleDisperser:
    """Tests for SimpleDisperser."""

    def test_construction_with_R(self):
        wl = np.linspace(5000, 7000, 100) * u.AA
        d = SimpleDisperser(wavelength=wl, R=3000.0)
        assert d.unit == u.AA

    def test_construction_with_dlam(self):
        wl = np.linspace(5000, 7000, 100) * u.AA
        d = SimpleDisperser(wavelength=wl, dlam=2.0 * u.AA)
        r = d.R(jnp.array([6000.0]))
        np.testing.assert_allclose(r, 3000.0, rtol=1e-3)

    def test_construction_with_dvel(self):
        wl = np.linspace(5000, 7000, 100) * u.AA
        d = SimpleDisperser(wavelength=wl, dvel=100.0 * u.km / u.s)
        r = d.R(jnp.array([6000.0]))
        # R = c / dvel = 299792.458 / 100 ≈ 2998
        np.testing.assert_allclose(r, 299792.458 / 100, rtol=1e-3)

    def test_no_resolution_raises(self):
        wl = np.linspace(5000, 7000, 100) * u.AA
        with pytest.raises(ValueError, match='Exactly one of'):
            SimpleDisperser(wavelength=wl)

    def test_multiple_resolution_raises(self):
        wl = np.linspace(5000, 7000, 100) * u.AA
        with pytest.raises(ValueError, match='Exactly one of'):
            SimpleDisperser(wavelength=wl, R=3000, dlam=2.0 * u.AA)

    def test_wrong_shape_R_raises(self):
        wl = np.linspace(5000, 7000, 100) * u.AA
        with pytest.raises(ValueError, match='same shape'):
            SimpleDisperser(wavelength=wl, R=np.ones(50))

    def test_scalar_R_broadcast(self):
        wl = np.linspace(5000, 7000, 100) * u.AA
        d = SimpleDisperser(wavelength=wl, R=3000.0)
        r = d.R(jnp.asarray(wl))
        np.testing.assert_allclose(r, 3000.0)

    def test_array_R(self):
        wl = np.linspace(5000, 7000, 100) * u.AA
        r_arr = np.linspace(2000, 4000, 100)
        d = SimpleDisperser(wavelength=wl, R=r_arr)
        r = d.R(jnp.asarray(wl))
        np.testing.assert_allclose(r, r_arr, rtol=1e-5)

    def test_dlam_dpix_from_gradient(self):
        """dlam_dpix should be derived from pixel spacing."""
        wl = np.linspace(5000, 7000, 100) * u.AA
        d = SimpleDisperser(wavelength=wl, R=3000.0)
        dlam = d.dlam_dpix(jnp.asarray(wl))
        # Uniform grid: spacing = (7000-5000)/99 ≈ 20.2
        expected = np.gradient(wl.value)
        np.testing.assert_allclose(dlam, expected, rtol=1e-5)

    def test_interpolation_outside_grid(self):
        """Values outside the grid should be extrapolated via jnp.interp."""
        wl = np.linspace(5000, 7000, 100) * u.AA
        d = SimpleDisperser(wavelength=wl, R=3000.0)
        # Should not raise, jnp.interp clamps at boundaries
        r = d.R(jnp.array([4000.0, 8000.0]))
        assert jnp.all(jnp.isfinite(r))

    def test_name_kwarg(self):
        wl = np.linspace(5000, 7000, 100) * u.AA
        d = SimpleDisperser(wavelength=wl, R=3000.0, name='test')
        assert d.name == 'test'
