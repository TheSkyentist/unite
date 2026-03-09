"""Tests for unite.continuum.fit (fit_continuum_form and helpers)."""

import jax.numpy as jnp
import pytest

from unite.continuum.fit import ContinuumFitResult, _initial_guess, fit_continuum_form
from unite.continuum.library import (
    Blackbody,
    Chebyshev,
    Linear,
    ModifiedBlackbody,
    Polynomial,
    PowerLaw,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CENTER = 1.0  # microns
N = 30
WL = jnp.linspace(0.8, 1.2, N)
ERROR = jnp.ones(N) * 0.01


# ---------------------------------------------------------------------------
# Linear forms
# ---------------------------------------------------------------------------


class TestFitLinear:
    """fit_continuum_form with is_linear=True forms."""

    def test_linear_recovers_params(self):
        """Fit a known linear continuum and check params are close."""
        # True: scale=2.0, slope=1.0, center=1.0
        form = Linear()
        true_params = {'scale': 2.0, 'slope': 1.0, 'normalization_wavelength': CENTER}
        flux = form.evaluate(WL, CENTER, true_params)

        result = fit_continuum_form(form, WL, flux, ERROR, CENTER)

        assert isinstance(result, ContinuumFitResult)
        assert result.params['scale'] == pytest.approx(2.0, abs=1e-4)
        assert result.params['slope'] == pytest.approx(1.0, abs=1e-4)
        assert result.params['normalization_wavelength'] == pytest.approx(CENTER)

    def test_linear_normalization_wavelength_default(self):
        """normalization_wavelength defaults to center when not provided."""
        form = Linear()
        flux = jnp.ones(N)
        result = fit_continuum_form(form, WL, flux, ERROR, CENTER)
        assert result.params['normalization_wavelength'] == pytest.approx(CENTER)

    def test_linear_explicit_normalization_wavelength(self):
        """Passing normalization_wavelength explicitly uses that value."""
        form = Linear()
        flux = jnp.ones(N)
        result = fit_continuum_form(
            form, WL, flux, ERROR, CENTER, normalization_wavelength=0.9
        )
        assert result.params['normalization_wavelength'] == pytest.approx(0.9)

    def test_linear_chi2_red_computed(self):
        """chi2_red is a float when dof > 0."""
        form = Linear()
        flux = jnp.ones(N)
        result = fit_continuum_form(form, WL, flux, ERROR, CENTER)
        assert result.chi2_red is not None
        assert isinstance(result.chi2_red, float)
        assert result.dof > 0

    def test_polynomial_fit(self):
        """Polynomial (degree=2) is linear and recovers params."""
        form = Polynomial(degree=2)
        true_params = {
            'scale': 2.0,
            'c1': 0.5,
            'c2': -0.2,
            'normalization_wavelength': CENTER,
        }
        flux = form.evaluate(WL, CENTER, true_params)

        result = fit_continuum_form(form, WL, flux, ERROR, CENTER)

        assert result.params['scale'] == pytest.approx(2.0, abs=1e-3)
        assert result.params['c1'] == pytest.approx(0.5, abs=1e-3)

    def test_chebyshev_fit(self):
        """Chebyshev (order=2) is linear."""
        form = Chebyshev(order=2, half_width=0.2)
        true_params = {
            'scale': 1.0,
            'c1': 0.3,
            'c2': 0.05,
            'normalization_wavelength': CENTER,
        }
        flux = form.evaluate(WL, CENTER, true_params)

        result = fit_continuum_form(form, WL, flux, ERROR, CENTER)
        assert isinstance(result, ContinuumFitResult)
        assert result.params['scale'] == pytest.approx(1.0, abs=1e-3)

    def test_chi2_red_none_when_dof_zero(self):
        """chi2_red is None when n_data == n_params (dof=0)."""
        # Linear has 2 params; use 2 data points
        form = Linear()
        wl2 = jnp.array([0.9, 1.1])
        flux2 = jnp.array([1.0, 2.0])
        err2 = jnp.array([0.01, 0.01])
        result = fit_continuum_form(form, wl2, flux2, err2, CENTER)
        assert result.chi2_red is None
        assert result.dof == 0

    def test_model_shape_matches_input(self):
        """model in result has same shape as input wavelength."""
        form = Linear()
        flux = jnp.ones(N)
        result = fit_continuum_form(form, WL, flux, ERROR, CENTER)
        assert result.model.shape == (N,)


# ---------------------------------------------------------------------------
# Nonlinear forms
# ---------------------------------------------------------------------------


class TestFitNonlinear:
    """fit_continuum_form with is_linear=False forms (Gauss-Newton)."""

    def test_blackbody_fit_converges(self):
        """Blackbody fit converges to reasonable parameters."""
        form = Blackbody()
        true_params = {
            'scale': 1e-3,
            'temperature': 5000.0,
            'normalization_wavelength': CENTER,
        }
        flux = form.evaluate(WL, CENTER, true_params)

        result = fit_continuum_form(form, WL, flux, ERROR * 1e-3, CENTER)

        assert isinstance(result, ContinuumFitResult)
        assert result.params['temperature'] == pytest.approx(5000.0, rel=0.05)
        assert result.params['scale'] == pytest.approx(1e-3, rel=0.1)

    def test_powerlaw_fit(self):
        """PowerLaw is nonlinear (Gauss-Newton) and recovers params."""
        form = PowerLaw()
        true_params = {'scale': 1.5, 'beta': -1.0, 'normalization_wavelength': CENTER}
        flux = form.evaluate(WL, CENTER, true_params)

        result = fit_continuum_form(form, WL, flux, ERROR, CENTER)

        assert result.params['scale'] == pytest.approx(1.5, rel=0.05)
        assert result.params['beta'] == pytest.approx(-1.0, abs=0.1)

    def test_modified_blackbody_fit_converges(self):
        """ModifiedBlackbody fit runs and returns valid params."""
        form = ModifiedBlackbody()
        true_params = {
            'scale': 1e-3,
            'temperature': 5000.0,
            'beta': 2.0,
            'normalization_wavelength': CENTER,
        }
        flux = form.evaluate(WL, CENTER, true_params)

        result = fit_continuum_form(form, WL, flux, ERROR * 1e-3, CENTER)

        assert isinstance(result, ContinuumFitResult)
        assert 'temperature' in result.params
        assert 'scale' in result.params
        assert 'beta' in result.params

    def test_nonlinear_chi2_red_computed(self):
        """chi2_red is a float (not None) for well-constrained nonlinear fit."""
        form = Blackbody()
        true_params = {
            'scale': 1e-3,
            'temperature': 5000.0,
            'normalization_wavelength': CENTER,
        }
        flux = form.evaluate(WL, CENTER, true_params)

        result = fit_continuum_form(form, WL, flux, ERROR * 1e-3, CENTER)
        assert result.chi2_red is not None

    def test_nonlinear_explicit_normalization_wavelength(self):
        """Explicit normalization_wavelength is used in nonlinear fit."""
        form = Blackbody()
        true_params = {
            'scale': 1e-3,
            'temperature': 5000.0,
            'normalization_wavelength': 0.9,
        }
        flux = form.evaluate(WL, CENTER, true_params)

        result = fit_continuum_form(
            form, WL, flux, ERROR * 1e-3, CENTER, normalization_wavelength=0.9
        )
        assert result.params['normalization_wavelength'] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# _initial_guess helper
# ---------------------------------------------------------------------------


class TestInitialGuess:
    """_initial_guess covers all name branches."""

    def test_scale_branch(self):
        """'scale' param gets median of flux."""
        flux = jnp.array([2.0, 4.0, 6.0])
        x0 = _initial_guess(['scale'], flux)
        assert float(x0[0]) == pytest.approx(4.0, abs=0.1)

    def test_temperature_branch(self):
        """'temperature' param gets 5000.0."""
        flux = jnp.ones(5)
        x0 = _initial_guess(['temperature'], flux)
        assert float(x0[0]) == pytest.approx(5000.0)

    def test_other_branch(self):
        """Unknown param names get 0.0."""
        flux = jnp.ones(5)
        x0 = _initial_guess(['beta'], flux)
        assert float(x0[0]) == pytest.approx(0.0)

    def test_multiple_params(self):
        """Multiple params are stacked in order."""
        flux = jnp.array([3.0, 3.0, 3.0])
        x0 = _initial_guess(['scale', 'temperature', 'beta'], flux)
        assert x0.shape == (3,)
        assert float(x0[0]) == pytest.approx(3.0, abs=0.1)
        assert float(x0[1]) == pytest.approx(5000.0)
        assert float(x0[2]) == pytest.approx(0.0)
