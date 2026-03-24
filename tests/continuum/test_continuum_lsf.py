"""Tests for continuum LSF convolution and pixel integration."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter1d

from unite.continuum.library import (
    Bernstein,
    Blackbody,
    Chebyshev,
    Linear,
    Polynomial,
    PowerLaw,
    _gaussian_convolve_poly,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# FWHM to sigma
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _numerical_convolve(wavelength, flux, lsf_fwhm):
    """Convolve flux with a Gaussian LSF numerically (constant sigma approx)."""
    # Use the median LSF FWHM for a uniform convolution
    sigma_pix = float(np.median(lsf_fwhm)) * _FWHM_TO_SIGMA
    dlam = float(wavelength[1] - wavelength[0])  # assume uniform grid
    sigma_in_pixels = sigma_pix / dlam
    if sigma_in_pixels < 0.01:
        return flux
    return gaussian_filter1d(flux, sigma_in_pixels, mode='nearest')


# ---------------------------------------------------------------------------
# _gaussian_convolve_poly unit tests
# ---------------------------------------------------------------------------


class TestGaussianConvolvePoly:
    """Direct tests for the polynomial convolution kernel."""

    def test_zero_fwhm_identity(self):
        """With FWHM=0 the polynomial should be unchanged."""
        coeffs = jnp.array([3.0, -2.0, 1.0])  # 3x^2 - 2x + 1
        result = _gaussian_convolve_poly(coeffs, 0.0)
        np.testing.assert_allclose(result, coeffs, atol=1e-12)

    def test_constant_unchanged(self):
        """A constant polynomial is unchanged by any convolution."""
        coeffs = jnp.array([5.0])
        result = _gaussian_convolve_poly(coeffs, 1.0)
        np.testing.assert_allclose(result, [5.0], atol=1e-12)

    def test_linear_unchanged(self):
        """A linear polynomial (ax + b) is unchanged by Gaussian convolution."""
        coeffs = jnp.array([3.0, 7.0])  # 3x + 7
        result = _gaussian_convolve_poly(coeffs, 2.5)
        np.testing.assert_allclose(result, coeffs, atol=1e-12)

    def test_quadratic_adds_variance(self):
        """For c*x^2, convolution with N(0, s^2) gives c*x^2 + c*s^2."""
        c = 2.0
        fwhm = 1.0
        sigma2 = (fwhm * _FWHM_TO_SIGMA) ** 2
        coeffs = jnp.array([c, 0.0, 0.0])  # 2x^2
        result = _gaussian_convolve_poly(coeffs, fwhm)
        expected = jnp.array([c, 0.0, c * sigma2])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_cubic(self):
        """For x^3, convolution gives x^3 + 3*s^2*x."""
        fwhm = 0.5
        sigma2 = (fwhm * _FWHM_TO_SIGMA) ** 2
        coeffs = jnp.array([1.0, 0.0, 0.0, 0.0])  # x^3
        result = _gaussian_convolve_poly(coeffs, fwhm)
        # x^3 + 3*s^2*x (3 = C(3,2) * 1!! = 3)
        expected = jnp.array([1.0, 0.0, 3.0 * sigma2, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_matches_numerical_convolution(self):
        """Analytic poly convolution matches numerical on a fine grid."""
        coeffs = jnp.array([0.5, -1.0, 3.0, 2.0])  # 0.5x^3 - x^2 + 3x + 2
        fwhm = 0.1  # smaller FWHM for better numerical accuracy
        x = jnp.linspace(-2, 2, 20000)
        # Evaluate original polynomial
        orig = jnp.polyval(coeffs, x)
        # Numerical convolution
        numerical = _numerical_convolve(x, np.array(orig), fwhm * np.ones_like(x))
        # Analytic convolution
        conv_coeffs = _gaussian_convolve_poly(coeffs, fwhm)
        analytic = jnp.polyval(conv_coeffs, x)
        # Compare in interior (avoid edge effects from gaussian_filter1d)
        mask = (x > -1.0) & (x < 1.0)
        np.testing.assert_allclose(
            analytic[mask], numerical[mask], rtol=1e-4, atol=1e-4
        )


# ---------------------------------------------------------------------------
# ContinuumForm.evaluate with lsf_fwhm=0 (backward compatibility)
# ---------------------------------------------------------------------------


class TestEvaluateZeroLSF:
    """All forms should produce identical results with lsf_fwhm=0."""

    wavelength = jnp.linspace(1.0, 2.0, 100)
    center = 1.5
    obs_low = 1.0
    obs_high = 2.0

    def _params(self, form, **overrides):
        priors = form.default_priors(self.center)
        params = {}
        for k, v in priors.items():
            if k in overrides:
                params[k] = overrides[k]
            elif hasattr(v, 'value'):
                params[k] = v.value
            else:
                params[k] = (v.low + v.high) / 2
        return params

    @pytest.mark.parametrize(
        'form',
        [
            Linear(),
            Polynomial(degree=2),
            Polynomial(degree=3),
            Chebyshev(order=2),
            Bernstein(degree=3),
            PowerLaw(),
        ],
    )
    def test_zero_lsf_matches_no_lsf(self, form):
        params = self._params(form)
        result_default = form.evaluate(
            self.wavelength, self.center, params, self.obs_low, self.obs_high
        )
        result_zero = form.evaluate(
            self.wavelength, self.center, params, self.obs_low, self.obs_high, 0.0
        )
        np.testing.assert_allclose(result_zero, result_default, atol=1e-12)


# ---------------------------------------------------------------------------
# Linear is unchanged by LSF
# ---------------------------------------------------------------------------


class TestLinearLSFInvariance:
    """Linear continuum should be unchanged by any LSF FWHM."""

    def test_linear_unchanged_by_lsf(self):
        form = Linear()
        wavelength = jnp.linspace(1.0, 2.0, 100)
        params = {'scale': 1.5, 'angle': 0.3, 'norm_wav': 1.5}
        result_no_lsf = form.evaluate(wavelength, 1.5, params, 1.0, 2.0, 0.0)
        result_with_lsf = form.evaluate(wavelength, 1.5, params, 1.0, 2.0, 0.05)
        np.testing.assert_allclose(result_with_lsf, result_no_lsf, atol=1e-12)

    def test_linear_integrate_equals_evaluate(self):
        """For Linear, integrate should equal evaluate at midpoints."""
        form = Linear()
        low = jnp.linspace(1.0, 1.9, 90)
        high = low + 0.01
        params = {'scale': 1.5, 'angle': 0.3, 'norm_wav': 1.5}
        mid = (low + high) / 2.0
        integrated = form.integrate(low, high, 1.5, params, 1.0, 2.0, 0.05)
        evaluated = form.evaluate(mid, 1.5, params, 1.0, 2.0, 0.05)
        np.testing.assert_allclose(integrated, evaluated, atol=1e-12)


# ---------------------------------------------------------------------------
# Polynomial LSF convolution
# ---------------------------------------------------------------------------


class TestPolynomialLSF:
    """Test that Polynomial.evaluate with LSF matches numerical convolution."""

    def test_degree2_lsf(self):
        form = Polynomial(degree=2)
        wavelength = jnp.linspace(1.0, 2.0, 10000)
        nw = 1.5
        params = {'scale': 1.0, 'c1': 0.5, 'c2': 0.3, 'norm_wav': nw}
        fwhm = 0.02

        result_lsf = form.evaluate(wavelength, 1.5, params, 1.0, 2.0, fwhm)
        result_no_lsf = form.evaluate(wavelength, 1.5, params, 1.0, 2.0, 0.0)

        # Numerical convolution for comparison
        numerical = _numerical_convolve(
            wavelength, np.array(result_no_lsf), fwhm * np.ones_like(wavelength)
        )

        # Compare in interior
        mask = (wavelength > 1.2) & (wavelength < 1.8)
        np.testing.assert_allclose(
            result_lsf[mask], numerical[mask], rtol=1e-4, atol=1e-6
        )

    def test_degree3_lsf(self):
        form = Polynomial(degree=3)
        wavelength = jnp.linspace(1.0, 2.0, 10000)
        nw = 1.5
        params = {'scale': 1.0, 'c1': 0.5, 'c2': 0.3, 'c3': -0.1, 'norm_wav': nw}
        fwhm = 0.02

        result_lsf = form.evaluate(wavelength, 1.5, params, 1.0, 2.0, fwhm)
        result_no_lsf = form.evaluate(wavelength, 1.5, params, 1.0, 2.0, 0.0)

        numerical = _numerical_convolve(
            wavelength, np.array(result_no_lsf), fwhm * np.ones_like(wavelength)
        )

        mask = (wavelength > 1.2) & (wavelength < 1.8)
        np.testing.assert_allclose(
            result_lsf[mask], numerical[mask], rtol=1e-4, atol=1e-6
        )


# ---------------------------------------------------------------------------
# Chebyshev LSF convolution
# ---------------------------------------------------------------------------


class TestChebyshevLSF:
    """Test that Chebyshev.evaluate with LSF matches numerical convolution."""

    def test_order2_lsf(self):
        form = Chebyshev(order=2)
        wavelength = jnp.linspace(1.0, 2.0, 10000)
        center = 1.5
        params = {'scale': 1.0, 'c1': 0.1, 'c2': 0.05, 'norm_wav': center}
        fwhm = 0.02

        result_lsf = form.evaluate(wavelength, center, params, 1.0, 2.0, fwhm)
        result_no_lsf = form.evaluate(wavelength, center, params, 1.0, 2.0, 0.0)

        numerical = _numerical_convolve(
            wavelength, np.array(result_no_lsf), fwhm * np.ones_like(wavelength)
        )

        mask = (wavelength > 1.2) & (wavelength < 1.8)
        np.testing.assert_allclose(
            result_lsf[mask], numerical[mask], rtol=1e-3, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Bernstein LSF convolution
# ---------------------------------------------------------------------------


class TestBernsteinLSF:
    """Test that Bernstein.evaluate with LSF matches numerical convolution."""

    def test_degree3_lsf(self):
        form = Bernstein(degree=3)
        wavelength = jnp.linspace(1.0, 2.0, 10000)
        center = 1.5
        params = {
            'scale': 1.0,
            'coeff_1': 1.1,
            'coeff_2': 0.9,
            'coeff_3': 1.2,
            'norm_wav': center,
        }
        fwhm = 0.02

        result_lsf = form.evaluate(wavelength, center, params, 1.0, 2.0, fwhm)
        result_no_lsf = form.evaluate(wavelength, center, params, 1.0, 2.0, 0.0)

        numerical = _numerical_convolve(
            wavelength, np.array(result_no_lsf), fwhm * np.ones_like(wavelength)
        )

        mask = (wavelength > 1.2) & (wavelength < 1.8)
        np.testing.assert_allclose(
            result_lsf[mask], numerical[mask], rtol=1e-3, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Non-convolvable forms ignore LSF gracefully
# ---------------------------------------------------------------------------


class TestNonConvolvableFormsIgnoreLSF:
    """PowerLaw, Blackbody, etc. should work with lsf_fwhm but not change."""

    wavelength = jnp.linspace(1.0, 2.0, 100)

    def test_powerlaw_ignores_lsf(self):
        form = PowerLaw()
        params = {'scale': 1.0, 'beta': -1.5, 'norm_wav': 1.5}
        r0 = form.evaluate(self.wavelength, 1.5, params, 1.0, 2.0, 0.0)
        r1 = form.evaluate(self.wavelength, 1.5, params, 1.0, 2.0, 0.05)
        np.testing.assert_allclose(r1, r0, atol=1e-12)

    def test_blackbody_ignores_lsf(self):
        form = Blackbody()
        form._prepare(
            1.0 * __import__('astropy').units.um, 2.0 * __import__('astropy').units.um
        )
        params = {'scale': 1.0, 'temperature': 5000.0, 'norm_wav': 1.5}
        r0 = form.evaluate(self.wavelength, 1.5, params, 1.0, 2.0, 0.0)
        r1 = form.evaluate(self.wavelength, 1.5, params, 1.0, 2.0, 0.05)
        np.testing.assert_allclose(r1, r0, atol=1e-12)


# ---------------------------------------------------------------------------
# ContinuumForm.integrate (forms without override: evaluate at midpoints)
# ---------------------------------------------------------------------------


class TestDefaultIntegrate:
    """Forms without an integrate override should evaluate at pixel midpoints."""

    def test_blackbody_integrate_is_midpoint(self):
        from astropy import units as u

        form = Blackbody()
        form._prepare(1.0 * u.um, 2.0 * u.um)
        low = jnp.linspace(1.0, 1.9, 90)
        high = low + 0.01
        params = {'scale': 1.0, 'temperature': 5000.0, 'norm_wav': 1.5}
        mid = (low + high) / 2.0
        integrated = form.integrate(low, high, 1.5, params, 1.0, 2.0, 0.0)
        evaluated = form.evaluate(mid, 1.5, params, 1.0, 2.0, 0.0)
        np.testing.assert_allclose(integrated, evaluated, atol=1e-12)


# ---------------------------------------------------------------------------
# Analytic integrate: Polynomial
# ---------------------------------------------------------------------------


class TestPolynomialIntegrate:
    """Polynomial.integrate should compute exact pixel-averaged values."""

    def test_linear_integrate_exact(self):
        """For degree-1 poly, integrate should match midpoint (exact for linear)."""
        form = Polynomial(degree=1)
        low = jnp.linspace(1.0, 1.9, 90)
        high = low + 0.01
        params = {'scale': 1.0, 'c1': 2.0, 'norm_wav': 1.5}
        mid = (low + high) / 2.0
        integrated = form.integrate(low, high, 1.5, params, 1.0, 2.0, 0.0)
        evaluated = form.evaluate(mid, 1.5, params, 1.0, 2.0, 0.0)
        np.testing.assert_allclose(integrated, evaluated, atol=1e-12)

    def test_quadratic_integrate_exact(self):
        """For degree-2 poly with wide pixels, integrate != midpoint eval."""
        form = Polynomial(degree=2)
        # Use wide pixels so midpoint approximation differs from exact integral
        low = jnp.array([1.0, 1.2, 1.4, 1.6, 1.8])
        high = jnp.array([1.2, 1.4, 1.6, 1.8, 2.0])
        params = {'scale': 1.0, 'c1': 0.5, 'c2': 3.0, 'norm_wav': 1.5}
        integrated = form.integrate(low, high, 1.5, params, 1.0, 2.0, 0.0)
        # Verify against numerical integration (scipy)
        from scipy.integrate import quad

        def f(w):
            x = w - 1.5
            return 1.0 + 0.5 * x + 3.0 * x**2

        for i in range(len(low)):
            lo, hi = float(low[i]), float(high[i])
            expected, _ = quad(f, lo, hi)
            expected /= hi - lo
            np.testing.assert_allclose(float(integrated[i]), expected, rtol=1e-10)

    def test_integrate_with_lsf(self):
        """Polynomial integrate with LSF should differ from no-LSF for degree >= 2."""
        form = Polynomial(degree=2)
        low = jnp.array([1.0, 1.2, 1.4, 1.6, 1.8])
        high = jnp.array([1.2, 1.4, 1.6, 1.8, 2.0])
        params = {'scale': 1.0, 'c1': 0.5, 'c2': 3.0, 'norm_wav': 1.5}
        int_no_lsf = form.integrate(low, high, 1.5, params, 1.0, 2.0, 0.0)
        int_with_lsf = form.integrate(low, high, 1.5, params, 1.0, 2.0, 0.05)
        # Should differ (LSF adds variance to the quadratic term)
        assert not jnp.allclose(int_no_lsf, int_with_lsf)


# ---------------------------------------------------------------------------
# Analytic integrate: Chebyshev
# ---------------------------------------------------------------------------


class TestChebyshevIntegrate:
    """Chebyshev.integrate should compute exact pixel-averaged values."""

    def test_order0_integrate(self):
        """Order-0 Chebyshev is constant: integrate == evaluate at midpoint."""
        form = Chebyshev(order=0)
        low = jnp.linspace(1.0, 1.8, 40)
        high = low + 0.05
        params = {'scale': 2.5, 'norm_wav': 1.5}
        mid = (low + high) / 2.0
        integrated = form.integrate(low, high, 1.5, params, 1.0, 2.0, 0.0)
        evaluated = form.evaluate(mid, 1.5, params, 1.0, 2.0, 0.0)
        np.testing.assert_allclose(integrated, evaluated, atol=1e-10)

    def test_order2_integrate_matches_scipy(self):
        """Order-2 Chebyshev integrate matches numerical integration."""
        form = Chebyshev(order=2)
        low = jnp.array([1.0, 1.2, 1.4, 1.6, 1.8])
        high = jnp.array([1.2, 1.4, 1.6, 1.8, 2.0])
        center = 1.5
        params = {'scale': 1.0, 'c1': 0.2, 'c2': 0.1, 'norm_wav': center}
        integrated = form.integrate(low, high, center, params, 1.0, 2.0, 0.0)
        # Compare against evaluate at many sub-points (numerical average)
        for i in range(len(low)):
            lo, hi = float(low[i]), float(high[i])
            wl = jnp.linspace(lo, hi, 10000)
            vals = form.evaluate(wl, center, params, 1.0, 2.0, 0.0)
            numerical_avg = float(jnp.mean(vals))
            np.testing.assert_allclose(float(integrated[i]), numerical_avg, rtol=1e-4)


# ---------------------------------------------------------------------------
# Analytic integrate: Bernstein
# ---------------------------------------------------------------------------


class TestBernsteinIntegrate:
    """Bernstein.integrate should compute exact pixel-averaged values."""

    def test_degree3_integrate_matches_numerical(self):
        form = Bernstein(degree=3)
        low = jnp.array([1.0, 1.2, 1.4, 1.6, 1.8])
        high = jnp.array([1.2, 1.4, 1.6, 1.8, 2.0])
        center = 1.5
        params = {
            'scale': 1.0,
            'coeff_1': 1.1,
            'coeff_2': 0.9,
            'coeff_3': 1.2,
            'norm_wav': center,
        }
        integrated = form.integrate(low, high, center, params, 1.0, 2.0, 0.0)
        for i in range(len(low)):
            lo, hi = float(low[i]), float(high[i])
            wl = jnp.linspace(lo, hi, 10000)
            vals = form.evaluate(wl, center, params, 1.0, 2.0, 0.0)
            numerical_avg = float(jnp.mean(vals))
            np.testing.assert_allclose(float(integrated[i]), numerical_avg, rtol=1e-4)


# ---------------------------------------------------------------------------
# Analytic integrate: PowerLaw
# ---------------------------------------------------------------------------


# class TestPowerLawIntegrate:
#     """PowerLaw.integrate should compute exact pixel-averaged values."""

#     def test_powerlaw_integrate_matches_scipy(self):
#         form = PowerLaw()
#         low = jnp.array([1.0, 1.2, 1.4, 1.6, 1.8])
#         high = jnp.array([1.2, 1.4, 1.6, 1.8, 2.0])
#         params = {'scale': 2.0, 'beta': -1.5, 'norm_wav': 1.5}
#         integrated = form.integrate(low, high, 1.5, params, 1.0, 2.0, 0.0)
#         from scipy.integrate import quad

#         def f(w):
#             return 2.0 * (w / 1.5) ** (-1.5)

#         for i in range(len(low)):
#             lo, hi = float(low[i]), float(high[i])
#             expected, _ = quad(f, lo, hi)
#             expected /= hi - lo
#             np.testing.assert_allclose(float(integrated[i]), expected, rtol=1e-10)

#     def test_powerlaw_narrow_pixels_close_to_midpoint(self):
#         """With narrow pixels, integrate should be close to midpoint eval."""
#         form = PowerLaw()
#         low = jnp.linspace(1.0, 1.9, 90)
#         high = low + 0.001  # very narrow pixels
#         params = {'scale': 1.0, 'beta': -1.5, 'norm_wav': 1.5}
#         mid = (low + high) / 2.0
#         integrated = form.integrate(low, high, 1.5, params, 1.0, 2.0, 0.0)
#         evaluated = form.evaluate(mid, 1.5, params, 1.0, 2.0, 0.0)
#         np.testing.assert_allclose(integrated, evaluated, rtol=1e-5)

#     def test_powerlaw_beta_zero(self):
#         """With beta=0, power law is constant: integrate == scale everywhere."""
#         form = PowerLaw()
#         low = jnp.array([1.0, 1.5, 1.8])
#         high = jnp.array([1.2, 1.7, 2.0])
#         params = {'scale': 3.0, 'beta': 0.0, 'norm_wav': 1.5}
#         integrated = form.integrate(low, high, 1.5, params, 1.0, 2.0, 0.0)
#         np.testing.assert_allclose(integrated, 3.0, atol=1e-12)
