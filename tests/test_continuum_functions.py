"""Tests for JAX-jitted continuum evaluation kernels."""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import comb as scipy_comb

from unite.continuum.functions import (
    bernstein_eval,
    bspline_basis,
    bspline_eval,
    chebval,
    planck_function,
)


# ---------------------------------------------------------------------------
# Planck function
# ---------------------------------------------------------------------------


class TestPlanckFunction:
    """Tests for the normalized Planck function."""

    def test_unity_at_pivot(self):
        """Planck function should equal 1.0 at the pivot wavelength."""
        pivot = 0.5
        result = planck_function(jnp.array([pivot]), jnp.array(5000.0), pivot)
        np.testing.assert_allclose(result, 1.0, rtol=1e-5)

    def test_monotonic_for_rayleigh_jeans(self):
        """In the Rayleigh-Jeans regime (long λ), B_λ should decrease with λ."""
        # Hot star, far-red wavelengths
        wl = jnp.linspace(2.0, 5.0, 50)
        result = planck_function(wl, jnp.array(10000.0), 0.5)
        # Should be monotonically decreasing
        assert jnp.all(jnp.diff(result) < 0)

    def test_positive_everywhere(self):
        """Planck function should always be positive."""
        wl = jnp.linspace(0.1, 10.0, 100)
        result = planck_function(wl, jnp.array(5000.0), 0.5)
        assert jnp.all(result > 0)

    def test_finite_at_extremes(self):
        """Should remain finite at extreme temperatures."""
        wl = jnp.array([0.5])
        for temp in [50.0, 100.0, 50000.0, 100000.0]:
            result = planck_function(wl, jnp.array(temp), 0.5)
            assert jnp.all(jnp.isfinite(result))

    def test_hotter_is_bluer(self):
        """A hotter blackbody should be brighter in the blue relative to pivot."""
        wl_blue = jnp.array([0.3])
        cool = planck_function(wl_blue, jnp.array(3000.0), 0.5)
        hot = planck_function(wl_blue, jnp.array(10000.0), 0.5)
        assert hot.item() > cool.item()


# ---------------------------------------------------------------------------
# Chebyshev polynomial
# ---------------------------------------------------------------------------


class TestChebval:
    """Tests for Chebyshev polynomial evaluation."""

    def test_constant(self):
        """T0(x) = 1 → chebval([c0], x) = c0."""
        x = jnp.linspace(-1, 1, 10)
        result = chebval(x, [3.0])
        np.testing.assert_allclose(result, 3.0)

    def test_linear(self):
        """c0 + c1*x."""
        x = jnp.linspace(-1, 1, 10)
        result = chebval(x, [2.0, 3.0])
        expected = 2.0 + 3.0 * x
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_quadratic(self):
        """c0*T0 + c1*T1 + c2*T2 where T2 = 2x^2 - 1."""
        x = jnp.linspace(-1, 1, 20)
        result = chebval(x, [1.0, 0.0, 1.0])
        # T0 + T2 = 1 + 2x^2 - 1 = 2x^2
        expected = 2.0 * x**2
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_agrees_with_numpy(self):
        """Should agree with numpy.polynomial.chebyshev.chebval."""
        from numpy.polynomial.chebyshev import chebval as np_chebval

        x = jnp.linspace(-1, 1, 50)
        coeffs = [1.5, -0.3, 0.7, 0.1]
        result = chebval(x, coeffs)
        expected = np_chebval(np.array(x), coeffs)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# ---------------------------------------------------------------------------
# B-spline
# ---------------------------------------------------------------------------


class TestBSpline:
    """Tests for B-spline basis and evaluation."""

    def _make_clamped_knots(self, n_basis, degree, low=0.0, high=1.0):
        """Create a clamped knot vector."""
        n_internal = n_basis - degree + 1
        internal = np.linspace(low, high, n_internal)
        knots = np.concatenate([
            np.full(degree, low),
            internal,
            np.full(degree, high),
        ])
        return jnp.asarray(knots)

    def test_basis_partition_of_unity(self):
        """B-spline basis should sum to 1 at every interior point."""
        n_basis = 8
        degree = 3
        knots = self._make_clamped_knots(n_basis, degree)
        t = jnp.linspace(0.01, 0.99, 50)
        basis = bspline_basis(t, knots, degree)
        row_sums = jnp.sum(basis, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_basis_nonnegative(self):
        """B-spline basis should be non-negative."""
        knots = self._make_clamped_knots(6, 3)
        t = jnp.linspace(0.0, 1.0, 50)
        basis = bspline_basis(t, knots, 3)
        assert jnp.all(basis >= -1e-10)

    def test_basis_shape(self):
        """Basis matrix should have shape (n_points, n_basis)."""
        n_basis = 8
        knots = self._make_clamped_knots(n_basis, 3)
        t = jnp.linspace(0.0, 1.0, 30)
        basis = bspline_basis(t, knots, 3)
        assert basis.shape == (30, n_basis)

    def test_eval_constant(self):
        """Constant coefficients → constant output."""
        n_basis = 6
        degree = 3
        knots = self._make_clamped_knots(n_basis, degree)
        coeffs = jnp.full(n_basis, 5.0)
        wl = jnp.linspace(0.01, 0.99, 20)
        result = bspline_eval(wl, coeffs, knots, degree)
        np.testing.assert_allclose(result, 5.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Bernstein polynomial
# ---------------------------------------------------------------------------


class TestBernstein:
    """Tests for Bernstein polynomial evaluation."""

    def test_constant(self):
        """All coefficients equal → constant output."""
        n = 4
        coeffs = jnp.full(n + 1, 3.0)
        binom = jnp.array([float(scipy_comb(n, i, exact=True)) for i in range(n + 1)])
        wl = jnp.linspace(0.5, 1.5, 20)
        result = bernstein_eval(wl, coeffs, 0.5, 1.5, binom)
        np.testing.assert_allclose(result, 3.0, atol=1e-5)

    def test_linear(self):
        """Degree-1 Bernstein: c0*(1-t) + c1*t = linear interpolation."""
        coeffs = jnp.array([2.0, 8.0])
        binom = jnp.array([1.0, 1.0])
        wl = jnp.array([0.0, 0.5, 1.0])
        result = bernstein_eval(wl, coeffs, 0.0, 1.0, binom)
        np.testing.assert_allclose(result, [2.0, 5.0, 8.0], atol=1e-5)

    def test_nonnegative_with_positive_coeffs(self):
        """Positive coefficients → positive output."""
        n = 5
        coeffs = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0, 4.0])
        binom = jnp.array([float(scipy_comb(n, i, exact=True)) for i in range(n + 1)])
        wl = jnp.linspace(0.0, 1.0, 50)
        result = bernstein_eval(wl, coeffs, 0.0, 1.0, binom)
        assert jnp.all(result > 0)

    def test_endpoints(self):
        """Bernstein evaluates to c0 at low and c_n at high."""
        n = 3
        coeffs = jnp.array([2.0, 5.0, 3.0, 7.0])
        binom = jnp.array([float(scipy_comb(n, i, exact=True)) for i in range(n + 1)])
        wl = jnp.array([0.0, 1.0])
        result = bernstein_eval(wl, coeffs, 0.0, 1.0, binom)
        np.testing.assert_allclose(result[0], 2.0, atol=1e-5)
        np.testing.assert_allclose(result[1], 7.0, atol=1e-5)
