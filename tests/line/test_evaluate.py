"""Tests for pointwise evaluate functions and absorption profiles."""

import jax.numpy as jnp
import pytest

from unite.line.compute import evaluate_lines
from unite.line.functions import (
    evaluate_gaussHermite,
    evaluate_gaussian,
    evaluate_gaussianLaplace,
    evaluate_split_normal,
    evaluate_voigt,
    integrate_gaussHermite,
    integrate_gaussian,
    integrate_gaussianLaplace,
    integrate_split_normal,
    integrate_voigt,
)
from unite.line.profiles import (
    SEMG,
    Cauchy,
    GaussHermite,
    Gaussian,
    Laplace,
    PseudoVoigt,
    SplitNormal,
)

# ---------------------------------------------------------------------------
# Evaluate function normalization
# ---------------------------------------------------------------------------


class TestEvaluateNormalization:
    """Evaluate functions should integrate to ~1 over a wide wavelength range."""

    center = 5000.0
    lsf_fwhm = 5.0
    wavelength = jnp.linspace(4000.0, 6000.0, 10000)

    def test_gaussian(self):
        vals = evaluate_gaussian(self.wavelength, self.center, self.lsf_fwhm, 100.0)
        integral = jnp.trapezoid(vals, self.wavelength)
        assert float(integral) == pytest.approx(1.0, rel=1e-3)

    def test_voigt(self):
        vals = evaluate_voigt(self.wavelength, self.center, self.lsf_fwhm, 80.0, 50.0)
        integral = jnp.trapezoid(vals, self.wavelength)
        assert float(integral) == pytest.approx(1.0, rel=0.05)

    def test_cauchy_via_voigt(self):
        vals = evaluate_voigt(self.wavelength, self.center, self.lsf_fwhm, 0.0, 100.0)
        integral = jnp.trapezoid(vals, self.wavelength)
        # Cauchy has heavy tails so allow wider tolerance
        assert float(integral) == pytest.approx(1.0, rel=0.15)

    def test_gaussianLaplace(self):
        vals = evaluate_gaussianLaplace(
            self.wavelength, self.center, self.lsf_fwhm, 80.0, 50.0
        )
        integral = jnp.trapezoid(vals, self.wavelength)
        assert float(integral) == pytest.approx(1.0, rel=0.05)

    def test_laplace_via_gaussianLaplace(self):
        vals = evaluate_gaussianLaplace(
            self.wavelength, self.center, self.lsf_fwhm, 0.0, 100.0
        )
        integral = jnp.trapezoid(vals, self.wavelength)
        assert float(integral) == pytest.approx(1.0, rel=0.05)

    def test_split_normal(self):
        vals = evaluate_split_normal(
            self.wavelength, self.center, self.lsf_fwhm, 100.0, 60.0
        )
        integral = jnp.trapezoid(vals, self.wavelength)
        assert float(integral) == pytest.approx(1.0, rel=1e-3)

    def test_gaussHermite(self):
        vals = evaluate_gaussHermite(
            self.wavelength, self.center, self.lsf_fwhm, 80.0, 0.1, 0.05
        )
        integral = jnp.trapezoid(vals, self.wavelength)
        assert float(integral) == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Consistency: evaluate * pixel_width ≈ integrate for narrow pixels
# ---------------------------------------------------------------------------


class TestEvaluateIntegrateConsistency:
    """For narrow pixels, evaluate(mid)*dλ should converge to integrate(lo, hi)."""

    center = 5000.0
    lsf_fwhm = 5.0

    def _check_convergence(
        self, integrate_fn, evaluate_fn, args_integrate, args_evaluate
    ):
        """Verify convergence as pixel width shrinks."""
        for n_pix, tol in [(100, 0.1), (500, 0.01), (2000, 1e-3)]:
            edges = jnp.linspace(4900.0, 5100.0, n_pix + 1)
            lo, hi = edges[:-1], edges[1:]
            mid = (lo + hi) / 2.0
            dlam = hi - lo

            integrated = integrate_fn(lo, hi, *args_integrate)
            evaluated = evaluate_fn(mid, *args_evaluate) * dlam

            # Compare near the peak where values are significant
            mask = integrated > 1e-6
            if jnp.any(mask):
                rel_err = jnp.max(
                    jnp.abs(integrated[mask] - evaluated[mask]) / integrated[mask]
                )
                assert float(rel_err) < tol, (
                    f'n_pix={n_pix}: max relative error {float(rel_err):.4e} > {tol}'
                )

    def test_gaussian(self):
        self._check_convergence(
            integrate_gaussian,
            evaluate_gaussian,
            (self.center, self.lsf_fwhm, 100.0),
            (self.center, self.lsf_fwhm, 100.0),
        )

    def test_voigt(self):
        self._check_convergence(
            integrate_voigt,
            evaluate_voigt,
            (self.center, self.lsf_fwhm, 80.0, 50.0),
            (self.center, self.lsf_fwhm, 80.0, 50.0),
        )

    def test_gaussianLaplace(self):
        self._check_convergence(
            integrate_gaussianLaplace,
            evaluate_gaussianLaplace,
            (self.center, self.lsf_fwhm, 80.0, 50.0),
            (self.center, self.lsf_fwhm, 80.0, 50.0),
        )

    def test_split_normal(self):
        self._check_convergence(
            integrate_split_normal,
            evaluate_split_normal,
            (self.center, self.lsf_fwhm, 100.0, 60.0),
            (self.center, self.lsf_fwhm, 100.0, 60.0),
        )

    def test_gaussHermite(self):
        self._check_convergence(
            integrate_gaussHermite,
            evaluate_gaussHermite,
            (self.center, self.lsf_fwhm, 80.0, 0.1, 0.05),
            (self.center, self.lsf_fwhm, 80.0, 0.1, 0.05),
        )


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


class TestEvaluateSymmetry:
    """Symmetric profiles should be symmetric about center."""

    center = 5000.0
    lsf_fwhm = 5.0
    offsets = jnp.array([1.0, 5.0, 20.0, 50.0, 100.0])

    def test_gaussian_symmetric(self):
        left = evaluate_gaussian(
            self.center - self.offsets, self.center, self.lsf_fwhm, 100.0
        )
        right = evaluate_gaussian(
            self.center + self.offsets, self.center, self.lsf_fwhm, 100.0
        )
        assert jnp.allclose(left, right, rtol=1e-10)

    def test_voigt_symmetric(self):
        left = evaluate_voigt(
            self.center - self.offsets, self.center, self.lsf_fwhm, 80.0, 50.0
        )
        right = evaluate_voigt(
            self.center + self.offsets, self.center, self.lsf_fwhm, 80.0, 50.0
        )
        assert jnp.allclose(left, right, rtol=1e-10)


# ---------------------------------------------------------------------------
# Profile.evaluate() method
# ---------------------------------------------------------------------------

_PROFILE_EVALUATE_PARAMS = [
    (Gaussian(), {'fwhm_gauss': 100.0}),
    (Cauchy(), {'fwhm_lorentz': 100.0}),
    (PseudoVoigt(), {'fwhm_gauss': 80.0, 'fwhm_lorentz': 50.0}),
    (Laplace(), {'fwhm_exp': 100.0}),
    (SEMG(), {'fwhm_gauss': 80.0, 'fwhm_exp': 50.0}),
    (GaussHermite(), {'fwhm_gauss': 80.0, 'h3': 0.1, 'h4': 0.05}),
    (SplitNormal(), {'fwhm_blue': 100.0, 'fwhm_red': 60.0}),
]


class TestProfileEvaluateMethod:
    """Test Profile.evaluate() dispatches correctly via evaluate_branch()."""

    @pytest.mark.parametrize('profile,extra_kwargs', _PROFILE_EVALUATE_PARAMS)
    def test_evaluate_integrates_to_near_unity(self, profile, extra_kwargs):
        wavelength = jnp.linspace(4000.0, 6000.0, 10000)
        result = profile.evaluate(
            wavelength, center=5000.0, lsf_fwhm=5.0, **extra_kwargs
        )
        integral = jnp.trapezoid(result, wavelength)
        # Cauchy has heavy tails
        tol = 0.15 if isinstance(profile, Cauchy) else 0.05
        assert float(integral) == pytest.approx(1.0, rel=tol)

    @pytest.mark.parametrize('profile,extra_kwargs', _PROFILE_EVALUATE_PARAMS)
    def test_evaluate_returns_nonneg_near_peak(self, profile, extra_kwargs):
        wavelength = jnp.linspace(4900.0, 5100.0, 200)
        result = profile.evaluate(
            wavelength, center=5000.0, lsf_fwhm=5.0, **extra_kwargs
        )
        # GaussHermite can go slightly negative in the wings
        if not isinstance(profile, GaussHermite):
            assert jnp.all(result >= 0.0)


# ---------------------------------------------------------------------------
# evaluate_lines vmapped dispatch
# ---------------------------------------------------------------------------


class TestEvaluateLines:
    """Test the vmapped evaluate_lines dispatch function."""

    def test_basic_dispatch(self):
        wavelength = jnp.linspace(4900.0, 5100.0, 100)
        centers = jnp.array([5000.0, 5050.0])
        lsf_fwhms = jnp.array([5.0, 5.0])
        p0 = jnp.array([100.0, 80.0])
        p1 = jnp.zeros(2)
        p2 = jnp.zeros(2)
        codes = jnp.array([0, 0])  # both Gaussian

        result = evaluate_lines(wavelength, centers, lsf_fwhms, p0, p1, p2, codes)
        assert result.shape == (2, 100)
        # Each line should have a peak near its center
        assert jnp.argmax(result[0]) != jnp.argmax(result[1])
