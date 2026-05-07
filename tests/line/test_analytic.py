"""Tests for profile integration: raw function normalization and Profile.integrate() dispatch."""

import jax.numpy as jnp
import pytest

from unite.line import functions
from unite.line.library import (
    SEMG,
    BoxGauss,
    Cauchy,
    GaussHermite,
    Gaussian,
    Laplace,
    PseudoVoigt,
    SkewNormal,
    SkewVoigt,
    SplitNormal,
)

# ---------------------------------------------------------------------------
# Integration function normalization
# ---------------------------------------------------------------------------


class TestIntegrationNormalization:
    """Verify that all integration kernels sum to ~1.0 over wide bins."""

    low = jnp.linspace(0, 10000, 1000)[:-1]
    high = jnp.linspace(0, 10000, 1000)[1:]
    center = jnp.array([5000.0])
    lsf_fwhm = jnp.array([10.0])

    def test_gaussian(self):
        result = functions.integrate_gaussian(
            self.low, self.high, self.center, self.lsf_fwhm, jnp.array([100.0])
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-5)

    def test_cauchy(self):
        result = functions._cauchy_cdf_diff(
            self.low - self.center, self.high - self.center, self.lsf_fwhm
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=5e-3)

    def test_voigt_pure_lorentzian(self):
        result = functions.integrate_voigt(
            self.low, self.high, self.center, 0.0, 0.0, jnp.array([100.0])
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=5e-2)

    def test_laplace(self):
        result = functions._laplace_cdf_diff(
            self.low - self.center, self.high - self.center, jnp.array([100.0])
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-4)

    def test_voigt(self):
        result = functions.integrate_voigt(
            self.low,
            self.high,
            self.center,
            self.lsf_fwhm,
            jnp.array([80.0]),
            jnp.array([50.0]),
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=5e-3)

    def test_gaussianLaplace(self):
        result = functions.integrate_gaussianLaplace(
            self.low,
            self.high,
            self.center,
            self.lsf_fwhm,
            jnp.array([80.0]),
            jnp.array([50.0]),
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-4)

    def test_gaussHermite(self):
        result = functions.integrate_gaussHermite(
            self.low,
            self.high,
            self.center,
            self.lsf_fwhm,
            jnp.array([80.0]),
            jnp.array([0.1]),
            jnp.array([0.05]),
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-5)

    def test_split_normal(self):
        result = functions.integrate_split_normal(
            self.low,
            self.high,
            self.center,
            self.lsf_fwhm,
            jnp.array([100.0]),
            jnp.array([60.0]),
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-5)

    def test_boxGauss(self):
        result = functions.integrate_boxGauss(
            self.low,
            self.high,
            self.center,
            self.lsf_fwhm,
            jnp.array([300.0]),
            jnp.array([100.0]),
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-5)

    def test_skewNormal(self):
        result = functions.integrate_skewNormal(
            self.low,
            self.high,
            self.center,
            self.lsf_fwhm,
            jnp.array([80.0]),
            jnp.array([2.0]),
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-5)

    def test_boxGauss_wide_box(self):
        result = functions.integrate_boxGauss(
            self.low,
            self.high,
            self.center,
            self.lsf_fwhm,
            jnp.array([500.0]),
            jnp.array([0.0]),
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-5)

    def test_boxGauss_large_gauss(self):
        result = functions.integrate_boxGauss(
            self.low,
            self.high,
            self.center,
            self.lsf_fwhm,
            jnp.array([0.1]),
            jnp.array([500.0]),
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# Profile.integrate() method dispatch
# ---------------------------------------------------------------------------

_PROFILE_PARAMS = [
    (Gaussian(), {'fwhm_gauss': 100.0}),
    (Cauchy(), {'fwhm_lorentz': 100.0}),
    (PseudoVoigt(), {'fwhm_gauss': 80.0, 'fwhm_lorentz': 50.0}),
    (Laplace(), {'fwhm_exp': 100.0}),
    (SEMG(), {'fwhm_gauss': 80.0, 'fwhm_exp': 50.0}),
    (GaussHermite(), {'fwhm_gauss': 80.0, 'h3': 0.1, 'h4': 0.05}),
    (SplitNormal(), {'fwhm_blue': 100.0, 'fwhm_red': 60.0}),
    (SkewVoigt(), {'fwhm_gauss': 80.0, 'fwhm_lorentz': 50.0, 'alpha': 2.0}),
    (BoxGauss(), {'fwhm_box': 300.0, 'fwhm_gauss': 100.0}),
    (SkewNormal(), {'fwhm_gauss': 80.0, 'alpha': 2.0}),
]


class TestProfileIntegrateMethod:
    """Test Profile.integrate() dispatches correctly via functions.integrate_branch()."""

    @pytest.mark.parametrize('profile,extra_kwargs', _PROFILE_PARAMS)
    def test_integrate_sums_to_near_unity(self, profile, extra_kwargs):
        low = jnp.linspace(4500.0, 5500.0, 500)[:-1]
        high = jnp.linspace(4500.0, 5500.0, 500)[1:]
        result = profile.integrate(
            low, high, center=5000.0, lsf_fwhm=5.0, **extra_kwargs
        )
        # Cauchy has heavy tails so allow wider tolerance
        assert float(jnp.sum(result)) == pytest.approx(1.0, rel=0.1)

    @pytest.mark.parametrize('profile,extra_kwargs', _PROFILE_PARAMS)
    def test_integrate_returns_nonneg(self, profile, extra_kwargs):
        low = jnp.linspace(4900.0, 5100.0, 50)[:-1]
        high = jnp.linspace(4900.0, 5100.0, 50)[1:]
        result = profile.integrate(
            low, high, center=5000.0, lsf_fwhm=5.0, **extra_kwargs
        )
        assert jnp.all(result >= 0.0)


# ---------------------------------------------------------------------------
# integrate_skewNormal
# ---------------------------------------------------------------------------


class TestIntegrateSkewNormal:
    """Tests for the public integrate_skewNormal kernel."""

    center = 5000.0
    lsf_fwhm = 5.0
    edges = jnp.linspace(4000.0, 6000.0, 2001)
    lo, hi = edges[:-1], edges[1:]

    def test_alpha_zero_matches_gaussian(self):
        """integrate_skewNormal with alpha=0 is identical to integrate_gaussian."""
        skew = functions.integrate_skewNormal(
            self.lo, self.hi, self.center, self.lsf_fwhm, 80.0, 0.0
        )
        gauss = functions.integrate_gaussian(
            self.lo, self.hi, self.center, self.lsf_fwhm, 80.0
        )
        assert jnp.allclose(skew, gauss, rtol=1e-6)

    def test_normalization_positive_alpha(self):
        """integrate_skewNormal sums to 1 for positive alpha."""
        result = functions.integrate_skewNormal(
            self.lo, self.hi, self.center, self.lsf_fwhm, 80.0, 3.0
        )
        assert float(jnp.sum(result)) == pytest.approx(1.0, rel=1e-5)

    def test_normalization_negative_alpha(self):
        """integrate_skewNormal sums to 1 for negative alpha."""
        result = functions.integrate_skewNormal(
            self.lo, self.hi, self.center, self.lsf_fwhm, 80.0, -3.0
        )
        assert float(jnp.sum(result)) == pytest.approx(1.0, rel=1e-5)

    def test_skew_shifts_mass(self):
        """Positive alpha shifts mass to the red side of center."""
        result_pos = functions.integrate_skewNormal(
            self.lo, self.hi, self.center, self.lsf_fwhm, 80.0, 5.0
        )
        result_neg = functions.integrate_skewNormal(
            self.lo, self.hi, self.center, self.lsf_fwhm, 80.0, -5.0
        )
        red_mask = self.hi > self.center
        assert float(jnp.sum(result_pos[red_mask])) > float(
            jnp.sum(result_neg[red_mask])
        )

    def test_alpha_antisymmetry(self):
        """Flipping the sign of alpha mirrors the profile: f(x; +alpha) = f(-x; -alpha)."""
        result_pos = functions.integrate_skewNormal(
            self.lo, self.hi, self.center, self.lsf_fwhm, 80.0, 3.0
        )
        result_neg = functions.integrate_skewNormal(
            # Mirror the bins about center
            2 * self.center - self.hi,
            2 * self.center - self.lo,
            self.center,
            self.lsf_fwhm,
            80.0,
            -3.0,
        )
        assert jnp.allclose(result_pos, result_neg, rtol=1e-6)
