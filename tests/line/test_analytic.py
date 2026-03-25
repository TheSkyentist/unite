"""Tests for profile integration: raw function normalization and Profile.integrate() dispatch."""

import jax.numpy as jnp
import pytest

from unite.line import functions
from unite.line.library import (
    SEMG,
    Cauchy,
    GaussHermite,
    Gaussian,
    Laplace,
    PseudoVoigt,
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
