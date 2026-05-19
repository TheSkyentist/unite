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


def _per_pixel(cum):
    """Convert a length-E cumulative array to length-(E-1) per-pixel integrals."""
    return jnp.diff(cum)


# ---------------------------------------------------------------------------
# Integration function normalization
# ---------------------------------------------------------------------------


class TestIntegrationNormalization:
    """Verify that all integration kernels sum to ~1.0 over wide bins.

    The new kernels return cumulative arrays at edges; ``cdf[-1] - cdf[0]``
    is the total probability mass and ``jnp.diff(cdf)`` gives per-pixel
    integrals.
    """

    edges = jnp.linspace(0.0, 10000.0, 1000)
    lsf_at_edges = jnp.full_like(edges, 10.0)
    center = 5000.0

    def test_gaussian(self):
        cdf = functions.integrate_gaussian(
            self.edges, self.lsf_at_edges, self.center, 100.0
        )
        assert jnp.isclose(cdf[-1] - cdf[0], 1.0, rtol=1e-5)

    def test_voigt_pure_lorentzian(self):
        cdf = functions.integrate_voigt(
            self.edges, jnp.zeros_like(self.edges), self.center, 0.0, 100.0
        )
        # Heavy tails: looser tolerance.
        assert jnp.isclose(cdf[-1] - cdf[0], 1.0, rtol=5e-2)

    def test_voigt(self):
        cdf = functions.integrate_voigt(
            self.edges, self.lsf_at_edges, self.center, 80.0, 50.0
        )
        assert jnp.isclose(cdf[-1] - cdf[0], 1.0, rtol=5e-3)

    def test_gaussianLaplace(self):
        cdf = functions.integrate_gaussianLaplace(
            self.edges, self.lsf_at_edges, self.center, 80.0, 50.0
        )
        assert jnp.isclose(cdf[-1] - cdf[0], 1.0, rtol=1e-4)

    def test_gaussHermite(self):
        cdf = functions.integrate_gaussHermite(
            self.edges, self.lsf_at_edges, self.center, 80.0, 0.1, 0.05
        )
        assert jnp.isclose(cdf[-1] - cdf[0], 1.0, rtol=1e-5)

    def test_split_normal(self):
        cdf = functions.integrate_split_normal(
            self.edges, self.lsf_at_edges, self.center, 100.0, 60.0
        )
        assert jnp.isclose(cdf[-1] - cdf[0], 1.0, rtol=1e-5)

    def test_boxGauss(self):
        cdf = functions.integrate_boxGauss(
            self.edges, self.lsf_at_edges, self.center, 300.0, 100.0
        )
        assert jnp.isclose(cdf[-1] - cdf[0], 1.0, rtol=1e-5)

    def test_skewNormal(self):
        cdf = functions.integrate_skewNormal(
            self.edges, self.lsf_at_edges, self.center, 80.0, 2.0
        )
        assert jnp.isclose(cdf[-1] - cdf[0], 1.0, rtol=1e-5)

    def test_boxGauss_wide_box(self):
        cdf = functions.integrate_boxGauss(
            self.edges, self.lsf_at_edges, self.center, 500.0, 0.0
        )
        assert jnp.isclose(cdf[-1] - cdf[0], 1.0, rtol=1e-5)

    def test_boxGauss_large_gauss(self):
        cdf = functions.integrate_boxGauss(
            self.edges, self.lsf_at_edges, self.center, 0.1, 500.0
        )
        assert jnp.isclose(cdf[-1] - cdf[0], 1.0, rtol=1e-5)


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
        edges = jnp.linspace(4500.0, 5500.0, 500)
        lsf = jnp.full_like(edges, 5.0)
        cdf = profile.integrate(edges, lsf, center=5000.0, **extra_kwargs)
        per_pixel = _per_pixel(cdf)
        # Cauchy has heavy tails so allow wider tolerance.
        assert float(jnp.sum(per_pixel)) == pytest.approx(1.0, rel=0.1)

    @pytest.mark.parametrize('profile,extra_kwargs', _PROFILE_PARAMS)
    def test_integrate_returns_nonneg(self, profile, extra_kwargs):
        edges = jnp.linspace(4900.0, 5100.0, 50)
        lsf = jnp.full_like(edges, 5.0)
        cdf = profile.integrate(edges, lsf, center=5000.0, **extra_kwargs)
        per_pixel = _per_pixel(cdf)
        assert jnp.all(per_pixel >= 0.0)


# ---------------------------------------------------------------------------
# integrate_skewNormal
# ---------------------------------------------------------------------------


class TestIntegrateSkewNormal:
    """Tests for the public integrate_skewNormal kernel."""

    center = 5000.0
    edges = jnp.linspace(4000.0, 6000.0, 2001)
    lsf_at_edges = jnp.full_like(edges, 5.0)

    def test_alpha_zero_matches_gaussian(self):
        """integrate_skewNormal with alpha=0 is identical to integrate_gaussian."""
        skew = functions.integrate_skewNormal(
            self.edges, self.lsf_at_edges, self.center, 80.0, 0.0
        )
        gauss = functions.integrate_gaussian(
            self.edges, self.lsf_at_edges, self.center, 80.0
        )
        assert jnp.allclose(skew, gauss, rtol=1e-6)

    def test_normalization_positive_alpha(self):
        """integrate_skewNormal sums to 1 for positive alpha."""
        cdf = functions.integrate_skewNormal(
            self.edges, self.lsf_at_edges, self.center, 80.0, 3.0
        )
        assert float(cdf[-1] - cdf[0]) == pytest.approx(1.0, rel=1e-5)

    def test_normalization_negative_alpha(self):
        """integrate_skewNormal sums to 1 for negative alpha."""
        cdf = functions.integrate_skewNormal(
            self.edges, self.lsf_at_edges, self.center, 80.0, -3.0
        )
        assert float(cdf[-1] - cdf[0]) == pytest.approx(1.0, rel=1e-5)

    def test_skew_shifts_mass(self):
        """Positive alpha shifts mass to the red side of center."""
        cdf_pos = functions.integrate_skewNormal(
            self.edges, self.lsf_at_edges, self.center, 80.0, 5.0
        )
        cdf_neg = functions.integrate_skewNormal(
            self.edges, self.lsf_at_edges, self.center, 80.0, -5.0
        )
        pp_pos = _per_pixel(cdf_pos)
        pp_neg = _per_pixel(cdf_neg)
        red_mask = self.edges[1:] > self.center
        assert float(jnp.sum(pp_pos[red_mask])) > float(jnp.sum(pp_neg[red_mask]))

    def test_alpha_antisymmetry(self):
        """Per-pixel result for +alpha is the mirror of -alpha about center."""
        cdf_pos = functions.integrate_skewNormal(
            self.edges, self.lsf_at_edges, self.center, 80.0, 3.0
        )
        # Reflect edges about center so the per-pixel array gets reversed.
        edges_mirror = 2 * self.center - self.edges[::-1]
        lsf_mirror = self.lsf_at_edges[::-1]
        cdf_neg = functions.integrate_skewNormal(
            edges_mirror, lsf_mirror, self.center, 80.0, -3.0
        )
        pp_pos = _per_pixel(cdf_pos)
        pp_neg = _per_pixel(cdf_neg)
        assert jnp.allclose(pp_pos, pp_neg[::-1], rtol=1e-6)
