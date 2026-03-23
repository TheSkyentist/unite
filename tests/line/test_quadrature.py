"""Tests for Gauss-Legendre quadrature integration of line profiles.

These tests verify that GL quadrature of ``evaluate_lines`` converges to
``analytic_integrate_lines`` and that profiles integrate to unity over wide bins.
"""

import jax
import jax.numpy as jnp
import pytest

from unite.line.compute import analytic_integrate_lines, evaluate_lines
from unite.line.profiles import (
    SEMG,
    Cauchy,
    GaussHermite,
    Gaussian,
    Laplace,
    PseudoVoigt,
    SplitNormal,
)


def _gl_nodes_weights(n):
    """Compute Gauss-Legendre nodes and weights as JAX arrays."""
    import numpy as np

    nodes, weights = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(nodes), jnp.asarray(weights)


def _quadrature_integrate(
    low, high, centers, lsf, p0s, p1s, p2s, codes, nodes, weights
):
    """Integrate line profiles over pixels using GL quadrature of evaluate_lines."""
    mid = (low + high) / 2.0
    half_width = (high - low) / 2.0
    # Sub-pixel points: (n_nodes, n_pix)
    x = mid[None, :] + half_width[None, :] * nodes[:, None]

    # Evaluate all profiles at each set of node points: (n_nodes, n_lines, n_pix)
    phi_at_nodes = jax.vmap(
        lambda wav: evaluate_lines(wav, centers, lsf, p0s, p1s, p2s, codes)
    )(x)

    # GL weighted sum: (n_lines, n_pix)
    return half_width[None, :] * jnp.einsum('n,nlp->lp', weights, phi_at_nodes)


# ---------------------------------------------------------------------------
# Quadrature normalization: all profiles should sum to ~1 over wide bins
# ---------------------------------------------------------------------------

_ALL_PROFILES = [
    (Gaussian(), {'fwhm_gauss': 100.0}),
    (Cauchy(), {'fwhm_lorentz': 100.0}),
    (PseudoVoigt(), {'fwhm_gauss': 80.0, 'fwhm_lorentz': 50.0}),
    (Laplace(), {'fwhm_exp': 100.0}),
    (SEMG(), {'fwhm_gauss': 80.0, 'fwhm_exp': 50.0}),
    (GaussHermite(), {'fwhm_gauss': 80.0, 'h3': 0.1, 'h4': 0.05}),
    (SplitNormal(), {'fwhm_blue': 100.0, 'fwhm_red': 60.0}),
]


class TestQuadratureNormalization:
    """Quadrature-integrated profiles should sum to ~1 over wide bins."""

    low = jnp.linspace(0, 10000, 1000)[:-1]
    high = jnp.linspace(0, 10000, 1000)[1:]
    center = 5000.0
    lsf_fwhm = 10.0
    nodes, weights = _gl_nodes_weights(7)

    @pytest.mark.parametrize('profile,extra_kwargs', _ALL_PROFILES)
    def test_sums_to_unity(self, profile, extra_kwargs):
        pnames = profile.param_names()
        p0 = extra_kwargs.get(pnames[0], 0.0) if len(pnames) > 0 else 0.0
        p1 = extra_kwargs.get(pnames[1], 0.0) if len(pnames) > 1 else 0.0
        p2 = extra_kwargs.get(pnames[2], 0.0) if len(pnames) > 2 else 0.0

        # Shape arrays for vmap: (1,) for single line
        centers = jnp.array([self.center])
        lsf = jnp.array([self.lsf_fwhm])
        p0s = jnp.array([p0])
        p1s = jnp.array([p1])
        p2s = jnp.array([p2])
        codes = jnp.array([profile.code])

        result = _quadrature_integrate(
            self.low,
            self.high,
            centers,
            lsf,
            p0s,
            p1s,
            p2s,
            codes,
            self.nodes,
            self.weights,
        )
        total = float(jnp.sum(result[0]))
        # Cauchy has heavy tails
        tol = 0.1 if isinstance(profile, Cauchy) else 0.01
        assert total == pytest.approx(1.0, rel=tol)


# ---------------------------------------------------------------------------
# Convergence: quadrature should match analytic integration
# ---------------------------------------------------------------------------


class TestQuadratureMatchesAnalytic:
    """Quadrature integration should converge to analytic integration
    as n_nodes increases."""

    center = 5000.0
    lsf_fwhm = 10.0

    @pytest.mark.parametrize('profile,extra_kwargs', _ALL_PROFILES)
    def test_convergence(self, profile, extra_kwargs):
        # Use reasonably wide pixels to make the test meaningful
        edges = jnp.linspace(4800.0, 5200.0, 200)
        lo, hi = edges[:-1], edges[1:]

        pnames = profile.param_names()
        p0 = extra_kwargs.get(pnames[0], 0.0) if len(pnames) > 0 else 0.0
        p1 = extra_kwargs.get(pnames[1], 0.0) if len(pnames) > 1 else 0.0
        p2 = extra_kwargs.get(pnames[2], 0.0) if len(pnames) > 2 else 0.0

        centers = jnp.array([self.center])
        lsf = jnp.array([self.lsf_fwhm])
        p0s = jnp.array([p0])
        p1s = jnp.array([p1])
        p2s = jnp.array([p2])
        codes = jnp.array([profile.code])

        analytic = analytic_integrate_lines(lo, hi, centers, lsf, p0s, p1s, p2s, codes)[
            0
        ]

        # Test convergence: more nodes → smaller error
        prev_err = float('inf')
        for n_nodes in [3, 5, 7, 11]:
            nodes, weights = _gl_nodes_weights(n_nodes)
            quad = _quadrature_integrate(
                lo, hi, centers, lsf, p0s, p1s, p2s, codes, nodes, weights
            )[0]

            # Compare near peak where values are significant
            mask = analytic > 1e-8
            if jnp.any(mask):
                err = float(
                    jnp.max(jnp.abs(analytic[mask] - quad[mask]) / analytic[mask])
                )
                assert err < prev_err or err < 1e-6, (
                    f'n_nodes={n_nodes}: error {err:.2e} not improving'
                )
                prev_err = err

        # With 11 nodes, should be very close to analytic
        assert prev_err < 1e-4, f'11-node quadrature error {prev_err:.2e} too large'


# ---------------------------------------------------------------------------
# Multi-line dispatch: quadrature handles mixed profile codes
# ---------------------------------------------------------------------------


class TestQuadratureMultiLine:
    """Quadrature integration should handle multiple lines with different
    profile codes dispatched via lax.switch."""

    def test_mixed_profiles(self):
        edges = jnp.linspace(4800.0, 5200.0, 100)
        lo, hi = edges[:-1], edges[1:]

        # Two lines: Gaussian (code 0) and PseudoVoigt (code 2)
        centers = jnp.array([5000.0, 5050.0])
        lsf = jnp.array([10.0, 10.0])
        p0 = jnp.array([100.0, 80.0])
        p1 = jnp.array([0.0, 20.0])
        p2 = jnp.zeros(2)
        codes = jnp.array([0, 2])

        nodes, weights = _gl_nodes_weights(7)
        result = _quadrature_integrate(
            lo, hi, centers, lsf, p0, p1, p2, codes, nodes, weights
        )
        assert result.shape == (2, 99)
        # Both should sum to ~1
        assert float(jnp.sum(result[0])) == pytest.approx(1.0, rel=0.01)
        assert float(jnp.sum(result[1])) == pytest.approx(1.0, rel=0.05)
        # Peaks at different locations
        assert jnp.argmax(result[0]) != jnp.argmax(result[1])
