"""Tests for Gauss-Legendre quadrature integration of line profiles."""

import jax.numpy as jnp
import pytest

from unite.line.profiles import (
    SEMG,
    Cauchy,
    GaussHermite,
    Gaussian,
    GaussianAbsorption,
    Laplace,
    LorentzianAbsorption,
    PseudoVoigt,
    SplitNormal,
    VoigtAbsorption,
    analytic_integrate_lines,
    quadrature_integrate_lines,
)


def _gl_nodes_weights(n):
    """Compute Gauss-Legendre nodes and weights as JAX arrays."""
    import numpy as np

    nodes, weights = np.polynomial.legendre.leggauss(n)
    return jnp.asarray(nodes), jnp.asarray(weights)


# ---------------------------------------------------------------------------
# Quadrature normalization: all profiles should sum to ~1 over wide bins
# ---------------------------------------------------------------------------

_EMISSION_PROFILES = [
    (Gaussian(), {'fwhm_gauss': 100.0}),
    (Cauchy(), {'fwhm_lorentz': 100.0}),
    (PseudoVoigt(), {'fwhm_gauss': 80.0, 'fwhm_lorentz': 50.0}),
    (Laplace(), {'fwhm_exp': 100.0}),
    (SEMG(), {'fwhm_gauss': 80.0, 'fwhm_exp': 50.0}),
    (GaussHermite(), {'fwhm_gauss': 80.0, 'h3': 0.1, 'h4': 0.05}),
    (SplitNormal(), {'fwhm_blue': 100.0, 'fwhm_red': 60.0}),
]

_ABSORPTION_PROFILES = [
    (GaussianAbsorption(), {'fwhm_gauss': 100.0}),
    (VoigtAbsorption(), {'fwhm_gauss': 80.0, 'fwhm_lorentz': 50.0}),
    (LorentzianAbsorption(), {'fwhm_lorentz': 100.0}),
]

_ALL_PROFILES = _EMISSION_PROFILES + _ABSORPTION_PROFILES


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

        result = quadrature_integrate_lines(
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
        # Cauchy/Lorentzian have heavy tails
        tol = 0.1 if isinstance(profile, (Cauchy, LorentzianAbsorption)) else 0.01
        assert total == pytest.approx(1.0, rel=tol)


# ---------------------------------------------------------------------------
# Convergence: quadrature should match analytic integration for emission
# ---------------------------------------------------------------------------


class TestQuadratureMatchesAnalytic:
    """For emission profiles, quadrature integration should converge to
    analytic integration as n_nodes increases."""

    center = 5000.0
    lsf_fwhm = 10.0

    @pytest.mark.parametrize('profile,extra_kwargs', _EMISSION_PROFILES)
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
            quad = quadrature_integrate_lines(
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
# Absorption profiles: quadrature is more accurate than pixel-center
# ---------------------------------------------------------------------------


class TestAbsorptionQuadratureImprovement:
    """Quadrature should be more accurate than the pixel-center approximation
    used by analytic mode for absorption profiles."""

    center = 5000.0
    lsf_fwhm = 10.0

    def test_gaussian_absorption_vs_reference(self):
        """For wide pixels, quadrature should match the analytic Gaussian
        integral better than the pixel-center approximation does."""
        # Use wide pixels where pixel-center approximation is less accurate
        edges = jnp.linspace(4950.0, 5050.0, 20)
        lo, hi = edges[:-1], edges[1:]

        centers = jnp.array([self.center])
        lsf = jnp.array([self.lsf_fwhm])
        p0s = jnp.array([100.0])
        p1s = jnp.zeros(1)
        p2s = jnp.zeros(1)

        # Reference: use the emission Gaussian (code 0) analytic integration
        # which is exact via erf.
        ref_codes = jnp.array([0])  # Gaussian emission (exact)
        reference = analytic_integrate_lines(
            lo, hi, centers, lsf, p0s, p1s, p2s, ref_codes
        )[0]

        # Absorption pixel-center approximation (code 7)
        abs_codes = jnp.array([7])  # GaussianAbsorption
        approx = analytic_integrate_lines(
            lo, hi, centers, lsf, p0s, p1s, p2s, abs_codes
        )[0]

        # Quadrature on the absorption profile
        nodes, weights = _gl_nodes_weights(7)
        quad = quadrature_integrate_lines(
            lo, hi, centers, lsf, p0s, p1s, p2s, abs_codes, nodes, weights
        )[0]

        # Both should be close to reference, but quadrature should be closer
        mask = reference > 1e-8
        err_approx = float(
            jnp.max(jnp.abs(reference[mask] - approx[mask]) / reference[mask])
        )
        err_quad = float(
            jnp.max(jnp.abs(reference[mask] - quad[mask]) / reference[mask])
        )

        assert err_quad < err_approx, (
            f'Quadrature error ({err_quad:.2e}) should be smaller than '
            f'pixel-center error ({err_approx:.2e})'
        )


# ---------------------------------------------------------------------------
# Multi-line dispatch: quadrature handles mixed profile codes
# ---------------------------------------------------------------------------


class TestQuadratureMultiLine:
    """Quadrature integration should handle multiple lines with different
    profile codes dispatched via lax.switch."""

    def test_mixed_profiles(self):
        edges = jnp.linspace(4800.0, 5200.0, 100)
        lo, hi = edges[:-1], edges[1:]

        # Two lines: Gaussian (code 0) and GaussianAbsorption (code 7)
        centers = jnp.array([5000.0, 5050.0])
        lsf = jnp.array([10.0, 10.0])
        p0 = jnp.array([100.0, 80.0])
        p1 = jnp.zeros(2)
        p2 = jnp.zeros(2)
        codes = jnp.array([0, 7])

        nodes, weights = _gl_nodes_weights(7)
        result = quadrature_integrate_lines(
            lo, hi, centers, lsf, p0, p1, p2, codes, nodes, weights
        )
        assert result.shape == (2, 99)
        # Both should sum to ~1
        assert float(jnp.sum(result[0])) == pytest.approx(1.0, rel=0.01)
        assert float(jnp.sum(result[1])) == pytest.approx(1.0, rel=0.01)
        # Peaks at different locations
        assert jnp.argmax(result[0]) != jnp.argmax(result[1])
