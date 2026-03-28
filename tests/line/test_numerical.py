"""Numerical validation: compare analytic integrals against scipy.integrate.quad."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from scipy import integrate, special

from unite.line import functions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numerical_integral(pdf_fn, center, lsf_fwhm, low, high, *params):
    """Integrate pdf_fn over each bin [low_i, high_i] using scipy.quad."""
    return jnp.array([
        integrate.quad(pdf_fn, float(lo), float(hi), args=(float(center), float(lsf_fwhm), *map(float, params)))[0]
        for lo, hi in zip(low, high)
    ])


# ---------------------------------------------------------------------------
# Reference PDFs (using scipy for the ground truth)
# ---------------------------------------------------------------------------

def _skew_gaussian_pdf(x, center, lsf_fwhm, fwhm, alpha):
    """Skew-normal PDF with Gaussian LSF convolved in (sigma_tot = combined sigma)."""
    import math
    sigma_tot = math.sqrt((fwhm**2 + lsf_fwhm**2)) / (2 * math.sqrt(2 * math.log(2)))
    # Effective alpha after LSF convolution
    alpha_eff = alpha * sigma_tot / math.sqrt(sigma_tot**2 + alpha**2 * (lsf_fwhm / (2 * math.sqrt(2 * math.log(2))))**2)
    dx = x - center
    gauss = math.exp(-0.5 * (dx / sigma_tot)**2) / (math.sqrt(2 * math.pi) * sigma_tot)
    skew = 1.0 + special.erf(alpha_eff * dx / sigma_tot)
    return gauss * skew


def _skew_voigt_pdf(x, center, lsf_fwhm, fwhm_g, fwhm_l, alpha):
    """Skew pseudo-Voigt PDF: evaluate_skewVoigt at a single point."""
    return float(functions.evaluate_skewVoigt(
        jnp.array([float(x)]),
        jnp.array([float(center)]),
        jnp.array([float(lsf_fwhm)]),
        jnp.array([float(fwhm_g)]),
        jnp.array([float(fwhm_l)]),
        jnp.array([float(alpha)]),
    )[0])


# ---------------------------------------------------------------------------
# Owen's T accuracy
# ---------------------------------------------------------------------------

class TestOwensT:
    """Verify _owens_t against scipy.stats.norm-based reference."""

    @pytest.mark.parametrize('h,a', [
        (0.0, 1.0),
        (0.0, 5.0),
        (1.0, 0.5),
        (1.0, 2.0),
        (1.0, 100.0),   # large a — problematic for naive GL on [0, a]
        (3.0, 0.3),
        (3.0, 50.0),
        (-1.5, 2.0),    # negative h (T is even in h)
        (1.0, -2.0),    # negative a
    ])
    def test_owens_t_vs_reference(self, h, a):
        """Compare _owens_t to scipy.integrate.quad of the defining integral."""
        if a >= 0:
            ref, _ = integrate.quad(
                lambda x: jnp.exp(-0.5 * h**2 * (1.0 + x**2)) / (1.0 + x**2),
                0.0, abs(a),
            )
            ref /= 2.0 * jnp.pi
        else:
            ref, _ = integrate.quad(
                lambda x: jnp.exp(-0.5 * h**2 * (1.0 + x**2)) / (1.0 + x**2),
                0.0, abs(a),
            )
            ref = -ref / (2.0 * jnp.pi)

        result = float(functions._owens_t(jnp.array(h), jnp.array(a)))
        assert result == pytest.approx(ref, rel=1e-8, abs=1e-14)


# ---------------------------------------------------------------------------
# SkewGaussian: analytic vs numerical integral
# ---------------------------------------------------------------------------

_CENTER = 5000.0
_LSF_FWHM = 10.0
_BINS_FINE = jnp.linspace(4900.0, 5100.0, 201)
_LOW = _BINS_FINE[:-1]
_HIGH = _BINS_FINE[1:]


class TestSkewGaussianNumerical:
    """Compare integrate_skewGaussian to scipy numerical integrals."""

    @pytest.mark.parametrize('fwhm,alpha', [
        (80.0,   0.0),    # symmetric (should match Gaussian)
        (80.0,   2.0),    # mild positive skew
        (80.0,  -2.0),    # mild negative skew
        (80.0,  10.0),    # strong skew
        (80.0, 100.0),    # very large alpha
        (40.0,   5.0),    # narrow line, moderate skew
    ])
    def test_matches_numerical(self, fwhm, alpha):
        analytic = functions.integrate_skewGaussian(
            _LOW, _HIGH,
            jnp.array([_CENTER]),
            jnp.array([_LSF_FWHM]),
            jnp.array([fwhm]),
            jnp.array([alpha]),
        )
        numerical = _numerical_integral(
            _skew_gaussian_pdf, _CENTER, _LSF_FWHM, _LOW, _HIGH, fwhm, alpha,
        )
        # Allow 0.5% relative tolerance per bin (profiles are smooth)
        assert jnp.allclose(analytic, numerical, rtol=5e-3, atol=1e-6), (
            f'Max deviation: {float(jnp.max(jnp.abs(analytic - numerical))):.2e}'
        )

    @pytest.mark.parametrize('fwhm,alpha', [
        (80.0, 2.0),
        (80.0, -5.0),
        (40.0, 10.0),
    ])
    def test_normalizes_to_unity(self, fwhm, alpha):
        """Integral over a wide range should be ~1."""
        low = jnp.linspace(4000.0, 6000.0, 2001)[:-1]
        high = jnp.linspace(4000.0, 6000.0, 2001)[1:]
        result = functions.integrate_skewGaussian(
            low, high,
            jnp.array([_CENTER]),
            jnp.array([_LSF_FWHM]),
            jnp.array([fwhm]),
            jnp.array([alpha]),
        )
        assert float(jnp.sum(result)) == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# SkewVoigt: analytic vs numerical integral
# ---------------------------------------------------------------------------


class TestSkewVoigtNumerical:
    """Compare integrate_skewVoigt to scipy numerical integrals."""

    @pytest.mark.parametrize('fwhm_g,fwhm_l,alpha', [
        (80.0, 0.0,  0.0),   # pure Gaussian, no skew
        (80.0, 50.0, 0.0),   # Voigt, no skew
        (80.0, 50.0, 2.0),   # Voigt with skew
        (80.0, 50.0, -2.0),  # Voigt with negative skew
        (80.0, 30.0, 5.0),   # stronger skew
    ])
    def test_matches_numerical(self, fwhm_g, fwhm_l, alpha):
        analytic = functions.integrate_skewVoigt(
            _LOW, _HIGH,
            jnp.array([_CENTER]),
            jnp.array([_LSF_FWHM]),
            jnp.array([fwhm_g]),
            jnp.array([fwhm_l]),
            jnp.array([alpha]),
        )
        numerical = _numerical_integral(
            _skew_voigt_pdf, _CENTER, _LSF_FWHM, _LOW, _HIGH, fwhm_g, fwhm_l, alpha,
        )
        # SkewVoigt uses a midpoint-skew approximation, so allow wider tolerance
        assert jnp.allclose(analytic, numerical, rtol=0.05, atol=1e-5), (
            f'Max deviation: {float(jnp.max(jnp.abs(analytic - numerical))):.2e}'
        )

    @pytest.mark.parametrize('fwhm_g,fwhm_l,alpha', [
        (80.0, 50.0, 2.0),
        (80.0, 50.0, -3.0),
    ])
    def test_normalizes_to_unity(self, fwhm_g, fwhm_l, alpha):
        """Integral over a wide range should be ~1 (SkewVoigt is not analytic so wider tolerance)."""
        low = jnp.linspace(4000.0, 6000.0, 2001)[:-1]
        high = jnp.linspace(4000.0, 6000.0, 2001)[1:]
        result = functions.integrate_skewVoigt(
            low, high,
            jnp.array([_CENTER]),
            jnp.array([_LSF_FWHM]),
            jnp.array([fwhm_g]),
            jnp.array([fwhm_l]),
            jnp.array([alpha]),
        )
        assert float(jnp.sum(result)) == pytest.approx(1.0, rel=0.1)
