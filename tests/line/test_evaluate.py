"""Tests for pointwise evaluate functions and absorption profiles."""

from types import MappingProxyType

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.signal import fftconvolve

from unite.line import functions
from unite.line.compute import evaluate_lines
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
# Evaluate function normalization
# ---------------------------------------------------------------------------


class TestEvaluateNormalization:
    """Evaluate functions should integrate to ~1 over a wide wavelength range."""

    center = 5000.0
    lsf_fwhm = 5.0
    wavelength = jnp.linspace(4000.0, 6000.0, 10000)

    def test_gaussian(self):
        vals = functions.evaluate_gaussian(
            self.wavelength, self.center, self.lsf_fwhm, 100.0
        )
        integral = jnp.trapezoid(vals, self.wavelength)
        assert float(integral) == pytest.approx(1.0, rel=1e-3)

    def test_voigt(self):
        vals = functions.evaluate_voigt(
            self.wavelength, self.center, self.lsf_fwhm, 80.0, 50.0
        )
        integral = jnp.trapezoid(vals, self.wavelength)
        assert float(integral) == pytest.approx(1.0, rel=0.05)

    def test_cauchy_via_voigt(self):
        vals = functions.evaluate_voigt(
            self.wavelength, self.center, self.lsf_fwhm, 0.0, 100.0
        )
        integral = jnp.trapezoid(vals, self.wavelength)
        # Cauchy has heavy tails so allow wider tolerance
        assert float(integral) == pytest.approx(1.0, rel=0.15)

    def test_gaussian_limit(self):
        # fwhm_l -> 0: Voigt should reduce to Gaussian
        vals_voigt = functions.evaluate_voigt(
            self.wavelength, self.center, self.lsf_fwhm, 100.0, 1e-6
        )
        vals_gaussian = functions.evaluate_gaussian(
            self.wavelength, self.center, self.lsf_fwhm, 100.0
        )
        assert jnp.allclose(vals_voigt, vals_gaussian, rtol=1e-4)

    def test_gaussianLaplace(self):
        vals = functions.evaluate_gaussianLaplace(
            self.wavelength, self.center, self.lsf_fwhm, 80.0, 50.0
        )
        integral = jnp.trapezoid(vals, self.wavelength)
        assert float(integral) == pytest.approx(1.0, rel=0.05)

    def test_laplace_via_gaussianLaplace(self):
        vals = functions.evaluate_gaussianLaplace(
            self.wavelength, self.center, self.lsf_fwhm, 0.0, 100.0
        )
        integral = jnp.trapezoid(vals, self.wavelength)
        assert float(integral) == pytest.approx(1.0, rel=0.05)

    def test_split_normal(self):
        vals = functions.evaluate_split_normal(
            self.wavelength, self.center, self.lsf_fwhm, 100.0, 60.0
        )
        integral = jnp.trapezoid(vals, self.wavelength)
        assert float(integral) == pytest.approx(1.0, rel=1e-3)

    def test_gaussHermite(self):
        vals = functions.evaluate_gaussHermite(
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
        self, integrate_fn, evaluate_fn, args_integrate, args_evaluate, tolerances=None
    ):
        """Verify convergence as pixel width shrinks."""
        if tolerances is None:
            tolerances = [(100, 0.1), (500, 0.01), (2000, 1e-3)]
        for n_pix, tol in tolerances:
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
            functions.integrate_gaussian,
            functions.evaluate_gaussian,
            (self.center, self.lsf_fwhm, 100.0),
            (self.center, self.lsf_fwhm, 100.0),
        )

    def test_voigt_ida_vs_faddeeva(self):
        # functions.integrate_voigt uses the Ida extended pseudo-Voigt CDF (< 0.12% peak error).
        # functions.evaluate_voigt is the exact Faddeeva Voigt.  At fine pixels the Riemann error
        # is negligible, so the residual reflects the Ida approximation error: < 1%.
        self._check_convergence(
            functions.integrate_voigt,
            functions.evaluate_voigt,
            (self.center, self.lsf_fwhm, 80.0, 50.0),
            (self.center, self.lsf_fwhm, 80.0, 50.0),
            tolerances=[(100, 0.1), (500, 0.02), (2000, 0.01)],
        )

    def test_gaussianLaplace(self):
        self._check_convergence(
            functions.integrate_gaussianLaplace,
            functions.evaluate_gaussianLaplace,
            (self.center, self.lsf_fwhm, 80.0, 50.0),
            (self.center, self.lsf_fwhm, 80.0, 50.0),
        )

    def test_split_normal(self):
        self._check_convergence(
            functions.integrate_split_normal,
            functions.evaluate_split_normal,
            (self.center, self.lsf_fwhm, 100.0, 60.0),
            (self.center, self.lsf_fwhm, 100.0, 60.0),
        )

    def test_gaussHermite(self):
        self._check_convergence(
            functions.integrate_gaussHermite,
            functions.evaluate_gaussHermite,
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
        left = functions.evaluate_gaussian(
            self.center - self.offsets, self.center, self.lsf_fwhm, 100.0
        )
        right = functions.evaluate_gaussian(
            self.center + self.offsets, self.center, self.lsf_fwhm, 100.0
        )
        assert jnp.allclose(left, right, rtol=1e-10)

    def test_voigt_symmetric(self):
        left = functions.evaluate_voigt(
            self.center - self.offsets, self.center, self.lsf_fwhm, 80.0, 50.0
        )
        right = functions.evaluate_voigt(
            self.center + self.offsets, self.center, self.lsf_fwhm, 80.0, 50.0
        )
        assert jnp.allclose(left, right, rtol=1e-10)


# ---------------------------------------------------------------------------
# Voigt implementation agreement (uses private helpers for comparison)
# ---------------------------------------------------------------------------


class TestVoigtAgreement:
    """
    Validates Thompson, Ida, and Faddeeva approximations against each other
    and against a numerically convolved baseline on a super-fine grid.
    """

    center = 5000.0
    lsf_fwhm = 5.0
    # Standard evaluation grid
    wavelength = jnp.linspace(4000.0, 6000.0, 4000)

    # Parameters for the three test regimes: (fwhm_g, fwhm_l)
    regimes = MappingProxyType(
        {
            'gaussian_dominated': (100.0, 10.0),
            'mixed': (80.0, 50.0),
            'lorentzian_dominated': (20.0, 100.0),
        }
    )

    def _compute_baseline_voigt(self, fwhm_g_total, fwhm_l):
        """Generates a ground-truth Voigt via fine-grid FFT convolution."""
        # 1. Create a fine grid centered at 0 (0.1 Å step is >10 pts/FWHM for all regimes)
        fine_dx = 0.1
        half_width = 600.0  # Wide enough to capture wings
        x_fine = jnp.arange(-half_width, half_width, fine_dx)

        # 2. Define kernels
        sigma = fwhm_g_total / (2 * jnp.sqrt(2 * jnp.log(2)))
        gamma = fwhm_l / 2.0

        gauss = (1.0 / (sigma * jnp.sqrt(2 * jnp.pi))) * jnp.exp(
            -0.5 * (x_fine / sigma) ** 2
        )
        lorentz = (gamma / jnp.pi) / (x_fine**2 + gamma**2)

        # 3. FFT-based convolution — O(N log N) instead of O(N²)
        # We multiply by fine_dx to keep the integral normalized to ~1.0
        baseline_pdf_fine = fftconvolve(gauss, lorentz, mode='same') * fine_dx

        # 4. Interpolate back to the evaluation grid
        return jnp.interp(self.wavelength - self.center, x_fine, baseline_pdf_fine)

    def _get_pdfs(self, fwhm_g, fwhm_l):
        """Helper to compute all three PDF approximations."""
        fwhm_g_total = functions._combine_fwhm(self.lsf_fwhm, fwhm_g)
        dx = self.wavelength - self.center

        t = functions._voigt_thompson_pdf(dx, fwhm_g_total, fwhm_l)
        f = functions._voigt_faddeeva_pdf(dx, fwhm_g_total, fwhm_l)

        # Ida is typically a CDF diff, but we can approximate its PDF
        # on a fine grid for L1 comparison.
        # Alternatively, we compare Ida primarily in the CDF test.
        return t, f, fwhm_g_total

    @pytest.mark.parametrize(
        'regime', ['gaussian_dominated', 'mixed', 'lorentzian_dominated']
    )
    def test_pdf_agreement_to_baseline(self, regime):
        fg, fl = self.regimes[regime]
        t_pdf, f_pdf, fg_total = self._get_pdfs(fg, fl)
        baseline_pdf = self._compute_baseline_voigt(fg_total, fl)

        # L1 Rel Error vs Baseline
        # Faddeeva is usually the most accurate (>0.1% error)
        # Thompson is an approximation (~1-2% error)
        err_faddeeva = jnp.trapezoid(jnp.abs(f_pdf - baseline_pdf), self.wavelength)
        err_thompson = jnp.trapezoid(jnp.abs(t_pdf - baseline_pdf), self.wavelength)

        assert err_faddeeva < 0.01  # Faddeeva should be very close to baseline
        assert err_thompson < 0.05  # Thompson has wider tolerance

    @pytest.mark.parametrize(
        'regime', ['gaussian_dominated', 'mixed', 'lorentzian_dominated']
    )
    def test_inter_approximation_agreement(self, regime):
        """Test Thompson vs Faddeeva directly."""
        fg, fl = self.regimes[regime]
        t_pdf, f_pdf, _ = self._get_pdfs(fg, fl)

        rel_l1 = jnp.trapezoid(jnp.abs(t_pdf - f_pdf), self.wavelength) / jnp.trapezoid(
            f_pdf, self.wavelength
        )

        assert rel_l1 < 0.05

    def test_ida_vs_thompson_cdf(self):
        """
        Specific test for CDF differences (integrated flux per pixel).
        Ida (rational approx) vs Thompson (integrated PDF).
        """
        # Define pixel edges
        edges = jnp.linspace(4900.0, 5100.0, 100)  # Focus on the core
        lo, hi = edges[:-1], edges[1:]
        mid = (lo + hi) / 2.0
        dlam = hi - lo

        fg, fl = 80.0, 50.0
        fg_total = functions._combine_fwhm(self.lsf_fwhm, fg)

        # 1. Ida CDF Diff
        ida_flux = functions._voigt_ida_cdf_diff(
            lo - self.center, hi - self.center, fg_total, fl
        )

        # 2. Thompson Integrated (PDF * dx)
        thompson_flux = (
            functions._voigt_thompson_pdf(mid - self.center, fg_total, fl) * dlam
        )

        # Compare integrated flux L1
        l1_err = float(jnp.sum(jnp.abs(ida_flux - thompson_flux)))
        # Total area is ~1.0, so sum of abs diff is roughly the L1 error
        assert l1_err < 0.05


# ---------------------------------------------------------------------------
# Voigt limits: pure Gaussian and pure Cauchy
# ---------------------------------------------------------------------------


class TestIntegrateVoigtLimits:
    """functions.integrate_voigt should reduce exactly to Gaussian/Cauchy at the limits."""

    center = 5000.0
    lsf_fwhm = 5.0
    edges = jnp.linspace(4900.0, 5100.0, 501)
    lo, hi = edges[:-1], edges[1:]

    def test_gaussian_limit(self):
        # fwhm_l -> 0: rho -> 0, Ida weights collapse to pure Gaussian
        voigt = functions.integrate_voigt(
            self.lo, self.hi, self.center, self.lsf_fwhm, 100.0, 1e-6
        )
        gauss = functions.integrate_gaussian(
            self.lo, self.hi, self.center, self.lsf_fwhm, 100.0
        )
        assert jnp.allclose(voigt, gauss, rtol=1e-4)

    def test_cauchy_limit(self):
        # fwhm_g -> 0 (only LSF), fwhm_l large: both functions.integrate_voigt and functions.evaluate_voigt
        # should agree closely when pixel-integrated.
        voigt = functions.integrate_voigt(
            self.lo, self.hi, self.center, self.lsf_fwhm, 1e-6, 100.0
        )
        faddeeva_sum = float(
            jnp.sum(
                functions.evaluate_voigt(
                    (self.lo + self.hi) / 2, self.center, self.lsf_fwhm, 1e-6, 100.0
                )
                * (self.hi - self.lo)
            )
        )
        assert float(jnp.sum(voigt)) == pytest.approx(faddeeva_sum, rel=0.01)


# ---------------------------------------------------------------------------
# Voigt vs numerical convolution
# ---------------------------------------------------------------------------


class TestVoigtNumericalConvolution:
    """functions.evaluate_voigt should match a direct numerical G⊗L convolution.

    This is the physical ground truth: the Voigt profile is by definition the
    convolution of a Gaussian with a Lorentzian.  We pass lsf_fwhm=0 so the
    Gaussian component is purely the intrinsic fwhm_g.
    """

    fwhm_g = 80.0
    fwhm_l = 50.0

    def test_faddeeva_matches_convolution(self):
        # Fine symmetric grid — wide enough for tails, fine enough for accuracy
        x = np.linspace(-600.0, 600.0, 24001)
        dx = float(x[1] - x[0])

        sigma_g = self.fwhm_g / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        gamma_l = self.fwhm_l / 2.0
        gauss = np.exp(-0.5 * x**2 / sigma_g**2) / (sigma_g * np.sqrt(2.0 * np.pi))
        lorentz = (gamma_l / np.pi) / (x**2 + gamma_l**2)
        conv = np.array(fftconvolve(gauss, lorentz, mode='same')) * dx

        # lsf_fwhm=0 means no additional LSF — Voigt is purely G(fwhm_g)⊗L(fwhm_l)
        faddeeva = np.array(
            functions.evaluate_voigt(jnp.array(x), 0.0, 0.0, self.fwhm_g, self.fwhm_l)
        )

        # Compare in the central region; trim edges where the finite-grid
        # convolution loses accuracy due to wrap-around
        trim = len(x) // 5
        region = slice(trim, -trim)
        l1_err = np.trapezoid(np.abs(conv[region] - faddeeva[region]), dx=dx)
        l1_ref = np.trapezoid(faddeeva[region], dx=dx)
        assert l1_err / l1_ref < 0.005  # < 0.5% L1 agreement


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
    (SkewVoigt(), {'fwhm_gauss': 80.0, 'fwhm_lorentz': 50.0, 'alpha': 0.5}),
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


# ---------------------------------------------------------------------------
# Private Voigt helper functions
# ---------------------------------------------------------------------------


class TestPrivateVoigtHelpers:
    """Tests for private Voigt helper functions not exercised elsewhere."""

    center = 5000.0
    lsf_fwhm = 5.0
    edges = jnp.linspace(4000.0, 6000.0, 2001)
    lo, hi = edges[:-1], edges[1:]
    wavelength = jnp.linspace(4500.0, 5500.0, 2000)

    def test_thompson_cdf_diff_normalizes(self):
        """_voigt_thompson_cdf_diff should sum to ~1 over a wide range."""
        lo = self.edges[:-1] - self.center
        hi = self.edges[1:] - self.center
        result = functions._voigt_thompson_cdf_diff(lo, hi, 80.0, 50.0)
        # Thompson pseudo-Voigt has Lorentzian component with heavy tails; allow 5% tolerance
        assert float(jnp.sum(result)) == pytest.approx(1.0, rel=0.05)

    def test_ida_pdf_normalizes(self):
        """_voigt_ida_pdf integrates to ~1, exercising _f_l_pdf, _f_p_pdf, _sech2."""
        dx = self.wavelength - self.center
        result = functions._voigt_ida_pdf(dx, 80.0, 50.0)
        step = float(self.wavelength[1] - self.wavelength[0])
        assert float(jnp.sum(result) * step) == pytest.approx(1.0, rel=0.05)


# ---------------------------------------------------------------------------
# integrate_skewVoigt
# ---------------------------------------------------------------------------


class TestIntegrateSkewVoigt:
    """Tests for the public integrate_skewVoigt kernel."""

    center = 5000.0
    lsf_fwhm = 5.0
    edges = jnp.linspace(4000.0, 6000.0, 2001)
    lo, hi = edges[:-1], edges[1:]

    def test_alpha_zero_matches_voigt(self):
        """integrate_skewVoigt with alpha=0 is identical to integrate_voigt."""
        skew = functions.integrate_skewVoigt(
            self.lo, self.hi, self.center, self.lsf_fwhm, 80.0, 50.0, 0.0
        )
        voigt = functions.integrate_voigt(
            self.lo, self.hi, self.center, self.lsf_fwhm, 80.0, 50.0
        )
        assert jnp.allclose(skew, voigt, rtol=1e-6)

    def test_normalization_with_skew(self):
        """integrate_skewVoigt sums to ~1 for nonzero alpha (anti-symmetric cancellation)."""
        result = functions.integrate_skewVoigt(
            self.lo, self.hi, self.center, self.lsf_fwhm, 80.0, 50.0, 1.0
        )
        # Midpoint skew approximation has ~2% discretization error for large alpha_eff
        assert float(jnp.sum(result)) == pytest.approx(1.0, rel=0.05)
