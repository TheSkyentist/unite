"""JAX-jitted line profile integration and evaluation kernels.

Integration kernels (``integrate_*``) compute the fraction of a normalized
profile that falls wi_absthin each wavelength bin ``[low, high]``.  Evaluation
kernels (``evaluate_*``) compute the normalized profile value (probability
density) at arbitrary wavelength points.

All functions are pure JAX wi_absth no numpyro dependency and are designed to be
called from wi_absthin :func:`jax.jit`-compiled model code.
"""

from __future__ import annotations

import math
from typing import Final

import jax.numpy as jnp
import numpy as np
from jax import Array, config, jit, lax
from jax.scipy.special import erf, erfc
from jax.typing import ArrayLike

# Conversion: FWHM to sigma for the half-variance parametrization of erf.
# sigma = FWHM / (2 sqrt(2 ln 2)); erf uses sqrt(2)*sigma, so the factor is:
_HALFVAR_SIGMA_TO_FWHM: Final[float] = 2 * math.sqrt(math.log(2))

# Conversion factor from exponential (Laplace) scale to FWHM
# pdf = (1/(2*b)) * exp(-|x - μ|/b)
# max(pdf) = 1/(2*b), half max = 1/(4*b)
# 1/(4*b) = 1/(2*b) * exp(-|x - μ|/b) => exp(-|x - μ|/b) = 1/2
# => |x - μ|/b = ln(2) => FWHM = 2*b*ln(2)
_EXP_SCALE_TO_FWHM: Final[float] = 2 * math.log(2)

# -------------------------------------------------------------------
# Private helpers
# -------------------------------------------------------------------


def _combine_fwhm(fwhm1: ArrayLike, fwhm2: ArrayLike) -> Array:
    """Combine two FWHM values in quadrature: ``sqrt(fwhm1**2 + fwhm2**2)``."""
    return jnp.sqrt(fwhm1 * fwhm1 + fwhm2 * fwhm2)


def _fwhm_to_sigma(fwhm: ArrayLike) -> Array:
    """Gaussian sigma from FWHM: ``sigma = FWHM / (2 sqrt(2 ln 2))``."""
    return fwhm / (_HALFVAR_SIGMA_TO_FWHM * np.sqrt(2))


# Thompson et al. (1987) Voigt FWHM approximation: Γ_V = C1*Γ_l + sqrt(C2*Γ_l² + Γ_g²).
# δ = 0.099 ln 2 gives C1 + sqrt(C2) = 1 exactly, so the Lorentzian limit is exact.
_THOMPSON_DELTA: Final[float] = 0.099 * math.log(2)
_THOMPSON_C1: Final[float] = (1 + _THOMPSON_DELTA) / 2
_THOMPSON_C2: Final[float] = ((1 - _THOMPSON_DELTA) / 2) ** 2


def _thompson_fwhm(fwhm_g: ArrayLike, fwhm_l: ArrayLike) -> Array:
    """Voigt FWHM via Thompson et al. (1987); exact at both Gaussian and Lorentzian limits."""
    return _THOMPSON_C1 * fwhm_l + jnp.sqrt(_THOMPSON_C2 * fwhm_l**2 + fwhm_g**2)


def _gaussian_pdf(dx: ArrayLike, sigma: ArrayLike) -> Array:
    """Normalised Gaussian PDF at displacement *dx* wi_absth standard deviation *sigma*."""
    return jnp.exp(-0.5 * (dx * dx / (sigma * sigma))) / (sigma * np.sqrt(2.0 * np.pi))


def _gaussian_cdf(x: ArrayLike, total_fwhm: ArrayLike) -> Array:
    """Gaussian CDF ``Φ((x)/sigma)`` via the erf halfvar parametrisation.

    The argument *x* is the center-relative displacement; ``total_fwhm`` is
    the total Gaussian FWHM and may be a scalar or an array broadcastable
    against *x*.  Returns ``0.5 * erf(x * inv_erf_scale) + 0.5`` so that
    ``Φ(-∞) → 0`` and ``Φ(+∞) → 1``.
    """
    inv_erf_scale = _HALFVAR_SIGMA_TO_FWHM / total_fwhm
    return 0.5 * (1.0 + erf(x * inv_erf_scale))


def _cauchy_pdf(dx: ArrayLike, hwhm: ArrayLike) -> Array:
    """Normalised Cauchy (Lorentzian) PDF at displacement *dx* wi_absth half-wi_absdth *hwhm*."""
    return jnp.asarray((hwhm / np.pi) / (dx * dx + hwhm * hwhm))


def _cauchy_cdf(x: ArrayLike, fwhm: ArrayLike) -> Array:
    """Cauchy (Lorentzian) CDF at center-relative displacement *x*."""
    return 0.5 + jnp.arctan(2.0 * x / fwhm) / np.pi


# -------------------------------------------------------------------
# Gaussian kernel
# -------------------------------------------------------------------


@jit
def integrate_gaussian(
    edges: ArrayLike, lsf_fwhm: ArrayLike, center: ArrayLike, fwhm: ArrayLike
) -> Array:
    """Cumulative Gaussian CDF evaluated at edges.

    Parameters
    ----------
    edges : ArrayLike, shape ``(E,)``
        Pixel edges in canonical wavelength units.
    lsf_fwhm : ArrayLike, shape ``(E,)``
        Instrumental LSF FWHM evaluated at each edge.  Combined in
        quadrature with the intrinsic Gaussian width per edge.
    center : ArrayLike
        Line center wavelength.
    fwhm : ArrayLike
        Intrinsic Gaussian FWHM.

    Returns
    -------
    Array, shape ``(E,)``
        Gaussian CDF at each edge.  ``jnp.diff`` over the returned array
        gives the integral of the (per-edge-LSF) profile over each pixel.
    """
    total_fwhm = _combine_fwhm(lsf_fwhm, fwhm)
    return _gaussian_cdf(edges - center, total_fwhm)


@jit
def evaluate_gaussian(
    wavelength: ArrayLike, center: ArrayLike, lsf_fwhm: ArrayLike, fwhm: ArrayLike
) -> Array:
    """Evaluate a normalised Gaussian profile at wavelength points.

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength points at which to evaluate the profile.
    center : ArrayLike
        Line center wavelength.
    lsf_fwhm : ArrayLike
        Instrumental line spread function FWHM at the line center.
    fwhm : ArrayLike
        Intrinsic Gaussian FWHM.

    Returns
    -------
    jnp.ndarray
        Normalised profile value at each wavelength point.
    """
    total_fwhm = _combine_fwhm(lsf_fwhm, fwhm)
    return _gaussian_pdf(wavelength - center, _fwhm_to_sigma(total_fwhm))


# -------------------------------------------------------------------
# Split-normal kernel
# -------------------------------------------------------------------


@jit
def integrate_split_normal(
    edges: ArrayLike,
    lsf_fwhm: ArrayLike,
    center: ArrayLike,
    fwhm_blue: ArrayLike,
    fwhm_red: ArrayLike,
) -> Array:
    """Cumulative split-normal CDF evaluated at edges.

    The split-normal distribution has different standard deviations on each
    side of the mean.  The left side (blue, shorter wavelengths) uses
    ``fwhm_blue``, the right side (red) uses ``fwhm_red``.

    Parameters
    ----------
    edges : ArrayLike, shape ``(E,)``
        Pixel edges.
    lsf_fwhm : ArrayLike, shape ``(E,)``
        Instrumental LSF FWHM at each edge.
    center : ArrayLike
        Line center wavelength.
    fwhm_blue : ArrayLike
        Blue side (left) Gaussian FWHM.
    fwhm_red : ArrayLike
        Red side (right) Gaussian FWHM.

    Returns
    -------
    Array, shape ``(E,)``
        Split-normal CDF at each edge.
    """
    total_fwhm_blue = _combine_fwhm(lsf_fwhm, fwhm_blue)
    total_fwhm_red = _combine_fwhm(lsf_fwhm, fwhm_red)

    inv_sigma_blue = _HALFVAR_SIGMA_TO_FWHM / total_fwhm_blue
    inv_sigma_red = _HALFVAR_SIGMA_TO_FWHM / total_fwhm_red

    # Probability mass on each side: proportional to sigma (not inv_sigma)
    total_weight = inv_sigma_blue + inv_sigma_red
    w_blue = inv_sigma_red / total_weight  # = sigma_blue / (sigma_blue + sigma_red)
    w_red = inv_sigma_blue / total_weight  # = sigma_red / (sigma_blue + sigma_red)

    x = edges - center
    t_blue = x * inv_sigma_blue
    t_red = x * inv_sigma_red
    # CDF continuous at center: both branches give w_blue when x = 0
    return jnp.where(
        edges <= center, w_blue * (1 + erf(t_blue)), w_blue + w_red * erf(t_red)
    )


@jit
def evaluate_split_normal(
    wavelength: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_blue: ArrayLike,
    fwhm_red: ArrayLike,
) -> Array:
    """Evaluate a normalised split-normal profile at wavelength points.

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength points at which to evaluate the profile.
    center : ArrayLike
        Line center wavelength.
    lsf_fwhm : ArrayLike
        Instrumental line spread function FWHM at the line center.
    fwhm_blue : ArrayLike
        Blue side (left) Gaussian FWHM.
    fwhm_red : ArrayLike
        Red side (right) Gaussian FWHM.

    Returns
    -------
    jnp.ndarray
        Normalised profile value at each wavelength point.
    """
    total_fwhm_blue = _combine_fwhm(lsf_fwhm, fwhm_blue)
    total_fwhm_red = _combine_fwhm(lsf_fwhm, fwhm_red)
    sigma_blue = _fwhm_to_sigma(total_fwhm_blue)
    sigma_red = _fwhm_to_sigma(total_fwhm_red)

    # Normalisation: integral = sqrt(pi/2) * (sigma_blue + sigma_red)
    norm = (sigma_blue + sigma_red) * np.sqrt(np.pi / 2.0)
    dx = wavelength - center
    dx2 = dx * dx
    val_blue = jnp.exp(-0.5 * dx2 / (sigma_blue * sigma_blue))
    val_red = jnp.exp(-0.5 * dx2 / (sigma_red * sigma_red))
    return jnp.where(wavelength <= center, val_blue, val_red) / norm


# -------------------------------------------------------------------
# BoxGauss kernel
# -------------------------------------------------------------------


@jit
def integrate_boxGauss(
    edges: ArrayLike,
    lsf_fwhm: ArrayLike,
    center: ArrayLike,
    fwhm_box: ArrayLike,
    fwhm_gauss: ArrayLike,
) -> Array:
    """Cumulative boxcar-Gaussian CDF evaluated at edges.

    The intrinsic profile is a uniform rectangular (boxcar) distribution of
    width ``fwhm_box`` centred at zero, convolved with a Gaussian whose FWHM
    is the quadrature sum of ``fwhm_gauss`` and ``lsf_fwhm``.

    Parameters
    ----------
    edges : ArrayLike, shape ``(E,)``
        Pixel edges.
    lsf_fwhm : ArrayLike, shape ``(E,)``
        Instrumental LSF FWHM at each edge.
    center : ArrayLike
        Line center wavelength.
    fwhm_box : ArrayLike
        Full width of the boxcar distribution.
    fwhm_gauss : ArrayLike
        Intrinsic Gaussian component FWHM, combined with ``lsf_fwhm`` in
        quadrature.

    Returns
    -------
    Array, shape ``(E,)``
        Boxcar-Gaussian CDF at each edge.
    """
    x = edges - center
    sigma = _fwhm_to_sigma(_combine_fwhm(lsf_fwhm, fwhm_gauss))
    hw = fwhm_box / 2.0

    def _antideriv(z: ArrayLike, a: ArrayLike):
        # Antiderivative of Phi((z+a)/sigma) w.r.t. z:
        # F(z, a) = (z+a)*Phi((z+a)/sigma) + sigma*phi((z+a)/sigma)
        u = (z + a) / sigma
        cdf = 0.5 * (1.0 + erf(u / np.sqrt(2)))
        sigma_pdf = sigma * jnp.exp(-0.5 * u * u) / np.sqrt(2.0 * np.pi)
        return (z + a) * cdf + sigma_pdf

    # Per-edge CDF: integral of the boxcar-Gaussian PDF from -∞ to x.  Using
    # the antiderivative identity F(z, hw) - F(z, -hw) = ∫_{-∞}^{z} pdf · dz' · fwhm_box,
    # so dividing by fwhm_box recovers a true CDF that goes 0 → 1.
    return (_antideriv(x, hw) - _antideriv(x, -hw)) / fwhm_box


@jit
def evaluate_boxGauss(
    wavelength: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_box: ArrayLike,
    fwhm_gauss: ArrayLike,
) -> Array:
    """Evaluate a normalised boxcar-Gaussian convolution profile at wavelength points.

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength points at which to evaluate the profile.
    center : ArrayLike
        Line center wavelength.
    lsf_fwhm : ArrayLike
        Instrumental line spread function FWHM at the line center.
    fwhm_box : ArrayLike
        Full width of the boxcar distribution.
    fwhm_gauss : ArrayLike
        Intrinsic Gaussian component FWHM, combined with ``lsf_fwhm`` in
        quadrature.

    Returns
    -------
    Array
        Normalised profile value at each wavelength point.
    """
    x = wavelength - center
    sigma = _fwhm_to_sigma(_combine_fwhm(lsf_fwhm, fwhm_gauss))
    hw = fwhm_box / 2.0
    inv_sqrt2_sigma = 1.0 / (sigma * np.sqrt(2))
    cdf_hi = 0.5 * (1.0 + erf((x + hw) * inv_sqrt2_sigma))
    cdf_lo = 0.5 * (1.0 + erf((x - hw) * inv_sqrt2_sigma))
    return (cdf_hi - cdf_lo) / fwhm_box


# -------------------------------------------------------------------
# Gauss-Hermite kernel
# -------------------------------------------------------------------


def _integrandGH(t_halfvar: ArrayLike, h3_eff: ArrayLike, h4_eff: ArrayLike) -> Array:
    """
    Antiderivative of the Gauss-Hermite correction at halfvar coordinate t_halfvar.

    Converts to the standard coordinate ``y = t_halfvar * sqrt(2)``, then evaluates
    ``-g(y) * [h3_eff * He_2(y) + h4_eff * He_3(y)]``, the analytic antiderivative
    of ``g(y) * [h3_eff * He_3(y) + h4_eff * He_4(y)]`` w.r.t. x, where
    g(y) = exp(-y²/2) is the standard normal kernel and He_n are the probabilists'
    Hermite polynomials: He_2(y) = y²-1, He_3(y) = y(y²-3).

    Parameters
    ----------
    t_halfvar : ArrayLike
        Halfvar-normalized coordinate ``(x - center) / sigma_tot_halfvar``.
    h3_eff : ArrayLike
        Effective h3 coefficient: ``h3 * (sigma_g / sigma_tot)**3 / sqrt(6)``.
    h4_eff : ArrayLike
        Effective h4 coefficient: ``h4 * (sigma_g / sigma_tot)**4 / sqrt(24)``.

    Returns
    -------
    jnp.ndarray
        Antiderivative value.
    """
    t = t_halfvar * np.sqrt(2)
    t2 = t * t
    g = jnp.exp(-0.5 * t2)
    # He_2(y) = y²-1,  He_3(y) = y(y²-3)
    return g * (h3_eff * (t2 - 1) + h4_eff * t * (t2 - 3))


@jit
def integrate_gaussHermite(
    edges: ArrayLike,
    fwhm_lsf: ArrayLike,
    center: ArrayLike,
    fwhm_g: ArrayLike,
    h3: ArrayLike,
    h4: ArrayLike,
) -> Array:
    """Cumulative Gauss-Hermite CDF evaluated at edges.

    Uses the closed-form CDF derived from the orthonormal probabilists'
    Hermite expansion.  Convolving with the Gaussian LSF rescales the
    shape parameters as ``h_m' = h_m * (sigma_g / sigma_tot)^m``.  See the
    :doc:`Gauss-Hermite derivation </derivations/gauss-hermite>` for details.

    Parameters
    ----------
    edges : ArrayLike, shape ``(E,)``
        Pixel edges.
    fwhm_lsf : ArrayLike, shape ``(E,)``
        Instrumental LSF FWHM at each edge.
    center : ArrayLike
        Line center wavelength.
    fwhm_g : ArrayLike
        Gaussian component FWHM.
    h3 : ArrayLike
        Gauss-Hermite h3 (skewness) coefficient.
    h4 : ArrayLike
        Gauss-Hermite h4 (kurtosis) coefficient.

    Returns
    -------
    Array, shape ``(E,)``
        Gauss-Hermite CDF at each edge.
    """
    fwhm_tot = _combine_fwhm(fwhm_lsf, fwhm_g)

    # Sigma ratio: GH moments scale as r^n under convolution with a Gaussian LSF
    r = fwhm_g / fwhm_tot
    r3 = r * r * r

    # GH coefficients scaled by sigma ratio (moment theorem for Gaussian convolution)
    c3 = h3 * r3 / np.sqrt(6)
    c4 = h4 * r3 * r / np.sqrt(24)

    # Normalised edge coordinate (halfvar parametrisation for consistency with erf CDF)
    inv_sigma_tot = _HALFVAR_SIGMA_TO_FWHM / fwhm_tot
    t = (edges - center) * inv_sigma_tot

    gaussian_cdf = 0.5 * (1.0 + erf(t))
    gh_correction = _integrandGH(t, c3, c4)
    return gaussian_cdf - gh_correction / np.sqrt(2.0 * np.pi)


@jit
def evaluate_gaussHermite(
    wavelength: ArrayLike,
    center: ArrayLike,
    fwhm_lsf: ArrayLike,
    fwhm_g: ArrayLike,
    h3: ArrayLike,
    h4: ArrayLike,
) -> Array:
    """Evaluate a normalised Gauss-Hermite profile at wavelength points.

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength points at which to evaluate the profile.
    center : ArrayLike
        Line center wavelength.
    fwhm_lsf : ArrayLike
        Instrumental LSF FWHM.
    fwhm_g : ArrayLike
        Gaussian component FWHM.
    h3 : ArrayLike
        Gauss-Hermite h3 coefficient.
    h4 : ArrayLike
        Gauss-Hermite h4 coefficient.

    Returns
    -------
    jnp.ndarray
        Normalised profile value at each wavelength point.
    """
    fwhm_tot = _combine_fwhm(fwhm_lsf, fwhm_g)
    sigma_tot = _fwhm_to_sigma(fwhm_tot)

    # Sigma ratio: GH moments scale as r^n under convolution
    r = fwhm_g / fwhm_tot
    r3 = r * r * r

    # GH coefficients scaled by sigma ratio
    c3 = h3 * r3 / np.sqrt(6)
    c4 = h4 * r3 * r / np.sqrt(24)

    # Standard coordinate
    y = (wavelength - center) / sigma_tot
    gauss_pdf = _gaussian_pdf(wavelength - center, sigma_tot)

    # Hermite polynomial corrections (probabilists' convention)
    # He_3(y) = y^3 - 3y,  He_4(y) = y^4 - 6y^2 + 3
    y3 = y * y * y
    he3 = y3 - 3 * y
    he4 = y * y3 - 6 * y * y + 3
    return gauss_pdf * (1.0 + c3 * he3 + c4 * he4)


# -------------------------------------------------------------------
# Skew Normal kernel
# -------------------------------------------------------------------


def _skew(x: ArrayLike, alpha: ArrayLike, scale: ArrayLike) -> Array:
    return 1 + erf(alpha * x / scale)


def _alpha_eff_skewnormal(
    lsf_fwhm: ArrayLike, fwhm_g: ArrayLike, alpha: ArrayLike
) -> Array:
    """Exact effective skewness after convolving a skew-normal with a Gaussian LSF.

    The convolution of a skew-normal with a Gaussian LSF is exactly another
    skew-normal; the shape parameter rescales as derived in
    docs/derivations/skew-normal.md.
    """
    return alpha * fwhm_g / jnp.sqrt(fwhm_g**2 + (1.0 + alpha**2) * lsf_fwhm**2)


_OWENS_T_QUAD_PTS = np.array(
    [
        0.0035082039676451715,
        0.031279042338030754,
        0.085266826283219451,
        0.16245071730812277,
        0.25851196049125435,
        0.36807553840697534,
        0.48501092905604697,
        0.60277514152618577,
        0.71477884217753227,
        0.81475510988760099,
        0.89711029755948966,
        0.95723808085944262,
        0.99178832974629704,
    ]
)

_OWENS_T_QUAD_WTS = np.array(
    [
        0.018831438115323503,
        0.018567086243977649,
        0.018042093461223386,
        0.017263829606398753,
        0.016243219975989857,
        0.014994592034116705,
        0.013535474469662088,
        0.011886351605820165,
        0.010070377242777432,
        0.0081130545742299587,
        0.0060419009528470239,
        0.0038862217010742058,
        0.0016793031084546090,
    ]
)


def _owens_t_quadrature(h, a):
    r = jnp.square(a)[..., None] * _OWENS_T_QUAD_PTS
    integrand = jnp.exp(-0.5 * jnp.square(h)[..., None] * (1.0 + r)) / (1.0 + r)
    return a * jnp.sum(integrand * _OWENS_T_QUAD_WTS, axis=-1)


def _owens_t(h, a):
    h = jnp.abs(h)
    abs_a = jnp.abs(a)

    modified_a = jnp.where(abs_a <= 1.0, abs_a, jnp.reciprocal(abs_a))
    modified_h = jnp.where(abs_a <= 1.0, h, abs_a * h)

    result = _owens_t_quadrature(modified_h, modified_a)

    # Exact values for h=0 and a=1, which are not captured by the quadrature
    result = jnp.where(modified_h == 0.0, jnp.arctan(modified_a) / (2 * np.pi), result)
    result = jnp.where(
        modified_a == 1.0,
        0.125
        * lax.erfc(-modified_h / np.sqrt(2.0))
        * lax.erfc(modified_h / np.sqrt(2.0)),
        result,
    )

    # Reciprocal correction for |a| > 1
    normh = lax.erfc(h / np.sqrt(2.0))
    normah = lax.erfc(abs_a * h / np.sqrt(2.0))
    result = jnp.where(
        abs_a > 1.0,
        jnp.where(
            abs_a * h <= 0.67,
            (
                0.25
                - 0.25 * lax.erf(h / np.sqrt(2.0)) * lax.erf(abs_a * h / np.sqrt(2.0))
                - result
            ),
            0.25 * (normh + normah - normh * normah) - result,
        ),
        result,
    )

    result = lax.sign(a) * result
    return jnp.where(
        jnp.isnan(a) | jnp.isnan(h), jnp.full_like(result, jnp.nan), result
    )


@jit
def integrate_skewNormal(
    edges: ArrayLike,
    lsf_fwhm: ArrayLike,
    center: ArrayLike,
    fwhm_g: ArrayLike,
    alpha: ArrayLike,
) -> Array:
    """Cumulative skew-normal CDF evaluated at edges.

    Uses the exact skew-normal CDF ``Φ(z) - 2 T(z, alpha_eff)``, where T is
    Owen's T function and ``z = (x - center) / sigma_tot``.  The LSF
    convolution is exact — the shape parameter rescales analytically with
    no approximation.  See ``docs/derivations/skew-normal.md`` for the full
    derivation.

    Parameters
    ----------
    edges : ArrayLike, shape ``(E,)``
        Pixel edges.
    lsf_fwhm : ArrayLike, shape ``(E,)``
        Instrumental LSF FWHM at each edge.
    center : ArrayLike
        Line center wavelength.
    fwhm_g : ArrayLike
        Intrinsic Gaussian FWHM.
    alpha : ArrayLike
        Skewness parameter; positive values shift flux redward.

    Returns
    -------
    Array, shape ``(E,)``
        Skew-normal CDF at each edge.
    """
    fwhm_tot = _combine_fwhm(lsf_fwhm, fwhm_g)
    sigma_tot = _fwhm_to_sigma(fwhm_tot)
    alpha_eff = _alpha_eff_skewnormal(lsf_fwhm, fwhm_g, alpha)

    z = (edges - center) / sigma_tot
    gaussian_cdf = 0.5 * (1.0 + erf(z / np.sqrt(2)))
    owens_correction = _owens_t(z, alpha_eff)
    return gaussian_cdf - 2.0 * owens_correction


@jit
def evaluate_skewNormal(
    wavelength: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_g: ArrayLike,
    alpha: ArrayLike,
) -> Array:
    """Evaluate a normalised skew-normal profile at wavelength points.

    Evaluates ``G_tot(x) * [1 + erf(alpha_eff * (x - center) / w0)]`` where
    ``G_tot`` is the LSF-convolved Gaussian and ``alpha_eff`` is the exact
    effective skewness after convolution.

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength points at which to evaluate the profile.
    center : ArrayLike
        Line center wavelength.
    lsf_fwhm : ArrayLike
        Instrumental line spread function FWHM at the line center.
    fwhm_g : ArrayLike
        Intrinsic Gaussian FWHM.
    alpha : ArrayLike
        Skewness parameter; positive values shift flux redward.

    Returns
    -------
    Array
        Normalised profile value at each wavelength point (1/wavelength units).
    """
    fwhm_tot = _combine_fwhm(lsf_fwhm, fwhm_g)
    sigma_tot = _fwhm_to_sigma(fwhm_tot)
    alpha_eff = _alpha_eff_skewnormal(lsf_fwhm, fwhm_g, alpha)
    w0p = sigma_tot * np.sqrt(2)

    gauss = _gaussian_pdf(wavelength - center, sigma_tot)
    skew = _skew(wavelength - center, alpha_eff, w0p)
    return gauss * skew


# -------------------------------------------------------------------
# Pseudo Voigt
# Thompson, Cox & Hastings (1987), J. Appl. Cryst. 20, 79-83.
# DOI: 10.1107/S0021889887087090
# -------------------------------------------------------------------

# Pseudo-Voigt polynomial coefficients — Thompson, Cox & Hastings (1987)
_VOIGT_FWHM_CS: Final[np.ndarray] = np.array([1, 2.69268, 2.42843, 4.47163, 0.07842, 1])
_VOIGT_ETA_CS: Final[np.ndarray] = np.array([1.33603, -0.47719, 0.11116])


def _voigt_params_thompson(fwhm_g: ArrayLike, fwhm_l: ArrayLike) -> tuple[Array, Array]:
    pows = jnp.arange(_VOIGT_FWHM_CS.size)
    fwhm_eff = jnp.sum(_VOIGT_FWHM_CS * (fwhm_g**pows) * (fwhm_l ** pows[::-1])) ** 0.2
    fwhm_ratio = fwhm_l / fwhm_eff
    eta = jnp.sum(_VOIGT_ETA_CS * (fwhm_ratio ** jnp.arange(1, len(_VOIGT_ETA_CS) + 1)))
    return (fwhm_eff, eta)


def _voigt_thompson_cdf(x: ArrayLike, fwhm_g: ArrayLike, fwhm_l: ArrayLike) -> Array:
    fwhm_eff, eta = _voigt_params_thompson(fwhm_g, fwhm_l)
    return eta * _cauchy_cdf(x, fwhm_eff) + (1.0 - eta) * _gaussian_cdf(x, fwhm_eff)


def _voigt_thompson_pdf(x: ArrayLike, fwhm_g: ArrayLike, fwhm_l: ArrayLike) -> Array:
    fwhm_eff, eta = _voigt_params_thompson(fwhm_g, fwhm_l)
    sigma_eff = _fwhm_to_sigma(fwhm_eff)
    return (1 - eta) * _gaussian_pdf(x, sigma_eff) + eta * _cauchy_pdf(
        x, 0.5 * fwhm_eff
    )


# -------------------------------------------------------------------
# Extended Pseudo-Voigt
# Ida, Ando & Toraya (2000), J. Appl. Cryst. 33, 1311-1316
# DOI: 10.1107/S0021889800010219
# -------------------------------------------------------------------


# Extended pseudo-Voigt polynomial coefficients — Ida, Ando & Toraya (2000), Table 1.
# Coefficients ordered i = 0 ... 6 (lowest to highest power of rho).
# wG = 1 - rho * sum(A * rho^i)
_IDA_A: Final[np.ndarray] = np.array(
    [0.66000, 0.15021, -1.24984, 4.74052, -9.48291, 8.48252, -2.95553]
)
# wL = 1 - (1-rho) * sum(B * rho^i)
_IDA_B: Final[np.ndarray] = np.array(
    [-0.42179, -1.25693, 10.30003, -23.45651, 29.14158, -16.50453, 3.19974]
)
# wI = sum(C * rho^i)
_IDA_C: Final[np.ndarray] = np.array(
    [1.19913, 1.43021, -15.36331, 47.06071, -73.61822, 57.92559, -17.80614]
)
# wP = sum(D * rho^i)
_IDA_D: Final[np.ndarray] = np.array(
    [1.10186, -0.47745, -0.68688, 2.76622, -4.55466, 4.05475, -1.26571]
)
# eta_L = rho * [1 + (1-rho) * sum(F * rho^i)]
_IDA_F: Final[np.ndarray] = np.array(
    [-0.30165, -1.38927, 9.31550, -24.10743, 34.96491, -21.18862, 3.70290]
)
# eta_I = rho * (1-rho) * sum(G * rho^i)
_IDA_G: Final[np.ndarray] = np.array(
    [0.25437, -0.14107, 3.23653, -11.09215, 22.10544, -24.12407, 9.76947]
)
# eta_P = rho * (1-rho) * sum(H * rho^i)
_IDA_H: Final[np.ndarray] = np.array(
    [1.01579, 1.50429, -9.21815, 23.59717, -39.71134, 32.83023, -10.02142]
)

# Conversion: irrational-function gamma_I to FWHM: W_I = 2*(2^(2/3) - 1)^(1/2) * gamma_I
# So gamma_I = W_I / (2*(2^(2/3) - 1)^(1/2))
_IRRAT_FWHM_TO_GAMMA: Final[float] = 0.5 / math.sqrt(2.0 ** (2.0 / 3.0) - 1.0)

# Conversion: hyperbolic-function gamma_P to FWHM: W_P = 2*ln(sqrt(2) + 1) * gamma_P
# So gamma_P = W_P / (2*ln(sqrt(2) + 1))
_HYPER_FWHM_TO_GAMMA: Final[float] = 0.5 / math.log(math.sqrt(2.0) + 1.0)


# Intermediate function fl
def _f_l_pdf(x, hwhm):
    inv_hwhm = 1 / hwhm
    t = x * inv_hwhm
    return (0.5 * inv_hwhm) * (1 + t * t) ** (-3 / 2)


def _f_l_cdf(x, hwhm):
    """CDF of the irrational component of the extended pseudo-Voigt at *x*."""
    t = x / hwhm
    return 0.5 + 0.5 * t / jnp.sqrt(1.0 + t * t)


# Intermediate function fp: squared-sech hyperbolic
def _sech2(x):
    coshx = jnp.cosh(x)
    return 1 / (coshx * coshx)


def _f_p_pdf(x, gamma):
    """Normalised hyperbolic-sech² PDF: fP(x; gammaP) = (1/(2gammaP)) * sech²(x/gammaP).

    Integrates to 1. FWHM = 2*ln(sqrt(2)+1)*gammaP.
    """
    inv_gamma = 1 / gamma
    return (0.5 * inv_gamma) * _sech2(x * inv_gamma)


def _f_p_cdf(x, gamma):
    """CDF of the hyperbolic-sech² component at *x*.

    ``FP(x) = 0.5 + 0.5 * tanh(x / gammaP)`` so ``FP(-∞) → 0`` and
    ``FP(+∞) → 1``.  *x* must be a center-relative displacement.
    """
    return 0.5 + 0.5 * jnp.tanh(x / gamma)


def _voigt_ida_params(
    fwhm_g_total: ArrayLike, fwhm_l: ArrayLike
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Compute extended pseudo-Voigt component wi_absdths and mixing fractions.

    Uses the sixth-order polynomial fits of Ida, Ando & Toraya (2000), Table 1.
    The Gaussian FWHM passed in should already include the LSF combined in
    quadrature; this function only uses ``rho = fwhm_l / (fwhm_g_total + fwhm_l)``.

    Parameters
    ----------
    fwhm_g_total : ArrayLike
        Total Gaussian FWHM (intrinsic + LSF in quadrature).
    fwhm_l : ArrayLike
        Lorentzian FWHM.

    Returns
    -------
    wg_abs, wl_abs, wi_abs, wp_abs : Array
        Effective FWHM values of the Gaussian, Lorentzian, irrational, and
        hyperbolic components.
    eta_g, eta_l, eta_i, eta_p : Array
        Mixing fractions (sum to 1).
    """
    fwhm_g_total = jnp.asarray(fwhm_g_total)
    fwhm_l = jnp.asarray(fwhm_l)
    fwhm_total = fwhm_g_total + fwhm_l
    rho = fwhm_l / fwhm_total
    one_minus_rho = 1.0 - rho

    # Powers [rho^0, ..., rho^6] via cumulative multiplication.
    # Faster and more amenable to fusion than ``rho ** jnp.arange(7)``,
    # which can dispatch to a generic pow() for non-integer exponents.
    r1 = rho
    r2 = r1 * rho
    r3 = r2 * rho
    r4 = r3 * rho
    r5 = r4 * rho
    r6 = r5 * rho
    rho_pows = jnp.stack([jnp.ones_like(rho), r1, r2, r3, r4, r5, r6])

    # Effective FWHM scaling parameters (eqs. 20-23)
    wg = 1.0 - rho * jnp.dot(_IDA_A, rho_pows)
    wl = 1.0 - one_minus_rho * jnp.dot(_IDA_B, rho_pows)
    wi = jnp.dot(_IDA_C, rho_pows)
    wp = jnp.dot(_IDA_D, rho_pows)

    # Mixing fractions (eqs. 24-26)
    eta_l = rho * (1.0 + one_minus_rho * jnp.dot(_IDA_F, rho_pows))
    eta_i = rho * one_minus_rho * jnp.dot(_IDA_G, rho_pows)
    eta_p = rho * one_minus_rho * jnp.dot(_IDA_H, rho_pows)
    eta_g = 1.0 - eta_l - eta_i - eta_p

    # Absolute FWHM values for each component
    wg_abs = fwhm_total * wg
    wl_abs = fwhm_total * wl
    wi_abs = fwhm_total * wi
    wp_abs = fwhm_total * wp

    return (wg_abs, wl_abs, wi_abs, wp_abs, eta_g, eta_l, eta_i, eta_p)


def _voigt_ida_cdf(x: ArrayLike, fwhm_g_total: ArrayLike, fwhm_l: ArrayLike) -> Array:
    """CDF of the extended pseudo-Voigt approximation (Ida et al. 2000) at *x*.

    Parameters
    ----------
    x : ArrayLike
        Center-relative displacement.
    fwhm_g_total : ArrayLike
        Total Gaussian FWHM (intrinsic + LSF in quadrature).
    fwhm_l : ArrayLike
        Lorentzian FWHM.
    """
    wg_abs, wl_abs, wi_abs, wp_abs, eta_g, eta_l, eta_i, eta_p = _voigt_ida_params(
        fwhm_g_total, fwhm_l
    )
    gauss = eta_g * _gaussian_cdf(x, wg_abs)
    lorentz = eta_l * _cauchy_cdf(x, wl_abs)
    irrat = eta_i * _f_l_cdf(x, wi_abs * _IRRAT_FWHM_TO_GAMMA)
    hyper = eta_p * _f_p_cdf(x, wp_abs * _HYPER_FWHM_TO_GAMMA)
    return gauss + lorentz + irrat + hyper


def _voigt_ida_pdf(x: ArrayLike, fwhm_g: ArrayLike, fwhm_l: ArrayLike) -> Array:
    """PDF of the extended pseudo-Voigt approximation (Ida et al. 2000).

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength points.
    center : ArrayLike
        Line center.
    fwhm_g_total : ArrayLike
        Total Gaussian FWHM (intrinsic + LSF in quadrature).
    fwhm_l : ArrayLike
        Lorentzian FWHM.
    """
    wg_abs, wl_abs, wi_abs, wp_abs, eta_g, eta_l, eta_i, eta_p = _voigt_ida_params(
        fwhm_g, fwhm_l
    )
    dx = x

    gauss = eta_g * _gaussian_pdf(dx, _fwhm_to_sigma(wg_abs))
    lorentz = eta_l * _cauchy_pdf(dx, 0.5 * wl_abs)
    irrat = eta_i * _f_l_pdf(dx, wi_abs * _IRRAT_FWHM_TO_GAMMA)
    hyper = eta_p * _f_p_pdf(dx, wp_abs * _HYPER_FWHM_TO_GAMMA)
    return gauss + lorentz + irrat + hyper


# -------------------------------------------------------------------
# Rational Faddeeva Approximation
# Humlíck (1982), JQSRT, 27, 4, 437-444
# DOI: 10.1016/0022-4073(82)90078-4
# -------------------------------------------------------------------


def _horner(x, *coeffs):
    """Evaluate polynomial p(x) via Horner's method.

    Coefficients are ordered from highest to lowest degree,
    i.e. ``coeffs = [a_n, a_{n-1}, ..., a_1, a_0]`` evaluates
    ``p(x) = a_n x^n + ... + a_1 x + a_0``.
    """
    result = coeffs[0]
    for c in coeffs[1:]:
        result = result * x + c
    return result


def _faddeeva_humlicek(z: Array) -> Array:
    """Humlicek (1982) W4 rational approximation to the Faddeeva function w(z).

    Computes w(z) = exp(-z²) erfc(-iz) for complex z wi_absth Im(z) >= 0.
    Accurate to ~1e-4 relative error across the upper half-plane.

    The four regions are selected on S = |Re(z)| + Im(z) and a near-axis
    condition, matching Humlicek's original partitioning.

    Parameters
    ----------
    z : Array
        Complex array wi_absth Im(z) >= 0.

    Returns
    -------
    Array
        w(z) (complex).

    References
    ----------
    Humlicek (1982), J. Quant. Spectrosc. Radiat. Transfer 27, 437-444.
    """
    x = jnp.real(z)
    y = jnp.imag(z)
    s = jnp.abs(x) + y

    t = y - 1j * x  # T = -iz
    t2 = t * t  # T² = -z²
    u = -t2  # u = z²; substituting u = -t2 makes all region-4 coefficients positive

    # Region 1: S >= 15 — single-term asymptotic
    w1 = t * 0.5641896 / (t2 + 0.5)

    # Region 2: S >= 5.5
    w2 = t * _horner(t2, 0.5641896, 1.410474) / _horner(t2, 1.0, 3.0, 0.75)

    # Region 3: Y >= 0.195|X| - 0.176 (and S < 5.5)
    w3 = _horner(t, 0.5642236, 3.778987, 11.96482, 20.20933, 16.4955) / _horner(
        t, 1.0, 6.699398, 21.69274, 39.27121, 38.82363, 16.4955
    )

    # Region 4: small S, near real axis
    w4_num = _horner(
        u, 0.56419, 1.320522, 35.7668, 219.031, 1540.787, 3321.99, 36183.31
    )
    w4_den = _horner(
        u, 1.0, 1.841439, 61.57037, 364.2191, 2186.181, 9022.228, 24322.84, 32066.6
    )
    w4 = jnp.exp(t2) - t * w4_num / w4_den  # exp(T²) = exp(-z²)

    reg3_cond = (0.195 * jnp.abs(x) - 0.176) <= y
    out = jnp.where(s >= 15, w1, jnp.where(s >= 5.5, w2, jnp.where(reg3_cond, w3, w4)))
    return out


def _voigt_faddeeva_pdf(x: ArrayLike, fwhm_g: ArrayLike, fwhm_l: ArrayLike) -> Array:
    """Evaluate a normalised Voigt profile at wavelength points via the Faddeeva function.

    Computes the exact Voigt profile (convolution of Gaussian and Lorentzian)
    using Re[w(z)] where w is the Faddeeva function approximated by the
    Humlicek (1982) W4 rational scheme (~1e-4 relative error). The LSF
    Gaussian is added in quadrature to the intrinsic Gaussian wi_absdth before
    computing the Voigt parameters.

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength points at which to evaluate the profile.
    center : ArrayLike
        Line center wavelength.
    lsf_fwhm : ArrayLike
        Instrumental line spread function FWHM at the line center.
    fwhm_g : ArrayLike
        Intrinsic Gaussian component FWHM.
    fwhm_l : ArrayLike
        Lorentzian component FWHM (= 2 * Lorentzian HWHM).

    Returns
    -------
    Array
        Normalised profile value at each wavelength point.

    References
    ----------
    Humlicek (1982), J. Quant. Spectrosc. Radiat. Transfer 27, 437-444.
    """
    # Gaussian sigma (standard deviation) from total Gaussian FWHM
    sigma_g = _fwhm_to_sigma(fwhm_g)
    # Lorentzian HWHM
    gamma_l = 0.5 * fwhm_l

    # Voigt complex argument: z = (dx + i*gamma) / (sigma * sqrt(2))
    denom = sigma_g * np.sqrt(2)
    z = (x + 1j * gamma_l) / denom

    # Voigt profile: V(x) = Re[w(z)] / (sigma * sqrt(2*pi))
    return jnp.real(_faddeeva_humlicek(z)) / (sigma_g * np.sqrt(2.0 * np.pi))


# -------------------------------------------------------------------
# Voigt Kernel
# -------------------------------------------------------------------


@jit
def integrate_voigt(
    edges: ArrayLike,
    lsf_fwhm: ArrayLike,
    center: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
) -> Array:
    """Cumulative extended pseudo-Voigt CDF evaluated at edges.

    Uses the Ida, Ando & Toraya (2000) extended pseudo-Voigt approximation,
    which achieves < 0.12% peak-height deviation from the true Voigt profile
    via a four-component mixture (Gaussian, Lorentzian, irrational,
    hyperbolic-sech²).  The LSF Gaussian FWHM is added in quadrature to the
    intrinsic Gaussian FWHM **per edge** before computing the mixture
    parameters.

    Parameters
    ----------
    edges : ArrayLike, shape ``(E,)``
        Pixel edges.
    lsf_fwhm : ArrayLike, shape ``(E,)``
        Instrumental LSF FWHM at each edge.
    center : ArrayLike
        Line center wavelength.
    fwhm_g : ArrayLike
        Intrinsic Gaussian component FWHM.
    fwhm_l : ArrayLike
        Lorentzian component FWHM.

    Returns
    -------
    Array, shape ``(E,)``
        Extended-pseudo-Voigt CDF at each edge.
    """
    fwhm_g_total = _combine_fwhm(lsf_fwhm, fwhm_g)
    return _voigt_ida_cdf(edges - center, fwhm_g_total, fwhm_l)


@jit
def evaluate_voigt(
    wavelength: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
) -> Array:
    """Evaluate a normalised Voigt profile at wavelength points via the Faddeeva function.

    Computes the exact Voigt profile using the Humlicek (1982) W4 rational approximation
    to the Faddeeva function, which is accurate to ~1e-4 relative error across the upper
    half-plane.

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength points at which to evaluate the profile.
    center : ArrayLike
        Line center wavelength.
    lsf_fwhm : ArrayLike
        Instrumental line spread function FWHM at the line center.
    fwhm_g : ArrayLike
        Intrinsic Gaussian component FWHM.
    fwhm_l : ArrayLike
        Lorentzian component FWHM.

    Returns
    -------
    jnp.ndarray
        Normalised profile value at each wavelength point.
    """
    fwhm_g_total = _combine_fwhm(lsf_fwhm, fwhm_g)
    return _voigt_faddeeva_pdf(wavelength - center, fwhm_g_total, fwhm_l)


# -------------------------------------------------------------------
# Skew Voigt kernel  (evaluate only — no analytic integrate)
# -------------------------------------------------------------------


# FXIG2 boost-correction parameters from numerical fit over (lor, alpha, eta) grid.
# log_boost = k * xi^a * eta^b / (1 + q*xi^c) / |alpha|^d
# where xi = gamma/sigma_lsf, eta = sigma_lsf/sigma_g, gamma = fwhm_l/2.
# w0 uses the Thompson FWHM erf scale; see docs/derivations/skew-voigt.md.
_FXIG_K: Final[float] = 0.27045
_FXIG_A: Final[float] = 0.53872
_FXIG_B: Final[float] = 1.0461
_FXIG_C: Final[float] = 1.7778
_FXIG_Q: Final[float] = 1.1286
_FXIG_D: Final[float] = 0.34693


def _alpha_eff(
    lsf_fwhm: ArrayLike, fwhm_g: ArrayLike, fwhm_l: ArrayLike, alpha: ArrayLike
) -> Array:
    """Effective skewness after convolving a skew Voigt with a Gaussian LSF.

    Applies the Gaussian-body exact formula as a base, then multiplies by the
    FXIG2 boost correction that accounts for the Lorentzian component.
    """
    sigma_g = _fwhm_to_sigma(fwhm_g)
    sigma_lsf = _fwhm_to_sigma(lsf_fwhm)
    gamma = fwhm_l / 2

    fwhm_cg = _combine_fwhm(fwhm_g, lsf_fwhm)
    w0 = _thompson_fwhm(fwhm_g, fwhm_l) / _HALFVAR_SIGMA_TO_FWHM
    w0p = _thompson_fwhm(fwhm_cg, fwhm_l) / _HALFVAR_SIGMA_TO_FWHM

    # Gaussian-body exact formula
    a_gauss = alpha * w0 / jnp.sqrt(w0p**2 + 2 * alpha**2 * sigma_lsf**2)

    # FXIG2 boost correction for Lorentzian component
    lor = gamma / (sigma_g + 1e-30)
    eta = sigma_lsf / (sigma_g + 1e-30)
    xi = lor / (eta + 1e-30)
    # Safe |alpha|^d: replace 0 with 1 so the power is finite; a_gauss=0 when alpha=0
    # so the boost value doesn't matter, but NaN must be avoided (0*exp(NaN)=NaN).
    abs_alpha = jnp.abs(alpha)
    safe_alpha = jnp.where(abs_alpha > 0, abs_alpha, jnp.ones_like(abs_alpha))
    log_boost = (
        _FXIG_K
        * xi**_FXIG_A
        * eta**_FXIG_B
        / (1 + _FXIG_Q * xi**_FXIG_C)
        / safe_alpha**_FXIG_D
    )

    return a_gauss * jnp.exp(log_boost)


@jit
def integrate_skewVoigt(
    edges: ArrayLike,
    lsf_fwhm: ArrayLike,
    center: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
    alpha: ArrayLike,
) -> Array:
    """Cumulative skew pseudo-Voigt at edges.

    The skew pseudo-Voigt has no closed-form CDF.  This implementation
    reuses the analytic Ida extended pseudo-Voigt CDF (cheap, exact for
    the symmetric part) for the per-pixel symmetric Voigt fraction, then
    multiplies by the skew correction ``[1 + erf(alpha_eff · (mid - c) / w0)]``
    evaluated at pixel midpoints.  This matches the prior implementation
    semantically (exact symmetric integration, midpoint rule on the skew
    factor only) while fitting the edges-only contract.  Avoiding the
    Faddeeva W4 PDF gives roughly an order-of-magnitude speedup over a
    pure cumsum-midpoint construction.

    Parameters
    ----------
    edges : ArrayLike, shape ``(E,)``
        Pixel edges.
    lsf_fwhm : ArrayLike, shape ``(E,)``
        Instrumental LSF FWHM at each edge.
    center : ArrayLike
        Line center wavelength.
    fwhm_g : ArrayLike
        Intrinsic Gaussian component FWHM.
    fwhm_l : ArrayLike
        Lorentzian component FWHM.
    alpha : ArrayLike
        Skewness parameter; erf scale is w0 = Gamma_V / (2 sqrt(ln 2)).

    Returns
    -------
    Array, shape ``(E,)``
        Cumulative skew pseudo-Voigt array.
    """
    edges_arr = jnp.asarray(edges)
    lsf_arr = jnp.asarray(lsf_fwhm)
    # Analytic per-pixel pseudo-Voigt fractions via Ida CDF diff (per-edge LSF).
    voigt_cum = integrate_voigt(edges_arr, lsf_arr, center, fwhm_g, fwhm_l)
    voigt_per_pixel = jnp.diff(voigt_cum)
    # Skew correction at pixel midpoints.  The skew factor is a second-order
    # asymmetry correction and varies negligibly across an unresolved line, so
    # ``a_eff`` and ``w0`` are computed from a single scalar LSF at the line
    # center (cheap) rather than from per-edge LSF — the per-edge variant
    # multiplies the cost of the polynomial/power-law ``_alpha_eff`` boost
    # correction by the number of edges with no scientifically meaningful gain.
    mids = 0.5 * (edges_arr[1:] + edges_arr[:-1])
    lsf_center = jnp.interp(center, edges_arr, lsf_arr)
    fwhm_g_tot = _combine_fwhm(lsf_center, fwhm_g)
    w0 = _thompson_fwhm(fwhm_g_tot, fwhm_l) / _HALFVAR_SIGMA_TO_FWHM
    a_eff = _alpha_eff(lsf_center, fwhm_g, fwhm_l, alpha)
    skew = _skew(mids - center, a_eff, w0)
    per_pixel = voigt_per_pixel * skew
    return jnp.concatenate([jnp.zeros(1, edges_arr.dtype), jnp.cumsum(per_pixel)])


@jit
def evaluate_skewVoigt(
    wavelength: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
    alpha: ArrayLike,
) -> Array:
    """Evaluate a normalised skew pseudo-Voigt profile at wavelength points.

    Evaluates ``V_pV(x) * [1 + erf(alpha_eff * (x-c) / w0)]`` where ``V_pV``
    is the LSF-convolved pseudo-Voigt (Thompson et al.) and ``alpha_eff`` is
    the effective skewness after convolution, computed via the Gaussian-body
    exact formula with an FXIG boost correction for the Lorentzian component.

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength points at which to evaluate the profile.
    center : ArrayLike
        Line center wavelength.
    lsf_fwhm : ArrayLike
        Instrumental line spread function FWHM.
    fwhm_g : ArrayLike
        Intrinsic Gaussian component FWHM.
    fwhm_l : ArrayLike
        Lorentzian component FWHM.
    alpha : ArrayLike
        Skewness parameter; erf scale is w0 = Gamma_V / (2 sqrt(ln 2)).

    Returns
    -------
    Array
        Normalised profile value at each wavelength point (1/wavelength units).
    """
    voigt_pdf = evaluate_voigt(wavelength, center, lsf_fwhm, fwhm_g, fwhm_l)

    # Get effective skew
    fwhm_g_tot = _combine_fwhm(lsf_fwhm, fwhm_g)
    w0 = _thompson_fwhm(fwhm_g_tot, fwhm_l) / _HALFVAR_SIGMA_TO_FWHM
    a_eff = _alpha_eff(lsf_fwhm, fwhm_g, fwhm_l, alpha)

    skew = _skew(wavelength - center, a_eff, w0)
    return voigt_pdf * skew


# -------------------------------------------------------------------
# Gaussian-Laplace (symmetric EMG) kernel
# -------------------------------------------------------------------

# Threshold for when exp(x*x) overflows
_OVERFLOW_THRESHOLD: Final[float] = (
    26.0 if getattr(config, 'jax_enable_x64', True) else 9.0
)


def _integrandGL(t: ArrayLike, a: ArrayLike) -> Array:
    """
    Antiderivative of the Gaussian-Laplace exponential correction.

    Evaluates exp(-a²) * [exp(2ta) * erfc(a+t) - exp(-2ta) * erfc(a-t)].
    Exploits odd symmetry to cover overflow at large a + t.

    Parameters
    ----------
    t : jnp.ndarray
        Normalized distance from center
    a : jnp.ndarray
        Convolution parameter (sigma *λ/2)

    Returns
    -------
    jnp.ndarray
        Integrand value wi_absth numerical stability
    """
    # Exploit odd symmetry: I(-t,a) = -I(t,a)
    t_abs = jnp.abs(t)
    ta = t_abs + a
    twota = 2 * t_abs * a
    posterm = jnp.where(ta > _OVERFLOW_THRESHOLD, 0, jnp.exp(twota) * erfc(ta))
    return jnp.sign(t) * jnp.exp(a * a) * (posterm - jnp.exp(-twota) * erfc(-t_abs + a))


@jit
def integrate_gaussianLaplace(
    edges: ArrayLike,
    lsf_fwhm: ArrayLike,
    center: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
) -> Array:
    """Cumulative symmetric-EMG CDF (Gaussian ⊛ Laplace) evaluated at edges.

    The LSF is added in quadrature to the Gaussian component **per edge**,
    then the symmetric-EMG CDF is evaluated analytically using a
    numerically stable erfcx form.  See the :doc:`SEMG derivation
    </derivations/semg>` for details.

    Parameters
    ----------
    edges : ArrayLike, shape ``(E,)``
        Pixel edges.
    lsf_fwhm : ArrayLike, shape ``(E,)``
        Instrumental LSF FWHM at each edge.
    center : ArrayLike
        Line center wavelength.
    fwhm_g : ArrayLike
        Intrinsic Gaussian component FWHM.
    fwhm_l : ArrayLike
        Laplace component FWHM (related to scale *b* by ``FWHM = 2 b ln 2``).

    Returns
    -------
    Array, shape ``(E,)``
        Symmetric-EMG CDF at each edge.
    """
    fwhm_g_total = _combine_fwhm(lsf_fwhm, fwhm_g)

    # sigma here is the halfvar-parametrised sigma (= sqrt(2) * standard sigma)
    σ = fwhm_g_total / _HALFVAR_SIGMA_TO_FWHM
    λ = _EXP_SCALE_TO_FWHM / fwhm_l

    # Convolution parameter
    a = 0.5 * σ * λ

    # Gaussian component (CDF at each edge)
    gaussian_cdf = _gaussian_cdf(edges - center, fwhm_g_total)

    # Exponential correction: antiderivative-style integrand at each edge
    t = (edges - center) / σ
    exp_correction = _integrandGL(t, a)

    # Guard against large a, the Gaussian limit
    return gaussian_cdf + jnp.where(a > _OVERFLOW_THRESHOLD, 0.0, 0.25 * exp_correction)


@jit
def evaluate_gaussianLaplace(
    wavelength: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
) -> Array:
    """Evaluate a normalised Gaussian-Laplace (SEMG) profile at wavelength points.

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength points at which to evaluate the profile.
    center : ArrayLike
        Line center wavelength.
    lsf_fwhm : ArrayLike
        Instrumental line spread function FWHM at the line center.
    fwhm_g : ArrayLike
        Intrinsic Gaussian component FWHM.
    fwhm_l : ArrayLike
        Laplacian component FWHM.

    Returns
    -------
    jnp.ndarray
        Normalised profile value at each wavelength point.
    """
    fwhm_g_total = _combine_fwhm(lsf_fwhm, fwhm_g)
    sigma = _fwhm_to_sigma(fwhm_g_total)
    lam = _EXP_SCALE_TO_FWHM / fwhm_l

    # Symmetric EMG PDF: Gaussian convolved wi_absth symmetric Laplace.
    # f(x) = (lam/4) * exp(a^2) * [exp(2|u|a)*erfc(|u|+a) + exp(-2|u|a)*erfc(a-|u|)]
    # where u = (x - center) / (sigma * sqrt(2)), a = lam * sigma / sqrt(2).
    # Uses the same overflow-protection pattern as _integrandGL.
    t = (wavelength - center) / sigma
    a = lam * sigma / np.sqrt(2.0)
    u_abs = jnp.abs(t) / np.sqrt(2.0)
    ua = u_abs + a
    two_ua = 2 * u_abs * a

    # Overflow-protected positive term: exp(2|u|a) * erfc(|u| + a)
    posterm = jnp.where(ua > _OVERFLOW_THRESHOLD, 0.0, jnp.exp(two_ua) * erfc(ua))
    negterm = jnp.exp(-two_ua) * erfc(a - u_abs)
    result = (0.25 * lam) * jnp.exp(a * a) * (posterm + negterm)

    # Pure Gaussian limit when exponential component is negligible
    return jnp.where(
        a > _OVERFLOW_THRESHOLD, _gaussian_pdf(wavelength - center, sigma), result
    )


# -------------------------------------------------------------------
# Gaussian-SplitLaplace (asymmetric EMG) kernel
# -------------------------------------------------------------------


def _integrandGSL(t: ArrayLike, a: ArrayLike) -> Array:
    """
    Antiderivative of the Gaussian-SplitLaplace exponential correction.

    Parameters
    ----------
    t : jnp.ndarray
        Normalized distance from center
    a : jnp.ndarray
        Convolution parameter (sigma * lambda / 2)

    Returns
    -------
    jnp.ndarray
        Integrand value with numerical stability
    """
    u = a * (a - 2 * t)
    return jnp.where(u > _OVERFLOW_THRESHOLD**2, 0.0, jnp.exp(u) * erfc(-t + a))


@jit
def integrate_gaussianSplitLaplace(
    edges: ArrayLike,
    lsf_fwhm: ArrayLike,
    center: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l_blue: ArrayLike,
    fwhm_l_red: ArrayLike,
) -> Array:
    """Cumulative asymmetric-EMG CDF (Gaussian ⊛ split-Laplace) evaluated at edges.

    The LSF is added in quadrature to the Gaussian component **per edge**,
    then the asymmetric-EMG CDF is evaluated analytically using a
    numerically stable erfcx form.

    Parameters
    ----------
    edges : ArrayLike, shape ``(E,)``
        Pixel edges.
    lsf_fwhm : ArrayLike, shape ``(E,)``
        Instrumental LSF FWHM at each edge.
    center : ArrayLike
        Line center wavelength.
    fwhm_g : ArrayLike
        Intrinsic Gaussian component FWHM.
    fwhm_l_blue : ArrayLike
        Blue-side Laplace component FWHM.
    fwhm_l_red : ArrayLike
        Red-side Laplace component FWHM.

    Returns
    -------
    Array, shape ``(E,)``
        Asymmetric-EMG CDF at each edge.
    """
    fwhm_g_total = _combine_fwhm(lsf_fwhm, fwhm_g)

    σ = fwhm_g_total / _HALFVAR_SIGMA_TO_FWHM
    λ_red = _EXP_SCALE_TO_FWHM / fwhm_l_red
    λ_blue = _EXP_SCALE_TO_FWHM / fwhm_l_blue

    # Convolution parameters for each side
    a_red = σ * λ_red / 2
    a_blue = σ * λ_blue / 2

    # Gaussian component (CDF at each edge)
    gaussian_cdf = _gaussian_cdf(edges - center, fwhm_g_total)

    # Exponential correction antiderivatives at each edge
    t = (edges - center) / σ
    red_t = _integrandGSL(t, a_red)
    blue_t = _integrandGSL(-t, a_blue)

    exp_correction = (fwhm_l_red * red_t - fwhm_l_blue * blue_t) / 2

    return gaussian_cdf - exp_correction / (fwhm_l_red + fwhm_l_blue)


@jit
def evaluate_gaussianSplitLaplace(
    wavelength: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l_blue: ArrayLike,
    fwhm_l_red: ArrayLike,
) -> Array:
    """Evaluate a normalised Gaussian-SplitLaplace profile at wavelength points.

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength points at which to evaluate the profile.
    center : ArrayLike
        Line center wavelength.
    lsf_fwhm : ArrayLike
        Instrumental line spread function FWHM at the line center.
    fwhm_g : ArrayLike
        Intrinsic Gaussian component FWHM.
    fwhm_l_r : ArrayLike
        Right-side Laplacian component FWHM.
    fwhm_l_l : ArrayLike
        Left-side Laplacian component FWHM.

    Returns
    -------
    jnp.ndarray
        Normalised profile value at each wavelength point.
    """
    fwhm_g_total = _combine_fwhm(lsf_fwhm, fwhm_g)
    sigma = fwhm_g_total / _HALFVAR_SIGMA_TO_FWHM
    lam_red = _EXP_SCALE_TO_FWHM / fwhm_l_red
    lam_blue = _EXP_SCALE_TO_FWHM / fwhm_l_blue

    t = (wavelength - center) / (sigma)
    a_red = lam_red * sigma / 2
    a_blue = lam_blue * sigma / 2

    u_right = a_red * (a_red - 2 * t)
    u_left = a_blue * (a_blue + 2 * t)

    right = jnp.where(
        u_right > _OVERFLOW_THRESHOLD**2, 0.0, jnp.exp(u_right) * erfc(-t + a_red)
    )
    left = jnp.where(
        u_left > _OVERFLOW_THRESHOLD**2, 0.0, jnp.exp(u_left) * erfc(t + a_blue)
    )

    return lam_red * lam_blue * (right + left) / (2 * (lam_red + lam_blue))
