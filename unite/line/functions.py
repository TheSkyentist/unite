"""JAX-jitted line profile integration and evaluation kernels.

Integration kernels (``integrate_*``) compute the fraction of a normalized
profile that falls within each wavelength bin ``[low, high]``.  Evaluation
kernels (``evaluate_*``) compute the normalized profile value (probability
density) at arbitrary wavelength points.

All functions are pure JAX with no numpyro dependency and are designed to be
called from within :func:`jax.jit`-compiled model code.
"""

from __future__ import annotations

from typing import Final

import jax.numpy as jnp
from jax import Array, config, jit
from jax.scipy.special import erf, erfc
from jax.typing import ArrayLike

# Conversion: FWHM to sigma for the half-variance parametrization of erf.
# sigma = FWHM / (2 sqrt(2 ln 2)); erf uses sqrt(2)*sigma, so the factor is:
_HALFVAR_SIGMA_TO_FWHM: Final[Array] = 2 * jnp.sqrt(jnp.log(2))

# Pseudo-Voigt polynomial coefficients from Thompson et al. (1987)
# DOI: 10.1107/S0021889887087090
_VOIGT_FWHM_CS: Final[Array] = jnp.array([1, 2.69268, 2.42843, 4.47163, 0.07842, 1])
_VOIGT_ETA_CS: Final[Array] = jnp.array([1.33603, -0.47719, 0.11116])

# Threshold for when exp(x*x) overflows
_OVERFLOW_THRESHOLD: Final[float] = 26.0 if config.jax_enable_x64 else 9.0

# Conversion factor from exponential (Laplace) scale to FWHM
# pdf = (1/(2*b)) * exp(-|x - μ|/b)
# max(pdf) = 1/(2*b), half max = 1/(4*b)
# 1/(4*b) = 1/(2*b) * exp(-|x - μ|/b) => exp(-|x - μ|/b) = 1/2
# => |x - μ|/b = ln(2) => FWHM = 2*b*ln(2)
_EXP_SCALE_TO_FWHM: Final[Array] = 2 * jnp.log(2)

# Precompute constants
_INV_SQRT2PI: Final[Array] = 1.0 / jnp.sqrt(2 * jnp.pi)
_SQRT6: Final[Array] = jnp.sqrt(6.0)
_SQRT24: Final[Array] = jnp.sqrt(24.0)


@jit
def integrate_gaussian(
    low: ArrayLike,
    high: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm: ArrayLike,
) -> Array:
    """Integrate a Gaussian profile over wavelength bins.

    Parameters
    ----------
    low : ArrayLike
        Lower bin edges.
    high : ArrayLike
        Upper bin edges.
    center : ArrayLike
        Line center wavelength.
    lsf_fwhm : ArrayLike
        Instrumental line spread function FWHM at the line center.
    fwhm : ArrayLike
        Intrinsic Gaussian FWHM.

    Returns
    -------
    jnp.ndarray
        Integrated fraction per bin.
    """
    # Add LSF in quadrature
    total_fwhm = jnp.sqrt(lsf_fwhm**2 + fwhm**2)
    inv_sigma = _HALFVAR_SIGMA_TO_FWHM / total_fwhm
    t_low = (low - center) * inv_sigma
    t_high = (high - center) * inv_sigma
    return (erf(t_high) - erf(t_low)) / 2


def _integrate_cauchy(
    low: ArrayLike,
    high: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm: ArrayLike,
) -> Array:
    """Private: Integrate a pure Cauchy (Lorentzian) profile over wavelength bins.

    This function is kept for reference but is no longer used directly.
    Cauchy profiles are now implemented as PseudoVoigt with LSF=0.

    Parameters
    ----------
    low : ArrayLike
        Lower bin edges.
    high : ArrayLike
        Upper bin edges.
    center : ArrayLike
        Line center wavelength.
    lsf_fwhm : ArrayLike
        Instrumental line spread function FWHM at the line center.
    fwhm : ArrayLike
        Intrinsic Lorentzian FWHM.

    Returns
    -------
    jnp.ndarray
        Integrated fraction per bin.
    """
    # For Cauchy, we don't convolve with LSF - it's a pure Lorentzian
    # This function is kept for reference but no longer used
    inv_hwhm = 2 / fwhm
    t_low = (low - center) * inv_hwhm
    t_high = (high - center) * inv_hwhm
    return (jnp.arctan(t_high) - jnp.arctan(t_low)) / jnp.pi


@jit
def integrate_voigt(
    low: ArrayLike,
    high: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
) -> Array:
    """Integrate a pseudo-Voigt profile over wavelength bins.

    Uses the Thompson et al. (1987) approximation: a weighted sum of
    Gaussian and Lorentzian components with an effective FWHM computed
    from the individual Gaussian and Lorentzian fwhms.

    Parameters
    ----------
    low : ArrayLike
        Lower bin edges.
    high : ArrayLike
        Upper bin edges.
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
        Integrated fraction per bin.

    References
    ----------
    Thompson, Cox & Hastings (1987), J. Appl. Cryst. 20, 79-83.
    DOI: 10.1107/S0021889887087090
    """
    # Add LSF in quadrature to Gaussian component only
    fwhm_g_tot = jnp.sqrt(lsf_fwhm**2 + fwhm_g**2)

    # Effective FWHM from the 5th-order polynomial
    pows = jnp.arange(_VOIGT_FWHM_CS.size)
    fwhm = jnp.sum(_VOIGT_FWHM_CS * (fwhm_g_tot**pows) * (fwhm_l ** pows[::-1])) ** (
        1 / 5
    )

    # Lorentzian mixing fraction η
    fwhm_ratio = fwhm_l / fwhm
    eta = jnp.sum(_VOIGT_ETA_CS * (fwhm_ratio ** jnp.arange(1, len(_VOIGT_ETA_CS) + 1)))

    # Use effective FWHM directly — LSF is already incorporated via fwhm_g_tot.
    # Pass lsf_fwhm=0 to avoid double-counting in integrate_gaussian.
    lorentzian = eta * _integrate_cauchy(low, high, center, lsf_fwhm, fwhm)
    gaussian = (1 - eta) * integrate_gaussian(low, high, center, 0.0, fwhm)
    return lorentzian + gaussian


# Don't need this since we always convolve with LSF
# But keeping anyways just in case
@jit
def _integrate_laplace(
    low: ArrayLike,
    high: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm: ArrayLike,
) -> Array:
    """Integrate a Laplace (double-exponential) profile over wavelength bins.

    The LSF is **not** convolved --- this profile is a pure Laplace
    distribution.

    Parameters
    ----------
    low : ArrayLike
        Lower bin edges.
    high : ArrayLike
        Upper bin edges.
    center : ArrayLike
        Line center wavelength.
    lsf_fwhm : ArrayLike
        Instrumental line spread function FWHM at the line center.
    fwhm : ArrayLike
        Laplace FWHM (related to scale parameter *b* by
        ``FWHM = 2 b ln 2``).

    Returns
    -------
    jnp.ndarray
        Integrated fraction per bin.
    """
    # For Laplace, we don't convolve with LSF - it's a pure Laplace distribution
    b = fwhm / _EXP_SCALE_TO_FWHM

    def _cdf(x):
        t = (x - center) / b
        return 0.5 + 0.5 * jnp.sign(t) * (1.0 - jnp.exp(-jnp.abs(t)))

    return _cdf(high) - _cdf(low)


def _integrandGL(t: ArrayLike, a: ArrayLike) -> Array:
    """
    Compute exponential correction integrand for Gaussian-Laplace convolution.

    Evaluates exp(-a²) * [exp(2ta) * erfcx(a+t) - exp(-2ta) * erfcx(a-t)].
    Exploits odd symmetry to cover overflow at large a + t.

    Parameters
    ----------
    t : jnp.ndarray
        Normalized distance from center
    a : jnp.ndarray
        Convolution parameter (σλ/2)

    Returns
    -------
    jnp.ndarray
        Integrand value with numerical stability
    """
    # Exploit odd symmetry: I(-t,a) = -I(t,a)
    # Then we only need one overflow protection
    t_abs = jnp.abs(t)

    # Shorthand terms
    ta = t_abs + a
    twota = 2 * t_abs * a

    # Overflow protection
    posterm = jnp.where(ta > _OVERFLOW_THRESHOLD, 0, jnp.exp(twota) * erfc(ta))

    return jnp.sign(t) * jnp.exp(a * a) * (posterm - jnp.exp(-twota) * erfc(-t_abs + a))


@jit
def integrate_gaussianLaplace(
    low: ArrayLike,
    high: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
) -> Array:
    """
    Integrate exponentially modified Gaussian (EMG) emission lines over wavelength bins.

    Computes the integral of Gaussian convolved with symmetric double exponential
    (Laplace distribution). This profile combines instrumental broadening (Gaussian)
    with natural broadening (Laplacian). Uses asymptotic approximations for
    numerical stability.

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins
    high : jnp.ndarray
        High wavelength edges of bins
    center : jnp.ndarray
        Line centers
    lsf_fwhm : jnp.ndarray
        Instrumental LSF FWHM
    fwhm_g : jnp.ndarray
        Intrinsic Gaussian component FWHM
    fwhm_l : jnp.ndarray
        Laplacian component FWHM (natural broadening)

    Returns
    -------
    jnp.ndarray
        Integrated flux in each bin for each line
    """
    # This function breaks for pure Laplace, does that matter? lsf is never zero?

    # Add LSF in quadrature to Gaussian component
    fwhm_g_total = jnp.sqrt(lsf_fwhm**2 + fwhm_g**2)

    # Convert FWHM to distribution parameters
    σ = fwhm_g_total / _HALFVAR_SIGMA_TO_FWHM
    λ = _EXP_SCALE_TO_FWHM / fwhm_l

    # Normalized distance from center
    scale = 1 / σ
    t_low = (low - center) * scale
    t_high = (high - center) * scale

    # Convolution parameter
    a = σ * λ / 2

    # Gaussian component (via error function CDF)
    gaussian_cdf = (erf(t_high) - erf(t_low)) / 2

    # Exponential correction
    exp_correction = _integrandGL(t_high, a) - _integrandGL(t_low, a)

    # Guard against large a, the Gaussian limit
    return gaussian_cdf + jnp.where(a > _OVERFLOW_THRESHOLD, 0, exp_correction / 4)


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
    t = t_halfvar * jnp.sqrt(2.0)
    g = jnp.exp(-t * t / 2)
    # He_2(y) = y²-1,  He_3(y) = y(y²-3)
    return g * (h3_eff * (t * t - 1) + h4_eff * t * (t * t - 3))


@jit
def integrate_split_normal(
    low: ArrayLike,
    high: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_blue: ArrayLike,
    fwhm_red: ArrayLike,
) -> Array:
    """
    Integrate a split-normal (two-sided Gaussian) profile over wavelength bins.

    The split-normal distribution has different standard deviations on each side
    of the mean. The left side (blue, shorter wavelengths) uses fwhm_blue, and
    the right side (red, longer wavelengths) uses fwhm_red.

    Parameters
    ----------
    low : ArrayLike
        Lower bin edges.
    high : ArrayLike
        Upper bin edges.
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
        Integrated fraction per bin.
    """
    # Add LSF in quadrature to both components
    total_fwhm_blue = jnp.sqrt(lsf_fwhm**2 + fwhm_blue**2)
    total_fwhm_red = jnp.sqrt(lsf_fwhm**2 + fwhm_red**2)

    # Convert FWHM to sigma for the half-variance parametrization of erf
    inv_sigma_blue = _HALFVAR_SIGMA_TO_FWHM / total_fwhm_blue
    inv_sigma_red = _HALFVAR_SIGMA_TO_FWHM / total_fwhm_red

    # Calculate normalization factor to ensure the total integral is 1
    # The split-normal PDF is proportional to:
    #   exp(-(x-center)^2/(2*sigma_blue^2)) for x <= center
    #   exp(-(x-center)^2/(2*sigma_red^2)) for x > center
    # The normalization constant is: 1 / [sqrt(2*pi) * (sigma_blue + sigma_red) / 2]
    # But since we're working with CDFs, we need a different approach

    # Probability mass on each side: proportional to sigma (not inv_sigma)
    total_weight = inv_sigma_blue + inv_sigma_red
    w_blue = inv_sigma_red / total_weight  # = sigma_blue / (sigma_blue + sigma_red)
    w_red = inv_sigma_blue / total_weight  # = sigma_red / (sigma_blue + sigma_red)

    def _split_normal_cdf(x, center, inv_sigma_blue, inv_sigma_red, w_blue, w_red):
        t_blue = (x - center) * inv_sigma_blue
        t_red = (x - center) * inv_sigma_red

        # CDF continuous at center: both branches give w_blue when x = center
        return jnp.where(
            x <= center, w_blue * (1 + erf(t_blue)), w_blue + w_red * erf(t_red)
        )

    # Calculate CDF at high and low edges
    cdf_high = _split_normal_cdf(
        high, center, inv_sigma_blue, inv_sigma_red, w_blue, w_red
    )
    cdf_low = _split_normal_cdf(
        low, center, inv_sigma_blue, inv_sigma_red, w_blue, w_red
    )

    return cdf_high - cdf_low


@jit
def integrate_gaussHermite(
    low: ArrayLike,
    high: ArrayLike,
    center: ArrayLike,
    fwhm_lsf: ArrayLike,
    fwhm_g: ArrayLike,
    h3: ArrayLike,
    h4: ArrayLike,
) -> Array:
    """
    Integrate Gauss-Hermite emission lines over wavelength bins.

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins
    high : jnp.ndarray
        High wavelength edges of bins
    center : jnp.ndarray
        Line centers
    fwhm_lsf : jnp.ndarray
        Instrumental LSF FWHM
    fwhm_g : jnp.ndarray
        Gaussian component FWHM
    h3 : jnp.ndarray
        Gauss-Hermite h3 coefficient
    h4 : jnp.ndarray
        Gauss-Hermite h4 coefficient

    Returns
    -------
    jnp.ndarray
        Integrated flux in each bin for each line
    """
    # Effective FWHM from convolution of LSF and Gaussian component
    fwhm_tot = jnp.sqrt(fwhm_lsf**2 + fwhm_g**2)

    # Convert FWHM to sigma
    inv_sigma_tot = _HALFVAR_SIGMA_TO_FWHM / fwhm_tot

    # Sigma ratio: GH moments scale as r^n under convolution with a Gaussian LSF
    r = fwhm_g / fwhm_tot
    r3 = r * r * r

    # Normalized bin edges
    t_low = (low - center) * inv_sigma_tot
    t_high = (high - center) * inv_sigma_tot

    # Gaussian CDF contribution
    gaussian_cdf = (erf(t_high) - erf(t_low)) / 2

    # GH coefficients scaled by sigma ratio (moment theorem for Gaussian convolution)
    c3 = h3 * r3 / _SQRT6
    c4 = h4 * r3 * r / _SQRT24

    # GH correction via analytic antiderivative
    gh_correction = _integrandGH(t_high, c3, c4) - _integrandGH(t_low, c3, c4)

    return gaussian_cdf - _INV_SQRT2PI * gh_correction


# _integrate_single_line and integrate_lines live in unite.line.profiles,
# where each Profile subclass owns its JAX branch via integrate_branch().
# _evaluate_single_line and evaluate_lines also live there, dispatching
# via evaluate_branch().


# -------------------------------------------------------------------
# Pointwise evaluation kernels
# -------------------------------------------------------------------
#
# Each evaluate_* function returns the normalised profile value
# (probability density in 1/wavelength units) at arbitrary wavelength
# points.  Integrating over all wavelength recovers unity.


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
    total_fwhm = jnp.sqrt(lsf_fwhm**2 + fwhm**2)
    sigma = total_fwhm / (_HALFVAR_SIGMA_TO_FWHM * jnp.sqrt(2.0))
    t = (wavelength - center) / sigma
    return jnp.exp(-0.5 * t**2) / (sigma * jnp.sqrt(2.0 * jnp.pi))


@jit
def evaluate_voigt(
    wavelength: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
) -> Array:
    """Evaluate a normalised pseudo-Voigt profile at wavelength points.

    Uses the Thompson et al. (1987) approximation: a weighted sum of
    Gaussian and Lorentzian components with an effective FWHM.

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
    # Add LSF in quadrature to Gaussian component only
    fwhm_g_tot = jnp.sqrt(lsf_fwhm**2 + fwhm_g**2)

    # Effective FWHM from the 5th-order polynomial
    pows = jnp.arange(_VOIGT_FWHM_CS.size)
    fwhm = jnp.sum(_VOIGT_FWHM_CS * (fwhm_g_tot**pows) * (fwhm_l ** pows[::-1])) ** (
        1 / 5
    )

    # Lorentzian mixing fraction η
    fwhm_ratio = fwhm_l / fwhm
    eta = jnp.sum(_VOIGT_ETA_CS * (fwhm_ratio ** jnp.arange(1, len(_VOIGT_ETA_CS) + 1)))

    # Gaussian PDF with effective FWHM
    sigma_eff = fwhm / (_HALFVAR_SIGMA_TO_FWHM * jnp.sqrt(2.0))
    t = (wavelength - center) / sigma_eff
    gauss_pdf = jnp.exp(-0.5 * t**2) / (sigma_eff * jnp.sqrt(2.0 * jnp.pi))

    # Lorentzian (Cauchy) PDF with effective FWHM
    hwhm = fwhm / 2.0
    lorentz_pdf = (hwhm / jnp.pi) / ((wavelength - center) ** 2 + hwhm**2)

    return (1 - eta) * gauss_pdf + eta * lorentz_pdf


@jit
def evaluate_gaussianLaplace(
    wavelength: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
) -> Array:
    """Evaluate a normalised Gaussian-Laplace (EMG) profile at wavelength points.

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
    # Add LSF in quadrature to Gaussian component
    fwhm_g_total = jnp.sqrt(lsf_fwhm**2 + fwhm_g**2)

    # Convert FWHM to distribution parameters
    sigma = fwhm_g_total / (_HALFVAR_SIGMA_TO_FWHM * jnp.sqrt(2.0))
    lam = _EXP_SCALE_TO_FWHM / fwhm_l

    # Symmetric EMG PDF: Gaussian convolved with symmetric Laplace.
    # f(x) = (lam/4) * exp(a^2) * [exp(2|u|a)*erfc(|u|+a) + exp(-2|u|a)*erfc(a-|u|)]
    # where u = (x - center) / (sigma * sqrt(2)), a = lam * sigma / sqrt(2).
    # Uses the same overflow-protection pattern as _integrandGL.
    t = (wavelength - center) / sigma
    a = lam * sigma / jnp.sqrt(2.0)

    u_abs = jnp.abs(t) / jnp.sqrt(2.0)
    ua = u_abs + a
    two_ua = 2 * u_abs * a

    # Overflow-protected positive term: exp(2|u|a) * erfc(|u| + a)
    posterm = jnp.where(ua > _OVERFLOW_THRESHOLD, 0.0, jnp.exp(two_ua) * erfc(ua))

    # Negative term (always numerically safe)
    negterm = jnp.exp(-two_ua) * erfc(a - u_abs)

    result = (lam / 4.0) * jnp.exp(a * a) * (posterm + negterm)

    # Pure Gaussian limit when exponential component is negligible
    gauss_pdf = jnp.exp(-0.5 * t**2) / (sigma * jnp.sqrt(2.0 * jnp.pi))
    return jnp.where(a > _OVERFLOW_THRESHOLD, gauss_pdf, result)


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
    # Add LSF in quadrature to both components
    total_fwhm_blue = jnp.sqrt(lsf_fwhm**2 + fwhm_blue**2)
    total_fwhm_red = jnp.sqrt(lsf_fwhm**2 + fwhm_red**2)

    sigma_blue = total_fwhm_blue / (_HALFVAR_SIGMA_TO_FWHM * jnp.sqrt(2.0))
    sigma_red = total_fwhm_red / (_HALFVAR_SIGMA_TO_FWHM * jnp.sqrt(2.0))

    # Normalisation: integral = sqrt(pi/2) * (sigma_blue + sigma_red)
    norm = jnp.sqrt(jnp.pi / 2.0) * (sigma_blue + sigma_red)

    dx = wavelength - center
    val_blue = jnp.exp(-0.5 * (dx / sigma_blue) ** 2)
    val_red = jnp.exp(-0.5 * (dx / sigma_red) ** 2)

    return jnp.where(wavelength <= center, val_blue, val_red) / norm


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
    # Effective FWHM from convolution
    fwhm_tot = jnp.sqrt(fwhm_lsf**2 + fwhm_g**2)
    sigma_tot = fwhm_tot / (_HALFVAR_SIGMA_TO_FWHM * jnp.sqrt(2.0))

    # Sigma ratio: GH moments scale as r^n under convolution
    r = fwhm_g / fwhm_tot
    r3 = r * r * r

    # Standard coordinate
    y = (wavelength - center) / sigma_tot

    # Base Gaussian PDF
    gauss_pdf = jnp.exp(-0.5 * y**2) / (sigma_tot * jnp.sqrt(2.0 * jnp.pi))

    # Hermite polynomial corrections (probabilists' convention)
    # He_3(y) = y^3 - 3y,  He_4(y) = y^4 - 6y^2 + 3
    he3 = y**3 - 3 * y
    he4 = y**4 - 6 * y**2 + 3

    # Scale by sigma ratio
    c3 = h3 * r3 / _SQRT6
    c4 = h4 * r3 * r / _SQRT24

    return gauss_pdf * (1.0 + c3 * he3 + c4 * he4)
