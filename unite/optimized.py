"""
Optimized line profile integrals
"""

# Typing
from typing import Final

# JAX packages
from jax.scipy.special import erf, erfc
from jax import config, jit, vmap, lax, numpy as jnp

# Conversion factor from FWHM to sigma for variance = 1/2
# σ = fwhm / ( 2 * sqrt( 2 * ln(2) ) )
# σ_halfvar = sqrt(2) * σ
_HALFVAR_SIGMA_TO_FWHM: Final[float] = 2 * jnp.sqrt(jnp.log(2))


@jit
def integrateGaussian(
    low: jnp.ndarray, high: jnp.ndarray, center: jnp.ndarray, fwhm: jnp.ndarray
) -> jnp.ndarray:
    """
    Integrate Gaussian emission lines over wavelength bins.

    Computes the integral of N Gaussian profiles over wavelength bins using
    the error function (Gaussian CDF).

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins
    high : jnp.ndarray
        High wavelength edges of bins
    center : jnp.ndarray
        Line centers
    fwhm : jnp.ndarray
        Gaussian FWHM at each line

    Returns
    -------
    jnp.ndarray
        Integrated flux in each bin for each line
    """
    # Convert FWHM to inverse sigma (variance = 1/2)
    inv_sigma = _HALFVAR_SIGMA_TO_FWHM / fwhm

    # Normalized distance from center
    t_low = (low - center) * inv_sigma
    t_high = (high - center) * inv_sigma

    # Integrate using Gaussian CDF (error function)
    return (erf(t_high) - erf(t_low)) / 2


@jit
def integrateCauchy(
    low: jnp.ndarray, high: jnp.ndarray, center: jnp.ndarray, fwhm: jnp.ndarray
) -> jnp.ndarray:
    """
    Integrate Cauchy (Lorentzian) emission lines over wavelength bins.

    Computes the integral of N Cauchy/Lorentzian profiles over wavelength bins
    using the arctan function (Cauchy CDF).

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins
    high : jnp.ndarray
        High wavelength edges of bins
    center : jnp.ndarray
        Line centers
    fwhm : jnp.ndarray
        Lorentzian FWHM at each line

    Returns
    -------
    jnp.ndarray
        Integrated flux in each bin for each line
    """
    # Convert FWHM to inverse half-width
    inv_hwhm = 2 / fwhm

    # Normalized distance from center
    t_low = (low - center) * inv_hwhm
    t_high = (high - center) * inv_hwhm

    # Integrate using Cauchy CDF (arctan)
    return (jnp.arctan(t_high) - jnp.arctan(t_low)) / jnp.pi


# Pseudo-Voigt profile magic numbers from Thompson+ (1987) DOI:10.1107/S0021889887087090
_VOIGT_FWHM_CS: Final[jnp.ndarray] = jnp.array(
    [1, 2.69268, 2.42843, 4.47163, 0.07842, 1]
)
_VOIGT_ETA_CS: Final[jnp.ndarray] = jnp.array([1.33603, -0.47719, 0.11116])


@jit
def integrateVoigt(
    low: jnp.ndarray,
    high: jnp.ndarray,
    center: jnp.ndarray,
    fwhm_g: jnp.ndarray,
    fwhm_γ: jnp.ndarray,
) -> jnp.ndarray:
    """
    Integrate Voigt emission lines over wavelength bins.

    Computes the integral of N Voigt profiles (Gaussian convolved with Lorentzian)
    using the pseudo-Voigt approximation from Thompson+ (1987).
    DOI:10.1107/S0021889887087090

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins
    high : jnp.ndarray
        High wavelength edges of bins
    center : jnp.ndarray
        Line centers
    fwhm_g : jnp.ndarray
        Gaussian component FWHM at each line
    fwhm_γ : jnp.ndarray
        Lorentzian component FWHM at each line

    Returns
    -------
    jnp.ndarray
        Integrated flux in each bin for each line
    """
    # Compute effective FWHM for pseudo-Voigt
    pows = jnp.arange(_VOIGT_FWHM_CS.size)
    fwhm = jnp.sum(_VOIGT_FWHM_CS * (fwhm_g**pows) * (fwhm_γ ** pows[::-1])) ** (1 / 5)

    # Compute Lorentzian mixing parameter η
    fwhm_ratio = fwhm_γ / fwhm
    η = jnp.sum(_VOIGT_ETA_CS * (fwhm_ratio ** jnp.arange(1, len(_VOIGT_ETA_CS) + 1)))

    # Weighted sum of Lorentzian and Gaussian components
    lorentzian = η * integrateCauchy(low, high, center, fwhm)
    gaussian = (1 - η) * integrateGaussian(low, high, center, fwhm)

    return lorentzian + gaussian


# Conversion factor from exponential (Laplace) scale to FWHM
# pdf = (1/(2*b)) * exp(-|x - μ|/b)
# max(pdf) = 1/(2*b), half max = 1/(4*b)
# 1/(4*b) = 1/(2*b) * exp(-|x - μ|/b) => exp(-|x - μ|/b) = 1/2
# => |x - μ|/b = ln(2) => FWHM = 2*b*ln(2)
_EXP_SCALE_TO_FWHM: Final[float] = 2 * jnp.log(2)


@jit
def integrateLaplace(
    low: jnp.ndarray, high: jnp.ndarray, center: jnp.ndarray, fwhm: jnp.ndarray
) -> jnp.ndarray:
    """
    Integrate Laplace (double exponential) emission lines over wavelength bins.

    Computes the integral of N Laplace/double exponential profiles over wavelength
    bins using the exponential CDF.

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins
    high : jnp.ndarray
        High wavelength edges of bins
    center : jnp.ndarray
        Line centers
    fwhm : jnp.ndarray
        Laplace FWHM at each line

    Returns
    -------
    jnp.ndarray
        Integrated flux in each bin for each line
    """
    # Convert FWHM to scale parameter
    λ = _EXP_SCALE_TO_FWHM / fwhm

    # Normalized distance from center
    t_low = (low - center) * λ
    t_high = (high - center) * λ

    # Laplace CDF: F(t) = 1/2 + 1/2 * sign(t) * (1 - exp(-|t|))
    @jit
    def laplace_cdf(t):
        return jnp.sign(t) * (1 - jnp.exp(-jnp.abs(t)))

    return (laplace_cdf(t_high) - laplace_cdf(t_low)) / 2


# Threshold for when exp(x*x) overflows
_OVERFLOW_THRESHOLD: Final[float] = 26.0 if config.jax_enable_x64 else 9.0


@jit
def _integrandGL(t: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
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
def integrateGaussianLaplace(
    low: jnp.ndarray,
    high: jnp.ndarray,
    center: jnp.ndarray,
    fwhm_g: jnp.ndarray,
    fwhm_l: jnp.ndarray,
) -> jnp.ndarray:
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
    fwhm_g : jnp.ndarray
        Gaussian component FWHM (instrumental broadening)
    fwhm_l : jnp.ndarray
        Laplacian component FWHM (natural broadening)

    Returns
    -------
    jnp.ndarray
        Integrated flux in each bin for each line
    """
    # This function breaks for pure Laplace, does that matter? lsf is never zero?

    # Convert FWHM to distribution parameters
    σ = fwhm_g / _HALFVAR_SIGMA_TO_FWHM
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


@jit
def integrateCond(
    low: jnp.ndarray,
    high: jnp.ndarray,
    center: jnp.ndarray,
    lsf: jnp.ndarray,
    fwhm: jnp.ndarray,
    is_voigt: jnp.ndarray,
) -> jnp.ndarray:
    """
    Integrate a single emission line profile (either Voigt or Gaussian)
    depending on whether the line is broad.

    Parameters
    ----------
    low : jnp.ndarray
        Low edge of the bin
    high : jnp.ndarray
        High edge of the bin
    center : jnp.ndarray
        Center of the emission line
    lsf : jnp.ndarray
        Line spread function
    fwhm : jnp.ndarray
        Full width at half maximum
    is_voigt : jnp.ndarray
        Whether the line is Voigt

    Returns
    -------
    jnp.ndarray
        Integral across wavelenths
    """
    return lax.cond(
        is_voigt,
        lambda _: integrateVoigt(low, high, center, lsf, fwhm),
        lambda _: integrateGaussian(
            low, high, center, jnp.sqrt(lsf * lsf + fwhm * fwhm)
        ),
        operand=None,  # No extra operand needed
    )


@jit
def integrate(
    low: jnp.ndarray,
    high: jnp.ndarray,
    cent: jnp.ndarray,
    lsf: jnp.ndarray,
    fwhm: jnp.ndarray,
    is_voigt: jnp.ndarray,
) -> jnp.ndarray:
    """
    Integrate N emission lines over λ bins.
    Returns a matrix of integrals in each bin for each line.
    Uses Voigt profile if broad, otherwise Gaussian.

    Parameters
    ----------
    low : jnp.ndarray
        Low edge of the bin
    high : jnp.ndarray
        High edge of the bin
    center : jnp.ndarray
        Center of the emission line
    lsf : jnp.ndarray
        Line spread function
    fwfm : jnp.ndarray
        Full width at half maximum
    is_broad : jnp.ndarray
        Whether the line is broad

    Returns
    -------
    jnp.ndarray
        Integral across wavelenths
    """
    # Vectorize the integration across the lines
    vectorized_integrate = vmap(integrateCond, in_axes=(None, None, 0, 0, 0, 0))

    # Perform the integration for all lines
    return vectorized_integrate(low, high, cent, lsf, fwhm, is_voigt)


@jit
def linearContinua(
    λ: jnp.ndarray,
    cont_center: jnp.ndarray,
    angles: jnp.ndarray,
    offsets: jnp.ndarray,
    continuum_regions: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the linear model

    Parameters
    ----------
    λ : jnp.ndarray
        Wavelength values
    cont_center : jnp.ndarray
        Centers of the continua
    angles : jnp.ndarray
        Angles of the continua
    offsets : jnp.ndarray
        Offset of the continua
    continuum_regions : jnp.ndarray
        Bounds of the continuum region

    Returns
    -------
    jnp.ndarray
        Flux values
    """

    # Evaluate the linear model
    λ = λ[:, jnp.newaxis]
    continuum = jnp.tan(angles) * (λ - cont_center) + offsets

    return jnp.where(
        jnp.logical_and(continuum_regions[:, 0] < λ, λ < continuum_regions[:, 1]),
        continuum,
        0.0,
    )


@jit
def powerLawContinuum(λ: jnp.ndarray, λ0: float, a: float, β: float) -> jnp.ndarray:
    """
    Compute the power law continuum

    Parameters
    ----------
    λ : jnp.ndarray
        Wavelength values
    λ0 : float
        Reference wavelength
    a : float
        Amplitude of the power law
    β : float
        Power law index

    Returns
    -------
    jnp.ndarray
        Flux values
    """

    return a * ((λ / λ0) ** β)
