"""JAX-jitted line profile integration and evaluation kernels.

Integration kernels (``integrate_*``) compute the fraction of a normalized
profile that falls wi_absthin each wavelength bin ``[low, high]``.  Evaluation
kernels (``evaluate_*``) compute the normalized profile value (probability
density) at arbitrary wavelength points.

All functions are pure JAX wi_absth no numpyro dependency and are designed to be
called from wi_absthin :func:`jax.jit`-compiled model code.
"""

from __future__ import annotations

from typing import Final, cast

import jax.numpy as jnp
from jax import Array, config, jit, lax
from jax.scipy.special import erf, erfc
from jax.typing import ArrayLike

# Conversion: FWHM to sigma for the half-variance parametrization of erf.
# sigma = FWHM / (2 sqrt(2 ln 2)); erf uses sqrt(2)*sigma, so the factor is:
_HALFVAR_SIGMA_TO_FWHM: Final[Array] = 2 * jnp.sqrt(jnp.log(2))

# Conversion factor from exponential (Laplace) scale to FWHM
# pdf = (1/(2*b)) * exp(-|x - μ|/b)
# max(pdf) = 1/(2*b), half max = 1/(4*b)
# 1/(4*b) = 1/(2*b) * exp(-|x - μ|/b) => exp(-|x - μ|/b) = 1/2
# => |x - μ|/b = ln(2) => FWHM = 2*b*ln(2)
_EXP_SCALE_TO_FWHM: Final[Array] = 2 * jnp.log(2)

# Precompute constants
_INV_SQRt2PI: Final[Array] = 1.0 / jnp.sqrt(2.0 * jnp.pi)
_SQRT2: Final[Array] = jnp.sqrt(2.0)
_SQRT6: Final[Array] = jnp.sqrt(6.0)
_SQRt24: Final[Array] = jnp.sqrt(24.0)

# 20-point Gauss-Legendre nodes transformed to [0, 1] and weights scaled by 1/2.
# Used by _owens_t for vectorized quadrature (no loop).
# nodes: t_i = (s_i + 1) / 2 where s_i are the GL nodes on [-1, 1]
# weights: w_i / 2 (Jacobian for the [-1,1] → [0,1] substitution)
_OWENS_T_NODES: Final[Array] = 0.5 * (
    jnp.array([
        -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
        -0.8391169718222188, -0.7463064833401507, -0.6360536807265150,
        -0.5108670019508271, -0.3737060887154195, -0.2277858511416451,
        -0.0765265211334973,  0.0765265211334973,  0.2277858511416451,
         0.3737060887154195,  0.5108670019508271,  0.6360536807265150,
         0.7463064833401507,  0.8391169718222188,  0.9122344282513259,
         0.9639719272779138,  0.9931285991850949,
    ])
    + 1.0
)
_OWENS_T_WEIGHTS: Final[Array] = 0.5 * jnp.array([
    0.0176140071391521, 0.0406014298003869, 0.0626720483341091, 0.0832767415767047,
    0.1019301198172404, 0.1181945319615184, 0.1316886384491766, 0.1420961093183820,
    0.1491729864726037, 0.1527533871307258, 0.1527533871307258, 0.1491729864726037,
    0.1420961093183820, 0.1316886384491766, 0.1181945319615184, 0.1019301198172404,
    0.0832767415767047, 0.0626720483341091, 0.0406014298003869, 0.0176140071391521,
])

# -------------------------------------------------------------------
# Private helpers
# -------------------------------------------------------------------


def _combine_fwhm(fwhm1: ArrayLike, fwhm2: ArrayLike) -> Array:
    """Combine two FWHM values in quadrature: ``sqrt(fwhm1**2 + fwhm2**2)``."""
    return jnp.sqrt(fwhm1 * fwhm1 + fwhm2 * fwhm2)


def _fwhm_to_sigma(fwhm: ArrayLike) -> Array:
    """Gaussian sigma from FWHM: ``sigma = FWHM / (2 sqrt(2 ln 2))``."""
    return cast(Array, fwhm / (_HALFVAR_SIGMA_TO_FWHM * _SQRT2))


def _gaussian_pdf(x: ArrayLike, fwhm: ArrayLike) -> Array:
    """Normalised Gaussian PDF at displacement *dx* wi_absth standard deviation *sigma*."""
    inv_sigma = 1 / _fwhm_to_sigma(fwhm)
    return cast(
        Array,
        jnp.exp(-0.5 * (x * x * inv_sigma * inv_sigma)) * _INV_SQRt2PI * inv_sigma,
    )


def _gaussian_cdf_diff(low: ArrayLike, high: ArrayLike, total_fwhm: ArrayLike) -> Array:
    """Gaussian CDF difference ``Φ(high) - Φ(low)`` via the erf halfvar parametrization."""
    inv_erf_scale = _HALFVAR_SIGMA_TO_FWHM / total_fwhm
    return cast(Array, 0.5 * (erf(high * inv_erf_scale) - erf(low * inv_erf_scale)))


def _cauchy_pdf(x: ArrayLike, hwhm: ArrayLike) -> Array:
    """Normalised Cauchy (Lorentzian) PDF at displacement *dx* wi_absth half-wi_absdth *hwhm*."""
    return cast(Array, (hwhm / jnp.pi) / (x * x + hwhm * hwhm))


def _cauchy_cdf_diff(low: ArrayLike, high: ArrayLike, fwhm: ArrayLike) -> Array:
    """Cauchy (Lorentzian) CDF difference ``F(high) - F(low)``."""
    inv_hwhm = 2 / fwhm
    t_low = low * inv_hwhm
    t_high = high * inv_hwhm
    return cast(Array, (jnp.arctan(t_high) - jnp.arctan(t_low)) / jnp.pi)


# Don't need this but keeping anyways just in case
def _laplace_cdf_diff(low: ArrayLike, high: ArrayLike, fwhm: ArrayLike) -> Array:
    """Laplace (double exponential) CDF difference ``F(high) - F(low)``."""
    b = fwhm / _EXP_SCALE_TO_FWHM

    t_high = high / b
    t_low = low / b

    # Explot symmetry and parts that cancel out.
    int_high = jnp.sign(t_high) * (1.0 - jnp.exp(-jnp.abs(t_high)))
    int_low = jnp.sign(t_low) * (1.0 - jnp.exp(-jnp.abs(t_low)))

    return cast(Array, 0.5 * (int_high - int_low))


def _skew(x: ArrayLike, alpha: ArrayLike, scale: ArrayLike) -> Array:
    return 1 + erf(alpha * x / scale)


def _skew_normal_pdf(x: ArrayLike, fwhm: ArrayLike, alpha: ArrayLike) -> Array:
    """Normalised skew-normal PDF at displacement *dx* wi_absth FWHM *fwhm* and skew *alpha*."""
    return _gaussian_pdf(x, fwhm) * _skew(x, alpha, _fwhm_to_sigma(fwhm))


def _owens_t(h: ArrayLike, a: ArrayLike) -> Array:
    r"""Owen's T function via 20-point Gauss-Legendre quadrature.

    Uses the trigonometric substitution :math:`x = \tan\theta` to map the
    integration domain :math:`[0, |a|]` to :math:`[0, \arctan|a|] \subset [0, \pi/2)`:

    .. math::

        T(h, a) = \frac{1}{2\pi} \int_0^{\arctan|a|}
                  \exp\!\left(-\frac{h^2}{2\cos^2\!\theta}\right) d\theta

    The domain is always bounded by :math:`\pi/2` regardless of :math:`|a|`, so
    the 20 GL nodes are distributed uniformly over the function's natural support,
    giving near-machine-precision accuracy for any finite :math:`h` and :math:`a`
    without branching or loops.  Sign is restored via :math:`T(h,-a) = -T(h,a)`.

    Parameters
    ----------
    h : ArrayLike
        Real-valued input.
    a : ArrayLike
        Real-valued input.

    Returns
    -------
    Array
        Value of Owen's T function.
    """
    h = jnp.asarray(h)
    a = jnp.asarray(a)
    theta_max = jnp.arctan(jnp.abs(a))                               # ∈ [0, π/2)
    theta = theta_max[..., None] * _OWENS_T_NODES                    # (..., 20)
    cos2 = jnp.cos(theta) ** 2
    integrand = jnp.exp(-0.5 * h[..., None] ** 2 / cos2)            # (..., 20)
    result = theta_max / (2.0 * jnp.pi) * jnp.sum(_OWENS_T_WEIGHTS * integrand, axis=-1)
    return cast(Array, jnp.where(a < 0.0, -result, result))


def _skew_normal_cdf_diff(
    low: ArrayLike, high: ArrayLike, fwhm: ArrayLike, alpha: ArrayLike
) -> Array:
    """Skew-normal CDF difference ``F(high) - F(low)``."""
    sigma = _fwhm_to_sigma(fwhm)
    cdf = _gaussian_cdf_diff(low, high, fwhm)
    return cdf + 2 * (_owens_t(high / sigma, alpha) - _owens_t(low / sigma, alpha))


# -------------------------------------------------------------------
# Gaussian kernel
# -------------------------------------------------------------------


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
    total_fwhm = _combine_fwhm(lsf_fwhm, fwhm)
    return _gaussian_cdf_diff(low - center, high - center, total_fwhm)


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
    return _gaussian_pdf(wavelength - center, total_fwhm)


# -------------------------------------------------------------------
# Pseudo Voigt
# Thompson, Cox & Hastings (1987), J. Appl. Cryst. 20, 79-83.
# DOI: 10.1107/S0021889887087090
# -------------------------------------------------------------------

# Pseudo-Voigt polynomial coefficients — Thompson, Cox & Hastings (1987)
_VOIGT_FWHM_CS: Final[Array] = jnp.array([1, 2.69268, 2.42843, 4.47163, 0.07842, 1])
_VOIGT_ETA_CS: Final[Array] = jnp.array([1.33603, -0.47719, 0.11116])


def _voigt_params_thompson(fwhm_g: ArrayLike, fwhm_l: ArrayLike) -> tuple[Array, Array]:
    pows = jnp.arange(_VOIGT_FWHM_CS.size)
    fwhm_eff = cast(
        Array, jnp.sum(_VOIGT_FWHM_CS * (fwhm_g**pows) * (fwhm_l ** pows[::-1])) ** 0.2
    )
    fwhm_ratio = fwhm_l / fwhm_eff
    eta = cast(
        Array,
        jnp.sum(_VOIGT_ETA_CS * (fwhm_ratio ** jnp.arange(1, len(_VOIGT_ETA_CS) + 1))),
    )
    return fwhm_eff, eta


def _voigt_thompson_cdf_diff(
    low: ArrayLike, high: ArrayLike, fwhm_g: ArrayLike, fwhm_l: ArrayLike
) -> Array:
    fwhm_eff, eta = _voigt_params_thompson(fwhm_g, fwhm_l)
    lorentzian = eta * _cauchy_cdf_diff(low, high, fwhm_eff)
    gaussian = (1 - eta) * _gaussian_cdf_diff(low, high, fwhm_eff)
    return cast(Array, lorentzian + gaussian)


def _voigt_thompson_pdf(x: ArrayLike, fwhm_g: ArrayLike, fwhm_l: ArrayLike) -> Array:
    fwhm_eff, eta = _voigt_params_thompson(fwhm_g, fwhm_l)
    return cast(
        Array,
        (1 - eta) * _gaussian_pdf(x, fwhm_eff) + eta * _cauchy_pdf(x, 0.5 * fwhm_eff),
    )


# -------------------------------------------------------------------
# Extended Pseudo-Voigt
# Ida, Ando & Toraya (2000), J. Appl. Cryst. 33, 1311-1316
# DOI: 10.1107/S0021889800010219
# -------------------------------------------------------------------


# Extended pseudo-Voigt polynomial coefficients — Ida, Ando & Toraya (2000), Table 1.
# Coefficients ordered i = 0 ... 6 (lowest to highest power of rho).
# wG = 1 - rho * sum(A * rho^i)
_IDA_A: Final[Array] = jnp.array(
    [0.66000, 0.15021, -1.24984, 4.74052, -9.48291, 8.48252, -2.95553]
)
# wL = 1 - (1-rho) * sum(B * rho^i)
_IDA_B: Final[Array] = jnp.array(
    [-0.42179, -1.25693, 10.30003, -23.45651, 29.14158, -16.50453, 3.19974]
)
# wI = sum(C * rho^i)
_IDA_C: Final[Array] = jnp.array(
    [1.19913, 1.43021, -15.36331, 47.06071, -73.61822, 57.92559, -17.80614]
)
# wP = sum(D * rho^i)
_IDA_D: Final[Array] = jnp.array(
    [1.10186, -0.47745, -0.68688, 2.76622, -4.55466, 4.05475, -1.26571]
)
# eta_L = rho * [1 + (1-rho) * sum(F * rho^i)]
_IDA_F: Final[Array] = jnp.array(
    [-0.30165, -1.38927, 9.31550, -24.10743, 34.96491, -21.18862, 3.70290]
)
# eta_I = rho * (1-rho) * sum(G * rho^i)
_IDA_G: Final[Array] = jnp.array(
    [0.25437, -0.14107, 3.23653, -11.09215, 22.10544, -24.12407, 9.76947]
)
# eta_P = rho * (1-rho) * sum(H * rho^i)
_IDA_H: Final[Array] = jnp.array(
    [1.01579, 1.50429, -9.21815, 23.59717, -39.71134, 32.83023, -10.02142]
)

# Conversion: irrational-function gamma_I to FWHM: W_I = 2*(2^(2/3) - 1)^(1/2) * gamma_I
# So gamma_I = W_I / (2*(2^(2/3) - 1)^(1/2))
_IRRAT_FWHM_TO_GAMMA: Final[Array] = 0.5 / jnp.sqrt(2.0 ** (2.0 / 3.0) - 1.0)

# Conversion: hyperbolic-function gamma_P to FWHM: W_P = 2*ln(sqrt(2) + 1) * gamma_P
# So gamma_P = W_P / (2*ln(sqrt(2) + 1))
_HYPER_FWHM_TO_GAMMA: Final[Array] = 0.5 / jnp.log(jnp.sqrt(2.0) + 1.0)


# Intermediate function fl
def _f_l_pdf(x, hwhm):
    inv_hwhm = 1 / hwhm
    t = x * inv_hwhm
    return (0.5 * inv_hwhm) * (1 + t * t) ** (-3 / 2)


def _f_l_cdf_diff(low, high, hwhm):
    inv_hwhm = 1 / hwhm
    t_low = low * inv_hwhm
    t_high = high * inv_hwhm
    return (
        t_high / jnp.sqrt(1 + t_high * t_high) - t_low / jnp.sqrt(1 + t_low * t_low)
    ) / 2


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


def _f_p_cdf_diff(low, high, gamma):
    """CDF difference for the hyperbolic-sech² function.

    FP(x) = (1/2) * tanh(x/gammaP), so FP(∞) - FP(-∞) = 1.
    *low* and *high* must be center-relative displacements.
    """
    inv_gamma = 1 / gamma
    return 0.5 * (jnp.tanh(high * inv_gamma) - jnp.tanh(low * inv_gamma))


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
    fwhm_total = fwhm_g_total + fwhm_l
    rho = fwhm_l / fwhm_total
    one_minus_rho = 1.0 - rho

    # Powers [rho^0, ..., rho^6] for polynomial evaluation
    rho_pows = rho ** jnp.arange(7)

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

    return (  # type: ignore[return-value]
        cast(Array, wg_abs),
        cast(Array, wl_abs),
        cast(Array, wi_abs),
        cast(Array, wp_abs),
        cast(Array, eta_g),
        cast(Array, eta_l),
        cast(Array, eta_i),
        cast(Array, eta_p),
    )


def _voigt_ida_cdf_diff(
    low: ArrayLike, high: ArrayLike, fwhm_g_total: ArrayLike, fwhm_l: ArrayLike
) -> Array:
    """CDF difference of the extended pseudo-Voigt approximation (Ida et al. 2000).

    Parameters
    ----------
    low, high : ArrayLike
        Absolute bin edges.
    center : ArrayLike
        Line center.
    fwhm_g_total : ArrayLike
        Total Gaussian FWHM (intrinsic + LSF in quadrature).
    fwhm_l : ArrayLike
        Lorentzian FWHM.
    """
    wg_abs, wl_abs, wi_abs, wp_abs, eta_g, eta_l, eta_i, eta_p = _voigt_ida_params(
        fwhm_g_total, fwhm_l
    )
    # Center-relative displacements needed by the non-standard CDFs

    gauss = eta_g * _gaussian_cdf_diff(low, high, wg_abs)
    lorentz = eta_l * _cauchy_cdf_diff(low, high, wl_abs)
    irrat = eta_i * _f_l_cdf_diff(low, high, wi_abs * _IRRAT_FWHM_TO_GAMMA)
    hyper = eta_p * _f_p_cdf_diff(low, high, wp_abs * _HYPER_FWHM_TO_GAMMA)
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

    gauss = eta_g * _gaussian_pdf(dx, wg_abs)
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
    return cast(
        Array,
        jnp.where(s >= 15, w1, jnp.where(s >= 5.5, w2, jnp.where(reg3_cond, w3, w4))),
    )


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
    denom = sigma_g * _SQRT2
    z = cast(Array, (x + 1j * gamma_l) / denom)

    # Voigt profile: V(x) = Re[w(z)] / (sigma * sqrt(2*pi))
    return jnp.real(_faddeeva_humlicek(z)) * _INV_SQRt2PI / sigma_g


# -------------------------------------------------------------------
# Voigt Kernel
# -------------------------------------------------------------------


@jit
def integrate_voigt(
    low: ArrayLike,
    high: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
) -> Array:
    """Integrate a Voigt profile over wavelength bins via the extended pseudo-Voigt.

    Uses the Ida, Ando & Toraya (2000) extended pseudo-Voigt approximation, which
    achieves < 0.12% peak-height deviation from the true Voigt profile using a
    four-component mixture: Gaussian, Lorentzian, irrational, and hyperbolic-sech².
    The LSF Gaussian FWHM is added in quadrature to the intrinsic Gaussian FWHM
    before computing the extended-pseudo-Voigt parameters.

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
    """
    fwhm_g_total = _combine_fwhm(lsf_fwhm, fwhm_g)
    return _voigt_ida_cdf_diff(low - center, high - center, fwhm_g_total, fwhm_l)


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
# Gaussian-Laplace (symmetric EMG) kernel
# -------------------------------------------------------------------

# Threshold for when exp(x*x) overflows
_OVERFLOW_THRESHOLD: Final[float] = (
    26.0 if getattr(config, 'jax_enable_x64', True) else 9.0
)


def _integrandGL(t: ArrayLike, a: ArrayLike) -> Array:
    """
    Antiderivative of the Gaussian-Laplace exponential correction.

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
    low: ArrayLike,
    high: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
) -> Array:
    """Integrate a symmetric EMG (Gaussian convolved wi_absth Laplace) over wavelength bins.

    The LSF is added in quadrature to the Gaussian component, then the
    symmetric EMG CDF is evaluated analytically using a numerically stable
    erfcx form. See the :doc:`SEMG derivation </derivations/semg>` for details.

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins.
    high : jnp.ndarray
        High wavelength edges of bins.
    center : jnp.ndarray
        Line centers.
    lsf_fwhm : jnp.ndarray
        Instrumental LSF FWHM.
    fwhm_g : jnp.ndarray
        Intrinsic Gaussian component FWHM.
    fwhm_l : jnp.ndarray
        Laplace component FWHM (related to scale *b* by ``FWHM = 2 b ln 2``).

    Returns
    -------
    jnp.ndarray
        Integrated fraction per bin (sums to 1 over all bins).
    """
    # This function breaks for pure Laplace, does that matter? lsf is never zero?

    fwhm_g_total = _combine_fwhm(lsf_fwhm, fwhm_g)

    # sigma here is the halfvar-parametrised sigma (= sqrt(2) * standard sigma)
    σ = fwhm_g_total / _HALFVAR_SIGMA_TO_FWHM
    λ = _EXP_SCALE_TO_FWHM / fwhm_l

    # Convolution parameter
    a = 0.5 * σ * λ

    # Gaussian component (via error function CDF)
    gaussian_cdf = _gaussian_cdf_diff(low - center, high - center, fwhm_g_total)

    # Exponential correction
    scale = 1 / σ
    t_low = (low - center) * scale
    t_high = (high - center) * scale
    exp_correction = _integrandGL(t_high, a) - _integrandGL(t_low, a)

    # Guard against large a, the Gaussian limit
    return gaussian_cdf + jnp.where(a > _OVERFLOW_THRESHOLD, 0, 0.25 * exp_correction)


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
    fwhm_g_total = _combine_fwhm(lsf_fwhm, fwhm_g)
    sigma = _fwhm_to_sigma(fwhm_g_total)
    lam = _EXP_SCALE_TO_FWHM / fwhm_l

    # Symmetric EMG PDF: Gaussian convolved wi_absth symmetric Laplace.
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
    negterm = jnp.exp(-two_ua) * erfc(a - u_abs)
    result = (0.25 * lam) * jnp.exp(a * a) * (posterm + negterm)

    # Pure Gaussian limit when exponential component is negligible
    return jnp.where(
        a > _OVERFLOW_THRESHOLD,
        _gaussian_pdf(wavelength - center, fwhm_g_total),
        result,
    )


# -------------------------------------------------------------------
# Split-normal kernel
# -------------------------------------------------------------------


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
    total_fwhm_blue = _combine_fwhm(lsf_fwhm, fwhm_blue)
    total_fwhm_red = _combine_fwhm(lsf_fwhm, fwhm_red)

    inv_sigma_blue = _HALFVAR_SIGMA_TO_FWHM / total_fwhm_blue
    inv_sigma_red = _HALFVAR_SIGMA_TO_FWHM / total_fwhm_red

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

    cdf_high = _split_normal_cdf(
        high, center, inv_sigma_blue, inv_sigma_red, w_blue, w_red
    )
    cdf_low = _split_normal_cdf(
        low, center, inv_sigma_blue, inv_sigma_red, w_blue, w_red
    )
    return cdf_high - cdf_low


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
    norm = (sigma_blue + sigma_red) * jnp.sqrt(jnp.pi / 2.0)
    dx = wavelength - center
    dx2 = dx * dx
    val_blue = jnp.exp(-0.5 * dx2 / (sigma_blue * sigma_blue))
    val_red = jnp.exp(-0.5 * dx2 / (sigma_red * sigma_red))
    return jnp.where(wavelength <= center, val_blue, val_red) / norm


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
    t = t_halfvar * _SQRT2
    t2 = t * t
    g = jnp.exp(-0.5 * t2)
    # He_2(y) = y²-1,  He_3(y) = y(y²-3)
    return g * (h3_eff * (t2 - 1) + h4_eff * t * (t2 - 3))


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
    """Integrate a Gauss-Hermite profile over wavelength bins.

    Uses the closed-form CDF derived from the orthonormal probabilists'
    Hermite expansion. Convolving wi_absth the Gaussian LSF rescales the shape
    parameters as ``h_m' = h_m * (sigma_g / sigma_tot)^m``. See the
    :doc:`Gauss-Hermite derivation </derivations/gauss-hermite>` for details.

    Parameters
    ----------
    low : jnp.ndarray
        Low wavelength edges of bins.
    high : jnp.ndarray
        High wavelength edges of bins.
    center : jnp.ndarray
        Line centers.
    fwhm_lsf : jnp.ndarray
        Instrumental LSF FWHM.
    fwhm_g : jnp.ndarray
        Gaussian component FWHM.
    h3 : jnp.ndarray
        Gauss-Hermite h3 (skewness) coefficient.
    h4 : jnp.ndarray
        Gauss-Hermite h4 (kurtosis) coefficient.

    Returns
    -------
    jnp.ndarray
        Integrated fraction per bin (sums to 1 over all bins).
    """
    fwhm_tot = _combine_fwhm(fwhm_lsf, fwhm_g)

    # Sigma ratio: GH moments scale as r^n under convolution wi_absth a Gaussian LSF
    r = fwhm_g / fwhm_tot
    r3 = r * r * r

    # GH coefficients scaled by sigma ratio (moment theorem for Gaussian convolution)
    c3 = h3 * r3 / _SQRT6
    c4 = h4 * r3 * r / _SQRt24

    # Normalized bin edges (halfvar parametrisation for consistency wi_absth erf CDF)
    inv_sigma_tot = _HALFVAR_SIGMA_TO_FWHM / fwhm_tot
    t_low = (low - center) * inv_sigma_tot
    t_high = (high - center) * inv_sigma_tot

    gaussian_cdf = 0.5 * (erf(t_high) - erf(t_low))
    gh_correction = _integrandGH(t_high, c3, c4) - _integrandGH(t_low, c3, c4)
    return gaussian_cdf - _INV_SQRt2PI * gh_correction


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
    c3 = h3 * r3 / _SQRT6
    c4 = h4 * r3 * r / _SQRt24

    # Standard coordinate
    gauss_pdf = _gaussian_pdf(wavelength - center, fwhm_tot)

    # Hermite polynomial corrections (probabilists' convention)
    # He_3(y) = y^3 - 3y,  He_4(y) = y^4 - 6y^2 + 3
    y = (wavelength - center) / sigma_tot
    y3 = y * y * y
    he3 = y3 - 3 * y
    he4 = y * y3 - 6 * y * y + 3
    return gauss_pdf * (1.0 + c3 * he3 + c4 * he4)


# -------------------------------------------------------------------
# Skew Gaussian kernel
# -------------------------------------------------------------------


def _alpha_gauss(alpha: ArrayLike, fwhm_lsf: ArrayLike, fwhm_tot: ArrayLike) -> Array:
    """Effective skewness after convolving a skew Gaussian with a Gaussian LSF."""
    return alpha * fwhm_tot / jnp.sqrt(fwhm_tot**2 + 2 * alpha**2 * fwhm_lsf**2)


@jit
def integrate_skewGaussian(
    low: ArrayLike,
    high: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm: ArrayLike,
    alpha: ArrayLike,
) -> Array:
    """Integrate a skew Gaussian over wavelength bins.

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
    fwhm_tot = _combine_fwhm(lsf_fwhm, fwhm)
    alpha_gauss = _alpha_gauss(alpha, lsf_fwhm, fwhm_tot)
    return _skew_normal_cdf_diff(low - center, high - center, fwhm_tot, alpha_gauss)


@jit
def evaluate_skewGaussian(
    wavelength: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm: ArrayLike,
    alpha: ArrayLike,
) -> Array:
    """Evaluate a skew Gaussian profile at wavelength points.

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
    fwhm_tot = _combine_fwhm(lsf_fwhm, fwhm)
    alpha_gauss = _alpha_gauss(alpha, lsf_fwhm, fwhm_tot)
    return _skew_normal_pdf(wavelength - center, fwhm_tot, alpha_gauss)


# -------------------------------------------------------------------
# Skew Voigt kernel  (evaluate only — no analytic integrate)
# -------------------------------------------------------------------


# FXIG boost-correction parameters from numerical fit over (lor, alpha, eta) grid.
# log_boost = k * xi^a * eta^b / (1 + q*xi^c) / (1 + r*|alpha|^d)
# where xi = gamma/sigma_lsf, eta = sigma_lsf/sigma_g, gamma = fwhm_l/2.
_FXIG_K: Final[float] = 9.9126
_FXIG_A: Final[float] = 0.43576
_FXIG_B: Final[float] = 0.97281
_FXIG_C: Final[float] = 2.1469
_FXIG_Q: Final[float] = 2.3396
_FXIG_R: Final[float] = 26.449
_FXIG_D: Final[float] = 0.36404


def _alpha_eff(
    lsf_fwhm: ArrayLike, fwhm_g: ArrayLike, fwhm_l: ArrayLike, alpha: ArrayLike
) -> Array:
    """Effective skewness after convolving a skew Voigt with a Gaussian LSF.

    Applies the Gaussian-body exact formula as a base, then multiplies by the
    FXIG boost correction that accounts for the Lorentzian component.
    """
    sigma_g = _fwhm_to_sigma(fwhm_g)
    sigma_lsf = _fwhm_to_sigma(lsf_fwhm)
    gamma = fwhm_l / 2

    w0 = _combine_fwhm(fwhm_g, fwhm_l)
    w0p = _combine_fwhm(w0, lsf_fwhm)

    # Gaussian-body exact formula
    a_gauss = alpha * w0 / jnp.sqrt(w0p**2 + 2 * alpha**2 * sigma_lsf**2)

    # FXIG boost correction for Lorentzian component
    lor = gamma / (sigma_g + 1e-30)
    eta = sigma_lsf / (sigma_g + 1e-30)
    xi = lor / (eta + 1e-30)
    log_boost = (
        _FXIG_K
        * xi**_FXIG_A
        * eta**_FXIG_B
        / (1 + _FXIG_Q * xi**_FXIG_C)
        / (1 + _FXIG_R * jnp.abs(alpha) ** _FXIG_D)
    )

    return cast(Array, a_gauss * jnp.exp(log_boost))


@jit
def integrate_skewVoigt(
    low: ArrayLike,
    high: ArrayLike,
    center: ArrayLike,
    lsf_fwhm: ArrayLike,
    fwhm_g: ArrayLike,
    fwhm_l: ArrayLike,
    alpha: ArrayLike,
) -> Array:
    """Integrate a skew pseudo-Voigt profile over wavelength bins.

    Uses the same extended pseudo-Voigt approximation as integrate_voigt,
    multiplied by the skew correction integrated via the error function. The skewness
    parameter is rescaled after convolution with the LSF via the Gaussian-body exact
    formula with an FXIG boost correction for the Lorentzian component.

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
    alpha : ArrayLike
        Skewness parameter relative to sqrt(fwhm_g**2 + fwhm_l**2).

    Returns
    -------
    Array
        Integrated fraction per bin.
    """
    # Compute Voigt CDF
    voigt_cdf = integrate_voigt(low, high, center, lsf_fwhm, fwhm_g, fwhm_l)

    # Get effective skew
    fwhm_g_tot = _combine_fwhm(lsf_fwhm, fwhm_g)
    w0 = _combine_fwhm(fwhm_g_tot, fwhm_l)
    a_eff = _alpha_eff(lsf_fwhm, fwhm_g, fwhm_l, alpha)

    # Evaluate skew correction at bin center (midpoint of low and high)
    x = 0.5 * (low + high) - center
    skew = _skew(x, a_eff, w0)
    return voigt_cdf * skew


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
        Skewness parameter relative to sqrt(fwhm_g**2 + fwhm_l**2).

    Returns
    -------
    Array
        Normalised profile value at each wavelength point (1/wavelength units).
    """
    voigt_pdf = evaluate_voigt(wavelength, center, lsf_fwhm, fwhm_g, fwhm_l)

    # Get effective skew
    fwhm_g_tot = _combine_fwhm(lsf_fwhm, fwhm_g)
    w0 = _combine_fwhm(fwhm_g_tot, fwhm_l)
    a_eff = _alpha_eff(lsf_fwhm, fwhm_g, fwhm_l, alpha)

    # Evaluate skew correction at bin center (midpoint of low and high)
    skew = _skew(wavelength - center, _alpha_eff(lsf_fwhm, fwhm_g, fwhm_l, a_eff), w0)
    return voigt_pdf * skew
