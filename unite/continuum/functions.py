"""JAX-jitted continuum evaluation kernels.

All functions are pure JAX with no numpyro dependency and are designed to
be called from within :func:`jax.jit`-compiled model code.
"""

from __future__ import annotations

from functools import partial
from typing import Final

import jax.nn as jnn
import jax.numpy as jnp
from astropy import constants
from jax import Array, jit
from jax.typing import ArrayLike

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------

_H: Final[float] = constants.h.si.value  # Planck constant (J·s)
_C_SI: Final[float] = constants.c.si.value  # Speed of light (m/s)
_KB: Final[float] = constants.k_B.si.value  # Boltzmann constant (J/K)


# ---------------------------------------------------------------------------
# Planck blackbody
# ---------------------------------------------------------------------------


@jit
def _safe_log_expm1(x: ArrayLike) -> Array:
    """Numerically stable ``log(exp(x) - 1)``.

    For large *x* (≳ 10) the expression equals *x*; for small *x* it is
    computed via ``log(expm1(x))``.  A smooth sigmoid blend avoids
    discontinuous gradients at the transition.

    Parameters
    ----------
    x : ArrayLike
        Input values (must be positive).

    Returns
    -------
    Array
        ``log(exp(x) - 1)`` computed with numerical stability.
    """
    alpha = jnn.sigmoid((x - 10.0) / 3.0)
    x_safe = jnp.minimum(x, 50.0)
    small = jnp.log(jnp.maximum(jnp.expm1(x_safe), 1e-100))
    return jnp.where(x > 50.0, x, alpha * x + (1 - alpha) * small)


@jit
def planck_function(
    wavelength_micron: ArrayLike, temperature_k: ArrayLike, pivot_micron: float = 0.5
) -> Array:
    """Return the normalized Planck function ``B_λ(T) / B_λ(pivot, T)``.

    Returns the blackbody spectral radiance normalized to unity at
    *pivot_micron*, so the fitted amplitude directly represents the
    observed flux at the pivot wavelength.

    Parameters
    ----------
    wavelength_micron : ArrayLike
        Rest-frame wavelengths in microns.
    temperature_k : ArrayLike
        Temperature in Kelvin.
    pivot_micron : float
        Normalization wavelength in microns.  Default ``0.5``.

    Returns
    -------
    Array
        Normalized Planck function (= 1 at *pivot_micron*).

    Notes
    -----
    Physical constants are pre-combined to avoid gradient overflow in JAX
    when differentiating ``exp(hc / λkT)`` with respect to temperature.
    """
    temperature_k = jnp.clip(temperature_k, 20.0, 1e5)

    wavelength_m = wavelength_micron * 1e-6
    pivot_m = pivot_micron * 1e-6

    const_wave = (_H * _C_SI) / (wavelength_m * _KB)
    const_pivot = (_H * _C_SI) / (pivot_m * _KB)

    x_wave = jnp.clip(const_wave / temperature_k, 0.01, 200.0)
    x_pivot = jnp.clip(const_pivot / temperature_k, 0.01, 200.0)

    log_ratio = -5.0 * jnp.log(wavelength_micron / pivot_micron)
    log_expm1_diff = _safe_log_expm1(x_pivot) - _safe_log_expm1(x_wave)

    return jnp.exp(jnp.clip(log_ratio + log_expm1_diff, -100.0, 100.0))


# ---------------------------------------------------------------------------
# Chebyshev polynomial
# ---------------------------------------------------------------------------


def chebval(x: ArrayLike, coeffs: list) -> Array:
    """Evaluate a Chebyshev series via Clenshaw recurrence.

    Parameters
    ----------
    x : ArrayLike
        Evaluation points, typically normalized to ``[-1, 1]``.
    coeffs : list of ArrayLike
        Chebyshev coefficients ``[c0, c1, ..., cN]``.

    Returns
    -------
    Array
        Series value at each point in *x*.
    """
    n = len(coeffs)
    if n == 1:
        return coeffs[0] + jnp.zeros_like(x)
    if n == 2:
        return coeffs[0] + coeffs[1] * x
    x2 = 2 * x
    c0, c1 = coeffs[-2], coeffs[-1]
    for k in range(3, n + 1):
        c0, c1 = coeffs[-k] - c1, c0 + c1 * x2
    return c0 + c1 * x


# ---------------------------------------------------------------------------
# B-spline
# ---------------------------------------------------------------------------


def bspline_basis(t: ArrayLike, knots: ArrayLike, degree: int) -> Array:
    """Compute the B-spline basis matrix via iterative Cox-de Boor recursion.

    The Python loop over *degree* is unrolled at JAX trace time because
    *degree* is a concrete ``int``, not a traced value.

    Parameters
    ----------
    t : ArrayLike
        Evaluation points, shape ``(N,)``.
    knots : ArrayLike
        Clamped knot vector, shape ``(M,)``.
    degree : int
        Spline degree (e.g. 3 for cubic).

    Returns
    -------
    Array
        Basis matrix, shape ``(N, n_basis)`` where ``n_basis = M - degree - 1``.
    """
    t = jnp.asarray(t)
    knots = jnp.asarray(knots)
    n_knots = len(knots)

    t_safe = jnp.where(t >= knots[-1], knots[-1] * (1 - 1e-14), t)

    basis = jnp.where(
        (t_safe[:, None] >= knots[None, :-1]) & (t_safe[:, None] < knots[None, 1:]),
        1.0,
        0.0,
    )

    for d in range(1, degree + 1):
        n_basis = n_knots - d - 1
        left_denom = knots[d : d + n_basis] - knots[:n_basis]
        right_denom = knots[d + 1 : d + 1 + n_basis] - knots[1 : 1 + n_basis]

        safe_left = jnp.where(left_denom > 0, left_denom, 1.0)
        safe_right = jnp.where(right_denom > 0, right_denom, 1.0)

        left_w = jnp.where(
            left_denom > 0,
            (t[:, None] - knots[None, :n_basis]) / safe_left[None, :],
            0.0,
        )
        right_w = jnp.where(
            right_denom > 0,
            (knots[None, d + 1 : d + 1 + n_basis] - t[:, None]) / safe_right[None, :],
            0.0,
        )
        basis = left_w * basis[:, :n_basis] + right_w * basis[:, 1 : n_basis + 1]

    return basis


@partial(jit, static_argnums=(3,))
def bspline_eval(
    wavelength: ArrayLike, coeffs: ArrayLike, knots: ArrayLike, degree: int
) -> Array:
    """Evaluate a B-spline continuum model.

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength values, shape ``(N,)``.
    coeffs : ArrayLike
        B-spline coefficients, shape ``(n_basis,)``.
    knots : ArrayLike
        Clamped knot vector.
    degree : int
        Spline degree (static for JIT).

    Returns
    -------
    Array
        Continuum flux, shape ``(N,)``.
    """
    basis = bspline_basis(wavelength, knots, degree)
    return basis @ coeffs


# ---------------------------------------------------------------------------
# Bernstein polynomial
# ---------------------------------------------------------------------------


@jit
def bernstein_eval(
    wavelength: ArrayLike,
    coeffs: ArrayLike,
    wavelength_min: float,
    wavelength_max: float,
    binom_coeffs: ArrayLike,
) -> Array:
    """Evaluate a Bernstein polynomial continuum model.

    Bernstein basis polynomials are non-negative on ``[0, 1]``, so
    positive coefficients guarantee a positive continuum everywhere.

    Parameters
    ----------
    wavelength : ArrayLike
        Wavelength values, shape ``(N,)``.
    coeffs : ArrayLike
        Bernstein coefficients, shape ``(n+1,)``.
    wavelength_min, wavelength_max : float
        Wavelength range for normalization to ``[0, 1]``.
    binom_coeffs : ArrayLike
        Pre-computed binomial coefficients ``C(n, i)``, shape ``(n+1,)``.

    Returns
    -------
    Array
        Continuum flux, shape ``(N,)``.
    """
    wavelength = jnp.asarray(wavelength)
    coeffs = jnp.asarray(coeffs)
    n = len(coeffs) - 1
    t = jnp.clip(
        (wavelength - wavelength_min) / (wavelength_max - wavelength_min), 0.0, 1.0
    )
    i = jnp.arange(n + 1)
    basis = (
        binom_coeffs
        * (t[:, None] ** i[None, :])
        * ((1 - t[:, None]) ** (n - i)[None, :])
    )
    return basis @ coeffs
