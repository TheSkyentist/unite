"""JAX-jitted continuum evaluation kernels.

All functions are pure JAX with no numpyro dependency and are designed to
be called from within :func:`jax.jit`-compiled model code.
"""

from __future__ import annotations

from functools import partial
from typing import Final, cast

import jax.numpy as jnp
from astropy import units as u
from astropy.constants import (
    c as _c,  # pyright: ignore[reportAttributeAccessIssue]
    h as _h,  # pyright: ignore[reportAttributeAccessIssue]
    k_B as _k_B,  # pyright: ignore[reportAttributeAccessIssue]
)
from jax import Array, jit
from jax.typing import ArrayLike

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

_HC_KB: Final[float] = ((_c * _h) / _k_B).to(u.um * u.K).value

# ---------------------------------------------------------------------------
# Planck blackbody
# ---------------------------------------------------------------------------


@jit
def planck_function(
    wavelength_micron: ArrayLike, temperature_k: ArrayLike, pivot_micron: float
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
        Normalization wavelength in microns.

    Returns
    -------
    Array
        Normalized Planck function (= 1 at *pivot_micron*).

    Notes
    -----
    Physical constants are pre-combined to avoid gradient overflow in JAX
    when differentiating ``exp(hc / λkT)`` with respect to temperature.
    """
    hc_kbt = _HC_KB / temperature_k
    x = hc_kbt / wavelength_micron
    x_p = hc_kbt / pivot_micron

    return cast(
        Array,
        ((pivot_micron / wavelength_micron) ** 5) * (jnp.expm1(x_p) / jnp.expm1(x)),
    )


# ---------------------------------------------------------------------------
# Chebyshev polynomial
# ---------------------------------------------------------------------------


def chebval(x: ArrayLike, coeffs: ArrayLike) -> Array:
    """Evaluate a Chebyshev series using the trigonometric identity.

    Parameters
    ----------
    x : ArrayLike
        Evaluation points, normalized to ``[-1, 1]``.
    coeffs : list of ArrayLike
        Chebyshev coefficients ``[c0, c1, ..., cN]``.

    Returns
    -------
    Array
        Series value at each point in *x*.
    """
    x = jnp.atleast_1d(x)
    coeffs = jnp.asarray(coeffs)
    # Create an array of degrees: [0, 1, 2, ..., n]
    degrees = jnp.arange(len(coeffs), dtype=x.dtype)

    # Calculate the theta values: theta = acos(x)
    # Resulting shape: (len(x),)
    theta = jnp.acos(x)

    # Calculate the basis: cos(n * theta)
    # We use broadcasting to get a matrix of shape (len(coeffs), len(x))
    basis = jnp.cos(degrees[:, None] * theta[None, :])

    # Weighted sum: coeffs @ basis
    return jnp.dot(coeffs, basis)

    # ---------------------------------------------------------------------------
    # B-spline
    # ---------------------------------------------------------------------------


@jit
def bernstein_eval(x, coeffs, binom_coeffs):
    """
    Evaluate a Bernstein polynomial series using a vectorized basis matrix.

    Parameters
    ----------
    x : ArrayLike
        Evaluation points, must be normalized to the range [0, 1].
        Shape: (N,).
    coeffs : ArrayLike
        Bernstein coefficients (control points). Shape: (n + 1,).
    binom_coeffs : ArrayLike
        Pre-computed binomial coefficients for degree n, where
        binom_coeffs[i] = C(n, i). Shape: (n + 1,).

    Returns
    -------
    Array
        The evaluated polynomial values at each point in x. Shape: (N,).
    """
    x = jnp.atleast_1d(x)
    n = coeffs.shape[0] - 1
    i = jnp.arange(n + 1)

    # Compute the basis functions using broadcasting
    # For n=10, this is numerically safe and extremely fast
    # Resulting shape: (len(wavelength), n+1)
    basis = binom_coeffs * (x[:, None] ** i) * ((1.0 - x[:, None]) ** (n - i))

    return basis @ coeffs


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

    # Handle the right-boundary condition (x == knots[-1])
    # by pushing points slightly inside the last interval.
    t = jnp.clip(t, knots[0], knots[-1] - 1e-14)

    # Degree 0 basis: indicator functions
    basis = jnp.where(
        (t[:, None] >= knots[None, :-1]) & (t[:, None] < knots[None, 1:]), 1.0, 0.0
    )

    # Recursive Cox-de Boor steps
    for d in range(1, degree + 1):
        n_basis = n_knots - d - 1

        # Denominators
        dt_left = knots[d : d + n_basis] - knots[:n_basis]
        dt_right = knots[d + 1 : d + 1 + n_basis] - knots[1 : 1 + n_basis]

        # Avoid division by zero for repeated knots
        # Using 1.0 as a dummy denominator; the 'where' will zero out the result anyway
        left_w = jnp.where(dt_left > 0, (t[:, None] - knots[:n_basis]) / dt_left, 0.0)
        right_w = jnp.where(
            dt_right > 0, (knots[d + 1 : d + 1 + n_basis] - t[:, None]) / dt_right, 0.0
        )

        # Update basis: linear combination of lower-degree bases
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
