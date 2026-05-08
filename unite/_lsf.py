"""Wavelength-dependent Gaussian LSF convolution for ``integration_mode='convolution'``.

Implements a banded spatially-varying Gaussian convolution on a non-uniform
wavelength grid via :func:`jax.lax.dynamic_slice` inside :func:`jax.vmap`.
All operations are JAX-jittable and compatible with numpyro inference engines.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

#: Conversion factor from FWHM to Gaussian sigma: ``1 / (2 * sqrt(2 * ln 2))``.
_FWHM_TO_SIGMA: float = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def _lsf_convolve(
    x_fine: ArrayLike, model_fine: ArrayLike, sigma_fine: ArrayLike, half_width: int
) -> jnp.ndarray:
    """Apply a spatially-varying Gaussian LSF convolution on a non-uniform grid.

    For each output point ``j``, the convolved value is a normalized Gaussian-
    weighted average of neighbouring input values within a window of
    ``2 * half_width + 1`` points.  The Gaussian width at each point is given
    by ``sigma_fine[j]`` in the same wavelength units as ``x_fine``.

    **Boundary treatment**: outside the array the wavelength grid is linearly
    extrapolated using the local pixel spacing at each edge (``dx`` of the
    first/last pixel pair).  This gives padded positions their correct distance
    from the kernel centre so the Gaussian falls off naturally.  The model
    values are constant-extrapolated (edge-extended) beyond the boundary,
    reflecting the assumption that the continuum does not abruptly drop to zero
    outside the fit region.  Using edge-extension for wavelengths instead would
    stack all padded positions at the boundary wavelength, making every kernel
    weight exactly 1 and collapsing the normalised result toward zero for edge
    pixels.

    Parameters
    ----------
    x_fine : array_like, shape ``(N,)``
        Wavelengths of grid points in canonical units.
    model_fine : array_like, shape ``(N,)``
        Intrinsic model flux values at grid points (no LSF applied).
    sigma_fine : array_like, shape ``(N,)``
        Gaussian LSF sigma (``= FWHM / 2.355``) at each grid point,
        in the same wavelength units as ``x_fine``.  May be a traced JAX
        value (e.g. when ``r_scale`` is a sampled parameter).
    half_width : int
        Kernel half-width in grid indices.  Must be a Python ``int``
        (not a traced value) so it can be used as a static size in
        :func:`jax.lax.dynamic_slice`.  Should satisfy
        ``half_width >= ceil(4 * max(sigma_fine) / min(dx))`` to
        capture at least 4 sigma of the broadest kernel.

    Returns
    -------
    jnp.ndarray, shape ``(N,)``
        LSF-convolved model flux at the input grid wavelengths.
    """
    x_fine = jnp.asarray(x_fine)
    model_fine = jnp.asarray(model_fine)
    sigma_fine = jnp.asarray(sigma_fine)

    pad = half_width
    w = 2 * half_width + 1

    # Linearly extrapolate the wavelength grid beyond both edges.
    # Using mode='edge' would stack every padded position at the boundary
    # wavelength, making dx=0 for all of them and inflating every kernel
    # weight to 1.0 — the normalised average then collapses toward zero for
    # edge pixels.  Linear extrapolation gives each padded position its
    # correct distance from the kernel centre so the Gaussian falls off
    # naturally, and the normalised result converges to the edge model value.
    dx_left = x_fine[1] - x_fine[0]
    dx_right = x_fine[-1] - x_fine[-2]
    x_pad = jnp.concatenate(
        [
            x_fine[0] + jnp.arange(-pad, 0) * dx_left,
            x_fine,
            x_fine[-1] + jnp.arange(1, pad + 1) * dx_right,
        ]
    )
    # Edge-extend the model: the continuum does not drop to zero at the
    # spectrum boundary, so the boundary value is a better prior than zero.
    m_pad = jnp.pad(model_fine, pad, mode='edge')
    s_pad = jnp.pad(sigma_fine, pad, mode='edge')

    def _at(j: jax.Array) -> jnp.ndarray:
        # Extract fixed-size window around padded index j + pad.
        wx = jax.lax.dynamic_slice(x_pad, (j,), (w,))
        wm = jax.lax.dynamic_slice(m_pad, (j,), (w,))
        dx = wx - x_pad[j + pad]
        sigma_j = s_pad[j + pad]
        # Compute exponent without per-element division: dx*dx * (-1 / (2*sigma*sigma)).
        neg_inv_two_sigma_sq = -0.5 / (sigma_j * sigma_j)
        kernel = jnp.exp(dx * dx * neg_inv_two_sigma_sq)
        # Normalise via a single scalar division at the end (flux-conserving),
        # avoiding the broadcast division of every kernel element.
        return jnp.dot(kernel, wm) / jnp.sum(kernel)

    return jax.vmap(_at)(jnp.arange(len(x_fine)))
