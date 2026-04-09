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
    ``2 * half_width + 1`` fine-grid points.  The Gaussian width at each
    point is given by ``sigma_fine[j]`` in the same wavelength units as
    ``x_fine``.

    The input arrays are zero-padded at the spectral edges (model values outside
    the array are treated as zero flux), with ``x_fine`` and ``sigma_fine``
    edge-padded so the kernel remains well-defined at the boundaries.

    Parameters
    ----------
    x_fine : array_like, shape ``(N,)``
        Wavelengths of fine-grid points in canonical units.
    model_fine : array_like, shape ``(N,)``
        Intrinsic model flux values at fine-grid points (no LSF applied).
    sigma_fine : array_like, shape ``(N,)``
        Gaussian LSF sigma (``= FWHM / 2.355``) at each fine-grid point,
        in the same wavelength units as ``x_fine``.  May be a traced JAX
        value (e.g. when ``r_scale`` is a sampled parameter).
    half_width : int
        Kernel half-width in fine-grid indices.  Must be a Python ``int``
        (not a traced value) so it can be used as a static size in
        :func:`jax.lax.dynamic_slice`.  Should satisfy
        ``half_width >= ceil(4 * max(sigma_fine) / min(dx_fine))`` to
        capture at least 4 sigma of the broadest kernel.

    Returns
    -------
    jnp.ndarray, shape ``(N,)``
        LSF-convolved model flux at the fine-grid wavelengths.
    """
    x_fine = jnp.asarray(x_fine)
    model_fine = jnp.asarray(model_fine)
    sigma_fine = jnp.asarray(sigma_fine)

    pad = half_width
    w = 2 * half_width + 1

    # Pad: edge-extend wavelengths and sigmas so the kernel centre is always
    # well-defined; zero-pad model so out-of-range flux contributes nothing.
    x_pad = jnp.pad(x_fine, pad, mode='edge')
    m_pad = jnp.pad(model_fine, pad, constant_values=0.0)
    s_pad = jnp.pad(sigma_fine, pad, mode='edge')

    def _at(j: jax.Array) -> jnp.ndarray:
        # Extract fixed-size window around padded index j + pad.
        wx = jax.lax.dynamic_slice(x_pad, (j,), (w,))
        wm = jax.lax.dynamic_slice(m_pad, (j,), (w,))
        dx = wx - x_pad[j + pad]
        sigma_j = s_pad[j + pad]
        kernel = jnp.exp(-0.5 * (dx / sigma_j) ** 2)
        kernel = kernel / jnp.sum(kernel)  # normalize → flux-conserving
        return jnp.dot(kernel, wm)

    return jax.vmap(_at)(jnp.arange(len(x_fine)))
