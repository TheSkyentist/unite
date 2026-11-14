"""Wavelength-dependent Gaussian LSF convolution for ``integration_mode='convolution'``.

Implements a banded spatially-varying Gaussian convolution on a non-uniform
wavelength grid via :func:`jax.lax.dynamic_slice` inside :func:`jax.vmap`.
All operations are JAX-jittable and compatible with numpyro inference engines.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer
from jax.typing import ArrayLike

#: Conversion factor from FWHM to Gaussian sigma: ``1 / (2 * sqrt(2 * ln 2))``.
_FWHM_TO_SIGMA: float = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def _padded_grid(x_fine: jnp.ndarray, half_width: int) -> jnp.ndarray:
    """Extend the fine grid by ``half_width`` points on each side.

    The grid is linearly extrapolated with the local spacing at each edge, so
    padded positions keep their true distance from the kernel centre and the
    Gaussian falls off naturally (see :func:`_lsf_convolve`).
    """
    dx_left = x_fine[1] - x_fine[0]
    dx_right = x_fine[-1] - x_fine[-2]
    return jnp.concatenate(
        [
            x_fine[0] + jnp.arange(-half_width, 0) * dx_left,
            x_fine,
            x_fine[-1] + jnp.arange(1, half_width + 1) * dx_right,
        ]
    )


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
    x_pad = _padded_grid(x_fine, half_width)
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


def _pixel_kernel(
    x_fine: ArrayLike, sigma_fine: ArrayLike, half_width: int, n_super: int
) -> jnp.ndarray:
    """Pixel-averaged form of the kernel applied by :func:`_lsf_convolve`.

    :func:`_lsf_convolve` followed by an average over the ``n_super`` fine
    points of each pixel is linear in the model, and its weights depend only
    on the grid and the LSF.  This returns those weights with the pixel
    average folded in, so that the convolved, pixel-averaged model is a short
    weighted sum per pixel (:func:`_apply_pixel_kernel`) with no exponentials.
    The ``n_super`` rows of a pixel have windows shifted by one fine point
    each, so their union spans ``2 * half_width + n_super`` consecutive
    points; it is stored in ``n_blocks`` blocks of ``n_super`` points.

    Parameters
    ----------
    x_fine, sigma_fine : array_like, shape ``(n_pixels * n_super,)``
        Fine-grid wavelengths and LSF sigmas, as for :func:`_lsf_convolve`.
    half_width : int
        Kernel half-width in fine-grid indices.
    n_super : int
        Fine points per pixel.

    Returns
    -------
    jnp.ndarray, shape ``(n_pixels, n_blocks, n_super)``
        Entry ``[p, a, b]`` is the weight of padded model point
        ``(p + a) * n_super + b`` in pixel ``p``.
    """
    x_fine = jnp.asarray(x_fine)
    sigma_fine = jnp.asarray(sigma_fine)
    n_fine = x_fine.shape[0]
    n_pixels = n_fine // n_super
    w = 2 * half_width + 1

    # Same padded grid and kernel as _lsf_convolve.
    x_pad = _padded_grid(x_fine, half_width)
    dx = jnp.stack([x_pad[k : k + n_fine] for k in range(w)], axis=1) - x_fine[:, None]
    kernel = jnp.exp(dx * dx * (-0.5 / (sigma_fine * sigma_fine))[:, None])
    kernel = kernel / (n_super * jnp.sum(kernel, axis=1, keepdims=True))

    # Sub-pixel row s of pixel p starts at padded index p * n_super + s.
    n_blocks = -(-(w + n_super - 1) // n_super)
    length = n_blocks * n_super
    rows = kernel.reshape(n_pixels, n_super, w)
    weights = jnp.zeros((n_pixels, length))
    for s in range(n_super):
        weights = weights + jnp.pad(rows[:, s, :], ((0, 0), (s, length - w - s)))
    return weights.reshape(n_pixels, n_blocks, n_super)


def _apply_pixel_kernel(
    weights: ArrayLike, model_fine: ArrayLike, half_width: int
) -> jnp.ndarray:
    """Convolve and pixel-average a fine-grid model with :func:`_pixel_kernel`.

    Parameters
    ----------
    weights : array_like, shape ``(n_pixels, n_blocks, n_super)``
        Output of :func:`_pixel_kernel`.
    model_fine : array_like, shape ``(..., n_pixels * n_super)``
        Intrinsic model on the fine grid; leading axes are components that
        share the kernel.
    half_width : int
        Kernel half-width used to build *weights*.

    Returns
    -------
    jnp.ndarray, shape ``(..., n_pixels)``
        Equal to ``_lsf_convolve(...)`` averaged over each pixel.
    """
    weights = jnp.asarray(weights)
    model_fine = jnp.asarray(model_fine)
    n_pixels, n_blocks, n_super = weights.shape
    lead = model_fine.shape[:-1]
    tail = (n_blocks - 1) * n_super - half_width
    # Edge-extend the model, as _lsf_convolve does.
    padded = jnp.concatenate(
        [
            jnp.broadcast_to(model_fine[..., :1], (*lead, half_width)),
            model_fine,
            jnp.broadcast_to(model_fine[..., -1:], (*lead, tail)),
        ],
        axis=-1,
    ).reshape(*lead, n_pixels + n_blocks - 1, n_super)
    out = jnp.zeros((*lead, n_pixels))
    for a in range(n_blocks):
        out = out + jnp.sum(weights[:, a, :] * padded[..., a : a + n_pixels, :], -1)
    return out


def _lsf_convolve_components(
    x_fine: ArrayLike, model_fine: ArrayLike, sigma_fine: ArrayLike, half_width: int
) -> jnp.ndarray:
    """:func:`_lsf_convolve` for a stack of components sharing one kernel.

    Accumulates the window one offset at a time, so the kernel is never
    materialised: memory is that of the components themselves rather than an
    ``(N, 2 * half_width + 1)`` matrix (per posterior draw under ``vmap``),
    and the kernel is evaluated once for all components.  Meant for
    prediction; reverse-mode differentiation through the unrolled loop works
    but is slow, so the likelihood uses :func:`_lsf_convolve`.

    Parameters
    ----------
    x_fine, sigma_fine : array_like, shape ``(N,)``
        Fine-grid wavelengths and LSF sigmas.
    model_fine : array_like, shape ``(..., N)``
        Components to convolve.
    half_width : int
        Kernel half-width in fine-grid indices.

    Returns
    -------
    jnp.ndarray, shape ``(..., N)``
    """
    x_fine = jnp.asarray(x_fine)
    model_fine = jnp.asarray(model_fine)
    sigma_fine = jnp.asarray(sigma_fine)
    n_fine = x_fine.shape[0]
    x_pad = _padded_grid(x_fine, half_width)
    widths = [(0, 0)] * (model_fine.ndim - 1) + [(half_width, half_width)]
    m_pad = jnp.pad(model_fine, widths, mode='edge')
    coeff = -0.5 / (sigma_fine * sigma_fine)
    num = jnp.zeros_like(model_fine)
    den = jnp.zeros_like(x_fine)
    for k in range(2 * half_width + 1):
        dx = x_pad[k : k + n_fine] - x_fine
        kernel = jnp.exp(dx * dx * coeff)
        num = num + kernel * m_pad[..., k : k + n_fine]
        den = den + kernel
    return num / den


def _is_constant(*arrays: ArrayLike) -> bool:
    """Whether every array is concrete (not a traced JAX value)."""
    return not any(isinstance(a, Tracer) for a in arrays)


def _convolve_to_pixels(
    x_fine: ArrayLike,
    model_fine: ArrayLike,
    sigma_fine: ArrayLike,
    half_width: int,
    n_super: int,
    *,
    cache: dict | None = None,
    key: object = None,
) -> jnp.ndarray:
    """LSF-convolve a fine-grid model and average it over each pixel.

    Numerically equivalent to ``_lsf_convolve(...)`` reshaped to
    ``(n_pixels, n_super)`` and averaged, but picks the cheapest route:

    - If ``x_fine`` and ``sigma_fine`` are concrete at trace time (the
      disperser's ``RScale`` and ``PixOffset`` are fixed), the pixel kernel
      is a constant.  It is computed once, optionally stored in *cache*, and
      applied as a weighted sum: no exponentials per model evaluation.
    - Otherwise, several components sharing one traced kernel go through
      :func:`_lsf_convolve_components`, which never materialises the kernel.
    - Otherwise (one component, traced kernel) :func:`_lsf_convolve` is used.

    Parameters
    ----------
    x_fine, sigma_fine : array_like, shape ``(n_pixels * n_super,)``
        Fine-grid wavelengths and LSF sigmas.
    model_fine : array_like, shape ``(..., n_pixels * n_super)``
        Intrinsic model; leading axes are components sharing the kernel.
    half_width, n_super : int
        Kernel half-width (fine indices) and fine points per pixel.
    cache : dict, optional
        Store for constant pixel kernels, reused across traces.
    key : hashable, optional
        Key identifying this grid within *cache*.

    Returns
    -------
    jnp.ndarray, shape ``(..., n_pixels)``
    """
    model_fine = jnp.asarray(model_fine)
    static = _is_constant(x_fine, sigma_fine)
    if static:
        x_np = np.asarray(x_fine)
        s_np = np.asarray(sigma_fine)
        # The stamp guards a cached kernel against a changed grid or LSF.
        stamp = (x_np.shape[0], half_width, n_super, x_np.tobytes(), s_np.tobytes())
        entry = cache.get(key) if cache is not None else None
        if entry is None or entry[0] != stamp:
            with jax.ensure_compile_time_eval():
                weights = np.asarray(_pixel_kernel(x_np, s_np, half_width, n_super))
            entry = (stamp, weights)
            if cache is not None:
                cache[key] = entry
        return _apply_pixel_kernel(entry[1], model_fine, half_width)
    if model_fine.ndim > 1:
        conv = _lsf_convolve_components(x_fine, model_fine, sigma_fine, half_width)
    else:
        conv = _lsf_convolve(x_fine, model_fine, sigma_fine, half_width)
    return conv.reshape(*model_fine.shape[:-1], -1, n_super).mean(axis=-1)


def _convolve_stacks_to_pixels(
    x_fine: ArrayLike,
    stacks: Sequence[ArrayLike],
    sigma_fine: ArrayLike,
    half_width: int,
    n_super: int,
    *,
    cache: dict | None = None,
    key: object = None,
) -> list[jnp.ndarray]:
    """:func:`_convolve_to_pixels` for several component stacks at once.

    A constant kernel is applied to each stack in turn, so the stacks are
    never joined into one array; a traced kernel is evaluated once on their
    concatenation.

    Parameters
    ----------
    x_fine, sigma_fine, half_width, n_super, cache, key
        As for :func:`_convolve_to_pixels`.
    stacks : sequence of array_like
        Each of shape ``(n_components, n_pixels * n_super)``.

    Returns
    -------
    list of jnp.ndarray
        One ``(n_components, n_pixels)`` array per stack.
    """
    arrays = [jnp.asarray(m) for m in stacks]
    if _is_constant(x_fine, sigma_fine):
        return [
            _convolve_to_pixels(
                x_fine, m, sigma_fine, half_width, n_super, cache=cache, key=key
            )
            for m in arrays
        ]
    joined = _convolve_to_pixels(
        x_fine, jnp.concatenate(arrays, axis=0), sigma_fine, half_width, n_super
    )
    sizes = np.cumsum([m.shape[0] for m in arrays])[:-1]
    return jnp.split(joined, sizes, axis=0)
