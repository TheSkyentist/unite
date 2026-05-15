"""Line profile evaluation and analytic integration.

Provides two vmapped dispatch functions:

- :func:`evaluate_lines` — pointwise evaluation of all line profiles at
  arbitrary wavelength arrays (used by the Gauss-Legendre quadrature path).
- :func:`integrate_lines` — exact CDF-based integration of all
  line profiles over pixel bins (used by the analytic integration path).

Both dispatch to per-profile JAX kernels via ``lax.switch`` keyed on
:attr:`~unite.line.library.Profile.code`.
"""

from __future__ import annotations

import jax

from unite.line.library import _EVALUATE_BRANCHES, _INTEGRATE_BRANCHES

# -------------------------------------------------------------------
# Pointwise evaluation
# -------------------------------------------------------------------


def _evaluate_single_line(wavelength, center, lsf_fwhm, p0, p1, p2, code):
    """Evaluate one line profile at wavelength points, dispatched by ``code``.

    Parameters
    ----------
    wavelength : jnp.ndarray, shape ``(n_points,)``
        Wavelength points at which to evaluate.
    center, lsf_fwhm, p0, p1, p2 : float
        Per-line scalars.
    code : int
        ``Profile.code`` for the line.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Normalised profile value at each wavelength point.
    """
    return jax.lax.switch(
        code, _EVALUATE_BRANCHES, wavelength, center, lsf_fwhm, p0, p1, p2
    )


evaluate_lines = jax.vmap(_evaluate_single_line, in_axes=(None, 0, 0, 0, 0, 0, 0))
"""Vectorised evaluation over all lines simultaneously.

Input shapes: ``wavelength (n_points,)``, all others ``(n_lines,)``.
Output shape: ``(n_lines, n_points)``.
"""


evaluate_lines_at_own_centers = jax.vmap(
    _evaluate_single_line, in_axes=(0, 0, 0, 0, 0, 0, 0)
)
"""Evaluate each line at its own center wavelength.

Each input has shape ``(n_lines,)``; line *j* is evaluated only at
``wavelength[j]``.  Output shape: ``(n_lines,)``.  Cheaper than calling
:data:`evaluate_lines` and taking the diagonal — avoids the O(n²) work
of evaluating every profile at every center.
"""


def _peak_to_area_tau(tau_per_line, centers, p0, p1, p2, profile_codes, is_tau, *, _eval_fn=None):
    """Convert peak optical depth τ₀ to area-tau for :func:`~unite._compose.compose_from_profiles`.

    Tau tokens parametrize the peak optical depth of the *intrinsic* (pre-LSF)
    profile at the nominal line center.  The compose path expects area-tau
    (equal to the integral of the optical-depth profile over wavelength), so
    each value must be divided by the intrinsic profile peak φ_intrinsic(center).

    A tiny stand-in LSF of 1e-10 avoids 0/0 singularities in EMG-based
    profiles (Laplace, SEMG) while leaving φ numerically equal to the
    intrinsic limit.  Only tau-parametrized lines are converted; emission
    lines (is_tau=False) pass through unchanged via the ``jnp.where`` mask.

    Parameters
    ----------
    tau_per_line : jnp.ndarray, shape ``(n_lines,)``
        Raw optical depths as sampled (peak-tau convention).
    centers : jnp.ndarray, shape ``(n_lines,)``
        Observed-frame line centers in canonical wavelength units.
    p0, p1, p2 : jnp.ndarray, shape ``(n_lines,)``
        Per-line profile parameter slots in canonical wavelength units.
    profile_codes : jnp.ndarray, shape ``(n_lines,)``
        Integer dispatch codes from :attr:`~unite.line.library.Profile.code`.
    is_tau : jnp.ndarray, shape ``(n_lines,)``
        Boolean mask — True for absorption lines.
    _eval_fn : callable or None, optional
        Specialized vmapped evaluate-at-own-centers function.  Defaults to
        the module-level :data:`evaluate_lines_at_own_centers`.

    Returns
    -------
    jnp.ndarray, shape ``(n_lines,)``
        Area-tau per line (emission lines unchanged).
    """
    import jax.numpy as jnp

    if _eval_fn is None:
        _eval_fn = evaluate_lines_at_own_centers
    _tiny_lsf = jnp.full_like(centers, 1e-10)
    _phi_center = _eval_fn(
        centers, centers, _tiny_lsf, p0, p1, p2, profile_codes
    )
    _phi_safe = jnp.where(is_tau, _phi_center, 1.0)
    return tau_per_line / _phi_safe


# -------------------------------------------------------------------
# Analytic (CDF-based) integration
# -------------------------------------------------------------------


def _integrate_single_line(low, high, center, lsf_fwhm, p0, p1, p2, code):
    """Integrate one line profile over pixel bins analytically.

    Dispatches to the correct CDF-based kernel via ``lax.switch`` on
    ``code``.  All FWHM parameters are in wavelength units; shape
    parameters (h3, h4) are dimensionless.

    Parameters
    ----------
    low, high : jnp.ndarray, shape ``(n_pixels,)``
        Pixel bin edges.
    center, lsf_fwhm, p0, p1, p2 : float
        Per-line scalars: observed center, LSF FWHM, and three profile
        parameter slots (in :meth:`Profile.param_names` order).  Slots
        unused by a given profile receive zero.
    code : int
        ``Profile.code`` for the line.

    Returns
    -------
    jnp.ndarray, shape ``(n_pixels,)``
        Integrated profile fraction per pixel bin.
    """
    return jax.lax.switch(
        code, _INTEGRATE_BRANCHES, low, high, center, lsf_fwhm, p0, p1, p2
    )


integrate_lines = jax.vmap(
    _integrate_single_line, in_axes=(None, None, 0, 0, 0, 0, 0, 0)
)
"""Vectorised analytic integration over all lines simultaneously.

Input shapes: ``low/high (n_pixels,)``, all others ``(n_lines,)``.
Output shape: ``(n_lines, n_pixels)``.
"""
