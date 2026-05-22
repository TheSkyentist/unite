"""Line profile evaluation and analytic integration.

Provides two vmapped dispatch functions:

- :func:`evaluate_lines` — pointwise evaluation of all line profiles at
  arbitrary wavelength arrays (used by the convolution-mode fine-grid path).
- :func:`integrate_lines` — exact CDF-based integration of all
  line profiles over pixel bins (used by the analytic integration path).

Both dispatch to per-profile JAX kernels via ``lax.switch`` keyed on
:attr:`~unite.line.library.Profile.code`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from unite._utils import C_KMS
from unite.line.library import _EVALUATE_BRANCHES, _INTEGRATE_BRANCHES

# -------------------------------------------------------------------
# Tiny stand-in LSF for avoiding 0/0 in EMG-based profile peaks
# -------------------------------------------------------------------

_TINY_LSF: float = 1e-10


# -------------------------------------------------------------------
# Per-line parameter arrays from ConfigMatrices
# -------------------------------------------------------------------


def _build_line_params(cm, context, n_lines, z_sys):
    """Unpack :class:`~unite.line.config.ConfigMatrices` into per-line arrays.

    Shared across :func:`~unite.model.unite_model` (at sample time) and
    :func:`~unite.compute.evaluate_model` (at posterior-evaluation time).

    Parameters
    ----------
    cm : ConfigMatrices
        Precomputed parameter matrices and line metadata.
    context : dict of str to array
        Parameter values keyed by numpyro site name.
    n_lines : int
        Number of lines.
    z_sys : float
        Systemic redshift.

    Returns
    -------
    flux_per_line, tau_per_line, centers, p0, p1, p2 : jnp.ndarray
        All shape ``(n_lines,)``.
    """
    if cm.flux_names:
        flux_per_line = (
            jnp.stack([context[n] for n in cm.flux_names])
            @ cm.flux_matrix
            * cm.strengths
        )
    else:
        flux_per_line = jnp.zeros(n_lines)

    if cm.tau_names:
        tau_per_line = (
            jnp.stack([context[n] for n in cm.tau_names]) @ cm.tau_matrix * cm.strengths
        )
    else:
        tau_per_line = jnp.zeros(n_lines)

    z_vec = jnp.stack([context[n] for n in cm.z_names])
    z_per_line = z_vec @ cm.z_matrix
    centers = cm.wavelengths * (1.0 + z_sys + z_per_line)

    p0_kms = (
        jnp.stack([context[n] for n in cm.p0_names]) @ cm.p0_matrix
        if cm.p0_names
        else jnp.zeros(n_lines)
    )
    p0 = centers * p0_kms / C_KMS

    p1v_kms = (
        jnp.stack([context[n] for n in cm.p1v_names]) @ cm.p1v_matrix
        if cm.p1v_names
        else jnp.zeros(n_lines)
    )
    p1v = centers * p1v_kms / C_KMS
    p1d = (
        jnp.stack([context[n] for n in cm.p1d_names]) @ cm.p1d_matrix
        if cm.p1d_names
        else jnp.zeros(n_lines)
    )
    p1 = p1v + p1d

    p2v_kms = (
        jnp.stack([context[n] for n in cm.p2v_names]) @ cm.p2v_matrix
        if cm.p2v_names
        else jnp.zeros(n_lines)
    )
    p2d = (
        jnp.stack([context[n] for n in cm.p2d_names]) @ cm.p2d_matrix
        if cm.p2d_names
        else jnp.zeros(n_lines)
    )
    p2 = centers * p2v_kms / C_KMS + p2d

    return flux_per_line, tau_per_line, centers, p0, p1, p2


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


def _peak_to_area_tau(
    tau_per_line, centers, p0, p1, p2, profile_codes, is_tau, *, _eval_fn=None
):
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
    if _eval_fn is None:
        _eval_fn = evaluate_lines_at_own_centers
    _tiny_lsf = jnp.full_like(centers, _TINY_LSF)
    _phi_center = _eval_fn(centers, centers, _tiny_lsf, p0, p1, p2, profile_codes)
    _phi_safe = jnp.where(is_tau, _phi_center, 1.0)
    return tau_per_line / _phi_safe


# -------------------------------------------------------------------
# Analytic (CDF-based) integration
# -------------------------------------------------------------------


def _integrate_single_line(edges, center, lsf_fwhm, p0, p1, p2, code):
    """Cumulative line profile at edges, dispatched by ``code``.

    Dispatches to the correct kernel via ``lax.switch`` on ``code``.
    The returned array has the same length as ``edges`` — applying
    ``jnp.diff`` (then masking inter-pixel gaps) recovers the per-pixel
    profile integral.  All FWHM parameters are in wavelength units;
    shape parameters (h3, h4, alpha) are dimensionless.

    Parameters
    ----------
    edges : jnp.ndarray, shape ``(E,)``
        Unique pixel edges (shared across all lines of a spectrum).
    center, p0, p1, p2 : float
        Per-line scalars: observed center wavelength and three profile
        parameter slots (in :meth:`Profile.param_names` order).  Slots
        unused by a given profile receive zero.
    lsf_fwhm : jnp.ndarray, shape ``(E,)``
        Instrumental LSF FWHM evaluated at each edge (shared across all
        lines of a spectrum).
    code : int
        ``Profile.code`` for the line.

    Returns
    -------
    jnp.ndarray, shape ``(E,)``
        Cumulative profile array evaluated at the edges.
    """
    return jax.lax.switch(
        code, _INTEGRATE_BRANCHES, edges, center, lsf_fwhm, p0, p1, p2
    )


integrate_lines = jax.vmap(_integrate_single_line, in_axes=(None, 0, None, 0, 0, 0, 0))
"""Vectorised cumulative integration over all lines simultaneously.

Input shapes: ``edges (E,)`` and ``lsf_fwhm (E,)`` (both shared across lines);
``center / p0 / p1 / p2 / code`` each ``(n_lines,)``.  Output shape:
``(n_lines, E)``.  Applying ``jnp.diff`` along axis 1 (then masking
inter-pixel gap entries) recovers per-pixel profile integrals.
"""
