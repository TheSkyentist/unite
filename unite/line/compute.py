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
