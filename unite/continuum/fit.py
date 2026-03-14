"""Least-squares fitting utilities for continuum forms.

Provides :func:`fit_continuum_form` which fits a
:class:`~unite.continuum.library.ContinuumForm` to data using weighted
least squares (for linear forms) or Gauss-Newton iteration (for nonlinear
forms).  All computation uses JAX — no additional dependencies required.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from unite.continuum.library import ContinuumForm


@dataclass
class ContinuumFitResult:
    """Result of a continuum form least-squares fit.

    Attributes
    ----------
    model : jnp.ndarray
        Best-fit model values at the *fitted* wavelengths (i.e. the
        unmasked pixels passed to :func:`fit_continuum_form`).
    params : dict of str to float
        Best-fit parameter values (including ``norm_wav``).
    chi2_red : float or None
        Reduced chi-squared of the fit, or ``None`` if degrees of
        freedom ≤ 0.
    dof : int
        Degrees of freedom (n_data - n_fitted_params).
    """

    model: jnp.ndarray
    params: dict[str, float]
    chi2_red: float | None
    dof: int


def fit_continuum_form(
    form: ContinuumForm,
    wavelength: ArrayLike,
    flux: ArrayLike,
    error: ArrayLike,
    center: float,
    obs_low: float,
    obs_high: float,
    norm_wav: float | None = None,
) -> ContinuumFitResult:
    """Fit a continuum form to data using least squares.

    For :attr:`~ContinuumForm.is_linear` forms the solution is exact
    (weighted least squares via ``jnp.linalg.lstsq``).  For nonlinear
    forms, a Gauss-Newton iteration is used with ``jax.jacfwd`` for
    the Jacobian.

    Parameters
    ----------
    form : ContinuumForm
        Continuum functional form to fit.
    wavelength : ArrayLike
        Wavelength array (1-D).
    flux : ArrayLike
        Flux density array (same length as *wavelength*).
    error : ArrayLike
        Flux density uncertainty (same length as *wavelength*).
    center : float
        Region midpoint wavelength (passed to ``form.evaluate``).
    obs_low : float
        Lower observed-frame wavelength bound of the region.
    obs_high : float
        Upper observed-frame wavelength bound of the region.
    norm_wav : float, optional
        Reference wavelength.  Defaults to *center*.

    Returns
    -------
    ContinuumFitResult
    """
    wavelength = jnp.asarray(wavelength, dtype=float)
    flux = jnp.asarray(flux, dtype=float)
    error = jnp.asarray(error, dtype=float)

    if norm_wav is None:
        norm_wav = center
    nw = norm_wav

    param_names = list(form.param_names())
    fitted_names = [p for p in param_names if p != 'norm_wav']
    n_params = len(fitted_names)
    n_data = wavelength.shape[0]
    dof = n_data - n_params

    if form.is_linear:
        return _fit_linear(
            form,
            wavelength,
            flux,
            error,
            center,
            obs_low,
            obs_high,
            nw,
            fitted_names,
            dof,
        )
    return _fit_nonlinear(
        form, wavelength, flux, error, center, obs_low, obs_high, nw, fitted_names, dof
    )


# ------------------------------------------------------------------
# Linear solver
# ------------------------------------------------------------------


def _fit_linear(
    form, wavelength, flux, error, center, obs_low, obs_high, nw, fitted_names, dof
):
    """Weighted least squares for linear forms."""
    # Build design matrix: evaluate form with one parameter = 1, rest = 0.
    columns = []
    for target_name in fitted_names:
        params = {p: 0.0 for p in form.param_names()}
        params['norm_wav'] = nw
        params[target_name] = 1.0
        col = form.evaluate(wavelength, center, params, obs_low, obs_high)
        columns.append(col)

    design = jnp.stack(columns, axis=-1)  # (n_data, n_params)
    weights = 1.0 / error
    design_w = design * weights[:, None]
    bw = flux * weights

    coeffs, _, _, _ = jnp.linalg.lstsq(design_w, bw)

    params_dict: dict[str, float] = {
        name: float(coeffs[i]) for i, name in enumerate(fitted_names)
    }
    params_dict['norm_wav'] = nw

    model = design @ coeffs

    residuals = (flux - model) / error
    chi2_red = float(jnp.sum(residuals**2) / dof) if dof > 0 else None

    return ContinuumFitResult(
        model=model, params=params_dict, chi2_red=chi2_red, dof=dof
    )


# ------------------------------------------------------------------
# Nonlinear solver (Gauss-Newton)
# ------------------------------------------------------------------


def _fit_nonlinear(
    form,
    wavelength,
    flux,
    error,
    center,
    obs_low,
    obs_high,
    nw,
    fitted_names,
    dof,
    max_iter=30,
    tol=1e-8,
):
    """Gauss-Newton iteration for nonlinear forms."""
    x = _initial_guess(fitted_names, flux)

    def residual_fn(param_vec):
        params_dict = {name: param_vec[i] for i, name in enumerate(fitted_names)}
        params_dict['norm_wav'] = nw
        model = form.evaluate(wavelength, center, params_dict, obs_low, obs_high)
        return (flux - model) / error

    for _ in range(max_iter):
        r = residual_fn(x)
        jac = jax.jacfwd(residual_fn)(x)
        dx, _, _, _ = jnp.linalg.lstsq(jac, -r)
        x = x + dx
        if float(jnp.max(jnp.abs(dx))) < tol * (float(jnp.max(jnp.abs(x))) + 1e-10):
            break

    params_dict: dict[str, float] = {
        name: float(x[i]) for i, name in enumerate(fitted_names)
    }
    params_dict['norm_wav'] = nw

    # Evaluate final model.
    full_params = dict(params_dict)
    model = form.evaluate(wavelength, center, full_params, obs_low, obs_high)

    residuals = (flux - model) / error
    chi2_red = float(jnp.sum(residuals**2) / dof) if dof > 0 else None

    return ContinuumFitResult(
        model=model, params=params_dict, chi2_red=chi2_red, dof=dof
    )


def _initial_guess(fitted_names: list[str], flux: jnp.ndarray) -> jnp.ndarray:
    """Generate initial parameter guess for nonlinear optimisation."""
    median_flux = float(jnp.median(jnp.abs(flux)))
    x0 = []
    for name in fitted_names:
        if name == 'scale':
            x0.append(median_flux)
        elif name == 'temperature':
            x0.append(5000.0)
        else:
            x0.append(0.0)
    return jnp.array(x0)
