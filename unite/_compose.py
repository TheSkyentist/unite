"""Model composition from pre-evaluated line profiles and continuum.

Private module providing pure JAX functions that combine pre-evaluated
emission profiles, absorption profiles, and continuum into a total model
prediction.  Used by both :mod:`unite.model` (likelihood) and
:mod:`unite.compute` (posterior decomposition).
"""

from __future__ import annotations

import jax.numpy as jnp


def compose_from_profiles(
    profiles, flux_per_line, tau_per_line, is_absorption, continuum, absorber_position
):
    """Compose the total model from pre-evaluated profiles and continuum.

    Parameters
    ----------
    profiles : jnp.ndarray, shape ``(n_lines, n_points)``
        Normalised profile values at wavelength points (from
        :func:`~unite.line.compute.evaluate_lines` or pixel-averaged
        from :func:`~unite.line.compute.integrate_lines`).
    flux_per_line : jnp.ndarray, shape ``(n_lines,)``
        Flux per line (zero for tau-parametrized lines).
    tau_per_line : jnp.ndarray, shape ``(n_lines,)``
        Optical depth per line (zero for flux-parametrized lines).
    is_absorption : jnp.ndarray, shape ``(n_lines,)``
        Boolean mask: True for tau-parametrized lines.
    continuum : jnp.ndarray, shape ``(n_points,)``
        Continuum flux density at each point.
    absorber_position : str
        One of ``'foreground'``, ``'behind_lines'``, ``'behind_continuum'``.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Total model flux density (before ``flux_scale`` and normalisation).
    """
    emission, transmission = _emission_and_transmission(
        profiles, flux_per_line, tau_per_line, is_absorption
    )
    return _combine(emission, transmission, continuum, absorber_position)


def compose_leave_one_out(
    profiles, flux_per_line, tau_per_line, is_absorption, continuum, absorber_position
):
    """Compose the total model and exact per-line contributions.

    For each line *j*, computes ``total - total_without_j`` to give the
    exact flux contribution (positive for emission, negative for
    absorption).  Profile evaluation is shared; only the composition
    is repeated per line.

    Parameters
    ----------
    profiles : jnp.ndarray, shape ``(n_lines, n_points)``
        Normalised profile values at wavelength points.
    flux_per_line : jnp.ndarray, shape ``(n_lines,)``
        Flux per line (zero for tau-parametrized lines).
    tau_per_line : jnp.ndarray, shape ``(n_lines,)``
        Optical depth per line (zero for flux-parametrized lines).
    is_absorption : jnp.ndarray, shape ``(n_lines,)``
        Boolean mask: True for tau-parametrized lines.
    continuum : jnp.ndarray, shape ``(n_points,)``
        Continuum flux density at each point.
    absorber_position : str
        One of ``'foreground'``, ``'behind_lines'``, ``'behind_continuum'``.

    Returns
    -------
    total : jnp.ndarray, shape ``(n_points,)``
        Full model flux density.
    per_line_delta : jnp.ndarray, shape ``(n_lines, n_points)``
        Per-line contribution: ``total - total_without_line_j``.
        Positive for emission lines, negative for absorption lines.
    """
    emission, transmission = _emission_and_transmission(
        profiles, flux_per_line, tau_per_line, is_absorption
    )
    total = _combine(emission, transmission, continuum, absorber_position)

    n_lines = profiles.shape[0]
    per_line_delta = jnp.zeros_like(profiles)

    for j in range(n_lines):
        # Build leave-one-out: zero this line's flux or tau.
        flux_loo = flux_per_line.at[j].set(
            jnp.where(is_absorption[j], flux_per_line[j], 0.0)
        )
        tau_loo = tau_per_line.at[j].set(
            jnp.where(is_absorption[j], 0.0, tau_per_line[j])
        )
        emission_loo, transmission_loo = _emission_and_transmission(
            profiles, flux_loo, tau_loo, is_absorption
        )
        total_loo = _combine(
            emission_loo, transmission_loo, continuum, absorber_position
        )
        per_line_delta = per_line_delta.at[j].set(total - total_loo)

    return total, per_line_delta


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------


def _emission_and_transmission(profiles, flux_per_line, tau_per_line, is_absorption):
    """Compute emission sum and combined transmission from profiles.

    Parameters
    ----------
    profiles : jnp.ndarray, shape ``(n_lines, n_points)``
    flux_per_line : jnp.ndarray, shape ``(n_lines,)``
    tau_per_line : jnp.ndarray, shape ``(n_lines,)``
    is_absorption : jnp.ndarray, shape ``(n_lines,)``

    Returns
    -------
    emission : jnp.ndarray, shape ``(n_points,)``
        Sum of flux-weighted emission profiles.
    transmission : jnp.ndarray, shape ``(n_points,)``
        Product of per-line transmissions ``exp(-tau_j * phi_j)``.
    """
    emission_phi = jnp.where(is_absorption[:, None], 0.0, profiles)
    emission = (flux_per_line[:, None] * emission_phi).sum(axis=0)

    absorption_phi = jnp.where(is_absorption[:, None], profiles, 0.0)
    total_tau = (tau_per_line[:, None] * absorption_phi).sum(axis=0)
    transmission = jnp.exp(-total_tau)

    return emission, transmission


def _combine(emission, transmission, continuum, absorber_position):
    """Combine emission, transmission, and continuum by absorber position.

    Parameters
    ----------
    emission : jnp.ndarray, shape ``(n_points,)``
    transmission : jnp.ndarray, shape ``(n_points,)``
    continuum : jnp.ndarray, shape ``(n_points,)``
    absorber_position : str

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
    """
    if absorber_position == 'foreground':
        return transmission * (emission + continuum)
    elif absorber_position == 'behind_lines':
        return emission + transmission * continuum
    else:  # behind_continuum
        return transmission * emission + continuum
