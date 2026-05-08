"""Model composition from pre-evaluated line profiles and continuum.

Private module providing pure JAX functions that combine pre-evaluated
emission profiles, absorption profiles, and continuum into a total model
prediction.  Used by both :mod:`unite.model` (likelihood) and
:mod:`unite.compute` (posterior decomposition).
"""

from __future__ import annotations

import jax.numpy as jnp


def compose_from_profiles(
    profiles,
    flux_per_line,
    tau_per_line,
    is_tau,
    applies_matrix,
    cont_applies,
    continuum,
):
    """Compose the total model from pre-evaluated profiles and continuum.

    Each emission/continuum component has its own effective transmission
    determined by which tau absorbers have a higher zorder (i.e. are in
    front of it).  ``applies_matrix[j, k]`` encodes this: True when tau
    line *k* applies to component *j*.

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
    is_tau : jnp.ndarray, shape ``(n_lines,)``
        Boolean mask: True for tau-parametrized lines.
    applies_matrix : jnp.ndarray, shape ``(n_lines, n_lines)``
        Static boolean matrix.  ``applies_matrix[j, k]`` is True when tau
        line *k* applies to emission line *j* (i.e. ``zorder_k > zorder_j``
        and ``is_tau[k]``).
    cont_applies : jnp.ndarray, shape ``(n_lines,)``
        Boolean mask: True for tau lines whose zorder exceeds the
        continuum zorder (i.e. they attenuate the continuum).
    continuum : jnp.ndarray, shape ``(n_points,)``
        Continuum flux density at each point.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Total model flux density (before ``flux_scale`` and normalisation).
    """
    emission_phi = jnp.where(is_tau[:, None], 0.0, profiles)  # (n_lines, n_points)
    absorption_phi = jnp.where(is_tau[:, None], profiles, 0.0)  # (n_lines, n_points)

    # Per-component effective tau field:
    # tau_fields[j, p] = sum_k applies_matrix[j,k] * tau_per_line[k] * absorption_phi[k,p]
    tau_fields = (
        applies_matrix * tau_per_line[None, :]
    ) @ absorption_phi  # (n_lines, n_points)
    t_eff = jnp.exp(-tau_fields)  # (n_lines, n_points)

    attenuated_emission = (flux_per_line[:, None] * emission_phi * t_eff).sum(axis=0)

    cont_tau = (cont_applies * tau_per_line) @ absorption_phi  # (n_points,)
    t_cont = jnp.exp(-cont_tau)  # (n_points,)

    return attenuated_emission + t_cont * continuum


def compose_leave_one_out(
    profiles,
    flux_per_line,
    tau_per_line,
    is_tau,
    applies_matrix,
    cont_applies,
    continuum,
):
    """Compose the total model and per-line contributions.

    For **emission** lines, ``per_line_delta[j] = flux_per_line[j] * profiles[j]``
    — the intrinsic (un-attenuated) profile contribution.  Using intrinsic
    emission preserves the identity
    ``sum(per_line_delta) + continuum == total`` for all zorder configurations,
    because the absorption delta absorbs the difference between total and
    the no-absorber baseline.

    For **absorption** lines, ``per_line_delta[j] = total - total_without_j``
    — the exact flux removed by that absorber (negative).  Removing tau *j*
    changes all effective transmissions that depend on it.

    Parameters
    ----------
    profiles : jnp.ndarray, shape ``(n_lines, n_points)``
        Normalised profile values at wavelength points.
    flux_per_line : jnp.ndarray, shape ``(n_lines,)``
        Flux per line (zero for tau-parametrized lines).
    tau_per_line : jnp.ndarray, shape ``(n_lines,)``
        Optical depth per line (zero for flux-parametrized lines).
    is_tau : jnp.ndarray, shape ``(n_lines,)``
        Boolean mask: True for tau-parametrized lines.
    applies_matrix : jnp.ndarray, shape ``(n_lines, n_lines)``
        Static boolean matrix (see :func:`compose_from_profiles`).
    cont_applies : jnp.ndarray, shape ``(n_lines,)``
        Boolean mask: tau lines that attenuate the continuum.
    continuum : jnp.ndarray, shape ``(n_points,)``
        Continuum flux density at each point.

    Returns
    -------
    total : jnp.ndarray, shape ``(n_points,)``
        Full model flux density.
    per_line_delta : jnp.ndarray, shape ``(n_lines, n_points)``
        Per-line contribution.  Emission lines: attenuated flux profile
        (positive).  Absorption lines: flux removed (negative).
    """
    emission_phi = jnp.where(is_tau[:, None], 0.0, profiles)  # (n_lines, n_points)
    absorption_phi = jnp.where(is_tau[:, None], profiles, 0.0)  # (n_lines, n_points)

    tau_fields = (applies_matrix * tau_per_line[None, :]) @ absorption_phi
    t_eff = jnp.exp(-tau_fields)  # (n_lines, n_points)

    # Precompute attenuated per-line terms; reused in the tau LOO loop below.
    attenuated_per_line = (
        flux_per_line[:, None] * emission_phi * t_eff
    )  # (n_lines, n_points)
    attenuated_emission = attenuated_per_line.sum(axis=0)

    cont_tau = (cont_applies * tau_per_line) @ absorption_phi
    t_cont = jnp.exp(-cont_tau)
    total = attenuated_emission + t_cont * continuum

    n_lines = profiles.shape[0]
    delta_list = []

    for j in range(n_lines):
        if bool(is_tau[j]):
            # Rank-1 removal: removing tau j subtracts its contribution from
            # tau_fields, which is equivalent to multiplying t_eff by
            # exp(+contribution).  Avoids recomputing the full matmul.
            contrib = applies_matrix[:, j, None] * (tau_per_line[j] * absorption_phi[j])
            attenuated_loo = (attenuated_per_line * jnp.exp(contrib)).sum(axis=0)
            cont_contrib = cont_applies[j] * (tau_per_line[j] * absorption_phi[j])
            total_loo = attenuated_loo + t_cont * jnp.exp(cont_contrib) * continuum
            delta_list.append(total - total_loo)
        else:
            # Emission: intrinsic profile contribution (no transmission factor).
            # Preserves: sum(deltas) + continuum == total for all zorder configs.
            delta_list.append(flux_per_line[j] * emission_phi[j])

    return total, jnp.stack(delta_list)
