"""Model composition from pre-evaluated line profiles and continuum.

Private module providing pure JAX functions that combine pre-evaluated
emission profiles, absorption profiles, and continuum into a total model
prediction.  Used by both :mod:`unite.model` (likelihood) and
:mod:`unite.compute` (posterior decomposition).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.core import Tracer


def _effective_depth(tau_per_line, absorption_phi, cover_per_line):
    """Per-absorber effective optical depth including covering fraction.

    A partially covering absorber transmits ``T = 1 - cf * (1 - exp(-x))`` with
    ``x = tau * phi``.  Writing ``T = exp(-x_eff)`` lets absorbers keep
    composing by summing optical depths (independent screens multiply their
    transmissions), so all downstream code is unchanged.  ``cf = 1`` gives
    ``x_eff = x`` exactly.

    ``cf = 1`` is special-cased: the general formula would evaluate
    ``log1p(-1) = -inf`` once ``exp(-x)`` underflows relative to 1 (``x`` of
    a few tens), which is reachable in mixed configurations where an absorber
    without a covering-fraction token implicitly has ``cf = 1`` while another
    absorber's ``cf`` is free.  The ``where`` guard also keeps gradients finite.

    Parameters
    ----------
    tau_per_line : jnp.ndarray, shape ``(n_lines,)``
        Area-tau per line (zero for emission lines).
    absorption_phi : jnp.ndarray, shape ``(n_lines, n_points)``
        Absorption profiles (zero rows for emission lines).
    cover_per_line : jnp.ndarray, shape ``(n_lines,)``
        Covering fraction per line in ``[0, 1]``.

    Returns
    -------
    jnp.ndarray, shape ``(n_lines, n_points)``
        Effective optical depth ``-log(1 - cf * (1 - exp(-tau * phi)))``.
    """
    x = tau_per_line[:, None] * absorption_phi
    full = cover_per_line[:, None] >= 1.0
    cf_safe = jnp.where(full, 0.0, cover_per_line[:, None])
    return jnp.where(full, x, -jnp.log1p(cf_safe * jnp.expm1(-x)))


def _attenuation_classes(is_tau, applies_matrix, cont_applies):
    """Group emission lines and continuum groups by the absorbers in front.

    A component's transmission depends only on *which* tau lines attenuate
    it.  Components with the same set share one transmission curve, and the
    number of distinct sets is the number of distinct depth orderings
    (usually one or two), not the number of lines.  The grouping is static,
    so it is resolved here in NumPy at trace time.

    Parameters
    ----------
    is_tau : array-like of bool, shape ``(n_lines,)``
    applies_matrix : array-like of bool, shape ``(n_lines, n_lines)``
    cont_applies : array-like of bool, shape ``(n_groups, n_lines)``

    Returns
    -------
    tuple or None
        ``(tau_idx, patterns, line_class, cont_class)`` with ``tau_idx`` the
        indices of the tau lines, ``patterns`` a bool ``(n_classes, n_lines)``
        array (True where a tau line attenuates the class), ``line_class`` a
        0/1 ``(n_classes, n_lines)`` array assigning emission lines to
        classes and ``cont_class`` a 0/1 ``(n_classes, n_groups)`` array
        assigning continuum groups.  ``None`` if any input is a traced
        value, in which case the caller uses the dense per-line form.
    """
    inputs = (is_tau, applies_matrix, cont_applies)
    if any(isinstance(a, Tracer) for a in inputs):
        return None
    tau = np.asarray(is_tau, dtype=bool)
    app = np.asarray(applies_matrix, dtype=bool) & tau[None, :]
    capp = np.atleast_2d(np.asarray(cont_applies, dtype=bool)) & tau[None, :]
    em_idx = np.flatnonzero(~tau)
    rows = np.concatenate([app[em_idx], capp], axis=0)
    patterns, inverse = np.unique(rows, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    n_classes = patterns.shape[0]
    line_class = np.zeros((n_classes, tau.shape[0]))
    line_class[inverse[: em_idx.size], em_idx] = 1.0
    cont_class = np.zeros((n_classes, capp.shape[0]))
    cont_class[inverse[em_idx.size :], np.arange(capp.shape[0])] = 1.0
    return np.flatnonzero(tau), patterns, line_class, cont_class


def _class_terms(
    classes, profiles, flux_per_line, tau_per_line, continuum, cover_per_line
):
    """Attenuated flux of each attenuation class and the tau-line depths.

    Returns
    -------
    attenuated : jnp.ndarray, shape ``(n_classes, n_points)``
        ``exp(-tau field) * (emission + continuum)`` of each class; the
        total model is its sum over classes.
    depth : jnp.ndarray, shape ``(n_tau, n_points)``
        Effective optical depth of each tau line (in ``tau_idx`` order).
    """
    tau_idx, patterns, line_class, cont_class = classes
    depth_tau = tau_per_line[tau_idx]
    if cover_per_line is None:
        depth = depth_tau[:, None] * profiles[tau_idx]
    else:
        depth = _effective_depth(depth_tau, profiles[tau_idx], cover_per_line[tau_idx])
    fields = patterns[:, tau_idx].astype(float) @ depth  # (n_classes, n_points)
    source = (line_class * flux_per_line[None, :]) @ profiles + cont_class @ continuum
    return jnp.exp(-fields) * source, depth


def compose_from_profiles(
    profiles,
    flux_per_line,
    tau_per_line,
    is_tau,
    applies_matrix,
    cont_applies,
    continuum,
    *,
    has_tau: bool = True,
    cover_per_line=None,
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
    cont_applies : jnp.ndarray, shape ``(n_groups, n_lines)`` or ``(n_lines,)``
        Boolean mask per continuum group (regions sharing a zorder): True
        for tau lines whose zorder exceeds the group's zorder (i.e. they
        attenuate that group).  A 1-D array is treated as a single group.
    continuum : jnp.ndarray, shape ``(n_groups, n_points)`` or ``(n_points,)``
        Continuum flux density of each group at each point (unattenuated).
    has_tau : bool, optional
        Static (Python-time) flag indicating whether the configuration
        contains any tau-parametrized lines.  When ``False`` the
        absorption pipeline is skipped entirely; the compiled graph
        becomes a single sum + add.  Default ``True``.
    cover_per_line : jnp.ndarray, shape ``(n_lines,)``, optional
        Covering fraction of each absorber (1 for emission lines or absorbers
        without one).  ``None`` (static) means every ``cf = 1`` and the
        transform is skipped, leaving the graph identical to the fully
        covering model.

    Returns
    -------
    jnp.ndarray, shape ``(n_points,)``
        Total model flux density (before ``flux_scale`` and normalisation).
    """
    # Continuum may be 1-D (single group) or (n_groups, n_points): groups of
    # regions sharing a zorder, each with its own row of ``cont_applies``.
    # Resolved before the jnp calls below, which stage constants under jit.
    classes = (
        _attenuation_classes(is_tau, applies_matrix, cont_applies) if has_tau else None
    )
    continuum = jnp.atleast_2d(continuum)
    cont_applies = jnp.atleast_2d(cont_applies)

    if not has_tau:
        return (flux_per_line[:, None] * profiles).sum(axis=0) + continuum.sum(axis=0)

    # Static zorder structure: one transmission curve per distinct set of
    # absorbers instead of one per line.
    if classes is not None:
        attenuated, _ = _class_terms(
            classes, profiles, flux_per_line, tau_per_line, continuum, cover_per_line
        )
        return attenuated.sum(axis=0)

    emission_phi = jnp.where(is_tau[:, None], 0.0, profiles)  # (n_lines, n_points)
    absorption_phi = jnp.where(is_tau[:, None], profiles, 0.0)  # (n_lines, n_points)

    # Per-component effective tau field:
    # tau_fields[j, p] = sum_k applies_matrix[j,k] * tau_per_line[k] * absorption_phi[k,p]
    if cover_per_line is None:
        tau_fields = (
            applies_matrix * tau_per_line[None, :]
        ) @ absorption_phi  # (n_lines, n_points)
        cont_tau = (
            cont_applies * tau_per_line[None, :]
        ) @ absorption_phi  # (G, n_points)
    else:
        depth = _effective_depth(tau_per_line, absorption_phi, cover_per_line)
        tau_fields = applies_matrix @ depth
        cont_tau = cont_applies @ depth
    t_eff = jnp.exp(-tau_fields)  # (n_lines, n_points)

    attenuated_emission = (flux_per_line[:, None] * emission_phi * t_eff).sum(axis=0)

    t_cont = jnp.exp(-cont_tau)  # (G, n_points)

    return attenuated_emission + (t_cont * continuum).sum(axis=0)


def compose_leave_one_out(
    profiles,
    flux_per_line,
    tau_per_line,
    is_tau,
    applies_matrix,
    cont_applies,
    continuum,
    *,
    has_tau: bool = True,
    cover_per_line=None,
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
    cont_applies : jnp.ndarray, shape ``(n_groups, n_lines)`` or ``(n_lines,)``
        Boolean mask per continuum group: tau lines that attenuate it.
    continuum : jnp.ndarray, shape ``(n_groups, n_points)`` or ``(n_points,)``
        Continuum flux density of each group at each point (unattenuated).
    has_tau : bool, optional
        Static (Python-time) flag indicating whether the configuration
        contains any tau-parametrized lines.  When ``False`` the
        absorption pipeline is skipped entirely.  Default ``True``.
    cover_per_line : jnp.ndarray, shape ``(n_lines,)``, optional
        Covering fraction of each absorber (1 for emission lines or absorbers
        without one).  ``None`` (static) means every ``cf = 1`` and the
        transform is skipped, leaving the graph identical to the fully
        covering model.

    Returns
    -------
    total : jnp.ndarray, shape ``(n_points,)``
        Full model flux density.
    per_line_delta : jnp.ndarray, shape ``(n_lines, n_points)``
        Per-line contribution.  Emission lines: attenuated flux profile
        (positive).  Absorption lines: flux removed (negative).
    """
    # Resolved before the jnp calls below, which stage constants under jit.
    classes = (
        _attenuation_classes(is_tau, applies_matrix, cont_applies) if has_tau else None
    )
    continuum = jnp.atleast_2d(continuum)
    cont_applies = jnp.atleast_2d(cont_applies)

    if not has_tau:
        emission_deltas = flux_per_line[:, None] * profiles  # (n_lines, n_points)
        total = emission_deltas.sum(axis=0) + continuum.sum(axis=0)
        return total, emission_deltas

    emission_phi = jnp.where(is_tau[:, None], 0.0, profiles)  # (n_lines, n_points)

    if classes is not None:
        # total = sum over classes of A_u = T_u * S_u.  Removing tau line j
        # multiplies T_u by exp(+depth_j) for every class it attenuates, so
        # total - total_without_j = -expm1(depth_j) * sum_{u attenuated by j} A_u.
        tau_idx, patterns, _, _ = classes
        attenuated, depth_tau = _class_terms(
            classes, profiles, flux_per_line, tau_per_line, continuum, cover_per_line
        )
        total = attenuated.sum(axis=0)
        deltas = flux_per_line[:, None] * emission_phi  # (n_lines, n_points)
        if tau_idx.size == 0:
            return total, deltas
        behind = patterns[:, tau_idx].T.astype(float) @ attenuated  # (n_tau, n_pts)
        return total, deltas.at[tau_idx].set(-jnp.expm1(depth_tau) * behind)

    absorption_phi = jnp.where(is_tau[:, None], profiles, 0.0)  # (n_lines, n_points)

    if cover_per_line is None:
        depth = tau_per_line[:, None] * absorption_phi  # (n_lines, n_points)
    else:
        depth = _effective_depth(tau_per_line, absorption_phi, cover_per_line)
    tau_fields = applies_matrix @ depth
    t_eff = jnp.exp(-tau_fields)  # (n_lines, n_points)

    # Precompute attenuated per-line terms; reused in the tau LOO loop below.
    attenuated_per_line = (
        flux_per_line[:, None] * emission_phi * t_eff
    )  # (n_lines, n_points)
    attenuated_emission = attenuated_per_line.sum(axis=0)

    cont_tau = cont_applies @ depth  # (G, n)
    t_cont = jnp.exp(-cont_tau)  # (G, n)
    total = attenuated_emission + (t_cont * continuum).sum(axis=0)

    # Emission deltas are vectorised (independent of tau), so compute them all
    # at once rather than in a Python loop.
    emission_deltas = flux_per_line[:, None] * emission_phi  # (n_lines, n_points)

    # is_tau is captured via closure as a concrete jnp array (matrices are
    # built before tracing), so we can resolve tau indices Python-side.
    tau_idx_np = np.where(np.asarray(is_tau))[0]

    if len(tau_idx_np) == 0:
        return total, emission_deltas

    tau_idx = jnp.asarray(tau_idx_np)

    def _tau_delta(j):
        # Rank-1 removal: removing tau j subtracts its contribution from
        # tau_fields, equivalent to multiplying t_eff by exp(+contribution).
        contrib = applies_matrix[:, j, None] * depth[j]
        attenuated_loo = (attenuated_per_line * jnp.exp(contrib)).sum(axis=0)
        cont_contrib = cont_applies[:, j, None] * depth[j]  # (G, n)
        total_loo = attenuated_loo + (t_cont * jnp.exp(cont_contrib) * continuum).sum(
            axis=0
        )
        return total - total_loo

    tau_deltas = jax.vmap(_tau_delta)(tau_idx)  # (n_tau, n_points)

    deltas = emission_deltas.at[tau_idx].set(tau_deltas)
    return total, deltas
