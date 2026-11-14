"""The grouped absorber composition equals the dense per-line form."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from unite import _compose
from unite._compose import compose_from_profiles, compose_leave_one_out

_N_POINTS = 40

# (line zorders, is_tau, continuum group zorders)
_LAYOUTS = {
    'one_screen': ([0, 0, 0, 1], [0, 0, 0, 1], [0]),
    'two_screens': ([0, 0, 1, 2, 0], [0, 0, 1, 1, 0], [0]),
    'emission_in_front': ([0, 3, 1, 2, 2], [0, 0, 1, 1, 0], [0]),
    'continuum_groups': ([0, 1, 1, 2, 3], [0, 0, 1, 1, 1], [0, 2]),
    'continuum_in_front': ([0, 0, 1], [0, 0, 1], [5]),
    'only_absorbers': ([1, 2], [1, 1], [0, 1]),
}


def _inputs(layout, cover):
    zorders, is_tau, group_z = (np.asarray(a) for a in _LAYOUTS[layout])
    is_tau = is_tau.astype(bool)
    n_lines = zorders.size
    rng = np.random.default_rng(1)
    x = np.linspace(-1.0, 1.0, _N_POINTS)
    centers = rng.uniform(-0.6, 0.6, n_lines)
    profiles = np.exp(-0.5 * ((x[None, :] - centers[:, None]) / 0.2) ** 2)
    flux = np.where(is_tau, 0.0, rng.uniform(0.5, 2.0, n_lines))
    tau = np.where(is_tau, rng.uniform(0.2, 3.0, n_lines), 0.0)
    applies = (zorders[None, :] > zorders[:, None]) & is_tau[None, :]
    cont_applies = (zorders[None, :] > group_z[:, None]) & is_tau[None, :]
    continuum = rng.uniform(0.5, 1.5, (group_z.size, _N_POINTS))
    cover_per_line = None
    if cover:
        cover_per_line = np.where(is_tau, rng.uniform(0.2, 1.0, n_lines), 1.0)
        # One absorber fully covering, exercising the cf = 1 special case.
        cover_per_line[np.flatnonzero(is_tau)[0]] = 1.0
    arrays = (profiles, flux, tau, is_tau, applies, cont_applies, continuum)
    return tuple(jnp.asarray(a) for a in arrays), cover_per_line


@pytest.mark.parametrize('cover', [False, True])
@pytest.mark.parametrize('layout', list(_LAYOUTS))
@pytest.mark.parametrize('jit', [False, True])
def test_grouped_matches_dense(monkeypatch, layout, cover, jit):
    args, cover_per_line = _inputs(layout, cover)
    kwargs = {} if cover_per_line is None else {'cover_per_line': cover_per_line}

    def run():
        wrap = jax.jit if jit else (lambda f: f)
        total = wrap(lambda p: compose_from_profiles(p, *args[1:], **kwargs))(args[0])
        loo = wrap(lambda p: compose_leave_one_out(p, *args[1:], **kwargs))(args[0])
        return total, loo

    total, (loo_total, deltas) = run()
    # The dense form is used when the zorder structure is not static.
    monkeypatch.setattr(_compose, '_attenuation_classes', lambda *a: None)
    d_total, (d_loo_total, d_deltas) = run()

    scale = float(jnp.max(jnp.abs(d_total)))
    np.testing.assert_allclose(total, d_total, rtol=1e-12)
    np.testing.assert_allclose(loo_total, d_loo_total, rtol=1e-12)
    np.testing.assert_allclose(deltas, d_deltas, rtol=1e-9, atol=1e-13 * scale)


def test_classes_follow_depth_ordering():
    """One class per distinct set of absorbers in front, not one per line."""
    args, _ = _inputs('emission_in_front', cover=False)
    tau_idx, patterns, line_class, cont_class = _compose._attenuation_classes(
        args[3], args[4], args[5]
    )
    np.testing.assert_array_equal(tau_idx, [2, 3])
    # zorder 0 (line 0, continuum): both absorbers; zorder 2 (line 4): none
    # (zorder 2 is not in front of itself); zorder 3 (line 1): none.
    assert patterns.shape[0] == 2
    assert line_class.sum() == 3
    np.testing.assert_array_equal(line_class[:, tau_idx], 0.0)
    np.testing.assert_array_equal(cont_class.sum(axis=0), [1.0])


def test_traced_structure_falls_back():
    args, _ = _inputs('two_screens', cover=False)
    want = compose_from_profiles(*args)
    got = jax.jit(compose_from_profiles)(*args)
    np.testing.assert_allclose(got, want, rtol=1e-12)
