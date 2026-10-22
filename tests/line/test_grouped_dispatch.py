"""Grouped-by-profile dispatch must match the ``lax.switch`` reference."""

import jax.numpy as jnp
import numpy as np
import pytest

from unite.line.compute import (
    evaluate_lines,
    evaluate_lines_at_own_centers,
    integrate_lines,
    make_grouped_dispatch,
)
from unite.line.library import _PROFILE_REGISTRY

_BY_CODE = {cls.code: cls for cls in _PROFILE_REGISTRY.values()}
_ALL = sorted(_BY_CODE)

# Unsorted, repeated, mixed — exercises the gather/permute path.
CODE_SETS = {
    'single': [0, 0, 0],
    'sorted_mix': [0, 0, 1, 2],
    'unsorted_mix': [2, 0, 1, 0, 2, 1],
    'all_profiles': _ALL + _ALL[::-1],
}


def _inputs(codes):
    n = len(codes)
    slots = []
    for c in codes:
        names = _BY_CODE[c]().param_names()
        vals = [
            0.05 if n_ in ('h3', 'h4') else 0.5 if n_ == 'alpha' else 3.0
            for n_ in names
        ]
        slots.append(vals + [0.0] * (3 - len(vals)))
    p = jnp.asarray(slots)
    centers = jnp.linspace(6450.0, 6750.0, n)
    return centers, p[:, 0], p[:, 1], p[:, 2], jnp.asarray(codes)


@pytest.fixture(scope='module')
def grid():
    edges = jnp.linspace(6400.0, 6800.0, 201)
    return edges, jnp.full(edges.shape, 2.0), (edges[:-1] + edges[1:]) / 2


@pytest.mark.parametrize('name', CODE_SETS)
def test_integrate_matches_switch(grid, name):
    edges, lsf_edges, _ = grid
    centers, p0, p1, p2, codes = _inputs(CODE_SETS[name])
    int_fn, _, _ = make_grouped_dispatch(codes)
    ref = integrate_lines(edges, centers, lsf_edges, p0, p1, p2, codes)
    out = int_fn(edges, centers, lsf_edges, p0, p1, p2, codes)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('name', CODE_SETS)
def test_evaluate_matches_switch(grid, name):
    _, _, wl = grid
    centers, p0, p1, p2, codes = _inputs(CODE_SETS[name])
    lsf = jnp.full(centers.shape, 2.0)
    _, eval_fn, _ = make_grouped_dispatch(codes)
    ref = evaluate_lines(wl, centers, lsf, p0, p1, p2, codes)
    out = eval_fn(wl, centers, lsf, p0, p1, p2, codes)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('name', CODE_SETS)
def test_evaluate_at_centers_matches_switch(name):
    centers, p0, p1, p2, codes = _inputs(CODE_SETS[name])
    lsf = jnp.full(centers.shape, 1e-10)
    _, _, ctr_fn = make_grouped_dispatch(codes)
    ref = evaluate_lines_at_own_centers(centers, centers, lsf, p0, p1, p2, codes)
    out = ctr_fn(centers, centers, lsf, p0, p1, p2, codes)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


def test_empty_codes_fall_back_to_switch():
    fns = make_grouped_dispatch(np.array([], dtype=int))
    assert fns == (integrate_lines, evaluate_lines, evaluate_lines_at_own_centers)
