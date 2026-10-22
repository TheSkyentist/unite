"""Profile-dispatch benchmarks: ``lax.switch`` vs. grouped-by-profile.

``integrate_lines`` vmaps a ``lax.switch`` over a per-line profile ``code``.
Because the predicate is batched, JAX lowers the switch to "run every branch
for every line, then select" — each line pays for every profile kernel in the
switch.  ``_make_specialized_dispatch`` trims the switch to the profiles a
config actually uses, but each line still computes all *used* branches.

This module compares two implementations on realistic profile mixtures:

- ``switch``  — the full 11-branch ``integrate_lines``.
- ``grouped`` — ``make_grouped_dispatch`` (the model path): lines grouped by
  profile at build time; each kernel is vmapped only over its own lines.

Each is timed forward-only and under ``value_and_grad`` (the NUTS cost).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks._helpers import PROFILE_CLASSES, block, default_param_for
from unite.line.compute import integrate_lines, make_grouped_dispatch

N_PIXELS = 1024
N_LINES = 20

_BY_NAME = {cls.__name__: cls for cls in PROFILE_CLASSES}


def _mix(**counts: int) -> list[str]:
    return [name for name, n in counts.items() for _ in range(n)]


MIXES = {
    'all_gaussian': _mix(Gaussian=20),
    'gauss_plus_skewvoigt': _mix(Gaussian=19, SkewVoigt=1),
    'gauss_pv_semg': _mix(Gaussian=12, PseudoVoigt=5, SEMG=3),
    'all_profiles': [
        PROFILE_CLASSES[i % len(PROFILE_CLASSES)].__name__ for i in range(N_LINES)
    ],
}


@pytest.fixture(scope='module')
def grid():
    edges = jnp.linspace(6400.0, 6800.0, N_PIXELS + 1)
    return edges, jnp.full((N_PIXELS + 1,), 2.0)


def _line_inputs(names):
    """Per-line ``(codes, centers, p0, p1, p2)`` for a list of profile names."""
    codes, slots = [], []
    for name in names:
        profile = _BY_NAME[name]()
        codes.append(profile.code)
        vals = [default_param_for(p) for p in profile.param_names()]
        slots.append(vals + [0.0] * (3 - len(vals)))
    p = jnp.asarray(slots)
    centers = jnp.linspace(6450.0, 6750.0, len(names))
    return np.asarray(codes), centers, p[:, 0], p[:, 1], p[:, 2]


def _build(impl, codes):
    """Return ``(fn, codes_arg)`` with the common integrate signature."""
    if impl == 'switch':
        return integrate_lines, jnp.asarray(codes)
    return make_grouped_dispatch(codes)[0], None


IMPLS = ('switch', 'grouped')


@pytest.mark.parametrize('mix', MIXES)
def test_grouped_matches_switch(grid, mix):
    """Correctness: grouped output equals the reference switch dispatch."""
    edges, lsf = grid
    codes, centers, p0, p1, p2 = _line_inputs(MIXES[mix])
    ref = integrate_lines(edges, centers, lsf, p0, p1, p2, jnp.asarray(codes))
    fn, cargs = _build('grouped', codes)
    out = fn(edges, centers, lsf, p0, p1, p2, cargs)
    np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('impl', IMPLS)
@pytest.mark.parametrize('mix', MIXES)
def test_integrate_forward(benchmark, grid, mix, impl):
    edges, lsf = grid
    codes, centers, p0, p1, p2 = _line_inputs(MIXES[mix])
    fn, cargs = _build(impl, codes)
    jfn = jax.jit(lambda c, a, b, d: fn(edges, c, lsf, a, b, d, cargs))
    jfn(centers, p0, p1, p2).block_until_ready()
    benchmark(lambda: block(jfn(centers, p0, p1, p2)))


@pytest.mark.parametrize('impl', IMPLS)
@pytest.mark.parametrize('mix', MIXES)
def test_integrate_value_and_grad(benchmark, grid, mix, impl):
    edges, lsf = grid
    codes, centers, p0, p1, p2 = _line_inputs(MIXES[mix])
    fn, cargs = _build(impl, codes)

    def loss(c, a, b, d):
        return jnp.sum(jnp.diff(fn(edges, c, lsf, a, b, d, cargs), axis=-1) ** 2)

    vg = jax.jit(jax.value_and_grad(loss, argnums=(0, 1, 2, 3)))
    jax.block_until_ready(vg(centers, p0, p1, p2))
    benchmark(lambda: jax.block_until_ready(vg(centers, p0, p1, p2)))
