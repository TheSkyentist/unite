"""Microbenchmarks for inner kernels.

These exercise the hot JAX primitives in isolation with synthetic inputs.
Compared to the end-to-end benchmarks they are quick (sub-millisecond) and
give a tight regression signal for kernel-level optimisations.

JAX gotcha: every benched call ends in :func:`block` so the async dispatch
completes inside the timing window.  Compilation is paid once during the
benchmark's automatic warmup; steady-state numbers are what get reported.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from benchmarks._helpers import PROFILE_CLASSES, block, default_param_for
from unite._compose import compose_from_profiles
from unite.line.compute import evaluate_lines, integrate_lines
from unite.line.functions import (
    evaluate_gaussian,
    integrate_gaussian,
    integrate_split_normal,
)

# ----------------------------------------------------------------------
# Per-kernel benchmarks (synthetic inputs)
# ----------------------------------------------------------------------

# Representative scales: NIRSpec-like spectrum (~1k pixels), modest line count.
N_PIXELS = 1024
N_LINES = 8


@pytest.fixture(scope='module')
def synthetic_inputs():
    """Pre-built JAX arrays for kernel benchmarks (avoids dispatch overhead)."""
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, 8)

    low = jnp.linspace(6400.0, 6800.0, N_PIXELS)
    high = low + (low[1] - low[0])
    wl = (low + high) / 2.0

    centers = 6500.0 + 300.0 * jax.random.uniform(keys[0], (N_LINES,))
    lsf_fwhm = jnp.full((N_LINES,), 2.0)
    p0 = jnp.full((N_LINES,), 3.0)  # FWHM-like
    p1 = jnp.zeros((N_LINES,))
    p2 = jnp.zeros((N_LINES,))
    codes = jnp.zeros((N_LINES,), dtype=jnp.int32)  # all Gaussian

    profiles = jax.random.uniform(keys[1], (N_LINES, N_PIXELS))
    flux = jax.random.uniform(keys[2], (N_LINES,), minval=10.0, maxval=200.0)
    tau = jnp.zeros((N_LINES,))
    is_tau = jnp.zeros((N_LINES,), dtype=jnp.bool_)
    applies = jnp.zeros((N_LINES, N_LINES), dtype=jnp.bool_)
    cont_applies = jnp.zeros((N_LINES,), dtype=jnp.bool_)
    continuum = jnp.full((N_PIXELS,), 5.0)

    return {
        'low': low,
        'high': high,
        'wl': wl,
        'centers': centers,
        'lsf_fwhm': lsf_fwhm,
        'p0': p0,
        'p1': p1,
        'p2': p2,
        'codes': codes,
        'profiles': profiles,
        'flux': flux,
        'tau': tau,
        'is_tau': is_tau,
        'applies': applies,
        'cont_applies': cont_applies,
        'continuum': continuum,
    }


# ----------------------------------------------------------------------


def test_integrate_gaussian_kernel(benchmark, synthetic_inputs):
    """Single-profile Gaussian CDF integration over pixels."""
    s = synthetic_inputs
    fn = jax.jit(integrate_gaussian)
    # Warm cache before timing.
    fn(
        s['low'], s['high'], s['centers'][0], s['lsf_fwhm'][0], s['p0'][0]
    ).block_until_ready()

    def run():
        return block(
            fn(s['low'], s['high'], s['centers'][0], s['lsf_fwhm'][0], s['p0'][0])
        )

    benchmark(run)


def test_evaluate_gaussian_kernel(benchmark, synthetic_inputs):
    """Single-profile Gaussian pointwise evaluation."""
    s = synthetic_inputs
    fn = jax.jit(evaluate_gaussian)
    fn(s['wl'], s['centers'][0], s['lsf_fwhm'][0], s['p0'][0]).block_until_ready()

    def run():
        return block(fn(s['wl'], s['centers'][0], s['lsf_fwhm'][0], s['p0'][0]))

    benchmark(run)


def test_integrate_split_normal_kernel(benchmark, synthetic_inputs):
    """Asymmetric kernel — represents the more expensive profile family."""
    s = synthetic_inputs
    fn = jax.jit(integrate_split_normal)
    # SplitNormal needs two FWHMs in p0, p1.
    fn(
        s['low'], s['high'], s['centers'][0], s['lsf_fwhm'][0], s['p0'][0], s['p0'][0]
    ).block_until_ready()

    def run():
        return block(
            fn(
                s['low'],
                s['high'],
                s['centers'][0],
                s['lsf_fwhm'][0],
                s['p0'][0],
                s['p0'][0],
            )
        )

    benchmark(run)


def test_integrate_lines_dispatch(benchmark, synthetic_inputs):
    """Vmapped + lax.switch dispatch over N_LINES Gaussians."""
    s = synthetic_inputs
    fn = jax.jit(integrate_lines)
    fn(
        s['low'],
        s['high'],
        s['centers'],
        s['lsf_fwhm'],
        s['p0'],
        s['p1'],
        s['p2'],
        s['codes'],
    ).block_until_ready()

    def run():
        return block(
            fn(
                s['low'],
                s['high'],
                s['centers'],
                s['lsf_fwhm'],
                s['p0'],
                s['p1'],
                s['p2'],
                s['codes'],
            )
        )

    benchmark(run)


def test_evaluate_lines_dispatch(benchmark, synthetic_inputs):
    """Vmapped + lax.switch evaluation (quadrature path)."""
    s = synthetic_inputs
    fn = jax.jit(evaluate_lines)
    fn(
        s['wl'], s['centers'], s['lsf_fwhm'], s['p0'], s['p1'], s['p2'], s['codes']
    ).block_until_ready()

    def run():
        return block(
            fn(
                s['wl'],
                s['centers'],
                s['lsf_fwhm'],
                s['p0'],
                s['p1'],
                s['p2'],
                s['codes'],
            )
        )

    benchmark(run)


def test_compose_from_profiles_no_tau(benchmark, synthetic_inputs):
    """Pure-emission compose path (cheap branch)."""
    s = synthetic_inputs
    fn = jax.jit(compose_from_profiles, static_argnames=('has_tau',))
    fn(
        s['profiles'],
        s['flux'],
        s['tau'],
        s['is_tau'],
        s['applies'],
        s['cont_applies'],
        s['continuum'],
        has_tau=False,
    ).block_until_ready()

    def run():
        return block(
            fn(
                s['profiles'],
                s['flux'],
                s['tau'],
                s['is_tau'],
                s['applies'],
                s['cont_applies'],
                s['continuum'],
                has_tau=False,
            )
        )

    benchmark(run)


def test_compose_from_profiles_with_tau(benchmark, synthetic_inputs):
    """Emission + absorption compose path (the expensive branch)."""
    s = synthetic_inputs
    is_tau = s['is_tau'].at[-1].set(True)
    tau = s['tau'].at[-1].set(0.5)
    applies = s['applies'].at[:-1, -1].set(True)
    cont_applies = s['cont_applies'].at[-1].set(True)
    fn = jax.jit(compose_from_profiles, static_argnames=('has_tau',))
    fn(
        s['profiles'],
        s['flux'],
        tau,
        is_tau,
        applies,
        cont_applies,
        s['continuum'],
        has_tau=True,
    ).block_until_ready()

    def run():
        return block(
            fn(
                s['profiles'],
                s['flux'],
                tau,
                is_tau,
                applies,
                cont_applies,
                s['continuum'],
                has_tau=True,
            )
        )

    benchmark(run)


# ----------------------------------------------------------------------
# Per-profile benchmarks (parametrized over every registered Profile)
# ----------------------------------------------------------------------


def _profile_kwargs(profile):
    """Build a kwargs dict for ``profile.integrate(...)`` / ``.evaluate(...)``."""
    return {name: default_param_for(name) for name in profile.param_names()}


@pytest.fixture(scope='module')
def profile_edges():
    """Pixel edges for per-profile integrate benchmarks (shared, immutable)."""
    low = jnp.linspace(6400.0, 6800.0, N_PIXELS)
    high = low + (low[1] - low[0])
    return low, high, (low + high) / 2.0


@pytest.mark.parametrize('profile_cls', PROFILE_CLASSES, ids=lambda c: c.__name__)
def test_profile_integrate(benchmark, profile_edges, profile_cls):
    """CDF integration over pixels, one bench per registered Profile."""
    low, high, _ = profile_edges
    profile = profile_cls()
    kwargs = _profile_kwargs(profile)
    center = jnp.asarray(6500.0)
    lsf_fwhm = jnp.asarray(2.0)

    fn = jax.jit(lambda lo, hi, c, lsf: profile.integrate(lo, hi, c, lsf, **kwargs))
    fn(low, high, center, lsf_fwhm).block_until_ready()

    benchmark(lambda: block(fn(low, high, center, lsf_fwhm)))


@pytest.mark.parametrize('profile_cls', PROFILE_CLASSES, ids=lambda c: c.__name__)
def test_profile_evaluate(benchmark, profile_edges, profile_cls):
    """Pointwise evaluation, one bench per registered Profile."""
    _, _, wl = profile_edges
    profile = profile_cls()
    kwargs = _profile_kwargs(profile)
    center = jnp.asarray(6500.0)
    lsf_fwhm = jnp.asarray(2.0)

    fn = jax.jit(lambda w, c, lsf: profile.evaluate(w, c, lsf, **kwargs))
    fn(wl, center, lsf_fwhm).block_until_ready()

    benchmark(lambda: block(fn(wl, center, lsf_fwhm)))
