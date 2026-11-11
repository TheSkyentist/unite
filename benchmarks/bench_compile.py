"""JIT compile-time benchmarks.

The cost of *tracing* the numpyro model into XLA HLO + lowering is paid once
per fresh model build.  These benchmarks measure that cost in isolation by
clearing JAX's compilation cache before each round.

Marked ``slow`` because each call is order-of-seconds.
"""

from __future__ import annotations

import jax
import pytest
from numpyro.infer.util import log_density

from benchmarks._helpers import (
    build_model,
    cfg_minimal,
    cfg_multi_grating,
    cfg_single_grating,
    make_spectrum,
    one_prior_draw,
)
from unite.spectrum import Spectra


def _compile_once_minimal():
    """Build a model and force its log-density JIT to compile."""
    lc, cc = cfg_minimal()
    spec = make_spectrum(6500, 6600, 100, 'min_compile')
    spectra = Spectra([spec], redshift=0.0)
    model_fn, args = build_model(lc, cc, spectra)
    params = one_prior_draw(model_fn, args)
    fn = jax.jit(lambda p: log_density(model_fn, (args,), {}, p)[0])
    fn(params).block_until_ready()


def _compile_once_single_grating():
    lc, cc = cfg_single_grating()
    spec = make_spectrum(6400, 6800, 400, 'sg_compile')
    spectra = Spectra([spec], redshift=0.0)
    model_fn, args = build_model(lc, cc, spectra)
    params = one_prior_draw(model_fn, args)
    fn = jax.jit(lambda p: log_density(model_fn, (args,), {}, p)[0])
    fn(params).block_until_ready()


def _compile_once_multi_grating():
    lc, cc = cfg_multi_grating()
    spec_a = make_spectrum(5800, 6000, 400, 'mg_blue_compile')
    spec_b = make_spectrum(6400, 6800, 400, 'mg_red_compile')
    spectra = Spectra([spec_a, spec_b], redshift=0.0)
    model_fn, args = build_model(lc, cc, spectra)
    params = one_prior_draw(model_fn, args)
    fn = jax.jit(lambda p: log_density(model_fn, (args,), {}, p)[0])
    fn(params).block_until_ready()


def _clear():
    """Force JAX to forget every compiled trace before the next round."""
    jax.clear_caches()


def _pedantic(benchmark, fn):
    """Time ``fn`` cold: clear every JAX cache before each of 3 rounds."""
    benchmark.pedantic(fn, setup=_clear, rounds=3, iterations=1)


@pytest.mark.slow
def test_compile_minimal(benchmark):
    """Cold-cache compile time for the minimal model."""
    _pedantic(benchmark, _compile_once_minimal)


@pytest.mark.slow
def test_compile_single_grating(benchmark):
    _pedantic(benchmark, _compile_once_single_grating)


@pytest.mark.slow
def test_compile_multi_grating(benchmark):
    _pedantic(benchmark, _compile_once_multi_grating)
