"""Benchmarks for higher-order grating overlap.

Every benchmark runs on the ``overlap_by_mode`` fixture: one 2000-pixel
spectrum with 7 lines, built under each integration mode both without
(``off``) and with (``on``) a second-order overlap on its disperser.  The two
members of a pair differ only by the overlap term, so comparing
``[analytic-on]`` with ``[analytic-off]`` gives its cost.

Covered paths: log-density, value+grad (the NUTS hot path),
``evaluate_model`` (shares the overlap code with the likelihood) and, as a
slow benchmark, cold-cache compile time.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from numpyro.infer.util import log_density

from benchmarks._helpers import (
    block,
    build_model,
    cfg_overlap,
    make_overlap_spectrum,
    one_prior_draw,
    write_overlap_ratio,
)
from unite.compute import evaluate_model
from unite.spectrum import Spectra

N_SAMPLES = 100


def _logp(bench):
    return jax.jit(lambda p: log_density(bench.model_fn, (bench.args,), {}, p)[0])


def test_logp_overlap(benchmark, overlap_by_mode):
    """log_density with and without the overlap term, per integration mode."""
    fn = _logp(overlap_by_mode)
    fn(overlap_by_mode.sample).block_until_ready()
    benchmark(lambda: block(fn(overlap_by_mode.sample)))


def test_logp_grad_overlap(benchmark, overlap_by_mode):
    """value+grad with and without the overlap term, per integration mode."""
    fn = jax.jit(
        jax.value_and_grad(
            lambda p: log_density(
                overlap_by_mode.model_fn, (overlap_by_mode.args,), {}, p
            )[0]
        )
    )
    block(fn(overlap_by_mode.sample))
    benchmark(lambda: block(fn(overlap_by_mode.sample)))


def test_evaluate_model_overlap(benchmark, overlap_by_mode):
    """evaluate_model on 100 samples, with and without the overlap term."""
    b = overlap_by_mode
    samples = {
        k: np.broadcast_to(np.asarray(v), (N_SAMPLES,)) for k, v in b.sample.items()
    }
    evaluate_model(samples, b.args)  # warm the cached evaluator
    benchmark(lambda: evaluate_model(samples, b.args))


@pytest.mark.slow
@pytest.mark.parametrize('state', ['off', 'on'])
def test_compile_overlap(benchmark, tmp_path, state):
    """Cold-cache build + log-density compile, with and without the overlap."""
    ratio = None
    if state == 'on':
        ratio = tmp_path / 'r2.ecsv'
        write_overlap_ratio(ratio)

    def compile_once():
        lc, cc = cfg_overlap()
        spectra = Spectra([make_overlap_spectrum(f'ov_compile_{state}', ratio)], 0.0)
        model_fn, args = build_model(lc, cc, spectra)
        params = one_prior_draw(model_fn, args)
        fn = jax.jit(lambda p: log_density(model_fn, (args,), {}, p)[0])
        fn(params).block_until_ready()

    benchmark.pedantic(compile_once, setup=jax.clear_caches, rounds=3, iterations=1)
