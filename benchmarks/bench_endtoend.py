"""End-to-end benchmarks of the numpyro model.

These cover the full forward pass, log-density evaluation, and value+grad —
the latter being what NUTS' leapfrog integrator does on every step.  A short
MCMC run is included as a `@pytest.mark.slow` benchmark so the *full* path
(JIT trace, sample, accept/reject) is exercised too.

These are the benchmarks most closely correlated with user-perceived
performance.  Kernel-level benchmarks live in ``bench_kernels.py``.
"""

from __future__ import annotations

import jax
import pytest
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density

from benchmarks._helpers import block

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_logp(model_fn, args):
    """Return a JIT-compiled scalar log-density function of the params."""

    def logp(params):
        lp, _ = log_density(model_fn, (args,), {}, params)
        return lp

    return jax.jit(logp)


def _make_logp_grad(model_fn, args):
    """Return JIT-compiled value+grad of the log-density."""

    def logp(params):
        lp, _ = log_density(model_fn, (args,), {}, params)
        return lp

    return jax.jit(jax.value_and_grad(logp))


# ----------------------------------------------------------------------
# Forward (log-density only)
# ----------------------------------------------------------------------


def test_logp_minimal(benchmark, minimal_bench):
    fn = _make_logp(minimal_bench.model_fn, minimal_bench.args)
    fn(minimal_bench.sample).block_until_ready()
    benchmark(lambda: block(fn(minimal_bench.sample)))


def test_logp_single_grating(benchmark, single_grating_bench):
    fn = _make_logp(single_grating_bench.model_fn, single_grating_bench.args)
    fn(single_grating_bench.sample).block_until_ready()
    benchmark(lambda: block(fn(single_grating_bench.sample)))


def test_logp_multi_grating(benchmark, multi_grating_bench):
    fn = _make_logp(multi_grating_bench.model_fn, multi_grating_bench.args)
    fn(multi_grating_bench.sample).block_until_ready()
    benchmark(lambda: block(fn(multi_grating_bench.sample)))


# ----------------------------------------------------------------------
# Value + grad (the NUTS hot path)
# ----------------------------------------------------------------------


def test_logp_grad_minimal(benchmark, minimal_bench):
    fn = _make_logp_grad(minimal_bench.model_fn, minimal_bench.args)
    block(fn(minimal_bench.sample))
    benchmark(lambda: block(fn(minimal_bench.sample)))


def test_logp_grad_single_grating(benchmark, single_grating_bench):
    fn = _make_logp_grad(single_grating_bench.model_fn, single_grating_bench.args)
    block(fn(single_grating_bench.sample))
    benchmark(lambda: block(fn(single_grating_bench.sample)))


def test_logp_grad_multi_grating(benchmark, multi_grating_bench):
    fn = _make_logp_grad(multi_grating_bench.model_fn, multi_grating_bench.args)
    block(fn(multi_grating_bench.sample))
    benchmark(lambda: block(fn(multi_grating_bench.sample)))


# ----------------------------------------------------------------------
# Short MCMC — slow, opt-in
# ----------------------------------------------------------------------


def _run_short_mcmc(model_fn, args, seed: int = 0):
    kernel = NUTS(model_fn)
    mcmc = MCMC(kernel, num_warmup=20, num_samples=20, num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), args)
    samples = mcmc.get_samples()
    block(samples)
    return samples


@pytest.mark.slow
def test_mcmc_minimal(benchmark, minimal_bench):
    benchmark(lambda: _run_short_mcmc(minimal_bench.model_fn, minimal_bench.args))


@pytest.mark.slow
def test_mcmc_single_grating(benchmark, single_grating_bench):
    benchmark(
        lambda: _run_short_mcmc(
            single_grating_bench.model_fn, single_grating_bench.args
        )
    )


# ----------------------------------------------------------------------
# PseudoVoigt profile — exercises the Faddeeva / wofz path
# ----------------------------------------------------------------------


def test_logp_single_grating_voigt(benchmark, single_grating_voigt_bench):
    """log_density on single-grating with PseudoVoigt profiles (analytic mode)."""
    fn = _make_logp(
        single_grating_voigt_bench.model_fn, single_grating_voigt_bench.args
    )
    fn(single_grating_voigt_bench.sample).block_until_ready()
    benchmark(lambda: block(fn(single_grating_voigt_bench.sample)))


def test_logp_grad_single_grating_voigt(benchmark, single_grating_voigt_bench):
    """value+grad on single-grating with PseudoVoigt profiles (analytic mode)."""
    fn = _make_logp_grad(
        single_grating_voigt_bench.model_fn, single_grating_voigt_bench.args
    )
    block(fn(single_grating_voigt_bench.sample))
    benchmark(lambda: block(fn(single_grating_voigt_bench.sample)))


def test_logp_voigt_by_mode(benchmark, single_grating_voigt_by_mode):
    """log_density on PseudoVoigt single-grating under each integration mode."""
    fn = _make_logp(
        single_grating_voigt_by_mode.model_fn, single_grating_voigt_by_mode.args
    )
    fn(single_grating_voigt_by_mode.sample).block_until_ready()
    benchmark(lambda: block(fn(single_grating_voigt_by_mode.sample)))


def test_logp_grad_voigt_by_mode(benchmark, single_grating_voigt_by_mode):
    """value+grad on PseudoVoigt single-grating under each integration mode."""
    fn = _make_logp_grad(
        single_grating_voigt_by_mode.model_fn, single_grating_voigt_by_mode.args
    )
    block(fn(single_grating_voigt_by_mode.sample))
    benchmark(lambda: block(fn(single_grating_voigt_by_mode.sample)))


# ----------------------------------------------------------------------
# Integration modes — parametrized over analytic / convolution
# ----------------------------------------------------------------------


def test_logp_by_mode(benchmark, single_grating_by_mode):
    """log_density on single-grating under each integration mode."""
    fn = _make_logp(single_grating_by_mode.model_fn, single_grating_by_mode.args)
    fn(single_grating_by_mode.sample).block_until_ready()
    benchmark(lambda: block(fn(single_grating_by_mode.sample)))


def test_logp_grad_by_mode(benchmark, single_grating_by_mode):
    """value+grad on single-grating under each integration mode."""
    fn = _make_logp_grad(single_grating_by_mode.model_fn, single_grating_by_mode.args)
    block(fn(single_grating_by_mode.sample))
    benchmark(lambda: block(fn(single_grating_by_mode.sample)))


# ----------------------------------------------------------------------
# Absorber path — direct A/B against pure-emission single_grating_bench
# ----------------------------------------------------------------------


def test_logp_single_grating_absorber(benchmark, single_grating_absorber_bench):
    """log_density with one tau absorber added to the single-grating config."""
    fn = _make_logp(
        single_grating_absorber_bench.model_fn, single_grating_absorber_bench.args
    )
    fn(single_grating_absorber_bench.sample).block_until_ready()
    benchmark(lambda: block(fn(single_grating_absorber_bench.sample)))


def test_logp_grad_single_grating_absorber(benchmark, single_grating_absorber_bench):
    """value+grad with one tau absorber added to the single-grating config."""
    fn = _make_logp_grad(
        single_grating_absorber_bench.model_fn, single_grating_absorber_bench.args
    )
    block(fn(single_grating_absorber_bench.sample))
    benchmark(lambda: block(fn(single_grating_absorber_bench.sample)))
