# Benchmarks

Performance suite for `unite`. Three layers:

1. **`bench_kernels.py`** — microbenchmarks for inner kernels (`integrate_gaussian`, `integrate_lines` dispatch, `compose_from_profiles`, …). Sub-millisecond each; tight regression signal.
2. **`bench_endtoend.py`** — `log_density` and `value_and_grad` of the full numpyro model on three representative configs. Closest to user-perceived performance.
3. **`bench_compile.py`** — JIT trace + lowering cost, measured with `jax.clear_caches()` between rounds. Marked `@pytest.mark.slow`.

There is also `profiling/profile_fit.py` for **interactive** profiling (Perfetto trace + optional pyinstrument + optional memory pprof).

## Running locally

`pixi run bench` runs the wall-time suite. Slow benchmarks (`@pytest.mark.slow`, mostly compile-time + short MCMC) are included by default — pass `-m "not slow"` to skip them.

```bash
pixi run bench                              # all benchmarks
pixi run -e bench pytest benchmarks/ --benchmark-only -m "not slow"  # fast only
```

### A/B comparing branches (e.g. for design decisions)

```bash
# On main
pixi run bench-save                          # saves baseline under .benchmarks/

# On feature branch
pixi run bench-compare                       # diff vs baseline, exits non-zero on >10% regression
```

`bench-save` and `bench-compare` use `pytest-benchmark`'s built-in JSON storage. To compare two arbitrary saved runs, see `pytest-benchmark compare --help`.

### Interactive profiling

```bash
pixi run profile                             # default: single_grating config, writes Perfetto trace
pixi run profile-mem                         # also dumps device memory pprof
pixi run -e bench python profiling/profile_fit.py --config multi --pyinstrument
```

Open the trace from `traces/` at <https://ui.perfetto.dev> for a flame graph of XLA ops and host-side Python.

## How CI works

`.github/workflows/bench.yml` runs on every PR to `main`. It uses **pytest-codspeed** (instruction-count benchmarking via Cachegrind) so the noisy GitHub Actions runners don't pollute measurements. CodSpeed posts a PR comment with per-benchmark deltas — informational only, never blocks the merge.

**Onboarding**: the repo must be added to your CodSpeed organisation at <https://codspeed.io> for the action to post results. The token comes from `secrets.CODSPEED_TOKEN`; for public OSS repos this is set automatically. Before onboarding, the action will still run but uploads will no-op.

`@pytest.mark.slow` benchmarks are excluded from CI (`-m "not slow"`) since compile-time and short-MCMC benchmarks aren't useful as regression signals.

## Authoring guidelines

### JAX gotchas

1. **Always block on the result.** Use `block(x)` from `_helpers.py`:

   ```python
   def run():
       return block(jit_fn(args))
   benchmark(run)
   ```

   Without this, JAX returns immediately with an unevaluated future and the benchmark measures dispatch overhead (~nothing).

2. **Warm the JIT cache before the timed call.** Call the function once outside `benchmark(...)` so compile cost isn't charged to the steady-state number:

   ```python
   fn = jax.jit(...)
   fn(args).block_until_ready()       # compile here
   benchmark(lambda: block(fn(args))) # measure here
   ```

3. **Compile-time benchmarks need `jax.clear_caches()`** between rounds (handled by `bench_compile.py` via `benchmark.pedantic(setup=...)`).

### Shape stability

JAX recompiles on shape or dtype changes. If a benchmark accidentally feeds different shapes on each call you'll measure compile cost every iteration. Keep inputs identical across rounds; use module-scoped fixtures for shared arrays.

### Naming

Functions starting with `test_` are auto-collected. Use the `benchmark` fixture (provided by both `pytest-benchmark` and `pytest-codspeed`) — files don't need to know which plugin is active.

### Slow vs fast

Mark anything taking >0.5s with `@pytest.mark.slow`:

- compile-time benches (cold-cache rebuilds)
- short MCMC benches
- any future end-to-end fit benchmark

Slow benches still run locally with `pixi run bench`; they're filtered out in CI.

## Adding a benchmark

1. Add a function `test_<name>` to the relevant `bench_*.py` (or create a new file). It must take `benchmark` and optionally the existing fixtures (`minimal_bench`, `single_grating_bench`, `multi_grating_bench`, `rng_key`).
2. JIT-compile the target, warm it, then `benchmark(lambda: block(fn(...)))`.
3. Locally: `pixi run bench -- benchmarks/bench_<file>.py::test_<name>` to verify it runs.
4. On PR: CodSpeed will start tracking it once merged to `main`.
