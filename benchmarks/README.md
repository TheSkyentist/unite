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

### Recording runs and comparing backwards

Results are stored privately in `.benchmarks/<machine>/` (gitignored), so history is per machine and per Python.

```bash
pixi run bench-save                          # fast set -> .benchmarks/<machine>/0004_<commit>_<date>.json
pixi run bench-compare                       # fast set, compared to the latest saved run
pixi run bench-compare --benchmark-compare=0002              # ...or to an older run
pixi run bench-compare --benchmark-compare-fail=median:25%   # exit non-zero on a big regression
pixi run -e bench pytest-benchmark compare 0002 0004         # two saved runs, no re-run
```

- The file name carries the commit hash, plus `_uncommitted-changes` if the tree was dirty. Dependencies are pinned per commit by `pixi.lock`, so the commit identifies the environment.
- Benchmarks are matched by name; ones missing on either side are shown as new or dropped, not compared.
- A single run is noisy. Keep the machine quiet and plugged in, and treat differences under ~10% as nothing until a repeat agrees. If a result is surprising, re-run the baseline on the same machine.
- Changing a benchmark's workload without renaming it makes old results incomparable; rename it when you do.

### Interactive profiling

```bash
pixi run profile                             # default: single_grating config, writes Perfetto trace
pixi run profile-mem                         # also dumps device memory pprof
pixi run -e bench python profiling/profile_fit.py --config multi --pyinstrument
```

Open the trace from `traces/` at <https://ui.perfetto.dev> for a flame graph of XLA ops and host-side Python.

## CI

There is none for benchmarks; they are run locally before merging (see above).

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

Functions starting with `test_` are auto-collected. Use the `benchmark` fixture (from `pytest-benchmark`).

### Slow vs fast

Mark anything taking >0.5s with `@pytest.mark.slow`:

- compile-time benches (cold-cache rebuilds)
- short MCMC benches
- any future end-to-end fit benchmark

Slow benches still run locally with `pixi run bench`; `bench-save` and `bench-compare` skip them.

## Adding a benchmark

1. Add a function `test_<name>` to the relevant `bench_*.py` (or create a new file). It must take `benchmark` and optionally the existing fixtures (`minimal_bench`, `single_grating_bench`, `multi_grating_bench`, `rng_key`).
2. JIT-compile the target, warm it, then `benchmark(lambda: block(fn(...)))`.
3. Locally: `pixi run bench -- benchmarks/bench_<file>.py::test_<name>` to verify it runs.
4. `pixi run bench-save` to include it in the recorded history.
