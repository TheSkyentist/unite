"""Profile a representative unite fit.

Runs a short NUTS fit under :func:`jax.profiler.trace` and writes a Perfetto
trace to ``traces/``.  Open it at https://ui.perfetto.dev to get a flame graph
of XLA ops, kernel launches, and host-side Python time.

Optional flags:

- ``--memory`` — also dump a pprof device-memory profile after the fit.
- ``--pyinstrument`` — wrap the fit in a pyinstrument sampling profiler and
  print a tree to stdout afterwards.  Useful for catching surprises in the
  Python layer (accidental retracing, host-side loops in tracing time, ...).
- ``--config {minimal,single,multi}`` — which canonical config to profile.

Usage::

    pixi run profile
    pixi run profile-mem
    pixi run -e bench python scripts/profile_fit.py --config multi --pyinstrument
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the benchmark helpers importable without installing them.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax
from numpyro.infer import MCMC, NUTS

from benchmarks._helpers import (
    build_model,
    cfg_minimal,
    cfg_multi_grating,
    cfg_single_grating,
    make_spectrum,
)
from unite.spectrum import Spectra

CONFIGS = {
    'minimal': (cfg_minimal, [('min', 6500, 6600, 100)]),
    'single': (cfg_single_grating, [('sg', 6400, 6800, 400)]),
    'multi': (
        cfg_multi_grating,
        [('mg_blue', 5800, 6000, 400), ('mg_red', 6400, 6800, 400)],
    ),
}


def build(cfg_name: str):
    cfg_fn, spec_args = CONFIGS[cfg_name]
    lc, cc = cfg_fn()
    specs = [make_spectrum(lo, hi, n, name) for (name, lo, hi, n) in spec_args]
    spectra = Spectra(specs, redshift=0.0)
    return build_model(lc, cc, spectra)


def run_fit(model_fn, args, *, num_warmup: int, num_samples: int, seed: int):
    """Run a short NUTS fit and block on the result."""
    kernel = NUTS(model_fn)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(seed), args)
    samples = mcmc.get_samples()
    # Force completion before returning.
    for v in samples.values():
        v.block_until_ready()
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', choices=list(CONFIGS), default='single')
    p.add_argument('--num-warmup', type=int, default=200)
    p.add_argument('--num-samples', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument(
        '--out-dir',
        type=Path,
        default=ROOT / 'artifacts' / 'profile',
        help='Directory for the Perfetto trace and memory pprof.',
    )
    p.add_argument(
        '--memory',
        action='store_true',
        help='Also dump a device-memory pprof profile after the fit.',
    )
    p.add_argument(
        '--pyinstrument',
        action='store_true',
        help='Wrap the fit in a pyinstrument sampling profiler.',
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[profile_fit] Building config={args.config!r} ...', flush=True)
    model_fn, model_args = build(args.config)

    # ------------------------------------------------------------------
    # Warm the JIT cache so the trace measures steady-state behavior.
    # ------------------------------------------------------------------
    print('[profile_fit] Warming up (JIT compile + 1 short run) ...', flush=True)
    _ = run_fit(model_fn, model_args, num_warmup=5, num_samples=5, seed=args.seed)

    # ------------------------------------------------------------------
    # Optional pyinstrument wrapper for the Python-side profile.
    # ------------------------------------------------------------------
    pyi = None
    if args.pyinstrument:
        try:
            from pyinstrument import Profiler
        except ImportError as e:  # pragma: no cover
            raise SystemExit(
                '--pyinstrument requested but pyinstrument is not installed. '
                'Install via the `bench` env: `pixi install -e bench`.'
            ) from e
        pyi = Profiler()
        pyi.start()

    # ------------------------------------------------------------------
    # The actual measured run, traced by jax.profiler.
    # ------------------------------------------------------------------
    print(
        f'[profile_fit] Tracing fit: warmup={args.num_warmup}, '
        f'samples={args.num_samples} -> {args.out_dir}',
        flush=True,
    )
    t0 = time.perf_counter()
    with jax.profiler.trace(str(args.out_dir), create_perfetto_link=False):
        run_fit(
            model_fn,
            model_args,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            seed=args.seed + 1,
        )
    elapsed = time.perf_counter() - t0
    print(f'[profile_fit] Fit took {elapsed:.2f} s wall time.', flush=True)

    if pyi is not None:
        pyi.stop()
        print('\n[profile_fit] pyinstrument tree:\n', flush=True)
        print(pyi.output_text(unicode=True, color=True), flush=True)

    if args.memory:
        mem_path = args.out_dir / 'memory.pprof'
        jax.profiler.save_device_memory_profile(str(mem_path))
        print(
            f'[profile_fit] Device memory profile written to {mem_path}\n'
            f'  Inspect with: `go tool pprof -text {mem_path}` '
            f'(or any pprof viewer).',
            flush=True,
        )

    print(
        '\n[profile_fit] Done. Open the Perfetto trace from '
        f'{args.out_dir} at https://ui.perfetto.dev',
        flush=True,
    )


if __name__ == '__main__':
    main()
