"""Batch-size scaling of the unite log-density gradient.

NUTS needs ``value_and_grad`` of the log-density once per leapfrog step.  For a
single source this is a small kernel, so on an accelerator it is typically
launch-latency bound and the device is mostly idle.  This script vmaps
``value_and_grad`` over ``B`` parameter vectors — equivalent to running ``B``
vectorized chains of one source, and, compute-wise, to ``B`` same-shape sources
— and reports how wall time grows with ``B``.

Reading the table:

- ``ms/call`` flat as ``B`` grows → the device is latency bound; batching is
  nearly free.
- ``us/eval`` (time per parameter vector) falls until the device saturates,
  then plateaus.  The knee is the useful batch size for this device/config.
- ``speedup`` is throughput relative to ``B`` sequential single evaluations.

Run it on the hardware you care about (CPU and GPU give very different curves)::

    pixi run -e bench python profiling/batch_scaling.py
    pixi run -e bench python profiling/batch_scaling.py --config mixed --mode convolution
    pixi run -e bench python profiling/batch_scaling.py --batches 1 8 64 512 4096
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import jax
from jax import random
from numpyro.infer import Predictive
from numpyro.infer.util import log_density

from benchmarks._helpers import (
    build_model,
    cfg_minimal,
    cfg_multi_grating,
    cfg_single_grating,
    cfg_single_grating_mixed,
    make_spectrum,
)
from unite.spectrum import Spectra

CONFIGS = {
    'minimal': (cfg_minimal, [(6500, 6600, 100)]),
    'single': (cfg_single_grating, [(6400, 6800, 400)]),
    'mixed': (cfg_single_grating_mixed, [(6400, 6800, 400)]),
    'multi': (cfg_multi_grating, [(5800, 6000, 400), (6400, 6800, 400)]),
}


def _build(config: str, mode: str, n_pixels: int | None):
    cfg_fn, windows = CONFIGS[config]
    lc, cc = cfg_fn()
    spectra = Spectra(
        [
            make_spectrum(lo, hi, n_pixels or n, f'{config}_{i}')
            for i, (lo, hi, n) in enumerate(windows)
        ],
        redshift=0.0,
    )
    return build_model(lc, cc, spectra, integration_mode=mode)


def _prior_draws(model_fn, args, n: int) -> dict:
    """``n`` prior draws of all latent sites, each leaf shaped ``(n, ...)``."""
    draws = Predictive(model_fn, num_samples=n)(random.PRNGKey(0), args)
    return {k: v for k, v in draws.items() if not k.startswith('obs_')}


def _time(fn, params, min_seconds: float = 0.3, max_iters: int = 2000) -> float:
    """Median-of-batches wall time (seconds) per call, after warmup."""
    jax.block_until_ready(fn(params))
    jax.block_until_ready(fn(params))
    t0 = time.perf_counter()
    jax.block_until_ready(fn(params))
    one = max(time.perf_counter() - t0, 1e-7)
    iters = int(min(max_iters, max(5, min_seconds / one)))
    samples = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            out = fn(params)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) / iters)
    return sorted(samples)[len(samples) // 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--config', choices=CONFIGS, default='single')
    ap.add_argument('--mode', choices=['analytic', 'convolution'], default='analytic')
    ap.add_argument(
        '--batches',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    )
    ap.add_argument(
        '--n-pixels', type=int, default=None, help='override pixels per spectrum'
    )
    ns = ap.parse_args()

    model_fn, args = _build(ns.config, ns.mode, ns.n_pixels)
    n_pix = sum(int(s.low.shape[0]) for s in args.spectra)
    n_lines = int(args.matrices.wavelengths.shape[0])

    def logp(params):
        lp, _ = log_density(model_fn, (args,), {}, params)
        return lp

    batched = jax.jit(jax.vmap(jax.value_and_grad(logp)))
    draws = _prior_draws(model_fn, args, max(ns.batches))

    dev = jax.devices()[0]
    print(
        f'device={dev.platform}:{dev.device_kind}  x64={jax.config.jax_enable_x64}  '
        f'config={ns.config}  mode={ns.mode}  lines={n_lines}  pixels={n_pix}'
    )
    print(f'{'B':>6} {'ms/call':>10} {'us/eval':>10} {'speedup':>9}')

    t1 = None
    for b in ns.batches:
        params = {k: v[:b] for k, v in draws.items()}
        t = _time(batched, params)
        if t1 is None:
            t1 = t / b  # per-eval cost of the first (smallest) batch
        print(f'{b:>6} {t * 1e3:>10.3f} {t / b * 1e6:>10.2f} {t1 / (t / b):>8.1f}x')


if __name__ == '__main__':
    main()
