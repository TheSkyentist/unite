"""Benchmarks for result extraction.

Covers the three main post-fit paths:

- :func:`~unite.compute.evaluate_model` — vmapped JAX forward pass over
  posterior samples; produces per-line and per-continuum-region predictions.
- :func:`~unite.results.make_parameter_table` — builds an Astropy table of
  posterior parameters in physical units.  For configs with absorbers it calls
  ``evaluate_model`` internally to compute absorption equivalent widths.
- :func:`~unite.results.make_spectra_tables` — always calls ``evaluate_model``
  then does NumPy post-processing to build per-spectrum decomposition tables.

JAX gotcha: ``evaluate_model`` defines its vmapped kernel (``_single``) as a
new closure on each call, so JAX retraces the Python function every invocation.
The XLA compilation may be cached, but Python-level tracing is not.  The
warmup call below populates the XLA cache; the benchmark then measures Python
tracing + XLA dispatch — i.e., the real user-facing cost.
"""

from __future__ import annotations

import numpy as np

from benchmarks._helpers import Bench
from unite.compute import evaluate_model
from unite.results import make_parameter_table, make_spectra_tables

N_SAMPLES = 100


def _mock_samples(bench: Bench, n: int) -> dict[str, np.ndarray]:
    """Broadcast a single prior draw to a (n,) leading axis for every parameter."""
    return {k: np.broadcast_to(np.asarray(v), (n,)) for k, v in bench.sample.items()}


# ----------------------------------------------------------------------
# evaluate_model — vmapped JAX, measures Python trace + XLA dispatch
# ----------------------------------------------------------------------


def test_evaluate_model_single_grating(benchmark, single_grating_bench):
    """evaluate_model on 100 samples: single-grating config (emission + continuum)."""
    b = single_grating_bench
    samples = _mock_samples(b, N_SAMPLES)
    evaluate_model(samples, b.args)  # warm XLA cache
    benchmark(lambda: evaluate_model(samples, b.args))


def test_evaluate_model_multi_grating(benchmark, multi_grating_bench):
    """evaluate_model on 100 samples: multi-grating (2 spectra, 1 absorber)."""
    b = multi_grating_bench
    samples = _mock_samples(b, N_SAMPLES)
    evaluate_model(samples, b.args)
    benchmark(lambda: evaluate_model(samples, b.args))


# ----------------------------------------------------------------------
# make_parameter_table
# Two paths depending on whether absorbers are present:
#   - emission-only → pure Python/NumPy, no JAX (no warmup needed)
#   - with absorber → _compute_rew_columns calls evaluate_model internally
# ----------------------------------------------------------------------


def test_make_parameter_table_emission(benchmark, single_grating_bench):
    """make_parameter_table with no absorbers: pure NumPy post-processing."""
    b = single_grating_bench
    samples = _mock_samples(b, N_SAMPLES)
    benchmark(lambda: make_parameter_table(samples, b.args))


def test_make_parameter_table_with_absorber(benchmark, single_grating_absorber_bench):
    """make_parameter_table with one absorber: triggers evaluate_model for absorption REW."""
    b = single_grating_absorber_bench
    samples = _mock_samples(b, N_SAMPLES)
    make_parameter_table(samples, b.args)  # warm XLA cache inside _compute_rew_columns
    benchmark(lambda: make_parameter_table(samples, b.args))


# ----------------------------------------------------------------------
# make_spectra_tables — always calls evaluate_model, then NumPy table-building
# ----------------------------------------------------------------------


def test_make_spectra_tables_single_grating(benchmark, single_grating_bench):
    """make_spectra_tables on 100 samples: single-grating (evaluate_model + table build)."""
    b = single_grating_bench
    samples = _mock_samples(b, N_SAMPLES)
    make_spectra_tables(samples, b.args)  # warm XLA cache
    benchmark(lambda: make_spectra_tables(samples, b.args))


def test_make_spectra_tables_multi_grating(benchmark, multi_grating_bench):
    """make_spectra_tables on 100 samples: multi-grating (2 spectra, absorber)."""
    b = multi_grating_bench
    samples = _mock_samples(b, N_SAMPLES)
    make_spectra_tables(samples, b.args)
    benchmark(lambda: make_spectra_tables(samples, b.args))
