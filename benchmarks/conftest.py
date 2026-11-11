"""Pytest fixtures for the unite benchmark suite.

Three canonical configs are exposed:

- ``minimal_bench``       — 1 line, 100 pixels, no continuum.
- ``single_grating_bench`` — 4 narrow + 1 broad line + linear continuum, 1 spectrum.
- ``multi_grating_bench``  — 4 narrow + 1 absorption + 2 continuum regions over
                             2 spectra (the multi-spectrum/tau path).

``overlap_by_mode`` adds a matched pair (with and without second-order
overlap) for each integration mode.

Each fixture returns a :class:`benchmarks._helpers.Bench` exposing the prepared
model fn, model args, and a single prior draw.  Models are built **once per
session** and shared across benchmark files.

We use synthetic spectra (not DJA fixtures) so the suite runs in CI without
data dependencies.
"""

from __future__ import annotations

import pytest
from jax import random

from benchmarks._helpers import (
    Bench,
    build_model,
    cfg_minimal,
    cfg_multi_grating,
    cfg_overlap,
    cfg_single_grating,
    cfg_single_grating_mixed,
    cfg_single_grating_voigt,
    cfg_single_grating_with_absorber,
    make_overlap_spectrum,
    make_spectrum,
    one_prior_draw,
    write_overlap_ratio,
)
from unite.spectrum import Spectra

INTEGRATION_MODES = ('analytic', 'convolution')


@pytest.fixture(scope='session')
def minimal_bench() -> Bench:
    """1 line, 100 pixels, no continuum."""
    lc, cc = cfg_minimal()
    spec = make_spectrum(6500, 6600, 100, 'min')
    spectra = Spectra([spec], redshift=0.0)
    model_fn, args = build_model(lc, cc, spectra)
    sample = one_prior_draw(model_fn, args)
    return Bench(model_fn=model_fn, args=args, sample=sample)


@pytest.fixture(scope='session')
def single_grating_bench() -> Bench:
    """5 lines + linear continuum, 1 spectrum, ~400 pixels."""
    lc, cc = cfg_single_grating()
    spec = make_spectrum(6400, 6800, 400, 'sg')
    spectra = Spectra([spec], redshift=0.0)
    model_fn, args = build_model(lc, cc, spectra)
    sample = one_prior_draw(model_fn, args)
    return Bench(model_fn=model_fn, args=args, sample=sample)


@pytest.fixture(scope='session')
def multi_grating_bench() -> Bench:
    """5 lines + 1 absorber across 2 spectra (~800 pixels total)."""
    lc, cc = cfg_multi_grating()
    spec_a = make_spectrum(5800, 6000, 400, 'mg_blue')
    spec_b = make_spectrum(6400, 6800, 400, 'mg_red')
    spectra = Spectra([spec_a, spec_b], redshift=0.0)
    model_fn, args = build_model(lc, cc, spectra)
    sample = one_prior_draw(model_fn, args)
    return Bench(model_fn=model_fn, args=args, sample=sample)


@pytest.fixture(scope='session')
def single_grating_absorber_bench() -> Bench:
    """Single-grating config with one extra tau absorber.

    Direct emission-only/with-absorber comparison: same emission lines and
    continuum as :func:`single_grating_bench`, plus a NaI-D-like absorber.
    """
    lc, cc = cfg_single_grating_with_absorber()
    spec = make_spectrum(6400, 6800, 400, 'sg_abs')
    spectra = Spectra([spec], redshift=0.0)
    model_fn, args = build_model(lc, cc, spectra)
    sample = one_prior_draw(model_fn, args)
    return Bench(model_fn=model_fn, args=args, sample=sample)


@pytest.fixture(scope='session')
def single_grating_voigt_bench() -> Bench:
    """5 PseudoVoigt lines + linear continuum, 1 spectrum, ~400 pixels."""
    lc, cc = cfg_single_grating_voigt()
    spec = make_spectrum(6400, 6800, 400, 'sg_voigt')
    spectra = Spectra([spec], redshift=0.0)
    model_fn, args = build_model(lc, cc, spectra)
    sample = one_prior_draw(model_fn, args)
    return Bench(model_fn=model_fn, args=args, sample=sample)


@pytest.fixture(scope='session', params=INTEGRATION_MODES)
def single_grating_voigt_by_mode(request) -> Bench:
    """PseudoVoigt single-grating built under each integration mode."""
    lc, cc = cfg_single_grating_voigt()
    spec = make_spectrum(6400, 6800, 400, f'sg_voigt_{request.param}')
    spectra = Spectra([spec], redshift=0.0)
    model_fn, args = build_model(lc, cc, spectra, integration_mode=request.param)
    sample = one_prior_draw(model_fn, args)
    return Bench(model_fn=model_fn, args=args, sample=sample)


@pytest.fixture(scope='session', params=INTEGRATION_MODES)
def single_grating_by_mode(request) -> Bench:
    """Single-grating config built under each integration mode.

    Parametrized over ``('analytic', 'convolution')`` so every benchmark
    that takes this fixture is automatically run once per mode.  Use this
    to compare mode-level cost.
    """
    lc, cc = cfg_single_grating()
    spec = make_spectrum(6400, 6800, 400, f'sg_{request.param}')
    spectra = Spectra([spec], redshift=0.0)
    model_fn, args = build_model(lc, cc, spectra, integration_mode=request.param)
    sample = one_prior_draw(model_fn, args)
    return Bench(model_fn=model_fn, args=args, sample=sample)


@pytest.fixture(scope='session', params=INTEGRATION_MODES)
def single_grating_mixed_by_mode(request) -> Bench:
    """Mixed-profile config (Gaussian + SkewVoigt + SEMG) under each mode."""
    lc, cc = cfg_single_grating_mixed()
    spec = make_spectrum(6400, 6800, 400, f'sg_mixed_{request.param}')
    spectra = Spectra([spec], redshift=0.0)
    model_fn, args = build_model(lc, cc, spectra, integration_mode=request.param)
    sample = one_prior_draw(model_fn, args)
    return Bench(model_fn=model_fn, args=args, sample=sample)


@pytest.fixture(scope='session')
def overlap_ratio_path(tmp_path_factory):
    """A second-order ``throughput_ratio`` file shared by the overlap fixtures."""
    path = tmp_path_factory.mktemp('overlap') / 'r2.ecsv'
    write_overlap_ratio(path)
    return path


@pytest.fixture(
    scope='session',
    params=[(m, o) for m in INTEGRATION_MODES for o in ('off', 'on')],
    ids=lambda p: f'{p[0]}-{p[1]}',
)
def overlap_by_mode(request, overlap_ratio_path) -> Bench:
    """7 lines over 3000-8000 Å (2000 pixels), with and without overlap.

    Parametrized over integration mode and ``'off'``/``'on'``: the two members
    of a pair differ only by the second-order term (free ``OrderScale``), so
    their ratio is the cost of the overlap.  Both have a bounded ``PixOffset``,
    which sets the overlap window margin.
    """
    mode, state = request.param
    ratio = overlap_ratio_path if state == 'on' else None
    lc, cc = cfg_overlap()
    spec = make_overlap_spectrum(f'ov_{mode}_{state}', ratio)
    spectra = Spectra([spec], redshift=0.0)
    model_fn, args = build_model(lc, cc, spectra, integration_mode=mode)
    sample = one_prior_draw(model_fn, args)
    return Bench(model_fn=model_fn, args=args, sample=sample)


@pytest.fixture(scope='session')
def rng_key():
    """A deterministic JAX PRNG key for benchmark inputs."""
    return random.PRNGKey(0)
