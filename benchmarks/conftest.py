"""Pytest fixtures for the unite benchmark suite.

Three canonical configs are exposed:

- ``minimal_bench``       — 1 line, 100 pixels, no continuum.
- ``single_grating_bench`` — 4 narrow + 1 broad line + linear continuum, 1 spectrum.
- ``multi_grating_bench``  — 4 narrow + 1 absorption + 2 continuum regions over
                             2 spectra (the multi-spectrum/tau path).

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
    cfg_single_grating,
    cfg_single_grating_voigt,
    cfg_single_grating_with_absorber,
    make_spectrum,
    one_prior_draw,
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


@pytest.fixture(scope='session')
def rng_key():
    """A deterministic JAX PRNG key for benchmark inputs."""
    return random.PRNGKey(0)
