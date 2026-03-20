"""Tests for the model evaluator (unite.evaluate)."""

import astropy.units as u
import numpy as np
import pytest
from jax import random
from numpyro.infer import Predictive

from unite import line, model, prior
from unite.continuum.config import ContinuumConfiguration, ContinuumRegion
from unite.continuum.library import Linear
from unite.evaluate import SpectrumPrediction, evaluate_model
from unite.instrument.base import PixOffset
from unite.instrument.generic import GenericDisperser, SimpleDisperser
from unite.spectrum import Spectra, Spectrum


def _make_spectrum(name='test', npix=80):

    wavelength = np.linspace(6500, 6600, npix) * u.AA
    disperser = SimpleDisperser(wavelength=wavelength, R=3000.0, name=name)
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)
    sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    line_flux = 50 * np.exp(-0.5 * ((wavelength.value - 6563.0) / sigma) ** 2)
    rng = np.random.default_rng(42)
    flux = (line_flux + 5.0 + rng.normal(0, 1, npix)) * flux_unit
    error = np.full(npix, 1.0) * flux_unit
    return Spectrum(
        low=low, high=high, flux=flux, error=error, disperser=disperser, name=name
    )


def _build_model(spectrum, lc, cc=None):
    spectra = Spectra([spectrum], redshift=0.0)
    spectra.prepare(lc, cc)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    return model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build()


def _simple_lc():
    lc = line.LineConfiguration()
    lc.add_line(
        'Ha',
        6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Uniform(-0.005, 0.005)),
        fwhm_gauss=line.FWHM(prior=prior.Uniform(100, 1000)),
        flux=line.Flux(prior=prior.Uniform(0, 5)),
    )
    return lc


@pytest.fixture(scope='module')
def simple_setup():
    """Lines-only model with predictive samples."""
    spec = _make_spectrum()
    model_fn, args = _build_model(spec, _simple_lc())
    samples = Predictive(model_fn, num_samples=5)(random.PRNGKey(0), args)
    return samples, args, spec


@pytest.fixture(scope='module')
def fixed_args():
    """Model where all line parameters are Fixed (no sampled params)."""
    spec = _make_spectrum(name='fixed_spec')
    lc = line.LineConfiguration()
    lc.add_line(
        'Ha',
        6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Fixed(0.0)),
        fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
        flux=line.Flux(prior=prior.Fixed(1.0)),
    )
    _model_fn, args = _build_model(spec, lc)
    return args


@pytest.fixture(scope='module')
def pix_offset_args():
    """Model with a disperser that has a PixOffset token."""
    import jax.numpy as jnp

    wavelength = np.linspace(6500, 6600, 80) * u.AA
    pix_off = PixOffset(prior=prior.Fixed(0.0), name='test_pix_off')
    disperser = GenericDisperser(
        R_func=lambda w: jnp.full_like(w, 3000.0),
        dlam_dpix_func=lambda w: w / 3000.0,
        unit=u.AA,
        name='pix_test',
        pix_offset=pix_off,
    )
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    spec = Spectrum(
        low=low,
        high=high,
        flux=np.ones(len(wavelength)) * flux_unit,
        error=np.ones(len(wavelength)) * flux_unit,
        disperser=disperser,
        name='pix_spec',
    )
    lc = line.LineConfiguration()
    lc.add_line(
        'Ha',
        6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Fixed(0.0)),
        fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
        flux=line.Flux(prior=prior.Fixed(1.0)),
    )
    spectra = Spectra([spec], redshift=0.0)
    spectra.prepare(lc)
    spectra.compute_scales(spectra.prepared_line_config)
    _model_fn, args = model.ModelBuilder(
        spectra.prepared_line_config, None, spectra
    ).build()
    return args


@pytest.fixture(scope='module')
def continuum_setup():
    """Model with a continuum configuration and predictive samples."""
    spec = _make_spectrum(name='cont_spec')
    lc = line.LineConfiguration()
    lc.add_line(
        'Ha',
        6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Fixed(0.0)),
        fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
        flux=line.Flux(prior=prior.Fixed(1.0)),
    )
    cc = ContinuumConfiguration(
        [ContinuumRegion(6490.0 * u.AA, 6620.0 * u.AA, form=Linear())]
    )
    model_fn, args = _build_model(spec, lc, cc)
    samples = Predictive(model_fn, num_samples=3)(random.PRNGKey(1), args)
    return samples, args


class TestEvaluateModel:
    """Tests for evaluate_model with a lines-only model."""

    def test_returns_predictions(self, simple_setup):
        samples, args, _spec = simple_setup
        results = evaluate_model(samples, args)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], SpectrumPrediction)

    def test_prediction_shapes(self, simple_setup):
        samples, args, spec = simple_setup
        pred = evaluate_model(samples, args)[0]
        assert pred.wavelength.shape == (spec.npix,)
        assert pred.total.shape == (5, spec.npix)
        assert len(pred.lines) == 1
        assert len(pred.continuum_regions) == 0

    def test_total_is_finite_and_lines_sum(self, simple_setup):
        """Total is finite; line contributions sum to total (no continuum)."""
        samples, args, _ = simple_setup
        pred = evaluate_model(samples, args)[0]
        assert np.all(np.isfinite(pred.total))
        line_sum = sum(arr for arr in pred.lines.values())
        np.testing.assert_allclose(pred.total, line_sum, rtol=1e-5)


class TestEvaluateFixed:
    """Tests covering Fixed-prior paths in evaluate_model."""

    def test_fixed_priors_evaluate(self, fixed_args):
        """All-Fixed model: empty samples dict → n_samples=1, finite output."""
        results = evaluate_model({}, fixed_args)
        assert len(results) == 1
        pred = results[0]
        assert pred.total.shape[0] == 1
        assert np.all(np.isfinite(pred.total))


class TestEvaluatePixOffset:
    """Tests covering pix_offset paths in evaluate_model."""

    def test_pix_offset_runs(self, pix_offset_args):
        """evaluate_model with pix_offset disperser completes successfully."""
        results = evaluate_model({}, pix_offset_args)
        assert len(results) == 1
        assert np.all(np.isfinite(results[0].total))


class TestEvaluateContinuum:
    """Tests covering continuum evaluation paths in evaluate_model."""

    def test_continuum_populated_and_valid(self, continuum_setup):
        """Continuum regions: non-empty, string keys, correct shapes, finite total."""
        samples, args = continuum_setup
        pred = evaluate_model(samples, args)[0]
        assert len(pred.continuum_regions) > 0
        n_pix = pred.wavelength.shape[0]
        for key, arr in pred.continuum_regions.items():
            assert isinstance(key, str)
            assert arr.shape == (3, n_pix)
        assert np.all(np.isfinite(pred.total))
