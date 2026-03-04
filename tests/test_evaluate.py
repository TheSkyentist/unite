"""Tests for the model evaluator (unite.evaluate)."""

import astropy.units as u
import numpy as np
from jax import random
from numpyro.infer import Predictive

from unite import line, model, prior
from unite.disperser.generic import SimpleDisperser
from unite.evaluate import SpectrumPrediction, evaluate_model
from unite.spectrum import Spectra, Spectrum


def _setup():
    """Create a simple model and get posterior samples."""
    wavelength = np.linspace(6500, 6600, 80) * u.AA
    disperser = SimpleDisperser(
        wavelength=wavelength.value, unit=u.AA, R=3000.0, name='test'
    )
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)

    sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
    line_flux = 50 * np.exp(-0.5 * ((wavelength.value - 6563.0) / sigma) ** 2)
    rng = np.random.default_rng(42)
    flux = line_flux + 5.0 + rng.normal(0, 1, len(wavelength))
    error = np.full(len(wavelength), 1.0)

    spectrum = Spectrum(
        low=low,
        high=high,
        flux=flux,
        error=error,
        disperser=disperser,
        name='test_spec',
    )

    lc = line.LineConfiguration()
    lc.add_line(
        'Ha',
        6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Uniform(-0.005, 0.005)),
        fwhm_gauss=line.FWHM(prior=prior.Uniform(2.0, 8.0)),
        flux=line.Flux(prior=prior.Uniform(0, 5)),
    )

    spectra = Spectra([spectrum], redshift=0.0)
    spectra.prepare(lc)
    spectra.compute_scales(spectra.prepared_line_config)

    model_fn, args = model.ModelBuilder(
        spectra.prepared_line_config, None, spectra
    ).build()

    # Get a few predictive samples (faster than MCMC for testing).
    rng_key = random.PRNGKey(0)
    predictive = Predictive(model_fn, num_samples=5)
    samples = predictive(rng_key, args)

    return samples, args, spectrum


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_returns_list_of_predictions(self):
        samples, args, _ = _setup()
        results = evaluate_model(samples, args)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], SpectrumPrediction)

    def test_prediction_shapes(self):
        samples, args, spectrum = _setup()
        results = evaluate_model(samples, args)
        pred = results[0]

        assert pred.wavelength.shape == (spectrum.npix,)
        assert pred.total.shape[0] == 5  # n_samples
        assert pred.total.shape[1] == spectrum.npix

    def test_lines_dict_populated(self):
        samples, args, _ = _setup()
        results = evaluate_model(samples, args)
        pred = results[0]

        # Should have one line entry.
        assert len(pred.lines) == 1
        for _key, arr in pred.lines.items():
            assert arr.shape[0] == 5  # n_samples

    def test_no_continuum_regions_for_lines_only(self):
        samples, args, _ = _setup()
        results = evaluate_model(samples, args)
        pred = results[0]
        assert len(pred.continuum_regions) == 0

    def test_total_is_finite(self):
        samples, args, _ = _setup()
        results = evaluate_model(samples, args)
        pred = results[0]
        assert np.all(np.isfinite(pred.total))

    def test_lines_sum_to_total_without_continuum(self):
        """For a lines-only model, the sum of line contributions should equal total."""
        samples, args, _ = _setup()
        results = evaluate_model(samples, args)
        pred = results[0]

        line_sum = sum(arr for arr in pred.lines.values())
        np.testing.assert_allclose(pred.total, line_sum, rtol=1e-5)
