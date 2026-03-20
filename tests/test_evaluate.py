"""Tests for the model evaluator (unite.evaluate)."""

import astropy.units as u
import numpy as np
from jax import random
from numpyro.infer import Predictive

from unite import line, model, prior
from unite.continuum.config import ContinuumConfiguration, ContinuumRegion
from unite.continuum.library import Linear
from unite.evaluate import SpectrumPrediction, evaluate_model
from unite.instrument.base import PixOffset
from unite.instrument.generic import GenericDisperser, SimpleDisperser
from unite.spectrum import Spectra, Spectrum


def _setup():
    """Create a simple model and get posterior samples."""
    wavelength = np.linspace(6500, 6600, 80) * u.AA
    disperser = SimpleDisperser(wavelength=wavelength, R=3000.0, name='test')
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)

    sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
    line_flux = 50 * np.exp(-0.5 * ((wavelength.value - 6563.0) / sigma) ** 2)
    rng = np.random.default_rng(42)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    flux = (line_flux + 5.0 + rng.normal(0, 1, len(wavelength))) * flux_unit
    error = np.full(len(wavelength), 1.0) * flux_unit

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
        fwhm_gauss=line.FWHM(prior=prior.Uniform(100, 1000)),
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


# ---------------------------------------------------------------------------
# Fixed-prior coverage (evaluate.py lines 74-75, 82-83)
# ---------------------------------------------------------------------------


def _setup_fixed():
    """Model where all line parameters are Fixed (no sampled params)."""
    wavelength = np.linspace(6500, 6600, 80) * u.AA
    disperser = SimpleDisperser(wavelength=wavelength, R=3000.0, name='fixed_test')
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    flux = np.ones(len(wavelength)) * flux_unit
    error = np.ones(len(wavelength)) * flux_unit

    spectrum = Spectrum(
        low=low,
        high=high,
        flux=flux,
        error=error,
        disperser=disperser,
        name='fixed_spec',
    )

    lc = line.LineConfiguration()
    lc.add_line(
        'Ha',
        6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Fixed(0.0)),
        fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
        flux=line.Flux(prior=prior.Fixed(1.0)),
    )

    spectra = Spectra([spectrum], redshift=0.0)
    spectra.prepare(lc)
    spectra.compute_scales(spectra.prepared_line_config)

    _model_fn, args = model.ModelBuilder(
        spectra.prepared_line_config, None, spectra
    ).build()

    return args


class TestEvaluateFixed:
    """Tests covering Fixed-prior paths in evaluate_model."""

    def test_fixed_priors_no_samples_needed(self):
        """evaluate_model with all Fixed priors: empty samples dict, n_samples=1."""
        args = _setup_fixed()
        # All params are Fixed → pass empty samples dict (line 75 branch)
        results = evaluate_model({}, args)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_fixed_n_samples_is_1(self):
        """When all params are Fixed, n_samples defaults to 1 (line 83)."""
        args = _setup_fixed()
        results = evaluate_model({}, args)
        pred = results[0]
        # n_samples=1 → shape (1, n_pix)
        assert pred.total.shape[0] == 1


# ---------------------------------------------------------------------------
# PixOffset coverage (evaluate.py lines 104-106, 177-179)
# ---------------------------------------------------------------------------


def _setup_pix_offset():
    """Model with a disperser that has a PixOffset token."""
    wavelength = np.linspace(6500, 6600, 80) * u.AA
    import jax.numpy as jnp

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
    flux = np.ones(len(wavelength)) * flux_unit
    error = np.ones(len(wavelength)) * flux_unit

    spectrum = Spectrum(
        low=low, high=high, flux=flux, error=error, disperser=disperser, name='pix_spec'
    )

    lc = line.LineConfiguration()
    lc.add_line(
        'Ha',
        6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Fixed(0.0)),
        fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
        flux=line.Flux(prior=prior.Fixed(1.0)),
    )

    spectra = Spectra([spectrum], redshift=0.0)
    spectra.prepare(lc)
    spectra.compute_scales(spectra.prepared_line_config)

    _model_fn, args = model.ModelBuilder(
        spectra.prepared_line_config, None, spectra
    ).build()

    return args


class TestEvaluatePixOffset:
    """Tests covering pix_offset paths in evaluate_model."""

    def test_pix_offset_runs(self):
        """evaluate_model with pix_offset disperser completes successfully."""
        args = _setup_pix_offset()
        results = evaluate_model({}, args)
        assert isinstance(results, list)
        assert len(results) == 1
        pred = results[0]
        assert np.all(np.isfinite(pred.total))


# ---------------------------------------------------------------------------
# Continuum coverage (evaluate.py lines 194-214, 230-232)
# ---------------------------------------------------------------------------


def _setup_continuum():
    """Model with a continuum configuration."""
    wavelength = np.linspace(6500, 6600, 80) * u.AA
    disperser = SimpleDisperser(wavelength=wavelength, R=3000.0, name='cont_test')
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    flux = np.ones(len(wavelength)) * 5.0 * flux_unit
    error = np.ones(len(wavelength)) * flux_unit

    spectrum = Spectrum(
        low=low,
        high=high,
        flux=flux,
        error=error,
        disperser=disperser,
        name='cont_spec',
    )

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

    spectra = Spectra([spectrum], redshift=0.0)
    spectra.prepare(lc, cc)
    spectra.compute_scales(spectra.prepared_line_config)

    model_fn, args = model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build()

    rng_key = random.PRNGKey(1)
    predictive = Predictive(model_fn, num_samples=3)
    samples = predictive(rng_key, args)

    return samples, args


class TestEvaluateContinuum:
    """Tests covering continuum evaluation paths in evaluate_model."""

    def test_continuum_regions_populated(self):
        """evaluate_model with continuum: continuum_regions dict is non-empty."""
        samples, args = _setup_continuum()
        results = evaluate_model(samples, args)
        pred = results[0]
        assert len(pred.continuum_regions) > 0

    def test_continuum_keys_are_strings(self):
        """Continuum region keys are informative string labels."""
        samples, args = _setup_continuum()
        results = evaluate_model(samples, args)
        pred = results[0]
        for key in pred.continuum_regions:
            assert isinstance(key, str)

    def test_continuum_shapes(self):
        """Continuum region arrays have shape (n_samples, n_pix)."""
        samples, args = _setup_continuum()
        results = evaluate_model(samples, args)
        pred = results[0]
        n_pix = pred.wavelength.shape[0]
        for arr in pred.continuum_regions.values():
            assert arr.shape == (3, n_pix)

    def test_total_includes_continuum(self):
        """Total prediction is finite when continuum is present."""
        samples, args = _setup_continuum()
        results = evaluate_model(samples, args)
        pred = results[0]
        assert np.all(np.isfinite(pred.total))
