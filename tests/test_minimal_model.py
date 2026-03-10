"""Minimal tests for unite_model function without full MCMC sampling."""

import warnings

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro.infer import Predictive

from unite import line, model, prior
from unite.instrument import Spectra
from unite.instrument.generic import GenericSpectrum, SimpleDisperser


def create_simple_spectrum():
    """Create a simple test spectrum with a single Gaussian line."""
    # Create a simple wavelength grid around 6563 Å (H-alpha)
    wavelength = np.linspace(6500, 6600, 100) * u.AA

    # Create a simple disperser with constant R=3000
    disperser = SimpleDisperser(
        wavelength=wavelength.value, unit=u.AA, R=3000.0, name='test_disperser'
    )

    # Create pixel edges (low and high)
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)

    # Create a simple Gaussian line profile centered at 6563 Å with FWHM ~5Å
    center_wl = 6563.0
    fwhm_wl = 5.0  # FWHM in wavelength units
    sigma = fwhm_wl / (2 * np.sqrt(2 * np.log(2)))

    # Create Gaussian line flux
    line_flux = 100 * np.exp(-0.5 * ((wavelength.value - center_wl) / sigma) ** 2)

    # Add some noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 2, len(wavelength))
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    flux = (line_flux + noise) * flux_unit
    error = np.full(len(wavelength), 2.0) * flux_unit

    # Create spectrum
    spectrum = GenericSpectrum(
        low=low,
        high=high,
        flux=flux,
        error=error,
        disperser=disperser,
        name='test_spectrum',
    )

    return spectrum


def create_minimal_config():
    """Create minimal line configuration with one line."""
    # Create line configuration
    line_config = line.LineConfiguration()

    # Add a single H-alpha line with variable parameters.
    # Redshift prior is kept narrow enough that the line center stays within
    # the 6500-6600 Å spectral range (z=±0.005 → Δλ ≈ ±33 Å from 6563 Å).
    # FWHM prior uses physically sensible km/s values that are wide enough
    # to span multiple pixels (pixel width ≈ 1 Å ≈ 46 km/s at H-alpha).
    line_config.add_line(
        'H_alpha',
        center=6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Uniform(-0.005, 0.005)),
        fwhm_gauss=line.FWHM(prior=prior.Uniform(100.0, 1000.0)),
        flux=line.Flux(prior=prior.Uniform(50.0, 150.0)),
    )

    return line_config


def _prepare_and_build(line_config, cont_config, spectra):
    """Helper: prepare spectra and build model (no warnings)."""
    spectra.prepare(line_config, cont_config)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    return model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build()


class TestMinimalModel:
    """Test suite for minimal unite_model validation."""

    def test_model_building(self):
        """Test that unite_model can be built without errors."""
        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()
        spectra = Spectra([spectrum], redshift=0.0)

        unite_model, unite_args = _prepare_and_build(line_config, None, spectra)

        # Check that we have the expected components
        assert unite_args is not None
        assert unite_model is not None
        assert len(unite_args.matrices.flux_names) == 1
        assert len(unite_args.matrices.z_names) == 1
        assert len(unite_args.matrices.p0_names) == 1

    def test_predictive_execution(self):
        """Test that the model can execute predictively."""
        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()
        spectra = Spectra([spectrum], redshift=0.0)

        unite_model, unite_args = _prepare_and_build(line_config, None, spectra)

        # Test predictive execution
        rng_key = random.PRNGKey(0)
        predictive = Predictive(unite_model, num_samples=3)
        samples = predictive(rng_key, unite_args)

        # Check that we get the expected auto-named parameters
        # Auto-naming uses "{name}-{wavelength}-{param}" pattern
        actual_params = set(samples.keys())

        # Check that line parameters exist (at least flux, z, fwhm)
        flux_params = [k for k in actual_params if 'flux' in k and 'scale' not in k]
        z_params = [k for k in actual_params if k.endswith('-z')]
        fwhm_params = [k for k in actual_params if 'fwhm' in k]

        assert len(flux_params) >= 1, f'No flux parameters found in {actual_params}'
        assert len(z_params) >= 1, f'No z parameters found in {actual_params}'
        assert len(fwhm_params) >= 1, f'No fwhm parameters found in {actual_params}'

        # Check shapes
        for param in flux_params + z_params + fwhm_params:
            assert samples[param].shape == (3,)  # 3 samples

    def test_model_with_continuum(self):
        """Test model building with continuum."""
        from unite.continuum import ContinuumConfiguration, Linear

        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()

        # Create simple continuum (note: must pass Quantity with units)
        cont_config = ContinuumConfiguration.from_lines(
            [6563.0] * u.AA, pad=0.1, form=Linear()
        )

        spectra = Spectra([spectrum], redshift=0.0)

        # Build model with continuum
        unite_model, unite_args = _prepare_and_build(line_config, cont_config, spectra)

        # Check that continuum configuration is included
        assert unite_args.cont_config is not None

        # Test predictive execution
        rng_key = random.PRNGKey(1)
        predictive = Predictive(unite_model, num_samples=2)
        samples = predictive(rng_key, unite_args)

        # Check for continuum parameters
        continuum_params = [k for k in samples if k.startswith('cont_')]
        assert len(continuum_params) > 0, 'No continuum parameters found'

    @pytest.mark.slow
    def test_quick_mcmc(self):
        """Test that MCMC can run and produce reasonable results."""
        from numpyro import infer

        spectrum = create_simple_spectrum()

        # Create line configuration with default priors (normalization makes
        # the default Uniform(-3, 3) flux prior appropriate).
        line_config = line.LineConfiguration()
        line_config.add_line(
            'H_alpha',
            center=6563.0 * u.AA,
            redshift=line.Redshift(prior=prior.Uniform(-0.005, 0.005)),
            fwhm_gauss=line.FWHM(prior=prior.Uniform(10, 100)),
        )

        spectra = Spectra([spectrum], redshift=0.0)
        unite_model, unite_args = _prepare_and_build(line_config, None, spectra)

        # Run a very quick MCMC
        rng_key = random.PRNGKey(2)
        kernel = infer.NUTS(unite_model)
        mcmc = infer.MCMC(kernel, num_samples=10, num_warmup=5, progress_bar=False)
        mcmc.run(rng_key, unite_args)

        # Check that we got samples for all parameters
        samples = mcmc.get_samples()

        # Find flux, z, fwhm parameters by pattern
        flux_key = next(k for k in samples if 'flux' in k and 'scale' not in k)
        z_key = next(k for k in samples if k.endswith('-z'))
        fwhm_key = next(k for k in samples if 'fwhm' in k)

        assert len(samples[flux_key]) == 10
        assert len(samples[z_key]) == 10
        assert len(samples[fwhm_key]) == 10

    def test_parameter_matrices(self):
        """Test that parameter matrices are correctly constructed."""
        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()
        spectra = Spectra([spectrum], redshift=0.0)

        _, unite_args = _prepare_and_build(line_config, None, spectra)
        matrices = unite_args.matrices

        # Check matrix shapes
        assert matrices.wavelengths.shape == (1,)
        assert matrices.strengths.shape == (1,)
        assert matrices.profile_codes.shape == (1,)
        assert matrices.flux_matrix.shape == (1, 1)
        assert matrices.z_matrix.shape == (1, 1)
        assert matrices.p0_matrix.shape == (1, 1)

        # Check that matrices are properly populated
        assert jnp.sum(matrices.flux_matrix) == 1.0
        assert jnp.sum(matrices.z_matrix) == 1.0
        assert jnp.sum(matrices.p0_matrix) == 1.0

    def test_spectra_integration(self):
        """Test that spectra are properly integrated into the model."""
        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()
        spectra = Spectra([spectrum], redshift=0.0)

        _, unite_args = _prepare_and_build(line_config, None, spectra)

        # Check spectra properties
        assert len(unite_args.spectra) == 1
        assert unite_args.spectra[0].npix == 100
        assert unite_args.redshift == 0.0

    def test_auto_prepare_warns(self):
        """ModelBuilder should warn when spectra are not prepared."""
        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()
        spectra = Spectra([spectrum], redshift=0.0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            model.ModelBuilder(line_config, None, spectra)
            msgs = [str(x.message) for x in w]
            assert any('Spectra not prepared' in m for m in msgs)
            assert any('Line scale not set' in m for m in msgs)


class TestModelOutputValidation:
    """Test that model outputs are reasonable."""

    def test_model_predictions_shape(self):
        """Test that model predictions have the correct shape."""
        from numpyro import infer

        from unite.evaluate import evaluate_model

        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()
        spectra = Spectra([spectrum], redshift=0.0)

        unite_model, unite_args = _prepare_and_build(line_config, None, spectra)

        # Run quick MCMC
        rng_key = random.PRNGKey(3)
        kernel = infer.NUTS(unite_model)
        mcmc = infer.MCMC(kernel, num_samples=5, num_warmup=3, progress_bar=False)
        mcmc.run(rng_key, unite_args)

        posterior_samples = mcmc.get_samples()

        # obs sites are still in the trace.
        obs_pred = posterior_samples.get('obs_test_spectrum')
        # obs site is observed so it won't appear in posterior samples;
        # use Predictive to get it.
        predictive = Predictive(unite_model, posterior_samples)
        predictions = predictive(rng_key, unite_args)
        obs_pred = predictions['obs_test_spectrum']
        assert obs_pred.shape == (5, 100)
        assert jnp.all(jnp.isfinite(obs_pred))

        # Model predictions via evaluate_model (physical units, decomposed).
        preds = evaluate_model(posterior_samples, unite_args)
        assert len(preds) == 1
        assert preds[0].total.shape == (5, 100)  # 5 samples, 100 pixels
        assert jnp.all(jnp.isfinite(preds[0].total))

    def test_model_fits_data_range(self):
        """Test that model predictions are finite and not all zeros."""
        from numpyro import infer

        from unite.evaluate import evaluate_model

        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()
        spectra = Spectra([spectrum], redshift=0.0)

        unite_model, unite_args = _prepare_and_build(line_config, None, spectra)

        # Run quick MCMC
        rng_key = random.PRNGKey(4)
        kernel = infer.NUTS(unite_model)
        mcmc = infer.MCMC(kernel, num_samples=5, num_warmup=3, progress_bar=False)
        mcmc.run(rng_key, unite_args)

        posterior_samples = mcmc.get_samples()
        preds = evaluate_model(posterior_samples, unite_args)
        mean_model = jnp.mean(jnp.asarray(preds[0].total), axis=0)

        print(mean_model)

        # evaluate_model returns physical flux units.
        assert jnp.all(jnp.isfinite(mean_model))
        assert unite_args.norm_factors[0] > 0
        assert jnp.max(jnp.abs(mean_model)) > 0  # not all zeros


# ---------------------------------------------------------------------------
# Wavelength unit consistency and normalization tests
# ---------------------------------------------------------------------------


class TestWavelengthUnitConsistency:
    """Tests for cross-unit wavelength handling and flux normalization."""

    def _make_spectrum(self, unit, *, name=''):
        """Create a test spectrum in the given wavelength unit."""
        wl_aa = np.linspace(6500, 6600, 100) * u.AA
        wl = wl_aa.to(unit)
        disperser = SimpleDisperser(
            wavelength=wl.value, unit=unit, R=3000.0, name=name or str(unit)
        )
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)

        center_wl = (6563.0 * u.AA).to(unit).value
        fwhm_wl = (5.0 * u.AA).to(unit).value
        sigma = fwhm_wl / (2 * np.sqrt(2 * np.log(2)))
        line_flux = 100 * np.exp(-0.5 * ((wl.value - center_wl) / sigma) ** 2)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 2, len(wl))
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        flux = (line_flux + noise) * flux_unit
        error = np.full(len(wl), 2.0) * flux_unit
        return GenericSpectrum(
            low=low,
            high=high,
            flux=flux,
            error=error,
            disperser=disperser,
            name=name or str(unit),
        )

    def test_cross_unit_identical_results(self):
        """Same physical spectrum in AA and um should give identical model output."""
        spec_aa = self._make_spectrum(u.AA, name='spec_aa')
        spec_um = self._make_spectrum(u.um, name='spec_um')

        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)

        # Build model with AA spectrum.
        spectra_aa = Spectra([spec_aa], redshift=0.0)
        _, args_aa = _prepare_and_build(lc, None, spectra_aa)

        # Build model with um spectrum (need fresh config for fresh tokens).
        lc2 = line.LineConfiguration()
        lc2.add_line('Ha', 6563.0 * u.AA)
        spectra_um = Spectra([spec_um], redshift=0.0)
        _, args_um = _prepare_and_build(lc2, None, spectra_um)

        # Wavelengths in canonical unit should be identical.
        # AA model uses AA as canonical; um model uses um as canonical.
        # The line wavelength should be converted correctly.
        assert args_aa.matrices.wavelengths[0] == pytest.approx(6563.0, rel=1e-6)
        assert args_um.matrices.wavelengths[0] == pytest.approx(0.6563, rel=1e-6)

    def test_normalization_brings_flux_to_o1(self):
        """Spectrum with flux ~1.5 (in 1e-17 unit) should have norm_factor ~1.5."""
        wl = np.linspace(6500, 6600, 50) * u.AA
        disperser = SimpleDisperser(
            wavelength=wl.value, unit=u.AA, R=3000.0, name='faint'
        )
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        flux = np.full(50, 1.5) * flux_unit
        error = np.full(50, 0.3) * flux_unit

        spectrum = GenericSpectrum(
            low=low,
            high=high,
            flux=flux,
            error=error,
            disperser=disperser,
            name='faint',
        )

        spectra = Spectra([spectrum], redshift=0.0)
        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)
        _, args = _prepare_and_build(lc, None, spectra)

        # norm_factor should be close to the median flux value (bare float in flux_unit).
        assert args.norm_factors[0] == pytest.approx(1.5, rel=0.1)
        # After normalization, flux/norm should be ~O(1).
        assert jnp.max(jnp.abs(spectrum.flux / args.norm_factors[0])) == pytest.approx(
            1.0, rel=0.1
        )

    def test_mixed_unit_spectra_builds(self):
        """Model with spectra in different units should build and execute."""
        spec_aa = self._make_spectrum(u.AA, name='spec_aa')
        spec_um = self._make_spectrum(u.um, name='spec_um')

        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)

        spectra = Spectra([spec_aa, spec_um], redshift=0.0)
        unite_model, unite_args = _prepare_and_build(lc, None, spectra)

        # Should have two spectra with different conversion factors.
        assert len(unite_args.spec_to_canonical) == 2
        assert unite_args.spec_to_canonical[0] == pytest.approx(1.0)  # AA→AA
        assert unite_args.spec_to_canonical[1] == pytest.approx(1e4, rel=1e-3)  # um→AA

        # Should be able to execute predictive.
        rng_key = random.PRNGKey(42)
        predictive = Predictive(unite_model, num_samples=2)
        samples = predictive(rng_key, unite_args)
        # Model predictions are now accessed via evaluate_model, not the trace.
        assert 'obs_spec_aa' in samples
        assert 'obs_spec_um' in samples

    def test_line_flux_scale_positive(self):
        """Per-spectrum line_flux_scales should always be positive."""
        spectrum = create_simple_spectrum()
        lc = create_minimal_config()
        spectra = Spectra([spectrum], redshift=0.0)
        _, args = _prepare_and_build(lc, None, spectra)
        assert all(s > 0 for s in args.line_flux_scales)

    def test_continuum_bounds_converted(self):
        """Continuum region bounds should be pre-converted to canonical unit."""
        from unite.continuum import ContinuumConfiguration, ContinuumRegion, Linear

        spectrum = self._make_spectrum(u.AA, name='test')
        spectra = Spectra([spectrum], redshift=0.0)

        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)

        # Continuum regions in microns.
        cont = ContinuumConfiguration(
            [ContinuumRegion(0.65 * u.um, 0.66 * u.um, Linear())]
        )

        _, args = _prepare_and_build(lc, cont, spectra)

        # Canonical unit is AA (first spectrum); bounds should be in AA.
        assert args.cont_low[0] == pytest.approx(6500.0, rel=1e-3)
        assert args.cont_high[0] == pytest.approx(6600.0, rel=1e-3)


class TestModelBuilderFit:
    """Test suite for ModelBuilder.fit() convenience method."""

    @pytest.mark.slow
    def test_fit_basic(self):
        """Test that fit() runs and returns samples."""
        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()
        spectra = Spectra([spectrum], redshift=0.0)

        spectra.prepare(line_config, None)
        spectra.compute_scales(
            spectra.prepared_line_config, spectra.prepared_cont_config
        )
        builder = model.ModelBuilder(
            spectra.prepared_line_config, spectra.prepared_cont_config, spectra
        )
        samples, _args = builder.fit(
            num_warmup=10, num_samples=15, num_chains=1, seed=42
        )

        # Check that samples dictionary has expected keys
        flux_key = next(k for k in samples if 'flux' in k and 'scale' not in k)
        z_key = next(k for k in samples if k.endswith('-z'))
        fwhm_key = next(k for k in samples if 'fwhm' in k)

        assert len(samples[flux_key]) == 15  # num_samples
        assert len(samples[z_key]) == 15
        assert len(samples[fwhm_key]) == 15

    @pytest.mark.slow
    def test_fit_with_continuum(self):
        """Test fit() with continuum configuration."""
        from unite.continuum import ContinuumConfiguration, Linear

        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()
        cont_config = ContinuumConfiguration.from_lines(
            [6563.0] * u.AA, pad=0.1, form=Linear()
        )
        spectra = Spectra([spectrum], redshift=0.0)

        spectra.prepare(line_config, cont_config)
        spectra.compute_scales(
            spectra.prepared_line_config, spectra.prepared_cont_config
        )
        builder = model.ModelBuilder(
            spectra.prepared_line_config, spectra.prepared_cont_config, spectra
        )
        samples, _args = builder.fit(num_warmup=10, num_samples=15, seed=43)

        # Check that we have samples for both line and continuum parameters
        assert 'H_alpha-6563.0-redshift' in samples or next(
            (k for k in samples if 'z' in k), None
        )
        assert next((k for k in samples if 'cont_' in k), None) is not None

    @pytest.mark.slow
    def test_fit_multiple_chains(self):
        """Test fit() with multiple chains."""
        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()
        spectra = Spectra([spectrum], redshift=0.0)

        spectra.prepare(line_config, None)
        spectra.compute_scales(
            spectra.prepared_line_config, spectra.prepared_cont_config
        )
        builder = model.ModelBuilder(
            spectra.prepared_line_config, spectra.prepared_cont_config, spectra
        )

        # Suppress numpyro's device warning when running on CPU
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore', message='.*There are not enough devices.*'
            )
            samples, _args = builder.fit(
                num_warmup=5, num_samples=10, num_chains=2, seed=44, progress_bar=False
            )

        # With 2 chains * 10 samples, should have 20 samples total per parameter
        # (shape may be (2, 10) or (20,) depending on device availability)
        flux_key = next(k for k in samples if 'flux' in k and 'scale' not in k)
        assert samples[flux_key].size == 20

    @pytest.mark.slow
    def test_fit_reproducible_seed(self):
        """Test that fit() with same seed produces same results."""
        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()
        spectra = Spectra([spectrum], redshift=0.0)

        # First fit
        spectra.prepare(line_config, None)
        spectra.compute_scales(
            spectra.prepared_line_config, spectra.prepared_cont_config
        )
        builder1 = model.ModelBuilder(
            spectra.prepared_line_config, spectra.prepared_cont_config, spectra
        )
        samples1, _args = builder1.fit(
            num_warmup=10, num_samples=20, num_chains=1, seed=999, progress_bar=False
        )

        # Second fit with same config (need fresh spectrum and line config for fresh tokens)
        spectrum2 = create_simple_spectrum()
        line_config2 = create_minimal_config()
        spectra2 = Spectra([spectrum2], redshift=0.0)
        spectra2.prepare(line_config2, None)
        spectra2.compute_scales(
            spectra2.prepared_line_config, spectra2.prepared_cont_config
        )
        builder2 = model.ModelBuilder(
            spectra2.prepared_line_config, spectra2.prepared_cont_config, spectra2
        )
        samples2, _args = builder2.fit(
            num_warmup=10, num_samples=20, num_chains=1, seed=999, progress_bar=False
        )

        # Samples should be identical (same seed, same data)
        flux_key = next(k for k in samples1 if 'flux' in k and 'scale' not in k)
        flux_key2 = next(k for k in samples2 if 'flux' in k and 'scale' not in k)
        assert jnp.allclose(samples1[flux_key], samples2[flux_key2], rtol=1e-6)


class TestFluxUnitValidation:
    """Tests for Spectrum flux/error Quantity validation."""

    def test_valid_flux_unit_accepted(self):
        """f_lambda Quantity should be accepted."""
        wl = np.linspace(6500, 6600, 10) * u.AA
        disperser = SimpleDisperser(wavelength=wl.value, unit=u.AA, R=3000.0)
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.erg / u.s / u.cm**2 / u.AA
        flux = np.ones(10) * flux_unit
        error = np.ones(10) * flux_unit

        spectrum = GenericSpectrum(
            low=low, high=high, flux=flux, error=error, disperser=disperser
        )
        assert spectrum.flux_unit == flux_unit

    def test_invalid_flux_unit_raises(self):
        """Non-f_lambda unit (e.g. Jy) should raise."""
        wl = np.linspace(6500, 6600, 10) * u.AA
        disperser = SimpleDisperser(wavelength=wl.value, unit=u.AA, R=3000.0)
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)

        with pytest.raises(ValueError, match='spectral flux density'):
            GenericSpectrum(
                low=low,
                high=high,
                flux=np.ones(10) * u.Jy,
                error=np.ones(10) * u.Jy,
                disperser=disperser,
            )

    def test_bare_array_raises(self):
        """Bare arrays (not Quantity) should raise TypeError."""
        wl = np.linspace(6500, 6600, 10) * u.AA
        disperser = SimpleDisperser(wavelength=wl.value, unit=u.AA, R=3000.0)
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)

        with pytest.raises(TypeError, match='flux must be an astropy Quantity'):
            GenericSpectrum(
                low=low,
                high=high,
                flux=np.ones(10),
                error=np.ones(10),
                disperser=disperser,
            )
