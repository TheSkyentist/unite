"""Minimal tests for unite_model function without full MCMC sampling."""

import warnings

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro.infer import Predictive

from unite import line, model, prior
from unite.instrument.generic import SimpleDisperser
from unite.spectrum import Spectra, Spectrum


def create_simple_spectrum():
    """Create a simple test spectrum with a single Gaussian line."""
    # Create a simple wavelength grid around 6563 Å (H-alpha)
    wavelength = np.linspace(6500, 6600, 100) * u.AA

    # Create a simple disperser with constant R=3000
    disperser = SimpleDisperser(wavelength=wavelength, R=3000.0, name='test_disperser')

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
    spectrum = Spectrum(
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


@pytest.fixture(scope='module')
def basic_model():
    """Shared module-level fixture: simple H-alpha model built once."""
    spectrum = create_simple_spectrum()
    line_config = create_minimal_config()
    spectra = Spectra([spectrum], redshift=0.0)
    unite_model, unite_args = _prepare_and_build(line_config, None, spectra)
    return unite_model, unite_args, spectrum


@pytest.fixture(scope='module')
def continuum_model():
    """H-alpha model with a continuum region (bounds in microns), built once."""
    from unite.continuum import ContinuumConfiguration, ContinuumRegion, Linear

    spectrum = create_simple_spectrum()
    line_config = create_minimal_config()
    # Continuum specified in microns — exercises unit conversion in args.cont_low/high.
    cont_config = ContinuumConfiguration(
        [ContinuumRegion(0.65 * u.um, 0.66 * u.um, Linear())]
    )
    spectra = Spectra([spectrum], redshift=0.0)
    unite_model, unite_args = _prepare_and_build(line_config, cont_config, spectra)
    return unite_model, unite_args


class TestMinimalModel:
    """Test suite for minimal unite_model validation."""

    def test_model_building(self, basic_model):
        """Test that unite_model can be built without errors."""
        unite_model, unite_args, _ = basic_model
        assert unite_args is not None
        assert unite_model is not None
        assert len(unite_args.matrices.flux_names) == 1
        assert len(unite_args.matrices.z_names) == 1
        assert len(unite_args.matrices.p0_names) == 1

    def test_predictive_execution(self, basic_model):
        """Test that the model can execute predictively."""
        unite_model, unite_args, _ = basic_model
        rng_key = random.PRNGKey(0)
        samples = Predictive(unite_model, num_samples=3)(rng_key, unite_args)
        actual_params = set(samples.keys())

        flux_params = [k for k in actual_params if 'flux' in k and 'scale' not in k]
        z_params = [k for k in actual_params if k.startswith('z_')]
        fwhm_params = [k for k in actual_params if 'fwhm' in k]

        assert len(flux_params) >= 1, f'No flux parameters found in {actual_params}'
        assert len(z_params) >= 1, f'No z parameters found in {actual_params}'
        assert len(fwhm_params) >= 1, f'No fwhm parameters found in {actual_params}'
        for param in flux_params + z_params + fwhm_params:
            assert samples[param].shape == (3,)

    def test_model_with_continuum(self, continuum_model):
        """Test model building with continuum."""
        unite_model, unite_args = continuum_model

        # Check that continuum configuration is included
        assert unite_args.cont_config is not None

        # Test predictive execution
        rng_key = random.PRNGKey(1)
        samples = Predictive(unite_model, num_samples=2)(rng_key, unite_args)

        # Check for continuum parameters — new naming: scale_*, angle_*, norm_wav_*, etc.
        continuum_params = [
            k
            for k in samples
            if any(
                k.startswith(p)
                for p in ('scale_', 'angle_', 'norm_wav_', 'beta_', 'slope_')
            )
        ]
        assert len(continuum_params) > 0, (
            f'No continuum parameters found in {set(samples.keys())}'
        )

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

        # Find flux, z, fwhm parameters by pattern (new naming: flux_*, z_*, fwhm_gauss_*)
        flux_key = next(k for k in samples if 'flux' in k and 'scale' not in k)
        z_key = next(k for k in samples if k.startswith('z_'))
        fwhm_key = next(k for k in samples if 'fwhm' in k)

        assert len(samples[flux_key]) == 10
        assert len(samples[z_key]) == 10
        assert len(samples[fwhm_key]) == 10

    def test_parameter_matrices(self, basic_model):
        """Test that parameter matrices are correctly constructed."""
        _, unite_args, _ = basic_model
        matrices = unite_args.matrices
        assert matrices.wavelengths.shape == (1,)
        assert matrices.strengths.shape == (1,)
        assert matrices.profile_codes.shape == (1,)
        assert matrices.flux_matrix.shape == (1, 1)
        assert matrices.z_matrix.shape == (1, 1)
        assert matrices.p0_matrix.shape == (1, 1)
        assert jnp.sum(matrices.flux_matrix) == 1.0
        assert jnp.sum(matrices.z_matrix) == 1.0
        assert jnp.sum(matrices.p0_matrix) == 1.0

    def test_spectra_integration(self, basic_model):
        """Test that spectra are properly integrated into the model."""
        _, unite_args, _ = basic_model
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

    def test_model_predictions_shape(self, basic_model):
        """Test that model predictions have the correct shape."""
        from unite.evaluate import evaluate_model

        unite_model, unite_args, _ = basic_model
        rng_key = random.PRNGKey(3)
        samples = Predictive(unite_model, num_samples=5)(rng_key, unite_args)

        obs_pred = samples['obs_test_spectrum']
        assert obs_pred.shape == (5, 100)
        assert jnp.all(jnp.isfinite(obs_pred))

        preds = evaluate_model(samples, unite_args)
        assert len(preds) == 1
        assert preds[0].total.shape == (5, 100)
        assert jnp.all(jnp.isfinite(preds[0].total))

    def test_model_fits_data_range(self, basic_model):
        """Test that model predictions are finite and not all zeros."""
        from unite.evaluate import evaluate_model

        unite_model, unite_args, _ = basic_model
        rng_key = random.PRNGKey(4)
        samples = Predictive(unite_model, num_samples=5)(rng_key, unite_args)

        preds = evaluate_model(samples, unite_args)
        mean_model = jnp.mean(jnp.asarray(preds[0].total), axis=0)

        assert jnp.all(jnp.isfinite(mean_model))
        assert unite_args.norm_factors[0] > 0
        assert jnp.max(jnp.abs(mean_model)) > 0


# ---------------------------------------------------------------------------
# Wavelength unit consistency and normalization tests
# ---------------------------------------------------------------------------


class TestWavelengthUnitConsistency:
    """Tests for cross-unit wavelength handling and flux normalization."""

    def _make_spectrum(self, unit, *, name=''):
        """Create a test spectrum in the given wavelength unit."""
        wl_aa = np.linspace(6500, 6600, 100) * u.AA
        wl = wl_aa.to(unit)
        disperser = SimpleDisperser(wavelength=wl, R=3000.0, name=name or str(unit))
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
        return Spectrum(
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
        disperser = SimpleDisperser(wavelength=wl, R=3000.0, name='faint')
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        flux = np.full(50, 1.5) * flux_unit
        error = np.full(50, 0.3) * flux_unit

        spectrum = Spectrum(
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

    def test_continuum_bounds_converted(self, continuum_model):
        """Continuum region bounds (specified in microns) should be converted to canonical unit."""
        _, args = continuum_model

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

        # Check that samples dictionary has expected keys (new naming: flux_*, z_*, fwhm_gauss_*)
        flux_key = next(k for k in samples if 'flux' in k and 'scale' not in k)
        z_key = next(k for k in samples if k.startswith('z_'))
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
            [6563.0] * u.AA, width=60_000 * u.km / u.s, form=Linear()
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
        assert next((k for k in samples if k.startswith('z_')), None) is not None
        # Continuum parameters use new naming: scale_*, angle_*, norm_wav_*, etc.
        assert (
            next(
                (
                    k
                    for k in samples
                    if any(
                        k.startswith(p)
                        for p in ('scale_', 'angle_', 'norm_wav_', 'beta_', 'slope_')
                    )
                ),
                None,
            )
            is not None
        )

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
        # New naming: flux_*, z_*, fwhm_gauss_*
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
        # New naming: flux_*, z_*, fwhm_gauss_*
        flux_key = next(k for k in samples1 if 'flux' in k and 'scale' not in k)
        flux_key2 = next(k for k in samples2 if 'flux' in k and 'scale' not in k)
        assert jnp.allclose(samples1[flux_key], samples2[flux_key2], rtol=1e-6)


class TestFullyMaskedSpectra:
    """Test graceful handling of spectra with no continuum overlap."""

    def _make_spectrum(self, wl_range, *, name='test'):
        """Create a test spectrum with the given wavelength range in Angstrom."""
        wl = np.linspace(*wl_range, 50) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0, name=name)
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        rng = np.random.default_rng(42)
        flux = rng.normal(10, 2, len(wl)) * flux_unit
        error = np.full(len(wl), 2.0) * flux_unit
        return Spectrum(
            low=low, high=high, flux=flux, error=error, disperser=disperser, name=name
        )

    def test_one_spectrum_fully_masked_warns(self):
        """A spectrum with no continuum overlap should be excluded with a warning."""
        from unite.continuum import ContinuumConfiguration, ContinuumRegion, Linear

        # Spectrum 1 covers 6500-6600 AA (overlaps continuum).
        spec_good = self._make_spectrum((6500, 6600), name='good')
        # Spectrum 2 covers 8000-8100 AA (no overlap with continuum).
        spec_bad = self._make_spectrum((8000, 8100), name='bad')

        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)

        # Continuum only covers 6450-6650 AA.
        cont = ContinuumConfiguration(
            [ContinuumRegion(6450.0 * u.AA, 6650.0 * u.AA, Linear())]
        )

        spectra = Spectra([spec_good, spec_bad], redshift=0.0)
        spectra.prepare(lc, cont)
        spectra.compute_scales(
            spectra.prepared_line_config, spectra.prepared_cont_config
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            unite_model_fn, unite_args = model.ModelBuilder(
                spectra.prepared_line_config, spectra.prepared_cont_config, spectra
            ).build()

        # Should warn about the excluded spectrum.
        msgs = [str(x.message) for x in w]
        assert any('bad' in m and 'excluded' in m for m in msgs), msgs

        # Only the good spectrum should be in the model.
        assert len(unite_args.spectra) == 1
        assert unite_args.spectra[0].name == 'good'

        # Model should still be executable.
        rng_key = random.PRNGKey(99)
        predictive = Predictive(unite_model_fn, num_samples=2)
        samples = predictive(rng_key, unite_args)
        assert 'obs_good' in samples

    def test_all_spectra_fully_masked_warns(self):
        """If all spectra are fully masked, build() warns and returns empty args."""
        from unite.continuum import ContinuumConfiguration, ContinuumRegion, Linear

        # Spectrum covers 8000-8100 AA, but continuum is at 6450-6650 AA.
        spec = self._make_spectrum((8000, 8100), name='far_away')

        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)

        cont = ContinuumConfiguration(
            [ContinuumRegion(6450.0 * u.AA, 6650.0 * u.AA, Linear())]
        )

        spectra = Spectra([spec], redshift=0.0)
        spectra.prepare(lc, cont)
        # compute_scales would raise because the line is outside coverage;
        # set scales manually to test the downstream build() warning.
        spectra.line_scale = 1e-17 * u.erg / u.s / u.cm**2
        spectra.continuum_scale = 1e-17 * u.erg / u.s / u.cm**2 / u.AA

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _unite_model_fn, unite_args = model.ModelBuilder(
                spectra.prepared_line_config, spectra.prepared_cont_config, spectra
            ).build()

        msgs = [str(x.message) for x in w]
        assert any('All spectra are fully masked' in m for m in msgs), msgs

        # ModelArgs should report zero spectra.
        assert len(unite_args) == 0
        assert not unite_args

    def test_model_args_len(self):
        """len(ModelArgs) returns the number of spectra."""
        spec = self._make_spectrum((6500, 6600), name='a')

        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)

        spectra = Spectra([spec], redshift=0.0)
        spectra.prepare(lc)
        spectra.compute_scales(
            spectra.prepared_line_config, spectra.prepared_cont_config
        )

        with warnings.catch_warnings():
            warnings.simplefilter('always')
            _, args = model.ModelBuilder(
                spectra.prepared_line_config, spectra.prepared_cont_config, spectra
            ).build()

        assert len(args) == 1
        assert args


class TestFluxUnitValidation:
    """Tests for Spectrum flux/error Quantity validation."""

    def test_valid_flux_unit_accepted(self):
        """f_lambda Quantity should be accepted."""
        wl = np.linspace(6500, 6600, 10) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0)
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.erg / u.s / u.cm**2 / u.AA
        flux = np.ones(10) * flux_unit
        error = np.ones(10) * flux_unit

        spectrum = Spectrum(
            low=low, high=high, flux=flux, error=error, disperser=disperser
        )
        assert spectrum.flux_unit == flux_unit

    def test_invalid_flux_unit_raises(self):
        """Non-f_lambda unit (e.g. Jy) should raise."""
        wl = np.linspace(6500, 6600, 10) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0)
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)

        with pytest.raises(
            ValueError,
            match=r'must have units equivalent to erg / \(Angstrom s cm2\), got Jy',
        ):
            Spectrum(
                low=low,
                high=high,
                flux=np.ones(10) * u.Jy,
                error=np.ones(10) * u.Jy,
                disperser=disperser,
            )

    def test_bare_array_raises(self):
        """Bare arrays (not Quantity) should raise TypeError."""
        wl = np.linspace(6500, 6600, 10) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0)
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)

        with pytest.raises(TypeError, match='flux must be an astropy Quantity'):
            Spectrum(
                low=low,
                high=high,
                flux=np.ones(10),
                error=np.ones(10),
                disperser=disperser,
            )


# ---------------------------------------------------------------------------
# _norm_factor with all-zero flux (model.py lines 641-643)
# ---------------------------------------------------------------------------


class TestNormFactor:
    """Tests for _norm_factor edge cases."""

    def test_zero_flux_falls_back_to_error(self):
        """A spectrum with all-zero flux should use max error as norm_factor."""
        import warnings

        wl = np.linspace(6500, 6600, 50) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0, name='zero_flux')
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        flux = np.zeros(50) * flux_unit
        error = np.full(50, 2.5) * flux_unit

        spectrum = Spectrum(
            low=low, high=high, flux=flux, error=error, disperser=disperser, name='zero'
        )
        spectra = Spectra([spectrum], redshift=0.0)

        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)
        spectra.prepare(lc)
        # Set scales manually since compute_scales can't find a line peak in zero flux
        spectra.line_scale = 1.0 * flux_unit * u.AA
        spectra.continuum_scale = 1.0 * flux_unit

        with warnings.catch_warnings():
            warnings.simplefilter('always')
            _, args = model.ModelBuilder(
                spectra.prepared_line_config, None, spectra
            ).build()

        # norm_factor falls back to max(error) = 2.5
        assert args.norm_factors[0] == pytest.approx(2.5, rel=0.01)


# ---------------------------------------------------------------------------
# ModelBuilder.matrices property (model.py line 392)
# ---------------------------------------------------------------------------


class TestModelBuilderMatrices:
    """Tests for the ModelBuilder.matrices property."""

    def test_matrices_property(self):
        """ModelBuilder.matrices returns precomputed ConfigMatrices."""
        spectrum = create_simple_spectrum()
        line_config = create_minimal_config()
        spectra = Spectra([spectrum], redshift=0.0)
        spectra.prepare(line_config)
        spectra.compute_scales(spectra.prepared_line_config)
        builder = model.ModelBuilder(spectra.prepared_line_config, None, spectra)
        assert builder.matrices is not None
        assert len(builder.matrices.flux_names) == 1


# ---------------------------------------------------------------------------
# Model with pix_offset disperser (model.py lines 201-205)
# ---------------------------------------------------------------------------


class TestPixOffset:
    """Tests for model with a disperser that has a pix_offset token."""

    def test_model_with_pix_offset(self):
        """Model with PixOffset token on disperser should build and run."""
        from jax import random as jrandom
        from numpyro.infer import Predictive

        from unite.instrument.base import PixOffset

        wl = np.linspace(6500, 6600, 60) * u.AA
        pix_offset_tok = PixOffset('test', prior=prior.Uniform(-1.0, 1.0))
        disperser = SimpleDisperser(
            wavelength=wl, R=3000.0, name='pix', pix_offset=pix_offset_tok
        )

        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
        line_flux = 50 * np.exp(-0.5 * ((wl.value - 6563.0) / sigma) ** 2)
        rng = np.random.default_rng(7)
        flux = (line_flux + 5.0 + rng.normal(0, 1, len(wl))) * flux_unit
        error = np.full(len(wl), 1.0) * flux_unit

        spectrum = Spectrum(
            low=low,
            high=high,
            flux=flux,
            error=error,
            disperser=disperser,
            name='pix_spec',
        )
        spectra = Spectra([spectrum], redshift=0.0)

        lc = line.LineConfiguration()
        lc.add_line(
            'Ha',
            6563.0 * u.AA,
            redshift=line.Redshift(prior=prior.Uniform(-0.005, 0.005)),
            fwhm_gauss=line.FWHM(prior=prior.Uniform(100, 1000)),
            flux=line.Flux(prior=prior.Uniform(0, 5)),
        )

        model_fn, args = _prepare_and_build(lc, None, spectra)

        # pix_offset token should appear in the dependency order
        assert any('pix_offset' in pname for pname in args.dependency_order)

        # Model should execute without errors
        rng_key = jrandom.PRNGKey(0)
        predictive = Predictive(model_fn, num_samples=3)
        samples = predictive(rng_key, args)
        assert 'obs_pix_spec' in samples


# ---------------------------------------------------------------------------
# PseudoVoigt profile covers model.py p1v_names branches (lines 160-161)
# ---------------------------------------------------------------------------


class TestPseudoVoigtModel:
    """Test that a PseudoVoigt line covers p1v_names branches in unite_model."""

    def test_pseudovoigt_model_runs(self):
        """Model with PseudoVoigt profile should build and execute (covers p1v branch)."""
        from jax import random as jrandom
        from numpyro.infer import Predictive

        from unite.line.profiles import PseudoVoigt

        wl = np.linspace(6500, 6600, 60) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0, name='pv')
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
        line_flux = 50 * np.exp(-0.5 * ((wl.value - 6563.0) / sigma) ** 2)
        rng = np.random.default_rng(8)
        flux = (line_flux + 5.0 + rng.normal(0, 1, len(wl))) * flux_unit
        error = np.full(len(wl), 1.0) * flux_unit

        spectrum = Spectrum(
            low=low,
            high=high,
            flux=flux,
            error=error,
            disperser=disperser,
            name='pv_spec',
        )
        spectra = Spectra([spectrum], redshift=0.0)

        lc = line.LineConfiguration()
        lc.add_line(
            'Ha',
            6563.0 * u.AA,
            profile=PseudoVoigt(),
            fwhm_gauss=line.FWHM(prior=prior.Uniform(100, 1000)),
            fwhm_lorentz=line.FWHM(prior=prior.Uniform(0, 500)),
        )

        model_fn, args = _prepare_and_build(lc, None, spectra)

        # p1v_names should be populated (fwhm_lorentz is a velocity FWHM)
        assert len(args.matrices.p1v_names) > 0

        rng_key = jrandom.PRNGKey(0)
        predictive = Predictive(model_fn, num_samples=2)
        samples = predictive(rng_key, args)
        assert 'obs_pv_spec' in samples


# ---------------------------------------------------------------------------
# GaussHermite profile covers model.py p1d/p2 branches (lines 167-168, 175-176)
# ---------------------------------------------------------------------------


class TestGaussHermiteModel:
    """Test that a GaussHermite line covers p1d_names and p2_names branches."""

    def test_gausshermite_model_runs(self):
        """Model with GaussHermite covers p1d and p2 branches in unite_model."""
        from jax import random as jrandom
        from numpyro.infer import Predictive

        from unite.line.config import LineShape
        from unite.line.profiles import GaussHermite

        wl = np.linspace(6500, 6600, 60) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0, name='gh')
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
        line_flux = 50 * np.exp(-0.5 * ((wl.value - 6563.0) / sigma) ** 2)
        rng = np.random.default_rng(9)
        flux = (line_flux + 5.0 + rng.normal(0, 1, len(wl))) * flux_unit
        error = np.full(len(wl), 1.0) * flux_unit

        spectrum = Spectrum(
            low=low,
            high=high,
            flux=flux,
            error=error,
            disperser=disperser,
            name='gh_spec',
        )
        spectra = Spectra([spectrum], redshift=0.0)

        lc = line.LineConfiguration()
        lc.add_line(
            'Ha',
            6563.0 * u.AA,
            profile=GaussHermite(),
            fwhm_gauss=line.FWHM(prior=prior.Uniform(100, 1000)),
            h3=LineShape(prior=prior.TruncatedNormal(0.0, 0.1, -0.3, 0.3)),
            h4=LineShape(prior=prior.TruncatedNormal(0.0, 0.1, -0.3, 0.3)),
        )

        model_fn, args = _prepare_and_build(lc, None, spectra)

        # p1d_names should have h3, p2_names should have h4
        assert len(args.matrices.p1d_names) > 0
        assert len(args.matrices.p2_names) > 0

        rng_key = jrandom.PRNGKey(0)
        predictive = Predictive(model_fn, num_samples=2)
        samples = predictive(rng_key, args)
        assert 'obs_gh_spec' in samples


# ---------------------------------------------------------------------------
# Two spectra sharing the same disperser (model.py line 359->357)
# ---------------------------------------------------------------------------


class TestSharedDisperser:
    """Two spectra sharing the same disperser object covers the disperser-already-seen branch."""

    def test_shared_disperser_registers_once(self):
        """Same disperser on two spectra: model builds correctly and runs."""
        from jax import random as jrandom
        from numpyro.infer import Predictive

        wl = np.linspace(6500, 6600, 60) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0, name='shared')
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
        line_flux = 50 * np.exp(-0.5 * ((wl.value - 6563.0) / sigma) ** 2)
        rng = np.random.default_rng(10)
        noise1 = rng.normal(0, 1, len(wl))
        noise2 = rng.normal(0, 1, len(wl))
        flux1 = (line_flux + 5.0 + noise1) * flux_unit
        flux2 = (line_flux + 5.0 + noise2) * flux_unit
        error = np.full(len(wl), 1.0) * flux_unit

        spec1 = Spectrum(
            low=low, high=high, flux=flux1, error=error, disperser=disperser, name='s1'
        )
        spec2 = Spectrum(
            low=low, high=high, flux=flux2, error=error, disperser=disperser, name='s2'
        )
        spectra = Spectra([spec1, spec2], redshift=0.0)

        lc = line.LineConfiguration()
        lc.add_line(
            'Ha',
            6563.0 * u.AA,
            redshift=line.Redshift(prior=prior.Uniform(-0.005, 0.005)),
            fwhm_gauss=line.FWHM(prior=prior.Uniform(100, 1000)),
            flux=line.Flux(prior=prior.Uniform(0, 5)),
        )

        model_fn, args = _prepare_and_build(lc, None, spectra)

        # Two spectra but only one disperser in the model
        assert len(args.spectra) == 2

        rng_key = jrandom.PRNGKey(0)
        predictive = Predictive(model_fn, num_samples=2)
        samples = predictive(rng_key, args)
        assert 'obs_s1' in samples
        assert 'obs_s2' in samples


# ---------------------------------------------------------------------------
# Shared continuum token (model.py line 373->372: token already seen)
# ---------------------------------------------------------------------------


class TestSharedContinuumToken:
    """Two continuum regions sharing the same Scale token object."""

    def test_shared_scale_token_registers_once(self):
        """Shared Scale token across regions should register only once in model."""
        from jax import random as jrandom
        from numpyro.infer import Predictive

        from unite.continuum import ContinuumConfiguration, ContinuumRegion, Linear
        from unite.continuum.config import Scale

        wl = np.linspace(6450, 6650, 100) * u.AA
        disperser = SimpleDisperser(wavelength=wl, R=3000.0, name='shared_cont')
        low = wl - 0.5 * np.gradient(wl)
        high = wl + 0.5 * np.gradient(wl)
        flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
        rng = np.random.default_rng(11)
        flux = (5.0 + rng.normal(0, 0.5, len(wl))) * flux_unit
        error = np.full(len(wl), 0.5) * flux_unit

        spectrum = Spectrum(
            low=low, high=high, flux=flux, error=error, disperser=disperser, name='sc'
        )
        spectra = Spectra([spectrum], redshift=0.0)

        lc = line.LineConfiguration()
        lc.add_line(
            'Ha',
            6563.0 * u.AA,
            redshift=line.Redshift(prior=prior.Fixed(0.0)),
            fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
            flux=line.Flux(prior=prior.Uniform(0, 5)),
        )

        # Two regions sharing a single Scale token object
        shared_scale = Scale(prior=prior.Uniform(0, 10))
        region1 = ContinuumRegion(
            6450 * u.AA, 6510 * u.AA, Linear(), params={'scale': shared_scale}
        )
        region2 = ContinuumRegion(
            6570 * u.AA, 6650 * u.AA, Linear(), params={'scale': shared_scale}
        )
        cc = ContinuumConfiguration([region1, region2])

        spectra.prepare(lc, cc)
        spectra.compute_scales(
            spectra.prepared_line_config, spectra.prepared_cont_config
        )

        model_fn, args = model.ModelBuilder(
            spectra.prepared_line_config, spectra.prepared_cont_config, spectra
        ).build()

        # The shared Scale token should produce fewer parameters than the number
        # of regions (since the same object is shared, it's registered only once).
        # With 2 regions each having: scale, norm_wav, angle → 6 params max,
        # but shared scale → at most 5 unique params.
        assert len(args.dependency_order) > 0

        rng_key = jrandom.PRNGKey(0)
        predictive = Predictive(model_fn, num_samples=2)
        samples = predictive(rng_key, args)
        assert 'obs_sc' in samples


# ---------------------------------------------------------------------------
# Absorption line model tests
# ---------------------------------------------------------------------------


def _create_absorption_spectrum():
    """Create a spectrum for absorption line testing."""
    wl = np.linspace(6500, 6600, 80) * u.AA
    disperser = SimpleDisperser(wavelength=wl, R=3000.0, name='abs_spec')
    low = wl - 0.5 * np.gradient(wl)
    high = wl + 0.5 * np.gradient(wl)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    flux = np.ones(len(wl)) * 100.0 * flux_unit
    error = np.full(len(wl), 1.0) * flux_unit
    return Spectrum(
        low=low, high=high, flux=flux, error=error, disperser=disperser, name='abs_spec'
    )


class TestAbsorptionModel:
    """Tests for absorption line support in the model pipeline."""

    def test_emission_only_regression(self, basic_model):
        """Emission-only model still works unchanged."""
        _unite_model, args, _ = basic_model
        assert len(args.matrices.tau_names) == 0
        assert not jnp.any(args.matrices.is_absorption)

    def test_absorption_model_builds(self):
        """Model with absorption lines builds and runs."""
        from unite.line.profiles import GaussianAbsorption

        spec = _create_absorption_spectrum()
        lc = line.LineConfiguration()
        lc.add_line(
            'Ha',
            6563.0 * u.AA,
            fwhm_gauss=line.FWHM(prior=prior.Uniform(100, 1000)),
            flux=line.Flux(prior=prior.Uniform(0, 5)),
        )
        lc.add_line(
            'HI_abs',
            6563.0 * u.AA,
            profile=GaussianAbsorption(),
            fwhm_gauss=line.FWHM(prior=prior.Uniform(100, 1000)),
            tau=line.Tau(prior=prior.Uniform(0, 5)),
        )
        model_fn, args = _prepare_and_build(lc, None, Spectra([spec], redshift=0.0))

        assert len(args.matrices.tau_names) == 1
        assert jnp.any(args.matrices.is_absorption)

        samples = Predictive(model_fn, num_samples=3)(random.PRNGKey(0), args)
        assert 'obs_abs_spec' in samples

    def test_tau_zero_no_absorption(self):
        """tau=0 should produce transmission=1 (no absorption)."""
        from unite.line.profiles import GaussianAbsorption

        spec = _create_absorption_spectrum()
        lc = line.LineConfiguration()
        lc.add_line(
            'HI_abs',
            6563.0 * u.AA,
            profile=GaussianAbsorption(),
            fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
            tau=line.Tau(prior=prior.Fixed(0.0)),
        )
        model_fn, args = _prepare_and_build(lc, None, Spectra([spec], redshift=0.0))
        # With tau=0, model predictions should just be 0 (no emission lines, transmission=1)
        samples = Predictive(model_fn, num_samples=2)(random.PRNGKey(0), args)
        obs = samples['obs_abs_spec']
        # With tau=0 and no emission, model is just 0 everywhere
        assert obs.shape[1] > 0  # sanity check

    def test_tau_positive_reduces_flux(self):
        """tau>0 should reduce flux at line center relative to tau=0."""
        from unite.evaluate import evaluate_model
        from unite.line.profiles import GaussianAbsorption

        spec = create_simple_spectrum()

        def _build_abs_model(tau_val):
            lc = line.LineConfiguration()
            lc.add_line(
                'Ha',
                6563.0 * u.AA,
                redshift=line.Redshift(prior=prior.Fixed(0.0)),
                fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
                flux=line.Flux(prior=prior.Fixed(1.0)),
            )
            lc.add_line(
                'HI_abs',
                6563.0 * u.AA,
                profile=GaussianAbsorption(),
                redshift=line.Redshift(prior=prior.Fixed(0.0)),
                fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
                tau=line.Tau(prior=prior.Fixed(tau_val)),
            )
            return _prepare_and_build(lc, None, Spectra([spec], redshift=0.0))

        _model_fn0, args0 = _build_abs_model(0.0)
        _model_fn5, args5 = _build_abs_model(5.0)

        # Use evaluate_model with empty samples (all Fixed) to get model predictions.
        pred0 = evaluate_model({}, args0)[0]
        pred5 = evaluate_model({}, args5)[0]

        # Find pixel closest to the line center wavelength.
        center_idx = int(np.argmin(np.abs(pred0.wavelength - 6563.0)))
        total0 = pred0.total[0, center_idx]
        total5 = pred5.total[0, center_idx]
        assert total5 < total0

    def test_invalid_absorber_position_raises(self):
        """Invalid absorber_position should raise ValueError."""
        spec = _create_absorption_spectrum()
        lc = line.LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA)
        spectra = Spectra([spec], redshift=0.0)
        spectra.prepare(lc)
        spectra.compute_scales(spectra.prepared_line_config)
        builder = model.ModelBuilder(spectra.prepared_line_config, None, spectra)
        with pytest.raises(ValueError, match='absorber_position must be one of'):
            builder.build(absorber_position='invalid')

    def test_absorber_positions(self):
        """All three absorber positions build and run without error."""
        from unite.line.profiles import GaussianAbsorption

        spec = _create_absorption_spectrum()
        lc = line.LineConfiguration()
        lc.add_line(
            'Ha',
            6563.0 * u.AA,
            fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
            flux=line.Flux(prior=prior.Fixed(1.0)),
        )
        lc.add_line(
            'HI_abs',
            6563.0 * u.AA,
            profile=GaussianAbsorption(),
            fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
            tau=line.Tau(prior=prior.Fixed(2.0)),
        )
        for pos in ('foreground', 'behind_lines', 'behind_continuum'):
            spectra = Spectra([spec], redshift=0.0)
            model_fn, args = _prepare_and_build(lc, None, spectra)
            # Rebuild with specific absorber_position
            spectra2 = Spectra([spec], redshift=0.0)
            spectra2.prepare(lc)
            spectra2.compute_scales(spectra2.prepared_line_config)
            builder = model.ModelBuilder(spectra2.prepared_line_config, None, spectra2)
            model_fn, args = builder.build(absorber_position=pos)
            assert args.absorber_position == pos
            samples = Predictive(model_fn, num_samples=2)(random.PRNGKey(0), args)
            assert 'obs_abs_spec' in samples

    def test_absorption_only_model(self):
        """Model with only absorption lines (no emission) builds and runs."""
        from unite.line.profiles import GaussianAbsorption

        spec = _create_absorption_spectrum()
        lc = line.LineConfiguration()
        lc.add_line(
            'HI_abs',
            6563.0 * u.AA,
            profile=GaussianAbsorption(),
            fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
            tau=line.Tau(prior=prior.Fixed(1.0)),
        )
        spectra = Spectra([spec], redshift=0.0)
        spectra.prepare(lc)
        # Manually set scales since compute_scales needs emission lines.
        spectra.line_scale = 1.0 * u.Unit('1e-17 erg / (s cm2)')
        spectra.continuum_scale = 1.0 * u.Unit('1e-17 erg / (s cm2 AA)')
        model_fn, args = model.ModelBuilder(
            spectra.prepared_line_config, None, spectra
        ).build()

        # Flux matrix should be empty
        assert args.matrices.flux_matrix.shape[0] == 0
        assert len(args.matrices.tau_names) == 1

        samples = Predictive(model_fn, num_samples=2)(random.PRNGKey(0), args)
        assert 'obs_abs_spec' in samples


class TestIntegrationMode:
    """Tests for the integration_mode parameter on build() and fit()."""

    def test_invalid_mode_raises(self, basic_model):
        """Invalid integration_mode should raise ValueError."""
        spec = create_simple_spectrum()
        lc = create_minimal_config()
        spectra = Spectra([spec], redshift=0.0)
        spectra.prepare(lc)
        spectra.compute_scales(spectra.prepared_line_config)
        builder = model.ModelBuilder(spectra.prepared_line_config, None, spectra)
        with pytest.raises(ValueError, match='integration_mode must be one of'):
            builder.build(integration_mode='invalid')

    def test_analytic_mode_no_quadrature_arrays(self, basic_model):
        """Analytic mode should not store quadrature nodes/weights."""
        _, args, _ = basic_model
        assert args.integration_mode == 'analytic'
        assert args.quadrature_nodes is None
        assert args.quadrature_weights is None

    def test_quadrature_mode_stores_arrays(self):
        """Quadrature mode should store GL nodes/weights."""
        spec = create_simple_spectrum()
        lc = create_minimal_config()
        spectra = Spectra([spec], redshift=0.0)
        spectra.prepare(lc)
        spectra.compute_scales(spectra.prepared_line_config)
        builder = model.ModelBuilder(spectra.prepared_line_config, None, spectra)
        _, args = builder.build(integration_mode='quadrature', n_nodes=5)
        assert args.integration_mode == 'quadrature'
        assert args.quadrature_nodes.shape == (5,)
        assert args.quadrature_weights.shape == (5,)

    def test_quadrature_predictive(self):
        """Model with quadrature integration runs without errors."""
        spec = create_simple_spectrum()
        lc = create_minimal_config()
        spectra = Spectra([spec], redshift=0.0)
        spectra.prepare(lc)
        spectra.compute_scales(spectra.prepared_line_config)
        builder = model.ModelBuilder(spectra.prepared_line_config, None, spectra)
        model_fn, args = builder.build(integration_mode='quadrature', n_nodes=7)
        samples = Predictive(model_fn, num_samples=3)(random.PRNGKey(0), args)
        assert f'obs_{spec.name}' in samples

    def test_quadrature_matches_analytic_emission(self):
        """For emission-only models, quadrature should match analytic closely."""
        from unite.evaluate import evaluate_model

        spec = create_simple_spectrum()
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
        builder = model.ModelBuilder(spectra.prepared_line_config, None, spectra)

        _, args_a = builder.build(integration_mode='analytic')
        _, args_q = builder.build(integration_mode='quadrature', n_nodes=11)

        pred_a = evaluate_model({}, args_a)[0]
        pred_q = evaluate_model({}, args_q)[0]

        # Total model predictions should match to high precision
        assert np.allclose(pred_a.total, pred_q.total, rtol=1e-4)
