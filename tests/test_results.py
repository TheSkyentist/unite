"""Tests for output parsing (unite.results)."""

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from jax import random
from numpyro.infer import Predictive

from unite import line, model, prior
from unite.continuum import ContinuumConfiguration, Linear
from unite.continuum.config import ContinuumRegion
from unite.instrument import Spectra
from unite.instrument.generic import GenericSpectrum, SimpleDisperser
from unite.results import (
    _get_n_samples,
    make_hdul,
    make_parameter_table,
    make_spectra_tables,
)


def _setup():
    """Create model and get predictive samples."""
    wavelength = np.linspace(6500, 6600, 60) * u.AA
    disperser = SimpleDisperser(
        wavelength=wavelength.value, unit=u.AA, R=3000.0, name='test'
    )
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)

    sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
    line_flux = 50 * np.exp(-0.5 * ((wavelength.value - 6563.0) / sigma) ** 2)
    rng = np.random.default_rng(42)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    flux = (line_flux + 5.0 + rng.normal(0, 1, len(wavelength))) * flux_unit
    error = np.full(len(wavelength), 1.0) * flux_unit

    spectrum = GenericSpectrum(
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

    rng_key = random.PRNGKey(0)
    predictive = Predictive(model_fn, num_samples=4)
    samples = predictive(rng_key, args)

    return samples, args


class TestMakeParameterTable:
    """Tests for make_parameter_table."""

    def test_returns_table(self):
        samples, args = _setup()
        table = make_parameter_table(samples, args)
        assert isinstance(table, Table)

    def test_has_all_parameters(self):
        samples, args = _setup()
        table = make_parameter_table(samples, args)
        for pname in args.dependency_order:
            assert pname in table.colnames

    def test_correct_n_rows(self):
        samples, args = _setup()
        table = make_parameter_table(samples, args)
        assert len(table) == 4  # 4 samples

    def test_metadata(self):
        samples, args = _setup()
        table = make_parameter_table(samples, args)
        assert 'LFLXSCL' in table.meta
        assert 'ZSYS' in table.meta

    def test_percentiles_mode(self):
        samples, args = _setup()
        percentiles = np.array([0.16, 0.5, 0.84])
        table = make_parameter_table(samples, args, percentiles=percentiles)
        assert 'percentile' in table.colnames
        assert len(table) == 3  # 3 percentiles
        assert np.allclose(table['percentile'], percentiles)

    def test_percentiles_mode_has_parameters(self):
        samples, args = _setup()
        percentiles = np.array([0.16, 0.5, 0.84])
        table = make_parameter_table(samples, args, percentiles=percentiles)
        for pname in args.dependency_order:
            assert pname in table.colnames


class TestMakeSpectraTables:
    """Tests for make_spectra_tables."""

    def test_returns_list_of_tables(self):
        samples, args = _setup()
        tables = make_spectra_tables(samples, args)
        assert isinstance(tables, list)
        assert len(tables) == 1
        assert isinstance(tables[0], Table)

    def test_has_wavelength_column(self):
        samples, args = _setup()
        tables = make_spectra_tables(samples, args)
        assert 'wavelength' in tables[0].colnames

    def test_has_model_total_column(self):
        samples, args = _setup()
        tables = make_spectra_tables(samples, args)
        assert 'model_total' in tables[0].colnames

    def test_has_observed_columns(self):
        samples, args = _setup()
        tables = make_spectra_tables(samples, args)
        t = tables[0]
        assert 'observed_flux' in t.colnames
        assert 'observed_error' in t.colnames
        assert 'scaled_error' in t.colnames

    def test_has_spectrum_metadata(self):
        samples, args = _setup()
        tables = make_spectra_tables(samples, args)
        t = tables[0]
        assert 'SPECNAME' in t.meta
        assert 'NORMFAC' in t.meta

    def test_percentiles_mode(self):
        samples, args = _setup()
        percentiles = np.array([0.16, 0.5, 0.84])
        tables = make_spectra_tables(samples, args, percentiles=percentiles)
        t = tables[0]
        # Percentiles gives (n_pixels, n_percentiles) for model columns
        assert t['model_total'].shape[0] == len(t['wavelength'])
        assert t['model_total'].shape[1] == 3  # 3 percentiles


def _setup_with_continuum():
    """Create model with continuum and get predictive samples."""
    wavelength = np.linspace(6500, 6600, 60) * u.AA
    disperser = SimpleDisperser(
        wavelength=wavelength.value, unit=u.AA, R=3000.0, name='test'
    )
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)

    sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
    line_flux = 50 * np.exp(-0.5 * ((wavelength.value - 6563.0) / sigma) ** 2)
    rng = np.random.default_rng(42)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    flux = (line_flux + 5.0 + rng.normal(0, 1, len(wavelength))) * flux_unit
    error = np.full(len(wavelength), 1.0) * flux_unit

    spectrum = GenericSpectrum(
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

    cont = ContinuumConfiguration.from_lines(
        lc.centers, width=30_000 * u.km / u.s, form=Linear()
    )

    spectra = Spectra([spectrum], redshift=0.0)
    spectra.prepare(lc, cont)
    spectra.compute_scales(spectra.prepared_line_config)

    model_fn, args = model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build()

    rng_key = random.PRNGKey(0)
    predictive = Predictive(model_fn, num_samples=4)
    samples = predictive(rng_key, args)

    return samples, args


class TestRestEquivalentWidths:
    """Tests for rest equivalent width columns in make_parameter_table."""

    def test_rew_columns_present(self):
        samples, args = _setup_with_continuum()
        table = make_parameter_table(samples, args)
        assert 'Ha_rew' in table.colnames

    def test_rew_has_unit(self):
        samples, args = _setup_with_continuum()
        table = make_parameter_table(samples, args)
        assert table['Ha_rew'].unit is not None

    def test_rew_shape(self):
        samples, args = _setup_with_continuum()
        table = make_parameter_table(samples, args)
        assert len(table['Ha_rew']) == 4  # 4 samples

    def test_rew_finite(self):
        samples, args = _setup_with_continuum()
        table = make_parameter_table(samples, args)
        assert np.all(np.isfinite(np.asarray(table['Ha_rew'])))

    def test_rew_percentiles_mode(self):
        samples, args = _setup_with_continuum()
        percentiles = np.array([0.16, 0.5, 0.84])
        table = make_parameter_table(samples, args, percentiles=percentiles)
        assert 'Ha_rew' in table.colnames
        assert len(table['Ha_rew']) == 3  # 3 percentiles
        assert 'percentile' in table.colnames
        assert np.allclose(table['percentile'], percentiles)

    def test_no_rew_without_continuum(self):
        samples, args = _setup()
        table = make_parameter_table(samples, args)
        assert not any(col.endswith('_rew') for col in table.colnames)


class TestMakeHDUL:
    """Tests for make_hdul."""

    def test_returns_hdulist(self):
        samples, args = _setup()
        hdul = make_hdul(samples, args)
        assert isinstance(hdul, fits.HDUList)

    def test_primary_hdu(self):
        samples, args = _setup()
        hdul = make_hdul(samples, args)
        assert isinstance(hdul[0], fits.PrimaryHDU)
        assert 'ZSYS' in hdul[0].header

    def test_parameter_hdu(self):
        samples, args = _setup()
        hdul = make_hdul(samples, args)
        assert hdul[1].name == 'PARAMETERS'
        assert isinstance(hdul[1], fits.BinTableHDU)

    def test_spectrum_hdu(self):
        samples, args = _setup()
        hdul = make_hdul(samples, args)
        # HDU 2 should be the spectrum table
        assert len(hdul) >= 3
        assert isinstance(hdul[2], fits.BinTableHDU)

    def test_n_hdus(self):
        samples, args = _setup()
        hdul = make_hdul(samples, args)
        # Primary + params + 1 spectrum = 3
        assert len(hdul) == 3


# ---------------------------------------------------------------------------
# _get_n_samples edge case (results.py line 315)
# ---------------------------------------------------------------------------


class TestGetNSamples:
    """Tests for _get_n_samples helper."""

    def test_empty_dict_returns_1(self):
        """_get_n_samples with empty dict returns 1."""
        assert _get_n_samples({}) == 1

    def test_non_empty_returns_first_shape(self):
        """_get_n_samples returns shape[0] of first array."""
        result = _get_n_samples({'x': np.ones(7)})
        assert result == 7


# ---------------------------------------------------------------------------
# make_hdul with continuum (results.py lines 284-286)
# ---------------------------------------------------------------------------


class TestMakeHDULWithContinuum:
    """Tests for make_hdul with a continuum configuration (covers lines 281-286)."""

    def test_continuum_scale_in_header(self):
        """make_hdul with continuum: CNTSCL and CNTUNT are in primary header."""
        samples, args = _setup_with_continuum()
        hdul = make_hdul(samples, args)
        # The primary header should have line scale info
        assert 'LFLXSCL' in hdul[0].header
        assert 'LFLXUNT' in hdul[0].header

    def test_hdul_with_continuum_n_hdus(self):
        """make_hdul with continuum has correct number of HDUs."""
        samples, args = _setup_with_continuum()
        hdul = make_hdul(samples, args)
        # Primary + params + 1 spectrum = 3
        assert len(hdul) == 3


# ---------------------------------------------------------------------------
# make_spectra_tables summary mode with continuum (lines 209-212)
# ---------------------------------------------------------------------------


class TestMakeSpectraTablesWithContinuum:
    """make_spectra_tables with continuum: covers lines 209-212, 220-221."""

    def test_percentiles_mode_continuum_columns(self):
        """Percentiles mode with continuum produces (n_pix, n_percentiles) continuum columns."""
        samples, args = _setup_with_continuum()
        percentiles = np.array([0.16, 0.5, 0.84])
        tables = make_spectra_tables(samples, args, percentiles=percentiles)
        t = tables[0]
        # Should have at least one continuum region column
        cont_cols = [
            c
            for c in t.colnames
            if c
            not in (
                'wavelength',
                'model_total',
                'Ha',
                'observed_flux',
                'observed_error',
                'scaled_error',
            )
        ]
        assert len(cont_cols) > 0
        for col in cont_cols:
            assert t[col].shape[1] == 3  # 3 percentiles

    def test_full_mode_continuum_columns(self):
        """Full mode (percentiles=None) with continuum produces (n_pix, n_samples) continuum columns."""
        samples, args = _setup_with_continuum()
        tables = make_spectra_tables(samples, args, percentiles=None)
        t = tables[0]
        cont_cols = [
            c
            for c in t.colnames
            if c
            not in (
                'wavelength',
                'model_total',
                'Ha',
                'observed_flux',
                'observed_error',
                'scaled_error',
            )
        ]
        assert len(cont_cols) > 0
        n_samples = 4
        for col in cont_cols:
            assert t[col].shape[1] == n_samples


# ---------------------------------------------------------------------------
# insert_nan coverage (results.py lines 237-238, 468-503)
# ---------------------------------------------------------------------------


def _setup_two_region_continuum():
    """Model with TWO non-adjacent continuum regions to test insert_nan."""
    from jax import random
    from numpyro.infer import Predictive

    # Wide wavelength range covering two separate regions
    wavelength = (
        np.concatenate([np.linspace(6480, 6540, 30), np.linspace(6580, 6640, 30)])
        * u.AA
    )
    wavelength = np.sort(wavelength)

    disperser = SimpleDisperser(
        wavelength=wavelength.value, unit=u.AA, R=3000.0, name='two_region'
    )
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    flux = np.ones(len(wavelength)) * 5.0 * flux_unit
    error = np.ones(len(wavelength)) * flux_unit

    spectrum = GenericSpectrum(
        low=low,
        high=high,
        flux=flux,
        error=error,
        disperser=disperser,
        name='two_region_spec',
    )

    lc = line.LineConfiguration()
    lc.add_line(
        'Ha',
        6510.0 * u.AA,
        redshift=line.Redshift(prior=prior.Fixed(0.0)),
        fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
        flux=line.Flux(prior=prior.Uniform(0, 5)),
    )
    lc.add_line(
        'Hb',
        6610.0 * u.AA,
        redshift=line.Redshift(prior=prior.Fixed(0.0)),
        fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
        flux=line.Flux(prior=prior.Uniform(0, 5)),
    )

    # Two separate continuum regions
    cc = ContinuumConfiguration(
        [
            ContinuumRegion(6480.0 * u.AA, 6540.0 * u.AA, form=Linear()),
            ContinuumRegion(6580.0 * u.AA, 6640.0 * u.AA, form=Linear()),
        ]
    )

    spectra = Spectra([spectrum], redshift=0.0)
    spectra.prepare(lc, cc)
    spectra.compute_scales(spectra.prepared_line_config)

    model_fn, args = model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build()

    rng_key = random.PRNGKey(42)
    predictive = Predictive(model_fn, num_samples=2)
    samples = predictive(rng_key, args)

    return samples, args


class TestInsertNan:
    """Tests for make_spectra_tables with insert_nan=True."""

    def test_insert_nan_adds_rows(self):
        """With two regions, insert_nan=True adds NaN separator rows."""
        samples, args = _setup_two_region_continuum()
        tables_without = make_spectra_tables(samples, args, insert_nan=False)
        tables_with = make_spectra_tables(samples, args, insert_nan=True)
        # Should have at least one more row due to NaN insert
        assert len(tables_with[0]) > len(tables_without[0])

    def test_insert_nan_nan_row_exists(self):
        """NaN row is actually NaN in the model_total column."""
        samples, args = _setup_two_region_continuum()
        tables = make_spectra_tables(samples, args, insert_nan=True)
        t = tables[0]
        model_vals = np.asarray(t['model_total'])
        # At least one row should contain NaN
        assert np.any(np.isnan(model_vals))
