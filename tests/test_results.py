"""Tests for output parsing (unite.results)."""

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from jax import random
from numpyro.infer import Predictive

from unite import line, model, prior
from unite.continuum import ContinuumConfiguration, Linear
from unite.continuum.config import ContinuumRegion, ContShape, Scale
from unite.instrument.generic import SimpleDisperser
from unite.results import (
    _get_n_samples,
    count_parameters,
    make_hdul,
    make_parameter_table,
    make_spectra_tables,
)
from unite.spectrum import Spectra, Spectrum

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_spectrum(name='test', npix=60):
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


def _build_model(spectrum, lc, cc=None):
    spectra = Spectra([spectrum], redshift=0.0)
    spectra.prepare(lc, cc)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    return model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build()


# ---------------------------------------------------------------------------
# Module-level fixtures (each model built once per module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def simple_setup():
    """Lines-only model with 4 predictive samples."""
    lc = _simple_lc()
    model_fn, args = _build_model(_make_spectrum(), lc)
    samples = Predictive(model_fn, num_samples=4)(random.PRNGKey(0), args)
    return samples, args


@pytest.fixture(scope='module')
def continuum_setup():
    """Lines + continuum model with 4 predictive samples."""
    lc = _simple_lc()
    cont = ContinuumConfiguration.from_lines(
        lc.centers, width=30_000 * u.km / u.s, form=Linear()
    )
    model_fn, args = _build_model(_make_spectrum(), lc, cont)
    samples = Predictive(model_fn, num_samples=4)(random.PRNGKey(0), args)
    return samples, args


@pytest.fixture(scope='module')
def model_fn_args():
    """Return (model_fn, args) without samples, for count_parameters."""
    lc = _simple_lc()
    model_fn, args = _build_model(_make_spectrum(name='s'), lc)
    return model_fn, args


@pytest.fixture(scope='module')
def calib_param_setup():
    """Model with a FluxScale calibration parameter and 4 predictive samples."""
    from unite.instrument.base import FluxScale

    wavelength = np.linspace(6500, 6600, 60) * u.AA
    flux_scale_tok = FluxScale('test', prior=prior.Uniform(0.8, 1.2))
    disperser = SimpleDisperser(
        wavelength=wavelength, R=3000.0, name='test', flux_scale=flux_scale_tok
    )
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)
    sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    line_flux = 50 * np.exp(-0.5 * ((wavelength.value - 6563.0) / sigma) ** 2)
    rng = np.random.default_rng(42)
    flux = (line_flux + 5.0 + rng.normal(0, 1, len(wavelength))) * flux_unit
    error = np.full(len(wavelength), 1.0) * flux_unit
    spec = Spectrum(
        low=low, high=high, flux=flux, error=error, disperser=disperser, name='calib'
    )
    lc = _simple_lc()
    model_fn, args = _build_model(spec, lc)
    samples = Predictive(model_fn, num_samples=4)(random.PRNGKey(0), args)
    return samples, args


@pytest.fixture(scope='module')
def continuum_scale_setup():
    """Lines + continuum model with continuum_scale computed, 4 samples."""
    lc = _simple_lc()
    cont = ContinuumConfiguration.from_lines(
        lc.centers, width=30_000 * u.km / u.s, form=Linear()
    )
    spectra = Spectra([_make_spectrum(name='cs')], redshift=0.0)
    spectra.prepare(lc, cont)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    model_fn, args = model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build()
    samples = Predictive(model_fn, num_samples=4)(random.PRNGKey(0), args)
    return samples, args


@pytest.fixture(scope='module')
def two_region_setup():
    """Model with two non-adjacent continuum regions and 2 predictive samples."""
    wavelength = (
        np.concatenate([np.linspace(6480, 6540, 30), np.linspace(6580, 6640, 30)])
        * u.AA
    )
    wavelength = np.sort(wavelength)
    disperser = SimpleDisperser(wavelength=wavelength, R=3000.0, name='two_region')
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    spec = Spectrum(
        low=low,
        high=high,
        flux=np.ones(len(wavelength)) * 5.0 * flux_unit,
        error=np.ones(len(wavelength)) * flux_unit,
        disperser=disperser,
        name='two_region_spec',
    )
    lc = line.LineConfiguration()
    for name, center in [('Ha', 6510.0), ('Hb', 6610.0)]:
        lc.add_line(
            name,
            center * u.AA,
            redshift=line.Redshift(prior=prior.Fixed(0.0)),
            fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
            flux=line.Flux(prior=prior.Uniform(0, 5)),
        )
    cc = ContinuumConfiguration(
        [
            ContinuumRegion(6480.0 * u.AA, 6540.0 * u.AA, form=Linear()),
            ContinuumRegion(6580.0 * u.AA, 6640.0 * u.AA, form=Linear()),
        ]
    )
    spectra = Spectra([spec], redshift=0.0)
    spectra.prepare(lc, cc)
    spectra.compute_scales(spectra.prepared_line_config)
    model_fn, args = model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build()
    samples = Predictive(model_fn, num_samples=2)(random.PRNGKey(42), args)
    return samples, args


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


_PERCENTILES = np.array([0.16, 0.5, 0.84])


class TestMakeParameterTable:
    """Tests for make_parameter_table."""

    def test_full_mode(self, simple_setup):
        """Table has correct type, all parameter columns, row count, and metadata."""
        samples, args = simple_setup
        table = make_parameter_table(samples, args)
        assert isinstance(table, Table)
        assert len(table) == 4
        for pname in args.dependency_order:
            assert pname in table.colnames
        assert 'LFLXSCL' in table.meta
        assert 'ZSYS' in table.meta

    def test_percentiles_mode(self, simple_setup):
        """Percentile mode adds 'percentile' column with correct values."""
        samples, args = simple_setup
        table = make_parameter_table(samples, args, percentiles=_PERCENTILES)
        assert 'percentile' in table.colnames
        assert len(table) == 3
        assert np.allclose(table['percentile'], _PERCENTILES)
        for pname in args.dependency_order:
            assert pname in table.colnames


class TestMakeSpectraTables:
    """Tests for make_spectra_tables."""

    def test_full_mode(self, simple_setup):
        """Returns list of one Table with correct columns and metadata."""
        samples, args = simple_setup
        tables = make_spectra_tables(samples, args)
        assert isinstance(tables, list) and len(tables) == 1
        t = tables[0]
        assert isinstance(t, Table)
        for col in (
            'wavelength',
            'model_total',
            'observed_flux',
            'observed_error',
            'scaled_error',
        ):
            assert col in t.colnames
        assert 'SPECNAME' in t.meta
        assert 'NORMFAC' in t.meta

    def test_percentiles_mode(self, simple_setup):
        """Percentile mode: model_total has shape (n_pix, n_percentiles)."""
        samples, args = simple_setup
        t = make_spectra_tables(samples, args, percentiles=_PERCENTILES)[0]
        assert t['model_total'].shape == (len(t['wavelength']), 3)


class TestMakeHDUL:
    """Tests for make_hdul (lines-only model)."""

    def test_structure(self, simple_setup):
        """HDUList: Primary + PARAMETERS + spectrum = 3 HDUs with correct types."""
        samples, args = simple_setup
        hdul = make_hdul(samples, args)
        assert isinstance(hdul, fits.HDUList)
        assert len(hdul) == 3
        assert isinstance(hdul[0], fits.PrimaryHDU)
        assert 'ZSYS' in hdul[0].header
        assert hdul[1].name == 'PARAMETERS'
        assert isinstance(hdul[1], fits.BinTableHDU)
        assert isinstance(hdul[2], fits.BinTableHDU)


class TestRestEquivalentWidths:
    """Tests for rest equivalent width columns in make_parameter_table."""

    def test_rew_basic(self, continuum_setup):
        """REW column present, has unit, correct length, all finite."""
        samples, args = continuum_setup
        table = make_parameter_table(samples, args)
        assert 'rew_Ha' in table.colnames
        assert table['rew_Ha'].unit is not None
        assert len(table['rew_Ha']) == 4
        assert np.all(np.isfinite(np.asarray(table['rew_Ha'])))

    def test_rew_percentiles_mode(self, continuum_setup):
        """REW present in percentile mode with correct shape."""
        samples, args = continuum_setup
        table = make_parameter_table(samples, args, percentiles=_PERCENTILES)
        assert 'rew_Ha' in table.colnames
        assert len(table['rew_Ha']) == 3
        assert 'percentile' in table.colnames

    def test_no_rew_without_continuum(self, simple_setup):
        """No REW columns when no continuum is configured."""
        samples, args = simple_setup
        table = make_parameter_table(samples, args)
        assert not any(col.startswith('rew_') for col in table.colnames)


class TestMakeHDULWithContinuum:
    """Tests for make_hdul with continuum."""

    def test_continuum_header_keys(self, continuum_scale_setup):
        """LFLXSCL, LFLXUNT, CNTSCL, CNTUNT appear in the primary header."""
        samples, args = continuum_scale_setup
        hdul = make_hdul(samples, args)
        for key in ('LFLXSCL', 'LFLXUNT', 'CNTSCL', 'CNTUNT'):
            assert key in hdul[0].header
        assert len(hdul) == 3


class TestMakeSpectraTablesWithContinuum:
    """make_spectra_tables with continuum (summary and full modes)."""

    def _get_cont_cols(self, t):
        known = {
            'wavelength',
            'model_total',
            'Ha',
            'observed_flux',
            'observed_error',
            'scaled_error',
        }
        return [c for c in t.colnames if c not in known]

    def test_percentiles_mode(self, continuum_setup):
        """Percentile mode: continuum columns have shape (n_pix, 3)."""
        samples, args = continuum_setup
        t = make_spectra_tables(samples, args, percentiles=_PERCENTILES)[0]
        cont_cols = self._get_cont_cols(t)
        assert len(cont_cols) > 0
        for col in cont_cols:
            assert t[col].shape[1] == 3

    def test_full_mode(self, continuum_setup):
        """Full mode: continuum columns have shape (n_pix, n_samples)."""
        samples, args = continuum_setup
        t = make_spectra_tables(samples, args, percentiles=None)[0]
        cont_cols = self._get_cont_cols(t)
        assert len(cont_cols) > 0
        for col in cont_cols:
            assert t[col].shape[1] == 4


class TestInsertNan:
    """Tests for make_spectra_tables with insert_nan=True."""

    def test_insert_nan(self, two_region_setup):
        """With two regions: insert_nan=True adds NaN separator rows."""
        samples, args = two_region_setup
        tables_without = make_spectra_tables(samples, args, insert_nan=False)
        tables_with = make_spectra_tables(samples, args, insert_nan=True)
        assert len(tables_with[0]) > len(tables_without[0])
        model_vals = np.asarray(tables_with[0]['model_total'])
        assert np.any(np.isnan(model_vals))


class TestGetNSamples:
    """Tests for _get_n_samples helper."""

    def test_empty_returns_1(self):
        assert _get_n_samples({}) == 1

    def test_non_empty_returns_shape(self):
        assert _get_n_samples({'x': np.ones(7)}) == 7


class TestInstrumentGroupInParameterTable:
    """Calibration parameters appear in the parameter table."""

    def test_instrument_param_in_table(self, calib_param_setup):
        samples, args = calib_param_setup
        table = make_parameter_table(samples, args)
        assert any('flux_scale' in col for col in table.colnames)


class TestCountParameters:
    """Tests for count_parameters."""

    def test_count_free_params(self, model_fn_args):
        """Ha has 3 free params (z, fwhm_gauss, flux); count matches."""
        model_fn, args = model_fn_args
        n = count_parameters(model_fn, args)
        assert isinstance(n, int) and n == 3


class TestFixedParamPercentiles:
    """Fixed prior params handled correctly in percentile mode."""

    def test_fixed_constant_across_percentiles(self, two_region_setup):
        """Fixed params have identical value for all percentile rows."""
        samples, args = two_region_setup
        table = make_parameter_table(samples, args, percentiles=_PERCENTILES)
        assert len(table) == 3
        for pname in args.dependency_order:
            p = args.all_priors[pname]
            if isinstance(p, prior.Fixed):
                vals = np.asarray(table[pname])
                assert np.all(vals == vals[0]), f'{pname} should be constant'


# ---------------------------------------------------------------------------
# REW accuracy tests with fully-determined (Fixed prior) models
# ---------------------------------------------------------------------------


def _make_rew_spectrum(npix=500):
    """High-resolution flat-continuum spectrum centred on Ha at z=0."""
    wavelength = np.linspace(6400, 6700, npix) * u.AA
    disperser = SimpleDisperser(wavelength=wavelength, R=5000.0, name='rew')
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
    line_flux = 50 * np.exp(-0.5 * ((wavelength.value - 6563.0) / sigma) ** 2)
    flux = (line_flux + 10.0) * flux_unit
    error = np.full(npix, 1.0) * flux_unit
    return Spectrum(
        low=low, high=high, flux=flux, error=error, disperser=disperser, name='rew'
    )


@pytest.fixture(scope='module')
def emission_rew_setup():
    """Deterministic emission model: known flux and flat continuum, z=0."""
    lc = line.LineConfiguration()
    lc.add_line(
        'Ha',
        6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Fixed(0.0)),
        fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
        flux=line.Flux(prior=prior.Fixed(1.0)),
    )
    cont = ContinuumConfiguration.from_lines(
        lc.centers, width=30_000 * u.km / u.s, form=Linear()
    )
    # Override continuum priors: flat at scale=1, angle=0.
    for region in cont:
        region.params['scale'] = Scale(prior=prior.Fixed(1.0))
        region.params['angle'] = ContShape(prior=prior.Fixed(0.0))
    spec = _make_rew_spectrum()
    spectra = Spectra([spec], redshift=0.0)
    spectra.prepare(lc, cont)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    _model_fn, args = model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build()
    return args


@pytest.fixture(scope='module')
def absorption_rew_setup():
    """Deterministic absorption model: known tau on flat continuum, z=0."""
    from unite.line.config import Tau
    from unite.line.library import Gaussian

    lc = line.LineConfiguration()
    # Need an emission line for compute_scales to work.
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
        profile=Gaussian(),
        redshift=line.Redshift(prior=prior.Fixed(0.0)),
        fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
        tau=Tau(prior=prior.Fixed(2.0)),
    )
    cont = ContinuumConfiguration.from_lines(
        lc.centers, width=30_000 * u.km / u.s, form=Linear()
    )
    for region in cont:
        region.params['scale'] = Scale(prior=prior.Fixed(1.0))
        region.params['angle'] = ContShape(prior=prior.Fixed(0.0))
    spec = _make_rew_spectrum()
    spectra = Spectra([spec], redshift=0.0)
    spectra.prepare(lc, cont)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    _model_fn, args = model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build()
    return args


class TestREWAccuracy:
    """Verify REW values against analytic expectations."""

    def test_emission_rew_value(self, emission_rew_setup):
        """Emission REW = flux / continuum at z=0.

        With scale=1 (un-scaled), cont_physical = 1 * cont_scale.
        flux_physical = 1 * line_flux_scale.
        REW = line_flux_scale / (cont_scale * 1.0) at z=0.
        """
        args = emission_rew_setup
        from unite.results import _compute_rew_columns

        rew_cols = _compute_rew_columns({}, args)
        assert 'rew_Ha' in rew_cols
        rew = rew_cols['rew_Ha']
        # Expected: flux_physical / cont_physical = line_flux_scale / cont_scale
        expected = args.line_flux_scales[0] / args.continuum_scales[0]
        np.testing.assert_allclose(rew, expected, rtol=1e-4)

    def test_absorption_rew_negative(self, absorption_rew_setup):
        """Absorption REW should be negative (flux is removed)."""
        args = absorption_rew_setup
        from unite.results import _compute_rew_columns

        rew_cols = _compute_rew_columns({}, args)
        assert 'rew_HI_abs' in rew_cols
        rew = rew_cols['rew_HI_abs']
        assert np.all(rew < 0), f'Absorption REW should be negative, got {rew}'

    def test_absorption_rew_appears_in_table(self, absorption_rew_setup):
        """Absorption REW appears in parameter table."""
        args = absorption_rew_setup
        table = make_parameter_table({}, args)
        assert 'rew_HI_abs' in table.colnames
        assert table['rew_HI_abs'].unit is not None

    def test_absorption_rew_magnitude(self, absorption_rew_setup):
        """Absorption REW magnitude is physical: non-zero, finite, order-of-AA.

        For a Gaussian absorber with tau=2 and FWHM=300 km/s at 6563 AA,
        the effective FWHM in AA is ~6.56 AA.  The integral of (T-1) for
        tau=2 is between -FWHM (weak limit) and 0 (no absorption), so the
        REW should be a few AA (negative).
        """
        args = absorption_rew_setup
        from unite.results import _compute_rew_columns

        rew_cols = _compute_rew_columns({}, args)
        rew = rew_cols['rew_HI_abs']
        assert np.all(np.isfinite(rew))
        assert np.all(np.abs(rew) > 0.1)  # not negligibly small
        assert np.all(np.abs(rew) < 100)  # not unphysically large

    def test_absorption_line_in_spectra_table(self, absorption_rew_setup):
        """Absorption line appears as a regular (negative) column in spectra tables."""
        args = absorption_rew_setup
        tables = make_spectra_tables({}, args)
        t = tables[0]
        # Absorption line appears under its label, not as trans_*.
        assert 'HI_abs' in t.colnames
        assert 'trans_HI_abs' not in t.colnames
        delta_vals = np.asarray(t['HI_abs'])
        # Should be non-positive (flux removed).
        assert np.all(delta_vals <= 0)

    def test_absorption_rew_in_percentile_mode(self, absorption_rew_setup):
        """Absorption REW column appears in parameter table in percentile mode."""
        args = absorption_rew_setup
        table = make_parameter_table({}, args, percentiles=np.array([0.16, 0.5, 0.84]))
        assert 'rew_HI_abs' in table.colnames
        assert len(table['rew_HI_abs']) == 3
        assert np.all(np.isfinite(np.asarray(table['rew_HI_abs'])))
