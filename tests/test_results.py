"""Tests for output parsing (unite.results)."""

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.table import Table
from jax import random
from numpyro.infer import Predictive

from unite import line, model, prior
from unite.instrument.generic import GenericSpectrum, SimpleDisperser
from unite.instrument.spectrum import Spectra
from unite.results import make_hdul, make_parameter_table, make_spectra_tables


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

    def test_has_spectrum_metadata(self):
        samples, args = _setup()
        tables = make_spectra_tables(samples, args)
        t = tables[0]
        assert 'SPECNAME' in t.meta
        assert 'NORMFAC' in t.meta

    def test_summary_mode(self):
        samples, args = _setup()
        tables = make_spectra_tables(samples, args, summary=True)
        t = tables[0]
        # Summary gives (n_pixels, 3) for model columns
        assert t['model_total'].shape[0] == len(t['wavelength'])
        assert t['model_total'].shape[1] == 3  # median, p16, p84


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
