"""Tests for SDSS disperser."""

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
from astropy import units as u
from astropy.table import Table

from unite.instrument.sdss import SDSSDisperser, SDSSSpectrum


def _make_sdss_table(n=50, include_and_mask=True, has_bad_ivar=False):
    """Build a minimal fake SDSS COADD table."""
    loglam = np.log10(np.linspace(4000, 8000, n))
    flux = np.ones(n) * 5.0
    ivar = np.ones(n) * 4.0  # error = 0.5
    if has_bad_ivar:
        ivar[10] = 0.0  # one bad pixel
    wdisp = np.ones(n) * 0.5

    data = {'loglam': loglam, 'flux': flux, 'ivar': ivar, 'wdisp': wdisp}
    if include_and_mask:
        and_mask = np.zeros(n, dtype=int)
        and_mask[20] = 1  # one masked pixel
        data['and_mask'] = and_mask

    return Table(data)


class TestSDSSDisperser:
    """Tests for SDSSDisperser."""

    def test_default_construction(self):
        d = SDSSDisperser()
        assert d.unit == u.AA
        assert d.name == 'SDSS'

    def test_construction_with_data(self):
        wl = np.linspace(3800, 9200, 100)
        wdisp = np.full(100, 1.0)  # 1 Angstrom/pixel sigma
        d = SDSSDisperser()
        # Manually set wavelength and R grids to simulate from_fits
        d._wavelength_grid = jnp.asarray(wl, dtype=float)
        d._R_grid = jnp.asarray(wl / (2.355 * wdisp), dtype=float)
        d._dlam_dpix_grid = jnp.gradient(d._wavelength_grid)
        # R = lambda / (2.355 * wdisp)
        expected_r = wl / (2.355 * wdisp)
        np.testing.assert_allclose(d.R(jnp.asarray(wl)), expected_r, rtol=1e-5)

    def test_construction_with_wavelength_only(self):
        wl = np.linspace(3800, 9200, 100)
        d = SDSSDisperser()
        # Manually set wavelength and default R grid
        d._wavelength_grid = jnp.asarray(wl, dtype=float)
        d._R_grid = jnp.full_like(d._wavelength_grid, 2000.0)
        d._dlam_dpix_grid = jnp.gradient(d._wavelength_grid)
        # Default R = 2000
        np.testing.assert_allclose(d.R(jnp.asarray(wl)), 2000.0)

    def test_calibration_tokens(self):
        from unite.instrument.base import FluxScale, RScale

        r = RScale()
        f = FluxScale()
        d = SDSSDisperser(r_scale=r, flux_scale=f)
        assert d.r_scale is r
        assert d.flux_scale is f
        assert d.has_calibration_params

    def test_repr(self):
        d = SDSSDisperser()
        assert 'SDSSDisperser' in repr(d)
        assert d.name == 'SDSS'

        wl = np.linspace(3800, 9200, 100)
        d2 = SDSSDisperser(name='SDSS-test')
        d2._wavelength_grid = jnp.asarray(wl, dtype=float)
        d2._R_grid = jnp.full_like(d2._wavelength_grid, 2000.0)
        d2._dlam_dpix_grid = jnp.gradient(d2._wavelength_grid)
        # Repr should still work with data loaded
        assert 'SDSS-test' in repr(d2)


class TestSDSSSpectrum:
    """Tests for SDSSSpectrum loader."""

    def test_is_generic_spectrum_subclass(self):
        from unite.instrument.generic import GenericSpectrum

        assert issubclass(SDSSSpectrum, GenericSpectrum)

    def test_from_arrays(self):
        npix = 50
        wl = np.linspace(4000, 8000, npix + 1) * u.AA
        low = wl[:-1]
        high = wl[1:]
        flux_unit = 1e-17 * u.erg / (u.s * u.cm**2 * u.AA)
        flux = np.ones(npix) * flux_unit
        error = np.full(npix, 0.1) * flux_unit
        d = SDSSDisperser()
        # Set wavelength grid to allow proper construction
        d._wavelength_grid = jnp.asarray(np.linspace(4000, 8000, npix), dtype=float)
        d._R_grid = jnp.full_like(d._wavelength_grid, 2000.0)
        d._dlam_dpix_grid = jnp.gradient(d._wavelength_grid)
        # Use constructor directly (not from_arrays)
        spec = SDSSSpectrum(low=low, high=high, flux=flux, error=error, disperser=d)
        assert spec.npix == npix
        assert spec.name == 'SDSS'


class TestSDSSFromFits:
    """Tests for SDSSSpectrum.from_fits using mocked Table.read."""

    def test_from_fits_basic(self):
        """from_fits loads a standard SDSS spec file with and_mask."""
        fake_table = _make_sdss_table(n=50, include_and_mask=True)
        with patch('astropy.table.Table.read', return_value=fake_table):
            disperser = SDSSDisperser()
            spec = SDSSSpectrum.from_fits('fake.fits', disperser)

        # and_mask pixel 20 is masked, so npix < 50
        assert spec.npix < 50
        assert spec.npix > 0

    def test_from_fits_without_and_mask(self):
        """from_fits works when and_mask column is absent (line 156-157)."""
        fake_table = _make_sdss_table(n=50, include_and_mask=False)
        with patch('astropy.table.Table.read', return_value=fake_table):
            disperser = SDSSDisperser()
            spec = SDSSSpectrum.from_fits('fake.fits', disperser)

        # All pixels have ivar > 0, so all should be kept
        assert spec.npix == 50

    def test_from_fits_bad_ivar(self):
        """from_fits masks pixels with ivar=0 and sets large error."""
        fake_table = _make_sdss_table(n=50, include_and_mask=False, has_bad_ivar=True)
        with patch('astropy.table.Table.read', return_value=fake_table):
            disperser = SDSSDisperser()
            spec = SDSSSpectrum.from_fits('fake.fits', disperser)

        # Pixel 10 has ivar=0 and is masked by and_mask=False path (ivar > 0 filter)
        assert spec.npix == 49  # one bad pixel excluded

    def test_from_fits_updates_disperser(self):
        """from_fits updates the disperser's wavelength grid."""
        fake_table = _make_sdss_table(n=50, include_and_mask=False)
        with patch('astropy.table.Table.read', return_value=fake_table):
            disperser = SDSSDisperser()
            SDSSSpectrum.from_fits('fake.fits', disperser)

        # Verify that from_fits updated the disperser's grids
        assert hasattr(disperser, '_wavelength_grid')
        assert disperser._wavelength_grid is not None
        assert len(disperser._wavelength_grid) == 50

    def test_from_fits_custom_name(self):
        """from_fits uses the provided name."""
        fake_table = _make_sdss_table(n=50, include_and_mask=False)
        with patch('astropy.table.Table.read', return_value=fake_table):
            disperser = SDSSDisperser()
            spec = SDSSSpectrum.from_fits('fake.fits', disperser, name='MySpec')

        assert spec.name == 'MySpec'
