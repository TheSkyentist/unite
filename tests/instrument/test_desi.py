"""Tests for DESI disperser and from_desi_fits loader."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u

from unite.instrument.desi import DESIDisperser
from unite.spectrum import from_desi_fits

SPECTRA_DIR = Path(__file__).parent.parent / 'spectra'
DESI_FILE = SPECTRA_DIR / 'desi-spectra-af558177.fits.gz'


class TestDESIDisperser:
    """Unit tests for DESIDisperser."""

    def test_construction_b(self):
        d = DESIDisperser('B')
        assert d.arm == 'B'
        assert d.unit == u.AA
        assert d.name == 'b'

    def test_construction_lowercase(self):
        d = DESIDisperser('r')
        assert d.arm == 'R'
        assert d.name == 'r'

    def test_custom_name(self):
        d = DESIDisperser('Z', name='my_z')
        assert d.name == 'my_z'

    def test_invalid_arm_raises(self):
        with pytest.raises(ValueError, match='arm must be one of'):
            DESIDisperser('X')

    def test_placeholder_R(self):
        d = DESIDisperser('B')
        wave = jnp.array([3600.0, 4700.0, 5800.0])
        r = d.R(wave)
        assert r.shape == wave.shape
        assert float(r[0]) > 0

    def test_placeholder_dlam_dpix(self):
        d = DESIDisperser('R')
        wave = jnp.array([6000.0])
        dlam = d.dlam_dpix(wave)
        assert float(dlam[0]) > 0

    def test_calibration_tokens(self):
        from unite.instrument.base import FluxScale, RScale
        from unite.prior import Uniform

        r = RScale(prior=Uniform(0.8, 1.2))
        f = FluxScale(prior=Uniform(0.5, 1.5))
        d = DESIDisperser('Z', r_scale=r, flux_scale=f)
        assert d.r_scale is r
        assert d.flux_scale is f
        assert d.has_calibration_params

    def test_repr(self):
        d = DESIDisperser('B')
        assert 'DESIDisperser' in repr(d)
        assert 'B' in repr(d)

    def test_grids_populated_after_load(self):
        d = DESIDisperser('B')
        wave = np.linspace(3600.0, 5800.0, 100)
        d._wavelength_grid = jnp.asarray(wave)
        d._R_grid = jnp.full(100, 3000.0)
        d._dlam_dpix_grid = jnp.full(100, 0.8)
        np.testing.assert_allclose(
            float(d.R(jnp.array([4700.0]))[0]), 3000.0, rtol=1e-5
        )


class TestFromDesiFits:
    """Tests for from_desi_fits using the real example file."""

    def test_default_loads_three_arms(self):
        specs = from_desi_fits(DESI_FILE)
        assert len(specs) == 3

    def test_arm_names_default(self):
        specs = from_desi_fits(DESI_FILE)
        names = [s.name for s in specs]
        assert names == ['b', 'r', 'z']

    def test_arm_names_with_base_name(self):
        specs = from_desi_fits(DESI_FILE, name='target')
        names = [s.name for s in specs]
        assert names == ['target_b', 'target_r', 'target_z']

    def test_b_arm_npix(self):
        specs = from_desi_fits(DESI_FILE)
        b = next(s for s in specs if s.name == 'b')
        assert b.npix == 2751

    def test_r_arm_npix(self):
        specs = from_desi_fits(DESI_FILE)
        r = next(s for s in specs if s.name == 'r')
        assert r.npix == 2326

    def test_z_arm_npix(self):
        specs = from_desi_fits(DESI_FILE)
        z = next(s for s in specs if s.name == 'z')
        assert z.npix == 2881

    def test_b_wavelength_range(self):
        specs = from_desi_fits(DESI_FILE)
        b = next(s for s in specs if s.name == 'b')
        low_val = float(b.low[0]) * u.AA
        high_val = float(b.high[-1]) * u.AA
        assert low_val < 3700 * u.AA
        assert high_val > 5700 * u.AA

    def test_r_wavelength_range(self):
        specs = from_desi_fits(DESI_FILE)
        r = next(s for s in specs if s.name == 'r')
        low_val = float(r.low[0])
        high_val = float(r.high[-1])
        assert low_val < 5850
        assert high_val > 7500

    def test_z_wavelength_range(self):
        specs = from_desi_fits(DESI_FILE)
        z = next(s for s in specs if s.name == 'z')
        low_val = float(z.low[0])
        high_val = float(z.high[-1])
        assert low_val < 7600
        assert high_val > 9700

    def test_flux_units(self):
        specs = from_desi_fits(DESI_FILE)
        for s in specs:
            assert s.flux_unit.is_equivalent(
                u.erg / (u.s * u.cm**2 * u.AA),
                equivalencies=u.spectral_density(1 * u.AA),
            )

    def test_wavelength_unit_angstrom(self):
        specs = from_desi_fits(DESI_FILE)
        for s in specs:
            assert s.unit == u.AA

    def test_R_values_physically_reasonable(self):
        specs = from_desi_fits(DESI_FILE)
        for s in specs:
            wave_mid = jnp.array([float(s.low[s.npix // 2])])
            r_val = float(s.disperser.R(wave_mid)[0])
            assert 1000 < r_val < 10000, f'R={r_val} out of range for arm {s.name}'

    def test_single_arm_selection(self):
        specs = from_desi_fits(DESI_FILE, arms=['R'])
        assert len(specs) == 1
        assert specs[0].name == 'r'

    def test_two_arm_selection(self):
        specs = from_desi_fits(DESI_FILE, arms=['B', 'Z'])
        assert len(specs) == 2
        assert specs[0].name == 'b'
        assert specs[1].name == 'z'

    def test_index_zero_works(self):
        specs = from_desi_fits(DESI_FILE, index=0)
        assert len(specs) == 3

    def test_index_out_of_range_raises(self):
        with pytest.raises(IndexError, match='out of range'):
            from_desi_fits(DESI_FILE, index=99)

    def test_custom_disperser_gets_populated(self):
        disp = DESIDisperser('B')
        specs = from_desi_fits(DESI_FILE, dispersers={'B': disp})
        b = next(s for s in specs if s.name == 'b')
        assert b.disperser is disp
        assert len(disp._wavelength_grid) == 2751

    def test_calibration_token_survives_load(self):
        from unite.instrument.base import RScale

        r = RScale()
        disp = DESIDisperser('R', r_scale=r)
        specs = from_desi_fits(DESI_FILE, dispersers={'R': disp})
        r_spec = next(s for s in specs if s.name == 'r')
        assert r_spec.disperser.r_scale is r

    def test_invalid_arm_raises(self):
        with pytest.raises(ValueError, match='arm must be one of'):
            from_desi_fits(DESI_FILE, arms=['X'])

    def test_missing_arm_warns(self, tmp_path):
        """from_desi_fits warns and skips arms absent from the file."""
        from astropy.io import fits

        # Build a minimal file with only the B arm.
        src = fits.open(DESI_FILE)
        mini = fits.HDUList(
            [
                src['PRIMARY'],
                src['B_WAVELENGTH'],
                src['B_FLUX'],
                src['B_IVAR'],
                src['B_MASK'],
                src['B_RESOLUTION'],
            ]
        )
        mini_path = tmp_path / 'mini.fits'
        mini.writeto(mini_path)
        src.close()

        with pytest.warns(UserWarning, match="DESI arm 'R' not found"):
            specs = from_desi_fits(mini_path, arms=['B', 'R'])

        assert len(specs) == 1
        assert specs[0].name == 'b'

    def test_spectra_collection_integration(self):
        """Loaded arms can be wrapped in a Spectra collection."""
        from unite.spectrum import Spectra

        specs = from_desi_fits(DESI_FILE)
        spectra = Spectra(specs, redshift=1.290)
        assert len(spectra) == 3
