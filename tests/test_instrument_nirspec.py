"""Tests for NIRSpec disperser and spectrum loader."""

from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u

from unite.instrument.nirspec import (
    G140H,
    G140M,
    G235H,
    G235M,
    G395H,
    G395M,
    PRISM,
    NIRSpecDisperser,
    NIRSpecSpectrum,
)

# ---------------------------------------------------------------------------
# NIRSpecDisperser — construction and validation
# ---------------------------------------------------------------------------


class TestNIRSpecDisperserConstruction:
    """Tests for NIRSpecDisperser construction and basic properties."""

    def test_default_construction(self):
        d = NIRSpecDisperser('g235h')
        assert d.grating == 'g235h'
        assert d.r_source == 'point'
        assert d.unit == u.um
        assert d.name == 'G235H'

    def test_case_insensitive_grating(self):
        d = NIRSpecDisperser('G395M')
        assert d.grating == 'g395m'

    def test_invalid_grating_raises(self):
        with pytest.raises(ValueError, match='Unknown NIRSpec grating'):
            NIRSpecDisperser('invalid')

    def test_r_source_uniform(self):
        d = NIRSpecDisperser('g235h', r_source='uniform')
        assert d.r_source == 'uniform'

    def test_r_source_aliases(self):
        d1 = NIRSpecDisperser('g235h', r_source='uniformly-illuminated')
        assert d1.r_source == 'uniform'
        d2 = NIRSpecDisperser('g235h', r_source='point-source')
        assert d2.r_source == 'point'

    def test_invalid_r_source_raises(self):
        with pytest.raises(ValueError, match='Unknown r_source'):
            NIRSpecDisperser('g235h', r_source='invalid')

    def test_custom_name(self):
        d = NIRSpecDisperser('prism', name='my_prism')
        assert d.name == 'my_prism'


# ---------------------------------------------------------------------------
# Resolving power and dispersion — all gratings
# ---------------------------------------------------------------------------

_ALL_GRATINGS = ['prism', 'g140m', 'g140h', 'g235m', 'g235h', 'g395m', 'g395h']

_GRATING_WL_RANGES = {
    'prism': (0.6, 5.3),
    'g140m': (0.97, 1.87),
    'g140h': (0.97, 1.87),
    'g235m': (1.66, 3.17),
    'g235h': (1.66, 3.17),
    'g395m': (2.87, 5.27),
    'g395h': (2.87, 5.27),
}


class TestNIRSpecR:
    """Tests for resolving power computation across all gratings."""

    @pytest.mark.parametrize('grating', _ALL_GRATINGS)
    def test_R_point_returns_positive(self, grating):
        d = NIRSpecDisperser(grating, r_source='point')
        lo, hi = _GRATING_WL_RANGES[grating]
        wl = jnp.linspace(lo, hi, 50)
        r = d.R(wl)
        assert jnp.all(r > 0), f'{grating} point R not all positive'

    @pytest.mark.parametrize('grating', _ALL_GRATINGS)
    def test_R_uniform_returns_positive(self, grating):
        d = NIRSpecDisperser(grating, r_source='uniform')
        lo, hi = _GRATING_WL_RANGES[grating]
        wl = jnp.linspace(lo, hi, 50)
        r = d.R(wl)
        assert jnp.all(r > 0), f'{grating} uniform R not all positive'

    @pytest.mark.parametrize('grating', _ALL_GRATINGS)
    def test_dlam_dpix_positive(self, grating):
        d = NIRSpecDisperser(grating)
        lo, hi = _GRATING_WL_RANGES[grating]
        wl = jnp.linspace(lo, hi, 50)
        dlam = d.dlam_dpix(wl)
        assert jnp.all(dlam > 0), f'{grating} dlam_dpix not all positive'

    def test_high_R_grating_has_higher_R_than_medium(self):
        """G235H should have higher R than G235M at the same wavelength."""
        wl = jnp.array([2.0, 2.5])
        r_h = G235H().R(wl)
        r_m = G235M().R(wl)
        assert jnp.all(r_h > r_m)

    def test_point_vs_uniform_differ(self):
        """Point and uniform R sources should give different values."""
        wl = jnp.array([2.0, 2.5, 3.0])
        r_point = NIRSpecDisperser('g235h', r_source='point').R(wl)
        r_uniform = NIRSpecDisperser('g235h', r_source='uniform').R(wl)
        # They should not be identical
        assert not jnp.allclose(r_point, r_uniform, atol=1.0)

    @pytest.mark.parametrize('grating', _ALL_GRATINGS)
    def test_R_reasonable_range(self, grating):
        """R should be in a physically reasonable range."""
        d = NIRSpecDisperser(grating)
        lo, hi = _GRATING_WL_RANGES[grating]
        wl = jnp.linspace(lo, hi, 50)
        r = d.R(wl)
        if grating == 'prism':
            assert jnp.all(r > 10) and jnp.all(r < 600)
        elif grating.endswith('m'):
            assert jnp.all(r > 500) and jnp.all(r < 3000)
        else:
            assert jnp.all(r > 1000) and jnp.all(r < 8000)


# ---------------------------------------------------------------------------
# Convenience subclasses
# ---------------------------------------------------------------------------

_CONVENIENCE_CLASSES = [PRISM, G140M, G140H, G235M, G235H, G395M, G395H]
_CONVENIENCE_NAMES = ['PRISM', 'G140M', 'G140H', 'G235M', 'G235H', 'G395M', 'G395H']


class TestConvenienceClasses:
    """Tests for per-grating convenience subclasses."""

    @pytest.mark.parametrize(
        'cls,name', zip(_CONVENIENCE_CLASSES, _CONVENIENCE_NAMES, strict=True)
    )
    def test_construction(self, cls, name):
        d = cls()
        assert d.name == name
        assert d.grating == name.lower()
        assert d.unit == u.um

    @pytest.mark.parametrize('cls', _CONVENIENCE_CLASSES)
    def test_r_source_kwarg(self, cls):
        d = cls(r_source='uniform')
        assert d.r_source == 'uniform'

    @pytest.mark.parametrize('cls', _CONVENIENCE_CLASSES)
    def test_is_nirspec_disperser(self, cls):
        d = cls()
        assert isinstance(d, NIRSpecDisperser)

    def test_repr(self):
        d = G235H()
        assert 'G235H' in repr(d)
        assert 'point' in repr(d)


# ---------------------------------------------------------------------------
# Calibration tokens
# ---------------------------------------------------------------------------


class TestNIRSpecCalibration:
    """Tests for calibration tokens on NIRSpec dispersers."""

    def test_no_calibration_by_default(self):
        d = G235H()
        assert not d.has_calibration_params
        assert d.r_scale is None
        assert d.flux_scale is None
        assert d.pix_offset is None

    def test_r_scale_token(self):
        from unite.instrument.base import RScale

        r = RScale()
        d = G235H(r_scale=r)
        assert d.has_calibration_params
        assert d.r_scale is r

    def test_flux_scale_token(self):
        from unite.instrument.base import FluxScale

        f = FluxScale()
        d = G235H(flux_scale=f)
        assert d.has_calibration_params
        assert d.flux_scale is f

    def test_pix_offset_token(self):
        from unite.instrument.base import PixOffset

        p = PixOffset()
        d = G235H(pix_offset=p)
        assert d.has_calibration_params
        assert d.pix_offset is p

    def test_shared_token_identity(self):
        """Two dispersers sharing a token should reference the same object."""
        from unite.instrument.base import RScale

        r = RScale()
        d1 = G235H(r_scale=r)
        d2 = G395H(r_scale=r)
        assert d1.r_scale is d2.r_scale


# ---------------------------------------------------------------------------
# NIRSpecSpectrum loader
# ---------------------------------------------------------------------------


class TestNIRSpecSpectrum:
    """Tests for NIRSpecSpectrum loader class."""

    def test_is_generic_spectrum_subclass(self):
        from unite.instrument.generic import GenericSpectrum

        assert issubclass(NIRSpecSpectrum, GenericSpectrum)

    def test_from_arrays(self):
        npix = 200
        wl = np.linspace(1.66, 3.17, npix + 1) * u.um
        low = wl[:-1]
        high = wl[1:]
        flux_unit = 1e-20 * u.erg / (u.s * u.cm**2 * u.AA)
        flux = np.ones(npix) * flux_unit
        error = np.full(npix, 0.1) * flux_unit
        d = G235H()
        spec = NIRSpecSpectrum.from_arrays(low, high, flux, error, d)
        assert spec.npix == npix
        assert spec.name == 'G235H'
        assert spec.unit == u.um

    def test_from_arrays_custom_name(self):
        npix = 50
        wl = np.linspace(1.66, 3.17, npix + 1) * u.um
        flux_unit = 1e-20 * u.erg / (u.s * u.cm**2 * u.AA)
        d = G235H()
        spec = NIRSpecSpectrum.from_arrays(
            wl[:-1],
            wl[1:],
            np.ones(npix) * flux_unit,
            np.full(npix, 0.1) * flux_unit,
            d,
            name='custom',
        )
        assert spec.name == 'custom'

    def test_from_arrays_angstrom_input(self):
        """Wavelengths in Angstroms should be accepted and converted."""
        npix = 50
        wl = np.linspace(16600, 31700, npix + 1) * u.AA
        flux_unit = 1e-20 * u.erg / (u.s * u.cm**2 * u.AA)
        d = G235H()
        spec = NIRSpecSpectrum.from_arrays(
            wl[:-1],
            wl[1:],
            np.ones(npix) * flux_unit,
            np.full(npix, 0.1) * flux_unit,
            d,
        )
        assert spec.npix == npix
        # Internal storage should be in um (disperser unit)
        assert spec.unit == u.um


# ---------------------------------------------------------------------------
# NIRSpecSpectrum.from_DJA — mocked FITS loading
# ---------------------------------------------------------------------------


class _FakeFluxCol:
    """Minimal column-like object with .to() and .mask, mimicking DJA masked columns."""

    def __init__(self, data, unit, mask=None):
        self._data = data
        self._unit = unit
        self.mask = mask if mask is not None else np.zeros(len(data), dtype=bool)

    def to(self, unit, equivalencies=None):
        q = u.Quantity(self._data, self._unit)
        return q.to(unit, equivalencies=equivalencies)


def _make_nirspec_dja_table(n=50, n_masked=0):
    """Create a minimal fake DJA SPEC1D table."""
    fλ_unit = 1e-20 * u.erg / (u.s * u.cm**2 * u.AA)
    wave_aa = np.linspace(16000, 20000, n)  # Angstroms
    flux_vals = np.ones(n) * 1.0
    err_vals = np.ones(n) * 0.1
    mask = np.zeros(n, dtype=bool)
    if n_masked > 0:
        mask[:n_masked] = True

    class FakeDJATable:
        def __getitem__(self, key):
            if key == 'wave':
                return wave_aa * u.AA
            elif key == 'flux':
                return _FakeFluxCol(flux_vals, fλ_unit, mask)
            elif key == 'err':
                return _FakeFluxCol(err_vals, fλ_unit, mask)
            raise KeyError(key)

    return FakeDJATable()


class TestNIRSpecFromDJA:
    """Tests for NIRSpecSpectrum.from_DJA using mocked Table.read."""

    def test_from_dja_basic(self):
        """from_DJA constructs a valid spectrum from mocked DJA FITS."""
        fake_table = _make_nirspec_dja_table(n=50)
        with patch('astropy.table.Table.read', return_value=fake_table):
            disperser = G235H()
            spec = NIRSpecSpectrum.from_DJA('fake.fits', disperser)

        assert spec.npix == 50
        assert spec.unit == u.um  # NIRSpec uses microns

    def test_from_dja_mask_applied(self):
        """from_DJA removes masked pixels from the output."""
        fake_table = _make_nirspec_dja_table(n=50, n_masked=5)
        with patch('astropy.table.Table.read', return_value=fake_table):
            disperser = G235H()
            spec = NIRSpecSpectrum.from_DJA('fake.fits', disperser)

        # 5 pixels masked → 45 remaining
        assert spec.npix == 45

    def test_from_dja_custom_name(self):
        """from_DJA uses the provided name."""
        fake_table = _make_nirspec_dja_table(n=50)
        with patch('astropy.table.Table.read', return_value=fake_table):
            disperser = G235H()
            spec = NIRSpecSpectrum.from_DJA('fake.fits', disperser, name='MyGalaxy')

        assert spec.name == 'MyGalaxy'

    def test_from_dja_default_name_is_disperser(self):
        """from_DJA defaults name to disperser.name when not provided."""
        fake_table = _make_nirspec_dja_table(n=50)
        with patch('astropy.table.Table.read', return_value=fake_table):
            disperser = G235H()
            spec = NIRSpecSpectrum.from_DJA('fake.fits', disperser)

        assert spec.name == disperser.name
