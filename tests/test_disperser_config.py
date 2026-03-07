"""Tests for DispersersConfiguration serialization and validation."""

import warnings

import pytest
from astropy import units as u

from unite.disperser.base import FluxScale, PixOffset, RScale
from unite.disperser.config import DispersersConfiguration
from unite.instruments.nirspec import G235H, G395H
from unite.instruments.sdss import SDSSDisperser
from unite.prior import TruncatedNormal, Uniform

# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestDispersersConfigurationConstruction:
    """Tests for DispersersConfiguration construction and validation."""

    def test_basic_construction(self):
        cfg = DispersersConfiguration([G235H(), G395H()])
        assert len(cfg) == 2
        assert cfg.names == ['G235H', 'G395H']

    def test_empty_name_raises(self):
        d = G235H(name='')
        # G235H default name is 'G235H' not '', so force an empty name
        d.name = ''
        with pytest.raises(ValueError, match='empty names'):
            DispersersConfiguration([d])

    def test_duplicate_names_raise(self):
        with pytest.raises(ValueError, match='Duplicate'):
            DispersersConfiguration([G235H(), G235H()])

    def test_getitem_by_index(self):
        cfg = DispersersConfiguration([G235H(), G395H()])
        assert cfg[0].name == 'G235H'
        assert cfg[1].name == 'G395H'

    def test_getitem_by_name(self):
        cfg = DispersersConfiguration([G235H(), G395H()])
        d = cfg['G395H']
        assert d.name == 'G395H'

    def test_getitem_missing_name_raises(self):
        cfg = DispersersConfiguration([G235H()])
        with pytest.raises(KeyError, match='No disperser named'):
            cfg['G999']

    def test_iter(self):
        dispersers = [G235H(), G395H()]
        cfg = DispersersConfiguration(dispersers)
        names = [d.name for d in cfg]
        assert names == ['G235H', 'G395H']

    def test_repr(self):
        cfg = DispersersConfiguration([G235H(), G395H()])
        r = repr(cfg)
        assert 'G235H' in r
        assert 'G395H' in r
        assert '2 disperser(s)' in r


# ---------------------------------------------------------------------------
# Degeneracy warnings
# ---------------------------------------------------------------------------


class TestDegeneracyWarnings:
    """Tests for validate() degeneracy warnings."""

    def test_no_warning_with_anchor(self):
        """No warning when one disperser has flux_scale=None (anchor)."""
        f = FluxScale()
        cfg = DispersersConfiguration([G235H(), G395H(flux_scale=f)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg.validate()
            flux_warnings = [x for x in w if 'flux_scale=None' in str(x.message)]
            assert len(flux_warnings) == 0

    def test_warning_all_flux_scale_set(self):
        """Warning when all dispersers have flux_scale set."""
        f1 = FluxScale()
        f2 = FluxScale()
        cfg = DispersersConfiguration([G235H(flux_scale=f1), G395H(flux_scale=f2)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg.validate()
            msgs = [str(x.message) for x in w]
            assert any('flux_scale=None' in m for m in msgs)

    def test_warning_all_pix_offset_set(self):
        """Warning when all dispersers have pix_offset set."""
        p1 = PixOffset()
        p2 = PixOffset()
        cfg = DispersersConfiguration([G235H(pix_offset=p1), G395H(pix_offset=p2)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg.validate()
            msgs = [str(x.message) for x in w]
            assert any('pix_offset=None' in m for m in msgs)

    def test_no_warning_single_disperser(self):
        """No degeneracy warning for single disperser."""
        f = FluxScale()
        cfg = DispersersConfiguration([G235H(flux_scale=f)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg.validate()
            assert len(w) == 0


# ---------------------------------------------------------------------------
# make_spectrum
# ---------------------------------------------------------------------------


class TestMakeSpectrum:
    """Tests for make_spectrum factory method."""

    def test_make_spectrum(self):
        import numpy as np

        cfg = DispersersConfiguration([G235H()])
        npix = 50
        wl = np.linspace(1.66, 3.17, npix + 1) * u.um
        flux_unit = 1e-20 * u.erg / (u.s * u.cm**2 * u.AA)
        spec = cfg.make_spectrum(
            'G235H',
            wl[:-1],
            wl[1:],
            np.ones(npix) * flux_unit,
            np.full(npix, 0.1) * flux_unit,
        )
        assert spec.npix == npix
        assert spec.name == 'G235H'

    def test_make_spectrum_wrong_name_raises(self):
        import numpy as np

        cfg = DispersersConfiguration([G235H()])
        npix = 10
        wl = np.linspace(1.66, 3.17, npix + 1) * u.um
        flux_unit = 1e-20 * u.erg / (u.s * u.cm**2 * u.AA)
        with pytest.raises(KeyError):
            cfg.make_spectrum(
                'G999',
                wl[:-1],
                wl[1:],
                np.ones(npix) * flux_unit,
                np.full(npix, 0.1) * flux_unit,
            )


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestDispersersConfigSerialization:
    """Tests for to_dict/from_dict and YAML round-trips."""

    def test_dict_roundtrip_nirspec(self):
        r = RScale(prior=TruncatedNormal(1.0, 0.05, 0.8, 1.2), name='shared_r')
        cfg = DispersersConfiguration([G235H(r_scale=r), G395H(r_scale=r)])
        d = cfg.to_dict()
        cfg2 = DispersersConfiguration.from_dict(d)
        assert cfg2.names == ['G235H', 'G395H']
        # Shared token should be the same object after round-trip
        assert cfg2[0].r_scale is cfg2[1].r_scale
        assert cfg2[0].r_scale.name == 'shared_r'

    def test_dict_roundtrip_sdss(self):
        d_sdss = SDSSDisperser(name='SDSS')
        cfg = DispersersConfiguration([d_sdss])
        d = cfg.to_dict()
        cfg2 = DispersersConfiguration.from_dict(d)
        assert cfg2.names == ['SDSS']

    def test_yaml_roundtrip(self):
        f = FluxScale(prior=Uniform(0.5, 2.0), name='f1')
        cfg = DispersersConfiguration([G235H(flux_scale=f), G395H()])
        yaml_str = cfg.to_yaml()
        cfg2 = DispersersConfiguration.from_yaml(yaml_str)
        assert cfg2.names == ['G235H', 'G395H']
        assert cfg2[0].flux_scale is not None
        assert cfg2[1].flux_scale is None

    def test_file_roundtrip(self, tmp_path):
        r = RScale(prior=Uniform(0.9, 1.1), name='r1')
        p = PixOffset(prior=Uniform(-1, 1), name='p1')
        cfg = DispersersConfiguration([G235H(r_scale=r, pix_offset=p), G395H()])
        path = tmp_path / 'dispersers.yaml'
        cfg.save(path)
        cfg2 = DispersersConfiguration.load(path)
        assert cfg2.names == ['G235H', 'G395H']
        assert cfg2[0].r_scale is not None
        assert cfg2[0].pix_offset is not None
        assert cfg2[1].r_scale is None

    def test_calib_params_section_in_dict(self):
        r = RScale(name='r1')
        f = FluxScale(name='f1')
        cfg = DispersersConfiguration([G235H(r_scale=r, flux_scale=f)])
        d = cfg.to_dict()
        assert 'calib_params' in d
        assert 'r1' in d['calib_params']
        assert 'f1' in d['calib_params']

    def test_no_calib_params_when_none(self):
        cfg = DispersersConfiguration([G235H()])
        d = cfg.to_dict()
        assert d['calib_params'] == {}

    def test_r_source_preserved(self):
        cfg = DispersersConfiguration([G235H(r_source='uniform')])
        d = cfg.to_dict()
        cfg2 = DispersersConfiguration.from_dict(d)
        assert cfg2[0].r_source == 'uniform'

    def test_mixed_instruments_roundtrip(self):
        r = RScale(name='shared_r')
        cfg = DispersersConfiguration([G235H(r_scale=r), SDSSDisperser(name='SDSS')])
        d = cfg.to_dict()
        cfg2 = DispersersConfiguration.from_dict(d)
        assert cfg2.names == ['G235H', 'SDSS']
        assert cfg2[0].r_scale is not None
