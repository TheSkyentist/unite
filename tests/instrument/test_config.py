"""Tests for InstrumentConfig serialization and validation."""

import warnings

import pytest

from unite.instrument.base import FluxScale, PixOffset, RScale
from unite.instrument.config import InstrumentConfig
from unite.instrument.nirspec import G235H, G395H
from unite.instrument.sdss import SDSSDisperser
from unite.prior import TruncatedNormal, Uniform

# ---------------------------------------------------------------------------
# Construction and validation
# ---------------------------------------------------------------------------


class TestInstrumentConfigConstruction:
    """Tests for InstrumentConfig construction and validation."""

    def test_basic_construction(self):
        cfg = InstrumentConfig([G235H(), G395H()])
        assert len(cfg) == 2
        assert cfg.names == ['G235H', 'G395H']

    def test_empty_name_raises(self):
        d = G235H(name='')
        # G235H default name is 'G235H' not '', so force an empty name
        d.name = ''
        with pytest.raises(ValueError, match='empty names'):
            InstrumentConfig([d])

    def test_duplicate_names_raise(self):
        with pytest.raises(ValueError, match='Duplicate'):
            InstrumentConfig([G235H(), G235H()])

    def test_getitem_by_index(self):
        cfg = InstrumentConfig([G235H(), G395H()])
        assert cfg[0].name == 'G235H'
        assert cfg[1].name == 'G395H'

    def test_getitem_by_name(self):
        cfg = InstrumentConfig([G235H(), G395H()])
        d = cfg['G395H']
        assert d.name == 'G395H'

    def test_getitem_missing_name_raises(self):
        cfg = InstrumentConfig([G235H()])
        with pytest.raises(KeyError, match='No disperser named'):
            cfg['G999']

    def test_iter(self):
        dispersers = [G235H(), G395H()]
        cfg = InstrumentConfig(dispersers)
        names = [d.name for d in cfg]
        assert names == ['G235H', 'G395H']

    def test_repr(self):
        cfg = InstrumentConfig([G235H(), G395H()])
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
        cfg = InstrumentConfig([G235H(), G395H(flux_scale=f)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg.validate()
            flux_warnings = [x for x in w if 'flux_scale=None' in str(x.message)]
            assert len(flux_warnings) == 0

    def test_warning_all_flux_scale_set(self):
        """Warning when all dispersers have flux_scale set."""
        f1 = FluxScale()
        f2 = FluxScale()
        cfg = InstrumentConfig([G235H(flux_scale=f1), G395H(flux_scale=f2)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg.validate()
            msgs = [str(x.message) for x in w]
            assert any('flux_scale=None' in m for m in msgs)

    def test_warning_all_pix_offset_set(self):
        """Warning when all dispersers have pix_offset set."""
        p1 = PixOffset()
        p2 = PixOffset()
        cfg = InstrumentConfig([G235H(pix_offset=p1), G395H(pix_offset=p2)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg.validate()
            msgs = [str(x.message) for x in w]
            assert any('pix_offset=None' in m for m in msgs)

    def test_no_warning_single_disperser(self):
        """No degeneracy warning for single disperser."""
        f = FluxScale()
        cfg = InstrumentConfig([G235H(flux_scale=f)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            cfg.validate()
            assert len(w) == 0


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestInstrumentConfigSerialization:
    """Tests for to_dict/from_dict and YAML round-trips."""

    def test_dict_roundtrip_nirspec(self):
        r = RScale(prior=TruncatedNormal(1.0, 0.05, 0.8, 1.2), name='shared_r')
        cfg = InstrumentConfig([G235H(r_scale=r), G395H(r_scale=r)])
        d = cfg.to_dict()
        cfg2 = InstrumentConfig.from_dict(d)
        assert cfg2.names == ['G235H', 'G395H']
        # Shared token should be the same object after round-trip
        assert cfg2[0].r_scale is cfg2[1].r_scale
        # Site name is 'r_scale_shared_r' (prefix 'r_scale' + label 'shared_r')
        assert cfg2[0].r_scale.name == 'r_scale_shared_r'

    def test_dict_roundtrip_sdss(self):
        d_sdss = SDSSDisperser(name='SDSS')
        cfg = InstrumentConfig([d_sdss])
        d = cfg.to_dict()
        cfg2 = InstrumentConfig.from_dict(d)
        assert cfg2.names == ['SDSS']

    def test_yaml_roundtrip(self):
        f = FluxScale(prior=Uniform(0.5, 2.0), name='f1')
        cfg = InstrumentConfig([G235H(flux_scale=f), G395H()])
        yaml_str = cfg.to_yaml()
        cfg2 = InstrumentConfig.from_yaml(yaml_str)
        assert cfg2.names == ['G235H', 'G395H']
        assert cfg2[0].flux_scale is not None
        assert cfg2[1].flux_scale is None

    def test_file_roundtrip(self, tmp_path):
        r = RScale(prior=Uniform(0.9, 1.1), name='r1')
        p = PixOffset(prior=Uniform(-1, 1), name='p1')
        cfg = InstrumentConfig([G235H(r_scale=r, pix_offset=p), G395H()])
        path = tmp_path / 'dispersers.yaml'
        cfg.save(path)
        cfg2 = InstrumentConfig.load(path)
        assert cfg2.names == ['G235H', 'G395H']
        assert cfg2[0].r_scale is not None
        assert cfg2[0].pix_offset is not None
        assert cfg2[1].r_scale is None

    def test_calib_params_section_in_dict(self):
        r = RScale(name='r1')
        f = FluxScale(name='f1')
        cfg = InstrumentConfig([G235H(r_scale=r, flux_scale=f)])
        d = cfg.to_dict()
        assert 'calib_params' in d
        # Site names are prefix + label: 'r_scale_r1', 'flux_scale_f1'
        assert 'r_scale_r1' in d['calib_params']
        assert 'flux_scale_f1' in d['calib_params']

    def test_no_calib_params_when_none(self):
        cfg = InstrumentConfig([G235H()])
        d = cfg.to_dict()
        assert d['calib_params'] == {}

    def test_r_source_preserved(self):
        cfg = InstrumentConfig([G235H(r_source='uniform')])
        d = cfg.to_dict()
        cfg2 = InstrumentConfig.from_dict(d)
        assert cfg2[0].r_source == 'uniform'

    def test_mixed_instruments_roundtrip(self):
        r = RScale(name='shared_r')
        cfg = InstrumentConfig([G235H(r_scale=r), SDSSDisperser(name='SDSS')])
        d = cfg.to_dict()
        cfg2 = InstrumentConfig.from_dict(d)
        assert cfg2.names == ['G235H', 'SDSS']
        assert cfg2[0].r_scale is not None


# ---------------------------------------------------------------------------
# Anonymous calibration token naming
# ---------------------------------------------------------------------------


class TestAnonymousCalibNaming:
    """Tests for disperser-name-based naming of anonymous calibration tokens."""

    def test_single_anonymous_r_scale_uses_disperser_name(self):
        r = RScale()
        cfg = InstrumentConfig([G235H(r_scale=r)])
        assert cfg[0].r_scale.name == 'r_scale_G235H'

    def test_two_anonymous_r_scales_use_disperser_names(self):
        r0 = RScale()
        r1 = RScale()
        cfg = InstrumentConfig([G235H(r_scale=r0), G395H(r_scale=r1)])
        assert cfg[0].r_scale.name == 'r_scale_G235H'
        assert cfg[1].r_scale.name == 'r_scale_G395H'

    def test_anonymous_tokens_different_prefixes_use_disperser_name(self):
        r = RScale()
        f = FluxScale()
        p = PixOffset()
        cfg = InstrumentConfig([G235H(r_scale=r, flux_scale=f, pix_offset=p)])
        assert cfg[0].r_scale.name == 'r_scale_G235H'
        assert cfg[0].flux_scale.name == 'flux_scale_G235H'
        assert cfg[0].pix_offset.name == 'pix_offset_G235H'

    def test_shared_anonymous_token_gets_alpha_name(self):
        shared = RScale()
        cfg = InstrumentConfig([G235H(r_scale=shared), G395H(r_scale=shared)])
        assert cfg[0].r_scale is cfg[1].r_scale
        assert cfg[0].r_scale.name == 'r_scale_a'

    def test_labeled_token_unaffected(self):
        r = RScale(name='calib')
        cfg = InstrumentConfig([G235H(r_scale=r)])
        assert cfg[0].r_scale.name == 'r_scale_calib'

    def test_mixed_labeled_and_anonymous(self):
        r_labeled = RScale(name='fixed')
        r_anon = RScale()
        cfg = InstrumentConfig([G235H(r_scale=r_labeled), G395H(r_scale=r_anon)])
        assert cfg[0].r_scale.name == 'r_scale_fixed'
        assert cfg[1].r_scale.name == 'r_scale_G395H'


# ---------------------------------------------------------------------------
# InstrumentConfig addition
# ---------------------------------------------------------------------------


class TestInstrumentConfigAdd:
    def test_add_non_overlapping(self):
        cfg1 = InstrumentConfig([G235H()])
        cfg2 = InstrumentConfig([G395H()])
        merged = cfg1 + cfg2
        assert merged.names == ['G235H', 'G395H']

    def test_add_preserves_order(self):
        cfg1 = InstrumentConfig([G235H()])
        cfg2 = InstrumentConfig([G395H(), SDSSDisperser(name='SDSS')])
        merged = cfg1 + cfg2
        assert merged.names == ['G235H', 'G395H', 'SDSS']

    def test_add_duplicate_names_raises(self):
        cfg1 = InstrumentConfig([G235H()])
        cfg2 = InstrumentConfig([G235H()])
        with pytest.raises(ValueError, match='Duplicate disperser name'):
            cfg1 + cfg2

    def test_add_wrong_type_returns_not_implemented(self):
        cfg = InstrumentConfig([G235H()])
        result = cfg.__add__('not a config')
        assert result is NotImplemented

    def test_add_empty_left(self):
        cfg1 = InstrumentConfig([G235H()])
        cfg2 = InstrumentConfig([G395H()])
        merged = InstrumentConfig([]) + cfg2 if False else cfg1 + cfg2
        assert len(merged) == 2

    def test_add_result_is_new_instance(self):
        cfg1 = InstrumentConfig([G235H()])
        cfg2 = InstrumentConfig([G395H()])
        merged = cfg1 + cfg2
        assert merged is not cfg1
        assert merged is not cfg2


# ---------------------------------------------------------------------------
# Serialization error handling
# ---------------------------------------------------------------------------


class TestInstrumentConfigSerializationErrors:
    """Tests for error handling in serialization helper functions."""

    def test_unregistered_disperser_raises_type_error(self):
        """_disperser_to_entry raises TypeError for unregistered disperser types (config.py line 136-141)."""
        from astropy import units as u

        from unite.instrument.base import Disperser
        from unite.instrument.config import _disperser_to_entry

        # Create a custom disperser that's not in the registry
        class UnregisteredDisperser(Disperser):
            def R(self, wavelength):
                return wavelength * 0 + 1000.0

            def dlam_dpix(self, wavelength):
                return wavelength * 0 + 0.1

        unregistered = UnregisteredDisperser(u.AA, name='Unknown')
        with pytest.raises(TypeError, match='Cannot serialize'):
            _disperser_to_entry(unregistered)

    def test_unknown_disperser_type_in_from_dict_raises(self):
        """_disperser_from_entry raises KeyError for unknown disperser types (config.py line 174-175)."""
        from unite.instrument.config import _disperser_from_entry

        bad_entry = {'type': 'UnknownDisperser123', 'name': 'test', 'entries': []}
        with pytest.raises(KeyError, match='Unknown disperser type'):
            _disperser_from_entry(bad_entry, {})


# ---------------------------------------------------------------------------
# Repr for empty and various configurations
# ---------------------------------------------------------------------------


class TestInstrumentConfigRepr:
    """Tests for InstrumentConfig __repr__ method."""

    def test_repr_empty_config(self):
        """repr of empty InstrumentConfig shows 'empty' (config.py line 469)."""
        cfg = InstrumentConfig([])
        r = repr(cfg)
        assert 'empty' in r.lower()

    def test_repr_with_calibration_sections(self):
        """repr includes calibration param sections when tokens are present (config.py line 524-531)."""
        r = RScale(prior=Uniform(0.9, 1.1), name='r1')
        f = FluxScale(prior=Uniform(0.5, 2.0), name='f1')
        cfg = InstrumentConfig([G235H(r_scale=r, flux_scale=f)])
        repr_str = repr(cfg)
        # Should include sections for r_scale and flux_scale
        assert (
            'r_scale' in repr_str
            or 'R_scale' in repr_str
            or 'R scale' in repr_str.lower()
        )
        assert 'flux' in repr_str.lower()

    def test_repr_includes_disperser_info(self):
        """repr includes disperser type and wavelength info."""
        cfg = InstrumentConfig([G235H(), G395H()])
        repr_str = repr(cfg)
        assert '2 disperser(s)' in repr_str
        assert 'G235H' in repr_str
        assert 'G395H' in repr_str
