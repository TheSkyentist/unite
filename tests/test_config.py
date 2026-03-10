"""Tests for top-level Configuration container."""

import warnings

from astropy import units as u

from unite.config import Configuration
from unite.continuum import ContinuumConfiguration, Linear
from unite.instrument.base import FluxScale, RScale
from unite.instrument.config import InstrumentConfig
from unite.instrument.nirspec import G235H, G395H
from unite.line import FWHM, LineConfiguration, Redshift
from unite.prior import TruncatedNormal, Uniform


def _make_line_config():
    """Create a minimal line config."""
    lc = LineConfiguration()
    z = Redshift(prior=Uniform(-0.01, 0.01))
    w = FWHM(prior=Uniform(1.0, 10.0))
    lc.add_line('Ha', 6564.61 * u.AA, redshift=z, fwhm_gauss=w)
    lc.add_line('NII', 6585.27 * u.AA, redshift=z, fwhm_gauss=w)
    return lc


def _make_continuum_config():
    """Create a simple continuum config."""
    return ContinuumConfiguration.from_lines(
        [6564.61, 6585.27] * u.AA, width=30_000 * u.km / u.s, form=Linear()
    )


def _make_dispersers_config():
    """Create a dispersers config with shared token."""
    r = RScale(prior=TruncatedNormal(1.0, 0.05, 0.8, 1.2), name='r_shared')
    return InstrumentConfig([G235H(r_scale=r), G395H(r_scale=r)])


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConfigurationConstruction:
    """Tests for Configuration construction."""

    def test_lines_only(self):
        lc = _make_line_config()
        cfg = Configuration(lc)
        assert cfg.lines is lc
        assert cfg.continuum is None
        assert cfg.dispersers is None

    def test_with_continuum(self):
        lc = _make_line_config()
        cc = _make_continuum_config()
        cfg = Configuration(lc, cc)
        assert cfg.continuum is cc

    def test_with_dispersers(self):
        lc = _make_line_config()
        dc = _make_dispersers_config()
        cfg = Configuration(lc, dispersers=dc)
        assert cfg.dispersers is dc

    def test_dispersers_validate_called(self):
        """Configuration should call validate() on dispersers."""
        f1 = FluxScale()
        f2 = FluxScale()
        dc = InstrumentConfig([G235H(flux_scale=f1), G395H(flux_scale=f2)])
        lc = _make_line_config()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Configuration(lc, dispersers=dc)
            msgs = [str(x.message) for x in w]
            assert any('flux_scale=None' in m for m in msgs)

    def test_repr(self):
        lc = _make_line_config()
        cc = _make_continuum_config()
        cfg = Configuration(lc, cc)
        r = repr(cfg)
        assert 'Configuration' in r
        assert 'lines=' in r


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestConfigurationSerialization:
    """Tests for Configuration dict/YAML/file serialization."""

    def test_dict_roundtrip_lines_only(self):
        lc = _make_line_config()
        cfg = Configuration(lc)
        d = cfg.to_dict()
        cfg2 = Configuration.from_dict(d)
        assert len(cfg2.lines) == 2
        assert cfg2.continuum is None
        assert cfg2.dispersers is None

    def test_dict_roundtrip_full(self):
        lc = _make_line_config()
        cc = _make_continuum_config()
        dc = _make_dispersers_config()
        cfg = Configuration(lc, cc, dispersers=dc)
        d = cfg.to_dict()
        cfg2 = Configuration.from_dict(d)
        assert len(cfg2.lines) == 2
        assert cfg2.continuum is not None
        assert cfg2.dispersers is not None
        assert cfg2.dispersers.names == ['G235H', 'G395H']

    def test_yaml_roundtrip(self):
        lc = _make_line_config()
        cc = _make_continuum_config()
        cfg = Configuration(lc, cc)
        yaml_str = cfg.to_yaml()
        cfg2 = Configuration.from_yaml(yaml_str)
        assert len(cfg2.lines) == 2
        assert cfg2.continuum is not None

    def test_file_roundtrip(self, tmp_path):
        lc = _make_line_config()
        cc = _make_continuum_config()
        dc = _make_dispersers_config()
        cfg = Configuration(lc, cc, dispersers=dc)
        path = tmp_path / 'config.yaml'
        cfg.save(path)
        cfg2 = Configuration.load(path)
        assert len(cfg2.lines) == 2
        assert cfg2.continuum is not None
        assert cfg2.dispersers is not None

    def test_shared_tokens_preserved(self):
        """Line tokens should be shared after round-trip."""
        lc = _make_line_config()
        cfg = Configuration(lc)
        d = cfg.to_dict()
        cfg2 = Configuration.from_dict(d)
        # Both lines share the same redshift — verify via to_dict
        d2 = cfg2.lines.to_dict()
        # Both lines should reference the same redshift name
        z_names = [line['redshift'] for line in d2['lines']]
        assert z_names[0] == z_names[1]

    def test_disperser_shared_tokens_preserved(self):
        """Disperser calibration tokens should be shared after round-trip."""
        lc = _make_line_config()
        dc = _make_dispersers_config()
        cfg = Configuration(lc, dispersers=dc)
        d = cfg.to_dict()
        cfg2 = Configuration.from_dict(d)
        assert cfg2.dispersers[0].r_scale is cfg2.dispersers[1].r_scale
