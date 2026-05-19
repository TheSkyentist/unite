"""Tests for Template continuum form."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from unite.continuum.config import ContinuumConfiguration, ContinuumRegion, Scale
from unite.continuum.library import Template
from unite.prior import Fixed, Uniform

# ---------------------------------------------------------------------------
# Fixtures: write minimal FITS/ECSV tables to tmp_path
# ---------------------------------------------------------------------------


_FLAM = u.Unit('erg s-1 cm-2 AA-1')


def _make_table(
    lam: np.ndarray,
    flux_dict: dict[str, np.ndarray],
    *,
    lam_unit: u.UnitBase = u.um,
    flux_unit: u.UnitBase = _FLAM,
    no_flux_unit: bool = False,
) -> Table:
    cols = {'wavelength': u.Quantity(lam, lam_unit)}
    for name, arr in flux_dict.items():
        cols[name] = u.Quantity(arr, flux_unit) if not no_flux_unit else arr
    return Table(cols)


@pytest.fixture
def single_template_file(tmp_path: Path) -> Path:
    lam = np.linspace(0.9, 2.5, 200)
    flux = 1.0 + 0.5 * (lam - 1.5)
    t = _make_table(lam, {'qso': flux})
    p = tmp_path / 'single.ecsv'
    t.write(p, overwrite=True)
    return p


@pytest.fixture
def multi_template_file(tmp_path: Path) -> Path:
    lam = np.linspace(0.9, 2.5, 200)
    t = _make_table(
        lam,
        {
            'age_1gyr': 1.0 + 0.1 * lam,
            'age_3gyr': 0.8 + 0.2 * lam,
            'age_10gyr': 0.5 + 0.3 * lam,
        },
    )
    p = tmp_path / 'multi.ecsv'
    t.write(p, overwrite=True)
    return p


@pytest.fixture
def no_unit_file(tmp_path: Path) -> Path:
    lam = np.linspace(0.9, 2.5, 200)
    flux = 1.0 + 0.5 * lam
    t = _make_table(lam, {'flux': flux}, no_flux_unit=True)
    p = tmp_path / 'no_unit.ecsv'
    t.write(p, overwrite=True)
    return p


@pytest.fixture
def angstrom_file(tmp_path: Path) -> Path:
    lam = np.linspace(9000.0, 25000.0, 200)
    flux = 1.0 + 0.5 * (lam - 15000) / 8000
    t = _make_table(lam, {'flux': flux}, lam_unit=u.AA)
    p = tmp_path / 'angstrom.ecsv'
    t.write(p, overwrite=True)
    return p


# ---------------------------------------------------------------------------
# Construction-time validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_single_column_inferred(self, single_template_file):
        tpl = Template(single_template_file)
        assert tpl._flux_cols == ['qso']
        assert tpl.param_names() == ('qso_scale', 'norm_wav')

    def test_multi_column(self, multi_template_file):
        tpl = Template(multi_template_file)
        assert tpl._flux_cols == ['age_1gyr', 'age_3gyr', 'age_10gyr']
        assert tpl.param_names() == (
            'age_1gyr_scale',
            'age_3gyr_scale',
            'age_10gyr_scale',
            'norm_wav',
        )

    def test_usecols_subset(self, multi_template_file):
        tpl = Template(multi_template_file, usecols=['age_1gyr', 'age_10gyr'])
        assert tpl._flux_cols == ['age_1gyr', 'age_10gyr']
        assert tpl.param_names() == ('age_1gyr_scale', 'age_10gyr_scale', 'norm_wav')

    def test_wavelength_colname_override(self, tmp_path):
        lam = np.linspace(0.9, 2.5, 100)
        t = Table(
            {
                'lam': u.Quantity(lam, u.um),
                'flux': u.Quantity(np.ones(100), u.Unit('erg s-1 cm-2 AA-1')),
            }
        )
        p = tmp_path / 'custom_wl.ecsv'
        t.write(p, overwrite=True)
        tpl = Template(p, wavelength_colname='lam')
        assert tpl._flux_cols == ['flux']

    def test_missing_wavelength_colname_raises(self, tmp_path):
        lam = np.linspace(0.9, 2.5, 100)
        t = Table({'lam': lam, 'flux': np.ones(100)})  # no units on either column
        p = tmp_path / 'no_wl_unit.ecsv'
        t.write(p, overwrite=True)
        with pytest.raises(ValueError, match='No column with a length unit'):
            Template(p)

    def test_ambiguous_wavelength_raises(self, tmp_path):
        lam = np.linspace(0.9, 2.5, 100)
        t = Table(
            {
                'wave1': u.Quantity(lam, u.um),
                'wave2': u.Quantity(lam * 1e4, u.AA),
                'flux': np.ones(100),
            }
        )
        p = tmp_path / 'ambiguous.ecsv'
        t.write(p, overwrite=True)
        with pytest.raises(ValueError, match='Multiple length-unit columns'):
            Template(p)

    def test_explicit_wavelength_colname_missing_raises(self, single_template_file):
        with pytest.raises(ValueError, match='not found'):
            Template(single_template_file, wavelength_colname='nonexistent')

    def test_usecols_missing_column_raises(self, multi_template_file):
        with pytest.raises(ValueError, match='usecols columns not found'):
            Template(multi_template_file, usecols=['age_1gyr', 'missing_col'])

    def test_non_monotonic_wavelength_raises(self, tmp_path):
        lam = np.array([0.9, 1.5, 1.2, 2.0])
        t = _make_table(lam, {'flux': np.ones(4)})
        p = tmp_path / 'non_mono.ecsv'
        t.write(p, overwrite=True)
        with pytest.raises(ValueError, match='strictly monotonically increasing'):
            Template(p)

    def test_nan_in_flux_raises(self, tmp_path):
        lam = np.linspace(0.9, 2.5, 100)
        flux = np.ones(100)
        flux[50] = np.nan
        t = _make_table(lam, {'flux': flux})
        p = tmp_path / 'nan_flux.ecsv'
        t.write(p, overwrite=True)
        with pytest.raises(ValueError, match='NaN or inf'):
            Template(p)

    def test_no_flux_unit_warns(self, no_unit_file):
        with pytest.warns(UserWarning, match='no units'):
            Template(no_unit_file)

    def test_repr(self, single_template_file):
        tpl = Template(single_template_file)
        assert 'single.ecsv' in repr(tpl)
        assert 'qso' in repr(tpl)


# ---------------------------------------------------------------------------
# _prepare() validation
# ---------------------------------------------------------------------------


class TestPrepare:
    def test_coverage_check_low(self, tmp_path):
        lam = np.linspace(1.0, 2.5, 100)  # starts at 1.0, not 0.9
        t = _make_table(lam, {'flux': np.ones(100)})
        p = tmp_path / 'short_low.ecsv'
        t.write(p, overwrite=True)
        with pytest.raises(ValueError, match='does not cover the full region'):
            ContinuumRegion(0.9 * u.um, 2.0 * u.um, form=Template(p))

    def test_coverage_check_high(self, tmp_path):
        lam = np.linspace(0.9, 1.8, 100)  # ends at 1.8, not 2.5
        t = _make_table(lam, {'flux': np.ones(100)})
        p = tmp_path / 'short_high.ecsv'
        t.write(p, overwrite=True)
        with pytest.raises(ValueError, match='does not cover the full region'):
            ContinuumRegion(0.9 * u.um, 2.5 * u.um, form=Template(p))

    def test_prepare_sets_internal_arrays(self, single_template_file):
        tpl = Template(single_template_file)
        ContinuumRegion(1.0 * u.um, 2.0 * u.um, form=tpl)
        assert tpl._lam_um is not None
        assert tpl._flam_eval is not None
        assert tpl._flam_eval.shape == (1, 200)

    def test_prepare_multi_column_shape(self, multi_template_file):
        tpl = Template(multi_template_file)
        ContinuumRegion(1.0 * u.um, 2.0 * u.um, form=tpl)
        assert tpl._flam_eval.shape == (3, 200)


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


class TestEvaluate:
    def _make_and_prepare(self, path, low=1.0, high=2.0):
        tpl = Template(path)
        ContinuumRegion(low * u.um, high * u.um, form=tpl, name='tpl')
        obs_low = low
        obs_high = high
        obs_center = (low + high) / 2.0
        return tpl, obs_low, obs_high, obs_center

    def test_equals_scale_at_norm_wav(self, single_template_file):
        tpl, obs_low, obs_high, obs_center = self._make_and_prepare(
            single_template_file
        )
        norm_wav = obs_center
        params = {'qso_scale': jnp.array(2.0), 'norm_wav': jnp.array(norm_wav)}
        result = tpl.evaluate(
            jnp.array([norm_wav]), obs_center, params, obs_low, obs_high
        )
        np.testing.assert_allclose(float(result[0]), 2.0, rtol=1e-5)

    def test_z_sys_shifts_evaluation(self, single_template_file):
        tpl, _, _, _ = self._make_and_prepare(single_template_file, low=1.0, high=2.0)
        z = 0.5
        obs_low = 1.0 * (1 + z)
        obs_high = 2.0 * (1 + z)
        obs_center = (obs_low + obs_high) / 2.0
        norm_wav_obs = 1.5 * (1 + z)

        params_z = {'qso_scale': jnp.array(1.0), 'norm_wav': jnp.array(norm_wav_obs)}
        params_0 = {'qso_scale': jnp.array(1.0), 'norm_wav': jnp.array(1.5)}

        wl_obs = jnp.linspace(obs_low, obs_high, 50)
        wl_rest = jnp.linspace(1.0, 2.0, 50)

        tpl0, obs_low0, obs_high0, obs_center0 = self._make_and_prepare(
            single_template_file
        )

        result_z = tpl.evaluate(
            wl_obs, obs_center, params_z, obs_low, obs_high, z_sys=z
        )
        result_0 = tpl0.evaluate(
            wl_rest, obs_center0, params_0, obs_low0, obs_high0, z_sys=0.0
        )
        np.testing.assert_allclose(result_z, result_0, rtol=1e-5)

    def test_multi_column_sum(self, multi_template_file):
        tpl, obs_low, obs_high, obs_center = self._make_and_prepare(multi_template_file)
        norm_wav = obs_center
        params = {
            'age_1gyr_scale': jnp.array(1.0),
            'age_3gyr_scale': jnp.array(0.0),
            'age_10gyr_scale': jnp.array(0.0),
            'norm_wav': jnp.array(norm_wav),
        }
        wl = jnp.array([norm_wav])
        result = tpl.evaluate(wl, obs_center, params, obs_low, obs_high)
        np.testing.assert_allclose(float(result[0]), 1.0, rtol=1e-5)

    def test_angstrom_wavelength_column(self, angstrom_file):
        tpl = Template(angstrom_file)
        ContinuumRegion(0.9 * u.um, 2.5 * u.um, form=tpl, name='tpl')
        norm_wav = 1.5
        params = {'flux_scale': jnp.array(3.0), 'norm_wav': jnp.array(norm_wav)}
        result = tpl.evaluate(jnp.array([norm_wav]), 1.5, params, 0.9, 2.5)
        np.testing.assert_allclose(float(result[0]), 3.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# integrate() — base-class midpoint rule
# ---------------------------------------------------------------------------


class TestIntegrate:
    def test_integrate_consistent_with_evaluate(self, single_template_file):
        tpl = Template(single_template_file)
        ContinuumRegion(1.0 * u.um, 2.0 * u.um, form=tpl, name='tpl')
        edges = jnp.linspace(1.0, 2.0, 101)
        mids = 0.5 * (edges[1:] + edges[:-1])
        params = {'qso_scale': jnp.array(1.0), 'norm_wav': jnp.array(1.5)}
        cum = tpl.integrate(edges, 1.5, params, 1.0, 2.0)
        widths = jnp.diff(edges)
        pixel_avg_from_integrate = jnp.diff(cum) / widths
        pixel_avg_from_evaluate = tpl.evaluate(mids, 1.5, params, 1.0, 2.0)
        np.testing.assert_allclose(
            pixel_avg_from_integrate, pixel_avg_from_evaluate, rtol=1e-4
        )


# ---------------------------------------------------------------------------
# param_names / default_priors / param_units
# ---------------------------------------------------------------------------


class TestParams:
    def test_param_names_single(self, single_template_file):
        tpl = Template(single_template_file)
        assert 'qso_scale' in tpl.param_names()
        assert 'norm_wav' in tpl.param_names()

    def test_default_priors_single(self, single_template_file):
        tpl = Template(single_template_file)
        priors = tpl.default_priors(region_center=1.5)
        assert isinstance(priors['qso_scale'], Uniform)
        assert isinstance(priors['norm_wav'], Fixed)
        assert float(priors['norm_wav'].value) == 1.5

    def test_param_units(self, single_template_file):
        tpl = Template(single_template_file)
        pu = tpl.param_units(u.Unit('erg s-1 cm-2 AA-1'), u.um)
        apply_cs, unit = pu['qso_scale']
        assert apply_cs is True
        assert unit == u.Unit('erg s-1 cm-2 AA-1')
        apply_cs_nw, _ = pu['norm_wav']
        assert apply_cs_nw is False


# ---------------------------------------------------------------------------
# Integration with ContinuumConfiguration / parameter naming
# ---------------------------------------------------------------------------


class TestContinuumRegionIntegration:
    def test_auto_param_naming(self, single_template_file):
        tpl = Template(single_template_file)
        cc = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, form=tpl, name='qso_region')]
        )
        param_names = [tok.name for d in cc.resolved_params for tok in d.values()]
        assert any('qso_scale' in n for n in param_names)
        assert any('norm_wav' in n for n in param_names)

    def test_scale_override_with_scale_token(self, single_template_file):
        tpl = Template(single_template_file)
        custom = Scale('my_qso', prior=Uniform(0, 10))
        region = ContinuumRegion(
            1.0 * u.um, 2.0 * u.um, form=tpl, params={'qso_scale': custom}
        )
        cc = ContinuumConfiguration([region])
        param_names = [tok.name for d in cc.resolved_params for tok in d.values()]
        # Naming convention: {param_slot}_{label}, so Scale('my_qso') → 'qso_scale_my_qso'
        assert 'qso_scale_my_qso' in param_names

    def test_multi_template_independent_scales(self, multi_template_file):
        tpl = Template(multi_template_file, usecols=['age_1gyr', 'age_3gyr'])
        cc = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, form=tpl, name='stellar')]
        )
        param_names = [tok.name for d in cc.resolved_params for tok in d.values()]
        assert any('age_1gyr_scale' in n for n in param_names)
        assert any('age_3gyr_scale' in n for n in param_names)


# ---------------------------------------------------------------------------
# Serialization (to_dict / from_dict)
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_roundtrip_single(self, single_template_file):
        tpl = Template(single_template_file)
        d = tpl.to_dict()
        assert d['type'] == 'Template'
        assert 'path' in d
        tpl2 = Template.from_dict(d)
        assert tpl2._flux_cols == tpl._flux_cols
        assert tpl2._path == tpl._path

    def test_roundtrip_with_usecols(self, multi_template_file):
        tpl = Template(multi_template_file, usecols=['age_1gyr', 'age_3gyr'])
        d = tpl.to_dict()
        assert d['usecols'] == ['age_1gyr', 'age_3gyr']
        tpl2 = Template.from_dict(d)
        assert tpl2._flux_cols == ['age_1gyr', 'age_3gyr']

    def test_roundtrip_with_wavelength_colname(self, tmp_path):
        lam = np.linspace(0.9, 2.5, 100)
        t = Table(
            {
                'lam': u.Quantity(lam, u.um),
                'flux': u.Quantity(np.ones(100), u.Unit('erg s-1 cm-2 AA-1')),
            }
        )
        p = tmp_path / 'custom.ecsv'
        t.write(p, overwrite=True)
        tpl = Template(p, wavelength_colname='lam')
        d = tpl.to_dict()
        assert d['wavelength_colname'] == 'lam'
        tpl2 = Template.from_dict(d)
        assert tpl2._wavelength_colname == 'lam'

    def test_no_usecols_key_when_none(self, single_template_file):
        tpl = Template(single_template_file)
        d = tpl.to_dict()
        assert 'usecols' not in d

    def test_eq_and_hash(self, single_template_file, multi_template_file):
        t1 = Template(single_template_file)
        t2 = Template(single_template_file)
        t3 = Template(multi_template_file)
        assert t1 == t2
        assert t1 != t3
        assert hash(t1) == hash(t2)
