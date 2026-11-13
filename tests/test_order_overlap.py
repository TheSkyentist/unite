"""Tests for higher-order grating overlap (``OrderOverlap`` / ``OrderScale``)."""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table
from numpyro import handlers
from numpyro.infer.util import log_density

from unite import _overlap, model, prior
from unite.compute import evaluate_model
from unite.config import Configuration
from unite.continuum import ContinuumConfiguration, ContinuumRegion, PowerLaw
from unite.continuum.config import ContShape, NormWavelength, Scale
from unite.instrument import (
    FluxScale,
    InstrumentConfig,
    OrderOverlap,
    OrderScale,
    PixOffset,
    RScale,
    nirspec,
)
from unite.instrument.desi import DESIDisperser
from unite.instrument.generic import GenericDisperser
from unite.instrument.sdss import SDSSDisperser
from unite.line import FWHM, Flux, LineConfiguration, Redshift
from unite.results import make_parameter_table, make_spectra_tables
from unite.spectrum import Spectra, from_arrays

erf = np.vectorize(math.erf)
C_KMS = 299792.458
FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

# ---------------------------------------------------------------------------
# Toy G140M/F070LP-like scenario
# ---------------------------------------------------------------------------

Z_SYS = 0.22
LINE_REST_AA = 6563.0
LINE_FLUX = 0.02  # in flux_unit * um
LINE_FWHM_KMS = 400.0
PIX = 0.00064  # um, ~G140M dispersion
FLUX_UNIT = u.Unit('1e-20 erg s-1 cm-2 AA-1')
# Rest-frame continuum regions (Angstrom); one power law across both, so the
# regions join smoothly: F = (lam_obs / 1 um) ** CONT_BETA.
REGIONS = {'opt': (3646.0, 8204.0), 'nir': (8204.0, 20000.0)}
CONT_BETA = -1.5
R2_AMP = 0.1  # 10% second-order copy
TABLE_WL = np.linspace(0.6, 3.6, 3001)  # um, step 1 nm (contains 1.3 and 1.5)


def _r2(w):
    """Smooth second-order ratio: 0 below 1.3 um, 0.1 above 1.5 um."""
    return R2_AMP * np.clip((np.asarray(w) - 1.3) / 0.2, 0.0, 1.0)


def _r2_sharp(w):
    """Second-order ratio with a sharp (one table step) edge at 1.3 um."""
    return np.where(np.asarray(w) > 1.3005, R2_AMP, 0.0)


def _write_ratio(path, ratio_fn=_r2, *, wl=TABLE_WL, unit=u.um, colname=None):
    """Write an ECSV ratio table; return its path."""
    cols = {'wavelength': wl * unit, colname or 'throughput_ratio': ratio_fn(wl)}
    Table(cols).write(path, overwrite=True)
    return path


@pytest.fixture(scope='module')
def ratio_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp('overlap')
    _write_ratio(d / 'r2.ecsv')
    _write_ratio(d / 'r2_sharp.ecsv', _r2_sharp)
    _write_ratio(d / 'zero.ecsv', lambda w: np.zeros_like(w))
    _write_ratio(d / 'far.ecsv', lambda w: np.where(w > 3.0, 0.1, 0.0))
    _write_ratio(d / 'r3.ecsv', lambda w: np.where(w > 1.6, 0.02, 0.0))
    return d


def _region_params(lo, hi):
    """Power-law ``(scale, beta, norm_wav)`` of region [lo, hi] (Angstrom)."""
    nw_rest = 0.5 * (lo + hi)
    nw_obs = nw_rest * 1e-4 * (1 + Z_SYS)
    return nw_obs**CONT_BETA, CONT_BETA, nw_rest


def _pixel_gaussian(low, high, center, sigma, area):
    """Pixel-averaged Gaussian of integrated flux *area* over [low, high]."""
    cdf = lambda x: 0.5 * (1.0 + erf((x - center) / (math.sqrt(2.0) * sigma)))  # noqa: E731
    return area * (cdf(high) - cdf(low)) / (high - low)


def _continuum(wl_obs, regions=REGIONS):
    """Observed-frame f_lambda continuum of the toy source (um)."""
    out = np.zeros_like(wl_obs)
    for lo, hi in regions.values():
        lo_obs, hi_obs = lo * 1e-4 * (1 + Z_SYS), hi * 1e-4 * (1 + Z_SYS)
        scale, beta, nw_rest = _region_params(lo, hi)
        nw = nw_rest * 1e-4 * (1 + Z_SYS)
        inside = (wl_obs >= lo_obs) & (wl_obs <= hi_obs)
        out = np.where(inside, scale * (wl_obs / nw) ** beta, out)
    return out


def _make_toy(
    disperser,
    *,
    ratio_fn=_r2,
    order_scale=1.0,
    pix_shift=0.0,
    wl_range=(0.70, 1.80),
    name='g140m',
):
    """Toy spectrum: continuum + line + order_scale * second-order image.

    The second-order image of light of wavelength ``lam1`` lands at
    ``2 * lam1`` with ``R_2(lam1) = 2 R(lam1)``: in observed wavelength its
    LSF FWHM is the first-order LSF of the photon wavelength.  With
    *pix_shift*, the data are generated on the true wavelengths
    ``edge - pix_shift * dlam_dpix(edge)`` of the nominal pixel edges.
    """
    low = np.arange(wl_range[0], wl_range[1], PIX)
    high = low + PIX
    dl = lambda w: np.asarray(disperser.dlam_dpix(jnp.asarray(w)))  # noqa: E731
    lo_t, hi_t = low - pix_shift * dl(low), high - pix_shift * dl(high)
    wl = 0.5 * (lo_t + hi_t)
    lam_c = LINE_REST_AA * 1e-4 * (1 + Z_SYS)

    def _line_image(order):
        c = order * lam_c
        lsf = lam_c / float(disperser.R(lam_c))  # first-order LSF of the photons
        sigma = math.hypot(c * LINE_FWHM_KMS / C_KMS, lsf) * FWHM_TO_SIGMA
        # F(lam / order) integrates to order * LINE_FLUX over observed lam.
        return _pixel_gaussian(lo_t, hi_t, c, sigma, order * LINE_FLUX)

    flux = _continuum(wl) + _line_image(1)
    flux = flux + order_scale * ratio_fn(wl) * (_continuum(wl / 2) + _line_image(2))
    err = np.full_like(flux, 0.01)
    return from_arrays(
        low=low * u.um,
        high=high * u.um,
        flux=flux * FLUX_UNIT,
        error=err * FLUX_UNIT,
        disperser=disperser,
        name=name,
    )


def _line_config(free=False):
    lc = LineConfiguration()
    if free:
        lc.add_line(
            'Ha',
            LINE_REST_AA * u.AA,
            flux=Flux('Ha', prior=prior.Uniform(0, 1)),
            fwhm_gauss=FWHM('c', prior=prior.Uniform(100, 1000)),
            redshift=Redshift('c', prior=prior.Uniform(-0.002, 0.002)),
        )
    else:
        lc.add_line(
            'Ha',
            LINE_REST_AA * u.AA,
            flux=Flux('Ha', prior=prior.Fixed(LINE_FLUX)),
            fwhm_gauss=FWHM('c', prior=prior.Fixed(LINE_FWHM_KMS)),
            redshift=Redshift('c', prior=prior.Fixed(0.0)),
        )
    return lc


def _cont_config(regions=REGIONS):
    out = []
    for name, (lo, hi) in regions.items():
        scale, beta, nw_rest = _region_params(lo, hi)
        out.append(
            ContinuumRegion(
                lo * u.AA,
                hi * u.AA,
                form=PowerLaw(),
                name=name,
                params={
                    'scale': Scale(prior=prior.Fixed(scale)),
                    'beta': ContShape(prior=prior.Fixed(beta)),
                    'norm_wav': NormWavelength(prior=prior.Fixed(nw_rest)),
                },
            )
        )
    return ContinuumConfiguration(out)


def _build(
    disperser, *, integration_mode='analytic', regions=REGIONS, spectra_list=None, **toy
):
    """Build the toy model with every source parameter fixed at the truth."""
    specs = spectra_list or [_make_toy(disperser, **toy)]
    spectra = Spectra(specs, redshift=Z_SYS)
    spectra.line_scale = 1.0 * FLUX_UNIT * u.um
    spectra.continuum_scale = 1.0 * FLUX_UNIT
    spectra.prepare(_line_config(), _cont_config(regions), drop_empty_regions=False)
    builder = model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    )
    model_fn, args = builder.build(integration_mode=integration_mode)
    # build() trims spectra to the continuum regions; return the trimmed copy.
    return model_fn, args, args.spectra[0]


def _overlap_region(spec):
    return np.asarray(spec.wavelength) > 1.3  # _r2 is zero at and below 1.3 um


def _g140m(path, **kw):
    scale = kw.pop('scale', None)
    return nirspec.G140M(order_overlap=[OrderOverlap(2, path, scale=scale)], **kw)


# ---------------------------------------------------------------------------
# OrderOverlap
# ---------------------------------------------------------------------------


class TestOrderOverlap:
    def test_reads_default_column(self, ratio_dir):
        ov = OrderOverlap(2, ratio_dir / 'r2.ecsv')
        assert ov.order == 2
        assert ov.column == 'throughput_ratio'
        assert isinstance(ov.scale, OrderScale)
        assert isinstance(ov.scale.prior, prior.Fixed)
        assert ov.scale.prior.value == 1.0
        lo, hi = ov.support.to_value(u.um)
        assert 1.3 < lo < 1.302 and hi == pytest.approx(3.6)

    def test_evaluate_interpolates_and_converts_units(self, ratio_dir):
        ov = OrderOverlap(2, ratio_dir / 'r2.ecsv')
        w = np.array([1.0, 1.4, 1.6, 3.0])
        np.testing.assert_allclose(ov.evaluate(w, u.um), _r2(w), atol=1e-12)
        np.testing.assert_allclose(ov.evaluate(w * 1e4, u.AA), _r2(w), atol=1e-12)
        # Exactly zero outside the table.
        assert float(ov.evaluate(0.5, u.um)) == 0.0
        assert float(ov.evaluate(4.0, u.um)) == 0.0
        assert jax.jit(lambda x: ov.evaluate(x, u.um))(jnp.ones(3) * 2).shape == (3,)

    def test_support_bounds_contain_nonzero_values(self, ratio_dir):
        ov = OrderOverlap(2, ratio_dir / 'r2_sharp.ecsv')
        lo, hi = ov.support_bounds(u.um)
        w = np.linspace(0.6, 3.6, 30001)
        r = np.asarray(ov.evaluate(w, u.um))
        assert np.all(r[(w < lo) | (w > hi)] == 0.0)
        assert np.all(r[(w > lo + 1e-3) & (w < hi)] > 0.0)
        assert OrderOverlap(2, ratio_dir / 'zero.ecsv').support_bounds(u.um) is None
        assert OrderOverlap(2, ratio_dir / 'zero.ecsv').support is None

    def test_single_other_column_is_used(self, tmp_path):
        p = _write_ratio(tmp_path / 'x.ecsv', colname='r')
        assert OrderOverlap(2, p).column == 'r'

    def test_explicit_column(self, tmp_path):
        t = Table({'wl': TABLE_WL * u.um, 'a': _r2(TABLE_WL), 'b': 2 * _r2(TABLE_WL)})
        t.write(tmp_path / 'x.ecsv')
        ov = OrderOverlap(2, tmp_path / 'x.ecsv', column='b')
        np.testing.assert_allclose(ov.evaluate(2.0, u.um), 0.2)
        with pytest.raises(ValueError, match='ambiguous'):
            OrderOverlap(2, tmp_path / 'x.ecsv')
        with pytest.raises(ValueError, match='not found'):
            OrderOverlap(2, tmp_path / 'x.ecsv', column='c')

    def test_dimensionless_unit_is_accepted(self, tmp_path):
        t = Table({'wl': TABLE_WL * u.um, 'throughput_ratio': _r2(TABLE_WL) * 100})
        t['throughput_ratio'].unit = u.percent
        t.write(tmp_path / 'x.ecsv')
        ov = OrderOverlap(2, tmp_path / 'x.ecsv')
        np.testing.assert_allclose(ov.evaluate(2.0, u.um), 0.1)

    @pytest.mark.parametrize(
        ('table', 'match'),
        [
            ({'wl': TABLE_WL, 'throughput_ratio': _r2(TABLE_WL)}, 'wavelength'),
            (
                {'wl': TABLE_WL * u.um, 'w2': TABLE_WL * u.AA, 'r': _r2(TABLE_WL)},
                'Multiple',
            ),
            ({'wl': TABLE_WL * u.um}, 'ambiguous'),
            (
                {
                    'wl': TABLE_WL * u.um,
                    'throughput_ratio': _r2(TABLE_WL) * u.Unit('erg s-1 cm-2 AA-1'),
                },
                'dimensionless',
            ),
            ({'wl': TABLE_WL[::-1] * u.um, 'r': _r2(TABLE_WL)}, 'increasing'),
            ({'wl': TABLE_WL * u.um, 'r': np.full(TABLE_WL.size, np.nan)}, 'finite'),
        ],
    )
    def test_construction_errors(self, tmp_path, table, match):
        Table(table).write(tmp_path / 'bad.ecsv')
        with pytest.raises(ValueError, match=match):
            OrderOverlap(2, tmp_path / 'bad.ecsv')

    def test_order_and_scale_validation(self, ratio_dir):
        with pytest.raises(ValueError, match='>= 2'):
            OrderOverlap(1, ratio_dir / 'r2.ecsv')
        with pytest.raises(TypeError, match='integer'):
            OrderOverlap(2.0, ratio_dir / 'r2.ecsv')  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match='OrderScale'):
            OrderOverlap(2, ratio_dir / 'r2.ecsv', scale=FluxScale())  # ty: ignore[invalid-argument-type]

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            OrderOverlap(2, tmp_path / 'nope.ecsv')

    def test_eq_and_repr(self, ratio_dir):
        a = OrderOverlap(2, ratio_dir / 'r2.ecsv')
        b = OrderOverlap(2, ratio_dir / 'r2.ecsv', scale=OrderScale('x'))
        assert a == b  # scale does not take part
        assert a != OrderOverlap(3, ratio_dir / 'r2.ecsv')
        assert a != OrderOverlap(2, ratio_dir / 'r2_sharp.ecsv')
        assert 'order=2' in repr(a) and 'r2.ecsv' in repr(a)
        assert 'column' not in repr(a)


# ---------------------------------------------------------------------------
# Disperser attachment, naming and sharing
# ---------------------------------------------------------------------------


NIRSPEC_GRATINGS = [
    nirspec.G140M,
    nirspec.G140H,
    nirspec.G235M,
    nirspec.G235H,
    nirspec.G395M,
    nirspec.G395H,
]


class TestDisperserAttachment:
    def test_default_empty(self):
        assert nirspec.G140M().order_overlap == ()
        assert [s for s, _ in nirspec.G140M().calibration_tokens()] == [
            'r_scale',
            'flux_scale',
            'pix_offset',
        ]

    @pytest.mark.parametrize('cls', NIRSPEC_GRATINGS)
    def test_gratings_accept_and_sort(self, cls, ratio_dir):
        ov2 = OrderOverlap(2, ratio_dir / 'r2.ecsv')
        ov3 = OrderOverlap(3, ratio_dir / 'r3.ecsv')
        d = cls(order_overlap=[ov3, ov2])
        assert d.order_overlap == (ov2, ov3)
        slots = [s for s, _ in d.calibration_tokens()]
        assert slots[-2:] == ['order_scale_2', 'order_scale_3']

    def test_generic_and_nirspec_accept(self, ratio_dir):
        ov = OrderOverlap(2, ratio_dir / 'r2.ecsv')
        g = GenericDisperser(
            R_func=lambda w: jnp.full_like(w, 1000.0),
            dlam_dpix_func=lambda w: w / 1000.0,
            unit=u.um,
            order_overlap=[ov],
        )
        assert g.order_overlap == (ov,)
        assert nirspec.NIRSpec('g140m', order_overlap=[ov]).order_overlap == (ov,)

    @pytest.mark.parametrize(
        'make',
        [
            lambda ov: nirspec.PRISM(order_overlap=[ov]),
            lambda ov: nirspec.NIRSpec('prism', order_overlap=[ov]),
            lambda ov: DESIDisperser('B', order_overlap=[ov]),
            lambda ov: SDSSDisperser(order_overlap=[ov]),
        ],
    )
    def test_carve_outs_warn_but_accept(self, make, ratio_dir):
        ov = OrderOverlap(2, ratio_dir / 'r2.ecsv')
        with pytest.warns(UserWarning, match='does not normally have'):
            d = make(ov)
        assert d.order_overlap == (ov,)
        assert d.supports_order_overlap is False

    def test_carve_outs_silent_without_overlap(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            nirspec.PRISM()
            DESIDisperser('B')
            SDSSDisperser()

    def test_validation(self, ratio_dir):
        ov = OrderOverlap(2, ratio_dir / 'r2.ecsv')
        with pytest.raises(TypeError, match='sequence of OrderOverlap'):
            nirspec.G140M(order_overlap=ov)  # ty: ignore[invalid-argument-type]
        with pytest.raises(TypeError, match='entries must be OrderOverlap'):
            nirspec.G140M(order_overlap=[ov, 'x'])  # ty: ignore[invalid-argument-type]
        with pytest.raises(ValueError, match='Duplicate order'):
            nirspec.G140M(order_overlap=[ov, OrderOverlap(2, ratio_dir / 'r3.ecsv')])

    def test_str_lists_overlap_and_order_scale(self, ratio_dir):
        d = _g140m(ratio_dir / 'r2.ecsv')
        InstrumentConfig([d])
        text = str(d)
        assert 'Order Overlap:' in text
        assert 'Order Scale:' in text
        assert 'order_scale_G140M_2' in text
        assert 'Order Overlap' not in str(nirspec.G140M())


class TestNamingAndSharing:
    def test_unshared_anonymous_named_by_disperser_and_order(self, ratio_dir):
        d = _g140m(ratio_dir / 'r2.ecsv')
        InstrumentConfig([d])
        assert d.order_overlap[0].scale.name == 'order_scale_G140M_2'

    def test_user_label(self, ratio_dir):
        d = _g140m(ratio_dir / 'r2.ecsv', scale=OrderScale('g140m2'))
        assert d.order_overlap[0].scale.name == 'order_scale_g140m2'

    def test_shared_across_dispersers_gets_alpha_name(self, ratio_dir):
        s = OrderScale()
        a = _g140m(ratio_dir / 'r2.ecsv', scale=s)
        b = nirspec.G140H(
            order_overlap=[OrderOverlap(2, ratio_dir / 'r2.ecsv', scale=s)]
        )
        InstrumentConfig([a, b])
        assert s.name == 'order_scale_a'

    def test_same_instance_on_two_orders_is_one_parameter(self, ratio_dir):
        s = OrderScale(prior=prior.Uniform(0.5, 1.5))
        d = nirspec.G140M(
            order_overlap=[
                OrderOverlap(2, ratio_dir / 'r2.ecsv', scale=s),
                OrderOverlap(3, ratio_dir / 'r3.ecsv', scale=s),
            ]
        )
        _, args, _ = _build(d)
        assert s.name == 'order_scale_a'
        assert [n for n in args.all_priors if n.startswith('order_scale')] == [s.name]
        assert {w.scale_name for w in args.overlap_windows[0]} == {s.name}

    def test_parameter_table_lists_order_scale(self, ratio_dir):
        d = _g140m(ratio_dir / 'r2.ecsv')
        _, args, _ = _build(d)
        table = make_parameter_table({}, args)
        assert 'order_scale_G140M_2' in table.colnames
        np.testing.assert_allclose(table['order_scale_G140M_2'], 1.0)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip_with_sharing(self, ratio_dir):
        s = OrderScale(prior=prior.TruncatedNormal(1.0, 0.2, 0.5, 1.5))
        a = nirspec.G140M(
            order_overlap=[
                OrderOverlap(2, ratio_dir / 'r2.ecsv', scale=s),
                OrderOverlap(3, ratio_dir / 'r3.ecsv'),
            ]
        )
        b = nirspec.G235M(
            order_overlap=[OrderOverlap(2, ratio_dir / 'r2.ecsv', scale=s)]
        )
        cfg = InstrumentConfig([a, b])
        d = cfg.to_dict()
        assert d['calib_params'][s.name]['type'] == 'OrderScale'
        entry = d['entries'][0]['order_overlap']
        assert entry[0] == {
            'order': 2,
            'path': str(ratio_dir / 'r2.ecsv'),
            'scale': s.name,
        }

        cfg2 = InstrumentConfig.from_yaml(cfg.to_yaml())
        a2, b2 = cfg2['G140M'], cfg2['G235M']
        assert a2.order_overlap == a.order_overlap
        assert b2.order_overlap == b.order_overlap
        assert a2.order_overlap[0].scale is b2.order_overlap[0].scale
        assert a2.order_overlap[1].scale is not a2.order_overlap[0].scale
        assert a2.order_overlap[1].scale.name == 'order_scale_G140M_3'
        assert isinstance(a2.order_overlap[0].scale.prior, prior.TruncatedNormal)

    def test_column_written_only_when_not_default(self, tmp_path):
        p = _write_ratio(tmp_path / 'x.ecsv', colname='r')
        cfg = InstrumentConfig([nirspec.G140M(order_overlap=[OrderOverlap(2, p)])])
        entry = cfg.to_dict()['entries'][0]['order_overlap'][0]
        assert entry['column'] == 'r'
        cfg2 = InstrumentConfig.from_dict(cfg.to_dict())
        assert cfg2['G140M'].order_overlap[0].column == 'r'

    def test_no_key_without_overlap(self):
        d = InstrumentConfig([nirspec.G140M()]).to_dict()
        assert 'order_overlap' not in d['entries'][0]
        assert set(d['calib_params']) == {
            'r_scale_G140M',
            'flux_scale_G140M',
            'pix_offset_G140M',
        }

    def test_missing_file_raises_naming_path(self, tmp_path):
        p = _write_ratio(tmp_path / 'gone.ecsv')
        d = InstrumentConfig(
            [nirspec.G140M(order_overlap=[OrderOverlap(2, p)])]
        ).to_dict()
        p.unlink()
        with pytest.raises(FileNotFoundError, match=r'gone\.ecsv'):
            InstrumentConfig.from_dict(d)

    def test_configuration_yaml_round_trip(self, ratio_dir, tmp_path):
        cfg = Configuration(
            _line_config(),
            _cont_config(),
            InstrumentConfig([_g140m(ratio_dir / 'r2.ecsv')]),
        )
        path = tmp_path / 'cfg.yaml'
        cfg.save(path)
        loaded = Configuration.load(path)
        assert loaded.dispersers is not None
        assert loaded.dispersers['G140M'].order_overlap == (
            OrderOverlap(2, ratio_dir / 'r2.ecsv'),
        )


# ---------------------------------------------------------------------------
# Model behaviour
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def built(ratio_dir):
    """Toy model with and without overlap, in both integration modes."""
    out = {}
    for mode in ('analytic', 'convolution'):
        out[mode, True] = _build(_g140m(ratio_dir / 'r2.ecsv'), integration_mode=mode)
        # Same (contaminated) data, no overlap term.
        spec = _make_toy(nirspec.G140M())
        out[mode, False] = _build(
            nirspec.G140M(), integration_mode=mode, spectra_list=[spec]
        )
    return out


class TestOverlapModel:
    @pytest.mark.parametrize('mode', ['analytic', 'convolution'])
    def test_reproduces_injected_second_order(self, built, mode):
        _, args_on, spec = built[mode, True]
        pred_on = evaluate_model({}, args_on)[0]
        data = np.asarray(spec.flux)
        region = _overlap_region(spec)
        scale = np.max(data)
        interior = slice(10, -10)
        resid_on = (np.abs(pred_on.total[0] - data) / scale)[interior]
        assert resid_on.max() < (1e-4 if mode == 'analytic' else 1e-3)

        _, args_off, _ = built[mode, False]
        pred_off = evaluate_model({}, args_off)[0]
        resid_off = np.abs(pred_off.total[0] - data) / scale
        assert resid_off[region].max() > 1e-2
        assert resid_off[~region][10:].max() < 1e-3

    @pytest.mark.parametrize('mode', ['analytic', 'convolution'])
    def test_decomposition_identity(self, built, mode):
        _, args, spec = built[mode, True]
        pred = evaluate_model({}, args)[0]
        assert list(pred.overlap) == [2]
        parts = (
            sum(pred.lines.values())
            + sum(pred.continuum_regions.values())
            + sum(pred.overlap.values())
        )
        np.testing.assert_allclose(parts, pred.total, rtol=1e-10, atol=1e-12)
        region = _overlap_region(spec)
        if mode == 'analytic':
            assert np.all(pred.overlap[2][0][~region] == 0.0)
        assert np.all(pred.overlap[2][0][region] > 0.0)
        assert evaluate_model({}, built[mode, False][1])[0].overlap == {}

    @pytest.mark.parametrize('mode', ['analytic', 'convolution'])
    def test_model_matches_evaluate_model(self, built, mode):
        """The likelihood mean in unite_model equals evaluate_model's total."""
        model_fn, args, spec = built[mode, True]
        tr = handlers.trace(handlers.seed(model_fn, 0)).get_trace(args)
        mu = np.asarray(tr[f'obs_{spec.name}']['fn'].loc) * args.norm_factors[0]
        pred = evaluate_model({}, args)[0]
        np.testing.assert_allclose(mu, pred.total[0], rtol=1e-10)

    def test_analytic_and_convolution_agree(self, built):
        pred_a = evaluate_model({}, built['analytic', True][1])[0]
        pred_c = evaluate_model({}, built['convolution', True][1])[0]
        interior = slice(10, -10)
        np.testing.assert_allclose(
            pred_a.total[0, interior], pred_c.total[0, interior], rtol=2e-3
        )
        np.testing.assert_allclose(
            pred_a.overlap[2][0, interior],
            pred_c.overlap[2][0, interior],
            atol=2e-3 * np.max(pred_a.overlap[2]),
        )

    @staticmethod
    def _image_sigma(args, spec):
        """Gaussian sigma (observed um) of the second-order line image."""
        pred = evaluate_model({}, args)[0]
        wl = np.asarray(spec.wavelength)
        lam_c2 = 2 * LINE_REST_AA * 1e-4 * (1 + Z_SYS)
        window = np.abs(wl - lam_c2) < 0.01
        image = pred.overlap[2][0][window] - np.median(pred.overlap[2][0][window])
        image = np.clip(image, 0, None)
        x = wl[window]
        mean = np.sum(x * image) / np.sum(image)
        return math.sqrt(np.sum((x - mean) ** 2 * image) / np.sum(image))

    def test_second_order_line_image_resolution(self, built):
        """The image has R_2(lam1) = 2 R(lam1) at the photon wavelength."""
        _, args, spec = built['analytic', True]
        lam_c = LINE_REST_AA * 1e-4 * (1 + Z_SYS)
        lsf = lam_c / float(spec.disperser.R(lam_c))
        expected = math.hypot(2 * lam_c * LINE_FWHM_KMS / C_KMS, lsf) * FWHM_TO_SIGMA
        np.testing.assert_allclose(self._image_sigma(args, spec), expected, rtol=0.05)

    @pytest.mark.parametrize('mode', ['analytic', 'convolution'])
    def test_resolution_uses_photon_wavelength(self, ratio_dir, mode):
        """With constant R the order-2 LSF is R(lam1), not R(2 lam1)."""
        r_const = 1000.0
        d = GenericDisperser(
            R_func=lambda w: jnp.full_like(jnp.asarray(w, dtype=float), r_const),
            dlam_dpix_func=lambda w: w / 1000.0,
            unit=u.um,
            name='g140m',
            order_overlap=[OrderOverlap(2, ratio_dir / 'r2.ecsv')],
        )
        _, args, spec = _build(d, integration_mode=mode)
        lam_c = LINE_REST_AA * 1e-4 * (1 + Z_SYS)
        vel = 2 * lam_c * LINE_FWHM_KMS / C_KMS
        expected = math.hypot(vel, lam_c / r_const) * FWHM_TO_SIGMA
        landing = math.hypot(vel, 2 * lam_c / r_const) * FWHM_TO_SIGMA
        assert landing / expected > 1.15
        np.testing.assert_allclose(self._image_sigma(args, spec), expected, rtol=0.02)

    @pytest.mark.parametrize('mode', ['analytic', 'convolution'])
    def test_zero_ratio_matches_no_overlap_exactly(self, ratio_dir, built, mode):
        spec = _make_toy(nirspec.G140M())
        d = nirspec.G140M(order_overlap=[OrderOverlap(2, ratio_dir / 'zero.ecsv')])
        spec_zero = _make_toy(d)
        with pytest.warns(UserWarning, match='no active pixels'):
            _, args_zero, _ = _build(d, integration_mode=mode, spectra_list=[spec_zero])
        _, args_off, _ = _build(
            nirspec.G140M(), integration_mode=mode, spectra_list=[spec]
        )
        a = evaluate_model({}, args_off)[0].total
        b = evaluate_model({}, args_zero)[0].total
        assert np.array_equal(a, b)

    @pytest.mark.parametrize('mode', ['analytic', 'convolution'])
    def test_order_scale_is_linear(self, ratio_dir, mode):
        s = OrderScale(prior=prior.Uniform(0.0, 3.0))
        _, args, _ = _build(
            _g140m(ratio_dir / 'r2.ecsv', scale=s), integration_mode=mode
        )
        name = s.name
        p1 = evaluate_model({name: np.array([1.0])}, args)[0]
        p2 = evaluate_model({name: np.array([2.0])}, args)[0]
        np.testing.assert_allclose(p2.overlap[2], 2 * p1.overlap[2], rtol=1e-12)
        np.testing.assert_allclose(
            p2.total - p1.total, p1.overlap[2], rtol=1e-9, atol=1e-12
        )

    def test_flux_scale_multiplies_overlap(self, ratio_dir):
        fs = FluxScale(prior=prior.Uniform(0.5, 1.5))
        _, args, _ = _build(_g140m(ratio_dir / 'r2.ecsv', flux_scale=fs))
        p1 = evaluate_model({fs.name: np.array([1.0])}, args)[0]
        p2 = evaluate_model({fs.name: np.array([1.3])}, args)[0]
        np.testing.assert_allclose(p2.overlap[2], 1.3 * p1.overlap[2], rtol=1e-12)

    def test_batch_size_matches(self, ratio_dir):
        s = OrderScale(prior=prior.Uniform(0.0, 3.0))
        _, args, _ = _build(_g140m(ratio_dir / 'r2.ecsv', scale=s))
        samples = {s.name: np.linspace(0.5, 1.5, 5)}
        full = evaluate_model(samples, args)[0]
        chunked = evaluate_model(samples, args, batch_size=2)[0]
        np.testing.assert_allclose(chunked.overlap[2], full.overlap[2])

    def test_spectra_tables_overlap_column(self, built):
        _, args, spec = built['analytic', True]
        t = make_spectra_tables({}, args)[spec.name]
        assert 'overlap_2' in t.colnames
        assert t['overlap_2'].unit == FLUX_UNIT
        pct = make_spectra_tables({}, args, percentiles=np.array([0.5]))[spec.name]
        assert pct['overlap_2'].shape == (len(pct), 1)
        _, args_off, _ = built['analytic', False]
        cols = make_spectra_tables({}, args_off)[spec.name].colnames
        assert not any(c.startswith('overlap') for c in cols)

    def test_two_orders(self, ratio_dir):
        d = nirspec.G140M(
            order_overlap=[
                OrderOverlap(2, ratio_dir / 'r2.ecsv'),
                OrderOverlap(3, ratio_dir / 'r3.ecsv'),
            ]
        )
        _, args, _ = _build(d)
        pred = evaluate_model({}, args)[0]
        assert sorted(pred.overlap) == [2, 3]
        parts = (
            sum(pred.lines.values())
            + sum(pred.continuum_regions.values())
            + sum(pred.overlap.values())
        )
        np.testing.assert_allclose(parts, pred.total, rtol=1e-10)

    @pytest.mark.parametrize('mode', ['analytic', 'convolution'])
    def test_sampled_model_is_differentiable(self, ratio_dir, mode):
        d = nirspec.G140M(
            order_overlap=[
                OrderOverlap(
                    2,
                    ratio_dir / 'r2.ecsv',
                    scale=OrderScale(prior=prior.Uniform(0.5, 1.5)),
                )
            ],
            r_scale=RScale(prior=prior.Uniform(0.8, 1.2)),
            flux_scale=FluxScale(prior=prior.Uniform(0.5, 1.5)),
            pix_offset=PixOffset(prior=prior.Uniform(-1.0, 1.0)),
        )
        spectra = Spectra([_make_toy(d)], redshift=Z_SYS)
        cc = ContinuumConfiguration(
            [
                ContinuumRegion(lo * u.AA, hi * u.AA, form=PowerLaw(), name=name)
                for name, (lo, hi) in REGIONS.items()
            ]
        )
        spectra.prepare(_line_config(free=True), cc, drop_empty_regions=False)
        spectra.compute_scales(
            spectra.prepared_line_config, spectra.prepared_cont_config
        )
        model_fn, args = model.ModelBuilder(
            spectra.prepared_line_config, spectra.prepared_cont_config, spectra
        ).build(integration_mode=mode)
        tr = handlers.trace(handlers.seed(model_fn, 0)).get_trace(args)
        params = {
            k: v['value']
            for k, v in tr.items()
            if v['type'] == 'sample' and not v['is_observed']
        }
        assert 'order_scale_G140M_2' in params

        def _ld(p):
            return log_density(model_fn, (args,), {}, p)[0]

        ld, grad = jax.jit(jax.value_and_grad(_ld))(params)
        assert np.isfinite(float(ld))
        assert all(np.isfinite(float(g)) for g in grad.values())
        assert float(grad['order_scale_G140M_2']) != 0.0

    def test_evaluate_intrinsic_excludes_overlap(self, built):
        from unite.compute import evaluate_intrinsic

        _, args_on, _ = built['analytic', True]
        _, args_off, _ = built['analytic', False]
        wl = np.linspace(1.4, 1.7, 50) * u.um
        a = evaluate_intrinsic({}, args_on, wl)
        b = evaluate_intrinsic({}, args_off, wl)
        np.testing.assert_allclose(a.total, b.total)
        assert a.overlap == {}


# ---------------------------------------------------------------------------
# Windows, PixOffset margin and recovery
# ---------------------------------------------------------------------------


class TestWindows:
    @pytest.mark.parametrize(
        ('p', 'margin'),
        [
            (prior.Fixed(0.0), 1),
            (prior.Fixed(-2.3), 4),
            (prior.Uniform(-1.0, 1.0), 2),
            (prior.TruncatedNormal(0.0, 0.5, -0.4, 2.5), 4),
        ],
    )
    def test_pix_offset_margin(self, p, margin):
        assert _overlap.pix_offset_margin(p) == margin

    def test_dependent_bound_means_full_range(self):
        other = PixOffset(prior=prior.Uniform(-1, 1))
        assert _overlap.pix_offset_margin(prior.Uniform(other - 1, other + 1)) is None

    def test_window_matches_support(self, built):
        _, args, spec = built['analytic', True]
        (win,) = args.overlap_windows[0]
        wl = np.asarray(spec.wavelength)
        active = np.flatnonzero(_r2(wl) > 0)
        # Fixed(0) PixOffset: one-pixel margin on each side.
        assert win.pix0 == active[0] - 1
        assert win.pix0 + win.n_pix == min(active[-1] + 2, wl.size)
        assert win.region_mask.shape == (2, win.i1 - win.i0)

    def test_unbounded_pix_offset_uses_full_range(self, ratio_dir):
        d = _g140m(
            ratio_dir / 'r2.ecsv', pix_offset=PixOffset(prior=prior.Uniform(-1e9, 1e9))
        )
        _, args, spec = _build(d)
        (win,) = args.overlap_windows[0]
        assert (win.pix0, win.n_pix) == (0, spec.npix)


def _profile(model_fn, args, site, grid, fixed):
    """Log density along *grid* for parameter *site*, others at *fixed*."""

    @jax.jit
    def ld(v):
        return log_density(model_fn, (args,), {}, {**fixed, site: v})[0]

    return np.array([float(ld(jnp.asarray(v))) for v in grid])


PIX_TRUE = 0.3
SCALE_TRUE = 1.2


@pytest.fixture(scope='module', params=['analytic', 'convolution'])
def shifted(request, ratio_dir):
    po = PixOffset(prior=prior.Uniform(-1.0, 1.0))
    osc = OrderScale(prior=prior.Uniform(0.5, 1.5))
    d = nirspec.G140M(
        pix_offset=po,
        order_overlap=[OrderOverlap(2, ratio_dir / 'r2_sharp.ecsv', scale=osc)],
    )
    spec = _make_toy(d, ratio_fn=_r2_sharp, order_scale=SCALE_TRUE, pix_shift=PIX_TRUE)
    model_fn, args, _ = _build(d, integration_mode=request.param, spectra_list=[spec])
    return model_fn, args, po.name, osc.name


class TestPixOffsetRecovery:
    """A known pixel shift plus overlap is recovered, with a sharp r_m edge."""

    def test_window_margin_from_prior(self, shifted):
        _, args, _, _ = shifted
        (win,) = args.overlap_windows[0]
        wl = np.asarray(args.spectra[0].wavelength)
        first_active = int(np.flatnonzero(_r2_sharp(wl) > 0)[0])
        assert win.pix0 <= first_active - 2  # Uniform(-1, 1): margin 2 pixels

    def test_pix_offset_recovered(self, shifted):
        model_fn, args, po, osc = shifted
        grid = np.linspace(-0.2, 0.8, 51)
        ll = _profile(model_fn, args, po, grid, {osc: SCALE_TRUE})
        assert abs(grid[np.argmax(ll)] - PIX_TRUE) <= 0.02
        # Continuous: no steps from pixels crossing region or support edges.
        d2 = np.diff(ll, 2)
        assert np.all(np.isfinite(ll))
        assert np.max(np.abs(d2)) < 0.05 * (ll.max() - ll.min())

    def test_order_scale_recovered(self, shifted):
        model_fn, args, po, osc = shifted
        grid = np.linspace(0.8, 1.5, 71)
        ll = _profile(model_fn, args, osc, grid, {po: PIX_TRUE})
        assert abs(grid[np.argmax(ll)] - SCALE_TRUE) <= 0.01


# ---------------------------------------------------------------------------
# Build-time warnings
# ---------------------------------------------------------------------------


class TestBuildWarnings:
    def test_clean_build_is_silent(self, ratio_dir):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _build(_g140m(ratio_dir / 'r2.ecsv'))

    def test_no_active_pixels(self, ratio_dir):
        d = _g140m(ratio_dir / 'far.ecsv')  # support beyond the spectrum
        with pytest.warns(UserWarning, match='order 2 overlap has no active') as rec:
            _build(d)
        assert 'constrained only by its prior' not in str(rec[0].message)

    def test_no_active_pixels_free_scale(self, ratio_dir):
        d = _g140m(
            ratio_dir / 'far.ecsv', scale=OrderScale(prior=prior.Uniform(0.5, 1.5))
        )
        with pytest.warns(UserWarning, match='constrained only by its prior'):
            _, args, _ = _build(d)
        assert args.overlap_windows[0] == []

    def test_source_window_without_continuum_region(self, ratio_dir):
        # Only the nir region: photons at 0.65-0.9 um (rest 0.53-0.74 um) have
        # first-order data in the untrimmed spectrum but no region.
        regions = {'nir': (8204.0, 20000.0)}
        spec = _make_toy(nirspec.G140M())
        d = _g140m(ratio_dir / 'r2.ecsv')
        spec = _make_toy(d)
        with pytest.warns(UserWarning, match='no continuum region'):
            _build(d, regions=regions, spectra_list=[spec])

    def test_region_edge_inside_source_window(self, ratio_dir):
        # Region edge at rest 6000 AA -> obs 0.732 um, inside 0.65-0.9 um.
        regions = {'a': (6000.0, 8204.0), 'nir': (8204.0, 20000.0)}
        d = _g140m(ratio_dir / 'r2.ecsv')
        with pytest.warns(UserWarning, match='region edge falls inside'):
            _build(d, regions=regions)

    def test_source_window_without_first_order_data_is_silent(self, ratio_dir):
        # Spectrum starts at 1.0 um: photons at 0.65-0.9 um have no data and
        # no region; nothing to warn about.
        regions = {'nir': (8204.0, 20000.0)}
        d = _g140m(ratio_dir / 'r2.ecsv')
        spec = _make_toy(d, wl_range=(1.0, 1.8))
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            _build(d, regions=regions, spectra_list=[spec])


class TestMsaexpOrderRatio:
    """``msaexp_order_ratio`` converts sensitivities to the f_lambda ratio."""

    @staticmethod
    def _table():
        wl = np.linspace(1.0, 4.0, 301)
        return Table(
            {
                'wavelength': wl,
                'sensitivity': np.full_like(wl, 2.0),
                'sensitivity_2': np.where(wl < 1.2, 0.0, 1.0),
            }
        )

    def test_values_and_zero_outside(self, tmp_path):
        from unite.instrument.nirspec import msaexp_order_ratio

        out = msaexp_order_ratio(self._table(), 2, path=tmp_path / 'r.fits')
        wl_ph = np.linspace(1.0, 4.0, 301)
        assert np.allclose(out['wavelength'].value, 2 * wl_ph)
        r = np.asarray(out['throughput_ratio'])
        # S_2 = 0 blueward of 1.2 um; S_1 undefined redward of 4 um observed
        assert np.all(r[wl_ph < 1.2] == 0.0)
        assert np.all(r[wl_ph > 2.0 + 1e-9] == 0.0)
        mid = (wl_ph >= 1.2) & (wl_ph <= 2.0)
        assert np.allclose(r[mid], 1.0 / (4 * 2.0))
        assert (tmp_path / 'r.fits').exists()

    def test_bad_input(self):
        from unite.instrument.nirspec import msaexp_order_ratio

        with pytest.raises(ValueError, match='order'):
            msaexp_order_ratio(self._table(), 1)
        with pytest.raises(ValueError, match='sensitivity_3'):
            msaexp_order_ratio(self._table(), 3)
