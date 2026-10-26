"""Tests for opt-in starting values (``init``) on priors."""

import astropy.units as u
import numpy as np
import pytest
from jax import random

from unite import line, model, prior
from unite.instrument.generic import SimpleDisperser
from unite.line.config import FWHM, LineConfiguration
from unite.prior import Fixed, TruncatedNormal, Uniform, prior_from_dict
from unite.spectrum import Spectra, Spectrum

# ---------------------------------------------------------------------------
# Prior-level behaviour
# ---------------------------------------------------------------------------


def _tok(name):
    return FWHM(name=name)


class TestPriorInit:
    """``init`` on Uniform / TruncatedNormal / Fixed."""

    def test_default_is_none(self):
        assert Uniform(0, 1).init is None
        assert TruncatedNormal(0.5, 0.1, 0, 1).init is None
        assert Fixed(1.0).init is None

    def test_stored_as_float(self):
        assert Uniform(0, 10, init=3).init == 3.0
        assert isinstance(Uniform(0, 10, init=3).init, float)
        assert TruncatedNormal(5, 1, 0, 10, init=4).init == 4.0

    @pytest.mark.parametrize('bad', [0.0, -1.0, 10.0, 11.0])
    def test_uniform_outside_float_bounds_raises(self, bad):
        with pytest.raises(ValueError, match='init='):
            Uniform(0, 10, init=bad)

    @pytest.mark.parametrize('bad', [0.0, 10.0, 12.0])
    def test_truncated_normal_outside_float_bounds_raises(self, bad):
        with pytest.raises(ValueError, match='init='):
            TruncatedNormal(5, 1, 0, 10, init=bad)

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
    def test_non_finite_raises(self, bad):
        with pytest.raises(ValueError, match='finite'):
            Uniform(0, 10, init=bad)

    def test_dependent_bound_not_checked_at_construction(self):
        # The lower bound depends on another parameter, so it cannot be
        # checked here; the plain-float upper bound still is.
        tok = _tok('x')
        assert Uniform(low=tok * 2 + 100, high=1000, init=50.0).init == 50.0
        with pytest.raises(ValueError, match='upper bound'):
            Uniform(low=tok * 2 + 100, high=1000, init=1500.0)

    def test_keyword_only(self):
        with pytest.raises(TypeError):
            Uniform(0, 10, 5.0)  # ty: ignore[too-many-positional-arguments]

    def test_fixed_rejects_init(self):
        with pytest.raises(TypeError):
            Fixed(1.0, init=1.0)  # ty: ignore[unknown-argument]
        with pytest.raises(ValueError, match='cannot have an init'):
            Fixed.from_dict({'type': 'Fixed', 'value': 1.0, 'init': 1.0})

    def test_to_dist_unaffected(self):
        d = Uniform(0, 10, init=3.0).to_dist({})
        assert float(d.low) == 0.0
        assert float(d.high) == 10.0

    def test_repr_shows_init_only_when_set(self):
        assert 'init' not in repr(Uniform(0, 1))
        assert 'init=0.5' in repr(Uniform(0, 1, init=0.5))
        assert 'init=4.5' in repr(TruncatedNormal(5, 1, 0, 10, init=4.5))


class TestInitSerialization:
    """``init`` round-trips and stays backward compatible."""

    def test_to_dict_omits_init_when_unset(self):
        assert 'init' not in Uniform(0, 10).to_dict()
        assert 'init' not in TruncatedNormal(5, 1, 0, 10).to_dict()

    def test_uniform_roundtrip(self):
        d = Uniform(0, 10, init=3.5).to_dict()
        assert d['init'] == 3.5
        assert prior_from_dict(d).init == 3.5

    def test_truncated_normal_roundtrip(self):
        d = TruncatedNormal(5, 1, 0, 10, init=4.5).to_dict()
        assert d['init'] == 4.5
        p2 = prior_from_dict(d)
        assert isinstance(p2, TruncatedNormal)
        assert p2.init == 4.5

    def test_dict_without_init_key_loads(self):
        """YAML written before ``init`` existed still loads."""
        p = prior_from_dict({'type': 'Uniform', 'low': 0.0, 'high': 1.0})
        assert p.init is None

    def test_dependent_bound_roundtrip(self):
        tok = _tok('x')
        p = Uniform(low=tok * 2 + 100, high=1000, init=500.0)
        d = p.to_dict({tok: 'x'})
        p2 = Uniform.from_dict(d, token_registry={'x': tok})
        assert p2.init == 500.0

    def test_line_configuration_yaml_roundtrip(self):
        lc = LineConfiguration()
        lc.add_line(
            'Ha',
            center=6563.0 * u.AA,
            fwhm_gauss=FWHM('narrow', prior=Uniform(0, 1000, init=250.0)),
        )
        lc2 = LineConfiguration.from_yaml(lc.to_yaml())
        inits = [tok.prior.init for e in lc2._entries for tok in e.fwhms.values()]
        assert inits == [250.0]


# ---------------------------------------------------------------------------
# Model-level behaviour
# ---------------------------------------------------------------------------


def _spectrum(wmin=6500, wmax=6600, npix=100):
    wavelength = np.linspace(wmin, wmax, npix) * u.AA
    disperser = SimpleDisperser(wavelength=wavelength, R=3000.0, name='d')
    low = wavelength - 0.5 * np.gradient(wavelength)
    high = wavelength + 0.5 * np.gradient(wavelength)
    sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
    rng = np.random.default_rng(42)
    unit = u.Unit('1e-17 erg / (s cm2 AA)')
    flux = (
        100 * np.exp(-0.5 * ((wavelength.value - 6563.0) / sigma) ** 2)
        + rng.normal(0, 2, npix)
    ) * unit
    error = np.full(npix, 2.0) * unit
    return Spectrum(
        low=low, high=high, flux=flux, error=error, disperser=disperser, name='s'
    )


def _two_component_config(*, narrow_init=None, broad_init=None, z_init=None):
    """Narrow + broad H-alpha; the broad FWHM lower bound depends on narrow."""
    narrow = line.FWHM('narrow', prior=prior.Uniform(100.0, 400.0, init=narrow_init))
    broad = line.FWHM(
        'broad', prior=prior.Uniform(low=narrow * 2 + 150, high=3000.0, init=broad_init)
    )
    z = line.Redshift('z', prior=prior.Uniform(-0.005, 0.005, init=z_init))
    lc = line.LineConfiguration()
    for name, fwhm in (('Ha_n', narrow), ('Ha_b', broad)):
        lc.add_line(
            name,
            center=6563.0 * u.AA,
            redshift=z,
            fwhm_gauss=fwhm,
            flux=line.Flux(prior=prior.Uniform(0.0, 150.0)),
        )
    return lc, narrow, broad, z


def _builder(lc, **spectrum_kwargs):
    spectra = Spectra([_spectrum(**spectrum_kwargs)], redshift=0.0)
    spectra.prepare(lc, None)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    return model.ModelBuilder(spectra.prepared_line_config, None, spectra)


class TestFarSEMGInitialisation:
    """A Gaussian-limit SEMG line far from other pixels must have finite gradients."""

    def test_init_to_uniform_finds_valid_start(self):
        from numpyro.infer import init_to_uniform
        from numpyro.infer.util import initialize_model

        # [O III] and H-alpha are ~7e4 km/s apart; fwhm_exp << fwhm_gauss.
        z = line.Redshift('z', prior=prior.Uniform(-0.005, 0.005))
        lc = line.LineConfiguration()
        lc.add_line(
            'OIII',
            center=5007.0 * u.AA,
            redshift=z,
            fwhm_gauss=line.FWHM('n', prior=prior.Fixed(200.0)),
            flux=line.Flux(prior=prior.Fixed(50.0)),
        )
        lc.add_line(
            'Ha_wing',
            center=6563.0 * u.AA,
            profile='semg',
            redshift=z,
            fwhm_gauss=line.FWHM('g', prior=prior.Fixed(480.0)),
            fwhm_exp=line.FWHM('e', prior=prior.Fixed(5.0)),
            flux=line.Flux(prior=prior.Fixed(50.0)),
        )
        model_fn, args = _builder(lc, wmin=4900, wmax=6700, npix=600).build()
        info = initialize_model(
            random.PRNGKey(0),
            model_fn,
            model_args=(args,),
            init_strategy=init_to_uniform,
        )
        assert np.all(np.isfinite(np.asarray(info.param_info.potential_energy)))


class TestModelArgsInitValues:
    """``ModelArgs.init_values`` and build-time validation."""

    def test_empty_when_no_init_set(self):
        lc, *_ = _two_component_config()
        _, args = _builder(lc).build()
        assert args.init_values == {}

    def test_keyed_by_site_name(self):
        lc, narrow, _broad, z = _two_component_config(narrow_init=250.0, z_init=0.001)
        _, args = _builder(lc).build()
        assert args.init_values == {narrow.name: 250.0, z.name: 0.001}
        assert set(args.init_values) <= set(args.dependency_order)

    def test_returns_fresh_dict(self):
        lc, *_ = _two_component_config(narrow_init=250.0)
        _, args = _builder(lc).build()
        args.init_values['junk'] = 1.0
        assert 'junk' not in args.init_values

    def test_valid_dependent_init_builds(self):
        # broad lower bound resolves to 250 * 2 + 150 = 650; 1500 is inside.
        lc, *_ = _two_component_config(narrow_init=250.0, broad_init=1500.0)
        _, args = _builder(lc).build()
        assert len(args.init_values) == 2

    def test_dependent_init_outside_resolved_bounds_raises(self):
        # broad lower bound resolves to 300 * 2 + 150 = 750; 500 is below it.
        lc, _narrow, broad, _z = _two_component_config(
            narrow_init=300.0, broad_init=500.0
        )
        with pytest.raises(ValueError, match='lies outside its prior bounds') as exc:
            _builder(lc).build()
        assert broad.name in str(exc.value)

    def test_dependent_init_skipped_when_dependency_unknown(self):
        # narrow has no init, so broad's resolved bounds are not knowable at
        # build time; build must not raise.
        lc, *_ = _two_component_config(broad_init=500.0)
        _, args = _builder(lc).build()
        assert len(args.init_values) == 1

    def test_init_to_value_starts_at_specified_values(self):
        from numpyro.infer import init_to_value
        from numpyro.infer.util import initialize_model

        lc, narrow, broad, z = _two_component_config(
            narrow_init=250.0, broad_init=1500.0, z_init=0.001
        )
        model_fn, args = _builder(lc).build()
        info = initialize_model(
            random.PRNGKey(0),
            model_fn,
            model_args=(args,),
            init_strategy=init_to_value(values=args.init_values),
        )
        start = info.postprocess_fn(info.param_info.z)
        assert float(start[narrow.name]) == pytest.approx(250.0)
        assert float(start[broad.name]) == pytest.approx(1500.0)
        assert float(start[z.name]) == pytest.approx(0.001)


class TestFitInitStrategy:
    """``fit()`` applies ``init_values`` and honours an explicit strategy."""

    @staticmethod
    def _fit_and_capture(monkeypatch, builder, **fit_kwargs):
        import numpyro.infer as infer

        seen = {}
        real_nuts = infer.NUTS

        def spy(*args, **kwargs):
            seen['init_strategy'] = kwargs['init_strategy']
            return real_nuts(*args, **kwargs)

        monkeypatch.setattr(infer, 'NUTS', spy)
        builder.fit(
            num_warmup=2, num_samples=2, progress_bar=False, seed=0, **fit_kwargs
        )
        return seen['init_strategy']

    @staticmethod
    def _single_line_builder(**z_kwargs):
        lc = line.LineConfiguration()
        lc.add_line(
            'H_alpha',
            center=6563.0 * u.AA,
            redshift=line.Redshift(prior=prior.Uniform(-0.005, 0.005, **z_kwargs)),
            fwhm_gauss=line.FWHM(prior=prior.Uniform(100.0, 1000.0)),
            flux=line.Flux(prior=prior.Uniform(50.0, 150.0)),
        )
        return _builder(lc)

    @pytest.mark.slow
    def test_default_strategy_when_no_init(self, monkeypatch):
        from numpyro.infer import init_to_uniform

        strategy = self._fit_and_capture(monkeypatch, self._single_line_builder())
        assert strategy is init_to_uniform

    @pytest.mark.slow
    def test_init_values_applied_automatically(self, monkeypatch):
        from numpyro.infer import init_to_value

        builder = self._single_line_builder(init=0.002)
        strategy = self._fit_and_capture(monkeypatch, builder)
        assert strategy.func is init_to_value
        assert list(strategy.keywords['values'].values()) == [0.002]

    @pytest.mark.slow
    def test_explicit_strategy_wins(self, monkeypatch):
        from numpyro.infer import init_to_median

        builder = self._single_line_builder(init=0.002)
        strategy = self._fit_and_capture(
            monkeypatch, builder, init_strategy=init_to_median
        )
        assert strategy is init_to_median
