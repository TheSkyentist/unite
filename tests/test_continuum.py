"""Tests for unite.continuum (forms, regions, and configuration)."""

import jax.numpy as jnp
import pytest

from unite.continuum import (
    AttenuatedBlackbody,
    Bernstein,
    Blackbody,
    BSpline,
    Chebyshev,
    ContinuumConfiguration,
    ContinuumRegion,
    Linear,
    ModifiedBlackbody,
    Polynomial,
    PowerLaw,
    form_from_dict,
)
from unite.prior import Uniform

# ---------------------------------------------------------------------------
# ContinuumRegion
# ---------------------------------------------------------------------------


class TestContinuumRegion:
    def test_construction(self):
        r = ContinuumRegion(1.0, 2.0, Linear())
        assert r.low == pytest.approx(1.0)
        assert r.high == pytest.approx(2.0)

    def test_center(self):
        r = ContinuumRegion(1.0, 3.0)
        assert r.center == pytest.approx(2.0)

    def test_low_ge_high_raises(self):
        with pytest.raises(ValueError, match='low must be < high'):
            ContinuumRegion(5.0, 3.0)

    def test_low_equals_high_raises(self):
        with pytest.raises(ValueError, match='low must be < high'):
            ContinuumRegion(3.0, 3.0)

    def test_default_form_is_linear(self):
        r = ContinuumRegion(1.0, 2.0)
        assert isinstance(r.form, Linear)

    def test_prior_override_stored(self):
        r = ContinuumRegion(1.0, 2.0, Linear(), priors={'offset': Uniform(-5, 5)})
        assert 'offset' in r.priors


# ---------------------------------------------------------------------------
# ContinuumConfiguration construction
# ---------------------------------------------------------------------------


class TestContinuumConfigurationConstruction:
    def test_empty(self):
        config = ContinuumConfiguration()
        assert len(config) == 0

    def test_sorted_on_input(self):
        r1 = ContinuumRegion(6.0, 7.0)
        r2 = ContinuumRegion(1.0, 2.0)
        config = ContinuumConfiguration([r1, r2])
        assert config[0].low == pytest.approx(1.0)
        assert config[1].low == pytest.approx(6.0)

    def test_overlap_raises(self):
        r1 = ContinuumRegion(1.0, 3.0)
        r2 = ContinuumRegion(2.5, 4.0)
        with pytest.raises(ValueError, match='overlap'):
            ContinuumConfiguration([r1, r2])

    def test_touching_boundaries_allowed(self):
        r1 = ContinuumRegion(1.0, 2.0)
        r2 = ContinuumRegion(2.0, 3.0)
        config = ContinuumConfiguration([r1, r2])
        assert len(config) == 2

    def test_iter_and_getitem(self):
        regions = [ContinuumRegion(float(i), float(i) + 0.5) for i in range(3)]
        config = ContinuumConfiguration(regions)
        assert list(config) == list(regions)
        assert config[1].low == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ContinuumConfiguration.from_lines
# ---------------------------------------------------------------------------


class TestFromLines:
    def test_basic(self):
        config = ContinuumConfiguration.from_lines([5000.0, 6560.0])
        assert len(config) == 2

    def test_all_regions_share_form(self):
        config = ContinuumConfiguration.from_lines([5000.0, 6560.0])
        assert config[0].form is config[1].form

    def test_overlapping_lines_merged(self):
        # Two lines close together → their padded windows merge into one region.
        config = ContinuumConfiguration.from_lines([6549.86, 6585.27], pad=0.05)
        assert len(config) == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError, match='must not be empty'):
            ContinuumConfiguration.from_lines([])

    def test_custom_form(self):
        pl = PowerLaw()
        config = ContinuumConfiguration.from_lines([5000.0], form=pl)
        assert config[0].form is pl

    def test_regions_sorted(self):
        config = ContinuumConfiguration.from_lines([6560.0, 4860.0])
        assert config[0].low < config[1].low


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    def test_linear_region(self):
        config = ContinuumConfiguration([ContinuumRegion(4600.0, 5200.0, Linear())])
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        assert len(config2) == 1
        assert isinstance(config2[0].form, Linear)
        assert config2[0].low == pytest.approx(4600.0)

    def test_shared_form_identity_preserved(self):
        pl = PowerLaw()
        config = ContinuumConfiguration(
            [
                ContinuumRegion(1.0, 1.5, pl),
                ContinuumRegion(2.0, 2.5, pl),
                ContinuumRegion(3.0, 3.5, pl),
            ]
        )
        d = config.to_dict()
        # One form entry in the serialized dict
        assert len(d['forms']) == 1

        config2 = ContinuumConfiguration.from_dict(d)
        assert config2[0].form is config2[1].form
        assert config2[1].form is config2[2].form

    def test_distinct_forms_serialized_separately(self):
        config = ContinuumConfiguration(
            [ContinuumRegion(1.0, 1.5, Linear()), ContinuumRegion(2.0, 2.5, PowerLaw())]
        )
        d = config.to_dict()
        assert len(d['forms']) == 2

    def test_prior_overrides_roundtrip(self):
        r = ContinuumRegion(1.0, 2.0, Linear(), priors={'offset': Uniform(-5, 5)})
        config = ContinuumConfiguration([r])
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        assert 'offset' in config2[0].priors
        prior = config2[0].priors['offset']
        assert isinstance(prior, Uniform)
        assert prior.low == pytest.approx(-5.0)

    def test_polynomial_degree_preserved(self):
        config = ContinuumConfiguration(
            [ContinuumRegion(1.0, 2.0, Polynomial(degree=3))]
        )
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        assert config2[0].form.degree == 3

    def test_from_lines_roundtrip(self):
        config = ContinuumConfiguration.from_lines([5000.0, 6560.0])
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        assert len(config2) == len(config)
        assert config2[0].form is config2[1].form


# ---------------------------------------------------------------------------
# form_from_dict dispatch
# ---------------------------------------------------------------------------


class TestFormFromDict:
    def test_linear(self):
        f = form_from_dict({'type': 'Linear'})
        assert isinstance(f, Linear)

    def test_power_law(self):
        f = form_from_dict({'type': 'PowerLaw'})
        assert isinstance(f, PowerLaw)

    def test_polynomial(self):
        f = form_from_dict({'type': 'Polynomial', 'degree': 2})
        assert isinstance(f, Polynomial)
        assert f.degree == 2

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match='Unknown ContinuumForm type'):
            form_from_dict({'type': 'NoSuchForm'})


# ---------------------------------------------------------------------------
# Individual form contracts
# ---------------------------------------------------------------------------


_WL = jnp.linspace(0.9, 1.1, 40)
_CENTER = 1.0


class TestLinear:
    def test_param_names(self):
        assert Linear().param_names() == ('angle', 'offset')

    def test_n_params(self):
        assert Linear().n_params == 2

    def test_default_priors_keys(self):
        assert set(Linear().default_priors()) == {'angle', 'offset'}

    def test_evaluate_shape(self):
        params = {'angle': jnp.array(0.0), 'offset': jnp.array(1.0)}
        result = Linear().evaluate(_WL, _CENTER, params)
        assert result.shape == _WL.shape

    def test_evaluate_flat_at_zero_angle(self):
        params = {'angle': jnp.array(0.0), 'offset': jnp.array(2.0)}
        result = Linear().evaluate(_WL, _CENTER, params)
        assert jnp.allclose(result, 2.0)

    def test_roundtrip(self):
        f = Linear()
        assert form_from_dict(f.to_dict()) == f

    def test_equality(self):
        assert Linear() == Linear()


class TestPowerLaw:
    def test_param_names(self):
        assert PowerLaw().param_names() == ('amplitude', 'beta')

    def test_evaluate_shape(self):
        params = {'amplitude': jnp.array(1.0), 'beta': jnp.array(0.0)}
        result = PowerLaw().evaluate(_WL, _CENTER, params)
        assert result.shape == _WL.shape

    def test_evaluate_flat_at_zero_beta(self):
        params = {'amplitude': jnp.array(2.0), 'beta': jnp.array(0.0)}
        result = PowerLaw().evaluate(_WL, _CENTER, params)
        assert jnp.allclose(result, 2.0)

    def test_roundtrip(self):
        f = PowerLaw()
        assert form_from_dict(f.to_dict()) == f


class TestPolynomial:
    def test_param_names_degree1(self):
        assert Polynomial(degree=1).param_names() == ('c0', 'c1')

    def test_param_names_degree2(self):
        assert Polynomial(degree=2).param_names() == ('c0', 'c1', 'c2')

    def test_negative_degree_raises(self):
        with pytest.raises(ValueError, match='>= 0'):
            Polynomial(degree=-1)

    def test_evaluate_shape(self):
        p = Polynomial(degree=2)
        params = {k: jnp.array(1.0) for k in p.param_names()}
        result = p.evaluate(_WL, _CENTER, params)
        assert result.shape == _WL.shape

    def test_roundtrip(self):
        f = Polynomial(degree=3)
        f2 = form_from_dict(f.to_dict())
        assert f2.degree == 3
        assert f2 == f

    def test_equality_different_degree(self):
        assert Polynomial(1) != Polynomial(2)

    def test_hash_consistent(self):
        assert hash(Polynomial(2)) == hash(Polynomial(2))


class TestChebyshev:
    def test_param_names_order2(self):
        assert Chebyshev(order=2).param_names() == ('c0', 'c1', 'c2')

    def test_negative_order_raises(self):
        with pytest.raises(ValueError, match='>= 0'):
            Chebyshev(order=-1)

    def test_evaluate_shape(self):
        f = Chebyshev(order=2, half_width=0.1)
        params = {k: jnp.array(0.0) for k in f.param_names()}
        params['c0'] = jnp.array(1.0)
        result = f.evaluate(_WL, _CENTER, params)
        assert result.shape == _WL.shape

    def test_roundtrip(self):
        f = Chebyshev(order=3, half_width=0.2)
        f2 = form_from_dict(f.to_dict())
        assert f2.order == 3
        assert f2.half_width == pytest.approx(0.2)
        assert f2 == f

    def test_equality_checks_half_width(self):
        assert Chebyshev(2, 0.1) != Chebyshev(2, 0.2)


class TestBlackbody:
    def test_param_names(self):
        assert set(Blackbody().param_names()) == {'amplitude', 'temperature'}

    def test_evaluate_shape(self):
        wl = jnp.linspace(0.5, 2.0, 30)
        params = {'amplitude': jnp.array(1.0), 'temperature': jnp.array(5000.0)}
        result = Blackbody(pivot_micron=1.0).evaluate(wl, 1.0, params)
        assert result.shape == wl.shape

    def test_evaluate_positive(self):
        wl = jnp.linspace(0.5, 2.0, 30)
        params = {'amplitude': jnp.array(1.0), 'temperature': jnp.array(5000.0)}
        result = Blackbody(pivot_micron=1.0).evaluate(wl, 1.0, params)
        assert jnp.all(result > 0)

    def test_roundtrip(self):
        f = Blackbody(pivot_micron=1.5)
        f2 = form_from_dict(f.to_dict())
        assert f2.pivot_micron == pytest.approx(1.5)
        assert f2 == f


class TestModifiedBlackbody:
    def test_param_names(self):
        assert set(ModifiedBlackbody().param_names()) == {
            'amplitude',
            'temperature',
            'beta',
        }

    def test_roundtrip(self):
        f = ModifiedBlackbody(pivot_micron=2.0)
        f2 = form_from_dict(f.to_dict())
        assert f2 == f


class TestAttenuatedBlackbody:
    def test_param_names(self):
        assert set(AttenuatedBlackbody().param_names()) == {
            'amplitude',
            'temperature',
            'tau_v',
            'alpha',
        }

    def test_roundtrip(self):
        f = AttenuatedBlackbody(pivot_micron=1.5, lambda_v_micron=0.6)
        f2 = form_from_dict(f.to_dict())
        assert f2.pivot_micron == pytest.approx(1.5)
        assert f2.lambda_v_micron == pytest.approx(0.6)
        assert f2 == f

    def test_equality_checks_both_params(self):
        assert AttenuatedBlackbody(1.0, 0.55) != AttenuatedBlackbody(1.0, 0.50)


class TestBSpline:
    @pytest.fixture
    def cubic_knots(self):
        # Clamped cubic knot vector: repeat end knots degree+1 times.
        return jnp.array([0.9, 0.9, 0.9, 0.9, 1.0, 1.1, 1.1, 1.1, 1.1])

    def test_n_basis(self, cubic_knots):
        b = BSpline(cubic_knots, degree=3)
        assert b.n_basis == 5  # 9 knots - 3 - 1

    def test_param_names(self, cubic_knots):
        b = BSpline(cubic_knots, degree=3)
        assert b.param_names() == tuple(f'coeff_{i}' for i in range(5))

    def test_evaluate_shape(self, cubic_knots):
        b = BSpline(cubic_knots, degree=3)
        params = {f'coeff_{i}': jnp.array(1.0) for i in range(b.n_basis)}
        result = b.evaluate(_WL, _CENTER, params)
        assert result.shape == _WL.shape

    def test_roundtrip(self, cubic_knots):
        b = BSpline(cubic_knots, degree=3)
        b2 = form_from_dict(b.to_dict())
        assert b2 == b
        assert b2.n_basis == b.n_basis


class TestBernstein:
    @pytest.fixture
    def bernstein(self):
        return Bernstein(degree=3, wavelength_min=0.9, wavelength_max=1.1)

    def test_param_names(self, bernstein):
        assert bernstein.param_names() == tuple(f'coeff_{i}' for i in range(4))

    def test_evaluate_shape(self, bernstein):
        params = {f'coeff_{i}': jnp.array(1.0) for i in range(4)}
        result = bernstein.evaluate(_WL, _CENTER, params)
        assert result.shape == _WL.shape

    def test_evaluate_nonnegative_with_positive_coeffs(self, bernstein):
        params = {f'coeff_{i}': jnp.array(1.0) for i in range(4)}
        result = bernstein.evaluate(_WL, _CENTER, params)
        assert jnp.all(result >= 0)

    def test_roundtrip(self, bernstein):
        b2 = form_from_dict(bernstein.to_dict())
        assert b2.degree == 3
        assert b2 == bernstein

    def test_equality_checks_bounds(self):
        b1 = Bernstein(3, 0.9, 1.1)
        b2 = Bernstein(3, 0.9, 1.2)
        assert b1 != b2
