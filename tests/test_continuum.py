"""Tests for unite.continuum (forms, regions, and configuration)."""

import jax.numpy as jnp
import pytest
from astropy import units as u

from unite.continuum import (
    AttenuatedBlackbody,
    Bernstein,
    Blackbody,
    BSpline,
    Chebyshev,
    ContinuumConfiguration,
    ContinuumNormalizationWavelength,
    ContinuumRegion,
    ContinuumScale,
    Linear,
    ModifiedBlackbody,
    Polynomial,
    PowerLaw,
    form_from_dict,
)
from unite.prior import Fixed, Parameter, Uniform

# ---------------------------------------------------------------------------
# Parameter
# ---------------------------------------------------------------------------


class TestParameterAsContinuumToken:
    def test_custom_prior(self):
        p = Parameter('amp', prior=Uniform(0, 10))
        assert p.prior.low == pytest.approx(0.0)
        assert p.prior.high == pytest.approx(10.0)

    def test_name_stored(self):
        p = Parameter('beta', prior=Uniform(-5, 5))
        assert p.name == 'beta'

    def test_repr(self):
        p = Parameter('x', prior=Uniform(0, 1))
        assert 'x' in repr(p)


# ---------------------------------------------------------------------------
# ContinuumScale
# ---------------------------------------------------------------------------


class TestContinuumScale:
    def test_default_prior_is_uniform_positive(self):
        tok = ContinuumScale('s')
        assert isinstance(tok.prior, Uniform)
        assert tok.prior.low == pytest.approx(0.0)
        assert tok.prior.high == pytest.approx(10.0)

    def test_custom_prior(self):
        tok = ContinuumScale('s', prior=Uniform(0, 5))
        assert tok.prior.high == pytest.approx(5.0)

    def test_name_stored(self):
        tok = ContinuumScale('my_scale')
        assert tok.name == 'my_scale'

    def test_is_subclass_of_continuum_param(self):
        assert isinstance(ContinuumScale('s'), Parameter)

    def test_wrong_slot_raises(self):
        tok = ContinuumScale('s')
        region = ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear(), params={'slope': tok})
        with pytest.raises(ValueError, match=r'ContinuumScale.*"scale"'):
            ContinuumConfiguration([region])

    def test_correct_slot_ok(self):
        tok = ContinuumScale('s', prior=Uniform(0, 5))
        region = ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear(), params={'scale': tok})
        config = ContinuumConfiguration([region])
        assert config.resolved_params[0]['scale'] is tok

    def test_scale_slot_accepts_generic_continuum_param(self):
        # Generic Parameter is allowed in any slot including 'scale'.
        tok = Parameter('generic', prior=Uniform(0, 5))
        region = ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear(), params={'scale': tok})
        config = ContinuumConfiguration([region])
        assert config.resolved_params[0]['scale'] is tok


# ---------------------------------------------------------------------------
# ContinuumNormalizationWavelength
# ---------------------------------------------------------------------------


class TestContinuumNormalizationWavelength:
    def test_default_prior_is_fixed(self):
        tok = ContinuumNormalizationWavelength('nw')
        assert isinstance(tok.prior, Fixed)

    def test_custom_prior(self):
        tok = ContinuumNormalizationWavelength('nw', prior=Fixed(2.5))
        assert tok.prior.value == pytest.approx(2.5)

    def test_name_stored(self):
        tok = ContinuumNormalizationWavelength('my_nw')
        assert tok.name == 'my_nw'

    def test_is_subclass_of_continuum_param(self):
        assert isinstance(ContinuumNormalizationWavelength('nw'), Parameter)

    def test_wrong_slot_raises(self):
        tok = ContinuumNormalizationWavelength('nw', prior=Fixed(1.5))
        region = ContinuumRegion(1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'scale': tok})
        with pytest.raises(ValueError, match=r'ContinuumNormalizationWavelength.*"normalization_wavelength"'):
            ContinuumConfiguration([region])

    def test_correct_slot_ok(self):
        tok = ContinuumNormalizationWavelength('nw', prior=Fixed(1.5))
        region = ContinuumRegion(1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'normalization_wavelength': tok})
        config = ContinuumConfiguration([region])
        assert config.resolved_params[0]['normalization_wavelength'] is tok

    def test_normalization_wavelength_slot_accepts_generic_continuum_param(self):
        # Generic Parameter is allowed in any slot.
        tok = Parameter('generic', prior=Fixed(1.5))
        region = ContinuumRegion(1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'normalization_wavelength': tok})
        config = ContinuumConfiguration([region])
        assert config.resolved_params[0]['normalization_wavelength'] is tok


# ---------------------------------------------------------------------------
# ContinuumRegion
# ---------------------------------------------------------------------------


class TestContinuumRegion:
    def test_construction(self):
        r = ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())
        assert r.low == pytest.approx(1.0)
        assert r.high == pytest.approx(2.0)

    def test_center(self):
        r = ContinuumRegion(1.0 * u.um, 3.0 * u.um)
        assert r.center == pytest.approx(2.0)

    def test_low_ge_high_raises(self):
        with pytest.raises(ValueError, match='low must be < high'):
            ContinuumRegion(5.0 * u.um, 3.0 * u.um)

    def test_low_equals_high_raises(self):
        with pytest.raises(ValueError, match='low must be < high'):
            ContinuumRegion(3.0 * u.um, 3.0 * u.um)

    def test_no_units_raises(self):
        with pytest.raises(TypeError, match='must be an astropy Quantity'):
            ContinuumRegion(1.0, 2.0)

    def test_wrong_units_raises(self):
        with pytest.raises(ValueError, match=r'wavelength.*units'):
            ContinuumRegion(1.0 * u.kg, 2.0 * u.kg)

    def test_unit_conversion(self):
        # high in AA, low in um — should convert high to um.
        r = ContinuumRegion(1.0 * u.um, 20000.0 * u.AA)
        assert r.low == pytest.approx(1.0)
        assert r.high == pytest.approx(2.0)

    def test_default_form_is_linear(self):
        r = ContinuumRegion(1.0 * u.um, 2.0 * u.um)
        assert isinstance(r.form, Linear)

    def test_params_stored(self):
        p = Parameter('my_slope', prior=Uniform(-5, 5))
        r = ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear(), params={'slope': p})
        assert r.params['slope'] is p


# ---------------------------------------------------------------------------
# ContinuumConfiguration construction
# ---------------------------------------------------------------------------


class TestContinuumConfigurationConstruction:
    def test_empty(self):
        config = ContinuumConfiguration()
        assert len(config) == 0

    def test_sorted_on_input(self):
        r1 = ContinuumRegion(6.0 * u.um, 7.0 * u.um)
        r2 = ContinuumRegion(1.0 * u.um, 2.0 * u.um)
        config = ContinuumConfiguration([r1, r2])
        assert config[0].low == pytest.approx(1.0)
        assert config[1].low == pytest.approx(6.0)

    def test_overlap_warns(self):
        r1 = ContinuumRegion(1.0 * u.um, 3.0 * u.um)
        r2 = ContinuumRegion(2.5 * u.um, 4.0 * u.um)
        with pytest.warns(UserWarning, match='overlap'):
            ContinuumConfiguration([r1, r2])

    def test_overlapping_regions_still_constructed(self):
        r1 = ContinuumRegion(1.0 * u.um, 3.0 * u.um)
        r2 = ContinuumRegion(2.5 * u.um, 4.0 * u.um)
        with pytest.warns(UserWarning):
            config = ContinuumConfiguration([r1, r2])
        assert len(config) == 2

    def test_wide_region_warns_against_all_overlapping(self):
        # A wide region [1, 10] should warn against both [3, 5] and [7, 9].
        r_wide = ContinuumRegion(1.0 * u.um, 10.0 * u.um)
        r_mid = ContinuumRegion(3.0 * u.um, 5.0 * u.um)
        r_late = ContinuumRegion(7.0 * u.um, 9.0 * u.um)
        with pytest.warns(UserWarning):
            ContinuumConfiguration([r_wide, r_mid, r_late])

    def test_touching_boundaries_allowed(self):
        r1 = ContinuumRegion(1.0 * u.um, 2.0 * u.um)
        r2 = ContinuumRegion(2.0 * u.um, 3.0 * u.um)
        config = ContinuumConfiguration([r1, r2])
        assert len(config) == 2

    def test_iter_and_getitem(self):
        regions = [ContinuumRegion(float(i) * u.um, (float(i) + 0.5) * u.um) for i in range(3)]
        config = ContinuumConfiguration(regions)
        assert list(config) == list(regions)
        assert config[1].low == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Auto-naming via resolved_params
# ---------------------------------------------------------------------------


class TestResolvedParams:
    def test_resolved_params_length_matches_regions(self):
        config = ContinuumConfiguration([
            ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear()),
            ContinuumRegion(3.0 * u.um, 4.0 * u.um, Linear()),
        ])
        assert len(config.resolved_params) == 2

    def test_auto_names_use_form_type_and_index(self):
        config = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())])
        resolved = config.resolved_params[0]
        assert 'scale' in resolved
        assert resolved['scale'].name == 'cont_linear_0_scale'
        assert resolved['slope'].name == 'cont_linear_0_slope'
        assert resolved['normalization_wavelength'].name == 'cont_linear_0_normalization_wavelength'

    def test_two_same_type_regions_get_different_indices(self):
        config = ContinuumConfiguration([
            ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear()),
            ContinuumRegion(3.0 * u.um, 4.0 * u.um, Linear()),
        ])
        r0 = config.resolved_params[0]
        r1 = config.resolved_params[1]
        assert r0['scale'].name == 'cont_linear_0_scale'
        assert r1['scale'].name == 'cont_linear_1_scale'
        # Must be distinct token objects.
        assert r0['scale'] is not r1['scale']

    def test_normalization_wavelength_default_fixed_at_region_center(self):
        config = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())])
        nw_tok = config.resolved_params[0]['normalization_wavelength']
        assert isinstance(nw_tok.prior, Fixed)
        assert nw_tok.prior.value == pytest.approx(1.5)  # center of [1, 2] um

    def test_explicit_token_used_as_is(self):
        scale = Parameter('my_scale', prior=Uniform(0, 5))
        config = ContinuumConfiguration([
            ContinuumRegion(1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'scale': scale}),
        ])
        resolved = config.resolved_params[0]
        assert resolved['scale'] is scale
        # beta and normalization_wavelength should be auto-created.
        assert resolved['beta'].name == 'cont_powerlaw_0_beta'
        assert resolved['normalization_wavelength'].name == 'cont_powerlaw_0_normalization_wavelength'
        # normalization_wavelength default is Fixed at region center.
        assert isinstance(resolved['normalization_wavelength'].prior, Fixed)
        assert resolved['normalization_wavelength'].prior.value == pytest.approx(1.5)

    def test_shared_tokens_are_same_object(self):
        scale = Parameter('pl_scale', prior=Uniform(0, 10))
        beta = Parameter('pl_beta', prior=Uniform(-5, 5))
        nw = Parameter('pl_nw', prior=Fixed(2.5))
        pl = PowerLaw()
        config = ContinuumConfiguration([
            ContinuumRegion(1.0 * u.um, 2.0 * u.um, pl, params={'scale': scale, 'beta': beta, 'normalization_wavelength': nw}),
            ContinuumRegion(3.0 * u.um, 4.0 * u.um, pl, params={'scale': scale, 'beta': beta, 'normalization_wavelength': nw}),
        ])
        r0 = config.resolved_params[0]
        r1 = config.resolved_params[1]
        assert r0['scale'] is r1['scale']
        assert r0['beta'] is r1['beta']
        assert r0['normalization_wavelength'] is r1['normalization_wavelength']

    def test_unique_param_count_with_sharing(self):
        scale = Parameter('pl_scale', prior=Uniform(0, 10))
        beta = Parameter('pl_beta', prior=Uniform(-5, 5))
        nw = Parameter('pl_nw', prior=Fixed(2.5))
        pl = PowerLaw()
        config = ContinuumConfiguration([
            ContinuumRegion(1.0 * u.um, 2.0 * u.um, pl, params={'scale': scale, 'beta': beta, 'normalization_wavelength': nw}),
            ContinuumRegion(3.0 * u.um, 4.0 * u.um, pl, params={'scale': scale, 'beta': beta, 'normalization_wavelength': nw}),
        ])
        # 3 shared tokens → repr says 3 parameters, not 6.
        assert '3 parameter(s)' in repr(config)

    def test_resolved_params_is_copy(self):
        config = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())])
        rp1 = config.resolved_params
        rp2 = config.resolved_params
        assert rp1 is not rp2  # New list each time.
        assert rp1[0] is not rp2[0]  # Inner dicts are copies too.


# ---------------------------------------------------------------------------
# ContinuumConfiguration.from_lines
# ---------------------------------------------------------------------------


class TestFromLines:
    def test_basic(self):
        config = ContinuumConfiguration.from_lines([5000.0, 6560.0] * u.AA)
        assert len(config) == 2

    def test_all_regions_share_form_object(self):
        # from_lines creates one form instance for efficiency — form objects
        # are stateless so sharing is harmless.
        config = ContinuumConfiguration.from_lines([5000.0, 6560.0] * u.AA)
        assert config[0].form is config[1].form

    def test_all_regions_have_independent_params(self):
        # Even though form is shared, each region's params are independent.
        config = ContinuumConfiguration.from_lines([5000.0, 6560.0] * u.AA)
        r0 = config.resolved_params[0]
        r1 = config.resolved_params[1]
        assert r0['scale'] is not r1['scale']
        assert r0['slope'] is not r1['slope']

    def test_overlapping_lines_merged(self):
        # Two lines close together → their padded windows merge into one region.
        config = ContinuumConfiguration.from_lines([6549.86, 6585.27] * u.AA, pad=0.05)
        assert len(config) == 1

    def test_empty_raises(self):
        with pytest.raises(ValueError, match='must not be empty'):
            ContinuumConfiguration.from_lines([] * u.AA)

    def test_no_units_raises(self):
        with pytest.raises(TypeError, match='must be an astropy Quantity'):
            ContinuumConfiguration.from_lines([5000.0, 6560.0])

    def test_wrong_units_raises(self):
        with pytest.raises(ValueError, match=r'wavelength.*units'):
            ContinuumConfiguration.from_lines([5000.0, 6560.0] * u.km / u.s)

    def test_custom_form(self):
        pl = PowerLaw()
        config = ContinuumConfiguration.from_lines([5000.0] * u.AA, form=pl)
        assert config[0].form is pl

    def test_regions_sorted(self):
        config = ContinuumConfiguration.from_lines([6560.0, 4860.0] * u.AA)
        assert config[0].low < config[1].low


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    def test_linear_region(self):
        config = ContinuumConfiguration([ContinuumRegion(4600.0 * u.AA, 5200.0 * u.AA, Linear())])
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        assert len(config2) == 1
        assert isinstance(config2[0].form, Linear)
        assert config2[0].low == pytest.approx(4600.0)

    def test_dict_has_params_forms_regions_keys(self):
        config = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())])
        d = config.to_dict()
        assert 'params' in d
        assert 'forms' in d
        assert 'regions' in d

    def test_dict_regions_have_wavelength_unit(self):
        config = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())])
        d = config.to_dict()
        assert 'wavelength_unit' in d['regions'][0]

    def test_auto_param_names_roundtrip(self):
        config = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())])
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        resolved = config2.resolved_params[0]
        assert resolved['scale'].name == 'cont_linear_0_scale'
        assert isinstance(resolved['scale'].prior, Uniform)

    def test_shared_form_object_preserved_after_roundtrip(self):
        # Regions sharing the same form Python object should still share it
        # after serialization (forms are de-duplicated in the forms section).
        pl = PowerLaw()
        config = ContinuumConfiguration([
            ContinuumRegion(1.0 * u.um, 1.5 * u.um, pl),
            ContinuumRegion(2.0 * u.um, 2.5 * u.um, pl),
            ContinuumRegion(3.0 * u.um, 3.5 * u.um, pl),
        ])
        d = config.to_dict()
        assert len(d['forms']) == 1  # one form entry for all three regions.
        config2 = ContinuumConfiguration.from_dict(d)
        assert config2[0].form is config2[1].form
        assert config2[1].form is config2[2].form

    def test_distinct_forms_serialized_separately(self):
        config = ContinuumConfiguration([
            ContinuumRegion(1.0 * u.um, 1.5 * u.um, Linear()),
            ContinuumRegion(2.0 * u.um, 2.5 * u.um, PowerLaw()),
        ])
        d = config.to_dict()
        assert len(d['forms']) == 2

    def test_explicit_param_token_roundtrip(self):
        scale = Parameter('pl_scale', prior=Uniform(0, 8))
        r = ContinuumRegion(1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'scale': scale})
        config = ContinuumConfiguration([r])
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        resolved = config2.resolved_params[0]
        assert resolved['scale'].name == 'pl_scale'
        assert isinstance(resolved['scale'].prior, Uniform)
        assert resolved['scale'].prior.high == pytest.approx(8.0)

    def test_shared_token_roundtrip(self):
        scale = Parameter('pl_scale', prior=Uniform(0, 10))
        beta = Parameter('pl_beta', prior=Uniform(-5, 5))
        nw = Parameter('pl_nw', prior=Fixed(2.5))
        pl = PowerLaw()
        config = ContinuumConfiguration([
            ContinuumRegion(1.0 * u.um, 2.0 * u.um, pl, params={'scale': scale, 'beta': beta, 'normalization_wavelength': nw}),
            ContinuumRegion(3.0 * u.um, 4.0 * u.um, pl, params={'scale': scale, 'beta': beta, 'normalization_wavelength': nw}),
        ])
        d = config.to_dict()
        # Three shared tokens → three param entries.
        assert len(d['params']) == 3
        config2 = ContinuumConfiguration.from_dict(d)
        r0 = config2.resolved_params[0]
        r1 = config2.resolved_params[1]
        # After round-trip: same name → same object.
        assert r0['scale'] is r1['scale']
        assert r0['beta'] is r1['beta']
        assert r0['normalization_wavelength'] is r1['normalization_wavelength']
        assert isinstance(r0['normalization_wavelength'].prior, Fixed)
        assert r0['normalization_wavelength'].prior.value == pytest.approx(2.5)

    def test_polynomial_degree_preserved(self):
        config = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 2.0 * u.um, Polynomial(degree=3))])
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        assert config2[0].form.degree == 3

    def test_from_lines_roundtrip(self):
        config = ContinuumConfiguration.from_lines([5000.0, 6560.0] * u.AA)
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        assert len(config2) == len(config)
        # Params are independent after round-trip.
        assert config2.resolved_params[0]['scale'] is not config2.resolved_params[1]['scale']

    def test_wavelength_unit_preserved_in_roundtrip(self):
        config = ContinuumConfiguration([ContinuumRegion(4600.0 * u.AA, 5200.0 * u.AA, Linear())])
        d = config.to_dict()
        assert d['regions'][0]['wavelength_unit'] == str(u.AA)
        config2 = ContinuumConfiguration.from_dict(d)
        assert config2[0].low == pytest.approx(4600.0)
        assert config2[0]._unit.is_equivalent(u.AA)

    def test_powerlaw_normalization_wavelength_param_roundtrip(self):
        nw = Parameter('my_nw', prior=Fixed(3.5))
        config = ContinuumConfiguration([
            ContinuumRegion(1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'normalization_wavelength': nw}),
        ])
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        resolved = config2.resolved_params[0]
        assert resolved['normalization_wavelength'].name == 'my_nw'
        assert isinstance(resolved['normalization_wavelength'].prior, Fixed)
        assert resolved['normalization_wavelength'].prior.value == pytest.approx(3.5)


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

    def test_power_law_no_constructor_args(self):
        f = form_from_dict({'type': 'PowerLaw'})
        assert isinstance(f, PowerLaw)
        assert f == PowerLaw()

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
        assert Linear().param_names() == ('scale', 'slope', 'normalization_wavelength')

    def test_n_params(self):
        assert Linear().n_params == 3

    def test_default_priors_keys(self):
        assert set(Linear().default_priors()) == {'scale', 'slope', 'normalization_wavelength'}

    def test_default_priors_normalization_wavelength_uses_region_center(self):
        priors = Linear().default_priors(region_center=2.5)
        assert isinstance(priors['normalization_wavelength'], Fixed)
        assert priors['normalization_wavelength'].value == pytest.approx(2.5)

    def test_evaluate_shape(self):
        params = {
            'scale': jnp.array(1.0),
            'slope': jnp.array(0.0),
            'normalization_wavelength': jnp.array(_CENTER),
        }
        result = Linear().evaluate(_WL, _CENTER, params)
        assert result.shape == _WL.shape

    def test_evaluate_flat_at_zero_slope(self):
        params = {
            'scale': jnp.array(2.0),
            'slope': jnp.array(0.0),
            'normalization_wavelength': jnp.array(_CENTER),
        }
        result = Linear().evaluate(_WL, _CENTER, params)
        assert jnp.allclose(result, 2.0)

    def test_evaluate_equals_scale_at_normalization_wavelength(self):
        nw = 1.05
        params = {
            'scale': jnp.array(3.0),
            'slope': jnp.array(5.0),
            'normalization_wavelength': jnp.array(nw),
        }
        val = Linear().evaluate(jnp.array([nw]), _CENTER, params)
        assert val[0] == pytest.approx(3.0)

    def test_roundtrip(self):
        f = Linear()
        assert form_from_dict(f.to_dict()) == f

    def test_equality(self):
        assert Linear() == Linear()


class TestPowerLaw:
    def test_param_names(self):
        assert PowerLaw().param_names() == ('scale', 'beta', 'normalization_wavelength')

    def test_default_priors_normalization_wavelength_uses_region_center(self):
        priors = PowerLaw().default_priors(region_center=3.5)
        assert isinstance(priors['normalization_wavelength'], Fixed)
        assert priors['normalization_wavelength'].value == pytest.approx(3.5)

    def test_default_priors_normalization_wavelength_default_region_center(self):
        priors = PowerLaw().default_priors()
        assert isinstance(priors['normalization_wavelength'], Fixed)
        assert priors['normalization_wavelength'].value == pytest.approx(1.0)

    def test_evaluate_uses_normalization_wavelength(self):
        params = {
            'scale': jnp.array(1.0),
            'beta': jnp.array(1.0),
            'normalization_wavelength': jnp.array(2.0),
        }
        result = PowerLaw().evaluate(_WL, _CENTER, params)
        expected = _WL / 2.0
        assert jnp.allclose(result, expected)

    def test_evaluate_shape(self):
        params = {
            'scale': jnp.array(1.0),
            'beta': jnp.array(0.0),
            'normalization_wavelength': jnp.array(_CENTER),
        }
        result = PowerLaw().evaluate(_WL, _CENTER, params)
        assert result.shape == _WL.shape

    def test_evaluate_flat_at_zero_beta(self):
        params = {
            'scale': jnp.array(2.0),
            'beta': jnp.array(0.0),
            'normalization_wavelength': jnp.array(_CENTER),
        }
        result = PowerLaw().evaluate(_WL, _CENTER, params)
        assert jnp.allclose(result, 2.0)

    def test_evaluate_equals_scale_at_normalization_wavelength(self):
        nw = 1.05
        params = {
            'scale': jnp.array(3.7),
            'beta': jnp.array(2.0),
            'normalization_wavelength': jnp.array(nw),
        }
        val = PowerLaw().evaluate(jnp.array([nw]), _CENTER, params)
        assert val[0] == pytest.approx(3.7)

    def test_roundtrip(self):
        f = PowerLaw()
        f2 = form_from_dict(f.to_dict())
        assert f2 == f

    def test_equality(self):
        assert PowerLaw() == PowerLaw()

    def test_repr(self):
        assert repr(PowerLaw()) == 'PowerLaw()'


class TestPolynomial:
    def test_param_names_degree0(self):
        assert Polynomial(degree=0).param_names() == ('scale', 'normalization_wavelength')

    def test_param_names_degree1(self):
        assert Polynomial(degree=1).param_names() == ('scale', 'c1', 'normalization_wavelength')

    def test_param_names_degree2(self):
        assert Polynomial(degree=2).param_names() == ('scale', 'c1', 'c2', 'normalization_wavelength')

    def test_negative_degree_raises(self):
        with pytest.raises(ValueError, match='>= 0'):
            Polynomial(degree=-1)

    def test_evaluate_shape(self):
        p = Polynomial(degree=2)
        params = {k: jnp.array(1.0) for k in p.param_names()}
        result = p.evaluate(_WL, _CENTER, params)
        assert result.shape == _WL.shape

    def test_evaluate_equals_scale_at_normalization_wavelength(self):
        nw = 1.0
        params = {
            'scale': jnp.array(4.0),
            'c1': jnp.array(2.0),
            'c2': jnp.array(3.0),
            'normalization_wavelength': jnp.array(nw),
        }
        val = Polynomial(degree=2).evaluate(jnp.array([nw]), _CENTER, params)
        assert val[0] == pytest.approx(4.0)

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
    def test_param_names_order0(self):
        assert Chebyshev(order=0).param_names() == ('scale', 'normalization_wavelength')

    def test_param_names_order2(self):
        assert Chebyshev(order=2).param_names() == ('scale', 'c1', 'c2', 'normalization_wavelength')

    def test_negative_order_raises(self):
        with pytest.raises(ValueError, match='>= 0'):
            Chebyshev(order=-1)

    def test_evaluate_shape(self):
        f = Chebyshev(order=2, half_width=0.1)
        params = {k: jnp.array(0.0) for k in f.param_names()}
        params['scale'] = jnp.array(1.0)
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
        assert set(Blackbody().param_names()) == {'scale', 'temperature', 'normalization_wavelength'}

    def test_default_priors_normalization_wavelength_uses_region_center(self):
        priors = Blackbody().default_priors(region_center=2.0)
        assert isinstance(priors['normalization_wavelength'], Fixed)
        assert priors['normalization_wavelength'].value == pytest.approx(2.0)

    def test_evaluate_shape(self):
        wl = jnp.linspace(0.5, 2.0, 30)
        params = {
            'scale': jnp.array(1.0),
            'temperature': jnp.array(5000.0),
            'normalization_wavelength': jnp.array(1.0),
        }
        result = Blackbody().evaluate(wl, 1.0, params)
        assert result.shape == wl.shape

    def test_evaluate_positive(self):
        wl = jnp.linspace(0.5, 2.0, 30)
        params = {
            'scale': jnp.array(1.0),
            'temperature': jnp.array(5000.0),
            'normalization_wavelength': jnp.array(1.0),
        }
        result = Blackbody().evaluate(wl, 1.0, params)
        assert jnp.all(result > 0)

    def test_evaluate_equals_scale_at_normalization_wavelength(self):
        nw = 1.0
        params = {
            'scale': jnp.array(2.5),
            'temperature': jnp.array(5000.0),
            'normalization_wavelength': jnp.array(nw),
        }
        val = Blackbody().evaluate(jnp.array([nw]), nw, params)
        assert val[0] == pytest.approx(2.5, rel=1e-5)

    def test_roundtrip(self):
        f = Blackbody()
        f2 = form_from_dict(f.to_dict())
        assert f2 == f


class TestModifiedBlackbody:
    def test_param_names(self):
        assert set(ModifiedBlackbody().param_names()) == {
            'scale',
            'temperature',
            'beta',
            'normalization_wavelength',
        }

    def test_evaluate_equals_scale_at_normalization_wavelength(self):
        nw = 1.0
        params = {
            'scale': jnp.array(3.0),
            'temperature': jnp.array(5000.0),
            'beta': jnp.array(1.5),
            'normalization_wavelength': jnp.array(nw),
        }
        val = ModifiedBlackbody().evaluate(jnp.array([nw]), nw, params)
        assert val[0] == pytest.approx(3.0, rel=1e-5)

    def test_roundtrip(self):
        f = ModifiedBlackbody()
        f2 = form_from_dict(f.to_dict())
        assert f2 == f


class TestAttenuatedBlackbody:
    def test_param_names(self):
        assert set(AttenuatedBlackbody().param_names()) == {
            'scale',
            'temperature',
            'tau_v',
            'alpha',
            'normalization_wavelength',
        }

    def test_lambda_v_micron_stored(self):
        f = AttenuatedBlackbody(lambda_v_micron=0.6)
        assert f.lambda_v_micron == pytest.approx(0.6)

    def test_evaluate_equals_scale_at_normalization_wavelength(self):
        nw = 1.0
        params = {
            'scale': jnp.array(1.5),
            'temperature': jnp.array(5000.0),
            'tau_v': jnp.array(0.5),
            'alpha': jnp.array(-1.5),
            'normalization_wavelength': jnp.array(nw),
        }
        val = AttenuatedBlackbody().evaluate(jnp.array([nw]), nw, params)
        assert val[0] == pytest.approx(1.5, rel=1e-5)

    def test_roundtrip(self):
        f = AttenuatedBlackbody(lambda_v_micron=0.6)
        f2 = form_from_dict(f.to_dict())
        assert f2.lambda_v_micron == pytest.approx(0.6)
        assert f2 == f

    def test_equality_checks_lambda_v(self):
        assert AttenuatedBlackbody(0.55) != AttenuatedBlackbody(0.50)


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
        expected = ('scale', 'coeff_1', 'coeff_2', 'coeff_3', 'coeff_4', 'normalization_wavelength')
        assert b.param_names() == expected

    def test_evaluate_shape(self, cubic_knots):
        b = BSpline(cubic_knots, degree=3)
        params = {n: jnp.array(1.0) for n in b.param_names()}
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
        expected = ('scale', 'coeff_1', 'coeff_2', 'coeff_3', 'normalization_wavelength')
        assert bernstein.param_names() == expected

    def test_evaluate_shape(self, bernstein):
        params = {n: jnp.array(1.0) for n in bernstein.param_names()}
        result = bernstein.evaluate(_WL, _CENTER, params)
        assert result.shape == _WL.shape

    def test_evaluate_nonnegative_with_positive_coeffs(self, bernstein):
        params = {n: jnp.array(1.0) for n in bernstein.param_names()}
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
