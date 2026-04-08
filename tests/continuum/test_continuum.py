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
    ContinuumRegion,
    ContShape,
    Linear,
    ModifiedBlackbody,
    NormWavelength,
    Polynomial,
    PowerLaw,
    Scale,
    form_from_dict,
    get_form,
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
        # label stores user-supplied name; site name is set at registration
        assert p.label == 'beta'

    def test_repr(self):
        p = Parameter('x', prior=Uniform(0, 1))
        assert 'x' in repr(p)


# ---------------------------------------------------------------------------
# ContinuumScale
# ---------------------------------------------------------------------------


class TestScale:
    def test_default_prior_is_uniform_positive(self):
        tok = Scale('s')
        assert isinstance(tok.prior, Uniform)
        assert tok.prior.low == pytest.approx(0.0)
        assert tok.prior.high == pytest.approx(2.0)

    def test_custom_prior(self):
        tok = Scale('s', prior=Uniform(0, 5))
        assert tok.prior.high == pytest.approx(5.0)

    def test_name_stored(self):
        tok = Scale('my_scale')
        # label stores the user-supplied label; site name is set when attached to a region
        assert tok.label == 'my_scale'

    def test_is_subclass_of_parameter(self):
        assert isinstance(Scale('s'), Parameter)

    def test_wrong_slot_raises(self):
        tok = Scale('s')
        region = ContinuumRegion(
            1.0 * u.um, 2.0 * u.um, Linear(), params={'angle': tok}
        )
        with pytest.raises(TypeError, match=r'must be a ContShape'):
            ContinuumConfiguration([region])

    def test_correct_slot_ok(self):
        tok = Scale('s', prior=Uniform(0, 5))
        region = ContinuumRegion(
            1.0 * u.um, 2.0 * u.um, Linear(), params={'scale': tok}
        )
        config = ContinuumConfiguration([region])
        assert config.resolved_params[0]['scale'] is tok

    def test_scale_slot_rejects_generic_param(self):
        # Generic Parameter is no longer allowed; must be typed.
        tok = Parameter('generic', prior=Uniform(0, 5))
        region = ContinuumRegion(
            1.0 * u.um, 2.0 * u.um, Linear(), params={'scale': tok}
        )
        with pytest.raises(TypeError, match='must be a Scale'):
            ContinuumConfiguration([region])


# ---------------------------------------------------------------------------
# ContinuumNormalizationWavelength
# ---------------------------------------------------------------------------


class TestNormWavelength:
    def test_default_prior_is_fixed(self):
        tok = NormWavelength('nw')
        assert isinstance(tok.prior, Fixed)
        assert tok.prior.value == pytest.approx(1.0)

    def test_custom_prior(self):
        tok = NormWavelength('nw', prior=Fixed(2.5))
        assert tok.prior.value == pytest.approx(2.5)

    def test_name_stored(self):
        tok = NormWavelength('my_nw')
        # label stores the user-supplied label; site name is set when attached to a region
        assert tok.label == 'my_nw'

    def test_is_subclass_of_parameter(self):
        assert isinstance(NormWavelength('nw'), Parameter)

    def test_wrong_slot_raises(self):
        tok = NormWavelength('nw', prior=Fixed(1.5))
        region = ContinuumRegion(
            1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'scale': tok}
        )
        with pytest.raises(TypeError, match=r'must be a Scale'):
            ContinuumConfiguration([region])

    def test_correct_slot_ok(self):
        tok = NormWavelength('nw', prior=Fixed(1.5))
        region = ContinuumRegion(
            1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'norm_wav': tok}
        )
        config = ContinuumConfiguration([region])
        assert config.resolved_params[0]['norm_wav'] is tok

    def test_norm_wav_slot_rejects_generic_param(self):
        # Generic Parameter is no longer allowed; must be typed.
        tok = Parameter('generic', prior=Fixed(1.5))
        region = ContinuumRegion(
            1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'norm_wav': tok}
        )
        with pytest.raises(TypeError, match='must be a NormWavelength'):
            ContinuumConfiguration([region])


# ---------------------------------------------------------------------------
# ContShape
# ---------------------------------------------------------------------------


class TestContShape:
    def test_default_prior(self):
        tok = ContShape('beta')
        assert isinstance(tok.prior, Uniform)
        assert tok.prior.low == pytest.approx(-10.0)
        assert tok.prior.high == pytest.approx(10.0)

    def test_custom_prior(self):
        tok = ContShape('beta', prior=Uniform(-5, 5))
        assert tok.prior.low == pytest.approx(-5.0)
        assert tok.prior.high == pytest.approx(5.0)

    def test_name_stored(self):
        tok = ContShape('my_beta')
        # label stores the user-supplied label; site name is set when attached to a region
        assert tok.label == 'my_beta'

    def test_is_subclass_of_parameter(self):
        assert isinstance(ContShape('beta'), Parameter)

    def test_wrong_slot_for_contshape_raises(self):
        # ContShape should not be used in 'scale' or 'norm_wav' slots
        tok = ContShape('not_a_beta')
        region = ContinuumRegion(
            1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'scale': tok}
        )
        with pytest.raises(TypeError, match='must be a Scale'):
            ContinuumConfiguration([region])

    def test_correct_slot_for_form_param(self):
        tok = ContShape('beta', prior=Uniform(-5, 5))
        region = ContinuumRegion(
            1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'beta': tok}
        )
        config = ContinuumConfiguration([region])
        assert config.resolved_params[0]['beta'] is tok


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
        with pytest.raises(ValueError, match=r'low.*units'):
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
        p = Parameter('my_angle', prior=Uniform(-5, 5))
        r = ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear(), params={'angle': p})
        assert r.params['angle'] is p

    def test_bspline_knots_within_region_ok(self):
        """BSpline knots within region bounds should not raise."""
        knots = [1.0] * u.um
        r = ContinuumRegion(0.9 * u.um, 1.1 * u.um, BSpline(knots, degree=3))
        assert r.form.n_basis == 5

    def test_bspline_knots_not_array_raises(self):
        """BSpline knots within region bounds should not raise."""
        knots = 1.0 * u.um
        with pytest.raises(ValueError, match=r'knots must be 1-D, got 0-D array.'):
            ContinuumRegion(0.9 * u.um, 1.1 * u.um, BSpline(knots, degree=3))

    def test_bspline_knots_outside_region_raises(self):
        """BSpline knots outside region bounds should raise."""
        knots = [0.8] * u.um
        with pytest.raises(
            ValueError, match=r'All knots must be within the region bounds'
        ):
            ContinuumRegion(0.9 * u.um, 1.1 * u.um, BSpline(knots, degree=3))


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
        regions = [
            ContinuumRegion(float(i) * u.um, (float(i) + 0.5) * u.um) for i in range(3)
        ]
        config = ContinuumConfiguration(regions)
        assert list(config) == list(regions)
        assert config[1].low == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Auto-naming via resolved_params
# ---------------------------------------------------------------------------


class TestResolvedParams:
    def test_resolved_params_length_matches_regions(self):
        config = ContinuumConfiguration(
            [
                ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear()),
                ContinuumRegion(3.0 * u.um, 4.0 * u.um, Linear()),
            ]
        )
        assert len(config.resolved_params) == 2

    def test_auto_names_use_alpha_counter(self):
        config = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())]
        )
        resolved = config.resolved_params[0]
        assert 'scale' in resolved
        assert isinstance(resolved['scale'], Scale)
        assert resolved['scale'].name == 'scale_a'
        assert isinstance(resolved['angle'], ContShape)
        assert resolved['angle'].name == 'angle_a'
        assert isinstance(resolved['norm_wav'], NormWavelength)
        assert resolved['norm_wav'].name == 'norm_wav_a'

    def test_two_same_type_regions_get_sequential_alpha_names(self):
        config = ContinuumConfiguration(
            [
                ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear()),
                ContinuumRegion(3.0 * u.um, 4.0 * u.um, Linear()),
            ]
        )
        r0 = config.resolved_params[0]
        r1 = config.resolved_params[1]
        assert r0['scale'].name == 'scale_a'
        assert r1['scale'].name == 'scale_b'
        # Must be distinct token objects.
        assert r0['scale'] is not r1['scale']

    def test_norm_wav_default_fixed_at_region_center(self):
        config = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())]
        )
        nw_tok = config.resolved_params[0]['norm_wav']
        assert isinstance(nw_tok, NormWavelength)
        assert isinstance(nw_tok.prior, Fixed)
        assert nw_tok.prior.value == pytest.approx(1.5)  # center of [1, 2] um

    def test_explicit_token_used_as_is(self):
        scale = Scale('my_scale', prior=Uniform(0, 5))
        config = ContinuumConfiguration(
            [
                ContinuumRegion(
                    1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'scale': scale}
                )
            ]
        )
        resolved = config.resolved_params[0]
        assert resolved['scale'] is scale
        # scale site name is derived from its label: 'scale_my_scale'
        assert resolved['scale'].name == 'scale_my_scale'
        # beta and norm_wav should be auto-created with alpha names.
        assert resolved['beta'].name == 'beta_a'
        assert resolved['norm_wav'].name == 'norm_wav_a'
        # norm_wav default is Fixed at region center.
        assert isinstance(resolved['norm_wav'].prior, Fixed)
        assert resolved['norm_wav'].prior.value == pytest.approx(1.5)

    def test_shared_tokens_are_same_object(self):
        scale = Scale('pl_scale', prior=Uniform(0, 10))
        beta = ContShape('pl_beta', prior=Uniform(-5, 5))
        nw = NormWavelength('pl_nw', prior=Fixed(2.5))
        pl = PowerLaw()
        config = ContinuumConfiguration(
            [
                ContinuumRegion(
                    1.0 * u.um,
                    2.0 * u.um,
                    pl,
                    params={'scale': scale, 'beta': beta, 'norm_wav': nw},
                ),
                ContinuumRegion(
                    3.0 * u.um,
                    4.0 * u.um,
                    pl,
                    params={'scale': scale, 'beta': beta, 'norm_wav': nw},
                ),
            ]
        )
        r0 = config.resolved_params[0]
        r1 = config.resolved_params[1]
        assert r0['scale'] is r1['scale']
        assert r0['beta'] is r1['beta']
        assert r0['norm_wav'] is r1['norm_wav']

    def test_unique_param_count_with_sharing(self):
        scale = Scale('pl_scale', prior=Uniform(0, 10))
        beta = ContShape('pl_beta', prior=Uniform(-5, 5))
        nw = NormWavelength('pl_nw', prior=Fixed(2.5))
        pl = PowerLaw()
        config = ContinuumConfiguration(
            [
                ContinuumRegion(
                    1.0 * u.um,
                    2.0 * u.um,
                    pl,
                    params={'scale': scale, 'beta': beta, 'norm_wav': nw},
                ),
                ContinuumRegion(
                    3.0 * u.um,
                    4.0 * u.um,
                    pl,
                    params={'scale': scale, 'beta': beta, 'norm_wav': nw},
                ),
            ]
        )
        # 3 shared tokens → repr says 3 parameters, not 6.
        assert '3 parameter(s)' in repr(config)

    def test_resolved_params_is_copy(self):
        config = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())]
        )
        rp1 = config.resolved_params
        rp2 = config.resolved_params
        assert rp1 is not rp2  # New list each time.
        assert rp1[0] is not rp2[0]  # Inner dicts are copies too.

    def test_named_region_auto_params_use_region_name(self):
        config = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear(), name='blue')]
        )
        resolved = config.resolved_params[0]
        assert resolved['scale'].name == 'scale_blue'
        assert resolved['angle'].name == 'angle_blue'
        assert resolved['norm_wav'].name == 'norm_wav_blue'

    def test_two_named_regions_use_region_names(self):
        config = ContinuumConfiguration(
            [
                ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear(), name='blue'),
                ContinuumRegion(3.0 * u.um, 4.0 * u.um, Linear(), name='red'),
            ]
        )
        r0 = config.resolved_params[0]
        r1 = config.resolved_params[1]
        assert r0['scale'].name == 'scale_blue'
        assert r1['scale'].name == 'scale_red'

    def test_duplicate_region_names_raise(self):
        with pytest.raises(ValueError, match='Duplicate ContinuumRegion name'):
            ContinuumConfiguration(
                [
                    ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear(), name='blue'),
                    ContinuumRegion(3.0 * u.um, 4.0 * u.um, Linear(), name='blue'),
                ]
            )

    def test_shared_anonymous_param_across_regions_gets_alpha_name(self):
        beta = ContShape(prior=Uniform(-5, 5))
        config = ContinuumConfiguration(
            [
                ContinuumRegion(
                    1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'beta': beta}
                ),
                ContinuumRegion(
                    3.0 * u.um, 4.0 * u.um, PowerLaw(), params={'beta': beta}
                ),
            ]
        )
        r0 = config.resolved_params[0]
        r1 = config.resolved_params[1]
        assert r0['beta'] is r1['beta']
        assert r0['beta'].name == 'beta_a'

    def test_named_region_roundtrip(self):
        config = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear(), name='blue')]
        )
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        assert config2._regions[0].name == 'blue'
        assert config2.resolved_params[0]['scale'].name == 'scale_blue'


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
        assert r0['angle'] is not r1['angle']

    def test_overlapping_lines_merged(self):
        # Two lines close together → their padded windows merge into one region.
        config = ContinuumConfiguration.from_lines(
            [6549.86, 6585.27] * u.AA, width=30_000 * u.km / u.s
        )
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
        config = ContinuumConfiguration(
            [ContinuumRegion(4600.0 * u.AA, 5200.0 * u.AA, Linear())]
        )
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        assert len(config2) == 1
        assert isinstance(config2[0].form, Linear)
        assert config2[0].low == pytest.approx(4600.0)

    def test_dict_has_params_forms_regions_keys(self):
        config = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())]
        )
        d = config.to_dict()
        assert 'params' in d
        assert 'forms' in d
        assert 'regions' in d

    def test_dict_regions_have_wavelength_unit(self):
        config = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())]
        )
        d = config.to_dict()
        assert 'wavelength_unit' in d['regions'][0]

    def test_auto_param_names_roundtrip(self):
        config = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, Linear())]
        )
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        resolved = config2.resolved_params[0]
        assert resolved['scale'].name == 'scale_a'
        assert isinstance(resolved['scale'].prior, Uniform)

    def test_shared_form_object_preserved_after_roundtrip(self):
        # Regions sharing the same form Python object should still share it
        # after serialization (forms are de-duplicated in the forms section).
        pl = PowerLaw()
        config = ContinuumConfiguration(
            [
                ContinuumRegion(1.0 * u.um, 1.5 * u.um, pl),
                ContinuumRegion(2.0 * u.um, 2.5 * u.um, pl),
                ContinuumRegion(3.0 * u.um, 3.5 * u.um, pl),
            ]
        )
        d = config.to_dict()
        assert len(d['forms']) == 1  # one form entry for all three regions.
        config2 = ContinuumConfiguration.from_dict(d)
        assert config2[0].form is config2[1].form
        assert config2[1].form is config2[2].form

    def test_distinct_forms_serialized_separately(self):
        config = ContinuumConfiguration(
            [
                ContinuumRegion(1.0 * u.um, 1.5 * u.um, Linear()),
                ContinuumRegion(2.0 * u.um, 2.5 * u.um, PowerLaw()),
            ]
        )
        d = config.to_dict()
        assert len(d['forms']) == 2

    def test_explicit_param_token_roundtrip(self):
        scale = Scale('pl_scale', prior=Uniform(0, 8))
        r = ContinuumRegion(1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'scale': scale})
        config = ContinuumConfiguration([r])
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        resolved = config2.resolved_params[0]
        # Site name uses prefix + label: 'scale_pl_scale'
        assert resolved['scale'].name == 'scale_pl_scale'
        assert isinstance(resolved['scale'], Scale)
        assert isinstance(resolved['scale'].prior, Uniform)
        assert resolved['scale'].prior.high == pytest.approx(8.0)

    def test_shared_token_roundtrip(self):
        scale = Scale('pl_scale', prior=Uniform(0, 10))
        beta = ContShape('pl_beta', prior=Uniform(-5, 5))
        nw = NormWavelength('pl_nw', prior=Fixed(2.5))
        pl = PowerLaw()
        config = ContinuumConfiguration(
            [
                ContinuumRegion(
                    1.0 * u.um,
                    2.0 * u.um,
                    pl,
                    params={'scale': scale, 'beta': beta, 'norm_wav': nw},
                ),
                ContinuumRegion(
                    3.0 * u.um,
                    4.0 * u.um,
                    pl,
                    params={'scale': scale, 'beta': beta, 'norm_wav': nw},
                ),
            ]
        )
        d = config.to_dict()
        # Three shared tokens → three param entries.
        assert len(d['params']) == 3
        config2 = ContinuumConfiguration.from_dict(d)
        r0 = config2.resolved_params[0]
        r1 = config2.resolved_params[1]
        # After round-trip: same name → same object.
        assert r0['scale'] is r1['scale']
        assert r0['beta'] is r1['beta']
        assert r0['norm_wav'] is r1['norm_wav']
        assert isinstance(r0['norm_wav'].prior, Fixed)
        assert r0['norm_wav'].prior.value == pytest.approx(2.5)

    def test_polynomial_degree_preserved(self):
        config = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, Polynomial(degree=3))]
        )
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        assert config2[0].form.degree == 3

    def test_from_lines_roundtrip(self):
        config = ContinuumConfiguration.from_lines([5000.0, 6560.0] * u.AA)
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        assert len(config2) == len(config)
        # Params are independent after round-trip.
        assert (
            config2.resolved_params[0]['scale']
            is not config2.resolved_params[1]['scale']
        )

    def test_wavelength_unit_preserved_in_roundtrip(self):
        config = ContinuumConfiguration(
            [ContinuumRegion(4600.0 * u.AA, 5200.0 * u.AA, Linear())]
        )
        d = config.to_dict()
        assert d['regions'][0]['wavelength_unit'] == str(u.AA)
        config2 = ContinuumConfiguration.from_dict(d)
        assert config2[0].low == pytest.approx(4600.0)
        assert config2[0].unit.is_equivalent(u.AA)

    def test_powerlaw_norm_wav_param_roundtrip(self):
        nw = NormWavelength('my_nw', prior=Fixed(3.5))
        config = ContinuumConfiguration(
            [
                ContinuumRegion(
                    1.0 * u.um, 2.0 * u.um, PowerLaw(), params={'norm_wav': nw}
                )
            ]
        )
        d = config.to_dict()
        config2 = ContinuumConfiguration.from_dict(d)
        resolved = config2.resolved_params[0]
        # Site name uses prefix + label: 'norm_wav_my_nw'
        assert resolved['norm_wav'].name == 'norm_wav_my_nw'
        assert isinstance(resolved['norm_wav'], NormWavelength)
        assert isinstance(resolved['norm_wav'].prior, Fixed)
        assert resolved['norm_wav'].prior.value == pytest.approx(3.5)


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
_OBS_LOW = 0.9
_OBS_HIGH = 1.1


class TestLinear:
    def test_param_names(self):
        assert Linear().param_names() == ('scale', 'angle', 'norm_wav')

    def test_n_params(self):
        assert Linear().n_params == 3

    def test_default_priors_keys(self):
        assert set(Linear().default_priors()) == {'scale', 'angle', 'norm_wav'}

    def test_default_priors_norm_wav_uses_region_center(self):
        priors = Linear().default_priors(region_center=2.5)
        assert isinstance(priors['norm_wav'], Fixed)
        assert priors['norm_wav'].value == pytest.approx(2.5)

    def test_evaluate_shape(self):
        params = {
            'scale': jnp.array(1.0),
            'angle': jnp.array(0.0),
            'norm_wav': jnp.array(_CENTER),
        }
        result = Linear().evaluate(_WL, _CENTER, params, _OBS_LOW, _OBS_HIGH)
        assert result.shape == _WL.shape

    def test_evaluate_flat_at_zero_angle(self):
        params = {
            'scale': jnp.array(2.0),
            'angle': jnp.array(0.0),
            'norm_wav': jnp.array(_CENTER),
        }
        result = Linear().evaluate(_WL, _CENTER, params, _OBS_LOW, _OBS_HIGH)
        assert jnp.allclose(result, 2.0)

    def test_evaluate_equals_scale_at_norm_wav(self):
        nw = 1.05
        params = {
            'scale': jnp.array(3.0),
            'angle': jnp.array(5.0),
            'norm_wav': jnp.array(nw),
        }
        val = Linear().evaluate(jnp.array([nw]), _CENTER, params, _OBS_LOW, _OBS_HIGH)
        assert val[0] == pytest.approx(3.0)

    def test_roundtrip(self):
        f = Linear()
        assert form_from_dict(f.to_dict()) == f

    def test_equality(self):
        assert Linear() == Linear()


class TestPowerLaw:
    def test_param_names(self):
        assert PowerLaw().param_names() == ('scale', 'beta', 'norm_wav')

    def test_default_priors_norm_wav_uses_region_center(self):
        priors = PowerLaw().default_priors(region_center=3.5)
        assert isinstance(priors['norm_wav'], Fixed)
        assert priors['norm_wav'].value == pytest.approx(3.5)

    def test_default_priors_norm_wav_default_region_center(self):
        priors = PowerLaw().default_priors()
        assert isinstance(priors['norm_wav'], Fixed)
        assert priors['norm_wav'].value == pytest.approx(1.0)

    def test_evaluate_uses_norm_wav(self):
        params = {
            'scale': jnp.array(1.0),
            'beta': jnp.array(1.0),
            'norm_wav': jnp.array(2.0),
        }
        result = PowerLaw().evaluate(_WL, _CENTER, params, _OBS_LOW, _OBS_HIGH)
        expected = _WL / 2.0
        assert jnp.allclose(result, expected)

    def test_evaluate_shape(self):
        params = {
            'scale': jnp.array(1.0),
            'beta': jnp.array(0.0),
            'norm_wav': jnp.array(_CENTER),
        }
        result = PowerLaw().evaluate(_WL, _CENTER, params, _OBS_LOW, _OBS_HIGH)
        assert result.shape == _WL.shape

    def test_evaluate_flat_at_zero_beta(self):
        params = {
            'scale': jnp.array(2.0),
            'beta': jnp.array(0.0),
            'norm_wav': jnp.array(_CENTER),
        }
        result = PowerLaw().evaluate(_WL, _CENTER, params, _OBS_LOW, _OBS_HIGH)
        assert jnp.allclose(result, 2.0)

    def test_evaluate_equals_scale_at_norm_wav(self):
        nw = 1.05
        params = {
            'scale': jnp.array(3.7),
            'beta': jnp.array(2.0),
            'norm_wav': jnp.array(nw),
        }
        val = PowerLaw().evaluate(jnp.array([nw]), _CENTER, params, _OBS_LOW, _OBS_HIGH)
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
        assert Polynomial(degree=0).param_names() == ('scale', 'norm_wav')

    def test_param_names_degree1(self):
        assert Polynomial(degree=1).param_names() == ('scale', 'c1', 'norm_wav')

    def test_param_names_degree2(self):
        assert Polynomial(degree=2).param_names() == ('scale', 'c1', 'c2', 'norm_wav')

    def test_negative_degree_raises(self):
        with pytest.raises(ValueError, match='>= 0'):
            Polynomial(degree=-1)

    def test_evaluate_shape(self):
        p = Polynomial(degree=2)
        params = {k: jnp.array(1.0) for k in p.param_names()}
        result = p.evaluate(_WL, _CENTER, params, _OBS_LOW, _OBS_HIGH)
        assert result.shape == _WL.shape

    def test_evaluate_equals_scale_at_norm_wav(self):
        nw = 1.0
        params = {
            'scale': jnp.array(4.0),
            'c1': jnp.array(2.0),
            'c2': jnp.array(3.0),
            'norm_wav': jnp.array(nw),
        }
        val = Polynomial(degree=2).evaluate(
            jnp.array([nw]), _CENTER, params, _OBS_LOW, _OBS_HIGH
        )
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
        assert Chebyshev(order=0).param_names() == ('scale', 'norm_wav')

    def test_param_names_order2(self):
        assert Chebyshev(order=2).param_names() == ('scale', 'c1', 'c2', 'norm_wav')

    def test_negative_order_raises(self):
        with pytest.raises(ValueError, match='>= 0'):
            Chebyshev(order=-1)

    def test_evaluate_shape(self):
        f = Chebyshev(order=2)
        params = {k: jnp.array(0.0) for k in f.param_names()}
        params['scale'] = jnp.array(1.0)
        result = f.evaluate(_WL, _CENTER, params, _OBS_LOW, _OBS_HIGH)
        assert result.shape == _WL.shape

    def test_evaluate_equals_scale_at_norm_wav(self):
        """Chebyshev should equal scale at norm_wav."""
        nw = 1.05
        params = {
            'scale': jnp.array(2.0),
            'c1': jnp.array(0.3),
            'c2': jnp.array(0.1),
            'norm_wav': jnp.array(nw),
        }
        val = Chebyshev(order=2).evaluate(
            jnp.array([nw]), _CENTER, params, _OBS_LOW, _OBS_HIGH
        )
        assert val[0] == pytest.approx(2.0, rel=1e-5)

    def test_roundtrip(self):
        f = Chebyshev(order=3)
        f2 = form_from_dict(f.to_dict())
        assert f2.order == 3
        assert f2.stretch == pytest.approx(1)
        assert f2 == f

    def test_equality_checks_stretch(self):
        assert Chebyshev(2, 0.1) != Chebyshev(2, 0.2)


class TestBlackbody:
    def test_param_names(self):
        assert set(Blackbody().param_names()) == {'scale', 'temperature', 'norm_wav'}

    def test_default_priors_norm_wav_uses_region_center(self):
        priors = Blackbody().default_priors(region_center=2.0)
        assert isinstance(priors['norm_wav'], Fixed)
        assert priors['norm_wav'].value == pytest.approx(2.0)

    def test_evaluate_shape(self):
        wl = jnp.linspace(0.5, 2.0, 30)
        params = {
            'scale': jnp.array(1.0),
            'temperature': jnp.array(5000.0),
            'norm_wav': jnp.array(1.0),
        }
        result = Blackbody().evaluate(wl, 1.0, params, 0.5, 2.0)
        assert result.shape == wl.shape

    def test_evaluate_positive(self):
        wl = jnp.linspace(0.5, 2.0, 30)
        params = {
            'scale': jnp.array(1.0),
            'temperature': jnp.array(5000.0),
            'norm_wav': jnp.array(1.0),
        }
        result = Blackbody().evaluate(wl, 1.0, params, 0.5, 2.0)
        assert jnp.all(result > 0)

    def test_evaluate_equals_scale_at_norm_wav(self):
        nw = 1.0
        params = {
            'scale': jnp.array(2.5),
            'temperature': jnp.array(5000.0),
            'norm_wav': jnp.array(nw),
        }
        val = Blackbody().evaluate(jnp.array([nw]), nw, params, _OBS_LOW, _OBS_HIGH)
        assert val[0] == pytest.approx(2.5, rel=1e-5)

    def test_evaluate_consistent_across_unit_conversions(self):
        """Blackbody evaluation should be consistent regardless of wavelength unit."""
        # Create a region in Angstroms
        region_aa = ContinuumRegion(5000.0 * u.AA, 6000.0 * u.AA, Blackbody())
        # Create a region in microns
        region_um = ContinuumRegion(0.5 * u.um, 0.6 * u.um, Blackbody())
        # Evaluate at the same physical wavelength (5500 AA = 0.55 um)
        params_aa = {
            'scale': jnp.array(1.0),
            'temperature': jnp.array(5000.0),
            'norm_wav': jnp.array(5500.0),
        }
        params_um = {
            'scale': jnp.array(1.0),
            'temperature': jnp.array(5000.0),
            'norm_wav': jnp.array(0.55),
        }
        val_aa = region_aa.form.evaluate(
            jnp.array([5500.0]), 5500.0, params_aa, 5000.0, 6000.0
        )
        val_um = region_um.form.evaluate(jnp.array([0.55]), 0.55, params_um, 0.5, 0.6)
        # Results should be equal at the same physical wavelength
        assert val_aa[0] == pytest.approx(val_um[0], rel=1e-5)

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
            'norm_wav',
        }

    def test_evaluate_equals_scale_at_norm_wav(self):
        nw = 1.0
        params = {
            'scale': jnp.array(3.0),
            'temperature': jnp.array(5000.0),
            'beta': jnp.array(1.5),
            'norm_wav': jnp.array(nw),
        }
        val = ModifiedBlackbody().evaluate(
            jnp.array([nw]), nw, params, _OBS_LOW, _OBS_HIGH
        )
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
            'norm_wav',
        }

    def test_lambda_ext_micron_stored(self):
        f = AttenuatedBlackbody(lambda_ext=0.6 * u.um)
        assert f.lambda_ext.value == pytest.approx(0.6)

    def test_evaluate_equals_scale_at_norm_wav(self):
        nw = 1.0
        params = {
            'scale': jnp.array(1.5),
            'temperature': jnp.array(5000.0),
            'tau_v': jnp.array(0.5),
            'alpha': jnp.array(-1.5),
            'norm_wav': jnp.array(nw),
        }
        val = AttenuatedBlackbody().evaluate(
            jnp.array([nw]), nw, params, _OBS_LOW, _OBS_HIGH
        )
        assert val[0] == pytest.approx(1.5, rel=1e-5)

    def test_roundtrip(self):
        f = AttenuatedBlackbody(lambda_ext=0.6 * u.um)
        f2 = form_from_dict(f.to_dict())
        assert f2.lambda_ext.value == pytest.approx(0.6)
        assert f2 == f

    def test_equality_checks_lambda_ext(self):
        assert AttenuatedBlackbody(0.55 * u.um) != AttenuatedBlackbody(0.50 * u.um)


class TestBSpline:
    @pytest.fixture
    def cubic_knots(self):
        # Clamped cubic knot vector: repeat end knots degree+1 times.
        return [1.0] * u.um

    def test_n_basis(self, cubic_knots):
        b = BSpline(cubic_knots, degree=3)
        assert b.n_basis == 5  # 9 knots - 3 - 1

    def test_param_names(self, cubic_knots):
        b = BSpline(cubic_knots, degree=3)
        expected = ('scale', 'coeff_1', 'coeff_2', 'coeff_3', 'coeff_4', 'norm_wav')
        assert b.param_names() == expected

    def test_evaluate_shape(self, cubic_knots):
        b = BSpline(cubic_knots, degree=3)
        b._prepare(0.9 * u.um, 1.1 * u.um)
        params = {n: jnp.array(1.0) for n in b.param_names()}
        result = b.evaluate(_WL, _CENTER, params, _OBS_LOW, _OBS_HIGH)
        assert result.shape == _WL.shape

    def test_evaluate_equals_scale_at_norm_wav(self, cubic_knots):
        """BSpline should equal scale at norm_wav."""
        b = BSpline(cubic_knots, degree=3)
        b._prepare(0.9 * u.um, 1.1 * u.um)
        nw = 1.0
        params = {n: jnp.array(0.5) for n in b.param_names()}
        params['scale'] = jnp.array(2.0)
        params['norm_wav'] = jnp.array(nw)
        val = b.evaluate(jnp.array([nw]), _CENTER, params, _OBS_LOW, _OBS_HIGH)
        assert val[0] == pytest.approx(2.0, rel=1e-5)

    def test_roundtrip(self, cubic_knots):
        b = BSpline(cubic_knots, degree=3)
        b2 = form_from_dict(b.to_dict())
        assert b2 == b
        assert b2.n_basis == b.n_basis


class TestBernstein:
    @pytest.fixture
    def bernstein(self):
        return Bernstein(degree=3)

    def test_param_names(self, bernstein):
        expected = ('scale', 'coeff_1', 'coeff_2', 'coeff_3', 'norm_wav')
        assert bernstein.param_names() == expected

    def test_evaluate_shape(self, bernstein):
        params = {n: jnp.array(1.0) for n in bernstein.param_names()}
        result = bernstein.evaluate(_WL, _CENTER, params, _OBS_LOW, _OBS_HIGH)
        assert result.shape == _WL.shape

    def test_evaluate_nonnegative_with_positive_coeffs(self, bernstein):
        params = {n: jnp.array(1.0) for n in bernstein.param_names()}
        result = bernstein.evaluate(_WL, _CENTER, params, _OBS_LOW, _OBS_HIGH)
        assert jnp.all(result >= 0)

    def test_evaluate_equals_scale_at_norm_wav(self, bernstein):
        """Bernstein should equal scale at norm_wav."""
        nw = 1.05
        params = {n: jnp.array(0.5) for n in bernstein.param_names()}
        params['scale'] = jnp.array(3.0)
        params['norm_wav'] = jnp.array(nw)
        val = bernstein.evaluate(jnp.array([nw]), _CENTER, params, _OBS_LOW, _OBS_HIGH)
        assert val[0] == pytest.approx(3.0, rel=1e-5)

    def test_roundtrip(self, bernstein):
        b2 = form_from_dict(bernstein.to_dict())
        assert b2.degree == 3
        assert b2 == bernstein

    def test_equality_checks_stretch(self):
        b1 = Bernstein(degree=3, stretch=1.0)
        b2 = Bernstein(degree=3, stretch=1.5)
        assert b1 != b2
        b3 = Bernstein(degree=3, stretch=1.0)
        assert b1 == b3


# ---------------------------------------------------------------------------
# get_form string registry
# ---------------------------------------------------------------------------


class TestGetForm:
    def test_string_returns_instance(self):
        f = get_form('Linear')
        assert isinstance(f, Linear)

    def test_string_with_kwargs(self):
        f = get_form('Polynomial', degree=3)
        assert isinstance(f, Polynomial)
        assert f.degree == 3

    def test_passthrough_instance(self):
        original = PowerLaw()
        assert get_form(original) is original

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match='Unknown ContinuumForm type'):
            get_form('NoSuchForm')

    def test_region_accepts_string(self):
        r = ContinuumRegion(1.0 * u.um, 2.0 * u.um, 'PowerLaw')
        assert isinstance(r.form, PowerLaw)

    def test_region_string_unknown_raises(self):
        with pytest.raises(ValueError, match='Unknown ContinuumForm type'):
            ContinuumRegion(1.0 * u.um, 2.0 * u.um, 'FakeForm')


# ---------------------------------------------------------------------------
# Params validation
# ---------------------------------------------------------------------------


class TestParamsValidation:
    def test_invalid_param_name_raises(self):
        region = ContinuumRegion(
            1.0 * u.um,
            2.0 * u.um,
            Linear(),
            params={'amplitude': Parameter('amp', prior=Uniform(0, 10))},
        )
        with pytest.raises(ValueError, match='does not have parameter'):
            ContinuumConfiguration([region])

    def test_typo_param_name_raises(self):
        region = ContinuumRegion(
            1.0 * u.um,
            2.0 * u.um,
            PowerLaw(),
            params={'betta': ContShape('b', prior=Uniform(-5, 5))},
        )
        with pytest.raises(ValueError, match='does not have parameter'):
            ContinuumConfiguration([region])

    def test_valid_param_names_pass(self):
        region = ContinuumRegion(
            1.0 * u.um,
            2.0 * u.um,
            Linear(),
            params={'scale': Scale('s', prior=Uniform(0, 5))},
        )
        config = ContinuumConfiguration([region])
        assert len(config) == 1


# ---------------------------------------------------------------------------
# _prepare: static wavelength conversion
# ---------------------------------------------------------------------------


# class TestPrepare:
#     def test_blackbody_prepares_micron_factor(self):
#         f = Blackbody()
#         f._prepare(u.AA, u.um)
#         # Convert from AA to um: 1 AA = 1e-4 um
#         assert f._micron_factor == pytest.approx(1e-4)
#         # Prepare with canonical unit already microns
#         prepared_um = f._prepare(u.um, u.um)
#         assert prepared_um._micron_factor == pytest.approx(1.0)

#     def test_attenuated_blackbody_converts_lambda_ext(self):
#         f = AttenuatedBlackbody(lambda_ext=0.55 * u.um)
#         prepared = f._prepare(u.AA, u.um)
#         # 0.55 um = 5500 AA
#         assert prepared._lambda_ext_eval == pytest.approx(5500.0)
#         # Original unchanged
#         assert f._lambda_ext_eval == pytest.approx(0.55)

#     def test_attenuated_blackbody_quantity_input(self):
#         f = AttenuatedBlackbody(lambda_ext=5500.0 * u.AA)
#         assert f.lambda_ext.value == pytest.approx(5500)
#         prepared = f._prepare(u.AA, u.um)
#         assert prepared._lambda_ext_eval == pytest.approx(5500.0)

#     def test_attenuated_blackbody_evaluate_uses_prepared_value(self):
#         # Evaluate in Angstrom space after _prepare
#         f = AttenuatedBlackbody(lambda_ext=0.55 * u.um)
#         prepared = f._prepare(u.AA, u.um)
#         wl = jnp.linspace(4000.0, 8000.0, 20)  # Angstroms
#         nw = 5500.0
#         params = {
#             'scale': jnp.array(1.0),
#             'temperature': jnp.array(5000.0),
#             'tau_v': jnp.array(0.5),
#             'alpha': jnp.array(-1.5),
#             'norm_wav': jnp.array(nw),
#         }
#         result = prepared.evaluate(wl, nw, params, 4000.0, 8000.0)
#         assert result.shape == wl.shape
#         # At normalization wavelength, result should equal scale
#         val = prepared.evaluate(jnp.array([nw]), nw, params, 4000.0, 8000.0)
#         assert val[0] == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------------------------
# is_linear property
# ---------------------------------------------------------------------------

_FLUX_UNIT = u.erg / (u.s * u.cm**2 * u.AA)
_WL_UNIT = u.um
_LINEAR_FORMS = [Linear(), Polynomial(2)]
_NONLINEAR_FORMS = [
    PowerLaw(),
    Blackbody(),
    ModifiedBlackbody(),
    AttenuatedBlackbody(),
    Chebyshev(2, 0.1),
    BSpline([1.0] * u.um, degree=3),
    Bernstein(degree=3),
]


class TestIsLinear:
    @pytest.mark.parametrize('form', _LINEAR_FORMS)
    def test_linear_forms_return_true(self, form):
        assert form.is_linear is True

    @pytest.mark.parametrize('form', _NONLINEAR_FORMS)
    def test_nonlinear_forms_return_false(self, form):
        assert form.is_linear is False


# ---------------------------------------------------------------------------
# param_units method
# ---------------------------------------------------------------------------


class TestParamUnits:
    @pytest.mark.parametrize(
        'form',
        [
            Linear(),
            PowerLaw(),
            Polynomial(2),
            Chebyshev(2, 0.1),
            BSpline([1.0] * u.um, degree=3),
            Bernstein(degree=3),
            Blackbody(),
            ModifiedBlackbody(),
            AttenuatedBlackbody(),
        ],
    )
    def test_param_units_returns_dict(self, form):
        pu = form.param_units(_FLUX_UNIT, _WL_UNIT)
        assert isinstance(pu, dict)
        assert 'scale' in pu
        # scale should have apply_cs=True and flux_unit
        apply_cs, _phys_unit = pu['scale']
        assert apply_cs is True

    def test_linear_angle_dimensionless(self):
        pu = Linear().param_units(_FLUX_UNIT, _WL_UNIT)
        _, angle_unit = pu['angle']
        assert angle_unit is None

    def test_powerlaw_beta_dimensionless(self):
        pu = PowerLaw().param_units(_FLUX_UNIT, _WL_UNIT)
        _, beta_unit = pu['beta']
        assert beta_unit is None

    def test_blackbody_temperature_unit(self):
        pu = Blackbody().param_units(_FLUX_UNIT, _WL_UNIT)
        _, temp_unit = pu['temperature']
        assert temp_unit == u.K


# ---------------------------------------------------------------------------
# default_priors for parameterized forms
# ---------------------------------------------------------------------------


class TestDefaultPriors:
    def test_chebyshev_default_priors_order2(self):
        priors = Chebyshev(order=2).default_priors(region_center=1.5)
        assert 'c1' in priors
        assert 'c2' in priors
        assert isinstance(priors['norm_wav'], Fixed)
        assert priors['norm_wav'].value == pytest.approx(1.5)

    def test_polynomial_default_priors_degree2(self):
        priors = Polynomial(degree=2).default_priors(region_center=2.0)
        assert 'c1' in priors
        assert 'c2' in priors

    def test_bspline_default_priors(self):
        knots = [1.0, 1.05, 1.1] * u.um
        b = BSpline(knots, degree=3)
        priors = b.default_priors(region_center=1.0)
        assert 'scale' in priors
        for i in range(1, b.n_basis):
            assert f'coeff_{i}' in priors

    def test_bernstein_default_priors(self):
        b = Bernstein(degree=3)
        priors = b.default_priors(region_center=1.0)
        assert 'scale' in priors
        assert 'coeff_1' in priors


# ---------------------------------------------------------------------------
# __eq__ cross-type (NotImplemented) and __hash__ for all forms
# ---------------------------------------------------------------------------


class TestFormEqHash:
    @pytest.mark.parametrize(
        'form',
        [
            Linear(),
            PowerLaw(),
            Polynomial(2),
            Chebyshev(2, 0.1),
            Blackbody(),
            ModifiedBlackbody(),
            AttenuatedBlackbody(),
            BSpline([1] * u.um, degree=3),
            Bernstein(degree=3),
        ],
    )
    def test_hashable(self, form):
        assert isinstance(hash(form), int)

    def test_different_types_not_equal(self):
        # ContinuumForm base __eq__ returns NotImplemented for different types
        assert Linear() != PowerLaw()
        assert Blackbody() != ModifiedBlackbody()
        assert Polynomial(2) != Chebyshev(2)

    def test_polynomial_eq_hash(self):
        assert Polynomial(2) == Polynomial(2)
        assert Polynomial(2) != Polynomial(3)
        assert hash(Polynomial(2)) == hash(Polynomial(2))

    def test_chebyshev_eq_hash(self):
        assert Chebyshev(2, 0.1) == Chebyshev(2, 0.1)
        assert Chebyshev(2, 0.1) != Chebyshev(2, 0.2)
        assert isinstance(hash(Chebyshev(2, 0.1)), int)

    def test_attenuated_blackbody_eq_hash(self):
        assert AttenuatedBlackbody(0.55 * u.um) == AttenuatedBlackbody(0.55 * u.um)
        assert AttenuatedBlackbody(0.55 * u.um) != AttenuatedBlackbody(0.50 * u.um)
        assert isinstance(hash(AttenuatedBlackbody(0.55 * u.um)), int)

    def test_bspline_eq_hash(self):
        knots = [1.0] * u.um
        b1 = BSpline(knots, degree=3)
        b2 = BSpline(knots, degree=3)
        assert b1 == b2
        assert isinstance(hash(b1), int)

    def test_bernstein_eq_hash(self):
        b1 = Bernstein(degree=3)
        b2 = Bernstein(degree=3)
        assert b1 == b2
        assert isinstance(hash(b1), int)


# ---------------------------------------------------------------------------
# ContinuumConfiguration addition
# ---------------------------------------------------------------------------


class TestContinuumConfigurationAdd:
    def test_add_basic(self):
        cc1 = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 2.0 * u.um)])
        cc2 = ContinuumConfiguration([ContinuumRegion(3.0 * u.um, 4.0 * u.um)])
        merged = cc1 + cc2
        assert len(merged) == 2
        assert merged[0].low == pytest.approx(1.0)
        assert merged[1].low == pytest.approx(3.0)

    def test_add_multiple_regions(self):
        cc1 = ContinuumConfiguration(
            [
                ContinuumRegion(1.0 * u.um, 2.0 * u.um),
                ContinuumRegion(3.0 * u.um, 4.0 * u.um),
            ]
        )
        cc2 = ContinuumConfiguration(
            [
                ContinuumRegion(5.0 * u.um, 6.0 * u.um),
                ContinuumRegion(7.0 * u.um, 8.0 * u.um),
            ]
        )
        merged = cc1 + cc2
        assert len(merged) == 4

    def test_add_preserves_forms(self):
        cc1 = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, form=Linear())]
        )
        cc2 = ContinuumConfiguration(
            [ContinuumRegion(3.0 * u.um, 4.0 * u.um, form=PowerLaw())]
        )
        merged = cc1 + cc2
        assert isinstance(merged[0].form, Linear)
        assert isinstance(merged[1].form, PowerLaw)

    def test_add_overlapping_regions_allowed(self):
        """Overlapping regions are summed in the model — addition should not block this."""
        cc1 = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 3.0 * u.um)])
        cc2 = ContinuumConfiguration([ContinuumRegion(2.0 * u.um, 4.0 * u.um)])
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            merged = cc1 + cc2
        assert len(merged) == 2

    def test_add_colliding_param_names_raises(self):
        shared = Scale('my_scale')
        cc1 = ContinuumConfiguration(
            [ContinuumRegion(1.0 * u.um, 2.0 * u.um, params={'scale': shared})]
        )
        cc2 = ContinuumConfiguration(
            [
                ContinuumRegion(
                    3.0 * u.um, 4.0 * u.um, params={'scale': Scale('my_scale')}
                )
            ]
        )
        with pytest.raises(ValueError, match='my_scale'):
            cc1 + cc2

    def test_add_auto_named_params_do_not_collide(self):
        """Auto-created param names are re-indexed on construction — no collision."""
        cc1 = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 2.0 * u.um)])
        cc2 = ContinuumConfiguration([ContinuumRegion(3.0 * u.um, 4.0 * u.um)])
        merged = cc1 + cc2
        assert len(merged) == 2

    def test_add_wrong_type_returns_not_implemented(self):
        cc = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 2.0 * u.um)])
        result = cc.__add__('not a config')
        assert result is NotImplemented

    def test_add_empty_left(self):
        cc1 = ContinuumConfiguration()
        cc2 = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 2.0 * u.um)])
        merged = cc1 + cc2
        assert len(merged) == 1

    def test_add_empty_right(self):
        cc1 = ContinuumConfiguration([ContinuumRegion(1.0 * u.um, 2.0 * u.um)])
        cc2 = ContinuumConfiguration()
        merged = cc1 + cc2
        assert len(merged) == 1


# ---------------------------------------------------------------------------
# Form validation errors and repr
# ---------------------------------------------------------------------------


class TestChebyshevValidation:
    def test_invalid_stretch_raises(self):
        with pytest.raises(ValueError, match='stretch factor must be > 0'):
            Chebyshev(stretch=0.0)

    def test_negative_stretch_raises(self):
        with pytest.raises(ValueError, match='stretch factor must be > 0'):
            Chebyshev(stretch=-1.0)

    def test_repr(self):
        c = Chebyshev(order=3, stretch=2.0)
        r = repr(c)
        assert 'Chebyshev' in r
        assert '3' in r
        assert '2.0' in r


class TestBernsteinValidation:
    def test_invalid_stretch_raises(self):
        with pytest.raises(ValueError, match='stretch factor must be > 0'):
            Bernstein(stretch=0.0)

    def test_negative_stretch_raises(self):
        with pytest.raises(ValueError, match='stretch factor must be > 0'):
            Bernstein(stretch=-0.5)

    def test_repr(self):
        b = Bernstein(degree=3, stretch=1.5)
        r = repr(b)
        assert 'Bernstein' in r
        assert '3' in r


class TestBSplineValidation:
    def test_non_quantity_knots_raises(self):
        with pytest.raises(ValueError, match='knots must be an astropy Quantity'):
            BSpline(knots=[6500.0, 6550.0, 6600.0])

    def test_degree_property(self):
        knots = [6520.0, 6560.0] * u.AA
        b = BSpline(knots=knots, degree=2)
        assert b.degree == 2

    def test_knots_property(self):
        knots = [6520.0, 6560.0] * u.AA
        b = BSpline(knots=knots)
        assert len(b.knots) == 2

    def test_repr(self):
        knots = [6520.0, 6560.0] * u.AA
        b = BSpline(knots=knots, degree=3)
        r = repr(b)
        assert 'BSpline' in r
        assert '3' in r


class TestPolynomialRepr:
    def test_repr(self):
        p = Polynomial(degree=2)
        r = repr(p)
        assert 'Polynomial' in r
        assert '2' in r


class TestAttenuatedBlackbodyValidation:
    def test_non_quantity_lambda_ext_raises(self):
        with pytest.raises(ValueError, match='lambda_ext must be an astropy Quantity'):
            AttenuatedBlackbody(lambda_ext=0.55)

    def test_repr(self):
        a = AttenuatedBlackbody()
        r = repr(a)
        assert 'AttenuatedBlackbody' in r

    def test_eq_different_type(self):
        a = AttenuatedBlackbody()
        assert a.__eq__('other') is NotImplemented

    def test_default_priors(self):
        a = AttenuatedBlackbody()
        priors = a.default_priors(region_center=1.0)
        assert 'scale' in priors
        assert 'temperature' in priors
        assert 'tau_v' in priors
        assert 'alpha' in priors
        assert 'norm_wav' in priors


class TestModifiedBlackbodyDefaultPriors:
    def test_default_priors(self):
        m = ModifiedBlackbody()
        priors = m.default_priors(region_center=1.5)
        assert 'scale' in priors
        assert 'temperature' in priors
        assert 'beta' in priors
        assert 'norm_wav' in priors
        # norm_wav should be fixed at region_center
        assert isinstance(priors['norm_wav'], Fixed)


# ---------------------------------------------------------------------------
