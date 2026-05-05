"""Tests for Profile class behaviour: serialization, default priors, and aliases."""

import pytest

from unite.line.library import (
    SEMG,
    BoxGauss,
    Cauchy,
    GaussHermite,
    Gaussian,
    Laplace,
    PseudoVoigt,
    SkewVoigt,
    SplitNormal,
    profile_from_dict,
    resolve_profile,
)
from unite.prior import Prior

# ---------------------------------------------------------------------------
# Profile serialization: to_dict / from_dict
# ---------------------------------------------------------------------------

_ALL_PROFILES = [
    Gaussian(),
    Cauchy(),
    PseudoVoigt(),
    Laplace(),
    SEMG(),
    GaussHermite(),
    SplitNormal(),
    SkewVoigt(),
    BoxGauss(),
]


class TestProfileSerialization:
    @pytest.mark.parametrize('profile', _ALL_PROFILES)
    def test_roundtrip(self, profile):
        d = profile.to_dict()
        restored = profile_from_dict(d)
        assert type(restored) is type(profile)

    @pytest.mark.parametrize('profile', _ALL_PROFILES)
    def test_repr_contains_class_name(self, profile):
        assert type(profile).__name__ in repr(profile)


# ---------------------------------------------------------------------------
# default_priors
# ---------------------------------------------------------------------------


class TestDefaultPriors:
    @pytest.mark.parametrize('profile', _ALL_PROFILES)
    def test_keys_match_param_names(self, profile):
        priors = profile.default_priors()
        assert set(priors.keys()) == set(profile.param_names())

    @pytest.mark.parametrize('profile', _ALL_PROFILES)
    def test_values_are_priors(self, profile):
        for v in profile.default_priors().values():
            assert isinstance(v, Prior)


# ---------------------------------------------------------------------------
# resolve_profile
# ---------------------------------------------------------------------------


def test_resolve_profile_returns_instance_unchanged():
    g = Gaussian()
    assert resolve_profile(g) is g


@pytest.mark.parametrize(
    ('alias', 'expected_type'),
    [
        ('gaussian', Gaussian),
        ('normal', Gaussian),
        ('cauchy', Cauchy),
        ('lorentzian', Cauchy),
        ('laplace', Laplace),
        ('exponential', Laplace),
        ('voigt', PseudoVoigt),
        ('pseudovoigt', PseudoVoigt),
        ('semg', SEMG),
        ('exp-gaussian', SEMG),
        ('hermite', GaussHermite),
        ('gauss-hermite', GaussHermite),
        ('split-normal', SplitNormal),
        ('two-sided', SplitNormal),
        ('skew-voigt', SkewVoigt),
        ('skewvoigt', SkewVoigt),
        ('boxgauss', BoxGauss),
        ('box-gauss', BoxGauss),
        ('boxcar', BoxGauss),
    ],
)
def test_resolve_profile_valid_string(alias, expected_type):
    assert isinstance(resolve_profile(alias), expected_type)


def test_resolve_profile_case_insensitive():
    assert isinstance(resolve_profile('Gaussian'), Gaussian)
    assert isinstance(resolve_profile('CAUCHY'), Cauchy)


def test_resolve_profile_invalid_string_raises():
    with pytest.raises(ValueError, match='Unknown profile'):
        resolve_profile('not_a_profile')


def test_resolve_profile_invalid_type_raises():
    with pytest.raises(TypeError, match='str or Profile'):
        resolve_profile(42)
