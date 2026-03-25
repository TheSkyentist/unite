"""Tests for Profile class behaviour: serialization, equality, hashing, aliases."""

import pytest

from unite.line.library import (
    SEMG,
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
# Profile equality and hashing
# ---------------------------------------------------------------------------

_ALL_PROFILE_CLASSES = [
    Gaussian,
    Cauchy,
    PseudoVoigt,
    Laplace,
    SEMG,
    GaussHermite,
    SplitNormal,
    SkewVoigt,
]


class TestProfileEqHash:
    @pytest.mark.parametrize('profile_cls', _ALL_PROFILE_CLASSES)
    def test_same_type_equal(self, profile_cls):
        assert profile_cls() == profile_cls()

    def test_different_types_not_equal(self):
        assert Gaussian() != Cauchy()
        assert PseudoVoigt() != Laplace()

    @pytest.mark.parametrize('profile_cls', _ALL_PROFILE_CLASSES)
    def test_hashable(self, profile_cls):
        p = profile_cls()
        assert isinstance(hash(p), int)

    def test_can_be_used_as_dict_key(self):
        d = {Gaussian(): 'gauss', Cauchy(): 'cauchy'}
        assert d[Gaussian()] == 'gauss'


# ---------------------------------------------------------------------------
# resolve_profile
# ---------------------------------------------------------------------------


def test_resolve_profile_invalid_type_raises():
    with pytest.raises(TypeError, match='str or Profile'):
        resolve_profile(42)


# ---------------------------------------------------------------------------
# default_priors
# ---------------------------------------------------------------------------


def test_semg_default_priors():
    """SEMG.default_priors() returns a dict with fwhm_gauss and fwhm_exp."""
    priors = SEMG().default_priors()
    assert 'fwhm_gauss' in priors
    assert 'fwhm_exp' in priors
