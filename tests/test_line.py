"""Tests for unite.line (parameter tokens and LineConfiguration)."""

import pytest
from astropy import units as u

from unite.line.config import FWHM, Flux, LineConfiguration, Param, Redshift
from unite.line.profiles import (
    Cauchy,
    Gaussian,
    GaussHermite,
    Laplace,
    PseudoVoigt,
    SEMG,
    SplitNormal,
    profile_from_dict,
)
from unite.prior import Fixed, TruncatedNormal, Uniform

# ---------------------------------------------------------------------------
# Parameter tokens
# ---------------------------------------------------------------------------


class TestRedshift:
    def test_default_prior(self):
        z = Redshift()
        assert isinstance(z.prior, Uniform)

    def test_named(self):
        z = Redshift('nlr')
        assert z.name == 'nlr'

    def test_custom_prior(self):
        p = TruncatedNormal(0.0, 0.01, -0.05, 0.05)
        z = Redshift('nlr', prior=p)
        assert z.prior is p

    def test_arithmetic_returns_ref(self):
        from unite.prior import ParameterRef

        z = Redshift()
        ref = z * 2
        assert isinstance(ref, ParameterRef)
        assert ref.scale == pytest.approx(2.0)

    def test_add_returns_ref(self):
        from unite.prior import ParameterRef

        z = Redshift()
        ref = z + 0.01
        assert isinstance(ref, ParameterRef)
        assert ref.offset == pytest.approx(0.01)

    def test_cross_kind_ref_raises(self):
        fwhm = FWHM()
        with pytest.raises(
            TypeError, match='ParameterRefs must reference the same kind'
        ):
            Redshift(prior=Uniform(low=fwhm * 2, high=1000))


class TestFWHM:
    def test_default_prior(self):
        f = FWHM()
        assert isinstance(f.prior, Uniform)

    def test_named_with_prior(self):
        f = FWHM('broad', prior=Uniform(500, 5000))
        assert f.name == 'broad'
        assert f.prior.low == pytest.approx(500.0)

    def test_dependent_bound(self):
        narrow = FWHM('narrow', prior=Uniform(0, 1000))
        broad = FWHM('broad', prior=Uniform(low=narrow * 2 + 150, high=5000))
        assert narrow in broad.prior.dependencies()


class TestFlux:
    def test_default_prior(self):
        f = Flux()
        assert isinstance(f.prior, Uniform)

    def test_named(self):
        f = Flux('Ha_flux')
        assert f.name == 'Ha_flux'


class TestParam:
    def test_default_prior(self):
        p = Param()
        assert isinstance(p.prior, Uniform)

    def test_custom_prior(self):
        p = Param('h3', prior=Uniform(-0.5, 0.5))
        assert p.prior.low == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# LineConfiguration.add_line
# ---------------------------------------------------------------------------


class TestAddLine:
    def test_single_gaussian(self):
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA)
        assert len(config) == 1

    def test_wavelength_unit_preserved(self):
        config = LineConfiguration()
        config.add_line('Ha', 0.656461 * u.micron)
        entry = config._entries[0]
        assert entry.wavelength.unit == u.micron

    def test_shared_redshift(self):
        z = Redshift('nlr')
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA, redshift=z)
        config.add_line('Hb', 4862.68 * u.AA, redshift=z)
        assert config._entries[0].redshift is config._entries[1].redshift

    def test_shared_fwhm(self):
        f = FWHM('narrow')
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA, fwhm_gauss=f)
        config.add_line('Hb', 4862.68 * u.AA, fwhm_gauss=f)
        assert (
            config._entries[0].fwhms['fwhm_gauss']
            is config._entries[1].fwhms['fwhm_gauss']
        )

    def test_duplicate_raises(self):
        z = Redshift('z')
        f = FWHM('f')
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA, redshift=z, fwhm_gauss=f)
        with pytest.raises(ValueError, match='identical line'):
            config.add_line('Ha', 6564.61 * u.AA, redshift=z, fwhm_gauss=f)

    def test_two_components_same_wavelength_allowed(self):
        z = Redshift('z')
        f_narrow = FWHM('narrow')
        f_broad = FWHM('broad')
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA, redshift=z, fwhm_gauss=f_narrow)
        config.add_line('Ha', 6564.61 * u.AA, redshift=z, fwhm_gauss=f_broad)
        assert len(config) == 2

    def test_wrong_redshift_type_raises(self):
        config = LineConfiguration()
        with pytest.raises(TypeError, match='Redshift'):
            config.add_line('Ha', 6564.61 * u.AA, redshift=FWHM())  # type: ignore[arg-type]

    def test_wrong_flux_type_raises(self):
        config = LineConfiguration()
        with pytest.raises(TypeError, match='Flux'):
            config.add_line('Ha', 6564.61 * u.AA, flux=FWHM())  # type: ignore[arg-type]

    def test_unknown_param_raises(self):
        config = LineConfiguration()
        with pytest.raises(ValueError, match='Unexpected parameter'):
            config.add_line('Ha', 6564.61 * u.AA, not_a_param=FWHM())

    def test_wrong_param_type_for_slot_raises(self):
        config = LineConfiguration()
        with pytest.raises(TypeError, match='must be a FWHM'):
            config.add_line('Ha', 6564.61 * u.AA, fwhm_gauss=Param())  # type: ignore[arg-type]

    def test_non_quantity_center_raises(self):
        config = LineConfiguration()
        with pytest.raises(TypeError):
            config.add_line('Ha', 6564.61)  # bare float, no unit

    def test_strength_stored(self):
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA, strength=2.5)
        assert config._entries[0].strength == pytest.approx(2.5)


class TestProfileAliases:
    @pytest.mark.parametrize(
        'alias',
        [
            'gaussian',
            'normal',
            'lorentzian',
            'cauchy',
            'voigt',
            'hermite',
            'laplace',
            'split-normal',
        ],
    )
    def test_string_alias_accepted(self, alias):
        config = LineConfiguration()
        config.add_line('X', 5000.0 * u.AA, profile=alias)
        assert len(config) == 1

    def test_unknown_profile_raises(self):
        config = LineConfiguration()
        with pytest.raises(ValueError, match='Unknown profile'):
            config.add_line('X', 5000.0 * u.AA, profile='notaprofile')

    def test_profile_instance_accepted(self):
        from unite.line.profiles import Gaussian

        config = LineConfiguration()
        config.add_line('X', 5000.0 * u.AA, profile=Gaussian())
        assert len(config) == 1


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------


class TestLineConfigurationRoundTrip:
    def test_single_gaussian_roundtrip(self):
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA)
        d = config.to_dict()
        config2 = LineConfiguration.from_dict(d)
        assert len(config2) == 1
        assert config2._entries[0].name == 'Ha'
        wl = config2._entries[0].wavelength
        assert wl.value == pytest.approx(6564.61)
        assert wl.unit == u.AA

    def test_wavelength_unit_preserved_roundtrip(self):
        config = LineConfiguration()
        config.add_line('Ha', 0.656461 * u.micron)
        d = config.to_dict()
        config2 = LineConfiguration.from_dict(d)
        assert config2._entries[0].wavelength.unit == u.micron
        assert config2._entries[0].wavelength.value == pytest.approx(0.656461)

    def test_shared_token_roundtrip(self):
        z = Redshift('nlr', prior=TruncatedNormal(0.0, 0.01, -0.05, 0.05))
        f = FWHM('narrow', prior=Uniform(50, 800))
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA, redshift=z, fwhm_gauss=f)
        config.add_line('Hb', 4862.68 * u.AA, redshift=z, fwhm_gauss=f)

        d = config.to_dict()
        config2 = LineConfiguration.from_dict(d)

        assert len(config2) == 2
        # Shared redshift → same object in both entries
        assert config2._entries[0].redshift is config2._entries[1].redshift
        # Shared fwhm → same object
        assert (
            config2._entries[0].fwhms['fwhm_gauss']
            is config2._entries[1].fwhms['fwhm_gauss']
        )

    def test_dependent_prior_roundtrip(self):
        narrow = FWHM('narrow', prior=Uniform(0, 1000))
        broad = FWHM('broad', prior=Uniform(low=narrow * 2 + 150, high=5000))
        config = LineConfiguration()
        config.add_line('Ha_n', 6564.61 * u.AA, fwhm_gauss=narrow)
        config.add_line('Ha_b', 6564.61 * u.AA, fwhm_gauss=broad)

        d = config.to_dict()
        config2 = LineConfiguration.from_dict(d)

        # broad prior depends on narrow token
        broad2 = config2._entries[1].fwhms['fwhm_gauss']
        narrow2 = config2._entries[0].fwhms['fwhm_gauss']
        assert narrow2 in broad2.prior.dependencies()

    def test_profile_roundtrip(self):
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA, profile='voigt')
        d = config.to_dict()
        config2 = LineConfiguration.from_dict(d)
        from unite.line.profiles import PseudoVoigt

        assert isinstance(config2._entries[0].profile, PseudoVoigt)

    def test_strength_roundtrip(self):
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA, strength=3.0)
        d = config.to_dict()
        config2 = LineConfiguration.from_dict(d)
        assert config2._entries[0].strength == pytest.approx(3.0)

    def test_fixed_prior_roundtrip(self):
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA, redshift=Redshift(prior=Fixed(0.0)))
        d = config.to_dict()
        config2 = LineConfiguration.from_dict(d)
        assert isinstance(config2._entries[0].redshift.prior, Fixed)
        assert config2._entries[0].redshift.prior.value == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Profile integration normalization tests
# ---------------------------------------------------------------------------


class TestProfileNormalization:
    def test_all_profiles_integrate_to_unity(self):
        """Verify that all profile integration functions sum to ~1.0."""
        import jax.numpy as jnp

        from unite.line.functions import (
            _integrate_cauchy,
            _integrate_laplace,
            integrate_gaussHermite,
            integrate_gaussian,
            integrate_gaussianLaplace,
            integrate_split_normal,
            integrate_voigt,
        )

        # Create test bins centered around 5000 with very wide coverage to capture full profiles
        # Use 1000 bins from 0 to 10000 to ensure we capture even heavy-tailed distributions
        # Note: Cauchy (Lorentzian) has infinite support, so we can only approximate unity
        low = jnp.linspace(0, 10000, 1000)[:-1]  # 1000 bins
        high = jnp.linspace(0, 10000, 1000)[1:]
        center = jnp.array([5000.0])
        lsf_fwhm = jnp.array([10.0])

        # Test Gaussian
        fwhm = jnp.array([100.0])
        lsf_fwhm = jnp.array([10.0])  # Small LSF
        result = integrate_gaussian(low, high, center, lsf_fwhm, fwhm)
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-5)

        # Test Cauchy (Lorentzian) - has heavy tails, so we expect a looser tolerance
        fwhm = jnp.array([100.0])
        lsf_fwhm = jnp.array([10.0])  # Small LSF (ignored for Cauchy)
        result = _integrate_cauchy(low, high, center, lsf_fwhm, lsf_fwhm)
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=5e-3)

        # With our wide range (0-10000), we should get close to 1.0
        fwhm = jnp.array([100.0])
        lsf_fwhm = jnp.array([10.0])  # Small LSF (ignored for Cauchy)
        result = integrate_voigt(low, high, center, 0.0, 0.0, fwhm)
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=5e-2)

        # Test Laplace - has exponential tails
        fwhm = jnp.array([100.0])
        lsf_fwhm = jnp.array([10.0])  # Small LSF
        result = _integrate_laplace(low, high, center, lsf_fwhm, fwhm)
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-4)

        # Test PseudoVoigt - has Lorentzian component with heavy tails
        fwhm_g = jnp.array([80.0])
        fwhm_l = jnp.array([50.0])
        lsf_fwhm = jnp.array([10.0])  # Small LSF
        result = integrate_voigt(low, high, center, lsf_fwhm, fwhm_g, fwhm_l)
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=5e-3)

        # Test SEMG (Gaussian-Laplace convolution) - has exponential tails
        fwhm_g = jnp.array([80.0])
        fwhm_l = jnp.array([50.0])
        lsf_fwhm = jnp.array([10.0])  # Small LSF
        result = integrate_gaussianLaplace(low, high, center, lsf_fwhm, fwhm_g, fwhm_l)
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-4)

        # Test Gauss-Hermite
        fwhm_lsf = jnp.array([10.0])
        fwhm_g = jnp.array([80.0])
        h3 = jnp.array([0.1])
        h4 = jnp.array([0.05])
        result = integrate_gaussHermite(low, high, center, fwhm_lsf, fwhm_g, h3, h4)
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-5)

        # Test SplitNormal
        fwhm_blue = jnp.array([100.0])
        fwhm_red = jnp.array([60.0])
        result = integrate_split_normal(
            low, high, center, lsf_fwhm, fwhm_blue, fwhm_red
        )
        assert jnp.isclose(jnp.sum(result), 1.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# add_lines
# ---------------------------------------------------------------------------


class TestAddLines:
    def test_broadcast_single_redshift(self):
        z = Redshift('z')
        config = LineConfiguration()
        config.add_lines('[NII]', [6585.27 * u.AA, 6549.86 * u.AA], redshift=z)
        assert len(config) == 2
        assert config._entries[0].redshift is config._entries[1].redshift

    def test_per_line_flux(self):
        config = LineConfiguration()
        f1 = Flux('f1')
        f2 = Flux('f2')
        config.add_lines('X', [5000.0 * u.AA, 5100.0 * u.AA], flux=[f1, f2])
        assert config._entries[0].flux is f1
        assert config._entries[1].flux is f2

    def test_empty_centers_raises(self):
        config = LineConfiguration()
        with pytest.raises(ValueError, match="'centers' must be non-empty"):
            config.add_lines('X', [])

    def test_wrong_length_raises(self):
        config = LineConfiguration()
        z = Redshift('z')
        with pytest.raises(ValueError, match="'redshift' has"):
            config.add_lines(
                'X',
                [5000.0 * u.AA, 5100.0 * u.AA],
                redshift=[z, z, z],  # 3 values for 2 centers
            )


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_strict_no_collision(self):
        a = LineConfiguration()
        a.add_line('Ha', 6563.0 * u.AA, redshift=Redshift('z_a'))
        b = LineConfiguration()
        b.add_line('Hb', 4861.0 * u.AA, redshift=Redshift('z_b'))
        merged = a.merge(b, strict=True)
        assert len(merged) == 2

    def test_merge_strict_collision_raises(self):
        z = Redshift('shared_z')
        a = LineConfiguration()
        a.add_line('Ha', 6563.0 * u.AA, redshift=z)
        # b uses a different Redshift instance but with the same name
        b = LineConfiguration()
        b.add_line('Hb', 4861.0 * u.AA, redshift=Redshift('shared_z'))
        with pytest.raises(ValueError, match='Token name collision'):
            a.merge(b, strict=True)

    def test_merge_lenient_shares_tokens(self):
        a = LineConfiguration()
        a.add_line('Ha', 6563.0 * u.AA, redshift=Redshift('shared_z'))
        b = LineConfiguration()
        b.add_line('Hb', 4861.0 * u.AA, redshift=Redshift('shared_z'))
        merged = a.merge(b, strict=False)
        assert len(merged) == 2
        # Both lines should share the same redshift token (from a)
        z_ids = {id(e.redshift) for e in merged._entries}
        assert len(z_ids) == 1

    def test_add_operator(self):
        a = LineConfiguration()
        a.add_line('Ha', 6563.0 * u.AA, redshift=Redshift('z_a'))
        b = LineConfiguration()
        b.add_line('Hb', 4861.0 * u.AA, redshift=Redshift('z_b'))
        merged = a + b
        assert len(merged) == 2


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_yaml_roundtrip(self):
        lc = LineConfiguration()
        z = Redshift('nlr')
        lc.add_line('Ha', 6563.0 * u.AA, redshift=z)
        lc.add_line('Hb', 4861.0 * u.AA, redshift=z)
        text = lc.to_yaml()
        lc2 = LineConfiguration.from_yaml(text)
        assert len(lc2) == 2

    def test_file_roundtrip(self, tmp_path):
        lc = LineConfiguration()
        z = Redshift('nlr')
        lc.add_line('Ha', 6563.0 * u.AA, redshift=z)
        path = tmp_path / 'lines.yaml'
        lc.save(path)
        lc2 = LineConfiguration.load(path)
        assert len(lc2) == 1


# ---------------------------------------------------------------------------
# Profile.integrate() method (direct call covers lines 97-101)
# ---------------------------------------------------------------------------

_PROFILE_PARAMS = [
    (Gaussian(), {'fwhm_gauss': 100.0}),
    (Cauchy(), {'fwhm_lorentzian': 100.0}),
    (PseudoVoigt(), {'fwhm_gauss': 80.0, 'fwhm_lorentz': 50.0}),
    (Laplace(), {'fwhm_exp': 100.0}),
    (SEMG(), {'fwhm_gauss': 80.0, 'fwhm_exp': 50.0}),
    (GaussHermite(), {'fwhm_gauss': 80.0, 'h3': 0.1, 'h4': 0.05}),
    (SplitNormal(), {'fwhm_blue': 100.0, 'fwhm_red': 60.0}),
]


class TestProfileIntegrateMethod:
    """Test Profile.integrate() dispatches correctly via jax_branch()."""

    @pytest.mark.parametrize('profile,extra_kwargs', _PROFILE_PARAMS)
    def test_integrate_sums_to_near_unity(self, profile, extra_kwargs):
        import jax.numpy as jnp

        low = jnp.linspace(4500.0, 5500.0, 500)[:-1]
        high = jnp.linspace(4500.0, 5500.0, 500)[1:]
        result = profile.integrate(low, high, center=5000.0, lsf_fwhm=5.0, **extra_kwargs)
        # Cauchy has heavy tails so allow wider tolerance
        assert float(jnp.sum(result)) == pytest.approx(1.0, rel=0.1)

    @pytest.mark.parametrize('profile,extra_kwargs', _PROFILE_PARAMS)
    def test_integrate_returns_nonneg(self, profile, extra_kwargs):
        import jax.numpy as jnp

        low = jnp.linspace(4900.0, 5100.0, 50)[:-1]
        high = jnp.linspace(4900.0, 5100.0, 50)[1:]
        result = profile.integrate(low, high, center=5000.0, lsf_fwhm=5.0, **extra_kwargs)
        assert jnp.all(result >= 0.0)


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

_ALL_PROFILE_CLASSES = [Gaussian, Cauchy, PseudoVoigt, Laplace, SEMG, GaussHermite, SplitNormal]


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
# resolve_profile: TypeError for invalid input (lines 507-508)
# ---------------------------------------------------------------------------


def test_resolve_profile_invalid_type_raises():
    from unite.line.profiles import resolve_profile

    with pytest.raises(TypeError, match='str or Profile'):
        resolve_profile(42)


# ---------------------------------------------------------------------------
# SEMG.default_priors (profiles.py line 348)
# ---------------------------------------------------------------------------


def test_semg_default_priors():
    """SEMG.default_priors() returns a dict with fwhm_gauss and fwhm_exp (line 348)."""
    priors = SEMG().default_priors()
    assert 'fwhm_gauss' in priors
    assert 'fwhm_exp' in priors
