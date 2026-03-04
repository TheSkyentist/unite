"""Tests for unite.line (parameter tokens and LineConfiguration)."""

import pytest
from astropy import units as u

from unite.line.config import FWHM, Flux, LineConfiguration, Param, Redshift
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
