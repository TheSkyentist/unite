"""Tests for unite.line parameter tokens and LineConfiguration."""

import pytest
from astropy import units as u

from unite.line.config import FWHM, Flux, LineConfiguration, LineShape, Redshift, Tau
from unite.line.profiles import GaussHermite, Gaussian, PseudoVoigt
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
        assert z.label == 'nlr'
        assert z.name is None  # site name is set at registration time

    def test_custom_prior(self):
        p = TruncatedNormal(0.0, 0.01, -0.05, 0.05)
        z = Redshift('nlr', prior=p)
        assert z.prior is p

    def test_arithmetic_returns_expr(self):
        from unite.prior import _Expr

        z = Redshift()
        ref = z * 2
        assert isinstance(ref, _Expr)
        assert ref.resolve({z: 3.0}) == pytest.approx(6.0)

    def test_add_returns_expr(self):
        from unite.prior import _Expr

        z = Redshift()
        ref = z + 0.01
        assert isinstance(ref, _Expr)
        assert ref.resolve({z: 0.0}) == pytest.approx(0.01)

    def test_cross_kind_ref_raises(self):
        fwhm = FWHM()
        with pytest.raises(TypeError, match='same kind of parameter'):
            Redshift(prior=Uniform(low=fwhm * 2, high=1000))


class TestFWHM:
    def test_default_prior(self):
        f = FWHM()
        assert isinstance(f.prior, Uniform)

    def test_named_with_prior(self):
        f = FWHM('broad', prior=Uniform(500, 5000))
        assert f.label == 'broad'
        assert f.name is None  # site name is set at registration time
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
        assert f.label == 'Ha_flux'
        assert f.name is None  # site name is set at registration time


class TestLineShape:
    def test_default_prior(self):
        p = LineShape()
        assert isinstance(p.prior, TruncatedNormal)

    def test_custom_prior(self):
        p = LineShape('h3', prior=Uniform(-0.5, 0.5))
        assert p.prior.low == pytest.approx(-0.5)


class TestTau:
    def test_default_prior(self):
        t = Tau()
        assert isinstance(t.prior, Uniform)
        assert t.prior.low == pytest.approx(0.0)
        assert t.prior.high == pytest.approx(10.0)

    def test_named(self):
        t = Tau('hi_absorber')
        assert t.label == 'hi_absorber'
        assert t.name is None  # site name set at registration time

    def test_custom_prior(self):
        p = Uniform(0, 50)
        t = Tau('deep', prior=p)
        assert t.prior is p

    def test_category_prefix(self):
        from unite.line.config import _prefix_for

        assert _prefix_for('tau') == 'tau'


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

    def test_duplicate_name_raises(self):
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA)
        with pytest.raises(ValueError, match='already used'):
            config.add_line('Ha', 6564.61 * u.AA)

    def test_duplicate_name_raises_even_with_different_tokens(self):
        z = Redshift('z')
        f_narrow = FWHM('narrow')
        f_broad = FWHM('broad')
        config = LineConfiguration()
        config.add_line('Ha', 6564.61 * u.AA, redshift=z, fwhm_gauss=f_narrow)
        with pytest.raises(ValueError, match='already used'):
            config.add_line('Ha', 6564.61 * u.AA, redshift=z, fwhm_gauss=f_broad)

    def test_two_components_require_distinct_names(self):
        z = Redshift('z')
        f_narrow = FWHM('narrow')
        f_broad = FWHM('broad')
        config = LineConfiguration()
        config.add_line('Ha_narrow', 6564.61 * u.AA, redshift=z, fwhm_gauss=f_narrow)
        config.add_line('Ha_broad', 6564.61 * u.AA, redshift=z, fwhm_gauss=f_broad)
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
            config.add_line('Ha', 6564.61 * u.AA, fwhm_gauss=LineShape())  # type: ignore[arg-type]

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
# add_lines
# ---------------------------------------------------------------------------


class TestAddLines:
    def test_broadcast_single_redshift(self):
        z = Redshift('z')
        config = LineConfiguration()
        config.add_lines('[NII]', [6585.27, 6549.86] * u.AA, redshift=z)
        assert len(config) == 2
        assert config._entries[0].redshift is config._entries[1].redshift

    def test_auto_names_use_center_value(self):
        config = LineConfiguration()
        config.add_lines('[NII]', [6585.27, 6549.86] * u.AA)
        assert config._entries[0].name == '[NII]_6585.27'
        assert config._entries[1].name == '[NII]_6549.86'

    def test_explicit_names_array(self):
        config = LineConfiguration()
        config.add_lines(['NII_6585', 'NII_6550'], [6585.27, 6549.86] * u.AA)
        assert config._entries[0].name == 'NII_6585'
        assert config._entries[1].name == 'NII_6550'

    def test_names_wrong_length_raises(self):
        config = LineConfiguration()
        with pytest.raises(ValueError, match="'name' sequence has length"):
            config.add_lines(['only_one'], [5000.0, 5100.0] * u.AA)

    def test_per_line_flux(self):
        config = LineConfiguration()
        f1 = Flux('f1')
        f2 = Flux('f2')
        config.add_lines('X', [5000.0, 5100.0] * u.AA, flux=[f1, f2])
        assert config._entries[0].flux is f1
        assert config._entries[1].flux is f2

    def test_empty_centers_raises(self):
        config = LineConfiguration()
        with pytest.raises(ValueError, match="'centers' must be non-empty"):
            config.add_lines('X', [] * u.AA)

    def test_wrong_length_raises(self):
        config = LineConfiguration()
        z = Redshift('z')
        with pytest.raises(ValueError, match="'redshift' has"):
            config.add_lines(
                'X',
                [5000.0, 5100.0] * u.AA,
                redshift=[z, z, z],  # 3 values for 2 centers
            )


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


class TestMerge:
    def test_merge_strict_no_collision(self):
        a = LineConfiguration()
        a.add_line('Ha', 6563.0 * u.AA, redshift=Redshift('a'))
        b = LineConfiguration()
        b.add_line('Hb', 4861.0 * u.AA, redshift=Redshift('b'))
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
        a.add_line('Ha', 6563.0 * u.AA, redshift=Redshift('a'))
        b = LineConfiguration()
        b.add_line('Hb', 4861.0 * u.AA, redshift=Redshift('b'))
        merged = a + b
        assert len(merged) == 2

    def test_merge_type_mismatch_raises(self):
        """Merging tokens with the same name but different types raises TypeError."""
        # Build lc1 with a FWHM token that gets name 'fwhm_gauss_foo'
        lc1 = LineConfiguration()
        lc1.add_line('Ha', 6563.0 * u.AA, fwhm_gauss=FWHM('foo'))

        # Build lc2 with a Flux token that we manually give the same site name
        lc2 = LineConfiguration()
        flux_tok = Flux()
        flux_tok.name = 'fwhm_gauss_foo'  # same site name, wrong type
        lc2.add_line('Hb', 4861.0 * u.AA, flux=flux_tok)

        with pytest.raises(TypeError, match='type'):
            lc1.merge(lc2, strict=False)


# ---------------------------------------------------------------------------
# LineShape via GaussHermite roundtrip (covers from_dict LineShape branch)
# ---------------------------------------------------------------------------


class TestGaussHermiteRoundtrip:
    def test_gausshermite_roundtrip(self):
        """GaussHermite config survives to_dict/from_dict (tests LineShape section)."""
        lc = LineConfiguration()
        lc.add_line('Ha', 6563.0 * u.AA, profile=GaussHermite())
        d = lc.to_dict()
        lc2 = LineConfiguration.from_dict(d)
        assert len(lc2) == 1
        assert isinstance(lc2._entries[0].profile, GaussHermite)
        # h3 and h4 should be LineShape tokens
        assert isinstance(lc2._entries[0].fwhms.get('h3'), LineShape)
        assert isinstance(lc2._entries[0].fwhms.get('h4'), LineShape)


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
