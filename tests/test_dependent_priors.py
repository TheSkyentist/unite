"""Tests for deep and complex dependent prior chains.

This tests the core power feature of unite: arbitrary-depth dependency
chains between parameter tokens (FWHM, Redshift, Flux) expressed via
ParameterRef arithmetic.  The test scenarios model real astrophysical
constraints such as:

- Narrow → broad → outflow velocity ordering
- Doublet flux ratios (e.g. [NII] 3:1 ratio with flexibility)
- Multi-component lines with cascading FWHM constraints
- Complex redshift hierarchies (systemic → NLR → BLR → outflow)
"""

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random
from numpyro.infer import Predictive

from unite import model
from unite.instrument import Spectra
from unite.instrument.generic import GenericSpectrum, SimpleDisperser
from unite.line.config import FWHM, Flux, LineConfiguration, Redshift
from unite.prior import ParameterRef, TruncatedNormal, Uniform, topological_sort

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spectrum(wl_range=(6400, 6700), npix=200, name='test'):
    """Create a test spectrum covering the given range."""
    wl = np.linspace(*wl_range, npix) * u.AA
    disperser = SimpleDisperser(wavelength=wl.value, unit=u.AA, R=3000.0, name=name)
    low = wl - 0.5 * np.gradient(wl)
    high = wl + 0.5 * np.gradient(wl)
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    rng = np.random.default_rng(42)
    flux = (10.0 + rng.normal(0, 1, npix)) * flux_unit
    error = np.full(npix, 1.0) * flux_unit
    return GenericSpectrum(
        low=low, high=high, flux=flux, error=error, disperser=disperser, name=name
    )


def _prepare_and_build(line_config, spectra, cont_config=None):
    """Prepare spectra and build model."""
    spectra.prepare(line_config, cont_config)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    return model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build()


# ---------------------------------------------------------------------------
# Deep topological sort chains (3+ levels)
# ---------------------------------------------------------------------------


class TestDeepChains:
    """Tests for dependency chains deeper than 2 levels."""

    def test_three_level_fwhm_chain(self):
        """A → B → C: narrow → medium → broad FWHM ordering."""
        narrow = FWHM('narrow', prior=Uniform(50, 300))
        medium = FWHM('medium', prior=Uniform(low=narrow + 100, high=1000))
        broad = FWHM('broad', prior=Uniform(low=medium + 200, high=3000))

        named_priors = {
            'narrow': narrow.prior,
            'medium': medium.prior,
            'broad': broad.prior,
        }
        param_to_name = {narrow: 'narrow', medium: 'medium', broad: 'broad'}
        order = topological_sort(named_priors, param_to_name)

        assert order.index('narrow') < order.index('medium')
        assert order.index('medium') < order.index('broad')

    def test_four_level_chain(self):
        """A → B → C → D: four-level deep chain."""
        a = FWHM('a', prior=Uniform(10, 100))
        b = FWHM('b', prior=Uniform(low=a + 50, high=500))
        c = FWHM('c', prior=Uniform(low=b * 1.5, high=1000))
        d = FWHM('d', prior=Uniform(low=c + 100, high=5000))

        named_priors = {'a': a.prior, 'b': b.prior, 'c': c.prior, 'd': d.prior}
        param_to_name = {a: 'a', b: 'b', c: 'c', d: 'd'}
        order = topological_sort(named_priors, param_to_name)

        assert order == ['a', 'b', 'c', 'd']

    def test_three_level_redshift_chain(self):
        """Systemic → NLR → outflow redshift hierarchy."""
        z_sys = Redshift('z_sys', prior=Uniform(-0.01, 0.01))
        z_nlr = Redshift(
            'z_nlr',
            prior=TruncatedNormal(
                loc=z_sys, scale=0.001, low=z_sys - 0.005, high=z_sys + 0.005
            ),
        )
        z_out = Redshift('z_out', prior=Uniform(low=z_nlr - 0.01, high=z_nlr))

        named_priors = {
            'z_sys': z_sys.prior,
            'z_nlr': z_nlr.prior,
            'z_out': z_out.prior,
        }
        param_to_name = {z_sys: 'z_sys', z_nlr: 'z_nlr', z_out: 'z_out'}
        order = topological_sort(named_priors, param_to_name)

        assert order.index('z_sys') < order.index('z_nlr')
        assert order.index('z_nlr') < order.index('z_out')


# ---------------------------------------------------------------------------
# Parameter token directly as bound (_normalize_bound path)
# ---------------------------------------------------------------------------


class TestParameterAsBound:
    """Tests for passing a Parameter token directly as a prior bound."""

    def test_fwhm_token_as_low_bound(self):
        """FWHM token passed directly (not via arithmetic) as low bound."""
        narrow = FWHM('narrow', prior=Uniform(50, 300))
        # Passing narrow directly, no arithmetic
        broad = FWHM('broad', prior=Uniform(low=narrow, high=3000))

        assert narrow in broad.prior.dependencies()
        # The low bound should be a ParameterRef with scale=1 offset=0
        assert isinstance(broad.prior.low, ParameterRef)
        assert broad.prior.low.scale == 1.0
        assert broad.prior.low.offset == 0.0

    def test_parameter_as_bound_resolves(self):
        """Parameter used directly as bound should resolve correctly."""
        narrow = FWHM('narrow', prior=Uniform(50, 300))
        broad = FWHM('broad', prior=Uniform(low=narrow, high=3000))

        # Resolve: narrow sampled at 200 → broad.low = 200
        context = {narrow: 200.0}
        d = broad.prior.to_dist(context)
        assert float(d.low) == pytest.approx(200.0)

    def test_redshift_token_as_loc(self):
        """Redshift token passed directly as TruncatedNormal loc."""
        z_sys = Redshift('z_sys', prior=Uniform(-0.01, 0.01))
        z_nlr = Redshift(
            'z_nlr', prior=TruncatedNormal(loc=z_sys, scale=0.001, low=-0.02, high=0.02)
        )
        assert z_sys in z_nlr.prior.dependencies()


# ---------------------------------------------------------------------------
# TruncatedNormal with multiple dependent bounds
# ---------------------------------------------------------------------------


class TestTruncatedNormalDependencies:
    """Tests for TruncatedNormal where loc, low, and high all depend."""

    def test_all_three_bounds_depend_on_different_params(self):
        """loc, low, high each reference different tokens."""
        center = FWHM('center', prior=Uniform(200, 800))
        lower = FWHM('lower', prior=Uniform(50, 200))
        upper = FWHM('upper', prior=Uniform(800, 2000))

        constrained = FWHM(
            'constrained',
            prior=TruncatedNormal(loc=center, scale=50.0, low=lower, high=upper),
        )

        deps = constrained.prior.dependencies()
        assert center in deps
        assert lower in deps
        assert upper in deps

    def test_loc_and_low_same_token(self):
        """loc and low both reference the same token."""
        base = FWHM('base', prior=Uniform(100, 500))
        derived = FWHM(
            'derived',
            prior=TruncatedNormal(loc=base + 100, scale=30.0, low=base, high=2000),
        )

        deps = derived.prior.dependencies()
        assert base in deps
        assert len(deps) == 1  # only one unique dependency

        # Resolve — just verify the distribution is created without error
        context = {base: 300.0}
        d = derived.prior.to_dist(context)
        assert float(d.low) == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Flux ratio constraints
# ---------------------------------------------------------------------------


class TestFluxDependencies:
    """Tests for dependent flux priors (doublet ratios etc.)."""

    def test_flux_ratio_via_parameter_ref(self):
        """[NII] doublet: f_6585 and f_6549 with ratio ≈ 2.95."""
        f_strong = Flux('NII_6585', prior=Uniform(0, 10))
        # Weak line flux is constrained to be ~ 1/2.95 of the strong line
        f_weak = Flux(
            'NII_6549',
            prior=TruncatedNormal(
                loc=f_strong / 2.95, scale=0.1, low=f_strong / 4.0, high=f_strong / 2.0
            ),
        )

        deps = f_weak.prior.dependencies()
        assert f_strong in deps

        # Resolve
        context = {f_strong: 5.0}
        d = f_weak.prior.to_dist(context)
        assert float(d.low) == pytest.approx(5.0 / 4.0)
        assert float(d.high) == pytest.approx(5.0 / 2.0)

    def test_flux_chain(self):
        """Three-line flux chain: Ha → [NII]6585 → [NII]6549."""
        f_ha = Flux('Ha', prior=Uniform(0, 20))
        f_nii_s = Flux('NII_6585', prior=Uniform(low=0, high=f_ha * 2))
        f_nii_w = Flux(
            'NII_6549',
            prior=TruncatedNormal(
                loc=f_nii_s / 2.95, scale=0.05, low=f_nii_s / 4.0, high=f_nii_s / 2.0
            ),
        )

        named_priors = {
            'Ha': f_ha.prior,
            'NII_6585': f_nii_s.prior,
            'NII_6549': f_nii_w.prior,
        }
        param_to_name = {f_ha: 'Ha', f_nii_s: 'NII_6585', f_nii_w: 'NII_6549'}
        order = topological_sort(named_priors, param_to_name)

        assert order.index('Ha') < order.index('NII_6585')
        assert order.index('NII_6585') < order.index('NII_6549')

    def test_cross_kind_ref_forbidden(self):
        """An FWHM bound on a Flux prior should raise TypeError."""
        fwhm_tok = FWHM('w', prior=Uniform(50, 500))
        with pytest.raises(TypeError, match='same kind'):
            Flux('f', prior=Uniform(low=fwhm_tok * 0.1, high=10))


# ---------------------------------------------------------------------------
# Complex real-world line configuration
# ---------------------------------------------------------------------------


class TestComplexLineConfig:
    """Tests for realistic multi-component line configurations with deep priors."""

    def _make_three_component_config(self):
        """Build a 3-component H-alpha + [NII] config.

        Components:
          narrow (NLR): shared z_nlr, fwhm_narrow
          broad (BLR): z_blr depends on z_nlr, fwhm_broad > fwhm_narrow + 200
          outflow: z_out = z_nlr - offset, fwhm_out > fwhm_broad
        """
        # -- Redshift hierarchy --
        z_nlr = Redshift('z_nlr', prior=Uniform(-0.005, 0.005))
        z_blr = Redshift(
            'z_blr',
            prior=TruncatedNormal(
                loc=z_nlr, scale=0.002, low=z_nlr - 0.01, high=z_nlr + 0.01
            ),
        )
        z_out = Redshift('z_out', prior=Uniform(low=z_nlr - 0.02, high=z_nlr))

        # -- FWHM hierarchy --
        fwhm_narrow = FWHM('fwhm_narrow', prior=Uniform(50, 500))
        fwhm_broad = FWHM('fwhm_broad', prior=Uniform(low=fwhm_narrow + 200, high=5000))
        fwhm_out = FWHM('fwhm_out', prior=Uniform(low=fwhm_broad, high=8000))

        # -- Flux with doublet ratio --
        f_ha_n = Flux('Ha_n', prior=Uniform(0, 10))
        f_ha_b = Flux('Ha_b', prior=Uniform(0, 10))
        f_ha_out = Flux('Ha_out', prior=Uniform(0, 10))
        f_nii_s = Flux('NII_s', prior=Uniform(0, 10))
        f_nii_w = Flux(
            'NII_w',
            prior=TruncatedNormal(
                loc=f_nii_s / 2.95, scale=0.1, low=f_nii_s / 4.0, high=f_nii_s / 2.0
            ),
        )

        lc = LineConfiguration()

        # Narrow lines
        lc.add_line(
            'Ha', 6564.61 * u.AA, redshift=z_nlr, fwhm_gauss=fwhm_narrow, flux=f_ha_n
        )
        lc.add_line(
            'NII_6585',
            6585.27 * u.AA,
            redshift=z_nlr,
            fwhm_gauss=fwhm_narrow,
            flux=f_nii_s,
        )
        lc.add_line(
            'NII_6549',
            6549.86 * u.AA,
            redshift=z_nlr,
            fwhm_gauss=fwhm_narrow,
            flux=f_nii_w,
        )

        # Broad lines
        lc.add_line(
            'Ha', 6564.61 * u.AA, redshift=z_blr, fwhm_gauss=fwhm_broad, flux=f_ha_b
        )

        # Outflow lines
        lc.add_line(
            'Ha', 6564.61 * u.AA, redshift=z_out, fwhm_gauss=fwhm_out, flux=f_ha_out
        )

        return lc

    def test_three_component_builds(self):
        """Three-component config should build without errors."""
        lc = self._make_three_component_config()
        assert len(lc) == 5

    def test_three_component_serialization_roundtrip(self):
        """Full serialization round-trip preserves deep dependencies."""
        lc = self._make_three_component_config()
        d = lc.to_dict()
        lc2 = LineConfiguration.from_dict(d)
        assert len(lc2) == 5

        # Verify the z_blr → z_nlr dependency survived
        z_blr = lc2._entries[3].redshift
        z_nlr = lc2._entries[0].redshift
        assert z_nlr in z_blr.prior.dependencies()

        # Verify the z_out → z_nlr dependency survived
        z_out = lc2._entries[4].redshift
        assert z_nlr in z_out.prior.dependencies()

        # Verify the fwhm chain survived: fwhm_out → fwhm_broad → fwhm_narrow
        fwhm_narrow = lc2._entries[0].fwhms['fwhm_gauss']
        fwhm_broad = lc2._entries[3].fwhms['fwhm_gauss']
        fwhm_out = lc2._entries[4].fwhms['fwhm_gauss']
        assert fwhm_narrow in fwhm_broad.prior.dependencies()
        assert fwhm_broad in fwhm_out.prior.dependencies()

        # Verify the flux ratio survived: NII_w → NII_s
        f_nii_s = lc2._entries[1].flux
        f_nii_w = lc2._entries[2].flux
        assert f_nii_s in f_nii_w.prior.dependencies()

    def test_three_component_yaml_roundtrip(self):
        """YAML round-trip preserves deep dependencies."""
        lc = self._make_three_component_config()
        yaml_str = lc.to_yaml()
        lc2 = LineConfiguration.from_yaml(yaml_str)
        assert len(lc2) == 5

        # Verify z chain
        z_nlr = lc2._entries[0].redshift
        z_blr = lc2._entries[3].redshift
        z_out = lc2._entries[4].redshift
        assert z_nlr in z_blr.prior.dependencies()
        assert z_nlr in z_out.prior.dependencies()

    def test_three_component_model_builds(self):
        """The full 3-component config should produce a valid numpyro model."""
        lc = self._make_three_component_config()
        spec = _make_spectrum()
        spectra = Spectra([spec], redshift=0.0)
        unite_model, unite_args = _prepare_and_build(lc, spectra)
        assert unite_model is not None
        assert unite_args is not None

    def test_three_component_model_samples(self):
        """The model with deep dependencies should execute via Predictive."""
        lc = self._make_three_component_config()
        spec = _make_spectrum()
        spectra = Spectra([spec], redshift=0.0)
        unite_model, unite_args = _prepare_and_build(lc, spectra)

        rng_key = random.PRNGKey(0)
        predictive = Predictive(unite_model, num_samples=5)
        samples = predictive(rng_key, unite_args)

        # Should have sampled all parameters
        param_keys = set(samples.keys())

        # Check that the z hierarchy parameters exist
        z_keys = [k for k in param_keys if 'z_nlr' in k or 'z_blr' in k or 'z_out' in k]
        assert len(z_keys) >= 3, f'Expected 3 z params, found {z_keys} in {param_keys}'

        # Check FWHM hierarchy
        fwhm_keys = [k for k in param_keys if 'fwhm' in k]
        assert len(fwhm_keys) >= 3  # fwhm_narrow, fwhm_broad, fwhm_out

        # Check all values are finite
        for k, v in samples.items():
            assert jnp.all(jnp.isfinite(v)), f'Non-finite values in {k}'


# ---------------------------------------------------------------------------
# Diamond dependencies (multiple parents)
# ---------------------------------------------------------------------------


class TestDiamondDependencies:
    """Tests for parameters that depend on multiple parents."""

    def test_fwhm_depends_on_two_parents(self):
        """A FWHM with low from one token and high from another."""
        lower = FWHM('lower', prior=Uniform(50, 200))
        upper = FWHM('upper', prior=Uniform(800, 2000))
        mid = FWHM('mid', prior=Uniform(low=lower + 50, high=upper - 50))

        deps = mid.prior.dependencies()
        assert lower in deps
        assert upper in deps

        named_priors = {'lower': lower.prior, 'upper': upper.prior, 'mid': mid.prior}
        param_to_name = {lower: 'lower', upper: 'upper', mid: 'mid'}
        order = topological_sort(named_priors, param_to_name)
        assert order.index('lower') < order.index('mid')
        assert order.index('upper') < order.index('mid')

    def test_diamond_with_convergent_child(self):
        """A → C, B → C, A → D, B → D (diamond with two children)."""
        a = FWHM('a', prior=Uniform(10, 100))
        b = FWHM('b', prior=Uniform(100, 500))
        c = FWHM('c', prior=Uniform(low=a, high=b))
        d = FWHM('d', prior=Uniform(low=a + 50, high=b - 50))

        named_priors = {'a': a.prior, 'b': b.prior, 'c': c.prior, 'd': d.prior}
        param_to_name = {a: 'a', b: 'b', c: 'c', d: 'd'}
        order = topological_sort(named_priors, param_to_name)

        # a and b before both c and d
        assert order.index('a') < order.index('c')
        assert order.index('a') < order.index('d')
        assert order.index('b') < order.index('c')
        assert order.index('b') < order.index('d')


# ---------------------------------------------------------------------------
# Complex arithmetic expressions
# ---------------------------------------------------------------------------


class TestComplexArithmetic:
    """Tests for complex ParameterRef arithmetic chains."""

    def test_scale_then_add(self):
        """param * 2 + 150."""
        tok = FWHM('x', prior=Uniform(0, 500))
        ref = tok * 2 + 150
        assert ref.resolve({tok: 100.0}) == pytest.approx(350.0)

    def test_divide_then_subtract(self):
        """param / 3 - 50."""
        tok = FWHM('x', prior=Uniform(0, 500))
        ref = tok / 3 - 50
        assert ref.resolve({tok: 300.0}) == pytest.approx(50.0)

    def test_rsub(self):
        """1000 - param."""
        tok = FWHM('x', prior=Uniform(0, 500))
        ref = 1000 - tok
        assert ref.resolve({tok: 300.0}) == pytest.approx(700.0)

    def test_chained_multiply_add(self):
        """(param * 0.5 + 100) — verifying resolve."""
        tok = FWHM('x', prior=Uniform(0, 500))
        ref = tok * 0.5 + 100
        dependent = FWHM('y', prior=Uniform(low=ref, high=5000))
        context = {tok: 400.0}
        d = dependent.prior.to_dist(context)
        assert float(d.low) == pytest.approx(300.0)

    def test_nested_ref_in_truncated_normal(self):
        """TruncatedNormal with all bounds as ParameterRef expressions."""
        base = FWHM('base', prior=Uniform(100, 500))
        derived = FWHM(
            'derived',
            prior=TruncatedNormal(
                loc=base * 1.5 + 50, scale=30.0, low=base + 20, high=base * 3
            ),
        )
        context = {base: 200.0}
        d = derived.prior.to_dist(context)
        assert float(d.low) == pytest.approx(220.0)
        assert float(d.high) == pytest.approx(600.0)


# ---------------------------------------------------------------------------
# Serialization of deep chains
# ---------------------------------------------------------------------------


class TestDeepChainSerialization:
    """Tests for serialization of deep dependency chains via LineConfiguration."""

    def test_three_level_fwhm_serialization(self):
        """A → B → C FWHM chain round-trips through to_dict/from_dict."""
        narrow = FWHM('narrow', prior=Uniform(50, 300))
        medium = FWHM('medium', prior=Uniform(low=narrow + 100, high=1500))
        broad = FWHM('broad', prior=Uniform(low=medium * 1.5, high=5000))

        lc = LineConfiguration()
        lc.add_line('Ha_n', 6564.61 * u.AA, fwhm_gauss=narrow)
        lc.add_line('Ha_m', 6564.61 * u.AA, fwhm_gauss=medium)
        lc.add_line('Ha_b', 6564.61 * u.AA, fwhm_gauss=broad)

        d = lc.to_dict()
        lc2 = LineConfiguration.from_dict(d)

        narrow2 = lc2._entries[0].fwhms['fwhm_gauss']
        medium2 = lc2._entries[1].fwhms['fwhm_gauss']
        broad2 = lc2._entries[2].fwhms['fwhm_gauss']

        # medium depends on narrow
        assert narrow2 in medium2.prior.dependencies()
        # broad depends on medium
        assert medium2 in broad2.prior.dependencies()
        # Verify the scale was preserved
        assert isinstance(broad2.prior.low, ParameterRef)
        assert broad2.prior.low.scale == pytest.approx(1.5)

    def test_flux_ratio_serialization(self):
        """Flux ratio dependencies survive serialization."""
        f_strong = Flux('f_s', prior=Uniform(0, 10))
        f_weak = Flux(
            'f_w',
            prior=TruncatedNormal(
                loc=f_strong / 2.95, scale=0.1, low=f_strong / 4.0, high=f_strong / 2.0
            ),
        )

        lc = LineConfiguration()
        lc.add_line('NII_6585', 6585.27 * u.AA, flux=f_strong)
        lc.add_line('NII_6549', 6549.86 * u.AA, flux=f_weak)

        d = lc.to_dict()
        lc2 = LineConfiguration.from_dict(d)

        f_s2 = lc2._entries[0].flux
        f_w2 = lc2._entries[1].flux
        assert f_s2 in f_w2.prior.dependencies()

        # Verify scale and offset preserved
        assert isinstance(f_w2.prior.loc, ParameterRef)
        assert f_w2.prior.loc.scale == pytest.approx(1.0 / 2.95, rel=1e-5)

    def test_file_roundtrip_deep_chain(self, tmp_path):
        """Deep chain survives save/load to file."""
        a = FWHM('a', prior=Uniform(10, 100))
        b = FWHM('b', prior=Uniform(low=a + 20, high=500))
        c = FWHM('c', prior=Uniform(low=b * 2, high=2000))

        lc = LineConfiguration()
        lc.add_line('X1', 5000.0 * u.AA, fwhm_gauss=a)
        lc.add_line('X2', 5100.0 * u.AA, fwhm_gauss=b)
        lc.add_line('X3', 5200.0 * u.AA, fwhm_gauss=c)

        path = tmp_path / 'deep_chain.yaml'
        lc.save(path)
        lc2 = LineConfiguration.load(path)

        a2 = lc2._entries[0].fwhms['fwhm_gauss']
        b2 = lc2._entries[1].fwhms['fwhm_gauss']
        c2 = lc2._entries[2].fwhms['fwhm_gauss']

        assert a2 in b2.prior.dependencies()
        assert b2 in c2.prior.dependencies()


# ---------------------------------------------------------------------------
# End-to-end model with deep dependencies
# ---------------------------------------------------------------------------


class TestEndToEndDeepDependencies:
    """End-to-end model building and sampling with deep dependency chains."""

    def test_narrow_broad_model_respects_ordering(self):
        """Verify sampled broad FWHM > sampled narrow FWHM + offset."""
        fwhm_narrow = FWHM('fwhm_narrow', prior=Uniform(50, 300))
        fwhm_broad = FWHM('fwhm_broad', prior=Uniform(low=fwhm_narrow + 200, high=3000))

        lc = LineConfiguration()
        z = Redshift('z', prior=Uniform(-0.005, 0.005))
        lc.add_line('Ha', 6564.61 * u.AA, redshift=z, fwhm_gauss=fwhm_narrow)
        lc.add_line('Ha', 6564.61 * u.AA, redshift=z, fwhm_gauss=fwhm_broad)

        spec = _make_spectrum()
        spectra = Spectra([spec], redshift=0.0)
        unite_model, unite_args = _prepare_and_build(lc, spectra)

        rng_key = random.PRNGKey(42)
        predictive = Predictive(unite_model, num_samples=50)
        samples = predictive(rng_key, unite_args)

        # Find the FWHM parameter keys
        narrow_key = next(k for k in samples if 'fwhm_narrow' in k)
        broad_key = next(k for k in samples if 'fwhm_broad' in k)

        # For ALL samples, broad > narrow + 200 (the prior constraint)
        assert jnp.all(samples[broad_key] >= samples[narrow_key] + 200)

    def test_three_level_fwhm_model_respects_ordering(self):
        """Verify A < B < C ordering in sampled values."""
        a = FWHM('fwhm_a', prior=Uniform(50, 200))
        b = FWHM('fwhm_b', prior=Uniform(low=a + 50, high=800))
        c = FWHM('fwhm_c', prior=Uniform(low=b + 50, high=3000))

        lc = LineConfiguration()
        z = Redshift('z', prior=Uniform(-0.005, 0.005))
        lc.add_line('L1', 6564.61 * u.AA, redshift=z, fwhm_gauss=a)
        lc.add_line('L2', 6564.61 * u.AA, redshift=z, fwhm_gauss=b)
        lc.add_line('L3', 6564.61 * u.AA, redshift=z, fwhm_gauss=c)

        spec = _make_spectrum()
        spectra = Spectra([spec], redshift=0.0)
        unite_model, unite_args = _prepare_and_build(lc, spectra)

        rng_key = random.PRNGKey(99)
        predictive = Predictive(unite_model, num_samples=50)
        samples = predictive(rng_key, unite_args)

        a_key = next(k for k in samples if 'fwhm_a' in k)
        b_key = next(k for k in samples if 'fwhm_b' in k)
        c_key = next(k for k in samples if 'fwhm_c' in k)

        assert jnp.all(samples[b_key] >= samples[a_key] + 50)
        assert jnp.all(samples[c_key] >= samples[b_key] + 50)

    def test_redshift_hierarchy_model(self):
        """Verify redshift hierarchy produces valid samples."""
        z_sys = Redshift('z_sys', prior=Uniform(-0.005, 0.005))
        z_nlr = Redshift(
            'z_nlr',
            prior=TruncatedNormal(
                loc=z_sys, scale=0.001, low=z_sys - 0.003, high=z_sys + 0.003
            ),
        )

        lc = LineConfiguration()
        lc.add_line('Ha_sys', 6564.61 * u.AA, redshift=z_sys)
        lc.add_line('Ha_nlr', 6564.61 * u.AA, redshift=z_nlr)

        spec = _make_spectrum()
        spectra = Spectra([spec], redshift=0.0)
        unite_model, unite_args = _prepare_and_build(lc, spectra)

        rng_key = random.PRNGKey(7)
        predictive = Predictive(unite_model, num_samples=50)
        samples = predictive(rng_key, unite_args)

        sys_key = next(k for k in samples if 'z_sys' in k)
        nlr_key = next(k for k in samples if 'z_nlr' in k)

        # z_nlr should be within z_sys ± 0.003
        assert jnp.all(samples[nlr_key] >= samples[sys_key] - 0.003 - 1e-6)
        assert jnp.all(samples[nlr_key] <= samples[sys_key] + 0.003 + 1e-6)
