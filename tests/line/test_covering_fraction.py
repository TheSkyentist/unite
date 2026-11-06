"""Tests for the covering-fraction parameter of absorption lines."""

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from unite import line, model, prior
from unite._compose import compose_from_profiles, compose_leave_one_out
from unite.compute import evaluate_model
from unite.continuum.config import (
    ContinuumConfiguration,
    ContinuumRegion,
    ContShape,
    Scale,
)
from unite.continuum.library import Linear
from unite.instrument.generic import SimpleDisperser
from unite.line import CoveringFraction, Tau
from unite.line.compute import _build_cover_params
from unite.spectrum import Spectra, Spectrum

FLUX_UNIT = u.Unit('1e-17 erg / (s cm2 AA)')

# ---------------------------------------------------------------------------
# compose-level tests (pure JAX)
# ---------------------------------------------------------------------------


def _compose_inputs():
    """One emission line (z0), two tau absorbers at z1 (cf, tau free per test)."""
    n_pts = 5
    profiles = jnp.array(
        [
            [0.1, 0.5, 1.0, 0.5, 0.1],  # emission
            [0.2, 0.6, 1.0, 0.6, 0.2],  # absorber A
            [0.0, 0.4, 0.8, 0.4, 0.0],  # absorber B
        ]
    )
    flux = jnp.array([3.0, 0.0, 0.0])
    is_tau = jnp.array([False, True, True])
    zo = np.array([0, 1, 1])
    applies = jnp.array((zo[None, :] > zo[:, None]) & np.asarray(is_tau)[None, :])
    cont = jnp.full((1, n_pts), 2.0)
    cont_applies = jnp.asarray(
        (zo[None, :] > np.array([0])[:, None]) & np.asarray(is_tau)[None, :]
    )
    return profiles, flux, is_tau, applies, cont_applies, cont


class TestCompose:
    def test_cf_one_matches_none(self):
        profiles, flux, is_tau, applies, cont_applies, cont = _compose_inputs()
        tau = jnp.array([0.0, 1.5, 0.7])
        base = compose_from_profiles(
            profiles, flux, tau, is_tau, applies, cont_applies, cont
        )
        cov = compose_from_profiles(
            profiles,
            flux,
            tau,
            is_tau,
            applies,
            cont_applies,
            cont,
            cover_per_line=jnp.ones(3),
        )
        np.testing.assert_allclose(cov, base, rtol=1e-12)

    def test_cf_zero_removes_absorber(self):
        profiles, flux, is_tau, applies, cont_applies, cont = _compose_inputs()
        tau = jnp.array([0.0, 1.5, 0.7])
        # cf = 0 for both absorbers -> same as tau = 0.
        cov = compose_from_profiles(
            profiles,
            flux,
            tau,
            is_tau,
            applies,
            cont_applies,
            cont,
            cover_per_line=jnp.zeros(3),
        )
        none = compose_from_profiles(
            profiles, flux, jnp.zeros(3), is_tau, applies, cont_applies, cont
        )
        np.testing.assert_allclose(cov, none, rtol=1e-12)

    def test_partial_cover_hand_computed(self):
        """Two independent screens: transmissions multiply."""
        profiles, flux, is_tau, applies, cont_applies, cont = _compose_inputs()
        tau = jnp.array([0.0, 1.5, 0.7])
        cf = jnp.array([1.0, 0.3, 0.6])
        got = compose_from_profiles(
            profiles, flux, tau, is_tau, applies, cont_applies, cont, cover_per_line=cf
        )
        t_a = 1 - 0.3 * (1 - np.exp(-1.5 * np.asarray(profiles[1])))
        t_b = 1 - 0.6 * (1 - np.exp(-0.7 * np.asarray(profiles[2])))
        expected = (3.0 * np.asarray(profiles[0]) + 2.0) * t_a * t_b
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_leave_one_out_consistent(self):
        profiles, flux, is_tau, applies, cont_applies, cont = _compose_inputs()
        tau = jnp.array([0.0, 1.5, 0.7])
        cf = jnp.array([1.0, 0.3, 0.6])
        kw = {'cover_per_line': cf}
        total, deltas = compose_leave_one_out(
            profiles, flux, tau, is_tau, applies, cont_applies, cont, **kw
        )
        total_ref = compose_from_profiles(
            profiles, flux, tau, is_tau, applies, cont_applies, cont, **kw
        )
        np.testing.assert_allclose(total, total_ref, rtol=1e-12)
        for j in (1, 2):
            tau_wo = tau.at[j].set(0.0)
            without = compose_from_profiles(
                profiles, flux, tau_wo, is_tau, applies, cont_applies, cont, **kw
            )
            np.testing.assert_allclose(deltas[j], total_ref - without, rtol=1e-10)

    def test_cf_one_exact_at_large_depth(self):
        """cf=1 must reduce to tau*phi even where exp(-x) underflows vs 1."""
        from unite._compose import _effective_depth

        phi = jnp.array([[1.0, 0.5], [1.0, 0.5]])
        tau = jnp.array([50.0, 50.0])
        depth = _effective_depth(tau, phi, jnp.array([1.0, 0.3]))
        np.testing.assert_allclose(depth[0], [50.0, 25.0], rtol=1e-12)
        assert np.all(np.isfinite(depth))

    def test_mixed_cover_deep_absorber_finite(self):
        """A deep cf=1 absorber next to a free-cf one stays finite with finite grads."""
        import jax

        profiles, flux, is_tau, applies, cont_applies, cont = _compose_inputs()

        def f(cf_b):
            return compose_from_profiles(
                profiles,
                flux,
                jnp.array([0.0, 80.0, 0.7]),  # A deep, implicitly cf=1
                is_tau,
                applies,
                cont_applies,
                cont,
                cover_per_line=jnp.array([1.0, 1.0, cf_b]),
            ).sum()

        assert np.isfinite(f(0.5))
        assert np.isfinite(jax.grad(f)(0.5))

    def test_gradient_finite_at_cf_one_and_large_tau(self):
        import jax

        profiles, flux, is_tau, applies, cont_applies, cont = _compose_inputs()

        def f(cf1, tau1):
            return compose_from_profiles(
                profiles,
                flux,
                jnp.array([0.0, tau1, 0.7]),
                is_tau,
                applies,
                cont_applies,
                cont,
                cover_per_line=jnp.array([1.0, cf1, 0.6]),
            ).sum()

        for cf1, tau1 in [(1.0, 2.0), (0.5, 300.0), (0.0, 1.0)]:
            g = jax.grad(f, argnums=(0, 1))(cf1, tau1)
            assert np.all(np.isfinite(g))


# ---------------------------------------------------------------------------
# configuration tests
# ---------------------------------------------------------------------------


def _abs_lc(cf=None, *, share=False):
    lc = line.LineConfiguration()
    z = line.Redshift(prior=prior.Fixed(0.0))
    fw = line.FWHM(prior=prior.Fixed(300.0))
    tau = Tau(prior=prior.Fixed(2.0))
    lc.add_line(
        'abs1', 6563.0 * u.AA, redshift=z, fwhm_gauss=fw, tau=tau, covering_fraction=cf
    )
    if share:
        lc.add_line(
            'abs2',
            6570.0 * u.AA,
            redshift=z,
            fwhm_gauss=fw,
            tau=Tau(prior=prior.Fixed(1.0)),
            covering_fraction=cf,
        )
    return lc


class TestConfig:
    def test_default_prior_is_fixed_one(self):
        cf = CoveringFraction()
        assert isinstance(cf.prior, prior.Fixed)
        assert cf.prior.value == 1.0

    def test_omitted_creates_no_parameter(self):
        cm = _abs_lc().build_matrices()
        assert cm.cover_names == []
        assert cm.cover_matrix.shape == (0, 1)
        assert cm.has_cover is False

    def test_fixed_one_token_is_noop_flag(self):
        cm = _abs_lc(CoveringFraction()).build_matrices()
        assert cm.cover_names == ['cf_abs1']
        assert cm.has_cover is False
        assert _build_cover_params(cm, {'cf_abs1': jnp.asarray(1.0)}, 1) is None

    def test_free_token_sets_flag_and_site_name(self):
        cf = CoveringFraction('abs', prior=prior.Uniform(0, 1))
        cm = _abs_lc(cf, share=True).build_matrices()
        assert cm.cover_names == ['cf_abs']
        assert cm.has_cover is True
        # shared token -> one parameter, both lines
        np.testing.assert_array_equal(np.asarray(cm.cover_matrix), [[1.0, 1.0]])

    def test_fixed_other_value_sets_flag(self):
        cm = _abs_lc(CoveringFraction(prior=prior.Fixed(0.5))).build_matrices()
        assert cm.has_cover is True

    def test_per_line_cover_with_mixed_tokens(self):
        lc = line.LineConfiguration()
        z = line.Redshift(prior=prior.Fixed(0.0))
        lc.add_line('a', 6563.0 * u.AA, redshift=z, tau=Tau())
        lc.add_line(
            'b',
            6570.0 * u.AA,
            redshift=z,
            tau=Tau(),
            covering_fraction=CoveringFraction('b', prior=prior.Uniform(0, 1)),
        )
        cm = lc.build_matrices()
        cover = _build_cover_params(cm, {'cf_b': jnp.asarray(0.4)}, 2)
        np.testing.assert_allclose(cover, [1.0, 0.4])

    def test_requires_tau(self):
        lc = line.LineConfiguration()
        with pytest.raises(TypeError, match='absorption'):
            lc.add_line('Ha', 6563.0 * u.AA, covering_fraction=CoveringFraction())

    def test_wrong_type(self):
        lc = line.LineConfiguration()
        with pytest.raises(TypeError, match='CoveringFraction'):
            lc.add_line(
                'abs',
                6563.0 * u.AA,
                tau=Tau(),
                covering_fraction=Tau(),  # ty: ignore[invalid-argument-type]
            )

    def test_add_lines_shared_token(self):
        lc = line.LineConfiguration()
        cf = CoveringFraction('dbl', prior=prior.Uniform(0, 1))
        lc.add_lines('abs', [6563.0, 6585.0] * u.AA, tau=Tau('t'), covering_fraction=cf)
        cm = lc.build_matrices()
        assert cm.cover_names == ['cf_dbl']

    def test_yaml_roundtrip(self):
        cf = CoveringFraction('abs', prior=prior.Uniform(0.1, 1.0))
        lc = _abs_lc(cf, share=True)
        lc2 = line.LineConfiguration.from_yaml(lc.to_yaml())
        assert lc2.to_dict() == lc.to_dict()
        assert 'covering_fraction' in lc.to_dict()
        assert isinstance(lc2['abs1'].covering_fraction, CoveringFraction)
        assert lc2['abs1'].covering_fraction is lc2['abs2'].covering_fraction
        assert lc2['abs1'].covering_fraction.prior.low == 0.1

    def test_yaml_omitted_has_no_section(self):
        d = _abs_lc().to_dict()
        assert 'covering_fraction' not in d
        assert 'covering_fraction' not in d['lines'][0]

    def test_merge_and_filter_keep_token(self):
        cf = CoveringFraction('abs', prior=prior.Uniform(0, 1))
        a = _abs_lc(cf)
        b = line.LineConfiguration()
        b.add_line(
            'Ha', 6563.0 * u.AA, redshift=line.Redshift('other', prior=prior.Fixed(0.0))
        )
        merged = a + b
        assert merged['abs1'].covering_fraction is cf
        assert merged._filter([True, False])['abs1'].covering_fraction is cf
        merged.remove_line('Ha')
        assert merged['abs1'].covering_fraction is cf

    def test_repr_lists_section(self):
        cf = CoveringFraction('abs', prior=prior.Uniform(0, 1))
        assert 'CoveringFraction' in repr(_abs_lc(cf))


# ---------------------------------------------------------------------------
# end-to-end model tests
# ---------------------------------------------------------------------------


def _spectrum(npix=80):
    wavelength = np.linspace(6500, 6600, npix) * u.AA
    disperser = SimpleDisperser(wavelength=wavelength, R=3000.0, name='cf')
    grad = np.gradient(wavelength)
    rng = np.random.default_rng(1)
    flux = (5.0 + rng.normal(0, 0.5, npix)) * FLUX_UNIT
    return Spectrum(
        low=wavelength - 0.5 * grad,
        high=wavelength + 0.5 * grad,
        flux=flux,
        error=np.full(npix, 1.0) * FLUX_UNIT,
        disperser=disperser,
        name='cf',
    )


def _build(lc, mode='analytic'):
    region = ContinuumRegion(6480.0 * u.AA, 6650.0 * u.AA, form=Linear())
    region.params['scale'] = Scale(prior=prior.Fixed(1.0))
    region.params['angle'] = ContShape(prior=prior.Fixed(0.0))
    cc = ContinuumConfiguration([region])
    spectra = Spectra([_spectrum()], redshift=0.0)
    lc_prep, cc_prep = spectra.prepare(lc, cc)
    spectra.line_scale = 1.0 * FLUX_UNIT * u.AA
    spectra.continuum_scale = 1.0 * FLUX_UNIT
    return model.ModelBuilder(lc_prep, cc_prep, spectra).build(integration_mode=mode)


@pytest.mark.parametrize('mode', ['analytic', 'convolution'])
def test_model_partial_cover_relation(mode):
    """With one absorber over a flat continuum, T_cf = 1 - cf (1 - T_1)."""
    cf_val = 0.35
    _, args1 = _build(_abs_lc(), mode)
    _, args_cf = _build(_abs_lc(CoveringFraction(prior=prior.Fixed(cf_val))), mode)
    t1 = evaluate_model({}, args1)[0].total[0]
    tcf = evaluate_model({}, args_cf)[0].total[0]
    assert t1.min() < 0.9  # absorption is actually visible
    if mode == 'analytic':
        np.testing.assert_allclose(tcf, 1 - cf_val * (1 - t1), rtol=1e-10)
    else:
        # Convolution mode applies the LSF after the nonlinearity, but the map
        # T -> 1 - cf(1 - T) is affine in the *intrinsic* transmission, so the
        # relation still holds for a single absorber.
        np.testing.assert_allclose(tcf, 1 - cf_val * (1 - t1), rtol=1e-8)


def test_model_cf_one_token_equals_no_token():
    _, a = _build(_abs_lc())
    _, b = _build(_abs_lc(CoveringFraction()))
    np.testing.assert_allclose(
        evaluate_model({}, a)[0].total, evaluate_model({}, b)[0].total, rtol=1e-12
    )


def test_model_cf_zero_is_no_absorption():
    _, args = _build(_abs_lc(CoveringFraction(prior=prior.Fixed(0.0))))
    pred = evaluate_model({}, args)[0]
    np.testing.assert_allclose(pred.total[0], 1.0, rtol=1e-12)
    np.testing.assert_allclose(pred.lines['abs1'], 0.0, atol=1e-12)


def test_model_samples_cf_site_and_matches_evaluate():
    cf = CoveringFraction('abs', prior=prior.Uniform(0.0, 1.0))
    model_fn, args = _build(_abs_lc(cf))
    trace = handlers.trace(handlers.seed(model_fn, rng_seed=3)).get_trace(args)
    assert 'cf_abs' in trace
    assert trace['cf_abs']['type'] == 'sample'
    cf_val = float(trace['cf_abs']['value'])
    assert 0.0 <= cf_val <= 1.0

    loc = None
    for site in trace.values():
        if site['type'] == 'sample' and site.get('is_observed', False):
            loc = np.asarray(site['fn'].loc) * args.norm_factors[0]
    assert loc is not None
    pred = evaluate_model({'cf_abs': jnp.asarray([cf_val])}, args)[0]
    np.testing.assert_allclose(pred.total[0], loc, rtol=1e-4)
