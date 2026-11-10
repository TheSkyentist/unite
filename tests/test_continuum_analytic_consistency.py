"""Regression: analytic-mode continuum is one computation shared by fit and results.

Polynomial-family continua (``Polynomial``, ``Chebyshev``, ``Legendre``,
``Bernstein``) used to be pixel-averaged with one LSF per region in the
likelihood but evaluated at pixel centres with a per-pixel LSF in
``evaluate_model``, so the predictions were not the fitted model (GH #27).
Both now call ``integrate_continuum_regions`` with the LSF of each pixel.
"""

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro import handlers

from unite import line, model, prior
from unite.compute import evaluate_model
from unite.continuum import (
    Bernstein,
    Chebyshev,
    ContinuumConfiguration,
    ContinuumRegion,
    Legendre,
    Polynomial,
)
from unite.continuum.library import (
    ContinuumForm,
    _convolved_polyint,
    _gaussian_convolve_poly,
    _polyint_at,
)
from unite.instrument.generic import GenericDisperser
from unite.spectrum import Spectra, Spectrum

_FLUX_UNIT = u.Unit('1e-17 erg / (s cm2 AA)')
_WL = np.linspace(6300, 6800, 250)

# (form factory, site-name prefix -> value) giving strong curvature.  The
# polynomial coefficients are per Angstrom^i, the others are relative.
_FORMS = {
    'polynomial': (
        lambda: Polynomial(degree=4),
        {'c1': 2e-3, 'c2': 2e-3, 'c3': 0.0, 'c4': 1e-8},
    ),
    'chebyshev': (
        lambda: Chebyshev(order=4),
        {'c1': 0.3, 'c2': -0.4, 'c3': 0.2, 'c4': 0.3},
    ),
    'legendre': (
        lambda: Legendre(order=4),
        {'p1': 0.3, 'p2': -0.4, 'p3': 0.2, 'p4': 0.3},
    ),
    'bernstein': (
        lambda: Bernstein(degree=4),
        {'coeff_1': 1.2, 'coeff_2': 0.7, 'coeff_3': 1.5, 'coeff_4': 0.9},
    ),
}


def _build(form_name, mode, r_blue=500.0, r_red=1700.0):
    factory, _ = _FORMS[form_name]
    wl = _WL * u.AA

    def r_func(w):
        return r_blue + (r_red - r_blue) * (w - 6300.0) / 500.0

    disp = GenericDisperser(
        R_func=r_func,
        dlam_dpix_func=lambda w: jnp.full_like(w, _WL[1] - _WL[0]),
        unit=u.AA,
        name='d',
    )
    grad = np.gradient(wl)
    sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
    flux = (
        10.0
        + 50 * np.exp(-0.5 * ((wl.value - 6563.0) / sigma) ** 2)
        + np.random.default_rng(1).normal(0, 0.1, wl.size)
    ) * _FLUX_UNIT
    spec = Spectrum(
        low=wl - 0.5 * grad,
        high=wl + 0.5 * grad,
        flux=flux,
        error=np.full(wl.size, 0.5) * _FLUX_UNIT,
        disperser=disp,
        name='s',
    )
    spectra = Spectra([spec], redshift=0.0)
    lc = line.LineConfiguration()
    lc.add_line(
        'Ha',
        6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Fixed(0.0)),
        fwhm_gauss=line.FWHM(prior=prior.Fixed(300.0)),
        flux=line.Flux(prior=prior.Fixed(0.0)),  # isolate the continuum
    )
    cc = ContinuumConfiguration(
        [ContinuumRegion(6320 * u.AA, 6780 * u.AA, form=factory())]
    )
    spectra.prepare(lc, cc)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    kwargs = {'n_super': 30} if mode == 'convolution' else {}
    return model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build(integration_mode=mode, **kwargs)


def _params(model_fn, args, form_name):
    """Prior draw with the continuum shape coefficients set to strong curvature."""
    with handlers.seed(rng_seed=0):
        tr = handlers.trace(model_fn).get_trace(args)
    p = {
        k: v['value']
        for k, v in tr.items()
        if v['type'] == 'sample' and not v.get('is_observed')
    }
    overrides = _FORMS[form_name][1]
    for key in p:
        prefix = key.rsplit('_', 1)[0]
        if prefix in overrides:
            p[key] = jnp.asarray(overrides[prefix])
        elif prefix == 'scale':
            p[key] = jnp.asarray(1.0)
    return p


def _likelihood_mean(model_fn, args, p):
    t = handlers.trace(handlers.substitute(model_fn, data=p)).get_trace(args)
    return np.asarray(t['obs_s']['fn'].loc)


@pytest.mark.parametrize('form_name', list(_FORMS))
def test_evaluate_model_equals_likelihood_mean(form_name):
    """evaluate_model's analytic total is the fitted mean, with a varying LSF."""
    model_fn, args = _build(form_name, 'analytic')
    p = _params(model_fn, args, form_name)
    loc = _likelihood_mean(model_fn, args, p) * args.norm_factors[0]
    (pred,) = evaluate_model({k: np.atleast_1d(v) for k, v in p.items()}, args)
    np.testing.assert_allclose(pred.total[0], loc, rtol=1e-9, atol=0)
    # The continuum decomposition sums to the total (the line flux is zero).
    cont = sum(np.asarray(r)[0] for r in pred.continuum_regions.values())
    np.testing.assert_allclose(cont, pred.total[0], rtol=1e-9, atol=0)


@pytest.mark.parametrize('form_name', list(_FORMS))
def test_analytic_matches_convolution_mode(form_name):
    """The per-pixel LSF reproduces the numerical variable-LSF convolution.

    Compared in the interior: convolution mode edge-extends the model at the
    spectral boundaries.  With a region-scalar LSF the analytic result was
    ~5e-4 of the continuum range away; the per-pixel LSF is closer than 5e-5.
    """
    fn_a, args_a = _build(form_name, 'analytic')
    fn_c, args_c = _build(form_name, 'convolution')
    p = _params(fn_a, args_a, form_name)
    loc_a = _likelihood_mean(fn_a, args_a, p)
    loc_c = _likelihood_mean(fn_c, args_c, p)
    interior = slice(40, -40)
    rng = np.ptp(loc_c[interior])
    assert rng > 0
    np.testing.assert_allclose(
        loc_a[interior], loc_c[interior], rtol=0, atol=5e-5 * rng
    )


@pytest.mark.parametrize('degree', [2, 4])
def test_constant_lsf_array_matches_scalar(degree):
    """A per-interval LSF that is constant gives the scalar-LSF pixel integrals."""
    rng = np.random.default_rng(0)
    coeffs = jnp.asarray(rng.normal(size=degree + 1))
    x = jnp.linspace(-30.0, 30.0, 61)
    per_interval = _convolved_polyint(coeffs, jnp.full(60, 7.0), x)
    scalar = _polyint_at(_gaussian_convolve_poly(coeffs, 7.0), x)
    np.testing.assert_allclose(
        np.diff(per_interval), np.diff(scalar), rtol=1e-9, atol=1e-8
    )


def test_interval_lsf_support_is_polynomial_family():
    """Only forms that convolve exactly with the LSF take a per-pixel LSF."""
    supported = {
        cls.__name__
        for cls in ContinuumForm.__subclasses__()
        if getattr(cls, 'supports_interval_lsf', False)
    }
    assert supported == {'Polynomial', 'Chebyshev', 'Legendre', 'Bernstein'}
