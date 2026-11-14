"""Fixed ``RScale``/``PixOffset`` are evaluated at trace time.

With literal ``Fixed`` priors on both tokens the pixel grid, the LSF and (in
convolution mode) the pixel-averaged convolution kernel are constants.  The
result must equal the general path, where the same values arrive as sampled
parameters.
"""

import astropy.units as u
import jax
import numpy as np
import pytest
from numpyro import handlers

from unite import line, model, prior
from unite.compute import evaluate_model
from unite.continuum import ContinuumConfiguration, ContinuumRegion, Linear
from unite.instrument.base import PixOffset, RScale
from unite.instrument.generic import SimpleDisperser
from unite.model import _static_calibration
from unite.spectrum import Spectra, Spectrum

_FLUX_UNIT = u.Unit('1e-17 erg / (s cm2 AA)')
_R_SCALE = 1.15
_PIX_OFFSET = 0.3


def _build(mode, *, fixed, gap):
    wl = np.linspace(6400, 6700, 150)
    if gap:
        # Drop a block of pixels so the edge topology has an inter-pixel gap.
        wl = np.concatenate([wl[:70], wl[80:]])
    dlam = 300.0 / 149
    wl = wl * u.AA
    if fixed:
        r_tok = RScale('r', prior=prior.Fixed(_R_SCALE))
        p_tok = PixOffset('p', prior=prior.Fixed(_PIX_OFFSET))
    else:
        r_tok = RScale('r', prior=prior.Uniform(0.5, 2.0))
        p_tok = PixOffset('p', prior=prior.Uniform(-1.0, 1.0))
    disp = SimpleDisperser(
        wavelength=wl, R=3000.0, name='d', r_scale=r_tok, pix_offset=p_tok
    )
    rng = np.random.default_rng(3)
    flux = (
        10.0
        + 50 * np.exp(-0.5 * ((wl.value - 6563.0) / 2.0) ** 2)
        + rng.normal(0, 0.5, wl.size)
    ) * _FLUX_UNIT
    half = 0.5 * dlam * u.AA
    spec = Spectrum(
        low=wl - half,
        high=wl + half,
        flux=flux,
        error=np.full(wl.size, 0.5) * _FLUX_UNIT,
        disperser=disp,
        name='s',
    )
    spectra = Spectra([spec], redshift=0.0)
    lc = line.LineConfiguration()
    z = line.Redshift(prior=prior.Uniform(-0.005, 0.005))
    fwhm = line.FWHM(prior=prior.Uniform(100, 1000))
    for name, wav in (('X', 6480.0), ('Ha', 6563.0)):
        lc.add_line(
            name,
            wav * u.AA,
            redshift=z,
            fwhm_gauss=fwhm,
            flux=line.Flux(prior=prior.Uniform(0, 500)),
        )
    lc.add_line(
        'abs',
        6600.0 * u.AA,
        redshift=z,
        fwhm_gauss=line.FWHM(prior=prior.Uniform(100, 800)),
        tau=line.Tau(prior=prior.Uniform(0.0, 2.0)),
    )
    cc = ContinuumConfiguration(
        [ContinuumRegion(6405 * u.AA, 6695 * u.AA, form=Linear())]
    )
    spectra.prepare(lc, cc)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    return model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build(integration_mode=mode)


def _params(model_fn, args):
    with handlers.seed(rng_seed=0):
        tr = handlers.trace(model_fn).get_trace(args)
    return {
        k: v['value']
        for k, v in tr.items()
        if v['type'] == 'sample' and not v.get('is_observed')
    }


def _loglik(model_fn, args, params):
    def fn(p):
        tr = handlers.trace(handlers.substitute(model_fn, data=p)).get_trace(args)
        return tr['obs_s']['fn'].log_prob(tr['obs_s']['value']).sum()

    return jax.jit(jax.value_and_grad(fn))(params)


def test_static_calibration_detection():
    _, fixed_args = _build('analytic', fixed=True, gap=False)
    _, free_args = _build('analytic', fixed=False, gap=False)
    assert _static_calibration(fixed_args.spectra[0].disperser) == (
        _R_SCALE,
        _PIX_OFFSET,
    )
    assert _static_calibration(free_args.spectra[0].disperser) is None


def test_dependent_fixed_is_not_static():
    """A ``Fixed`` that references another parameter is not a constant."""
    wl = np.linspace(6400, 6700, 50) * u.AA
    shared = RScale('a', prior=prior.Uniform(0.5, 2.0))
    disp = SimpleDisperser(
        wavelength=wl,
        R=3000.0,
        name='d',
        r_scale=RScale('b', prior=prior.Fixed(shared)),
    )
    assert _static_calibration(disp) is None


@pytest.mark.parametrize('gap', [False, True])
@pytest.mark.parametrize('mode', ['analytic', 'convolution'])
def test_static_matches_sampled(mode, gap):
    fixed_fn, fixed_args = _build(mode, fixed=True, gap=gap)
    free_fn, free_args = _build(mode, fixed=False, gap=gap)
    params = _params(fixed_fn, fixed_args)
    r_name = free_args.spectra[0].disperser.r_scale.name
    p_name = free_args.spectra[0].disperser.pix_offset.name
    free_params = {**params, r_name: _R_SCALE, p_name: _PIX_OFFSET}
    assert r_name not in params

    # Likelihood and its gradient with respect to the shared parameters.
    ll_fixed, grad_fixed = _loglik(fixed_fn, fixed_args, params)
    ll_free, grad_free = _loglik(free_fn, free_args, free_params)
    np.testing.assert_allclose(ll_fixed, ll_free, rtol=1e-10)
    for name, g in grad_fixed.items():
        np.testing.assert_allclose(g, grad_free[name], rtol=1e-7, atol=1e-7)

    # Predictions, including the per-line and continuum decomposition.
    def as_samples(p):
        return {k: np.full(3, float(v)) for k, v in p.items()}

    fixed_pred = evaluate_model(as_samples(params), fixed_args)[0]
    free_pred = evaluate_model(as_samples(free_params), free_args)[0]
    scale = np.abs(free_pred.total).max()
    kw = {'rtol': 1e-9, 'atol': 1e-12 * scale}
    np.testing.assert_allclose(fixed_pred.total, free_pred.total, **kw)
    assert len(fixed_pred.tau_profiles) == 1
    for group in ('lines', 'continuum_regions', 'tau_profiles'):
        got, want = getattr(fixed_pred, group), getattr(free_pred, group)
        assert got.keys() == want.keys()
        for key in want:
            np.testing.assert_allclose(got[key], want[key], **kw)

    if mode == 'convolution':
        assert (0, 1) in fixed_args._conv_kernels
        assert not free_args._conv_kernels
