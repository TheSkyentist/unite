"""Regression: the likelihood must be continuous in ``PixOffset``.

Continuum-region membership used to be evaluated on the PixOffset-shifted
wavelengths with a hard mask, so edge pixels jumped in and out of their region
as the offset varied and the log-likelihood became a step function.
Membership is now fixed at build time from the unshifted grid
(``ModelArgs.region_masks``).
"""

import astropy.units as u
import jax
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
    Linear,
    Polynomial,
)
from unite.instrument.base import PixOffset
from unite.instrument.generic import SimpleDisperser
from unite.spectrum import Spectra, Spectrum

_FLUX_UNIT = u.Unit('1e-17 erg / (s cm2 AA)')

_WL = np.linspace(6400, 6700, 150)
_DLAM = _WL[1] - _WL[0]


def _lo(k):
    """Lower bound just below pixel k's low edge.

    Pixel k is the first kept pixel; its midpoint sits 0.55 px above the bound,
    so it crosses the bound at pix_offset ~ +0.55.
    """
    return (_WL[k] - 0.55 * _DLAM) * u.AA


def _hi(k):
    """Upper bound just above pixel k's high edge (crossing at ~ -0.55)."""
    return (_WL[k] + 0.55 * _DLAM) * u.AA


_REGIONS = {
    'single': lambda: [ContinuumRegion(_lo(16), _hi(134), form=Linear())],
    # Two regions separated by a gap, different forms.
    'gap': lambda: [
        ContinuumRegion(_lo(16), _hi(61), form=Linear()),
        ContinuumRegion(_lo(69), _hi(134), form=Chebyshev(order=2)),
    ],
    # Two regions sharing a boundary: pixel 61 is the last of the first region
    # and crosses the shared bound at pix_offset ~ -0.55; pixel 62 straddles
    # the bound and is trimmed.
    'touching': lambda: [
        ContinuumRegion(_lo(16), _hi(61), form=Polynomial(degree=2)),
        ContinuumRegion(_hi(61), _hi(134), form=Bernstein(degree=3)),
    ],
}


def _build(integration_mode, regions):
    wl = _WL * u.AA
    tok = PixOffset('p', prior=prior.Uniform(-1.0, 1.0))
    disp = SimpleDisperser(wavelength=wl, R=3000.0, name='d', pix_offset=tok)
    grad = np.gradient(wl)
    sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
    rng = np.random.default_rng(3)
    flux = (
        10.0
        + 50 * np.exp(-0.5 * ((wl.value - 6563.0) / sigma) ** 2)
        + 20 * np.exp(-0.5 * ((wl.value - 6480.0) / sigma) ** 2)
        + rng.normal(0, 0.5, wl.size)
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
    z = line.Redshift(prior=prior.Uniform(-0.005, 0.005))
    fwhm = line.FWHM(prior=prior.Uniform(100, 1000))
    # One line in each region of every configuration so none is dropped.
    for name, wav in (('X', 6480.0), ('Ha', 6563.0)):
        lc.add_line(
            name,
            wav * u.AA,
            redshift=z,
            fwhm_gauss=fwhm,
            flux=line.Flux(prior=prior.Uniform(0, 500)),
        )
    cc = ContinuumConfiguration(_REGIONS[regions]())
    spectra.prepare(lc, cc)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    model_fn, args = model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    ).build(integration_mode=integration_mode)
    assert len(args.region_masks[0]) == len(_REGIONS[regions]())
    return model_fn, args


def _base_params(model_fn, args):
    with handlers.seed(rng_seed=0):
        tr = handlers.trace(model_fn).get_trace(args)
    base = {
        k: v['value']
        for k, v in tr.items()
        if v['type'] == 'sample' and not v.get('is_observed')
    }
    pix_name = next(k for k in base if 'pix_offset' in k)
    return base, pix_name


@pytest.mark.parametrize('regions', list(_REGIONS))
@pytest.mark.parametrize('mode', ['analytic', 'convolution'])
def test_loglik_continuous_in_pix_offset(mode, regions):
    model_fn, args = _build(mode, regions)
    base, pix_name = _base_params(model_fn, args)

    def loglik(x):
        p = {**base, pix_name: x}
        t = handlers.trace(handlers.substitute(model_fn, data=p)).get_trace(args)
        site = t['obs_s']
        return site['fn'].log_prob(site['value']).sum()

    # Every region bound is crossed by an edge-pixel midpoint at |x| ~ 0.55, and
    # the fine grid (n_super=10) crosses them every 0.1 px.
    xs = np.linspace(-0.9, 0.9, 901)
    ll = np.asarray(jax.jit(jax.vmap(loglik))(xs))
    assert np.all(np.isfinite(ll))
    # A jump shows up as a spike in the second difference; a smooth loglik has
    # a nearly constant one.  (First differences are not used: prior draws can
    # put the model far from the data, and the steep slope then hides jumps.)
    curv = np.abs(np.diff(ll, 2))
    assert curv.max() < 20 * np.median(curv), (
        f'{mode}/{regions}: loglik second difference spikes to '
        f'{curv.max():.3g} (median {np.median(curv):.3g}) near '
        f'pix_offset={xs[np.argmax(curv) + 1]:.4f}'
    )


@pytest.mark.parametrize('mode', ['analytic', 'convolution'])
def test_evaluate_model_matches_model_with_pix_offset(mode):
    """evaluate_model uses the same static region membership as the model."""
    model_fn, args = _build(mode, 'gap')
    base, pix_name = _base_params(model_fn, args)
    spectrum = args.spectra[0]
    expected = args.region_masks[0][:, spectrum.keep_mask]  # (n_regions, npix)
    # Each kept pixel lies in at least one region (spectra are trimmed to them).
    assert expected.any(axis=0).all()

    # Offsets large enough to push edge midpoints across the region bounds.
    for x in (-0.8, 0.0, 0.8):
        p = {**base, pix_name: x}
        (pred,) = evaluate_model({k: np.atleast_1d(v) for k, v in p.items()}, args)

        # Total matches the likelihood mean.  (Analytic evaluate_model samples
        # the continuum at pixel midpoints rather than averaging it over the
        # pixel, hence the loose tolerance.)
        t = handlers.trace(handlers.substitute(model_fn, data=p)).get_trace(args)
        loc = np.asarray(t['obs_s']['fn'].loc) * args.norm_factors[0]
        np.testing.assert_allclose(
            pred.total[0], loc, rtol=0, atol=1e-6 * np.abs(loc).max()
        )

        # Per-region membership is the static one.  Only checkable in analytic
        # mode: in convolution mode the LSF spreads each region past its pixels.
        regions = list(pred.continuum_regions.values())
        assert len(regions) == expected.shape[0]
        if mode == 'analytic':
            for k, region in enumerate(regions):
                np.testing.assert_array_equal(region[0] != 0, expected[k])
