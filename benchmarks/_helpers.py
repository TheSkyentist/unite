"""Shared helpers for the unite benchmark suite.

Lives in a regular module (not conftest) so it can be imported from both
``conftest.py`` and individual ``bench_*.py`` files via
``from benchmarks._helpers import ...``.
"""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from jax import random
from numpyro.infer import Predictive

from unite import line, model, prior
from unite.continuum import ContinuumConfiguration, ContinuumRegion, Linear
from unite.instrument.generic import SimpleDisperser
from unite.line.library import (
    SEMG,
    BoxGauss,
    Cauchy,
    GaussHermite,
    Gaussian,
    GaussianSplitLaplace,
    Laplace,
    Profile,
    PseudoVoigt,
    SkewNormal,
    SkewVoigt,
    SplitNormal,
)
from unite.spectrum import Spectra, Spectrum


@dataclass
class Bench:
    """Container for a benchmark-ready model.

    Attributes
    ----------
    model_fn
        The numpyro model function returned by :meth:`ModelBuilder.build`.
    args
        ``ModelArgs`` bundle passed to ``model_fn``.
    sample
        One prior draw of all sampled sites, ready to pass into a
        ``substitute``-wrapped model or a log-density call.
    """

    model_fn: object
    args: object
    sample: dict


# ----------------------------------------------------------------------
# Block helper
# ----------------------------------------------------------------------


def block(x):
    """Force JAX async dispatch to complete.

    ``benchmark()`` measures the return-from-call wall time; without this
    JAX returns immediately with an unevaluated future and the benchmark
    measures ~nothing.  Call ``block(result)`` at the end of every benched
    fn that returns a JAX array.
    """
    if isinstance(x, jnp.ndarray):
        return x.block_until_ready()
    if isinstance(x, (tuple, list)):
        for item in x:
            block(item)
        return x
    if isinstance(x, dict):
        for item in x.values():
            block(item)
        return x
    return x


# ----------------------------------------------------------------------
# Synthetic spectrum factory
# ----------------------------------------------------------------------


_FLUX_UNIT = u.Unit('1e-17 erg / (s cm2 AA)')


def make_spectrum(
    wl_min: float,
    wl_max: float,
    n_pixels: int,
    name: str,
    R: float = 3000.0,
    seed: int = 42,
) -> Spectrum:
    """Build a synthetic ``Spectrum`` on a uniform wavelength grid (Å)."""
    wavelength = np.linspace(wl_min, wl_max, n_pixels) * u.AA
    disperser = SimpleDisperser(wavelength=wavelength, R=R, name=f'{name}_disp')
    dx = np.gradient(wavelength)
    low = wavelength - 0.5 * dx
    high = wavelength + 0.5 * dx

    rng = np.random.default_rng(seed)
    # A flat continuum + a couple of Gaussians + noise — content doesn't
    # matter for benchmarking shape/compile time, only that the spectrum
    # has structure compute_scales() can latch onto.
    flux = np.full(n_pixels, 5.0)
    for centre, amp in [(6563.0, 80.0), (6585.0, 25.0), (6548.0, 20.0)]:
        flux = flux + amp * np.exp(-0.5 * ((wavelength.value - centre) / 1.5) ** 2)
    flux = flux + rng.normal(0.0, 1.5, n_pixels)
    error = np.full(n_pixels, 1.5)

    return Spectrum(
        low=low,
        high=high,
        flux=flux * _FLUX_UNIT,
        error=error * _FLUX_UNIT,
        disperser=disperser,
        name=name,
    )


# ----------------------------------------------------------------------
# Build + sample helpers
# ----------------------------------------------------------------------


def build_model(line_cfg, cont_cfg, spectra: Spectra, *, integration_mode='analytic'):
    """Prepare spectra + build the model.  Returns (model_fn, args).

    ``integration_mode`` is forwarded to :meth:`ModelBuilder.build`; valid
    values are ``'analytic'`` (default) and ``'convolution'``.
    """
    spectra.prepare(line_cfg, cont_cfg)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    builder = model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    )
    return builder.build(integration_mode=integration_mode)


def one_prior_draw(model_fn, args, seed: int = 0) -> dict:
    """Pull a single sample from the prior so benchmarks have valid inputs."""
    predictive = Predictive(model_fn, num_samples=1)
    samples = predictive(random.PRNGKey(seed), args)
    # Drop the observation site; we only want the sampled latent params.
    return {k: v[0] for k, v in samples.items() if not k.startswith('obs_')}


# ----------------------------------------------------------------------
# Canonical configurations
# ----------------------------------------------------------------------


def cfg_minimal():
    """1 line, no continuum — the smallest forward pass."""
    lc = line.LineConfiguration()
    lc.add_line(
        'H_alpha',
        center=6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Uniform(-0.005, 0.005)),
        fwhm_gauss=line.FWHM(prior=prior.Uniform(100.0, 1000.0)),
        flux=line.Flux(prior=prior.Uniform(50.0, 150.0)),
    )
    return lc, None


def cfg_single_grating():
    """5 lines (4 narrow + 1 broad) + linear continuum."""
    lc = line.LineConfiguration()
    z_narrow = line.Redshift(prior=prior.Uniform(-0.005, 0.005))
    fwhm_narrow = line.FWHM(prior=prior.Uniform(100.0, 500.0))
    for name, centre in [
        ('H_alpha_narrow', 6563.0),
        ('NII_6548', 6548.0),
        ('NII_6584', 6584.0),
        ('SII_6717', 6717.0),
    ]:
        lc.add_line(
            name,
            center=centre * u.AA,
            redshift=z_narrow,
            fwhm_gauss=fwhm_narrow,
            flux=line.Flux(prior=prior.Uniform(0.0, 200.0)),
        )
    lc.add_line(
        'H_alpha_broad',
        center=6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Uniform(-0.01, 0.01)),
        fwhm_gauss=line.FWHM(prior=prior.Uniform(1000.0, 5000.0)),
        flux=line.Flux(prior=prior.Uniform(0.0, 300.0)),
    )
    cc = ContinuumConfiguration(
        [ContinuumRegion(6400.0 * u.AA, 6800.0 * u.AA, Linear())]
    )
    return lc, cc


def cfg_single_grating_with_absorber():
    """5 emission lines + 1 absorber + linear continuum on a single spectrum.

    Same emission lines as :func:`cfg_single_grating`, plus a tau-parametrized
    NaI-D-like absorber.  Use this to isolate the cost of the absorption
    pipeline against the pure-emission baseline.
    """
    lc = line.LineConfiguration()
    z_narrow = line.Redshift(prior=prior.Uniform(-0.005, 0.005))
    fwhm_narrow = line.FWHM(prior=prior.Uniform(100.0, 500.0))
    for name, centre in [
        ('H_alpha_narrow', 6563.0),
        ('NII_6548', 6548.0),
        ('NII_6584', 6584.0),
        ('SII_6717', 6717.0),
    ]:
        lc.add_line(
            name,
            center=centre * u.AA,
            redshift=z_narrow,
            fwhm_gauss=fwhm_narrow,
            flux=line.Flux(prior=prior.Uniform(0.0, 200.0)),
        )
    lc.add_line(
        'H_alpha_broad',
        center=6563.0 * u.AA,
        redshift=line.Redshift(prior=prior.Uniform(-0.01, 0.01)),
        fwhm_gauss=line.FWHM(prior=prior.Uniform(1000.0, 5000.0)),
        flux=line.Flux(prior=prior.Uniform(0.0, 300.0)),
    )
    lc.add_line(
        'NaI_D_abs',
        center=6500.0 * u.AA,
        redshift=z_narrow,
        fwhm_gauss=line.FWHM(prior=prior.Uniform(100.0, 800.0)),
        tau=line.Tau(prior=prior.Uniform(0.0, 2.0)),
        zorder=1,
    )
    cc = ContinuumConfiguration(
        [ContinuumRegion(6400.0 * u.AA, 6800.0 * u.AA, Linear())]
    )
    return lc, cc


def cfg_single_grating_voigt():
    """5 PseudoVoigt lines (4 narrow + 1 broad) + linear continuum.

    Mirrors :func:`cfg_single_grating` but uses PseudoVoigt profiles so that
    ``evaluate_voigt`` (the Faddeeva / wofz path) is exercised in convolution
    mode.  Shared ``fwhm_lorentz`` tokens across the narrow group and a
    separate one for the broad component.
    """
    lc = line.LineConfiguration()
    z_narrow = line.Redshift(prior=prior.Uniform(-0.005, 0.005))
    fwhm_g_narrow = line.FWHM(prior=prior.Uniform(100.0, 500.0))
    fwhm_l_narrow = line.FWHM(prior=prior.Uniform(50.0, 300.0))
    for name, centre in [
        ('H_alpha_narrow', 6563.0),
        ('NII_6548', 6548.0),
        ('NII_6584', 6584.0),
        ('SII_6717', 6717.0),
    ]:
        lc.add_line(
            name,
            center=centre * u.AA,
            profile='pseudovoigt',
            redshift=z_narrow,
            fwhm_gauss=fwhm_g_narrow,
            fwhm_lorentz=fwhm_l_narrow,
            flux=line.Flux(prior=prior.Uniform(0.0, 200.0)),
        )
    lc.add_line(
        'H_alpha_broad',
        center=6563.0 * u.AA,
        profile='pseudovoigt',
        redshift=line.Redshift(prior=prior.Uniform(-0.01, 0.01)),
        fwhm_gauss=line.FWHM(prior=prior.Uniform(1000.0, 5000.0)),
        fwhm_lorentz=line.FWHM(prior=prior.Uniform(200.0, 1000.0)),
        flux=line.Flux(prior=prior.Uniform(0.0, 300.0)),
    )
    cc = ContinuumConfiguration(
        [ContinuumRegion(6400.0 * u.AA, 6800.0 * u.AA, Linear())]
    )
    return lc, cc


def cfg_multi_grating():
    """4 narrow + 1 absorption, intended for use with two spectra."""
    lc = line.LineConfiguration()
    z_narrow = line.Redshift(prior=prior.Uniform(-0.005, 0.005))
    fwhm_narrow = line.FWHM(prior=prior.Uniform(100.0, 500.0))
    for name, centre in [
        ('H_alpha', 6563.0),
        ('NII_6548', 6548.0),
        ('NII_6584', 6584.0),
        ('SII_6717', 6717.0),
    ]:
        lc.add_line(
            name,
            center=centre * u.AA,
            redshift=z_narrow,
            fwhm_gauss=fwhm_narrow,
            flux=line.Flux(prior=prior.Uniform(0.0, 200.0)),
        )
    lc.add_line(
        'NaI_D_abs',
        center=5895.0 * u.AA,
        redshift=z_narrow,
        fwhm_gauss=line.FWHM(prior=prior.Uniform(100.0, 800.0)),
        tau=line.Tau(prior=prior.Uniform(0.0, 2.0)),
        zorder=1,
    )
    cc = ContinuumConfiguration(
        [
            ContinuumRegion(5800.0 * u.AA, 6000.0 * u.AA, Linear()),
            ContinuumRegion(6400.0 * u.AA, 6800.0 * u.AA, Linear()),
        ]
    )
    return lc, cc


# ----------------------------------------------------------------------
# Profile registry — used by parametrized per-profile microbenchmarks
# ----------------------------------------------------------------------


PROFILE_CLASSES: tuple[type[Profile], ...] = (
    Gaussian,
    Cauchy,
    PseudoVoigt,
    Laplace,
    SEMG,
    GaussHermite,
    SplitNormal,
    GaussianSplitLaplace,
    SkewNormal,
    BoxGauss,
    SkewVoigt,
)


def default_param_for(name: str) -> float:
    """Plausible benchmark value for a given profile parameter name.

    Microbenchmarks don't care about physical realism, only that the kernel
    receives values in its valid domain.  Returns:

    - ``3.0`` for any FWHM-like parameter (matched by ``'fwhm'`` substring),
    - ``0.05`` for Gauss-Hermite shape coefficients (``h3``, ``h4``),
    - ``0.5``  for asymmetry/skew parameters (``alpha``).
    """
    if 'fwhm' in name:
        return 3.0
    if name in ('h3', 'h4'):
        return 0.05
    if name == 'alpha':
        return 0.5
    raise KeyError(f'No default benchmark value for profile parameter {name!r}')
