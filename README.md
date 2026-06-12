# unite — Unified liNe Integration Turbo Engine

[![PyPI](https://img.shields.io/pypi/v/unite)](https://pypi.org/project/unite/)
[![Python](https://img.shields.io/pypi/pyversions/unite)](https://pypi.org/project/unite/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![Tests](https://github.com/TheSkyentist/unite/actions/workflows/ci.yml/badge.svg)](https://github.com/TheSkyentist/unite/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TheSkyentist/unite/branch/main/graph/badge.svg)](https://codecov.io/gh/TheSkyentist/unite)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://unite.readthedocs.io/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.15585034-blue)](https://doi.org/10.5281/zenodo.15585034)

`unite` is a Python package for fast, Bayesian inference of emission lines from astronomical spectra. It is built on [JAX](https://jax.readthedocs.io/), [NumPyro](https://num.pyro.ai/), and [Astropy](https://www.astropy.org/), and supports fitting multiple spectra simultaneously with shared kinematics, calibration tokens, and flexible priors.

Originally designed for JWST/NIRSpec but extensible to any spectrograph.

## What it does

- **Two pixel-integration modes**: analytic (exact CDF-based, default) and numerical LSF convolution (`n_super` uniform fine-grid points per pixel + banded wavelength-varying Gaussian convolution, correctly computes `LSF ⊗ [F · exp(-τ · φ_intrinsic)]` for absorption lines)
- **Simultaneous multi-spectrum fitting** across gratings and instruments with shared kinematic parameters (redshift, FWHM)
- **Multiple line profiles**: Gaussian, Cauchy, Pseudo-Voigt, Laplace, SEMG, Gauss-Hermite, Split-Normal, Skew-Normal, Skew-Voigt, Box-Gauss, Gaussian-Split-Laplace (asymmetric EMG)
- **Emission and absorption lines**: flux-parametrized additive profiles and tau-parametrized multiplicative transmission `exp(-tau * phi)`, with per-component depth ordering (`zorder`) so each absorber selectively attenuates only the sources behind it
- **Flexible continuum models**: Linear, Polynomial, Chebyshev, Legendre, Bernstein, B-Spline, Power-Law, Blackbody, Modified Blackbody, Attenuated Blackbody, Template (user-supplied file) — auto-generated from line configurations
- **Calibration tokens** (flux scale, resolution scale, pixel offset) with free or fixed priors, shared across spectra
- **YAML serialization** for reproducible, human-editable configurations
- **User-controlled sampler** — `ModelBuilder` returns `(model_fn, model_args)` for use with any NumPyro backend (NUTS, SVI, nested sampling, ...)
- **Instrument support** for JWST/NIRSpec (all gratings + PRISM), SDSS, and any custom spectrograph via generic dispersers

## Installation

```bash
pip install unite
```

Or with [Pixi](https://pixi.sh/):

```bash
pixi add unite --pypi
```

## Quick Start

```python
import jax
import astropy.units as u
from numpyro import infer

from unite import line, model, prior
from unite.continuum import ContinuumConfiguration, Linear
from unite.instrument import nirspec
from unite.results import make_parameter_table, make_spectra_tables
from unite.spectrum import Spectra, from_DJA

# 1. Configure lines with shared kinematics
z    = line.Redshift('z', prior=prior.Uniform(-0.005, 0.005))
fwhm = line.FWHM('narrow', prior=prior.Uniform(100, 1000))

lc = line.LineConfiguration()
lc.add_line(
    'H_alpha',
    6563.0 * u.AA,
    redshift=z,
    fwhm_gauss=fwhm,
    flux=line.Flux(prior=prior.Uniform(0, 10))
)
lc.add_line(
    'NII_6585',
    6585.0 * u.AA,
    redshift=z,
    fwhm_gauss=fwhm,
    flux=line.Flux(prior=prior.Uniform(0, 10))
)
# Tau-parametrized absorption line: transmission = exp(-tau * phi)
lc.add_line(
    'HI_abs',
    6563.0 * u.AA,
    redshift=z,
    fwhm_gauss=line.FWHM('abs', prior=prior.Uniform(50, 500)),
    tau=line.Tau(prior=prior.Uniform(0, 5))
)

cc = ContinuumConfiguration.from_lines(lc.centers, width=15_000*u.km/u.s, form=Linear())

# 2. Load spectra (NIRSpec example; any instrument works)
g395m = nirspec.G395M()
spec = from_DJA('dja-spectrum.fits', disperser=g395m)

spectra = Spectra([spec], redshift=5.28)
filtered_lines, filtered_cont = spectra.prepare(lc, cc)
spectra.compute_scales(filtered_lines, filtered_cont, error_scale=True)

# 3. Build and run with any NumPyro sampler
# integration_mode='analytic' (default) uses exact CDF integration;
# integration_mode='convolution' convolves intrinsic model with LSF on a fine grid
#   (n_super uniform points per pixel) — most accurate for absorption lines
builder = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
model_fn, model_args = builder.build(integration_mode='analytic')

mcmc = infer.MCMC(infer.NUTS(model_fn), num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), model_args)

# 4. Extract results
# Get summary statistics at specific percentiles
samples = mcmc.get_samples()
param_table = make_parameter_table(samples, model_args, percentiles=[0.16, 0.5, 0.84])
spectra_tables = make_spectra_tables(samples, model_args, percentiles=[0.16, 0.5, 0.84])
```

## Contributing

Bug reports, feature requests, and pull requests are welcome on [GitHub](https://github.com/TheSkyentist/unite/issues). If you find a bug or have an idea for an improvement, please open an issue — even a brief description is helpful.

## Documentation

Full documentation, tutorials, and API reference at **[unite.readthedocs.io](https://unite.readthedocs.io/)**.

## Citing

If you use `unite` in your research, please cite the appropriate software version on [Zenodo](https://doi.org/10.5281/zenodo.15585034). If you use the built in NIRSpec LSF data, please also cite the appropriate LSF source (de Graaff et al. 2024 for `point`, Jakobsen et al. 2022 for `uniform`).

See [CITATION.md](CITATION.md) for BibTeX entries and full details.

## License

GPL v3 or later. See [LICENSE](LICENSE) for details.
