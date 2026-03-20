# unite — Unified liNe Integration Turbo Engine

[![PyPI](https://img.shields.io/pypi/v/unite)](https://pypi.org/project/unite/)
[![Python](https://img.shields.io/pypi/pyversions/unite)](https://pypi.org/project/unite/)
[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)
[![Tests](https://github.com/TheSkyentist/unite/actions/workflows/ci.yml/badge.svg)](https://github.com/TheSkyentist/unite/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/TheSkyentist/unite/branch/main/graph/badge.svg)](https://codecov.io/gh/TheSkyentist/unite)
[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://unite.readthedocs.io/)

`unite` is a Python package for fast, Bayesian inference of emission lines from astronomical spectra. It is built on [JAX](https://jax.readthedocs.io/), [NumPyro](https://num.pyro.ai/), and [Astropy](https://www.astropy.org/), and supports fitting multiple spectra simultaneously with shared kinematics, calibration tokens, and flexible priors.

Originally designed for JWST/NIRSpec but extensible to any spectrograph.

## What it does

- **Exact pixel integration** of line profiles — fast, memory-efficient, and correct even for undersampled data
- **Simultaneous multi-spectrum fitting** across gratings and instruments with shared kinematic parameters (redshift, FWHM)
- **Multiple line profiles**: Gaussian, Voigt, Cauchy, Pseudo-Voigt, Laplace, Gauss-Hermite, Split-Normal
- **Flexible continuum models**: Linear, Power-Law, Polynomial — auto-generated from line configurations
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
pixi add unite
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

cc = ContinuumConfiguration.from_lines(lc.centers, pad=0.05, form=Linear())

# 2. Load spectra (NIRSpec example; any instrument works)
g395m = nirspec.G395M()
spec = from_DJA('dja-spectrum.fits', disperser=g395m)

spectra = Spectra([spec], redshift=5.28)
filtered_lines, filtered_cont = spectra.prepare(lc, cc)
spectra.compute_scales(filtered_lines, filtered_cont, error_scale=True)

# 3. Build and run with any NumPyro sampler
builder = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
model_fn, model_args = builder.build()

mcmc = infer.MCMC(infer.NUTS(model_fn), num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), model_args)

# 4. Extract results
param_table = make_parameter_table(mcmc.get_samples(), model_args)
spectra_tables = make_spectra_tables(mcmc.get_samples(), model_args)
```

## Contributing

Bug reports, feature requests, and pull requests are welcome on [GitHub](https://github.com/TheSkyentist/unite/issues). If you find a bug or have an idea for an improvement, please open an issue — even a brief description is helpful.

## Documentation

Full documentation, tutorials, and API reference at **[unite.readthedocs.io](https://unite.readthedocs.io/)**.

## Citing

If you use `unite` in your research, please cite the Zenodo software release. Each versioned release has a unique DOI minted automatically when a GitHub release is created.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15585034.svg)](https://doi.org/10.5281/zenodo.15585034)

See [CITATION.md](CITATION.md) for BibTeX and details. The Zenodo record lists all releases — visit the link to cite a specific version. GitHub's "Cite this repository" button (top-right of the repo page) also generates citation text directly from [`CITATION.cff`](CITATION.cff).

## License

GPL v3 or later. See [LICENSE](LICENSE) for details.
