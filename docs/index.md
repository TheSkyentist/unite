# unite

***Unified liNe Integration Turbo Engine**

`unite` is a Python package for Bayesian inference of emission lines from astronomical spectra.
It is built on [JAX](https://jax.readthedocs.io/), [NumPyro](https://num.pyro.ai/), and
[Astropy](https://www.astropy.org/), and supports fitting multiple spectra simultaneously
with shared kinematics, calibration tokens, and flexible priors.

**Key features:**

- Fast, exact pixel integration of line profiles (Gaussian, Voigt, Hermite, and more)
- Simultaneous fitting across multiple spectra and gratings
- Shared kinematic parameters (redshift, FWHM) across lines and components
- Dependent priors with arbitrary-depth parameter chains
- Calibration tokens (flux scale, resolution scale, pixel offset) with free or fixed priors
- YAML serialization for reproducible, human-editable configurations
- Instrument support for JWST/NIRSpec (all gratings + PRISM), SDSS, and generic spectrographs
- User-controlled sampler — `ModelBuilder` returns `(model_fn, model_args)` for use with
  any NumPyro backend (NUTS, SVI, nested sampling, ...)

---

```{toctree}
:maxdepth: 1
:caption: Getting Started

installation
concepts
contributing
```

```{toctree}
:maxdepth: 1
:caption: Tutorials

auto_tutorials/tutorial_nirspec
auto_tutorials/tutorial_generic
auto_tutorials/tutorial_config
sg_execution_times
```

```{toctree}
:maxdepth: 3
:caption: Usage

usage/priors
usage/line_configuration
usage/continuum_configuration
usage/instrument
usage/serialization
usage/build_model
usage/inference
usage/results
```

```{toctree}
:maxdepth: 1
:caption: API Reference

api/index
```

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
<!--- {ref}`search`-->
