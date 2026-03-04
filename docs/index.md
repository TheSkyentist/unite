# unite

**Unified Numerical Integration Tool for spEctroscopy**

`unite` is a Python package for Bayesian inference of emission lines from astronomical spectra.
It is built on [JAX](https://jax.readthedocs.io/), [NumPyro](https://num.pyro.ai/), and
[Astropy](https://www.astropy.org/), and supports fitting multiple spectra simultaneously
with shared kinematics, calibration tokens, and flexible priors.

**Key features:**

- Fast, exact pixel integration of line profiles (Gaussian, Voigt, Hermite, and more)
- Simultaneous fitting across multiple spectra and gratings
- Shared kinematic parameters (redshift, FWHM) across lines and components
- Calibration tokens (flux scale, resolution scale, pixel offset) with free or fixed priors
- YAML serialization for reproducible, human-editable configurations
- Instrument support for JWST/NIRSpec (all gratings + PRISM) and generic spectrographs
- User-controlled sampler — `ModelBuilder` returns `(model_fn, model_args)` for use with
  any NumPyro backend (NUTS, SVI, nested sampling, …)

---

## Getting Started

```{toctree}
:maxdepth: 1
:caption: Getting Started

getting_started
concepts
```

## Tutorials

```{toctree}
:maxdepth: 1
:caption: Tutorials

tutorials/index
tutorials/generic
tutorials/nirspec
```

## How-To Guides

```{toctree}
:maxdepth: 1
:caption: How-To Guides

guides/index
guides/line_profiles
guides/serialization
guides/results
guides/advanced
```

## API Reference

```{toctree}
:maxdepth: 1
:caption: API Reference

api/index
```

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
