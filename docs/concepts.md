# Core Concepts

This page explains the key ideas behind `unite`'s design. Understanding these concepts will
make the tutorials much easier to follow.

---

## Tokens and Parameter Sharing

The central design pattern in `unite` is the **token**. A token is a named Python object
that represents a model parameter. Sharing a token between two lines means those lines share
the *same* parameter in the model.

```python
from unite import line, prior

# One Redshift token → one model parameter
z = line.Redshift('nlr_z', prior=prior.Uniform(-0.005, 0.005))

lc = line.LineConfiguration()
lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z, ...)   # ← same z
lc.add_line('NII_6585', 6585.0 * u.AA, redshift=z, ...)  # ← same z
```

When `ModelBuilder` sees the same token object (by Python identity) attached to multiple
lines, it generates a single shared latent variable in the NumPyro model.

**Separate tokens → separate parameters:**

```python
z_narrow = line.Redshift('narrow_z', prior=prior.Uniform(-0.01, 0.01))
z_broad  = line.Redshift('broad_z',  prior=prior.Uniform(-0.01, 0.01))

lc.add_line('H_alpha_narrow', 6563.0 * u.AA, redshift=z_narrow, ...)
lc.add_line('H_alpha_broad',  6563.0 * u.AA, redshift=z_broad,  ...)
```

### Available Tokens

| Token | Role | Unit |
|-------|------|------|
| {class}`~unite.line.Redshift` | Kinematic redshift | dimensionless |
| {class}`~unite.line.FWHM` | Line width (Gaussian FWHM) | km/s |
| {class}`~unite.line.Flux` | Line flux normalization | (internal units) |
| {class}`~unite.line.Param` | Arbitrary profile parameter | user-defined |
| {class}`~unite.disperser.RScale` | Resolution scale on a disperser | dimensionless |
| {class}`~unite.disperser.FluxScale` | Flux calibration scale on a disperser | dimensionless |
| {class}`~unite.disperser.PixOffset` | Pixel wavelength offset on a disperser | pixels |

---

## Priors

Every token carries a **prior distribution** that is sampled in the NumPyro model:

```python
from unite import prior

prior.Uniform(low, high)
prior.TruncatedNormal(loc, scale, low, high)
prior.Fixed(value)          # not sampled; held constant
```

Priors can also be *dependent* on other tokens. You can pass another token (or an arithmetic
expression involving tokens) as a bound:

```python
fwhm_narrow = line.FWHM('fwhm_narrow', prior=prior.Uniform(100, 1000))

# Broad component must be at least 150 km/s wider than the narrow component
fwhm_broad = line.FWHM('fwhm_broad', prior=prior.Uniform(fwhm_narrow + 150, 5000))
```

`ModelBuilder` uses topological sorting to sample `fwhm_narrow` first and then construct the
prior for `fwhm_broad` at runtime.

---

## Line Configuration

{class}`~unite.line.LineConfiguration` is the container for emission line specifications.

```python
lc = line.LineConfiguration()
lc.add_line(
    name='H_alpha',
    center=6563.0 * u.AA,
    redshift=z,
    fwhm_gauss=fwhm,          # Gaussian FWHM token (required for Gaussian profile)
    flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 10)),
    profile='Gaussian',        # default
)
```

Multiple calls with the same `name` create **multiple components** (e.g., broad + narrow):

```python
lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z_narrow, fwhm_gauss=fwhm_narrow, ...)
lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z_broad,  fwhm_gauss=fwhm_broad,  ...)
```

---

## Continuum Configuration

{class}`~unite.continuum.ContinuumConfiguration` defines wavelength regions with a
functional continuum form attached to each.

The easiest way to build a continuum configuration is automatically from the line centers:

```python
from unite.continuum import ContinuumConfiguration, Linear

cc = ContinuumConfiguration.from_lines(
    lc.centers,    # wavelength array of line rest-frame centers
    pad=0.05,      # fractional padding on each side
    form=Linear(), # continuum form: Linear, PowerLaw, Polynomial, …
)
```

This pads around each set of lines, merges overlapping regions, and assigns the chosen form
to every region.

---

## The Prepare → Scale → Build Pipeline

Before running the sampler, three preparation steps are needed:

### 1. `Spectra.prepare()`

```python
spectra = Spectra([spectrum], redshift=0.0)
filtered_lines, filtered_cont = spectra.prepare(lc, cc)
```

This filters out lines and continuum regions not covered by the spectrum's wavelength range.
The result is two new configuration objects containing only the relevant features.

### 2. `Spectra.compute_scales()`

```python
spectra.compute_scales(filtered_lines, filtered_cont, error_scale=True)
```

Computes the median flux level of the continuum regions and uses it to set internal
normalization constants (`line_scale`, `continuum_scale`). This makes the sampler's job
easier by keeping parameter values near unity.

When `error_scale=True` is passed, it also estimates and applies per-spectrum error
rescaling factors. This is useful when the pipeline uncertainties are unreliable.

### 3. `ModelBuilder.build()`

```python
builder = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
model_fn, model_args = builder.build()
```

Returns a NumPyro model function and a {class}`~unite.model.ModelArgs` dataclass containing
all pre-computed matrices, scales, and data arrays. You then pass these directly to any
NumPyro inference algorithm.

---

## User Controls the Sampler

`unite` does not bundle a fitting loop. The `(model_fn, model_args)` tuple can be used with:

```python
# NUTS (recommended for well-behaved posteriors)
from numpyro import infer
kernel = infer.NUTS(model_fn)
mcmc = infer.MCMC(kernel, num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), model_args)

# SVI (variational inference — faster but approximate)
from numpyro import infer
guide = infer.autoguide.AutoNormal(model_fn)
svi = infer.SVI(model_fn, guide, infer.optim.Adam(0.01), infer.Trace_ELBO())
```

---

## Configuration Serialization

Every configuration object can be saved to and loaded from YAML:

```python
from unite.config import Configuration

config = Configuration(line=lc, continuum=cc, dispersers=dc)
config.to_yaml('my_fit.yaml')

# Load back — all tokens and sharing relationships are preserved
config2 = Configuration.from_yaml('my_fit.yaml')
```

See {doc}`guides/serialization` for the full workflow.

---

## Spectra Collection

{class}`~unite.spectrum.Spectra` is a container for one or more spectra:

```python
from unite.spectrum import Spectra

# Single spectrum
spectra = Spectra([spectrum], redshift=0.0)

# Multiple spectra (e.g., two NIRSpec gratings)
spectra = Spectra([prism_spectrum, g395m_spectrum], redshift=5.28)
```

The `redshift` argument shifts all line centers to the observed frame before coverage
filtering. This is necessary when the lines fall far from their rest-frame wavelengths.
