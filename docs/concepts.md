# Core Concepts

This page explains the key ideas behind `unite`'s design. Understanding these concepts will
make the tutorials and guides much easier to follow. Please reference the relevant usage sections
for detailed examples, explanation, and API documentation.

---

## Key Assumptions

`unite` makes several scientific and practical assumptions. Understanding these upfront will
help you assess whether the package is appropriate for your data and avoid common pitfalls.

### Same-Source Assumption

The model is evaluated independently for each disperser/spectrum, but all spectra are assumed
to observe the **same physical source** with the same intrinsic properties (redshift, line
widths, line fluxes). This is what enables simultaneous fitting across multiple gratings.

This assumption may **not** hold if:

- **The source varies between observations** — e.g., time-variable AGN or transients observed
  at different epochs.
- **Different slit positions or fiber placements** sample different spatial regions of an
  extended source, leading to different line ratios or kinematics.
- **Different aperture sizes** capture different fractions of the source flux, biasing
  relative line strengths.

For **JWST/NIRSpec MSA** observations, objects observed with the same mask configuration are
typically observed simultaneously across all gratings, so the same-source assumption is
usually valid. Calibration tokens ({class}`~unite.disperser.base.FluxScale`,
{class}`~unite.disperser.base.RScale`) can absorb some inter-disperser differences in flux
calibration or resolution, but they cannot account for fundamentally different source
properties.

### Gaussian Line Spread Function

The spectral LSF is modelled as a **Gaussian** at every pixel, with FWHM determined by the
disperser's resolution curve $R(\lambda)$:

$$\mathrm{lsf\_fwhm}(\lambda) = \frac{\lambda}{R(\lambda)}$$

For profiles with a Gaussian component (`Gaussian`, `PseudoVoigt`, `SEMG`, `GaussHermite`,
`SplitNormal`), the intrinsic and LSF widths are added in quadrature:

$$\mathrm{fwhm\_total} = \sqrt{\mathrm{fwhm\_intrinsic}^2 + \mathrm{lsf\_fwhm}^2}$$

For purely Lorentzian components (`Cauchy`, and the Lorentzian part of `PseudoVoigt`), the
LSF is **not** convolved into the Lorentzian width. This means a "Cauchy" profile in `unite`
is effectively a Voigt-like profile (Lorentzian convolved with the Gaussian LSF), which is
physically appropriate since the instrumental broadening is always present.

### Pixel Integration

`unite` provides two integration modes, selectable via `integration_mode` on
{meth}`~unite.model.ModelBuilder.build`:

**Analytic mode** (default) integrates each line profile over pixel bins using its CDF:

- **Exact** for emission lines — no discretisation error from summing sub-pixel samples
- **Fast** — one CDF evaluation per pixel edge, independent of line width
- **Robust for undersampled data** — critical for NIRSpec PRISM where lines can be narrower
  than a single pixel
- **Approximate for absorption lines** — each profile is integrated independently before the
  nonlinear transmission `exp(-τ·φ)` is applied.  This computes `exp(-τ·∫φ)` rather than
  `∫F·exp(-τ·φ)`, which is accurate when the absorber is well-resolved but introduces an
  approximation for unresolved or marginally resolved lines.

**Quadrature mode** evaluates the full composed model — emission, absorption, and
continuum together — at Gauss-Legendre sub-pixel nodes and integrates via weighted sum:

- **Exact** for both emission and absorption — properly computes `∫F(λ)·exp(-τ·φ(λ)) dλ`
  over each pixel
- **Slower** — requires `n_nodes` (default 7) profile evaluations per pixel instead of one
- **Recommended when** tau-parametrized lines are unresolved or marginally resolved, or when
  mixing emission and absorption at similar wavelengths

```python
# Analytic (default) — fast, exact for emission lines
model_fn, args = builder.build(integration_mode='analytic')

# Quadrature — exact for all lines, including absorption
model_fn, args = builder.build(integration_mode='quadrature', n_nodes=7)
```

The continuum is evaluated at pixel centers in both modes, since it varies slowly enough
that sub-pixel variation is negligible.

---

## Tokens and Parameter Sharing

The central design pattern in `unite` is the **token**. A token is a named Python object
that represents a model parameter. Sharing a token between two lines means those lines share
the *same* parameter in the model.

```python
from unite import line, prior
from astropy import units as u

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
z_narrow = line.Redshift('narrow', prior=prior.Uniform(-0.01, 0.01))
z_broad  = line.Redshift('broad',  prior=prior.Uniform(-0.01, 0.01))

lc.add_line('Ha_narrow', 6563.0 * u.AA, redshift=z_narrow, ...)
lc.add_line('Ha_broad',  6563.0 * u.AA, redshift=z_broad,  ...)
```

---

## Priors

Every token carries a **prior distribution** that is sampled in the NumPyro model:

```python
from unite import prior

prior.Uniform(low, high)                       # flat prior
prior.TruncatedNormal(loc, scale, low, high)   # Gaussian with hard bounds
prior.Fixed(value)                             # not sampled; held constant
```

Priors can be *dependent* on other tokens — for example, constraining a broad FWHM to always
exceed a narrow FWHM by at least 150 km/s:

```python
fwhm_narrow = line.FWHM('narrow', prior=prior.Uniform(100, 1000))
fwhm_broad  = line.FWHM('broad',  prior=prior.Uniform(fwhm_narrow + 150, 5000))
```

See {doc}`usage/priors` for the full reference on supported priors, dependent priors, and
topological sorting.

---

## Line Configuration

{class}`~unite.line.LineConfiguration` is the container for emission line specifications.
Lines are added with `add_line`, specifying the rest-frame center wavelength, kinematic
tokens, flux, and (optionally) a profile shape.

```python
lc = line.LineConfiguration()
lc.add_line(
    name='H_alpha',
    center=6563.0 * u.AA,
    redshift=z,
    fwhm_gauss=fwhm,
    flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 10)),
    profile='Gaussian',   # default
)
```

Multiple calls with the same name create **multiple components** (e.g., narrow + broad) whose
fluxes are summed. See {doc}`usage/line_configuration` for the full guide including all
seven profile shapes, parameter sharing patterns, and merging configurations.

---

## Continuum Configuration

{class}`~unite.continuum.ContinuumConfiguration` defines wavelength regions with a
functional continuum form attached to each. The easiest way to build one is automatically
from the line centers:

```python
from unite.continuum import ContinuumConfiguration, Linear

cc = ContinuumConfiguration.from_lines(
    lc.centers,    # wavelength array of line rest-frame centers
    form=Linear(), # continuum form (or string name, e.g. 'Linear')
)
```

Nine built-in forms are available: {class}`~unite.continuum.Linear`,
{class}`~unite.continuum.PowerLaw`, {class}`~unite.continuum.Polynomial`,
{class}`~unite.continuum.Chebyshev`, {class}`~unite.continuum.BSpline`,
{class}`~unite.continuum.Bernstein`, {class}`~unite.continuum.Blackbody`,
{class}`~unite.continuum.ModifiedBlackbody`, and
{class}`~unite.continuum.AttenuatedBlackbody`.

Each form's model parameters (e.g. `scale`, `slope`, `temperature`) receive default priors
that can be overridden per-region via `ContinuumRegion(params={...})`. Sharing the same
{class}`~unite.prior.Parameter` instance across regions ties them to a single model
parameter — the same token pattern used for emission lines.

See {doc}`usage/continuum_configuration` for all available functional forms, custom priors,
parameter sharing, and the quick-reference table.

---

## Dispersers & Spectra

A **disperser** represents an instrument's wavelength dispersion and resolution properties. It encodes two calibrations: the resolving power $R(\lambda)$ (used to compute the instrumental line spread function) and the wavelength dispersion per pixel $d\lambda/\mathrm{dpix}(\lambda)$. Built-in support is provided for JWST/NIRSpec and SDSS; custom instruments can be configured using generic disperser classes. Optional calibration tokens ({class}`~unite.instrument.RScale`, {class}`~unite.instrument.FluxScale`, {class}`~unite.instrument.PixOffset`) can absorb uncertainties in resolution, flux calibration, and wavelength solution.

{class}`~unite.spectrum.Spectra` is a container for one or more spectra:

```python
from unite.instrument import nirspec
from unite.spectrum import Spectra

# Configure dispersers
g235h = nirspec.G235H()
g395m = nirspec.G395M()

# Load spectra (NIRSpec example)
spectrum1 = nirspec.NIRSpecSpectrum.from_DJA('g235h.fits', disperser=g235h)
spectrum2 = nirspec.NIRSpecSpectrum.from_DJA('g395m.fits', disperser=g395m)

# Wrap in Spectra container
spectra = Spectra([spectrum1, spectrum2], redshift=5.28)
```

The `redshift` argument shifts all line centers to the observed frame before coverage
filtering. See {doc}`usage/instrument` for more on data handling, error scaling, and
multi-spectrum fits.

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

This performs two tasks:

**Flux normalisation.** Estimates characteristic flux scales (`line_scale`,
`continuum_scale`) so that sampler parameters are near unity. This is important for efficient
MCMC sampling — parameters spanning many orders of magnitude lead to poor posterior geometry
and slow convergence.

**Error scaling** (when `error_scale=True`). Spectral reduction pipelines — including
NIRSpec's — often produce error arrays that over/underestimate the true noise. `unite` can optionally rescale errorbars to mitigate this effect, though it does not currently take into account correlated noise.

### 3. `ModelBuilder.build()`

```python
builder = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
model_fn, model_args = builder.build()
```

Returns a NumPyro model function and a {class}`~unite.model.ModelArgs` dataclass containing
all pre-computed matrices, scales, and data arrays.

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
guide = infer.autoguide.AutoNormal(model_fn)
svi = infer.SVI(model_fn, guide, infer.optim.Adam(0.01), infer.Trace_ELBO())
```

---

## Configuration Serialization

Every configuration object can be saved to and loaded from YAML:

```python
from unite.config import Configuration

config = Configuration(lines=lc, continuum=cc, dispersers=dc)
config.save('my_fit.yaml')

# Load back — all tokens and sharing relationships are preserved
config2 = Configuration.load('my_fit.yaml')
```

See {doc}`usage/serialization` for the full workflow, YAML format, and sub-configuration
serialization.
