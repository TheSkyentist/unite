# Spectra and Data

{class}`~unite.instrument.generic.GenericSpectrum` holds a single observed spectrum;
{class}`~unite.instrument.spectrum.Spectra` is a collection that handles coverage filtering, flux
normalization, and error scaling.

---

## GenericSpectrum

A {class}`~unite.instrument.generic.GenericSpectrum` is defined by pixel bin edges, flux and
error arrays, and a {class}`~unite.instrument.base.Disperser`:

```python
from unite.instrument import generic
from astropy import units as u
import numpy as np

wavelength = np.linspace(6400, 6700, 300) * u.AA
low = wavelength - 0.5 * np.gradient(wavelength)
high = wavelength + 0.5 * np.gradient(wavelength)
flux = np.random.normal(10, 2, 300) * u.erg / u.s / u.cm**2 / u.AA
error = np.full(300, 2.0) * u.erg / u.s / u.cm**2 / u.AA

disperser = generic.SimpleDisperser(wavelength=wavelength.value, unit=u.AA, R=3000, name='sim')
spectrum = generic.GenericSpectrum(low=low, high=high, flux=flux, error=error, disperser=disperser)
```

### Requirements

- `low` and `high` must be astropy Quantities with wavelength (length) dimensions
- `flux` and `error` must be astropy Quantities with spectral flux density per wavelength
  units ($f_\lambda$, e.g., `erg / s / cm^2 / AA`)
- `flux` and `error` must have compatible units
- `disperser` must be a {class}`~unite.instrument.base.Disperser` instance
- All arrays must be 1-D with the same length

### Name

The `name` keyword defaults to `disperser.name`. It appears in results tables and FITS
headers:

```python
spectrum = GenericSpectrum(..., name='G235H_obs1')
```

### Instrument-specific Subclasses

Instrument-specific spectrum classes are **subclasses** of `GenericSpectrum`:

```python
from unite.instrument import nirspec, sdss, generic

nirspec_spec = nirspec.NIRSpecSpectrum.from_DJA('file.fits', disperser=g235h)
sdss_spec    = sdss.SDSSSpectrum.from_fits('spec.fits', disperser=sdss_disp)

# Both are GenericSpectrum instances
assert isinstance(nirspec_spec, generic.GenericSpectrum)  # True
assert isinstance(sdss_spec, generic.GenericSpectrum)     # True

# And their repr shows the concrete class name
print(nirspec_spec)  # NIRSpecSpectrum 'G235H': 1200 px, λ ∈ [1.66, 3.17] um
```

:::{note}
`GenericSpectrum`, `GenericDisperser`, and `SimpleDisperser` are intentionally not
re-exported from `unite.instrument`. Import them explicitly from
`unite.instrument.generic` to make the generic nature of your code clear.
:::

---

## Spectra Collection

{class}`~unite.instrument.spectrum.Spectra` is the main container for fitting. It holds one
or more {class}`~unite.instrument.generic.GenericSpectrum` objects:

```python
from unite.instrument import Spectra

# Single spectrum
spectra = Spectra([spectrum], redshift=0.0)

# Multiple spectra (e.g., two NIRSpec gratings)
spectra = Spectra([g235h_spectrum, g395h_spectrum], redshift=5.28)
```

The `redshift` argument shifts all line centres to the observed frame before coverage
filtering. Essential when fitting high-redshift sources where lines fall far from their
rest-frame wavelengths.

---

## The Prepare → Scale → Build Pipeline

### 1. prepare() — Coverage Filtering

```python
filtered_lines, filtered_cont = spectra.prepare(lc, cc)
```

This:

- Shifts line centres by $(1 + z)$ to the observed frame
- Checks which lines fall within each spectrum's wavelength range
- Drops lines and continuum regions with no spectral coverage
- Returns filtered copies of the line and continuum configurations

A line that falls in one grating but not another is only modelled in the grating where it is
covered.

### 2. compute_scales() — Normalization and Error Scaling

```python
spectra.compute_scales(filtered_lines, filtered_cont, error_scale=True)
```

This performs two tasks:

#### Flux Normalization

Estimates characteristic flux scales so that sampler parameters are near unity. This is
important for efficient MCMC sampling — parameters spanning many orders of magnitude lead to
poor posterior geometry and slow convergence.

- **line_scale**: Estimated from peak flux above the continuum times the expected line width.
  Stored as a {class}`~astropy.units.Quantity` in integrated flux units.
- **continuum_scale**: Maximum median $|f|$ across continuum regions after masking lines.
  Stored as a {class}`~astropy.units.Quantity` in flux density units.

#### Error Scaling

When `error_scale=True`, `compute_scales` also estimates per-region error scaling factors.

Spectral reduction pipelines often produce error arrays that **underestimate the true
noise**. This is because:

- Pipeline errors typically capture photon noise and read noise but not **correlated noise**
  from resampling (drizzling), flat-fielding, or background subtraction.
- The reduction process — especially for JWST/NIRSpec and other IFU/MSA instruments —
  introduces pixel-to-pixel correlations that the formal error bars do not account for.
- The result is artificially small error bars, leading to overconfident posteriors.

The error scaling algorithm:

1. Masks pixels near emission lines (using a configurable FWHM, convolved in quadrature
   with the local LSF)
2. Fits a low-order polynomial to the unmasked pixels in each continuum region
3. Computes the reduced $\chi^2$ ($\chi^2_\mathrm{red}$) of the residuals
4. Inflates errors by $\sqrt{\max(\chi^2_\mathrm{red},\; 1)}$ per region

This is a **conservative** correction:

- Regions where the pipeline errors are already correct ($\chi^2_\mathrm{red} \approx 1$)
  are left unchanged.
- Regions with underestimated errors are inflated to match the observed scatter.

The per-pixel error scaling factors are stored on each spectrum via the `error_scale`
attribute:

```python
for spec in spectra:
    print(spec.name, spec.error_scale)
```

#### Inspecting the Continuum Fit

After calling `compute_scales`, the fitted continuum model and per-region diagnostics are
available via `Spectra.scale_diagnostics`. This is a list of
`SpectrumScaleDiagnostic` objects — one per spectrum — each containing:

| Attribute | Description |
|---|---|
| `wavelength` | Pixel-centre wavelengths (disperser unit) |
| `flux` / `error` | Observed flux and uncertainty arrays |
| `line_mask` | Boolean array — `True` where a pixel was excluded near an emission line |
| `continuum_model` | Full-length continuum model array; `NaN` outside any fitted region |
| `regions` | List of `RegionDiagnostic` objects, one per continuum region |

Each `RegionDiagnostic` holds `obs_low`, `obs_high`, `in_region`, `good_mask`,
`model_on_region`, `chi2_red`, and `fit_params`.

A typical inspection loop:

```python
spectra.compute_scales(filtered_lines, filtered_cont, error_scale=True)

import numpy as np
for diag in spectra.scale_diagnostics:
    wl = np.asarray(diag.wavelength)
    flux = np.asarray(diag.flux)
    cont = np.asarray(diag.continuum_model)   # NaN outside regions
    mask = np.asarray(diag.line_mask)

    for rinfo in diag.regions:
        good = np.asarray(rinfo.good_mask)
        model = np.asarray(rinfo.model_on_region)  # evaluated on in_region pixels
        print(f'  chi2_red = {rinfo.chi2_red:.2f}')
```

See `examples/scale_diagnostic_example.py` for a complete plotting script that renders
three-panel figures (spectrum + fit, residuals in σ, residual histogram) for every
continuum region in every spectrum.

### 3. ModelBuilder.build()

```python
from unite import model

builder = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
model_fn, model_args = builder.build()
```

Returns a NumPyro model function and a {class}`~unite.model.ModelArgs` dataclass containing
all pre-computed matrices, scales, and data arrays. You then run any NumPyro sampler:

```python
import jax
from numpyro import infer

mcmc = infer.MCMC(infer.NUTS(model_fn), num_warmup=500, num_samples=1000)
mcmc.run(jax.random.PRNGKey(0), model_args)
```

---

## Multiple Spectra

When fitting multiple spectra simultaneously, the model is evaluated independently for each
spectrum but parameters are **shared** according to token identity. This is the core of
simultaneous fitting.

```python
from unite.instrument import nirspec, RScale, Spectra
from unite import prior

# Shared resolution calibration across both gratings
r = RScale(prior=prior.TruncatedNormal(1.0, 0.05, 0.8, 1.2), name='r_shared')
g235h = nirspec.G235H(r_scale=r)
g395h = nirspec.G395H(r_scale=r)

spec1 = nirspec.NIRSpecSpectrum.from_DJA('g235h.fits', disperser=g235h)
spec2 = nirspec.NIRSpecSpectrum.from_DJA('g395h.fits', disperser=g395h)

spectra = Spectra([spec1, spec2], redshift=5.28)
```

See {doc}`instruments` for details on calibration tokens and disperser configuration,
including degeneracy warnings.
