# Generic Spectrograph Tutorial

This tutorial walks through a complete `unite` fit on simulated data, using a generic
spectrograph model. No real data files are required — we generate a synthetic spectrum
in the first step.

**What we fit:** H$\alpha$ + [NII]$\lambda\lambda$6549,6585 doublet with a linear continuum.

---

## Imports

```python
import astropy.units as u
import jax
import numpy as np
from numpyro import infer

from unite import line, model, prior
from unite.continuum import ContinuumConfiguration, Linear
from unite.instrument import generic, Spectra
from unite.results import make_hdul, make_parameter_table, make_spectra_tables
```

---

## Step 1 — Simulate a Spectrum

We create a 300-pixel spectrum covering the H$\alpha$ region with three Gaussian emission
lines on a linear continuum, plus Gaussian noise.

```python
rng = np.random.default_rng(42)
wavelength = np.linspace(6400, 6700, 300) * u.AA
wl = wavelength.value

# Three Gaussian lines
sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
line_flux = (
    80 * np.exp(-0.5 * ((wl - 6563.0) / sigma) ** 2)   # H-alpha
    + 30 * np.exp(-0.5 * ((wl - 6549.0) / sigma) ** 2) # [NII] 6549
    + 90 * np.exp(-0.5 * ((wl - 6585.0) / sigma) ** 2) # [NII] 6585
)

# Linear continuum
continuum = 10.0 + 0.005 * (wl - 6550.0)

# Total flux with noise
flux = line_flux + continuum + rng.normal(0, 2, len(wl))
error = np.full_like(flux, 2.0)

# Pixel edges (required by unite for exact pixel integration)
low  = wavelength - 0.5 * np.gradient(wavelength)
high = wavelength + 0.5 * np.gradient(wavelength)
```

---

## Step 2 — Create the Disperser and Spectrum

{class}`~unite.instrument.generic.SimpleDisperser` takes a wavelength array and a constant
spectral resolution $R = \lambda / \Delta\lambda$. It uses these to compute the LSF FWHM at
each pixel for the line integration.

```python
disperser = generic.SimpleDisperser(wavelength=wl, unit=u.AA, R=3000.0, name='sim_grating')

spectrum = generic.GenericSpectrum(
    low=low,
    high=high,
    flux=flux,
    error=error,
    disperser=disperser,
    name='sim_spectrum',
)
```

---

## Step 3 — Build the Line Configuration

We define **shared tokens** for the redshift and FWHM — all three lines will share the same
kinematic parameters. Each line gets its own flux token with an independent prior.

```python
z    = line.Redshift('nlr_z',  prior=prior.Uniform(-0.005, 0.005))
fwhm = line.FWHM('nlr_fwhm',  prior=prior.Uniform(1.0, 10.0))

lc = line.LineConfiguration()

lc.add_line(
    'H_alpha',
    6563.0 * u.AA,
    redshift=z,
    fwhm_gauss=fwhm,
    flux=line.Flux('Ha_flux',      prior=prior.Uniform(0, 5)),
)
lc.add_line(
    'NII_6585',
    6585.0 * u.AA,
    redshift=z,
    fwhm_gauss=fwhm,
    flux=line.Flux('NII6585_flux', prior=prior.Uniform(0, 5)),
)
lc.add_line(
    'NII_6549',
    6549.0 * u.AA,
    redshift=z,
    fwhm_gauss=fwhm,
    flux=line.Flux('NII6549_flux', prior=prior.Uniform(0, 5)),
)
```

:::{note}
The `profile` argument defaults to `'Gaussian'`. See {doc}`../guides/line_configuration` for other
available profiles and their parameters.
:::

---

## Step 4 — Build the Continuum Configuration

{meth}`~unite.continuum.ContinuumConfiguration.from_lines` automatically pads around the
line centers and merges overlapping regions:

```python
cc = ContinuumConfiguration.from_lines(
    lc.centers,    # rest-frame line centers
    pad=0.05,      # 5% fractional padding on each side
    form=Linear(), # linear continuum in each region
)
```

---

## Step 5 — Prepare the Spectra

Wrap the spectrum in a {class}`~unite.instrument.spectrum.Spectra` collection and run the
three preparation steps.

```python
spectra = Spectra([spectrum], redshift=0.0)

# Filter out lines / regions not covered by this spectrum
filtered_lines, filtered_cont = spectra.prepare(lc, cc)
print(f'Filtered lines: {len(filtered_lines)}, regions: {len(filtered_cont)}')

# Compute internal flux normalization and estimate per-spectrum error rescaling
spectra.compute_scales(filtered_lines, filtered_cont, error_scale=True)
print(f'Line scale:      {spectra.line_scale:.4g}')
print(f'Continuum scale: {spectra.continuum_scale:.4g}')
for s in spectra:
    print(f'Error scale for {s.name}: {s.error_scale:.4f}')
```

---

## Step 6 — Build and Sample the Model

{class}`~unite.model.ModelBuilder` assembles a NumPyro model from the filtered
configurations and spectra. You then run it with any NumPyro inference kernel.

```python
builder = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
model_fn, model_args = builder.build()

rng_key = jax.random.PRNGKey(0)
kernel  = infer.NUTS(model_fn)
mcmc    = infer.MCMC(kernel, num_warmup=200, num_samples=500, progress_bar=True)
mcmc.run(rng_key, model_args)

samples = mcmc.get_samples()
```

---

## Step 7 — Extract Results

`unite` provides three output functions for working with posterior samples.

### Parameter table

A flat {class}`~astropy.table.Table` with one row per posterior sample and one column per
parameter:

```python
param_table = make_parameter_table(samples, model_args)

for col in ['nlr_z', 'nlr_fwhm', 'Ha_flux']:
    vals = param_table[col]
    print(f'{col}: median={np.median(vals):.4f}, std={np.std(vals):.4f}')
```

### Per-spectrum model tables

One {class}`~astropy.table.Table` per spectrum with wavelength, model total, continuum, and
individual line contributions:

```python
spectra_tables = make_spectra_tables(samples, model_args, summary=True)

for t in spectra_tables:
    print(t.meta['SPECNAME'], t.colnames)
```

### FITS output

```python
hdul = make_hdul(samples, model_args, summary=True)
hdul.writeto('results.fits', overwrite=True)
```

See {doc}`../guides/results` for a full description of the output format.

---

## Complete Script

The full runnable version of this tutorial is in
[`examples/example_workflow.py`](https://github.com/rhviding/unite/blob/main/examples/example_workflow.py).
