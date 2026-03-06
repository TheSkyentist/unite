# NIRSpec Tutorial

This tutorial demonstrates fitting real JWST/NIRSpec spectra from the
[Dawn JWST Archive (DJA)](https://dawn-cph.github.io/dja/) using `unite`'s built-in NIRSpec
support. We fit H$\alpha$ with a narrow + broad decomposition simultaneously across the
PRISM and G395M gratings.

**What we fit:** Narrow + broad H$\alpha$ in a $z \approx 5.28$ galaxy, using two gratings
simultaneously.

---

## NIRSpec Dispersers

`unite` ships calibration data for all NIRSpec grating/filter combinations:

| Class | Grating | Filter | $R$ range |
|-------|---------|--------|-----------|
| `G140H` | G140H | F070LP or F100LP | 1700–3600 |
| `G140M` | G140M | F070LP or F100LP | 500–1340 |
| `G235H` | G235H | F170LP | 1700–3600 |
| `G235M` | G235M | F170LP | 700–1340 |
| `G395H` | G395H | F290LP | 1700–3600 |
| `G395M` | G395M | F290LP | 700–1340 |
| `PRISM` | PRISM | CLEAR | 30–300 |

Each class accepts an `r_source` argument (`'point'` or `'extended'`) that selects the
appropriate LSF calibration.

---

## Imports

```python
import astropy.units as u
import jax
import numpy as np
from matplotlib import pyplot as plt
from numpyro import infer

from unite import disperser, line, model, prior
from unite.continuum import ContinuumConfiguration, Linear
from unite.disperser.nirspec import G395M, PRISM, NIRSpecSpectrum
from unite.results import make_hdul, make_parameter_table, make_spectra_tables
from unite.spectrum import Spectra
```

---

## Step 1 — Create Dispersers

Instantiate the disperser objects. Here we attach a {class}`~unite.disperser.FluxScale`
calibration token to the G395M grating. This adds a free multiplicative scaling parameter
(with a $\mathcal{U}(0.5, 2.0)$ prior) that accounts for flux calibration uncertainty
between the two gratings.

```python
g395m_disperser = G395M(
    r_source='point',
    flux_scale=disperser.FluxScale(prior.Uniform(0.5, 2.0)),
)
prism_disperser = PRISM(r_source='point')
```

:::{note}
Calibration tokens (`FluxScale`, `RScale`, `PixOffset`) are attached to the **disperser**,
not the spectrum. If two spectra share the same disperser instance, they share the same
calibration parameter. See {doc}`../guides/instruments` for details.
:::

---

## Step 2 — Load Spectra from DJA

{class}`~unite.disperser.nirspec.NIRSpecSpectrum` wraps NIRSpec 1D extractions in the DJA
`msaexp` format. The `from_DJA` class method downloads and parses the FITS file, extracts
wavelength, flux, and error arrays, and returns a {class}`~unite.spectrum.Spectrum`.

```python
g395m_spectrum = NIRSpecSpectrum.from_DJA(
    'https://s3.amazonaws.com/msaexp-nirspec/extractions/'
    'rubies-egs53-v4/rubies-egs53-v4_g395m-f290lp_4233_42046.spec.fits',
    disperser=g395m_disperser,
)
prism_spectrum = NIRSpecSpectrum.from_DJA(
    'https://s3.amazonaws.com/msaexp-nirspec/extractions/'
    'rubies-egs53-v4/rubies-egs53-v4_prism-clear_4233_42046.spec.fits',
    disperser=prism_disperser,
)
```

---

## Step 3 — Build the Line Configuration

We fit H$\alpha$ with two components: a narrow and a broad one. They share the same
redshift but have independent FWHM parameters. The broad FWHM is constrained to be at
least 150 km/s wider than the narrow FWHM using a **dependent prior**.

```python
z = line.Redshift('z', prior=prior.Uniform(-0.005, 0.005))

fwhm_narrow = line.FWHM('fwhm_narrow', prior=prior.Uniform(100, 1000))

# Dependent prior: broad FWHM lower bound is fwhm_narrow + 150 km/s
fwhm_broad  = line.FWHM('fwhm_broad',  prior=prior.Uniform(fwhm_narrow + 150, 5000))

lc = line.LineConfiguration()

# Narrow component
lc.add_line(
    'H_alpha',
    6563.0 * u.AA,
    redshift=z,
    fwhm_gauss=fwhm_narrow,
    flux=line.Flux('Ha_narrow_flux', prior=prior.Uniform(0, 10)),
)

# Broad component — same line name, different token instances → separate parameters
lc.add_line(
    'H_alpha',
    6563.0 * u.AA,
    redshift=z,
    fwhm_gauss=fwhm_broad,
    flux=line.Flux('Ha_broad_flux', prior=prior.Uniform(0, 10)),
)

cc = ContinuumConfiguration.from_lines(lc.centers, pad=0.05, form=Linear())
```

:::{note}
`fwhm_narrow + 150` returns a {class}`~unite.prior.ParameterRef` expression, not a number.
`ModelBuilder` evaluates this expression at sample time after drawing `fwhm_narrow`.
See {doc}`../guides/priors` for more on dependent priors.
:::

---

## Step 4 — Prepare the Spectra

Pass both spectra to a {class}`~unite.spectrum.Spectra` collection and set the source
redshift. `prepare()` will shift line centers to the observed frame and filter coverage for
each spectrum independently.

```python
spectra = Spectra([prism_spectrum, g395m_spectrum], redshift=5.2772)

filtered_lines, filtered_cont = spectra.prepare(lc, cc)
print(f'Lines: {len(filtered_lines)}, regions: {len(filtered_cont)}')

spectra.compute_scales(filtered_lines, filtered_cont, error_scale=True)
print(f'Line scale:      {spectra.line_scale:.4g}')
print(f'Continuum scale: {spectra.continuum_scale:.4g}')
for s in spectra:
    print(f'Error scale for {s.name}: {s.error_scale:.4f}')
```

---

## Step 5 — Build and Sample

```python
builder   = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
model_fn, model_args = builder.build()

mcmc = infer.MCMC(
    infer.NUTS(model_fn),
    num_warmup=200,
    num_samples=500,
    progress_bar=True,
)
mcmc.run(jax.random.PRNGKey(0), model_args)
samples = mcmc.get_samples()
```

---

## Step 6 — Results and Plotting

```python
param_table   = make_parameter_table(samples, model_args)
spectra_tables = make_spectra_tables(samples, model_args, summary=True)

# Save to FITS
hdul = make_hdul(samples, model_args)
hdul.writeto('results.fits', overwrite=True)

# Quick plot
fig, ax = plt.subplots()
colors = ['tab:blue', 'tab:red']

for i, s in enumerate(spectra):
    ax.step(s.wavelength, s.flux, color=colors[i], label=s.name, where='mid')
    ax.fill_between(
        s.wavelength,
        s.flux - s.error,
        s.flux + s.error,
        alpha=0.3, step='mid', color=colors[i],
    )

for i, table in enumerate(spectra_tables):
    ax.step(table['wavelength'], table['model_total'],
            color=colors[i], lw=2, where='mid')

ax.set_xlabel('Wavelength (μm)')
ax.set_ylabel('Flux')
ax.set_xlim([3.75, 4.5])
ax.legend()
fig.savefig('halpha_fit.pdf')
```

---

## Multi-Grating Design Notes

When fitting multiple gratings simultaneously:

- **Coverage filtering** is done per-spectrum: lines only visible in one grating are
  modelled only in that grating.
- **Flux scale tokens** (`FluxScale`) on individual dispersers introduce a free scaling
  factor between gratings. This is useful when relative flux calibration is uncertain.
- **Resolution scale tokens** (`RScale`) allow the LSF width to be a free parameter,
  accounting for LSF calibration uncertainty or spatially extended sources.
- The PRISM's extremely variable resolution (R ≈ 30–300) is handled by the
  wavelength-dependent LSF calibration data bundled with `unite`.

---

## Complete Script

The full runnable version of this tutorial is in
[`examples/nirspec_workflow.py`](https://github.com/rhviding/unite/blob/main/examples/nirspec_workflow.py).
