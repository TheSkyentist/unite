# Results and Output

After running the sampler, `unite` provides three output functions that transform raw
posterior samples into user-friendly tables and FITS files.

All functions share the same signature:

```python
from unite.results import make_parameter_table, make_spectra_tables, make_hdul

samples = mcmc.get_samples()   # dict of str → ndarray, shape (n_samples,)
```

---

## Parameter Table

{func}`~unite.results.make_parameter_table` returns a flat {class}`~astropy.table.Table`
with one column per model parameter.

### Full posterior (default)

```python
table = make_parameter_table(samples, model_args)
# One row per posterior sample, one column per parameter

print(table.colnames)
# ['nlr_z', 'nlr_fwhm', 'Ha_flux', 'NII6585_flux', 'NII6549_flux',
#  'cont_slope_0', 'cont_intercept_0', ...]

import numpy as np
print(f"Ha_flux: {np.median(table['Ha_flux']):.3f} ± {np.std(table['Ha_flux']):.3f}")
```

### Summary (median + 16th/84th percentiles)

```python
table = make_parameter_table(samples, model_args, summary=True)
# Three rows: 'median', 'p16', 'p84'

print(table['stat', 'Ha_flux'])
```

### Column units

All columns carry physical {class}`~astropy.units.Quantity` units where known:

| Column type | Unit |
|-------------|------|
| Line flux | `flux_unit * canonical_unit` |
| FWHM | km/s |
| Continuum `scale` | `flux_unit` |
| Continuum `slope` / polynomial coefficients | `flux_unit / canonical_unit^n` |
| Shape / index parameters (`beta`, `temperature`, …) | dimensionless or K |
| **Rest equivalent width** (`{line_label}_rew`) | `canonical_unit` (rest frame) |

The **canonical unit** is the wavelength unit of the first spectrum's disperser (e.g.
`u.AA` or `u.um`).  It can be overridden when constructing `Spectra`.  See
{doc}`spectra` for details.

### Rest equivalent widths

When a continuum is included in the fit, `make_parameter_table` automatically appends one
rest equivalent width (REW) column per emission line.  Columns are named
`{line_label}_rew` — e.g. `Ha_rew`, `[NII]_6585_rew`.

The REW is computed per posterior sample as:

$$\mathrm{REW}_j = \frac{F_j}{C_j^\mathrm{obs} \,(1 + z_j)}$$

where $F_j$ is the physical integrated line flux (in `flux_unit * canonical_unit`),
$C_j^\mathrm{obs}$ is the continuum flux density evaluated at the observed-frame line centre
(in `flux_unit`), and the $(1 + z_j)$ factor (with $z_j = z_\mathrm{sys} + \Delta z_j$)
converts the observer-frame equivalent width to **rest frame**.  The result is in
`canonical_unit`.

A few things to keep in mind:

- **Sign follows line flux** — emission lines yield positive REW; absorption lines (negative
  flux) yield negative REW.  This matches the standard spectroscopic sign convention.
- **Lines without continuum coverage are omitted** — if a line's rest-frame wavelength falls
  outside every continuum region, no `_rew` column is produced for it.
- **No continuum, no REW** — if the model was built without a continuum, no `_rew` columns
  appear at all.

### Table metadata

The table carries useful metadata:

| Key | Content |
|-----|---------|
| `LFLXSCL` | Line flux scale factor (multiply flux parameters by this) |
| `NRMFCTRS` | Per-spectrum continuum normalization factors |
| `ZSYS` | Systemic redshift used for coverage filtering |

---

## Per-Spectrum Model Tables

{func}`~unite.results.make_spectra_tables` returns a list of {class}`~astropy.table.Table`
objects — one per spectrum — with the model decomposed into individual components.

```python
tables = make_spectra_tables(samples, model_args, summary=True)

for t in tables:
    print(t.meta['SPECNAME'])   # spectrum name
    print(t.colnames)
    # ['wavelength', 'model_total', 'H_alpha_0', 'H_alpha_1', 'NII_6585_0',
    #  'NII_6549_0', 'cont_region_0', 'observed_flux', 'observed_error']
```

### Columns

| Column | Description |
|--------|-------------|
| `wavelength` | Observed-frame wavelength (trimmed to continuum regions) |
| `model_total` | Total model (lines + continuum) |
| `<line_name>_<i>` | Individual line component `i` |
| `cont_region_<k>` | Continuum contribution from region `k` |
| `observed_flux` | Observed flux from the input spectrum |
| `observed_error` | Scaled uncertainty |

### Array shapes

- **Full mode** (default): each column except `wavelength` has shape `(n_pixels, n_samples)`.
- **Summary mode** (`summary=True`): shape is `(n_pixels, 3)` — columns are
  `[median, p16, p84]` along the last axis.

### NaN separators between regions

If your fit has multiple disjoint continuum regions, pass `insert_nan=True` to insert a NaN
row at the wavelength gap between each pair of regions. This is useful for clean matplotlib
plots:

```python
tables = make_spectra_tables(samples, model_args, summary=True, insert_nan=True)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for t in tables:
    ax.step(t['wavelength'], t['model_total'][:, 0],  # median
            where='mid', label=t.meta['SPECNAME'])
ax.set_xlabel('Wavelength')
ax.legend()
```

---

## FITS Output

{func}`~unite.results.make_hdul` wraps everything in an {class}`~astropy.io.fits.HDUList`:

```python
hdul = make_hdul(samples, model_args, summary=True)
hdul.writeto('results.fits', overwrite=True)
```

### HDU structure

| HDU | Name | Type | Content |
|-----|------|------|---------|
| 0 | `PRIMARY` | `PrimaryHDU` | Empty data; header with global metadata |
| 1 | `PARAMETERS` | `BinTableHDU` | Parameter posterior table |
| 2+ | `<SPECNAME>` | `BinTableHDU` | Per-spectrum decomposition table |

### Primary header keywords

| Keyword | Content |
|---------|---------|
| `ZSYS` | Systemic redshift |
| `LFLXSCL` | Line flux scale |
| `NSPEC` | Number of spectra |

### Reading the FITS file

```python
from astropy.io import fits
from astropy.table import Table

with fits.open('results.fits') as hdul:
    param_table = Table.read(hdul['PARAMETERS'])
    spec_table  = Table.read(hdul[2])   # first spectrum

print(param_table['Ha_flux'])
```

---

## Evaluating the Model at Arbitrary Samples

For more advanced use (e.g., plotting individual draws or computing derived quantities),
use {func}`~unite.evaluate.evaluate_model` directly:

```python
from unite.evaluate import evaluate_model

predictions = evaluate_model(samples, model_args)

for pred, spectrum in zip(predictions, model_args.spectra):
    # pred.total         — (n_samples, n_pixels)
    # pred.lines         — dict of name → (n_samples, n_pixels)
    # pred.continuum_regions — dict of name → (n_samples, n_pixels)
    # pred.wavelength    — (n_pixels,)
    print(pred.wavelength.shape, pred.total.shape)
```

{class}`~unite.evaluate.SpectrumPrediction` is a simple dataclass; use standard NumPy
operations to compute any derived quantity you need.
