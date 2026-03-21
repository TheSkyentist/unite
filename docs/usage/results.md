# Results and Output

After running the sampler, `unite` provides three output functions that transform raw
posterior samples into user-friendly FITS Tables.

All functions share the same signature:

```python
from unite.results import make_parameter_table, make_spectra_tables, make_hdul

samples = mcmc.get_samples()   # dict of str → ndarray, shape (n_samples,)
```

:::{tip}
{meth}`~unite.model.ModelBuilder.fit` returns `(samples, model_args)` as a tuple, which
can be passed directly to all three output functions:

```python
samples, model_args = builder.fit()
table = make_parameter_table(samples, model_args)
```
:::

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

### Percentile summaries

```python
import numpy as np

# Return median and 68% credible interval
percentiles = np.array([0.16, 0.5, 0.84])
table = make_parameter_table(samples, model_args, percentiles=percentiles)
# Three rows: percentiles [0.16, 0.5, 0.84]

print(table)
print(table['Ha_flux'])  # shape (3,) for the 3 percentiles
```

Alternatively, you can return all posterior samples (default):

```python
table = make_parameter_table(samples, model_args)
# One row per sample
print(table.colnames)
```

### Column units

All columns carry physical {class}`~astropy.units.Quantity` units where known:

| Column type | Unit |
|-------------|------|
| Line flux | `flux_unit * canonical_unit` |
| FWHM | km/s |
| Continuum `scale` | `flux_unit` |
| Shape / index parameters (`beta`, `temperature`, …) | dimensionless or K |
| Rest equivalent width | `canonical_unit` (rest frame) |

The **canonical unit** is the wavelength unit of the first spectrum's disperser (e.g.
`u.AA` or `u.um`).  It can be overridden when constructing `Spectra`.  See
{doc}`instrument` for details.

### Rest equivalent widths

When a continuum is included in the fit, `make_parameter_table` automatically appends one
rest equivalent width (REW) column per line (both emission and absorption).

#### Emission lines

For emission lines, the REW is computed analytically per posterior sample:

$$\mathrm{REW}_j = \frac{F_j}{C_j^\mathrm{obs} \,(1 + z_j)}$$

where $F_j$ is the physical integrated line flux (in `flux_unit * canonical_unit`),
$C_j^\mathrm{obs}$ is the total continuum flux density at the observed-frame line center
(summing all covering continuum regions, in `flux_unit`), and the $(1 + z_j)$ factor
(with $z_j = z_\mathrm{sys} + \Delta z_j$) converts the observer-frame equivalent
width to **rest frame**.  The result is in `canonical_unit`.

#### Absorption lines

For absorption lines (parameterized with optical depth $\tau$), the REW is computed
by numerical integration:

$$\mathrm{REW}_j = \frac{1}{1 + z_j} \int \frac{\Delta_j(\lambda)}{C_j^\mathrm{obs}} \, d\lambda$$

where $\Delta_j(\lambda) = F_\mathrm{total}(\lambda) \times (1 - 1/T_j(\lambda))$ is the
flux removed by absorber $j$, and $T_j = \exp(-\tau_j \, \phi_j)$ is its transmission.
The integral is evaluated via the trapezoidal rule on the spectrum with the finest pixel
grid covering the line.  Absorption REW values are negative.

:::{note}
Absorption REW values should be used with caution:

1. The integration uses the pixel grid of the highest-resolution spectrum covering
   the absorption line.  If no spectrum fully resolves the absorption profile, the
   REW may be underestimated.
2. The continuum normalization sums **all** continuum regions covering the line center.
   In regions where continuum regions overlap, this may differ from the user's intended
   local continuum level.
:::

#### General notes

- **Sign convention** — emission lines yield positive REW; absorption lines yield
  negative REW.
- **Lines without continuum coverage are omitted** — if a line's rest-frame wavelength
  falls outside every continuum region, no `rew_` column is produced for it.

### Table metadata

The table carries useful metadata:

| Key | Content |
|-----|---------|
| `LFLXSCL` | Line flux scale factor |
| `LFLXUNT` | Unit string for the line flux scale |
| `CNTSCL` | Continuum flux scale factor |
| `CNTUNT` | Unit string for the continuum flux scale |
| `NRMFCTRS` | Per-spectrum continuum normalization factors |
| `ZSYS` | Systemic redshift used for coverage filtering |

---

## Per-Spectrum Model Tables

{func}`~unite.results.make_spectra_tables` returns a list of {class}`~astropy.table.Table`
objects — one per spectrum — with the model decomposed into individual components.

```python
import numpy as np

# Get all posterior samples
tables = make_spectra_tables(samples, model_args)

# OR get specific percentiles
percentiles = np.array([0.16, 0.5, 0.84])
tables = make_spectra_tables(samples, model_args, percentiles=percentiles)

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
| `observed_error` | Pipeline uncertainty array (unscaled) |
| `scaled_error` | Error inflated by the per-spectrum `error_scale` factor |

### Array shapes

- **All samples (default, `percentiles=None`)**: each model column has shape `(n_pixels, n_samples)`.
- **Percentile mode** (e.g., `percentiles=[0.16, 0.5, 0.84]`): shape is `(n_pixels, n_percentiles)`.
  Each sample along the last axis corresponds to one percentile value.

### NaN separators between regions

If your fit has multiple disjoint continuum regions, pass `insert_nan=True` to insert a NaN
row at the wavelength gap between each pair of regions. This is useful for clean matplotlib
plots:

```python
import numpy as np
import matplotlib.pyplot as plt

percentiles = np.array([0.16, 0.5, 0.84])
tables = make_spectra_tables(samples, model_args, percentiles=percentiles, insert_nan=True)

fig, ax = plt.subplots()
for t in tables:
    ax.step(t['wavelength'], t['model_total'][:, 1],  # median (index 1 = 0.5 percentile)
            where='mid', label=t.meta['SPECNAME'])
ax.set_xlabel('Wavelength')
ax.legend()
```

---

## FITS Output

{func}`~unite.results.make_hdul` wraps everything in an {class}`~astropy.io.fits.HDUList`:

```python
import numpy as np

# Get all posterior samples
hdul = make_hdul(samples, model_args)

# OR save specific percentiles to FITS
percentiles = np.array([0.16, 0.5, 0.84])
hdul = make_hdul(samples, model_args, percentiles=percentiles)
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
| `LFLXUNT` | Unit string for the line flux scale |
| `CNTSCL` | Continuum flux scale |
| `CNTUNT` | Unit string for the continuum flux scale |
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

---

## Model Diagnostics

### Degrees of Freedom

{func}`~unite.results.count_parameters` traces the compiled model once (no sampling
required) and counts all free scalar parameters.

```python
from unite.results import count_parameters

model_fn, model_args = builder.build()
n_params = count_parameters(model_fn, model_args)
print(f'Free parameters: {n_params}')
```

### Reduced Chi-Square

Use the median model from {func}`~unite.results.make_spectra_tables` against the
scaled errors.  The `scaled_error` column reflects any per-region error rescaling
applied by {meth}`~unite.spectrum.Spectra.compute_scales`.

```python
import numpy as np

percentiles = np.array([0.16, 0.5, 0.84])
spectra_tables = make_spectra_tables(samples, model_args, percentiles=percentiles, insert_nan=True)

chi2_total = 0.0
n_pixels_total = 0
for t in spectra_tables:
    obs   = np.asarray(t['observed_flux'])
    err   = np.asarray(t['scaled_error'])
    med   = np.asarray(t['model_total'][:, 1])  # median (50th percentile)
    valid = np.isfinite(obs) & np.isfinite(err) & np.isfinite(med) & (err > 0)
    resid = (obs[valid] - med[valid]) / err[valid]
    chi2_total     += float(np.sum(resid**2))
    n_pixels_total += int(valid.sum())

dof      = n_pixels_total - n_params
chi2_red = chi2_total / dof
print(f'χ²_ν = {chi2_red:.3f}  ({n_pixels_total} pixels − {n_params} params = {dof} DoF)')
```

### Log-Likelihood and Log-Posterior

{func}`~numpyro.infer.util.log_likelihood` returns a dict of per-pixel log-likelihoods
(one entry per spectrum); {func}`~numpyro.infer.util.log_density` evaluates the full
log-joint density (likelihood + priors).  `jax.jit(jax.vmap(...))` compiles once and
evaluates all samples in parallel, which is efficient for the typical ~1 000-sample case.

```python
import jax
import jax.numpy as jnp
from numpyro.infer.util import log_density, log_likelihood

# Log-likelihood — shape (n_samples,) after summing over pixels
log_liks = log_likelihood(model_fn, samples, model_args)
n_samp   = next(iter(log_liks.values())).shape[0]
total_ll = sum(v.reshape(n_samp, -1).sum(-1) for v in log_liks.values())
print(f'Mean log-likelihood: {total_ll.mean():.2f}')

# Log-posterior (unnormalized log-joint: log p(θ, data))
def _log_joint(sample):
    ld, _ = log_density(model_fn, (model_args,), {}, sample)
    return ld

log_joint = jax.jit(jax.vmap(_log_joint))(samples)
print(f'Mean log-posterior: {log_joint.mean():.2f}')
```

### WAIC

WAIC is computed per pixel from the log-likelihood arrays.  Lower WAIC is better.

```python
# Per-pixel log-likelihoods: (n_samples, n_pixels_total)
ll_obs = jnp.concatenate(
    [v.reshape(n_samp, -1) for v in log_liks.values()], axis=-1
)

lppd   = jnp.sum(jax.nn.logsumexp(ll_obs, axis=0) - jnp.log(n_samp))
p_waic = jnp.sum(jnp.var(ll_obs, axis=0))
waic   = -2.0 * (lppd - p_waic)
print(f'WAIC: {waic:.2f}')
```

### Going further with ArviZ

[ArviZ](https://python.arviz.org) has first-class NumPyro support and adds WAIC standard
errors, PSIS-LOO, per-observation
diagnostics, and `az.compare()` for ranking multiple models.

```python
import arviz as az

# NUTS — pass the mcmc object directly; ArviZ extracts samples and log-likelihood
# idata = az.from_numpyro(mcmc, log_likelihood=log_liks)

# SVI — no mcmc object, so build InferenceData from dicts
idata = az.from_dict(
    posterior=samples,
    log_likelihood=log_liks,   # dict of site → (n_samples, n_pixels)
)

az.waic(idata)   # waic, waic_se, p_waic
az.loo(idata)    # PSIS-LOO; flags poorly-constrained pixels via Pareto-k
az.compare({'model_a': idata_a, 'model_b': idata_b})  # rank multiple models
```
