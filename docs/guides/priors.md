# Priors

Every parameter in `unite` carries a **prior distribution** that controls the range and shape
of the posterior. `unite` provides three built-in prior types plus a mechanism for
**dependent priors** that reference other parameters.

---

## Supported Priors

### Uniform

A flat prior between `low` and `high`:

```python
from unite import prior

p = prior.Uniform(low=0.0, high=1000.0)
```

- Most common choice for line fluxes, FWHM bounds, and other parameters with no strong
  prior expectation.
- Bounds can be floats or {class}`~unite.prior.ParameterRef` expressions
  (see [Dependent Priors](#dependent-priors) below).

### TruncatedNormal

A Gaussian centred at `loc` with standard deviation `scale`, truncated to `[low, high]`:

```python
p = prior.TruncatedNormal(loc=1.0, scale=0.05, low=0.8, high=1.2)
```

- Good for calibration parameters where you have a prior expectation (e.g.,
  {class}`~unite.disperser.base.RScale` near 1.0).
- `loc`, `low`, and `high` can all be {class}`~unite.prior.ParameterRef` expressions.

### Fixed

A constant value — not sampled:

```python
p = prior.Fixed(6564.61)
```

- The parameter is held constant at the given value.
- `to_dist()` returns `None` (no NumPyro distribution is created).
- Useful for:
  - Fixing line ratios to theoretical values
  - Fixing calibration parameters to known corrections
  - Fixing redshift when known precisely

---

## Dependent Priors

One of `unite`'s most powerful features. Prior bounds can reference other parameters using
arithmetic expressions on token objects. This creates **dependency chains** that are
automatically resolved via topological sorting at model-build time.

### Basic Example — Minimum FWHM Separation

```python
from unite import line, prior

fwhm_narrow = line.FWHM('fwhm_narrow', prior=prior.Uniform(50, 500))

# Broad component must be at least 150 km/s wider
fwhm_broad = line.FWHM('fwhm_broad', prior=prior.Uniform(fwhm_narrow + 150, 5000))
```

The expression `fwhm_narrow + 150` creates a {class}`~unite.prior.ParameterRef`. At runtime,
`fwhm_narrow` is sampled first, then the lower bound of `fwhm_broad`'s prior is evaluated as
`sampled_value + 150`.

### Supported Arithmetic

| Expression | Result |
|------------|--------|
| `token + constant` | value + constant |
| `token - constant` | value - constant |
| `constant - token` | constant - value (negates scale) |
| `token * constant` | value × constant |
| `constant * token` | value × constant |
| `token / constant` | value ÷ constant |

Expressions can be **chained**: `token * 2 + 150` produces `scale=2, offset=150`, so the
result is `2 * sampled_value + 150`.

:::{note}
Only **linear** expressions (scale × token + offset) are supported. Combinations of two
tokens in a single expression are not.
:::

### Deep Dependency Chains

Dependencies can be **arbitrarily deep**:

```python
fwhm_narrow    = line.FWHM('fwhm_n',  prior=prior.Uniform(50, 500))
fwhm_broad     = line.FWHM('fwhm_b',  prior=prior.Uniform(fwhm_narrow + 150, 3000))
fwhm_very_broad = line.FWHM('fwhm_vb', prior=prior.Uniform(fwhm_broad + 200, 5000))
```

Topological sorting ensures the sampling order: `fwhm_n` → `fwhm_b` → `fwhm_vb`.

### Flux Ratios

Dependent priors are especially useful for constraining flux ratios between lines:

```python
flux_ha = line.Flux('Ha_flux', prior=prior.Uniform(0, 10))

# [NII]6585 / Ha ratio between 0.1 and 3.0
flux_nii = line.Flux('NII_flux', prior=prior.Uniform(flux_ha * 0.1, flux_ha * 3.0))

# [NII]6549 is 1/3 of [NII]6585 (fixed ratio from atomic physics)
flux_nii_weak = line.Flux(
    'NII_weak_flux',
    prior=prior.Uniform(flux_nii / 3.0 - 0.01, flux_nii / 3.0 + 0.01),
)
```

### Using with TruncatedNormal

Both bounds **and** `loc` can be {class}`~unite.prior.ParameterRef` expressions:

```python
fwhm_narrow = line.FWHM('fwhm_n', prior=prior.Uniform(50, 500))

# Broad component centred at 2× narrow, with scatter
fwhm_broad = line.FWHM('fwhm_b', prior=prior.TruncatedNormal(
    loc=fwhm_narrow * 2,
    scale=50,
    low=fwhm_narrow + 100,
    high=5000,
))
```

---

## Topological Sorting

When parameters have dependent priors, {class}`~unite.model.ModelBuilder` uses
{func}`~unite.prior.topological_sort` (Kahn's algorithm) to determine the sampling order.
Parameters are always sampled **before** any parameter whose prior depends on them.

```python
from unite import model

builder = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
model_fn, model_args = builder.build()
print(model_args.dependency_order)
# e.g. ['fwhm_n', 'fwhm_b', 'fwhm_vb', 'Ha_flux', ...]
```

**Circular dependencies** (A depends on B, B depends on A) raise a `ValueError` at build
time with a clear error message.

---

## Serialization

Priors round-trip through YAML, including dependent priors with
{class}`~unite.prior.ParameterRef` expressions. In the YAML format, a dependent bound
appears as a reference with scale and offset:

```yaml
fwhm_broad:
  name: fwhm_b
  prior:
    type: Uniform
    low: {ref: fwhm_n, scale: 1.0, offset: 150.0}
    high: 5000.0
```

The full configuration round-trip preserves all dependency chains. See
{doc}`serialization` for the complete serialization workflow.
