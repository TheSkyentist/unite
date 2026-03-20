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
- Bounds can be floats or parameter expressions
  (see [Dependent Priors](#dependent-priors) below).

### TruncatedNormal

A Gaussian centered at `loc` with standard deviation `scale`, truncated to `[low, high]`:

```python
p = prior.TruncatedNormal(loc=1.0, scale=0.05, low=0.8, high=1.2)
```

- Good for calibration parameters where you have a prior expectation (e.g.,
  {class}`~unite.disperser.base.RScale` near 1.0).
- `loc`, `low`, and `high` can all be parameter expressions.

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

### Basic Example

```python
from unite import line, prior

fwhm_narrow = line.FWHM('narrow', prior=prior.Uniform(50, 500))

# Broad component must be at least 150 km/s wider
fwhm_broad = line.FWHM('broad', prior=prior.Uniform(fwhm_narrow + 150, 5000))
```

The expression `fwhm_narrow + 150` creates a parameter expression. At runtime,
`fwhm_narrow` is sampled first, then the lower bound of `fwhm_broad`'s prior is evaluated as
`sampled_value + 150`.

:::{note}
Only a {class}`~unite.prior.Parameter` of the same kind can be used in an expression (e.g., FWHM in FWHM).
You cannot, for example, use a line flux token in an expression for a redshift prior.
:::

### Supported Arithmetic

Parameters support `+`, `-`, `*`, and `/` against constants **or other parameter tokens**,
building an expression tree that is evaluated at sample time. Expressions can be arbitrarily
nested and combined.

Single-token expressions:

```python
fwhm1 = line.FWHM()
fwhm2 = line.FWHM(prior=prior.Uniform(fwhm_narrow * 2 + 150, 5000))
fwhm3 = line.FWHM(prior=prior.Uniform(0, fwhm_narrow / 2 - 150))
```

Two-token expressions — including ratios across parameters:

```python
# Flux ratio constraint: tie [OIII] 4363 ratio across kinematic components.
# Same electron temperature in both narrow and broad components.
flux_5007_narrow = line.Flux('5007_narrow', prior=prior.Uniform(0, 10))
flux_5007_broad  = line.Flux('5007_broad',  prior=prior.Uniform(0, 10))
flux_4363_narrow = line.Flux('4363_narrow', prior=prior.Uniform(0, 10))

# flux_4363_broad is fully determined — not a free parameter
flux_4363_broad = line.Flux(
    '4363_broad',
    prior=prior.Fixed(flux_4363_narrow * flux_5007_broad / flux_5007_narrow),
)
```

This also works for soft constraints using `Uniform` or `TruncatedNormal` bounds:

```python
# Allow some scatter around the expected ratio
flux_4363_broad = line.Flux(
    '4363_broad',
    prior=prior.TruncatedNormal(
        loc=flux_4363_narrow * flux_5007_broad / flux_5007_narrow,
        scale=0.05,
        low=0,
        high=10,
    ),
)
```

### Deep Dependency Chains

Dependencies can be **arbitrarily deep**:

```python
fwhm_narrow    = line.FWHM('n',  prior=prior.Uniform(50, 500))
fwhm_broad     = line.FWHM('b',  prior=prior.Uniform(fwhm_narrow + 150, 3000))
fwhm_very_broad = line.FWHM('vb', prior=prior.Uniform(fwhm_broad + 200, 5000))
```

Topological sorting ensures the sampling order: `fwhm_n` → `fwhm_b` → `fwhm_vb`.

### Using with TruncatedNormal

Both bounds **and** `loc` can be parameter expressions:

```python
fwhm_narrow = line.FWHM('n', prior=prior.Uniform(50, 500))

# Broad component centerd at 2× narrow, with scatter
fwhm_broad = line.FWHM('b', prior=prior.TruncatedNormal(
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
# e.g. ['n', 'b', 'vb', 'Ha_flux', ...]
```

**Circular dependencies** (A depends on B, B depends on A) raise a `ValueError` at build
time with a clear error message.

---

## Serialization

Priors round-trip through YAML, including all parameter expressions. Single-token
references serialize as `{ref: name}`; compound expressions serialize as nested
`{op, left, right}` trees:

```yaml
# fwhm_narrow + 150
fwhm_broad:
  prior:
    type: Uniform
    low: {op: '+', left: {ref: fwhm_gauss_narrow}, right: 150.0}
    high: 5000.0

# flux_4363_narrow * flux_5007_broad / flux_5007_narrow
flux_4363_broad:
  prior:
    type: Fixed
    value:
      op: /
      left:
        op: '*'
        left: {ref: flux_4363_narrow}
        right: {ref: flux_5007_broad}
      right: {ref: flux_5007_narrow}
```

The full configuration round-trip preserves all dependency chains. See
{doc}`serialization` for the complete serialization workflow.
