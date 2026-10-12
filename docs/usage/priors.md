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

:::{warning}
Dependent priors like this tie parameters' posteriors together. For NUTS,
pass `dense_mass=True` so warmup learns a full mass matrix instead of a
diagonal one; for SVI, use a full-rank guide (`AutoMultivariateNormal`)
instead of a mean-field one (see [Sampling & Optimization](inference.md))
:::

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

# Broad component centerd at 2x narrow, with scatter
fwhm_broad = line.FWHM('b', prior=prior.TruncatedNormal(
    loc=fwhm_narrow * 2,
    scale=50,
    low=fwhm_narrow + 100,
    high=5000,
))
```

---

## Starting Values

By default, the sampler chooses where each chain starts. `unite` does not set
starting values for you: the starting point of every parameter is drawn by NumPyro's
default [initialization strategy](https://num.pyro.ai/en/stable/utilities.html#initialization-strategies)
(`init_to_uniform`), which draws each chain's start at random from the central part of each
prior's range. Each chain gets its own random key, so chains start in different places.
To use a different strategy, build the kernel yourself and pass `init_strategy=` to it
(see {doc}`inference`), or pass `init_strategy=` to
{meth}`~unite.model.ModelBuilder.fit`.

If you know a good starting point, you can opt in by setting `init` on a
{class}`~unite.prior.Uniform` or {class}`~unite.prior.TruncatedNormal` prior:

```python
from unite import line, prior

z = line.Redshift('z', prior=prior.Uniform(-0.0025, 0.0025, init=0.0))
narrow = line.FWHM('narrow', prior=prior.Uniform(100.0, 600.0, init=250.0))
# Dependent bounds work too: broad's lower bound resolves to 250 * 2 + 150 = 650
broad = line.FWHM(
    'broad', prior=prior.Uniform(low=narrow * 2 + 150, high=3000.0, init=1500.0)
)
```

Parameters without an `init` keep NumPyro's default behavior, so you can set as few or
as many as you like. {class}`~unite.prior.Fixed` priors are never sampled and cannot
have an `init`.

:::{warning}
Setting `init` comes with two caveats:

1. **You are responsible for the value being valid.** It must lie strictly inside the
   prior's bounds, after resolving any [dependent bounds](#dependent-priors). A plain-float
   bound is checked when the prior is created and an out-of-range `init` raises a
   `ValueError`. For dependent bounds, the check happens in
   {meth}`~unite.model.ModelBuilder.build`, but only when every parameter the bound
   depends on is itself fixed or has an `init`. Otherwise the bound varies with that
   parameter's random start and can't be checked, so an invalid value can instead fail
   at sampling time with NumPyro's `Cannot find valid initial parameters`.
2. **All chains start from the same point.** The chains still diverge once sampling begins,
   but they no longer start dispersed, which makes multi-chain convergence diagnostics
   such as R-hat less sensitive to chains that are stuck or have found different modes.
   Treat a good R-hat from chains that share a start with more caution.
:::

### Using starting values with your own kernel

{meth}`~unite.model.ModelBuilder.fit` applies any `init` values automatically. If you
build the kernel yourself, it does **not**: starting values belong to the sampler
kernel, not the model, so {meth}`~unite.model.ModelBuilder.build` returns a `model_fn`
that knows nothing about them. A prior's `init` is silently ignored unless you pass it to
the kernel yourself (NumPyro's default random start is used instead). The values are
available as
{attr}`ModelArgs.init_values <unite.model.ModelArgs.init_values>`: a dictionary keyed by
NumPyro site name (with the type prefix, e.g. `z_...`, `fwhm_gauss_...`) holding only the
parameters that have an `init`. It is in constrained (physical) units, so it plugs
straight into NumPyro's `init_to_value`:

```python
from numpyro import infer

model_fn, model_args = builder.build()
print(model_args.init_values)
# e.g. {'z_z': 0.0, 'fwhm_gauss_narrow': 250.0, 'fwhm_gauss_broad': 1500.0}

kernel = infer.NUTS(
    model_fn,
    init_strategy=infer.init_to_value(values=model_args.init_values),
)
```

The dictionary is a copy, so you can edit it before passing it on. Any parameter missing
from it falls back to NumPyro's default, which is also what `init_to_value` does.
If no prior sets `init`, `init_values` is empty.

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
    low: { op: "+", left: { ref: fwhm_gauss_narrow }, right: 150.0 }
    high: 5000.0

# flux_4363_narrow * flux_5007_broad / flux_5007_narrow
flux_4363_broad:
  prior:
    type: Fixed
    value:
      op: /
      left:
        op: "*"
        left: { ref: flux_4363_narrow }
        right: { ref: flux_5007_broad }
      right: { ref: flux_5007_narrow }
```

A starting value is stored as an `init` key and is written only when set, so YAML saved
before `init` existed still loads unchanged:

```yaml
fwhm_narrow:
  prior:
    type: Uniform
    low: 100.0
    high: 600.0
    init: 250.0
```

The full configuration round-trip preserves all dependency chains. See
{doc}`serialization` for the complete serialization workflow.
