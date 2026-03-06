# Continuum Configuration

{class}`~unite.continuum.ContinuumConfiguration` defines wavelength regions where the
continuum is modelled, and attaches a functional form to each region. The continuum is
evaluated at pixel centres (not integrated), since it varies slowly enough that sub-pixel
variation is negligible.

---

## Creating a Continuum Configuration

### Automatic from Line Centres

The easiest approach — pads around each set of line centres, merges overlapping regions,
and assigns the same functional form to every region:

```python
from unite.continuum import ContinuumConfiguration, Linear
from astropy import units as u

cc = ContinuumConfiguration.from_lines(
    [6563.0, 6585.0, 6549.0] * u.AA,   # or lc.centers
    pad=0.05,       # 5% fractional padding on each side
    form=Linear(),  # functional form for all regions
)
```

### Manual Regions

For full control, define regions explicitly:

```python
from unite.continuum import ContinuumConfiguration, ContinuumRegion, Linear, PowerLaw

regions = [
    ContinuumRegion(6400 * u.AA, 6700 * u.AA, form=Linear()),
    ContinuumRegion(4800 * u.AA, 5100 * u.AA, form=PowerLaw()),
]
cc = ContinuumConfiguration(regions)
```

You can also pass the form as a **string** instead of an instance — it will be resolved
from the built-in registry:

```python
regions = [
    ContinuumRegion(6400 * u.AA, 6700 * u.AA, form='Linear'),
    ContinuumRegion(4800 * u.AA, 5100 * u.AA, form='PowerLaw'),
]
```

For programmatic access to the registry, use {func}`~unite.continuum.get_form`:

```python
from unite.continuum import get_form

form = get_form('Polynomial', degree=3)
```

---

## Continuum Forms

Each region has a functional form that defines how the continuum varies with wavelength. All
forms share a unified parameter interface: a `scale` parameter (the continuum flux at the
normalisation wavelength) plus form-specific shape parameters.

Forms have two kinds of configuration:

- **Constructor parameters** — set when you create the form (e.g., polynomial degree, knot
  vector). These are static and define the shape of the functional form.
- **Model parameters** — sampled during inference (e.g., `scale`, `slope`, `temperature`).
  These receive default priors that can be overridden via
  `ContinuumRegion(params={...})` — see [Custom Priors](#custom-priors-on-continuum-parameters)
  below.

### Linear

$f(\lambda) = \text{scale} + \text{slope} \times (\lambda - \lambda_\text{norm})$

```python
from unite.continuum import Linear
form = Linear()
```

Model parameters: `scale`, `slope`, `normalization_wavelength`.

Most common choice — sufficient when the continuum varies slowly across a narrow wavelength
range.

### PowerLaw

$f(\lambda) = \text{scale} \times (\lambda / \lambda_\text{norm})^\beta$

```python
from unite.continuum import PowerLaw
form = PowerLaw()
```

Model parameters: `scale`, `beta`, `normalization_wavelength`.

Good for UV/optical AGN continuum.

### Polynomial

$f(\lambda) = \text{scale} + c_1 x + c_2 x^2 + \ldots$ where $x = \lambda - \lambda_\text{norm}$

```python
from unite.continuum import Polynomial
form = Polynomial(degree=2)   # quadratic
```

Constructor parameter: `degree` (default 1).

Model parameters: `scale`, `c1`, `c2`, ..., `normalization_wavelength`.

### Chebyshev

Chebyshev polynomial expansion. More numerically stable than standard polynomials for wide
wavelength ranges. The x-coordinate is normalised to $[-1, 1]$ using the `half_width`.

```python
from unite.continuum import Chebyshev
form = Chebyshev(order=3, half_width=150.0)  # half_width in same units as region bounds
```

Constructor parameters: `order` (default 2), `half_width` (default 1.0, set to
`(high - low) / 2` of the region for proper orthogonality).

Model parameters: `scale`, `c1`, `c2`, ..., `normalization_wavelength`.

### BSpline

B-spline continuum with a user-defined knot vector. Provides local control through knot
placement.

```python
import jax.numpy as jnp
from unite.continuum import BSpline

# Clamped cubic knots (repeat end knots degree+1 times)
knots = jnp.array([4800, 4800, 4800, 4800, 4900, 5000, 5100, 5100, 5100, 5100])
form = BSpline(knots=knots, degree=3)
```

Constructor parameters: `knots` (clamped knot vector), `degree` (default 3 for cubic).

Model parameters: `scale`, `coeff_1`, `coeff_2`, ..., `normalization_wavelength`.

### Bernstein

Bernstein polynomial basis. **Guaranteed positive** when all coefficients are positive.

```python
from unite.continuum import Bernstein
form = Bernstein(degree=4, wavelength_min=4800, wavelength_max=5100)
```

Constructor parameters: `degree` (default 4), `wavelength_min`, `wavelength_max` (define
the normalisation range $[0, 1]$, in the same units as the region bounds).

Model parameters: `scale`, `coeff_1`, `coeff_2`, ..., `normalization_wavelength`.

### Blackbody

Planck function normalised at a reference wavelength:
$f(\lambda) = \text{scale} \times B_\lambda(T) / B_\lambda(\lambda_\text{norm}, T)$

```python
from unite.continuum import Blackbody
form = Blackbody()
```

Model parameters: `scale`, `temperature`, `normalization_wavelength`.

When fitting a single blackbody across disjoint spectral windows, share a
{class}`~unite.continuum.ContinuumNormalizationWavelength` token to enforce a consistent
reference wavelength — see [Sharing Parameters](#sharing-parameters-across-regions).

### ModifiedBlackbody

Blackbody with a power-law modifier:
$f(\lambda) = \text{scale} \times B_\lambda(T) / B_\lambda(\lambda_\text{norm}, T) \times (\lambda / \lambda_\text{norm})^\beta$

```python
from unite.continuum import ModifiedBlackbody
form = ModifiedBlackbody()
```

Model parameters: `scale`, `temperature`, `beta`, `normalization_wavelength`.

Setting $\beta = 0$ recovers a pure {class}`~unite.continuum.Blackbody`.

### AttenuatedBlackbody

Dust-attenuated blackbody with a power-law extinction curve:
$f(\lambda) = \text{scale} \times B_\lambda(T) / B_\lambda(\lambda_\text{norm}, T) \times \exp\!\bigl(-\tau_V [(\lambda/\lambda_V)^\alpha - (\lambda_\text{norm}/\lambda_V)^\alpha]\bigr)$

Extinction is normalised at `normalization_wavelength`, so `scale` is the **observed**
(attenuated) flux there.

```python
from unite.continuum import AttenuatedBlackbody
from astropy import units as u

form = AttenuatedBlackbody()                         # default: lambda_V = 0.55 μm (V band)
form = AttenuatedBlackbody(lambda_v_micron=0.44)     # B band
form = AttenuatedBlackbody(lambda_v_micron=4400 * u.AA)  # same, with units
```

Constructor parameter: `lambda_v_micron` — extinction reference wavelength (default 0.55 μm).
Accepts a bare float (interpreted as microns) or an astropy Quantity with any length unit.

Model parameters: `scale`, `temperature`, `tau_v`, `alpha`, `normalization_wavelength`.

---

## Continuum Parameters and Priors

Each continuum form generates model parameters with default priors. When a region is added
to a {class}`~unite.continuum.ContinuumConfiguration`, any parameter not explicitly provided
receives an auto-created token with the form's default prior.

The **normalisation wavelength** for each region defaults to the region midpoint (as a
{class}`~unite.prior.Fixed` prior).

### Custom Priors on Continuum Parameters

To override default priors, pass a {class}`~unite.prior.Parameter` token in the `params`
dict of {class}`~unite.continuum.ContinuumRegion`. Only the parameters you want to customise
need to be specified — the rest keep their defaults.

```python
from unite.continuum import ContinuumConfiguration, ContinuumRegion, PowerLaw
from unite.prior import Parameter, TruncatedNormal, Uniform, Fixed
from astropy import units as u

region = ContinuumRegion(
    1.0 * u.um, 1.5 * u.um,
    form=PowerLaw(),
    params={
        'scale': Parameter('my_scale', prior=TruncatedNormal(2.0, 0.5, 0.0, 10.0)),
        'beta': Parameter('my_beta', prior=Uniform(-3.0, 0.0)),
        # 'normalization_wavelength' keeps its default Fixed(region_center) prior
    },
)
cc = ContinuumConfiguration([region])
```

Invalid parameter names are caught at configuration time:

```python
# Raises ValueError — Linear has no 'amplitude' parameter
ContinuumRegion(
    1.0 * u.um, 2.0 * u.um,
    form=Linear(),
    params={'amplitude': Parameter('amp', prior=Uniform(0, 10))},
)
```

### Sharing Parameters Across Regions

Pass the **same** {class}`~unite.prior.Parameter` instance in the `params` dict of multiple
regions. The {class}`~unite.continuum.ContinuumConfiguration` detects shared identity and
creates a single numpyro site, coupling those regions to one sampled value.

This is essential for fitting a global continuum model (e.g. a single power law) across
disjoint spectral windows:

```python
from unite.continuum import (
    ContinuumConfiguration, ContinuumRegion, ContinuumNormalizationWavelength,
    PowerLaw,
)
from unite.prior import Parameter, Uniform, Fixed
from astropy import units as u

pl = PowerLaw()
shared_scale = Parameter('pl_scale', prior=Uniform(0, 10))
shared_beta  = Parameter('pl_beta',  prior=Uniform(-5, 5))
shared_nw    = ContinuumNormalizationWavelength('pl_nw', prior=Fixed(1.25))

cc = ContinuumConfiguration([
    ContinuumRegion(
        0.9 * u.um, 1.4 * u.um, form=pl,
        params={'scale': shared_scale, 'beta': shared_beta,
                'normalization_wavelength': shared_nw},
    ),
    ContinuumRegion(
        1.7 * u.um, 2.5 * u.um, form=pl,
        params={'scale': shared_scale, 'beta': shared_beta,
                'normalization_wavelength': shared_nw},
    ),
])
# → 3 model parameters: pl_scale, pl_beta, pl_nw
```

### Typed Tokens

Two specialised token classes add slot validation:

- {class}`~unite.continuum.ContinuumScale` — can only be placed in the `'scale'` slot.
- {class}`~unite.continuum.ContinuumNormalizationWavelength` — can only be placed in the
  `'normalization_wavelength'` slot.

These prevent accidental mis-assignment. Using a generic {class}`~unite.prior.Parameter` in
any slot is always allowed.

---

## Quick Reference: Form Parameters

| Form | Constructor args | Model parameters |
|------|-----------------|-----------------|
| `Linear()` | — | `scale`, `slope`, `normalization_wavelength` |
| `PowerLaw()` | — | `scale`, `beta`, `normalization_wavelength` |
| `Polynomial(degree=N)` | `degree` | `scale`, `c1`…`cN`, `normalization_wavelength` |
| `Chebyshev(order=N, half_width=H)` | `order`, `half_width` | `scale`, `c1`…`cN`, `normalization_wavelength` |
| `BSpline(knots=K, degree=D)` | `knots`, `degree` | `scale`, `coeff_1`…, `normalization_wavelength` |
| `Bernstein(degree=N, wavelength_min, wavelength_max)` | `degree`, `wavelength_min`, `wavelength_max` | `scale`, `coeff_1`…`coeff_N`, `normalization_wavelength` |
| `Blackbody()` | — | `scale`, `temperature`, `normalization_wavelength` |
| `ModifiedBlackbody()` | — | `scale`, `temperature`, `beta`, `normalization_wavelength` |
| `AttenuatedBlackbody(lambda_v_micron=0.55)` | `lambda_v_micron` | `scale`, `temperature`, `tau_v`, `alpha`, `normalization_wavelength` |

---

## Serialization

{class}`~unite.continuum.ContinuumConfiguration` supports standalone YAML serialization:

```python
cc.save('continuum.yaml')
cc2 = ContinuumConfiguration.load('continuum.yaml')
```

See {doc}`serialization` for the full workflow and YAML format examples.
