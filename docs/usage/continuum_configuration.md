# Continuum Configuration

{class}`~unite.continuum.ContinuumConfiguration` defines wavelength regions where the
continuum is modeled, and attaches a functional form to each region. The continuum is
evaluated at pixel centers (not integrated), since it varies slowly enough that sub-pixel
variation is negligible. Continua are also not convolved with the LSF.

---

## Creating a Continuum Configuration

### Automatic from Line Centers

The easiest approach is the `ContinuumConfiguration.from_lines` class method, 
which automatically creates regions around each set of line centers, 
merges overlapping regions, and assigns the same functional form to every region:

```python
from unite.continuum import ContinuumConfiguration, Linear
from astropy import units as u

lc = LineConfiguration(...)
cc = ContinuumConfiguration.from_lines(
    lc.centers, # Or any array of wavelenths definign cents.
    width=30_000 * u.km / u.s,  # Total width of each region in velocity units, default is 30,000 km/s
    form=Linear(),  # Functional for in each region, default is Linear which takes no arguments
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

Regions can be given an optional **name** which is used as the suffix for auto-created
parameter tokens (e.g. `scale_blue`, `beta_blue`). Region names must be unique within
a configuration.

```python
regions = [
    ContinuumRegion(6400 * u.AA, 6700 * u.AA, form=Linear(), name='red'),
    ContinuumRegion(4800 * u.AA, 5100 * u.AA, form=PowerLaw(), name='blue'),
]
# → auto-created params: 'scale_red', 'angle_red', 'scale_blue', 'beta_blue', …
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

### Combining Continuum Configurations

Two Continuum Configurations can be combined with the `+` operator. This is useful for building up a configuration iteratively, or for merging configurations defined in separate YAML files.

```python
cc1 = ContinuumConfiguration.load('continuum1.yaml')
cc2 = ContinuumConfiguration.load('continuum2.yaml')
cc_combined = cc1 + cc2
```

---

## Continuum Parameters and Priors

Each continuum region must be specified with the low and high 
wavelength bounds (as {class}`astropy.units.Quantity`), 
and a functional form. 

Each continuum form generates model parameters with default priors. When a region is added
to a {class}`~unite.continuum.ContinuumConfiguration`, any parameter not explicitly provided
receives an auto-created token with the form's default prior.

As with lines parameters, a shared token is the same parameter in the model.
These are the following token types:

| Token | Role | Unit |
|-------|------|------|
| {class}`~unite.continuum.NormWavelength` | Rest-wavelength at which continuum is scaled | same unit as low bound |
| {class}`~unite.continuum.Scale` | Continuum height at normalization wavelength | internal units |
| {class}`~unite.continuum.ContShape` | Arbitrary profile parameter (β, angle, etc.) | per-parameter basis |

The **normalization wavelength** for each region defaults to the region midpoint (as a
{class}`~unite.prior.Fixed` prior).

:::{note}
Scales are relative to a scale computed based on the spectrum. See Flux and Error Scales in {doc}`build_model` for details.
:::

### Custom Priors on Continuum Parameters

Similar to line parameters, you can override the default priors by passing a dict of continuum tokens.

```python
from unite import continuum
from unite.prior import TruncatedNormal, Uniform, Fixed
from astropy import units as u

region = continuum.ContinuumRegion(
    1.0 * u.um, 1.5 * u.um,
    form=continuum.PowerLaw(),
    params={
        'scale': continuum.Scale('pl', prior=TruncatedNormal(low = 0, high = 2.0, loc=1, scale=0.1)),
        'beta': continuum.ContShape('pl', prior=Uniform(-3.0, 0.0)),
        # 'norm_wav' keeps its default Fixed(region_center) prior
    },
)
cc = continuum.ContinuumConfiguration([region])
```

Invalid parameter names are caught when assembling the configuration:

```python
region = continuum.ContinuumRegion(
    1.0 * u.um, 2.0 * u.um,
    form=continuum.Linear(),
    params={'amplitude': continuum.Scale('amp', prior=Uniform(0, 10))},
)
cc = continuum.ContinuumConfiguration([region])
# Raises ValueError — Linear has no 'amplitude' parameter
```

### Parameter Naming

Each continuum parameter token accepts a semantic label as its first argument. A **type-specific prefix is automatically prepended** to form the final site name:

- `Scale('pl')` → site name `'scale_pl'` (prefix: `scale_`)
- `ContShape('pl')` → site name `'beta_pl'` (prefix: `beta_` for PowerLaw, `angle_` for Linear, etc.)
- `NormWavelength('pl')` → site name `'norm_wav_pl'` (prefix: `norm_wav_`)

Pass **only the semantic label**, not the full prefixed name. Good examples: `'pl'` (power law), `'poly'` (polynomial), `'red'` (red region), `'uv'` (UV region).

Alternatively, **use region names for automatic naming**. If you don't provide explicit tokens, the region's `name` parameter is used as a suffix:

```python
regions = [
    ContinuumRegion(0.9 * u.um, 1.4 * u.um, form=PowerLaw(), name='red'),
    # Auto-creates tokens: scale_red, beta_red, norm_wav_red
]
```

### Sharing Parameters Across Regions

Pass the **same** {class}`~unite.prior.Parameter` instance in the `params` dict of multiple
regions. The {class}`~unite.continuum.ContinuumConfiguration` detects shared identity and
creates a single numpyro site, coupling those regions to one sampled value.

This is essential for fitting a global continuum model (e.g. a single power law) across
disjoint spectral windows:

```python
from unite import continuum
from unite.prior import Uniform, Fixed
from astropy import units as u

pl = continuum.PowerLaw()
shared_scale = continuum.Scale('pl', prior=Uniform(0, 10))
shared_beta  = continuum.ContShape('pl', prior=Uniform(-5, 5))
shared_nw    = continuum.NormWavelength('pl', prior=Fixed(1.25))

cc = continuum.ContinuumConfiguration([
    continuum.ContinuumRegion(
        0.9 * u.um, 1.4 * u.um, form=pl,
        params={
            'scale': shared_scale,
            'beta': shared_beta,
            'norm_wav': shared_nw
        },
    ),
    continuum.ContinuumRegion(
        1.7 * u.um, 2.5 * u.um, form=pl,
        params={
            'scale': shared_scale,
            'beta': shared_beta,
            'norm_wav': shared_nw
        },
    ),
])
```

---

## Continuum Forms

Each region has a functional form that defines how the continuum varies with wavelength. All
forms share a unified parameter interface: a `norm_wav` parameter (the wavelength at which
the continuum is normalized), a `scale` parameter (the normalization of the continuum), and 
any additional form-specific shape parameters.

Forms have two kinds of configuration:

- **Constructor parameters** — set when you create the form (e.g., polynomial degree, knot
  vector). These are static and define the shape of the functional form.
- **Model parameters** — sampled during inference (e.g., `scale`, `angle`, `temperature`).
  These receive default priors that can be overridden via
  `ContinuumRegion(params={...})`

Here are the built-in forms, their constructor parameters, and their model parameters.
The **LSF** column indicates whether analytic LSF convolution is applied when the
model is evaluated with the instrument's line-spread function:

| Form | Constructor args | Additional Model parameters | LSF |
|------|------------------|-----------------------------|-----|
| `Linear` | — | `angle` | Yes (exact) |
| `Polynomial` | `degree` | `c1`…`cN` | Yes (exact) |
| `Chebyshev` | `order`, `stretch` | `c1`…`cN` | Yes (exact) |
| `Bernstein` | `degree`, `stretch` | `c1`…`cN` | Yes (exact) |
| `PowerLaw` | — | `beta` | No |
| `BSpline` | `knots`, `degree` | `c1`… | No |
| `Blackbody` | — | `temperature` | No |
| `ModifiedBlackbody` | — | `temperature`, `beta` | No |
| `AttenuatedBlackbody` | `lambda_ext` | `temperature`, `tau_ext`, `alpha` | No |

Forms marked **Yes (exact)** are polynomial-based and have their coefficients
analytically convolved with the Gaussian LSF before evaluation.  Forms marked
**No** ignore the LSF — their curvature is assumed to vary slowly enough that
the unconvolved value is a good approximation at the spectral resolution of
most instruments.  For `BSpline`, the non-polynomial basis makes analytic
convolution impractical; for `PowerLaw` and the blackbody family, the nonlinear
functional forms do not admit closed-form convolution.


### Linear

$f(\lambda) = \text{scale} + \tan(\theta) \times (\lambda - \lambda_\text{norm})$

```python
from unite.continuum import Linear
form = Linear()
```

Model parameters: `scale`, `angle`.

### PowerLaw

$f(\lambda) = \text{scale} \times (\lambda / \lambda_\text{norm})^\beta$

```python
from unite.continuum import PowerLaw
form = PowerLaw()
```

Model parameters: `scale`, `beta`.

### Polynomial

$f(\lambda) = \text{scale} + c_1 x + c_2 x^2 + \ldots$ where $x = \lambda - \lambda_\text{norm}$

```python
from unite.continuum import Polynomial
form = Polynomial(degree=2)   # quadratic
```

Constructor parameter: `degree` (default 1).

Model parameters: `scale`, `c1`, `c2`, .... `cn` where `n` is the degree.

### Chebyshev

Chebyshev (first-kind) polynomial expansion. Guarantees orthogonality and better numerical stability.
The x-coordinate is normalized to $[-1, 1]$ and then scaled by `stretch`.

:::{note}
Exercise extreme caution when using non-unity `stretch`, especially less than one.
This can lead to large instability outside the nominal range.
:::

```python
from unite.continuum import Chebyshev
form = Chebyshev(order=3, stretch=1)  # half_width in same units as region bounds
```

Constructor parameters: `order` (default 2), `stretch` (default 1.0)

Model parameters: `scale`, `c1`, `c2`, .... `cn` where `n` is the degree.

### Clamped BSpline

B-spline continuum with a user-defined knot vector. Spline is automatically clamped at the region edges. 
Therefore all knots should be within the region bounds.

```python
import jax.numpy as jnp
from unite.continuum import BSpline

# Knots (ends are automatically clamped)
knots = jnp.array([4050, 5000] * u.AA)
form = BSpline(knots=knots, degree=3)
```

Constructor parameters: `knots` (knot vector), `degree` (default 3 for cubic).

Model parameters: `scale`, `c1`, `c2`, .... `cn` where `n` is the degree.

### Bernstein

Bernstein polynomial basis. 
The x-coordinate is normalised to $[-1, 1]$, scaled by `stretch`, then shifted to $[0, 1]$ assuming it was still $[-1, 1]$ after stretching. **Guaranteed positive** when all coefficients are positive.

```python
from unite.continuum import Bernstein
form = Bernstein(degree=4, stretch=1)
```

Constructor parameters: `degree` (default 4), `stretch` (default 1.0, same caveats as Chebyshev).

Model parameters: `scale`, `c1`, `c2`, .... `cn` where `n` is the degree.

### Blackbody

Planck function normalised at a reference wavelength:
$f(\lambda) = \text{scale} \times B_\lambda(T) / B_{\lambda_\text{norm}}, T)$

```python
from unite.continuum import Blackbody
form = Blackbody()
```

Model parameters: `scale`, `temperature`.

When fitting a single blackbody across disjoint spectral windows, share a
{class}`~unite.continuum.ContinuumNormalizationWavelength` token to enforce a consistent
reference wavelength — see [Sharing Parameters](#sharing-parameters-across-regions).

### ModifiedBlackbody

Blackbody with a power-law modifier:
$f(\lambda) = \text{scale} \times \times B_\lambda(T) / B_{\lambda_\text{norm}}, T) \times (\lambda / \lambda_\text{norm})^\beta$

```python
from unite.continuum import ModifiedBlackbody
form = ModifiedBlackbody()
```

Model parameters: `scale`, `temperature`, `beta`.

Setting $\beta = 0$ recovers a pure {class}`~unite.continuum.Blackbody`.

### AttenuatedBlackbody

Dust-attenuated blackbody with a power-law extinction curve:  
$f(\lambda) = \text{scale} \times B_\lambda(T) / B_{\lambda_\text{norm}}, T) \times \exp\!\bigl(-\tau_{\text{ext}} [(\lambda/\lambda_{\text{ext}})^\alpha - (\lambda_\text{norm}/\lambda_{\text{ext}})^\alpha]\bigr)$

Extinction is normalised at `norm_wav`, so `scale` is the **observed**
(attenuated) flux there.

```python
from unite.continuum import AttenuatedBlackbody
from astropy import units as u

form = AttenuatedBlackbody()                         # default: lambda_V = 0.55 μm (V band)
form = AttenuatedBlackbody(lambda_ext=4400 * u.AA)  # B Band
```

Constructor parameter: `lambda_ext` — extinction reference wavelength (default 0.55 μm).
Accepts a bare float (interpreted as microns) or an {class}`astropy.units.Quantity` with any length unit.

Model parameters: `scale`, `temperature`, `tau_ext`, `alpha`.

---

## Serialization

{class}`~unite.continuum.ContinuumConfiguration` supports standalone YAML serialization:

```python
cc.save('continuum.yaml')
cc2 = ContinuumConfiguration.load('continuum.yaml')
```

See {doc}`serialization` for the full workflow and YAML format examples.
