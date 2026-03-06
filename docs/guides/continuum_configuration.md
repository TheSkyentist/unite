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

---

## Continuum Forms

Each region has a functional form that defines how the continuum varies with wavelength. All
forms share a unified parameter interface: a `scale` parameter (the continuum flux at the
normalisation wavelength) plus form-specific shape parameters.

### Linear

$f(\lambda) = \mathrm{scale} \times (1 + \mathrm{slope} \times (\lambda - \lambda_\mathrm{norm}))$

```python
from unite.continuum import Linear
form = Linear()
```

Two parameters: `scale` and `slope`. Most common choice — sufficient when the continuum
varies slowly across a narrow wavelength range.

### PowerLaw

$f(\lambda) = \mathrm{scale} \times (\lambda / \lambda_\mathrm{norm})^\alpha$

```python
from unite.continuum import PowerLaw
form = PowerLaw()
```

Two parameters: `scale` and `alpha` (power-law index). Good for UV/optical AGN continuum.

### Polynomial

Standard polynomial expansion about the normalisation wavelength.

```python
from unite.continuum import Polynomial
form = Polynomial(degree=2)   # quadratic
```

Parameters: `scale` plus `degree` polynomial coefficients.

### Chebyshev

Chebyshev polynomial expansion. More numerically stable than standard polynomials for wide
wavelength ranges.

```python
from unite.continuum import Chebyshev
form = Chebyshev(degree=3)
```

### BSpline

B-spline continuum with a configurable number of knots.

```python
from unite.continuum import BSpline
form = BSpline(n_knots=5)
```

### Bernstein

Bernstein polynomial basis. Guaranteed positive when all coefficients are positive.

```python
from unite.continuum import Bernstein
form = Bernstein(degree=3)
```

### Blackbody

Planck function continuum. Single shape parameter: temperature.

```python
from unite.continuum import Blackbody
form = Blackbody()
```

---

## Continuum Parameters and Priors

Each continuum form generates parameters with default priors that are set automatically by
{class}`~unite.model.ModelBuilder` based on the data scaling from
{meth}`~unite.spectrum.Spectra.compute_scales`.

The **normalisation wavelength** for each region defaults to the region midpoint (as a
{class}`~unite.prior.Fixed` prior). You can share a normalisation wavelength across regions
or set it explicitly.

---

## Serialization

{class}`~unite.continuum.ContinuumConfiguration` supports standalone YAML serialization:

```python
cc.save('continuum.yaml')
cc2 = ContinuumConfiguration.load('continuum.yaml')
```

See {doc}`serialization` for the full workflow and YAML format examples.
