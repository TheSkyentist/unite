# Advanced Topics

## Shared Kinematics

The most powerful feature of `unite` is the ability to share kinematic parameters across
lines, components, and even spectra. Sharing is controlled by **token identity**: two lines
that receive the *same Python object* as a `redshift` or `fwhm_gauss` argument will have a
single shared latent variable in the NumPyro model.

### Sharing redshift across a multiplet

```python
z    = line.Redshift('z',    prior=prior.Uniform(-0.01, 0.01))
fwhm = line.FWHM('fwhm',    prior=prior.Uniform(50, 500))

lc = line.LineConfiguration()
lc.add_line('H_alpha',  6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm, ...)
lc.add_line('NII_6585', 6585.0 * u.AA, redshift=z, fwhm_gauss=fwhm, ...)
lc.add_line('NII_6549', 6549.0 * u.AA, redshift=z, fwhm_gauss=fwhm, ...)
```

→ One `z` and one `fwhm` parameter in the model, shared by all three lines.

### Two kinematic components

```python
z = line.Redshift('z', prior=prior.Uniform(-0.01, 0.01))

fwhm_n = line.FWHM('fwhm_narrow', prior=prior.Uniform(50, 400))
fwhm_b = line.FWHM('fwhm_broad',  prior=prior.Uniform(500, 3000))

lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm_n,
            flux=line.Flux('Ha_n_flux', prior=prior.Uniform(0, 10)))
lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm_b,
            flux=line.Flux('Ha_b_flux', prior=prior.Uniform(0, 10)))
```

→ Shared `z`, but independent `fwhm_narrow` and `fwhm_broad`. Both H$\alpha$ components
appear in the model and their fluxes are summed.

---

## Dependent Priors

Prior bounds can reference other parameters using arithmetic expressions on token objects.
This creates a **dependency** between parameters: one must be sampled before the other.

### Minimum separation constraint

```python
fwhm_narrow = line.FWHM('fwhm_narrow', prior=prior.Uniform(50, 500))

# Broad component must be at least 150 km/s wider than the narrow component
fwhm_broad  = line.FWHM('fwhm_broad',  prior=prior.Uniform(fwhm_narrow + 150, 3000))
```

The expression `fwhm_narrow + 150` creates a {class}`~unite.prior.ParameterRef` object.
At model-build time, {func}`~unite.prior.topological_sort` orders the parameters so that
`fwhm_narrow` is sampled first, and the lower bound of `fwhm_broad`'s prior is then
evaluated at the sampled value.

### Supported arithmetic

| Expression | Effect |
|------------|--------|
| `token + constant` | Lower/upper bound = sampled value + constant |
| `token - constant` | Lower/upper bound = sampled value − constant |
| `token * constant` | Lower/upper bound = sampled value × constant |
| `token / constant` | Lower/upper bound = sampled value ÷ constant |

Expressions can appear as either the `low` or `high` bound of {class}`~unite.prior.Uniform`
or {class}`~unite.prior.TruncatedNormal`.

:::{note}
Only **linear** expressions (scale × token + offset) are supported. Combinations of two
tokens in a single expression are not.
:::

---

## Multiple Spectra

`unite` can fit any number of spectra simultaneously. The model evaluates each spectrum
independently but shares parameters according to token identity.

```python
from unite.spectrum import Spectra

spectra = Spectra([prism_spectrum, g395m_spectrum], redshift=5.28)
```

`prepare()` filters lines and continuum regions for each spectrum independently based on
wavelength coverage. A line that falls in the wavelength range of one grating but not another
will only be modelled in the grating where it is covered.

---

## Calibration Tokens

Calibration tokens attach free (or fixed) parameters to a disperser. They are useful for
absorbing flux calibration or resolution calibration uncertainty between gratings.

### FluxScale — relative flux calibration

```python
from unite.disperser import FluxScale
from unite.disperser.nirspec import G395M, PRISM
from unite import prior

# Free multiplicative flux scale on G395M; PRISM is the reference (no token)
g395m = G395M(r_source='point', flux_scale=FluxScale(prior.Uniform(0.5, 2.0)))
prism = PRISM(r_source='point')
```

The `FluxScale` multiplies the model flux for all pixels in that spectrum before comparing
to the data. A value of 1.0 means no correction; values above/below 1 rescale the model up
or down.

### RScale — resolution calibration

```python
from unite.disperser import RScale

g395m = G395M(r_source='point', r_scale=RScale(prior.Uniform(0.8, 1.2)))
```

`RScale` multiplies the nominal resolution $R(\lambda)$ by a free factor. This adjusts the
LSF width at every pixel. Useful when the actual LSF is broader or narrower than the
calibration data predicts (e.g., for extended sources in NIRSpec).

### PixOffset — wavelength offset

```python
from unite.disperser import PixOffset

g395m = G395M(r_source='point', pix_offset=PixOffset(prior.Uniform(-2, 2)))
```

`PixOffset` shifts the disperser's wavelength solution by a fixed number of pixels. Useful
for correcting systematic wavelength offsets.

### Sharing calibration tokens

If two dispersers should share a calibration parameter (e.g., a common flux scale applied
to all NIRSpec spectra), pass the same token instance to both:

```python
shared_scale = FluxScale(prior.Uniform(0.5, 2.0))
g235m = G235M(r_source='point', flux_scale=shared_scale)
g395m = G395M(r_source='point', flux_scale=shared_scale)
```

### Fixed calibration

To fix a calibration parameter (e.g., apply a known correction without uncertainty), use
{class}`~unite.prior.Fixed`:

```python
g395m = G395M(r_source='point', flux_scale=FluxScale(prior.Fixed(1.15)))
```

---

## DispersersConfiguration

When using calibration tokens, wrap your dispersers in a
{class}`~unite.disperser.DispersersConfiguration` before building the
{class}`~unite.config.Configuration`:

```python
from unite.disperser import DispersersConfiguration

dc = DispersersConfiguration(dispersers=[prism, g395m])
config = Configuration(lines=lc, continuum=cc, dispersers=dc)
```

`DispersersConfiguration.validate()` is called automatically on construction and will warn
if any calibration axis (flux, resolution, pixel offset) has no fixed anchor — meaning all
dispersers in that axis have a free scale, making the axis unidentifiable.

---

## Topological Sorting

When parameters have dependent priors, `ModelBuilder` uses
{func}`~unite.prior.topological_sort` to order the sampling so that a parameter is always
sampled *before* any other parameter whose prior depends on it.

You can inspect the sampling order by looking at `model_args.dependency_order` after
calling `build()`:

```python
builder = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
model_fn, model_args = builder.build()

print(model_args.dependency_order)
# e.g. ['z', 'fwhm_narrow', 'fwhm_broad', 'Ha_n_flux', 'Ha_b_flux', ...]
```

Parameters are listed in the order in which NumPyro will sample them. Dependent parameters
always appear after the parameters they depend on.
