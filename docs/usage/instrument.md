# Instruments & Spectrum Loading

`unite` splits instrumental configuration and spectral data into two modules:

- **`unite.instrument`** — disperser classes and calibration tokens (hardware configuration)
- **`unite.spectrum`** — spectrum containers and loader functions (data I/O)

This separation means you can configure and serialize dispersers independently of any particular
dataset, then load spectra against those configurations.

---

## Dispersers

A disperser describes an instrument's spectral characteristics:

- **R(λ)** — the resolving power as a function of wavelength, used to compute the LSF FWHM
  at each pixel.
- **(dλ/dpix)(λ)** — the wavelength dispersion per pixel as a function of wavelength, used to
  convert between pixel offsets and wavelength offsets.

### Calibration Tokens

For a given instrument the user can optionally attach calibration tokens with associated priors:

- {class}`~unite.instrument.RScale`: Resolution scaling factor (e.g. R_eff = R_nominal × RScale)
- {class}`~unite.instrument.FluxScale`: Flux scaling factor (e.g. F_eff = F_model × FluxScale)
- {class}`~unite.instrument.PixOffset`: Linear pixel offset (e.g. λ_eff = λ_model + (dλ/dpix)(λ_model) × PixOffset)

These absorb calibration offsets and uncertainties between instruments.

```python
from unite.instrument import RScale, FluxScale, PixOffset
```

In practice `FluxScale` and `PixOffset` are most useful for **relative** calibration between
spectra — one spectrum is treated as the reference with fixed calibration, and the other spectra
have free parameters to account for differences in flux calibration and wavelength solution. If a
`FluxScale` is applied to all spectra it is completely degenerate with the overall flux
normalization of the model.

#### RScale — Resolution Calibration

Multiplies the nominal $R(\lambda)$ by a free factor:
$R_\mathrm{eff}(\lambda) = R_\mathrm{nominal}(\lambda) \times \text{r\_scale}$

```python
r = RScale(prior=prior.TruncatedNormal(1.0, 0.05, 0.8, 1.2))
```

Useful when the actual LSF is broader or narrower than the calibration predicts — e.g., for
extended sources in NIRSpec, or when the slit filling fraction differs from the calibration
assumption.

#### FluxScale — Relative Flux Calibration

Multiplies the model flux for all pixels in a spectrum by a free factor. A value of 1.0 means no
correction.

```python
f = FluxScale(prior=prior.Uniform(0.5, 2.0))
```

#### PixOffset — Wavelength Offset

Shifts the wavelength solution by a number of pixels. A value of 0.0 means no correction.

```python
p = PixOffset(prior=prior.Uniform(-2, 2))
```

#### Parameter Naming

Each calibration token accepts an optional semantic label as its first argument. A
**type-specific prefix is automatically prepended** to form the final site name:

- `RScale('shared')` → site name `'r_scale_shared'` (prefix: `r_scale_`)
- `FluxScale('shared')` → site name `'flux_scale_shared'` (prefix: `flux_scale_`)
- `PixOffset('shared')` → site name `'pix_offset_shared'` (prefix: `pix_offset_`)

Pass **only the semantic label**, not the full prefixed name. For unshared tokens on a single named
disperser, the disperser name is used automatically (e.g., `RScale()` on `'G235H'` →
`'r_scale_G235H'`).

#### Sharing Calibration Tokens

Pass the **same token instance** to multiple dispersers to share a calibration parameter:

```python
shared_r = RScale(prior=prior.TruncatedNormal(1.0, 0.05, 0.8, 1.2), name='shared')
g235h = nirspec.G235H(r_scale=shared_r)
g395h = nirspec.G395H(r_scale=shared_r)
# → Both dispersers use the same 'r_scale_shared' parameter in the model
```

#### Fixed Calibration

Use {class}`~unite.prior.Fixed` to apply a known correction without uncertainty. This is the
default for all calibration tokens — if you don't specify a token the disperser is treated as the
reference with no correction.

```python
g395h = nirspec.G395H(flux_scale=FluxScale(prior=prior.Fixed(1.15)))
```

---

### JWST/NIRSpec

Convenience classes are provided for each NIRSpec disperser:

```python
from unite.instrument import nirspec

# nirspec.G140H, nirspec.G140M, nirspec.G235H, nirspec.G235M,
# nirspec.G395H, nirspec.G395M, nirspec.PRISM
```

NIRSpec dispersers support two resolution calibration modes via `r_source`:

```python
g235h = nirspec.G235H(r_source='point')     # point-source R (de Graaff et al. 2024)
g235h = nirspec.G235H(r_source='uniform')   # uniform illumination R from FITS tables
```

- **`'point'`** (default): Uses polynomial fits to the slit-centered point-source resolving power
  derived from
  [de Graaff et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024A%26A...684A..87D/abstract).
  Recommended for most sources. Even resolved sources can use this curve combined with an `RScale`
  token to account for the effective slit filling fraction or position.
- **`'uniform'`**: Uses tabulated resolving power for uniform illumination of the slit from
  [JDOX](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters#gsc.tab=0).

Here is an example applying calibration tokens to NIRSpec dispersers, using realistic offsets
between G395M and PRISM based on the RUBIES survey
([de Graaff et al. 2025](https://ui.adsabs.harvard.edu/abs/2025A%26A...697A.189D/abstract)):

```python
from unite.instrument import nirspec, RScale, FluxScale, PixOffset
from unite import prior

# Free resolution scale shared across both dispersers
r_scale = RScale(prior=prior.TruncatedNormal(low=0.7, high=1.3, loc=1.0, scale=0.3))

# G395M is the reference — no FluxScale or PixOffset
g395m = nirspec.G395M(r_scale=r_scale)

# PRISM has free flux and wavelength offsets relative to G395M
prism = nirspec.PRISM(
    r_scale=r_scale,
    flux_scale=FluxScale(prior=prior.TruncatedNormal(low=0.8, high=1.4, loc=1.1, scale=0.1)),  # de Graaff+25 Fig 8
    pix_offset=PixOffset(prior=prior.TruncatedNormal(low=-0.2, high=0.6, loc=0.2, scale=0.1)),  # de Graaff+25 Fig 9
)
```

---

### SDSS

The SDSS disperser computes $R(\lambda)$ from the `wdisp` column in the native SDSS pipeline
output. The disperser's resolution grid is populated at load time by {func}`~unite.spectrum.from_sdss_fits`.

```python
from unite.instrument import sdss

disperser = sdss.SDSSDisperser()
# Calibration tokens can be attached here as with any disperser
```

---

### Generic Dispersers

For instruments without built-in support, `unite` provides two generic disperser classes.
Import them explicitly from `unite.instrument.generic`.

#### SimpleDisperser

{class}`~unite.instrument.generic.SimpleDisperser` defines the pixel scale from the input
wavelength array (one wavelength element per pixel). The resolution can be a constant or an array.

```python
from unite.instrument import generic
import numpy as np
from astropy import units as u

wavelength = np.linspace(4000, 9000, 500) * u.AA
disperser = generic.SimpleDisperser(
    wavelength=wavelength,
    R=3000.0,
    name='my_spectrograph',
)
```

Three modes are supported for specifying the dispersion:

| Mode | Argument | Description |
|------|----------|-------------|
| Resolving power | `R=value` | $R = \lambda / \Delta\lambda$ (constant or array) |
| Wavelength dispersion | `dlam=value` | $\Delta\lambda$ per pixel (constant or array) |
| Velocity dispersion | `dvel=value` | $\Delta v$ per pixel in km/s (constant or array) |

#### GenericDisperser

For maximum flexibility, pass callables that return $R(\lambda)$ and $d\lambda/\mathrm{pix}(\lambda)$:

```python
from unite.instrument import generic

def my_R(wavelength):
    return 2000 + 500 * (wavelength - 4000) / 5000

def my_dlam(wavelength):
    return wavelength / my_R(wavelength)

disperser = generic.GenericDisperser(R_func=my_R, dlam_dpix_func=my_dlam, unit=u.AA, name='custom')
```

---

### InstrumentConfig & Serialization

{class}`~unite.instrument.config.InstrumentConfig` wraps a set of dispersers for serialization
and named lookup:

```python
from unite.instrument.config import InstrumentConfig

dc = InstrumentConfig([prism, g395m])
```

Dispersers can be retrieved by name:

```python
d = dc['G395M']      # returns the G395M disperser
d = dc['PRISM']      # returns the PRISM disperser
names = dc.names     # ['PRISM', 'G395M']
```

This is particularly useful after loading a configuration from YAML:

```python
dc = InstrumentConfig.load('dispersers.yaml')
g395m = dc['G395M']  # retrieve with calibration tokens intact
```

`validate()` is called automatically on construction and warns if any calibration axis has **no
fixed anchor** — meaning all dispersers have a free parameter on that axis, making it
unidentifiable.

Instrument configs can be combined with addition:

```python
dc1 = InstrumentConfig.load('dispersers1.yaml')
dc2 = InstrumentConfig.load('dispersers2.yaml')
dc_combined = dc1 + dc2
```

---

## Loading Spectra

All spectrum loaders live in `unite.spectrum` and return a {class}`~unite.spectrum.Spectrum`
object. Loaders always require a configured disperser.

### from_arrays

{func}`~unite.spectrum.from_arrays` is the generic loader — it works with any disperser and
accepts {class}`astropy.units.Quantity` arrays for the pixel bin edges, flux, and flux error.
Because `unite` performs exact pixel integration, bin edges (not centers) are required.

```python
from unite.spectrum import from_arrays
from astropy import units as u
import numpy as np

low  = wl_edges[:-1]   # lower pixel edges, Quantity
high = wl_edges[1:]    # upper pixel edges, Quantity
flux = flux_array * (1e-20 * u.erg / (u.s * u.cm**2 * u.AA))
err  = err_array  * (1e-20 * u.erg / (u.s * u.cm**2 * u.AA))

spectrum = from_arrays(low, high, flux, err, disperser=disperser)
```

This is the right loader for:
- Simulated spectra or pipeline-reduced data in custom formats
- Any instrument not covered by `from_DJA` or `from_sdss_fits`
- Quick testing with synthetic data

An optional `name` argument overrides the disperser name on the resulting spectrum.

---

### from_DJA

{func}`~unite.spectrum.from_DJA` loads a NIRSpec spectrum in the format of the
[DAWN JWST Archive](https://dawn-cph.github.io/dja/index.html). It accepts any configured
disperser and can load directly from a URL (with optional `astropy` caching).

```python
from unite.instrument import nirspec
from unite.spectrum import from_DJA

g235h = nirspec.G235H()

spectrum = from_DJA('spectrum_g235h.fits', disperser=g235h)
spectrum = from_DJA('https://url_to_dja_spectrum.fits', disperser=g235h, cache=True)
```

---

### from_sdss_fits

{func}`~unite.spectrum.from_sdss_fits` loads an SDSS spectrum from a native pipeline FITS file
(or URL). It **requires** an {class}`~unite.instrument.sdss.SDSSDisperser` — the disperser's
$R(\lambda)$ grid is populated from the `wdisp` column in the file.

```python
from unite.instrument import sdss
from unite.spectrum import from_sdss_fits

disperser = sdss.SDSSDisperser()
spectrum = from_sdss_fits('spec-PLATE-MJD-FIBER.fits', disperser=disperser)
```

---

## Mixing Instruments

`unite` can combine spectra from entirely different instruments in one fit. Each spectrum uses its
own disperser; shared line tokens ensure the same physical parameters are used across all spectra.

```python
from unite.instrument import nirspec, sdss, FluxScale
from unite import prior
from unite.spectrum import Spectra, from_DJA, from_sdss_fits

# NIRSpec grating — flux reference, no FluxScale
g395h = nirspec.G395H()

# SDSS disperser with a free flux scale relative to NIRSpec
sdss_disp = sdss.SDSSDisperser(
    flux_scale=FluxScale(prior=prior.Uniform(0.5, 2.0))
)

spec_nir  = from_DJA('g395h.fits', disperser=g395h)
spec_sdss = from_sdss_fits('spec-PLATE-MJD-FIBER.fits', disperser=sdss_disp)

spectra = Spectra([spec_nir, spec_sdss], redshift=z_sys)
```
