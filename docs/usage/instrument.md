# Instruments & Spectrum Loading

`unite` splits instrumental configuration and spectral loading. This enables the user to configure and save for easy reuse across datasets. In this section we describe the `unite.instrument` module, which defines dispersers and spectrum loading for supported instruments, as well as generic classes for custom instruments. We also introduce calibration tokens for handling uncertainties in flux calibration, resolution, and wavelength solution.

Currently `unite` supports JWST/NIRSpec and SDSS spectra out of the box, with more instruments to come. For unsupported instruments, the generic disperser and spectrum classes can be used to build custom configurations.

---

## Configuration Overview

`unite` describes a disperser with the following calibrations:

- **R(λ)** — the resolving power as a function of wavelength, used to compute the LSF FWHM
  at each pixel.
- **(dλ/dpix)(λ)** — the wavelength dispersion per pixel as a function of wavelength, used to convert between pixel offsets
  and wavelength offsets.

### Calibration Tokens

For a given instrument the user can optionally include calibration tokens with associated priors:
- {class}`~unite.instrument.RScale`: Resolution scaling factor (e.g. R_eff = R_nominal × RScale)
- {class}`~unite.instrument.FluxScale`: Flux scaling factor (e.g. F_eff = F_model × FluxScale)
- {class}`~unite.instrument.PixOffset`: Linear pixel offset (e.g. λ_eff = λ_model + (dλ/dpix)(λ_model) × PixOffset)
Which are used to absorb calibration offsets and/or uncertainties between instruments.

Import tokens from `unite.instrument`:

```python
from unite.instrument import RScale, FluxScale, PixOffset
```

While `RScale` might be used to apply to all spectra (e.g. to account for uncertainty in the resolution curve) in practice `FluxScale` and `PixOffset` are most useful for **relative** calibration between spectra, i.e. one spectrum is treated as the reference with fixed calibration, and the other spectra have free `FluxScale` and `PixOffset` parameters to account for differences in flux calibration and wavelength solution between the instruments. If, for example, a `FluxScale` is applied to all spectra this is completely degenerate with the overall flux normalization of the mode.

Tokens can also be named with the `name` argument, which is useful for sharing a token across multiple dispersers (see below) and for tracking parameters in the output tables.

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

Multiplies the model flux for all pixels in a spectrum by a free factor. A value of 1.0
means no correction.

```python
f = FluxScale(prior=prior.Uniform(0.5, 2.0))
```

#### PixOffset — Wavelength Offset

Shifts the wavelength solution by a number of pixels. A value of 0.0 means no correction.

```python
p = PixOffset(prior=prior.Uniform(-2, 2))
```

### Parameter Naming

Each calibration token accepts an optional semantic label as its first argument. A **type-specific prefix is automatically prepended** to form the final site name:

- `RScale('shared')` → site name `'r_scale_shared'` (prefix: `r_scale_`)
- `FluxScale('shared')` → site name `'flux_scale_shared'` (prefix: `flux_scale_`)
- `PixOffset('shared')` → site name `'pix_offset_shared'` (prefix: `pix_offset_`)

Pass **only the semantic label**, not the full prefixed name. For unshared tokens on a single named disperser, the disperser name is used automatically (e.g., `RScale()` on 'G235H' → `'r_scale_G235H'`).

### Sharing Calibration Tokens

Pass the **same token instance** to multiple dispersers to share a calibration parameter:

```python
shared_r = RScale(prior=prior.TruncatedNormal(1.0, 0.05, 0.8, 1.2), name='shared')
g235h = nirspec.G235H(r_scale=shared_r)
g395h = nirspec.G395H(r_scale=shared_r)
# → Both dispersers use the same 'r_scale_shared' parameter in the model
```

### Fixed Calibration

Use {class}`~unite.prior.Fixed` to apply a known correction without uncertainty.
This is the default for all calibration tokens, meaning that if you don't specify a token the disperser is treated as the reference with no correction. 
```python
g395h = nirspec.G395H(flux_scale=FluxScale(prior=prior.Fixed(1.15)))
```

To configure dispersers and load spectra from your instrument, please refer to the relevant section on this page.

---

## JWST/NIRSpec

### Dispersers: Gratings and PRISM

Convenience classes are provided for each NIRSpec disperser.

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

- **`'point'`** (default): Uses polynomial fits to the slit-centered point-source resolving power derived from
  [de Graaff et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024A%26A...684A..87D/abstract). 
  This is the default and recommended for most sources. Even resolved sources can use this curve
  when combined with an `RScale` token to account for the effective slit filling fraction/position.
- **`'uniform'`**: Uses tabulated resolving power for uniform illumination of the slit from [JDOX](https://jwst-docs.stsci.edu/jwst-near-infrared-spectrograph/nirspec-instrumentation/nirspec-dispersers-and-filters#gsc.tab=0).

In practice, the exact slit placement and source morphology may deviate from either of these two assumptions. In this case, the best approach is to use `r_source='point'` with a fitted `RScale` parameter to allow for uncertainty in the model.

### Calibration Tokens on NIRSpec

Here is an example of how to apply calibration tokens to a NIRSpec disperser. The same general approach applies for any instrument. In this example we use realistic offsets between the G395M and PRISM dispersers based on the observed offsets in calibrations for the RUBIES survey ([de Graaff et al. 2025](https://ui.adsabs.harvard.edu/abs/2025A%26A...697A.189D/abstract)).

```python
from unite.instrument import nirspec, RScale, FluxScale, PixOffset
from unite import prior

# Free resolution scale to account for uncertainty in the effective LSF (e.g. due to slit placement or filling fraction)
# In this example we assume it is the same for both dispersers, but it could be independent 
r_scale = RScale(prior=prior.TruncatedNormal(low = 0.7, high = 1.3, loc=1.0, scale=0.3))

# Assume G395M is our refernce grating, so it has a fixed FluxScale and PixOffset (default)
g395m = nirspec.PRISM(r_scale=r_scale)

# Apply reasonable priors on offsets
prism = nirspec.PRISM(
    r_scale=r_scale,
    flux_scale=FluxScale(prior=prior.TruncatedNormal(low=0.8, high=1.4, loc=1.1, scale=0.1)), # de Graaff+25 Fig 8.
    pix_offset=PixOffset(prior=prior.TruncatedNormal(low=-0.2, high=0.6, loc=0.2, scale=0.1)) # de Graaff+25 Fig 9.    
)
```

### Loading NIRSpec Spectra

Once dispersers have been configured, spectra can be loaded with the appropriate disperser with {class}`~unite.instrument.nirspec.NIRSpecSpectrum` which provides loaders for NIRSpec data.

{meth}`~unite.instrument.nirspec.NIRSpecSpectrum.from_DJA` loads a NIRSpec spectrum assuming the FITS file is in the format of the [DAWN JWST Archive](https://dawn-cph.github.io/dja/index.html). The spectra can even be loaded from a URL and cached in the `astropy` cache.

```python
from unite.instrument import nirspec

g235h = nirspec.G235H()

# From DJA pipeline output
spectrum = nirspec.NIRSpecSpectrum.from_DJA('spectrum_g235h.fits', disperser=g235h)
spectrum = nirspec.NIRSpecSpectrum.from_DJA('https://url_to_dja_spectrum.fits', disperser=g235h, cache=True)
```

{meth}`~unite.instrument.nirspec.NIRSpecSpectrum.from_arrays` loads a NIRSpec spectrum from arrays of {class}`astropy.unit.Quantity` for the lower pixel edges, upper pixel edges, flux, and flux error. Notably as `unite` performs pixel integration, it is necessary to pass the pixel bounds, not their centers. 

```python
# From astropy Quantity arrays
spectrum = nirspec.NIRSpecSpectrum.from_arrays(low, high, flux, error, disperser=g235h)
```
---

## SDSS

The SDSS disperser computes $R$ from the `wdisp` column (wavelength dispersion per pixel) in the native SDSS pipeline output. Therefore only loading from FITS files is currently supported. This can also be loaded via a URL and the cache keyword is also supported.

```python
from unite.instrument import sdss

# Calibration tokens can be passed here
disperser = sdss.SDSSDisperser()

# From FITS file — disperser R(λ) is updated from the wdisp column in the file
spectrum = sdss.SDSSSpectrum.from_fits('spec-PLATE-MJD-FIBER.fits', disperser=disperser)
```

{class}`~unite.instrument.sdss.SDSSSpectrum` is also a subclass of
{class}`~unite.instrument.generic.GenericSpectrum`.

---

## Generic Dispersers

For instruments without built-in support, `unite` provides two generic disperser classes.
Import them explicitly from `unite.instrument.generic`.

### SimpleDisperser

A SimpleDisperser defines the pixel scale from the input wavelength array (assuming each wavelength element corresponds to a pixel). In this case the user can specify either pass a constant or an array for the resolution.

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

{class}`~unite.instrument.generic.SimpleDisperser` supports three modes for specifying the
dispersion:

| Mode | Argument | Description |
|------|----------|-------------|
| Resolving power | `R=value` | $R = \lambda / \Delta\lambda$ (constant or array) |
| Wavelength dispersion | `dlam=value` | $\Delta\lambda$ per pixel (constant or array) |
| Velocity dispersion | `dvel=value` | $\Delta v$ per pixel in km/s (constant or array) |

### GenericDisperser

For maximum flexibility, pass callables that return $R(\lambda)$ and $d\lambda/\mathrm{pix}(\lambda)$:

```python
from unite.instrument import generic

def my_R(wavelength):
    return 2000 + 500 * (wavelength - 4000) / 5000

def my_dlam(wavelength):
    return wavelength / my_R(wavelength)

disperser = generic.GenericDisperser(R_func=my_R, dlam_dpix_func=my_dlam, unit=u.AA, name='custom')
```

### GenericSpectrum

To build a spectrum for a custom instrument, use
{class}`~unite.instrument.generic.GenericSpectrum`:

```python
from unite.instrument import generic

disperser = generic.SimpleDisperser(wavelength=wl, unit=u.AA, R=3000.0, name='sim')
spectrum = generic.GenericSpectrum(
    low=low, high=high, flux=flux, error=error, disperser=disperser
)
```

---

## InstrumentConfig and Serialization

{class}`~unite.instrument.config.InstrumentConfig` wraps a set of dispersers for
serialization and named lookup:

```python
from unite.instrument.config import InstrumentConfig

dc = InstrumentConfig([prism, g395m])
```

Dispersers can be retrieved by name using standard indexing:

```python
d = dc['G395M']          # returns the G395M disperser
d = dc['PRISM']          # returns the PRISM disperser
names = dc.names         # ['PRISM', 'G395M']
```

This is particularly useful after loading a configuration from YAML:

```python
dc = InstrumentConfig.load('dispersers.yaml')
g395m = dc['G395M']  # retrieve with calibration tokens intact
```

`validate()` is called automatically on construction and warns if any calibration axis (flux,
resolution, pixel offset) has **no fixed anchor** — meaning all dispersers have a free
parameter on that axis, making it unidentifiable.

Instrument configs can also be combined with addition:

```python
dc1 = InstrumentConfig.load('dispersers1.yaml')
dc2 = InstrumentConfig.load('dispersers2.yaml')
dc_combined = dc1 + dc2  # contains both dispersers
```

---

## Mixing Instruments

`unite` can combine spectra from entirely different instruments in one fit. Each spectrum
uses its own disperser; shared line tokens ensure the same physical parameters are used
across all spectra. Note, this may violate the assumption that the same input spectrum is 
being observed by different instruments.

```python
from unite.instrument import nirspec, sdss, FluxScale, Spectra
from unite import prior, line
import astropy.units as u

# NIRSpec grating (flux reference — no FluxScale)
g395h = nirspec.G395H()

# SDSS disperser with a free flux scale relative to NIRSpec
sdss_disp = sdss.SDSSDisperser()
sdss_disp.flux_scale = FluxScale(prior=prior.Uniform(0.5, 2.0))

# Both spectra share the same line and continuum configurations
# → same physical redshift, FWHM, and flux parameters
spec_nir  = nirspec.NIRSpecSpectrum.from_DJA('g395h.fits', disperser=g395h)
spec_sdss = sdss.SDSSSpectrum.from_fits('spec-PLATE-MJD-FIBER.fits', disperser=sdss_disp)

spectra = Spectra([spec_nir, spec_sdss], redshift=z_sys)
```
