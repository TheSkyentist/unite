# Instruments and Dispersers

Dispersers describe the instrumental characteristics (resolution, wavelength solution) of a
spectrograph. `unite` provides built-in support for **JWST/NIRSpec** (all gratings + PRISM),
**SDSS**, and **generic** spectrographs.

---

## Disperser Overview

A {class}`~unite.disperser.base.Disperser` provides:

- **R(λ)** — the resolving power as a function of wavelength, used to compute the LSF FWHM
  at each pixel
- **Calibration tokens** — optional free parameters
  ({class}`~unite.disperser.base.RScale`,
  {class}`~unite.disperser.base.FluxScale`,
  {class}`~unite.disperser.base.PixOffset`) for absorbing calibration uncertainties

---

## Built-in Instruments

### JWST/NIRSpec

Seven convenience classes for each NIRSpec grating/prism:

```python
from unite.instruments.nirspec import G140H, G140M, G235H, G235M, G395H, G395M, PRISM
```

Each class provides $R(\lambda)$ based on calibration data.

#### Resolution Sources

NIRSpec dispersers support two resolution calibration modes via `r_source`:

```python
g235h = G235H(r_source='point')     # point-source R (de Graaff et al. 2025)
g235h = G235H(r_source='uniform')   # uniform illumination R from FITS tables
```

- **`'point'`** (default): Uses polynomial fits to the point-source resolving power from
  de Graaff et al. (2025). Recommended for unresolved or compact sources observed through MSA
  shutters.
- **`'uniform'`**: Uses tabulated resolving power for uniform illumination of the slit. More
  appropriate for extended sources that fill the shutter.

#### Calibration Tokens on NIRSpec

```python
from unite.disperser import RScale, FluxScale, PixOffset
from unite import prior

# Free resolution scale (e.g., for extended sources)
g235h = G235H(r_scale=RScale(prior=prior.TruncatedNormal(1.0, 0.05, 0.8, 1.2)))

# Free flux calibration between gratings
g235h = G235H(flux_scale=FluxScale(prior=prior.Uniform(0.5, 2.0)))

# Wavelength offset in pixels
g235h = G235H(pix_offset=PixOffset(prior=prior.Uniform(-2, 2)))
```

#### Loading NIRSpec Spectra

```python
from unite.instruments.nirspec import NIRSpecSpectrum

# From DJA pipeline output
spectrum = NIRSpecSpectrum.from_DJA('spectrum_g235h.fits', disperser=g235h)

# From arrays (wavelength in microns)
spectrum = NIRSpecSpectrum.from_arrays(
    wavelength=wavelength,
    flux=flux,
    error=error,
    disperser=g235h,
)
```

### SDSS

```python
from unite.instruments.sdss import SDSSDisperser, SDSSSpectrum

disperser = SDSSDisperser(wdisp=wdisp_array, wavelength=wavelength, unit=u.AA)

# From FITS file
spectrum = SDSSSpectrum.from_fits('spec-PLATE-MJD-FIBER.fits')

# From arrays
spectrum = SDSSSpectrum.from_arrays(
    wavelength=wl, flux=flux, error=error, disperser=disperser,
)
```

The SDSS disperser computes $R$ from the `wdisp` column (wavelength dispersion per pixel) in
the native SDSS pipeline output.

---

## Generic Dispersers

For instruments without built-in support, `unite` provides two generic disperser classes.

### SimpleDisperser

Define the dispersion from a wavelength grid:

```python
from unite.disperser.generic import SimpleDisperser
import numpy as np
from astropy import units as u

wavelength = np.linspace(4000, 9000, 500)
disperser = SimpleDisperser(
    wavelength=wavelength,
    unit=u.AA,
    R=3000.0,        # constant R
    name='my_spectrograph',
)
```

{class}`~unite.disperser.generic.SimpleDisperser` supports three modes for specifying the
dispersion:

| Mode | Argument | Description |
|------|----------|-------------|
| Resolving power | `R=value` | $R = \lambda / \Delta\lambda$ (constant or array) |
| Wavelength dispersion | `dlam=value` | $\Delta\lambda$ per pixel (constant or array) |
| Velocity dispersion | `dvel=value` | $\Delta v$ per pixel in km/s (constant or array) |

### GenericDisperser

For maximum flexibility, pass a callable that returns $R$ as a function of wavelength:

```python
from unite.disperser.generic import GenericDisperser

def my_R(wavelength):
    return 2000 + 500 * (wavelength - 4000) / 5000

disperser = GenericDisperser(R=my_R, unit=u.AA, name='custom')
```

---

## Calibration Tokens

Calibration tokens are free (or fixed) parameters attached to a disperser to absorb
calibration uncertainties between spectra. They are subclasses of
{class}`~unite.prior.Parameter` and follow the same token-sharing pattern as line tokens.

### RScale — Resolution Calibration

Multiplies the nominal $R(\lambda)$ by a free factor:
$R_\mathrm{eff}(\lambda) = R_\mathrm{nominal}(\lambda) \times r\_scale$

```python
from unite.disperser import RScale

r = RScale(prior=prior.TruncatedNormal(1.0, 0.05, 0.8, 1.2))
g235h = G235H(r_scale=r)
```

Useful when the actual LSF is broader or narrower than the calibration predicts — e.g., for
extended sources in NIRSpec, or when the slit filling fraction differs from the calibration
assumption.

### FluxScale — Relative Flux Calibration

Multiplies the model flux for all pixels in a spectrum by a free factor. A value of 1.0
means no correction.

```python
from unite.disperser import FluxScale

f = FluxScale(prior=prior.Uniform(0.5, 2.0))
g395h = G395H(flux_scale=f)
```

### PixOffset — Wavelength Offset

Shifts the wavelength solution by a number of pixels.

```python
from unite.disperser import PixOffset

p = PixOffset(prior=prior.Uniform(-2, 2))
g395h = G395H(pix_offset=p)
```

### Sharing Calibration Tokens

Pass the **same token instance** to multiple dispersers to share a calibration parameter:

```python
shared_r = RScale(prior=prior.TruncatedNormal(1.0, 0.05, 0.8, 1.2), name='r_shared')
g235h = G235H(r_scale=shared_r)
g395h = G395H(r_scale=shared_r)
```

### Fixed Calibration

Use {class}`~unite.prior.Fixed` to apply a known correction without uncertainty:

```python
g395h = G395H(flux_scale=FluxScale(prior=prior.Fixed(1.15)))
```

---

## DispersersConfiguration

When using calibration tokens, wrap your dispersers in a
{class}`~unite.disperser.config.DispersersConfiguration`:

```python
from unite.disperser import DispersersConfiguration

dc = DispersersConfiguration([prism, g395m])
```

`validate()` is called automatically on construction and warns if any calibration axis (flux,
resolution, pixel offset) has **no fixed anchor** — meaning all dispersers have a free
parameter on that axis, making it unidentifiable.

### Degeneracy Warnings

If all dispersers have a free `FluxScale`, the absolute flux level is **degenerate**: the
model can absorb any global scaling into the `FluxScale` parameters. Similarly for `RScale`
and `PixOffset`. Best practice: leave one disperser as the **reference** with default (fixed)
calibration tokens.

---

## Serialization

```python
dc.save('dispersers.yaml')
dc2 = DispersersConfiguration.load('dispersers.yaml')
```

Calibration token sharing is preserved through YAML round-trips. See
{doc}`serialization` for the full workflow.

---

## Multi-Instrument Fitting

### NIRSpec Multi-Grating

A typical setup fits overlapping NIRSpec gratings with a single shared resolution
calibration and an independent flux scale per grating:

```python
from unite.instruments.nirspec import G235H, G395H, NIRSpecSpectrum
from unite.disperser import DispersersConfiguration, RScale, FluxScale
from unite import prior

# Shared resolution scale — same physical LSF for both gratings
r = RScale(prior=prior.TruncatedNormal(1.0, 0.05, 0.8, 1.2), name='r_shared')

# G235H is the flux reference; G395H has a free relative calibration
g235h = G235H(r_scale=r)
g395h = G395H(r_scale=r, flux_scale=FluxScale(prior=prior.Uniform(0.5, 2.0)))

spec1 = NIRSpecSpectrum.from_DJA('g235h.fits', disperser=g235h)
spec2 = NIRSpecSpectrum.from_DJA('g395h.fits', disperser=g395h)

dc = DispersersConfiguration([g235h, g395h])
```

Because `r` is the **same token instance** on both dispersers, it becomes a single
free parameter in the model that scales both LSFs simultaneously.

### Mixing Instruments

`unite` can combine spectra from entirely different instruments in one fit. Each spectrum
uses its own disperser; shared line tokens ensure the same physical parameters are used
across all spectra.

```python
from unite.instruments.nirspec import G395H, NIRSpecSpectrum
from unite.instruments.sdss import SDSSDisperser, SDSSSpectrum
from unite.disperser import DispersersConfiguration, FluxScale
from unite import prior, line
import astropy.units as u

# NIRSpec grating (flux reference — no FluxScale)
g395h = G395H()

# SDSS disperser with a free flux scale relative to NIRSpec
sdss = SDSSDisperser(wdisp=wdisp_array, wavelength=wl_sdss, unit=u.AA)
sdss.flux_scale = FluxScale(prior=prior.Uniform(0.5, 2.0))

# Both spectra share the same line and continuum configurations
# → same physical redshift, FWHM, and flux parameters
spec_nir  = NIRSpecSpectrum.from_DJA('g395h.fits', disperser=g395h)
spec_sdss = SDSSSpectrum.from_fits('spec-PLATE-MJD-FIBER.fits', disperser=sdss)

spectra = Spectra([spec_nir, spec_sdss], redshift=z_sys)
dc = DispersersConfiguration([g395h, sdss])
```
