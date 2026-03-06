# Building Configurations Programmatically

This tutorial shows how to construct `unite` configurations in code — using loops,
templates, and merging — rather than defining each line by hand. This is the recommended
approach for survey pipelines and multi-object analyses.

---

## Fitting the Same Lines Across Many Objects

When processing a catalog, you typically want to apply the same line set to every
object, with per-object priors informed by a photometric redshift or other prior
knowledge.

```python
from astropy.table import Table
from unite import line, prior, model
from unite.continuum import ContinuumConfiguration, Linear
from unite.config import Configuration
import astropy.units as u

# Emission lines to fit (rest-frame wavelengths)
EMISSION_LINES = {
    'H_alpha':   6563.0 * u.AA,
    'NII_6585':  6585.0 * u.AA,
    'NII_6549':  6549.0 * u.AA,
    'H_beta':    4861.0 * u.AA,
    'OIII_5007': 5007.0 * u.AA,
    'OIII_4959': 4959.0 * u.AA,
}

catalog = Table.read('catalog.fits')

for obj in catalog:
    # Per-object redshift prior centred on photometric redshift
    z_lo = obj['z_phot'] - 0.02
    z_hi = obj['z_phot'] + 0.02
    z    = line.Redshift('z',    prior=prior.Uniform(z_lo, z_hi))
    fwhm = line.FWHM('fwhm_nlr', prior=prior.Uniform(50, 500))

    lc = line.LineConfiguration()
    for name, center in EMISSION_LINES.items():
        lc.add_line(
            name, center,
            redshift=z,
            fwhm_gauss=fwhm,
            flux=line.Flux(f'{name}_flux', prior=prior.Uniform(0, 10)),
        )

    cc = ContinuumConfiguration.from_lines(lc.centers, pad=0.05, form=Linear())

    config = Configuration(lc, cc)
    config.save(f'configs/{obj["id"]}.yaml')
```

Each object gets its own YAML file that can be inspected and edited offline before
running the fit.

---

## Building Narrow and Broad Components Separately

For AGN fitting it is often cleaner to define the narrow-line and broad-line
configurations independently and then merge them:

```python
def build_narrow(z_lo, z_hi):
    z    = line.Redshift('z_nlr',  prior=prior.Uniform(z_lo, z_hi))
    fwhm = line.FWHM('fwhm_nlr',  prior=prior.Uniform(50, 400))

    lc = line.LineConfiguration()
    for name, wl in EMISSION_LINES.items():
        lc.add_line(name, wl, redshift=z, fwhm_gauss=fwhm,
                    flux=line.Flux(f'{name}_n', prior=prior.Uniform(0, 10)))
    return lc


def build_broad(z_lo, z_hi, fwhm_narrow_token):
    z    = line.Redshift('z_blr', prior=prior.Uniform(z_lo, z_hi))
    fwhm = line.FWHM('fwhm_blr', prior=prior.Uniform(fwhm_narrow_token + 200, 8000))

    lc = line.LineConfiguration()
    # Only Balmer lines have a broad component
    for name, wl in [('H_alpha', 6563.0 * u.AA), ('H_beta', 4861.0 * u.AA)]:
        lc.add_line(name, wl, redshift=z, fwhm_gauss=fwhm,
                    flux=line.Flux(f'{name}_b', prior=prior.Uniform(0, 20)))
    return lc


lc_narrow = build_narrow(0.0, 0.01)
lc_broad  = build_broad(0.0, 0.01, fwhm_narrow_token=lc_narrow['fwhm_nlr'])

# strict=True (default / __add__): raises if two tokens share the same name
# strict=False: same-named tokens of the same type are merged into one parameter
lc = lc_narrow + lc_broad
```

All components of the same line name are automatically summed in the model.

---

## Sharing Tokens Across Merged Configurations

If both sub-configurations define a token with the **same name and type**, use
`strict=False` to merge them into a single model parameter:

```python
# Both use 'z_sys' — we want one shared redshift
lc_optical = build_optical_config()   # defines Redshift('z_sys', ...)
lc_uv      = build_uv_config()        # also defines Redshift('z_sys', ...)

# strict=False: 'z_sys' appears once in the model
lc = lc_optical.merge(lc_uv, strict=False)
```

With `strict=True` (the default), a name collision raises an error so you do not
accidentally share parameters you intended to keep independent.

---

## Saving and Loading in Batch

Once a configuration is built, save it to YAML for reproducibility and offline editing:

```python
from unite.config import Configuration
from unite.continuum import ContinuumConfiguration, PowerLaw

cc = ContinuumConfiguration.from_lines(lc.centers, pad=0.08, form=PowerLaw())
config = Configuration(lc, cc)
config.save('agn_config.yaml')

# Later — or on another machine — reload and run
config2 = Configuration.load('agn_config.yaml')
```

See {doc}`../guides/serialization` for details on the YAML format and round-trip
guarantees.
