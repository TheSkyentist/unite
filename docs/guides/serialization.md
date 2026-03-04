# Configuration Serialization

`unite` configurations can be saved to and loaded from human-readable YAML files. This
supports reproducible fits, offline editing, and sharing configurations between collaborators.

---

## The Configuration Container

{class}`~unite.config.Configuration` is the top-level object that bundles everything needed
to describe a fit:

```python
from unite.config import Configuration
from unite.disperser import DispersersConfiguration

config = Configuration(
    lines=lc,           # LineConfiguration
    continuum=cc,       # ContinuumConfiguration (optional)
    dispersers=dc,      # DispersersConfiguration (optional)
)
```

---

## Saving to YAML

```python
# To a file
config.save('my_fit.yaml')

# To a YAML string (useful for logging or embedding in notebooks)
yaml_str = config.to_yaml()
print(yaml_str)
```

---

## Loading from YAML

```python
# From a file
config2 = Configuration.load('my_fit.yaml')

# From a YAML string
config2 = Configuration.from_yaml(yaml_str)

# Access the reconstructed objects
lc2 = config2.lines
cc2 = config2.continuum
dc2 = config2.dispersers
```

Token sharing is preserved across the round-trip: if two lines shared a `Redshift` token
when the config was saved, they will share the same reconstructed token object after loading.

---

## YAML Format

The YAML file is human-readable and editable. Here is the output of a simple two-line
configuration:

```yaml
lines:
  lines:
    - name: H_alpha
      center: 6563.0 AA
      profile: Gaussian
      redshift:
        name: nlr_z
        prior: {type: Uniform, low: -0.005, high: 0.005}
      fwhm_gauss:
        name: nlr_fwhm
        prior: {type: Uniform, low: 1.0, high: 10.0}
      flux:
        name: Ha_flux
        prior: {type: Uniform, low: 0, high: 5}
    - name: NII_6585
      center: 6585.0 AA
      profile: Gaussian
      redshift: {ref: nlr_z}       # ← shared token reference
      fwhm_gauss: {ref: nlr_fwhm}  # ← shared token reference
      flux:
        name: NII6585_flux
        prior: {type: Uniform, low: 0, high: 5}
continuum:
  regions:
    - low: 6241.85 AA
      high: 6931.05 AA
      form: {type: Linear}
```

Shared tokens appear **once** as a full definition and subsequently as a `{ref: name}`
reference. You can edit priors, add lines, or change profile types directly in the YAML.

---

## Including Dispersers

When a `DispersersConfiguration` is included, calibration tokens are also serialized:

```python
from unite.disperser import DispersersConfiguration, FluxScale
from unite.disperser.nirspec import G395M, PRISM

g395m = G395M(r_source='point', flux_scale=FluxScale(prior.Uniform(0.5, 2.0)))
prism = PRISM(r_source='point')

dc = DispersersConfiguration(dispersers=[g395m, prism])
config = Configuration(lines=lc, continuum=cc, dispersers=dc)
config.save('nirspec_fit.yaml')
```

The YAML output will include a `dispersers` section with `calib_params` (calibration token
definitions) and `entries` (disperser instances referencing those tokens):

```yaml
dispersers:
  calib_params:
    flux_scale_g395m:
      type: FluxScale
      prior: {type: Uniform, low: 0.5, high: 2.0}
  entries:
    - type: G395M
      r_source: point
      flux_scale: {ref: flux_scale_g395m}
    - type: PRISM
      r_source: point
```

---

## Round-Trip Guarantee

`unite` tests confirm that `Configuration.load(config.save(...))` reproduces an identical
configuration, including:

- All token names and priors
- Token sharing relationships (same token → same model parameter)
- Profile types and parameters
- Continuum region boundaries and functional forms
- Calibration token types and priors on dispersers

---

## Editing Configurations Offline

A common workflow is to build a configuration programmatically, save it, edit the YAML
by hand (e.g., to tighten priors based on a preliminary fit), and reload:

```python
# First pass — broad priors
config.save('fit_v1.yaml')

# Edit fit_v1.yaml in your editor, then:
config_v2 = Configuration.load('fit_v1.yaml')
```

:::{warning}
The YAML format is part of `unite`'s public API, but **do not rename token `name` fields**
without also updating all `{ref: ...}` references to that token. Broken references will
raise a `KeyError` on load.
:::
