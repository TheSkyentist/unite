# Configuration Serialization

`unite` configurations can be saved to and loaded from human-readable YAML files. This
supports reproducible fits, offline editing, and sharing configurations between collaborators.

---

## The Configuration Container

{class}`~unite.config.Configuration` is the top-level object that bundles everything needed
to describe a fit:

```python
from unite.config import Configuration
from unite.instrument.config import InstrumentConfig

config = Configuration(
    lines=lc,           # LineConfiguration
    continuum=cc,       # ContinuumConfiguration (optional)
    dispersers=dc,      # InstrumentConfig (optional)
)
```

---

## Saving and Loading

### Top-Level Configuration

```python
# To a file
config.save('my_fit.yaml')

# To a YAML string (useful for logging or embedding in notebooks)
yaml_str = config.to_yaml()

# From a file
config2 = Configuration.load('my_fit.yaml')

# From a YAML string
config2 = Configuration.from_yaml(yaml_str)

# Access the reconstructed objects
lc2 = config2.lines
cc2 = config2.continuum
dc2 = config2.dispersers
```

### Sub-Configuration Serialization

Each sub-configuration also supports standalone YAML I/O:

```python
# Lines only
lc.save('lines.yaml')
lc2 = LineConfiguration.load('lines.yaml')

# Continuum only
cc.save('continuum.yaml')
cc2 = ContinuumConfiguration.load('continuum.yaml')

# Dispersers only
dc.save('dispersers.yaml')
dc2 = InstrumentConfig.load('dispersers.yaml')
```

This is useful for reusing the same line configuration across different fits, or sharing
disperser setups between collaborators.

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

### Dependent Priors in YAML

When a prior bound references another parameter, it is serialized with `ref`, `scale`, and
`offset` fields:

```yaml
fwhm_broad:
  name: fwhm_b
  prior:
    type: Uniform
    low: {ref: fwhm_n, scale: 1.0, offset: 150.0}
    high: 5000.0
```

This represents the expression `fwhm_narrow * 1.0 + 150.0` as the lower bound.

### Dispersers in YAML

When a {class}`~unite.instrument.config.InstrumentConfig` is included, calibration
tokens are also serialized:

```yaml
dispersers:
  calib_params:
    r_shared:
      type: RScale
      prior: {type: TruncatedNormal, loc: 1.0, scale: 0.05, low: 0.8, high: 1.2}
  entries:
    - type: G235H
      r_source: point
      r_scale: {ref: r_shared}
    - type: G395H
      r_source: point
      r_scale: {ref: r_shared}
      flux_scale:
        name: flux_g395h
        prior: {type: Uniform, low: 0.5, high: 2.0}
```

---

## Round-Trip Guarantee

`unite` tests confirm that `Configuration.load(config.save(...))` reproduces an identical
configuration, including:

- All token names and priors
- Token sharing relationships (same token → same model parameter)
- Dependent prior expressions (ParameterRef chains)
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

---

## Dict Interface

All configuration objects also support `to_dict()` / `from_dict()` for programmatic
manipulation:

```python
d = config.to_dict()
# Modify the dict...
config2 = Configuration.from_dict(d)
```

This is useful for automated parameter sweeps or template-based configuration generation.
