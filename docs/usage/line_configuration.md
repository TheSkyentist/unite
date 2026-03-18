# Line Configuration

`unite`'s line configuration system is designed to be flexible and feature rich, allowing user to set up complex multi-line models with shared and correlated parameters across a wide range of line profiles. This page walks though the core concepts and usage patterns for building line configurations.

---

## Creating a Line Configuration

{class}`~unite.line.LineConfiguration` is the core container that defines which emission
lines to fit, their profile shapes, and how parameters are shared between lines.
The best way to get started is to create an empty configurations. We'll also need 
the `prior` module for defining priors on line parameters, and `astropy.units` for specifying line wavelengths.

```python
from unite import line, prior
from astropy import units as u

lc = line.LineConfiguration()
```

---

## Adding Lines

### Basic Usage

```python
lc.add_line('H_alpha', 6563.0 * u.AA)
```

:::{important}
Line names must be **unique** within a `LineConfiguration`. Adding a second line with the same name raises a `ValueError`. Use distinct names for different kinematic components, e.g. `'Ha_narrow'` and `'Ha_broad'`.
:::

At minimum each line requires:

- A **name** (string) — used in results tables and YAML output
- A **rest-frame center wavelength** 

Centers must be {class}`astropy.units.Quantity` with wavelength units:

```python
lc.add_line('Ly_alpha', 1216.0 * u.AA, ...)
lc.add_line('CO_10', 2.6 * u.um, ...)
```

### Adding a Single Line

Optionally each line can also include:

- A relative **redshift** token ({class}`~unite.line.Redshift`)
- A **flux** token ({class}`~unite.line.Flux`)
- A **profile shape** (e.g. `Gaussian`, `PseudoVoigt`, etc.)
- **FWHM token(s)** appropriate for the chosen profile
- Additional shape parameters (e.g. `h3`, `h4` for `GaussHermite`)
- A **strength** parameter which multiplies the flux token, mostly used for line ratios (see below)

If not specified, default priors are used for redshift, flux, and FWHM tokens, and the profile defaults to `Gaussian`.

```python
z = line.Redshift('z', prior=prior.Uniform(-0.01, 0.01))
fwhm = line.FWHM('fwhm', prior=prior.Uniform(50, 500))

lc.add_line(
    'H_alpha', 6563.0 * u.AA,
    redshift=z,
    fwhm_gauss=fwhm,
    flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 1)),
)
```

You can preview your line configuration by printing it:

```
print(lc)
LineConfiguration: 1 lines, 1 flux / 1 z / 1 profile params

  Name     Wavelength        Profile   Redshift  Params  Flux     Strength
  -------  ----------------  --------  --------  ------  -------  --------
  H_alpha  6563.00 Angstrom  Gaussian  z         fwhm    Ha_flux  1.00

  Redshift:
    z  Uniform(low=-0.01, high=0.01)

  Params (fwhm_gauss):
    fwhm  Uniform(low=50.0, high=500.0)

  Flux:
    Ha_flux  Uniform(low=0.0, high=1.0)
```


:::{note}
Redshift tokens/priors are relative to the systemic system redshift which is specified with the {doc}`instrument`. This allows configurations to be redshift-agnostic and reusable across different datasets.
:::

### Adding Multiple Lines

`unite` also supports adding multiple lines at once with `add_lines`. Each entry in
`centers` becomes an independent line with a unique auto-generated name of the form
`'{name}_{center_value}'` (e.g. `'OIII_4960'`, `'OIII_5008'`). You can also supply
an explicit `names` array of the same length as `centers`.

```python
# Auto-generated names: 'OIII_4960' and 'OIII_5008'
lc.add_lines('OIII', np.array([4960, 5008]) * u.AA, redshift=z, fwhm_gauss=fwhm)

# Explicit names
lc.add_lines('OIII', np.array([4960, 5008]) * u.AA,
             names=['OIII_blue', 'OIII_red'], redshift=z)
```

Note how we fix the line ratio here.

```python
flux = line.Flux('OIII')
lc.add_lines(
    'OIII', np.array([4960, 5008]) * u.AA,
    redshift = z, # Same redshift
    fwhm = [fwhm, fwhm], # Different FWHMs (just as an example)
    flux = [line.Flux(prior = prior.Fixed(flux / 3)), flux], # Fixed ratio
)
```

However, it is likely easier to specify multiples through the `strength` parameter, which multiplies the flux token when building the model. Both approaches yield the same result.

```python
flux = line.Flux('OIII')
lc.add_lines(
    'OIII', np.array([4960, 5008]) * u.AA,
    flux = line.Flux(), # Default behaviour, this would mean all lines have the same flux
    strength = [1/3, 1] # But we pass a strength per line which is multiplied by the flux token when building the model
)
```

---

## Parameter Sharing

This is `unite`'s central design pattern. A **token** is a named Python object representing a
model parameter. **Same Python object = same parameter in the model.**

These are the following token types:

| Token | Role | Unit |
|-------|------|------|
| {class}`~unite.line.Redshift` | Redshift offset from systemic | dimensionless |
| {class}`~unite.line.FWHM` | Line width | km/s |
| {class}`~unite.line.Flux` | Line flux normalization | internal units |
| {class}`~unite.line.LineShape` | Arbitrary profile parameter (h3, h4, etc.) | per-parameter basis |

:::{note}
Fluxes are relative to a scale computed based on the spectrum. See Flux and Error Scales in {doc}`build_model` for details.
:::

### Shared Kinematics

```python
# One redshift, one FWHM → shared across all lines
z = line.Redshift('z', prior=prior.Uniform(-0.01, 0.01))
fwhm = line.FWHM('fwhm', prior=prior.Uniform(50, 500))

lc.add_line('H_alpha',  6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm,
            flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 10)))
lc.add_line('NII_6585', 6585.0 * u.AA, redshift=z, fwhm_gauss=fwhm,
            flux=line.Flux('NII6585_flux', prior=prior.Uniform(0, 10)))
lc.add_line('NII_6549', 6549.0 * u.AA, redshift=z, fwhm_gauss=fwhm,
            flux=line.Flux('NII6549_flux', prior=prior.Uniform(0, 10)))
```

Result: one `z` parameter and one `fwhm` parameter in the model, shared by all three lines.

### Independent Parameters

```python
z_narrow = line.Redshift('z_narrow', prior=prior.Uniform(-0.01, 0.01))
z_broad  = line.Redshift('z_broad',  prior=prior.Uniform(-0.01, 0.01))

lc.add_line('Ha_narrow', 6563.0 * u.AA, redshift=z_narrow, ...)
lc.add_line('Ha_broad',  6563.0 * u.AA, redshift=z_broad,  ...)
```

### Parameter Names

All token classes accept an optional first positional argument that becomes a semantic label for the parameter. **It is highly recommended to name your tokens** — without a name, `unite` auto-generates one (e.g. `fwhm_0`, `redshift_2`) that becomes difficult to interpret in output.

A **type-specific prefix is automatically prepended** to your label to form the final site name:

- `Redshift('nlr')` → site name `'z_nlr'` (prefix: `z_`)
- `FWHM('broad')` → site name `'fwhm_broad'` (prefix: `fwhm_`)
- `Flux('Ha')` → site name `'flux_Ha'` (prefix: `flux_`)

So pass **only the semantic label**, not the full prefixed name:

```python
# Auto-generated names → based on line name, less explicit
z = line.Redshift()

# Explicit semantic labels → clear, shareable across lines
z_nlr = line.Redshift('nlr')        # → site name 'z_nlr'
z_blr = line.Redshift('blr')        # → site name 'z_blr'
fwhm = line.FWHM('broad')           # → site name 'fwhm_broad'
flux = line.Flux('Ha')              # → site name 'flux_Ha'
```

The site names show up in:

- The `samples` dict returned by `mcmc.get_samples()`
- Column names in the parameter table from `make_parameter_table()`

:::{tip}
Use short, semantic labels (without prefixes) that identify the physical component,
e.g. `'nlr'`, `'blr'`, `'broad'`, `'Ha'`, `'NII'`. The type-specific prefix is added
automatically. This makes site names concise and easy to navigate in results.
:::

Lines can share the same name but must have at least one different token, otherwise an error will be raised. This allows you to easily set up multi-component models with shared parameters.

```python
z = line.Redshift('nlr', prior=prior.Uniform(-0.01, 0.01))

fwhm_n = line.FWHM('narrow', prior=prior.Uniform(50, 400))
fwhm_b = line.FWHM('broad',  prior=prior.Uniform(500, 3000))

lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm_n,
            flux=line.Flux('Ha_n', prior=prior.Uniform(0, 10)))
lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm_b,
            flux=line.Flux('Ha_b', prior=prior.Uniform(0, 10)))
```

---

## Dependent Priors

Prior bounds on FWHM or Flux tokens can reference other tokens, creating dependency chains
that are automatically resolved at model-build time. See {doc}`priors` for the full
reference.

```python
fwhm_narrow = line.FWHM('fwhm_n', prior=prior.Uniform(50, 500))
fwhm_broad  = line.FWHM('fwhm_b', prior=prior.Uniform(fwhm_narrow + 150, 5000))
```

This ensures the broad component is always at least 150 km/s wider than the narrow component.

---

## Line Profiles

Here we list all currently supported profiles in `unite`.
All profiles are **analytically integrated** over pixels (not Riemann-summed) and convolved
with the instrumental LSF. See {doc}`/concepts` for the LSF convolution convention.
Profiles are set via the `profile` argument (case-insensitive strings or class instances):

| String | Profile | FWHM Parameter(s) | Shape Parameter(s) |
|--------|---------|-------------------|--------------------|
| `'Gaussian'`, `'gaussian'`, `'normal'` | `Gaussian` | `fwhm_gauss` |
| `'Cauchy'`, `'cauchy'`, `'Lorentzian'`, `'lorentzian'` | `Cauchy` | `fwhm_lorentz` |
| `'PseudoVoigt'`, `'pseudovoigt'`, `'voigt'` | `PseudoVoigt` | `fwhm_gauss`, `fwhm_lorentz` |
| `'Laplace'`, `'laplace'`, `'exponential'` | `Laplace` | `fwhm_exp` |
| `'SEMG'`, `'semg'`, `'exp-gaussian'` | `SEMG` | `fwhm_gauss`, `fwhm_exp |
| `'GaussHermite'`, `'hermite'`, `'gauss-hermite'` | `GaussHermite` | `fwhm_gauss`, `h3`, `h4` |
| `'SplitNormal'`, `'split-normal'`, `'two-sided'` | `SplitNormal` | `fwhm_blue`, `fwhm_red` |

### Gaussian (default)

The simplest and most common profile. A Gaussian intrinsic shape convolved with the
Gaussian LSF; the result is also Gaussian with
$\mathrm{FWHM} = \sqrt{\mathrm{fwhm\_gauss}^2 + \mathrm{lsf\_fwhm}^2}$.

**Parameters:** `fwhm_gauss` (km/s)

```python
lc.add_line('H_alpha', 6563.0 * u.AA, profile='Gaussian',
            redshift=z, fwhm_gauss=fwhm, flux=flux)
```

### PseudoVoigt

An accurate numerical approximation to the Voigt profile, which is the convolution of a Gaussian and a Lorentzian. 

**Parameters:** `fwhm_gauss` (km/s), `fwhm_lorentz` (km/s)

```python
lc.add_line('H_alpha', 6563.0 * u.AA, profile='PseudoVoigt',
            redshift=z,
            fwhm_gauss=line.FWHM('fwhm_g', prior=prior.Uniform(50, 500)),
            fwhm_lorentz=line.FWHM('fwhm_l', prior=prior.Uniform(0, 500)),
            flux=flux)
```

### Cauchy

A pure Lorentzian profile. Internally a `PseudoVoigt` with `fwhm_gauss = 0`.

**Parameters:** `fwhm_lorentz` (km/s)

```python
lc.add_line('H_alpha', 6563.0 * u.AA, profile='Cauchy',
            redshift=z,
            fwhm_lorentz=line.FWHM('fwhm', prior=prior.Uniform(0, 1000)),
            flux=flux)
```

### Laplace

A double-exponential (Laplace) profile convolved with the Gaussian LSF.

**Parameters:** `fwhm_exp` (km/s)

### SEMG — Symmetric Exponentially Modified Gaussian

A Gaussian convolved with a Laplace distribution.

**Parameters:** `fwhm_gauss` (km/s), `fwhm_exp` (km/s)

### GaussHermite

A Gaussian modified by Hermite polynomial corrections. `h3` controls
skewness; `h4` controls kurtosis.

**Parameters:** `fwhm_gauss` (km/s), `h3` (dimensionless), `h4` (dimensionless)

```python
lc.add_line('H_alpha', 6563.0 * u.AA, profile='GaussHermite',
            redshift=z,
            fwhm_gauss=line.FWHM('fwhm', prior=prior.Uniform(50, 1000)),
            h3=line.Param('h3', prior=prior.TruncatedNormal(0, 0.1, -0.3, 0.3)),
            h4=line.Param('h4', prior=prior.TruncatedNormal(0, 0.1, -0.3, 0.3)),
            flux=flux)
```

### SplitNormal

A two-sided Gaussian with independent widths on the blue and red sides.

**Parameters:** `fwhm_blue` (km/s), `fwhm_red` (km/s)

```python
lc.add_line('H_alpha', 6563.0 * u.AA, profile='SplitNormal',
            redshift=z,
            fwhm_blue=line.FWHM('fwhm_b', prior=prior.Uniform(50, 500)),
            fwhm_red=line.FWHM('fwhm_r',  prior=prior.Uniform(50, 500)),
            flux=flux)
```

---

## Merging Configurations

{class}`~unite.line.LineConfiguration` supports merging two configurations:

```python
lc_narrow = line.LineConfiguration()
# ... add narrow lines ...

lc_broad = line.LineConfiguration()
# ... add broad lines ...

# strict=True (default, also __add__): raises on token name collisions
lc_combined = lc_narrow + lc_broad

# strict=False: shares same-named tokens of same type
lc_combined = lc_narrow.merge(lc_broad, strict=False)
```

The `strict=False` mode is useful when both configurations define tokens with the same name
and you want them to be treated as the same model parameter. However, proceed with caution, it will not check that the priors for the same-named tokens are identical, it will choose the token from the first configuration. 

---

## Serialization

{class}`~unite.line.LineConfiguration` supports standalone YAML serialization:

```python
lc.save('lines.yaml')
lc2 = line.LineConfiguration.load('lines.yaml')
```

Token sharing is preserved: if two lines shared a `Redshift` token before saving, they will
share the same reconstructed token after loading. See {doc}`serialization` for the full
workflow and YAML format.
