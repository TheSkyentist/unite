# Line Configuration

{class}`~unite.line.LineConfiguration` is the core container that defines which emission
lines to fit, their profile shapes, and how parameters are shared between lines.

---

## Creating a Line Configuration

```python
from unite import line, prior
from astropy import units as u

lc = line.LineConfiguration()
```

---

## Adding Lines

### Basic Usage

```python
z = line.Redshift('z', prior=prior.Uniform(-0.01, 0.01))
fwhm = line.FWHM('fwhm', prior=prior.Uniform(50, 500))

lc.add_line(
    'H_alpha', 6563.0 * u.AA,
    redshift=z,
    fwhm_gauss=fwhm,
    flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 10)),
)
```

Each line requires:

- A **name** (string) — used in results tables and YAML output
- A **rest-frame centre wavelength** (Quantity with wavelength units)
- A **redshift** token ({class}`~unite.line.Redshift`)
- **FWHM token(s)** appropriate for the chosen profile
- A **flux** token ({class}`~unite.line.Flux`)

### Line Centres and Units

Centres must be astropy Quantities with wavelength units:

```python
lc.add_line('Ly_alpha', 1216.0 * u.AA, ...)
lc.add_line('CO_10', 2.6 * u.um, ...)       # microns work too
```

---

## Parameter Sharing (Tokens)

This is `unite`'s central design pattern. A **token** is a named Python object representing a
model parameter. **Same Python object = same parameter in the model.**

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

### Token Types

| Token | Role | Unit |
|-------|------|------|
| {class}`~unite.line.Redshift` | Kinematic redshift | dimensionless |
| {class}`~unite.line.FWHM` | Line width | km/s |
| {class}`~unite.line.Flux` | Line flux normalization | internal units |
| {class}`~unite.line.Param` | Arbitrary profile parameter (h3, h4, etc.) | user-defined |

---

## Multiple Components

Adding lines with the **same name** creates multiple components that are summed in the model:

```python
z = line.Redshift('z', prior=prior.Uniform(-0.01, 0.01))

fwhm_n = line.FWHM('fwhm_narrow', prior=prior.Uniform(50, 400))
fwhm_b = line.FWHM('fwhm_broad',  prior=prior.Uniform(500, 3000))

lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm_n,
            flux=line.Flux('Ha_n', prior=prior.Uniform(0, 10)))
lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm_b,
            flux=line.Flux('Ha_b', prior=prior.Uniform(0, 10)))
```

Both components contribute to the total H$\alpha$ flux. A common pattern is to share the
redshift but use independent FWHM tokens for each kinematic component (narrow, broad,
outflow, etc.).

---

## Dependent Priors on Line Parameters

Prior bounds on FWHM or Flux tokens can reference other tokens, creating dependency chains
that are automatically resolved at model-build time. See {doc}`priors` for the full
reference.

```python
fwhm_narrow = line.FWHM('fwhm_n', prior=prior.Uniform(50, 500))
fwhm_broad  = line.FWHM('fwhm_b', prior=prior.Uniform(fwhm_narrow + 150, 5000))
```

This ensures the broad component is always at least 150 km/s wider than the narrow component.

### Flux Ratio Chains

The same approach applies to flux tokens. A common case is enforcing physically motivated
flux ratios between related lines:

```python
flux_ha = line.Flux('Ha', prior=prior.Uniform(0, 10))

# [NII]6585 constrained relative to H-alpha
flux_nii = line.Flux('NII', prior=prior.Uniform(flux_ha * 0.05, flux_ha * 3.0))

# [NII]6549 is fixed at 1/2.94 of [NII]6585 (atomic physics)
flux_nii_w = line.Flux('NII_w', prior=prior.Uniform(
    flux_nii / 2.94 - 0.01, flux_nii / 2.94 + 0.01,
))
```

Dependency chains can go arbitrarily deep — `unite` resolves the topological ordering at
model-build time.

---

## Three-Component AGN Model

A common AGN setup with narrow, broad, and outflow components sharing a systemic redshift
but with ordering constraints between the widths:

```python
# Shared systemic redshift
z = line.Redshift('z', prior=prior.Uniform(-0.01, 0.01))

# Three FWHM tokens with ordering constraints
fwhm_n   = line.FWHM('fwhm_narrow',  prior=prior.Uniform(50, 400))
fwhm_b   = line.FWHM('fwhm_broad',   prior=prior.Uniform(fwhm_n + 150, 3000))
fwhm_out = line.FWHM('fwhm_outflow', prior=prior.Uniform(fwhm_n + 50, fwhm_b - 50))

lc = line.LineConfiguration()

# Narrow component
lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm_n,
            flux=line.Flux('Ha_n', prior=prior.Uniform(0, 10)))

# Broad component
lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm_b,
            flux=line.Flux('Ha_b', prior=prior.Uniform(0, 10)))

# Outflow component with independent velocity offset
z_out = line.Redshift('z_out', prior=prior.Uniform(-0.005, 0.001))
lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z_out, fwhm_gauss=fwhm_out,
            flux=line.Flux('Ha_out', prior=prior.Uniform(0, 5)))
```

All three components are summed when computing the total H$\alpha$ model flux.

---

## Line Profiles

All profiles are **analytically integrated** over pixels (not Riemann-summed) and convolved
with the instrumental LSF. See {doc}`/concepts` for the LSF convolution convention.

### Gaussian (default)

The simplest and most common profile. A Gaussian intrinsic shape convolved with the
Gaussian LSF; the result is also Gaussian with
$\mathrm{FWHM} = \sqrt{\mathrm{fwhm\_gauss}^2 + \mathrm{lsf\_fwhm}^2}$.

**Parameters:** `fwhm_gauss` (km/s)

```python
lc.add_line('H_alpha', 6563.0 * u.AA, profile='Gaussian',
            redshift=z, fwhm_gauss=fwhm, flux=flux)
```

**When to use:** Thermal or turbulent broadening; most nebular emission lines.

### PseudoVoigt

A linear combination of a Gaussian and a Lorentzian (Thompson et al. 1987). The Gaussian
component is convolved with the LSF; the Lorentzian component is not.

**Parameters:** `fwhm_gauss` (km/s), `fwhm_lorentz` (km/s)

```python
lc.add_line('H_alpha', 6563.0 * u.AA, profile='PseudoVoigt',
            redshift=z,
            fwhm_gauss=line.FWHM('fwhm_g', prior=prior.Uniform(50, 500)),
            fwhm_lorentz=line.FWHM('fwhm_l', prior=prior.Uniform(0, 500)),
            flux=flux)
```

**When to use:** Lines with both thermal (Gaussian) and natural/pressure (Lorentzian)
broadening — e.g., AGN broad lines, damped Lyman-alpha wings.

### Cauchy

A pure Lorentzian profile. Internally a `PseudoVoigt` with `fwhm_gauss = 0`. The LSF is
*not* convolved into the Lorentzian width.

**Parameters:** `fwhm_lorentzian` (km/s)

```python
lc.add_line('H_alpha', 6563.0 * u.AA, profile='Cauchy',
            redshift=z,
            fwhm_lorentzian=line.FWHM('fwhm', prior=prior.Uniform(0, 1000)),
            flux=flux)
```

**When to use:** Pure Lorentzian tails. Prefer `PseudoVoigt` for most applications.

### Laplace

A double-exponential (Laplace) profile convolved with the Gaussian LSF.

**Parameters:** `fwhm_exp` (km/s)

**When to use:** Lines with exponential wings (e.g., some outflow signatures).

### SEMG — Symmetric Exponentially Modified Gaussian

A Gaussian (convolved with LSF) additionally broadened by a symmetric Laplace distribution.
Equivalent to the convolution $G_\mathrm{LSF+intr} * \mathrm{Laplace}$.

**Parameters:** `fwhm_gauss` (km/s), `fwhm_exp` (km/s)

**When to use:** Profiles with mild exponential tails on top of a Gaussian core, e.g.
turbulent outflows.

### GaussHermite

A Gaussian (convolved with LSF) modified by Hermite polynomial corrections. `h3` controls
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

**When to use:** Kinematically complex lines (galaxy kinematics, merger signatures). The
Gauss-Hermite expansion is widely used in stellar kinematics and is orthogonal to the
Gaussian shape.

### SplitNormal

A two-sided Gaussian with independent widths on the blue and red sides. Each side is
independently convolved with the LSF.

**Parameters:** `fwhm_blue` (km/s), `fwhm_red` (km/s)

```python
lc.add_line('H_alpha', 6563.0 * u.AA, profile='SplitNormal',
            redshift=z,
            fwhm_blue=line.FWHM('fwhm_b', prior=prior.Uniform(50, 500)),
            fwhm_red=line.FWHM('fwhm_r',  prior=prior.Uniform(50, 500)),
            flux=flux)
```

**When to use:** Lines with a measurable asymmetry; simple outflow models.

### Profile String Aliases

Profiles are set via the `profile` argument (case-insensitive strings or class instances):

| String | Profile |
|--------|---------|
| `'Gaussian'`, `'gaussian'`, `'normal'` | `Gaussian` |
| `'Cauchy'`, `'cauchy'`, `'Lorentzian'`, `'lorentzian'` | `Cauchy` |
| `'PseudoVoigt'`, `'pseudovoigt'`, `'voigt'` | `PseudoVoigt` |
| `'Laplace'`, `'laplace'`, `'exponential'` | `Laplace` |
| `'SEMG'`, `'semg'`, `'exp-gaussian'` | `SEMG` |
| `'GaussHermite'`, `'hermite'`, `'gauss-hermite'` | `GaussHermite` |
| `'SplitNormal'`, `'split-normal'`, `'two-sided'` | `SplitNormal` |

### Summary Table

| Profile | Parameters | LSF on Gaussian? | Typical use |
|---------|-----------|-----------------|-------------|
| `Gaussian` | `fwhm_gauss` | Yes | Nebular / thermal |
| `PseudoVoigt` | `fwhm_gauss`, `fwhm_lorentz` | Yes (Gaussian only) | AGN broad lines |
| `Cauchy` | `fwhm_lorentzian` | No | Pure Lorentzian |
| `Laplace` | `fwhm_exp` | No | Exponential wings |
| `SEMG` | `fwhm_gauss`, `fwhm_exp` | Yes | Gaussian + exp wings |
| `GaussHermite` | `fwhm_gauss`, `h3`, `h4` | Yes | Non-Gaussian shape |
| `SplitNormal` | `fwhm_blue`, `fwhm_red` | Yes (both) | Asymmetric lines |

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
and you want them to be treated as the same model parameter.

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
