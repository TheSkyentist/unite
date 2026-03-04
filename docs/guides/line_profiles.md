# Line Profiles

`unite` supports seven line profile shapes. All profiles are **analytically integrated over
pixels** (not Riemann-summed), which is exact, fast, and handles undersampled data correctly.

All profiles are convolved with the instrumental LSF, which is modelled as a Gaussian with
FWHM determined by the disperser at each pixel's wavelength.

---

## Specifying a Profile

Profiles are set per-line via the `profile` argument of `add_line`:

```python
lc.add_line('H_alpha', 6563.0 * u.AA, profile='Gaussian', ...)
lc.add_line('H_alpha', 6563.0 * u.AA, profile='GaussHermite', ...)
```

Profiles can also be passed as class instances for access to default-prior inspection:

```python
from unite.line import GaussHermite
lc.add_line('H_alpha', 6563.0 * u.AA, profile=GaussHermite(), ...)
```

String aliases are case-insensitive:

| String | Profile |
|--------|---------|
| `'Gaussian'`, `'gaussian'`, `'normal'` | `Gaussian` |
| `'Cauchy'`, `'cauchy'`, `'Lorentzian'`, `'lorentzian'` | `Cauchy` |
| `'PseudoVoigt'`, `'pseudovoigt'`, `'voigt'` | `PseudoVoigt` |
| `'Laplace'`, `'laplace'`, `'exponential'` | `Laplace` |
| `'SEMG'`, `'semg'`, `'exp-gaussian'` | `SEMG` |
| `'GaussHermite'`, `'hermite'`, `'gauss-hermite'` | `GaussHermite` |
| `'SplitNormal'`, `'split-normal'`, `'two-sided'` | `SplitNormal` |

---

## Profile Reference

### Gaussian

The simplest and most common profile. A Gaussian intrinsic shape convolved with the
Gaussian LSF; the result is also Gaussian with FWHM = $\sqrt{\mathrm{fwhm\_gauss}^2 +
\mathrm{lsf\_fwhm}^2}$.

**Parameters:** `fwhm_gauss` (km/s)

```python
lc.add_line(
    'H_alpha', 6563.0 * u.AA,
    profile='Gaussian',
    redshift=z,
    fwhm_gauss=line.FWHM('fwhm', prior=prior.Uniform(50, 500)),
    flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 10)),
)
```

**When to use:** Thermal or turbulent broadening; most nebular emission lines.

---

### PseudoVoigt

A linear combination of a Gaussian and a Lorentzian (Thompson et al. 1987). The Gaussian
component is convolved with the LSF; the Lorentzian component is not.

**Parameters:** `fwhm_gauss` (km/s), `fwhm_lorentz` (km/s)

```python
lc.add_line(
    'H_alpha', 6563.0 * u.AA,
    profile='PseudoVoigt',
    redshift=z,
    fwhm_gauss=line.FWHM('fwhm_g', prior=prior.Uniform(50, 500)),
    fwhm_lorentz=line.FWHM('fwhm_l', prior=prior.Uniform(0, 500)),
    flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 10)),
)
```

**When to use:** Lines with both thermal (Gaussian) and natural/pressure (Lorentzian)
broadening, e.g. AGN broad lines, damped Lyman-alpha wings.

---

### Cauchy

A pure Lorentzian profile. Implemented internally as a `PseudoVoigt` with `fwhm_gauss = 0`.
The LSF is *not* convolved into the Lorentzian width (consistent with the package convention
that LSF adds in quadrature to the Gaussian component only).

**Parameters:** `fwhm_lorentzian` (km/s)

```python
lc.add_line(
    'H_alpha', 6563.0 * u.AA,
    profile='Cauchy',
    redshift=z,
    fwhm_lorentzian=line.FWHM('fwhm', prior=prior.Uniform(0, 1000)),
    flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 10)),
)
```

**When to use:** Pure Lorentzian tails; rarely needed — prefer `PseudoVoigt` for most AGN
applications.

---

### Laplace

A double-exponential (Laplace) profile convolved with the Gaussian LSF.

**Parameters:** `fwhm_exp` (km/s)

```python
lc.add_line(
    'H_alpha', 6563.0 * u.AA,
    profile='Laplace',
    redshift=z,
    fwhm_exp=line.FWHM('fwhm', prior=prior.Uniform(0, 1000)),
    flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 10)),
)
```

**When to use:** Lines with exponential wings (e.g., some outflow signatures).

---

### SEMG — Symmetric Exponentially Modified Gaussian

A Gaussian (convolved with LSF) additionally broadened by a symmetric Laplace distribution.
Equivalent to the convolution $G_\mathrm{LSF+intr} * \mathrm{Laplace}$.

**Parameters:** `fwhm_gauss` (km/s), `fwhm_exp` (km/s)

```python
lc.add_line(
    'H_alpha', 6563.0 * u.AA,
    profile='SEMG',
    redshift=z,
    fwhm_gauss=line.FWHM('fwhm_g', prior=prior.Uniform(0, 500)),
    fwhm_exp=line.FWHM('fwhm_e',   prior=prior.Uniform(0, 500)),
    flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 10)),
)
```

**When to use:** Profiles with mild exponential tails on top of a Gaussian core, e.g.
turbulent outflows.

---

### GaussHermite

A Gaussian (convolved with LSF) modified by Hermite polynomial corrections that encode
deviations from a symmetric Gaussian shape. `h3` controls skewness; `h4` controls kurtosis.

**Parameters:** `fwhm_gauss` (km/s), `h3` (dimensionless), `h4` (dimensionless)

```python
lc.add_line(
    'H_alpha', 6563.0 * u.AA,
    profile='GaussHermite',
    redshift=z,
    fwhm_gauss=line.FWHM('fwhm', prior=prior.Uniform(50, 1000)),
    h3=line.Param('h3', prior=prior.TruncatedNormal(0, 0.1, -0.3, 0.3)),
    h4=line.Param('h4', prior=prior.TruncatedNormal(0, 0.1, -0.3, 0.3)),
    flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 10)),
)
```

**When to use:** Kinematically complex lines (galaxy kinematics, merger signatures). The
Gauss-Hermite expansion is widely used in stellar kinematics and has the advantage of being
orthogonal to the Gaussian shape.

---

### SplitNormal

A two-sided Gaussian with independent widths on the blue and red sides of the line center.
Each side is independently convolved with the LSF.

**Parameters:** `fwhm_blue` (km/s), `fwhm_red` (km/s)

```python
lc.add_line(
    'H_alpha', 6563.0 * u.AA,
    profile='SplitNormal',
    redshift=z,
    fwhm_blue=line.FWHM('fwhm_b', prior=prior.Uniform(50, 500)),
    fwhm_red=line.FWHM('fwhm_r',  prior=prior.Uniform(50, 500)),
    flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 10)),
)
```

**When to use:** Lines with a measurable asymmetry that is better captured by two widths than
by a Gauss-Hermite `h3` correction; simple outflow models.

---

## LSF Convolution Convention

The spectral LSF is modelled as a **Gaussian** at every pixel, with FWHM determined by the
disperser's resolution curve $R(\lambda)$:

$$\mathrm{lsf\_fwhm}(\lambda) = \frac{\lambda}{R(\lambda)}$$

For profiles with a Gaussian component (`Gaussian`, `PseudoVoigt`, `SEMG`, `GaussHermite`,
`SplitNormal`), the intrinsic and LSF widths are added in quadrature:

$$\mathrm{total\_fwhm\_gauss} = \sqrt{\mathrm{fwhm\_gauss}^2 + \mathrm{lsf\_fwhm}^2}$$

For purely Lorentzian profiles (`Cauchy`, and the Lorentzian component of `PseudoVoigt`),
**no LSF convolution is applied** — the Lorentzian width is the intrinsic width.

---

## Summary Table

| Profile | Parameters | LSF on Gaussian? | Typical use |
|---------|-----------|-----------------|-------------|
| `Gaussian` | `fwhm_gauss` | ✓ | Nebular / thermal |
| `PseudoVoigt` | `fwhm_gauss`, `fwhm_lorentz` | ✓ (Gaussian only) | AGN broad lines |
| `Cauchy` | `fwhm_lorentzian` | — | Pure Lorentzian |
| `Laplace` | `fwhm_exp` | — | Exponential wings |
| `SEMG` | `fwhm_gauss`, `fwhm_exp` | ✓ | Gaussian + exponential wings |
| `GaussHermite` | `fwhm_gauss`, `h3`, `h4` | ✓ | Non-Gaussian shape |
| `SplitNormal` | `fwhm_blue`, `fwhm_red` | ✓ (both) | Asymmetric lines |
