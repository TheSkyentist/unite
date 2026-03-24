# Building the Model 

At this point a user should have:
- Configured their line model (see {doc}`line_configuration`)
- Configured their continuum model (see {doc}`continuum_configuration`)
- Configured their disperser(s) and loaded their spectrum(s) (see {doc}`instrument`)

In this section we will describe how to put these pieces together to build the full model for fitting. Before building the model, the line and continuum configurations must be filtered to match the spectral coverage, and characteristic flux scales must be computed for normalization.

## Spectra Collection

{class}`~unite.spectrum.Spectra` is the main container for fitting. It holds one
or more `Spectrum` objects:

```python
from unite.instrument import Spectra

# Single spectrum
spectra = Spectra([spectrum], redshift=0.0)

# Multiple spectra (e.g., two NIRSpec gratings)
spectra = Spectra([g235h_spectrum, g395h_spectrum], redshift=5.28)
```

The `redshift` argument shifts all line centers to the observed frame before coverage
filtering. All redshifts in `unite` are offset from this canonical redshift.

### Canonical Wavelength Unit

All internal model computations — line center arrays, continuum bounds, pixel integration,
and output quantities — use a single **canonical wavelength unit**.  By default this is the
wavelength unit of the **first spectrum's disperser** (e.g. `u.AA` for Å, `u.um` for
microns).  When mixing spectra from different instruments with different native units, the
first spectrum therefore sets the unit for everything.

You can override this explicitly:

```python
spectra = Spectra([g235h_spectrum, g395h_spectrum], redshift=5.28, canonical_unit=u.um)
```

The canonical unit propagates to all output tables and FITS files.  In particular:

- Line flux columns are in `flux_unit * canonical_unit` (integrated flux).
- Rest equivalent width columns (see {doc}`results`) are in `canonical_unit`.
- The `wavelength` column in per-spectrum tables is in `canonical_unit`.

When in doubt, inspect `spectra.canonical_unit` before running the model.

---

## Coverage Filtering

The goal of `prepare()` is to filter the line and continuum configurations to match the spectral coverage of the input spectra. This ensures that the model only includes lines and continuum regions that are actually observed, which improves efficiency and avoids fitting unconstrained parameters.

```python
filtered_lines, filtered_cont = spectra.prepare(line_configuration, continuum_configuration)

# Optional kwargs with defaults:
filtered_lines, filtered_cont = spectra.prepare(
    line_configuration, continuum_configuration,
    linedet_width=1000 * u.km / u.s,    # detection width for coverage check
    drop_empty_regions=True,# drop continuum regions with no covered lines
)
```

This:

- Shifts line centers by $(1 + z)$ to the observed frame
- Checks which lines fall within each spectrum's wavelength range, i.e. some pixels must fall within `line_center ± linedet_width / 2` for at least one spectrum.
- Drops lines with no coverage. 
- Drops continua with no coverage. If `drop_empty_regions=False`, empty continua are retained.
- Returns filtered copies of the line and continuum configurations.
covered.

After calling `prepare()`, the filtered configs are available as read-only properties:

```python
spectra.is_prepared          # True once prepare() has been called
spectra.prepared_line_config  # filtered LineConfiguration
spectra.prepared_cont_config  # filtered ContinuumConfiguration (or None)
```

## Flux and Error Scales

`unite` computes characteristic flux scales for the lines and continua to normalize the model parameters. Line Flux and Continuum Scale parameters and priors are defined in terms of these scales (see {doc}`line_configuration` and {doc}`continuum_configuration`), so this step is crucial for efficient sampling, accurate posteriors, and interpretable results.

In addition, error rescaling can also be performed to correct for under/overestimated noise in the input spectra.

The scales are computed by fitting the continuum model to the line-masked spectra and measuring the maximum continuum heights and line heights above the continua. The residual scatter in the continuum fit is used to inform error rescaling.

```python
spectra.compute_scales(filtered_lines, filtered_cont)

# Optional kwargs with defaults:
spectra.compute_scales(
    filtered_lines, filtered_cont,
    line_mask_width=1000 * u.km / u.s,  # FWHM for masking lines during continuum fit
    box_width=1000 * u.km / u.s,        # maximum expected intrinsic line FWHM for line scale
    error_scale=True,
)
```

### Continuum Fitting

The continuum fitting procedure is as follows:
- Emission lines are masked with a total width of `line_mask_width` (convolved in quadrature with the local LSF width) in each spectrum.
- The continuum model is fit to the unmasked pixels in each continuum region.
- The maximum of the median continuum heights across all regions is taken as the `continuum_scale` and is therefore in spectral flux density units.

### Flux Scale

Once the continuum model is fit, the a line height is computed for each line within it's line mask. The line heights are then multiplied by box_width in wavelength units to get an estimated line flux. The maximum of these line fluxes across all lines and spectra is taken as the `line_scale` and is therefore in flux units. This also ensures that all lines are tied to a global flux scale.

:::{note}
The box width is a tunable parameter that should be set to the maximum expected intrinsic line FWHM in wavelength units. However different line profiles will have different relationships between the line height and integrated flux. The default width of 1000 km/s is a reasonable choice but may need to be adjusted for very broad or very narrow lines.
:::

### Error Scaling

When `error_scale=True`, `compute_scales` also estimates per-region error scaling factors.

After the continuum fit, the an error scale is computed for all of the pixels for each continuum region for each spectrum. The error scale is computed such that the reduced chi-squared of the fit would become 1, which assumes that all remaining residual scatter is due to misestimated errors rather than errors in the model.

:::{note}
This is a simple heuristic that can be effective for correcting under/overestimated noise in the input spectra, but it should be used with caution. If the continuum model is a poor fit to the data, or if there are unmasked lines or other features in the residuals, then the error scaling may be inaccurate and could lead to biased posteriors. We recommend inspecting the continuum fits and residuals (see below) before deciding whether to apply error scaling.
:::

### Inspecting the Continuum Fit

After calling `compute_scales`, the fitted continuum model and per-region diagnostics are
attached directly to each spectrum via `spectrum.scale_diagnostic`.  This is a
`SpectrumScaleDiagnostic` object containing:

| Attribute | Description |
|---|---|
| `line_mask` | Boolean array — `True` where a pixel was excluded near an emission line |
| `continuum_model` | Full-length continuum model array; `NaN` outside any fitted region |
| `regions` | List of `RegionDiagnostic` objects, one per continuum region |

The spectrum's own `wavelength`, `flux`, `error`, `flux_unit`, and `unit` attributes
provide the rest of the picture — they are not duplicated in the diagnostic.

Each `RegionDiagnostic` holds `obs_low`, `obs_high`, `in_region`, `good_mask`,
`model_on_region`, `chi2_red`, and `fit_params`.

The preferred access pattern is to iterate directly over `spectra`:

```python
spectra.compute_scales(filtered_lines, filtered_cont, error_scale=True)

for s in spectra:
    diag = s.scale_diagnostic       # SpectrumScaleDiagnostic for this spectrum

    wl   = s.wavelength             # pixel-center wavelengths
    flux = s.flux                   # observed flux
    cont = diag.continuum_model     # NaN outside fitted regions
    mask = diag.line_mask           # True = excluded near a line

    for rinfo in diag.regions:
        model = rinfo.model_on_region   # evaluated on in_region pixels
        print(f'  chi2_red = {rinfo.chi2_red:.2f}')
```

You can also access diagnostics by index or name via `spectra.scale_diagnostics`:

```python
diag = spectra.scale_diagnostics[0]          # integer index
diag = spectra.scale_diagnostics['g235h']    # spectrum name (string key)
```

### Manual Scale Override

If `compute_scales()` produces scales that are not suitable for your data (e.g., lines too faint or continuum fit too poor), you can manually override the scales by setting them directly on the `Spectra` object:

```python
from astropy import units as u

# Set line flux scale manually
spectra.line_scale = 1.5e-14 * u.erg / u.s / u.cm**2

# Set continuum flux density scale manually
spectra.continuum_scale = 2.0e-15 * u.erg / u.s / u.cm**2 / u.AA
```

Both scales must be positive `astropy.units.Quantity` objects:

- `line_scale` must have flux dimensions (integrated flux). For example: `u.erg / u.s / u.cm**2`.
- `continuum_scale` must have flux density dimensions (flux per unit wavelength). For example: `u.erg / u.s / u.cm**2 / u.AA`.

Manual override is useful when:
- Either scale computation fails, produces unreasonable results, or is unstable.
- You have prior knowledge of the typical flux levels from other data or observations.

---

## Building the Model

```python
from unite import model

builder = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
model_fn, model_args = builder.build()
```

Returns a NumPyro model function and a {class}`~unite.model.ModelArgs` dataclass containing
all pre-computed matrices, scales, and data arrays. You can now proceed to inference with your favorite NumPyro inference algorithm (see {doc}`inference`).

:::{note}
Filtering the configurations and computing scales can be skipped; building the model will automatically call `prepare()` and `compute_scales()` with sensible defaults if they have not been called yet. However, we recommend calling them explicitly to inspect the filtered configurations and diagnostics before committing to the full model build.
:::

### Integration Mode

`build()` accepts an `integration_mode` parameter that controls how line profiles
are integrated over pixels.  The choice matters most when your model includes
tau-parametrized (absorption) lines:

```python
# Analytic (default) — fast, exact for emission
model_fn, args = builder.build(integration_mode='analytic')

# Quadrature — exact for emission and absorption
model_fn, args = builder.build(integration_mode='quadrature', n_nodes=7)
```

| Mode | How it works | Speed | Absorption accuracy |
|---|---|---|---|
| `'analytic'` | CDF-based integration of each profile over pixel bins | Fast | Approximate — integrates `φ` before applying `exp(-τ·φ)` |
| `'quadrature'` | Gauss-Legendre quadrature of the full composed model at `n_nodes` sub-pixel points | Slower | Exact — properly integrates `∫F(λ)·exp(-τ·φ(λ)) dλ` |

**When to use each:**

- **Analytic** is the right default for most models, and is exact for emission-only
  models.  For absorption lines, the approximation is accurate when the absorber
  is well-resolved (profile varies slowly across a pixel).
- **Quadrature** should be used when tau-parametrized lines are unresolved or
  marginally resolved — for example, narrow absorption in low-resolution spectra
  (NIRSpec PRISM), or when mixing emission and absorption at similar wavelengths
  across spectrographs with very different resolutions.

:::{warning}
In analytic mode the model computes `exp(-τ·∫φ)` rather than `∫F·exp(-τ·φ)`.
For unresolved absorbers (line narrower than a pixel), these differ because
pixel-averaging the profile before applying the exponential underestimates the
absorption depth.  If your fit includes absorption lines that are unresolved in
one or more spectra, use `integration_mode='quadrature'`.
:::

### Absorber Position

When your line configuration includes absorption lines, the `absorber_position`
parameter on `build()` controls where the absorbing material sits relative to
the emission and continuum sources.  This affects the model equation:

```python
# Default: absorber in front of everything
model_fn, args = builder.build(absorber_position='foreground')

# Absorber between continuum source and emission region
model_fn, args = builder.build(absorber_position='behind_lines')

# Absorber between emission region and observer, behind continuum
model_fn, args = builder.build(absorber_position='behind_continuum')
```

The three options correspond to different physical geometries:

| Position | Model equation | Physical meaning |
|---|---|---|
| `'foreground'` | `T × (emission + continuum)` | Absorber in front of lines and continua |
| `'behind_lines'` | `emission + T × continuum` | Absorber between continua and lines |
| `'behind_continuum'` | `T × emission + continuum` | Absorber between lines and continua |

where `T = exp(-τ·φ(λ))` is the transmission spectrum.

When no absorption lines are present, `T = 1` everywhere and the model reduces
to the standard emission + continuum equation regardless of `absorber_position`.

Both `absorber_position` and `integration_mode` can be passed to the `fit()`
convenience method:

```python
samples, args = builder.fit(
    absorber_position='foreground',
    integration_mode='quadrature',
)
```
