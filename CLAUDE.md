# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**unite** — Unified liNe Integration Turbo Engine. A Python package for fast, efficient Bayesian inference of emission lines from multiple spectra simultaneously, built on JAX, NumPyro, and Astropy. Originally designed for JWST/NIRSpec spectra but extensible to other instruments.

## Commands

This project uses [Pixi](https://pixi.sh/) for environment and task management.

```bash
pixi run format       # Format code with ruff
pixi run lint         # Check code with ruff
pixi run lint-fix     # Auto-fix linting issues
pixi run test         # Run pytest
pixi run test-cov     # Run tests with coverage report
pixi run test-lowest  # Test against lowest compatible dependency versions
pixi run docs         # Build Sphinx documentation
pixi run docs-clean   # Build docs from scratch (clean build directory first)
pixi run build        # Build sdist + wheel
pixi run type-check   # Run basedpyright for type checking
```

To run a single test file or specific test:

```bash
pixi run -e test pytest tests/test_instrument_base.py
pixi run -e test pytest tests/test_instrument_nirspec.py::test_function_name
```

## Architecture

### User Workflow

1. **Line configuration** — create `LineConfiguration` with lines, multiplets, shared kinematics (`Redshift`/`FWHM` tokens), priors, and profile shapes
2. **Continuum configuration** — auto-generate from line config (pad + merge) or specify regions manually; attach functional form per region
3. **Serialization** — save/load config to human-readable YAML; user can edit offline
4. **Spectrum loading** — create spectrum objects, wrap in `Spectra` collection; automatic coverage filtering drops unconstrained lines/regions
5. **Model building** — `ModelBuilder.build()` returns `(model_fn, model_args)` for the user to run with their own numpyro sampler (NUTS, NS, SVI, etc.); or call `ModelBuilder.fit()` as a NUTS convenience wrapper that returns `(samples, model_args)`
6. **Result extraction** — `evaluate_model`, `make_parameter_table`, `make_spectra_tables`, `make_hdul`; pass `percentiles=[0.16, 0.5, 0.84]` to get summary statistics instead of all samples

### Module Map

#### Core Modules

**`unite/config.py`** — Top-level `Configuration` container; serializes a complete line + continuum + instrument config to/from YAML.

**`unite/model.py`** — `ModelBuilder` assembles line/continuum configurations and spectra into a numpyro model. `build()` returns `(model_fn, model_args)` and accepts `integration_mode` (`'analytic'`, `'quadrature'`, or `'convolution'`) to select how line profiles are integrated over pixels. Absorber geometry is controlled by per-component `zorder` integers set at config time (on `add_line()` and `ContinuumConfiguration`), not at build time. `build()` derives `cont_applies` (which tau lines attenuate the continuum) from the line and continuum zorders. `fit()` is a convenience NUTS wrapper returning `(samples, model_args)`. Handles parameter sharing matrices, priors, topological sorting, and the emission/absorption split (tau is kept separate from flux_scale throughout).

**`unite/prior.py`** — `Prior` ABC with implementations (`Uniform`, `TruncatedNormal`, `Fixed`, etc.); `Parameter` base class; `topological_sort` for dependency ordering.

**`unite/compute.py`** — `SpectrumPrediction` dataclass and `evaluate_model` for computing model predictions from posterior samples. For emission lines, `pred.lines[label]` is the exact flux contribution (positive). For absorption lines, `pred.lines[label]` is the flux _removed_ by that absorber: `total - total_without_j` (negative). Both analytic and quadrature paths use `compose_leave_one_out` from `_compose.py` for uniform decomposition.

**`unite/_compose.py`** — Private module with pure JAX composition functions shared by `model.py` and `evaluate.py`. `compose_from_profiles` combines pre-evaluated emission profiles, absorption profiles, and continuum into a total model using per-component effective transmissions derived from `applies_matrix` (shape `(n_lines, n_lines)`, precomputed from line zorders) and `cont_applies` (shape `(n_lines,)`, which tau lines attenuate the continuum). `compose_leave_one_out` additionally computes exact per-line contributions: emission lines return their intrinsic (un-attenuated) flux profile; absorption lines return `total - total_without_j` (negative).

**`unite/results.py`** — `make_parameter_table`, `make_spectra_tables`, `make_hdul` for structured result extraction. All accept an optional `percentiles` array (values in 0–1) to return summary statistics instead of all posterior samples. REST equivalent widths are computed for both emission lines (analytical: `F / (C * (1+z))`) and absorption lines (numerical integration of the absorbed flux profile on the finest covering pixel grid).

**`unite/_utils.py`** — Constants (e.g., `C_KMS`) and wavelength/flux conversion utilities.

#### Line Fitting Subsystem

**`unite/line/config.py`** — `LineConfiguration`: main container for line specs (emission and absorption). Token classes: `Redshift`, `FWHM`, `LineShape`, `Flux`, `Tau` — shared tokens across lines produce a single model parameter. `Flux` is for emission lines, `Tau` (dimensionless optical depth) is for absorption lines. Whether a line is emission or absorption is determined by the token: pass `tau=Tau()` for absorption, otherwise defaults to `Flux()`. Any profile shape can be used for either emission or absorption. Passing both `flux` and `tau` raises `TypeError`. Each line carries a `zorder: int` (default 0 for emission, 1 for tau) set via `add_line(..., zorder=)`. `ConfigMatrices` includes `line_zorders` (shape `(n_lines,)`), `applies_matrix` (shape `(n_lines, n_lines)` bool — True when tau line k is in front of component j), and `is_tau` alongside the existing flux/redshift/fwhm matrices.

**`unite/line/library.py`** — `Profile` ABC plus concrete profiles: `Gaussian`, `Cauchy`, `PseudoVoigt`, `Laplace`, `SEMG`, `GaussHermite`, `SplitNormal`. All profiles can be used for both emission and absorption lines — the distinction is made at the configuration level, not the profile level. Each declares required parameters via `param_names()` and provides `integrate_branch()` / `evaluate_branch()` for dispatch.

**`unite/line/compute.py`** — Vmapped dispatch functions for line profile computation: `evaluate_lines` (pointwise evaluation at arbitrary wavelengths, used by the quadrature path) and `integrate_lines` (exact CDF-based integration over pixel bins, used by the analytic path). Both dispatch to per-profile JAX kernels via `lax.switch` keyed on `Profile.code`.

**`unite/line/functions.py`** — JAX-jitted integration kernels (`integrate_gaussian`, `integrate_voigt`, `integrate_gaussHermite`, etc.) and pointwise evaluation kernels (`evaluate_gaussian`, `evaluate_voigt`, `evaluate_lorentzian`). Integration kernels return normalized per-pixel fractions with LSF convolution and exact pixel integration. Evaluation kernels are used by Gauss-Legendre quadrature integration.

#### Continuum Modeling Subsystem

**`unite/continuum/config.py`** — `ContinuumConfiguration` managing `ContinuumRegion` objects; auto-generation from line configs (padding + merging). Token classes: `Scale`, `NormWavelength`, `ContShape`.

**`unite/continuum/library.py`** — `ContinuumForm` ABC and implementations: `Linear`, `PowerLaw`, `Polynomial`, `Bernstein`, `BSpline`, `Chebyshev`, `Blackbody`, `AttenuatedBlackbody`, `ModifiedBlackbody`. All forms accept an `lsf_fwhm` parameter on both `evaluate()` and `integrate()`. Polynomial-based forms (`Linear`, `Polynomial`, `Chebyshev`, `Bernstein`) analytically convolve with the Gaussian LSF via `_gaussian_convolve_poly`; others ignore it. Private helpers `_cheb_to_mono_matrix` and `_bernstein_to_mono_matrix` provide static basis-conversion matrices cached at form construction time.

**`unite/continuum/compute.py`** — Continuum evaluation functions shared by `model.py` and `evaluate.py`: `eval_continuum` (total continuum at arbitrary wavelength points), `eval_continuum_regions` (per-region contributions for decomposition), `integrate_continuum` (delegates to each form's `integrate()` method). All three accept an optional `lsf_fwhm` array (per-pixel or per-wavelength-point) that is forwarded to the form methods.

**`unite/continuum/functions.py`** — JAX-compatible continuum evaluation kernels used by `ContinuumForm.evaluate()`.

**`unite/continuum/fit.py`** — `fit_continuum_form`, `ContinuumFitResult` for non-Bayesian continuum fitting (used by `compute_scales`).

#### Spectrum Handling

**`unite/spectrum/`** — Spectrum and collection module (separate from instrument):

- `spectrum.py` — `Spectrum` single-spectrum container: pixel edges (`low`, `high`), flux/error as `astropy.units.Quantity`, `Disperser`, optional scalar or per-pixel `error_scale`.
- `collection.py` — `Spectra` collection with `redshift`, `canonical_unit`, `line_scale`/`continuum_scale` (Quantities). `compute_scales()` masks lines, fits polynomial continuum, measures peak for line scale; `error_scale=True` computes per-region error scaling. `prepare()` filters configs for coverage; sets `prepared_line_config`/`prepared_cont_config` on the object.
- `loaders.py` — `from_arrays()`, `from_DJA()` (JWST/NIRSpec DJA FITS), `from_sdss_fits()` loader functions.

#### Instrument Handling

**`unite/instrument/base.py`** — `Disperser` ABC with abstract `R(wavelength)` and `dlam_dpix(wavelength)`. Calibration tokens: `RScale` (resolving power scale), `FluxScale` (flux normalization), `PixOffset` (pixel shift). Sharing = same token instance on multiple dispersers → single model parameter.

**`unite/instrument/generic.py`** — `GenericDisperser` (user-supplied JAX callables for R and dλ/dpix), `SimpleDisperser` (pixel-grid wavelength array). **Only importable via `unite.instrument.generic`** — not re-exported from `unite.instrument`.

**`unite/instrument/config.py`** — `InstrumentConfig`: collects dispersers, serializes calibration tokens (hoisted to top-level `calib_params` in YAML). Disperser registry for round-trip deserialization.

**`unite/instrument/nirspec/`** — `G140H`, `G140M`, `G235H`, `G235M`, `G395H`, `G395M`, `PRISM` dispersers.

**`unite/instrument/sdss/`** — `SDSSDisperser`.

### Design Principles

- **Matrices encode sharing, priors encode relationships.** Matrices map shared tokens to lines. Dependent priors (topologically sorted at model-build time) handle inter-parameter constraints (e.g., broad fwhm > narrow fwhm + 150 km/s). This enables arbitrary-depth dependency chains.
- **Profiles declare their needs.** Each `Profile` subclass declares required fwhm parameters via `fwhm_names()`. The matrix and model systems handle them generically.
- **User controls the sampler.** `ModelBuilder.build()` returns `(model_fn, model_args)` for use with any numpyro sampler. `ModelBuilder.fit()` provides a NUTS convenience wrapper for common use cases.
- **Serialization is first-class.** Every config object has `to_dict()` / `from_dict()`. YAML is the interchange format. Round-trip correctness is tested.

## Key Conventions

- **JAX-first**: All internal arrays use `jax.numpy` (jnp); all math must be JAX-jittable. JAX 64-bit is enabled via `JAX_ENABLE_X64=1`.
- **Units at boundaries**: Astropy units for input validation; internal JAX arrays are unitless. NIRSpec dispersers expect wavelengths in microns. `Spectrum` flux/error must be `astropy.units.Quantity` with f_lambda units.
- **Imports**: `Spectrum`, `Spectra`, and loaders (`from_arrays`, `from_DJA`, `from_sdss_fits`) import from `unite.spectrum`. `GenericDisperser`, `SimpleDisperser` import from `unite.instrument.generic`. `unite.instrument` exports only `RScale`, `FluxScale`, `PixOffset`, `InstrumentConfig`.
- **Docstrings**: NumPy-style (numpydoc convention) required for all public classes and methods.
- **Type hints**: Use `jax.typing.ArrayLike` for flexible array inputs; `astropy.units.Quantity` for physical quantities.
- **Code style**: Ruff with single quotes, 88-char line length. `N802` is ignored (uppercase `R` function allowed per physics convention).
- **Parameter site names**: The `name` argument on tokens is a human-readable label. The numpyro site name is the label with a type-specific prefix prepended at model-build time (e.g. `z_` for redshifts, `flux_` for fluxes, `tau_` for optical depths).
- **Flux vs tau parametrization**: Lines parametrized by `Flux` are additive profiles scaled by integrated flux (can be positive or negative — negative flux is useful for placing non-detection constraints). Lines parametrized by `Tau` (dimensionless optical depth) produce multiplicative transmission `T = exp(-tau * phi)` applied to emission/continuum. Any profile shape can be used with either parametrization — the distinction is made by passing `tau=Tau()` to `add_line` (defaults to `Flux()` if neither is specified; passing both raises `TypeError`). Tau is never mixed with `flux_scale` — they flow through separate matrices (`flux_matrix` vs `tau_matrix`).
- **Tests**: `pytest` with `xfail_strict=True` and warnings treated as errors. Coverage tracks branches.
- **Formatting, Linting, and Type Checking**: We use ruff for formatting and linting as well as basedpyright type checking. These are all part of our CI tests with Pixi tasks.
- **`absorption.py`**: Top-level standalone module (not yet part of the package) implementing JAX-jittable Balmer absorption (`balmer_transmission`, `full_balmer_transmission`) via Voigt profiles using the Humlicek W4 Faddeeva approximation. Used for post-fit HI absorption diagnostics.

## Documentation

Docs are written in MyST Markdown (`.md`) and built with Sphinx. Tutorials are executable Python scripts (`tutorial_*.py`) in `docs/tutorials/` and are auto-executed by **sphinx-gallery** to produce an `auto_tutorials/` gallery — `abort_on_example_error = True` means a broken tutorial fails the build. Ruff format/lint checks also run against `docs/tutorials/`.

Structure:

- `docs/usage/` — narrative guides (line configuration, continuum, instrument, priors, serialization, model building, inference, results)
- `docs/tutorials/` — executable example scripts processed by sphinx-gallery
- `docs/api/` — API reference auto-generated from docstrings via `autodoc` + `numpydoc`

All public classes and methods require NumPy-style docstrings. Type hints appear in the description (not the signature) via `autodoc_typehints = 'description'`.

## CI/CD

Four GitHub Actions workflows, all targeting `main`:

| Workflow          | Trigger                                      | Purpose                                                                             |
| ----------------- | -------------------------------------------- | ----------------------------------------------------------------------------------- |
| `ci.yml`          | push/PR to `main`                            | format check, lint, tests on Python 3.12–3.14; coverage uploaded to Codecov on 3.14 |
| `docs.yml`        | PR to `main` + manual                        | builds Sphinx docs (including executing all tutorials)                              |
| `autorelease.yml` | push to `main` when `pyproject.toml` changes | creates a GitHub Release for the new version tag                                    |
| `publish.yml`     | GitHub Release published                     | builds sdist + wheel, publishes to PyPI via trusted publishing                      |

**Release process:** bump `version` in `pyproject.toml`, merge to `main` → `autorelease.yml` creates the tag/release → `publish.yml` pushes to PyPI automatically.

## README Maintenance

The `README.md` is the primary user-facing description of `unite`'s capabilities. Keep it in sync with the code when making significant changes. The sections most likely to drift:

- **"What it does"** bullet list — update when adding/removing profile types, continuum forms, integration modes, or major features (e.g. new absorber position options, new instruments).
- **"Quick Start"** code example — update the imports and API calls when the public interface changes (e.g. new token types, renamed arguments, new result-extraction signatures).
- **Profile list** (`unite/line/library.py`) — each registered `Profile` subclass should appear in the README bullet for line profiles.
- **Continuum form list** (`unite/continuum/library.py`) — each registered `ContinuumForm` subclass should appear in the README bullet for continuum models.
- **Integration modes** (`ModelBuilder.build(integration_mode=...)`) — all three modes (`'analytic'`, `'quadrature'`, `'convolution'`) should be described.
- **Absorption support** — mention tau-parametrized lines and per-component `zorder` depth ordering whenever documenting the model.

Do **not** add implementation detail, internal module paths, or scientific caveats to the README; those belong in `CLAUDE.md` (Scientific Assumptions) or the Sphinx docs.

## Scientific Assumptions

- Spectral LSF is assumed to be Gaussian. Therefore all lines, even if intrinsically Lorentzian, exponential, or Voigt, are analytically convolved with a Gaussian LSF. E.g. a Voigt adds the LSF fwhm in quadrature to the intrinsic Gaussian fwhm, but not the Lorentzian fwhm. So a Lorentzian line is actually a pseudo-Voigt with `fwhm_gauss` = LSF fwhm and `fwhm_lorentz` = intrinsic Lorentzian fwhm.
- Three integration modes are available, selectable via `integration_mode` on `ModelBuilder.build()`:
  - **Analytic** (default): All line profiles are analytically integrated over pixels via CDF differences (exact for each profile individually). However, for tau-parametrized lines the model computes `exp(-tau * ∫phi)` rather than `∫F * exp(-tau * phi)` — i.e. each profile is integrated before the nonlinear transmission is applied to the flux. This is accurate when the profile varies slowly across a pixel, but introduces an approximation for tau lines that are unresolved or marginally resolved.
  - **Quadrature**: The full composed model `(continuum + Σ flux_i * phi_i) * exp(-Σ tau_j * phi_j)` is evaluated at Gauss-Legendre quadrature nodes within each pixel and integrated via weighted sum with `n_nodes` nodes per pixel (default 7). This properly integrates `∫F * exp(-tau * phi)` over pixels, avoiding the pixel-integration approximation in analytic mode, at the cost of speed. Note: quadrature mode does not resolve the LSF pre-convolution approximation described below.
  - **Convolution**: The intrinsic model (LSF=0) is evaluated on a uniform fine sub-pixel grid of `n_super` points per pixel (default 10), numerically convolved with the wavelength-dependent Gaussian LSF via a banded spatially-varying kernel (`unite/_lsf.py`), then pixel-averaged. This correctly computes `LSF ⊗ [F · exp(-tau · phi_intrinsic)]`, eliminating the LSF pre-convolution approximation for absorption lines and applying LSF to all continuum forms (including non-polynomial). The kernel half-width `conv_half_width` is auto-computed at build time; the dominant cost is O(N_total × kernel_width) where N_total = n_pixels × n_super.
- **LSF pre-convolution of tau profiles (analytic and quadrature modes)**: In both integration modes, the absorption profile `phi(λ)` used in `exp(-tau * phi)` is the LSF-convolved profile, not the intrinsic one. The physically correct observable is `LSF ⊗ [F · exp(-tau · phi_intrinsic)]`, but computing this would require convolving the nonlinear product over the full multi-pixel LSF kernel — this is not supported. The approximation is accurate when (a) the absorber is well-resolved (intrinsic FWHM ≫ LSF FWHM) — phi_LSF ≈ phi_intrinsic — or (b) the line is optically thin (tau ≪ 1) — the integrand is linear in phi and convolution distributes. For **unresolved, optically thick absorbers** (e.g. narrow ISM lines at moderate spectral resolution) the approximation breaks down: the LSF-broadened profile has a lower peak than the intrinsic profile, so the code underestimates the absorption depth for a given tau, biasing inferred tau values high and misrepresenting the curve of growth.
- **Continuum LSF convolution**: Polynomial-based continuum forms (`Linear`, `Polynomial`, `Chebyshev`, `Bernstein`) are analytically convolved with the Gaussian LSF by transforming monomial coefficients using the Gaussian moment formula. Non-polynomial forms (`PowerLaw`, `BSpline`, `Blackbody`, `ModifiedBlackbody`, `AttenuatedBlackbody`) ignore the LSF — their curvature is assumed slowly varying at spectral resolution. For Chebyshev/Bernstein, the LSF FWHM is rescaled into the normalised coordinate domain before convolution. The default `ContinuumForm.integrate()` evaluates at pixel centres; subclasses may override for exact pixel integration.
