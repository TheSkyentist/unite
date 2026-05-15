---
title: 'unite: Unified liNe Integration Turbo Engine'
tags:
  - Python
  - astronomy
  - spectroscopy
  - JAX
  - NumPyro
  - inference
authors:
  - name: Raphael Erik Hviding
    orcid: 0000-0002-4684-9005
    corresponding: true
    affiliation: 1
authors:
  - name: Alberto Torralba
    orcid: 0000-0001-5586-6950
    corresponding: false
    affiliation: 2
affiliations:
  - name: Max-Planck-Institut für Astronomie
    index: 1
    ror: "01vhnrs90"
  - name: "Institute of Science and Technology Austria"
    index: 2
    ror: 03gnh5541"
date: X May 2026
bibliography: paper.bib
---

# Summary

Astronomical spectroscopy, whereby light from celestial sources is dispersed into its constituent wavelengths (colors), is a cornerstone of modern astrophysical research.
In particular the characterization of spectral lines, which arise from atomic and molecular transitions in astrophysical gas, allows astronomers to measure fundamental physical properties such as redshift, chemical composition, temperature, density, and kinematics of the emitting or absorbing material. 
Therefore, the flexibility, speed, and accuracy of spectral line-fitting tools directly impact the scientific return of spectroscopic observations.

`unite` (Unified liNe Integration Turbo Engine) is a Python package for fast and accurate Bayesian inference of spectral line features from one or more spectra simultaneously.
It is built primarily on JAX [@bradbury2018jax] (for speed and automatic differentiation), NumPyro [@phan2019numpyro; @bingham2019pyro] (for probabilistic programming), and Astropy [@astropy2013package; @astropy2018project; @astropy2022sustaining] (for units and FITS handling), and is designed to be flexible and extensible to a wide range of spectroscopic applications.

# Statement of need

The Near-Infrared Spectrograph (NIRSpec; @jakobsen2022nirspec) on the James Webb Space Telescope (JWST; @gardner2023jwst) is the frontier instrument for near-infrared spectroscopy and has already provided revolutionary insights across a wide range of astrophysical topics, from the characterization of exoplanet atmospheres to the discovery of the most distant galaxies known.
However, a defining challenge of NIRSpec spectroscopy is that the detector critically undersamples the instrumental line spread function (LSF) across all gratings and observing modes.
Therefore, evaluating the model at pixel centers rather than integrating over the pixel domains introduces systematic errors in recovered fluxes and line widths, which can bias scientific inferences.
When existing spectral line fitting tools can account for this, they typically do so via computationally and memory expensive super-sampling and convolution.
In addition, typical JWST/NIRSpec programs observe targets with multiple gratings to leverage their complementary strengths (i.e. high-resolution to measure kinematics, low-resolution to constrain fluxes and continuum shapes), so a statistically optimal analysis must fit all gratings simultaneously with shared astrophysical parameters and grating-specific calibration offsets.
`unite` is designed to address both of these challenges while additionally providing a flexible, extensible framework for spectroscopic analysis that can be applied to any instrument with user-supplied resolving power and pixel-scale functions.
By relying on existing optimized libraries for probabilistic programming and automatic differentiation, `unite` delivers scientifically rigorous Bayesian inference without compromising on speed, enabling the analysis of large spectroscopic datasets with accurate uncertainty quantification.
The target audience is observational astronomers, especially those working with JWST/NIRSpec data, that want to do inference over large spectroscopic samples, deal with undersampled data, and/or fit multiple spectra simultaneously with shared parameters.

# Software design

When fitting spectroscopic data sets, fitting routines typically the assumption that the model evaluated at the pixel center is a good representation of the average of the model over the pixel domain, which is what the instrument actually measures. 
This approximation is well justified when the spectrum is critically sampled or over-sampled, i.e. when the signal changes slowly over the pixel domain, but breaks down when the spectrum is undersampled and the signal changes rapidly over the pixel domain, as is the case for NIRSpec.
This can be addressed integrating the model over the pixel domain, providing the exact solution for the observed signal regardless of the degree of undersampling.
We rely on the assumption that the LSF is well approximated by a Gaussian kernel, which is a good approximation for NIRSpec and many other spectrographs, especially in the undersampled regime.

`unite` computes the integrals of continua and line models analytically where possible. However, analytic pixel integration is not possible for all model setups, in particular in the presence of optical-depth parametrized absorption lines where the nonlinear transmission $e^{-\tau\phi}$ couples the line depth and profile shape in a way that prevents closed-form solutions for the pixel integrals.
In these cases, `unite` provides two additional integration modes: quadrature mode, which evaluates the full model at Gauss-Legendre nodes within each pixel and weights by the corresponding quadrature weights; and convolution mode, which supersamples the intrinsic model on a fine wavelength grid and convolves with the wavelength-dependent LSF kernel to produce a pixel-convolved model. 
Both of these modes handle the nonlinear coupling to different degrees of accuracy and computational cost, and users can choose the appropriate mode for their specific application.

Despite implementing integration, quadrature, and convolution modes, `unite` is fast and efficient thanks to its JAX backend, which provides just-in-time (JIT) compilation and native GPU support.
At its core, `unite` is a domain-specific language for building probabilistic models of spectroscopic data.
Users build a declarative configuration of line and continuum components, assign priors to physical parameters via token instances, which can be shared across multiple model components with arithmetic combinations. 
In addition, users specify the instrumental configuration carrying empirical calibrations of the wavelength-dependent resolving power, pixel scale, and flux normalization for each disperser, which can be shared across instruments. 
One aspect that sets `unite` apart from other spectral fitting tools is that it treats instrumental calibration parameters as first-class citizens in the inference process; pixel offsets, resolution scales, and flux normalizations can be directly incorporated into the model with priors and sampled jointly with astrophysical parameters, allowing for instrumental uncertainties to directly propagate to the inferred properties.

All configurations are serializable to human-readable YAML for reproducibility and sharing.
`unite` assembles a NumPyro probabilistic model for inference with any compatible sampler, including SVI for quick exploratory fits, NUTS for full posterior sampling, and nested sampling for model comparison and evidence calculation.
Finally, `unite` provides convenience functions for extracting results as parameter tables and per-spectrum model predictions into domain-appropriate FITS HDU lists.

The package is publicly available on GitHub and PyPI under the GPL-3.0-or-later license, with a DOI minted via Zenodo [@hviding2026unite] and accompanied by Sphinx documentation including narrative guides, an API reference, and executable tutorials. CI/CD workflows ensure that the code is tested and documented with each update, and the project is open to feedback and contributions from the community.

# State of the field

Spectral line analysis is among the most common operations in observational astronomy, and the landscape of fitting software is correspondingly rich.

The dominant paradigm for emission-line fitting is non-linear least squares, with most Python codes built on `LMFIT` [@newville2014lmfit] or `scipy.optimize` [@virtanen2020scipy] directly.

`pPXF` [@cappellari2004ppxf; @cappellari2017ppxf; @cappellari2023ppxf] is the standard tool for stellar kinematics and stellar-population recovery, fitting observed galaxy spectra as non-negative linear combinations of SSP templates and emission lines convolved with a parametric line-of-sight velocity distribution with optional supersampling. 

`PySpecKit` [@ginsburg2022pyspeckit] is a general-purpose spectral fitting toolkit supporting a range of profile shapes and optional MCMC uncertainty estimation.

`LiMe` [@fernandez2024lime] is a modern library designed for large and complex datasets including JWST spectra, with batch fitting across many spectra and flexible profile shapes.

`Q3DFIT` [@rupke2014ifsfit; @rupke2021questfit; @rupke2023q3dfit] targets IFU spectroscopy of quasar host galaxies, fitting multi-component Gaussian profiles via least squares with the stellar continuum pre-subtracted using `pPXF`.

`BADASS` [@sexton2021badass; @sexton2024badass] is a comprehensive Bayesian emission-line code to simultaneously infer AGN power-law continuum, FeII pseudo-continuum, stellar LOSVD (via `pPXF`), and multi-component Gaussian emission lines in a single posterior.

These codes occupy various niches but none provide pixel integration or first-class joint multi-grating fitting. 
In addition, the fitting paradigm for many of these codes is least-squares optimization, which does not provide the posterior distribution over parameters. 
For those that do provide Bayesian inference, they typically rely on `emcee` [@foremanmackey2013emcee] or `pymc3` [@abrilpla2023pymc] which can struggle in high-dimensional parameter spaces and do not natively support (GPU) acceleration.
However many incorporate templates into their fitting frameworks, which `unite` does not currently support. 

# Research impact statement

`unite` has already been used in several published JWST/NIRSpec analyses, demonstrating its utility for emission line characterization. 

- **Accurate Line Fluxes and Kinematics**: With the accurate accounting for undersampling, `unite` has been used to robustly characterize fluxes and kinematics for both emission lines and absorption features in high-redshift galaxies, providing insights into the physical processes driving star formation and black hole growth. [@degraaff2025bhstar; @naidu2025bhstar; @sun2026bhstar; @wang2025photons; @wang2026water]
- **Multi-Component Line Fitting**: `unite` has been used to identify broad Balmer emission (broad H$\alpha$, H$\beta$) in high-redshift ($z \gtrsim 4$) galaxies observed with NIRSpec, providing evidence for active supermassive black hole growth in the early universe. Multi-grating joint fitting is critical as grating spectroscopy often lacks in signal-to-noise but by coupling physical constraints on flux from low-resolution gratings, `unite` enables robust detections of these components. [@hviding2025rubies; @degraaff2025bhstar; @hviding2026xraydot]
- **Redshift Precision**: `unite` was used in the analysis of MoM-z14, which at the time of publication was (and remains) the most distant spectroscopically confirmed galaxy known. Coupling line parameters and properly accounting for undersampling allowed for a factor of 5 improvement in redshift precision than from continuum estimates alone even with faint, marginally detected, emission lines. [@naidu2026cosmic]

# AI usage disclosure

Claude Code (Anthropic) was used as an AI coding assistant during the development of `unite`, its documentation, and the drafting of this paper. 
It was used to generate new code, refactor existing code, and write certain sections of the documentation. 
All scientific decisions, algorithmic design choices, and final content were made and verified by the authors.

# Acknowledgements

REH acknowledges support by the German Aerospace Center (DLR) and the Federal Ministry for Economic Affairs and Energy (BMWi) through program 50OR2403 `RUBIES'. 

# References
