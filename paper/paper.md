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
  - name: Raphael E. Hviding
    orcid: 0000-0002-4684-9005
    corresponding: true
    affiliation: 1
  - name: Anna de Graaff
    orcid: 0000-0002-2380-9801
    corresponding: false
    affiliation: "2, 1"
  # - name: Ivo Labbé
  #   orcid: 0000-0002-2057-5376
  #   corresponding: false
  #   affiliation: 3
  - name: Rohan P. Naidu
    orcid: 0000-0003-3997-5705
    corresponding: false
    affiliation: 4
  - name: Helena Treiber
    orcid: 0000-0003-0660-9776
    corresponding: false
    affiliation: 5
  - name: Rana Osman
    orcid: 0000-0002-6807-9611
    corresponding: false
    affiliation: 5
  - name: Jenny Greene
    orcid: 0000-0002-5612-3427
    corresponding: false
    affiliation: 5
  - name: Alberto Torralba
    orcid: 0000-0001-5586-6950
    corresponding: false
    affiliation: 6
  - name: Bingjie Wang
    orcid: 0000-0001-9269-5046
    corresponding: false
    affiliation: 5
  - name: Morgan Fouesneau
    orcid: 0000-0001-9256-5516
    corresponding: false
    affiliation: 1
affiliations:
  - name: Max-Planck-Institut für Astronomie
    index: 1
    ror: "01vhnrs90"
  - name: Center for Astrophysics Harvard & Smithsonian
    index: 2
    ror: "03c3r2d17"
  # - name: Swinburne University of Technology
  #   index: 3
  #   ror: "031rekg67"
  - name: MIT Kavli Institute
    index: 4
    ror: "042nb2s44"
  - name: Princeton University
    index: 5
    ror: "00hx57361"
  - name: Institute of Science and Technology Austria
    index: 6
    ror: "03gnh5541"


date: X May 2026
bibliography: paper.bib
---

# Summary

Astronomical spectroscopy, whereby light from celestial sources is dispersed into its constituent wavelengths (colors), is a cornerstone of modern astrophysical research.
In particular, spectral lines arising from atomic and molecular transitions in astrophysical gas encode fundamental physical properties such as redshift, chemical composition, temperature, density, and kinematics, making the flexibility, speed, and accuracy of spectral line-fitting tools directly impactful on the scientific return of spectroscopic observations.

`unite` (Unified liNe Integration Turbo Engine) is a Python package for fast and accurate Bayesian inference of spectral lines and continua with flexible configurations from one or more spectra simultaneously.
It is built on JAX [@bradbury2018jax], NumPyro [@phan2019numpyro; @bingham2019pyro], and Astropy [@astropy2013package; @astropy2018project; @astropy2022sustaining].

# Statement of need

Making the most of the growing volume and diversity of spectroscopic observations necessitates joint analysis across all available data to fully exploit their collective constraining power and deliver accurate measurements with representative uncertainties.
This requires enforcing shared astrophysical parameters while accounting for and propagating systematic uncertainties in instrument-specific calibrations. 
An additional challenge arises when the data are undersampled, as evaluating the model at pixel centers rather than integrating over the pixel domain introduces systematic errors in recovered line shapes, yet existing tools that account for this typically do so via computationally expensive supersampling and convolution, limiting their scalability to large datasets and heterogeneous, multi-spectrograph configurations.

The Near-Infrared Spectrograph (NIRSpec; @jakobsen2022nirspec) on the James Webb Space Telescope (JWST; @gardner2023jwst) is a prime example where all of these challenges arise simultaneously: it critically undersamples the LSF across all gratings and observing modes, observations routinely span multiple gratings with complementary strengths (high-resolution to measure kinematics, low-resolution to constrain fluxes and continuum shapes), and systematic uncertainties in resolving power, absolute normalization, and wavelength solution have measurable impacts on the data [@degraaff2025rubies].

`unite` is designed to address these challenges while providing a flexible, extensible, and reproducible framework for spectroscopic analysis applicable to any spectrograph, rigorously carrying instrumental systematics through to the final parameter uncertainties.
By leveraging optimized libraries for probabilistic programming and automatic differentiation, `unite` delivers scientifically rigorous Bayesian inference without compromising on computational efficiency, enabling the analysis of large and mixed spectroscopic datasets with accurate error estimation.

# Software design

Spectral fitting routines typically assume that the model evaluated at the pixel center is a good representation of the average of the model over the pixel domain, which is what the instrument actually records. 
This approximation is well justified when the spectrum is critically (Nyquist) sampled or over-sampled, i.e. when the signal changes slowly over the pixel domain, but breaks down when the spectrum is undersampled and the signal changes rapidly over the pixel domain.
In the case of NIRSpec, a point source is undersampled by a factor of $\sim 1.5$ [@graaff2024nirspec], leading to a bias of $10-20\%$ in recovered line widths if not properly accounted for.
This can be addressed by integrating the model over the pixel domain, providing the exact solution for the observed signal regardless of the degree of undersampling.
We rely on the assumption that the LSF is well approximated by a Gaussian kernel, which is representative of NIRSpec [@shajib2025nirspec] and many other spectrographs.

`unite` computes the integrals of continua and line models analytically where possible. However, analytic pixel integration is not possible for all model setups, in particular in the presence of optical-depth parametrized absorption lines where the nonlinear transmission $e^{-\tau\phi}$ couples the line depth and profile shape in a way that prevents closed-form solutions for the pixel integrals.
In these cases, `unite` provides a numerical convolution mode which supersamples the intrinsic model on a fine wavelength grid and convolves with the wavelength-dependent LSF kernel to produce a pixel-convolved model. 
This model accurately captures the nonlinear coupling of line depth and profile shape, but is more computationally expensive than the analytic integration mode; users can choose the appropriate mode for their application.

At its core, `unite` is a domain-specific language for building probabilistic models of spectroscopic data.
Users build a declarative configuration of line and continuum components and assign priors to physical parameters via token instances that can be shared across multiple model components and combined arithmetically.
The prior system is expressive: priors on any parameter can depend on other parameters through arithmetic expressions and topologically sorted dependency chains, enabling physical constraints such as requiring a broad component's velocity width to exceed that of a narrow component by a certain amount, or fixing flux ratios between doublet lines.
All configurations are serializable to human-readable YAML for reproducibility and sharing.

In addition, users specify the instrumental configuration carrying empirical calibrations of the wavelength-dependent resolving power, pixel scale, and flux normalization for each disperser, which can be shared across instruments. 
One aspect that sets `unite` apart from other spectral fitting tools is that it treats instrumental calibration parameters as first-class citizens in the inference process; priors can be specified on each of the aforementioned calibrations and are sampled jointly with astrophysical parameters, allowing for systematic instrumental uncertainties to directly propagate to the inferred properties.
For example, when fitting NIRSpec data, users can incorporate the empirically measured wavelength and flux calibration offsets from @degraaff2025rubies as priors to obtain realistic uncertainty estimates on fluxes and kinematics by marginalizing over instrumental uncertainties. 

`unite` leverages NumPyro's probabilistic programming framework, built on JAX for automatic differentiation, JIT compilation, and native GPU support, to assemble an inference model compatible with a wide range of samplers: SVI for quick fits, NUTS for full posteriors, or nested sampling for model comparison.
Finally, `unite` provides convenience functions for extracting results as parameter tables and per-spectrum model predictions into domain-appropriate FITS files, carrying physical units throughout, via Astropy.

The package is publicly available on GitHub and PyPI under the GPL-3.0-or-later license, with a DOI minted via Zenodo [@hviding2026unite] and accompanied by Sphinx documentation including narrative guides, an API reference, and executable tutorials. CI/CD workflows ensure that the code is tested and documented with each update, and the project is open to feedback and contributions from the community.

# State of the field

Spectral line analysis is among the most common operations in observational astronomy, and the landscape of fitting software is correspondingly rich.

`pPXF` [@cappellari2004ppxf; @cappellari2017ppxf; @cappellari2023ppxf] is the standard tool for stellar kinematics and stellar-population recovery, fitting observed galaxy spectra as non-negative linear combinations of SSP templates and emission lines convolved with a parametric line-of-sight velocity distribution (LOSVD), using an analytic Fourier representation of the LOSVD to accurately recover kinematics even when the LOSVD is smaller than the instrumental resolution.

`PySpecKit` [@ginsburg2022pyspeckit] is a general-purpose spectral fitting toolkit supporting a range of profile shapes and optional MCMC uncertainty estimation.

`LiMe` [@fernandez2024lime] is a modern library designed for large and complex datasets including JWST spectra, with batch fitting across many spectra and flexible profile shapes.

`Q3DFIT` [@rupke2014ifsfit; @rupke2021questfit; @rupke2023q3dfit] targets IFU spectroscopy of quasar host galaxies, fitting multi-component Gaussian profiles via least squares with the stellar continuum pre-subtracted using `pPXF`.

`BADASS` [@sexton2021badass; @sexton2024badass] is a comprehensive Bayesian emission-line code to simultaneously infer AGN power-law continuum, FeII pseudo-continuum, stellar LOSVD (via `pPXF`), and multi-component Gaussian emission lines in a single posterior.

Many of these codes rely on least-squares optimization (`LMFIT` [@newville2014lmfit] or `scipy.optimize` [@virtanen2020scipy]), which does not yield posterior distributions, or on Bayesian inference via `emcee` [@foremanmackey2013emcee] or `pymc` [@abrilpla2023pymc]. 
`unite` is more closely comparable to the latter category of Bayesian tools and builds natively on JAX and NumPyro, inheriting JIT compilation, automatic differentiation, and GPU support, enabling scalable inference across large spectroscopic samples.

# Research impact statement

`unite` has already been used in several published JWST/NIRSpec analyses, demonstrating its utility for emission line characterization. 

- **Accurate Line Fluxes and Kinematics**: By accurately accounting for undersampling, `unite` has been used to robustly characterize fluxes and kinematics for both emission lines and absorption features in high-redshift galaxies, providing insights into the physical processes driving star formation and black hole growth. [@degraaff2025bhstar; @naidu2025bhstar; @sun2026bhstar; @wang2025photons; @wang2026water]
- **Multi-Component Line Fitting**: `unite` has been used to identify broad Balmer emission (broad H$\alpha$, H$\beta$) in high-redshift ($z \gtrsim 4$) galaxies observed with NIRSpec, providing evidence for active supermassive black hole growth in the early universe. Individual dispersers often lack sufficient signal-to-noise for such detections, making multi-disperser joint fitting critical; by coupling flux constraints from low-resolution prism and kinematic constraints from medium-resolution grating observations, `unite` enables robust detections of these components. [@hviding2025rubies; @degraaff2025bhstar; @hviding2026xraydot]
- **Redshift Precision**: `unite` was used in the analysis of MoM-z14, which at the time of publication was (and is currently) the most distant spectroscopically confirmed galaxy. Even with faint, marginally detected emission lines, jointly fitting line parameters and accounting for undersampling yielded a factor of 5 improvement in redshift precision over continuum estimates alone. [@naidu2026cosmic]

# AI usage disclosure

Claude Code (Anthropic) was used as an AI coding assistant during the development of `unite`, its documentation, and the drafting of this paper. 
It was used to generate new code, refactor existing code, and write certain sections of the documentation. 
All scientific decisions, algorithmic design choices, and final content were made and verified by the authors.

# Acknowledgements

REH acknowledges support by the German Aerospace Center (DLR) and the Federal Ministry for Economic Affairs and Energy (BMWi) through program 50OR2403 `RUBIES'. 
AdG acknowledges support from a Clay Fellowship awarded by the Smithsonian Astrophysical Observatory.
RPN thanks Neil Pappalardo and Jane Pappalardo for their generous support of the MIT Pappalardo Fellowships in Physics.
BW acknowledges support provided by NASA through Hubble Fellowship grant HST-HF2-51592.001 awarded by the Space Telescope Science Institute, which is operated by the Association of Universities for Research in Astronomy, Inc., for NASA, under the contract NAS 5-26555.

We acknowledge the support of the Data Science Group at the Max Planck Institute for Astronomy (MPIA) and especially Morgan Fouesneau and Ivelina Momcheva for their invaluable assistance providing high-level feedback for the development of the core `unite` routines.

# References
