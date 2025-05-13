"""
Module for Handling LSF/lsf Curves
"""

# Typing
from typing import Final, Callable

# Import packages
import astropy.units as u
from astropy.table import Table

# Bayesian Inference
from numpyro import sample, deterministic as determ
from unite import priors

# JAX packages
from jax import jit, numpy as jnp

# Conversion factor from FWHM to sigma for variance = 1/2
SIGMA_TO_FWHM: Final[float] = 2 * jnp.sqrt(2 * jnp.log(2))


# Generic Calibration
def NIRSpecCalibration(names: list, fixed: list) -> None:
    """
    Initialize the calibration

    Parameters
    ----------
    spectra : Spectra
        Spectra Object

    Returns
    -------
    None
    """

    # Create Global LSF scale
    lsf_scale = sample('lsf_scale', priors.lsf_scale_prior())

    # Empty calibration dictionary
    calibration = {}

    # Iterate over spectra
    for name, fix in zip(names, fixed):
        if fix:
            # If fixed, deterministic offset and scale
            px_offset = determ(f'{name}_offset', 0)
            flux_scale = determ(f'{name}_flux', 1)

        else:
            # If not fixed, sample offset and scale
            px_offset = sample(f'{name}_offset', priors.pixel_offset_prior())
            flux_scale = sample(f'{name}_flux', priors.flux_scale_prior())

        # Add to calibration dictionary
        calibration[name] = (lsf_scale, px_offset, flux_scale)

    return calibration


# Interpolated lsf Curve
def InterpLSFCurve(lsf_file: str, λ_unit: u.Unit) -> Callable:
    """
    Initialize the lsf curve

    Parameters
    ----------
    lsf_file : str
        File containing the lsf curve
    λ_unit : u.Unit
        Wavelength target unit

    Returns
    -------
    None
    """

    # Load the lsf from file
    lsf_tab = Table.read(lsf_file)

    # Convert to JAX arrays in the correct units
    wave = jnp.array((lsf_tab['wave']).to(λ_unit))
    fwhm = jnp.array((lsf_tab['sigma']).to(λ_unit)) * SIGMA_TO_FWHM

    # Compute Interpolated lsf Curve
    @jit
    def lsf(λ, scale):
        lsf_interp = scale * jnp.interp(
            λ, wave, fwhm, left='extrapolate', right='extrapolate'
        )
        return lsf_interp

    return lsf


def PixelOffset(dispersion_file: str, λ_unit: u.Unit) -> Callable:
    """
    Return the pixel offset calibration function

    Parameters
    ----------
    dispersion_file : str
        File containing the dispersion curve
    λ_unit : u.Unit
        Wavelength target unit

    Returns
    -------
    Callable
        Pixel Offset Calibration Function
    """

    # Load the dispersion curve
    u.set_enabled_aliases(
        {'MICRONS': u.micron, 'PIXEL': u.pix, 'RESOLUTION': u.Angstrom / u.micron}
    )
    disp_tab = Table.read(dispersion_file)

    # Convert to JAX arrays in the correct units
    wave = jnp.array((disp_tab['WAVELENGTH']).to(λ_unit))
    disp = jnp.array((disp_tab['DLDS']).to(λ_unit / u.pix))

    # Compute Interpolated offset Curve
    @jit
    def pxoff(λ, offset):
        return offset * jnp.interp(
            λ, wave, disp, left='extrapolate', right='extrapolate'
        )

    return pxoff
