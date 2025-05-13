"""
Module for defining the priors for the model parameters
"""

# Typing
from typing import Optional

# JAX packages
from jax import numpy as jnp

# Bayesian Inference
from numpyro import distributions as dist

# unite
from unite import defaults

# Convert Prior dictionaries to Arrays
flux = defaults.convertToArray(defaults.flux)
redshift = defaults.convertToArray(defaults.redshift)
fwhm = defaults.convertToArray(defaults.fwhm)


def fwhm_prior(
    linetypes: jnp.ndarray, orig: Optional[jnp.ndarray] = None
) -> dist.Distribution:
    """
    Return a fwhm prior based on the linetype

    Parameters
    ----------
    linetypes : jnp.ndarray
        Integer array of line types
    orig : jnp.ndarray, optional
        Original values for additional lines

    Return
    ------
    dist.Distribution
        Prior distribution
    """

    # Get the low and high bounds
    low, high = fwhm[linetypes].T

    # If there are original values, set the low and high bounds
    if orig is not None:
        # Broad line must be 100 km/s higher than the original
        low = jnp.where(
            jnp.logical_or(
                linetypes == defaults.LINETYPES['broad'],
                linetypes == defaults.LINETYPES['cauchy'],
            ),
            orig + 100,
            low,
        )

    return dist.Uniform(low=low, high=high)


def redshift_prior(
    linetypes: jnp.ndarray, orig: Optional[jnp.ndarray] = None
) -> dist.Distribution:
    """
    Return a redshift prior based on the linetype

    Parameters
    ----------
    linetypes : jnp.ndarray
        Integer array of line types
    orig : jnp.ndarray, optional
        Original values for additional lines

    Return
    ------
    dist.Distribution
        Prior distribution
    """

    # Get the low and high bounds
    low, high = redshift[linetypes].T

    # If there are original values, set the low and high bounds
    if orig is not None:
        # Outflow line must be blueshifted relative to the original
        high = jnp.where(linetypes == defaults.LINETYPES['outflow'], orig, high)

    return dist.Uniform(low=low, high=high)


def flux_prior(
    linetypes: jnp.ndarray, orig: Optional[jnp.ndarray] = None
) -> dist.Distribution:
    """
    Return a flux prior based on the linetype

    Parameters
    ----------
    linetypes : jnp.ndarray
        Integer array of line types
    orig : jnp.ndarray, optional
        Original values for additional lines

    Return
    ------
    dist.Distribution
        Prior distribution
    """

    # Get the low and high bounds
    low, high = flux[linetypes].T

    return dist.Uniform(low=low, high=high)


def angle_prior() -> dist.Distribution:
    """
    Return a uniform prior for the angle of the angle of the continuum

    Parameters
    ----------
    None

    Return
    ------
    dist.Distribution
        Prior distribution for the angle of the continuum
    """

    return dist.Uniform(low=-jnp.pi / 2, high=jnp.pi / 2)


def height_prior(height_guess: float) -> dist.Distribution:
    """
    Return a uniform prior for the height of the continuum

    Parameters
    ----------
    intercept_guess : float
        Initial guess for the height of the continuum

    Return
    ------
    dist.Distribution
        Prior distribution for the height of the continuum
    """

    low = jnp.where(height_guess < 0, 2 * height_guess, -height_guess)
    high = jnp.where(height_guess < 0, -2 * height_guess, 2 * height_guess)
    return dist.Uniform(low=low, high=high)


def lsf_scale_prior(
    mean: float = 1.2, sig: float = 0.1, cutoff: float = 3.0
) -> dist.Distribution:
    """
    Return a truncated normal prior for the lsf scale
    Centered on 1.2 with a standard deviation of 0.1, but truncated at 3σ

    Parameters
    ----------
    mean : float, optional
        Mean of the prior
    sig : float, optional
        Standard deviation of the prior
    cutoff : float, optional
        Sigma cutoff for the prior


    Return
    ------
    dist.Distribution
        Prior distribution for the lsf scale
    """

    return dist.TruncatedNormal(
        loc=mean, scale=sig, low=mean - cutoff * sig, high=mean + cutoff * sig
    )


def pixel_offset_prior(mean: float = 0.2, half_width: float = 0.5) -> dist.Distribution:
    """
    Return a uniform prior for the pixel offset

    Parameters
    ----------
    mean : float, optional
        Mean of the prior
    half_width : float, optional
        Half width of the prior

    Return
    ------
    dist.Distribution
        Prior distribution for the pixel offset
    """

    return dist.Uniform(low=mean - half_width, high=mean + half_width)


def flux_scale_prior(mean=1.1, sig=0.2, cutoff=3.0) -> dist.Distribution:
    """
    Return a truncated normal prior for the flux scale

    Parameters
    ----------
    mean : float, optional
        Mean of the prior
    sig : float, optional
        Standard deviation of the prior
    cutoff : float, optional
        Sigma cutoff for the prior

    Return
    ------
    dist.Distribution
        Prior distribution for the flux scale
    """

    return dist.TruncatedNormal(
        loc=mean, scale=sig, low=mean - cutoff * sig, high=mean + cutoff * sig
    )
