"""
Functions for initial parameter estimation
"""

from typing import Tuple

# Astronomy packages
from astropy import units as u, constants as consts

# Numerical packages
import numpy as np
import jax.numpy as jnp

# Spectra class
from unite import defaults
from unite.spectra import Spectra, Spectrum


def linesFluxesGuess(
    config: list,
    spectra: Spectra,
    cont_regs: jnp.ndarray,
    cont_guesses: jnp.ndarray,
    inner: u.Quantity = defaults.LINEPAD,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Guess the line fluxes for a given configuration

    Parameters
    ----------
    spectra : Spectra
        Spectra
    config : dict
        Configuration of emission lines
    cont_regs : jnp.ndarray
        Continuum regions
    cont_guesses : jnp.ndarray
        Continuum height guesses
    inner : u.Quantity, optional
        Inner region to compute the flux

    Returns
    -------
    tuple(jnp.ndarray, jnp.ndarray)
        Line centers and line flux guesses

    """

    # Convert to resolution
    inner = (inner / consts.c).to(u.dimensionless_unscaled).value

    # Compute the line centers and relative strengths
    centers, strengths = jnp.array(
        [
            (
                li['Wavelength'],
                li['RelStrength'] if li['RelStrength'] is not None else 1,
            )
            for g in config['Groups'].values()
            for s in g['Species']
            for li in s['Lines']
        ]
    ).T
    centers = jnp.array(
        u.Quantity(centers, config['Unit']).to(spectra.λ_unit)
    )  # Correct units

    # Compute the relevant continuum guesses
    opz = 1 + spectra.redshift_initial
    line_conts = cont_guesses[
        jnp.argmax(
            (cont_regs[:, 0][None, :] <= centers[:, None] * opz)
            & (opz * centers[:, None] <= cont_regs[:, 1][None, :]),
            axis=1,
        )
    ]

    # Get the guesses
    guesses = jnp.array(
        [
            max(
                [
                    lineFluxGuess(spectrum, center, line_cont, inner)
                    for spectrum in spectra.spectra
                ]
            )
            for center, line_cont in zip(centers, line_conts)
        ]
    )
    guesses = jnp.abs(guesses) / strengths  # Divide by strengths to normalize

    # For all lines that are tied, guess to the max value divided
    i = 0
    for group in config['Groups'].values():
        for species in group['Species']:
            species_guesses, species_inds = [], []
            for line in species['Lines']:
                if line['RelStrength'] is not None:
                    species_guesses.append(guesses[i])
                    species_inds.append(i)
                i += 1
            if species_guesses:
                species_guess = max(species_guesses)
                for ind in species_inds:
                    guesses = guesses.at[ind].set(species_guess)

    return centers, jnp.array(guesses)


# Line Flux Guess
def lineFluxGuess(
    spectrum: Spectrum, center: float, line_cont: float, inner: u.Quantity
) -> float:
    """
    Compute the line flux guess as the sum of the flux in the inner region minus the continuum region guess

    Parameters
    ----------
    spectrum : Spectrum
        Spectrum
    center : float
        Line center
    line_cont : float
        Continuum height guess
    inner : u.Quantity
        Inner region to compute the flux (R)

    Returns
    -------
    float
        Line flux guess
    """

    # Redshift the line
    innerwidth = (linewav := center * (1 + spectrum.redshift_initial)) * inner

    # Compute the mask
    imask = spectrum.coverage((linewav - innerwidth), (linewav + innerwidth))

    # Check if the mask is empty
    empty = not imask.any()
    if empty:
        imask = True

    # Estimate flux as maximum deviation from zero times the width of the region
    flux = (
        jnp.abs(spectrum.flux[imask]).max()
        * (spectrum.high[imask] - spectrum.low[imask])
    ).sum()

    # If mask is empty, negate the sign
    if empty:
        flux = -flux

    return flux


def computeContinuumRegions(
    config: list, spectra: Spectra, pad: u.Quantity = defaults.CONTINUUM
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the continuum regions from the configuration

    Parameters
    ----------
    config : dict
        Configuration of emission lines
    spectra : Spectra
        Spectra
    pad : u.Quantity, optional
        Region width around the lines

    Returns
    -------
    (np.ndarray, np.ndarray)
        Continuum regions and continuum height guesses
    """

    # Get lines from config
    lines = np.sort(
        [
            line['Wavelength']
            for group in config['Groups'].values()
            for species in group['Species']
            for line in species['Lines']
        ]
    ) * u.Unit(config['Unit'])

    # Compute pad in correct units
    pad = (pad / consts.c).to(u.dimensionless_unscaled).value

    # Generate continuum regions
    allregs = lines[:, np.newaxis] + np.array([-1, 1]) * (pad * lines)[:, np.newaxis]
    cont_regs = [allregs[0]]
    for region in allregs[1:]:
        if region[0] < cont_regs[-1][1]:
            cont_regs[-1][1] = region[1]
        else:
            cont_regs.append(region)

    # Convert to correct units and redshift
    cont_regs_rest = jnp.array(
        [cont_regs.to(spectra.λ_unit).value for cont_regs in cont_regs]
    )
    cont_regs_obs = cont_regs_rest * (1 + spectra.redshift_initial)

    return cont_regs_obs, continuumHeightGuesses(cont_regs_obs, config, spectra)


def continuumHeightGuesses(
    continuum_regions: jnp.ndarray,
    config: list,
    spectra: Spectra,
    linepad: u.Quantity = defaults.LINEPAD,
    sigma: float = 0,
) -> jnp.ndarray:
    """
    Guess the continuum height for different

    Parameters
    ----------
    spectra : Spectra
        Spectra
    continuum_regions : list
        List of continuum regions
    config : dict
        Configuration of emission lines
    linepad : u.Quantity, optional
        Padding to mask line
    sigma : float, optional


    Returns
    -------
    jnp.ndarray
        Array of continuum height guesses
    """

    # Return the updated config
    return jnp.array(
        [
            max(
                [
                    continuumHeightGuess(
                        config, continuum_regions, spectrum, linepad, sigma
                    )
                    for spectrum in spectra.spectra
                ]
            )
            for continuum_regions in continuum_regions
        ]
    )


# Continuum Height Guess
def continuumHeightGuess(
    config: dict,
    continuum_region: jnp.ndarray,
    spectrum: Spectrum,
    linepad: u.Quantity,
    sigma: float,
) -> jnp.ndarray:
    """
    Guess the continuum height for a spectrum

    Parameters
    ----------
    continuum_region : jnp.ndarray
        Boundary of the continuum region
    config : dict
        Configuration of emission lines
    linepad : u.Quantity
        Padding to mask line
    sigma : float
        Upper bound for median calculation

    Returns
    -------
    float
        Continuum Height Estimate
    """

    # Mask the lines
    mask = spectrum.maskLines(config, continuum_region, linepad)

    # If no coverage, return very large guess, but negative so it can be overwritten by other disperser.
    if mask.sum() == 0:
        return -jnp.abs(spectrum.flux + sigma * spectrum.err).max()

    # Compute the median Nsigma upper bound
    return jnp.median(spectrum.flux[mask] + sigma * spectrum.err[mask])
