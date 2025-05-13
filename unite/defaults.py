"""
Default Values
"""

# Typing
from typing import Final

# Astropy packages
from astropy import units as u

# JAX packages
import jax.numpy as jnp

# Constants used for line detection, line padding, and continuum regions
LINEDETECT: Final[u.Quantity] = 1_000 * (u.km / u.s)
LINEPAD: Final[u.Quantity] = 3_500 * (u.km / u.s)
CONTINUUM: Final[u.Quantity] = 15_000 * (u.km / u.s)

# Dictionary that defines mapping from integers to line types
linetypes: list = ['narrow', 'broad', 'cauchy', 'absorption', 'emission', 'outflow']
LINETYPES: Final[dict] = {line: i for i, line in enumerate(linetypes)}

# Define the Flux priors (scale relative to the guess)
flux: Final[dict[str, tuple[float]]] = {
    'narrow': (-2, 2),
    'broad': (0, 3),
    'cauchy': (0, 3),
    'absorption': (-2, 0),
    'emission': (0, 2),
    'outflow': (-2, 2),
}


# Define the Redshift priors in dimensionless units
δz: Final[float] = 0.005
redshift: Final[dict[str, tuple[float]]] = {
    'narrow': (-δz, δz),
    'broad': (-2 * δz, 2 * δz),
    'cauchy': (-2 * δz, 2 * δz),
    'absorption': (-3 * δz, 3 * δz),
    'emission': (-δz, δz),
    'outflow': (-δz, 2 * δz),
}

# Define the Dispersion priors in km/s
fwhm: Final[dict[str, tuple[float]]] = {
    'narrow': (0, 750),
    'broad': (250, 2500),
    'cauchy': (250, 2500),
    'absorption': (0, 1000),
    'emission': (0, 1000),
    'outflow': (150, 2500),
}


def convertToArray(priorDict: dict[str, tuple[float]]) -> jnp.ndarray:
    """
    Convert dictionary of priors to JAX Array

    Parameters
    ----------
    priorDict : dict[str, tuple[float]]
        Dictionary of prior values

    Returns
    -------
    jnp.ndarray
        Array of prior values
    """

    # Initialize the array
    out = jnp.zeros((len(set(priorDict)), 2))

    # Fill the array
    for key, val in priorDict.items():
        out = out.at[LINETYPES[key]].set(val)

    return out
