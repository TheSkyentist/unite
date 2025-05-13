"""
Utility Functions
"""

# Typing
from typing import List

# Import packages
import copy

# Astronomy packages
from astropy import units as u, constants as consts

# Numerical packages
import jax.numpy as jnp

# Spectra class
from unite import defaults
from unite.spectra import Spectra


def restrictConfig(
    config: dict, spectra: Spectra, linedet: u.Quantity = defaults.LINEDETECT
) -> List:
    """
    Restrict the configuration to only include lines that are covered by the spectra

    Parameters
    ----------
    config : dict
        Configuration of emission lines
    linedet : u.Quantity, optional
        Padding around the lines necessary to cover the line
        In velocity space

    Returns
    -------
    list
        Updated configuration
    """

    # Set the default linetype as narrow
    for group in config['Groups'].values():
        for species in group['Species']:
            if 'LineType' not in species:
                species['LineType'] = 'narrow'

    # Add additional components
    new_config = copy.deepcopy(config)
    for group in config['Groups'].values():
        for species in group['Species']:
            if 'AdditionalComponents' in species:
                # Iterate over additional components
                for comp, dest in species['AdditionalComponents'].items():
                    # Add it to the correct group
                    new_group = new_config['Groups'][dest]

                    # Get copy of species
                    new_species = copy.deepcopy(species)

                    # Remove additional component and add LineType
                    new_species.pop('AdditionalComponents')
                    new_species['LineType'] = comp

                    # Add the new species
                    new_group['Species'].append(new_species)

    # Initialize dictionary
    config = copy.deepcopy(new_config)

    # Effective resolution
    lineres = (linedet / consts.c).to(u.dimensionless_unscaled).value

    # Loop over config
    new_groups = {}
    for gname, group in config['Groups'].items():
        new_species = []
        for species in group['Species']:
            new_lines = []
            for line in species['Lines']:
                # Compute line wavelength
                linewav = (line['Wavelength'] * u.Unit(config['Unit'])).to(
                    spectra.λ_unit
                )

                # Redshift the line
                linewav = linewav * (1 + spectra.redshift_initial)
                linewidth = linewav * lineres

                # Compute boundaries
                low, high = (linewav - linewidth).value, (linewav + linewidth).value

                # Check coverage
                if jnp.logical_or.reduce(
                    jnp.array([s.coverage(low, high).any() for s in spectra.spectra])
                ):
                    new_lines.append(line)

            # Add species only if it has remaining lines
            if new_lines:
                species['Lines'] = new_lines
                new_species.append(species)

        # Add group only if it has remaining species
        if new_species:
            group['Species'] = new_species
            new_groups[gname] = group

    # Return the updated config
    return config


# TODO: Maybe start with what we have from GELATO?
def validateConfig(config: dict) -> None:
    """
    Validate configution file is valid.
    Will raise and error if it is not

    Parameters
    ----------
    config : dict
        Configuration of emission lines

    Returns
    -------
    None
    """
    if False:
        raise  # What kind of error?
        exit(1)
    return
