"""
Compuation of parameter matrices
"""

# Typing
from typing import Dict, Tuple, List

# Numerical packages
import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

# unite
from unite import defaults


def configToMatrices(
    config: dict,
) -> Tuple[
    Tuple[List[BCOO], List[BCOO], List[BCOO]],
    Tuple[jnp.ndarray, List[jnp.ndarray], List[jnp.ndarray]],
]:
    """
    Convert the configuration to sparse matrices for the model

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    matrices:
        Tuple of parameter matrices for the model
    linetypes:
        Tuple of line types for the model

    """

    # Keep track of total line index
    i = 0

    # Keep track of unique index (i) and index pair (inds) between unique and total
    i_f, f_inds, fluxes = 0, {}, []  # for flux also keep track of the ratio
    i_z, z_inds = 0, {}
    i_σ, σ_inds = 0, {}

    # Iterate over groups, species, and lines
    linetypes = []
    for group in config['Groups'].values():
        for species in group['Species']:
            # Check if any fluxes in species are tied
            if not all([line['RelStrength'] is None for line in species['Lines']]):
                # Assign special index for tied fluxes
                species_f = i_f
                i_f += 1  # Increment for not tied fluxes
            # Iterate over lines
            for line in species['Lines']:
                # Keep track of lineTypes
                linetypes.append(species['LineType'])

                # Keep track of nonzero matrix elements
                z_inds[i] = i_z
                σ_inds[i] = i_σ

                # Associate line with it's total index
                line['Index'] = i

                # If the flux is not tied, increment
                if line['RelStrength'] is None:
                    fluxes.append(1)
                    f_inds[i] = i_f
                    i_f += 1
                else:
                    fluxes.append(line['RelStrength'])
                    f_inds[i] = species_f

                # Increment line index
                i += 1

            # If Group is not tied, increment
            if not group['TieRedshift']:
                i_z += 1
            if not group['TieDispersion']:
                i_σ += 1

        # Increment between groups if we didn't already and species is not empty
        if group['Species']:
            i_z += 1
        if group['Species']:
            i_σ += 1

    # Iterate again to find origin for each additional component
    add_inds = {}
    for group in config['Groups'].values():
        for species in group['Species']:
            for line in species['Lines']:
                # Check if there are additional components
                if 'AdditionalComponents' in species:
                    # Iterate over additional components
                    for comp, dest in species['AdditionalComponents'].items():
                        # Iterate again to find the additional components
                        for addSpecies in config['Groups'][dest]['Species']:
                            # Ensure additional component matches the species
                            if (
                                addSpecies['Name'] != species['Name']
                                or addSpecies['LineType'] != comp
                            ):
                                continue

                            # Iterate until we find the right line
                            for addLine in addSpecies['Lines']:
                                if not addLine['Wavelength'] == line['Wavelength']:
                                    continue
                                add_inds[addLine['Index']] = line['Index']

    # For now, sigma's always decouple from their parent group
    # Maybe tie this to linetype in the future?
    # Fluxes can never be tied since we always create a new species for fluxes
    # z_translation = {}
    σ_translation = {}
    for i_add, i_orig in add_inds.items():
        # Check if we should decouple z
        # if (z_inds[i_add] == z_inds[i_orig]) and (linetypes[i_add] in []):
        #     # Update translation to first available index
        #     if z_inds[i_add] not in z_translation:
        #         z_translation[z_inds[i_add]] = i_z
        #         i_z += 1
        #     # Update link
        #     z_inds[i_add] = z_translation[z_inds[i_add]]

        # Check if we should decouple σ
        if (σ_inds[i_add] == σ_inds[i_orig]) and (True):  # (linetypes[i_add] in []):
            # Update translation to first available index
            if σ_inds[i_add] not in σ_translation:
                σ_translation[σ_inds[i_add]] = i_σ
                i_σ += 1
            # Update link
            σ_inds[i_add] = σ_translation[σ_inds[i_add]]

    # Split into the origin components
    orig = [
        {i: j for i, j in inds.items() if (i not in add_inds)}
        for inds in (f_inds, z_inds, σ_inds)
    ]

    # Add additional components that are tied to the origin
    for o, inds in zip(orig, (f_inds, z_inds, σ_inds)):
        for key, val in inds.items():
            if key not in add_inds:
                continue
            if val in o.values():
                o[key] = val

    # Split into additional components
    add = [
        {i: j for i, j in inds.items() if ((i in add_inds) and (j not in o.values()))}
        for inds, o in zip((f_inds, z_inds, σ_inds), orig)
    ]

    # Don't have to worry about this for fluxes.
    fluxes_org = [f for i, f in enumerate(fluxes) if i not in add_inds]
    fluxes_add = [f for i, f in enumerate(fluxes) if i in add_inds]

    # Re-index unique values after the split
    orig = [reIndex(oi) for oi in orig]
    add = [reIndex(oi) for oi in add]

    # Mapping from unique origin indices to unique addition indices
    orig_add = [
        {ai[i_add]: oi[i_orig] for i_add, i_orig in add_inds.items() if i_add in ai}
        for oi, ai in zip(orig, add)
    ]

    # Convert linetypes to integers
    linetypes = jnp.array([defaults.LINETYPES[lt] for lt in linetypes])

    # Linetype in original components
    linetypes_orig = []
    for o in orig:
        new_array = np.full(max(o.values()) + 1, -1)
        for k, v in o.items():
            if new_array[v] == -1:
                new_array[v] = linetypes[k]
        linetypes_orig.append(jnp.array(new_array))

    # Linetype in additional components
    linetypes_add = []
    for a in add:
        if len(a) == 0:
            linetypes_add.append(jnp.array([]))
            continue
        new_array = np.full(max(a.values()) + 1, -1)
        for k, v in a.items():
            if new_array[v] == -1:
                new_array[v] = linetypes[k]
        linetypes_add.append(jnp.array(new_array))

    # Create flux matrices
    f_orig = BCOO(
        (fluxes_org, list(orig[0].items())),
        shape=(i, max(orig[0].values()) + 1 if len(orig[0]) else 1),
    ).T
    f_add = (
        BCOO((fluxes_add, list(add[0].items())), shape=(i, max(add[0].values()) + 1)).T
        if len(add[0])
        else jnp.zeros((0, i))
    )

    # Create the other matrices
    orig, add = [
        [
            BCOO(
                (jnp.ones(len(ind), int), list(ind.items())),
                shape=(i, max(ind.values()) + 1),
            ).T
            if len(ind)
            else jnp.zeros((0, i))
            for ind in inds
        ]
        for inds in (orig[1:], add[1:])
    ]

    # Concatenate the flux matrices
    orig = [f_orig] + orig
    add = [f_add] + add

    # Create the origin to additional matrices
    orig_add = [
        BCOO(
            (jnp.ones(len(ind), int), list(ind.items())), shape=(a.shape[0], o.shape[0])
        ).T
        if len(ind)
        else jnp.zeros((0, o.shape[0]))
        for ind, o, a in zip(orig_add, orig, add)
    ]

    return (orig, add, orig_add), (linetypes, linetypes_orig, linetypes_add)


def reIndex(indices: Dict[int, int]) -> Dict[int, int]:
    """
    Remake the indices such that there are no empty rows in the matrix

    Get
    Parameters
    ----------
    indices : dict
        Mapping from total index to unique index

    Returns
    -------
    dict
        Updating mapping
    """

    # Get set of unique indices
    uinds = set(indices.values())

    # Make the translation
    translation = {ui: i for ui, i in zip(uinds, range(len(uinds)))}

    # Translate
    return {i: translation[j] for i, j in indices.items()}
