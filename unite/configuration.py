"""
Unified Configuration Class for Emission Line Analysis
"""

# Typing
import copy
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Numerical packages
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

# Astronomy packages
from astropy import units as u
from astropy import constants as consts

# unite imports
from unite import defaults


class Configuration:
    """
    Unified configuration class for emission line analysis

    Handles building, modifying, and converting emission line configurations
    to the matrices needed for fitting, while managing defaults and priors.
    """

    def __init__(
        self,
        config_dict: Optional[Dict] = None,
        *,
        linedetect: u.Quantity = defaults.LINEDETECT,
        linepad: u.Quantity = defaults.LINEPAD,
        continuum_size: u.Quantity = defaults.CONTINUUM,
        flux_priors: Optional[Dict[str, Tuple[float, float]]] = None,
        redshift_priors: Optional[Dict[str, Tuple[float, float]]] = None,
        dispersion_priors: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """
        Initialize configuration

        Parameters
        ----------
        config_dict : dict, optional
            Initial configuration dictionary
        linedetect : u.Quantity, optional
            Padding around lines for detection, default from defaults.LINEDETECT
        linepad : u.Quantity, optional
            Padding around lines for masking, default from defaults.LINEPAD
        continuum_size : u.Quantity, optional
            Size of continuum regions, default from defaults.CONTINUUM
        flux_priors : dict, optional
            Custom flux priors by line type
        redshift_priors : dict, optional
            Custom redshift priors by line type
        dispersion_priors : dict, optional
            Custom dispersion priors by line type
        """

        # Store configuration
        self.config = config_dict or self._create_empty_config()

        # Store analysis parameters
        self.linedetect = linedetect
        self.linepad = linepad
        self.continuum_size = continuum_size

        # Set up priors (use defaults if not provided)
        self.flux_priors = flux_priors or defaults.flux
        self.redshift_priors = redshift_priors or defaults.redshift
        self.dispersion_priors = dispersion_priors or defaults.fwhm

        # Validate configuration
        self._validate_config()

    @classmethod
    def from_template(cls, template: str = 'basic', **kwargs) -> 'Configuration':
        """
        Create configuration from predefined templates

        Parameters
        ----------
        template : str
            Template name ('basic', 'full', 'agn', etc.)
        **kwargs
            Additional arguments passed to __init__

        Returns
        -------
        Configuration
            New configuration instance
        """
        templates = {
            'basic': cls._basic_template(),
            'full': cls._full_template(),
            'agn': cls._agn_template(),
        }

        if template not in templates:
            raise ValueError(
                f'Unknown template: {template}. Available: {list(templates.keys())}'
            )

        return cls(config_dict=templates[template], **kwargs)

    def add_group(
        self, name: str, tie_redshift: bool = False, tie_dispersion: bool = False
    ) -> None:
        """
        Add a new group to the configuration

        Parameters
        ----------
        name : str
            Group name
        tie_redshift : bool, optional
            Whether to tie redshift within group
        tie_dispersion : bool, optional
            Whether to tie dispersion within group
        """
        if name in self.config['Groups']:
            raise ValueError(f"Group '{name}' already exists")

        self.config['Groups'][name] = {
            'TieRedshift': tie_redshift,
            'TieDispersion': tie_dispersion,
            'Species': [],
        }

    def add_species(
        self,
        group_name: str,
        species_name: str,
        line_type: str = 'narrow',
        additional_components: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Add a species to a group

        Parameters
        ----------
        group_name : str
            Name of the group to add to
        species_name : str
            Name of the species
        line_type : str, optional
            Type of line, default "narrow"
        additional_components : dict, optional
            Additional components mapping {component_type: target_group}
        """
        if group_name not in self.config['Groups']:
            raise ValueError(f"Group '{group_name}' does not exist")

        if line_type not in defaults.LINETYPES:
            raise ValueError(f'Unknown line type: {line_type}')

        species = {'Name': species_name, 'LineType': line_type, 'Lines': []}

        if additional_components:
            species['AdditionalComponents'] = additional_components

        self.config['Groups'][group_name]['Species'].append(species)

    def add_line(
        self,
        group_name: str,
        species_name: str,
        wavelength: float,
        rel_strength: Optional[float] = None,
    ) -> None:
        """
        Add a line to a species

        Parameters
        ----------
        group_name : str
            Group name
        species_name : str
            Species name
        wavelength : float
            Line wavelength in config units
        rel_strength : float, optional
            Relative strength for tied lines
        """
        # Find the species
        species = self._find_species(group_name, species_name)
        if species is None:
            raise ValueError(
                f"Species '{species_name}' not found in group '{group_name}'"
            )

        line = {'Wavelength': wavelength, 'RelStrength': rel_strength}

        species['Lines'].append(line)

    def remove_group(self, name: str) -> None:
        """Remove a group from configuration"""
        if name not in self.config['Groups']:
            raise ValueError(f"Group '{name}' does not exist")
        del self.config['Groups'][name]

    def remove_species(self, group_name: str, species_name: str) -> None:
        """Remove a species from a group"""
        species = self._find_species(group_name, species_name)
        if species is None:
            raise ValueError(
                f"Species '{species_name}' not found in group '{group_name}'"
            )

        self.config['Groups'][group_name]['Species'].remove(species)

    def restrict_to_coverage(
        self, spectra, linedetect: Optional[u.Quantity] = None
    ) -> 'Configuration':
        """
        Create new configuration restricted to spectral coverage

        Parameters
        ----------
        spectra : Spectra
            Spectra object to check coverage against
        linedetect : u.Quantity, optional
            Line detection padding, uses instance default if None

        Returns
        -------
        Configuration
            New restricted configuration
        """
        # Import here to avoid circular imports
        from unite.spectra import Spectra

        if not isinstance(spectra, Spectra):
            raise TypeError('spectra must be a Spectra object')

        if linedetect is None:
            linedetect = self.linedetect

        # Create deep copy
        new_config = copy.deepcopy(self.config)

        # Effective resolution
        lineres = (linedetect / consts.c).to(u.dimensionless_unscaled).value

        # Filter groups
        new_groups = {}
        for gname, group in new_config['Groups'].items():
            new_species = []
            for species in group['Species']:
                new_lines = []
                for line in species['Lines']:
                    # Compute line wavelength
                    linewav = (line['Wavelength'] * u.Unit(new_config['Unit'])).to(
                        spectra.λ_unit
                    )

                    # Redshift the line
                    linewav = linewav * (1 + spectra.redshift)
                    linewidth = linewav * lineres

                    # Compute boundaries
                    low, high = (linewav - linewidth).value, (linewav + linewidth).value

                    # Check coverage
                    if jnp.logical_or.reduce(
                        jnp.array(
                            [s.coverage(low, high).any() for s in spectra.spectra]
                        )
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

        # Update config
        new_config['Groups'] = new_groups

        # Return new Configuration instance
        return Configuration(
            config_dict=new_config,
            linedetect=self.linedetect,
            linepad=self.linepad,
            continuum_size=self.continuum_size,
            flux_priors=self.flux_priors,
            redshift_priors=self.redshift_priors,
            dispersion_priors=self.dispersion_priors,
        )

    def expand_additional_components(self) -> 'Configuration':
        """
        Create new configuration with additional components expanded as separate species

        Returns
        -------
        Configuration
            New configuration with expanded additional components
        """
        # Set default line types
        new_config = copy.deepcopy(self.config)
        for group in new_config['Groups'].values():
            for species in group['Species']:
                if 'LineType' not in species:
                    species['LineType'] = 'narrow'

        # Add additional components
        for group in self.config['Groups'].values():
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

        return Configuration(
            config_dict=new_config,
            linedetect=self.linedetect,
            linepad=self.linepad,
            continuum_size=self.continuum_size,
            flux_priors=self.flux_priors,
            redshift_priors=self.redshift_priors,
            dispersion_priors=self.dispersion_priors,
        )

    def to_matrices(
        self,
    ) -> Tuple[
        Tuple[List[BCOO], List[BCOO], List[BCOO]],
        Tuple[jnp.ndarray, List[jnp.ndarray], List[jnp.ndarray]],
    ]:
        """
        Convert configuration to sparse matrices for fitting

        Returns
        -------
        matrices : tuple
            Tuple of parameter matrices (orig, add, orig_add)
        linetypes : tuple
            Tuple of line type arrays (all, orig, add)
        """
        # Expand additional components first
        expanded_config = self.expand_additional_components()

        # Use the existing configToMatrices function
        from unite.parameters import configToMatrices

        return configToMatrices(expanded_config.config)

    def get_priors(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Get prior arrays for flux, redshift, and dispersion

        Returns
        -------
        tuple
            Arrays of (flux_priors, redshift_priors, dispersion_priors)
        """
        flux_array = defaults.convertToArray(self.flux_priors)
        redshift_array = defaults.convertToArray(self.redshift_priors)
        dispersion_array = defaults.convertToArray(self.dispersion_priors)

        return flux_array, redshift_array, dispersion_array

    def update_priors(
        self,
        flux_priors: Optional[Dict[str, Tuple[float, float]]] = None,
        redshift_priors: Optional[Dict[str, Tuple[float, float]]] = None,
        dispersion_priors: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """
        Update prior distributions

        Parameters
        ----------
        flux_priors : dict, optional
            New flux priors
        redshift_priors : dict, optional
            New redshift priors
        dispersion_priors : dict, optional
            New dispersion priors
        """
        if flux_priors:
            self.flux_priors.update(flux_priors)
        if redshift_priors:
            self.redshift_priors.update(redshift_priors)
        if dispersion_priors:
            self.dispersion_priors.update(dispersion_priors)

    def copy(self) -> 'Configuration':
        """Create a deep copy of the configuration"""
        return Configuration(
            config_dict=copy.deepcopy(self.config),
            linedetect=self.linedetect,
            linepad=self.linepad,
            continuum_size=self.continuum_size,
            flux_priors=copy.deepcopy(self.flux_priors),
            redshift_priors=copy.deepcopy(self.redshift_priors),
            dispersion_priors=copy.deepcopy(self.dispersion_priors),
        )

    @property
    def groups(self) -> List[str]:
        """Get list of group names"""
        return list(self.config['Groups'].keys())

    @property
    def line_count(self) -> int:
        """Get total number of lines in configuration"""
        count = 0
        for group in self.config['Groups'].values():
            for species in group['Species']:
                count += len(species['Lines'])
        return count

    def _create_empty_config(self) -> Dict:
        """Create empty configuration dictionary"""
        return {'Unit': 'Angstrom', 'Groups': {}}

    def _find_species(self, group_name: str, species_name: str) -> Optional[Dict]:
        """Find species in configuration"""
        if group_name not in self.config['Groups']:
            return None

        for species in self.config['Groups'][group_name]['Species']:
            if species['Name'] == species_name:
                return species
        return None

    def _validate_config(self) -> None:
        """Validate configuration structure"""
        if not isinstance(self.config, dict):
            raise ValueError('Configuration must be a dictionary')

        if 'Groups' not in self.config:
            raise ValueError("Configuration must have 'Groups' key")

        if 'Unit' not in self.config:
            self.config['Unit'] = 'Angstrom'

        # Validate groups structure
        for group_name, group in self.config['Groups'].items():
            if not isinstance(group, dict):
                raise ValueError(f"Group '{group_name}' must be a dictionary")

            required_keys = ['TieRedshift', 'TieDispersion', 'Species']
            for key in required_keys:
                if key not in group:
                    if key in ['TieRedshift', 'TieDispersion']:
                        group[key] = False
                    elif key == 'Species':
                        group[key] = []

    @staticmethod
    def _basic_template() -> Dict:
        """Basic template with common lines"""
        return {
            'Unit': 'Angstrom',
            'Groups': {
                'narrow': {
                    'TieRedshift': False,
                    'TieDispersion': False,
                    'Species': [
                        {
                            'Name': 'H',
                            'LineType': 'narrow',
                            'Lines': [
                                {'Wavelength': 6562.8, 'RelStrength': None},  # H-alpha
                                {'Wavelength': 4861.3, 'RelStrength': None},  # H-beta
                            ],
                        },
                        {
                            'Name': 'OIII',
                            'LineType': 'narrow',
                            'Lines': [
                                {'Wavelength': 5006.8, 'RelStrength': None},
                                {
                                    'Wavelength': 4958.9,
                                    'RelStrength': 0.33,
                                },  # Tied to 5007
                            ],
                        },
                    ],
                }
            },
        }

    @staticmethod
    def _full_template() -> Dict:
        """Full template with many common lines"""
        return {
            'Unit': 'Angstrom',
            'Groups': {
                'narrow': {
                    'TieRedshift': False,
                    'TieDispersion': False,
                    'Species': [
                        {
                            'Name': 'H',
                            'LineType': 'narrow',
                            'Lines': [
                                {'Wavelength': 6562.8, 'RelStrength': None},
                                {'Wavelength': 4861.3, 'RelStrength': None},
                                {'Wavelength': 4340.5, 'RelStrength': None},
                                {'Wavelength': 4101.7, 'RelStrength': None},
                            ],
                        },
                        {
                            'Name': 'OIII',
                            'LineType': 'narrow',
                            'Lines': [
                                {'Wavelength': 5006.8, 'RelStrength': None},
                                {'Wavelength': 4958.9, 'RelStrength': 0.33},
                            ],
                        },
                        {
                            'Name': 'NII',
                            'LineType': 'narrow',
                            'Lines': [
                                {'Wavelength': 6583.5, 'RelStrength': None},
                                {'Wavelength': 6548.0, 'RelStrength': 0.33},
                            ],
                        },
                        {
                            'Name': 'SII',
                            'LineType': 'narrow',
                            'Lines': [
                                {'Wavelength': 6716.4, 'RelStrength': None},
                                {'Wavelength': 6730.8, 'RelStrength': None},
                            ],
                        },
                    ],
                }
            },
        }

    @staticmethod
    def _agn_template() -> Dict:
        """AGN template with broad and narrow components"""
        return {
            'Unit': 'Angstrom',
            'Groups': {
                'narrow': {
                    'TieRedshift': True,
                    'TieDispersion': True,
                    'Species': [
                        {
                            'Name': 'H',
                            'LineType': 'narrow',
                            'Lines': [
                                {'Wavelength': 6562.8, 'RelStrength': None},
                                {'Wavelength': 4861.3, 'RelStrength': None},
                            ],
                            'AdditionalComponents': {'broad': 'broad'},
                        },
                        {
                            'Name': 'OIII',
                            'LineType': 'narrow',
                            'Lines': [
                                {'Wavelength': 5006.8, 'RelStrength': None},
                                {'Wavelength': 4958.9, 'RelStrength': 0.33},
                            ],
                        },
                    ],
                },
                'broad': {'TieRedshift': True, 'TieDispersion': True, 'Species': []},
            },
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file

        Parameters
        ----------
        filepath : str or Path
            Path to save the configuration file
        """
        filepath = Path(filepath)

        # Create a serializable version of the config
        save_data = {
            'config': self.config,
            'parameters': {
                'linedetect': {
                    'value': self.linedetect.value,
                    'unit': str(self.linedetect.unit),
                },
                'linepad': {
                    'value': self.linepad.value,
                    'unit': str(self.linepad.unit),
                },
                'continuum_size': {
                    'value': self.continuum_size.value,
                    'unit': str(self.continuum_size.unit),
                },
            },
            'priors': {
                'flux': self.flux_priors,
                'redshift': self.redshift_priors,
                'dispersion': self.dispersion_priors,
            },
        }

        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Configuration':
        """
        Load configuration from a JSON file

        Parameters
        ----------
        filepath : str or Path
            Path to the configuration file

        Returns
        -------
        Configuration
            Loaded configuration instance
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            save_data = json.load(f)

        # Reconstruct astropy quantities
        linedetect = save_data['parameters']['linedetect']['value'] * u.Unit(
            save_data['parameters']['linedetect']['unit']
        )
        linepad = save_data['parameters']['linepad']['value'] * u.Unit(
            save_data['parameters']['linepad']['unit']
        )
        continuum_size = save_data['parameters']['continuum_size']['value'] * u.Unit(
            save_data['parameters']['continuum_size']['unit']
        )

        return cls(
            config_dict=save_data['config'],
            linedetect=linedetect,
            linepad=linepad,
            continuum_size=continuum_size,
            flux_priors=save_data['priors']['flux'],
            redshift_priors=save_data['priors']['redshift'],
            dispersion_priors=save_data['priors']['dispersion'],
        )

    def __str__(self) -> str:
        """Detailed string representation with proper indentation"""
        lines = [f'Configuration: {len(self.groups)} groups, {self.line_count} lines']
        lines.append(f'Unit: {self.config["Unit"]}')
        lines.append('Analysis Parameters:')
        lines.append(f'  Line Detection: {self.linedetect}')
        lines.append(f'  Line Padding: {self.linepad}')
        lines.append(f'  Continuum Size: {self.continuum_size}')
        lines.append('')

        for group_name, group in self.config['Groups'].items():
            lines.append(f"Group '{group_name}':")
            lines.append(f'  Tie Redshift: {group["TieRedshift"]}')
            lines.append(f'  Tie Dispersion: {group["TieDispersion"]}')
            lines.append(f'  Species ({len(group["Species"])}):')

            for species in group['Species']:
                lines.append(f'    {species["Name"]} ({species["LineType"]}):')
                if 'AdditionalComponents' in species:
                    components = []
                    for comp, target in species['AdditionalComponents'].items():
                        components.append(f'{comp} → {target}')
                    lines.append(
                        f'      Additional Components: {", ".join(components)}'
                    )
                lines.append(f'      Lines ({len(species["Lines"])}):')

                for line in species['Lines']:
                    rel_str = (
                        f', rel_strength={line["RelStrength"]}'
                        if line['RelStrength'] is not None
                        else ''
                    )
                    lines.append(
                        f'        {line["Wavelength"]} {self.config["Unit"]}{rel_str}'
                    )

            lines.append('')

        return '\n'.join(lines)

    def __repr__(self) -> str:
        """String representation"""
        return f'Configuration(groups={len(self.groups)}, lines={self.line_count})'
