"""
Module for Loading Spectra Data
"""

# Import packages
from os import path
from importlib import resources

# Astronomy packages
from astropy.table import Table
from astropy import units as u, constants as consts

# Numerical packages
import numpy as np

# Calibration
from unite import calibration, defaults


# Spectra class
class Spectra:
    """
    Generic Collection of Spectra
    """

    def __init__(
        self,
        spectra: list,
        redshift_initial: float,
        λ_unit: u.Unit,
        fλ_unit: u.Unit,
        continuum_regions: list = [],
    ) -> None:
        """
        Initialize the spectra

        Parameters
        ----------
        spectra : list
            List of Spectrum objects
        redshift_initial : float
            Initial redshift
        λ_unit : u.Unit
            Wavelength target unit
        fλ_unit : u.Unit
            Spectral flux target unit (fλ)
        continuum_regions : list, optional
            Continuum regions to use, if not specified they will be computed

        Returns
        -------
        None
        """

        # Keep track
        self.redshift_initial = redshift_initial

        # Keep track of units
        self.λ_unit = λ_unit
        self.fλ_unit = fλ_unit

        # Store the spectra
        self.spectra = spectra

        # Keep track of names
        self.names = [spectrum.name for spectrum in spectra]

    def restrict(self, continuum_regions: list) -> None:
        """
        Restrict the spectra to the continuum regions

        Parameters
        ----------
        continuum_regions : list
            List of continuum regions

        Returns
        -------
        None
        """

        # Loop over the spectra
        for spectrum in self.spectra:
            spectrum.restrict(continuum_regions)

        # Remove empty spectra
        self.spectra = [spectrum for spectrum in self.spectra if len(spectrum.wave) > 0]
        self.names = [spectrum.name for spectrum in self.spectra]

    def rescale(
        self, config: dict, continuum_regions: list, linepad: u.Quantity
    ) -> None:
        """
        Rescale the errorbars in each region

        Parameters
        ----------
        config : dict
            Configuration dictionary
        continuum_regions : list
            List of continuum regions
        linepad : u.Quantity
            Padding to mask emission lines

        Returns
        -------
        None
        """

        for spectrum in self.spectra:
            spectrum.rescale(config, continuum_regions, linepad)

    def restrictAndRescale(
        self,
        config: dict,
        continuum_regions: list,
        linepad: u.Quantity = defaults.LINEPAD,
    ) -> None:
        """
        Restrict the spectra to the continuum regions and rescale the errorbars

        Parameters
        ----------
        config : dict
            Configuration dictionary
        continuum_regions : list
            List of continuum regions

        Returns
        -------
        None
        """

        self.restrict(continuum_regions)
        self.rescale(config, continuum_regions, linepad)


# NIRSpec Spectra
class NIRSpecSpectra(Spectra):
    """
    Collection of NIRSpec spectra for a given source ID and mask
    """

    def __init__(
        self,
        rows: Table,
        spectra_directory: str,
        λ_unit: u.Unit = u.micron,
        fλ_unit: u.Unit = u.Unit(1e-20 * u.erg / u.s / u.cm**2 / u.angstrom),
    ) -> None:
        """
        Initialize the spectra

        Parameters
        ----------
        rows : Table
            Table of rows for the source
        spectra_directory : str
            Path to directory containing the spectra
        instrument_directory : str
            Path to directory containing the lsf curves
        λ_unit : u.Unit
            Wavelength target unit
        fλ_unit : u.Unit
            Spectral flux target unit (fλ)

        Returns
        -------
        None
        """

        # Get best initial redshift
        bestrow = rows[rows['grade'] == np.max(rows['grade'])]
        if len(bestrow) > 1:
            bestrow = bestrow[bestrow['grating'] == 'G395M']

        # If z isn't -1 else use fitz
        if bestrow['z'] == -1:
            redshift_initial = bestrow['zfit'][0]
        else:
            redshift_initial = bestrow['z'][0]

        # Compute the spectrum files
        spectrum_files = [path.join(spectra_directory, row['file']) for row in rows]

        # If there is only one spectrum, it is fixed, otherwise set PRISM to be free
        if len(spectrum_files) == 1:
            fixed = [True]
        else:
            fixed = [False if 'PRISM' in row['grating'] else True for row in rows]
        self.fixed = fixed

        # Load the spectra
        spectra = [
            NIRSpecSpectrum(row['grating'], sf, redshift_initial, λ_unit, fλ_unit, fix)
            for row, sf, fix in zip(rows, spectrum_files, fixed)
        ]

        # Initialize
        super().__init__(
            spectra=spectra,
            redshift_initial=redshift_initial,
            λ_unit=λ_unit,
            fλ_unit=fλ_unit,
        )

    def rescale(
        self, config: dict, continuum_regions: list, linepad: u.Quantity
    ) -> None:
        """
        Rescale the errorbars in each region

        Parameters
        ----------
        config : dict
            Configuration dictionary
        continuum_regions : list
            List of continuum regions
        linepad : u.Quantity
            Padding to mask emission lines

        Returns
        -------
        None
        """

        super().rescale(config, continuum_regions, linepad)

        # If no spectra are fixed, fix the first one
        if not any([spectrum.fixed for spectrum in self.spectra]):
            self.spectra[0].fixed = True
        self.fixed = [spectrum.fixed for spectrum in self.spectra]


# Spectrum class
class Spectrum:
    """
    Spectrum from a given disperser
    """

    def __init__(
        self,
        name: str,
        low: np.ndarray,
        wave: np.ndarray,
        high: np.ndarray,
        flux: np.ndarray,
        err: np.ndarray,
        redshift_initial: float,
        λ_unit: u.Unit,
        fλ_unit: u.Unit,
    ) -> None:
        """
        Initialize the spectrum

        Parameters
        ----------
        name : str
            Name of the spectrum
        low : np.ndarray
            Low edge of the bins
        wave : np.ndarray
            Central wavelength of the bins
        high : np.ndarray
            High edge of the bins
        flux : np.ndarray
            Observed flux values
        err : np.ndarray
            Error in the flux values
        redshift_initial : float
            Initial redshift
        λ_unit : u.Unit
            Wavelength target unit
        fλ_unit : u.Unit
            Spectral flux target unit (fλ)

        Returns
        -------
        None
        """

        # Keep track of the name:
        self.name = name

        # Keep track of redshift
        self.redshift_initial = redshift_initial

        # Keep track of units
        self.λ_unit = λ_unit
        self.fλ_unit = fλ_unit

        # Mask NaN values and store
        mask = np.invert(np.isnan(err))
        for key, array in zip(
            ['wave', 'low', 'high', 'flux', 'err'],
            [wave, low, high, flux, err],
        ):
            setattr(self, key, array[mask])

    def __call__(self):
        """
        Return the attributes we care about
        """

        return (getattr(self, key) for key in ['low', 'wave', 'high', 'flux', 'err'])

    # Calculate if range is covered
    def coverage(self, low: float, high: float, partial: bool = True) -> np.ndarray:
        """
        Check if a given range is covered by the spectrum

        Parameters
        ----------
        low : float
            Low edge of the range
        high : float
            High edge of the range
        halfok : bool, optional
            Whether partial coverage is enough, defaults to True

        Returns
        -------
        np.ndarray
           Boolean array of spectral coverage
        """

        # Check if the range is covered
        if partial:
            return np.logical_and(low < self.high, self.low < high)
        else:
            return np.logical_and(low <= self.low, self.high <= high)

    # Restrict to continuum regions
    def restrict(self, continuum_regions: list) -> None:
        """
        Restrict the spectrum to the continuum regions

        Parameters
        ----------
        continuum_regions : list
            List of continuum regions

        Returns
        -------
        None
        """

        # Compute the mask
        mask = np.logical_or.reduce(
            np.array(
                [
                    self.coverage(region[0], region[1], partial=False)
                    for region in continuum_regions
                ]
            )
        )

        # Apply the mask
        for key in ['wave', 'low', 'high', 'flux', 'err']:
            setattr(self, key, getattr(self, key)[mask])

    # Mask lines in continuum regions
    def maskLines(
        self,
        config: list,
        continuum_region: np.ndarray,
        linepad: u.Quantity,
    ) -> np.ndarray:
        """
        Mask the lines in the continuum region

        Parameters
        ----------
        continuum_region : np.ndarray
            Boundary of the continuum region
        config : dict
            Configuration of emission lines
        spectrum : Spectrum
            Spectrum
        linepad : u.Quantity
            Padding to mask emission lines

        Returns
        -------
        np.ndarray
            Masked region
        """

        # Compute redshift and dimensionless padding unit
        opz = 1 + self.redshift_initial
        pad = (linepad / consts.c).to(u.dimensionless_unscaled).value

        # Extract the region
        low, high = continuum_region
        mask = np.logical_and(low < self.wave, self.wave < high)

        # Mask each line
        λ_unit = u.Unit(config['Unit'])
        for group in config['Groups'].values():
            for species in group['Species']:
                for line in species['Lines']:
                    # Compute the line wavelength
                    linewav = (line['Wavelength'] * λ_unit).to(self.λ_unit).value * opz

                    # Get the effective padding
                    linepad = linewav * pad

                    # Compute the boundaries
                    low, high = linewav - linepad, linewav + linepad

                    # Mask the line
                    linemask = np.logical_and(low < self.wave, self.wave < high)
                    mask = np.logical_and(mask, np.invert(linemask))

        return mask

    # Rescale errorbars based on linear continuum
    def scaleErrorbars(self, region: np.ndarray) -> float:
        """
        Rescale errorbars in a region assuming a linear continuum
        Do a least squares fit to the region and scale the errorbars to have unit variance

        Parameters
        ----------
        region : np.ndarray
            Boolean array defining the region of interest

        Returns
        -------
        float
            Scale factor for the errorbars
        """
        # Scale the error based on ratio of σ(flux) / median(err)
        # return jnp.std(flux[continuum_region]) / jnp.median(err[continuum_region])

        # Compute least squares fit
        N = region.sum()
        X = np.vstack([self.wave[region], np.ones(N)]).T
        W = np.diag(1 / np.square(self.err[region]))
        y = np.atleast_2d(self.flux[region]).T
        β = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ y

        # Compute χ²/ν
        resid = X @ β - y
        χ2_ν = (resid.T @ W @ resid)[0][0] / (N - β.size)

        # Return scale that makes residuals have unit variance
        return np.sqrt(χ2_ν)

    def rescale(
        self, config: dict, continuum_regions: list, linepad: u.Quantity
    ) -> None:
        """
        Rescale the errorbars in each region

        Parameters
        ----------
        config : dict
            Configuration dictionary
        continuum_regions : list
            List of continuum regions
        linepad : u.Quantity
            Padding to mask emission lines

        Returns
        -------
        None
        """

        # Loop over the continuum regions
        newerr = np.zeros_like(self.err)
        scales = []
        for region in continuum_regions:
            # Compute the masks
            regmask = self.coverage(region[0], region[1], partial=False)
            linemask = self.maskLines(config, region, linepad)

            # If not enough data, don't change errors
            # Need at least three points for reduced χ² of a line
            if np.sum(linemask) <= 2:
                newerr = np.where(regmask, self.err, newerr)
                continue

            # Scale the errorbars
            scale = self.scaleErrorbars(linemask)
            scales.append(scale)

            # Apply the scaling
            newerr = np.where(regmask, self.err * scale, newerr)

        # Keep track of the scales
        self.errscales = scales

        # Store the new errorbars
        self.err = newerr


# NIRSpec Spectrum
class NIRSpecSpectrum(Spectrum):
    def __init__(
        self,
        disperser: str,
        spec_file: str,
        redshift_initial: float,
        λ_unit: u.Unit,
        fλ_unit: u.Unit,
        fixed: bool,
    ) -> None:
        """
        Load the spectrum from a file

        Parameters
        ----------
        disperser : str
            Name of the spectrum
        spec_file : str
            File containing the spectrum
        redshift_initial : float
            Initial redshift
        λ_unit : u.Unit
            Wavelength target unit
        fλ_unit : u.Unit
            Spectral flux target unit (fλ)
        fixed : bool
            Whether flux/pixel offset are fixed

        Returns
        -------
        None
        """

        # Keep track if fixed
        self.fixed = fixed

        # Compute resolution
        lsf_dir = resources.files('unite.data.resolution')
        lsf_file = f'jwst_nirspec_{disperser.lower()}_lsf.fits'
        self.lsf = calibration.InterpLSFCurve(lsf_dir.joinpath(lsf_file), λ_unit)

        # Compute pixel offset
        disp_dir = resources.files('unite.data.resolution')
        disp_file = f'jwst_nirspec_{disperser.lower()}_disp.fits'
        self.offset = calibration.PixelOffset(disp_dir.joinpath(disp_file), λ_unit)

        # Load the spectrum from file
        spec = Table.read(spec_file, 'SPEC1D')

        # Unpack relevant columns, convert
        wave = spec['wave'].to(λ_unit)
        flux = spec['flux'].to(fλ_unit, equivalencies=u.spectral_density(wave)).value
        err = spec['err'].to(fλ_unit, equivalencies=u.spectral_density(wave)).value
        wave = wave.value

        # Calculate bin edges
        δλ = np.diff(wave) / 2
        mid = wave[:-1] + δλ
        edges = np.concatenate([wave[0:1] - δλ[0:1], mid, wave[-2:-1] + δλ[-2:-1]])
        low = edges[:-1]
        high = edges[1:]

        # Initialize the spectrum
        super().__init__(
            name=disperser,
            low=low,
            wave=wave,
            high=high,
            flux=flux,
            err=err,
            redshift_initial=redshift_initial,
            λ_unit=λ_unit,
            fλ_unit=fλ_unit,
        )
