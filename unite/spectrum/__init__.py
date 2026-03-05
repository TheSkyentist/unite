"""Spectrum data containers for unite.

Typical usage::

    from unite.spectrum import Spectrum, Spectra
    from astropy import units as u

    # Create a Spectrum with a disperser (flux/error must be Quantities).
    from unite.disperser.nirspec import G235H
    flux_unit = u.Unit('1e-17 erg / (s cm2 AA)')
    spec = Spectrum(low, high, flux * flux_unit, error * flux_unit, G235H())
    spectra = Spectra([spec], redshift=1.8)
"""

from unite.spectrum.spectrum import Spectra, Spectrum

__all__ = ['Spectra', 'Spectrum']
