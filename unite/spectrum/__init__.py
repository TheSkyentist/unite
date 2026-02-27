"""Spectrum data containers for unite.

Typical usage::

    from unite.spectrum import Spectrum, Spectra

    # Create a Spectrum with a disperser.
    from unite.disperser.nirspec import G235H
    spec = Spectrum(low, high, flux, error, G235H())
    spectra = Spectra([spec], redshift=1.8)
"""

from unite.spectrum.spectrum import Spectra, Spectrum

__all__ = [
    'Spectra',
    'Spectrum',
]
