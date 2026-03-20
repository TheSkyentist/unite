"""Spectrum classes and loaders.

The core data container::

    from unite.spectrum import Spectrum

The spectrum collection::

    from unite.spectrum import Spectra

Loaders::

    from unite.spectrum import from_arrays, from_DJA, from_sdss_fits
"""

from unite.spectrum.collection import (
    RegionDiagnostic,
    ScaleDiagnosticList,
    Spectra,
    SpectrumScaleDiagnostic,
)
from unite.spectrum.loaders import from_arrays, from_DJA, from_sdss_fits
from unite.spectrum.spectrum import Spectrum

__all__ = [
    'RegionDiagnostic',
    'ScaleDiagnosticList',
    'Spectra',
    'Spectrum',
    'SpectrumScaleDiagnostic',
    'from_DJA',
    'from_arrays',
    'from_sdss_fits',
]
