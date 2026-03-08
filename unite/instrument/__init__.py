"""Instrument models: dispersers, spectrum containers, and instrument-specific loaders.

Generic building blocks are in :mod:`unite.instrument.generic`::

    from unite.instrument.generic import GenericDisperser, SimpleDisperser, GenericSpectrum

Instrument-specific subclasses::

    from unite.instrument.nirspec import G235H, G395H, NIRSpecSpectrum
    from unite.instrument.sdss import SDSSDisperser, SDSSSpectrum

Collection and configuration::

    from unite.instrument import Spectra, InstrumentConfig, RScale, FluxScale, PixOffset
"""

from unite.instrument.base import Disperser, FluxScale, PixOffset, RScale
from unite.instrument.config import InstrumentConfig
from unite.instrument.spectrum import Spectra

__all__ = [
    'Disperser',
    'FluxScale',
    'InstrumentConfig',
    'PixOffset',
    'RScale',
    'Spectra',
]
