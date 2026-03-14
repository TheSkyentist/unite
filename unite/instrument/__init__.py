"""Instrument models: dispersers, spectrum containers, and instrument-specific loaders.

Generic building blocks are in :mod:`unite.instrument.generic`::

    from unite.instrument import generic
    generic.GenericDisperser, generic.SimpleDisperser, generic.GenericSpectrum

Instrument-specific subclasses::

    from unite.instrument import nirspec, sdss
    nirspec.G235H, nirspec.G395H, nirspec.NIRSpecSpectrum
    sdss.SDSSDisperser, sdss.SDSSSpectrum

Collection and configuration::

    from unite.instrument import Spectra, InstrumentConfig, RScale, FluxScale, PixOffset
"""

from unite.instrument.base import FluxScale, PixOffset, RScale
from unite.instrument.config import InstrumentConfig
from unite.instrument.spectrum import Spectra

__all__ = ['FluxScale', 'InstrumentConfig', 'PixOffset', 'RScale', 'Spectra']
