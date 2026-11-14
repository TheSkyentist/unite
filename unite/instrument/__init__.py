"""Instrument models: dispersers and calibration tokens.

Generic building blocks are in :mod:`unite.instrument.generic`::

    from unite.instrument.generic import GenericDisperser, SimpleDisperser

Instrument-specific dispersers::

    from unite.instrument import nirspec, sdss
    nirspec.NIRSpec, nirspec.G235H, nirspec.G395H, nirspec.PRISM
    sdss.SDSSDisperser

Calibration tokens and configuration::

    from unite.instrument import RScale, FluxScale, PixOffset, InstrumentConfig

Higher-order grating overlap::

    from unite.instrument import OrderOverlap, OrderScale

Spectrum classes and loaders live in :mod:`unite.spectrum`::

    from unite.spectrum import Spectrum, Spectra, from_arrays, from_DJA, from_sdss_fits
"""

from unite.instrument.base import FluxScale, OrderScale, PixOffset, RScale
from unite.instrument.config import InstrumentConfig
from unite.instrument.overlap import OrderOverlap

__all__ = [
    'FluxScale',
    'InstrumentConfig',
    'OrderOverlap',
    'OrderScale',
    'PixOffset',
    'RScale',
]
