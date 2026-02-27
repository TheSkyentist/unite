"""JWST NIRSpec dispersers and spectrum loader."""

from unite.disperser.nirspec.disperser import (
    G140H,
    G140M,
    G235H,
    G235M,
    G395H,
    G395M,
    NIRSpecDisperser,
    Prism,
)
from unite.disperser.nirspec.spectrum import NIRSpecSpectrum

__all__ = [
    'G140H',
    'G140M',
    'G235H',
    'G235M',
    'G395H',
    'G395M',
    'NIRSpecDisperser',
    'NIRSpecSpectrum',
    'Prism',
]
