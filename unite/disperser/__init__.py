"""Instrument disperser models and calibration tokens for unite.

Typical usage::

    from unite.disperser import DispersersConfiguration, RScale, FluxScale
    from unite.disperser.nirspec import G235H, G395H, NIRSpecSpectrum
    from unite.prior import TruncatedNormal

    # Shared resolving-power scale (same token instance → one parameter).
    r = RScale(prior=TruncatedNormal(1.0, 0.05, 0.8, 1.2))
    flux_0 = FluxScale(prior=TruncatedNormal(1.0, 0.1, 0.5, 2.0))

    cfg = DispersersConfiguration([
        G235H(r_scale=r),                       # flux reference (flux_scale=None)
        G395H(r_scale=r, flux_scale=flux_0),    # free flux scale
    ])

    # Load spectra using NIRSpecSpectrum.
    spec = NIRSpecSpectrum.from_arrays(low, high, flux, err, cfg['G235H'])
"""

from unite.disperser.base import Disperser, FluxScale, PixOffset, RScale
from unite.disperser.config import DispersersConfiguration
from unite.disperser.generic import GenericDisperser, SimpleDisperser

__all__ = [
    'Disperser',
    'DispersersConfiguration',
    'FluxScale',
    'GenericDisperser',
    'PixOffset',
    'RScale',
    'SimpleDisperser',
]
