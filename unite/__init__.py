"""unite — Unified Numerical Integration Tool for spEctroscopy.

A Python package for fast, efficient Bayesian inference of emission lines
from multiple spectra simultaneously, built on JAX, NumPyro, and Astropy.

Typical workflow
----------------
1. Create a :class:`LineConfiguration` with lines, multiplets, shared
   kinematics, priors, and profile shapes.
2. Create a :class:`ContinuumConfiguration` (auto-generated or manual).
3. Optionally save/load the configuration to YAML via :func:`save_yaml` /
   :func:`load_yaml`.
4. Wrap observed spectra in a :class:`Spectra` collection.
5. Call :meth:`ModelBuilder.build` to obtain ``(model_fn, model_args)``
   and run with your own numpyro sampler.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('unite')
except PackageNotFoundError:
    __version__ = '0.0.0'

from unite.config import Configuration
from unite.disperser.config import DispersersConfiguration

__all__ = ['Configuration', 'DispersersConfiguration']
