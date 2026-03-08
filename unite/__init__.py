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
except PackageNotFoundError:  # pragma: no cover
    __version__ = '0.0.0'  # pragma: no cover

from unite.config import Configuration
from unite.evaluate import SpectrumPrediction, evaluate_model
from unite.results import make_hdul, make_parameter_table, make_spectra_tables

__all__ = [
    'Configuration',
    'SpectrumPrediction',
    'evaluate_model',
    'make_hdul',
    'make_parameter_table',
    'make_spectra_tables',
]
