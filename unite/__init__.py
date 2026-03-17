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

import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _read_pyproject_version():
    try:
        import tomllib

        pyproject = Path(__file__).resolve().parents[1] / 'pyproject.toml'
        data = tomllib.loads(pyproject.read_text())
        return data['project']['version']
    except Exception:
        return '0.0.0.dev0'


def _git_hash():
    try:
        return (
            subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


try:
    __version__ = version('unite')
except PackageNotFoundError:
    base = _read_pyproject_version()
    gh = _git_hash()
    __version__ = f'{base}.dev0+git.{gh}' if gh else f'{base}.dev0'
