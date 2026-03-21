"""Concrete line profile implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax
from jax import Array
from jax.typing import ArrayLike

from unite._utils import _make_register
from unite.line.functions import (
    evaluate_gaussHermite,
    evaluate_gaussian,
    evaluate_gaussianLaplace,
    evaluate_split_normal,
    evaluate_voigt,
    integrate_gaussHermite,
    integrate_gaussian,
    integrate_gaussianLaplace,
    integrate_split_normal,
    integrate_voigt,
)
from unite.prior import Prior, TruncatedNormal, Uniform

# -------------------------------------------------------------------
# Base Profile class
# -------------------------------------------------------------------


class Profile(ABC):
    """Abstract base class for spectral line profiles.

    A profile declares which parameters it requires (via :meth:`param_names`
    and :meth:`default_priors`) and provides an :meth:`integrate` method that
    computes the profile integral over wavelength bins.

    Each concrete subclass carries an integer :attr:`code` for dispatch
    in JAX arrays, and supports serialization via :meth:`to_dict` /
    :meth:`from_dict`.
    """

    #: Integer code for this profile type, used in JAX arrays.
    code: int

    @abstractmethod
    def param_names(self) -> tuple[str, ...]:
        """Return names of parameters this profile requires.

        Returns
        -------
        tuple of str
            For example, ``('fwhm_gauss',)`` for Gaussian,
            ``('fwhm_gauss', 'fwhm_lorentz')`` for pseudo-Voigt, or
            ``('fwhm_gauss', 'h3', 'h4')`` for Gauss-Hermite.
        """

    @abstractmethod
    def default_priors(self) -> dict[str, Prior]:
        """Return sensible default priors for each parameter.

        The keys must match :meth:`param_names`.  These are used when the
        user does not supply an explicit token for a parameter.

        Returns
        -------
        dict of str to Prior
            For example, ``{'fwhm_gauss': Uniform(0, 1000)}``.
        """

    @property
    def is_absorption(self) -> bool:
        """Whether this is an absorption profile."""
        return False

    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: ArrayLike,
        lsf_fwhm: ArrayLike,
        **params: ArrayLike,
    ) -> Array:
        r"""Integrate the profile over wavelength bins.

        Delegates to :meth:`integrate_branch` by mapping keyword arguments to
        positional slots (p0, p1, p2) in :meth:`param_names` order.

        Parameters
        ----------
        low : ArrayLike
            Lower wavelength edges of bins.
        high : ArrayLike
            Upper wavelength edges of bins.
        center : ArrayLike
            Line center wavelength.
        lsf_fwhm : ArrayLike
            Instrumental line spread function FWHM at the line center.
        \*\*params : ArrayLike
            Parameter values, keyed by the names from :meth:`param_names`.

        Returns
        -------
        Array
            Fractional flux integrated in each bin (sums to 1 over all bins).
        """
        pnames = self.param_names()
        p0 = params[pnames[0]] if len(pnames) > 0 else 0.0
        p1 = params[pnames[1]] if len(pnames) > 1 else 0.0
        p2 = params[pnames[2]] if len(pnames) > 2 else 0.0
        return self.integrate_branch()(low, high, center, lsf_fwhm, p0, p1, p2)

    def evaluate(
        self,
        wavelength: ArrayLike,
        center: ArrayLike,
        lsf_fwhm: ArrayLike,
        **params: ArrayLike,
    ) -> Array:
        r"""Evaluate the normalised profile at wavelength points.

        Delegates to :meth:`evaluate_branch` by mapping keyword arguments to
        positional slots (p0, p1, p2) in :meth:`param_names` order.

        Parameters
        ----------
        wavelength : ArrayLike
            Wavelength points at which to evaluate the profile.
        center : ArrayLike
            Line center wavelength.
        lsf_fwhm : ArrayLike
            Instrumental line spread function FWHM at the line center.
        \*\*params : ArrayLike
            Parameter values, keyed by the names from :meth:`param_names`.

        Returns
        -------
        Array
            Normalised profile value at each wavelength point
            (1/wavelength units).
        """
        pnames = self.param_names()
        p0 = params[pnames[0]] if len(pnames) > 0 else 0.0
        p1 = params[pnames[1]] if len(pnames) > 1 else 0.0
        p2 = params[pnames[2]] if len(pnames) > 2 else 0.0
        return self.evaluate_branch()(wavelength, center, lsf_fwhm, p0, p1, p2)

    @abstractmethod
    def integrate_branch(self):
        """Return a JAX-compatible branch callable for ``lax.switch`` dispatch.

        The returned function must have the fixed signature::

            fn(low, high, center, lsf_fwhm, p0, p1, p2) -> Array

        Parameters correspond to :meth:`param_names` in order: ``p0`` is
        ``param_names()[0]``, ``p1`` is ``param_names()[1]``, ``p2`` is
        ``param_names()[2]``.  Unused slots receive zero from the model
        builder and must be ignored.

        Returns
        -------
        callable
            A pure-JAX function suitable as a ``lax.switch`` branch.
        """

    @abstractmethod
    def evaluate_branch(self):
        """Return a JAX-compatible branch callable for pointwise evaluation.

        The returned function must have the fixed signature::

            fn(wavelength, center, lsf_fwhm, p0, p1, p2) -> Array

        Returns the normalised profile value at each wavelength point.

        Returns
        -------
        callable
            A pure-JAX function suitable as a ``lax.switch`` branch.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a YAML-safe dictionary."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> Profile:
        """Deserialize from a dictionary."""


# -------------------------------------------------------------------
# Registry for deserialization
# -------------------------------------------------------------------

_PROFILE_REGISTRY: dict[str, type[Profile]] = {}
_register = _make_register(_PROFILE_REGISTRY)


def profile_from_dict(d: dict) -> Profile:
    """Deserialize a Profile from a dictionary using the 'type' key.

    Parameters
    ----------
    d : dict
        Dictionary with a ``'type'`` key matching a registered profile class.

    Returns
    -------
    Profile

    Raises
    ------
    KeyError
        If the type is not registered.
    """
    cls = _PROFILE_REGISTRY[d['type']]
    return cls.from_dict(d)


# -------------------------------------------------------------------
# Concrete profiles
# -------------------------------------------------------------------


@_register
class Gaussian(Profile):
    """Gaussian (normal) line profile.

    Requires a single parameter ``fwhm_gauss``.  The instrumental LSF
    is added in quadrature: ``total_fwhm = sqrt(lsf_fwhm² + fwhm_gauss²)``.
    """

    code = 0

    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss',)

    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_gauss': Uniform(0, 1000)}

    def integrate_branch(self):
        def _fn(lo, hi, c, lsf, p0, p1, p2):
            # p0 = fwhm_gauss
            return integrate_gaussian(lo, hi, c, lsf, p0)

        return _fn

    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return evaluate_gaussian(wavelength, c, lsf, p0)

        return _fn

    def to_dict(self) -> dict:
        return {'type': 'Gaussian'}

    @classmethod
    def from_dict(cls, d: dict) -> Gaussian:
        return cls()

    def __repr__(self) -> str:
        return 'Gaussian()'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Gaussian)

    def __hash__(self) -> int:
        return hash(type(self))


@_register
class Cauchy(Profile):
    """Cauchy (Lorentzian) line profile.

    Requires a single parameter ``fwhm_lorentz``.  The LSF is **not**
    convolved — this profile is a pure Lorentzian.

    Note: This profile is implemented as a PseudoVoigt with LSF=0 for consistency
    with the scientific assumptions of the package (all lines are convolved with
    instrumental LSF).
    """

    code = 1

    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_lorentz',)

    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_lorentz': Uniform(0, 1000)}

    def integrate_branch(self):
        def _fn(lo, hi, c, lsf, p0, p1, p2):
            # p0 = fwhm_lorentz; pure Cauchy via PseudoVoigt with zero Gaussian width
            return integrate_voigt(lo, hi, c, lsf, 0.0, p0)

        return _fn

    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return evaluate_voigt(wavelength, c, lsf, 0.0, p0)

        return _fn

    def to_dict(self) -> dict:
        return {'type': 'Cauchy'}

    @classmethod
    def from_dict(cls, d: dict) -> Cauchy:
        return cls()

    def __repr__(self) -> str:
        return 'Cauchy()'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Cauchy)

    def __hash__(self) -> int:
        return hash(type(self))


@_register
class PseudoVoigt(Profile):
    """Pseudo-Voigt line profile (Thompson et al. 1987).

    Requires two parameters: ``fwhm_gauss`` for the Gaussian component
    and ``fwhm_lorentz`` for the Lorentzian component. The instrumental
    LSF is added in quadrature to the Gaussian component.
    """

    code = 2

    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss', 'fwhm_lorentz')

    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_gauss': Uniform(0, 1000), 'fwhm_lorentz': Uniform(0, 1000)}

    def integrate_branch(self):
        def _fn(lo, hi, c, lsf, p0, p1, p2):
            # p0 = fwhm_gauss, p1 = fwhm_lorentz
            return integrate_voigt(lo, hi, c, lsf, p0, p1)

        return _fn

    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return evaluate_voigt(wavelength, c, lsf, p0, p1)

        return _fn

    def to_dict(self) -> dict:
        return {'type': 'PseudoVoigt'}

    @classmethod
    def from_dict(cls, d: dict) -> PseudoVoigt:
        return cls()

    def __repr__(self) -> str:
        return 'PseudoVoigt()'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PseudoVoigt)

    def __hash__(self) -> int:
        return hash(type(self))


@_register
class Laplace(Profile):
    """Laplace (double-exponential) line profile.

    Requires a single parameter ``fwhm_exp``.  The LSF is **not**
    convolved --- this profile is a pure Laplace distribution.
    """

    code = 3

    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_exp',)

    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_exp': Uniform(0, 1000)}

    def integrate_branch(self):
        def _fn(lo, hi, c, lsf, p0, p1, p2):
            # p0 = fwhm_exp; pure Laplace convolved with Gaussian LSF
            return integrate_gaussianLaplace(lo, hi, c, lsf, 0.0, p0)

        return _fn

    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return evaluate_gaussianLaplace(wavelength, c, lsf, 0.0, p0)

        return _fn

    def to_dict(self) -> dict:
        return {'type': 'Laplace'}

    @classmethod
    def from_dict(cls, d: dict) -> Laplace:
        return cls()

    def __repr__(self) -> str:
        return 'Laplace()'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Laplace)

    def __hash__(self) -> int:
        return hash(type(self))


@_register
class SEMG(Profile):
    """Symmetric Exponentially Modified Gaussian (SEMG) line profile.

    A Gaussian (with LSF) convolved with a symmetric Laplace
    (double-exponential) distribution. Requires two parameters:
    ``fwhm_gauss`` for the intrinsic Gaussian component and ``fwhm_exp``
    for the Laplacian component. The instrumental LSF is added in
    quadrature to the Gaussian component.
    """

    code = 4

    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss', 'fwhm_exp')

    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_gauss': Uniform(0, 1000), 'fwhm_exp': Uniform(0, 1000)}

    def integrate_branch(self):
        def _fn(lo, hi, c, lsf, p0, p1, p2):
            # p0 = fwhm_gauss, p1 = fwhm_exp
            return integrate_gaussianLaplace(lo, hi, c, lsf, p0, p1)

        return _fn

    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return evaluate_gaussianLaplace(wavelength, c, lsf, p0, p1)

        return _fn

    def to_dict(self) -> dict:
        return {'type': 'SEMG'}

    @classmethod
    def from_dict(cls, d: dict) -> SEMG:
        return cls()

    def __repr__(self) -> str:
        return 'SEMG()'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SEMG)

    def __hash__(self) -> int:
        return hash(type(self))


@_register
class GaussHermite(Profile):
    """Gauss-Hermite line profile.

    A Gaussian (with LSF) modified by Hermite polynomial corrections for
    skewness (h3) and kurtosis (h4). Requires three parameters:
    ``fwhm_gauss`` for the intrinsic Gaussian FWHM, ``h3`` for the
    skewness coefficient, and ``h4`` for the kurtosis coefficient. The
    instrumental LSF is added in quadrature to the Gaussian component.
    """

    code = 5

    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss', 'h3', 'h4')

    def default_priors(self) -> dict[str, Prior]:
        return {
            'fwhm_gauss': Uniform(0, 1000),
            'h3': TruncatedNormal(loc=0, scale=0.1, low=-0.3, high=0.3),
            'h4': TruncatedNormal(loc=0, scale=0.1, low=-0.3, high=0.3),
        }

    def integrate_branch(self):
        def _fn(lo, hi, c, lsf, p0, p1, p2):
            # p0 = fwhm_gauss, p1 = h3, p2 = h4
            return integrate_gaussHermite(lo, hi, c, lsf, p0, p1, p2)

        return _fn

    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return evaluate_gaussHermite(wavelength, c, lsf, p0, p1, p2)

        return _fn

    def to_dict(self) -> dict:
        return {'type': 'GaussHermite'}

    @classmethod
    def from_dict(cls, d: dict) -> GaussHermite:
        return cls()

    def __repr__(self) -> str:
        return 'GaussHermite()'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GaussHermite)

    def __hash__(self) -> int:
        return hash(type(self))


@_register
class SplitNormal(Profile):
    """Split-normal (two-sided Gaussian) line profile.

    A Gaussian with different standard deviations on each side of the mean.
    Requires two parameters: ``fwhm_blue`` for the blue (left) side and
    ``fwhm_red`` for the red (right) side. The instrumental LSF is added
    in quadrature to both components.
    """

    code = 6

    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_blue', 'fwhm_red')

    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_blue': Uniform(0, 1000), 'fwhm_red': Uniform(0, 1000)}

    def integrate_branch(self):
        def _fn(lo, hi, c, lsf, p0, p1, p2):
            # p0 = fwhm_blue, p1 = fwhm_red
            return integrate_split_normal(lo, hi, c, lsf, p0, p1)

        return _fn

    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return evaluate_split_normal(wavelength, c, lsf, p0, p1)

        return _fn

    def to_dict(self) -> dict:
        return {'type': 'SplitNormal'}

    @classmethod
    def from_dict(cls, d: dict) -> SplitNormal:
        return cls()

    def __repr__(self) -> str:
        return 'SplitNormal()'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SplitNormal)

    def __hash__(self) -> int:
        return hash(type(self))


# -------------------------------------------------------------------
# Absorption profiles
# -------------------------------------------------------------------


@_register
class GaussianAbsorption(Profile):
    """Gaussian absorption profile.

    Uses the same Gaussian shape as :class:`Gaussian` but is designed for
    absorption lines.  The ``integrate_branch`` uses pixel-center
    evaluation (the known approximation for slowly-varying absorbers),
    while ``evaluate_branch`` returns the exact normalised density.
    """

    code = 7

    @property
    def is_absorption(self) -> bool:
        return True

    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss',)

    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_gauss': Uniform(0, 1000)}

    def integrate_branch(self):
        def _fn(lo, hi, c, lsf, p0, p1, p2):
            mid = (lo + hi) / 2.0
            return evaluate_gaussian(mid, c, lsf, p0) * (hi - lo)

        return _fn

    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return evaluate_gaussian(wavelength, c, lsf, p0)

        return _fn

    def to_dict(self) -> dict:
        return {'type': 'GaussianAbsorption'}

    @classmethod
    def from_dict(cls, d: dict) -> GaussianAbsorption:
        return cls()

    def __repr__(self) -> str:
        return 'GaussianAbsorption()'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GaussianAbsorption)

    def __hash__(self) -> int:
        return hash(type(self))


@_register
class VoigtAbsorption(Profile):
    """Pseudo-Voigt absorption profile.

    Uses the same Thompson et al. (1987) pseudo-Voigt shape as
    :class:`PseudoVoigt` but is designed for absorption lines.
    """

    code = 8

    @property
    def is_absorption(self) -> bool:
        return True

    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss', 'fwhm_lorentz')

    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_gauss': Uniform(0, 1000), 'fwhm_lorentz': Uniform(0, 1000)}

    def integrate_branch(self):
        def _fn(lo, hi, c, lsf, p0, p1, p2):
            mid = (lo + hi) / 2.0
            return evaluate_voigt(mid, c, lsf, p0, p1) * (hi - lo)

        return _fn

    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return evaluate_voigt(wavelength, c, lsf, p0, p1)

        return _fn

    def to_dict(self) -> dict:
        return {'type': 'VoigtAbsorption'}

    @classmethod
    def from_dict(cls, d: dict) -> VoigtAbsorption:
        return cls()

    def __repr__(self) -> str:
        return 'VoigtAbsorption()'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, VoigtAbsorption)

    def __hash__(self) -> int:
        return hash(type(self))


@_register
class LorentzianAbsorption(Profile):
    """Lorentzian (Cauchy) absorption profile.

    Uses the same Lorentzian shape as :class:`Cauchy` but is designed for
    absorption lines.
    """

    code = 9

    @property
    def is_absorption(self) -> bool:
        return True

    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_lorentz',)

    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_lorentz': Uniform(0, 1000)}

    def integrate_branch(self):
        def _fn(lo, hi, c, lsf, p0, p1, p2):
            mid = (lo + hi) / 2.0
            return evaluate_voigt(mid, c, lsf, 0.0, p0) * (hi - lo)

        return _fn

    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return evaluate_voigt(wavelength, c, lsf, 0.0, p0)

        return _fn

    def to_dict(self) -> dict:
        return {'type': 'LorentzianAbsorption'}

    @classmethod
    def from_dict(cls, d: dict) -> LorentzianAbsorption:
        return cls()

    def __repr__(self) -> str:
        return 'LorentzianAbsorption()'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LorentzianAbsorption)

    def __hash__(self) -> int:
        return hash(type(self))


_PROFILE_ALIASES: dict[str, Profile] = {
    'gaussian': Gaussian(),
    'normal': Gaussian(),
    'lorentzian': Cauchy(),
    'cauchy': Cauchy(),
    'exponential': Laplace(),
    'laplace': Laplace(),
    'voigt': PseudoVoigt(),
    'pseudovoigt': PseudoVoigt(),
    'semg': SEMG(),
    'exp-gaussian': SEMG(),
    'hermite': GaussHermite(),
    'gauss-hermite': GaussHermite(),
    'split-normal': SplitNormal(),
    'two-sided': SplitNormal(),
    'gaussian-absorption': GaussianAbsorption(),
    'voigt-absorption': VoigtAbsorption(),
    'lorentzian-absorption': LorentzianAbsorption(),
}


def resolve_profile(profile: str | Profile) -> Profile:
    """Convert a profile string or instance to a Profile object.

    Parameters
    ----------
    profile : str or Profile
        Profile name (case-insensitive) or instance.

    Returns
    -------
    Profile

    Raises
    ------
    ValueError
        If the string is not a recognized profile alias.
    """
    if isinstance(profile, Profile):
        return profile
    if isinstance(profile, str):
        key = profile.lower()
        if key not in _PROFILE_ALIASES:
            valid = ', '.join(sorted(_PROFILE_ALIASES))
            msg = f'Unknown profile {profile!r}. Valid names: {valid}'
            raise ValueError(msg)
        return _PROFILE_ALIASES[key]
    msg = f'profile must be a str or Profile, got {type(profile).__name__}'
    raise TypeError(msg)


# -------------------------------------------------------------------
# JAX dispatch: integration and evaluation branches
# -------------------------------------------------------------------

# Build the lax.switch branch lists once at import time.
# Each Profile subclass owns its branches via integrate_branch() and
# evaluate_branch(); sorted by code guarantees the list index matches
# Profile.code.
_INTEGRATE_BRANCHES = [
    cls().integrate_branch()
    for cls in sorted(_PROFILE_REGISTRY.values(), key=lambda c: c.code)
]

_EVALUATE_BRANCHES = [
    cls().evaluate_branch()
    for cls in sorted(_PROFILE_REGISTRY.values(), key=lambda c: c.code)
]


def _integrate_single_line(low, high, center, lsf_fwhm, p0, p1, p2, code):
    """Integrate one line profile over pixel bins, dispatched by ``code``.

    All FWHM parameters are in wavelength units.  Shape parameters (h3, h4)
    are dimensionless.  ``lax.switch`` selects the branch at trace time, so
    every branch must have identical output shape.

    Parameters
    ----------
    low, high : jnp.ndarray, shape (n_pixels,)
        Pixel bin edges.
    center, lsf_fwhm, p0, p1, p2 : float
        Per-line scalars: observed center, LSF FWHM, and three profile
        parameter slots (in :meth:`Profile.param_names` order).  Slots
        unused by a given profile receive zero.
    code : int
        ``Profile.code`` for the line.

    Returns
    -------
    jnp.ndarray, shape (n_pixels,)
        Integrated profile fraction per pixel bin.
    """
    return jax.lax.switch(
        code, _INTEGRATE_BRANCHES, low, high, center, lsf_fwhm, p0, p1, p2
    )


# vmap over lines: per-line scalars map to axis 0; pixel arrays are shared (None).
integrate_lines = jax.vmap(
    _integrate_single_line, in_axes=(None, None, 0, 0, 0, 0, 0, 0)
)
"""Vectorised integration over all lines simultaneously.

Input shapes: ``low/high (n_pixels,)``, all others ``(n_lines,)``.
Output shape: ``(n_lines, n_pixels)``.
"""


def _evaluate_single_line(wavelength, center, lsf_fwhm, p0, p1, p2, code):
    """Evaluate one line profile at wavelength points, dispatched by ``code``.

    Parameters
    ----------
    wavelength : jnp.ndarray, shape (n_points,)
        Wavelength points at which to evaluate.
    center, lsf_fwhm, p0, p1, p2 : float
        Per-line scalars.
    code : int
        ``Profile.code`` for the line.

    Returns
    -------
    jnp.ndarray, shape (n_points,)
        Normalised profile value at each wavelength point.
    """
    return jax.lax.switch(
        code, _EVALUATE_BRANCHES, wavelength, center, lsf_fwhm, p0, p1, p2
    )


evaluate_lines = jax.vmap(_evaluate_single_line, in_axes=(None, 0, 0, 0, 0, 0, 0))
"""Vectorised evaluation over all lines simultaneously.

Input shapes: ``wavelength (n_points,)``, all others ``(n_lines,)``.
Output shape: ``(n_lines, n_points)``.
"""
