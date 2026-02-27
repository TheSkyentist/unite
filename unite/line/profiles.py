"""Concrete line profile implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from unite.line.functions import (
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

    @abstractmethod
    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: ArrayLike,
        lsf_fwhm: ArrayLike,
        **params: ArrayLike,
    ) -> Array:
        """Integrate the profile over wavelength bins.

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
        **params : ArrayLike
            Parameter values, keyed by the names from :meth:`param_names`.

        Returns
        -------
        Array
            Fractional flux integrated in each bin (sums to 1 over all bins).
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


def _register(cls: type[Profile]) -> type[Profile]:
    """Register a profile class for deserialization."""
    _PROFILE_REGISTRY[cls.__name__] = cls
    return cls


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

    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: ArrayLike,
        lsf_fwhm: ArrayLike,
        **params: ArrayLike,
    ) -> Array:
        fwhm = params['fwhm_gauss']
        total_fwhm = jnp.sqrt(lsf_fwhm**2 + fwhm**2)
        return integrate_gaussian(low, high, center, total_fwhm)

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

    Requires a single parameter ``fwhm_lorentzian``.  The LSF is **not**
    convolved — this profile is a pure Lorentzian.

    Note: This profile is implemented as a PseudoVoigt with LSF=0 for consistency
    with the scientific assumptions of the package (all lines are convolved with
    instrumental LSF).
    """

    code = 1

    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_lorentzian',)

    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_lorentzian': Uniform(0, 1000)}

    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: ArrayLike,
        lsf_fwhm: ArrayLike,
        **params: ArrayLike,
    ) -> Array:
        # Implement as PseudoVoigt with LSF=0 to maintain consistency
        # with the scientific model (all profiles should handle LSF)
        fwhm_lorentzian = params['fwhm_lorentzian']
        return integrate_voigt(low, high, center, lsf_fwhm, 0.0, fwhm_lorentzian)

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

    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: ArrayLike,
        lsf_fwhm: ArrayLike,
        **params: ArrayLike,
    ) -> Array:
        fwhm_gauss = jnp.sqrt(lsf_fwhm**2 + params['fwhm_gauss'] ** 2)
        fwhm_lorentz = params['fwhm_lorentz']
        return integrate_voigt(low, high, center, fwhm_gauss, fwhm_lorentz)

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

    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: ArrayLike,
        lsf_fwhm: ArrayLike,
        **params: ArrayLike,
    ) -> Array:
        fwhm = params['fwhm_exp']
        return integrate_gaussianLaplace(low, high, center, lsf_fwhm, 0, fwhm)

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

    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: ArrayLike,
        lsf_fwhm: ArrayLike,
        **params: ArrayLike,
    ) -> Array:
        fwhm_gauss = jnp.sqrt(lsf_fwhm**2 + params['fwhm_gauss'] ** 2)
        return integrate_gaussianLaplace(
            low, high, center, fwhm_gauss, params['fwhm_exp']
        )

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

    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: ArrayLike,
        lsf_fwhm: ArrayLike,
        **params: ArrayLike,
    ) -> Array:
        return integrate_gaussHermite(
            low,
            high,
            center,
            lsf_fwhm,
            params['fwhm_gauss'],
            params['h3'],
            params['h4'],
        )

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

    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: ArrayLike,
        lsf_fwhm: ArrayLike,
        **params: ArrayLike,
    ) -> Array:
        return integrate_split_normal(
            low, high, center, lsf_fwhm, params['fwhm_blue'], params['fwhm_red']
        )

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
