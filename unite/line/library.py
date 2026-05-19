"""Concrete line profile implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import override

from jax import Array
from jax.typing import ArrayLike

from unite._utils import _make_register
from unite.line import functions
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

    def integrate(
        self,
        edges: ArrayLike,
        lsf_fwhm: ArrayLike,
        center: ArrayLike,
        **params: ArrayLike,
    ) -> Array:
        r"""Cumulative profile array evaluated at edges.

        Delegates to :meth:`integrate_branch` by mapping keyword arguments
        to positional slots (p0, p1, p2) in :meth:`param_names` order.

        Parameters
        ----------
        edges : ArrayLike, shape ``(E,)``
            Pixel edges in canonical wavelength units.
        lsf_fwhm : ArrayLike, shape ``(E,)``
            Instrumental LSF FWHM evaluated at each edge.
        center : ArrayLike
            Line center wavelength.
        \*\*params : ArrayLike
            Parameter values keyed by the names from :meth:`param_names`.

        Returns
        -------
        Array, shape ``(E,)``
            Cumulative profile array.  ``jnp.diff`` over the result, masked
            to drop inter-pixel gap entries, gives the per-pixel integral
            of the profile.
        """
        pnames = self.param_names()
        p0 = params[pnames[0]] if len(pnames) > 0 else 0.0
        p1 = params[pnames[1]] if len(pnames) > 1 else 0.0
        p2 = params[pnames[2]] if len(pnames) > 2 else 0.0
        return self.integrate_branch()(edges, center, lsf_fwhm, p0, p1, p2)

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
    def integrate_branch(self) -> Callable[..., Array]:
        """Return a JAX-compatible branch callable for ``lax.switch`` dispatch.

        The returned function must have the fixed signature::

            fn(edges, center, lsf_fwhm, p0, p1, p2) -> Array

        where *edges* has shape ``(E,)`` (pixel edges), *lsf_fwhm* has
        shape ``(E,)`` (instrumental LSF FWHM at each edge), and the
        returned array also has shape ``(E,)``.  The return is a
        cumulative profile array such that ``jnp.diff`` recovers per-pixel
        integrals.  Argument order matches :meth:`evaluate_branch`.

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
    def evaluate_branch(self) -> Callable[..., Array]:
        """Return a JAX-compatible branch callable for pointwise evaluation.

        The returned function must have the fixed signature::

            fn(wavelength, center, lsf_fwhm, p0, p1, p2) -> Array

        Returns the normalised profile value at each wavelength point.

        Returns
        -------
        callable
            A pure-JAX function suitable as a ``lax.switch`` branch.
        """

    def to_dict(self) -> dict:
        """Serialize to a YAML-safe dictionary.

        The default implementation returns ``{'type': type(self).__name__}``,
        which suffices for all no-argument profile subclasses.  Override for
        profiles that carry constructor parameters (e.g. order, degree).
        """
        return {'type': type(self).__name__}

    @classmethod
    def from_dict(cls, d: dict) -> Profile:
        """Deserialize from a dictionary.

        The default implementation calls ``cls()`` with no arguments, which
        suffices for all no-argument profile subclasses.  Override for
        profiles that require constructor parameters.
        """
        return cls()

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'


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

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss',)

    @override
    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_gauss': Uniform(0, 1000)}

    @override
    def integrate_branch(self):
        def _fn(edges, c, lsf, p0, p1, p2):
            # p0 = fwhm_gauss
            return functions.integrate_gaussian(edges, lsf, c, p0)

        return _fn

    @override
    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return functions.evaluate_gaussian(wavelength, c, lsf, p0)

        return _fn


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

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_lorentz',)

    @override
    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_lorentz': Uniform(0, 1000)}

    @override
    def integrate_branch(self):
        def _fn(edges, c, lsf, p0, p1, p2):
            # p0 = fwhm_lorentz; pure Cauchy via PseudoVoigt with zero Gaussian width
            return functions.integrate_voigt(edges, lsf, c, 0.0, p0)

        return _fn

    @override
    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return functions.evaluate_voigt(wavelength, c, lsf, 0.0, p0)

        return _fn


@_register
class PseudoVoigt(Profile):
    """Pseudo-Voigt line profile (Thompson et al. 1987).

    Requires two parameters: ``fwhm_gauss`` for the Gaussian component
    and ``fwhm_lorentz`` for the Lorentzian component. The instrumental
    LSF is added in quadrature to the Gaussian component.
    """

    code = 2

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss', 'fwhm_lorentz')

    @override
    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_gauss': Uniform(0, 1000), 'fwhm_lorentz': Uniform(0, 1000)}

    @override
    def integrate_branch(self):
        def _fn(edges, c, lsf, p0, p1, p2):
            # p0 = fwhm_gauss, p1 = fwhm_lorentz
            return functions.integrate_voigt(edges, lsf, c, p0, p1)

        return _fn

    @override
    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return functions.evaluate_voigt(wavelength, c, lsf, p0, p1)

        return _fn


@_register
class Laplace(Profile):
    """Laplace (double-exponential) line profile.

    Requires a single parameter ``fwhm_exp``.  The LSF is **not**
    convolved --- this profile is a pure Laplace distribution.
    """

    code = 3

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_exp',)

    @override
    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_exp': Uniform(0, 1000)}

    @override
    def integrate_branch(self):
        def _fn(edges, c, lsf, p0, p1, p2):
            # p0 = fwhm_exp; pure Laplace convolved with Gaussian LSF
            return functions.integrate_gaussianLaplace(edges, lsf, c, 0.0, p0)

        return _fn

    @override
    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return functions.evaluate_gaussianLaplace(wavelength, c, lsf, 0.0, p0)

        return _fn


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

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss', 'fwhm_exp')

    @override
    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_gauss': Uniform(0, 1000), 'fwhm_exp': Uniform(0, 1000)}

    @override
    def integrate_branch(self):
        def _fn(edges, c, lsf, p0, p1, p2):
            # p0 = fwhm_gauss, p1 = fwhm_exp
            return functions.integrate_gaussianLaplace(edges, lsf, c, p0, p1)

        return _fn

    @override
    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return functions.evaluate_gaussianLaplace(wavelength, c, lsf, p0, p1)

        return _fn


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

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss', 'h3', 'h4')

    @override
    def default_priors(self) -> dict[str, Prior]:
        return {
            'fwhm_gauss': Uniform(0, 1000),
            'h3': TruncatedNormal(loc=0, scale=0.1, low=-0.3, high=0.3),
            'h4': TruncatedNormal(loc=0, scale=0.1, low=-0.3, high=0.3),
        }

    @override
    def integrate_branch(self):
        def _fn(edges, c, lsf, p0, p1, p2):
            # p0 = fwhm_gauss, p1 = h3, p2 = h4
            return functions.integrate_gaussHermite(edges, lsf, c, p0, p1, p2)

        return _fn

    @override
    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return functions.evaluate_gaussHermite(wavelength, c, lsf, p0, p1, p2)

        return _fn


@_register
class SplitNormal(Profile):
    """Split-normal (two-sided Gaussian) line profile.

    A Gaussian with different standard deviations on each side of the mean.
    Requires two parameters: ``fwhm_blue`` for the blue (left) side and
    ``fwhm_red`` for the red (right) side. The instrumental LSF is added
    in quadrature to both components.
    """

    code = 6

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_blue', 'fwhm_red')

    @override
    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_blue': Uniform(0, 1000), 'fwhm_red': Uniform(0, 1000)}

    @override
    def integrate_branch(self):
        def _fn(edges, c, lsf, p0, p1, p2):
            # p0 = fwhm_blue, p1 = fwhm_red
            return functions.integrate_split_normal(edges, lsf, c, p0, p1)

        return _fn

    @override
    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return functions.evaluate_split_normal(wavelength, c, lsf, p0, p1)

        return _fn


@_register
class GaussianSplitLaplace(Profile):
    """Asymmetric Exponentially Modified Gaussian (asymmetric EMG) line profile.

    A Gaussian (with LSF) convolved with a split-Laplace (asymmetric
    double-exponential) distribution, where the blue (short-wavelength)
    and red (long-wavelength) exponential tails are controlled
    independently.  When ``fwhm_l_blue == fwhm_l_red`` this reduces
    exactly to :class:`SEMG`.

    Pixel integration is analytic via the closed-form antiderivative of
    the Gaussian-split-Laplace CDF.

    Requires three parameters: ``fwhm_gauss`` for the intrinsic Gaussian
    component, ``fwhm_l_blue`` for the blue-side Laplacian exponential
    tail, and ``fwhm_l_red`` for the red-side Laplacian exponential tail.
    The instrumental LSF is added in quadrature to the Gaussian component.
    """

    code = 10

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss', 'fwhm_l_blue', 'fwhm_l_red')

    @override
    def default_priors(self) -> dict[str, Prior]:
        return {
            'fwhm_gauss': Uniform(0, 1000),
            'fwhm_l_blue': Uniform(0, 1000),
            'fwhm_l_red': Uniform(0, 1000),
        }

    @override
    def integrate_branch(self):
        def _fn(edges, c, lsf, p0, p1, p2):
            # p0 = fwhm_gauss, p1 = fwhm_l_blue, p2 = fwhm_l_red
            return functions.integrate_gaussianSplitLaplace(edges, lsf, c, p0, p1, p2)

        return _fn

    @override
    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            # p0 = fwhm_gauss, p1 = fwhm_l_blue, p2 = fwhm_l_red
            return functions.evaluate_gaussianSplitLaplace(
                wavelength, c, lsf, p0, p1, p2
            )

        return _fn


@_register
class SkewNormal(Profile):
    """Skew-normal line profile.

    A Gaussian (with LSF) modulated by an erf skew factor
    ``[1 + erf(alpha_eff * (x - c) / w0)]``.  For ``alpha = 0`` this reduces
    exactly to a Gaussian.

    Unlike :class:`SkewVoigt`, the convolution with the Gaussian LSF is
    **exact**: the shape parameter rescales analytically as
    ``alpha_eff = alpha * sigma_g / sqrt(sigma_g^2 + (1 + alpha^2) sigma_lsf^2)``,
    with no numerical correction required.  Pixel integration uses the
    closed-form skew-normal CDF ``Phi(z) - 2 T(z, alpha_eff)`` via Owen's T
    function.  See ``docs/derivations/skew-normal.md`` for the full derivation.

    Requires two parameters: ``fwhm_gauss`` for the intrinsic Gaussian FWHM
    and ``alpha`` for the skewness (positive values shift flux redward).
    """

    code = 9

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss', 'alpha')

    @override
    def default_priors(self) -> dict[str, Prior]:
        return {
            'fwhm_gauss': Uniform(0, 1000),
            'alpha': TruncatedNormal(loc=0, scale=100, low=-300, high=300),
        }

    @override
    def integrate_branch(self):
        def _fn(edges, c, lsf, p0, p1, p2):
            # p0 = fwhm_gauss, p1 = alpha
            return functions.integrate_skewNormal(edges, lsf, c, p0, p1)

        return _fn

    @override
    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return functions.evaluate_skewNormal(wavelength, c, lsf, p0, p1)

        return _fn


@_register
class BoxGauss(Profile):
    """Boxcar distribution convolved with a Gaussian.

    The intrinsic profile is a uniform rectangular (boxcar) distribution of
    full width ``fwhm_box`` centred at zero (area = 1), convolved with a
    Gaussian whose FWHM is the quadrature sum of ``fwhm_gauss`` and
    ``lsf_fwhm``.  As ``fwhm_box`` → 0 the profile reduces to a pure
    Gaussian; as ``fwhm_gauss`` → 0 (and ``lsf_fwhm`` → 0) it approaches
    the sharp rectangular distribution.

    Requires two parameters: ``fwhm_box`` for the boxcar full width and
    ``fwhm_gauss`` for the intrinsic Gaussian component.
    """

    code = 8

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_box', 'fwhm_gauss')

    @override
    def default_priors(self) -> dict[str, Prior]:
        return {'fwhm_box': Uniform(0, 1000), 'fwhm_gauss': Uniform(0, 1000)}

    @override
    def integrate_branch(self):
        def _fn(edges, c, lsf, p0, p1, p2):
            # p0 = fwhm_box, p1 = fwhm_gauss
            return functions.integrate_boxGauss(edges, lsf, c, p0, p1)

        return _fn

    @override
    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return functions.evaluate_boxGauss(wavelength, c, lsf, p0, p1)

        return _fn


@_register
class SkewVoigt(Profile):
    r"""Skew pseudo-Voigt line profile.

    A pseudo-Voigt profile multiplied by a skew factor
    ``[1 + erf(alpha * (x - c) / w0)]``, where
    ``w0 = Gamma_V / (2 sqrt(ln 2)) = sigma_V * sqrt(2)`` is the erf scale
    derived from the Thompson et al. Voigt FWHM ``Gamma_V``.  For a pure
    Gaussian (``fwhm_lorentz = 0``) this reduces to the standard skew-normal
    with shape parameter ``alpha`` and dispersion ``sigma_g``.  The profile
    integrates to 1 for any value of ``alpha`` because the skew factor is odd
    and the pseudo-Voigt is even.

    Convolution with the Gaussian LSF rescales the skewness to an effective
    :math:`\\alpha_\\text{eff}` via the Gaussian-body exact formula with an
    FXIG2 boost correction for the Lorentzian component.  See
    ``docs/derivations/skew-voigt.md`` for the full derivation.

    Requires three parameters: ``fwhm_gauss`` for the Gaussian component,
    ``fwhm_lorentz`` for the Lorentzian component, and ``alpha`` for the
    skewness (positive values shift flux redward).
    """

    code = 7

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('fwhm_gauss', 'fwhm_lorentz', 'alpha')

    @override
    def default_priors(self) -> dict[str, Prior]:
        return {
            'fwhm_gauss': Uniform(0, 1000),
            'fwhm_lorentz': Uniform(0, 1000),
            'alpha': TruncatedNormal(loc=0, scale=1, low=-5, high=5),
        }

    @override
    def integrate_branch(self):
        def _fn(edges, c, lsf, p0, p1, p2):
            # p0 = fwhm_gauss, p1 = fwhm_lorentz, p2 = alpha
            return functions.integrate_skewVoigt(edges, lsf, c, p0, p1, p2)

        return _fn

    @override
    def evaluate_branch(self):
        def _fn(wavelength, c, lsf, p0, p1, p2):
            return functions.evaluate_skewVoigt(wavelength, c, lsf, p0, p1, p2)

        return _fn


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
    'skew-voigt': SkewVoigt(),
    'skewvoigt': SkewVoigt(),
    'skew-normal': SkewNormal(),
    'skewnormal': SkewNormal(),
    'boxgauss': BoxGauss(),
    'box-gauss': BoxGauss(),
    'boxcar': BoxGauss(),
    'gaussiansplitlaplace': GaussianSplitLaplace(),
    'gaussian-split-laplace': GaussianSplitLaplace(),
    'asymmetric-emg': GaussianSplitLaplace(),
    'aemg': GaussianSplitLaplace(),
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


# Re-export from compute module for backward compatibility.
from unite.line.compute import integrate_lines, evaluate_lines  # noqa: I001, E402, F401
