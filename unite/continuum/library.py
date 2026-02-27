"""Continuum functional forms: abstract base and concrete implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from unite.continuum.functions import (
    bernstein_eval,
    bspline_eval,
    chebval,
    planck_function,
)
from unite.prior import Prior, Uniform

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FORM_REGISTRY: dict[str, type[ContinuumForm]] = {}


def _register(cls: type[ContinuumForm]) -> type[ContinuumForm]:
    """Register a ContinuumForm subclass for deserialization."""
    _FORM_REGISTRY[cls.__name__] = cls
    return cls


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ContinuumForm(ABC):
    """Abstract base class for continuum functional forms.

    Each subclass defines a parameterised continuum model that can be
    evaluated at an array of wavelengths.  Forms are stateless (or carry
    only static configuration like polynomial degree) and are fully
    serialisable for YAML round-trip.

    Subclasses must implement :meth:`param_names`, :meth:`default_priors`,
    :meth:`evaluate`, :meth:`to_dict`, and :meth:`from_dict`.
    """

    @abstractmethod
    def param_names(self) -> tuple[str, ...]:
        """Names of the parameters this form requires.

        Returns
        -------
        tuple of str
            Parameter names (e.g. ``('angle', 'offset')``).
        """

    @abstractmethod
    def default_priors(self) -> dict[str, Prior]:
        """Return sensible default priors for each parameter.

        The keys must match :meth:`param_names`.

        Returns
        -------
        dict of str to Prior
        """

    @property
    def n_params(self) -> int:
        """Number of free parameters."""
        return len(self.param_names())

    @abstractmethod
    def evaluate(
        self, wavelength: ArrayLike, center: float, params: dict[str, ArrayLike]
    ) -> Array:
        """Evaluate the continuum model.

        All operations must use :mod:`jax.numpy` so the function is
        compatible with JAX tracing / JIT compilation.

        Parameters
        ----------
        wavelength : ArrayLike
            Wavelength array.
        center : float
            Reference wavelength (typically the midpoint of the
            continuum region).
        params : dict of str to ArrayLike
            Parameter values keyed by :meth:`param_names`.

        Returns
        -------
        Array
            Continuum flux at each wavelength.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a YAML-safe dictionary."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> ContinuumForm:
        """Deserialize from a dictionary."""

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self).__name__)

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'


# ---------------------------------------------------------------------------
# Deserialization helper
# ---------------------------------------------------------------------------


def form_from_dict(d: dict) -> ContinuumForm:
    """Reconstruct a :class:`ContinuumForm` from a serialised dictionary.

    Parameters
    ----------
    d : dict
        Must contain a ``'type'`` key matching a registered form name.

    Returns
    -------
    ContinuumForm

    Raises
    ------
    ValueError
        If the type is not recognised.
    """
    type_name = d['type']
    if type_name not in _FORM_REGISTRY:
        msg = f'Unknown ContinuumForm type: {type_name!r}'
        raise ValueError(msg)
    return _FORM_REGISTRY[type_name].from_dict(d)


# ---------------------------------------------------------------------------
# Piecewise / region-local forms
# ---------------------------------------------------------------------------


@_register
class Linear(ContinuumForm):
    """Linear continuum: ``tan(angle) * (wavelength - center) + offset``.

    Uses an angle parameterisation for the slope, which is better behaved
    for sampling than a raw slope coefficient.

    Parameters (sampled)
    --------------------
    angle : float
        Slope angle in radians.
    offset : float
        Intercept at the region centre.
    """

    def param_names(self) -> tuple[str, ...]:
        return ('angle', 'offset')

    def default_priors(self) -> dict[str, Prior]:
        return {'angle': Uniform(-jnp.pi / 2, jnp.pi / 2), 'offset': Uniform(-10, 10)}

    def evaluate(
        self, wavelength: ArrayLike, center: float, params: dict[str, ArrayLike]
    ) -> Array:
        return jnp.tan(params['angle']) * (wavelength - center) + params['offset']

    def to_dict(self) -> dict:
        return {'type': 'Linear'}

    @classmethod
    def from_dict(cls, d: dict) -> Linear:
        return cls()


@_register
class PowerLaw(ContinuumForm):
    """Power-law continuum: ``amplitude * (wavelength / center) ** beta``.

    Parameters (sampled)
    --------------------
    amplitude : float
        Normalisation at the region centre.
    beta : float
        Power-law index.
    """

    def param_names(self) -> tuple[str, ...]:
        return ('amplitude', 'beta')

    def default_priors(self) -> dict[str, Prior]:
        return {'amplitude': Uniform(0, 10), 'beta': Uniform(-5, 5)}

    def evaluate(
        self, wavelength: ArrayLike, center: float, params: dict[str, ArrayLike]
    ) -> Array:
        return params['amplitude'] * (wavelength / center) ** params['beta']

    def to_dict(self) -> dict:
        return {'type': 'PowerLaw'}

    @classmethod
    def from_dict(cls, d: dict) -> PowerLaw:
        return cls()


@_register
class Polynomial(ContinuumForm):
    """Polynomial continuum of configurable degree.

    Evaluates ``c0 + c1 * x + c2 * x**2 + ...`` where
    ``x = wavelength - center``.

    Parameters
    ----------
    degree : int
        Polynomial degree (default 1).
    """

    def __init__(self, degree: int = 1) -> None:
        if degree < 0:
            msg = f'Polynomial degree must be >= 0, got {degree}'
            raise ValueError(msg)
        self._degree = degree

    @property
    def degree(self) -> int:
        """Polynomial degree."""
        return self._degree

    def param_names(self) -> tuple[str, ...]:
        return tuple(f'c{i}' for i in range(self._degree + 1))

    def default_priors(self) -> dict[str, Prior]:
        return {f'c{i}': Uniform(-10, 10) for i in range(self._degree + 1)}

    def evaluate(
        self, wavelength: ArrayLike, center: float, params: dict[str, ArrayLike]
    ) -> Array:
        x = wavelength - center
        result = jnp.zeros_like(x, dtype=float)
        for i in range(self._degree + 1):
            result = result + params[f'c{i}'] * x**i
        return result

    def to_dict(self) -> dict:
        return {'type': 'Polynomial', 'degree': self._degree}

    @classmethod
    def from_dict(cls, d: dict) -> Polynomial:
        return cls(degree=d['degree'])

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._degree == other._degree  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(('Polynomial', self._degree))

    def __repr__(self) -> str:
        return f'Polynomial(degree={self._degree})'


@_register
class Chebyshev(ContinuumForm):
    """Chebyshev polynomial continuum of configurable order.

    Evaluates a Chebyshev series on coordinates normalized to ``[-1, 1]``
    within the continuum region.  Numerically more stable than a standard
    polynomial basis for higher orders.

    The x-coordinate is ``(wavelength - center) / half_width`` where
    *half_width* should be set to the half-width of the region for proper
    orthogonality.  Defaults to ``1.0`` (identity normalization).

    Parameters
    ----------
    order : int
        Chebyshev order (default 2).  Number of coefficients = order + 1.
    half_width : float
        Half-width of the continuum region in the same units as wavelength.
        Set to ``(high - low) / 2`` of the :class:`ContinuumRegion`.
        Default ``1.0``.
    """

    def __init__(self, order: int = 2, half_width: float = 1.0) -> None:
        if order < 0:
            msg = f'Chebyshev order must be >= 0, got {order}'
            raise ValueError(msg)
        self._order = order
        self._half_width = half_width

    @property
    def order(self) -> int:
        """Chebyshev order."""
        return self._order

    @property
    def half_width(self) -> float:
        """Normalization half-width."""
        return self._half_width

    def param_names(self) -> tuple[str, ...]:
        return tuple(f'c{i}' for i in range(self._order + 1))

    def default_priors(self) -> dict[str, Prior]:
        return {f'c{i}': Uniform(-10, 10) for i in range(self._order + 1)}

    def evaluate(
        self, wavelength: ArrayLike, center: float, params: dict[str, ArrayLike]
    ) -> Array:
        x = (wavelength - center) / self._half_width
        coeffs = [params[f'c{i}'] for i in range(self._order + 1)]
        return chebval(x, coeffs)

    def to_dict(self) -> dict:
        return {
            'type': 'Chebyshev',
            'order': self._order,
            'half_width': self._half_width,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Chebyshev:
        return cls(order=d['order'], half_width=d.get('half_width', 1.0))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._order == other._order and self._half_width == other._half_width  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(('Chebyshev', self._order, self._half_width))

    def __repr__(self) -> str:
        return f'Chebyshev(order={self._order}, half_width={self._half_width})'


# ---------------------------------------------------------------------------
# Global (non-piecewise) physical forms
# ---------------------------------------------------------------------------


@_register
class Blackbody(ContinuumForm):
    """Planck blackbody continuum normalized at a pivot wavelength.

    Evaluates ``amplitude * B_λ(T) / B_λ(pivot, T)`` so that *amplitude*
    directly represents the continuum flux at *pivot_micron*.

    Parameters
    ----------
    pivot_micron : float
        Normalization wavelength in microns (default ``1.0``).

    Parameters (sampled)
    --------------------
    amplitude : float
        Continuum flux at the pivot wavelength.
    temperature : float
        Blackbody temperature in Kelvin.
    """

    def __init__(self, pivot_micron: float = 1.0) -> None:
        self._pivot_micron = pivot_micron

    @property
    def pivot_micron(self) -> float:
        """Pivot wavelength in microns."""
        return self._pivot_micron

    def param_names(self) -> tuple[str, ...]:
        return ('amplitude', 'temperature')

    def default_priors(self) -> dict[str, Prior]:
        return {'amplitude': Uniform(0, 10), 'temperature': Uniform(100, 50000)}

    def evaluate(
        self, wavelength: ArrayLike, center: float, params: dict[str, ArrayLike]
    ) -> Array:
        bb = planck_function(wavelength, params['temperature'], self._pivot_micron)
        return params['amplitude'] * bb

    def to_dict(self) -> dict:
        return {'type': 'Blackbody', 'pivot_micron': self._pivot_micron}

    @classmethod
    def from_dict(cls, d: dict) -> Blackbody:
        return cls(pivot_micron=d.get('pivot_micron', 1.0))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._pivot_micron == other._pivot_micron  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(('Blackbody', self._pivot_micron))

    def __repr__(self) -> str:
        return f'Blackbody(pivot_micron={self._pivot_micron})'


@_register
class ModifiedBlackbody(ContinuumForm):
    """Modified blackbody: ``amplitude * B_λ(T) * (λ / pivot)^beta``.

    The power-law modifier *beta* broadens (beta > 0) or narrows (beta < 0)
    the SED relative to a pure blackbody.  *beta = 0* recovers
    :class:`Blackbody`.

    Parameters
    ----------
    pivot_micron : float
        Normalization wavelength in microns (default ``1.0``).

    Parameters (sampled)
    --------------------
    amplitude : float
        Continuum flux at the pivot wavelength.
    temperature : float
        Blackbody temperature in Kelvin.
    beta : float
        Power-law modifier index.
    """

    def __init__(self, pivot_micron: float = 1.0) -> None:
        self._pivot_micron = pivot_micron

    @property
    def pivot_micron(self) -> float:
        """Pivot wavelength in microns."""
        return self._pivot_micron

    def param_names(self) -> tuple[str, ...]:
        return ('amplitude', 'temperature', 'beta')

    def default_priors(self) -> dict[str, Prior]:
        return {
            'amplitude': Uniform(0, 10),
            'temperature': Uniform(100, 50000),
            'beta': Uniform(-4, 4),
        }

    def evaluate(
        self, wavelength: ArrayLike, center: float, params: dict[str, ArrayLike]
    ) -> Array:
        bb = planck_function(wavelength, params['temperature'], self._pivot_micron)
        modifier = (wavelength / self._pivot_micron) ** params['beta']
        return params['amplitude'] * bb * modifier

    def to_dict(self) -> dict:
        return {'type': 'ModifiedBlackbody', 'pivot_micron': self._pivot_micron}

    @classmethod
    def from_dict(cls, d: dict) -> ModifiedBlackbody:
        return cls(pivot_micron=d.get('pivot_micron', 1.0))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._pivot_micron == other._pivot_micron  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(('ModifiedBlackbody', self._pivot_micron))

    def __repr__(self) -> str:
        return f'ModifiedBlackbody(pivot_micron={self._pivot_micron})'


@_register
class AttenuatedBlackbody(ContinuumForm):
    """Dust-attenuated blackbody continuum.

    Evaluates ``amplitude * B_λ(T) * exp(-tau_v * [(λ/lambda_v)^alpha - (pivot/lambda_v)^alpha])``.

    Extinction is normalized at the pivot wavelength so that *amplitude*
    represents the **observed** (attenuated) flux there.  Negative *alpha*
    gives steeper extinction at short wavelengths (typical dust law).

    Parameters
    ----------
    pivot_micron : float
        Normalization wavelength in microns (default ``1.0``).
    lambda_v_micron : float
        Reference wavelength for the extinction law in microns
        (default ``0.55``, corresponding to optical V band).

    Parameters (sampled)
    --------------------
    amplitude : float
        Observed continuum flux at the pivot wavelength.
    temperature : float
        Blackbody temperature in Kelvin.
    tau_v : float
        Optical depth at *lambda_v_micron*.
    alpha : float
        Dust extinction power-law index (negative = steeper at short λ).
    """

    def __init__(
        self, pivot_micron: float = 1.0, lambda_v_micron: float = 0.55
    ) -> None:
        self._pivot_micron = pivot_micron
        self._lambda_v_micron = lambda_v_micron

    @property
    def pivot_micron(self) -> float:
        """Pivot wavelength in microns."""
        return self._pivot_micron

    @property
    def lambda_v_micron(self) -> float:
        """Extinction reference wavelength in microns."""
        return self._lambda_v_micron

    def param_names(self) -> tuple[str, ...]:
        return ('amplitude', 'temperature', 'tau_v', 'alpha')

    def default_priors(self) -> dict[str, Prior]:
        return {
            'amplitude': Uniform(0, 10),
            'temperature': Uniform(100, 50000),
            'tau_v': Uniform(0, 5),
            'alpha': Uniform(-2, 0),
        }

    def evaluate(
        self, wavelength: ArrayLike, center: float, params: dict[str, ArrayLike]
    ) -> Array:
        bb = planck_function(wavelength, params['temperature'], self._pivot_micron)
        ext_data = (wavelength / self._lambda_v_micron) ** params['alpha']
        ext_pivot = (self._pivot_micron / self._lambda_v_micron) ** params['alpha']
        extinction = jnp.exp(-params['tau_v'] * (ext_data - ext_pivot))
        return params['amplitude'] * bb * extinction

    def to_dict(self) -> dict:
        return {
            'type': 'AttenuatedBlackbody',
            'pivot_micron': self._pivot_micron,
            'lambda_v_micron': self._lambda_v_micron,
        }

    @classmethod
    def from_dict(cls, d: dict) -> AttenuatedBlackbody:
        return cls(
            pivot_micron=d.get('pivot_micron', 1.0),
            lambda_v_micron=d.get('lambda_v_micron', 0.55),
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (  # type: ignore[attr-defined]
            self._pivot_micron == other._pivot_micron
            and self._lambda_v_micron == other._lambda_v_micron
        )

    def __hash__(self) -> int:
        return hash(('AttenuatedBlackbody', self._pivot_micron, self._lambda_v_micron))

    def __repr__(self) -> str:
        return (
            f'AttenuatedBlackbody(pivot_micron={self._pivot_micron}, '
            f'lambda_v_micron={self._lambda_v_micron})'
        )


@_register
class BSpline(ContinuumForm):
    """Global B-spline continuum with local knot control.

    The knot vector must be set via *knots* at construction time (typically
    derived from the wavelength coverage of the spectrum).

    Parameters
    ----------
    knots : array-like
        Clamped knot vector.  The number of basis functions is
        ``len(knots) - degree - 1``.
    degree : int
        Spline degree (default 3 for cubic).

    Parameters (sampled)
    --------------------
    coeff_0, coeff_1, …, coeff_{n_basis-1} : float
        B-spline coefficients.
    """

    def __init__(self, knots, degree: int = 3) -> None:
        self._knots = jnp.asarray(knots)
        self._degree = degree
        self._n_basis = len(self._knots) - degree - 1

    @property
    def n_basis(self) -> int:
        """Number of B-spline basis functions."""
        return self._n_basis

    def param_names(self) -> tuple[str, ...]:
        return tuple(f'coeff_{i}' for i in range(self._n_basis))

    def default_priors(self) -> dict[str, Prior]:
        return {f'coeff_{i}': Uniform(-10, 10) for i in range(self._n_basis)}

    def evaluate(
        self, wavelength: ArrayLike, center: float, params: dict[str, ArrayLike]
    ) -> Array:
        coeffs = jnp.stack([params[f'coeff_{i}'] for i in range(self._n_basis)])
        return bspline_eval(wavelength, coeffs, self._knots, self._degree)

    def to_dict(self) -> dict:
        return {
            'type': 'BSpline',
            'knots': self._knots.tolist(),
            'degree': self._degree,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BSpline:
        return cls(knots=d['knots'], degree=d.get('degree', 3))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (  # type: ignore[attr-defined]
            bool(jnp.array_equal(self._knots, other._knots))
            and self._degree == other._degree
        )

    def __hash__(self) -> int:
        return hash(('BSpline', tuple(float(k) for k in self._knots), self._degree))

    def __repr__(self) -> str:
        return f'BSpline(n_basis={self._n_basis}, degree={self._degree})'


@_register
class Bernstein(ContinuumForm):
    """Global Bernstein polynomial continuum with positivity guarantee.

    Bernstein basis polynomials are non-negative on ``[0, 1]``, so fitting
    with positive coefficients guarantees a positive continuum everywhere.

    Parameters
    ----------
    degree : int
        Polynomial degree (default 4).  Number of coefficients = degree + 1.
    wavelength_min, wavelength_max : float
        Wavelength range for normalization to ``[0, 1]``.

    Parameters (sampled)
    --------------------
    coeff_0, coeff_1, …, coeff_{degree} : float
        Bernstein coefficients (positive values give positive continuum).
    """

    def __init__(
        self, degree: int = 4, wavelength_min: float = 0.0, wavelength_max: float = 1.0
    ) -> None:
        from scipy.special import comb

        self._degree = degree
        self._wavelength_min = wavelength_min
        self._wavelength_max = wavelength_max
        self._binom = jnp.array(
            [comb(degree, i, exact=True) for i in range(degree + 1)], dtype=float
        )

    @property
    def degree(self) -> int:
        """Polynomial degree."""
        return self._degree

    def param_names(self) -> tuple[str, ...]:
        return tuple(f'coeff_{i}' for i in range(self._degree + 1))

    def default_priors(self) -> dict[str, Prior]:
        return {f'coeff_{i}': Uniform(0, 10) for i in range(self._degree + 1)}

    def evaluate(
        self, wavelength: ArrayLike, center: float, params: dict[str, ArrayLike]
    ) -> Array:
        coeffs = jnp.stack([params[f'coeff_{i}'] for i in range(self._degree + 1)])
        return bernstein_eval(
            wavelength, coeffs, self._wavelength_min, self._wavelength_max, self._binom
        )

    def to_dict(self) -> dict:
        return {
            'type': 'Bernstein',
            'degree': self._degree,
            'wavelength_min': self._wavelength_min,
            'wavelength_max': self._wavelength_max,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Bernstein:
        return cls(
            degree=d['degree'],
            wavelength_min=d.get('wavelength_min', 0.0),
            wavelength_max=d.get('wavelength_max', 1.0),
        )

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (  # type: ignore[attr-defined]
            self._degree == other._degree
            and self._wavelength_min == other._wavelength_min
            and self._wavelength_max == other._wavelength_max
        )

    def __hash__(self) -> int:
        return hash(
            ('Bernstein', self._degree, self._wavelength_min, self._wavelength_max)
        )

    def __repr__(self) -> str:
        return f'Bernstein(degree={self._degree})'
