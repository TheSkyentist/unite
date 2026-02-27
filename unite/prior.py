"""Declarative, serializable prior distributions for model parameters.

Priors are inert data containers that describe numpyro distributions.
At model-build time, a topological sort of their dependencies determines
sampling order, and each prior's :meth:`to_dist` method receives a context
dict of already-sampled values to resolve :class:`ParameterRef` expressions.

Examples
--------
Simple fixed-bound prior:

>>> from unite.config.prior import Uniform
>>> p = Uniform(0, 750)
>>> p.to_dist({})
Uniform(low=0, high=750)

Dependent prior using arithmetic on parameter tokens:

>>> from unite.config.base import FWHM
>>> from unite.config.prior import Uniform
>>> narrow = FWHM('narrow', prior=Uniform(0, 750))
>>> broad = FWHM('broad', prior=Uniform(low=narrow * 2 + 150, high=2500))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpyro.distributions as dist

# -------------------------------------------------------------------
# ParameterRef: arithmetic expression referencing a parameter token
# -------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterRef:
    """A linear expression referencing a parameter token.

    Created by arithmetic on :class:`~unite.config.base.FWHM` or
    :class:`~unite.config.base.Redshift` objects.  The resolved value is
    ``context[param] * scale + offset``.

    Parameters
    ----------
    param : object
        The FWHM or Redshift token this expression references.
    scale : float
        Multiplicative factor.
    offset : float
        Additive offset (applied after scaling).
    """

    param: object
    scale: float = 1.0
    offset: float = 0.0

    def resolve(self, context: dict) -> float:
        """Evaluate against already-sampled parameter values.

        Parameters
        ----------
        context : dict
            Mapping of parameter token objects to their sampled values.

        Returns
        -------
        float
        """
        return context[self.param] * self.scale + self.offset

    # -- Chained arithmetic -------------------------------------------

    def __mul__(self, other: int | float) -> ParameterRef:
        if not isinstance(other, int | float):
            return NotImplemented
        return ParameterRef(self.param, self.scale * other, self.offset * other)

    def __rmul__(self, other: int | float) -> ParameterRef:
        return self.__mul__(other)

    def __truediv__(self, other: int | float) -> ParameterRef:
        if not isinstance(other, int | float):
            return NotImplemented
        return ParameterRef(self.param, self.scale / other, self.offset / other)

    def __add__(self, other: int | float) -> ParameterRef:
        if not isinstance(other, int | float):
            return NotImplemented
        return ParameterRef(self.param, self.scale, self.offset + other)

    def __radd__(self, other: int | float) -> ParameterRef:
        return self.__add__(other)

    def __sub__(self, other: int | float) -> ParameterRef:
        if not isinstance(other, int | float):
            return NotImplemented
        return ParameterRef(self.param, self.scale, self.offset - other)

    def __rsub__(self, other: int | float) -> ParameterRef:
        if not isinstance(other, int | float):
            return NotImplemented
        return ParameterRef(self.param, -self.scale, other - self.offset)

    def __repr__(self) -> str:
        parts = []
        if self.scale != 1.0:
            parts.append(f'{self.scale} *')
        label = getattr(self.param, 'name', None) or type(self.param).__name__
        parts.append(label)
        if self.offset > 0:
            parts.append(f'+ {self.offset}')
        elif self.offset < 0:
            parts.append(f'- {-self.offset}')
        return f'Reference({" ".join(parts)})'


# -------------------------------------------------------------------
# Bound type alias
# -------------------------------------------------------------------

Bound = float | ParameterRef
"""Type for a prior bound: either a fixed float or a parameter expression."""


# -------------------------------------------------------------------
# Prior ABC
# -------------------------------------------------------------------


class Prior(ABC):
    """Abstract base class for declarative prior descriptions.

    Subclasses must implement :meth:`to_dist`, :meth:`dependencies`,
    :meth:`to_dict`, and the classmethod :meth:`from_dict`.
    """

    @abstractmethod
    def to_dist(self, context: dict) -> dist.Distribution | None:
        """Convert to a numpyro distribution, or ``None`` for fixed values.

        Parameters
        ----------
        context : dict
            Mapping of parameter token objects to already-sampled values.
            Required for resolving :class:`ParameterRef` expressions.

        Returns
        -------
        numpyro.distributions.Distribution or None
            ``None`` indicates the parameter is fixed and should not be
            sampled.  The caller is responsible for injecting the fixed
            value (from :attr:`~Fixed.value`) into the model context.
        """

    @abstractmethod
    def dependencies(self) -> set:
        """Return parameter token objects this prior depends on.

        Returns
        -------
        set
            Set of FWHM / Redshift objects.  Empty for independent priors.
        """

    @abstractmethod
    def to_dict(self, param_namer: dict | None = None) -> dict:
        """Serialize to a YAML-safe dictionary.

        Parameters
        ----------
        param_namer : dict, optional
            Mapping from parameter token objects to string names,
            used to serialize :class:`ParameterRef` bounds.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> Prior:
        """Deserialize from a dictionary."""


# -------------------------------------------------------------------
# Bound helpers
# -------------------------------------------------------------------


def _resolve_bound(value: Bound, context: dict) -> float:
    """Resolve a bound that may be a fixed float or a ParameterRef."""
    if isinstance(value, ParameterRef):
        return value.resolve(context)
    return value


def _bound_deps(value: Bound) -> set:
    """Extract parameter token dependencies from a bound."""
    if isinstance(value, ParameterRef):
        return {value.param}
    return set()


def _serialize_bound(value: Bound, param_namer: dict | None) -> float | dict:
    """Serialize a bound to a YAML-safe value."""
    if isinstance(value, ParameterRef):
        if param_namer is None:
            msg = 'Cannot serialize ParameterRef without param_namer'
            raise ValueError(msg)
        d: dict = {'ref': param_namer[value.param]}
        if value.offset != 0.0:
            d['offset'] = value.offset
        if value.scale != 1.0:
            d['scale'] = value.scale
        return d
    return value


def _deserialize_bound(value: float | dict, token_registry: dict | None) -> Bound:
    """Deserialize a bound from a YAML-safe value.

    Parameters
    ----------
    value : float or dict
        The serialized bound.
    token_registry : dict, optional
        Mapping from string names to parameter token objects,
        used to resolve serialized :class:`ParameterRef` bounds.
    """
    if isinstance(value, dict):
        if token_registry is None:
            msg = 'Cannot deserialize ParameterRef without token_registry'
            raise ValueError(msg)
        param = token_registry[value['ref']]
        return ParameterRef(
            param=param, scale=value.get('scale', 1.0), offset=value.get('offset', 0.0)
        )
    return float(value)


# -------------------------------------------------------------------
# Concrete priors
# -------------------------------------------------------------------


class Uniform(Prior):
    """Uniform prior with bounds that may reference other parameters.

    Parameters
    ----------
    low : float or ParameterRef
        Lower bound.
    high : float or ParameterRef
        Upper bound.

    Examples
    --------
    Fixed bounds:

    >>> Uniform(0, 750)

    Dependent bound (broad fwhm > narrow fwhm + 150 km/s):

    >>> Uniform(low=narrow_fwhm * 2 + 150, high=2500)
    """

    def __init__(self, low: Bound = 0, high: Bound = 1) -> None:
        self.low = low if isinstance(low, ParameterRef) else float(low)
        self.high = high if isinstance(high, ParameterRef) else float(high)

    def to_dist(self, context: dict) -> dist.Distribution:
        return dist.Uniform(
            _resolve_bound(self.low, context), _resolve_bound(self.high, context)
        )

    def dependencies(self) -> set:
        return _bound_deps(self.low) | _bound_deps(self.high)

    def to_dict(self, param_namer: dict | None = None) -> dict:
        return {
            'type': 'Uniform',
            'low': _serialize_bound(self.low, param_namer),
            'high': _serialize_bound(self.high, param_namer),
        }

    @classmethod
    def from_dict(cls, d: dict, token_registry: dict | None = None) -> Uniform:
        return cls(
            low=_deserialize_bound(d['low'], token_registry),
            high=_deserialize_bound(d['high'], token_registry),
        )

    def __repr__(self) -> str:
        return f'Uniform(low={self.low!r}, high={self.high!r})'


class TruncatedNormal(Prior):
    """Truncated normal prior with bounds that may reference other parameters.

    Parameters
    ----------
    loc : float or ParameterRef
        Mean of the underlying normal distribution.
    scale : float
        Standard deviation of the underlying normal distribution.
    low : float or ParameterRef
        Lower truncation bound.
    high : float or ParameterRef
        Upper truncation bound.
    """

    def __init__(self, loc: Bound, scale: float, low: Bound, high: Bound) -> None:
        self.loc = loc if isinstance(loc, ParameterRef) else float(loc)
        self.scale = float(scale)
        self.low = low if isinstance(low, ParameterRef) else float(low)
        self.high = high if isinstance(high, ParameterRef) else float(high)

    def to_dist(self, context: dict) -> dist.Distribution:
        return dist.TruncatedNormal(
            loc=_resolve_bound(self.loc, context),
            scale=self.scale,
            low=_resolve_bound(self.low, context),
            high=_resolve_bound(self.high, context),
        )

    def dependencies(self) -> set:
        return _bound_deps(self.loc) | _bound_deps(self.low) | _bound_deps(self.high)

    def to_dict(self, param_namer: dict | None = None) -> dict:
        return {
            'type': 'TruncatedNormal',
            'loc': _serialize_bound(self.loc, param_namer),
            'scale': self.scale,
            'low': _serialize_bound(self.low, param_namer),
            'high': _serialize_bound(self.high, param_namer),
        }

    @classmethod
    def from_dict(cls, d: dict, token_registry: dict | None = None) -> TruncatedNormal:
        return cls(
            loc=_deserialize_bound(d['loc'], token_registry),
            scale=d['scale'],
            low=_deserialize_bound(d['low'], token_registry),
            high=_deserialize_bound(d['high'], token_registry),
        )

    def __repr__(self) -> str:
        return (
            f'TruncatedNormal(loc={self.loc!r}, scale={self.scale}, '
            f'low={self.low!r}, high={self.high!r})'
        )


class Fixed(Prior):
    """A fixed (non-sampled) constant value.

    ``Fixed`` parameters are injected directly into the model context as
    constants rather than being drawn from a distribution.  This avoids
    Delta distributions, which are not differentiable and would break
    gradient-based samplers.

    Parameters
    ----------
    value : float or int
        The constant value.

    Examples
    --------
    >>> Fixed(6564.61)
    Fixed(6564.61)
    """

    def __init__(self, value: float | int) -> None:
        if not isinstance(value, int | float):
            msg = f'Fixed value must be int or float, got {type(value).__name__}'
            raise TypeError(msg)
        self.value = float(value)

    def to_dist(self, context: dict) -> None:
        """Return ``None`` — the parameter is constant and must not be sampled."""
        return None

    def dependencies(self) -> set:
        return set()

    def to_dict(self, param_namer: dict | None = None) -> dict:
        return {'type': 'Fixed', 'value': self.value}

    @classmethod
    def from_dict(cls, d: dict, token_registry: dict | None = None) -> Fixed:
        return cls(d['value'])

    def __repr__(self) -> str:
        return f'Fixed({self.value!r})'


# -------------------------------------------------------------------
# Registry for deserialization
# -------------------------------------------------------------------

_PRIOR_REGISTRY: dict[str, type[Prior]] = {
    'Uniform': Uniform,
    'TruncatedNormal': TruncatedNormal,
    'Fixed': Fixed,
}


def prior_from_dict(d: dict, token_registry: dict | None = None) -> Prior:
    """Deserialize a Prior from a dictionary using the 'type' key.

    Parameters
    ----------
    d : dict
        Dictionary with a ``'type'`` key matching a registered prior class.
    token_registry : dict, optional
        Mapping from string names to parameter token objects.

    Returns
    -------
    Prior

    Raises
    ------
    KeyError
        If the type is not registered.
    """
    cls = _PRIOR_REGISTRY[d['type']]
    return cls.from_dict(d, token_registry=token_registry)


# -------------------------------------------------------------------
# Parameter token
# -------------------------------------------------------------------


class Parameter:
    """A named, shareable model parameter token.

    Any object referencing the same ``Parameter`` instance shares a single
    sampled value in the fitted model.  Sharing is identity-based — pass the
    **same instance** to multiple lines or dispersers.

    Arithmetic on a ``Parameter`` produces a :class:`ParameterRef` expression
    that can be used as a prior bound, enabling dependent priors such as
    ``broad_fwhm > narrow_fwhm + 150 km/s``.

    Parameters
    ----------
    name : str, optional
        Human-readable label used as the numpyro site name.  May be ``None``
        when the token will be auto-named later (e.g. by
        :class:`~unite.line.config.LineConfiguration`).
    prior : Prior
        Prior distribution for this parameter.

    Raises
    ------
    TypeError
        If any dependency of *prior* references a parameter that is not an
        instance of the same subclass as *self*.  Cross-kind
        :class:`ParameterRef` expressions (e.g. an ``FWHM`` bound on a
        ``Redshift`` prior) are forbidden.
    """

    def __init__(self, name: str | None = None, *, prior: Prior) -> None:
        for dep in prior.dependencies():
            if not isinstance(dep, type(self)):
                msg = (
                    f'{type(self).__name__} prior references a '
                    f'{type(dep).__name__} parameter. ParameterRefs must '
                    f'reference the same kind of parameter.'
                )
                raise TypeError(msg)
        self.name = name
        self.prior = prior

    def __repr__(self) -> str:
        """Return a readable string representation."""
        parts = []
        if self.name is not None:
            parts.append(repr(self.name))
        parts.append(f'prior={self.prior!r}')
        return f'{self.__class__.__name__}({", ".join(parts)})'

    def __mul__(self, other: int | float) -> ParameterRef:
        """Return a scaled :class:`ParameterRef`."""
        if not isinstance(other, int | float):
            return NotImplemented
        return ParameterRef(self, scale=float(other))

    def __rmul__(self, other: int | float) -> ParameterRef:
        """Return a scaled :class:`ParameterRef`."""
        return self.__mul__(other)

    def __truediv__(self, other: int | float) -> ParameterRef:
        """Return a :class:`ParameterRef` divided by a scalar."""
        if not isinstance(other, int | float):
            return NotImplemented
        return ParameterRef(self, scale=1.0 / float(other))

    def __add__(self, other: int | float) -> ParameterRef:
        """Return a :class:`ParameterRef` with an additive offset."""
        if not isinstance(other, int | float):
            return NotImplemented
        return ParameterRef(self, offset=float(other))

    def __radd__(self, other: int | float) -> ParameterRef:
        """Return a :class:`ParameterRef` with an additive offset."""
        return self.__add__(other)

    def __sub__(self, other: int | float) -> ParameterRef:
        """Return a :class:`ParameterRef` with a subtractive offset."""
        if not isinstance(other, int | float):
            return NotImplemented
        return ParameterRef(self, offset=-float(other))

    def __rsub__(self, other: int | float) -> ParameterRef:
        """Return a negated :class:`ParameterRef` with an offset."""
        if not isinstance(other, int | float):
            return NotImplemented
        return ParameterRef(self, scale=-1.0, offset=float(other))


# -------------------------------------------------------------------
# Topological sort for dependency resolution
# -------------------------------------------------------------------


def topological_sort(
    named_priors: dict[str, Prior], param_to_name: dict[object, str]
) -> list[str]:
    """Sort parameter names so that dependencies are sampled first.

    Parameters
    ----------
    named_priors : dict
        Mapping of parameter string names to :class:`Prior` objects.
    param_to_name : dict
        Mapping of parameter token objects to their string names.

    Returns
    -------
    list of str
        Parameter names in valid sampling order.

    Raises
    ------
    ValueError
        If there is a circular dependency.
    """
    # Kahn's algorithm
    in_degree: dict[str, int] = {name: 0 for name in named_priors}
    dependents: dict[str, list[str]] = {name: [] for name in named_priors}

    for name, prior in named_priors.items():
        for dep_obj in prior.dependencies():
            dep_name = param_to_name.get(dep_obj)
            if dep_name is not None and dep_name in named_priors:
                in_degree[name] += 1
                dependents[dep_name].append(name)

    queue = [name for name, deg in in_degree.items() if deg == 0]
    result: list[str] = []

    while queue:
        queue.sort()
        node = queue.pop(0)
        result.append(node)
        for dependent in dependents[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(named_priors):
        remaining = set(named_priors) - set(result)
        raise ValueError(f'Circular dependency among parameters: {remaining}')

    return result
