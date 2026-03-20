"""Declarative, serializable prior distributions for model parameters.

Priors are inert data containers that describe numpyro distributions.
At model-build time, a topological sort of their dependencies determines
sampling order, and each prior's :meth:`to_dist` method receives a context
dict of already-sampled values to resolve parameter expressions.

Examples
--------
Simple fixed-bound prior:

>>> from unite.prior import Uniform
>>> p = Uniform(0, 750)
>>> p.to_dist({})
Uniform(low=0, high=750)

Dependent prior using arithmetic on parameter tokens:

>>> from unite.line.config import FWHM
>>> from unite.prior import Uniform
>>> narrow = FWHM('narrow', prior=Uniform(0, 750))
>>> broad = FWHM('broad', prior=Uniform(low=narrow * 2 + 150, high=2500))

Ratio constraint tying a flux diagnostic across two kinematic components:

>>> flux_5007_narrow = Flux('5007_narrow')
>>> flux_5007_broad = Flux('5007_broad')
>>> flux_4363_narrow = Flux('4363_narrow')
>>> flux_4363_broad = Flux('4363_broad',
...     prior=Fixed(flux_4363_narrow * flux_5007_broad / flux_5007_narrow))
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpyro.distributions as dist

# -------------------------------------------------------------------
# Private expression tree for parameter arithmetic
# -------------------------------------------------------------------


class _Expr(ABC):
    """Private abstract base for parameter arithmetic expression trees.

    Instances are created by arithmetic on :class:`Parameter` tokens or
    other ``_Expr`` instances.  Never construct subclasses directly —
    use arithmetic operators on :class:`Parameter` tokens instead.

    All four binary operators (``+``, ``-``, ``*``, ``/``) are supported
    between any combination of :class:`Parameter` tokens, ``_Expr`` nodes,
    and plain scalars.  The result is always an ``_Expr`` node that can be
    passed as a prior bound or the value of a :class:`Fixed` prior.
    """

    @abstractmethod
    def resolve(self, context: dict) -> float:
        """Evaluate the expression against sampled parameter values.

        Parameters
        ----------
        context : dict
            Mapping of parameter token objects to their sampled values.

        Returns
        -------
        float
        """

    @abstractmethod
    def dependencies(self) -> set:
        """Return the set of Parameter tokens this expression depends on.

        Returns
        -------
        set
        """

    @abstractmethod
    def to_dict(self, param_namer: dict) -> float | dict:
        """Serialize to a YAML-safe value.

        Parameters
        ----------
        param_namer : dict
            Mapping from parameter token objects to string names.
        """

    # ------------------------------------------------------------------
    # Arithmetic to build expression trees
    # ------------------------------------------------------------------

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return _BinOpExpr('*', self, _LiteralLeaf(float(other)))
        if isinstance(other, _Expr):
            return _BinOpExpr('*', self, other)
        if isinstance(other, Parameter):
            return _BinOpExpr('*', self, _ParamLeaf(other))
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return _BinOpExpr('*', _LiteralLeaf(float(other)), self)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return _BinOpExpr('/', self, _LiteralLeaf(float(other)))
        if isinstance(other, _Expr):
            return _BinOpExpr('/', self, other)
        if isinstance(other, Parameter):
            return _BinOpExpr('/', self, _ParamLeaf(other))
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return _BinOpExpr('/', _LiteralLeaf(float(other)), self)
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return _BinOpExpr('+', self, _LiteralLeaf(float(other)))
        if isinstance(other, _Expr):
            return _BinOpExpr('+', self, other)
        if isinstance(other, Parameter):
            return _BinOpExpr('+', self, _ParamLeaf(other))
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return _BinOpExpr('+', _LiteralLeaf(float(other)), self)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return _BinOpExpr('-', self, _LiteralLeaf(float(other)))
        if isinstance(other, _Expr):
            return _BinOpExpr('-', self, other)
        if isinstance(other, Parameter):
            return _BinOpExpr('-', self, _ParamLeaf(other))
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return _BinOpExpr('-', _LiteralLeaf(float(other)), self)
        return NotImplemented


class _ParamLeaf(_Expr):
    """Expression leaf referencing a single Parameter token."""

    def __init__(self, param) -> None:
        self.param = param

    def resolve(self, context: dict) -> float:
        return context[self.param]

    def dependencies(self) -> set:
        return {self.param}

    def to_dict(self, param_namer: dict) -> dict:
        return {'ref': param_namer[self.param]}

    def __repr__(self) -> str:
        label = (
            getattr(self.param, 'label', None)
            or getattr(self.param, 'name', None)
            or type(self.param).__name__
        )
        return label


class _LiteralLeaf(_Expr):
    """Expression leaf holding a fixed float value."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def resolve(self, context: dict) -> float:
        return self.value

    def dependencies(self) -> set:
        return set()

    def to_dict(self, param_namer: dict) -> float:
        return self.value

    def __repr__(self) -> str:
        return repr(self.value)


class _BinOpExpr(_Expr):
    """Binary operation node: left op right."""

    _VALID_OPS = frozenset({'+', '-', '*', '/'})

    def __init__(self, op: str, left, right) -> None:
        if op not in self._VALID_OPS:
            msg = f'Unknown operator {op!r}. Must be one of {sorted(self._VALID_OPS)}.'
            raise ValueError(msg)
        self.op = op
        self.left = left if isinstance(left, _Expr) else _LiteralLeaf(float(left))
        self.right = right if isinstance(right, _Expr) else _LiteralLeaf(float(right))

    def resolve(self, context: dict) -> float:
        l = self.left.resolve(context)  # noqa: E741
        r = self.right.resolve(context)
        match self.op:
            case '+':
                return l + r
            case '-':
                return l - r
            case '*':
                return l * r
            case '/':
                return l / r
            case _:
                raise ValueError(f'Unsupported operator: {self.op}')

    def dependencies(self) -> set:
        return self.left.dependencies() | self.right.dependencies()

    def to_dict(self, param_namer: dict) -> dict:
        return {
            'op': self.op,
            'left': self.left.to_dict(param_namer),
            'right': self.right.to_dict(param_namer),
        }

    def __repr__(self) -> str:
        return f'({self.left!r} {self.op} {self.right!r})'


# -------------------------------------------------------------------
# Bound type alias
# -------------------------------------------------------------------

Bound = float | _Expr
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
            Required for resolving parameter expressions in bounds.

        Returns
        -------
        numpyro.distributions.Distribution or None
            ``None`` indicates the parameter is fixed and must not be
            sampled.  The caller is responsible for injecting the fixed
            value (from :meth:`~Fixed.resolved_value`) into the model context.
        """

    @abstractmethod
    def dependencies(self) -> set:
        """Return parameter token objects this prior depends on.

        Returns
        -------
        set
            Set of Parameter objects.  Empty for independent priors.
        """

    @abstractmethod
    def to_dict(self, param_namer: dict | None = None) -> dict:
        """Serialize to a YAML-safe dictionary.

        Parameters
        ----------
        param_namer : dict, optional
            Mapping from parameter token objects to string names,
            used to serialize parameter expressions in bounds.
        """

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> Prior:
        """Deserialize from a dictionary."""


# -------------------------------------------------------------------
# Bound helpers
# -------------------------------------------------------------------


def _resolve_bound(value: Bound, context: dict) -> float:
    """Resolve a bound that may be a fixed float or a parameter expression."""
    if isinstance(value, _Expr):
        return value.resolve(context)
    return value


def _bound_deps(value: Bound) -> set:
    """Extract parameter token dependencies from a bound."""
    if isinstance(value, _Expr):
        return value.dependencies()
    return set()


def _serialize_bound(value: Bound, param_namer: dict | None) -> float | dict:
    """Serialize a bound to a YAML-safe value."""
    if isinstance(value, _Expr):
        if param_namer is None:
            msg = 'Cannot serialize parameter expression without param_namer'
            raise ValueError(msg)
        return value.to_dict(param_namer)
    return value


def _deserialize_bound(value: float | dict, token_registry: dict | None) -> Bound:
    """Deserialize a bound from a YAML-safe value.

    Parameters
    ----------
    value : float or dict
        The serialized bound.
    token_registry : dict, optional
        Mapping from string names to parameter token objects,
        used to resolve serialized parameter expressions.
    """
    if isinstance(value, dict):
        if token_registry is None:
            msg = 'Cannot deserialize parameter expression without token_registry'
            raise ValueError(msg)
        if 'ref' in value:
            return _ParamLeaf(token_registry[value['ref']])
        if 'op' in value:
            left = _deserialize_bound(value['left'], token_registry)
            right = _deserialize_bound(value['right'], token_registry)
            return _BinOpExpr(value['op'], left, right)
    return float(value)


def _normalize_bound(value: Bound | Parameter) -> Bound:
    """Convert a bound value to a Bound type (float or _Expr).

    If value is a Parameter, wrap it in a _ParamLeaf expression.
    """
    if isinstance(value, _Expr):
        return value
    elif isinstance(value, Parameter):
        return _ParamLeaf(value)
    else:
        return float(value)


# -------------------------------------------------------------------
# Concrete priors
# -------------------------------------------------------------------


class Uniform(Prior):
    """Uniform prior with bounds that may reference other parameters.

    Parameters
    ----------
    low : float, or arithmetic expression on Parameter tokens
        Lower bound.
    high : float, or arithmetic expression on Parameter tokens
        Upper bound.

    Examples
    --------
    Fixed bounds:

    >>> Uniform(0, 750)

    Dependent bound (broad fwhm > narrow fwhm + 150 km/s):

    >>> Uniform(low=narrow_fwhm * 2 + 150, high=2500)

    Direct parameter reference:

    >>> Uniform(low=base_redshift, high=base_redshift + 0.1)
    """

    def __init__(self, low: Bound = 0, high: Bound = 1) -> None:
        self.low = _normalize_bound(low)
        self.high = _normalize_bound(high)

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
    loc : float, or arithmetic expression on Parameter tokens
        Mean of the underlying normal distribution.
    scale : float
        Standard deviation of the underlying normal distribution.
    low : float, or arithmetic expression on Parameter tokens
        Lower truncation bound.
    high : float, or arithmetic expression on Parameter tokens
        Upper truncation bound.
    """

    def __init__(self, loc: Bound, scale: float, low: Bound, high: Bound) -> None:
        self.loc = _normalize_bound(loc)
        self.scale = float(scale)
        self.low = _normalize_bound(low)
        self.high = _normalize_bound(high)

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
        return f'TruncatedNormal(loc={self.loc!r}, scale={self.scale}, low={self.low!r}, high={self.high!r})'


class Fixed(Prior):
    """A fixed (non-sampled) constant value or deterministic expression.

    ``Fixed`` parameters are injected directly into the model context as
    constants rather than being drawn from a distribution.  This avoids
    Delta distributions, which are not differentiable and would break
    gradient-based samplers.

    The value may be a literal number, a :class:`Parameter` token
    (automatically converted to an expression), or any arithmetic expression
    built from :class:`Parameter` tokens (e.g. ``flux_a * flux_b / flux_c``).
    Expressions are evaluated at model-build time after their dependencies
    have been sampled, enabling deterministic relationships between parameters.

    Parameters
    ----------
    value : float, int, or arithmetic expression on Parameter tokens
        The constant value or deterministic expression.

    Examples
    --------
    Literal value:

    >>> Fixed(6564.61)
    Fixed(6564.61)

    Tie a redshift to another parameter:

    >>> Fixed(narrow_z)

    Tie the [OIII] 4363 flux ratio across narrow and broad components
    (same electron temperature in both):

    >>> Fixed(flux_4363_narrow * flux_5007_broad / flux_5007_narrow)
    """

    def __init__(self, value: Bound | Parameter) -> None:
        if not isinstance(value, (int, float, _Expr, Parameter)):
            msg = (
                f'Fixed value must be int, float, or a parameter expression, '
                f'got {type(value).__name__}'
            )
            raise TypeError(msg)
        self.value = _normalize_bound(value)

    def to_dist(self, context: dict) -> None:
        """Return ``None`` — the parameter is constant and must not be sampled."""
        return None

    def resolved_value(self, context: dict) -> float:
        """Evaluate the fixed value against a context of sampled parameters.

        For literal values, returns the value directly.  For expression values
        (including single-parameter references), evaluates the expression tree
        against already-sampled parameter values.

        Parameters
        ----------
        context : dict
            Mapping of parameter token objects to their sampled values.

        Returns
        -------
        float
        """
        if isinstance(self.value, _Expr):
            return self.value.resolve(context)
        return self.value

    def dependencies(self) -> set:
        return _bound_deps(self.value)

    def to_dict(self, param_namer: dict | None = None) -> dict:
        return {'type': 'Fixed', 'value': _serialize_bound(self.value, param_namer)}

    @classmethod
    def from_dict(cls, d: dict, token_registry: dict | None = None) -> Fixed:
        return cls(_deserialize_bound(d['value'], token_registry))

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

    Arithmetic on a ``Parameter`` produces an expression tree that can be
    used as a prior bound, enabling dependent priors such as
    ``broad_fwhm > narrow_fwhm + 150 km/s`` or ratio constraints such as
    ``flux_4363_broad = flux_4363_narrow * flux_5007_broad / flux_5007_narrow``.

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
        instance of the same subclass as *self*.  Cross-kind expressions
        (e.g. an ``FWHM`` expression used as a ``Redshift`` prior bound)
        are forbidden.
    """

    def __init__(self, name: str | None = None, *, prior: Prior) -> None:
        for dep in prior.dependencies():
            if not isinstance(dep, type(self)):
                msg = (
                    f'{type(self).__name__} prior references a '
                    f'{type(dep).__name__} parameter. Parameter expressions must '
                    f'reference the same kind of parameter.'
                )
                raise TypeError(msg)
        self.label: str | None = name  # user-supplied semantic label
        self.name: str | None = None  # NumPyro site name (set at registration)
        self.prior = prior

    def __repr__(self) -> str:
        """Return a readable string representation."""
        parts = []
        label = self.label or self.name
        if label is not None:
            parts.append(repr(label))
        parts.append(f'prior={self.prior!r}')
        return f'{self.__class__.__name__}({", ".join(parts)})'

    def __mul__(self, other) -> _Expr:
        """Return a product expression."""
        if isinstance(other, (int, float)):
            return _BinOpExpr('*', _ParamLeaf(self), _LiteralLeaf(float(other)))
        if isinstance(other, _Expr):
            return _BinOpExpr('*', _ParamLeaf(self), other)
        if isinstance(other, Parameter):
            return _BinOpExpr('*', _ParamLeaf(self), _ParamLeaf(other))
        return NotImplemented

    def __rmul__(self, other) -> _Expr:
        """Return a product expression."""
        if isinstance(other, (int, float)):
            return _BinOpExpr('*', _LiteralLeaf(float(other)), _ParamLeaf(self))
        return NotImplemented

    def __truediv__(self, other) -> _Expr:
        """Return a division expression."""
        if isinstance(other, (int, float)):
            return _BinOpExpr('/', _ParamLeaf(self), _LiteralLeaf(float(other)))
        if isinstance(other, _Expr):
            return _BinOpExpr('/', _ParamLeaf(self), other)
        if isinstance(other, Parameter):
            return _BinOpExpr('/', _ParamLeaf(self), _ParamLeaf(other))
        return NotImplemented

    def __rtruediv__(self, other) -> _Expr:
        """Return a division expression."""
        if isinstance(other, (int, float)):
            return _BinOpExpr('/', _LiteralLeaf(float(other)), _ParamLeaf(self))
        return NotImplemented

    def __add__(self, other) -> _Expr:
        """Return an additive expression."""
        if isinstance(other, (int, float)):
            return _BinOpExpr('+', _ParamLeaf(self), _LiteralLeaf(float(other)))
        if isinstance(other, _Expr):
            return _BinOpExpr('+', _ParamLeaf(self), other)
        if isinstance(other, Parameter):
            return _BinOpExpr('+', _ParamLeaf(self), _ParamLeaf(other))
        return NotImplemented

    def __radd__(self, other) -> _Expr:
        """Return an additive expression."""
        if isinstance(other, (int, float)):
            return _BinOpExpr('+', _LiteralLeaf(float(other)), _ParamLeaf(self))
        return NotImplemented

    def __sub__(self, other) -> _Expr:
        """Return a subtractive expression."""
        if isinstance(other, (int, float)):
            return _BinOpExpr('-', _ParamLeaf(self), _LiteralLeaf(float(other)))
        if isinstance(other, _Expr):
            return _BinOpExpr('-', _ParamLeaf(self), other)
        if isinstance(other, Parameter):
            return _BinOpExpr('-', _ParamLeaf(self), _ParamLeaf(other))
        return NotImplemented

    def __rsub__(self, other) -> _Expr:
        """Return a subtractive expression."""
        if isinstance(other, (int, float)):
            return _BinOpExpr('-', _LiteralLeaf(float(other)), _ParamLeaf(self))
        return NotImplemented


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
