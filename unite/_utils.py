"""Internal utilities shared across unite submodules."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

from astropy import units as u
from astropy.constants import c

_T = TypeVar('_T')

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

C_KMS: float = c.to('km/s').value
"""Speed of light in km/s."""

# ---------------------------------------------------------------------------
# Registry decorator factory
# ---------------------------------------------------------------------------


def _make_register(registry: dict[str, type]) -> Callable[[type[_T]], type[_T]]:
    """Return a class decorator that registers classes by name.

    Parameters
    ----------
    registry : dict
        Mapping from class name to class.  Modified in place when the
        returned decorator is applied.

    Returns
    -------
    callable
        A decorator that inserts the decorated class into *registry* under
        ``cls.__name__`` and returns the class unchanged.

    Examples
    --------
    >>> _MY_REGISTRY: dict[str, type] = {}
    >>> _register = _make_register(_MY_REGISTRY)
    >>> @_register
    ... class Foo: pass
    >>> _MY_REGISTRY
    {'Foo': <class 'Foo'>}
    """

    def _register(cls: type[_T]) -> type[_T]:
        registry[cls.__name__] = cls
        return cls

    return _register


# ---------------------------------------------------------------------------
# Alphabet-based auto-naming
# ---------------------------------------------------------------------------

_LETTERS = 'abcdefghijklmnopqrstuvwxyz'


def _alpha_name(n: int) -> str:
    """Return Excel-style column name for index *n* (0 → 'a', 25 → 'z', 26 → 'aa', …).

    Parameters
    ----------
    n : int
        Non-negative integer index.

    Returns
    -------
    str
        Excel-style column name.
    """
    result = ''
    n += 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        result = _LETTERS[r] + result
    return result


# ---------------------------------------------------------------------------
# Broadcasting utilities
# ---------------------------------------------------------------------------


def _broadcast(val, arg_name: str, n: int) -> list:
    """Broadcast a scalar or sequence to a list of length *n*.

    Parameters
    ----------
    val : Any
        Either a scalar value or a sequence.
    arg_name : str
        Name of the argument (for error messages).
    n : int
        Expected length.

    Returns
    -------
    list
        A list of length *n*.

    Raises
    ------
    ValueError
        If *val* is a sequence with length neither 1 nor *n*.
    """
    if isinstance(val, list | tuple):
        if len(val) == 1:
            return list(val) * n
        if len(val) != n:
            raise ValueError(f"'{arg_name}' has {len(val)} entries; expected 1 or {n}.")
        return list(val)
    return [val] * n


# ---------------------------------------------------------------------------
# Core Validation Utility
# ---------------------------------------------------------------------------


def _ensure_quantity(
    value: u.Quantity, ref_unit: u.UnitBase, name: str, ndim: int | Iterable[int]
) -> u.Quantity:
    # 1. Type validation and casting
    if not isinstance(value, u.Quantity):
        if hasattr(value, 'unit'):
            value = u.Quantity(value)
        else:
            raise TypeError(
                f'{name} must be an astropy Quantity with {ref_unit.physical_type} units, '
                f'got {type(value).__name__}.'
            )

    # 2. Unit equivalence validation
    if not value.unit.is_equivalent(ref_unit):
        raise ValueError(
            f'{name} must have units equivalent to {ref_unit}, got {value.unit}.'
        )

    # 3. Flexible ndim validation (supports int or list/tuple)
    allowed_ndims = [ndim] if isinstance(ndim, int) else ndim
    if value.ndim not in allowed_ndims:
        allowed_str = ' or '.join(map(str, allowed_ndims))
        raise ValueError(f'{name} must be {allowed_str}-D, got {value.ndim}-D array.')

    return value


# ---------------------------------------------------------------------------
# Specific Physical Type Wrappers
# ---------------------------------------------------------------------------


def _ensure_wavelength(value, name, ndim):
    return _ensure_quantity(value, u.m, name, ndim)


def _ensure_velocity(value, name, ndim):
    return _ensure_quantity(value, u.km / u.s, name, ndim)


def _ensure_flux_density(value, name, ndim):
    ref = u.erg / (u.s * u.cm**2 * u.AA)
    return _ensure_quantity(value, ref, name, ndim)


# ---------------------------------------------------------------------------
# Unit Conversion
# ---------------------------------------------------------------------------


def _get_conversion_factor(from_unit: u.UnitBase, to_unit: u.UnitBase) -> float:
    """Return generic scalar multiplier to convert between equivalent units."""
    return float((1.0 * from_unit).to(to_unit).value)
