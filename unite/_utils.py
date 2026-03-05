"""Internal utilities shared across unite submodules."""

from __future__ import annotations

from collections.abc import Callable
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
# Wavelength validation
# ---------------------------------------------------------------------------


def _ensure_wavelength(
    value: u.Quantity, name: str = 'wavelength', *, ndim: int | None = None
) -> u.Quantity:
    """Validate that *value* is an astropy Quantity with wavelength (length) units.

    Parameters
    ----------
    value : astropy.units.Quantity
        Value to validate.
    name : str, optional
        Name used in error messages.  Defaults to ``'wavelength'``.
    ndim : int, optional
        If provided, also validates that the quantity has exactly this
        number of dimensions.

    Returns
    -------
    astropy.units.Quantity
        The validated quantity (unchanged).

    Raises
    ------
    TypeError
        If *value* is not a :class:`~astropy.units.Quantity`.
    ValueError
        If *value* does not have length (wavelength) units.
    ValueError
        If *ndim* is provided and *value* does not have that many dimensions.
    """
    if not isinstance(value, u.Quantity):
        raise TypeError(
            f'{name} must be an astropy Quantity with wavelength units, '
            f'got {type(value).__name__}.'
        )
    if not value.unit.is_equivalent(u.m):  # type: ignore[union-attr]
        raise ValueError(
            f'{name} must have wavelength (length) units, got {value.unit!r}.'
        )
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f'{name} must be {ndim}-D, got {value.ndim}-D array.')
    return value


# ---------------------------------------------------------------------------
# Unit conversion helpers
# ---------------------------------------------------------------------------


def _wavelength_conversion_factor(from_unit: u.UnitBase, to_unit: u.UnitBase) -> float:
    """Return the scalar multiplier to convert wavelengths from *from_unit* to *to_unit*.

    Parameters
    ----------
    from_unit : astropy.units.UnitBase
        Source wavelength unit.
    to_unit : astropy.units.UnitBase
        Target wavelength unit.

    Returns
    -------
    float
        Multiplicative factor such that ``value_in_to = value_in_from * factor``.
    """
    return float((1.0 * from_unit).to(to_unit).value)


def _ensure_velocity(value: u.Quantity, name: str = 'velocity') -> u.Quantity:
    """Validate that *value* is a Quantity with velocity units.

    Parameters
    ----------
    value : astropy.units.Quantity
        Value to validate.
    name : str, optional
        Name used in error messages.  Defaults to ``'velocity'``.

    Returns
    -------
    astropy.units.Quantity
        The validated quantity (unchanged).

    Raises
    ------
    TypeError
        If *value* is not a :class:`~astropy.units.Quantity`.
    ValueError
        If *value* does not have velocity units.
    """
    if not isinstance(value, u.Quantity):
        raise TypeError(
            f'{name} must be an astropy Quantity with velocity units '
            f'(e.g. km/s), got {type(value).__name__}.'
        )
    if not value.unit.is_equivalent(u.km / u.s):
        raise ValueError(
            f'{name} must have velocity units (e.g. km/s), got {value.unit!r}.'
        )
    return value


def _ensure_flux_density(unit: u.UnitBase) -> None:
    """Validate that *unit* is a spectral flux density per wavelength.

    Accepted units must be dimensionally equivalent to
    ``erg / s / cm^2 / Angstrom`` (i.e. energy per time per area per
    wavelength).

    Parameters
    ----------
    unit : astropy.units.UnitBase
        Unit to validate.

    Raises
    ------
    ValueError
        If the unit is not a valid f_lambda flux density.
    """
    ref = u.erg / u.s / u.cm**2 / u.AA
    if not unit.is_equivalent(ref):
        raise ValueError(
            f'flux_unit must be a spectral flux density per wavelength '
            f'(e.g. erg/s/cm^2/Angstrom), got {unit!r}.'
        )


def _ensure_flux_density_quantity(
    value: u.Quantity, name: str = 'flux', *, ndim: int | None = None
) -> u.Quantity:
    """Validate that *value* is a Quantity with f_lambda flux density units.

    Parameters
    ----------
    value : astropy.units.Quantity
        Value to validate.
    name : str, optional
        Name used in error messages.  Defaults to ``'flux'``.
    ndim : int, optional
        If provided, also validates that the quantity has exactly this
        number of dimensions.

    Returns
    -------
    astropy.units.Quantity
        The validated quantity (unchanged).

    Raises
    ------
    TypeError
        If *value* is not a :class:`~astropy.units.Quantity`.
    ValueError
        If *value* does not have f_lambda flux density units, or if
        *ndim* is provided and the array has the wrong dimensionality.
    """
    if not isinstance(value, u.Quantity):
        raise TypeError(
            f'{name} must be an astropy Quantity with flux density units '
            f'(e.g. erg/s/cm^2/Angstrom), got {type(value).__name__}.'
        )
    _ensure_flux_density(value.unit)
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f'{name} must be {ndim}-D, got {value.ndim}-D array.')
    return value


def _flux_density_conversion_factor(
    from_unit: u.UnitBase, to_unit: u.UnitBase
) -> float:
    """Return the scalar multiplier to convert f_lambda from *from_unit* to *to_unit*.

    Both units must be spectral flux density per wavelength (f_lambda).

    Parameters
    ----------
    from_unit : astropy.units.UnitBase
        Source f_lambda unit.
    to_unit : astropy.units.UnitBase
        Target f_lambda unit.

    Returns
    -------
    float
        Multiplicative factor such that ``value_in_to = value_in_from * factor``.
    """
    return float((1.0 * from_unit).to(to_unit).value)
