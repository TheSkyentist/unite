"""JWST NIRSpec disperser implementations.

This module provides :class:`NIRSpecDisperser`, a concrete
:class:`~unite.instrument.base.Disperser` for every NIRSpec grating/prism
configuration.  Two resolving-power calibrations are available:

* ``"uniform"`` or ``"uniformly-illuminated"`` — the official JDOX tabulated *R(λ)* curves shipped with the
  package as FITS files. This calibration is for uniformly illuminated sources.
* ``"point"`` or ``"point-source"`` — polynomial *R(λ)* fits from de Graaff et al. (2025). This
  calibration is tailored for point-source observations.

Both calibrations share the same JDOX linear-dispersion (*dλ/dpix*) tables.

In addition to the generic :class:`NIRSpecDisperser` class, convenience
subclasses are provided for each grating:

* :class:`Prism`
* :class:`G140M`, :class:`G140H`
* :class:`G235M`, :class:`G235H`
* :class:`G395M`, :class:`G395H`

These fix the grating name while still accepting ``r_source`` and calibration
token kwargs.
"""

from __future__ import annotations

from importlib import resources
from typing import Literal, cast

import jax.numpy as jnp
from astropy import units as u
from astropy.io import fits

from unite.instrument.base import Disperser, FluxScale, PixOffset, RScale

# ---------------------------------------------------------------------------
# Valid grating names
# ---------------------------------------------------------------------------

_NIRSPEC_GRATINGS: tuple[str, ...] = (
    'prism',
    'g140m',
    'g140h',
    'g235m',
    'g235h',
    'g395m',
    'g395h',
)
"""All supported NIRSpec grating / prism names (lower-case)."""

# ---------------------------------------------------------------------------
# de Graaff+25 polynomial coefficients for R(λ)
#
# Coefficients are in *descending* order of power (i.e. the convention used
# by ``jnp.polyval``): ``[c_n, c_{n-1}, ..., c_1, c_0]`` where
# ``R(λ) = c_n λ^n + ... + c_1 λ + c_0`` and λ is in **microns**.
# ---------------------------------------------------------------------------

# Derived from msafit: https://doi.org/10.1051/0004-6361/202347755
_DEGRAAFF25_R_COEFFS: dict[str, tuple[float, ...]] = {
    'prism': (
        0.6588751520824567,
        -13.160715906787065,
        105.20050050555237,
        -429.52868537465565,
        959.0507565400321,
        -1043.4918213547285,
        480.90575759267125,
    ),
    'g140m': (72.97401411114878, 1238.119318020148, 122.54619776393024),
    'g140h': (497.93050338537523, 2477.457080098987, 691.0215379986365),
    'g235m': (-103.96581777710607, 1216.7117974835414, -496.75905571502886),
    'g235h': (-290.05755034631613, 3480.1919687566588, -1422.7141098998798),
    'g395m': (-33.7506093199521, 571.9739062868341, -28.885468437995822),
    'g395h': (-113.72459145364421, 1717.7667125191442, -322.82363185174756),
}

# Type alias for the two supported R calibrations.
RSource = Literal['uniform', 'point']

# ---------------------------------------------------------------------------
# Mapping from lower-case grating name to class name
# ---------------------------------------------------------------------------

_GRATING_CLASS_NAMES: dict[str, str] = {
    'prism': 'PRISM',
    'g140m': 'G140M',
    'g140h': 'G140H',
    'g235m': 'G235M',
    'g235h': 'G235H',
    'g395m': 'G395M',
    'g395h': 'G395H',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_uniform_disp(grating: str) -> dict[str, jnp.ndarray]:
    """Load the JDOX dispersion FITS file for *grating*.

    Parameters
    ----------
    grating : str
        Lower-case grating name (e.g. ``"g140h"``).

    Returns
    -------
    dict
        Keys ``"wavelength"`` (µm), ``"dlds"`` (µm/pixel), ``"R"``
        (dimensionless), each as a 1-D ``jnp.ndarray``.
    """
    filename = f'jwst_nirspec_{grating}_disp.fits'
    ref = resources.files('unite.instrument.nirspec.data').joinpath(filename)
    with resources.as_file(ref) as path, fits.open(path) as hdul:
        data = hdul[1].data
        return {
            'wavelength': jnp.asarray(data['WAVELENGTH'], dtype=float),
            'dlds': jnp.asarray(data['DLDS'], dtype=float),
            'R': jnp.asarray(data['R'], dtype=float),
        }


# ---------------------------------------------------------------------------
# NIRSpecDisperser
# ---------------------------------------------------------------------------


class NIRSpecDisperser(Disperser):
    """Disperser for a JWST NIRSpec grating or prism configuration.

    Parameters
    ----------
    grating : str
        NIRSpec grating name.  One of ``"prism"``, ``"g140m"``,
        ``"g140h"``, ``"g235m"``, ``"g235h"``, ``"g395m"``, ``"g395h"``
        (case-insensitive).
    r_source : ``"uniform"``, ``"uniformly-illuminated"``, ``"point"``, or ``"point-source"``, optional
        Which resolving-power calibration to use.  ``"uniform"`` or
        ``"uniformly-illuminated"`` interpolates the tabulated JDOX *R(λ)* curve for uniformly illuminated
        sources; ``"point"`` or ``"point-source"`` (default) evaluates the polynomial fit from de
        Graaff et al. (2025), which is tailored for point-source observations.
    name : str, optional
        Human-readable label.  Defaults to the grating name (upper-case).
    r_scale : RScale, optional
        Token for the resolving-power scale.
    flux_scale : FluxScale, optional
        Token for the flux normalisation.
    pix_offset : PixOffset, optional
        Token for the pixel shift.

    Notes
    -----
    The linear dispersion *dλ/dpix* is always taken from the JDOX dispersion
    tables regardless of the ``r_source`` choice.

    All wavelengths are in **microns** (``astropy.units.um``).

    Examples
    --------
    >>> d = NIRSpecDisperser("g235h")
    >>> d.grating
    'g235h'
    >>> d.r_source
    'point'

    >>> d_uniform = NIRSpecDisperser("prism", r_source="uniform")
    >>> d_uniform.r_source
    'uniform'
    """

    def __init__(
        self,
        grating: str,
        r_source: RSource = 'point',
        *,
        name: str = '',
        r_scale: RScale | None = None,
        flux_scale: FluxScale | None = None,
        pix_offset: PixOffset | None = None,
    ) -> None:
        grating = grating.lower()
        if grating not in _NIRSPEC_GRATINGS:
            msg = f'Unknown NIRSpec grating {grating!r}. Must be one of {_NIRSPEC_GRATINGS}.'
            raise ValueError(msg)

        # Normalize r_source to 'uniform' or 'point'
        _r_source_str: str = r_source.lower()
        if _r_source_str in ('uniformly-illuminated', 'uniformly_illuminated'):
            _r_source_str = 'uniform'
        elif _r_source_str in ('point-source', 'point_source'):
            _r_source_str = 'point'

        if _r_source_str not in ('uniform', 'point'):
            msg = f"Unknown r_source {_r_source_str!r}. Must be 'uniform'(ly-illuminated), or 'point'(-source)."
            raise ValueError(msg)

        super().__init__(
            unit=u.um,
            name=name or grating.upper(),
            r_scale=r_scale,
            flux_scale=flux_scale,
            pix_offset=pix_offset,
        )

        self.grating: str = grating
        self.r_source: RSource = cast('RSource', _r_source_str)

        # Load JDOX dispersion table (shared by both R sources).
        uniform = _load_uniform_disp(grating)
        self._wavelength_grid = uniform['wavelength']
        self._dlds_grid = uniform['dlds']

        # Build the R source.
        if _r_source_str == 'uniform':
            self._R_grid: jnp.ndarray | None = uniform['R']
            self._R_poly: jnp.ndarray | None = None
            _r_grid = uniform['R']
            self.R = lambda wavelength: jnp.interp(  # ty: ignore[invalid-assignment]
                wavelength, self._wavelength_grid, _r_grid
            )
        else:
            self._R_grid = None
            self._R_poly = jnp.asarray(_DEGRAAFF25_R_COEFFS[grating], dtype=float)
            _r_poly = self._R_poly
            self.R = lambda wavelength: jnp.polyval(_r_poly, wavelength)  # ty: ignore[invalid-assignment]

    # -- Disperser interface -------------------------------------------------

    def R(self, wavelength):  # ty: ignore[invalid-method-override]
        """Return the resolving power at the given wavelengths (µm)."""
        raise NotImplementedError('R method should be set dynamically in __init__.')

    def dlam_dpix(self, wavelength):
        """Return *dλ/dpix* in µm/pixel, interpolated from the JDOX table."""
        return jnp.interp(wavelength, self._wavelength_grid, self._dlds_grid)

    # -- Convenience ---------------------------------------------------------

    def __repr__(self) -> str:
        """Return a readable string representation of this disperser."""
        cls = type(self).__name__
        return f'{cls}(grating={self.grating!r}, r_source={self.r_source!r})'


# ---------------------------------------------------------------------------
# Per-grating convenience subclasses, generated in a loop
# ---------------------------------------------------------------------------


def _make_grating_class(grating: str, class_name: str) -> type[NIRSpecDisperser]:
    """Create a ``NIRSpecDisperser`` subclass with a fixed grating.

    Parameters
    ----------
    grating : str
        Lower-case grating name.
    class_name : str
        PascalCase name for the generated class.

    Returns
    -------
    type[NIRSpecDisperser]
        A new subclass that fixes the *grating* argument.
    """

    class _GratingDisperser(NIRSpecDisperser):
        __doc__ = f"""NIRSpec **{grating.upper()}** disperser.

    Convenience subclass of :class:`NIRSpecDisperser` with the grating
    fixed to ``"{grating}"``.  Only ``r_source`` and calibration tokens
    need to be specified.

    Parameters
    ----------
    r_source : ``"uniform"``, ``"uniformly-illuminated"``, ``"point"``, or ``"point-source"``, optional
        Which resolving-power calibration to use (default ``"point"``).
    name : str, optional
        Human-readable label.  Defaults to ``"{class_name}"``.
    r_scale : RScale, optional
        Token for the resolving-power scale.
    flux_scale : FluxScale, optional
        Token for the flux normalisation.
    pix_offset : PixOffset, optional
        Token for the pixel shift.

    Examples
    --------
    >>> d = {class_name}()
    >>> d.grating
    '{grating}'
    """

        def __init__(
            self,
            r_source: RSource = 'point',
            *,
            name: str = '',
            r_scale: RScale | None = None,
            flux_scale: FluxScale | None = None,
            pix_offset: PixOffset | None = None,
        ) -> None:
            super().__init__(
                grating=grating,
                r_source=r_source,
                name=name or class_name,
                r_scale=r_scale,
                flux_scale=flux_scale,
                pix_offset=pix_offset,
            )

        def __repr__(self) -> str:
            """Return a readable string representation of this disperser."""
            return f'{class_name}(r_source={self.r_source!r})'

    _GratingDisperser.__name__ = class_name
    _GratingDisperser.__qualname__ = class_name

    return _GratingDisperser


# Build and inject per-grating classes into the module namespace.
for _grating, _class_name in _GRATING_CLASS_NAMES.items():
    globals()[_class_name] = _make_grating_class(_grating, _class_name)

# Explicit type-visible references to dynamically generated classes.
# These are set by the loop above; re-declared here for type checkers.
G140H: type[NIRSpecDisperser] = globals()['G140H']
G140M: type[NIRSpecDisperser] = globals()['G140M']
G235H: type[NIRSpecDisperser] = globals()['G235H']
G235M: type[NIRSpecDisperser] = globals()['G235M']
G395H: type[NIRSpecDisperser] = globals()['G395H']
G395M: type[NIRSpecDisperser] = globals()['G395M']
PRISM: type[NIRSpecDisperser] = globals()['PRISM']

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    'G140H',
    'G140M',
    'G235H',
    'G235M',
    'G395H',
    'G395M',
    'PRISM',
    'NIRSpecDisperser',
]
