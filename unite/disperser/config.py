"""Dispersers configuration: instrument models with calibration tokens.

:class:`DispersersConfiguration` collects one :class:`~unite.disperser.base.Disperser`
per observing configuration.  Each disperser carries optional
:class:`~unite.disperser.base.RScale` / :class:`~unite.disperser.base.FluxScale` /
:class:`~unite.disperser.base.PixOffset` tokens (``r_scale``, ``flux_scale``,
``pix_offset``) for shared calibration parameters.

Sharing tokens
--------------
Pass the **same** token instance to multiple dispersers to share a single
parameter in the fitted model::

    from unite.disperser import DispersersConfiguration, RScale
    from unite.disperser.nirspec import G235H, G395H
    from unite.prior import TruncatedNormal

    r = RScale(prior=TruncatedNormal(1.0, 0.05, 0.8, 1.2))
    cfg = DispersersConfiguration([
        G235H(r_scale=r),
        G395H(r_scale=r),   # same r — shared parameter
    ])

Degeneracy warning
------------------
A multi-disperser fit is only identified if at least one disperser has
``flux_scale=None`` (flux anchor) and at least one has ``pix_offset=None``
(pixel-offset anchor).  :meth:`validate` issues :class:`UserWarning` when
these conditions are not met.  Validation is called automatically when this
object is passed to :class:`~unite.config.Configuration`.

Serialization
-------------
:meth:`to_dict` hoists all unique calibration tokens to a top-level
``calib_params`` section keyed by token name.  Disperser entries reference
tokens by name.  Shared tokens round-trip correctly — the same object is
reconstructed for both entries.

Examples
--------
>>> from unite.disperser import DispersersConfiguration, RScale, FluxScale
>>> from unite.disperser.nirspec import G235H, G395H
>>> from unite.prior import TruncatedNormal
>>> r = RScale(prior=TruncatedNormal(1.0, 0.05, 0.8, 1.2))
>>> flux_0 = FluxScale(prior=TruncatedNormal(1.0, 0.1, 0.5, 2.0))
>>> cfg = DispersersConfiguration([
...     G235H(r_scale=r),
...     G395H(r_scale=r, flux_scale=flux_0),
... ])
>>> cfg.names
['G235H', 'G395H']
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from pathlib import Path

import yaml
from astropy import units as u

from unite.disperser.base import Disperser, FluxScale, PixOffset, RScale
from unite.disperser.nirspec.disperser import (
    G140H,
    G140M,
    G235H,
    G235M,
    G395H,
    G395M,
    PRISM,
)
from unite.disperser.sdss.disperser import SDSSDisperser
from unite.prior import Parameter, prior_from_dict

# ---------------------------------------------------------------------------
# Disperser serialization registry
# ---------------------------------------------------------------------------

_DISPERSER_REGISTRY: dict[str, type[Disperser]] = {
    'G140H': G140H,
    'G140M': G140M,
    'G235H': G235H,
    'G235M': G235M,
    'G395H': G395H,
    'G395M': G395M,
    'PRISM': PRISM,
    'SDSSDisperser': SDSSDisperser,
}

_CALIB_REGISTRY: dict[str, type[Parameter]] = {
    'RScale': RScale,
    'FluxScale': FluxScale,
    'PixOffset': PixOffset,
}

# ---------------------------------------------------------------------------
# CalibParam (de)serialization helpers
# ---------------------------------------------------------------------------


def _calib_param_to_dict(token: Parameter) -> dict:
    """Serialize a calibration token to a YAML-safe dictionary."""
    return {'type': type(token).__name__, 'prior': token.prior.to_dict()}


def _calib_param_from_dict(name: str, d: dict) -> Parameter:
    """Reconstruct a calibration token from its serialized dict."""
    cls = _CALIB_REGISTRY[d['type']]
    prior = prior_from_dict(d['prior'])
    return cls(prior=prior, name=name)


# ---------------------------------------------------------------------------
# Disperser (de)serialization helpers
# ---------------------------------------------------------------------------


def _disperser_to_entry(disperser: Disperser) -> dict:
    """Serialize a disperser to an entry dict (CalibParams referenced by name).

    Parameters
    ----------
    disperser : Disperser
        Must be a registered type (NIRSpec or SDSS).

    Returns
    -------
    dict
    """
    cls_name = type(disperser).__name__
    if cls_name not in _DISPERSER_REGISTRY:
        msg = (
            f'Cannot serialize disperser of type {cls_name!r}. '
            f'Only registered dispersers support serialization. '
            f'Registered types: {sorted(_DISPERSER_REGISTRY)}.'
        )
        raise TypeError(msg)

    d: dict = {'type': cls_name, 'name': disperser.name}
    if cls_name == 'NIRSpecDisperser':
        d['grating'] = disperser.grating
        d['r_source'] = disperser.r_source
    elif hasattr(disperser, 'r_source'):
        d['r_source'] = disperser.r_source

    # CalibParam references by token name (or null for fixed).
    for attr in ('r_scale', 'flux_scale', 'pix_offset'):
        token = getattr(disperser, attr)
        d[attr] = token.name if token is not None else None

    return d


def _disperser_from_entry(d: dict, token_registry: dict[str, Parameter]) -> Disperser:
    """Reconstruct a disperser from an entry dict and token registry.

    Parameters
    ----------
    d : dict
        As produced by :func:`_disperser_to_entry`.
    token_registry : dict
        Mapping from token names to reconstructed CalibParam objects.

    Returns
    -------
    Disperser
    """
    cls_name = d['type']
    if cls_name not in _DISPERSER_REGISTRY:
        msg = f'Unknown disperser type {cls_name!r}. Registered: {sorted(_DISPERSER_REGISTRY)}.'
        raise KeyError(msg)
    cls = _DISPERSER_REGISTRY[cls_name]

    kwargs: dict = {'name': d.get('name', '')}
    if cls_name == 'NIRSpecDisperser':
        kwargs['grating'] = d['grating']
        kwargs['r_source'] = d.get('r_source', 'point')
    elif 'r_source' in d:
        kwargs['r_source'] = d['r_source']

    for attr in ('r_scale', 'flux_scale', 'pix_offset'):
        ref = d.get(attr)
        kwargs[attr] = token_registry[ref] if ref is not None else None

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Degeneracy warnings
# ---------------------------------------------------------------------------

_FLUX_DEGENERACY_WARNING = (
    'DispersersConfiguration: no disperser has flux_scale=None (fixed). '
    'Relative flux scales are degenerate — set flux_scale=None on one '
    'disperser to anchor the flux calibration.'
)

_PIX_DEGENERACY_WARNING = (
    'DispersersConfiguration: no disperser has pix_offset=None (fixed). '
    'Pixel-offset (dispersion) scales are degenerate — set pix_offset=None '
    'on one disperser to anchor the wavelength solution.'
)


# ---------------------------------------------------------------------------
# DispersersConfiguration
# ---------------------------------------------------------------------------


class DispersersConfiguration:
    """Configuration for a multi-disperser spectral dataset.

    An ordered collection of :class:`~unite.disperser.base.Disperser` objects,
    one per observing disperser.  Each disperser carries optional
    :class:`~unite.disperser.base.CalibParam` tokens for calibration parameters.

    The configuration is **data-free**: it describes which dispersers are used
    and how they are calibrated.  Actual spectral data arrays are attached
    later via :meth:`make_spectrum`.

    Parameters
    ----------
    dispersers : sequence of Disperser
        One entry per disperser.  Names (``disperser.name``) must be unique.

    Raises
    ------
    ValueError
        If any disperser names are duplicated or empty.
    """

    def __init__(self, dispersers: Sequence[Disperser]) -> None:
        names = [d.name for d in dispersers]
        empty = [i for i, n in enumerate(names) if not n]
        if empty:
            msg = (
                f'Dispersers at indices {empty} have empty names. '
                'Set a non-empty name on each disperser.'
            )
            raise ValueError(msg)
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            msg = f'Duplicate disperser name(s) in DispersersConfiguration: {dupes}'
            raise ValueError(msg)
        self._dispersers: list[Disperser] = list(dispersers)

    # -- validation ----------------------------------------------------------

    def validate(self) -> None:
        """Check for flux and pixel-offset degeneracies.

        Issues a :class:`UserWarning` for each calibration axis (flux,
        dispersion) where no disperser is anchored (token ``None``).
        """
        if len(self._dispersers) > 1:
            if all(d.flux_scale is not None for d in self._dispersers):
                warnings.warn(_FLUX_DEGENERACY_WARNING, UserWarning, stacklevel=2)
            if all(d.pix_offset is not None for d in self._dispersers):
                warnings.warn(_PIX_DEGENERACY_WARNING, UserWarning, stacklevel=2)

    # -- data loading --------------------------------------------------------

    def make_spectrum(
        self,
        name: str,
        low: u.Quantity,
        high: u.Quantity,
        flux: u.Quantity,
        error: u.Quantity,
    ):
        """Create a :class:`~unite.spectrum.spectrum.Spectrum` from data arrays.

        Looks up the disperser by *name* and creates a Spectrum with it
        attached.  Calibration tokens on the disperser are accessible via
        ``spectrum.disperser.r_scale`` etc.

        Parameters
        ----------
        name : str
            Disperser name as registered in this configuration.
        low : astropy.units.Quantity
            Lower pixel-edge wavelengths (1-D, wavelength dimensions).
        high : astropy.units.Quantity
            Upper pixel-edge wavelengths.  Same shape as *low*.
        flux : astropy.units.Quantity
            Flux density values per pixel (f_lambda).
        error : astropy.units.Quantity
            Flux density uncertainty per pixel (same unit as *flux*).

        Returns
        -------
        Spectrum

        Raises
        ------
        KeyError
            If *name* is not found in this configuration.
        """
        from unite.spectrum.spectrum import Spectrum

        disperser = self[name]
        return Spectrum(
            low=low, high=high, flux=flux, error=error, disperser=disperser, name=name
        )

    # -- names ---------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Names of all dispersers in this configuration."""
        return [d.name for d in self._dispersers]

    # -- container interface -------------------------------------------------

    def __len__(self) -> int:
        """Return the number of dispersers."""
        return len(self._dispersers)

    def __iter__(self) -> Iterator[Disperser]:
        """Iterate over the dispersers."""
        return iter(self._dispersers)

    def __getitem__(self, key: int | str) -> Disperser:
        """Look up a disperser by index or name."""
        if isinstance(key, str):
            for d in self._dispersers:
                if d.name == key:
                    return d
            msg = f'No disperser named {key!r} in DispersersConfiguration.'
            raise KeyError(msg)
        return self._dispersers[key]

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a YAML-safe dictionary.

        Shared :class:`~unite.disperser.base.CalibParam` tokens are hoisted to a
        top-level ``calib_params`` section.  Disperser entries reference
        tokens by name, so sharing is preserved on round-trip.

        Returns
        -------
        dict
            Contains ``'calib_params'`` and ``'entries'`` keys.
        """
        # Collect all unique tokens (preserving insertion order).
        seen_ids: dict[int, Parameter] = {}
        for d in self._dispersers:
            for attr in ('r_scale', 'flux_scale', 'pix_offset'):
                token = getattr(d, attr)
                if token is not None and id(token) not in seen_ids:
                    seen_ids[id(token)] = token

        calib_params = {t.name: _calib_param_to_dict(t) for t in seen_ids.values()}
        entries = [_disperser_to_entry(d) for d in self._dispersers]

        return {'calib_params': calib_params, 'entries': entries}

    @classmethod
    def from_dict(cls, d: dict) -> DispersersConfiguration:
        """Deserialize from a dictionary.

        Parameters
        ----------
        d : dict
            As produced by :meth:`to_dict`.

        Returns
        -------
        DispersersConfiguration
        """
        # Reconstruct token registry (name → CalibParam).
        token_registry: dict[str, Parameter] = {}
        for name, token_d in d.get('calib_params', {}).items():
            token_registry[name] = _calib_param_from_dict(name, token_d)

        dispersers = [_disperser_from_entry(e, token_registry) for e in d['entries']]
        # Bypass validate() on load — already validated when saved.
        obj = cls.__new__(cls)
        obj._dispersers = dispersers
        return obj

    # -- YAML serialization --------------------------------------------------

    def to_yaml(self) -> str:
        """Serialize to a YAML string.

        Returns
        -------
        str
        """
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> DispersersConfiguration:
        """Deserialize from a YAML string.

        Parameters
        ----------
        text : str
            YAML string as produced by :meth:`to_yaml`.

        Returns
        -------
        DispersersConfiguration
        """
        return cls.from_dict(yaml.safe_load(text))

    # -- File I/O ------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a YAML file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        Path(path).write_text(self.to_yaml())

    @classmethod
    def load(cls, path: str | Path) -> DispersersConfiguration:
        """Load from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to a YAML file written by :meth:`save`.

        Returns
        -------
        DispersersConfiguration
        """
        return cls.from_yaml(Path(path).read_text())

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a multi-line summary of this configuration."""
        if not self._dispersers:
            return 'DispersersConfiguration: empty'

        header = f'DispersersConfiguration: {len(self._dispersers)} disperser(s)'

        def _tok_name(token: Parameter | None) -> str:
            if token is None:
                return '— (fixed)'
            return token.name or '—'

        rows = [
            (
                d.name,
                repr(d),
                _tok_name(d.r_scale),
                _tok_name(d.flux_scale),
                _tok_name(d.pix_offset),
            )
            for d in self._dispersers
        ]

        col_headers = ('Name', 'Disperser', 'r_scale', 'flux_scale', 'pix_offset')
        widths = [len(h) for h in col_headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        fmt = '  '.join(f'{{:<{w}}}' for w in widths)
        sep = '  '.join('-' * w for w in widths)
        lines = [header, '', '  ' + fmt.format(*col_headers), '  ' + sep]
        for row in rows:
            lines.append('  ' + fmt.format(*row))

        # Add calibration parameter details section (like Line config)
        calib_params: dict[str, list[tuple[str, Parameter]]] = {
            'r_scale': [],
            'flux_scale': [],
            'pix_offset': [],
        }

        seen_ids: dict[str, set[int]] = {
            'r_scale': set(),
            'flux_scale': set(),
            'pix_offset': set(),
        }

        for d in self._dispersers:
            for attr in ('r_scale', 'flux_scale', 'pix_offset'):
                token = getattr(d, attr)
                if token is not None:
                    token_id = id(token)
                    if token_id not in seen_ids[attr]:
                        seen_ids[attr].add(token_id)
                        calib_params[attr].append((token.name, token))

        # Format calibration parameter sections
        for attr, params in calib_params.items():
            if params:
                lines.append('')
                section_title = attr.replace('_', ' ').title()
                lines.append(f'  {section_title}:')
                name_width = max(len(name) for name, _ in params)
                for name, token in params:
                    lines.append(f'    {name:<{name_width}}  {token.prior!r}')

        return '\n'.join(lines)
