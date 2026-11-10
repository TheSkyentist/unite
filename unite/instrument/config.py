"""Instrument configuration: disperser models with calibration tokens.

:class:`InstrumentConfig` collects one :class:`~unite.instrument.base.Disperser`
per observing configuration.  Each disperser carries optional
:class:`~unite.instrument.base.RScale` / :class:`~unite.instrument.base.FluxScale` /
:class:`~unite.instrument.base.PixOffset` tokens (``r_scale``, ``flux_scale``,
``pix_offset``) for shared calibration parameters.

Sharing tokens
--------------
Pass the **same** token instance to multiple dispersers to share a single
parameter in the fitted model::

    from unite.instrument import InstrumentConfig, RScale
    from unite.instrument.nirspec import G235H, G395H
    from unite.prior import TruncatedNormal

    r = RScale(prior=TruncatedNormal(1.0, 0.05, 0.8, 1.2))
    cfg = InstrumentConfig([
        G235H(r_scale=r),
        G395H(r_scale=r),   # same r — shared parameter
    ])

Degeneracy warning
------------------
A multi-disperser fit is only identified if at least one disperser's
``flux_scale`` is anchored (a ``Fixed`` prior — the default when the token
is left unset) and at least one disperser's ``pix_offset`` is likewise
anchored.  :meth:`validate` issues :class:`UserWarning` when these
conditions are not met.  Validation is called automatically when this
object is passed to :class:`~unite.config.Configuration`.

Serialization
-------------
:meth:`to_dict` hoists all unique calibration tokens to a top-level
``calib_params`` section keyed by token name.  Disperser entries reference
tokens by name.  Shared tokens round-trip correctly — the same object is
reconstructed for both entries.

Examples
--------
>>> from unite.instrument import InstrumentConfig, RScale, FluxScale
>>> from unite.instrument.nirspec import G235H, G395H
>>> from unite.prior import TruncatedNormal
>>> r = RScale(prior=TruncatedNormal(1.0, 0.05, 0.8, 1.2))
>>> flux_0 = FluxScale(prior=TruncatedNormal(1.0, 0.1, 0.5, 2.0))
>>> cfg = InstrumentConfig([
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

from unite._utils import _alpha_name
from unite.instrument.base import (
    Disperser,
    FluxScale,
    OrderScale,
    PixOffset,
    RScale,
    format_calibration_sections,
)
from unite.instrument.nirspec.disperser import (
    G140H,
    G140M,
    G235H,
    G235M,
    G395H,
    G395M,
    PRISM,
    NIRSpec,
)
from unite.instrument.sdss.disperser import SDSSDisperser
from unite.prior import Fixed, Parameter, prior_from_dict

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
    'OrderScale': OrderScale,
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
    tok = cls(prior=prior)
    tok.name = name  # set finalized site name directly to avoid re-prefixing
    tok.label = name  # keep label consistent with name
    return tok


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
    if isinstance(disperser, NIRSpec):
        d['grating'] = disperser.grating
        d['r_source'] = disperser.r_source

    # CalibParam references by token name (fixed slots only).
    for slot, token in disperser.calibration_tokens():
        if slot in Disperser.CALIB_SLOTS:
            d[slot] = token.name

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
    if cls_name == 'NIRSpec':
        kwargs['grating'] = d['grating']
        kwargs['r_source'] = d.get('r_source', 'point')
    elif 'r_source' in d:
        kwargs['r_source'] = d['r_source']

    for slot in Disperser.CALIB_SLOTS:
        kwargs[slot] = token_registry[d[slot]]

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Degeneracy warnings
# ---------------------------------------------------------------------------

_FLUX_DEGENERACY_WARNING = (
    'InstrumentConfig: no disperser has an anchored (Fixed-prior) flux_scale. '
    'Relative flux scales are degenerate — leave flux_scale unset (or give it '
    'a Fixed prior) on one disperser to anchor the flux calibration.'
)

_PIX_DEGENERACY_WARNING = (
    'InstrumentConfig: no disperser has an anchored (Fixed-prior) pix_offset. '
    'Pixel-offset (dispersion) scales are degenerate — leave pix_offset unset '
    '(or give it a Fixed prior) on one disperser to anchor the wavelength solution.'
)


# ---------------------------------------------------------------------------
# InstrumentConfig
# ---------------------------------------------------------------------------


class InstrumentConfig:
    """Configuration for a multi-disperser spectral dataset.

    An ordered collection of :class:`~unite.instrument.base.Disperser` objects,
    one per observing disperser.  Each disperser carries optional
    :class:`~unite.instrument.base.CalibParam` tokens for calibration parameters.

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
            msg = f'Dispersers at indices {empty} have empty names. Set a non-empty name on each disperser.'
            raise ValueError(msg)
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            msg = f'Duplicate disperser name(s) in InstrumentConfig: {dupes}'
            raise ValueError(msg)
        self._dispersers: list[Disperser] = list(dispersers)
        self._assign_calib_names()

    # -- calibration token naming --------------------------------------------

    def _assign_calib_names(self) -> None:
        """Assign site names to any anonymous calibration tokens.

        Tokens with a user-supplied label already have their site name set by
        :class:`~unite.instrument.base.Disperser`.  This method handles the
        remaining anonymous tokens (``label=None``, ``name=None``):

        * **Unshared** token (one slot on a single named disperser) →
          ``'{prefix}_{disperser.name}{qualifier}'``, where the qualifier is
          the part of the slot label after the prefix (empty for the fixed
          slots), e.g. ``FluxScale()`` on ``G235H`` → ``'flux_scale_G235H'``.
        * **Shared** token (several slots or dispersers), or disperser with no
          name → alphabetic counter per prefix: ``'r_scale_a'``,
          ``'r_scale_b'``, …

        Slots are enumerated with
        :meth:`~unite.instrument.base.Disperser.calibration_tokens`.  A shared
        anonymous token is encountered once and receives exactly one name.
        """
        # Count the slots each anonymous token occupies (over all dispersers).
        tok_slots: dict[int, int] = {}
        for d in self._dispersers:
            for _, tok in d.calibration_tokens():
                if tok._name is None:
                    tok_slots[id(tok)] = tok_slots.get(id(tok), 0) + 1

        counters: dict[str, int] = {}  # site prefix → next alpha index
        seen: set[int] = set()  # id(tok) already named

        for d in self._dispersers:
            for slot, tok in d.calibration_tokens():
                if tok._name is not None or id(tok) in seen:
                    continue
                seen.add(id(tok))
                prefix = tok.site_prefix
                if tok.label is not None:
                    # User-supplied label takes priority (name already set by base.py).
                    pass
                elif tok_slots.get(id(tok), 0) == 1 and d.name:
                    # Unshared token on a single named disperser: use its name,
                    # plus any slot qualifier beyond the prefix (e.g. the order).
                    suffix = slot[len(prefix) :]
                    tok.name = f'{prefix}_{d.name}{suffix}'
                    tok.label = f'{d.name}{suffix}'
                else:
                    # Shared across several slots/dispersers, or unnamed
                    # disperser: alphabetic counter per prefix.
                    idx = counters.get(prefix, 0)
                    counters[prefix] = idx + 1
                    label = _alpha_name(idx)
                    tok.name = f'{prefix}_{label}'
                    tok.label = label

    # -- validation ----------------------------------------------------------

    def validate(self) -> None:
        """Check for flux and pixel-offset degeneracies.

        Issues a :class:`UserWarning` for each calibration axis (flux,
        dispersion) where no disperser is anchored (``Fixed`` prior).
        """
        if len(self._dispersers) > 1:
            if all(not isinstance(d.flux_scale.prior, Fixed) for d in self._dispersers):
                warnings.warn(_FLUX_DEGENERACY_WARNING, UserWarning, stacklevel=2)
            if all(not isinstance(d.pix_offset.prior, Fixed) for d in self._dispersers):
                warnings.warn(_PIX_DEGENERACY_WARNING, UserWarning, stacklevel=2)

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
            msg = f'No disperser named {key!r} in InstrumentConfig.'
            raise KeyError(msg)
        return self._dispersers[key]

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a YAML-safe dictionary.

        Shared :class:`~unite.instrument.base.CalibParam` tokens are hoisted to a
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
            for _, token in d.calibration_tokens():
                if id(token) not in seen_ids:
                    seen_ids[id(token)] = token

        calib_params = {t.name: _calib_param_to_dict(t) for t in seen_ids.values()}
        entries = [_disperser_to_entry(d) for d in self._dispersers]

        return {'calib_params': calib_params, 'entries': entries}

    @classmethod
    def from_dict(cls, d: dict) -> InstrumentConfig:
        """Deserialize from a dictionary.

        Parameters
        ----------
        d : dict
            As produced by :meth:`to_dict`.

        Returns
        -------
        InstrumentConfig
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
    def from_yaml(cls, text: str) -> InstrumentConfig:
        """Deserialize from a YAML string.

        Parameters
        ----------
        text : str
            YAML string as produced by :meth:`to_yaml`.

        Returns
        -------
        InstrumentConfig
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
    def load(cls, path: str | Path) -> InstrumentConfig:
        """Load from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to a YAML file written by :meth:`save`.

        Returns
        -------
        InstrumentConfig
        """
        return cls.from_yaml(Path(path).read_text())

    # -- addition ------------------------------------------------------------

    def __add__(self, other: InstrumentConfig) -> InstrumentConfig:
        """Combine two configurations (strict mode — raises on duplicate names).

        Parameters
        ----------
        other : InstrumentConfig

        Returns
        -------
        InstrumentConfig
            New configuration containing dispersers from both *self* and *other*.

        Raises
        ------
        ValueError
            If any disperser name appears in both configurations.
        TypeError
            If *other* is not an :class:`InstrumentConfig`.
        """
        if not isinstance(other, InstrumentConfig):
            return NotImplemented

        self_names = set(self.names)
        collisions = [n for n in other.names if n in self_names]
        if collisions:
            msg = f'Duplicate disperser name(s) in InstrumentConfig addition: {collisions}'
            raise ValueError(msg)

        return InstrumentConfig(list(self._dispersers) + list(other._dispersers))

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a multi-line summary of this configuration."""
        if not self._dispersers:
            return 'InstrumentConfig: empty'

        header = f'InstrumentConfig: {len(self._dispersers)} disperser(s)'

        def _tok_name(token: Parameter) -> str:
            suffix = ' (fixed)' if isinstance(token.prior, Fixed) else ''
            return f'{token.name}{suffix}'

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

        lines.extend(format_calibration_sections(self._dispersers))

        return '\n'.join(lines)
