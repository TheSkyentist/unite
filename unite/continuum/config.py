"""Continuum configuration for spectral fitting."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
from astropy import units as u

from unite._utils import _ensure_wavelength
from unite.continuum.library import ContinuumForm, Linear, form_from_dict
from unite.prior import Fixed, Parameter, Prior, Uniform, prior_from_dict

# ------------------------------------------------------------------
# Parameter token classes
# ------------------------------------------------------------------


class ContinuumScale(Parameter):
    """Typed token for the ``'scale'`` parameter slot.

    ``scale`` is the continuum flux at ``normalization_wavelength``.  When
    the same :class:`ContinuumScale` instance is placed in the ``'scale'``
    slot of multiple :class:`ContinuumRegion` objects, those regions share a
    single sampled amplitude in the model.

    Placing a :class:`ContinuumScale` token in any slot **other** than
    ``'scale'`` raises a :exc:`ValueError` at
    :class:`ContinuumConfiguration` construction time.

    Parameters
    ----------
    name : str, optional
        Human-readable label used as the numpyro site name.
    prior : Prior, optional
        Prior distribution.  Defaults to ``Uniform(0, 10)``.
    """

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Uniform(0, 10)
        super().__init__(name, prior=prior)


class ContinuumNormalizationWavelength(Parameter):
    """Typed token for the ``'normalization_wavelength'`` parameter slot.

    ``normalization_wavelength`` is the rest-frame reference wavelength at
    which the continuum equals ``scale``.  The model automatically applies
    the systemic redshift before evaluating the continuum form.

    Sharing the same :class:`ContinuumNormalizationWavelength` instance
    across multiple regions ties them to a single consistent reference
    wavelength — essential for globally-normalised forms such as
    :class:`~unite.continuum.library.PowerLaw` and
    :class:`~unite.continuum.library.Blackbody`.

    Placing a :class:`ContinuumNormalizationWavelength` token in any slot
    **other** than ``'normalization_wavelength'`` raises a :exc:`ValueError`
    at :class:`ContinuumConfiguration` construction time.

    Parameters
    ----------
    name : str, optional
        Human-readable label used as the numpyro site name.
    prior : Prior, optional
        Prior distribution.  Defaults to ``Fixed(1.0)``.  In practice,
        use ``Fixed(value)`` with your chosen rest-frame reference
        wavelength.
    """

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Fixed(1.0)
        super().__init__(name, prior=prior)


# ------------------------------------------------------------------
# ContinuumRegion
# ------------------------------------------------------------------


@dataclass
class ContinuumRegion:
    """A single continuum region with wavelength bounds and functional form.

    Parameters
    ----------
    low : astropy.units.Quantity
        Lower wavelength bound (must have length units).
    high : astropy.units.Quantity
        Upper wavelength bound (must have length units).
    form : ContinuumForm
        Functional form for the continuum in this region.
    params : dict of str to Parameter, optional
        Parameter tokens for the form's parameters, keyed by
        :meth:`ContinuumForm.param_names`.  Parameters not listed here
        receive auto-created tokens when the region is added to a
        :class:`ContinuumConfiguration`.

    Raises
    ------
    TypeError
        If ``low`` or ``high`` are not astropy Quantities with length units.
    ValueError
        If ``low >= high``.
    """

    low: u.Quantity
    high: u.Quantity
    form: ContinuumForm = field(default_factory=Linear)
    params: dict[str, Parameter] = field(default_factory=dict)

    def __post_init__(self) -> None:
        low_q = _ensure_wavelength(self.low, 'low')
        high_q = _ensure_wavelength(self.high, 'high')
        self._unit: u.UnitBase = low_q.unit
        self.low: float = float(low_q.value)
        self.high: float = float(high_q.to(low_q.unit).value)
        if self.low >= self.high:
            msg = (
                f'ContinuumRegion low must be < high, '
                f'got low={self.low}, high={self.high}'
            )
            raise ValueError(msg)

    @property
    def center(self) -> float:
        """Midpoint wavelength of the region."""
        return (self.low + self.high) / 2.0


# ------------------------------------------------------------------
# Region auto-generation helpers
# ------------------------------------------------------------------


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge overlapping intervals.

    Parameters
    ----------
    intervals : list of (low, high) tuples
        Must be sorted by ``low``.

    Returns
    -------
    list of (low, high) tuples
        Non-overlapping, sorted intervals.
    """
    if not intervals:
        return []
    merged = [intervals[0]]
    for low, high in intervals[1:]:
        if low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


# ------------------------------------------------------------------
# ContinuumConfiguration
# ------------------------------------------------------------------


class ContinuumConfiguration:
    """Collection of continuum regions for spectral fitting.

    Regions are sorted by wavelength at construction time and must not
    overlap.  Parameters for each region are represented as
    :class:`Parameter` tokens.  Sharing the **same** token instance
    across multiple regions ties those parameters to a single sampled
    value in the model — analogous to how :class:`~unite.line.config.FWHM`
    and :class:`~unite.line.config.Flux` tokens work for emission lines.

    Parameters
    ----------
    regions : list of ContinuumRegion, optional
        Continuum regions.  Will be sorted by ``low`` bound.

    Raises
    ------
    ValueError
        If any two regions overlap.

    Examples
    --------
    Auto-generate from line wavelengths (independent local continua):

    >>> from unite.continuum import ContinuumConfiguration
    >>> cont = ContinuumConfiguration.from_lines(
    ...     [6564.61, 4862.68, 6585.27, 6549.86])
    >>> len(cont)
    2

    Manual construction with a global power-law (shared parameters):

    >>> from unite.continuum import (
    ...     ContinuumConfiguration, Parameter, ContinuumRegion, PowerLaw)
    >>> from unite.prior import Uniform
    >>> pl = PowerLaw(pivot=2.5)
    >>> amp = Parameter('pl_amp', prior=Uniform(0, 10))
    >>> beta = Parameter('pl_beta', prior=Uniform(-5, 5))
    >>> cont = ContinuumConfiguration([
    ...     ContinuumRegion(4600, 5100, pl, params={'amplitude': amp, 'beta': beta}),
    ...     ContinuumRegion(6200, 6900, pl, params={'amplitude': amp, 'beta': beta}),
    ... ])
    """

    def __init__(self, regions: list[ContinuumRegion] | None = None) -> None:
        self._regions: list[ContinuumRegion] = sorted(
            list(regions) if regions else [], key=lambda r: r.low
        )
        self._check_overlaps()
        self._resolved_params: list[dict[str, Parameter]] = self._resolve_params()

    def _check_overlaps(self) -> None:
        """Warn if any two regions overlap; overlapping contributions are summed."""
        for i, a in enumerate(self._regions):
            for b in self._regions[i + 1 :]:
                if a.high <= b.low:
                    break  # sorted, no further overlap possible
                warnings.warn(
                    f'Continuum regions [{a.low}, {a.high}] and '
                    f'[{b.low}, {b.high}] overlap; their contributions '
                    f'will be summed in the model.',
                    UserWarning,
                    stacklevel=3,
                )

    def _resolve_params(self) -> list[dict[str, Parameter]]:
        """Assign auto-created tokens to any unspecified params.

        Each region receives independently named auto-created tokens for
        every parameter not provided in ``ContinuumRegion.params``.
        Naming follows ``cont_{form_type}_{region_idx}_{param_name}``
        where *region_idx* counts instances of each form type in sorted
        order.  Explicitly provided tokens are used as-is.

        Raises
        ------
        ValueError
            If a :class:`ContinuumScale` token is placed in a slot other
            than ``'scale'``, or a
            :class:`ContinuumNormalizationWavelength` token is placed in a
            slot other than ``'normalization_wavelength'``.
        """
        type_counts: dict[str, int] = {}
        resolved_list: list[dict[str, Parameter]] = []
        for region in self._regions:
            # Validate typed tokens before resolving.
            for pn, tok in region.params.items():
                if isinstance(tok, ContinuumScale) and pn != 'scale':
                    msg = (
                        f'ContinuumScale token can only be placed in the '
                        f'"scale" slot, got "{pn}"'
                    )
                    raise ValueError(msg)
                if (
                    isinstance(tok, ContinuumNormalizationWavelength)
                    and pn != 'normalization_wavelength'
                ):
                    msg = (
                        f'ContinuumNormalizationWavelength token can only be placed '
                        f'in the "normalization_wavelength" slot, got "{pn}"'
                    )
                    raise ValueError(msg)

            tk = type(region.form).__name__.lower()
            idx = type_counts.get(tk, 0)
            type_counts[tk] = idx + 1
            default_priors = region.form.default_priors(region_center=region.center)
            resolved: dict[str, Parameter] = {}
            for pn in region.form.param_names():
                if pn in region.params:
                    resolved[pn] = region.params[pn]
                else:
                    auto_name = f'cont_{tk}_{idx}_{pn}'
                    resolved[pn] = Parameter(auto_name, prior=default_priors[pn])
            resolved_list.append(resolved)
        return resolved_list

    @property
    def resolved_params(self) -> list[dict[str, Parameter]]:
        """Resolved parameter tokens for each region (read-only copy).

        Index *i* contains the full ``{param_name: Parameter}`` mapping
        for ``self.regions[i]``, including auto-created tokens for any
        parameters the user did not explicitly provide.
        """
        return [dict(d) for d in self._resolved_params]

    @classmethod
    def from_lines(
        cls,
        wavelengths: u.Quantity,
        pad: float = 0.05,
        form: ContinuumForm | None = None,
    ) -> ContinuumConfiguration:
        """Auto-generate continuum regions from line wavelengths.

        Each line is padded by a fractional amount in wavelength space,
        and overlapping regions are merged.  All generated regions are
        independent: each receives its own auto-created parameter tokens.

        Parameters
        ----------
        wavelengths : astropy.units.Quantity
            Rest-frame wavelengths of spectral lines (must have length units).
        pad : float
            Fractional wavelength padding (dimensionless, delta-z
            equivalent).  Default ``0.05`` corresponds to ~15,000 km/s.
        form : ContinuumForm, optional
            Functional form to assign to every region.  Defaults to
            :class:`~unite.continuum.library.Linear`.

        Returns
        -------
        ContinuumConfiguration

        Raises
        ------
        TypeError
            If *wavelengths* is not an astropy Quantity with length units.
        ValueError
            If *wavelengths* is empty.
        """
        wl_q = _ensure_wavelength(wavelengths, 'wavelengths')
        wl = np.asarray(wl_q.value, dtype=float)
        if wl.size == 0:
            msg = 'wavelengths must not be empty.'
            raise ValueError(msg)

        if form is None:
            form = Linear()

        intervals: list[tuple[float, float]] = sorted(
            (float(w * (1.0 - pad)), float(w * (1.0 + pad))) for w in wl
        )
        merged = _merge_intervals(intervals)
        regions = [
            ContinuumRegion(low=lo * wl_q.unit, high=hi * wl_q.unit, form=form)
            for lo, hi in merged
        ]
        return cls(regions)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a YAML-safe dictionary.

        Unique :class:`ContinuumForm` instances are collected into a
        ``'forms'`` section keyed by auto-generated names.  Unique
        :class:`Parameter` tokens are collected into a ``'params'``
        section keyed by their names.  Each region references its form
        and params by name, so shared token objects round-trip correctly.

        Returns
        -------
        dict
            Keys: ``'params'``, ``'forms'``, ``'regions'``.
        """
        # 1. Assign names to unique form objects.
        form_names: dict[int, str] = {}
        type_counts: dict[str, int] = {}
        forms_section: dict[str, dict] = {}
        for region in self._regions:
            fid = id(region.form)
            if fid not in form_names:
                type_key = type(region.form).__name__.lower()
                idx = type_counts.get(type_key, 0)
                name = f'{type_key}_{idx}'
                type_counts[type_key] = idx + 1
                form_names[fid] = name
                forms_section[name] = region.form.to_dict()

        # 2. Collect unique Parameter tokens in first-appearance order.
        #    Build param_namer first so dependent priors can reference other tokens.
        param_namer: dict[object, str] = {}
        params_order: list[tuple[str, Parameter]] = []
        seen_tok_ids: set[int] = set()
        for resolved in self._resolved_params:
            for tok in resolved.values():
                if id(tok) not in seen_tok_ids:
                    seen_tok_ids.add(id(tok))
                    param_namer[tok] = tok.name
                    params_order.append((tok.name, tok))

        params_section: dict[str, dict] = {
            name: tok.prior.to_dict(param_namer) for name, tok in params_order
        }

        # 3. Serialize regions.
        regions_section = []
        for region, resolved in zip(self._regions, self._resolved_params, strict=True):
            rd: dict = {
                'low': region.low,
                'high': region.high,
                'wavelength_unit': str(region._unit),
                'form': form_names[id(region.form)],
                'params': {pn: tok.name for pn, tok in resolved.items()},
            }
            regions_section.append(rd)

        return {
            'params': params_section,
            'forms': forms_section,
            'regions': regions_section,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ContinuumConfiguration:
        """Deserialize from a dictionary.

        Parameters
        ----------
        d : dict
            As produced by :meth:`to_dict`.

        Returns
        -------
        ContinuumConfiguration
        """
        # 1. Reconstruct form objects.
        form_objects: dict[str, ContinuumForm] = {
            name: form_from_dict(fd) for name, fd in d['forms'].items()
        }

        # 2. Reconstruct Parameter tokens (two passes for dependent priors).
        # Pass 1: create tokens with placeholder priors to build the registry.
        token_registry: dict[str, Parameter] = {
            name: Parameter(name, prior=Uniform(0, 1)) for name in d['params']
        }
        # Pass 2: overwrite with real priors (may reference tokens in registry).
        for name, pd in d['params'].items():
            token_registry[name].prior = prior_from_dict(
                pd, token_registry=token_registry
            )

        # 3. Reconstruct regions.
        regions = []
        for rd in d['regions']:
            params = {pn: token_registry[tn] for pn, tn in rd['params'].items()}
            wl_unit = u.Unit(rd.get('wavelength_unit', 'um'))
            regions.append(
                ContinuumRegion(
                    low=rd['low'] * wl_unit,
                    high=rd['high'] * wl_unit,
                    form=form_objects[rd['form']],
                    params=params,
                )
            )
        return cls(regions)

    # ------------------------------------------------------------------
    # Container interface
    # ------------------------------------------------------------------

    @property
    def regions(self) -> list[ContinuumRegion]:
        """List of continuum regions (read-only copy)."""
        return list(self._regions)

    def __len__(self) -> int:
        return len(self._regions)

    def __iter__(self) -> Iterator[ContinuumRegion]:
        return iter(self._regions)

    def __getitem__(self, idx: int) -> ContinuumRegion:
        return self._regions[idx]

    def __repr__(self) -> str:
        if not self._regions:
            return 'ContinuumConfiguration: empty'

        # Count unique param tokens.
        seen_tok_ids: set[int] = set()
        for resolved in self._resolved_params:
            for tok in resolved.values():
                seen_tok_ids.add(id(tok))

        header = (
            f'ContinuumConfiguration: {len(self._regions)} region(s), '
            f'{len(seen_tok_ids)} parameter(s)'
        )

        # Build table.
        rows = []
        for region, resolved in zip(self._regions, self._resolved_params, strict=True):
            range_str = f'[{region.low}, {region.high}]'
            form_str = repr(region.form)
            params_str = ', '.join(tok.name for tok in resolved.values())
            rows.append((range_str, form_str, params_str))

        col_headers = ('Range', 'Form', 'Parameters')
        widths = [len(h) for h in col_headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        fmt = '  '.join(f'{{:<{w}}}' for w in widths)
        sep = '  '.join('-' * w for w in widths)
        lines = [header, '', '  ' + fmt.format(*col_headers), '  ' + sep]
        for row in rows:
            lines.append('  ' + fmt.format(*row))

        # Parameters section: unique tokens in first-appearance order.
        seen_tok_ids2: set[int] = set()
        unique_toks: list[tuple[str, Parameter]] = []
        for resolved in self._resolved_params:
            for tok in resolved.values():
                if id(tok) not in seen_tok_ids2:
                    seen_tok_ids2.add(id(tok))
                    unique_toks.append((tok.name, tok))

        if unique_toks:
            lines.append('')
            lines.append('  Parameters:')
            name_width = max(len(n) for n, _ in unique_toks)
            for name, tok in unique_toks:
                lines.append(f'    {name:<{name_width}}  {tok.prior!r}')

        return '\n'.join(lines)
