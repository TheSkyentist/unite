"""Continuum configuration for spectral fitting."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np

from unite.continuum.library import ContinuumForm, Linear, form_from_dict
from unite.prior import Prior, prior_from_dict


@dataclass
class ContinuumRegion:
    """A single continuum region with wavelength bounds and functional form.

    Parameters
    ----------
    low : float
        Lower wavelength bound.
    high : float
        Upper wavelength bound.
    form : ContinuumForm
        Functional form for the continuum in this region.
    priors : dict of str to Prior
        Per-region priors for the form's parameters, keyed by
        :meth:`ContinuumForm.param_names`.  Parameters not listed here
        use the form's :meth:`~ContinuumForm.default_priors`.

    Raises
    ------
    ValueError
        If ``low >= high``.
    """

    low: float
    high: float
    form: ContinuumForm = field(default_factory=Linear)
    priors: dict[str, Prior] = field(default_factory=dict)

    def __post_init__(self) -> None:
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
    overlap.  A single :class:`ContinuumForm` instance may be shared
    across multiple regions to express a global continuum component (e.g.
    a power-law or blackbody that spans the whole spectrum) evaluated only
    where data exist.  Shared form identity is preserved through
    serialization round-trips.

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
    Auto-generate from line wavelengths:

    >>> from unite.continuum import ContinuumConfiguration
    >>> cont = ContinuumConfiguration.from_lines(
    ...     [6564.61, 4862.68, 6585.27, 6549.86])
    >>> len(cont)
    2

    Manual construction with a shared power-law:

    >>> from unite.continuum import ContinuumConfiguration, ContinuumRegion, PowerLaw
    >>> pl = PowerLaw()
    >>> cont = ContinuumConfiguration([
    ...     ContinuumRegion(4600, 5100, pl),
    ...     ContinuumRegion(6200, 6900, pl),
    ... ])
    """

    def __init__(self, regions: list[ContinuumRegion] | None = None) -> None:
        self._regions: list[ContinuumRegion] = sorted(
            list(regions) if regions else [], key=lambda r: r.low
        )
        self._check_overlaps()

    def _check_overlaps(self) -> None:
        for i in range(len(self._regions) - 1):
            a, b = self._regions[i], self._regions[i + 1]
            if a.high > b.low:
                msg = (
                    f'Continuum regions overlap: '
                    f'[{a.low}, {a.high}] and [{b.low}, {b.high}]'
                )
                raise ValueError(msg)

    @classmethod
    def from_lines(
        cls,
        wavelengths: Sequence[float],
        pad: float = 0.05,
        form: ContinuumForm | None = None,
    ) -> ContinuumConfiguration:
        """Auto-generate continuum regions from line wavelengths.

        Each line is padded by a fractional amount in wavelength space,
        and overlapping regions are merged.  All generated regions share
        the same ``form`` instance.

        Parameters
        ----------
        wavelengths : sequence of float
            Rest-frame wavelengths of spectral lines.
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
        ValueError
            If *wavelengths* is empty.
        """
        wl = np.asarray(wavelengths, dtype=float)
        if wl.size == 0:
            msg = 'wavelengths must not be empty.'
            raise ValueError(msg)

        if form is None:
            form = Linear()

        intervals: list[tuple[float, float]] = sorted(
            (float(w * (1.0 - pad)), float(w * (1.0 + pad))) for w in wl
        )
        merged = _merge_intervals(intervals)
        regions = [ContinuumRegion(low=lo, high=hi, form=form) for lo, hi in merged]
        return cls(regions)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a YAML-safe dictionary.

        Unique :class:`ContinuumForm` instances are collected into a
        top-level ``'forms'`` section keyed by auto-generated names
        (e.g. ``'powerlaw_0'``, ``'linear_0'``).  Each region entry
        references its form by name, so shared form objects round-trip
        correctly.

        Returns
        -------
        dict
            Keys: ``'forms'``, ``'regions'``.
        """
        # Collect unique form objects in first-appearance order and auto-name them.
        form_names: dict[int, str] = {}  # id(form) -> assigned name
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

        regions_section = []
        for region in self._regions:
            rd: dict = {
                'low': region.low,
                'high': region.high,
                'form': form_names[id(region.form)],
            }
            if region.priors:
                rd['priors'] = {k: v.to_dict() for k, v in region.priors.items()}
            regions_section.append(rd)

        return {'forms': forms_section, 'regions': regions_section}

    @classmethod
    def from_dict(cls, d: dict) -> ContinuumConfiguration:
        """Deserialize from a dictionary.

        Parameters
        ----------
        d : dict
            As produced by :meth:`to_dict`.  Must contain ``'forms'``
            and ``'regions'`` keys.

        Returns
        -------
        ContinuumConfiguration
        """
        form_objects: dict[str, ContinuumForm] = {
            name: form_from_dict(fd) for name, fd in d['forms'].items()
        }
        regions = []
        for rd in d['regions']:
            priors = {k: prior_from_dict(v) for k, v in rd.get('priors', {}).items()}
            regions.append(
                ContinuumRegion(
                    low=rd['low'],
                    high=rd['high'],
                    form=form_objects[rd['form']],
                    priors=priors,
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

        # Collect unique forms
        seen: dict[int, str] = {}
        type_counts: dict[str, int] = {}
        for r in self._regions:
            fid = id(r.form)
            if fid not in seen:
                tk = type(r.form).__name__.lower()
                idx = type_counts.get(tk, 0)
                seen[fid] = f'{type(r.form).__name__}_{idx}'
                type_counts[tk] = idx + 1

        header = (
            f'ContinuumConfiguration: {len(self._regions)} regions, '
            f'{len(seen)} unique form(s)'
        )

        rows = []
        for r in self._regions:
            form_name = seen[id(r.form)]
            # Show prior names in the table (only explicitly set ones)
            prior_names = ', '.join(r.priors.keys()) if r.priors else '—'
            rows.append((f'[{r.low}, {r.high}]', form_name, prior_names))

        col_headers = ('Range', 'Form', 'Priors')
        widths = [len(h) for h in col_headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        fmt = '  '.join(f'{{:<{w}}}' for w in widths)
        sep = '  '.join('-' * w for w in widths)
        lines = [header, '', '  ' + fmt.format(*col_headers), '  ' + sep]
        for row in rows:
            lines.append('  ' + fmt.format(*row))

        # Add prior details section (like Line config)
        # Show all priors (both explicit and default) for all regions
        
        # Collect all unique priors across all regions
        seen_priors: dict[int, tuple[str, object, str]] = {}  # (name, prior, region_info)
        for r in self._regions:
            # Add explicit priors
            for name, prior in r.priors.items():
                prior_id = id(prior)
                if prior_id not in seen_priors:
                    seen_priors[prior_id] = (name, prior, f'region [{r.low}, {r.high}]')
            
            # Add default priors that are not overridden
            default_priors = r.form.default_priors()
            for name, prior in default_priors.items():
                if name not in r.priors:  # Only show if not explicitly set
                    prior_id = id(prior)
                    if prior_id not in seen_priors:
                        seen_priors[prior_id] = (name, prior, f'region [{r.low}, {r.high}]')
        
        # Format priors section
        if seen_priors:
            lines.append('')
            lines.append('  Priors:')
            name_width = max(len(name) for name, _, _ in seen_priors.values())
            for name, prior, region_info in seen_priors.values():
                lines.append(f'    {name:<{name_width}}  {prior!r}  # {region_info}')

        return '\n'.join(lines)
