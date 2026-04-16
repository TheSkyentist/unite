"""Continuum configuration for spectral fitting."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from astropy import units as u

from unite._utils import C_KMS, _alpha_name, _ensure_velocity, _ensure_wavelength
from unite.continuum.library import ContinuumForm, Linear, form_from_dict, get_form
from unite.prior import Fixed, Parameter, Prior, Uniform, prior_from_dict

# ------------------------------------------------------------------
# Parameter token classes
# ------------------------------------------------------------------


class Scale(Parameter):
    """Typed token for the ``'scale'`` parameter slot.

    ``scale`` is the continuum flux at ``norm_wav``.  When
    the same :class:`Scale` instance is placed in the ``'scale'``
    slot of multiple :class:`ContinuumRegion` objects, those regions share a
    single sampled amplitude in the model.

    Placing a :class:`Scale` token in any slot **other** than
    ``'scale'`` raises a :exc:`ValueError` at
    :class:`ContinuumConfiguration` construction time.

    Parameters
    ----------
    name : str, optional
        Human-readable label used as the numpyro site name.
    prior : Prior, optional
        Prior distribution.  Defaults to ``Uniform(0, 2)``.
    """

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Uniform(0, 2)
        super().__init__(name, prior=prior)


class NormWavelength(Parameter):
    """Typed token for the ``'norm_wav'`` parameter slot.

    ``norm_wav`` is the rest-frame reference wavelength at
    which the continuum equals ``scale``.  The model automatically applies
    the systemic redshift before evaluating the continuum form.

    Sharing the same :class:`NormWavelength` instance
    across multiple regions ties them to a single consistent reference
    wavelength — essential for globally-normalised forms such as
    :class:`~unite.continuum.library.PowerLaw` and
    :class:`~unite.continuum.library.Blackbody`.

    Placing a :class:`NormWavelength` token in any slot
    **other** than ``'norm_wav'`` raises a :exc:`ValueError`
    at :class:`ContinuumConfiguration` construction time.

    Parameters
    ----------
    name : str, optional
        Human-readable label used as the numpyro site name.
    prior : Prior, optional
        Prior distribution. Defaults to ``Fixed(1.0)`` (1 micron).
    """

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Fixed(1.0)
        super().__init__(name, prior=prior)


class ContShape(Parameter):
    """Typed token for form-specific shape/parameter slots.

    Used for continuum form parameters such as spectral indices (``beta``),
    polynomial coefficients (``c1``, ``c2``, etc.), and other dimensionless
    or form-specific parameters (e.g. ``angle``, ``temperature``, ``tau_v``).
    The default prior is ``Uniform(-10, 10)``; in practice the form's
    :meth:`~unite.continuum.library.ContinuumForm.default_priors` provides
    a more specific default when no token is supplied explicitly.

    Parameters
    ----------
    name : str, optional
        Human-readable label used as the numpyro site name.
    prior : Prior, optional
        Prior distribution. Defaults to ``Uniform(-10, 10)``.
    """

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Uniform(-10, 10)
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
        receive auto-created tokens with the form's default priors when
        the region is added to a :class:`ContinuumConfiguration`.
    name : str, optional
        Human-readable label for this region.  When provided, auto-created
        parameter tokens use this as a suffix (e.g. ``scale_blue``,
        ``beta_blue``).  Region names must be unique within a
        :class:`ContinuumConfiguration`.

        **Custom priors** — supply a :class:`~unite.prior.Parameter`
        with the desired prior for any slot you want to override::

            from unite.prior import Parameter, TruncatedNormal
            region = ContinuumRegion(
                1.0 * u.um, 1.5 * u.um, form=PowerLaw(),
                params={
                    'scale': Parameter('my_scale',
                                       prior=TruncatedNormal(2.0, 0.5, 0, 10)),
                    # 'beta' and 'norm_wav' get default priors
                },
            )

        **Shared parameters** — pass the *same* :class:`~unite.prior.Parameter`
        instance in the ``params`` dict of multiple regions.  The
        :class:`ContinuumConfiguration` detects shared identity and creates
        a single numpyro site, coupling those regions to one sampled value::

            shared_scale = Parameter('global_scale', prior=Uniform(0, 10))
            region1 = ContinuumRegion(1.0 * u.um, 1.5 * u.um, form=PowerLaw(),
                                      params={'scale': shared_scale})
            region2 = ContinuumRegion(2.0 * u.um, 2.5 * u.um, form=PowerLaw(),
                                      params={'scale': shared_scale})

        Use :class:`Scale` and
        :class:`NormWavelength` typed tokens for the
        ``'scale'`` and ``'norm_wav'`` slots respectively;
        they add a validation check that prevents accidentally assigning
        them to the wrong slot.

    Raises
    ------
    TypeError
        If ``low`` or ``high`` are not astropy Quantities with length units.
    ValueError
        If ``low >= high``.
    """

    low: u.Quantity | float
    high: u.Quantity | float
    form: ContinuumForm | str = field(default_factory=Linear)
    params: dict[str, Parameter] = field(default_factory=dict)
    name: str | None = None

    def __post_init__(self) -> None:
        # Resolve string form names to ContinuumForm instances.
        if isinstance(self.form, str):
            self.form = get_form(self.form)

        # Ensure wavelength bounds are Quantities with length units, and that low < high.
        low_q = _ensure_wavelength(self.low, 'low', ndim=0)
        high_q = _ensure_wavelength(self.high, 'high', ndim=0)
        if low_q >= high_q:
            msg = f'ContinuumRegion low must be < high, got low={low_q}, high={high_q}'
            raise ValueError(msg)
        high_q = high_q.to(low_q.unit)

        # Convert to region wavelengths
        self.unit: u.UnitBase = low_q.unit
        self.low = float(low_q.value)
        self.high = float(high_q.value)

        # Prepare the form (e.g. for forms that need to know the wavelength range)
        assert isinstance(self.form, ContinuumForm)
        self.form._prepare(low_q, high_q)

    @property
    def center(self) -> float:
        """Midpoint wavelength of the region."""
        return float((self.low + self.high) / 2.0)


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
# Parameter type rules
# ------------------------------------------------------------------


def _param_class_for(pn: str) -> type[Parameter]:
    """Return the expected :class:`Parameter` subclass for a continuum form param.

    Parameters
    ----------
    pn : str
        Form parameter name (e.g. ``'scale'``, ``'beta'``, ``'angle'``).

    Returns
    -------
    type
        :class:`Scale` for ``'scale'``, :class:`NormWavelength` for
        ``'norm_wav'``, :class:`ContShape`
        for all others.
    """
    if pn == 'scale':
        return Scale
    if pn in 'norm_wav':
        return NormWavelength
    return ContShape


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

    >>> from astropy import units as u
    >>> from unite.continuum import ContinuumConfiguration
    >>> cont = ContinuumConfiguration.from_lines(
    ...     [6564.61, 4862.68, 6585.27, 6549.86] * u.AA)
    >>> len(cont)
    2

    Manual construction with independent regions and custom priors:

    >>> from astropy import units as u
    >>> from unite.continuum.config import (
    ...     ContinuumConfiguration, ContinuumRegion)
    >>> from unite.continuum.library import Linear
    >>> from unite.prior import Parameter, TruncatedNormal, Fixed
    >>> region = ContinuumRegion(
    ...     1.0 * u.um, 1.5 * u.um, form=Linear(),
    ...     params={
    ...         'scale': Parameter('my_scale',
    ...                            prior=TruncatedNormal(2.0, 0.5, 0.0, 10.0)),
    ...         'norm_wav': Parameter('my_nw', prior=Fixed(1.25)),
    ...         # 'slope' receives the default Uniform(-10, 10) prior
    ...     },
    ... )
    >>> cont = ContinuumConfiguration([region])

    Global power-law with parameters shared across two spectral windows:

    >>> from astropy import units as u
    >>> from unite.continuum.config import (
    ...     ContinuumConfiguration, ContinuumRegion, Scale, NormWavelength, ContShape)
    >>> from unite.continuum.library import PowerLaw
    >>> from unite.prior import Uniform, Fixed
    >>> pl = PowerLaw()
    >>> shared_scale = Scale('pl_scale', prior=Uniform(0, 10))
    >>> shared_beta  = ContShape('pl_beta',  prior=Uniform(-5, 5))
    >>> shared_nw    = NormWavelength(
    ...     'pl_nw', prior=Fixed(1.0))  # single reference wavelength
    >>> cont = ContinuumConfiguration([
    ...     ContinuumRegion(0.9 * u.um, 1.4 * u.um, form=pl,
    ...                     params={'scale': shared_scale, 'beta': shared_beta,
    ...                             'norm_wav': shared_nw}),
    ...     ContinuumRegion(1.7 * u.um, 2.5 * u.um, form=pl,
    ...                     params={'scale': shared_scale, 'beta': shared_beta,
    ...                             'norm_wav': shared_nw}),
    ... ])
    >>> # → one 'pl_scale', one 'pl_beta', one 'pl_nw' site in the numpyro model
    """

    def __init__(self, regions: list[ContinuumRegion] | None = None) -> None:
        self._regions: list[ContinuumRegion] = sorted(
            list(regions) if regions else [], key=lambda r: r.low
        )
        self._check_overlaps()
        self._check_duplicate_region_names()
        self._resolved_params: list[dict[str, Parameter]] = self._resolve_params()

    def _check_duplicate_region_names(self) -> None:
        """Raise if two regions share the same non-None name."""
        names = [r.name for r in self._regions if r.name is not None]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            msg = f'Duplicate ContinuumRegion name(s): {dupes}. Region names must be unique.'
            raise ValueError(msg)

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
        """Assign names to all parameter tokens.

        Naming priority for each token:

        1. ``tok.name`` already set (finalized, e.g. re-used across configs).
        2. ``tok.label`` set by user → ``'{param_name}_{label}'``.
        3. Region has a ``name`` → ``'{param_name}_{region.name}'``.
        4. Fallback: global alphabetic counter per param name
           (``scale_a``, ``scale_b``, …).

        **Shared tokens** (same instance in multiple regions) are named on
        first encounter and skipped on subsequent ones, so a shared anonymous
        token gets a single alphabetic name regardless of how many regions
        reference it.

        Raises
        ------
        ValueError
            If a parameter name is not recognized by the form.
        TypeError
            If a token has the wrong type for its slot.
        """
        resolved_list: list[dict[str, Parameter]] = []
        seen: set[int] = set()  # id(tok) → already named
        counters: dict[str, int] = {}  # param_name → next alpha index

        for region in self._regions:
            assert isinstance(region.form, ContinuumForm), (
                f'Expected ContinuumForm but got {type(region.form).__name__}. '
                'String form names should have been resolved in __post_init__.'
            )
            region_form: ContinuumForm = region.form
            # Validate that all param keys match the form's param_names.
            valid_names = set(region_form.param_names())
            invalid = set(region.params) - valid_names
            if invalid:
                msg = (
                    f'{type(region_form).__name__} does not have parameter(s) '
                    f'{invalid}. Valid parameters: {sorted(valid_names)}'
                )
                raise ValueError(msg)

            # Validate token types before resolving.
            for pn, tok in region.params.items():
                expected = _param_class_for(pn)
                if not isinstance(tok, expected):
                    msg = f"Parameter '{pn}' must be a {expected.__name__}, got {type(tok).__name__}."
                    raise TypeError(msg)

            default_priors = region_form.default_priors(region_center=region.center)
            resolved: dict[str, Parameter] = {}

            for pn in region_form.param_names():
                if pn in region.params:
                    tok = region.params[pn]
                    if id(tok) not in seen:
                        seen.add(id(tok))
                        if tok._name is None:
                            if tok.label is not None:
                                tok.name = f'{pn}_{tok.label}'
                            elif region.name is not None:
                                tok.name = f'{pn}_{region.name}'
                                tok.label = region.name
                            else:
                                idx = counters.get(pn, 0)
                                counters[pn] = idx + 1
                                label = _alpha_name(idx)
                                tok.name = f'{pn}_{label}'
                                tok.label = label
                    resolved[pn] = tok
                else:
                    # Auto-create a fresh token for this region.
                    if region.name is not None:
                        auto_label = region.name
                    else:
                        idx = counters.get(pn, 0)
                        counters[pn] = idx + 1
                        auto_label = _alpha_name(idx)
                    auto_name = f'{pn}_{auto_label}'
                    param_class = _param_class_for(pn)
                    new_tok = param_class(prior=default_priors[pn])
                    new_tok.name = auto_name
                    new_tok.label = auto_label
                    resolved[pn] = new_tok

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
        width: u.Quantity = 15_000.0 * u.km / u.s,
        form: ContinuumForm | None = None,
    ) -> ContinuumConfiguration:
        """Auto-generate continuum regions from line wavelengths.

        Each line gets a region of total width *width* (i.e. ``±width/2``
        in velocity space around the line center).  Overlapping regions
        are merged.  All generated regions are independent: each receives
        its own auto-created parameter tokens.

        Parameters
        ----------
        wavelengths : astropy.units.Quantity
            Rest-frame wavelengths of spectral lines (must have length units).
        width : astropy.units.Quantity, optional
            Total velocity width of each region (must have velocity units,
            e.g. ``km/s``).  Each line is padded by ``±width/2``.
            Defaults to ``3000 km/s``.
        form : ContinuumForm, optional
            Functional form to assign to every region.  Defaults to
            :class:`~unite.continuum.library.Linear`.

        Returns
        -------
        ContinuumConfiguration

        Raises
        ------
        TypeError
            If *wavelengths* is not an astropy Quantity with length units,
            or if *width* is not a Quantity with velocity units.
        ValueError
            If *wavelengths* is empty or *width* is not positive.
        """
        wl_q = _ensure_wavelength(wavelengths, 'wavelengths', ndim=1)
        width_q = _ensure_velocity(width, 'width', ndim=0)
        width_kms = float(width_q.to(u.km / u.s).value)
        if width_kms <= 0:
            msg = f'width must be positive, got {width_kms} km/s.'
            raise ValueError(msg)

        wl = np.asarray(wl_q.value, dtype=float)
        if wl.size == 0:
            msg = 'wavelengths must not be empty.'
            raise ValueError(msg)

        if form is None:
            form = Linear()

        # Convert half-width in velocity to a fractional wavelength offset.
        half_frac = (width_kms / 2.0) / C_KMS

        intervals: list[tuple[float, float]] = sorted(
            (float(w * (1.0 - half_frac)), float(w * (1.0 + half_frac))) for w in wl
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
            assert isinstance(region.form, ContinuumForm)
            region_form_to_dict: ContinuumForm = region.form
            fid = id(region_form_to_dict)
            if fid not in form_names:
                type_key = type(region_form_to_dict).__name__.lower()
                idx = type_counts.get(type_key, 0)
                name = f'{type_key}_{idx}'
                type_counts[type_key] = idx + 1
                form_names[fid] = name
                forms_section[name] = region_form_to_dict.to_dict()

        # 2. Collect unique Parameter tokens in first-appearance order.
        #    Build param_namer first so dependent priors can reference other tokens.
        param_namer: dict[object, str] = {}
        params_order: list[tuple[str, Parameter]] = []
        seen_tok_ids: set[int] = set()
        for resolved in self._resolved_params:
            for tok in resolved.values():
                if id(tok) not in seen_tok_ids:
                    seen_tok_ids.add(id(tok))
                    tok_name = tok.name
                    param_namer[tok] = tok_name
                    params_order.append((tok_name, tok))

        params_section: dict[str, dict] = {
            name: tok.prior.to_dict(param_namer) for name, tok in params_order
        }

        # 3. Serialize regions.
        regions_section = []
        for region, resolved in zip(self._regions, self._resolved_params, strict=True):
            rd: dict = {
                'low': region.low,
                'high': region.high,
                'wavelength_unit': str(region.unit),
                'form': form_names[id(region.form)],
                'params': {pn: tok.name for pn, tok in resolved.items()},
            }
            if region.name is not None:
                rd['name'] = region.name
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
        # Build a mapping of token names to their expected types based on usage in regions.
        token_name_to_type: dict[str, type[Parameter]] = {}
        token_name_to_slot: dict[str, str] = {}
        for rd in d['regions']:
            for param_name, token_name in rd['params'].items():
                # Map token_name to its expected parameter type
                token_name_to_type[token_name] = _param_class_for(param_name)
                token_name_to_slot[token_name] = param_name

        # Pass 1: create tokens with finalized names and labels to build the registry.
        token_registry: dict[str, Parameter] = {}
        for tok_name in d['params']:
            param_class = token_name_to_type.get(tok_name, ContShape)
            tok = param_class(prior=Uniform(0, 1))
            tok.name = tok_name
            slot = token_name_to_slot.get(tok_name, '')
            prefix = slot + '_' if slot else ''
            tok.label = (
                tok_name[len(prefix) :]
                if prefix and tok_name.startswith(prefix)
                else tok_name
            )
            token_registry[tok_name] = tok

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
                    name=rd.get('name', None),
                )
            )
        return cls(regions)

    # ------------------------------------------------------------------
    # YAML serialization
    # ------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialize to a YAML string.

        Returns
        -------
        str
        """
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> ContinuumConfiguration:
        """Deserialize from a YAML string.

        Parameters
        ----------
        text : str
            YAML string as produced by :meth:`to_yaml`.

        Returns
        -------
        ContinuumConfiguration
        """
        return cls.from_dict(yaml.safe_load(text))

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a YAML file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        Path(path).write_text(self.to_yaml())

    @classmethod
    def load(cls, path: str | Path) -> ContinuumConfiguration:
        """Load from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to a YAML file written by :meth:`save`.

        Returns
        -------
        ContinuumConfiguration
        """
        return cls.from_yaml(Path(path).read_text())

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

    def __add__(self, other: ContinuumConfiguration) -> ContinuumConfiguration:
        """Combine two configurations (strict mode — raises on parameter name collisions).

        Parameters
        ----------
        other : ContinuumConfiguration

        Returns
        -------
        ContinuumConfiguration
            New configuration containing regions from both *self* and *other*.

        Raises
        ------
        ValueError
            If any user-provided parameter token name appears in both configs.
        TypeError
            If *other* is not a :class:`ContinuumConfiguration`.
        """
        if not isinstance(other, ContinuumConfiguration):
            return NotImplemented

        # Collect user-provided token names from each config (region.params only;
        # auto-created tokens are re-indexed on construction and never collide).
        self_names = {
            tok.name
            for r in self._regions
            for tok in r.params.values()
            if tok._name is not None
        }
        other_names = {
            tok.name
            for r in other._regions
            for tok in r.params.values()
            if tok._name is not None
        }
        collisions = sorted(self_names & other_names)
        if collisions:
            msg = (
                f'Parameter name collision(s) when adding ContinuumConfigurations: '
                f'{collisions}. Rename the conflicting tokens before adding.'
            )
            raise ValueError(msg)

        return ContinuumConfiguration(list(self._regions) + list(other._regions))

    def __repr__(self) -> str:
        if not self._regions:
            return 'ContinuumConfiguration: empty'

        # Count unique param tokens.
        seen_tok_ids: set[int] = set()
        for resolved in self._resolved_params:
            for tok in resolved.values():
                seen_tok_ids.add(id(tok))

        header = f'ContinuumConfiguration: {len(self._regions)} region(s), {len(seen_tok_ids)} parameter(s)'

        # Build table.
        rows = []
        for region, resolved in zip(self._regions, self._resolved_params, strict=True):
            range_str = f'[{region.low}, {region.high}]'
            unit_str = str(region.unit)
            form_str = repr(region.form)
            params_str = ', '.join(tok.name for tok in resolved.values())
            rows.append((range_str, unit_str, form_str, params_str))

        col_headers = ('Range', 'Unit', 'Form', 'Parameters')
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
