"""Emission line configuration for spectral fitting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count

import jax.numpy as jnp
import numpy as np
from astropy import units as u

from unite._utils import _alpha_name, _broadcast, _ensure_wavelength
from unite.line.profiles import Profile, profile_from_dict, resolve_profile
from unite.prior import Parameter, Prior, Uniform, prior_from_dict

# ------------------------------------------------------------------
# Parameter token classes
# ------------------------------------------------------------------


class Redshift(Parameter):
    """
    A named delta redshift parameter that can be shared between lines.

    Default priors is uniform between -0.01 and 0.01.
    """

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Uniform(-0.0025, 0.0025)
        super().__init__(name, prior=prior)


class FWHM(Parameter):
    """
    A named FWHM parameter (km/s) that can be shared between lines.

    Default prior is uniform between 0 and 1000 km/s.
    """

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Uniform(0, 1000)
        super().__init__(name, prior=prior)


class Param(Parameter):
    """
    A named generic shape parameter that can be shared between lines.

    Used for dimensionless profile shape parameters such as Gauss-Hermite
    moments (h3, h4) or spectral indices.  The default prior is a wide
    uniform ``Uniform(-10, 10)``; in practice the profile's
    :meth:`~unite.profile.base.Profile.default_priors` provides a more
    specific default when no token is supplied explicitly.
    """

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Uniform(-10, 10)
        super().__init__(name, prior=prior)


class Flux(Parameter):
    """
    A named flux parameter that can be shared between lines.

    Default prior is uniform between -3 and 3 (relative to initial guess).
    """

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Uniform(-3, 3)
        super().__init__(name, prior=prior)


# ------------------------------------------------------------------
# Alphabet-based auto-naming
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Wavelength extraction
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Parameter type rules
# ------------------------------------------------------------------


def _param_class_for(pn: str) -> type[Parameter]:
    """Return the expected :class:`Parameter` subclass for a profile param name.

    Parameters
    ----------
    pn : str
        Profile parameter name (e.g. ``'fwhm_gauss'``, ``'h3'``).

    Returns
    -------
    type
        :class:`FWHM` for names starting with ``'fwhm'``, :class:`Param`
        otherwise.
    """
    return FWHM if pn.startswith('fwhm') else Param


# ------------------------------------------------------------------
# Profile param resolution
# ------------------------------------------------------------------


def _resolve_params(
    profile: Profile, param_kwargs: dict[str, Parameter]
) -> dict[str, Parameter]:
    """Resolve parameter tokens for a profile.

    Named tokens are extracted from *param_kwargs*.  Any missing tokens
    are filled with a fresh :class:`FWHM` or :class:`Param` instance
    whose prior comes from
    :meth:`~unite.profile.base.Profile.default_priors`.

    Parameters
    ----------
    profile : Profile
        The resolved profile object.
    param_kwargs : dict
        Named parameter tokens extracted from ``**kwargs``.

    Returns
    -------
    dict of str to Parameter
        Mapping from profile parameter name to token.

    Raises
    ------
    ValueError
        If an unrecognised parameter name is passed.
    TypeError
        If a token has the wrong type for its slot (e.g. a
        :class:`Redshift` passed as ``fwhm_gauss``).
    """
    pnames = profile.param_names()
    pname = type(profile).__name__

    # Detect unrecognised keyword arguments before building defaults
    unexpected = sorted(set(param_kwargs) - set(pnames))
    if unexpected:
        msg = (
            f'Unexpected parameter keyword argument(s): {unexpected}. '
            f'{pname} expects: {list(pnames)}.'
        )
        raise ValueError(msg)

    # Validate types and fill defaults
    defaults = profile.default_priors()
    result: dict[str, Parameter] = {}
    for pn in pnames:
        if pn in param_kwargs:
            tok = param_kwargs[pn]
            expected = _param_class_for(pn)
            if not isinstance(tok, expected):
                msg = (
                    f"Parameter '{pn}' must be a {expected.__name__}, "
                    f'got {type(tok).__name__}.'
                )
                raise TypeError(msg)
            result[pn] = tok
        else:
            result[pn] = _param_class_for(pn)(prior=defaults[pn])
    return result


# ------------------------------------------------------------------
# ConfigMatrices — precomputed parameter-to-line mappings
# ------------------------------------------------------------------


@dataclass
class ConfigMatrices:
    """Precomputed indicator matrices mapping unique parameters to per-line arrays.

    Each matrix has shape ``(n_unique_params, n_lines)``.  A ``1`` at
    position ``[i, j]`` means unique parameter ``i`` contributes to line
    ``j``.  Matrix-vector products turn a stacked vector of sampled values
    into a per-line array, replacing a Python loop at JIT trace time.

    Profile parameters are split into three *slots* matching the positional
    arguments of ``_integrate_single_line``:

    * **Slot 0** (``p0``): primary velocity FWHM (always ``fwhm_*``, converted
      to wavelength before vmap).
    * **Slot 1** (``p1``): secondary parameter — either a velocity FWHM
      (``fwhm_lorentz`` / ``fwhm_exp``, stored in ``p1v_*``) or a dimensionless
      shape parameter (``h3``, stored in ``p1d_*``).  The two sub-matrices are
      summed after appropriate unit conversion:
      ``p1 = p1v_kms * centers / C + p1d``.
    * **Slot 2** (``p2``): tertiary dimensionless parameter (``h4``), pass-through.
    """

    #: Rest-frame line wavelengths (raw floats; units must match spectra). Shape ``(n_lines,)``.
    wavelengths: jnp.ndarray
    #: Multiplet relative flux strengths. Shape ``(n_lines,)``.
    strengths: jnp.ndarray
    #: Integer profile codes for ``lax.switch`` dispatch. Shape ``(n_lines,)``.
    profile_codes: jnp.ndarray

    #: NumPyro site names for unique flux parameters (rows of ``flux_matrix``).
    flux_names: list[str]
    #: Indicator matrix mapping flux parameters to lines. Shape ``(n_flux, n_lines)``.
    flux_matrix: jnp.ndarray

    #: NumPyro site names for unique redshift parameters.
    z_names: list[str]
    #: Indicator matrix mapping redshift parameters to lines. Shape ``(n_z, n_lines)``.
    z_matrix: jnp.ndarray

    #: NumPyro site names for slot-0 (primary FWHM) parameters.
    p0_names: list[str]
    #: Indicator matrix for slot-0 parameters. Shape ``(n_p0, n_lines)``.
    p0_matrix: jnp.ndarray

    #: NumPyro site names for velocity FWHM parameters in slot 1.
    p1v_names: list[str]
    #: Indicator matrix for slot-1 velocity parameters. Shape ``(n_p1v, n_lines)``.
    p1v_matrix: jnp.ndarray

    #: NumPyro site names for dimensionless shape parameters in slot 1.
    p1d_names: list[str]
    #: Indicator matrix for slot-1 dimensionless parameters. Shape ``(n_p1d, n_lines)``.
    p1d_matrix: jnp.ndarray

    #: NumPyro site names for slot-2 (tertiary dimensionless) parameters.
    p2_names: list[str]
    #: Indicator matrix for slot-2 parameters. Shape ``(n_p2, n_lines)``.
    p2_matrix: jnp.ndarray

    priors: dict[str, Prior]


def _token_matrix(
    entries: list[_LineEntry], get_tok, n_lines: int
) -> tuple[list[str], jnp.ndarray]:
    """Build a ``(n_unique, n_lines)`` indicator matrix for a token accessor.

    Parameters
    ----------
    entries : list of _LineEntry
    get_tok : callable
        Extracts the token from a ``_LineEntry`` (e.g. ``lambda e: e.flux``).
    n_lines : int

    Returns
    -------
    names : list of str
        Numpyro site names in row order.
    matrix : jnp.ndarray, shape (n_unique, n_lines)
    """
    unique: list = []
    seen: dict[int, int] = {}
    for entry in entries:
        tok = get_tok(entry)
        if id(tok) not in seen:
            seen[id(tok)] = len(unique)
            unique.append(tok)
    mat = np.zeros((len(unique), n_lines), dtype=float)
    for j, entry in enumerate(entries):
        mat[seen[id(get_tok(entry))], j] = 1.0
    return [t.name for t in unique], jnp.array(mat)


def _slot_matrix(
    tok_pairs: list[tuple[int, object]], n_lines: int
) -> tuple[list[str], jnp.ndarray]:
    """Build an indicator matrix from ``(line_idx, token)`` pairs for one slot.

    Parameters
    ----------
    tok_pairs : list of (int, token)
        Each pair is ``(line_index, parameter_token)`` for lines that use this slot.
    n_lines : int

    Returns
    -------
    names : list of str
    matrix : jnp.ndarray, shape (n_unique, n_lines)
        Empty ``(0, n_lines)`` when *tok_pairs* is empty.
    """
    if not tok_pairs:
        return [], jnp.zeros((0, n_lines))
    unique: list = []
    seen: dict[int, int] = {}
    for _, tok in tok_pairs:
        if id(tok) not in seen:
            seen[id(tok)] = len(unique)
            unique.append(tok)
    mat = np.zeros((len(unique), n_lines), dtype=float)
    for j, tok in tok_pairs:
        mat[seen[id(tok)], j] = 1.0
    return [t.name for t in unique], jnp.array(mat)


# ------------------------------------------------------------------
# Internal entry
# ------------------------------------------------------------------


@dataclass
class _LineEntry:
    """Internal record for a single spectral line."""

    name: str
    wavelength: u.Quantity
    profile: Profile
    redshift: Redshift
    fwhms: dict[str, Parameter]  # profile param name -> parameter token
    flux: Flux
    strength: int | float = 1.0


# ------------------------------------------------------------------
# LineConfiguration
# ------------------------------------------------------------------


class LineConfiguration:
    """Configuration of spectral lines for spectral fitting.

    All lines are added via :meth:`add_line`.  Pass a scalar *center*
    for a single line, or a sequence of wavelengths for a multiplet
    (lines sharing one flux parameter with relative strengths).

    Shared kinematics are expressed by passing the same
    :class:`Redshift` or :class:`FWHM` instance to multiple calls.

    Examples
    --------
    >>> z = Redshift('nlr', prior=Uniform(0, 0.5))
    >>> fwhm = FWHM('nlr', prior=Uniform(0, 1000))
    >>> config = LineConfiguration()
    >>> config.add_line('Ha', 6564.61, z, fwhm=fwhm)
    >>> config.add_line('[NII]', [6585.27, 6549.86], z,
    ...     fwhm=fwhm, strength=[2.95, 1.0])
    """

    def __init__(self) -> None:
        self._entries: list[_LineEntry] = []
        # Token registry: id(token) -> short name (populated by add_line)
        self._token_registry: dict[int, str] = {}
        # Names already claimed per category, to detect clashes
        self._used_names: dict[str, set[str]] = {}
        # Per-category auto-name counter
        self._counters: dict[str, count] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_line(
        self,
        name: str,
        center: u.Quantity,
        *,
        profile: str | Profile = 'gaussian',
        redshift: Redshift | None = None,
        flux: Flux | None = None,
        strength: int | float = 1.0,
        **param_kwargs: Parameter | None,
    ) -> None:
        """Add one spectral line.

        Parameters
        ----------
        name : str
            Identifier for the line (e.g. ``'Ha'``, ``'[OIII]5007'``).
        center : float, Quantity, or sequence thereof
            Rest-frame wavelength(s).
        redshift : Redshift, optional
            Redshift parameter token.  A fresh ``Redshift()`` is created
            if not provided.
        flux : Flux, optional
            Flux parameter token.  A fresh ``Flux()`` is created if not
            provided.
        strength : float or sequence of float, optional
            Relative flux strength.  For multiplets, broadcast to match
            the number of centers.  Default ``1.0``.
        profile : str or Profile, optional
            Line profile.  Default ``'gaussian'``.
        **param_kwargs : Parameter
            Named parameter tokens keyed by the profile's
            :meth:`~unite.profile.base.Profile.param_names`.  Missing
            parameters are filled from
            :meth:`~unite.profile.base.Profile.default_priors`.
            Examples: ``fwhm_gauss=FWHM('narrow')``,
            ``h3=Param('narrow_h3', prior=Uniform(-0.5, 0.5))``.
        """
        # --- Validate wavelength and resolve all tokens first ---
        wl = _ensure_wavelength(center)

        if redshift is None:
            redshift = Redshift()
        elif not isinstance(redshift, Redshift):
            msg = f'redshift must be a Redshift, got {type(redshift).__name__}.'
            raise TypeError(msg)

        if flux is None:
            flux = Flux()
        elif not isinstance(flux, Flux):
            msg = f'flux must be a Flux, got {type(flux).__name__}.'
            raise TypeError(msg)

        prof = resolve_profile(profile)
        fwhms = _resolve_params(prof, param_kwargs)

        # --- Conflict check: (name, wavelength, redshift, fwhm) must be unique.
        # Same name + wavelength is allowed when redshift or fwhm differs
        # (e.g. narrow vs broad component of the same line). ---
        fwhm_ids = {k: id(v) for k, v in fwhms.items()}
        for entry in self._entries:
            if (
                entry.name == name
                and entry.wavelength == wl
                and entry.redshift is redshift
                and {k: id(v) for k, v in entry.fwhms.items()} == fwhm_ids
            ):
                msg = (
                    f'An identical line already exists: {name!r} at {wl} '
                    f'with the same redshift and fwhm tokens. '
                    f'Each (name, wavelength, redshift, fwhm) combination must be unique.'
                )
                raise ValueError(msg)

        # --- Register / auto-name all kinematic tokens ---
        # Auto-names follow the pattern "{line_name}-{wavelength}-{param}", e.g.
        # "Ha-6564-z", "Ha-6564-fwhm_gauss", "Ha-6564-flux".  When the hint is
        # already taken (two unnamed tokens of the same type on the same
        # line), the fallback appends an alphabet suffix: "Ha-6564-z_a", "_b", …
        self._register_token(redshift, 'redshift', hint=f'{name}-{wl.value}-z')
        for wn, w_obj in fwhms.items():
            self._register_token(w_obj, wn, hint=f'{name}-{wl.value}-{wn}')
        self._register_token(flux, 'flux', hint=f'{name}-{wl.value}-flux')

        self._entries.append(
            _LineEntry(
                name=name,
                wavelength=wl,
                profile=prof,
                redshift=redshift,
                fwhms=fwhms,
                flux=flux,
                strength=strength,
            )
        )

    def add_lines(
        self,
        name: str,
        centers: u.Quantity,
        *,
        profile: str | Profile = 'gaussian',
        redshift: Redshift | Sequence[Redshift | None] | None = None,
        flux: Flux | Sequence[Flux | None] | None = None,
        strength: int | float | Sequence[int | float] = 1.0,
        **param_kwargs,
    ) -> None:
        """Add multiple lines sharing the same name and profile.

        Each entry in *centers* becomes one line.  All other arguments
        may be supplied as a single value (broadcast to every line) or
        as a sequence of the same length as *centers*.

        Parameters
        ----------
        name : str
            Shared identifier for all lines.
        centers : Quantity
            Rest-frame wavelengths.  Must be non-empty.
        profile : str or Profile, optional
            Line profile shared by all lines.  Default ``'gaussian'``.
        redshift : Redshift or sequence thereof, optional
            Redshift token(s).  A single value is shared across all
            lines; a sequence assigns one token per line.
        flux : Flux or sequence thereof, optional
            Flux token(s).  Same broadcasting rule as *redshift*.
        strength : float or sequence of float, optional
            Relative flux strengths.  Default ``1.0``.
        **param_kwargs : Parameter or sequence of Parameter
            Profile parameter tokens.  Each value follows the same
            broadcasting rule.

        Raises
        ------
        ValueError
            If *centers* is empty, or if any sequence argument has a
            length other than 1 or ``len(centers)``.
        TypeError
            If any token has the wrong type for its slot.
        """
        n = len(centers)
        if n == 0:
            raise ValueError("'centers' must be non-empty.")

        redshifts = _broadcast(redshift, 'redshift', n)
        fluxes = _broadcast(flux, 'flux', n)
        strengths = _broadcast(strength, 'strength', n)
        broadcasted_kwargs = {k: _broadcast(v, k, n) for k, v in param_kwargs.items()}

        for i, center in enumerate(centers):
            kw = {k: v[i] for k, v in broadcasted_kwargs.items()}
            self.add_line(
                name,
                center,
                profile=profile,
                redshift=redshifts[i],
                flux=fluxes[i],
                strength=strengths[i],
                **kw,
            )

    def _register_token(
        self, token: Parameter, category: str, hint: str | None = None
    ) -> None:
        """Register a kinematic token, auto-naming it if it has no name yet.

        Sets ``token.name`` in-place when unnamed.  Unnamed tokens use *hint*
        as their name; when *hint* is already taken in the category, appends
        ``_a``, ``_b``, … until a unique name is found.
        Raises :exc:`ValueError` if a user-supplied name clashes with a name
        already registered in the same *category*.

        Parameters
        ----------
        token : Parameter
            The token to register.
        category : str
            Grouping key for duplicate detection (e.g. ``'z'``,
            ``'fwhm_gauss'``, ``'h3'``, ``'flux'``).
        hint : str, optional
            Preferred name to try first when the token is unnamed.
        """
        tid = id(token)
        if tid in self._token_registry:
            return  # already registered; name already assigned

        used = self._used_names.setdefault(category, set())

        if token.name is not None:
            name = token.name
            if name in used:
                msg = (
                    f'Duplicate {category} parameter name {name!r}. '
                    f'Each token in the same category must have a unique name.'
                )
                raise ValueError(msg)
        else:
            # Try the hint first; when taken, append _a, _b, _c, …
            if hint is not None and hint not in used:
                name = hint
            else:
                base = hint if hint is not None else category
                ctr = self._counters.setdefault(base, count())
                while True:
                    candidate = f'{base}_{_alpha_name(next(ctr))}'
                    if candidate not in used:
                        name = candidate
                        break
            token.name = name

        used.add(name)
        self._token_registry[tid] = name

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------

    def build_matrices(self) -> ConfigMatrices:
        """Build precomputed indicator matrices from this configuration.

        Returns
        -------
        ConfigMatrices
            Parameter-to-line mapping matrices and line metadata.
        """
        """Construct :class:`ConfigMatrices` from a list of line entries.

        Parameters
        ----------
        entries : list of _LineEntry

        Returns
        -------
        ConfigMatrices
        """
        n = len(self)

        wavelengths = jnp.array([float(e.wavelength.value) for e in self])
        strengths = jnp.array([float(e.strength) for e in self])
        profile_codes = jnp.array([e.profile.code for e in self], dtype=int)

        flux_names, flux_matrix = _token_matrix(self._entries, lambda e: e.flux, n)
        z_names, z_matrix = _token_matrix(self._entries, lambda e: e.redshift, n)

        # Assign profile params to slots based on each profile's param_names() order.
        # Slot 0: first param (always a velocity FWHM, name starts with 'fwhm').
        # Slot 1: second param — velocity FWHM (→ p1v) or dimensionless (→ p1d).
        # Slot 2: third param (always dimensionless, only h4 currently).
        p0_pairs: list[tuple[int, object]] = []
        p1v_pairs: list[tuple[int, object]] = []
        p1d_pairs: list[tuple[int, object]] = []
        p2_pairs: list[tuple[int, object]] = []

        for j, entry in enumerate(self):
            pnames = entry.profile.param_names()
            if len(pnames) >= 1:
                p0_pairs.append((j, entry.fwhms[pnames[0]]))
            if len(pnames) >= 2:
                pn1, tok1 = pnames[1], entry.fwhms[pnames[1]]
                (p1v_pairs if pn1.startswith('fwhm') else p1d_pairs).append((j, tok1))
            if len(pnames) >= 3:
                p2_pairs.append((j, entry.fwhms[pnames[2]]))

        p0_names, p0_matrix = _slot_matrix(p0_pairs, n)
        p1v_names, p1v_matrix = _slot_matrix(p1v_pairs, n)
        p1d_names, p1d_matrix = _slot_matrix(p1d_pairs, n)
        p2_names, p2_matrix = _slot_matrix(p2_pairs, n)

        # Collect priors from all unique tokens.
        priors: dict[str, Prior] = {}
        seen_ids: set[int] = set()
        for entry in self:
            for tok in (entry.flux, entry.redshift, *entry.fwhms.values()):
                if id(tok) not in seen_ids:
                    seen_ids.add(id(tok))
                    priors[tok.name] = tok.prior

        return ConfigMatrices(
            wavelengths=wavelengths,
            strengths=strengths,
            profile_codes=profile_codes,
            flux_names=flux_names,
            flux_matrix=flux_matrix,
            z_names=z_names,
            z_matrix=z_matrix,
            p0_names=p0_names,
            p0_matrix=p0_matrix,
            p1v_names=p1v_names,
            p1v_matrix=p1v_matrix,
            p1d_names=p1d_names,
            p1d_matrix=p1d_matrix,
            p2_names=p2_names,
            p2_matrix=p2_matrix,
            priors=priors,
        )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a YAML-safe dictionary.

        Parameters are grouped into named sections by kind (``'redshift'``,
        ``'fwhm_gauss'``, ``'flux'``, etc.).  Within each section the dict
        key is simply the parameter's name, so there is no redundant prefix.
        ``ParameterRef`` bounds may only reference parameters in the same
        section; cross-kind references raise :exc:`TypeError`.

        Returns
        -------
        dict
            Ordered keys are the section names followed by ``'lines'``.
        """
        # Collect unique tokens per section in first-appearance order.
        seen_ids: set[int] = set()
        sections: dict[str, list[tuple[str, Parameter]]] = {}

        def _collect(tok: Parameter, section: str) -> None:
            tid = id(tok)
            if tid not in seen_ids:
                seen_ids.add(tid)
                sections.setdefault(section, []).append((tok.name, tok))

        for entry in self._entries:
            _collect(entry.redshift, 'redshift')
            for pn, w_obj in entry.fwhms.items():
                _collect(w_obj, pn)
            _collect(entry.flux, 'flux')

        # Create a global param_namer that includes all tokens from all sections
        # This is needed for priors that reference parameters from other sections
        global_param_namer: dict[object, str] = {}
        for toks in sections.values():
            for name, tok in toks:
                global_param_namer[tok] = name

        result: dict = {}
        for sec, toks in sections.items():
            result[sec] = {
                name: {'prior': tok.prior.to_dict(global_param_namer)}
                for name, tok in toks
            }

        lines = []
        for entry in self._entries:
            item: dict = {
                'name': entry.name,
                'wavelength': float(entry.wavelength.value),
                'wavelength_unit': entry.wavelength.unit.to_string(),
                'redshift': entry.redshift.name,
                'params': {pn: tok.name for pn, tok in entry.fwhms.items()},
                'flux': entry.flux.name,
            }
            item['profile'] = entry.profile.to_dict()
            if entry.strength != 1.0:
                item['strength'] = entry.strength
            lines.append(item)

        result['lines'] = lines
        return result

    @property
    def wavelengths(self) -> list[u.Quantity]:
        """Rest-frame wavelengths as a list of Quantity values (one per line)."""
        return [e.wavelength for e in self._entries]

    @property
    def centers(self) -> u.Quantity:
        """Rest-frame wavelengths of all lines in this configuration."""
        return u.Quantity([e.wavelength for e in self._entries])

    def _filter(self, mask: list[bool]) -> LineConfiguration:
        """Return a new configuration keeping only entries where *mask* is True.

        Parameters
        ----------
        mask : list of bool
            Boolean mask with one entry per line.

        Returns
        -------
        LineConfiguration
            Filtered copy (tokens are shared, not duplicated).
        """
        new = LineConfiguration()
        for keep, entry in zip(mask, self._entries, strict=True):
            if keep:
                new.add_line(
                    entry.name,
                    entry.wavelength,
                    profile=entry.profile,
                    redshift=entry.redshift,
                    flux=entry.flux,
                    strength=entry.strength,
                    **entry.fwhms,
                )
        return new

    @classmethod
    def from_dict(cls, d: dict) -> LineConfiguration:
        """Deserialize from a dictionary.

        Parameters
        ----------
        d : dict
            As produced by :meth:`to_dict`.

        Returns
        -------
        LineConfiguration
        """

        def _class_for_section(section: str) -> type[Parameter]:
            if section == 'redshift':
                return Redshift
            if section == 'flux':
                return Flux
            if section.startswith('fwhm'):
                return FWHM
            return Param

        # Pass 1 — create token objects (name comes from the dict key).
        # All tokens must exist before priors are parsed because priors may
        # contain ParameterRefs that point to other tokens.
        section_tokens: dict[str, dict[str, Parameter]] = {}
        for section, params in d.items():
            if section == 'lines':
                continue
            klass = _class_for_section(section)
            section_tokens[section] = {name: klass(name=name) for name in params}

        # Pass 2 — assign deserialized priors.
        # ParameterRefs can reference parameters from any section, so we need a global registry.
        global_token_registry: dict[str, Parameter] = {}
        for tokens in section_tokens.values():
            global_token_registry.update(tokens)

        for section, params in d.items():
            if section == 'lines':
                continue
            for name, pdata in params.items():
                section_tokens[section][name].prior = prior_from_dict(
                    pdata['prior'], global_token_registry
                )

        config = cls()
        for line_data in d['lines']:
            profile = profile_from_dict(line_data['profile'])
            param_kwargs = {
                pn: section_tokens[pn][tok_name]
                for pn, tok_name in line_data['params'].items()
            }
            config.add_line(
                line_data['name'],
                line_data['wavelength'] * u.Unit(line_data['wavelength_unit']),
                profile=profile,
                redshift=section_tokens['redshift'][line_data['redshift']],
                flux=section_tokens['flux'][line_data['flux']],
                strength=line_data.get('strength', 1.0),
                **param_kwargs,
            )

        return config

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __repr__(self) -> str:
        if not self._entries:
            return 'LineConfiguration: empty'

        n_flux = len({id(e.flux) for e in self._entries})
        n_z = len({id(e.redshift) for e in self._entries})
        n_params = sum(
            len({id(e.fwhms.get(wn)) for e in self._entries if wn in e.fwhms})
            for wn in {wn for e in self._entries for wn in e.fwhms}
        )

        header = (
            f'LineConfiguration: {len(self._entries)} lines, '
            f'{n_flux} flux / {n_z} z / {n_params} profile params'
        )

        # --- Line table ---
        rows: list[tuple[str, str, str, str, str, str, str]] = []
        for entry in self._entries:
            prof_name = type(entry.profile).__name__
            # Tokens are always named after add_line registration
            fwhm_display = ', '.join(f.name for f in entry.fwhms.values())

            rows.append(
                (
                    entry.name,
                    f'{entry.wavelength:.2f}',
                    prof_name,
                    entry.redshift.name,
                    fwhm_display,
                    entry.flux.name,
                    f'{entry.strength:.2f}',
                )
            )

        headers = (
            'Name',
            'Wavelength',
            'Profile',
            'Redshift',
            'Params',
            'Flux',
            'Strength',
        )
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        fmt = '  '.join(f'{{:<{w}}}' for w in widths)
        sep = '  '.join('-' * w for w in widths)
        lines_out = [header, '', '  ' + fmt.format(*headers), '  ' + sep]
        for row in rows:
            lines_out.append('  ' + fmt.format(*row))

        # --- Parameter sections ---
        # Collect unique tokens in first-appearance order
        seen_z: set[int] = set()
        z_params: list[tuple[str, Prior]] = []
        for entry in self._entries:
            zid = id(entry.redshift)
            if zid not in seen_z:
                seen_z.add(zid)
                z_params.append((entry.redshift.name, entry.redshift.prior))

        seen_fwhm: dict[str, set[int]] = {}
        fwhm_key_order: list[str] = []
        fwhm_params: dict[str, list[tuple[str, Prior]]] = {}
        for entry in self._entries:
            for wn, w_obj in entry.fwhms.items():
                if wn not in seen_fwhm:
                    seen_fwhm[wn] = set()
                    fwhm_key_order.append(wn)
                    fwhm_params[wn] = []
                wid = id(w_obj)
                if wid not in seen_fwhm[wn]:
                    seen_fwhm[wn].add(wid)
                    fwhm_params[wn].append((w_obj.name, w_obj.prior))

        seen_flux: set[int] = set()
        flux_params: list[tuple[str, Prior]] = []
        for entry in self._entries:
            fid = id(entry.flux)
            if fid not in seen_flux:
                seen_flux.add(fid)
                flux_params.append((entry.flux.name, entry.flux.prior))

        def _fmt_section(title: str, params: list[tuple[str, Prior]]) -> list[str]:
            name_w = max(len(p[0]) for p in params)
            out = ['', f'  {title}:']
            for pname, prior in params:
                out.append(f'    {pname:<{name_w}}  {prior!r}')
            return out

        lines_out += _fmt_section('Redshift', z_params)

        for wn in fwhm_key_order:
            lines_out += _fmt_section(f'Params ({wn})', fwhm_params[wn])

        lines_out += _fmt_section('Flux', flux_params)

        return '\n'.join(lines_out)
