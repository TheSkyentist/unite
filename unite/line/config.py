"""Emission line configuration for spectral fitting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np
import yaml
from astropy import units as u

from unite._utils import _alpha_name, _broadcast, _ensure_wavelength
from unite.line.library import Profile, profile_from_dict, resolve_profile
from unite.prior import Parameter, Prior, TruncatedNormal, Uniform, prior_from_dict

# ------------------------------------------------------------------
# Parameter token classes
# ------------------------------------------------------------------

# Maps category name → type prefix for NumPyro site names.
# Categories not in this map use the category name itself as prefix.
_CATEGORY_PREFIX: dict[str, str] = {'redshift': 'z', 'flux': 'flux', 'tau': 'tau'}


def _prefix_for(category: str) -> str:
    """Return the NumPyro site-name prefix for a given category."""
    return _CATEGORY_PREFIX.get(category, category)


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


class LineShape(Parameter):
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
            prior = TruncatedNormal(low=0.3, high=0.3, loc=0, scale=0.1)
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


class Tau(Parameter):
    """
    A named optical depth parameter for absorption lines.

    Represents the absorption depth at the line center.  The normalised
    profile shape (from the absorption profile class) is multiplied by
    this value to obtain the wavelength-dependent optical depth
    ``tau(lam) = tau * phi(lam)``, and the resulting transmission is
    ``exp(-τ(λ))``.

    Default prior is uniform between 0 and 10.
    """

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Uniform(0, 10)
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
        :class:`FWHM` for names starting with ``'fwhm'``, :class:`LineShape`
        otherwise.
    """
    return FWHM if pn.startswith('fwhm') else LineShape


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
        msg = f'Unexpected parameter keyword argument(s): {unexpected}. {pname} expects: {list(pnames)}.'
        raise ValueError(msg)

    # Validate types and fill defaults
    defaults = profile.default_priors()
    result: dict[str, Parameter] = {}
    for pn in pnames:
        if pn in param_kwargs:
            tok = param_kwargs[pn]
            expected = _param_class_for(pn)
            if not isinstance(tok, expected):
                msg = f"Parameter '{pn}' must be a {expected.__name__}, got {type(tok).__name__}."
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

    #: NumPyro site names for unique tau (optical depth) parameters.
    tau_names: list[str]
    #: Indicator matrix mapping tau parameters to lines. Shape ``(n_tau, n_lines)``.
    tau_matrix: jnp.ndarray
    #: Boolean mask indicating which lines are absorption lines. Shape ``(n_lines,)``.
    is_absorption: jnp.ndarray

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
    return [cast(str, t.name) for t in unique], jnp.array(mat)


def _slot_matrix(
    tok_pairs: Sequence[tuple[int, object]], n_lines: int
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
    return [cast(str, t.name) for t in unique], jnp.array(mat)


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
    flux: Flux | None = None  # set for emission lines
    tau: Tau | None = None  # set for absorption lines
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
        # Line names already used (must be unique)
        self._used_line_names: set[str] = set()

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
        tau: Tau | None = None,
        strength: int | float = 1.0,
        **param_kwargs: Parameter | None,
    ) -> None:
        r"""Add one spectral line.

        Parameters
        ----------
        name : str
            Unique identifier for the line (e.g. ``'Ha'``, ``'[OIII]5007'``,
            ``'Ha_narrow'``).  Line names must be unique within a
            :class:`LineConfiguration`; adding a second line with the same
            name raises :exc:`ValueError`.
        center : float, Quantity, or sequence thereof
            Rest-frame wavelength(s).
        redshift : Redshift, optional
            Redshift parameter token.  A fresh ``Redshift()`` is created
            if not provided.
        flux : Flux, optional
            Flux parameter token for emission lines.  A fresh ``Flux()``
            is created if not provided.  Cannot be used with absorption
            profiles.
        tau : Tau, optional
            Optical depth parameter token for absorption lines.  A fresh
            ``Tau()`` is created if not provided.  Cannot be used with
            emission profiles.
        strength : float or sequence of float, optional
            Relative flux strength.  For multiplets, broadcast to match
            the number of centers.  Default ``1.0``.
        profile : str or Profile, optional
            Line profile.  Default ``'gaussian'``.
        \*\*param_kwargs : Parameter
            Named parameter tokens keyed by the profile's
            :meth:`~unite.profile.base.Profile.param_names`.  Missing
            parameters are filled from
            :meth:`~unite.profile.base.Profile.default_priors`.
            Examples: ``fwhm_gauss=FWHM('narrow')``,
            ``h3=Param('narrow_h3', prior=Uniform(-0.5, 0.5))``.

        Raises
        ------
        ValueError
            If *name* is already used by another line in this configuration.
        TypeError
            If *flux* is passed for an absorption profile, or *tau* is
            passed for an emission profile.
        """
        # --- Enforce unique line names ---
        if name in self._used_line_names:
            msg = (
                f'Line name {name!r} is already used. Each line must have a unique '
                f'name. Use distinct names such as {name!r}_narrow / {name!r}_broad '
                f'for multiple kinematic components.'
            )
            raise ValueError(msg)

        # --- Validate wavelength and resolve all tokens first ---
        wl = _ensure_wavelength(center, 'center', ndim=0)

        if redshift is None:
            redshift = Redshift()
        elif not isinstance(redshift, Redshift):
            msg = f'redshift must be a Redshift, got {type(redshift).__name__}.'
            raise TypeError(msg)

        prof = resolve_profile(profile)
        # Filter out None values from param_kwargs (broadcast may produce None).
        filtered_kwargs: dict[str, Parameter] = {
            k: v for k, v in param_kwargs.items() if v is not None
        }
        fwhms = _resolve_params(prof, filtered_kwargs)

        # --- Emission vs absorption: determined by flux/tau tokens ---
        if flux is not None and tau is not None:
            msg = (
                'Cannot specify both flux and tau for the same line. '
                'Use flux for emission lines or tau for absorption lines.'
            )
            raise TypeError(msg)
        if tau is not None:
            # Absorption line
            if not isinstance(tau, Tau):
                msg = f'tau must be a Tau, got {type(tau).__name__}.'
                raise TypeError(msg)
            flux_tok = None
            tau_tok = tau
        else:
            # Emission line (default)
            if flux is None:
                flux = Flux()
            elif not isinstance(flux, Flux):
                msg = f'flux must be a Flux, got {type(flux).__name__}.'
                raise TypeError(msg)
            flux_tok = flux
            tau_tok = None

        # --- Register / auto-name all kinematic tokens ---
        self._register_token(redshift, 'redshift', hint_label=name)
        for wn, w_obj in fwhms.items():
            self._register_token(w_obj, wn, hint_label=name)
        if flux_tok is not None:
            self._register_token(flux_tok, 'flux', hint_label=name)
        if tau_tok is not None:
            self._register_token(tau_tok, 'tau', hint_label=name)

        self._used_line_names.add(name)
        self._entries.append(
            _LineEntry(
                name=name,
                wavelength=wl,
                profile=prof,
                redshift=redshift,
                fwhms=fwhms,
                flux=flux_tok,
                tau=tau_tok,
                strength=strength,
            )
        )

    def add_lines(
        self,
        name: str | Sequence[str],
        centers: u.Quantity,
        *,
        profile: str | Profile = 'gaussian',
        redshift: Redshift | Sequence[Redshift | None] | None = None,
        flux: Flux | Sequence[Flux | None] | None = None,
        tau: Tau | Sequence[Tau | None] | None = None,
        strength: int | float | Sequence[int | float] = 1.0,
        **param_kwargs,
    ) -> None:
        r"""Add multiple lines, each with an independent name.

        Each entry in *centers* becomes one line with a unique name.  When
        *name* is a single string, names are auto-generated as
        ``'{name}_{center.value:g}'`` (e.g. ``'NII_6585'``, ``'NII_6550'``).
        When *name* is a sequence, it must have the same length as *centers*.
        All other arguments may be supplied as a single value (broadcast to
        every line) or as a sequence of the same length as *centers*.

        Parameters
        ----------
        name : str | Sequence[str]
            Single name or sequence of names, one per center.
        centers : Quantity
            Rest-frame wavelengths.  Must be non-empty.
        profile : str or Profile, optional
            Line profile shared by all lines.  Default ``'gaussian'``.
        redshift : Redshift or sequence thereof, optional
            Redshift token(s).  A single value is shared across all
            lines; a sequence assigns one token per line.
        flux : Flux or sequence thereof, optional
            Flux token(s).  Same broadcasting rule as *redshift*.
        tau : Tau or sequence thereof, optional
            Optical depth token(s).  Same broadcasting rule as *flux*.
        strength : float or sequence of float, optional
            Relative flux strengths.  Default ``1.0``.
        \*\*param_kwargs : Parameter or sequence of Parameter
            Profile parameter tokens.  Each value follows the same
            broadcasting rule.

        Raises
        ------
        ValueError
            If *centers* is empty, if *name* sequence has the wrong length, or if
            any sequence argument has a length other than 1 or
            ``len(centers)``.
        TypeError
            If any token has the wrong type for its slot.
        """
        # Convert list of Quantities to a single Quantity array if needed
        centers = _ensure_wavelength(centers, 'centers', ndim=1)
        n = len(centers)
        if n == 0:
            raise ValueError("'centers' must be non-empty.")

        # Generate or validate line names
        if isinstance(name, str):
            names_seq = [f'{name}_{center.value:g}' for center in centers]
        else:
            if len(name) != n:
                raise ValueError(
                    f"'name' sequence has length {len(name)}, but 'centers' has length {n}."
                )
            names_seq = list(name)

        redshifts = _broadcast(redshift, 'redshift', n)
        fluxes = _broadcast(flux, 'flux', n)
        taus = _broadcast(tau, 'tau', n)
        strengths = _broadcast(strength, 'strength', n)
        broadcasted_kwargs = {k: _broadcast(v, k, n) for k, v in param_kwargs.items()}

        for i, center in enumerate(centers):
            line_name = names_seq[i]
            kw = {k: v[i] for k, v in broadcasted_kwargs.items()}
            self.add_line(
                line_name,
                center,
                profile=profile,
                redshift=redshifts[i],
                flux=fluxes[i],
                tau=taus[i],
                strength=strengths[i],
                **kw,
            )

    def _register_token(
        self, token: Parameter, category: str, hint_label: str | None = None
    ) -> None:
        """Register a parameter token, assigning a prefixed site name.

        Sets ``token.name`` (NumPyro site name) and ``token.label`` (human
        label) in-place when not yet set.  Detects name collisions within
        each category.

        Parameters
        ----------
        token : Parameter
            The token to register.
        category : str
            Grouping key (e.g. ``'redshift'``, ``'fwhm_gauss'``, ``'flux'``).
            Determines the type prefix used to build the site name.
        hint_label : str, optional
            Label hint for auto-naming (the line name).  Used when the token
            has no user-supplied label.
        """
        tid = id(token)
        if tid in self._token_registry:
            return  # already registered; name already assigned

        prefix = _prefix_for(category)
        used = self._used_names.setdefault(category, set())

        if token.name is not None:
            # Already has a finalized site name (e.g. re-used from _filter or merge).
            site_name = token.name
            if site_name in used:
                raise ValueError(
                    f'Duplicate {category} site name {site_name!r}. '
                    f'Each token in the same category must have a unique name.'
                )
        elif token.label is not None:
            # User supplied a label; prefix it to get the site name.
            site_name = f'{prefix}_{token.label}'
            if site_name in used:
                raise ValueError(
                    f'Duplicate {category} parameter name {token.label!r} '
                    f'(site name {site_name!r}). Each token in the same '
                    f'category must have a unique name.'
                )
            token.name = site_name
        else:
            # Auto-name from hint (use line name, not wavelength).
            base = f'{prefix}_{hint_label}' if hint_label else prefix
            if base not in used:
                site_name = base
            else:
                ctr = self._counters.setdefault(base, count())
                while True:
                    site_name = f'{base}_{_alpha_name(next(ctr))}'
                    if site_name not in used:
                        break
            token.name = site_name
            # Derive label from site name by stripping the prefix.
            token.label = (
                site_name[len(prefix) + 1 :]
                if site_name.startswith(prefix + '_')
                else site_name
            )

        used.add(site_name)
        self._token_registry[tid] = site_name

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

        # Flux matrix — only for emission lines (absorption lines have flux=None).
        flux_entries = [(j, e.flux) for j, e in enumerate(self) if e.flux is not None]
        flux_names, flux_matrix = (
            _slot_matrix(flux_entries, n) if flux_entries else ([], jnp.zeros((0, n)))
        )

        # Tau matrix — only for absorption lines (emission lines have tau=None).
        tau_entries = [(j, e.tau) for j, e in enumerate(self) if e.tau is not None]
        tau_names, tau_matrix = (
            _slot_matrix(tau_entries, n) if tau_entries else ([], jnp.zeros((0, n)))
        )

        is_absorption = jnp.array([e.tau is not None for e in self], dtype=bool)

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
            for tok in (entry.flux, entry.tau, entry.redshift, *entry.fwhms.values()):
                if tok is not None and id(tok) not in seen_ids:
                    seen_ids.add(id(tok))
                    priors[cast(str, tok.name)] = tok.prior

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
            tau_names=tau_names,
            tau_matrix=tau_matrix,
            is_absorption=is_absorption,
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
        Parameter expression bounds may only reference parameters in the same
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
                sections.setdefault(section, []).append((cast(str, tok.name), tok))

        for entry in self._entries:
            _collect(entry.redshift, 'redshift')
            for pn, w_obj in entry.fwhms.items():
                _collect(w_obj, pn)
            if entry.flux is not None:
                _collect(entry.flux, 'flux')
            if entry.tau is not None:
                _collect(entry.tau, 'tau')

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
                'wavelength_unit': cast(u.UnitBase, entry.wavelength.unit).to_string(),
                'redshift': cast(str, entry.redshift.name),
                'params': {pn: cast(str, tok.name) for pn, tok in entry.fwhms.items()},
            }
            if entry.flux is not None:
                item['flux'] = entry.flux.name
            if entry.tau is not None:
                item['tau'] = entry.tau.name
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
                    tau=entry.tau,
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
            if section == 'tau':
                return Tau
            if section.startswith('fwhm'):
                return FWHM
            return LineShape

        # Pass 1 — create token objects (name comes from the dict key).
        # All tokens must exist before priors are parsed because priors may
        # contain parameter expressions that point to other tokens.
        sec_prefix = {
            section: _prefix_for(section) for section in d if section != 'lines'
        }
        section_tokens: dict[str, dict[str, Parameter]] = {}
        for section, params in d.items():
            if section == 'lines':
                continue
            klass = _class_for_section(section)
            prefix = sec_prefix[section]
            tokens: dict[str, Parameter] = {}
            for site_name in params:
                tok = klass(
                    prior=Uniform(0, 1)
                )  # placeholder prior, overwritten in pass 2
                tok.name = site_name  # set site name directly
                # Derive label by stripping prefix
                if site_name.startswith(prefix + '_'):
                    tok.label = site_name[len(prefix) + 1 :]
                else:
                    tok.label = site_name
                tokens[site_name] = tok
            section_tokens[section] = tokens

        # Pass 2 — assign deserialized priors.
        # Parameter expressions can reference parameters from any section, so we need a global registry.
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
            flux_tok = (
                section_tokens['flux'][line_data['flux']]
                if 'flux' in line_data
                else None
            )
            tau_tok = (
                section_tokens['tau'][line_data['tau']] if 'tau' in line_data else None
            )
            config.add_line(
                line_data['name'],
                line_data['wavelength'] * u.Unit(line_data['wavelength_unit']),
                profile=profile,
                redshift=cast(
                    Redshift, section_tokens['redshift'][line_data['redshift']]
                ),
                flux=cast(Flux | None, flux_tok),
                tau=cast(Tau | None, tau_tok),
                strength=line_data.get('strength', 1.0),
                **param_kwargs,
            )

        return config

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
    def from_yaml(cls, text: str) -> LineConfiguration:
        """Deserialize from a YAML string.

        Parameters
        ----------
        text : str
            YAML string as produced by :meth:`to_yaml`.

        Returns
        -------
        LineConfiguration
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
    def load(cls, path: str | Path) -> LineConfiguration:
        """Load from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to a YAML file written by :meth:`save`.

        Returns
        -------
        LineConfiguration
        """
        return cls.from_yaml(Path(path).read_text())

    # ------------------------------------------------------------------
    # Merging
    # ------------------------------------------------------------------

    def merge(
        self, other: LineConfiguration, *, strict: bool = True
    ) -> LineConfiguration:
        """Merge another :class:`LineConfiguration` into a new one.

        Parameters
        ----------
        other : LineConfiguration
            Configuration to merge.
        strict : bool
            If ``True`` (default), raise :exc:`ValueError` on any token
            name collision between the two configs.  If ``False``,
            same-named tokens of the same type are treated as shared
            (the token from *self* is kept); same-named tokens of
            different types still raise.

        Returns
        -------
        LineConfiguration
            New configuration containing lines from both *self* and *other*.

        Raises
        ------
        ValueError
            On name collisions (strict mode) or type mismatches.
        """
        if not isinstance(other, LineConfiguration):
            return NotImplemented

        # Build a mapping of token name → token from self's entries.
        self_tokens: dict[str, Parameter] = {}
        for entry in self._entries:
            for tok in (entry.flux, entry.tau, entry.redshift, *entry.fwhms.values()):
                if (
                    tok is not None
                    and tok.name is not None
                    and tok.name not in self_tokens
                ):
                    self_tokens[tok.name] = tok

        # Build a mapping from other's token id → replacement token.
        other_remap: dict[int, Parameter] = {}
        for entry in other._entries:
            for tok in (entry.flux, entry.tau, entry.redshift, *entry.fwhms.values()):
                if tok is None:
                    continue
                tid = id(tok)
                if tid in other_remap:
                    continue
                if tok.name in self_tokens:
                    existing = self_tokens[tok.name]
                    if strict:
                        msg = (
                            f'Token name collision: {tok.name!r} exists in both '
                            f'configs. Use strict=False to merge same-typed tokens.'
                        )
                        raise ValueError(msg)
                    if type(existing) is not type(tok):
                        msg = (
                            f'Token name {tok.name!r} has type '
                            f'{type(existing).__name__} in self but '
                            f'{type(tok).__name__} in other.'
                        )
                        raise TypeError(msg)
                    other_remap[tid] = existing
                else:
                    other_remap[tid] = tok

        # Build new config: copy self's entries, then add other's with remapped tokens.
        merged = LineConfiguration()
        for entry in self._entries:
            merged.add_line(
                entry.name,
                entry.wavelength,
                profile=entry.profile,
                redshift=entry.redshift,
                flux=entry.flux,
                tau=entry.tau,
                strength=entry.strength,
                **entry.fwhms,
            )
        for entry in other._entries:
            remapped_z = cast(Redshift, other_remap[id(entry.redshift)])
            remapped_flux = cast(
                Flux | None,
                other_remap[id(entry.flux)] if entry.flux is not None else None,
            )
            remapped_tau = cast(
                Tau | None,
                other_remap[id(entry.tau)] if entry.tau is not None else None,
            )
            remapped_fwhms = {k: other_remap[id(v)] for k, v in entry.fwhms.items()}
            merged.add_line(
                entry.name,
                entry.wavelength,
                profile=entry.profile,
                redshift=remapped_z,
                flux=remapped_flux,
                tau=remapped_tau,
                strength=entry.strength,
                **remapped_fwhms,
            )
        return merged

    def __add__(self, other: LineConfiguration) -> LineConfiguration:
        """Merge two configs (strict mode — raises on name collisions).

        Parameters
        ----------
        other : LineConfiguration

        Returns
        -------
        LineConfiguration
        """
        if not isinstance(other, LineConfiguration):
            return NotImplemented
        return self.merge(other, strict=True)

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

        n_flux = len({id(e.flux) for e in self._entries if e.flux is not None})
        n_z = len({id(e.redshift) for e in self._entries})
        n_params = sum(
            len({id(e.fwhms.get(wn)) for e in self._entries if wn in e.fwhms})
            for wn in {wn for e in self._entries for wn in e.fwhms}
        )

        header = f'LineConfiguration: {len(self._entries)} lines, {n_flux} flux / {n_z} z / {n_params} profile params'

        # --- Line table ---
        rows: list[tuple[str, str, str, str, str, str, str]] = []
        for entry in self._entries:
            prof_name = type(entry.profile).__name__
            # Tokens are always named after add_line registration
            fwhm_display = ', '.join(cast(str, f.name) for f in entry.fwhms.values())

            flux_or_tau = (
                cast(str, entry.flux.name)
                if entry.flux is not None
                else cast(str, entry.tau.name)
                if entry.tau is not None
                else ''
            )

            rows.append(
                (
                    entry.name,
                    f'{entry.wavelength:.2f}',
                    prof_name,
                    cast(str, entry.redshift.name),
                    fwhm_display,
                    flux_or_tau,
                    f'{entry.strength:.2f}',
                )
            )

        headers = (
            'Name',
            'Wavelength',
            'Profile',
            'Redshift',
            'Params',
            'Flux/Tau',
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
                z_params.append((cast(str, entry.redshift.name), entry.redshift.prior))

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
                    fwhm_params[wn].append((cast(str, w_obj.name), w_obj.prior))

        seen_flux: set[int] = set()
        flux_params: list[tuple[str, Prior]] = []
        for entry in self._entries:
            if entry.flux is not None:
                fid = id(entry.flux)
                if fid not in seen_flux:
                    seen_flux.add(fid)
                    flux_params.append((cast(str, entry.flux.name), entry.flux.prior))

        seen_tau: set[int] = set()
        tau_params: list[tuple[str, Prior]] = []
        for entry in self._entries:
            if entry.tau is not None:
                tid = id(entry.tau)
                if tid not in seen_tau:
                    seen_tau.add(tid)
                    tau_params.append((cast(str, entry.tau.name), entry.tau.prior))

        def _fmt_section(title: str, params: list[tuple[str, Prior]]) -> list[str]:
            name_w = max(len(p[0]) for p in params)
            out = ['', f'  {title}:']
            for pname, prior in params:
                out.append(f'    {pname:<{name_w}}  {prior!r}')
            return out

        lines_out += _fmt_section('Redshift', z_params)

        for wn in fwhm_key_order:
            lines_out += _fmt_section(f'Params ({wn})', fwhm_params[wn])

        if flux_params:
            lines_out += _fmt_section('Flux', flux_params)

        if tau_params:
            lines_out += _fmt_section('Tau', tau_params)

        return '\n'.join(lines_out)
