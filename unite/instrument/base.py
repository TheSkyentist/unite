"""Abstract base class for dispersers and calibration parameter tokens."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

from astropy import units as u
from jax.typing import ArrayLike

from unite.prior import Fixed, Parameter, Prior

if TYPE_CHECKING:
    from unite.instrument.overlap import OrderOverlap

# ---------------------------------------------------------------------------
# CalibParam tokens — subclasses of Parameter with instrument-specific defaults
# ---------------------------------------------------------------------------


class CalibParam(Parameter):
    """Base class of the calibration tokens carried by a :class:`Disperser`.

    Subclasses set :attr:`site_prefix`, the type prefix of the numpyro site
    name (e.g. ``'r_scale'`` gives sites named ``'r_scale_<label>'``).
    """

    #: Type prefix of the numpyro site name.
    site_prefix: str = ''


class RScale(CalibParam):
    """Multiplicative scale on the disperser resolving power *R*.

    Nominal value is 1 (no correction).  The model applies
    ``R_eff(λ) = R_nominal(λ) * r_scale``.

    Parameters
    ----------
    name : str, optional
        Human-readable label.  When attached to a :class:`Disperser`, the
        site name is auto-derived as ``'r_scale_{disperser_name}'`` if not provided.
    prior : Prior, optional
        Prior on the R scale factor.  Defaults to ``Fixed(1.0)`` (fixed at
        nominal).
    """

    #: Type prefix of the numpyro site name (``'r_scale_<label>'``).
    site_prefix: str = 'r_scale'

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Fixed(1.0)
        super().__init__(name=name, prior=prior)


class FluxScale(CalibParam):
    """Multiplicative flux normalisation between dispersers.

    Nominal value is 1.  The model flux is multiplied by ``flux_scale`` to
    predict the observed flux (``F_obs = flux_scale * F_model``), allowing
    relative flux calibration across multiple gratings.

    Parameters
    ----------
    name : str, optional
        Human-readable label.  When attached to a :class:`Disperser`, the
        site name is auto-derived as ``'flux_scale_{disperser_name}'`` if not provided.
    prior : Prior, optional
        Prior on the flux scale factor.  Defaults to ``Fixed(1.0)`` (fixed at
        nominal).
    """

    #: Type prefix of the numpyro site name (``'flux_scale_<label>'``).
    site_prefix: str = 'flux_scale'

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Fixed(1.0)
        super().__init__(name=name, prior=prior)


class PixOffset(CalibParam):
    """Pixel displacement of the spectrum on the detector relative to the wavelength calibration.

    Nominal value is 0 (no displacement).  A positive value indicates the
    spectrum is displaced toward longer wavelengths on the detector: the model
    subtracts ``pix_offset * dlam_dpix`` from the calibrated pixel-edge
    wavelengths, shifting them blueward to compensate.

    For example, if a disperser consistently returns redshifts that are too
    high compared to a reference, the spectrum is displaced redward by some
    number of pixels — set ``pix_offset`` to that (positive) number to correct
    the wavelength solution.

    Parameters
    ----------
    name : str, optional
        Human-readable label.  When attached to a :class:`Disperser`, the
        site name is auto-derived as ``'pix_offset_{disperser_name}'`` if not provided.
    prior : Prior, optional
        Prior on the pixel offset.  Defaults to ``Fixed(0.0)`` (no displacement).
    """

    #: Type prefix of the numpyro site name (``'pix_offset_<label>'``).
    site_prefix: str = 'pix_offset'

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Fixed(0.0)
        super().__init__(name=name, prior=prior)


class OrderScale(CalibParam):
    """Multiplicative scale on a higher-order throughput ratio ``r_m``.

    Nominal value is 1.  The order-*m* overlap term of the model is
    ``OrderScale * r_m(λ) * F(λ / m)`` (same direction as
    :class:`FluxScale`: the forward model multiplies), so a free prior
    represents the uncertainty on the order-*m* calibration curve.  Attach it
    through :class:`~unite.instrument.overlap.OrderOverlap`.

    Sharing: the same instance on several overlaps (orders or dispersers)
    gives a single model parameter.

    Parameters
    ----------
    name : str, optional
        Human-readable label.  The site name is ``'order_scale_{label}'``; an
        unlabelled, unshared token on a named disperser is auto-named
        ``'order_scale_{disperser_name}_{order}'``.
    prior : Prior, optional
        Prior on the scale factor.  Defaults to ``Fixed(1.0)``.
    """

    #: Type prefix of the numpyro site name (``'order_scale_<label>'``).
    site_prefix: str = 'order_scale'

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Fixed(1.0)
        super().__init__(name=name, prior=prior)


# ---------------------------------------------------------------------------
# Disperser ABC
# ---------------------------------------------------------------------------


class Disperser(ABC):
    """Abstract base class for dispersive optical elements.

    A disperser maps wavelength coordinates to instrumental quantities such as
    resolving power and plate scale.  Every concrete subclass must implement
    :meth:`R` and :meth:`dlam_dpix`, and must carry a ``unit`` attribute that
    records the wavelength unit the disperser expects.

    Calibration is encoded by optional tokens (:class:`RScale`,
    :class:`FluxScale`, :class:`PixOffset`):

    * ``r_scale`` — multiplicative scale on resolving power (*R*).
    * ``flux_scale`` — multiplicative flux normalisation.
    * ``pix_offset`` — pixel displacement of the spectrum on the detector (positive = redward; wavelength grid is corrected blueward).

    Every slot always carries a token — leaving one ``None`` at construction
    time creates a private, unshared token with the default ``Fixed`` prior
    (nominal 1.0 for ``r_scale``/``flux_scale``, 0.0 for ``pix_offset``)
    rather than omitting the parameter. This keeps the calibration state
    fully transparent: it always shows up in ``str(disperser)``, in
    parameter tables, and in serialized YAML, even when it is fixed at its
    nominal value. Pass an explicit token (e.g. with a sampled prior, or the
    same instance shared across dispersers) to make the parameter free or
    shared.

    Sharing: two dispersers that reference the **same token instance** share
    a single parameter in the model (identity-based, like ``FWHM``/
    ``Redshift`` on lines).

    Parameters
    ----------
    unit : astropy.units.UnitBase
        The wavelength unit that this disperser operates in (e.g.
        ``u.Angstrom``, ``u.nm``, ``u.um``).  All wavelength values passed
        to :meth:`R` and :meth:`dlam_dpix` are assumed to be in this unit.
    name : str, optional
        Human-readable label (e.g. ``'G235H'``).  Used in repr and as the
        default ``Spectrum.name`` when this disperser is attached.
    r_scale : RScale, optional
        Token for the resolving-power scale.  ``None`` (default) creates a
        private token fixed at nominal (1.0).
    flux_scale : FluxScale, optional
        Token for the flux normalisation.  ``None`` (default) creates a
        private token fixed at nominal (1.0).
    pix_offset : PixOffset, optional
        Token for the detector pixel displacement.  ``None`` (default)
        creates a private token fixed at nominal (0.0).
    order_overlap : sequence of OrderOverlap, optional
        Higher-order grating overlap, one
        :class:`~unite.instrument.overlap.OrderOverlap` per order (no
        duplicate orders).  Adds ``OrderScale_m * r_m(λ) * F(λ / m)`` to the
        first-order model.  ``None`` (default) disables the term.  Dispersers
        for which the term is not physical (prisms, sub-octave multi-arm
        spectrographs; :attr:`supports_order_overlap` is ``False``) warn but
        accept it.

    Notes
    -----
    **Order overlap.**  The term describes order-*m* light of the *same
    source* landing on its own first-order spectrum, in a spectrum that was
    flux-calibrated with the first-order throughput only.  It applies to
    gratings that cover more than an octave behind their order-sorting filter
    (NIRSpec gratings, MUSE extended mode, optical long-slit/MOS set-ups).
    It does not apply to prisms, cross-dispersed echelles, sub-octave
    multi-arm spectrographs, or slitless grisms (where contamination comes
    from other sources).

    ``RScale`` and ``PixOffset`` act on the overlap exactly as on first
    order: the order-*m* image has resolving power ``R_m(λ / m) = m R(λ / m)``
    at the photon wavelength, and the pixel displacement is the same in every
    order.  ``FluxScale`` multiplies the whole model.
    """

    #: Whether higher-order overlap is physical for this disperser.  When
    #: ``False``, passing ``order_overlap`` emits a :class:`UserWarning`.
    supports_order_overlap: bool = True

    def __init__(
        self,
        unit: u.UnitBase,
        *,
        name: str = '',
        r_scale: RScale | None = None,
        flux_scale: FluxScale | None = None,
        pix_offset: PixOffset | None = None,
        order_overlap: Sequence[OrderOverlap] | None = None,
    ) -> None:
        if r_scale is not None and not isinstance(r_scale, RScale):
            msg = f'r_scale must be an RScale token, got {type(r_scale).__name__}'
            raise TypeError(msg)
        if flux_scale is not None and not isinstance(flux_scale, FluxScale):
            msg = (
                f'flux_scale must be a FluxScale token, got {type(flux_scale).__name__}'
            )
            raise TypeError(msg)
        if pix_offset is not None and not isinstance(pix_offset, PixOffset):
            msg = (
                f'pix_offset must be a PixOffset token, got {type(pix_offset).__name__}'
            )
            raise TypeError(msg)
        self.unit = unit
        self.name = name
        # Every slot always carries a token; an omitted slot gets a private,
        # unshared token with the default `Fixed` prior (see class docstring).
        self.r_scale = r_scale if r_scale is not None else RScale()
        self.flux_scale = flux_scale if flux_scale is not None else FluxScale()
        self.pix_offset = pix_offset if pix_offset is not None else PixOffset()
        self.order_overlap: tuple[OrderOverlap, ...] = _validate_order_overlap(
            order_overlap
        )
        if self.order_overlap and not self.supports_order_overlap:
            warnings.warn(
                f'{type(self).__name__} does not normally have higher-order '
                'overlap (prisms, sub-octave multi-arm spectrographs); '
                'order_overlap is applied as given.',
                UserWarning,
                stacklevel=3,
            )

        # Apply the type prefix to any token that has a user-supplied label but
        # no site name yet.  Fully anonymous tokens (label=None, name=None) are
        # named later by InstrumentConfig, which has visibility over sharing.
        for _, tok in self.calibration_tokens():
            if tok._name is None and tok.label is not None:
                tok.name = f'{tok.site_prefix}_{tok.label}'

    @abstractmethod
    def R(self, wavelength: ArrayLike) -> ArrayLike:
        """Return the resolving power at the given wavelengths.

        Parameters
        ----------
        wavelength : ArrayLike
            Wavelength values in the unit specified by ``self.unit``.

        Returns
        -------
        ArrayLike
            Resolving power *R = λ / Δλ* evaluated at each wavelength.
        """

    @abstractmethod
    def dlam_dpix(self, wavelength: ArrayLike) -> ArrayLike:
        """Return the linear dispersion (wavelength per pixel).

        Parameters
        ----------
        wavelength : ArrayLike
            Wavelength values in the unit specified by ``self.unit``.

        Returns
        -------
        ArrayLike
            Dispersion *dλ/dpix* evaluated at each wavelength.
        """

    #: Fixed calibration slots, in order.  Each is an attribute holding a token.
    CALIB_SLOTS: tuple[str, ...] = ('r_scale', 'flux_scale', 'pix_offset')

    def calibration_tokens(self) -> Iterator[tuple[str, CalibParam]]:
        """Yield every calibration token on this disperser as ``(slot, token)``.

        ``slot`` is the fixed slot name (``'r_scale'``, ``'flux_scale'``,
        ``'pix_offset'``), followed by ``'order_scale_{m}'`` for the
        :class:`OrderScale` of each entry of :attr:`order_overlap`.  Tokens
        are yielded once per slot, so a token occupying several slots is
        yielded several times; callers that need unique tokens deduplicate by
        identity.

        Yields
        ------
        tuple of (str, CalibParam)
            The slot label and the token in it.
        """
        for slot in self.CALIB_SLOTS:
            yield slot, getattr(self, slot)
        for ov in getattr(self, 'order_overlap', ()):
            yield f'{ov.scale.site_prefix}_{ov.order}', ov.scale

    @property
    def calibrated_range(self) -> tuple[float, float] | None:
        """Wavelength range ``(low, high)`` over which :meth:`R` is calibrated.

        In :attr:`unit`.  ``None`` (default) when the disperser does not
        expose one.  Used to warn when the higher-order overlap evaluates
        ``R`` at photon wavelengths outside it.
        """
        return None

    @property
    def has_calibration_params(self) -> bool:
        """``True`` if any calibration token has a non-``Fixed`` (sampled) prior."""
        return any(
            not isinstance(tok.prior, Fixed) for _, tok in self.calibration_tokens()
        )

    def __str__(self) -> str:
        """Return :func:`repr` plus attached calibration tokens and their priors.

        ``__repr__`` stays a compact one-liner (used as a table cell by
        :class:`~unite.instrument.config.InstrumentConfig` and
        :class:`~unite.spectrum.Spectra`); ``print()``/``str()`` on a
        standalone disperser additionally shows what ``r_scale``,
        ``flux_scale``, and ``pix_offset`` are set to.
        """
        lines = [repr(self)]
        if self.order_overlap:
            lines.append('')
            lines.append('  Order Overlap:')
            lines.extend(f'    {ov!r}' for ov in self.order_overlap)
        lines.extend(format_calibration_sections([self]))
        return '\n'.join(lines)


def _validate_order_overlap(
    order_overlap: Sequence[OrderOverlap] | None,
) -> tuple[OrderOverlap, ...]:
    """Validate an ``order_overlap`` sequence and return it sorted by order.

    Raises
    ------
    TypeError
        If *order_overlap* is not a sequence of
        :class:`~unite.instrument.overlap.OrderOverlap`.
    ValueError
        If an order appears more than once.
    """
    from unite.instrument.overlap import OrderOverlap

    if order_overlap is None:
        return ()
    if isinstance(order_overlap, OrderOverlap) or not isinstance(
        order_overlap, Sequence
    ):
        msg = (
            'order_overlap must be a sequence of OrderOverlap, got '
            f'{type(order_overlap).__name__}'
        )
        raise TypeError(msg)
    for ov in order_overlap:
        if not isinstance(ov, OrderOverlap):
            msg = f'order_overlap entries must be OrderOverlap, got {type(ov).__name__}'
            raise TypeError(msg)
    orders = [ov.order for ov in order_overlap]
    dupes = sorted({m for m in orders if orders.count(m) > 1})
    if dupes:
        msg = f'Duplicate order(s) in order_overlap: {dupes}'
        raise ValueError(msg)
    return tuple(sorted(order_overlap, key=lambda ov: ov.order))


def format_calibration_sections(dispersers: Sequence[Disperser]) -> list[str]:
    """Format shared calibration-token sections for a multi-line ``__repr__``.

    Deduplicates the calibration tokens (:meth:`Disperser.calibration_tokens`)
    by identity across *dispersers* — a token shared by several dispersers (or
    several spectra pointing at the same disperser) is listed once — and
    renders each token type as an indented ``name  prior`` block.

    Parameters
    ----------
    dispersers : Sequence[Disperser]
        Dispersers to scan for calibration tokens. Duplicate disperser
        instances are ignored.

    Returns
    -------
    list[str]
        Lines to append to a ``__repr__``, each section preceded by a blank
        line. Empty if no disperser carries a calibration token.
    """
    # Grouped by site prefix (one section per token type), deduplicated by
    # token identity within each group, in first-seen order.
    calib_params: dict[str, list[tuple[str, CalibParam]]] = {}
    seen_ids: set[int] = set()
    seen_dispersers: set[int] = set()
    for d in dispersers:
        if id(d) in seen_dispersers:
            continue
        seen_dispersers.add(id(d))
        for _, token in d.calibration_tokens():
            if id(token) in seen_ids:
                continue
            seen_ids.add(id(token))
            group = calib_params.setdefault(token.site_prefix, [])
            group.append((token.display_name, token))

    lines: list[str] = []
    for attr, params in calib_params.items():
        if params:
            lines.append('')
            section_title = attr.replace('_', ' ').title()
            lines.append(f'  {section_title}:')
            name_width = max(len(name) for name, _ in params)
            for name, token in params:
                lines.append(f'    {name:<{name_width}}  {token.prior!r}')

    return lines
