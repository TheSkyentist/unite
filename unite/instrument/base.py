"""Abstract base class for dispersers and calibration parameter tokens."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

from astropy import units as u
from jax.typing import ArrayLike

from unite.prior import Fixed, Parameter, Prior

# ---------------------------------------------------------------------------
# CalibParam tokens — subclasses of Parameter with instrument-specific defaults
# ---------------------------------------------------------------------------


class RScale(Parameter):
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

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Fixed(1.0)
        super().__init__(name=name, prior=prior)


class FluxScale(Parameter):
    """Multiplicative flux normalisation between dispersers.

    Nominal value is 1.  The model divides observed flux by ``flux_scale``,
    allowing relative flux calibration across multiple gratings.

    Parameters
    ----------
    name : str, optional
        Human-readable label.  When attached to a :class:`Disperser`, the
        site name is auto-derived as ``'flux_scale_{disperser_name}'`` if not provided.
    prior : Prior, optional
        Prior on the flux scale factor.  Defaults to ``Fixed(1.0)`` (fixed at
        nominal).
    """

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Fixed(1.0)
        super().__init__(name=name, prior=prior)


class PixOffset(Parameter):
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

    def __init__(self, name: str | None = None, *, prior: Prior | None = None) -> None:
        if prior is None:
            prior = Fixed(0.0)
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

    Optionally, ``order_overlap`` describes contamination by higher
    diffraction orders (e.g. second-order light in NIRSpec G140M/F070LP
    redwards of ~1.3 µm) as a deterministic throughput-ratio curve per order.

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
    order_overlap : Mapping[int, Callable], optional
        Higher-order grating overlap: ``{m: r_m}`` where ``m >= 2`` is the
        diffraction order and ``r_m(wavelength)`` is a JAX-jittable callable
        returning the ratio of the order-*m* to first-order throughput at
        the **observed** (first-order-calibrated) wavelength, in this
        disperser's unit and in ``f_lambda`` convention.  The model
        then adds ``r_m(λ) * F_1(λ / m)`` to the first-order model
        ``F_1(λ)``, where ``F_1(λ / m)`` is pixel-integrated and
        LSF-convolved at the order-*m* resolution ``R_m(λ / m) =
        m R(λ / m)``, i.e. with ``R`` at the photon wavelength (see
        Notes).  ``None`` (default) disables the term.
        Runtime-only: not serialised by
        :class:`~unite.instrument.config.InstrumentConfig`.

    Notes
    -----
    **Scope of** ``order_overlap``.  The term describes order-*m* light of
    the *same source* landing on its own first-order spectrum, in a spectrum
    that the pipeline flux-calibrated with the first-order sensitivity only.
    This occurs for single-order gratings and grisms that cover more than
    an octave behind their order-sorting filter (cut-on ``λ_c``), at
    ``λ > m λ_c``: NIRSpec G140M/H with F070LP and G235M/H with F170LP in
    extended extractions, the MUSE extended mode, and low-resolution optical
    long-slit or multi-object set-ups.  It does not describe:

    - cross-dispersed echelles, where the orders are separated spatially;
    - spectrographs whose arms each cover less than an octave (e.g. SDSS,
      DESI), and prisms, which have no higher orders;
    - slitless spectroscopy, where the contamination comes from *other*
      sources;
    - light of another spectrum (or its higher orders) falling onto this
      one.

    Leave ``order_overlap`` unset in those cases.

    **Model.**  The full first-order model (lines, absorbers and continuum)
    is evaluated at ``λ / m`` in both integration modes.

    - The continuum at ``λ / m`` is only non-zero inside a configured
      continuum region, so the continuum configuration must cover both the
      contaminated range and the range it is contaminated by.  Lines that
      are covered only through a higher-order image are still dropped by
      the coverage filtering of :meth:`~unite.spectrum.Spectra.prepare`.
    - Light of wavelength ``λ / m`` forms the same image on the detector
      (in pixels) in every order, while the dispersion of order *m* is *m*
      times larger, hence ``R_m(λ / m) = m R(λ / m)``.  This holds for any
      grating or grism, not only NIRSpec.
    - ``r_scale``, ``flux_scale`` and ``pix_offset`` apply to the overlap
      term exactly as to the first-order term.
    - The ratios are deterministic: no parameter describes the uncertainty
      on the throughput calibration.

    **Atmosphere.**  Ground-based pipelines correct the telluric and
    extinction transmission ``T`` at the observed wavelength ``λ``, whereas
    the order-*m* photons were absorbed at ``λ / m``.  The effective ratio
    is then ``r_m(λ) T(λ / m) / T(λ)``, a factor that the caller must
    include in ``r_m``.  It is absent for space-based spectra.
    """

    def __init__(
        self,
        unit: u.UnitBase,
        *,
        name: str = '',
        r_scale: RScale | None = None,
        flux_scale: FluxScale | None = None,
        pix_offset: PixOffset | None = None,
        order_overlap: Mapping[int, Callable[[ArrayLike], ArrayLike]] | None = None,
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
        self.order_overlap = _validate_order_overlap(order_overlap)

        # Apply the type prefix to any token that has a user-supplied label but
        # no site name yet.  Fully anonymous tokens (label=None, name=None) are
        # named later by InstrumentConfig, which has visibility over sharing.
        for slot, tok in [
            ('r_scale', self.r_scale),
            ('flux_scale', self.flux_scale),
            ('pix_offset', self.pix_offset),
        ]:
            if tok._name is None and tok.label is not None:
                tok.name = f'{slot}_{tok.label}'

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

    @property
    def has_calibration_params(self) -> bool:
        """``True`` if any calibration token has a non-``Fixed`` (sampled) prior."""
        return any(
            not isinstance(tok.prior, Fixed)
            for tok in (self.r_scale, self.flux_scale, self.pix_offset)
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
        lines.extend(format_calibration_sections([self]))
        if self.order_overlap:
            lines.append('')
            orders = ', '.join(str(m) for m in self.order_overlap)
            lines.append(f'  Order Overlap: {orders}')
        return '\n'.join(lines)


def _validate_order_overlap(
    order_overlap: Mapping[int, Callable[[ArrayLike], ArrayLike]] | None,
) -> dict[int, Callable[[ArrayLike], ArrayLike]]:
    """Validate and copy an ``order_overlap`` mapping (see :class:`Disperser`).

    Parameters
    ----------
    order_overlap : Mapping[int, Callable] or None
        ``{order: ratio_callable}``; ``None`` means no overlap.

    Returns
    -------
    dict
        A new dict with integer keys sorted in ascending order.  Empty when
        *order_overlap* is ``None`` or empty.

    Raises
    ------
    TypeError
        If a key is not an integer or a value is not callable.
    ValueError
        If an order is smaller than 2.
    """
    if not order_overlap:
        return {}
    out: dict[int, Callable[[ArrayLike], ArrayLike]] = {}
    for order, ratio in order_overlap.items():
        if isinstance(order, bool) or not isinstance(order, int):
            msg = f'order_overlap keys must be integers, got {order!r}'
            raise TypeError(msg)
        if order < 2:
            msg = f'order_overlap orders must be >= 2, got {order}'
            raise ValueError(msg)
        if not callable(ratio):
            msg = (
                f'order_overlap[{order}] must be a callable ratio function, '
                f'got {type(ratio).__name__}'
            )
            raise TypeError(msg)
        out[order] = ratio
    return dict(sorted(out.items()))


def format_calibration_sections(dispersers: Sequence[Disperser]) -> list[str]:
    """Format shared calibration-token sections for a multi-line ``__repr__``.

    Deduplicates ``r_scale``/``flux_scale``/``pix_offset`` tokens by identity
    across *dispersers* — a token shared by several dispersers (or several
    spectra pointing at the same disperser) is listed once — and renders each
    attribute as an indented ``name  prior`` block.

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
    seen_dispersers: set[int] = set()
    for d in dispersers:
        if id(d) in seen_dispersers:
            continue
        seen_dispersers.add(id(d))
        for attr in ('r_scale', 'flux_scale', 'pix_offset'):
            token = getattr(d, attr)
            if token is not None:
                token_id = id(token)
                if token_id not in seen_ids[attr]:
                    seen_ids[attr].add(token_id)
                    calib_params[attr].append((token.display_name, token))

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
