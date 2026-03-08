"""Abstract base class for dispersers and calibration parameter tokens."""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import count

from astropy import units as u
from jax.typing import ArrayLike

from unite.prior import Fixed, Parameter, Prior

# ---------------------------------------------------------------------------
# Auto-naming counters (module-level, reset per Python session)
# ---------------------------------------------------------------------------

_rscale_counter = count(0)
_fluxscale_counter = count(0)
_pixoffset_counter = count(0)


# ---------------------------------------------------------------------------
# CalibParam tokens — subclasses of Parameter with instrument-specific defaults
# ---------------------------------------------------------------------------


class RScale(Parameter):
    """Multiplicative scale on the disperser resolving power *R*.

    Nominal value is 1 (no correction).  The model applies
    ``R_eff(λ) = R_nominal(λ) * r_scale``.

    Parameters
    ----------
    prior : Prior, optional
        Prior on the R scale factor.  Defaults to ``Fixed(1.0)`` (fixed at
        nominal).
    name : str, optional
        Auto-generated as ``'r_N'`` if not provided.
    """

    def __init__(self, prior: Prior | None = None, name: str | None = None) -> None:
        if prior is None:
            prior = Fixed(1.0)
        super().__init__(name=name or f'r_{next(_rscale_counter)}', prior=prior)


class FluxScale(Parameter):
    """Multiplicative flux normalisation between dispersers.

    Nominal value is 1.  The model divides observed flux by ``flux_scale``,
    allowing relative flux calibration across multiple gratings.

    Parameters
    ----------
    prior : Prior, optional
        Prior on the flux scale factor.  Defaults to ``Fixed(1.0)`` (fixed at
        nominal).
    name : str, optional
        Auto-generated as ``'flux_N'`` if not provided.
    """

    def __init__(self, prior: Prior | None = None, name: str | None = None) -> None:
        if prior is None:
            prior = Fixed(1.0)
        super().__init__(name=name or f'flux_{next(_fluxscale_counter)}', prior=prior)


class PixOffset(Parameter):
    """Additive pixel shift in the wavelength solution.

    Nominal value is 0 (no shift).  The model shifts pixel bin edges by
    ``pix_offset`` pixels.

    Parameters
    ----------
    prior : Prior, optional
        Prior on the pixel offset.  Defaults to ``Fixed(0.0)`` (no shift).
    name : str, optional
        Auto-generated as ``'pix_N'`` if not provided.
    """

    def __init__(self, prior: Prior | None = None, name: str | None = None) -> None:
        if prior is None:
            prior = Fixed(0.0)
        super().__init__(name=name or f'pix_{next(_pixoffset_counter)}', prior=prior)


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
    * ``pix_offset`` — additive pixel shift in the wavelength solution.

    ``None`` on any slot means that parameter is absent from the model
    entirely (equivalent to a fixed nominal value but without a token).
    To create a token that is fixed at its nominal value, pass
    e.g. ``r_scale=RScale()`` — the default prior is ``Fixed(1.0)``.

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
        Token for the resolving-power scale.  ``None`` → not in model.
    flux_scale : FluxScale, optional
        Token for the flux normalisation.  ``None`` → not in model.
    pix_offset : PixOffset, optional
        Token for the wavelength-solution pixel shift.  ``None`` → not in model.
    """

    def __init__(
        self,
        unit: u.UnitBase,
        *,
        name: str = '',
        r_scale: RScale | None = None,
        flux_scale: FluxScale | None = None,
        pix_offset: PixOffset | None = None,
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
        self.r_scale = r_scale
        self.flux_scale = flux_scale
        self.pix_offset = pix_offset

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
        """``True`` if any calibration token is attached."""
        return any(
            x is not None for x in (self.r_scale, self.flux_scale, self.pix_offset)
        )
