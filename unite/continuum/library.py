"""Continuum functional forms: abstract base and concrete implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import jax
import jax.numpy as jnp
import numpy as np
from astropy import units as u
from jax import Array
from jax.typing import ArrayLike

from unite._utils import _ensure_wavelength, _get_conversion_factor, _make_register
from unite.continuum.functions import (
    bernstein_eval,
    bspline_eval,
    chebval,
    planck_function,
)
from unite.prior import Fixed, Prior, Uniform

if TYPE_CHECKING:
    pass


def _gaussian_convolve_poly(coeffs: Array, lsf_fwhm: ArrayLike) -> Array:
    """Analytically convolve a polynomial with a Gaussian LSF.

    Given a polynomial ``p(x) = Σ c_i x^(n-i)`` (NumPy descending-order
    convention, degree *n* = ``len(coeffs) - 1``) and a Gaussian kernel with
    FWHM ``lsf_fwhm``, return the coefficients of the convolved polynomial
    ``(p * G)(x)``.

    A polynomial convolved with a Gaussian is still a polynomial of the same
    degree. Only even Gaussian moments contribute (odd moments vanish by
    symmetry), giving the coefficient update:

    .. code-block:: text

        c_j_new = sum_{k=0,2,4,...}^{N-j}  c_{j+k} * C(j+k, k) * (k-1)!! * sigma^k

    where ``(k-1)!!`` is the double factorial (the *k*-th even moment of
    a standard normal). See the
    :doc:`polynomial derivation </derivations/polynomial>` for a full derivation.

    For a monomial ``x^k`` convolved with ``N(0, s^2)``, the result is::

        sum_{j=0}^{floor(k/2)} C(k, 2j) * (2j-1)!! * s^{2j} * x^{k-2j}

    Parameters
    ----------
    coeffs : Array, shape ``(n+1,)``
        Polynomial coefficients in **descending** order
        (i.e. ``coeffs[0]`` is the leading coefficient for ``x^n``).
    lsf_fwhm : ArrayLike
        Scalar (or broadcastable) LSF FWHM in the same unit as *x*.

    Returns
    -------
    Array, shape ``(n+1,)``
        Convolved polynomial coefficients, same descending-order convention.
    """
    sigma2 = (jnp.asarray(lsf_fwhm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))) ** 2
    n = coeffs.shape[0] - 1  # polynomial degree; static at trace time
    max_half = n // 2 + 1

    # Even Gaussian moments M[j] = (2j-1)!! * sigma^{2j}.
    # M[0] = 1; M[j] = M[j-1] * (2j-1) * sigma^2 for j >= 1.
    if max_half > 1:

        def _moment_step(carry, j):
            cur = carry * (2 * j - 1) * sigma2
            return cur, cur

        _, rest = jax.lax.scan(_moment_step, 1.0, jnp.arange(1, max_half))
        moments = jnp.concatenate([jnp.array([1.0]), rest])
    else:
        moments = jnp.ones(1)

    # Binomial coefficients C(k, 2j): pure NumPy — depends only on n (static).
    binom_np = np.zeros((n + 1, max_half))
    for k in range(n + 1):
        for j in range(min(k // 2, max_half - 1) + 1):
            val = 1.0
            for m in range(2 * j):
                val = val * (k - m) / (m + 1)
            binom_np[k, j] = val

    # Static scatter tensor: A[out_idx, i, j] = binom_np[n-i, j] when
    # out_idx == i + 2*j (and the entry is in range), else 0.
    # out[out_idx] = sum_i sum_j coeffs[i] * A[out_idx, i, j] * moments[j]
    #              = einsum('oij,j,i->o', A, moments, coeffs).
    a_np = np.zeros((n + 1, n + 1, max_half))
    for i in range(n + 1):
        k = n - i
        for j in range(min(k // 2, max_half - 1) + 1):
            a_np[i + 2 * j, i, j] = binom_np[k, j]

    return jnp.einsum('oij,j,i->o', jnp.asarray(a_np), moments, coeffs)


def _polyint_at(coeffs: Array, x: ArrayLike) -> Array:
    """Antiderivative of a polynomial evaluated at *x*.

    Given descending-order coefficients ``[a_n, ..., a_0]``, return the
    antiderivative ``P(x) = ∫ p(x') dx' = a_n/(n+1) * x^{n+1} + ... + a_0 * x``
    evaluated at each entry of *x* (constant of integration zero).

    Used by polynomial-based continuum forms to produce a cumulative-at-edges
    array: ``jnp.diff(P_at_edges) / jnp.diff(edges)`` recovers the exact
    pixel-averaged value of *p* over each pixel.

    Parameters
    ----------
    coeffs : Array, shape ``(n+1,)``
        Polynomial coefficients in descending order.
    x : ArrayLike
        Points at which to evaluate the antiderivative.

    Returns
    -------
    Array
        Antiderivative value at each *x*.
    """
    # Antiderivative coefficients (also descending), with zero constant term.
    n = coeffs.shape[0]
    divisors = jnp.arange(n, 0, -1, dtype=coeffs.dtype)
    anti = jnp.concatenate([coeffs / divisors, jnp.array([0.0], dtype=coeffs.dtype)])
    return jnp.polyval(anti, x)


def _cheb_to_mono_matrix(n: int) -> Array:
    """Build the (n x n) Chebyshev-to-monomial conversion matrix.

    Returns a matrix *M* such that ``M @ cheb_coeffs`` gives monomial
    coefficients in **ascending** order (``[a0, a1, ..., a_{n-1}]``).

    ``cheb_coeffs = [c0, c1, ..., c_{n-1}]`` represents
    ``c0*T0(x) + c1*T1(x) + ... + c_{n-1}*T_{n-1}(x)``.

    Built at Python time (not traced), so the matrix is a static constant.
    """
    import numpy as np
    from numpy.polynomial.chebyshev import cheb2poly

    M = np.zeros((n, n))  # noqa: N806
    for k in range(n):
        basis = np.zeros(n)
        basis[k] = 1.0
        # cheb2poly returns ascending monomial coefficients for T_k
        poly = cheb2poly(basis)
        M[: len(poly), k] = poly
    return jnp.array(M)


def _bernstein_to_mono_matrix(n: int) -> Array:
    """Build the ((n+1) x (n+1)) Bernstein-to-monomial conversion matrix.

    Returns a matrix *M* such that ``M @ bern_coeffs`` gives monomial
    coefficients in **ascending** order for the polynomial on ``[0, 1]``.

    ``bern_coeffs = [b0, b1, ..., b_n]`` represents
    ``Σ b_i * C(n,i) * t^i * (1-t)^{n-i}``.
    """
    import numpy as np
    from scipy.special import comb

    M = np.zeros((n + 1, n + 1))  # noqa: N806
    # Monomial coeff for t^k: Σ_{i=0}^{k} C(n,i) C(n-i, k-i) (-1)^{k-i} b_i
    for k in range(n + 1):
        for i in range(k + 1):
            M[k, i] = (
                comb(n, i, exact=True)
                * comb(n - i, k - i, exact=True)
                * (-1) ** (k - i)
            )
    return jnp.array(M)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_FORM_REGISTRY: dict[str, type[ContinuumForm]] = {}
_register = _make_register(_FORM_REGISTRY)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class ContinuumForm(ABC):
    """Abstract base class for continuum functional forms.

    Each subclass defines a parameterised continuum model that can be
    evaluated at an array of wavelengths.  Forms are stateless (or carry
    only static configuration like polynomial degree) and are fully
    serialisable for YAML round-trip.

    All forms share a unified parameter interface:

    * ``scale`` — the continuum flux at ``norm_wav``.
    * ``norm_wav`` — rest-frame reference wavelength where
      the continuum equals ``scale``.  Default prior is
      ``Fixed(region_center)``, pinning it to the region midpoint.  Pass an
      explicit :class:`~unite.continuum.config.ContinuumNormalizationWavelength`
      token with ``Fixed(value)`` to share a consistent reference wavelength
      across multiple regions.

    Subclasses must implement :meth:`param_names`, :meth:`default_priors`,
    :meth:`evaluate`, :meth:`to_dict`, and :meth:`from_dict`.
    """

    @abstractmethod
    def param_names(self) -> tuple[str, ...]:
        """Names of the parameters this form requires.

        Returns
        -------
        tuple of str
            Parameter names.  Always includes ``'scale'`` and
            ``'norm_wav'``.
        """

    @abstractmethod
    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        """Return sensible default priors for each parameter.

        The keys must match :meth:`param_names`.

        Parameters
        ----------
        region_center : float
            Midpoint of the continuum region, used as the default
            ``Fixed`` value for ``norm_wav``.

        Returns
        -------
        dict of str to Prior
        """

    @property
    def n_params(self) -> int:
        """Number of free parameters (including Fixed reference parameters)."""
        return len(self.param_names())

    @abstractmethod
    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        """Evaluate the (optionally LSF-convolved) continuum model.

        All operations must use :mod:`jax.numpy` so the function is
        compatible with JAX tracing / JIT compilation.

        Parameters
        ----------
        wavelength : ArrayLike
            Wavelength array (observed frame).
        center : float
            Region midpoint in observed frame (passed for convenience;
            forms use ``params['norm_wav']`` instead).
        params : dict of str to ArrayLike
            Parameter values keyed by :meth:`param_names`.
            ``params['norm_wav']`` is already in observed
            frame (the model applies the systemic redshift before calling
            this method).
        obs_low : float
            Lower observed-frame wavelength bound of the region.
        obs_high : float
            Upper observed-frame wavelength bound of the region.
        lsf_fwhm : ArrayLike, optional
            LSF FWHM at each wavelength point (same unit as
            *wavelength*).  Default ``0.0`` means no LSF convolution.
        z_sys : float, optional
            Systemic redshift of the source.  Forms that evaluate in
            rest-frame (e.g. :class:`Template`) use this to convert
            observed-frame wavelengths to rest-frame.  Most forms ignore
            it.  Default ``0.0``.

        Returns
        -------
        Array
            Continuum flux at each wavelength.
        """

    def integrate(
        self,
        edges: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        """Cumulative continuum integral evaluated at edges.

        ``jnp.diff`` over the returned array (then divided by pixel widths)
        gives the pixel-averaged continuum.  The default implementation is
        a midpoint-rule cumulative sum via :meth:`evaluate`, which is exact
        for forms that vary linearly across a pixel (e.g. :class:`Linear`)
        and a reasonable approximation otherwise.  Polynomial-based forms
        override this method to use the exact antiderivative.

        Parameters
        ----------
        edges : ArrayLike, shape ``(E,)``
            Pixel edges (observed frame, canonical wavelength unit).
        center : float
            Region midpoint in observed frame.
        params : dict of str to ArrayLike
            Parameter values keyed by :meth:`param_names`.
        obs_low : float
            Lower observed-frame wavelength bound of the region.
        obs_high : float
            Upper observed-frame wavelength bound of the region.
        lsf_fwhm : ArrayLike, optional
            LSF FWHM scalar for the region; polynomial-based forms apply
            it via the analytic Gaussian-moment convolution.  Default
            ``0.0`` means no LSF convolution.
        z_sys : float, optional
            Systemic redshift; passed through to :meth:`evaluate`.
            Default ``0.0``.

        Returns
        -------
        Array, shape ``(E,)``
            Cumulative continuum integral at the edges.
        """
        edges_arr = jnp.asarray(edges)
        mids = 0.5 * (edges_arr[1:] + edges_arr[:-1])
        widths = jnp.diff(edges_arr)
        vals = self.evaluate(mids, center, params, obs_low, obs_high, lsf_fwhm, z_sys)
        return jnp.concatenate(
            [jnp.zeros(1, edges_arr.dtype), jnp.cumsum(vals * widths)]
        )

    @property
    def is_linear(self) -> bool:
        """Whether the form is linear in its fitted parameters.

        Linear forms can be solved exactly via weighted least squares.
        Nonlinear forms require iterative solvers (e.g. Gauss-Newton).

        Returns
        -------
        bool
        """
        return False

    @abstractmethod
    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        """Physical unit mapping for each parameter.

        Parameters
        ----------
        flux_unit : astropy.units.UnitBase
            Flux density unit of the spectrum (e.g. ``u.Unit('erg s-1 cm-2 AA-1')``).
        wl_unit : astropy.units.UnitBase
            Canonical wavelength unit (e.g. ``u.um``).

        Returns
        -------
        dict of str to (bool, unit)
            For each parameter name: ``(apply_continuum_scale, physical_unit)``.
            If ``apply_continuum_scale`` is ``True``, multiply the sampled value
            by ``continuum_scale`` to recover the physical quantity.
            ``physical_unit`` is the resulting astropy unit, or ``None`` for
            dimensionless parameters.
        """

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to a YAML-safe dictionary."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict) -> ContinuumForm:
        """Deserialize from a dictionary."""

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return True

    def __hash__(self) -> int:
        return hash(type(self).__name__)

    def _prepare(self, low: u.Quantity, high: u.Quantity) -> None:
        """Adapt the form to the specific continuum region.

        Used for ensuring units and ranges are consistent with the region bounds.

        Parameters
        ----------
        low : astropy.units.Quantity
            Lower bound of the continuum region in the region's wavelength unit.
        high : astropy.units.Quantity
            Upper bound of the continuum region in the region's wavelength unit.
        """
        return None

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'


# ---------------------------------------------------------------------------
# Deserialization helper
# ---------------------------------------------------------------------------


def form_from_dict(d: dict) -> ContinuumForm:
    """Reconstruct a :class:`ContinuumForm` from a serialised dictionary.

    Parameters
    ----------
    d : dict
        Must contain a ``'type'`` key matching a registered form name.

    Returns
    -------
    ContinuumForm

    Raises
    ------
    ValueError
        If the type is not recognised.
    """
    type_name = d['type']
    if type_name not in _FORM_REGISTRY:
        msg = f'Unknown ContinuumForm type: {type_name!r}'
        raise ValueError(msg)
    return _FORM_REGISTRY[type_name].from_dict(d)


def get_form(name_or_form: str | ContinuumForm, **kwargs) -> ContinuumForm:
    r"""Get a :class:`ContinuumForm` by name or pass through an existing instance.

    Parameters
    ----------
    name_or_form : str or ContinuumForm
        A registered form name (e.g. ``'Linear'``, ``'PowerLaw'``,
        ``'Polynomial'``) or an existing :class:`ContinuumForm` instance.
    \*\*kwargs
        Passed to the form constructor when *name_or_form* is a string
        (e.g. ``get_form('Polynomial', degree=3)``).

    Returns
    -------
    ContinuumForm

    Raises
    ------
    ValueError
        If the name is not recognised.

    Examples
    --------
    >>> get_form('Linear')
    Linear()
    >>> get_form('Polynomial', degree=3)
    Polynomial(degree=3)
    >>> get_form(Linear())  # pass-through
    Linear()
    """
    if isinstance(name_or_form, ContinuumForm):
        return name_or_form
    if name_or_form not in _FORM_REGISTRY:
        msg = f'Unknown ContinuumForm type: {name_or_form!r}. Available: {sorted(_FORM_REGISTRY)}'
        raise ValueError(msg)
    return _FORM_REGISTRY[name_or_form](**kwargs)


# ---------------------------------------------------------------------------
# Piecewise / region-local forms
# ---------------------------------------------------------------------------


@_register
class Linear(ContinuumForm):
    """Linear continuum: ``scale + slope * (wavelength - norm_wav)``.

    This form has no constructor parameters.

    Notes
    -----
    **Model parameters** (sampled with priors, overridable via
    ``ContinuumRegion(params={...})``):

    * ``scale`` — Continuum level at ``norm_wav``.
      Default prior: ``Uniform(0, 10)``.
    * ``slope`` — Continuum slope in flux per wavelength unit.
      Default prior: ``Uniform(-10, 10)``.
    * ``norm_wav`` — Reference wavelength where the
      continuum equals ``scale``.
      Default prior: ``Fixed(region_center)``.
    """

    @property
    @override
    def is_linear(self) -> bool:
        return True

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('scale', 'angle', 'norm_wav')

    @override
    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        return {
            'scale': Uniform(0, 2),
            'angle': Uniform(-np.pi / 2, np.pi / 2),
            'norm_wav': Fixed(region_center),
        }

    @override
    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        return {
            'scale': (True, flux_unit),
            'angle': (False, None),
            'norm_wav': (False, wl_unit),
        }

    @override
    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        nw = params['norm_wav']
        return params['scale'] + jnp.tan(params['angle']) * (wavelength - nw)

    @override
    def integrate(
        self,
        edges: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        # Linear is preserved exactly under Gaussian convolution, so LSF is a
        # no-op.  The antiderivative of ``scale + slope * (λ - nw)`` is
        # ``scale * (λ - nw) + slope * (λ - nw)² / 2`` (constant of integration
        # zero); ``jnp.diff`` then recovers the exact pixel integral.
        nw = params['norm_wav']
        x = jnp.asarray(edges) - nw
        slope = jnp.tan(params['angle'])
        return jnp.asarray(params['scale']) * x + 0.5 * slope * x * x

    @override
    def to_dict(self) -> dict:
        return {'type': 'Linear'}

    @classmethod
    @override
    def from_dict(cls, d: dict) -> Linear:
        return cls()


@_register
class Polynomial(ContinuumForm):
    """Polynomial continuum of configurable degree.

    Evaluates ``scale + c1*x + c2*x**2 + ...`` where
    ``x = wavelength - norm_wav``.

    Parameters
    ----------
    degree : int
        Polynomial degree (default 1).

    Notes
    -----
    **Model parameters** (sampled with priors, overridable via
    ``ContinuumRegion(params={...})``):

    * ``scale`` — Continuum level at ``norm_wav``.
      Default prior: ``Uniform(0, 10)``.
    * ``c1, c2, ...`` — Higher-order polynomial coefficients.
      Default prior: ``Uniform(-10, 10)`` each.
    * ``norm_wav`` — Reference wavelength.
      Default prior: ``Fixed(region_center)``.

    """

    def __init__(self, degree: int = 1) -> None:
        if degree < 0:
            msg = f'Polynomial degree must be >= 0, got {degree}'
            raise ValueError(msg)
        self._degree = degree

    @property
    @override
    def is_linear(self) -> bool:
        return True

    @property
    def degree(self) -> int:
        """Polynomial degree."""
        return self._degree

    @override
    def param_names(self) -> tuple[str, ...]:
        if self._degree == 0:
            return ('scale', 'norm_wav')
        return ('scale', *(f'c{i}' for i in range(1, self._degree + 1)), 'norm_wav')

    @override
    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        priors: dict[str, Prior] = {'scale': Uniform(0, 2)}
        for i in range(1, self._degree + 1):
            priors[f'c{i}'] = Uniform(-10, 10)
        priors['norm_wav'] = Fixed(region_center)
        return priors

    @override
    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        d: dict[str, tuple[bool, u.UnitBase | None]] = {'scale': (True, flux_unit)}
        for i in range(1, self._degree + 1):
            d[f'c{i}'] = (True, flux_unit / wl_unit**i)
        d['norm_wav'] = (False, wl_unit)
        return d

    @override
    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        nw = params['norm_wav']
        x = wavelength - nw
        # Monomial coefficients in descending order: c_n, ..., c_1, scale.
        mono = jnp.array(
            [params[f'c{i}'] for i in range(self._degree, 0, -1)] + [params['scale']]
        )
        convolved = _gaussian_convolve_poly(mono, lsf_fwhm)
        return jnp.polyval(convolved, x)

    @override
    def integrate(
        self,
        edges: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        nw = params['norm_wav']
        x = jnp.asarray(edges) - nw
        mono = jnp.array(
            [params[f'c{i}'] for i in range(self._degree, 0, -1)] + [params['scale']]
        )
        convolved = _gaussian_convolve_poly(mono, lsf_fwhm)
        return _polyint_at(convolved, x)

    @override
    def to_dict(self) -> dict:
        return {'type': 'Polynomial', 'degree': self._degree}

    @classmethod
    @override
    def from_dict(cls, d: dict) -> Polynomial:
        return cls(degree=d['degree'])

    @override
    def __repr__(self) -> str:
        return f'Polynomial(degree={self._degree})'


def _normalize_wavelength(
    wavelength: ArrayLike,
    center: float,
    obs_low: float,
    obs_high: float,
    stretch: float,
) -> tuple[Array, float]:
    """Map observed wavelengths to the normalised coordinate used by Chebyshev/Bernstein.

    Both forms normalize ``wavelength`` to the interval ``[-1, 1]`` (for
    Chebyshev) or ``[0, 1]`` (for Bernstein) using the same scale factor
    ``(obs_high - obs_low) / 2 * stretch``.  This helper computes the
    shared scale factor and the intermediate ``u = (w - center) / scale``
    coordinate; callers apply the form-specific final transformation.

    Returns ``(u, scale_factor)`` where ``u = (wavelength - center) / scale_factor``.
    """
    scale_factor = (obs_high - obs_low) / 2 * stretch
    u = (jnp.asarray(wavelength) - center) / scale_factor
    return u, scale_factor


@_register
class Chebyshev(ContinuumForm):
    """Chebyshev polynomial continuum of configurable order.

    Evaluates a Chebyshev series on coordinates normalized to ``[-1, 1]``
    within the continuum region, normalized so that the continuum equals
    ``scale`` at ``norm_wav``.  Numerically more stable than a standard
    polynomial basis for higher orders.

    The x-coordinate is ``(wavelength - center) / (half_width * stretch)``
    where ``half_width`` is derived from the region bounds passed to
    :meth:`evaluate`, and ``stretch`` is a form-specific scaling factor
    (default ``1.0`` for identity normalization).

    The continuum is parameterized as ``scale * T(x) / T(x_nw)`` where
    ``T`` is the Chebyshev series with constant term fixed at 1.0,
    and ``x_nw`` is the normalized coordinate at ``norm_wav``.

    Parameters
    ----------
    order : int
        Chebyshev order (default 2).  Number of coefficients = order + 1.
    stretch : float
        Stretch factor to scale the region normalization (default 1.0).

    Notes
    -----
    **Model parameters** (sampled with priors, overridable via
    ``ContinuumRegion(params={...})``):

    * ``scale`` — Continuum level at ``norm_wav``.
      Default prior: ``Uniform(0, 10)``.
    * ``c1, c2, ...`` — Higher-order Chebyshev coefficients (normalized to constant term 1.0).
      Default prior: ``Uniform(-10, 10)`` each.
    * ``norm_wav`` — Reference wavelength.
      Default prior: ``Fixed(region_center)``.

    """

    def __init__(self, order: int = 2, stretch: float = 1.0) -> None:
        if order < 0:
            msg = f'Chebyshev order must be >= 0, got {order}'
            raise ValueError(msg)
        self._order = order
        if stretch <= 0:
            msg = f'Chebyshev stretch factor must be > 0, got {stretch}'
            raise ValueError(msg)
        self._stretch = stretch
        # Static Chebyshev-to-monomial conversion matrix.
        self._cheb2mono = _cheb_to_mono_matrix(order + 1)

    @property
    @override
    def is_linear(self) -> bool:
        return False

    @property
    def order(self) -> int:
        """Chebyshev order."""
        return self._order

    @property
    def stretch(self) -> float:
        """Stretch factor for the region normalization."""
        return self._stretch

    @override
    def param_names(self) -> tuple[str, ...]:
        if self._order == 0:
            return ('scale', 'norm_wav')
        return ('scale', *(f'c{i}' for i in range(1, self._order + 1)), 'norm_wav')

    @override
    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        priors: dict[str, Prior] = {'scale': Uniform(0, 2)}
        for i in range(1, self._order + 1):
            priors[f'c{i}'] = Uniform(-10, 10)
        priors['norm_wav'] = Fixed(region_center)
        return priors

    @override
    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        # x is normalised to [-1, 1], so all coefficients have unit flux_unit.
        d: dict[str, tuple[bool, u.UnitBase | None]] = {'scale': (True, flux_unit)}
        for i in range(1, self._order + 1):
            d[f'c{i}'] = (True, flux_unit)
        d['norm_wav'] = (False, wl_unit)
        return d

    @override
    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        x, scale_factor = _normalize_wavelength(
            wavelength, center, obs_low, obs_high, self._stretch
        )
        x_nw, _ = _normalize_wavelength(
            params['norm_wav'], center, obs_low, obs_high, self._stretch
        )
        cheb_coeffs = jnp.array(
            [1.0] + [params[f'c{i}'] for i in range(1, self._order + 1)]
        )
        mono = (self._cheb2mono @ cheb_coeffs)[::-1]  # ascending → descending
        lsf_fwhm_scaled = jnp.asarray(lsf_fwhm) / scale_factor
        convolved = _gaussian_convolve_poly(mono, lsf_fwhm_scaled)
        shape = jnp.polyval(convolved, x)
        shape_nw = chebval(x_nw, cheb_coeffs)
        return params['scale'] * shape / shape_nw

    @override
    def integrate(
        self,
        edges: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        x, scale_factor = _normalize_wavelength(
            edges, center, obs_low, obs_high, self._stretch
        )
        x_nw, _ = _normalize_wavelength(
            params['norm_wav'], center, obs_low, obs_high, self._stretch
        )
        cheb_coeffs = jnp.array(
            [1.0] + [params[f'c{i}'] for i in range(1, self._order + 1)]
        )
        mono = (self._cheb2mono @ cheb_coeffs)[::-1]
        lsf_fwhm_scaled = jnp.asarray(lsf_fwhm) / scale_factor
        convolved = _gaussian_convolve_poly(mono, lsf_fwhm_scaled)
        # Antiderivative in normalised coord; rescale to λ by dλ/du = scale_factor.
        shape_anti = _polyint_at(convolved, x) * scale_factor
        shape_nw = chebval(x_nw, cheb_coeffs)
        return params['scale'] * shape_anti / shape_nw

    @override
    def to_dict(self) -> dict:
        return {'type': 'Chebyshev', 'order': self._order, 'stretch': self._stretch}

    @classmethod
    @override
    def from_dict(cls, d: dict) -> Chebyshev:
        return cls(order=d['order'], stretch=d.get('stretch', 1.0))

    @override
    def __repr__(self) -> str:
        return f'Chebyshev(order={self._order}, stretch={self._stretch})'


@_register
class BSpline(ContinuumForm):
    """B-spline continuum with local knot control.

    The knot vector must be set via *knots* at construction time (typically
    derived from the wavelength coverage of the spectrum).  Knots should be
    in the same wavelength unit as the :class:`ContinuumRegion` bounds;
    they are converted to the canonical unit at region construction time via
    :meth:`_prepare`. Knots must fall within the region bounds.

    The continuum is normalized so that it equals ``scale`` at ``norm_wav``,
    parameterized as ``scale * S(u) / S(u_nw)`` where ``S`` is the B-spline
    series with first coefficient fixed at 1.0, and ``u_nw`` is the normalized
    coordinate at ``norm_wav``.

    Parameters
    ----------
    knots : u.Quantity
        Knot vector in wavelength units. It is automatically clamped at the region bounds.
    degree : int
        Spline degree (default 3 for cubic).

    Notes
    -----
    **Model parameters** (sampled with priors, overridable via
    ``ContinuumRegion(params={...})``):

    * ``scale`` — Continuum level at ``norm_wav``.
      Default prior: ``Uniform(0, 10)``.
    * ``coeff_1, coeff_2, …`` — Remaining B-spline coefficients (normalized to first 1.0).
      Default prior: ``Uniform(-10, 10)`` each.
    * ``norm_wav`` — Reference wavelength.
      Default prior: ``Fixed(region_center)``.

    """

    def __init__(self, knots: u.Quantity, degree: int = 3) -> None:
        if isinstance(knots, u.Quantity):
            self._knots: u.Quantity = _ensure_wavelength(knots, 'knots', ndim=1)
        else:
            raise ValueError(
                f'knots must be an astropy Quantity with length units, got {knots}'
            )
        self._degree = degree
        self._n_basis = len(self._knots) + degree + 1

    @property
    @override
    def is_linear(self) -> bool:
        return False

    @property
    def degree(self) -> int:
        """Spline degree."""
        return self._degree

    @property
    def n_basis(self) -> int:
        """Number of B-spline basis functions."""
        return self._n_basis

    @property
    def knots(self) -> u.Quantity:
        """Knot vector (in the original units passed at construction)."""
        return self._knots

    @override
    def param_names(self) -> tuple[str, ...]:
        if self._n_basis == 1:
            return ('scale', 'norm_wav')
        return ('scale', *(f'coeff_{i}' for i in range(1, self._n_basis)), 'norm_wav')

    @override
    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        priors: dict[str, Prior] = {'scale': Uniform(0, 2)}
        for i in range(1, self._n_basis):
            priors[f'coeff_{i}'] = Uniform(-10, 10)
        priors['norm_wav'] = Fixed(region_center)
        return priors

    @override
    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        # B-spline coefficients share the same unit as the function value.
        d: dict[str, tuple[bool, u.UnitBase | None]] = {'scale': (True, flux_unit)}
        for i in range(1, self._n_basis):
            d[f'coeff_{i}'] = (True, flux_unit)
        d['norm_wav'] = (False, wl_unit)
        return d

    @override
    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        # LSF convolution is not supported for BSpline (non-polynomial basis).
        # Normalize wavelengths relative to the knot range
        obs_range = obs_high - obs_low

        # Map wavelengths to the same coordinate system as the knots
        u = 2 * (wavelength - obs_low) / obs_range - 1

        # Map norm_wav to the same coordinate system
        nw = params['norm_wav']
        u_nw = 2 * (nw - obs_low) / obs_range - 1

        # Also map knots to [-1, 1]
        knots_norm = 2 * (self._knots_eval - obs_low) / obs_range - 1

        shape_coeffs = jnp.concatenate(
            [jnp.array([1.0])]
            + [jnp.atleast_1d(params[f'coeff_{i}']) for i in range(1, self._n_basis)]
        )
        shape = bspline_eval(u, shape_coeffs, knots_norm, self._degree)
        _snw = bspline_eval(
            jnp.atleast_1d(u_nw), shape_coeffs, knots_norm, self._degree
        )
        shape_nw = _snw[0]
        return params['scale'] * shape / shape_nw

    @override
    def _prepare(self, low: u.Quantity, high: u.Quantity) -> None:
        """Validate and prepare the knot vector for evaluation within the region bounds."""
        if any((self._knots <= low) or (self._knots >= high)):
            msg = f'All knots must be within the region bounds [{low}, {high}], got {self._knots}'
            raise ValueError(msg)
        # Add n-1 extra knots at each end for clamping (B-spline convention)
        p = self._degree + 1
        knots_clamped = jnp.concatenate(
            [
                low.value * jnp.ones(p),
                self._knots.to(low.unit).value,
                high.value * jnp.ones(p),
            ]
        )
        self._knots_eval = knots_clamped

    @override
    def to_dict(self) -> dict:
        return {
            'type': 'BSpline',
            'knots': self._knots.value.tolist(),
            'unit': str(self._knots.unit),
            'degree': self._degree,
        }

    @classmethod
    @override
    def from_dict(cls, d: dict) -> BSpline:
        return cls(knots=d['knots'] * u.Unit(d['unit']), degree=d.get('degree', 3))

    @override
    def __repr__(self) -> str:
        return f'BSpline(degree={self._degree}, knots={self._knots})'


@_register
class Bernstein(ContinuumForm):
    """Global Bernstein polynomial continuum, normalized at a reference wavelength.

    Evaluates a Bernstein series on coordinates normalized to ``[0, 1]``
    within the continuum region, normalized so that the continuum equals
    ``scale`` at ``norm_wav``.

    The wavelength range is derived from the region bounds passed to
    :meth:`evaluate`, normalized to ``[0, 1]`` for the Bernstein basis.
    The ``stretch`` parameter optionally scales the region normalization.

    The continuum is parameterized as ``scale * B(t) / B(t_nw)`` where
    ``B`` is the Bernstein series with first term fixed at 1.0, and ``t_nw``
    is the normalized coordinate at ``norm_wav``.

    Parameters
    ----------
    degree : int
        Polynomial degree (default 4).  Number of coefficients = degree + 1.
    stretch : float
        Stretch factor to scale the region normalization (default 1.0).

    Notes
    -----
    **Model parameters** (sampled with priors, overridable via
    ``ContinuumRegion(params={...})``):

    * ``scale`` — Continuum level at ``norm_wav``.
      Default prior: ``Uniform(0, 10)``.
    * ``coeff_1, coeff_2, …`` — Remaining Bernstein coefficients (normalized to first term 1.0).
      Default prior: ``Uniform(-10, 10)`` each.
    * ``norm_wav`` — Reference wavelength.
      Default prior: ``Fixed(region_center)``.

    """

    def __init__(self, degree: int = 4, stretch: float = 1.0) -> None:
        from scipy.special import comb

        self._degree = degree
        if stretch <= 0:
            msg = f'Bernstein stretch factor must be > 0, got {stretch}'
            raise ValueError(msg)
        self._stretch = stretch
        self._binom = jnp.array(
            [comb(degree, i, exact=True) for i in range(degree + 1)], dtype=float
        )
        # Static Bernstein-to-monomial conversion matrix.
        self._bern2mono = _bernstein_to_mono_matrix(degree)

    @property
    @override
    def is_linear(self) -> bool:
        return False

    @property
    def degree(self) -> int:
        """Polynomial degree."""
        return self._degree

    @property
    def stretch(self) -> float:
        """Stretch factor for the region normalization."""
        return self._stretch

    @override
    def param_names(self) -> tuple[str, ...]:
        if self._degree == 0:
            return ('scale', 'norm_wav')
        return (
            'scale',
            *(f'coeff_{i}' for i in range(1, self._degree + 1)),
            'norm_wav',
        )

    @override
    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        priors: dict[str, Prior] = {'scale': Uniform(0, 2)}
        for i in range(1, self._degree + 1):
            priors[f'coeff_{i}'] = Uniform(0, 10)
        priors['norm_wav'] = Fixed(region_center)
        return priors

    @override
    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        # Bernstein coefficients share the same unit as the function value.
        d: dict[str, tuple[bool, u.UnitBase | None]] = {'scale': (True, flux_unit)}
        for i in range(1, self._degree + 1):
            d[f'coeff_{i}'] = (True, flux_unit)
        d['norm_wav'] = (False, wl_unit)
        return d

    @override
    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        # u ∈ [-1, 1]; t = (u + 1) / 2 ∈ [0, 1] for Bernstein basis.
        u, scale_factor = _normalize_wavelength(
            wavelength, center, obs_low, obs_high, self._stretch
        )
        u_nw, _ = _normalize_wavelength(
            params['norm_wav'], center, obs_low, obs_high, self._stretch
        )
        t = (u + 1) / 2
        t_nw = (u_nw + 1) / 2
        coeffs = jnp.concatenate(
            [jnp.array([1.0])]
            + [jnp.atleast_1d(params[f'coeff_{i}']) for i in range(1, self._degree + 1)]
        )
        mono = (self._bern2mono @ coeffs)[::-1]  # ascending → descending
        # LSF FWHM in t-coordinate: dt/dλ = 1 / (2 * scale_factor)
        lsf_fwhm_scaled = jnp.asarray(lsf_fwhm) / (2.0 * scale_factor)
        convolved = _gaussian_convolve_poly(mono, lsf_fwhm_scaled)
        shape = jnp.polyval(convolved, t)
        shape_nw = bernstein_eval(jnp.atleast_1d(t_nw), coeffs, self._binom)
        return cast(Array, params['scale'] * shape / shape_nw)

    @override
    def integrate(
        self,
        edges: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        u, scale_factor = _normalize_wavelength(
            edges, center, obs_low, obs_high, self._stretch
        )
        u_nw, _ = _normalize_wavelength(
            params['norm_wav'], center, obs_low, obs_high, self._stretch
        )
        t = (u + 1) / 2
        t_nw = (u_nw + 1) / 2
        coeffs = jnp.concatenate(
            [jnp.array([1.0])]
            + [jnp.atleast_1d(params[f'coeff_{i}']) for i in range(1, self._degree + 1)]
        )
        mono = (self._bern2mono @ coeffs)[::-1]
        lsf_fwhm_scaled = jnp.asarray(lsf_fwhm) / (2.0 * scale_factor)
        convolved = _gaussian_convolve_poly(mono, lsf_fwhm_scaled)
        # Antiderivative in t-coord; rescale to λ via dλ/dt = 2 * scale_factor.
        shape_anti = _polyint_at(convolved, t) * (2.0 * scale_factor)
        shape_nw = bernstein_eval(jnp.atleast_1d(t_nw), coeffs, self._binom)
        return params['scale'] * shape_anti / shape_nw

    @override
    def to_dict(self) -> dict:
        return {'type': 'Bernstein', 'degree': self._degree, 'stretch': self._stretch}

    @classmethod
    @override
    def from_dict(cls, d: dict) -> Bernstein:
        return cls(degree=d['degree'], stretch=d.get('stretch', 1.0))

    @override
    def __repr__(self) -> str:
        return f'Bernstein(degree={self._degree}, stretch={self._stretch})'


@_register
class PowerLaw(ContinuumForm):
    """Power-law continuum: ``scale * (wavelength / norm_wav) ** beta``.

    This form has no constructor parameters.

    To share a consistent reference wavelength across multiple regions
    (required for physically meaningful parameter sharing), pass a
    :class:`~unite.continuum.config.ContinuumNormalizationWavelength` with
    ``Fixed(value)`` carrying your chosen reference wavelength.

    Notes
    -----
    **Model parameters** (sampled with priors, overridable via
    ``ContinuumRegion(params={...})``):

    * ``scale`` — Continuum level at ``norm_wav``.
      Default prior: ``Uniform(0, 10)``.
    * ``beta`` — Power-law index (dimensionless).
      Default prior: ``Uniform(-5, 5)``.
    * ``norm_wav`` — Reference wavelength.
      Default prior: ``Fixed(region_center)``.
    """

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('scale', 'beta', 'norm_wav')

    @override
    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        return {
            'scale': Uniform(0, 2),
            'beta': Uniform(-5, 5),
            'norm_wav': Fixed(region_center),
        }

    @override
    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        return {
            'scale': (True, flux_unit),
            'beta': (False, None),
            'norm_wav': (False, wl_unit),
        }

    @override
    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        # LSF convolution is not supported for PowerLaw.
        nw = params['norm_wav']
        return cast(Array, params['scale'] * (wavelength / nw) ** params['beta'])

    @override
    def integrate(
        self,
        edges: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        # LSF convolution is not supported for PowerLaw (same as evaluate).
        # Returns cumulative antiderivative at edges; diff/widths gives pixel averages.
        edges_arr = jnp.asarray(edges)
        nw = params['norm_wav']
        beta = params['beta']
        bp1 = beta + 1.0
        # Power-rule antiderivative of (w/nw)^beta = w^bp1 / (bp1 * nw^beta).
        anti_power = edges_arr**bp1 / (bp1 * nw**beta)
        # beta = -1 fallback: antiderivative of 1/w is nw * ln(w).
        anti_log = nw * jnp.log(edges_arr)
        anti = jnp.where(jnp.abs(bp1) > 1e-10, anti_power, anti_log)
        return cast(Array, params['scale'] * (anti - anti[0]))

    @override
    def to_dict(self) -> dict:
        return {'type': 'PowerLaw'}

    @classmethod
    @override
    def from_dict(cls, d: dict) -> PowerLaw:
        return cls()


@_register
class Blackbody(ContinuumForm):
    """Planck blackbody continuum normalized at a reference wavelength.

    Evaluates ``scale * B_λ(T) / B_λ(norm_wav, T)`` so that
    *scale* directly represents the continuum flux at
    ``norm_wav``.  Wavelength parameters may be in any unit;
    automatic unit conversion to microns is applied internally.

    ``norm_wav`` is a named parameter with a default
    ``Fixed(region_center)`` prior.  Pass an explicit
    :class:`~unite.continuum.config.ContinuumNormalizationWavelength` with
    ``Fixed(value)`` to pin it to a specific wavelength across multiple
    regions — essential for physically consistent normalization when fitting
    a single blackbody across disjoint spectral windows.

    This form has no constructor parameters.

    Notes
    -----
    **Model parameters** (sampled with priors, overridable via
    ``ContinuumRegion(params={...})``):

    * ``scale`` — Continuum flux at ``norm_wav``
      (in units of ``continuum_scale``).
      Default prior: ``Uniform(0, 10)``.
    * ``temperature`` — Blackbody temperature in Kelvin.
      Default prior: ``Uniform(100, 50000)``.
    * ``norm_wav`` — Reference wavelength.
      Default prior: ``Fixed(region_center)``.
    """

    def __init__(self) -> None:
        self._micron_factor: float = 1.0

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('scale', 'temperature', 'norm_wav')

    @override
    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        return {
            'scale': Uniform(0, 2),
            'temperature': Uniform(100, 50000),
            'norm_wav': Fixed(region_center),
        }

    @override
    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        return {
            'scale': (True, flux_unit),
            'temperature': (False, u.K),
            'norm_wav': (False, wl_unit),
        }

    @override
    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        # LSF convolution is not supported for Blackbody.
        wl_um = wavelength * self._micron_factor
        nw_um = params['norm_wav'] * self._micron_factor
        bb = planck_function(wl_um, params['temperature'], nw_um)
        return params['scale'] * bb

    @override
    def _prepare(self, low: u.Quantity, high: u.Quantity) -> None:
        """Compute the micron conversion factor for the region's wavelength unit."""
        self._micron_factor = _get_conversion_factor(low.unit, u.um)

    @override
    def to_dict(self) -> dict:
        return {'type': 'Blackbody'}

    @classmethod
    @override
    def from_dict(cls, d: dict) -> Blackbody:
        return cls()


@_register
class ModifiedBlackbody(ContinuumForm):
    """Modified blackbody: ``scale * B_λ(T) * (λ / norm_wav)^beta / B_λ(nw, T)``.

    The power-law modifier *beta* broadens (beta > 0) or narrows (beta < 0)
    the SED relative to a pure blackbody.  *beta = 0* recovers
    :class:`Blackbody`.  Wavelength parameters may be in any unit;
    automatic unit conversion to microns is applied internally.

    ``norm_wav`` is a named parameter with a default
    ``Fixed(region_center)`` prior.  Share a
    :class:`~unite.continuum.config.ContinuumNormalizationWavelength` token
    across regions to enforce a consistent reference wavelength.

    This form has no constructor parameters.

    Notes
    -----
    **Model parameters** (sampled with priors, overridable via
    ``ContinuumRegion(params={...})``):

    * ``scale`` — Continuum flux at ``norm_wav``
      (in units of ``continuum_scale``).
      Default prior: ``Uniform(0, 10)``.
    * ``temperature`` — Blackbody temperature in Kelvin.
      Default prior: ``Uniform(100, 50000)``.
    * ``beta`` — Power-law modifier index (dimensionless).
      Default prior: ``Uniform(-4, 4)``.
    * ``norm_wav`` — Reference wavelength.
      Default prior: ``Fixed(region_center)``.
    """

    def __init__(self) -> None:
        self._micron_factor: float = 1.0

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('scale', 'temperature', 'beta', 'norm_wav')

    @override
    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        return {
            'scale': Uniform(0, 2),
            'temperature': Uniform(100, 50000),
            'beta': Uniform(-4, 4),
            'norm_wav': Fixed(region_center),
        }

    @override
    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        return {
            'scale': (True, flux_unit),
            'temperature': (False, u.K),
            'beta': (False, None),
            'norm_wav': (False, wl_unit),
        }

    @override
    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        # LSF convolution is not supported for ModifiedBlackbody.
        wl_um = wavelength * self._micron_factor
        nw_um = params['norm_wav'] * self._micron_factor
        bb = planck_function(wl_um, params['temperature'], nw_um)
        modifier = (wl_um / nw_um) ** params['beta']
        return params['scale'] * bb * modifier

    @override
    def _prepare(self, low: u.Quantity, high: u.Quantity) -> None:
        """Compute the micron conversion factor for the region's wavelength unit."""
        self._micron_factor = _get_conversion_factor(low.unit, u.um)

    @override
    def to_dict(self) -> dict:
        return {'type': 'ModifiedBlackbody'}

    @classmethod
    @override
    def from_dict(cls, d: dict) -> ModifiedBlackbody:
        return cls()


@_register
class AttenuatedBlackbody(ContinuumForm):
    """Dust-attenuated blackbody continuum.

    Evaluates
    ``scale * B_λ(T) / B_λ(nw,T) * exp(-tau_v * [(λ/lambda_ext)^alpha - (nw/lambda_ext)^alpha])``.

    Extinction is normalized at ``norm_wav`` so that *scale*
    represents the **observed** (attenuated) flux there.  Negative *alpha*
    gives steeper extinction at short wavelengths (typical dust law).
    Wavelength parameters may be in any unit; automatic unit conversion
    to microns is applied internally.

    Parameters
    ----------
    lambda_ext : astropy.units.Quantity
        Reference wavelength for the extinction law. Must be
        :class:`~astropy.units.Quantity` with any length unit — it will
        be converted automatically.  Defaults to  ``5500 * u.AA``.

    Notes
    -----
    ``lambda_ext`` is a *static* configuration parameter (not
    sampled).  It is stored in microns internally and, along with
    evaluation wavelengths, automatically converted at model-build time via
    :meth:`_prepare`.

    **Model parameters** (sampled with priors, overridable via
    ``ContinuumRegion(params={...})``):

    * ``scale`` — Observed continuum flux at ``norm_wav``
      (in units of ``continuum_scale``).
      Default prior: ``Uniform(0, 10)``.
    * ``temperature`` — Blackbody temperature in Kelvin.
      Default prior: ``Uniform(100, 50000)``.
    * ``tau_v`` — Optical depth at ``lambda_ext``.
      Default prior: ``Uniform(0, 5)``.
    * ``alpha`` — Dust extinction power-law index (negative = steeper at
      short λ).  Default prior: ``Uniform(-2, 0)``.
    * ``norm_wav`` — Reference wavelength.
      Default prior: ``Fixed(region_center)``.

    """

    def __init__(self, lambda_ext: u.Quantity = 5500 * u.AA) -> None:
        if isinstance(lambda_ext, u.Quantity):
            self._lambda_ext: u.Quantity = _ensure_wavelength(
                lambda_ext, 'lambda_ext', ndim=0
            )
        else:
            raise ValueError(
                f'lambda_ext must be an astropy Quantity with length units, got {lambda_ext}'
            )
        # Float used in evaluate(); initially microns, updated by _prepare().
        self._lambda_ext_um: float = float(self._lambda_ext.to(u.um).value)
        # Conversion factor from canonical unit to microns.
        self._micron_factor: float = 1.0

    @property
    def lambda_ext(self) -> u.Quantity:
        """Extinction reference wavelength as an astropy Quantity."""
        return self._lambda_ext

    @override
    def param_names(self) -> tuple[str, ...]:
        return ('scale', 'temperature', 'tau_v', 'alpha', 'norm_wav')

    @override
    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        return {
            'scale': Uniform(0, 2),
            'temperature': Uniform(100, 50000),
            'tau_v': Uniform(0, 5),
            'alpha': Uniform(-2, 0),
            'norm_wav': Fixed(region_center),
        }

    @override
    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        return {
            'scale': (True, flux_unit),
            'temperature': (False, u.K),
            'tau_v': (False, None),
            'alpha': (False, None),
            'norm_wav': (False, wl_unit),
        }

    @override
    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        # LSF convolution is not supported for AttenuatedBlackbody.
        wl_um = wavelength * self._micron_factor
        nw_um = params['norm_wav'] * self._micron_factor
        bb = planck_function(wl_um, params['temperature'], nw_um)
        ext_data = (wl_um / self._lambda_ext_um) ** params['alpha']
        ext_pivot = (nw_um / self._lambda_ext_um) ** params['alpha']
        extinction = jnp.exp(-params['tau_v'] * (ext_data - ext_pivot))
        return params['scale'] * bb * extinction

    @override
    def _prepare(self, low: u.Quantity, high: u.Quantity) -> None:
        """Compute the conversion factors for the region's wavelength unit."""
        self._micron_factor = _get_conversion_factor(low.unit, u.um)

    @override
    def to_dict(self) -> dict:
        return {'type': 'AttenuatedBlackbody', 'lambda_ext': self.lambda_ext}

    @classmethod
    @override
    def from_dict(cls, d: dict) -> AttenuatedBlackbody:
        return cls(lambda_ext=d.get('lambda_ext', 0.55))

    @override
    def __repr__(self) -> str:
        return f'AttenuatedBlackbody(lambda_ext={self.lambda_ext})'


@_register
class Template(ContinuumForm):
    """Interpolated spectral template continuum.

    Loads one or more template spectra from a file readable by
    :func:`astropy.table.Table.read`.  The file must contain a wavelength
    column (identified by its physical unit) and one or more flux columns.
    Each flux column becomes a separate scale parameter named
    ``{column}_scale``.

    The template is evaluated by linearly interpolating each column in the
    rest frame and normalising so that ``{col}_scale`` equals the flux at
    ``norm_wav``:

    .. code-block:: text

        F(λ) = Σ_i  {col_i}_scale * T_i(λ_rest) / T_i(norm_wav_rest)

    where ``λ_rest = λ_obs / (1 + z_sys)``.

    Parameters
    ----------
    path : str or Path
        Path to the template file.  Any format supported by
        :func:`~astropy.table.Table.read` is accepted (FITS, ECSV, …).
    wavelength_colname : str, optional
        Name of the wavelength column.  If omitted, the column whose unit
        has ``physical_type == 'length'`` is used; raises if the result is
        ambiguous.
    usecols : list or tuple of str, optional
        Flux columns to load.  Defaults to all non-wavelength columns.
        Raises if any requested column is absent.

    Notes
    -----
    The wavelength column must carry an astropy unit with
    ``physical_type == 'length'``.  Flux columns without units, or with
    units that are not spectral flux density (f_lambda), produce a
    :class:`UserWarning` but are still accepted.

    **Model parameters** (sampled with priors, overridable via
    ``ContinuumRegion(params={...})``):

    * ``{col}_scale`` — Template amplitude at ``norm_wav``, one per column.
      Default prior: ``Uniform(0, 2)``.
    * ``norm_wav`` — Rest-frame reference wavelength (shared across all
      columns).  Default prior: ``Fixed(region_center_rest)``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        wavelength_colname: str | None = None,
        usecols: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        import warnings
        from pathlib import Path as _Path

        from astropy.table import Table

        self._path = _Path(path)
        self._wavelength_colname = wavelength_colname
        self._usecols_arg = tuple(usecols) if usecols is not None else None

        table = Table.read(self._path)

        # --- identify wavelength column ---
        if wavelength_colname is not None:
            if wavelength_colname not in table.colnames:
                msg = (
                    f"wavelength_colname '{wavelength_colname}' not found in "
                    f'{self._path}. Available columns: {table.colnames}'
                )
                raise ValueError(msg)
            wl_col = wavelength_colname
        else:
            length_cols = [
                c
                for c in table.colnames
                if getattr(table[c], 'unit', None) is not None
                and u.Unit(table[c].unit).is_equivalent(u.m)
            ]
            if len(length_cols) == 0:
                msg = (
                    f'No column with a length unit found in {self._path}. '
                    'Set wavelength_colname explicitly.'
                )
                raise ValueError(msg)
            if len(length_cols) > 1:
                msg = (
                    f'Multiple length-unit columns in {self._path}: {length_cols}. '
                    'Set wavelength_colname explicitly.'
                )
                raise ValueError(msg)
            wl_col = length_cols[0]

        # --- wavelength array ---
        wl_col_data = table[wl_col]
        if not hasattr(wl_col_data, 'unit') or wl_col_data.unit is None:
            msg = f"Wavelength column '{wl_col}' has no unit."
            raise ValueError(msg)
        self._lam_qty: u.Quantity = u.Quantity(
            np.asarray(wl_col_data), unit=wl_col_data.unit
        )
        if not np.all(np.diff(self._lam_qty.value) > 0):
            msg = f"Wavelength column '{wl_col}' is not strictly monotonically increasing."
            raise ValueError(msg)

        # --- flux columns ---
        remaining = [c for c in table.colnames if c != wl_col]
        if usecols is not None:
            missing = [c for c in usecols if c not in table.colnames]
            if missing:
                msg = (
                    f'usecols columns not found in {self._path}: {missing}. '
                    f'Available columns: {table.colnames}'
                )
                raise ValueError(msg)
            flux_cols = list(usecols)
        else:
            flux_cols = remaining

        if len(flux_cols) == 0:
            msg = f'No flux columns found in {self._path} after excluding the wavelength column.'
            raise ValueError(msg)

        # --- unit warnings and NaN/inf checks ---
        _flam_ref = u.Unit('erg s-1 cm-2 AA-1')
        for col in flux_cols:
            col_data = table[col]
            col_unit = getattr(col_data, 'unit', None)
            if col_unit is None or str(col_unit) in ('', 'None'):
                warnings.warn(
                    f"Template column '{col}' in {self._path} has no units. "
                    'Assuming f_lambda (spectral flux density per wavelength). '
                    'scale parameters will be in units of continuum_scale.',
                    UserWarning,
                    stacklevel=2,
                )
            elif not u.Unit(col_unit).is_equivalent(_flam_ref):
                warnings.warn(
                    f"Template column '{col}' has unit '{col_unit}'. "
                    'Expected f_lambda (spectral flux density per wavelength).',
                    UserWarning,
                    stacklevel=2,
                )
            arr = np.asarray(col_data, dtype=float)
            if not np.all(np.isfinite(arr)):
                msg = f"Template column '{col}' contains NaN or inf values."
                raise ValueError(msg)

        lam_arr = self._lam_qty.to(u.um).value
        if not np.all(np.isfinite(lam_arr)):
            msg = f"Wavelength column '{wl_col}' contains NaN or inf values."
            raise ValueError(msg)

        self._flux_cols: list[str] = flux_cols
        self._flam_arrays: dict[str, np.ndarray] = {
            col: np.asarray(table[col], dtype=float) for col in flux_cols
        }

        # set by _prepare()
        self._lam_um: np.ndarray | None = None
        self._rest_low_um: float = 0.0
        self._flam_eval: Array | None = None
        self._lam_eval: Array | None = None

    # ------------------------------------------------------------------
    # ContinuumForm interface
    # ------------------------------------------------------------------

    @override
    def param_names(self) -> tuple[str, ...]:
        return (*[f'{c}_scale' for c in self._flux_cols], 'norm_wav')

    @override
    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        priors: dict[str, Prior] = {}
        for col in self._flux_cols:
            priors[f'{col}_scale'] = Uniform(0, 2)
        priors['norm_wav'] = Fixed(region_center)
        return priors

    @override
    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        d: dict[str, tuple[bool, u.UnitBase | None]] = {}
        for col in self._flux_cols:
            d[f'{col}_scale'] = (True, flux_unit)
        d['norm_wav'] = (False, wl_unit)
        return d

    @override
    def _prepare(self, low: u.Quantity, high: u.Quantity) -> None:
        """Convert template to microns and validate wavelength coverage."""
        self._lam_um = self._lam_qty.to(u.um).value
        self._rest_low_um = float(low.to(u.um).value)
        rest_high_um = float(high.to(u.um).value)

        # Use a small relative tolerance to absorb floating-point rounding from
        # unit conversions (e.g. 9000 AA * 1e-4 µm/AA = 0.9000000000000001 µm).
        _rtol = 1e-9
        if self._lam_um[0] > self._rest_low_um * (1.0 + _rtol):
            msg = (
                f'Template wavelength grid starts at {self._lam_um[0]:.4f} µm '
                f'but region lower bound is {low.to(u.um):.4f}. '
                'Template does not cover the full region.'
            )
            raise ValueError(msg)
        if self._lam_um[-1] < rest_high_um * (1.0 - _rtol):
            msg = (
                f'Template wavelength grid ends at {self._lam_um[-1]:.4f} µm '
                f'but region upper bound is {high.to(u.um):.4f}. '
                'Template does not cover the full region.'
            )
            raise ValueError(msg)

        # Stack flux arrays: shape (N_cols, N_lam)
        self._flam_eval = jnp.array(
            np.stack([self._flam_arrays[c] for c in self._flux_cols], axis=0)
        )
        self._lam_eval = jnp.array(self._lam_um)

    @override
    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
        z_sys: float = 0.0,
    ) -> Array:
        # Convert observed-frame wavelength to rest-frame microns.
        # obs_low = rest_low_canonical * (1 + z_sys), and _rest_low_um is
        # rest_low in microns, so wavelength * _rest_low_um / obs_low gives
        # rest-frame microns regardless of the canonical wavelength unit.
        assert self._flam_eval is not None and self._lam_eval is not None, (
            'Template._prepare() must be called before evaluate(). '
            'Wrap the Template in a ContinuumRegion to trigger _prepare().'
        )
        wl = jnp.asarray(wavelength)
        scale = self._rest_low_um / obs_low
        lam_rest_um = wl * scale
        norm_wav_rest_um = params['norm_wav'] * scale

        total = jnp.zeros_like(wl)
        for i, col in enumerate(self._flux_cols):
            flam_row = self._flam_eval[i]
            t_at_lam = jnp.interp(lam_rest_um, self._lam_eval, flam_row)
            t_at_norm = jnp.interp(norm_wav_rest_um, self._lam_eval, flam_row)
            total = total + params[f'{col}_scale'] * t_at_lam / t_at_norm
        return total

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    @override
    def to_dict(self) -> dict:
        d: dict = {'type': 'Template', 'path': str(self._path)}
        if self._wavelength_colname is not None:
            d['wavelength_colname'] = self._wavelength_colname
        if self._usecols_arg is not None:
            d['usecols'] = list(self._usecols_arg)
        return d

    @classmethod
    @override
    def from_dict(cls, d: dict) -> Template:
        return cls(
            d['path'],
            wavelength_colname=d.get('wavelength_colname'),
            usecols=d.get('usecols'),
        )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Template):
            return NotImplemented
        return (
            self._path == other._path
            and self._wavelength_colname == other._wavelength_colname
            and self._usecols_arg == other._usecols_arg
        )

    @override
    def __hash__(self) -> int:
        return hash(
            (
                type(self).__name__,
                self._path,
                self._wavelength_colname,
                self._usecols_arg,
            )
        )

    @override
    def __repr__(self) -> str:
        cols = ', '.join(self._flux_cols)
        return f'Template({self._path.name!r}, columns=[{cols}])'
