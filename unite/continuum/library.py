"""Continuum functional forms: abstract base and concrete implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
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

# FWHM to sigma conversion: sigma = FWHM / (2 * sqrt(2 * ln2))
_FWHM_TO_SIGMA = 1.0 / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))


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
    sigma2 = (jnp.asarray(lsf_fwhm) * _FWHM_TO_SIGMA) ** 2
    n = coeffs.shape[0] - 1  # polynomial degree

    # Pre-compute even moments of N(0, s^2): M[j] = (2j-1)!! s^{2j}.
    # M[0] = 1, M[j] = M[j-1] * (2j-1) * s^2.
    max_half = n // 2 + 1
    moments = jnp.zeros(max_half)
    moments = moments.at[0].set(1.0)

    def _moment_step(carry, j):
        prev = carry
        cur = prev * (2 * j - 1) * sigma2
        return cur, cur

    if max_half > 1:
        _, rest = jax.lax.scan(_moment_step, 1.0, jnp.arange(1, max_half))
        moments = jnp.concatenate([jnp.array([1.0]), rest])

    # Pre-compute binomial coefficients C(k, 2j) for all k in [0..n].
    # binom[k, j] = C(k, 2j). We build this with Pascal's rule.
    binom = jnp.zeros((n + 1, max_half))
    for k in range(n + 1):
        for j in range(min(k // 2, max_half - 1) + 1):
            two_j = 2 * j
            # C(k, 2j) via explicit formula: use lax-friendly recursion
            val = 1.0
            for m in range(two_j):
                val = val * (k - m) / (m + 1)
            binom = binom.at[k, j].set(val)

    # Build output coefficients (descending order, same as input).
    out = jnp.zeros(n + 1)
    # Input: coeffs[i] is the coefficient for x^(n-i).
    # For each input monomial c * x^k (k = n - i), the convolved result
    # adds c * C(k, 2j) * M[j] to the x^(k-2j) = x^(n - (i + 2j)) slot.
    for i in range(n + 1):
        k = n - i  # power of this monomial
        for j in range(min(k // 2, max_half - 1) + 1):
            out_idx = i + 2 * j  # slot for x^(k - 2j) in descending order
            out = out.at[out_idx].add(coeffs[i] * binom[k, j] * moments[j])

    return out


def _polyint_avg(coeffs: Array, x_low: ArrayLike, x_high: ArrayLike) -> Array:
    """Exact pixel-averaged value of a polynomial over ``[x_low, x_high]``.

    Given descending-order coefficients ``[a_n, ..., a_0]``, compute::

        (1 / (x_high - x_low)) * integral_{x_low}^{x_high} p(x) dx

    using the analytic antiderivative.

    Parameters
    ----------
    coeffs : Array, shape ``(n+1,)``
        Polynomial coefficients in descending order.
    x_low, x_high : ArrayLike
        Pixel bin edges (may be arrays).

    Returns
    -------
    Array
        Pixel-averaged polynomial values.
    """
    # Antiderivative: for descending [a_n, ..., a_0], the antiderivative
    # is [a_n/(n+1), a_{n-1}/n, ..., a_0/1, 0] (also descending).
    n = coeffs.shape[0]
    divisors = jnp.arange(n, 0, -1, dtype=coeffs.dtype)
    anti = jnp.concatenate([coeffs / divisors, jnp.array([0.0])])
    return (jnp.polyval(anti, x_high) - jnp.polyval(anti, x_low)) / (x_high - x_low)


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

        Returns
        -------
        Array
            Continuum flux at each wavelength.
        """

    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
    ) -> Array:
        """Pixel-integrated continuum model.

        Evaluates the (optionally LSF-convolved) continuum averaged over
        each pixel bin ``[low, high]``.  The default implementation
        evaluates at pixel centres, which is exact for forms whose
        variation across a pixel is negligible (e.g. :class:`Linear`).

        Subclasses may override this to provide analytic integration.

        Parameters
        ----------
        low, high : ArrayLike, shape ``(n_pixels,)``
            Pixel bin edges (observed frame, canonical wavelength unit).
        center : float
            Region midpoint in observed frame.
        params : dict of str to ArrayLike
            Parameter values keyed by :meth:`param_names`.
        obs_low : float
            Lower observed-frame wavelength bound of the region.
        obs_high : float
            Upper observed-frame wavelength bound of the region.
        lsf_fwhm : ArrayLike, optional
            LSF FWHM at each pixel centre (same unit as *low*/*high*).
            Default ``0.0`` means no LSF convolution.

        Returns
        -------
        Array
            Pixel-averaged continuum flux, shape ``(n_pixels,)``.
        """
        wavelength = (low + high) / 2.0
        return self.evaluate(wavelength, center, params, obs_low, obs_high, lsf_fwhm)

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
    def is_linear(self) -> bool:
        return True

    def param_names(self) -> tuple[str, ...]:
        return ('scale', 'angle', 'norm_wav')

    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        return {
            'scale': Uniform(0, 2),
            'angle': Uniform(-jnp.pi / 2, jnp.pi / 2),
            'norm_wav': Fixed(region_center),
        }

    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        return {
            'scale': (True, flux_unit),
            'angle': (False, None),
            'norm_wav': (False, wl_unit),
        }

    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
    ) -> Array:
        nw = params['norm_wav']
        return params['scale'] + jnp.tan(params['angle']) * (wavelength - nw)

    def to_dict(self) -> dict:
        return {'type': 'Linear'}

    @classmethod
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
    def is_linear(self) -> bool:
        return True

    @property
    def degree(self) -> int:
        """Polynomial degree."""
        return self._degree

    def param_names(self) -> tuple[str, ...]:
        if self._degree == 0:
            return ('scale', 'norm_wav')
        return ('scale', *(f'c{i}' for i in range(1, self._degree + 1)), 'norm_wav')

    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        priors: dict[str, Prior] = {'scale': Uniform(0, 2)}
        for i in range(1, self._degree + 1):
            priors[f'c{i}'] = Uniform(-10, 10)
        priors['norm_wav'] = Fixed(region_center)
        return priors

    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        d: dict[str, tuple[bool, u.UnitBase | None]] = {'scale': (True, flux_unit)}
        for i in range(1, self._degree + 1):
            d[f'c{i}'] = (True, flux_unit / wl_unit**i)
        d['norm_wav'] = (False, wl_unit)
        return d

    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
    ) -> Array:
        nw = params['norm_wav']
        x = wavelength - nw
        # Monomial coefficients in descending order: c_n, ..., c_1, scale.
        mono = jnp.array(
            [params[f'c{i}'] for i in range(self._degree, 0, -1)] + [params['scale']]
        )
        convolved = _gaussian_convolve_poly(mono, lsf_fwhm)
        return jnp.polyval(convolved, x)

    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
    ) -> Array:
        nw = params['norm_wav']
        x_low = low - nw
        x_high = high - nw
        mono = jnp.array(
            [params[f'c{i}'] for i in range(self._degree, 0, -1)] + [params['scale']]
        )
        convolved = _gaussian_convolve_poly(mono, lsf_fwhm)
        return _polyint_avg(convolved, x_low, x_high)

    def to_dict(self) -> dict:
        return {'type': 'Polynomial', 'degree': self._degree}

    @classmethod
    def from_dict(cls, d: dict) -> Polynomial:
        return cls(degree=d['degree'])

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._degree == other._degree  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(('Polynomial', self._degree))

    def __repr__(self) -> str:
        return f'Polynomial(degree={self._degree})'


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

    def param_names(self) -> tuple[str, ...]:
        if self._order == 0:
            return ('scale', 'norm_wav')
        return ('scale', *(f'c{i}' for i in range(1, self._order + 1)), 'norm_wav')

    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        priors: dict[str, Prior] = {'scale': Uniform(0, 2)}
        for i in range(1, self._order + 1):
            priors[f'c{i}'] = Uniform(-10, 10)
        priors['norm_wav'] = Fixed(region_center)
        return priors

    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        # x is normalised to [-1, 1], so all coefficients have unit flux_unit.
        d: dict[str, tuple[bool, u.UnitBase | None]] = {'scale': (True, flux_unit)}
        for i in range(1, self._order + 1):
            d[f'c{i}'] = (True, flux_unit)
        d['norm_wav'] = (False, wl_unit)
        return d

    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
    ) -> Array:
        half_width = (obs_high - obs_low) / 2
        scale_factor = half_width * self._stretch
        x = (wavelength - center) / scale_factor
        nw = params['norm_wav']
        x_nw = (nw - center) / scale_factor

        # Chebyshev coefficients → monomial (ascending) via static matrix,
        # then convolve with rescaled LSF and evaluate.
        cheb_coeffs = jnp.array(
            [1.0] + [params[f'c{i}'] for i in range(1, self._order + 1)]
        )
        mono_asc = self._cheb2mono @ cheb_coeffs  # ascending order
        mono = mono_asc[::-1]  # descending for jnp.polyval
        # Rescale LSF FWHM into the normalised coordinate system.
        lsf_fwhm_scaled = jnp.asarray(lsf_fwhm) / scale_factor
        convolved = _gaussian_convolve_poly(mono, lsf_fwhm_scaled)
        shape = jnp.polyval(convolved, x)
        # norm_wav is a scalar — no LSF convolution needed.
        shape_nw = chebval(x_nw, cheb_coeffs)
        return params['scale'] * shape / shape_nw

    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
    ) -> Array:
        half_width = (obs_high - obs_low) / 2
        scale_factor = half_width * self._stretch
        x_low = (low - center) / scale_factor
        x_high = (high - center) / scale_factor
        nw = params['norm_wav']
        x_nw = (nw - center) / scale_factor

        cheb_coeffs = jnp.array(
            [1.0] + [params[f'c{i}'] for i in range(1, self._order + 1)]
        )
        mono_asc = self._cheb2mono @ cheb_coeffs
        mono = mono_asc[::-1]
        lsf_fwhm_scaled = jnp.asarray(lsf_fwhm) / scale_factor
        convolved = _gaussian_convolve_poly(mono, lsf_fwhm_scaled)
        shape_avg = _polyint_avg(convolved, x_low, x_high)
        shape_nw = chebval(x_nw, cheb_coeffs)
        return params['scale'] * shape_avg / shape_nw

    def to_dict(self) -> dict:
        return {'type': 'Chebyshev', 'order': self._order, 'stretch': self._stretch}

    @classmethod
    def from_dict(cls, d: dict) -> Chebyshev:
        return cls(order=d['order'], stretch=d.get('stretch', 1.0))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self._order == other._order and self._stretch == other._stretch  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(('Chebyshev', self._order, self._stretch))

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

    def param_names(self) -> tuple[str, ...]:
        if self._n_basis == 1:
            return ('scale', 'norm_wav')
        return ('scale', *(f'coeff_{i}' for i in range(1, self._n_basis)), 'norm_wav')

    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        priors: dict[str, Prior] = {'scale': Uniform(0, 2)}
        for i in range(1, self._n_basis):
            priors[f'coeff_{i}'] = Uniform(-10, 10)
        priors['norm_wav'] = Fixed(region_center)
        return priors

    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        # B-spline coefficients share the same unit as the function value.
        d: dict[str, tuple[bool, u.UnitBase | None]] = {'scale': (True, flux_unit)}
        for i in range(1, self._n_basis):
            d[f'coeff_{i}'] = (True, flux_unit)
        d['norm_wav'] = (False, wl_unit)
        return d

    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
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
        shape_nw = bspline_eval(
            jnp.atleast_1d(u_nw), shape_coeffs, knots_norm, self._degree
        )[0]
        return params['scale'] * shape / shape_nw

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

    def to_dict(self) -> dict:
        return {
            'type': 'BSpline',
            'knots': self._knots.value.tolist(),
            'unit': str(self._knots.unit),
            'degree': self._degree,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BSpline:
        return cls(knots=d['knots'] * u.Unit(d['unit']), degree=d.get('degree', 3))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (  # type: ignore[attr-defined]
            bool(jnp.array_equal(self._knots, other._knots))
            and self._degree == other._degree
        )

    def __hash__(self) -> int:
        return hash(('BSpline', tuple(self._knots.to(u.um).value), self._degree))

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

    def param_names(self) -> tuple[str, ...]:
        if self._degree == 0:
            return ('scale', 'norm_wav')
        return (
            'scale',
            *(f'coeff_{i}' for i in range(1, self._degree + 1)),
            'norm_wav',
        )

    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        priors: dict[str, Prior] = {'scale': Uniform(0, 2)}
        for i in range(1, self._degree + 1):
            priors[f'coeff_{i}'] = Uniform(0, 10)
        priors['norm_wav'] = Fixed(region_center)
        return priors

    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        # Bernstein coefficients share the same unit as the function value.
        d: dict[str, tuple[bool, u.UnitBase | None]] = {'scale': (True, flux_unit)}
        for i in range(1, self._degree + 1):
            d[f'coeff_{i}'] = (True, flux_unit)
        d['norm_wav'] = (False, wl_unit)
        return d

    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
    ) -> Array:
        # 1. Coordinate Transformation
        half_width = (obs_high - obs_low) / 2
        stretch_factor = half_width * self._stretch

        # helper to transform wavelength to [0, 1]
        def to_t(w):
            uu = (w - center) / stretch_factor
            return (uu + 1) / 2

        t = to_t(wavelength)
        t_nw = to_t(params['norm_wav'])

        # 2. Bernstein coefficients → monomial (ascending) via static matrix,
        # convolve with rescaled LSF, then evaluate.
        coeffs = jnp.concatenate(
            [jnp.array([1.0])]
            + [jnp.atleast_1d(params[f'coeff_{i}']) for i in range(1, self._degree + 1)]
        )
        mono_asc = self._bern2mono @ coeffs  # ascending monomial in t
        mono = mono_asc[::-1]  # descending for jnp.polyval
        # LSF FWHM in t-coordinate: dt/dλ = 1 / (2 * stretch_factor)
        lsf_fwhm_scaled = jnp.asarray(lsf_fwhm) / (2.0 * stretch_factor)
        convolved = _gaussian_convolve_poly(mono, lsf_fwhm_scaled)
        shape = jnp.polyval(convolved, t)
        # norm_wav is a scalar — no LSF convolution needed.
        shape_nw = bernstein_eval(jnp.atleast_1d(t_nw), coeffs, self._binom)

        # 3. Normalize so that the continuum equals `scale` at `norm_wav`
        return params['scale'] * shape / shape_nw

    def integrate(
        self,
        low: ArrayLike,
        high: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
    ) -> Array:
        half_width = (obs_high - obs_low) / 2
        stretch_factor = half_width * self._stretch

        def to_t(w):
            uu = (w - center) / stretch_factor
            return (uu + 1) / 2

        t_low = to_t(low)
        t_high = to_t(high)
        t_nw = to_t(params['norm_wav'])

        coeffs = jnp.concatenate(
            [jnp.array([1.0])]
            + [jnp.atleast_1d(params[f'coeff_{i}']) for i in range(1, self._degree + 1)]
        )
        mono_asc = self._bern2mono @ coeffs
        mono = mono_asc[::-1]
        lsf_fwhm_scaled = jnp.asarray(lsf_fwhm) / (2.0 * stretch_factor)
        convolved = _gaussian_convolve_poly(mono, lsf_fwhm_scaled)
        shape_avg = _polyint_avg(convolved, t_low, t_high)
        shape_nw = bernstein_eval(jnp.atleast_1d(t_nw), coeffs, self._binom)
        return params['scale'] * shape_avg / shape_nw

    def to_dict(self) -> dict:
        return {'type': 'Bernstein', 'degree': self._degree, 'stretch': self._stretch}

    @classmethod
    def from_dict(cls, d: dict) -> Bernstein:
        return cls(degree=d['degree'], stretch=d.get('stretch', 1.0))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return (  # type: ignore[attr-defined]
            self._degree == other._degree and self._stretch == other._stretch
        )

    def __hash__(self) -> int:
        return hash(('Bernstein', self._degree, self._stretch))

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

    def param_names(self) -> tuple[str, ...]:
        return ('scale', 'beta', 'norm_wav')

    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        return {
            'scale': Uniform(0, 2),
            'beta': Uniform(-5, 5),
            'norm_wav': Fixed(region_center),
        }

    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        return {
            'scale': (True, flux_unit),
            'beta': (False, None),
            'norm_wav': (False, wl_unit),
        }

    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
    ) -> Array:
        # LSF convolution is not supported for PowerLaw.
        nw = params['norm_wav']
        return params['scale'] * (wavelength / nw) ** params['beta']

    # def integrate(
    #     self,
    #     low: ArrayLike,
    #     high: ArrayLike,
    #     center: float,
    #     params: dict[str, ArrayLike],
    #     obs_low: float,
    #     obs_high: float,
    #     lsf_fwhm: ArrayLike = 0.0,
    # ) -> Array:
    #     # Exact integral of scale * (wavelength / nw)^beta over [low, high]:
    #     # = scale / nw^beta * [w^{beta+1} / (beta+1)]_{low}^{high} / (high - low)
    #     nw = params['norm_wav']
    #     beta = params['beta']
    #     bp1 = beta + 1.0
    #     # For beta != -1 (the common case): use the power-rule antiderivative.
    #     # beta = -1 gives log, but that is physically unusual; we handle it
    #     # via jnp.where for safety.
    #     antideriv_high = high**bp1 / bp1
    #     antideriv_low = low**bp1 / bp1
    #     power_avg = (antideriv_high - antideriv_low) / (high - low)
    #     # beta = -1 fallback: integral of 1/w is ln(w)
    #     log_avg = (jnp.log(high) - jnp.log(low)) / (high - low)
    #     avg = jnp.where(jnp.abs(bp1) > 1e-10, power_avg, log_avg)
    #     return params['scale'] / nw**beta * avg

    def to_dict(self) -> dict:
        return {'type': 'PowerLaw'}

    @classmethod
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

    def param_names(self) -> tuple[str, ...]:
        return ('scale', 'temperature', 'norm_wav')

    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        return {
            'scale': Uniform(0, 2),
            'temperature': Uniform(100, 50000),
            'norm_wav': Fixed(region_center),
        }

    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        return {
            'scale': (True, flux_unit),
            'temperature': (False, u.K),
            'norm_wav': (False, wl_unit),
        }

    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
    ) -> Array:
        # LSF convolution is not supported for Blackbody.
        wl_um = wavelength * self._micron_factor
        nw_um = params['norm_wav'] * self._micron_factor
        bb = planck_function(wl_um, params['temperature'], nw_um)
        return params['scale'] * bb

    def _prepare(self, low: u.Quantity, high: u.Quantity) -> None:
        """Compute the micron conversion factor for the region's wavelength unit."""
        self._micron_factor = _get_conversion_factor(low.unit, u.um)

    def to_dict(self) -> dict:
        return {'type': 'Blackbody'}

    @classmethod
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

    def param_names(self) -> tuple[str, ...]:
        return ('scale', 'temperature', 'beta', 'norm_wav')

    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        return {
            'scale': Uniform(0, 2),
            'temperature': Uniform(100, 50000),
            'beta': Uniform(-4, 4),
            'norm_wav': Fixed(region_center),
        }

    def param_units(
        self, flux_unit: u.UnitBase, wl_unit: u.UnitBase
    ) -> dict[str, tuple[bool, u.UnitBase | None]]:
        return {
            'scale': (True, flux_unit),
            'temperature': (False, u.K),
            'beta': (False, None),
            'norm_wav': (False, wl_unit),
        }

    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
    ) -> Array:
        # LSF convolution is not supported for ModifiedBlackbody.
        wl_um = wavelength * self._micron_factor
        nw_um = params['norm_wav'] * self._micron_factor
        bb = planck_function(wl_um, params['temperature'], nw_um)
        modifier = (wl_um / nw_um) ** params['beta']
        return params['scale'] * bb * modifier

    def _prepare(self, low: u.Quantity, high: u.Quantity) -> None:
        """Compute the micron conversion factor for the region's wavelength unit."""
        self._micron_factor = _get_conversion_factor(low.unit, u.um)

    def to_dict(self) -> dict:
        return {'type': 'ModifiedBlackbody'}

    @classmethod
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

    def param_names(self) -> tuple[str, ...]:
        return ('scale', 'temperature', 'tau_v', 'alpha', 'norm_wav')

    def default_priors(self, region_center: float = 1.0) -> dict[str, Prior]:
        return {
            'scale': Uniform(0, 2),
            'temperature': Uniform(100, 50000),
            'tau_v': Uniform(0, 5),
            'alpha': Uniform(-2, 0),
            'norm_wav': Fixed(region_center),
        }

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

    def evaluate(
        self,
        wavelength: ArrayLike,
        center: float,
        params: dict[str, ArrayLike],
        obs_low: float,
        obs_high: float,
        lsf_fwhm: ArrayLike = 0.0,
    ) -> Array:
        # LSF convolution is not supported for AttenuatedBlackbody.
        wl_um = wavelength * self._micron_factor
        nw_um = params['norm_wav'] * self._micron_factor
        bb = planck_function(wl_um, params['temperature'], nw_um)
        ext_data = (wl_um / self._lambda_ext_um) ** params['alpha']
        ext_pivot = (nw_um / self._lambda_ext_um) ** params['alpha']
        extinction = jnp.exp(-params['tau_v'] * (ext_data - ext_pivot))
        return params['scale'] * bb * extinction

    def _prepare(self, low: u.Quantity, high: u.Quantity) -> None:
        """Compute the conversion factors for the region's wavelength unit."""
        self._micron_factor = _get_conversion_factor(low.unit, u.um)

    def to_dict(self) -> dict:
        return {'type': 'AttenuatedBlackbody', 'lambda_ext': self.lambda_ext}

    @classmethod
    def from_dict(cls, d: dict) -> AttenuatedBlackbody:
        return cls(lambda_ext=d.get('lambda_ext', 0.55))

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return NotImplemented
        return self.lambda_ext == other.lambda_ext  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(('AttenuatedBlackbody', self.lambda_ext))

    def __repr__(self) -> str:
        return f'AttenuatedBlackbody(lambda_ext={self.lambda_ext})'
