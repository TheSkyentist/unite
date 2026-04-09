"""Model builder and numpyro model function for spectral line fitting.

The :class:`ModelBuilder` assembles a :class:`~unite.line.config.LineConfiguration`,
an optional :class:`~unite.continuum.config.ContinuumConfiguration`, and a
:class:`~unite.spectrum.spectrum.Spectra` collection into a numpyro model
function that can be passed to any numpyro inference algorithm (NUTS, SVI, etc.).
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import jax
import jax.numpy as jnp
import numpyro
from astropy import units as u
from numpyro import distributions as dist

from unite._compose import compose_from_profiles
from unite._utils import C_KMS, _get_conversion_factor
from unite.continuum.compute import eval_continuum, integrate_continuum
from unite.continuum.config import ContinuumConfiguration
from unite.line.compute import evaluate_lines, integrate_lines
from unite.line.config import ConfigMatrices, LineConfiguration
from unite.prior import Fixed, Parameter, Prior, topological_sort
from unite.spectrum import Spectra, Spectrum

# Conversion factor from FWHM to Gaussian sigma: 1 / (2 * sqrt(2 * ln 2)).
_FWHM_TO_SIGMA: float = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))

# ------------------------------------------------------------------
# ModelArgs — data bundle for the numpyro model function
# ------------------------------------------------------------------


@dataclass
class ModelArgs:
    """Bundle of arguments passed to :func:`unite_model`.

    Created by :meth:`ModelBuilder.build`; not intended for direct
    construction by users.
    """

    #: Precomputed parameter matrices and line metadata.
    matrices: ConfigMatrices
    #: Individual spectra.
    spectra: list[Spectrum]
    #: Systemic redshift.
    redshift: float
    #: Continuum configuration, or ``None`` if not used.
    cont_config: ContinuumConfiguration | None
    #: Resolved ``{param_name: ContinuumParam}`` mappings per region, from :attr:`ContinuumConfiguration.resolved_params`.
    cont_resolved_params: list[dict[str, Parameter]] | None
    #: All parameters with their priors (line, calibration, continuum).
    all_priors: dict[str, Prior]
    #: Topological sampling order for all parameters.
    dependency_order: list[str]
    name_to_token: dict[str, object]
    # --- Wavelength unit conversion ---
    spec_to_canonical: list[float]
    # --- Pre-converted continuum bounds (rest-frame, canonical unit) ---
    cont_low: list[float] | None
    cont_high: list[float] | None
    cont_center: list[float] | None
    # --- Unit conversion factors: region.unit → canonical unit, per region ---
    cont_nw_conv: list[float] | None
    #: Pre-converted continuum forms (static wavelength config in canonical unit).
    cont_forms: list | None
    # --- Flux normalization ---
    norm_factors: list[float]
    #: Per-spectrum line flux scale (in each spectrum's flux_unit * canonical_wl_unit).
    line_flux_scales: list[float]
    #: Per-spectrum continuum scale (in each spectrum's flux_unit).
    continuum_scales: list[float]
    #: Wavelength unit of canonical frame (first spectrum's disperser unit).
    canonical_unit: object
    #: Per-spectrum flux density units.
    flux_units: list
    #: The Quantity line_scale and continuum_scale from Spectra (for results output).
    line_scale_quantity: u.Quantity | None
    continuum_scale_quantity: u.Quantity | None
    #: Human-readable column labels for each line, parallel to ``matrices.wavelengths``.
    #: Derived from user-supplied line names and rest-frame wavelengths.
    line_labels: list[str]
    #: Human-readable column labels for each continuum region, parallel to ``cont_config``.
    #: Derived from form type and wavelength bounds.
    continuum_labels: list[str]
    #: Absorber placement relative to emission and continuum sources.
    #: One of ``'foreground'``, ``'behind_lines'``, or ``'behind_continuum'``.
    absorber_position: str = 'foreground'
    #: Line integration mode: ``'analytic'`` (default) uses exact CDF-based
    #: integration for all line profiles individually;
    #: ``'quadrature'`` uses Gauss-Legendre quadrature to integrate the full
    #: composed model over pixels.
    integration_mode: str = 'analytic'
    #: Gauss-Legendre quadrature nodes on ``[-1, 1]``.
    #: ``None`` when ``integration_mode != 'quadrature'``.
    quadrature_nodes: jnp.ndarray | None = None
    #: Gauss-Legendre quadrature weights.
    #: ``None`` when ``integration_mode != 'quadrature'``.
    quadrature_weights: jnp.ndarray | None = None
    #: Number of uniform sub-pixel evaluation points per pixel for convolution mode.
    #: ``None`` when ``integration_mode != 'convolution'``.
    n_super: int | None = None
    #: Half-width of the banded LSF convolution kernel in fine-grid indices.
    #: Pre-computed at build time as a Python ``int`` (not a traced value).
    #: ``None`` when ``integration_mode != 'convolution'``.
    conv_half_width: int | None = None

    def __len__(self) -> int:
        """Return the number of spectra in the model."""
        return len(self.spectra)

    def __bool__(self) -> bool:
        """Return True if the model has at least one spectrum."""
        return len(self.spectra) > 0


# ------------------------------------------------------------------
# Numpyro model function
# ------------------------------------------------------------------


def unite_model(args: ModelArgs) -> None:
    """Numpyro model function for multi-spectrum emission-line fitting.

    All lines are integrated simultaneously via :func:`jax.vmap` with
    ``lax.switch`` dispatching to the correct profile kernel per line.
    Parameter broadcasting from unique tokens to per-line arrays is done
    with precomputed indicator matrices.

    Wavelength unit conversion is handled via pre-computed scalar factors
    stored in ``args.spec_to_canonical`` (one per spectrum).  Flux is
    normalized per spectrum so that the likelihood operates on O(1) values.

    Parameters
    ----------
    args : ModelArgs
        Pre-built data bundle from :meth:`ModelBuilder.build`.
    """
    cm = args.matrices
    z_sys = args.redshift
    n_lines = cm.wavelengths.shape[0]

    # --- 1. Sample all parameters in dependency order ---
    # Two parallel dicts are maintained:
    #   context    — str → value, used for all downstream name-based lookups.
    #   obj_ctx    — token_object → value, passed to prior.to_dist() so that
    #                parameter expressions can look up dependency values by
    #                token identity.
    context: dict[str, jnp.ndarray] = {}
    obj_ctx: dict[object, jnp.ndarray] = {}
    for pname in args.dependency_order:
        prior = args.all_priors[pname]
        if isinstance(prior, Fixed):
            val: jnp.ndarray = jnp.asarray(prior.resolved_value(obj_ctx))
        else:
            distribution = prior.to_dist(obj_ctx)
            assert distribution is not None
            val = cast(jnp.ndarray, numpyro.sample(pname, distribution))
        context[pname] = val
        tok = args.name_to_token.get(pname)
        if tok is not None:
            obj_ctx[tok] = val

    # --- 2. Per-line parameter arrays via matrix products ---
    # Flux: include multiplet strengths (emission lines only; absorption lines have 0).
    if cm.flux_names:
        flux_vec = jnp.stack([context[n] for n in cm.flux_names])
        flux_per_line = flux_vec @ cm.flux_matrix * cm.strengths  # (n_lines,)
    else:
        flux_per_line = jnp.zeros(n_lines)

    # Tau: optical depths for absorption lines (emission lines have 0).
    if cm.tau_names:
        tau_vec = jnp.stack([context[n] for n in cm.tau_names])
        tau_per_line = tau_vec @ cm.tau_matrix  # (n_lines,)
    else:
        tau_per_line = jnp.zeros(n_lines)

    # Redshift (delta from systemic).
    z_vec = jnp.stack([context[n] for n in cm.z_names])
    z_per_line = z_vec @ cm.z_matrix  # (n_lines,)

    # Observed-frame centers (in canonical wavelength unit).
    centers = cm.wavelengths * (1.0 + z_sys + z_per_line)  # (n_lines,)

    # Slot 0: primary velocity FWHM → canonical wavelength units.
    if cm.p0_names:
        p0_vec = jnp.stack([context[n] for n in cm.p0_names])
        p0_kms = p0_vec @ cm.p0_matrix  # (n_lines,)
    else:
        p0_kms = jnp.zeros(n_lines)
    p0 = centers * p0_kms / C_KMS  # canonical wavelength units

    # Slot 1: velocity FWHMs (p1v) + dimensionless params (p1d), summed.
    if cm.p1v_names:
        p1v_vec = jnp.stack([context[n] for n in cm.p1v_names])
        p1v_kms = p1v_vec @ cm.p1v_matrix  # (n_lines,)
    else:
        p1v_kms = jnp.zeros(n_lines)
    p1v = centers * p1v_kms / C_KMS

    if cm.p1d_names:
        p1d_vec = jnp.stack([context[n] for n in cm.p1d_names])
        p1d = p1d_vec @ cm.p1d_matrix  # (n_lines,), dimensionless
    else:
        p1d = jnp.zeros(n_lines)
    p1 = p1v + p1d  # for any line, only one sub-matrix is nonzero

    # Slot 2: dimensionless params only.
    if cm.p2_names:
        p2_vec = jnp.stack([context[n] for n in cm.p2_names])
        p2 = p2_vec @ cm.p2_matrix  # (n_lines,)
    else:
        p2 = jnp.zeros(n_lines)

    # --- 3. Per-spectrum likelihood ---
    for i, spectrum in enumerate(args.spectra):
        disp = spectrum.disperser
        wl_scale = args.spec_to_canonical[i]
        inv_wl_scale = 1.0 / wl_scale
        norm = args.norm_factors[i]
        lfs = args.line_flux_scales[i]
        cs = args.continuum_scales[i]

        # Calibration values (fall back to identity when no token is attached).
        r_scale = (
            context[cast(str, disp.r_scale.name)] if disp.r_scale is not None else 1.0
        )
        flux_scale = (
            context[cast(str, disp.flux_scale.name)]
            if disp.flux_scale is not None
            else 1.0
        )
        pix_offset = (
            context[cast(str, disp.pix_offset.name)]
            if disp.pix_offset is not None
            else 0.0
        )

        # Pixel edges in canonical wavelength unit.
        low = spectrum.low * wl_scale
        high = spectrum.high * wl_scale
        if disp.pix_offset is not None:
            mid_disp = (spectrum.low + spectrum.high) / 2.0  # disperser unit
            shift = pix_offset * disp.dlam_dpix(mid_disp) * wl_scale
            low = low + shift
            high = high + shift

        n_pixels = low.shape[0]

        # LSF FWHM at each line center (n_lines,).
        # R() expects disperser-unit wavelengths, result is in canonical unit.
        lsf_fwhm = centers / (disp.R(centers * inv_wl_scale) * r_scale)

        # LSF FWHM at pixel centres for continuum convolution.
        pix_mid = (low + high) / 2.0
        cont_lsf_fwhm = pix_mid / (disp.R(pix_mid * inv_wl_scale) * r_scale)

        # Scaled line fluxes and continuum for this spectrum.
        scaled_flux = flux_per_line * lfs / norm
        cont_scale_norm = cs / norm

        # --- Compute pixel-averaged model ---
        if args.integration_mode == 'quadrature':
            # Gauss-Legendre quadrature: evaluate full composed model at
            # sub-pixel nodes and integrate.  This properly computes
            # ∫ F(λ) · exp(-τ·φ(λ)) dλ over each pixel.
            mid = (low + high) / 2.0
            half_width = (high - low) / 2.0
            nodes = args.quadrature_nodes  # (n_nodes,)
            weights = args.quadrature_weights  # (n_nodes,)
            assert nodes is not None
            assert weights is not None

            # Map GL nodes to pixel sub-points: (n_nodes, n_pixels)
            x = mid[None, :] + half_width[None, :] * nodes[:, None]

            # Evaluate full model at each set of node points and integrate.
            # Default keyword args bind per-spectrum values at definition
            # time to avoid late-binding closure issues (ruff B023).
            def _glq_eval(
                wav,
                *,
                _lsf=lsf_fwhm,
                _scaled_flux=scaled_flux,
                _csn=cont_scale_norm,
                _fs=flux_scale,
                _inv_wl=inv_wl_scale,
                _rs=r_scale,
                _disp=disp,
            ):
                phi = evaluate_lines(wav, centers, _lsf, p0, p1, p2, cm.profile_codes)
                # LSF FWHM at quadrature node wavelengths for continuum.
                node_lsf = wav / (_disp.R(wav * _inv_wl) * _rs)
                cont = eval_continuum(wav, args, context, z_sys, node_lsf) * _csn
                total = compose_from_profiles(
                    phi,
                    _scaled_flux,
                    tau_per_line,
                    cm.is_absorption,
                    cont,
                    args.absorber_position,
                )
                return _fs * total

            model_at_nodes = jax.vmap(_glq_eval)(x)  # (n_nodes, n_pix)
            model = 0.5 * jnp.dot(weights, model_at_nodes)  # pixel-averaged
        elif args.integration_mode == 'convolution':
            # Numerical LSF convolution: evaluate the intrinsic model (lsf_fwhm=0)
            # on a uniform fine sub-pixel grid, convolve with the wavelength-
            # dependent Gaussian LSF, then pixel-average.  This correctly computes
            # LSF ⊗ [F · exp(-τ · φ_intrinsic)] rather than F · exp(-τ · LSF ⊗ φ).
            from unite._lsf import _lsf_convolve

            n_super = args.n_super
            half_width = args.conv_half_width
            assert n_super is not None
            assert half_width is not None

            # Fine grid: n_super uniform points per pixel, midpoints of sub-bins.
            # Shape: (n_super, n_pixels), flattened to (n_super * n_pixels,).
            offsets = (jnp.arange(n_super) + 0.5) / n_super  # (n_super,)
            x_fine = low[None, :] + offsets[:, None] * (high - low)[None, :]
            x_flat = x_fine.ravel()  # (n_super * n_pixels,)

            # Evaluate intrinsic model (lsf_fwhm=0) on fine grid.
            # _combine_fwhm(0, fwhm) = fwhm, so intrinsic profiles are used.
            zero_lsf = jnp.zeros_like(centers)

            def _conv_eval(
                wav,
                *,
                _zero_lsf=zero_lsf,
                _scaled_flux=scaled_flux,
                _csn=cont_scale_norm,
            ):
                phi = evaluate_lines(
                    wav, centers, _zero_lsf, p0, p1, p2, cm.profile_codes
                )
                cont = eval_continuum(wav, args, context, z_sys, 0.0) * _csn
                return compose_from_profiles(
                    phi,
                    _scaled_flux,
                    tau_per_line,
                    cm.is_absorption,
                    cont,
                    args.absorber_position,
                )

            model_fine_intrinsic = _conv_eval(x_flat)  # (n_super * n_pixels,)

            # Compute wavelength-varying LSF sigma at each fine-grid point.
            lsf_fwhm_fine = x_flat / (disp.R(x_flat * inv_wl_scale) * r_scale)
            sigma_fine = lsf_fwhm_fine * _FWHM_TO_SIGMA

            # Convolve intrinsic model with the spatially-varying Gaussian LSF.
            model_conv = _lsf_convolve(
                x_flat, model_fine_intrinsic, sigma_fine, half_width
            )

            # Apply flux_scale and pixel-average: reshape → (n_super, n_pixels).
            model = flux_scale * model_conv.reshape(n_super, n_pixels).mean(axis=0)
        else:
            # Analytic: integrate each line profile individually via CDF,
            # then combine.  Exact for flux-parametrized lines; approximate
            # for tau-parametrized lines (integrates phi before applying exp).
            pixints = integrate_lines(
                low, high, centers, lsf_fwhm, p0, p1, p2, cm.profile_codes
            ) / (high - low)
            cont = (
                integrate_continuum(low, high, args, context, z_sys, cont_lsf_fwhm)
                * cont_scale_norm
            )
            model = flux_scale * compose_from_profiles(
                pixints,
                scaled_flux,
                tau_per_line,
                cm.is_absorption,
                cont,
                args.absorber_position,
            )
        obs_name = f'obs_{spectrum.name}' if spectrum.name else f'obs_{i}'
        numpyro.sample(
            obs_name,
            dist.Normal(model, spectrum.scaled_error / norm),
            obs=spectrum.flux / norm,
        )


# ------------------------------------------------------------------
# ModelBuilder
# ------------------------------------------------------------------


class ModelBuilder:
    """Assemble configuration objects into a numpyro model.

    Collects all unique parameter tokens (line, calibration, continuum),
    builds precomputed indicator matrices, performs a topological sort for
    dependency resolution, and packages everything into a
    ``(model_fn, model_args)`` pair.

    Parameters
    ----------
    line_config : LineConfiguration
        Emission/absorption line configuration.
    continuum_config : ContinuumConfiguration or None
        Continuum configuration.  ``None`` for a lines-only model.
    spectra : Spectra
        Spectrum collection with systemic redshift.

    Examples
    --------
    >>> model_fn, args = ModelBuilder(line_config, cont, spectra).build()
    >>> kernel = numpyro.infer.NUTS(model_fn)
    >>> mcmc = numpyro.infer.MCMC(kernel, num_warmup=500, num_samples=1000)
    >>> mcmc.run(jax.random.PRNGKey(0), args)
    """

    def __init__(
        self,
        line_config: LineConfiguration,
        continuum_config: ContinuumConfiguration | None,
        spectra: Spectra,
    ) -> None:
        self._spectra = spectra

        # --- Auto-prepare if needed ---
        if not spectra.is_prepared:
            warnings.warn(
                'Spectra not prepared; calling spectra.prepare() with defaults. '
                'Call spectra.prepare(line_config, continuum_config) explicitly '
                'for full control.',
                UserWarning,
                stacklevel=2,
            )
            line_config, continuum_config = spectra.prepare(
                line_config, continuum_config
            )
        else:
            # Use the prepared configs.
            prepared_lc = spectra.prepared_line_config
            assert prepared_lc is not None, (
                'prepared_line_config is None despite is_prepared=True'
            )
            line_config = prepared_lc
            continuum_config = spectra.prepared_cont_config

        # --- Auto-compute scales if needed ---
        if spectra.line_scale is None:
            warnings.warn(
                'Line scale not set; calling spectra.compute_scales() with '
                'defaults. Call spectra.compute_scales(line_config, '
                'continuum_config) explicitly for full control.',
                UserWarning,
                stacklevel=2,
            )
            spectra.compute_scales(line_config, continuum_config)

        self._cont_config = continuum_config

        # --- Canonical wavelength unit: use Spectra's canonical_unit ---
        self._canonical_unit = spectra.canonical_unit

        # Build precomputed matrices from line entries.
        self._matrices = line_config.build_matrices()

        # Convert line wavelengths to canonical unit.
        # Each line may have a different Quantity unit, so convert per-line.
        if len(line_config) > 0:
            canon_wls = jnp.array(
                [
                    float(e.wavelength.to(self._canonical_unit).value)
                    for e in line_config._entries
                ]
            )
            self._matrices.wavelengths = canon_wls

        # --- Collect all unique parameter tokens for prior / topo-sort ---
        all_priors: dict[str, Prior] = dict(self._matrices.priors)
        param_to_name: dict[object, str] = {
            # We can reconstruct token→name from the matrices' name lists and
            # the original entries since tokens carry their .name attribute.
            tok: cast(str, tok.name)
            for entry in line_config._entries
            for tok in (entry.flux, entry.tau, entry.redshift, *entry.fwhms.values())
            if tok is not None
        }

        # Calibration tokens from each unique disperser.
        seen_dispersers: set[int] = set()
        seen_tok_ids: set[int] = set(id(t) for t in param_to_name)
        for spectrum in spectra:
            disp = spectrum.disperser
            if id(disp) not in seen_dispersers:
                seen_dispersers.add(id(disp))
                for tok in (disp.r_scale, disp.flux_scale, disp.pix_offset):
                    if tok is not None and id(tok) not in seen_tok_ids:
                        tok_name = cast(str, tok.name)
                        seen_tok_ids.add(id(tok))
                        all_priors[tok_name] = tok.prior
                        param_to_name[tok] = tok_name

        # Continuum parameters: collect unique ContinuumParam tokens by identity.
        # Shared tokens (same object in multiple regions) produce one numpyro site.
        if continuum_config is not None:
            seen_cont_ids: set[int] = set()
            for resolved in continuum_config.resolved_params:
                for tok in resolved.values():
                    if id(tok) not in seen_cont_ids:
                        tok_name = cast(str, tok.name)
                        seen_cont_ids.add(id(tok))
                        all_priors[tok_name] = tok.prior
                        param_to_name[tok] = tok_name

        self._all_priors = all_priors
        self._dep_order = (
            topological_sort(all_priors, param_to_name) if all_priors else []
        )
        # Reverse mapping: site name → token object, for obj_ctx in unite_model.
        # Continuum params have no token objects and are intentionally absent.
        self._line_config = line_config
        self._name_to_token: dict[str, object] = {
            name: tok for tok, name in param_to_name.items()
        }

    @property
    def matrices(self) -> ConfigMatrices:
        """Precomputed matrices (after coverage filtering)."""
        return self._matrices

    def build(
        self,
        *,
        absorber_position: str = 'foreground',
        integration_mode: str = 'analytic',
        n_nodes: int = 7,
        n_super: int = 10,
        conv_half_width: int | None = None,
    ) -> tuple[Callable, ModelArgs]:
        """Build the numpyro model function and its arguments.

        Parameters
        ----------
        absorber_position : str, optional
            Where the absorber sits relative to emission lines and
            continuum.  One of:

            * ``'foreground'`` (default) — absorbs both emission and
              continuum.
            * ``'behind_lines'`` — absorbs only the continuum (the
              absorber is between the continuum source and the emission
              region).
            * ``'behind_continuum'`` — absorbs only emission lines (the
              absorber is between the emission region and the observer,
              but behind the continuum source).

        integration_mode : str, optional
            How line profiles are integrated over pixels.  One of:

            * ``'analytic'`` (default) — exact CDF-based integration
              for emission profiles and pixel-center evaluation for
              absorption profiles.
            * ``'quadrature'`` — Gauss-Legendre quadrature for all
              profiles (both emission and absorption).  More accurate
              for absorption lines at the cost of speed.
            * ``'convolution'`` — evaluates the intrinsic model
              (``lsf_fwhm=0``) on a uniform fine sub-pixel grid of
              ``n_super`` points per pixel, numerically convolves with
              the wavelength-dependent Gaussian LSF, then pixel-averages.
              Correctly computes ``LSF ⊗ [F · exp(-τ · φ_intrinsic)]``
              rather than ``F · exp(-τ · LSF ⊗ φ)``, eliminating the
              LSF pre-convolution approximation for absorption lines.

        n_nodes : int, optional
            Number of Gauss-Legendre quadrature nodes per pixel
            (default: 7).  Only used when ``integration_mode='quadrature'``.
            Higher values give more accurate integration at greater
            computational cost.
        n_super : int, optional
            Number of uniform sub-pixel evaluation points per pixel
            (default: 10).  Only used when
            ``integration_mode='convolution'``.  Higher values resolve
            narrower intrinsic line profiles at the cost of speed.
            ``n_super=10`` is adequate for NIRSpec gratings; increase to
            20 for narrow absorbers at PRISM resolution.
        conv_half_width : int or None, optional
            Half-width of the banded LSF convolution kernel in fine-grid
            indices (default: ``None``).  When ``None``, auto-computed at
            build time as ``ceil(4 * max_sigma / min_dx_fine * 1.5)``
            where ``max_sigma`` is the largest LSF sigma across all
            spectra and ``min_dx_fine`` is the finest sub-pixel spacing.
            Only used when ``integration_mode='convolution'``.

        Returns
        -------
        model_fn : callable
            The numpyro model function (signature: ``model_fn(args)``).
        model_args : ModelArgs
            Pre-built data bundle to pass to the model function.

        Raises
        ------
        ValueError
            If *absorber_position* or *integration_mode* is not one of
            the valid values.
        """
        valid_positions = ('foreground', 'behind_lines', 'behind_continuum')
        if absorber_position not in valid_positions:
            msg = (
                f'absorber_position must be one of {valid_positions}, '
                f'got {absorber_position!r}.'
            )
            raise ValueError(msg)

        valid_modes = ('analytic', 'quadrature', 'convolution')
        if integration_mode not in valid_modes:
            msg = (
                f'integration_mode must be one of {valid_modes}, '
                f'got {integration_mode!r}.'
            )
            raise ValueError(msg)

        # Pre-compute Gauss-Legendre nodes and weights if needed.
        if integration_mode == 'quadrature':
            from numpy.polynomial.legendre import leggauss

            gl_nodes, gl_weights = leggauss(n_nodes)
            quadrature_nodes = jnp.asarray(gl_nodes)
            quadrature_weights = jnp.asarray(gl_weights)
        else:
            quadrature_nodes = None
            quadrature_weights = None

        # Trim spectra to union of continuum regions (observed frame).
        # Pixels outside all regions have model = 0 and would corrupt the
        # likelihood if observed flux is nonzero.  Trimming also reduces
        # array sizes passed to JAX.
        z = self._spectra.redshift
        if self._cont_config is not None:
            trimmed_spectra: list[Spectrum] = []
            for s in self._spectra:
                mask = jnp.zeros(s.npix, dtype=bool)
                for region in self._cont_config:
                    conv = _get_conversion_factor(region.unit, s.unit)
                    obs_low = region.low * conv * (1.0 + z)
                    obs_high = region.high * conv * (1.0 + z)
                    mask = mask | s.pixel_mask(obs_low, obs_high)
                trimmed = s._sliced(mask)
                if trimmed.npix > 0:
                    trimmed_spectra.append(trimmed)
                else:
                    warnings.warn(
                        f'Spectrum {s.name!r} has no pixels overlapping '
                        f'any continuum region and will be excluded from '
                        f'the fit.',
                        UserWarning,
                        stacklevel=2,
                    )
            if not trimmed_spectra:
                warnings.warn(
                    'All spectra are fully masked after trimming to '
                    'continuum regions. The resulting model has no spectra '
                    'and should not be used for fitting. Check that your '
                    'continuum configuration covers the observed wavelength '
                    'range.',
                    UserWarning,
                    stacklevel=2,
                )
        else:
            trimmed_spectra = list(self._spectra)

        # For convolution mode, compute the kernel half-width (in fine-grid indices)
        # from the maximum LSF sigma and minimum fine-grid spacing across all spectra.
        if integration_mode == 'convolution':
            if conv_half_width is None:
                max_lsf_fwhm = 0.0
                min_dx_fine = float('inf')
                for s in trimmed_spectra:
                    wl_scale = _get_conversion_factor(s.unit, self._canonical_unit)
                    pix_mid_disp = (s.low + s.high) / 2.0
                    r_arr = jnp.asarray(s.disperser.R(pix_mid_disp))
                    lsf = jnp.asarray(pix_mid_disp * wl_scale) / r_arr
                    max_lsf_fwhm = max(max_lsf_fwhm, float(jnp.max(lsf)))
                    pix_widths = (s.high - s.low) * wl_scale
                    min_dx_fine = min(min_dx_fine, float(jnp.min(pix_widths)) / n_super)
                max_sigma = max_lsf_fwhm * _FWHM_TO_SIGMA
                conv_half_width = max(1, math.ceil(4.0 * max_sigma / min_dx_fine * 1.5))
            n_super_val: int | None = n_super
            conv_half_width_val: int | None = conv_half_width
        else:
            n_super_val = None
            conv_half_width_val = None

        # Per-spectrum wavelength conversion factors.
        spec_to_canonical = [
            _get_conversion_factor(s.unit, self._canonical_unit)
            for s in trimmed_spectra
        ]

        # Per-spectrum flux normalization.
        norm_factors = [_compute_norm_factor(s) for s in trimmed_spectra]

        # Line flux scale (Quantity from Spectra.compute_scales).
        line_scale_qty = self._spectra.line_scale
        # Continuum scale (Quantity from Spectra.compute_scales; fallback).
        cont_scale_qty = self._spectra.continuum_scale

        # Convert Quantity scales to per-spectrum float values.
        # line_flux_scale needs units of [flux_density * canonical_wl_unit]
        # for each spectrum.
        # continuum_scale needs units of [flux_density] for each spectrum.
        canonical_unit = self._canonical_unit
        line_flux_scales: list[float] = []
        continuum_scales: list[float] = []
        assert line_scale_qty is not None, (
            'line_scale must be set before building the model'
        )
        for s in trimmed_spectra:
            target_line_unit = s.flux_unit * canonical_unit
            lfs = float(line_scale_qty.to(target_line_unit).value)
            line_flux_scales.append(lfs)

            if cont_scale_qty is not None:
                cs = float(cont_scale_qty.to(s.flux_unit).value)
            else:
                cs = 1.0
            continuum_scales.append(cs)

        flux_units = [s.flux_unit for s in trimmed_spectra]

        # Pre-convert continuum region bounds to canonical unit.
        if self._cont_config is not None:
            cont_low = []
            cont_high = []
            cont_center = []
            cont_nw_conv = []
            cont_forms = []
            for region in self._cont_config:
                conv = _get_conversion_factor(region.unit, self._canonical_unit)
                cont_low.append(region.low * conv)
                cont_high.append(region.high * conv)
                cont_center.append(region.center * conv)
                cont_nw_conv.append(conv)
                cont_forms.append(region.form)
        else:
            cont_low = None
            cont_high = None
            cont_center = None
            cont_nw_conv = None
            cont_forms = None

        line_labels = _make_line_labels(self._line_config)
        continuum_labels = (
            _make_continuum_labels(self._cont_config)
            if self._cont_config is not None
            else []
        )

        args = ModelArgs(
            matrices=self._matrices,
            spectra=trimmed_spectra,
            redshift=self._spectra.redshift,
            cont_config=self._cont_config,
            cont_resolved_params=(
                self._cont_config.resolved_params
                if self._cont_config is not None
                else None
            ),
            all_priors=self._all_priors,
            dependency_order=self._dep_order,
            name_to_token=self._name_to_token,
            spec_to_canonical=spec_to_canonical,
            cont_low=cont_low,
            cont_high=cont_high,
            cont_center=cont_center,
            cont_nw_conv=cont_nw_conv,
            cont_forms=cont_forms,
            norm_factors=norm_factors,
            line_flux_scales=line_flux_scales,
            continuum_scales=continuum_scales,
            canonical_unit=canonical_unit,
            flux_units=flux_units,
            line_scale_quantity=line_scale_qty,
            continuum_scale_quantity=cont_scale_qty,
            line_labels=line_labels,
            continuum_labels=continuum_labels,
            absorber_position=absorber_position,
            integration_mode=integration_mode,
            quadrature_nodes=quadrature_nodes,
            quadrature_weights=quadrature_weights,
            n_super=n_super_val,
            conv_half_width=conv_half_width_val,
        )
        return unite_model, args

    def fit(
        self,
        num_warmup: int = 250,
        num_samples: int = 1000,
        num_chains: int = 1,
        seed: int = 0,
        progress_bar: bool = True,
        absorber_position: str = 'foreground',
        integration_mode: str = 'analytic',
        n_nodes: int = 7,
        n_super: int = 10,
    ) -> tuple[dict, ModelArgs]:
        """Fit the model using NUTS sampling (convenience wrapper).

        This method builds the model, runs MCMC with the NUTS kernel, and
        returns the posterior samples. For more control over the sampler
        (e.g., custom kernel, SVI, nested sampling), call :meth:`build`
        directly and use numpyro's inference APIs.

        Parameters
        ----------
        num_warmup : int, optional
            Number of warmup iterations per chain (default: 1000).
        num_samples : int, optional
            Number of posterior samples per chain (default: 1000).
        num_chains : int, optional
            Number of MCMC chains to run in parallel (default: 1).
        seed : int, optional
            Random seed for JAX's PRNG (default: 0).
        progress_bar : bool, optional
            Whether to display a progress bar (default: True).
        absorber_position : str, optional
            Where the absorber sits relative to emission lines and
            continuum (default: ``'foreground'``).  See
            :meth:`build` for details.
        integration_mode : str, optional
            Line integration mode (default: ``'analytic'``).  See
            :meth:`build` for details.
        n_nodes : int, optional
            Gauss-Legendre quadrature nodes per pixel (default: 7).
            See :meth:`build` for details.
        n_super : int, optional
            Sub-pixel evaluation points per pixel for convolution mode
            (default: 10).  See :meth:`build` for details.

        Returns
        -------
        tuple
            ``(samples, model_args)`` where ``samples`` is a dictionary with
            parameter names as keys and shape ``(num_chains, num_samples)`` per
            parameter, and ``model_args`` is the :class:`ModelArgs` bundle.

        Examples
        --------
        >>> samples, model_args = builder.fit(num_warmup=200, num_samples=500, num_chains=4)
        """
        import jax
        from numpyro import infer

        model_fn, model_args = self.build(
            absorber_position=absorber_position,
            integration_mode=integration_mode,
            n_nodes=n_nodes,
            n_super=n_super,
        )
        mcmc = infer.MCMC(
            infer.NUTS(model_fn, dense_mass=True),
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )
        mcmc.run(jax.random.PRNGKey(seed), model_args)
        samples = mcmc.get_samples()
        return samples, model_args


# ------------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------------


def _make_line_labels(line_config: LineConfiguration) -> list[str]:
    """Return the unique name for every line entry.

    Since line names are required to be unique within a
    :class:`~unite.line.config.LineConfiguration`, the label for each entry
    is simply its name.  Use ``add_lines`` (which auto-generates names as
    ``'{name}_{center.value:g}'``) or supply explicit unique names to
    ``add_line`` for multiplets and multi-component lines.
    """
    return [entry.name for entry in line_config._entries]


def _make_continuum_labels(cont_config: ContinuumConfiguration) -> list[str]:
    """Build a human-readable label for every continuum region.

    Format: ``'{form_type}_{low:.4g}_{high:.4g}'`` where the wavelength
    values are in the region's native unit and ``form_type`` is the
    lower-cased class name of the functional form (e.g. ``'linear'``,
    ``'powerlaw'``, ``'polynomial'``).

    Examples: ``'linear_6400_6700'``, ``'powerlaw_0.95_2.5'``.
    """
    labels: list[str] = []
    for region in cont_config:
        form_type = type(region.form).__name__.lower()
        low_str = f'{region.low:.4g}'
        high_str = f'{region.high:.4g}'
        labels.append(f'{form_type}_{low_str}_{high_str}')
    return labels


def _compute_norm_factor(s: Spectrum) -> float:
    """Robust scale factor to bring a spectrum's flux to ~O(1).

    Uses the median of the absolute non-zero flux values.

    Parameters
    ----------
    s : Spectrum

    Returns
    -------
    float
        Positive normalization factor.
    """
    absflux = jnp.abs(s.flux)
    positive = absflux[absflux > 0]
    if positive.size == 0:
        fallback = float(jnp.max(s.error))
        return fallback if fallback > 0 else 1.0
    return float(jnp.median(positive))
