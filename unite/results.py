"""Output parsing: parameter tables, spectra tables, and FITS HDU lists.

These functions transform raw posterior samples + :class:`~unite.model.ModelArgs`
into user-friendly :class:`~astropy.table.Table` objects and FITS files.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable, Table

from unite.compute import evaluate_model
from unite.model import ModelArgs
from unite.prior import Fixed


def count_parameters(model_fn, model_args) -> int:
    """Count the number of free scalar parameters (degrees of freedom) in the model.

    Traces the model with a dummy PRNG key and counts every latent
    (non-observed) sample site, summing the sizes of all their shapes.
    This gives the total number of unconstrained scalar parameters —
    i.e. the model degrees of freedom.

    Parameters
    ----------
    model_fn : callable
        The numpyro model function returned by :meth:`~unite.model.ModelBuilder.build`.
    model_args : ModelArgs
        The model arguments returned by :meth:`~unite.model.ModelBuilder.build`.

    Returns
    -------
    int
        Total number of free scalar parameters.

    Examples
    --------
    >>> model_fn, model_args = builder.build()
    >>> print(f'Free parameters: {count_parameters(model_fn, model_args)}')
    Free parameters: 14
    """
    import jax
    from numpyro import handlers

    seeded = handlers.seed(model_fn, jax.random.PRNGKey(0))
    trace = handlers.trace(seeded).get_trace(model_args)
    return sum(
        int(np.prod(site['value'].shape))
        for site in trace.values()
        if site['type'] == 'sample' and not site.get('is_observed', False)
    )


def make_parameter_table(
    samples: dict[str, np.ndarray],
    args: ModelArgs,
    *,
    percentiles: np.ndarray | None = None,
) -> QTable:
    """Build an Astropy table of posterior parameter samples in physical units.

    Parameters
    ----------
    samples : dict of str to ndarray
        Posterior samples in physical space.  When using :meth:`ModelBuilder.fit`,
        samples are already transformed.  When calling ``mcmc.get_samples()``
        directly, first pass through :func:`~unite.model.transform_reparam_samples`
        to convert any reparameterized (unit-space) parameters back to physical values.
    args : ModelArgs
        Model arguments from :meth:`ModelBuilder.build`.
    percentiles : ndarray of float or None
        Array of percentile values in range (0, 1), e.g. ``[0.16, 0.5, 0.84]``.
        If provided, returns one row per percentile with percentile values.
        If ``None`` (default), returns one row per posterior sample.

    Returns
    -------
    astropy.table.QTable
        If ``percentiles`` is ``None``: one row per posterior sample.
        If ``percentiles`` is provided: one row per percentile with a ``percentile``
        column and one column per parameter.
        Columns carry physical units where known:

        * Line flux parameters — ``flux_unit * canonical_wl_unit``
        * FWHM parameters — km/s
        * Continuum ``scale`` parameters — ``flux_unit``
        * Continuum ``slope`` / polynomial coefficients — ``flux_unit / wl_unit^n``
        * Shape / index parameters (``beta``, ``temperature``, …) — raw values
        * Rest equivalent width columns (``rew_{line_label}``) —
          ``canonical_wl_unit``.  One column per line whose rest-frame
          wavelength falls within a continuum region.  Appended after all
          model parameters when a continuum is present.

    Notes
    -----
    For **absorption lines**, the rest equivalent width is computed by
    numerically integrating the absorbed flux profile over the spectrum with
    the finest pixel grid covering the line.  Use absorption REW values with
    caution when the covering spectrum does not fully resolve the absorption
    profile.
    """
    table = QTable()

    cm = args.matrices
    # Use the first spectrum's flux unit for parameter table units.
    flux_unit = args.flux_units[0]
    canonical_unit = args.canonical_unit
    line_flux_unit = flux_unit * canonical_unit

    # Line flux scale in the first spectrum's unit system for de-scaling.
    line_flux_scale_0 = args.line_flux_scales[0]
    cont_scale_0 = args.continuum_scales[0]

    # Classify parameter names.
    flux_names: set[str] = set(cm.flux_names)
    tau_names: set[str] = set(cm.tau_names)
    fwhm_names: set[str] = set(cm.p0_names or []) | set(cm.p1v_names or [])
    z_names: set[str] = set(cm.z_names or [])

    # Build continuum param lookup: param_name → (region_idx, param_slot_name)
    cont_param_lookup: dict[str, tuple[int, str]] = {}
    if args.cont_config is not None and args.cont_resolved_params is not None:
        for k, resolved in enumerate(args.cont_resolved_params):
            for pn, tok in resolved.items():
                cont_param_lookup[tok.name] = (k, pn)

    def _to_column(pname: str, arr: np.ndarray) -> np.ndarray | u.Quantity:
        """Convert a raw sample array to a physical Quantity where possible."""
        if pname in flux_names:
            phys = arr * line_flux_scale_0
            return u.Quantity(phys, unit=line_flux_unit)
        if pname in tau_names:
            return arr  # dimensionless optical depth
        if pname in fwhm_names:
            return u.Quantity(arr, unit=u.km / u.s)
        if pname in z_names:
            return arr  # dimensionless
        if pname in cont_param_lookup:
            k, pn = cont_param_lookup[pname]
            region = args.cont_config[k]
            pu = region.form.param_units(flux_unit, region._unit)
            apply_cs, phys_unit = pu.get(pn, (False, None))
            phys = arr * cont_scale_0 if apply_cs else arr
            return u.Quantity(phys, unit=phys_unit) if phys_unit is not None else phys
        return arr  # calibration or other dimensionless param

    ordered = _categorized_order(
        args.dependency_order,
        z_names,
        fwhm_names,
        flux_names,
        tau_names,
        cont_param_lookup,
    )
    rew_cols = _compute_rew_columns(samples, args)

    def _add_param(pname: str, *, pct_arr: np.ndarray | None = None) -> None:
        prior = args.all_priors[pname]
        if pct_arr is not None:
            if isinstance(prior, Fixed):
                val = float(prior.value)
                table[pname] = _to_column(pname, np.full(len(pct_arr), val))
            else:
                arr = np.asarray(samples[pname])
                table[pname] = _to_column(pname, np.percentile(arr, pct_arr * 100))
        else:
            n_samp = _get_n_samples(samples)
            if isinstance(prior, Fixed):
                table[pname] = _to_column(pname, np.full(n_samp, float(prior.value)))
            else:
                table[pname] = _to_column(pname, np.asarray(samples[pname]))

    def _add_rew(
        rew_arr: np.ndarray, col_name: str, *, pct_arr: np.ndarray | None = None
    ) -> None:
        if pct_arr is not None:
            vals = np.nanpercentile(rew_arr, pct_arr * 100)
        else:
            vals = rew_arr
        table[col_name] = u.Quantity(vals, unit=canonical_unit)

    # Split REW columns into emission (after flux) and absorption (after tau).
    abs_labels = {
        args.line_labels[j]
        for j in range(len(args.line_labels))
        if np.asarray(cm.is_absorption)[j]
    }
    emission_rew = {
        k: v for k, v in rew_cols.items() if k.removeprefix('rew_') not in abs_labels
    }
    absorption_rew = {
        k: v for k, v in rew_cols.items() if k.removeprefix('rew_') in abs_labels
    }

    if percentiles is not None:
        pct_arr = np.asarray(percentiles)
        table['percentile'] = pct_arr
        for category, pnames in ordered.items():
            for pname in pnames:
                _add_param(pname, pct_arr=pct_arr)
            if category == 'flux' and emission_rew:
                for col_name, rew_arr in emission_rew.items():
                    _add_rew(rew_arr, col_name, pct_arr=pct_arr)
            if category == 'tau' and absorption_rew:
                for col_name, rew_arr in absorption_rew.items():
                    _add_rew(rew_arr, col_name, pct_arr=pct_arr)
    else:
        for category, pnames in ordered.items():
            for pname in pnames:
                _add_param(pname)
            if category == 'flux' and emission_rew:
                for col_name, rew_arr in emission_rew.items():
                    _add_rew(rew_arr, col_name)
            if category == 'tau' and absorption_rew:
                for col_name, rew_arr in absorption_rew.items():
                    _add_rew(rew_arr, col_name)

    # Add metadata (short keys for FITS compatibility).
    lsq = args.line_scale_quantity
    csq = args.continuum_scale_quantity
    table.meta['LFLXSCL'] = float(lsq.value) if lsq is not None else None
    table.meta['LFLXUNT'] = str(lsq.unit) if lsq is not None else None
    table.meta['CNTSCL'] = float(csq.value) if csq is not None else None
    table.meta['CNTUNT'] = str(csq.unit) if csq is not None else None
    table.meta['NRMFCTRS'] = list(args.norm_factors)
    table.meta['ZSYS'] = args.redshift

    return table


def make_spectra_tables(
    samples: dict[str, np.ndarray],
    args: ModelArgs,
    *,
    insert_nan: bool = False,
    percentiles: np.ndarray | None = None,
) -> list[Table]:
    """Build per-spectrum tables of model decompositions.

    Parameters
    ----------
    samples : dict of str to ndarray
        Posterior samples.
    args : ModelArgs
        Model arguments from :meth:`ModelBuilder.build`.
    insert_nan : bool
        If ``True``, insert one NaN row at the midpoint wavelength between
        each pair of consecutive continuum regions.  Default ``False``.
    percentiles : ndarray of float or None
        Array of percentile values in range (0, 1), e.g. ``[0.16, 0.5, 0.84]``.
        If provided, collapses the sample dimension to those percentiles
        (shape ``(n_percentiles, n_pixels)``).
        If ``None`` (default), returns all samples (shape ``(n_samples, n_pixels)``).

    Returns
    -------
    list of astropy.table.QTable
        One table per spectrum.  Columns carry physical units where
        ``flux_unit`` was set on the spectrum.
    """
    predictions = evaluate_model(samples, args)
    tables: list[Table] = []

    for i, (pred, spectrum) in enumerate(zip(predictions, args.spectra, strict=True)):
        # Build trim mask: keep only pixels within any continuum region.
        wl = pred.wavelength
        if args.cont_config is not None and args.cont_low is not None:
            z = args.redshift
            inv_s2c = 1.0 / args.spec_to_canonical[i]
            pixel_mask = np.zeros(len(wl), dtype=bool)
            region_bounds: list[tuple[float, float]] = []
            for k in range(len(args.cont_config)):
                obs_low = args.cont_low[k] * (1.0 + z) * inv_s2c
                obs_high = args.cont_high[k] * (1.0 + z) * inv_s2c
                pixel_mask |= (wl >= obs_low) & (wl <= obs_high)
                region_bounds.append((obs_low, obs_high))
        else:
            pixel_mask = np.ones(len(wl), dtype=bool)
            region_bounds = []

        t = QTable()
        wl_unit = spectrum.unit
        spec_flux_unit = args.flux_units[i]

        t['wavelength'] = u.Quantity(wl[pixel_mask], unit=wl_unit)

        if percentiles is not None:
            # _compute_percentiles returns (n_percentiles, n_pixels) → trim → transpose to (n_pixels, n_percentiles)
            t['model_total'] = u.Quantity(
                _compute_percentiles(pred.total[:, pixel_mask], percentiles).T,
                unit=spec_flux_unit,
            )
            for name, arr in pred.lines.items():
                t[name] = u.Quantity(
                    _compute_percentiles(arr[:, pixel_mask], percentiles).T,
                    unit=spec_flux_unit,
                )
            for name, arr in pred.continuum_regions.items():
                t[name] = u.Quantity(
                    _compute_percentiles(arr[:, pixel_mask], percentiles).T,
                    unit=spec_flux_unit,
                )
        else:
            # (n_samples, n_pixels) → trim → transpose to (n_pixels, n_samples)
            t['model_total'] = u.Quantity(
                pred.total[:, pixel_mask].T, unit=spec_flux_unit
            )
            for name, arr in pred.lines.items():
                t[name] = u.Quantity(arr[:, pixel_mask].T, unit=spec_flux_unit)
            for name, arr in pred.continuum_regions.items():
                t[name] = u.Quantity(arr[:, pixel_mask].T, unit=spec_flux_unit)

        # Add observed data columns.
        t['observed_flux'] = u.Quantity(
            np.asarray(spectrum.flux)[pixel_mask], unit=spec_flux_unit
        )
        t['observed_error'] = u.Quantity(
            np.asarray(spectrum.error)[pixel_mask], unit=spec_flux_unit
        )
        t['scaled_error'] = u.Quantity(
            np.asarray(spectrum.scaled_error)[pixel_mask], unit=spec_flux_unit
        )

        t.meta['SPECNAME'] = spectrum.name
        t.meta['NORMFAC'] = float(args.norm_factors[i])

        if insert_nan and region_bounds:
            t = _insert_nan_between_regions(t, region_bounds)

        tables.append(t)

    return tables


def make_hdul(
    samples: dict[str, np.ndarray],
    args: ModelArgs,
    *,
    insert_nan: bool = False,
    percentiles: np.ndarray | None = None,
) -> fits.HDUList:
    """Build a FITS HDU list from posterior samples.

    Parameters
    ----------
    samples : dict of str to ndarray
        Posterior samples.
    args : ModelArgs
        Model arguments from :meth:`ModelBuilder.build`.
    insert_nan : bool
        Insert NaN rows between continuum regions.  Default ``False``.
    percentiles : ndarray of float or None
        Array of percentile values in range (0, 1).
        If provided, output tables contain percentile rows/columns.
        If ``None`` (default), output tables contain all samples.

    Returns
    -------
    astropy.io.fits.HDUList
        HDU 0: PrimaryHDU (empty, metadata in header).
        HDU 1: BinTableHDU from parameter table.
        HDU 2+: BinTableHDU per spectrum.
    """
    param_table = make_parameter_table(samples, args, percentiles=percentiles)
    spectra_tables = make_spectra_tables(
        samples, args, insert_nan=insert_nan, percentiles=percentiles
    )

    primary = fits.PrimaryHDU()
    primary.header['ZSYS'] = (args.redshift, 'Systemic redshift')
    lsq = args.line_scale_quantity
    csq = args.continuum_scale_quantity
    if lsq is not None:
        primary.header['LFLXSCL'] = (float(lsq.value), 'Line flux scale')
        primary.header['LFLXUNT'] = (str(lsq.unit), 'Line flux scale unit')
    if csq is not None:
        primary.header['CNTSCL'] = (float(csq.value), 'Continuum flux scale')
        primary.header['CNTUNT'] = (str(csq.unit), 'Continuum flux scale unit')
    primary.header['NSPEC'] = (len(args.spectra), 'Number of spectra')

    hdus = [primary]

    # Parameter table.
    param_hdu = fits.BinTableHDU(param_table, name='PARAMETERS')
    hdus.append(param_hdu)

    # Per-spectrum tables.
    for table in spectra_tables:
        name = table.meta.get('SPECNAME', 'SPECTRUM')
        spec_hdu = fits.BinTableHDU(table, name=name.upper())
        hdus.append(spec_hdu)

    return fits.HDUList(hdus)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _categorized_order(
    dependency_order: list[str],
    z_names: set[str],
    fwhm_names: set[str],
    flux_names: set[str],
    tau_names: set[str],
    cont_param_lookup: dict[str, tuple[int, str]],
) -> dict[str, list[str]]:
    """Return parameters grouped by category, preserving topological order within each group.

    Parameters
    ----------
    dependency_order : list of str
        Topologically sorted parameter names from ModelArgs.
    z_names, fwhm_names, flux_names, tau_names : set of str
        Parameter name sets from the coupling matrices.
    cont_param_lookup : dict
        Mapping of continuum param name → (region_idx, slot_name).

    Returns
    -------
    dict of str to list of str
        Ordered dict with keys ``'z'``, ``'fwhm'``, ``'flux'``, ``'tau'``,
        ``'cont'``, ``'instrument'`` and values being the parameter names in
        each category, in their original topological order.
    """
    groups: dict[str, list[str]] = {
        'z': [],
        'fwhm': [],
        'flux': [],
        'tau': [],
        'cont': [],
        'instrument': [],
    }
    for pname in dependency_order:
        if pname in z_names:
            groups['z'].append(pname)
        elif pname in fwhm_names:
            groups['fwhm'].append(pname)
        elif pname in flux_names:
            groups['flux'].append(pname)
        elif pname in tau_names:
            groups['tau'].append(pname)
        elif pname in cont_param_lookup:
            groups['cont'].append(pname)
        else:
            groups['instrument'].append(pname)
    return groups


def _get_n_samples(samples: dict[str, np.ndarray]) -> int:
    """Determine the number of samples from the first non-empty array."""
    for v in samples.values():
        arr = np.asarray(v)
        if arr.ndim >= 1:
            return arr.shape[0]
    return 1


def _compute_percentiles(
    arr: np.ndarray, percentiles: np.ndarray | list[float]
) -> np.ndarray:
    """Collapse (n_samples, n_pixels) to (n_percentiles, n_pixels).

    Parameters
    ----------
    arr : ndarray
        Shape (n_samples, n_pixels).
    percentiles : array-like of float
        Percentile values in range (0, 1), e.g., [0.16, 0.5, 0.84].

    Returns
    -------
    ndarray
        Shape (n_percentiles, n_pixels) with percentile values.
    """
    percentiles_arr = np.asarray(percentiles)
    return np.percentile(arr, percentiles_arr * 100, axis=0)


def _compute_rew_columns(
    samples: dict[str, np.ndarray], args: ModelArgs
) -> dict[str, np.ndarray]:
    """Compute rest equivalent width per line per posterior sample.

    For **emission lines**, the rest EW is::

        REW = F_line / (C_obs * (1 + z_total))

    where ``F_line`` is the physical integrated line flux, ``C_obs`` is the
    total continuum flux density evaluated at the observed-frame line center
    (summing all covering continuum regions), and the ``(1 + z_total)``
    factor converts the observer-frame equivalent width to rest frame.

    For **absorption lines**, the rest EW is computed numerically::

        REW = ∫ delta_j / C_center_j  dλ / (1 + z)

    where ``delta_j`` is the flux removed by the absorber
    (``total * (1 - 1/T_j)``, negative).  The integral is evaluated via the
    trapezoidal rule on the finest spectrum grid that covers the line.

    Parameters
    ----------
    samples : dict of str to ndarray
        Posterior samples.
    args : ModelArgs
        Model arguments from :meth:`ModelBuilder.build`.

    Returns
    -------
    dict of str to ndarray
        Mapping of ``'rew_{line_label}'`` → ``(n_samples,)`` array for
        both emission and absorption lines.  Lines without a covering
        continuum region are omitted.
    """
    if args.cont_config is None or args.cont_resolved_params is None:
        return {}

    cm = args.matrices
    n_samples = _get_n_samples(samples)
    n_lines = int(cm.wavelengths.shape[0])
    z_sys = args.redshift
    is_absorption = np.asarray(cm.is_absorption)
    has_absorption = bool(np.any(is_absorption))

    # --- flux per line: (n_samples, n_lines) ---
    if cm.flux_names:
        flux_vecs = np.column_stack(
            [
                np.full(n_samples, float(args.all_priors[n].value))
                if isinstance(args.all_priors[n], Fixed)
                else np.asarray(samples[n])
                for n in cm.flux_names
            ]
        )
        flux_per_line = (
            flux_vecs @ np.asarray(cm.flux_matrix) * np.asarray(cm.strengths)
        )
    else:
        flux_per_line = np.zeros((n_samples, n_lines))

    # --- redshift per line: (n_samples, n_lines) ---
    if cm.z_names:
        z_vecs = np.column_stack(
            [
                np.full(n_samples, float(args.all_priors[n].value))
                if isinstance(args.all_priors[n], Fixed)
                else np.asarray(samples[n])
                for n in cm.z_names
            ]
        )
        z_per_line = z_vecs @ np.asarray(cm.z_matrix)
    else:
        z_per_line = np.zeros((n_samples, n_lines))

    line_flux_scale = args.line_flux_scales[0]
    cont_scale = args.continuum_scales[0]

    # Compute predictions for absorption REW (numerical integration).
    predictions = evaluate_model(samples, args) if has_absorption else None

    result: dict[str, np.ndarray] = {}

    # --- Helper: evaluate total continuum at a point, summing all covering regions ---
    def _cont_at_point(obs_wl: np.ndarray) -> np.ndarray:
        """Total un-scaled continuum at obs_wl (n_samples,) → (n_samples,)."""
        total = np.zeros(n_samples)
        for k in range(len(args.cont_config)):
            obs_low = float(args.cont_low[k]) * (1.0 + z_sys)
            obs_high = float(args.cont_high[k]) * (1.0 + z_sys)
            obs_center = float(args.cont_center[k]) * (1.0 + z_sys)
            median_wl = float(np.median(obs_wl))
            if median_wl < obs_low or median_wl > obs_high:
                continue
            cont_p: dict[str, np.ndarray] = {}
            for pn, tok in args.cont_resolved_params[k].items():
                prior = args.all_priors[tok.name]
                val: np.ndarray = (
                    np.full(n_samples, float(prior.value))
                    if isinstance(prior, Fixed)
                    else np.asarray(samples[tok.name])
                )
                if pn == 'norm_wav':
                    val = val * args.cont_nw_conv[k] * (1.0 + z_sys)
                cont_p[pn] = val
            form = args.cont_forms[k]
            total = total + np.asarray(
                form.evaluate(obs_wl, obs_center, cont_p, obs_low, obs_high)
            )
        return total

    for j in range(n_lines):
        label = args.line_labels[j]
        rest_wl = float(cm.wavelengths[j])
        z_j = z_per_line[:, j]
        obs_wl_j = rest_wl * (1.0 + z_sys + z_j)
        z_total = z_sys + z_j

        if is_absorption[j]:
            # --- Absorption REW via numerical integration ---
            # delta_j = total * (1 - 1/T_j), already stored in pred.lines.
            # Find the finest spectrum grid that covers this line.
            obs_center_median = float(np.median(obs_wl_j))
            best_spec_idx = None
            best_dpix = np.inf
            for si, spectrum in enumerate(args.spectra):
                wl = np.asarray(spectrum.wavelength) * args.spec_to_canonical[si]
                if wl[0] <= obs_center_median <= wl[-1]:
                    cidx = np.argmin(np.abs(wl - obs_center_median))
                    dpix = float(wl[cidx] - wl[cidx - 1] if cidx > 0 else wl[1] - wl[0])
                    if dpix < best_dpix:
                        best_dpix = dpix
                        best_spec_idx = si
            if best_spec_idx is None:
                continue

            pred = predictions[best_spec_idx]
            if label not in pred.lines:
                continue
            delta_flux = pred.lines[label]  # (n_samples, n_pix), negative

            # Total continuum at line center, summing all covering regions.
            cont_center = _cont_at_point(obs_wl_j) * cont_scale  # (n_samples,)
            cont_center = np.where(np.abs(cont_center) > 1e-30, cont_center, 1e-30)

            # Trapezoid integration: REW = ∫ (delta / C_center) dλ / (1+z).
            wl_grid = pred.wavelength * args.spec_to_canonical[best_spec_idx]
            integrand = delta_flux / cont_center[:, None]  # (n_samples, n_pix)
            rew_obs = np.trapezoid(integrand, x=wl_grid, axis=1)  # (n_samples,)
            rew = rew_obs / (1.0 + z_total)

            result[f'rew_{label}'] = rew

        else:
            # --- Emission REW: F_line / (C_center * (1+z)) ---
            # Sum all covering continuum regions at line center.
            cont_val = _cont_at_point(obs_wl_j)
            if np.all(cont_val == 0.0):
                continue

            cont_physical = cont_val * cont_scale
            flux_physical = flux_per_line[:, j] * line_flux_scale

            rew = flux_physical / (cont_physical * (1.0 + z_total))
            result[f'rew_{label}'] = rew

    return result


def _insert_nan_between_regions(
    table: Table, region_bounds: list[tuple[float, float]]
) -> Table:
    """Insert NaN rows at region boundaries using local pixel spacing.

    For each gap between consecutive regions, inserts NaN rows at synthetic
    wavelengths estimated from the closest real pixels and their spacing.

    Parameters
    ----------
    table : Table
        Spectrum table with a ``'wavelength'`` column (already trimmed).
    region_bounds : list of (float, float)
        Observed-frame ``(low, high)`` bounds for each continuum region,
        in the spectrum's wavelength unit.  Need not be sorted.

    Returns
    -------
    Table
        New table with NaN rows inserted at region boundaries.
    """
    from astropy.table import vstack

    # Sort regions and find gaps between consecutive ones.
    sorted_bounds = sorted(region_bounds)
    wl = np.asarray(table['wavelength'])

    # Collect boundary wavelengths for each gap.
    boundary_wls = []
    for j in range(len(sorted_bounds) - 1):
        _, high_j = sorted_bounds[j]
        low_next, _ = sorted_bounds[j + 1]
        if low_next > high_j:
            # Find closest pixel to high_j and estimate boundary wavelength.
            idx_high = np.argmin(np.abs(wl - high_j))
            closest_high = wl[idx_high]
            if idx_high > 0:
                delta_high = closest_high - wl[idx_high - 1]
            else:
                delta_high = wl[1] - wl[0] if len(wl) > 1 else 1.0
            boundary_wls.append(closest_high + delta_high)

            # Find closest pixel to low_next and estimate boundary wavelength.
            idx_low = np.argmin(np.abs(wl - low_next))
            closest_low = wl[idx_low]
            if idx_low < len(wl) - 1:
                delta_low = wl[idx_low + 1] - closest_low
            else:
                delta_low = wl[-1] - wl[-2] if len(wl) > 1 else 1.0
            boundary_wls.append(closest_low - delta_low)

    if not boundary_wls:
        return table

    # Sort by wavelength to maintain order in table.
    boundary_wls.sort()

    # Build segments with NaN rows at boundaries.
    segments = []
    prev_idx = 0

    for wl_val in boundary_wls:
        idx = int(np.searchsorted(wl, wl_val))
        segments.append(table[prev_idx:idx])

        # Insert NaN row at boundary wavelength.
        nan_tbl = type(table)()  # QTable or Table, matching the input
        for col in table.colnames:
            col_obj = table[col]
            col_arr = np.asarray(col_obj)
            col_unit = getattr(col_obj, 'unit', None)
            if col == 'wavelength':
                val = u.Quantity([wl_val], unit=col_unit) if col_unit else [wl_val]
            elif col_arr.ndim == 1:
                val = u.Quantity([np.nan], unit=col_unit) if col_unit else [np.nan]
            else:
                arr = np.full((1, col_arr.shape[1]), np.nan)
                val = u.Quantity(arr, unit=col_unit) if col_unit else arr
            nan_tbl[col] = val
        segments.append(nan_tbl)

        prev_idx = idx

    segments.append(table[prev_idx:])

    return vstack(segments)
