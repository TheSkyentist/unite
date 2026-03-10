"""Output parsing: parameter tables, spectra tables, and FITS HDU lists.

These functions transform raw posterior samples + :class:`~unite.model.ModelArgs`
into user-friendly :class:`~astropy.table.Table` objects and FITS files.
"""

from __future__ import annotations

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import QTable, Table

from unite.evaluate import evaluate_model
from unite.model import ModelArgs
from unite.prior import Fixed


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
        * Rest equivalent width columns (``{line_label}_rew``) —
          ``canonical_wl_unit``.  One column per line whose rest-frame
          wavelength falls within a continuum region.  Appended after all
          model parameters when a continuum is present.
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
        if pname in fwhm_names:
            return u.Quantity(arr, unit=u.km / u.s)
        if pname in z_names:
            return arr  # dimensionless
        if pname in cont_param_lookup:
            k, pn = cont_param_lookup[pname]
            region = args.cont_config[k]
            pu = region.form.param_units(flux_unit, canonical_unit)
            apply_cs, phys_unit = pu.get(pn, (False, None))
            phys = arr * cont_scale_0 if apply_cs else arr
            return u.Quantity(phys, unit=phys_unit) if phys_unit is not None else phys
        return arr  # calibration or other dimensionless param

    if percentiles is not None:
        percentiles_arr = np.asarray(percentiles)
        table['percentile'] = percentiles_arr
        for pname in args.dependency_order:
            prior = args.all_priors[pname]
            if isinstance(prior, Fixed):
                val = float(prior.value)
                table[pname] = _to_column(pname, np.full(len(percentiles_arr), val))
            else:
                arr = np.asarray(samples[pname])
                percentile_vals = np.percentile(arr, percentiles_arr * 100)
                table[pname] = _to_column(pname, percentile_vals)
    else:
        n_samples = _get_n_samples(samples)
        for pname in args.dependency_order:
            prior = args.all_priors[pname]
            if isinstance(prior, Fixed):
                table[pname] = _to_column(pname, np.full(n_samples, float(prior.value)))
            else:
                arr = np.asarray(samples[pname])
                table[pname] = _to_column(pname, arr)

    # Add rest equivalent width columns.
    rew_cols = _compute_rew_columns(samples, args)
    if rew_cols:
        if percentiles is not None:
            percentiles_arr = np.asarray(percentiles)
            for col_name, rew_arr in rew_cols.items():
                percentile_vals = np.nanpercentile(rew_arr, percentiles_arr * 100)
                table[col_name] = u.Quantity(percentile_vals, unit=canonical_unit)
        else:
            for col_name, rew_arr in rew_cols.items():
                table[col_name] = u.Quantity(rew_arr, unit=canonical_unit)

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

    For each line whose rest-frame wavelength falls within a continuum
    region, the rest EW is::

        REW = F_line / (C_obs * (1 + z_total))

    where ``F_line`` is the physical integrated line flux
    (``flux_unit * canonical_wl_unit``), ``C_obs`` is the continuum flux
    density evaluated at the observed-frame line centre (``flux_unit``),
    and the ``(1 + z_total)`` factor converts the observer-frame equivalent
    width to rest frame.  The result is in rest-frame canonical wavelength
    units.

    Parameters
    ----------
    samples : dict of str to ndarray
        Posterior samples.
    args : ModelArgs
        Model arguments from :meth:`ModelBuilder.build`.

    Returns
    -------
    dict of str to ndarray
        Mapping of ``'{line_label}_rew'`` → ``(n_samples,)`` array.
        Lines without a covering continuum region are omitted.
    """
    if args.cont_config is None or args.cont_resolved_params is None:
        return {}

    cm = args.matrices
    n_samples = _get_n_samples(samples)
    n_lines = int(cm.wavelengths.shape[0])
    z_sys = args.redshift

    # --- flux per line: (n_samples, n_lines) ---
    flux_vecs = np.column_stack(
        [
            np.full(n_samples, float(args.all_priors[n].value))
            if isinstance(args.all_priors[n], Fixed)
            else np.asarray(samples[n])
            for n in cm.flux_names
        ]
    )
    flux_per_line = flux_vecs @ np.asarray(cm.flux_matrix) * np.asarray(cm.strengths)

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

    result: dict[str, np.ndarray] = {}

    for j in range(n_lines):
        label = args.line_labels[j]
        rest_wl = float(cm.wavelengths[j])  # rest-frame, canonical unit

        # Find the first continuum region whose rest-frame bounds cover this line.
        covering_k = None
        for k in range(len(args.cont_config)):
            if args.cont_low[k] <= rest_wl <= args.cont_high[k]:
                covering_k = k
                break

        if covering_k is None:
            continue

        k = covering_k
        form = args.cont_forms[k]
        obs_center = float(args.cont_center[k]) * (1.0 + z_sys)

        # Build cont_p with (n_samples,) arrays, mirroring evaluate.py.
        cont_p: dict[str, np.ndarray] = {}
        for pn, tok in args.cont_resolved_params[k].items():
            prior = args.all_priors[tok.name]
            val: np.ndarray = (
                np.full(n_samples, float(prior.value))
                if isinstance(prior, Fixed)
                else np.asarray(samples[tok.name])
            )
            if pn == 'normalization_wavelength':
                val = val * args.cont_nw_conv[k] * (1.0 + z_sys)
            cont_p[pn] = val

        # Observed-frame line centre per sample: (n_samples,)
        z_j = z_per_line[:, j]
        obs_wl_j = rest_wl * (1.0 + z_sys + z_j)

        # Continuum flux density at line centre (un-scaled, from form.evaluate).
        cont_val = np.asarray(form.evaluate(obs_wl_j, obs_center, cont_p))

        # Physical quantities.
        cont_physical = cont_val * cont_scale  # flux_unit
        flux_physical = (
            flux_per_line[:, j] * line_flux_scale
        )  # flux_unit * canonical_unit

        # Rest EW = obs EW / (1 + z_total).
        z_total = z_sys + z_j
        rew = flux_physical / (cont_physical * (1.0 + z_total))  # canonical_unit

        result[f'{label}_rew'] = rew

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
