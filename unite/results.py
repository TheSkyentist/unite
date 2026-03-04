"""Output parsing: parameter tables, spectra tables, and FITS HDU lists.

These functions transform raw posterior samples + :class:`~unite.model.ModelArgs`
into user-friendly :class:`~astropy.table.Table` objects and FITS files.
"""

from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.table import Table

from unite.evaluate import evaluate_model
from unite.model import ModelArgs
from unite.prior import Fixed


def make_parameter_table(
    samples: dict[str, np.ndarray], args: ModelArgs, *, summary: bool = False
) -> Table:
    """Build an Astropy table of posterior parameter samples.

    Parameters
    ----------
    samples : dict of str to ndarray
        Posterior samples (e.g. from ``mcmc.get_samples()``).
    args : ModelArgs
        Model arguments from :meth:`ModelBuilder.build`.
    summary : bool
        If ``True``, return a 3-row table with columns ``stat``
        (``'median'``, ``'p16'``, ``'p84'``) and one column per
        parameter.  Default ``False`` returns one row per sample.

    Returns
    -------
    astropy.table.Table
        One row per posterior sample (full mode) or 3 rows (summary mode).
    """
    table = Table()

    if summary:
        table['stat'] = ['median', 'p16', 'p84']
        for pname in args.dependency_order:
            prior = args.all_priors[pname]
            if isinstance(prior, Fixed):
                val = float(prior.value)
                table[pname] = [val, val, val]
            else:
                arr = np.asarray(samples[pname])
                table[pname] = [
                    float(np.median(arr)),
                    float(np.percentile(arr, 16)),
                    float(np.percentile(arr, 84)),
                ]
    else:
        for pname in args.dependency_order:
            prior = args.all_priors[pname]
            if isinstance(prior, Fixed):
                n_samples = _get_n_samples(samples)
                table[pname] = np.full(n_samples, prior.value)
            else:
                table[pname] = np.asarray(samples[pname])

    # Add metadata (short keys for FITS compatibility).
    table.meta['LFLXSCL'] = args.line_flux_scale
    table.meta['NRMFCTRS'] = list(args.norm_factors)
    table.meta['ZSYS'] = args.redshift

    return table


def make_spectra_tables(
    samples: dict[str, np.ndarray],
    args: ModelArgs,
    *,
    insert_nan: bool = False,
    summary: bool = False,
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
    summary : bool
        If ``True``, collapse the sample dimension to ``[median, p16, p84]``
        (shape ``(3, n_pixels)``).  Default ``False``.

    Returns
    -------
    list of astropy.table.Table
        One table per spectrum.
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

        t = Table()
        t['wavelength'] = wl[pixel_mask]

        if summary:
            # _summarize returns (3, n_pixels) → trim → transpose to (n_pixels, 3)
            t['model_total'] = _summarize(pred.total[:, pixel_mask]).T
            for name, arr in pred.lines.items():
                t[name] = _summarize(arr[:, pixel_mask]).T
            for name, arr in pred.continuum_regions.items():
                t[name] = _summarize(arr[:, pixel_mask]).T
        else:
            # (n_samples, n_pixels) → trim → transpose to (n_pixels, n_samples)
            t['model_total'] = pred.total[:, pixel_mask].T
            for name, arr in pred.lines.items():
                t[name] = arr[:, pixel_mask].T
            for name, arr in pred.continuum_regions.items():
                t[name] = arr[:, pixel_mask].T

        # Add observed data columns.
        t['observed_flux'] = np.asarray(spectrum.flux)[pixel_mask]
        t['observed_error'] = np.asarray(spectrum.scaled_error)[pixel_mask]

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
    summary: bool = False,
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
    summary : bool
        Collapse samples to ``[median, p16, p84]``.  Default ``False``.

    Returns
    -------
    astropy.io.fits.HDUList
        HDU 0: PrimaryHDU (empty, metadata in header).
        HDU 1: BinTableHDU from parameter table.
        HDU 2+: BinTableHDU per spectrum.
    """
    param_table = make_parameter_table(samples, args, summary=summary)
    spectra_tables = make_spectra_tables(
        samples, args, insert_nan=insert_nan, summary=summary
    )

    primary = fits.PrimaryHDU()
    primary.header['ZSYS'] = (args.redshift, 'Systemic redshift')
    primary.header['LFLXSCL'] = (args.line_flux_scale, 'Line flux scale')
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


def _summarize(arr: np.ndarray) -> np.ndarray:
    """Collapse (n_samples, n_pixels) to (3, n_pixels): [median, p16, p84]."""
    return np.array(
        [
            np.median(arr, axis=0),
            np.percentile(arr, 16, axis=0),
            np.percentile(arr, 84, axis=0),
        ]
    )


def _insert_nan_between_regions(
    table: Table, region_bounds: list[tuple[float, float]]
) -> Table:
    """Insert one NaN row at the midpoint between each pair of consecutive regions.

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
        New table with NaN rows inserted at gap midpoints.
    """
    from astropy.table import vstack

    # Sort regions and find gaps between consecutive ones.
    sorted_bounds = sorted(region_bounds)
    midpoints = []
    for j in range(len(sorted_bounds) - 1):
        _, high_j = sorted_bounds[j]
        low_next, _ = sorted_bounds[j + 1]
        if low_next > high_j:
            midpoints.append((high_j + low_next) / 2.0)

    if not midpoints:
        return table

    wl = np.asarray(table['wavelength'])

    # Build segments separated by NaN rows at each midpoint.
    segments = []
    prev_idx = 0
    for mid in midpoints:
        idx = int(np.searchsorted(wl, mid))
        segments.append(table[prev_idx:idx])
        nan_tbl = Table()
        for col in table.colnames:
            col_arr = np.asarray(table[col])
            if col == 'wavelength':
                nan_tbl[col] = [mid]
            elif col_arr.ndim == 1:
                nan_tbl[col] = [np.nan]
            else:
                nan_tbl[col] = [np.full(col_arr.shape[1], np.nan)]
        segments.append(nan_tbl)
        prev_idx = idx
    segments.append(table[prev_idx:])

    return vstack(segments)
