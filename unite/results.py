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
    samples: dict[str, np.ndarray],
    args: ModelArgs,
) -> Table:
    """Build an Astropy table of posterior parameter samples.

    Each sampled parameter becomes a column with shape ``(n_samples,)``.
    Fixed parameters are included as constant columns.

    Parameters
    ----------
    samples : dict of str to ndarray
        Posterior samples (e.g. from ``mcmc.get_samples()``).
    args : ModelArgs
        Model arguments from :meth:`ModelBuilder.build`.

    Returns
    -------
    astropy.table.Table
        One row per posterior sample.
    """
    table = Table()

    for pname in args.dependency_order:
        prior = args.all_priors[pname]
        if isinstance(prior, Fixed):
            # Determine n_samples from a sampled parameter.
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
    nan_fill: bool = False,
    summary: bool = False,
) -> list[Table]:
    """Build per-spectrum tables of model decompositions.

    Parameters
    ----------
    samples : dict of str to ndarray
        Posterior samples.
    args : ModelArgs
        Model arguments from :meth:`ModelBuilder.build`.
    nan_fill : bool
        If ``True``, insert NaN rows at wavelength gaps between
        continuum regions for cleaner plotting.  Default ``False``.
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
        t = Table()
        t['wavelength'] = pred.wavelength

        if summary:
            # _summarize returns (3, n_pixels) → transpose to (n_pixels, 3)
            t['model_total'] = _summarize(pred.total).T
            for name, arr in pred.lines.items():
                t[name] = _summarize(arr).T
            for name, arr in pred.continuum_regions.items():
                t[name] = _summarize(arr).T
        else:
            # (n_samples, n_pixels) → transpose to (n_pixels, n_samples)
            t['model_total'] = pred.total.T
            for name, arr in pred.lines.items():
                t[name] = arr.T
            for name, arr in pred.continuum_regions.items():
                t[name] = arr.T

        # Add observed data columns.
        t['observed_flux'] = np.asarray(spectrum.flux)
        t['observed_error'] = np.asarray(spectrum.scaled_error)

        t.meta['SPECNAME'] = spectrum.name
        t.meta['NORMFAC'] = float(args.norm_factors[i])

        if nan_fill:
            t = _insert_nan_gaps(t)

        tables.append(t)

    return tables


def make_hdul(
    samples: dict[str, np.ndarray],
    args: ModelArgs,
    *,
    nan_fill: bool = False,
    summary: bool = False,
) -> fits.HDUList:
    """Build a FITS HDU list from posterior samples.

    Parameters
    ----------
    samples : dict of str to ndarray
        Posterior samples.
    args : ModelArgs
        Model arguments from :meth:`ModelBuilder.build`.
    nan_fill : bool
        Insert NaN rows at wavelength gaps.  Default ``False``.
    summary : bool
        Collapse samples to ``[median, p16, p84]``.  Default ``False``.

    Returns
    -------
    astropy.io.fits.HDUList
        HDU 0: PrimaryHDU (empty, metadata in header).
        HDU 1: BinTableHDU from parameter table.
        HDU 2+: BinTableHDU per spectrum.
    """
    param_table = make_parameter_table(samples, args)
    spectra_tables = make_spectra_tables(
        samples, args, nan_fill=nan_fill, summary=summary
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
    return np.array([
        np.median(arr, axis=0),
        np.percentile(arr, 16, axis=0),
        np.percentile(arr, 84, axis=0),
    ])


def _insert_nan_gaps(table: Table) -> Table:
    """Insert NaN rows at wavelength gaps for cleaner plotting."""
    wl = np.asarray(table['wavelength'])
    if len(wl) < 2:
        return table

    # Detect gaps: where the step is > 3x the median step.
    diffs = np.diff(wl)
    median_step = np.median(diffs)
    if median_step <= 0:
        return table

    gap_indices = np.where(diffs > 3 * median_step)[0]
    if len(gap_indices) == 0:
        return table

    # Build new table with NaN rows inserted at gaps.
    rows_to_insert = []
    for idx in gap_indices:
        nan_row = {}
        for col in table.colnames:
            col_data = table[col]
            if col_data.ndim == 1:
                nan_row[col] = np.nan
            else:
                nan_row[col] = np.full(col_data.shape[1:], np.nan)
        rows_to_insert.append((idx + 1, nan_row))

    # Insert in reverse order to preserve indices.
    for insert_idx, nan_row in reversed(rows_to_insert):
        # Build a single-row table.
        nan_table = Table()
        for col in table.colnames:
            val = nan_row[col]
            if np.isscalar(val):
                nan_table[col] = [val]
            else:
                nan_table[col] = [val]
        from astropy.table import vstack
        top = table[:insert_idx]
        bottom = table[insert_idx:]
        table = vstack([top, nan_table, bottom])

    return table
