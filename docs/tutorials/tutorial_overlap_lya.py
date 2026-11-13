r"""
Order Overlap (Line)
====================

In this tutorial we fit the NIRSpec G140M/F070LP spectrum of the Lyman-alpha
emitter DIVER-GDN-1020514 at :math:`z \approx 5.36`, where a single emission
line is recorded in three diffraction orders.

As in :doc:`tutorial_overlap_broadline`, ``unite`` models additional orders
on a disperser through ``OrderOverlap``, given the relative throughput of
each order as a FITS table.  G140M/F070LP transmits from about 0.7 µm but the
detector records out to 3 µm, so redward of twice and three times the filter
cut-on, second- and third-order light lands on top of the first-order
spectrum.  Ly\ :math:`\alpha` at 0.773 µm therefore appears again at 1.546 µm
(second order) and 2.319 µm (third order).

With the overlap on the disperser, **one** Ly\ :math:`\alpha` line (one flux,
redshift and profile) describes all three images: their positions, widths and
fluxes follow from the model. Order-*m* provides *m* times the resolving
power of the first order, thereby providing additional constraints on the line kinematics.
"""

# %%
# Step 0 — Imports and Setup
# ---------------------------------
# We import the necessary libraries and set a plotting style for the tutorial.

import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table
from matplotlib import pyplot

from unite import continuum, instrument, line, model, prior, results, spectrum
from unite.instrument import nirspec

pyplot.style.use('unite.mplstyle')

zspec = 5.35904  # z_best in DJA v4.6
lya_rest = 1215.67 * u.AA  # vacuum

# Observed wavelength of the order-m image of Lyman-alpha
orders = (1, 2, 3)
colors = {1: 'C0', 2: 'C3', 3: 'C1'}  # one colour per order in every plot
lya_obs = {m: (m * lya_rest * (1 + zspec)).to(u.um) for m in orders}

# %%
# Step 1 — Configure the Disperser
# ----------------------------------
#
# As in :doc:`tutorial_overlap_broadline`, we derive the relative throughput
# :math:`r_m(\lambda)` of each order from the
# `msaexp <https://github.com/gbrammer/msaexp>`_ extended-sensitivity table of
# G140M/F070LP, which gives the sensitivity :math:`S_m` of every order against
# photon wavelength:
#
# .. math::
#
#    r_m(\lambda) = \frac{S_m(\lambda/m)}{m^2\, S_1(\lambda)}
#
# The factor :math:`m^2` converts to the :math:`f_\lambda` convention: the
# order-*m* image is dispersed *m* times as much, and the source flux density
# is taken per unit photon wavelength, another factor of *m*.  The conversion is
# done by :func:`unite.instrument.nirspec.msaexp_order_ratio`.

workdir = Path(tempfile.mkdtemp())
ratio_files = {}
for m in (2, 3):
    ratio_files[m] = workdir / f'g140m_f070lp_order{m}.fits'
    nirspec.msaexp_order_ratio('g140m_f070lp', m, path=ratio_files[m])

# %%
# The two ratios, with the three Ly\ :math:`\alpha` images marked.

fig, ax = pyplot.subplots(figsize=(10, 4))
for m, path in ratio_files.items():
    table = Table.read(path)
    ax.plot(
        table['wavelength'],
        table['throughput_ratio'],
        color=colors[m],
        label=f'$r_{m}$',
    )
    r_at_lya = np.interp(
        lya_obs[m].value, table['wavelength'], table['throughput_ratio']
    )
    print(f'r_{m} = {r_at_lya:.3f} at the order-{m} image of Lyman-alpha')
for m in orders:
    ax.axvline(lya_obs[m].value, color=colors[m], ls=':', lw=1)
ax.set(
    xlabel=r'$\lambda$ (Obs.) [µm]',
    ylabel=r'$r_m(\lambda)$',
    xlim=(0.6, 3.3),
    yscale='log',
    ylim=(1e-3, 1),
)
ax.legend()
pyplot.show()

# %%
# At the second-order image, :math:`r_2 \approx 0.18`: nearly a fifth of the
# first-order Ly\ :math:`\alpha` flux density is recorded a second time.
#
# Now the disperser.  Each order has its own ``OrderScale``, so the data can
# test each ratio separately (``OrderScale = 1`` means :math:`r_m` is exactly
# right).  We also fit a resolution scale relative to the point-source
# resolution curve.

order_scales = {
    m: instrument.OrderScale(
        f'order{m}', prior=prior.TruncatedNormal(1.0, 0.2, 0.5, 1.5)
    )
    for m in (2, 3)
}
g140m_disperser = nirspec.G140M(
    r_source='point',
    r_scale=instrument.RScale(prior=prior.TruncatedNormal(0.85, 0.15, 0.5, 1.1)),
    order_overlap=[
        instrument.OrderOverlap(m, ratio_files[m], scale=order_scales[m])
        for m in (2, 3)
    ],
)
print(g140m_disperser)

# %%
# Step 2 — Load the Spectrum from DJA
# ------------------------------------
#
# :func:`~unite.spectrum.from_DJA` downloads and parses the spectrum from the
# `DJA <https://dawn-cph.github.io/dja/>`_; ``cache=True`` stores the file
# locally so later runs do not re-download it.

g140m_spectrum = spectrum.from_DJA(
    'https://s3.amazonaws.com/msaexp-nirspec/extractions/diver-gdn-01a-v4/'
    'diver-gdn-01a-v4_g140m-f070lp_8018_1020514.spec.fits',
    disperser=g140m_disperser,
    cache=True,
)
spectra = spectrum.Spectra([g140m_spectrum], redshift=zspec)

# %%
# The vertical axis is linear within :math:`\pm 2` and logarithmic beyond, so
# all three images stand out above the noise.

fig, ax = pyplot.subplots(figsize=(12, 5))
ax.step(g140m_spectrum.wavelength, g140m_spectrum.flux, where='mid', color='k', lw=0.5)
for m in orders:
    ax.axvline(
        lya_obs[m].value,
        color=colors[m],
        ls=':',
        lw=1.2,
        label=rf'Ly$\alpha$, order {m}',
    )
ax.set(
    xlabel=rf'$\lambda$ (Obs.) [{g140m_spectrum.unit:latex}]',
    ylabel=rf'$f_\lambda$ [{g140m_spectrum.flux_unit:latex_inline}]',
    yscale='symlog',
    ylim=(-5, 500),
)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncols=3)
pyplot.show()

# %%
# Step 3 — Configure the Emission Line and Continuum
# ------------------------------------------------------
#
# The line is defined **once**, at its rest-frame wavelength.  We use an
# asymmetric profile (``'aemg'``: a Gaussian core with separate blue and red
# exponential tails).
#
# See :doc:`../usage/line_configuration` for the full line and profile
# reference.

line_configuration = line.LineConfiguration()
line_configuration.add_line(
    'Lya',
    lya_rest,
    profile='aemg',
    flux=line.Flux('Lya', prior=prior.Uniform(0, 10)),
    redshift=line.Redshift('Lya', prior=prior.Uniform(-0.02, 0.02)),
    fwhm_gauss=line.FWHM('Lya_gauss', prior=prior.Uniform(0, 800)),
    fwhm_l_blue=line.FWHM('Lya_blue', prior=prior.Uniform(0, 800)),
    fwhm_l_red=line.FWHM('Lya_red', prior=prior.Uniform(0, 1500)),
)
print(line_configuration)

# %%
# ``unite`` only fits pixels inside continuum regions, so we define one region
# around each image, ±3000 km/s wide.  Regions are in the rest frame of the
# first order, so the order-*m* image sits at :math:`m \times 1215.67` Å.
#
# The first-order region also supplies the source continuum of the overlap:
# the order-*m* model at :math:`\lambda` is only non-zero where
# :math:`\lambda/m` lies in a continuum region.  The power laws of the
# higher-order regions describe the first-order continuum at those
# wavelengths.
#
# See :doc:`../usage/continuum_configuration` for the other continuum forms.

half_width = 3000 / 299792.458  # ±3000 km/s, as a fraction of the speed of light
cc = continuum.ContinuumConfiguration(
    [
        continuum.ContinuumRegion(
            m * lya_rest * (1 - half_width),
            m * lya_rest * (1 + half_width),
            form=continuum.PowerLaw(),
            name=f'order{m}',
        )
        for m in orders
    ]
)
print(cc)

# %%
# Step 4 — Prepare the Spectrum and Fit
# ----------------------------------------
#
# The higher-order regions contain no line of their own, so
# :meth:`~unite.spectrum.Spectra.prepare` must be told to keep them
# (``drop_empty_regions=False``).
#
# :meth:`~unite.model.ModelBuilder.fit` warns that the second-order source
# window is only partly covered by continuum regions: the second-order light
# in the third-order region (2.30 to 2.34 µm) comes from 1.15 to 1.17 µm, where no
# region is defined.  That bit of continuum is left out of the overlap, and the
# free power law of the third-order region absorbs it instead.

filtered_lines, filtered_cont = spectra.prepare(
    line_configuration, cc, drop_empty_regions=False
)
spectra.compute_scales(filtered_lines, filtered_cont)

builder = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
samples, model_args = builder.fit(num_warmup=500, num_samples=500, progress_bar=False)

# %%
# Step 5 — Results
# ----------------
#
# The line and the calibration parameters.  Both ``OrderScale`` posteriors
# are consistent with one: the msaexp ratios describe the higher-order images
# well.

percentiles = np.array([0.16, 0.5, 0.84])
param_table = results.make_parameter_table(samples, model_args, percentiles=percentiles)
columns = ['percentile', 'flux_Lya', 'z_Lya']
columns += [c for c in param_table.colnames if c.startswith(('r_scale', 'order_'))]
param_table[columns].pprint(max_width=-1)

# %%
# The spectrum around each image, in velocity relative to that image, with the
# posterior median model split into its first-, second- and third-order parts.
# The three parts add up to the model.

table = results.make_spectra_tables(samples, model_args, percentiles=percentiles)[
    g140m_spectrum.name
]
wavelength = table['wavelength'].to_value(u.um)
total = table['model_total'][:, 1].value
parts = {m: table[f'overlap_{m}'][:, 1].value for m in (2, 3)}
parts[1] = total - parts[2] - parts[3]
labels = {1: 'First order', 2: 'Second order', 3: 'Third order'}

fig, axes = pyplot.subplots(1, 3, figsize=(16, 5.5))
fig.subplots_adjust(wspace=0.3, top=0.8)
for ax, m in zip(axes, orders, strict=True):
    velocity = (wavelength / lya_obs[m].value - 1) * 299792.458
    sel = np.abs(velocity) < 3000
    ax.step(
        velocity[sel],
        table['observed_flux'].value[sel],
        where='mid',
        color='0.6',
        lw=0.8,
        label='Data',
    )
    ax.step(velocity[sel], total[sel], where='mid', color='k', lw=1.0, label='Model')
    for k in orders:
        ax.step(
            velocity[sel],
            parts[k][sel],
            where='mid',
            color=colors[k],
            lw=1.0,
            label=labels[k],
        )
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set(
        title=f'Order {m} ({lya_obs[m].value:.3f} µm)',
        xlabel=r'$\Delta v$ [km s$^{-1}$]',
        xlim=(-3000, 3000),
    )
axes[0].set_ylabel(rf'$f_\lambda$ [{g140m_spectrum.flux_unit:latex_inline}]')
fig.legend(
    *axes[0].get_legend_handles_labels(),
    loc='lower center',
    bbox_to_anchor=(0.5, 0.88),
    ncols=5,
    handlelength=1.5,
    columnspacing=1.0,
)
pyplot.show()

# %%
# The second- and third-order images are reproduced by the first-order line
# alone, scaled by :math:`r_m` and broadened by the resolution of each order.
