r"""
Order Overlap (Continuum)
=========================

In this tutorial we fit all five NIRSpec spectra (PRISM, G140M, G235M, G395M
and G395H) of the broad-line source JADES-GS 187956 at :math:`z \approx 3.70`
(DJA ID 152632) with one shared model.

Light from higher diffraction orders can land on the detector of a grating
spectrograph: in order :math:`m`, photons at :math:`\lambda/m` are recorded at
:math:`\lambda`.  ``unite`` supports any number of additional orders on a
disperser through ``OrderOverlap``, provided the relative throughput of that
order (its sensitivity relative to first order, as a function of observed
wavelength) is given as a FITS table.

For this example we derive that relative throughput for JWST/NIRSpec.  The
medium-resolution gratings G140M/F070LP and G235M/F170LP are not blocked
against second-order light, and for a source with a blue rest-frame UV
continuum it raises the red half of both spectra by up to a quarter.  Left
out of the model, it is absorbed by whatever else can move: the flux scale
between gratings, the continuum shape, the broad-line wings.  Here we model it
and fit its strength with a single ``OrderScale``.
"""

# %%
# Step 0 — Imports and Setup
# ---------------------------------
# We import the necessary libraries and set a plotting style for the tutorial.

import tempfile
from pathlib import Path

import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from astropy.table import Table
from matplotlib import pyplot, ticker
from numpyro import infer

from unite import continuum, instrument, line, model, prior, results, spectrum
from unite.instrument import nirspec

pyplot.style.use('unite.mplstyle')

# %%
# Step 1 — Configure the Dispersers
# ----------------------------------
#
# Each disperser carries its calibration tokens:
#
# * ``RScale`` — resolution scale, shared by all five gratings
# * ``FluxScale`` — flux normalization of each grating relative to the PRISM
# * ``PixOffset`` — sub-pixel wavelength shift of the PRISM
# * ``OrderOverlap`` + ``OrderScale`` — second-order light on G140M and G235M
#
# See :doc:`../usage/instrument` for the full disperser and calibration token
# reference.
#
# The second-order term needs the throughput ratio :math:`r_2(\lambda)`: the
# fraction of second-order light, relative to first-order, that reaches the
# detector at observed wavelength :math:`\lambda`.  We compute it from the
# `msaexp <https://github.com/gbrammer/msaexp>`_ extended-sensitivity tables,
# which give the first- and second-order sensitivities :math:`S_1, S_2` of each
# grating/filter pair as a function of photon wavelength:
#
# .. math::
#
#    r_2(\lambda) = \frac{S_2(\lambda/2)}{4\, S_1(\lambda)}
#
# The factor of 4 converts to the :math:`f_\lambda` convention: the
# second-order image is dispersed twice as much, so its flux density per unit
# observed wavelength is diluted by 2, and the source flux density is taken
# per unit photon wavelength, another factor of 2.

workdir = Path(tempfile.mkdtemp())
g140m_ratio = workdir / 'g140m_f070lp_order2.fits'
g235m_ratio = workdir / 'g235m_f170lp_order2.fits'
nirspec.msaexp_order_ratio('g140m_f070lp', 2, path=g140m_ratio)
nirspec.msaexp_order_ratio('g235m_f170lp', 2, path=g235m_ratio)

# %%
# ``OrderOverlap`` reads a table with a wavelength column (with a length unit)
# and a dimensionless ``throughput_ratio`` column.  Let's look at what we wrote.

fig, ax = pyplot.subplots(figsize=(10, 4))
for i, (label, path) in enumerate(
    [('G140M/F070LP', g140m_ratio), ('G235M/F170LP', g235m_ratio)]
):
    table = Table.read(path)
    ax.plot(table['wavelength'], table['throughput_ratio'], color=f'C{i}', label=label)
ax.set(
    xlabel=r'$\lambda$ (Obs.) [µm]',
    ylabel=r'$r_2(\lambda)$',
    yscale='log',
    ylim=(1e-4, 1),
    xlim=(1.0, 5.5),
)
ax.legend()
pyplot.show()

# %%
# For G140M, :math:`r_2` reaches 0.17 at 1.4 µm.  For G235M it rises to 0.05
# between 3.2 and 4.2 µm, drops to about 0.001 and rises again beyond 4.9 µm.
#
# Now the dispersers.  The resolution scale is shared by all gratings and the
# PRISM is the flux-calibration reference (no ``flux_scale``), with a free
# pixel offset.  Each grating gets its own ``FluxScale``.  The ``init`` values
# are where the optimiser starts in Step 4.

# Shared resolution scale: same Python object → single model parameter
resolution_scale = instrument.RScale(
    prior=prior.TruncatedNormal(loc=0.85, scale=0.15, low=0.5, high=1.1, init=0.85)
)


def grating_flux_scale():
    """An independent flux scale for one grating, relative to the PRISM."""
    return instrument.FluxScale(
        prior=prior.TruncatedNormal(loc=1.0, scale=0.2, low=0.25, high=4.0, init=1.0)
    )


# The strength of the second-order light: one OrderScale shared by both
# gratings.  OrderScale = 1 means r_2 is exactly right.
order_scale = instrument.OrderScale(prior=prior.TruncatedNormal(1.0, 0.2, 0.5, 1.5))

prism_disperser = nirspec.PRISM(
    r_source='point',
    r_scale=resolution_scale,
    pix_offset=instrument.PixOffset(
        prior=prior.TruncatedNormal(loc=0.1, scale=0.15, low=-0.4, high=0.8, init=0.1)
    ),
)
g140m_disperser = nirspec.G140M(
    r_source='point',
    r_scale=resolution_scale,
    flux_scale=grating_flux_scale(),
    order_overlap=[instrument.OrderOverlap(2, g140m_ratio, scale=order_scale)],
)
g235m_disperser = nirspec.G235M(
    r_source='point',
    r_scale=resolution_scale,
    flux_scale=grating_flux_scale(),
    order_overlap=[instrument.OrderOverlap(2, g235m_ratio, scale=order_scale)],
)
g395m_disperser = nirspec.G395M(
    r_source='point', r_scale=resolution_scale, flux_scale=grating_flux_scale()
)
g395h_disperser = nirspec.G395H(
    r_source='point', r_scale=resolution_scale, flux_scale=grating_flux_scale()
)

instrument_config = instrument.InstrumentConfig(
    [
        prism_disperser,
        g140m_disperser,
        g235m_disperser,
        g395m_disperser,
        g395h_disperser,
    ]
)
print(instrument_config)

# %%
# Step 2 — Load the Spectra from DJA
# ------------------------------------
#
# :func:`~unite.spectrum.from_DJA` downloads and parses each spectrum from the
# `DJA <https://dawn-cph.github.io/dja/>`_; ``cache=True`` stores the files
# locally so later runs do not re-download them.

zspec = 3.70407  # z_best in DJA v4.6

DJA_URL = (
    'https://s3.amazonaws.com/msaexp-nirspec/extractions/jades-gds03-v4/'
    'jades-gds03-v4_{}_1286_187956.spec.fits'
)
spectra = spectrum.Spectra(
    [
        spectrum.from_DJA(DJA_URL.format(gf), disperser=d, cache=True)
        for gf, d in [
            ('prism-clear', prism_disperser),
            ('g140m-f070lp', g140m_disperser),
            ('g235m-f170lp', g235m_disperser),
            ('g395m-f290lp', g395m_disperser),
            ('g395h-f290lp', g395h_disperser),
        ]
    ],
    redshift=zspec,
)

# %%
# All five spectra together.  Redward of about 1.3 µm the G140M spectrum sits
# above the PRISM, and so does G235M redward of 3.2 µm: that is the
# second-order continuum.  G395M and G395H, behind the F290LP filter, agree
# with the PRISM.

fig, ax = pyplot.subplots(figsize=(12, 6))
# Gratings first, then the PRISM on top as the reference
for i, s in enumerate(spectra[1:]):
    ax.step(
        s.wavelength,
        s.flux,
        where='mid',
        color=f'C{i}',
        lw=0.5,
        alpha=0.8,
        label=s.disperser.name,
    )
ax.step(
    spectra[0].wavelength,
    spectra[0].flux,
    where='mid',
    color='k',
    lw=1.0,
    label=spectra[0].disperser.name,
)
ax.set(
    xlabel=rf'$\lambda$ (Obs.) [{spectra[0].unit:latex}]',
    ylabel=rf'$f_\lambda$ [{spectra[0].flux_unit:latex_inline}]',
    xlim=(0.7, 5.35),
    ylim=(-1, 25),
)
legend = ax.legend(loc='upper right')
for handle in legend.legend_handles:  # thicker lines in the legend
    handle.set(linewidth=2, alpha=1)
pyplot.show()

# %%
# Step 3 — Configure the Emission Lines and Continuum
# ------------------------------------------------------
#
# We build a narrow + broad model:
#
# * **Narrow**: every line shares one redshift ``z`` and FWHM ``narrow``
# * **Doublets/multiplets**: fixed ratios through one shared ``Flux`` and
#   ``strength``
# * **Broad hydrogen**: shared redshift and FWHM; free fluxes for
#   H\ :math:`\alpha` and H\ :math:`\beta`, while the broad fluxes of
#   H\ :math:`\gamma`, H\ :math:`\delta`, Pa\ :math:`\gamma` and
#   Pa\ :math:`\delta` are tied by dependent priors
#
# See :doc:`../usage/line_configuration` for the full line and profile
# reference and :doc:`../usage/priors` for dependent priors.

line_configuration = line.LineConfiguration()

# The redshift priors are relative to zspec (±0.1%)
dz = 0.001 * (1 + zspec)
z_narrow = line.Redshift('narrow', prior=prior.Uniform(-dz, dz, init=0.0))
z_broad = line.Redshift('broad', prior=prior.Uniform(-dz, dz, init=0.0))

# FWHMs in km/s
fwhm_narrow = line.FWHM('narrow', prior=prior.Uniform(10, 1000, init=200))
fwhm_broad = line.FWHM('broad', prior=prior.Uniform(800, 10_000, init=2500))


# Flux priors are half-normals in units of the line scale, so the width says
# how bright we expect a line to be: 0.5 for the strongest lines, 0.15 for
# typical ones, 0.05 for faint ones.
def flux_prior(width):
    """Half-normal flux prior, bounded at three line scales, started at its median."""
    return prior.TruncatedNormal(
        loc=0.0, scale=width, low=0.0, high=3.0, init=round(0.674 * width, 4)
    )


def narrow(name, center, width, **kwargs):
    """Add one narrow Gaussian line with its own flux and return that flux."""
    flux = line.Flux(name, prior=flux_prior(width))
    line_configuration.add_line(
        name,
        center * u.AA,
        profile='Gaussian',
        redshift=z_narrow,
        fwhm_gauss=fwhm_narrow,
        flux=flux,
        **kwargs,
    )
    return flux


def multiplet(name, centers, width, strength):
    """Add narrow lines that share one flux (named after the first), in fixed ratios."""
    names = [f'{name}_{c:.0f}' for c in centers]
    line_configuration.add_lines(
        names,
        centers * u.AA,
        profile='Gaussian',
        redshift=z_narrow,
        fwhm_gauss=fwhm_narrow,
        flux=line.Flux(names[0], prior=flux_prior(width)),
        strength=strength,
    )


# Hydrogen: each line has its own flux, which we keep to tie the broad lines
Ha = narrow('Ha', 6564.60, 0.5)
Hb = narrow('Hb', 4862.71, 0.5)
Hg = narrow('Hg', 4341.69, 0.5)
Hd = narrow('Hd', 4102.89, 0.15)
narrow('He', 3971.20, 0.05)
Pag = narrow('Pag', 10941.08, 0.15)
Pad = narrow('Pad', 10052.12, 0.05)
narrow('Pae', 9548.65, 0.05)

# Single forbidden and helium lines
narrow('OIII_4364', 4364.45, 0.15)
narrow('HeI_5877', 5877.29, 0.15)
narrow('HeI_7067', 7067.20, 0.05)
narrow('HeI_10833', 10833.14, 0.15)
narrow('OI_8449', 8448.78, 0.05)
narrow('SII_6718', 6718.30, 0.15)
narrow('SII_6733', 6732.67, 0.15)

# Doublets and multiplets with fixed ratios
multiplet('OII', [3727.09, 3729.84], 0.15, [1.0, 1.0])
multiplet('NeIII', [3869.85, 3968.59], 0.15, [3.32, 1.0])
multiplet('OIII', [4960.29, 5008.24], 0.5, [1.0, 2.98])
multiplet('OI', [6302.05, 6365.54], 0.15, [3.13, 1.0])
multiplet('NII', [6549.86, 6585.27], 0.15, [1.0, 2.95])
multiplet('OII', [7321.13, 7322.20, 7331.76, 7332.83], 0.05, [0.6, 2, 1, 1.05])
multiplet('SIII', [9070.05, 9532.25], 0.15, [1.0, 2.51])
multiplet('CI', [9826.80, 9852.94], 0.05, [1.0, 2.98])

# Broad hydrogen.  H-alpha and H-beta have free broad fluxes; the others keep
# the broad-to-narrow ratio of the nearest free line, through a Fixed prior
# that depends on other parameters.
Ha_broad = line.Flux('Ha_broad', prior=flux_prior(0.5))
Hb_broad = line.Flux('Hb_broad', prior=flux_prior(0.5))
broad_lines = {
    'Ha': (6564.60, Ha_broad),
    'Hb': (4862.71, Hb_broad),
    'Hg': (4341.69, line.Flux('Hg_broad', prior=prior.Fixed(Hg * Hb_broad / Hb))),
    'Hd': (4102.89, line.Flux('Hd_broad', prior=prior.Fixed(Hd * Hb_broad / Hb))),
    'Pag': (10941.08, line.Flux('Pag_broad', prior=prior.Fixed(Pag * Ha_broad / Ha))),
    'Pad': (10052.12, line.Flux('Pad_broad', prior=prior.Fixed(Pad * Ha_broad / Ha))),
}
for name, (center, flux) in broad_lines.items():
    line_configuration.add_line(
        f'{name}_broad',
        center * u.AA,
        profile='Gaussian',
        redshift=z_broad,
        fwhm_gauss=fwhm_broad,
        flux=flux,
    )

print(line_configuration)

# %%
# The continuum has to cover every wavelength the second-order term needs: the
# model at :math:`\lambda` includes :math:`F(\lambda/2)`, which is only
# non-zero where :math:`\lambda/2` lies inside a continuum region.  So instead
# of auto-generated windows around the lines, we use three regions that span
# all the spectra, split at the Balmer and Paschen limits, each with an
# order-5 Legendre polynomial.  For ``Legendre``, ``scale`` is the mean level
# over the region and the shape coefficients are in units of it.
#
# See :doc:`../usage/continuum_configuration` for the other continuum forms.

# The reddest pixel of any spectrum, in the rest frame
max_wavelength = max(float(s.high.max()) * s.unit for s in spectra) / (1 + zspec)
regions = {
    'UV': (1300.0 * u.AA, 3645.07 * u.AA),
    'optical': (3645.07 * u.AA, 8201.40 * u.AA),
    'NIR': (8201.40 * u.AA, max_wavelength.to(u.AA)),
}
cc = continuum.ContinuumConfiguration(
    [
        continuum.ContinuumRegion(low, high, form=continuum.Legendre(order=5), name=n)
        for n, (low, high) in regions.items()
    ]
)
print(cc)

# %%
# Step 4 — Prepare the Spectra and Fit with SVI
# ----------------------------------------------
#
# :meth:`~unite.spectrum.Spectra.prepare` filters lines and continuum regions
# to those covered by the spectra, and
# :meth:`~unite.spectrum.Spectra.compute_scales` sets the flux scales the
# priors are expressed in.  ``drop_empty_regions=False`` keeps continuum
# regions even where no lines fall, since the continuum is what we are after.
#
# See :doc:`../usage/build_model` for details.

filtered_lines, filtered_cont = spectra.prepare(
    line_configuration, cc, drop_empty_regions=False
)
spectra.compute_scales(
    filtered_lines,
    filtered_cont,
    line_mask_width=5000 * u.km / u.s,
    box_width=3000 * u.km / u.s,
)
print(f'Line scale:      {spectra.line_scale:.4g}')
print(f'Continuum scale: {spectra.continuum_scale:.4g}')

# %%
# With 50+ parameters and five spectra we use stochastic variational inference
# (SVI) rather than NUTS: a multivariate-normal guide started from the priors'
# ``init`` values and optimised with Adam.  The posterior samples are drawn
# from the guide.  See :doc:`../usage/inference` for other samplers.

model_fn, model_args = model.ModelBuilder(
    filtered_lines, filtered_cont, spectra
).build()

N_STEPS = 6000
guide = infer.autoguide.AutoMultivariateNormal(
    model_fn,
    init_loc_fn=infer.init_to_value(values=model_args.init_values),
    init_scale=0.01,
)
optimiser = numpyro.optim.ClippedAdam(
    step_size=lambda i: jnp.maximum(0.01 * 0.7 ** (8.0 * i / N_STEPS), 1e-4),
    clip_norm=10.0,
)
svi = infer.SVI(model_fn, guide, optimiser, loss=infer.Trace_ELBO(num_particles=4))
svi_result = svi.run(jax.random.PRNGKey(0), N_STEPS, model_args, progress_bar=False)
samples = guide.sample_posterior(
    jax.random.PRNGKey(1), svi_result.params, sample_shape=(200,)
)

# %%
# Step 5 — Results
# ----------------
#
# The calibration parameters, including the shared ``OrderScale``.  A value
# above one means the msaexp ratio underestimates the second-order light of
# this source.

percentiles = np.array([0.16, 0.5, 0.84])
param_table = results.make_parameter_table(samples, model_args, percentiles=percentiles)
calibration = [
    c
    for c in param_table.colnames
    if c.startswith(('order_scale', 'flux_scale', 'r_scale', 'pix_offset'))
    and np.ptp(param_table[c]) > 0  # skip the fixed ones
]
param_table[['percentile', *calibration]].pprint(max_width=-1)

# %%
# Each spectrum with the posterior median model.  For G140M and G235M the
# model is split into its first-order part and the second-order term; the two
# add up to the model.

spectra_tables = results.make_spectra_tables(
    samples, model_args, percentiles=percentiles
)


def break_gaps(wavelength, *columns):
    """Insert NaN where pixels are missing, so the steps do not bridge gaps.

    A gap is a step in wavelength more than twice as large as the steps on
    either side of it (pixel spacing varies smoothly, gaps do not).
    """
    wavelength = np.asarray(wavelength)
    step = np.diff(wavelength)
    neighbours = np.minimum(np.r_[step[:1], step[:-1]], np.r_[step[1:], step[-1:]])
    gaps = np.flatnonzero(step > 2 * neighbours)
    return [
        np.insert(np.asarray(c, dtype=float), gaps + 1, np.nan)
        for c in (wavelength, *columns)
    ]


fig, axes = pyplot.subplots(len(spectra_tables), 1, figsize=(12, 16), sharex=True)
fig.subplots_adjust(hspace=0.05, top=0.94, left=0.16)
# Lower y-limit per panel: just below each spectrum's continuum, and low
# enough in G235M to show its fainter second-order term
y_floor = {'PRISM': 2, 'G140M': 1, 'G235M': 0.1, 'G395M': 2, 'G395H': 2}
for ax, (name, table) in zip(axes, spectra_tables.items(), strict=True):
    has_overlap = 'overlap_2' in table.colnames
    wavelength, data, total, second = break_gaps(
        table['wavelength'].value,
        table['observed_flux'].value,
        table['model_total'][:, 1].value,
        table['overlap_2'][:, 1].value if has_overlap else np.zeros(len(table)),
    )
    ax.step(wavelength, data, where='mid', color='0.6', lw=0.6, label='Data')
    ax.step(wavelength, total, where='mid', color='k', lw=0.8, label='Model')
    if has_overlap:
        ax.step(
            wavelength,
            total - second,
            where='mid',
            color='C0',
            lw=0.8,
            label='First order',
        )
        ax.step(
            wavelength, second, where='mid', color='C3', lw=1.0, label='Second order'
        )
    ax.set(yscale='log', ylim=(y_floor[name], 2 * np.nanmax(total)), ylabel=name)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())  # label decades only
axes[0].set_xlim(0.6, 5.5)
# One legend for all panels, above the figure
fig.legend(
    *axes[1].get_legend_handles_labels(),
    loc='lower center',
    bbox_to_anchor=(0.5, 0.94),
    ncols=4,
    handlelength=1.5,
    columnspacing=1.0,
)
axes[-1].set_xlabel(rf'$\lambda$ (Obs.) [{spectra[0].unit:latex}]')
fig.supylabel(rf'$f_\lambda$ [{spectra[0].flux_unit:latex_inline}]', x=0.0)
pyplot.show()
