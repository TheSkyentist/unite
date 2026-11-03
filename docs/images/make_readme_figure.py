r"""Make the README hero figure: one ``unite`` model, two NIRSpec spectra.

Re-uses the setup of ``docs/tutorials/tutorial_nirspec.py`` (RUBIES-UDS-807469,
PRISM + G395M fit simultaneously with shared narrow + broad Balmer / [OIII] lines),
then draws the fit on top and the model's priors and dependencies below.

Usage (from the repository root)::

    pixi run -e test-py312 python docs/images/make_readme_figure.py
    pixi run -e test-py312 python docs/images/make_readme_figure.py --refit

The posterior samples are cached in ``docs/images/.readme_samples.npz`` so that
restyling the figure does not re-run NUTS. Pass ``--refit`` to force a new fit.
Needs a LaTeX installation (the ``unite.mplstyle`` uses ``text.usetex``).
"""

import argparse
import pickle
from pathlib import Path

import astropy.units as u
import jax
import numpy as np
import numpyro
from matplotlib import gridspec, pyplot
from numpyro import infer

from unite import continuum, instrument, line, model, prior, results, spectrum
from unite.instrument import nirspec

HERE = Path(__file__).resolve().parent
STYLE = HERE.parent / 'tutorials' / 'unite.mplstyle'
CACHE = HERE / '.readme_samples.pkl'
ZSPEC = 6.7754

# One colour per shared parameter, used for BOTH the model curves and the maths.
COL = {
    'z': '#2271B2',  # shared redshift
    'narrow': '#359B73',  # FWHM_narrow + narrow components
    'broad': '#D55E00',  # FWHM_broad + broad components
    'flux': '#7B4FA3',  # line fluxes
    'rscale': '#E69F00',  # resolution scale (shared by both dispersers)
    'fscale': '#F748A5',  # G395M flux scale
    'poff': '#3DB7E9',  # PRISM pixel offset
}


def rich(ax, x, y, segments, fontsize=20, **kw):
    """Draw coloured text segments left to right starting at axes coords (x, y).

    ``segments`` is a list of ``(latex, colour_key_or_None)``. Matplotlib's usetex
    backend drops LaTeX colour commands, so each piece is placed separately.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    for tex, key in segments:
        t = ax.text(
            x,
            y,
            tex,
            color=COL[key] if key else 'k',
            fontsize=fontsize,
            transform=ax.transAxes,
            **kw,
        )
        bb = t.get_window_extent(renderer)
        x = inv.transform((bb.x1, bb.y0))[0] + 0.009


# ---------------------------------------------------------------------------
# Model + fit (mirrors tutorial_nirspec.py)
# ---------------------------------------------------------------------------
def build_and_fit(refit: bool):
    """Configure, prepare and (optionally) fit; return everything needed to plot."""
    numpyro.set_host_device_count(2)

    resolution_scale = instrument.RScale(
        prior=prior.TruncatedNormal(low=0.6, high=1.4, loc=1.0, scale=0.1)
    )
    prism = nirspec.PRISM(
        r_source='point',
        r_scale=resolution_scale,
        pix_offset=instrument.PixOffset(
            prior=prior.TruncatedNormal(low=-0.2, high=0.6, loc=0.2, scale=0.1)
        ),
    )
    g395m = nirspec.G395M(
        r_source='point',
        r_scale=resolution_scale,
        flux_scale=instrument.FluxScale(
            prior=prior.TruncatedNormal(low=0.6, high=1.2, loc=0.9, scale=0.1)
        ),
    )

    base = 'https://s3.amazonaws.com/msaexp-nirspec/extractions/rubies-uds42-v4/'
    s_g395m = spectrum.from_DJA(
        base + 'rubies-uds42-v4_g395m-f290lp_4233_807469.spec.fits',
        disperser=g395m,
        cache=True,
    )
    s_prism = spectrum.from_DJA(
        base + 'rubies-uds42-v4_prism-clear_4233_807469.spec.fits',
        disperser=prism,
        cache=True,
    )
    spectra = spectrum.Spectra([s_g395m, s_prism], redshift=ZSPEC)

    lc = line.LineConfiguration()
    z = line.Redshift('shared', prior=prior.Uniform(-0.005, 0.005))
    fwhm_n = line.FWHM('narrow', prior=prior.Uniform(100, 500))
    fwhm_b = line.FWHM('broad', prior=prior.Uniform(fwhm_n + 150, 5000))
    lc.add_line('Ha', 6564.61 * u.AA, redshift=z, fwhm_gauss=fwhm_n)
    lc.add_line('Hb', 4862.68 * u.AA, redshift=z, fwhm_gauss=fwhm_n)
    lc.add_lines(
        'OIII',
        [4960.295, 5008.24] * u.AA,
        redshift=z,
        fwhm_gauss=fwhm_n,
        strength=[1.0, 3.0],
        flux=line.Flux(prior=prior.Uniform(0, 3)),
    )
    lc.add_lines(
        ['Ha_broad', 'Hb_broad'],
        [6564.61, 4862.68] * u.AA,
        profile='Exponential',
        redshift=z,
        fwhm_exp=fwhm_b,
        flux=[
            line.Flux(prior=prior.Uniform(0, 3)),
            line.Flux(prior=prior.Uniform(0, 3)),
        ],
    )
    cc = continuum.ContinuumConfiguration.from_lines(
        lc.centers, width=15_000 * u.km / u.s, form=continuum.Linear()
    )
    fl, fc = spectra.prepare(lc, cc)
    spectra.compute_scales(
        fl,
        fc,
        line_mask_width=5_000 * u.km / u.s,
        box_width=3_000 * u.km / u.s,
        error_scale=True,
    )
    builder = model.ModelBuilder(fl, fc, spectra)
    model_fn, model_args = builder.build()

    if CACHE.exists() and not refit:
        samples = pickle.loads(CACHE.read_bytes())
    else:
        mcmc = infer.MCMC(
            infer.NUTS(model_fn, dense_mass=True),
            num_warmup=500,
            num_samples=1000,
            num_chains=2,
            progress_bar=False,
        )
        mcmc.run(jax.random.PRNGKey(0), model_args)
        samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
        CACHE.write_bytes(pickle.dumps(samples))

    tables = results.make_spectra_tables(
        samples, model_args, insert_nan=True, percentiles=[0.16, 0.5, 0.84]
    )
    return spectra, tables


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def make_figure(spectra, tables, out: Path):
    """Draw spectra (top) and the prior / dependency maths (bottom)."""
    pyplot.style.use(str(STYLE))
    pyplot.rcParams.update(
        {
            'font.size': 16,
            'text.latex.preamble': r'\usepackage{amsmath}\usepackage{xcolor}',
        }
    )

    fig = pyplot.figure(figsize=(15, 17.5))
    outer = gridspec.GridSpec(
        3, 1, figure=fig, height_ratios=[0.11, 1.0, 0.84], hspace=0.17
    )
    ax_title = fig.add_subplot(outer[0])
    ax_title.axis('off')
    ax_title.text(
        0.5,
        -0.1,
        r'\textbf{unite: One Model, Many Spectra}',
        ha='center',
        va='bottom',
        fontsize=34,
        linespacing=1.25,
    )

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=outer[1], hspace=0, wspace=0.0
    )
    axes = np.array(
        [[fig.add_subplot(inner[i, j]) for j in range(2)] for i in range(2)]
    )
    for j in range(2):
        axes[0, j].sharex(axes[1, j])

    # x-limits: only where the model is defined (any spectrum), per column
    xlims = []
    for lo, hi in [(0.0, 4.5), (4.5, 99.0)]:
        w = np.concatenate(
            [
                np.asarray(t['wavelength'])[np.isfinite(t['model_total'][:, 1].value)]
                for t in tables.values()
            ]
        )
        w = w[(w > lo) & (w < hi)]
        pad = 0.02 * (w.max() - w.min())
        xlims.append((w.min() - pad, w.max() + pad))
    ylims = {'G395M': (0, 18), 'PRISM': (0, 7.5)}
    titles = [r'H$\beta$ + [O\textsc{iii}]', r'H$\alpha$']
    narrow_names = ['Ha', 'Hb', 'OIII_4960.3', 'OIII_5008.24']

    order = {s.disperser.name: s for s in spectra}
    for row, name in enumerate(['G395M', 'PRISM']):
        s = order[name]
        tab = tables[s.name]
        wl = np.asarray(tab['wavelength'])
        cont = sum(
            np.nan_to_num(np.asarray(tab[c][:, 1].value))
            for c in tab.colnames
            if c.startswith('linear_')
        )
        for col in range(2):
            ax = axes[row, col]
            ax.step(s.wavelength, s.flux, where='mid', color='0.55', lw=0.9, zorder=1)
            ax.step(
                wl,
                np.asarray(tab['model_total'][:, 1].value),
                where='mid',
                color='k',
                lw=1.6,
                zorder=5,
            )
            # components: only those present in this panel
            in_panel = (wl > xlims[col][0]) & (wl < xlims[col][1])
            yr = ylims[name][1] - ylims[name][0]
            for names, c in (
                (narrow_names, COL['narrow']),
                (['Ha_broad', 'Hb_broad'], COL['broad']),
            ):
                for nm in names:
                    if nm not in tab.colnames:
                        continue
                    comp = np.asarray(tab[nm][:, 1].value)
                    if not np.nanmax(np.abs(comp[in_panel])) > 0.01 * yr:
                        continue
                    y = comp + cont
                    ax.step(wl, y, where='mid', color=c, lw=3.0, zorder=4)
            ax.set(xlim=xlims[col], ylim=ylims[name])
            if col == 1:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.tick_params(labelbottom=False)
                if name == 'G395M':
                    ax.set_title(titles[col], pad=10, fontsize=24)
            else:
                ax.set_xlabel(r'$\lambda_{\rm obs}$ [$\mu$m]', fontsize=20)
        axes[row, 0].set_ylabel(rf'\textbf{{{name}}}  $f_\lambda$', fontsize=20)
        axes[row, 0].tick_params(labelsize=16)
        axes[row, 0].tick_params(labelsize=16)
    for ax in axes[1]:
        ax.tick_params(axis='x', labelsize=16)
    axes[0, 0].set_yticks([0, 5, 10, 15])
    axes[1, 0].set_yticks([0, 2, 4, 6])

    fig.canvas.draw()
    rich(
        axes[0, 0],
        0.03,
        0.94,
        [
            (r'\textbf{G395M}:', None),
            (r'\texttt{FluxScale}', 'fscale'),
            (r'\texttt{RScale}', 'rscale'),
        ],
        fontsize=16,
        va='top',
    )
    rich(
        axes[1, 0],
        0.03,
        0.94,
        [
            (r'\textbf{PRISM}:', None),
            (r'\texttt{PixOffset}', 'poff'),
            (r'\texttt{RScale}', 'rscale'),
        ],
        fontsize=16,
        va='top',
    )
    # legend (right of the Halpha panel, G395M row)
    lx, ly = 0.58, 0.94
    for k, (txt, key, col) in enumerate(
        [
            ('data', None, '0.55'),
            ('total model', None, 'k'),
            ('narrow lines', 'narrow', COL['narrow']),
            ('broad lines', 'broad', COL['broad']),
        ]
    ):
        yy = ly - 0.07 * k
        axes[0, 1].plot(
            [lx, lx + 0.06],
            [yy - 0.02] * 2,
            color=col,
            lw=3,
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].text(
            lx + 0.075,
            yy,
            txt,
            transform=axes[0, 1].transAxes,
            va='top',
            fontsize=17,
            color=COL[key] if key else 'k',
        )

    # ---- maths panel: one section per row ----
    axm = fig.add_subplot(outer[2])
    axm.axis('off')
    axm.add_patch(
        pyplot.Rectangle(
            (0, 0),
            1,
            1,
            transform=axm.transAxes,
            fc='0.96',
            ec='0.6',
            lw=1.5,
            clip_on=False,
        )
    )
    x0 = 0.025

    def title(y, text):
        axm.text(
            x0, y, rf'\textbf{{{text}}}', fontsize=24, va='top', transform=axm.transAxes
        )

    def line_(y, segs, x=x0, fs=21):
        rich(axm, x, y, segs, fontsize=fs, va='top')

    def divider(y):
        axm.plot([0.015, 0.985], [y, y], color='0.7', lw=1.2, transform=axm.transAxes)

    fwn = r'$\mathrm{FWHM}_{\rm n}$'
    fwb = r'$\mathrm{FWHM}_{\rm b}$'

    # 1. shared / dependent priors
    title(0.975, 'Shared and dependent priors')
    line_(0.895, [(r'$z$', 'z'), (r'$\sim\mathcal{U}(-0.005,\,0.005)$', None)], fs=19)
    line_(
        0.895, [(fwn, 'narrow'), (r'$\sim\mathcal{U}(100,\,500)$', None)], x=0.31, fs=19
    )
    line_(
        0.895,
        [
            (fwb, 'broad'),
            (r'$\sim\mathcal{U}($', None),
            (fwn, 'narrow'),
            (r'$+150,\,5000)$', None),
        ],
        x=0.585,
        fs=19,
    )
    line_(
        0.815,
        [
            (fwb, 'broad'),
            (r'depends on', None),
            (fwn, 'narrow'),
            (r'(broad lines are always broader),', None),
            (r'$z$', 'z'),
            (r'is shared by every line', None),
        ],
        fs=17,
    )
    divider(0.745)

    # 2. correlated fluxes
    title(0.715, 'Correlated fluxes')
    line_(0.635, [(r'$F_{5008}=3\,F_{4960}=3\,$', None), (r'$F_{[\rm OIII]}$', 'flux')])
    line_(
        0.635,
        [
            (r'the', None),
            (r'[OIII] doublet', 'flux'),
            (r'is one parameter with a fixed 1:3 ratio', None),
        ],
        x=0.40,
        fs=17,
    )
    divider(0.565)

    # 3. calibration
    title(0.535, 'Calibration offsets between spectra')
    line_(
        0.455,
        [
            (r'\texttt{RScale}', 'rscale'),
            (r'inflates the LSF to absorb slit and morphology uncertainty,', None),
            (r'shared', 'rscale'),
            (r'by both spectra', None),
        ],
        fs=18,
    )
    line_(
        0.385,
        [
            (r'\texttt{FluxScale}', 'fscale'),
            (r'absorbs the flux mismatch between', None),
            (r'G395M', 'fscale'),
            (r'and the', None),
            (r'PRISM', None),
            (r'reference', None),
        ],
        fs=18,
    )
    line_(
        0.315,
        [
            (r'\texttt{PixOffset}', 'poff'),
            (r'absorbs sub-pixel wavelength shifts in the', None),
            (r'PRISM', 'poff'),
            (r'spectrum', None),
        ],
        fs=18,
    )
    line_(
        0.245,
        [
            (r'each has its own prior, e.g.', None),
            (r'$\mathcal{N}_{[0.6,1.2]}(0.9,\,0.1)$', 'fscale'),
            (r'for the G395M flux scale', None),
        ],
        fs=17,
    )
    divider(0.18)

    # 4. what else
    axm.text(
        0.5,
        0.155,
        r'\textbf{This is only a fraction of what} \texttt{unite} \textbf{can do}',
        ha='center',
        va='top',
        fontsize=23,
        transform=axm.transAxes,
    )
    axm.text(
        0.5,
        0.095,
        r'JAX-powered and JIT-compiled $\cdot$ GPU-accelerated $\cdot$ '
        r'any NumPyro sampler (NUTS, nested sampling, SVI)'
        '\n'
        r'tau-parametrized absorbers $\cdot$ a library of line profiles and continuum '
        r'forms $\cdot$ human-readable, shareable configs',
        ha='center',
        va='top',
        fontsize=17,
        linespacing=1.5,
        transform=axm.transAxes,
    )

    fig.savefig(out.with_suffix('.png'), dpi=200, bbox_inches='tight')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    print(f'Wrote {out.with_suffix('.png')}')


def main():
    """Run the fit (or load cached samples) and draw the figure."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--refit', action='store_true', help='re-run NUTS')
    parser.add_argument('--out', type=Path, default=HERE / 'readme_figure')
    args = parser.parse_args()
    spectra, tables = build_and_fit(args.refit)
    make_figure(spectra, tables, args.out)


if __name__ == '__main__':
    main()
