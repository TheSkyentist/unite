import marimo

__generated_with = '0.25.1'
app = marimo.App(width='full')


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # unite explorer

    This notebook uses [marimo](https://marimo.io) to give you an interactive way
    of exploring your parameter choices: every free parameter gets a slider and a
    number box, and each spectrum shows the model updating as you move them.
    Parameters shared between lines (or spectra) appear under each of them and
    move together, and parameters whose priors depend on others are kept inside
    their allowed range.

    To explore your own data, replace the config cell so that it defines
    `model_args` (from `ModelBuilder.build()`) and `spectra` (the `Spectra` the
    model was built from).  You can copy this notebook as a Python file and
    edit it.
    """)
    return


@app.cell
def _():
    import astropy.units as u
    import marimo as mo

    from unite import continuum, instrument, line, model, prior, spectrum
    from unite.instrument import nirspec

    # --- Replace this cell with your own config and spectra -----------------
    # RUBIES-UDS-807469 (a little red dot at z ~ 6.78), PRISM + G395M, as in the
    # NIRSpec tutorial.
    _resolution_scale = instrument.RScale(
        prior=prior.TruncatedNormal(low=0.6, high=1.4, loc=1.0, scale=0.1)
    )
    _prism = nirspec.PRISM(
        r_source='point',
        r_scale=_resolution_scale,
        pix_offset=instrument.PixOffset(
            prior=prior.TruncatedNormal(low=-0.2, high=0.6, loc=0.2, scale=0.1)
        ),
    )
    _g395m = nirspec.G395M(
        r_source='point',
        r_scale=_resolution_scale,
        flux_scale=instrument.FluxScale(
            prior=prior.TruncatedNormal(low=0.6, high=1.2, loc=0.9, scale=0.1)
        ),
    )
    _base = (
        'https://s3.amazonaws.com/msaexp-nirspec/extractions/rubies-uds42-v4/'
        'rubies-uds42-v4_'
    )
    spectra = spectrum.Spectra(
        [
            spectrum.from_DJA(
                _base + 'g395m-f290lp_4233_807469.spec.fits',
                disperser=_g395m,
                cache=True,
            ),
            spectrum.from_DJA(
                _base + 'prism-clear_4233_807469.spec.fits',
                disperser=_prism,
                cache=True,
            ),
        ],
        redshift=6.7754,
    )

    line_configuration = line.LineConfiguration()
    _z = line.Redshift('shared', prior=prior.Uniform(-0.005, 0.005))
    _narrow = line.FWHM('narrow', prior=prior.Uniform(100, 500))
    _broad = line.FWHM('broad', prior=prior.Uniform(_narrow + 150, 5000))
    line_configuration.add_line(
        'Ha', 6564.61 * u.AA, redshift=_z, fwhm_gauss=_narrow, profile='Gaussian'
    )
    line_configuration.add_line(
        'Hb', 4862.68 * u.AA, redshift=_z, fwhm_gauss=_narrow, profile='Gaussian'
    )
    line_configuration.add_lines(
        'OIII',
        [4960.295, 5008.24] * u.AA,
        profile='Gaussian',
        redshift=_z,
        fwhm_gauss=_narrow,
        strength=[1.0, 3.0],
        flux=line.Flux(prior=prior.Uniform(0, 3)),
    )
    line_configuration.add_lines(
        ['Ha_broad', 'Hb_broad'],
        [6564.61, 4862.68] * u.AA,
        profile='Exponential',
        redshift=_z,
        fwhm_exp=_broad,
        flux=[
            line.Flux(prior=prior.Uniform(0, 3)),
            line.Flux(prior=prior.Uniform(0, 3)),
        ],
    )

    cc = continuum.ContinuumConfiguration.from_lines(
        line_configuration.centers, width=15_000 * u.km / u.s, form=continuum.Linear()
    )

    _lines, _cont = spectra.prepare(line_configuration, cc)
    spectra.compute_scales(
        _lines,
        _cont,
        line_mask_width=5_000 * u.km / u.s,
        box_width=3_000 * u.km / u.s,
        error_scale=True,
    )
    # ------------------------------------------------------------------------
    model_args = model.ModelBuilder(line_configuration, cc, spectra).build()[1]
    return mo, model_args, spectra


@app.cell(hide_code=True)
def _(mo):
    # Needs marimo, plotly and anywidget.  Defines `Explorer(model_args, spectra)`:
    # sliders for every free parameter and one figure per spectrum.
    import itertools

    import anywidget
    import numpy as np
    import plotly.graph_objects as go
    import traitlets

    from unite.compute import evaluate_model
    from unite.prior import Fixed

    # Raised by a prior whose bounds are not valid at the current values.
    _bad_prior = (ValueError, FloatingPointError)

    _js = r"""
    function fmt(v) { return String(Number(v.toPrecision(5))); }

    function render({ model, el }) {
      const specs = model.get('specs');
      let values = { ...model.get('values') };
      const bounds = {};  // name -> {lo, hi}: the range allowed right now
      const rows = {};    // name -> [{range, num, loSpan, hiSpan}], one per place shown

      const mk = (tag, cls, text) => {
        const e = document.createElement(tag);
        if (cls) e.className = cls;
        if (text !== undefined) e.textContent = text;
        return e;
      };

      let timer = null;
      function send() {
        if (timer) return;
        timer = setTimeout(() => {
          timer = null;
          model.set('values', { ...values });
          model.save_changes();
        }, 30);
      }

      function show(name) {
        const { lo, hi } = bounds[name];
        for (const r of rows[name]) {
          r.loSpan.textContent = lo === null ? '\u2212∞' : fmt(lo);
          r.hiSpan.textContent = hi === null ? '∞' : fmt(hi);
          if (r.range) {
            r.range.min = lo;
            r.range.max = hi;
            r.range.value = values[name];
          }
          if (el.getRootNode().activeElement !== r.num) r.num.value = fmt(values[name]);
        }
      }

      function setValue(name, v) {
        const { lo, hi } = bounds[name];
        if (!Number.isFinite(v)) return show(name);
        if (lo !== null) v = Math.max(lo, v);
        if (hi !== null) v = Math.min(hi, v);
        values[name] = v;
        show(name);
        send();
      }

      function makeRow(name) {
        const { lo, hi } = specs[name];
        const row = mk('div', 'up-row');
        const head = mk('div', 'up-head');
        head.append(mk('span', 'up-name', name));
        const num = mk('input', 'up-num');
        num.type = 'number';
        num.step = 'any';
        num.addEventListener('change', () => {
          setValue(name, parseFloat(num.value));
          num.value = fmt(values[name]);  // show the clamped value even while focused
        });
        head.append(num);
        row.append(head);

        let range = null;
        const bar = mk('div', 'up-bar');
        const loSpan = mk('span', 'up-bound');
        bar.append(loSpan);
        if (lo !== null && hi !== null) {
          range = mk('input', 'up-range');
          range.type = 'range';
          range.step = (hi - lo) / 1000;
          range.addEventListener('input', () => setValue(name, parseFloat(range.value)));
          bar.append(range);
        } else {
          bar.append(mk('span', 'up-fill'));
        }
        const hiSpan = mk('span', 'up-bound');
        bar.append(hiSpan);
        row.append(bar);

        (rows[name] ||= []).push({ range, num, loSpan, hiSpan });
        return row;
      }

      const root = mk('div', 'up-root');
      const msg = mk('div', 'up-msg');
      root.append(msg);
      for (const sec of model.get('sections')) {
        root.append(mk('h3', 'up-section', sec.title));
        for (const g of sec.groups) {
          const d = mk('details', 'up-group');
          d.open = true;
          d.append(mk('summary', '', g.title));
          for (const name of g.names) d.append(makeRow(name));
          root.append(d);
        }
      }
      el.append(root);

      function apply(u) {
        Object.assign(bounds, u.bounds);
        Object.assign(values, u.values);
        msg.textContent = u.message || '';
        for (const name of Object.keys(rows)) show(name);
      }
      for (const name of Object.keys(specs)) bounds[name] = { ...specs[name] };
      apply(model.get('update'));

      // Python sends back the allowed ranges, any values it had to move, and a
      // message when it rejected a change.
      model.on('change:update', () => apply(model.get('update')));
    }
    export default { render };
    """

    _style = """
    .up-root { font-size: 0.85rem; }
    .up-msg:not(:empty) { margin: 0.5rem 0; padding: 0.4rem 0.6rem; border-radius: 4px;
                          background: color-mix(in srgb, #d55e00 18%, transparent); }
    .up-section { margin: 0.75rem 0 0.25rem; font-size: 1rem; font-weight: 600; }
    .up-group { border-top: 1px solid color-mix(in srgb, currentColor 20%, transparent);
                padding: 0.4rem 0; }
    .up-group > summary { cursor: pointer; font-weight: 500; margin-bottom: 0.3rem; }
    .up-row { margin: 0.35rem 0 0.6rem; }
    .up-head { display: flex; justify-content: space-between; align-items: center;
               gap: 0.5rem; }
    .up-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .up-num { width: 6.5rem; text-align: right; font: inherit; color: inherit;
              background: transparent; padding: 0.1rem 0.3rem;
              border: 1px solid color-mix(in srgb, currentColor 30%, transparent);
              border-radius: 4px; }
    .up-bar { display: flex; align-items: center; gap: 0.4rem; }
    .up-bound { font-size: 0.7rem; opacity: 0.6; min-width: 2.2rem; }
    .up-bound:last-child { text-align: right; }
    .up-range, .up-fill { flex: 1; min-width: 0; }
    """

    class ParamPanel(anywidget.AnyWidget):
        """Sliders with synced number boxes, grouped into collapsible sections.

        Parameters
        ----------
        specs : dict
            ``{name: {'lo': float | None, 'hi': float | None}}``, the widest range
            each parameter can take; ``None`` is an unbounded end (a number box
            is shown without a slider).
        sections : list
            ``[{'title': str, 'groups': [{'title': str, 'names': [str]}]}]``.
            A name listed in several groups is one parameter, shown in each.
        values : dict
            ``{name: float}``, written by the front end whenever a control moves.
        update : dict
            Python to front end: ``{'id': int, 'bounds': {name: {'lo', 'hi'}},
            'values': {name: float}, 'message': str}``.  ``bounds`` is the range
            allowed at the current values, ``values`` are values to move, and
            ``message`` is shown above the controls.  ``id`` must change on every
            send so that identical updates are still delivered.
        """

        _esm = _js
        _css = _style
        specs = traitlets.Dict().tag(sync=True)
        sections = traitlets.List().tag(sync=True)
        values = traitlets.Dict().tag(sync=True)
        update = traitlets.Dict().tag(sync=True)

    class Explorer:
        """Sliders for every free parameter of a model, and a figure per spectrum.

        Parameters
        ----------
        model_args : unite.model.ModelArgs
            From :meth:`unite.model.ModelBuilder.build`.
        spectra : unite.spectrum.Spectra
            The (prepared) spectra the model was built from; the data is drawn
            from these.

        Attributes
        ----------
        controls : marimo.ui.anywidget
            The sliders and number boxes (display it, e.g. in ``mo.sidebar``).
        plots : marimo.Html
            One figure per spectrum.
        state : dict
            ``{site name: value}`` for the free parameters.

        Notes
        -----
        Parameters whose prior bounds depend on other parameters are kept
        inside their allowed range, in dependency order.  A change that would
        push a parameter out of range moves it to the edge (so raising
        ``fwhm_narrow`` raises ``fwhm_broad``); a parameter is never pulled back
        when the bound relaxes again.  A change that would leave some parameter
        with an empty range is rejected.
        """

        COLORS = ('#3DB7E9', '#F748A5', '#359B73', '#D55E00', '#E69F00', '#2271B2')

        def __init__(self, model_args, spectra):
            self.args = model_args
            self.specs = self._specs()
            self.state = {n: v for n, (_, _, v) in self.specs.items()}
            self._n_updates = 0
            _, self.bounds, _ = self._resolve(self.state)

            self._widget = ParamPanel(
                specs={
                    n: {'lo': self._finite(lo), 'hi': self._finite(hi)}
                    for n, (lo, hi, _) in self.specs.items()
                },
                sections=self._sections(spectra),
                values=dict(self.state),
            )
            self._send({}, '')
            self._widget.observe(self._on_values, names='values')

            self._make_figures(spectra)
            self._refresh()
            self.controls = mo.ui.anywidget(self._widget)
            self.plots = mo.vstack([mo.ui.anywidget(f) for f in self.figs])

        # -- parameters ---------------------------------------------------

        @staticmethod
        def _finite(x):
            return float(x) if np.isfinite(x) else None

        def _specs(self):
            """Return ``{site: (low, high, default)}`` for every free parameter.

            The default is the prior's ``init`` if set, else the midpoint of its
            bounds (its mean if a bound is infinite).  Bounds may depend on other
            parameters (e.g. broad fwhm > narrow fwhm + 150), so each is resolved
            at every corner of its dependencies' ranges and the widest interval
            is kept; the defaults are resolved in dependency order.
            """
            args = self.args
            defaults = {}  # token -> default value
            ranges = {}  # token -> (low, high)
            specs = {}
            for name in args.dependency_order:
                prior_ = args.all_priors[name]
                token = args.name_to_token.get(name)
                if isinstance(prior_, Fixed):
                    value = float(prior_.resolved_value(defaults))
                    lo = hi = value
                else:
                    # An infinite end of a dependency's range falls back to its
                    # default.
                    deps = list(prior_.dependencies())
                    ends = [
                        [
                            e if np.isfinite(e) else defaults[d]
                            for e in dict.fromkeys(ranges[d])
                        ]
                        for d in deps
                    ]
                    lows, highs = [], []
                    for corner in itertools.product(*ends):
                        context = {**defaults, **dict(zip(deps, corner, strict=True))}
                        support = prior_.to_dist(context).support
                        lows.append(float(support.lower_bound))
                        highs.append(float(support.upper_bound))
                    lo, hi = min(lows), max(highs)

                    support = prior_.to_dist(defaults).support
                    d_lo, d_hi = float(support.lower_bound), float(support.upper_bound)
                    if prior_.init is not None:
                        value = prior_.init
                    elif np.isfinite(d_lo) and np.isfinite(d_hi):
                        value = 0.5 * (d_lo + d_hi)
                    else:
                        value = float(prior_.to_dist(defaults).mean)
                    specs[name] = (lo, hi, value)
                if token is not None:
                    defaults[token] = value
                    ranges[token] = (lo, hi)
            return specs

        def _resolve(self, values):
            """Clamp *values* into their priors' support, in dependency order.

            Returns
            -------
            values, bounds, failed
                The clamped values and the allowed ``(low, high)`` of each
                parameter at those values.  If some parameter's range is empty,
                ``values`` and ``bounds`` are ``None`` and ``failed`` is its name.
            """
            args = self.args
            context = {}  # token -> value
            out, bounds = {}, {}
            for name in args.dependency_order:
                prior_ = args.all_priors[name]
                if isinstance(prior_, Fixed):
                    value = float(prior_.resolved_value(context))
                else:
                    try:
                        support = prior_.to_dist(context).support
                        lo, hi = float(support.lower_bound), float(support.upper_bound)
                    except _bad_prior:
                        return None, None, name
                    if not lo <= hi:  # also catches NaN
                        return None, None, name
                    value = min(max(values[name], lo), hi)
                    out[name] = value
                    bounds[name] = (lo, hi)
                token = args.name_to_token.get(name)
                if token is not None:
                    context[token] = value
            return out, bounds, None

        def _sections(self, spectra):
            """Group the free parameters by line, continuum region and spectrum."""
            args, cm = self.args, self.args.matrices
            matrices = [
                (cm.flux_names, cm.flux_matrix),
                (cm.tau_names, cm.tau_matrix),
                (cm.z_names, cm.z_matrix),
                (cm.p0_names, cm.p0_matrix),
                (cm.p1v_names, cm.p1v_matrix),
                (cm.p1d_names, cm.p1d_matrix),
                (cm.p2v_names, cm.p2v_matrix),
                (cm.p2d_names, cm.p2d_matrix),
            ]

            def group(title, names):
                return {'title': title, 'names': [n for n in names if n in self.specs]}

            lines = [
                group(
                    label,
                    [
                        name
                        for names, m in matrices
                        for i, name in enumerate(names)
                        if np.asarray(m)[i, j]
                    ],
                )
                for j, label in enumerate(args.line_labels)
            ]
            regions = [
                group(label, [tok.name for tok in resolved.values()])
                for label, resolved in zip(
                    args.continuum_labels, args.cont_resolved_params, strict=True
                )
            ]
            calibration = [
                group(
                    s.name,
                    [
                        tok.name
                        for tok in (
                            s.disperser.r_scale,
                            s.disperser.flux_scale,
                            s.disperser.pix_offset,
                        )
                    ],
                )
                for s in spectra
            ]
            return [
                {'title': 'Lines', 'groups': lines},
                {'title': 'Continuum', 'groups': regions},
                {'title': 'Calibration', 'groups': calibration},
            ]

        # -- reacting to the controls ---------------------------------------

        def _send(self, values, message):
            self._n_updates += 1
            self._widget.update = {
                'id': self._n_updates,
                'bounds': {
                    n: {'lo': self._finite(lo), 'hi': self._finite(hi)}
                    for n, (lo, hi) in self.bounds.items()
                },
                'values': values,
                'message': message,
            }

        def _on_values(self, change):
            new = dict(change['new'])
            changed = [n for n in new if new[n] != self.state[n]]
            if not changed:
                return
            values, bounds, failed = self._resolve(new)
            if values is None:
                # Put back what the user just moved.
                self._send(
                    {n: self.state[n] for n in changed},
                    f'Rejected: changing {', '.join(changed)} leaves {failed} '
                    'with an empty range.',
                )
                return
            moved = {n: values[n] for n in values if values[n] != new[n]}
            self.state, self.bounds = values, bounds
            self._send(moved, '')
            self._refresh()

        # -- figures --------------------------------------------------------

        def _make_figures(self, spectra):
            """One figure per spectrum, created once and updated in place.

            The model only covers the continuum windows, so NaNs are inserted at
            the gaps between them and the data is cropped to their span.
            """
            axis = {
                'showline': True,
                'linewidth': 2,
                'mirror': 'ticks',
                'ticks': 'inside',
                'ticklen': 8,
                'showgrid': False,
                'zeroline': False,
                'minor': {'ticks': 'inside', 'ticklen': 5, 'showgrid': False},
            }
            # The data are drawn in the rest frame (bottom axis); an invisible
            # trace on an overlaying axis gives the observed frame (top axis).
            # Plotly scales overlaying axes together, so they stay in step
            # when zooming.
            z1 = 1 + float(spectra.redshift)
            self.figs = []
            self._gaps = []  # per spectrum: insertion indices of the NaNs
            for s, ms in zip(spectra, self.args.spectra, strict=True):
                lo, hi = np.asarray(ms.low), np.asarray(ms.high)
                idx = np.where(lo[1:] - hi[:-1] > 1e-3 * (hi - lo)[:-1])[0] + 1
                mid = 0.5 * (hi[idx - 1] + lo[idx])
                self._gaps.append(idx)
                wl = np.insert(np.asarray(ms.wavelength), idx, mid) / z1
                nan = np.full_like(wl, np.nan)

                pad = 0.1 * (hi[-1] - lo[0])
                xrange_obs = [lo[0] - pad, hi[-1] + pad]
                xrange = [x / z1 for x in xrange_obs]
                w = np.asarray(s.wavelength)
                keep = (w > xrange_obs[0]) & (w < xrange_obs[1])
                unit = s.unit.to_string('unicode')
                flux_unit = s.flux_unit.to_string('unicode')

                fig = go.Figure()
                fig.add_scatter(
                    x=w[keep] / z1,
                    y=np.asarray(s.flux)[keep],
                    error_y={
                        'type': 'data',
                        'array': np.asarray(s.error)[keep],
                        'thickness': 0.8,
                        'width': 0,
                        'color': 'black',
                    },
                    mode='lines',
                    line={'shape': 'hvh', 'color': 'black', 'width': 1},
                    name='data',
                )
                fig.add_scatter(
                    x=wl,
                    y=nan,
                    mode='lines',
                    line={'shape': 'hvh', 'color': self.COLORS[0], 'width': 2},
                    name='model',
                )
                fig.add_scatter(
                    x=wl,
                    y=nan,
                    mode='lines',
                    line={
                        'shape': 'hvh',
                        'color': self.COLORS[3],
                        'width': 1.5,
                        'dash': 'dot',
                    },
                    name='continuum',
                )
                for k, label in enumerate(self.args.line_labels):
                    fig.add_scatter(
                        x=wl,
                        y=nan,
                        mode='lines',
                        line={
                            'shape': 'hvh',
                            'color': self.COLORS[(k + 1) % len(self.COLORS)],
                            'width': 1,
                        },
                        name=label,
                        visible='legendonly',
                    )
                fig.add_scatter(
                    x=xrange_obs,
                    y=[None, None],
                    xaxis='x2',
                    mode='markers',
                    showlegend=False,
                    hoverinfo='skip',
                )
                fig.update_layout(
                    template='simple_white',
                    title=s.name,
                    height=420,
                    margin={'t': 80, 'b': 50, 'l': 70, 'r': 20},
                    font={'family': 'serif', 'size': 14},
                    xaxis={
                        **axis,
                        'title': f'λ<sub>rest</sub> [{unit}]',
                        'range': xrange,
                        'mirror': False,
                    },
                    xaxis2={
                        **axis,
                        'title': f'λ<sub>obs</sub> [{unit}]',
                        'range': xrange_obs,
                        'overlaying': 'x',
                        'side': 'top',
                        'mirror': False,
                    },
                    yaxis={**axis, 'title': f'f<sub>λ</sub> [{flux_unit}]'},
                    legend={'bgcolor': 'rgba(0,0,0,0)'},
                )
                self.figs.append(go.FigureWidget(fig))

        def _refresh(self):
            """Evaluate the model at :attr:`state` and update the figures."""
            samples = {k: np.array([v]) for k, v in self.state.items()}
            preds = evaluate_model(samples, self.args)
            for fig, pred, idx in zip(self.figs, preds, self._gaps, strict=True):
                cont = sum(pred.continuum_regions.values(), np.zeros_like(pred.total))
                ys = [pred.total, cont, *pred.lines.values()]
                with fig.batch_update():
                    for trace, y in zip(fig.data[1:-1], ys, strict=True):
                        trace.y = np.insert(y[0], idx, np.nan)

    return (Explorer,)


@app.cell(hide_code=True)
def _(Explorer, mo, model_args, spectra):
    explorer = Explorer(model_args, spectra)
    _scroll = {'max-height': '92vh', 'overflow-y': 'auto'}
    mo.hstack(
        [explorer.controls.style(_scroll), explorer.plots.style(_scroll)],
        widths=[1, 3],
        align='start',
    )
    return


if __name__ == '__main__':
    app.run()
