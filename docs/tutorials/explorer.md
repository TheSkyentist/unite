# Exploring a Model Interactively

Before running a sampler it helps to see how your parameter and prior choices shape the
model. The explorer is a [marimo](https://marimo.io) notebook, `examples/explorer.py`,
that gives every free parameter of a built model a slider and a number box, and redraws
the model against the data of each spectrum as you move them.

```{image} ../images/explorer.png
:alt: The unite explorer showing sliders for each line and the G395M spectrum
:width: 100%
```

- **One figure per spectrum**, with the data, the total model, the continuum, and every
  line (click a line in the legend to show it). Figures can be zoomed and panned; the
  bottom axis is the rest frame and the top axis the observed frame.
- **Parameters are grouped** by line, by continuum region, and by spectrum (calibration).
  A parameter shared between lines appears under each of them, and changing it in one
  place changes it everywhere.
- **Dependent priors are respected.** If a prior's bounds depend on another parameter
  (for example `fwhm_broad > fwhm_narrow + 150`), the slider range follows the current
  value. Raising `fwhm_narrow` far enough pushes `fwhm_broad` up with it, but lowering it
  again does not pull `fwhm_broad` back. A change that would leave some parameter with no
  allowed values is rejected, with a message explaining why.
- **Starting values** are each prior's `init` if it has one, otherwise the midpoint of
  its bounds.

## Running it

The explorer needs `marimo`, `plotly` and `anywidget`. In the project's Pixi setup:

```bash
pixi run -e explore explore
```

which opens the notebook in the marimo editor. It is a plain Python file, so you can also
[download it from GitHub](https://github.com/TheSkyentist/unite/blob/main/examples/explorer.py),
install those three packages in your own environment, and run `marimo edit explorer.py`.

## Using your own data

The notebook's config cell builds the RUBIES NIRSpec example from the
{doc}`NIRSpec tutorial </auto_tutorials/tutorial_nirspec>`. To explore something else,
replace that cell so that it defines two names:

- `model_args`, the second item returned by
  {meth}`~unite.model.ModelBuilder.build`.
- `spectra`, the {class}`~unite.spectrum.Spectra` the model was built from.

The line and continuum configurations are only needed to build `model_args`; the explorer
itself does not use them.

```{raw} html
<p>A <a href="../_static/explorer.html">static snapshot</a> of the notebook is built with
these docs. Its sliders are not connected to Python, so run the notebook to interact
with it.</p>
```
