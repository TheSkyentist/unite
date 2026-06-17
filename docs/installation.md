# Installation

## From PyPI

`unite` is available on PyPI and can be installed with your favorite Python package manager:

```bash
pip install unite
uv pip install unite 
pixi add unite --pypi
```

---

## Dependencies

`unite` requires Python 3.12+ and depends on:

| Package | Role |
|---------|------|
| [JAX](https://jax.readthedocs.io/) | Fast array math and JIT compilation |
| [NumPyro](https://num.pyro.ai/) | Probabilistic programming and MCMC |
| [Astropy](https://www.astropy.org/) | Units, tables, and FITS I/O |
| [PyYAML](https://pyyaml.org/) | Configuration serialization |

---

## JAX 64-bit Mode

JAX defaults to 32-bit arithmetic, which is almost certainly sufficient for moderate
to high SNR spectroscopy. For very high-SNR data or long-tailed profiles (`SEMG`,
`GaussianSplitLaplace`), consider enabling 64-bit — `unite` is explicitly tested at
64-bit and it is not necessary for all science cases. Note that 64-bit may degrade
performance on GPUs. Enable it before any JAX import:

```python
import jax
jax.enable_64(True)

from unite import line, model, prior
```

Or via environment variable:

```bash
JAX_ENABLE_X64=1 python my_script.py
```

---

## Quick Start

The example below fits three emission lines (H$\alpha$ + [NII] doublet) in a simulated
spectrum. See {doc}`auto_tutorials/tutorial_generic` for the full annotated walkthrough.

```python
import jax
jax.config.update('jax_enable_x64', True)

import astropy.units as u
import numpy as np
from numpyro import infer

from unite import line, model, prior
from unite.continuum import ContinuumConfiguration, Linear
from unite.instrument.generic import SimpleDisperser
from unite.results import make_hdul, make_parameter_table
from unite.spectrum import Spectra, from_edges

# --- Simulate a spectrum ---
rng = np.random.default_rng(42)
wavelength = np.linspace(6400, 6700, 300) * u.AA
wl = wavelength.value

sigma = 5.0 / (2 * np.sqrt(2 * np.log(2)))
flux = (
    80 * np.exp(-0.5 * ((wl - 6563.0) / sigma) ** 2)
    + 30 * np.exp(-0.5 * ((wl - 6549.0) / sigma) ** 2)
    + 90 * np.exp(-0.5 * ((wl - 6585.0) / sigma) ** 2)
    + 10.0
    + rng.normal(0, 2, len(wl))
)
error = np.full_like(flux, 2.0)
low = wavelength - 0.5 * np.gradient(wavelength)
high = wavelength + 0.5 * np.gradient(wavelength)

# --- Build the model ---
disperser = SimpleDisperser(wavelength=wl, unit=u.AA, R=3000.0, name='sim')
spectrum = from_edges(low, high, flux, error, disperser)

z = line.Redshift('z', prior=prior.Uniform(-0.005, 0.005))
fwhm = line.FWHM('fwhm', prior=prior.Uniform(1.0, 10.0))

lc = line.LineConfiguration()
lc.add_line('H_alpha', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm,
            flux=line.Flux('Ha_flux', prior=prior.Uniform(0, 5)))
lc.add_line('NII_6585', 6585.0 * u.AA, redshift=z, fwhm_gauss=fwhm,
            flux=line.Flux('NII6585_flux', prior=prior.Uniform(0, 5)))
lc.add_line('NII_6549', 6549.0 * u.AA, redshift=z, fwhm_gauss=fwhm,
            flux=line.Flux('NII6549_flux', prior=prior.Uniform(0, 5)))

cc = ContinuumConfiguration.from_lines(lc.centers, pad=0.05, form=Linear())

spectra = Spectra([spectrum], redshift=0.0)
filtered_lines, filtered_cont = spectra.prepare(lc, cc)
spectra.compute_scales(filtered_lines, filtered_cont, error_scale=True)

# --- Sample ---
builder = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
model_fn, model_args = builder.build()

mcmc = infer.MCMC(infer.NUTS(model_fn), num_warmup=200, num_samples=500)
mcmc.run(jax.random.PRNGKey(0), model_args)

# --- Results ---
table = make_parameter_table(mcmc.get_samples(), model_args)
print(table['z', 'fwhm', 'Ha_flux'])
```

---

## Next Steps

- {doc}`concepts` — understand tokens, sharing, and the prepare → build pipeline
- {doc}`auto_tutorials/tutorial_generic` — full annotated generic spectrograph tutorial
- {doc}`auto_tutorials/tutorial_nirspec` — JWST/NIRSpec multi-grating fitting
- {doc}`api/index` — complete API reference
