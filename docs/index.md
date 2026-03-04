# unite

**Unified Numerical Integration Tool for spEctroscopy**

`unite` is a Python package for instrument modelling and spectroscopic data analysis,
built on top of [JAX](https://jax.readthedocs.io/),
[NumPyro](https://num.pyro.ai/), and [Astropy](https://www.astropy.org/).

## Getting Started

### Installation

```bash
pip install unite
```

For development, clone the repository and install with all extras:

```bash
git clone https://github.com/rhviding/unite.git
cd unite
pip install -e ".[dev]"
```

Or using [pixi](https://pixi.sh/):

```bash
pixi install
```

### Quick Example

```python
import unite
print(unite.__version__)
```

## Contents

```{toctree}
:maxdepth: 2

api/index
```

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
