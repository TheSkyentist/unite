# Sampling & Optimization

`unite` builds a NumPyro model function for you — it does not run inference itself.
{meth}`~unite.model.ModelBuilder.build` returns `(model_fn, model_args)`, and you
pass these to any [NumPyro](https://num.pyro.ai/en/stable/) inference kernel you choose.

In this section we only provide a quick introduction to running inference with NumPyro, for going beyond the basics please consult the [NumPyro documentation](https://num.pyro.ai/en/stable/). The recommended sampler is NUTS, but you can use any NumPyro inference method that suits your needs, or even extract the likelihood/posterior and pass it to an external optimizer or sampler.

---

## Setup & Performance

### JAX 64-bit Mode

JAX defaults to 32-bit. 64-bit is strongly recommended especially for line profiles with long tails. 

```python
import jax
jax.config.update('jax_enable_x64', True)

# Import unite and other JAX-dependent packages after
from unite import line, model, prior
```

Or via environment variable before starting Python:

```bash
JAX_ENABLE_X64=1 python my_script.py
```

Or via `pixi`'s environment configuration:

```toml
[activation.env]
JAX_ENABLE_X64 = "1"
```

### GPU Acceleration

JAX runs on GPU transparently — no code changes are needed. If a GPU is available and
JAX is installed with CUDA support, it will be used automatically.

```python
import jax
print(jax.devices())           # e.g. [CudaDevice(id=0)]
print(jax.default_backend())   # 'gpu', 'cpu', or 'tpu'

# Select a specific device
with jax.default_device(jax.devices('gpu')[0]):
    mcmc.run(jax.random.PRNGKey(0), model_args)
```

Install JAX with GPU support following the
[official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).

---

## Running the Model

`unite` provides a convenience function for running basic MCMC inference. This provides a thin wrapper to NUTS (see below).

```python
from unite import model
builder = model.ModelBuilder(line_config, cont_config, spectra)
samples = builder.fit(
    num_warmup = 250,       # Warmup samples
    num_samples = 1000,     # Number of Samples
    num_chains = 1,         # Number of Chains
    seed = 0,               # Random Seed
    progress_bar = True,    # Display a progress bar
)
```

---

## Inference Methods

### NUTS (No-U-Turn Sampler)

[NUTS](https://num.pyro.ai/en/stable/mcmc.html#numpyro.infer.mcmc.NUTS) is the
recommended sampler for most problems. It uses gradient information to efficiently
explore the posterior.

```python
import jax
from numpyro import infer

kernel = infer.NUTS(model_fn, dense_mass = True) # dense_mass=True helps with correlated parameters
mcmc = infer.MCMC(
    kernel,
    num_warmup=500,
    num_samples=1000,
    progress_bar=True,
)
mcmc.run(jax.random.PRNGKey(0), model_args)
samples = mcmc.get_samples()
```

Running multiple chains in parallel is the best way to assess convergence (R-hat):

```python
mcmc = infer.MCMC(
    infer.NUTS(model_fn),
    num_warmup=500,
    num_samples=1000,
    num_chains=4,
    progress_bar=True,
)
mcmc.run(jax.random.PRNGKey(0), model_args)
```

With a GPU, all chains run concurrently at essentially no extra cost. On CPU, chains
run sequentially by default, but can be configured otherwise.

For complex posteriors (many components, tight constraints) you may need more warmup or
a larger target acceptance probability:

```python
kernel = infer.NUTS(
    model_fn,
    target_accept_prob=0.9,   # default 0.8; increase for difficult posteriors
    max_tree_depth=12,        # default 10; increase if chains get stuck, or decrease to speed up if you get "divergent transition after max tree depth" warnings
    dense_mass=True,          # full mass matrix; helps with correlated parameters
)
```

---

### SVI — Stochastic Variational Inference

[SVI](https://num.pyro.ai/en/stable/svi.html) approximates the posterior with a simpler
family of distributions (e.g., multivariate normal). It is much faster than NUTS and
useful for:
- rapid iteration on priors and configuration
- large surveys where full MCMC is too slow
- initializing NUTS with a good starting point

```python
from numpyro import infer, optim

guide = autoguide.AutoMultivariateNormal(model_fn)
svi = infer.SVI(
    fit_func, 
    guide, 
    optim.Adam(step_size=0.01), 
    loss=infer.Trace_ELBO()
)
svi_result = svi.run(jax.random.PRNGKey(0), 10000, model_args)
params = svi_result.params
samples = guide.sample_posterior(jax.random.PRNGKey(1), params, sample_shape=(500,))
```

:::{note}
`AutoMultivariateNormal` fits a full-rank Gaussian. For simpler problems,
`AutoDiagonalNormal` is faster. For posteriors with strong non-Gaussianity, SVI results
should be treated as approximate and validated against NUTS on a representative object.
:::

---

### Using SVI to Initialize NUTS

SVI's variational posterior is an excellent starting point for NUTS, especially for
multi-component fits where NUTS warmup can be slow:

```python
from numpyro.infer import init_to_value

# 1. Run SVI to convergence (see above)
params = svi_result.params

# 2. Pass to NUTS via init_to_value
kernel = infer.NUTS(model_fn, init_strategy=init_to_value(values=init_values))
mcmc = infer.MCMC(kernel, num_warmup=50, num_samples=1000)
mcmc.run(jax.random.PRNGKey(1), model_args)
samples = mcmc.get_samples()
```

This can cut warmup steps.

---

### Nested Sampling with JAXNS

[JAXNS](https://github.com/Joshuaalbert/jaxns) is a JAX-native nested sampler. NumPyro
ships a thin wrapper for it in
[`numpyro.contrib.nested_sampling`](https://num.pyro.ai/en/stable/contrib.html#nested-sampling).
Nested sampling returns both posterior samples **and** the log Bayesian evidence (log Z),
which is useful for model comparison (e.g., is a broad component required?).

```bash
pip install jaxns
```

```python
from numpyro.contrib.nested_sampling import NestedSampler

ns = NestedSampler(
    model_fn,
    constructor_kwargs={'num_live_points': 500},
)
ns.run(jax.random.PRNGKey(0), model_args)

samples   = ns.get_samples(jax.random.PRNGKey(1), num_samples=1000)
log_Z     = ns.log_evidence_mean
log_Z_err = ns.log_evidence_uncert
print(f'log Z = {log_Z:.2f} ± {log_Z_err:.2f}')
```

Increase `num_live_points` for more accurate evidence estimates at the cost of runtime.
For model comparison, comparing log Z between two models is one of the most robust methods.
