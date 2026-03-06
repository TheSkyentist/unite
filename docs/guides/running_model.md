# Running the Model

`unite` builds a NumPyro model function for you — it does not run inference itself.
{meth}`~unite.model.ModelBuilder.build` returns `(model_fn, model_args)`, and you
pass these to any NumPyro inference kernel you choose.

---

## JAX 64-bit Mode

JAX defaults to 32-bit. 64-bit is strongly recommended — especially for posteriors with
long tails or tight dependent-prior constraints — and must be enabled before any JAX
computation:

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

See {doc}`../getting_started` for full details.

---

## Building the Model

```python
from unite import model

builder = model.ModelBuilder(filtered_lines, filtered_cont, spectra)
model_fn, model_args = builder.build()
```

`model_fn` is a standard NumPyro model function. `model_args` is a dataclass of
pre-computed arrays (pixel edges, LSF widths, scales, etc.) that `model_fn` accepts as
its sole argument.

---

## NUTS (No-U-Turn Sampler)

NUTS is the recommended sampler for most problems. It uses gradient information to
efficiently explore the posterior.

### Basic usage

```python
import jax
from numpyro import infer

kernel = infer.NUTS(model_fn)
mcmc = infer.MCMC(
    kernel,
    num_warmup=500,
    num_samples=1000,
    progress_bar=True,
)
mcmc.run(jax.random.PRNGKey(0), model_args)
samples = mcmc.get_samples()
```

### Multiple chains

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
run sequentially by default (set `chain_method='parallel'` to use multiprocessing).

### NUTS tuning

For complex posteriors (many components, tight constraints) you may need more warmup or
a larger target acceptance probability:

```python
kernel = infer.NUTS(
    model_fn,
    target_accept_prob=0.9,   # default 0.8; increase for difficult posteriors
    max_tree_depth=12,        # default 10; increase if chains get stuck
)
```

---

## GPU Acceleration

JAX runs on GPU transparently — no code changes are needed. If a GPU is available and
JAX is installed with CUDA support, it will be used automatically.

### Verify the device

```python
import jax
print(jax.devices())           # e.g. [CudaDevice(id=0)]
print(jax.default_backend())   # 'gpu', 'cpu', or 'tpu'
```

### Select a specific device

```python
with jax.default_device(jax.devices('gpu')[0]):
    mcmc.run(jax.random.PRNGKey(0), model_args)
```

### GPU installation

Install JAX with CUDA support following the
[official JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html).
The CPU-only JAX included by default in `unite`'s Pixi environment will not use a GPU.

### Performance tips

- **JIT compilation** happens on the first `mcmc.run` call. Subsequent calls with the
  same model are fast. The first call may take 30–120 seconds to compile.
- **Warm-up dominates** on first run. If you are iterating on priors, keep `num_warmup`
  low while testing, then increase for production.
- **Large `model_args` arrays** are transferred to the GPU once and reused. For very
  large spectra, check that your GPU has enough memory with `nvidia-smi`.

---

## SVI — Stochastic Variational Inference

SVI approximates the posterior with a simpler family of distributions (e.g.,
multivariate normal). It is much faster than NUTS and useful for:
- rapid iteration on priors and configuration
- large surveys where full MCMC is too slow
- initializing NUTS with a good starting point

```python
from numpyro import infer
from numpyro.infer import autoguide
import optax

guide = autoguide.AutoMultivariateNormal(model_fn)

optimizer = infer.SVI(
    model_fn,
    guide,
    optax.adam(1e-3),
    loss=infer.Trace_ELBO(),
)

rng_key = jax.random.PRNGKey(0)
svi_state = optimizer.init(rng_key, model_args)

# Run for N steps
for i in range(5000):
    svi_state, loss = optimizer.update(svi_state, model_args)
    if i % 500 == 0:
        print(f'step {i}: ELBO = {-loss:.2f}')

# Draw samples from the variational posterior
params = optimizer.get_params(svi_state)
predictive = infer.Predictive(guide, params=params, num_samples=1000)
samples = predictive(rng_key, model_args)
```

:::{note}
`AutoMultivariateNormal` fits a full-rank Gaussian. For simpler problems,
`AutoDiagonalNormal` is faster. For posteriors with strong non-Gaussianity, SVI results
should be treated as approximate and validated against NUTS on a representative object.
:::

---

## Nested Sampling with JAXNS

[JAXNS](https://github.com/Joshuaalbert/jaxns) is a JAX-native nested sampler. NumPyro
ships a thin wrapper for it in `numpyro.contrib.nested_sampling`. Nested sampling
returns both posterior samples **and** the log Bayesian evidence (log Z), which is
useful for model comparison (e.g., is a broad component required?).

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
For broad-component detection, compare log Z between a model with and without the
component; a difference of $\Delta \log Z \gtrsim 5$ is typically considered strong
evidence in favour of the more complex model.
