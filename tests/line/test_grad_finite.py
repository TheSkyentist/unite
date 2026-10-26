"""Gradients of the Gaussian-Laplace kernels stay finite where overflow guards fire.

``jnp.where(dead, 0, exp(x) * erfc(y))`` discards the forward value of the dead
branch but not its gradient: ``inf * 0`` there becomes NaN under ``jax.grad``.
These tests drive every guarded kernel into that regime (far from the line,
Gaussian limit, wide exponential) in both float32 and float64.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from unite.line import functions

LSF = 111.0
FWHM_G = 480.0

# (span, fwhm_exp): pixels within +-span of the line.  fwhm_exp=5 is the Gaussian
# limit in both dtypes (a > 26); fwhm_exp=1083 / 300 only overflow far from the line.
CASES = {
    'near': (5e3, 1083.0),
    'far': (7e4, 1083.0),
    'far_narrow_exp': (7e4, 300.0),
    'gaussian_limit_near': (5e3, 5.0),
    'gaussian_limit_far': (7e4, 5.0),
    'wide_exp': (7e4, 1e4),
}

# name, number of exponential FWHMs
KERNELS = [
    ('integrate_gaussianLaplace', 1),
    ('evaluate_gaussianLaplace', 1),
    ('integrate_gaussianSplitLaplace', 2),
    ('evaluate_gaussianSplitLaplace', 2),
]


@pytest.fixture(params=[True, False], ids=['float64', 'float32'])
def x64(request):
    with jax.enable_x64(request.param):
        yield request.param


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize(('name', 'n_exp'), KERNELS)
def test_grad_is_finite(x64, name, n_exp, case):
    span, fwhm_exp = CASES[case]
    kernel = getattr(functions, name)
    x = jnp.linspace(-span, span, 400)
    if name.startswith('integrate'):
        lsf = jnp.full_like(x, LSF)

        def loss(center, fwhm_g, *fwhm_exp):
            return kernel(x, lsf, center, fwhm_g, *fwhm_exp).sum()
    else:

        def loss(center, fwhm_g, *fwhm_exp):
            return kernel(x, center, LSF, fwhm_g, *fwhm_exp).sum()

    # Asymmetric split-Laplace widths exercise both sides independently.
    exp_widths = [fwhm_exp * (1 + i) for i in range(n_exp)]
    args = (0.0, FWHM_G, *exp_widths)
    argnums = tuple(range(len(args)))

    assert np.isfinite(float(loss(*args)))
    grads = jax.grad(loss, argnums=argnums)(*args)
    assert all(np.isfinite(float(g)) for g in grads), grads


def test_overflow_threshold_follows_dtype():
    with jax.enable_x64(True):
        assert functions._overflow_threshold(np.zeros(1, np.float64)) == 26
        assert functions._overflow_threshold(np.zeros(1, np.float32)) == 9
