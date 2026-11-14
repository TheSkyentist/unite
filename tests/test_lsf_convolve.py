"""Unit tests for the banded LSF convolution kernel in unite._lsf."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from unite._lsf import (
    _FWHM_TO_SIGMA,
    _apply_pixel_kernel,
    _convolve_stacks_to_pixels,
    _convolve_to_pixels,
    _lsf_convolve,
    _lsf_convolve_components,
    _pixel_kernel,
)


def _uniform_grid(n: int, lo: float = 0.0, hi: float = 1.0) -> jnp.ndarray:
    """Return n uniformly-spaced points in [lo, hi] (midpoints of sub-bins)."""
    dx = (hi - lo) / n
    return jnp.linspace(lo + dx / 2, hi - dx / 2, n)


def _gaussian(x: jnp.ndarray, center: float, sigma: float) -> jnp.ndarray:
    return jnp.exp(-0.5 * ((x - center) / sigma) ** 2) / (
        sigma * math.sqrt(2 * math.pi)
    )


class TestLsfConvolveBasic:
    """Basic correctness tests for _lsf_convolve."""

    def test_flat_conserved(self):
        """Convolving a flat signal with a normalized kernel returns the same flat signal."""
        n = 200
        x = _uniform_grid(n, 0.0, 10.0)
        flat = jnp.ones(n)
        sigma = jnp.full(n, 0.3)
        half_width = 30

        result = _lsf_convolve(x, flat, sigma, half_width)

        # Interior points (away from boundaries) should be 1.0.
        assert jnp.allclose(
            result[half_width:-half_width], jnp.ones(n - 2 * half_width), atol=1e-5
        )

    def test_zero_signal(self):
        """Convolving a zero signal returns zero everywhere."""
        n = 100
        x = _uniform_grid(n, 0.0, 5.0)
        zeros = jnp.zeros(n)
        sigma = jnp.full(n, 0.2)
        result = _lsf_convolve(x, zeros, sigma, half_width=20)
        assert jnp.allclose(result, jnp.zeros(n), atol=1e-10)

    def test_flux_conservation_impulse(self):
        """An impulse-like signal conserved after convolution (interior region)."""
        n = 300
        x = _uniform_grid(n, 0.0, 15.0)
        dx = float(x[1] - x[0])
        sigma_val = 0.5  # several pixels wide
        half_width = math.ceil(4 * sigma_val / dx) + 5

        # Delta-like impulse in the middle.
        impulse = jnp.zeros(n).at[n // 2].set(1.0 / dx)

        sigma = jnp.full(n, sigma_val)
        result = _lsf_convolve(x, impulse, sigma, half_width)

        # Total flux (integral ≈ sum * dx) should be conserved.
        assert abs(float(jnp.sum(result)) * dx - 1.0) < 0.02

    def test_output_shape(self):
        """Output shape matches input shape."""
        n = 80
        x = _uniform_grid(n)
        f = jnp.sin(x)
        sigma = jnp.full(n, 0.05)
        result = _lsf_convolve(x, f, sigma, half_width=10)
        assert result.shape == (n,)

    def test_broadening_fwhm(self):
        """Convolving a narrow Gaussian with a Gaussian LSF broadens FWHM correctly.

        A Gaussian with sigma_in convolved with Gaussian LSF of sigma_lsf gives
        a Gaussian with sigma_out = sqrt(sigma_in^2 + sigma_lsf^2).
        """
        n = 500
        x = _uniform_grid(n, -5.0, 5.0)
        dx = float(x[1] - x[0])

        sigma_in = 0.3
        sigma_lsf = 0.4
        sigma_out_expected = math.sqrt(sigma_in**2 + sigma_lsf**2)

        # Input: narrow Gaussian at center of grid.
        f_in = _gaussian(x, center=0.0, sigma=sigma_in)

        sigma_arr = jnp.full(n, sigma_lsf)
        half_width = math.ceil(4 * sigma_lsf / dx) + 10
        result = _lsf_convolve(x, f_in, sigma_arr, half_width)

        # Estimate output sigma from the second moment of the result.
        # Use only the central portion to avoid boundary effects.
        crop = slice(n // 4, 3 * n // 4)
        x_crop = np.asarray(x[crop])
        r_crop = np.asarray(result[crop])
        r_crop = np.clip(r_crop, 0, None)  # guard against tiny negatives from numerics

        norm = np.sum(r_crop) * dx
        if norm > 0:
            mean = np.sum(x_crop * r_crop) * dx / norm
            var = np.sum((x_crop - mean) ** 2 * r_crop) * dx / norm
            sigma_out_measured = math.sqrt(var)
            assert (
                abs(sigma_out_measured - sigma_out_expected) / sigma_out_expected < 0.05
            )

    def test_varying_sigma_runs(self):
        """Spatially varying sigma does not raise errors."""
        n = 100
        x = _uniform_grid(n, 1.0, 2.0)
        # LSF width that increases linearly (like lambda/R for constant R).
        sigma = x * 0.02
        half_width = math.ceil(4 * float(jnp.max(sigma)) / float(x[1] - x[0])) + 5
        result = _lsf_convolve(x, jnp.ones(n), sigma, half_width)
        assert result.shape == (n,)
        assert jnp.all(jnp.isfinite(result))


class TestLsfConvolveJAX:
    """JAX tracing and compilation tests."""

    def test_jit_compiles(self):
        """_lsf_convolve compiles under jax.jit without error."""
        n = 50
        x = _uniform_grid(n, 0.0, 5.0)
        f = jnp.ones(n)
        sigma = jnp.full(n, 0.2)
        jitted = jax.jit(_lsf_convolve, static_argnums=(3,))
        result = jitted(x, f, sigma, 10)
        assert result.shape == (n,)

    def test_grad_runs(self):
        """Gradient w.r.t. model_fine flows through _lsf_convolve."""
        n = 40
        x = _uniform_grid(n, 0.0, 4.0)
        sigma = jnp.full(n, 0.2)
        half_width = 8

        def loss(f):
            return jnp.sum(_lsf_convolve(x, f, sigma, half_width) ** 2)

        grad = jax.grad(loss)(jnp.ones(n))
        assert grad.shape == (n,)
        assert jnp.all(jnp.isfinite(grad))

    def test_vmap_over_batch(self):
        """_lsf_convolve can be vmapped over a batch of model arrays."""
        n = 60
        batch = 4
        x = _uniform_grid(n, 0.0, 6.0)
        sigma = jnp.full(n, 0.25)
        models = jnp.ones((batch, n)) * jnp.arange(1, batch + 1)[:, None]

        def _apply(f):
            return _lsf_convolve(x, f, sigma, half_width=12)

        results = jax.vmap(_apply)(models)
        assert results.shape == (batch, n)
        assert jnp.all(jnp.isfinite(results))


class TestFwhmToSigma:
    """Sanity check for the _FWHM_TO_SIGMA constant."""

    def test_value(self):
        expected = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
        assert abs(_FWHM_TO_SIGMA - expected) < 1e-12

    def test_round_trip(self):
        """FWHM * _FWHM_TO_SIGMA * 2.355 ≈ FWHM."""
        fwhm = 2.5
        sigma = fwhm * _FWHM_TO_SIGMA
        fwhm_back = sigma * 2.0 * math.sqrt(2.0 * math.log(2.0))
        assert abs(fwhm_back - fwhm) < 1e-12


def _pixel_grid(n_pixels: int, n_super: int):
    """Non-uniform pixels with a wavelength-dependent LSF."""
    edges = np.cumsum(np.linspace(0.8, 1.6, n_pixels + 1))
    low, high = edges[:-1], edges[1:]
    offsets = (np.arange(n_super) + 0.5) / n_super
    x = (low[:, None] + offsets[None, :] * (high - low)[:, None]).ravel()
    sigma = 0.9 + 0.004 * x
    return jnp.asarray(x), jnp.asarray(sigma)


def _reference(x, model, sigma, half_width, n_super):
    return _lsf_convolve(x, model, sigma, half_width).reshape(-1, n_super).mean(axis=1)


class TestPixelKernel:
    """The pixel-averaged kernel reproduces convolve-then-average."""

    @pytest.mark.parametrize(
        ('n_pixels', 'n_super', 'half_width'),
        [(60, 10, 37), (60, 4, 9), (5, 10, 80), (40, 1, 6), (1, 6, 4)],
    )
    def test_matches_convolve_then_average(self, n_pixels, n_super, half_width):
        x, sigma = _pixel_grid(n_pixels, n_super)
        model = jnp.sin(x / 3.0) + 0.01 * x + 2.0
        weights = _pixel_kernel(x, sigma, half_width, n_super)
        got = _apply_pixel_kernel(weights, model, half_width)
        want = _reference(x, model, sigma, half_width, n_super)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_component_axes(self):
        n_super, half_width = 10, 25
        x, sigma = _pixel_grid(30, n_super)
        models = jnp.stack([jnp.sin(x / (k + 2.0)) for k in range(6)]).reshape(2, 3, -1)
        weights = _pixel_kernel(x, sigma, half_width, n_super)
        got = _apply_pixel_kernel(weights, models, half_width)
        assert got.shape == (2, 3, 30)
        for a in range(2):
            for b in range(3):
                want = _reference(x, models[a, b], sigma, half_width, n_super)
                np.testing.assert_allclose(got[a, b], want, rtol=1e-12)

    def test_components_loop_matches(self):
        half_width = 25
        x, sigma = _pixel_grid(30, 10)
        models = jnp.stack([jnp.sin(x / (k + 2.0)) + k for k in range(4)])
        got = _lsf_convolve_components(x, models, sigma, half_width)
        for k in range(4):
            want = _lsf_convolve(x, models[k], sigma, half_width)
            np.testing.assert_allclose(got[k], want, rtol=1e-12)


class TestConvolveToPixels:
    """Every route of _convolve_to_pixels gives the same pixel averages."""

    n_super = 10
    half_width = 25

    def _setup(self):
        x, sigma = _pixel_grid(30, self.n_super)
        models = jnp.stack([jnp.sin(x / (k + 2.0)) + k for k in range(3)])
        want = jnp.stack(
            [_reference(x, m, sigma, self.half_width, self.n_super) for m in models]
        )
        return x, sigma, models, want

    def test_constant_kernel_is_cached(self):
        x, sigma, models, want = self._setup()
        cache: dict = {}
        fn = jax.jit(
            lambda m: _convolve_to_pixels(
                np.asarray(x),
                m,
                np.asarray(sigma),
                self.half_width,
                self.n_super,
                cache=cache,
                key='k',
            )
        )
        np.testing.assert_allclose(fn(models[0]), want[0], rtol=1e-12)
        np.testing.assert_allclose(fn(models), want, rtol=1e-12)
        assert list(cache) == ['k']
        weights = cache['k'][1]
        assert isinstance(weights, np.ndarray)
        # Same grid: the stored kernel is reused, not rebuilt.
        _convolve_to_pixels(
            np.asarray(x),
            models[0],
            np.asarray(sigma),
            self.half_width,
            self.n_super,
            cache=cache,
            key='k',
        )
        assert cache['k'][1] is weights

    def test_cached_kernel_not_reused_for_other_lsf(self):
        x, sigma, models, _ = self._setup()
        cache: dict = {}
        args = (self.half_width, self.n_super)
        _convolve_to_pixels(
            np.asarray(x), models[0], np.asarray(sigma), *args, cache=cache, key='k'
        )
        got = _convolve_to_pixels(
            np.asarray(x), models[0], np.asarray(2 * sigma), *args, cache=cache, key='k'
        )
        want = _reference(x, models[0], 2 * sigma, *args)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    def test_traced_kernel(self):
        x, sigma, models, want = self._setup()
        fn = jax.jit(
            lambda x, m, s: _convolve_to_pixels(x, m, s, self.half_width, self.n_super)
        )
        np.testing.assert_allclose(fn(x, models[0], sigma), want[0], rtol=1e-12)
        np.testing.assert_allclose(fn(x, models, sigma), want, rtol=1e-12)

    @pytest.mark.parametrize('traced', [False, True])
    def test_list_of_stacks(self, traced):
        x, sigma, models, want = self._setup()

        def run(x, s, m):
            return _convolve_stacks_to_pixels(
                x, [m[:1], m[1:]], s, self.half_width, self.n_super
            )

        if traced:
            first, rest = jax.jit(run)(x, sigma, models)
        else:
            first, rest = run(np.asarray(x), np.asarray(sigma), models)
        np.testing.assert_allclose(first, want[:1], rtol=1e-12)
        np.testing.assert_allclose(rest, want[1:], rtol=1e-12)

    def test_gradient_matches(self):
        x, sigma, models, _ = self._setup()
        weights = jnp.cos(jnp.arange(30) * 0.4)

        def constant(m):
            out = _convolve_to_pixels(
                np.asarray(x), m, np.asarray(sigma), self.half_width, self.n_super
            )
            return jnp.sum(out * weights)

        def reference(m):
            out = _reference(x, m, sigma, self.half_width, self.n_super)
            return jnp.sum(out * weights)

        got = jax.grad(constant)(models[1])
        want = jax.grad(reference)(models[1])
        np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-14)
