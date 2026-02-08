import jax.numpy as jnp
import pytest
from unite import optimized

# Test data for emission line integration
low_edge = jnp.array([0.0, 1.0])
high_edge = jnp.array([1.0, 2.0])
centers = jnp.array([0.5, 1.5])
fwhms = jnp.array([0.2, 0.3])
fwhm_g = jnp.array([0.2, 0.3])
fwhm_γ = jnp.array([0.1, 0.2])
lsf = jnp.array([0.1, 0.2])
is_voigt = jnp.array([True, False])

# Test data for continua
λ = jnp.linspace(0, 2, 5)
cont_centers = jnp.array([1.0, 1.5])
angles = jnp.array([0.1, 0.2])
offsets = jnp.array([0.0, 0.5])
continuum_regions = jnp.array([[0.0, 2.0], [1.0, 2.0]])


def test_integrateGaussian_shape():
    result = optimized.integrateGaussian(low_edge, high_edge, centers, fwhms)
    assert result.shape == (2, 2)


def test_integrateCauchy_shape():
    result = optimized.integrateCauchy(low_edge, high_edge, centers, fwhms)
    assert result.shape == (2, 2)


def test_integrateVoigt_shape():
    result = optimized.integrateVoigt(low_edge, high_edge, centers, fwhm_g, fwhm_γ)
    assert result.shape == (2, 2)


def test_integrateCond_gaussian():
    # Should use Gaussian branch
    result = optimized.integrateCond(
        low_edge, high_edge, centers, lsf, fwhms, jnp.array([False, False])
    )
    assert result.shape == (2,)


def test_integrateCond_voigt():
    # Should use Voigt branch
    result = optimized.integrateCond(
        low_edge, high_edge, centers, lsf, fwhms, jnp.array([True, True])
    )
    assert result.shape == (2,)


def test_integrate_shape():
    result = optimized.integrate(low_edge, high_edge, centers, lsf, fwhms, is_voigt)
    assert result.shape == (2, 2)


def test_linearContinua_shape():
    result = optimized.linearContinua(
        λ, cont_centers, angles, offsets, continuum_regions
    )
    assert result.shape == (5, 2)


def test_powerLawContinuum_shape():
    result = optimized.powerLawContinuum(λ, 1.0, 2.0, -1.0)
    assert result.shape == (5,)


def test_powerLawContinuum_values():
    result = optimized.powerLawContinuum(jnp.array([1.0]), 1.0, 2.0, -1.0)
    assert jnp.allclose(result, 2.0)
