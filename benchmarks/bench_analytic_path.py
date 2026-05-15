"""Breakdown benchmarks for the analytic integration path.

The goal is to measure what fraction of a single log-density evaluation is
spent in each sub-step of the analytic path so we can decide whether
optimizations like shared-edge CDF evaluation are worth the refactor cost.

The analytic forward pass decomposes into three phases:

1. **integrate_lines** — CDF diffs per pixel per line (``n_lines * 2N`` erfc
   calls for N contiguous pixels).
2. **integrate_continuum** — continuum form evaluation + pixel integration.
3. **compose_from_profiles** — linear combination of profiles, optional tau
   absorption (matrix-vector products + exp).

Each is benchmarked in isolation on representative arrays drawn from the
single-grating config.  Compare against ``test_logp_single_grating`` (the
full forward pass, ~50 µs on the same machine) to read off percentages.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from benchmarks._helpers import Bench, block
from unite._compose import compose_from_profiles
from unite._utils import C_KMS
from unite.continuum.compute import integrate_continuum
from unite.prior import Fixed

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _full_context(b: Bench) -> dict:
    """Build a complete context dict (sampled + Fixed params) from a Bench.

    ``b.sample`` only contains sampled latent parameters; Fixed params are
    absent.  ``integrate_continuum`` (and other downstream functions) need
    the full context, so we replicate the same logic as ``unite_model``.
    """
    args = b.args
    ctx: dict = {}
    for pname in args.dependency_order:
        prior = args.all_priors[pname]
        if isinstance(prior, Fixed):
            ctx[pname] = jnp.asarray(prior.value)
        else:
            ctx[pname] = jnp.asarray(b.sample[pname])
    return ctx


# ----------------------------------------------------------------------
# Helper: extract static per-line arrays from a Bench
# ----------------------------------------------------------------------


def _line_arrays(b: Bench):
    """Return (low, high, centers, lsf_fwhm, p0, p1, p2, pcodes) for spectrum 0.

    Uses the prior draw stored on ``b`` to compute representative per-line
    values; shapes match what ``integrate_lines`` receives at every MCMC step.
    """
    args = b.args
    cm = args.matrices
    sample = b.sample
    z_sys = args.redshift
    n_lines = int(cm.wavelengths.shape[0])

    ctx = {k: jnp.asarray(v) for k, v in sample.items()}

    z_per_line = (
        jnp.stack([ctx[n] for n in cm.z_names]) @ cm.z_matrix
        if cm.z_names
        else jnp.zeros(n_lines)
    )
    centers = cm.wavelengths * (1.0 + z_sys + z_per_line)

    p0_kms = (
        jnp.stack([ctx[n] for n in cm.p0_names]) @ cm.p0_matrix
        if cm.p0_names
        else jnp.zeros(n_lines)
    )
    p0 = centers * p0_kms / C_KMS

    p1v = (
        centers * (jnp.stack([ctx[n] for n in cm.p1v_names]) @ cm.p1v_matrix) / C_KMS
        if cm.p1v_names
        else jnp.zeros(n_lines)
    )
    p1d = (
        jnp.stack([ctx[n] for n in cm.p1d_names]) @ cm.p1d_matrix
        if cm.p1d_names
        else jnp.zeros(n_lines)
    )
    p1 = p1v + p1d

    p2v = (
        centers * (jnp.stack([ctx[n] for n in cm.p2v_names]) @ cm.p2v_matrix) / C_KMS
        if cm.p2v_names
        else jnp.zeros(n_lines)
    )
    p2d = (
        jnp.stack([ctx[n] for n in cm.p2d_names]) @ cm.p2d_matrix
        if cm.p2d_names
        else jnp.zeros(n_lines)
    )
    p2 = p2v + p2d

    spec = args.spectra[0]
    wl_scale = args.spec_to_canonical[0]
    low = spec.low * wl_scale
    high = spec.high * wl_scale
    disp = spec.disperser
    inv_wl = 1.0 / wl_scale
    lsf_fwhm = centers / disp.R(centers * inv_wl)

    pcodes = args._profile_codes_local
    if pcodes is None:
        pcodes = cm.profile_codes

    return low, high, centers, lsf_fwhm, p0, p1, p2, pcodes


# ----------------------------------------------------------------------
# Phase 1 — integrate_lines (CDF diffs)
# ----------------------------------------------------------------------


def test_analytic_integrate_lines(benchmark, single_grating_bench):
    """Time only the CDF-diff step for the single-grating config.

    This is the candidate for shared-edge optimization: N+1 erfc evals
    instead of 2N.  Compare the result here to ``test_logp_single_grating``
    to get the CDF fraction of total forward-pass time.
    """
    b = single_grating_bench
    low, high, centers, lsf_fwhm, p0, p1, p2, pcodes = _line_arrays(b)
    int_fn = b.args._integrate_fn

    jit_fn = jax.jit(lambda: int_fn(low, high, centers, lsf_fwhm, p0, p1, p2, pcodes))
    jit_fn()  # warm
    benchmark(lambda: block(jit_fn()))


def test_analytic_integrate_lines_multi(benchmark, multi_grating_bench):
    """Same as above for the first spectrum of the multi-grating config."""
    b = multi_grating_bench
    low, high, centers, lsf_fwhm, p0, p1, p2, pcodes = _line_arrays(b)
    int_fn = b.args._integrate_fn

    jit_fn = jax.jit(lambda: int_fn(low, high, centers, lsf_fwhm, p0, p1, p2, pcodes))
    jit_fn()
    benchmark(lambda: block(jit_fn()))


# ----------------------------------------------------------------------
# Phase 2 — integrate_continuum
# ----------------------------------------------------------------------


def test_analytic_integrate_continuum(benchmark, single_grating_bench):
    """Time only the continuum integration step.

    Uses a fixed context drawn from the prior so continuum parameters are
    valid but static across benchmark iterations.
    """
    b = single_grating_bench
    args = b.args
    ctx = _full_context(b)
    z_sys = args.redshift

    spec = args.spectra[0]
    wl_scale = args.spec_to_canonical[0]
    low = spec.low * wl_scale
    high = spec.high * wl_scale
    pix_mid = (low + high) / 2.0
    disp = spec.disperser
    cont_lsf = pix_mid / disp.R(pix_mid / wl_scale)

    jit_fn = jax.jit(lambda: integrate_continuum(low, high, args, ctx, z_sys, cont_lsf))
    jit_fn()
    benchmark(lambda: block(jit_fn()))


# ----------------------------------------------------------------------
# Phase 3 — compose_from_profiles (emission-only)
# ----------------------------------------------------------------------


def test_analytic_compose(benchmark, single_grating_bench):
    """Time only the composition step with no tau absorbers.

    Uses pre-computed static pixints so only the linear combination logic
    (matrix-vector products, no exp) is exercised.
    """
    b = single_grating_bench
    args = b.args
    cm = args.matrices
    low, high, centers, lsf_fwhm, p0, p1, p2, pcodes = _line_arrays(b)
    int_fn = b.args._integrate_fn

    # Pre-compute pixints once; treat as static input.
    pixints = jax.jit(
        lambda: int_fn(low, high, centers, lsf_fwhm, p0, p1, p2, pcodes) / (high - low)
    )()
    pixints.block_until_ready()

    n_lines = int(cm.wavelengths.shape[0])
    n_pixels = int(low.shape[0])
    ctx = {k: jnp.asarray(v) for k, v in b.sample.items()}
    flux_per_line = (
        jnp.stack([ctx[n] for n in cm.flux_names]) @ cm.flux_matrix * cm.strengths
    )
    scaled_flux = flux_per_line * float(args.line_flux_scales[0])
    tau_per_line = jnp.zeros(n_lines)
    continuum = jnp.ones(n_pixels) * 0.5

    jit_fn = jax.jit(
        lambda: compose_from_profiles(
            pixints,
            scaled_flux,
            tau_per_line,
            cm.is_tau,
            cm.applies_matrix,
            args.cont_applies,
            continuum,
            has_tau=False,
        )
    )
    jit_fn()
    benchmark(lambda: block(jit_fn()))


# ----------------------------------------------------------------------
# Phase 3b — compose_from_profiles with tau
# ----------------------------------------------------------------------


def test_analytic_compose_with_tau(benchmark, single_grating_absorber_bench):
    """Time the composition step with one tau absorber active.

    The exp() call for absorption makes this meaningfully more expensive
    than the emission-only case.
    """
    b = single_grating_absorber_bench
    args = b.args
    cm = args.matrices
    low, high, centers, lsf_fwhm, p0, p1, p2, pcodes = _line_arrays(b)
    int_fn = b.args._integrate_fn

    pixints = jax.jit(
        lambda: int_fn(low, high, centers, lsf_fwhm, p0, p1, p2, pcodes) / (high - low)
    )()
    pixints.block_until_ready()

    n_lines = int(cm.wavelengths.shape[0])
    n_pixels = int(low.shape[0])
    ctx = {k: jnp.asarray(v) for k, v in b.sample.items()}
    flux_per_line = (
        jnp.stack([ctx[n] for n in cm.flux_names]) @ cm.flux_matrix * cm.strengths
        if cm.flux_names
        else jnp.zeros(n_lines)
    )
    scaled_flux = flux_per_line * float(args.line_flux_scales[0])
    tau_per_line = jnp.stack([ctx[n] for n in cm.tau_names]) @ cm.tau_matrix
    continuum = jnp.ones(n_pixels) * 0.5

    jit_fn = jax.jit(
        lambda: compose_from_profiles(
            pixints,
            scaled_flux,
            tau_per_line,
            cm.is_tau,
            cm.applies_matrix,
            args.cont_applies,
            continuum,
            has_tau=True,
        )
    )
    jit_fn()
    benchmark(lambda: block(jit_fn()))
