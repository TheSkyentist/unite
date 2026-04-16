"""Tests for disperser base classes and calibration tokens."""

import pytest
from astropy import units as u

from unite.instrument.base import Disperser, FluxScale, PixOffset, RScale
from unite.prior import Fixed, TruncatedNormal, Uniform

# ---------------------------------------------------------------------------
# Calibration token construction
# ---------------------------------------------------------------------------


class TestRScale:
    """Tests for the RScale calibration token."""

    def test_default_prior_is_fixed_one(self):
        r = RScale()
        assert isinstance(r.prior, Fixed)
        assert r.prior.value == 1.0

    def test_custom_prior(self):
        r = RScale(prior=Uniform(0.8, 1.2))
        assert isinstance(r.prior, Uniform)

    def test_auto_name(self):
        # Before attachment to a Disperser, name is None (set at attachment time).
        r = RScale()
        assert r._name is None
        assert r.label is None

    def test_custom_name(self):
        # A custom name becomes the label; site name is set at attachment.
        r = RScale(name='my_r')
        assert r.label == 'my_r'
        assert r._name is None  # not yet attached to a disperser


class TestFluxScale:
    """Tests for the FluxScale calibration token."""

    def test_default_prior_is_fixed_one(self):
        f = FluxScale()
        assert isinstance(f.prior, Fixed)
        assert f.prior.value == 1.0

    def test_custom_prior(self):
        f = FluxScale(prior=TruncatedNormal(1.0, 0.1, 0.5, 2.0))
        assert isinstance(f.prior, TruncatedNormal)

    def test_auto_name(self):
        # Before attachment to a Disperser, name is None (set at attachment time).
        f = FluxScale()
        assert f._name is None
        assert f.label is None

    def test_custom_name(self):
        # A custom name becomes the label; site name is set at attachment.
        f = FluxScale(name='my_flux')
        assert f.label == 'my_flux'
        assert f._name is None  # not yet attached to a disperser


class TestPixOffset:
    """Tests for the PixOffset calibration token."""

    def test_default_prior_is_fixed_zero(self):
        p = PixOffset()
        assert isinstance(p.prior, Fixed)
        assert p.prior.value == 0.0

    def test_custom_prior(self):
        p = PixOffset(prior=Uniform(-2.0, 2.0))
        assert isinstance(p.prior, Uniform)

    def test_auto_name(self):
        # Before attachment to a Disperser, name is None (set at attachment time).
        p = PixOffset()
        assert p._name is None
        assert p.label is None

    def test_custom_name(self):
        # A custom name becomes the label; site name is set at attachment.
        p = PixOffset(name='my_pix')
        assert p.label == 'my_pix'
        assert p._name is None  # not yet attached to a disperser


# ---------------------------------------------------------------------------
# Disperser ABC — type validation
# ---------------------------------------------------------------------------


class TestDisperserTypeValidation:
    """Tests for Disperser base class type checking."""

    def test_wrong_type_r_scale(self):
        from unite.instrument.generic import GenericDisperser

        with pytest.raises(TypeError, match='r_scale must be an RScale'):
            GenericDisperser(
                R_func=lambda w: w * 0 + 1000,
                dlam_dpix_func=lambda w: w / 1000,
                unit=u.AA,
                r_scale='not_a_token',
            )

    def test_wrong_type_flux_scale(self):
        from unite.instrument.generic import GenericDisperser

        with pytest.raises(TypeError, match='flux_scale must be a FluxScale'):
            GenericDisperser(
                R_func=lambda w: w * 0 + 1000,
                dlam_dpix_func=lambda w: w / 1000,
                unit=u.AA,
                flux_scale=RScale(),
            )

    def test_wrong_type_pix_offset(self):
        from unite.instrument.generic import GenericDisperser

        with pytest.raises(TypeError, match='pix_offset must be a PixOffset'):
            GenericDisperser(
                R_func=lambda w: w * 0 + 1000,
                dlam_dpix_func=lambda w: w / 1000,
                unit=u.AA,
                pix_offset=FluxScale(),
            )

    def test_has_calibration_params_false(self):
        from unite.instrument.generic import GenericDisperser

        d = GenericDisperser(
            R_func=lambda w: w * 0 + 1000, dlam_dpix_func=lambda w: w / 1000, unit=u.AA
        )
        assert not d.has_calibration_params

    def test_has_calibration_params_true(self):
        from unite.instrument.generic import GenericDisperser

        d = GenericDisperser(
            R_func=lambda w: w * 0 + 1000,
            dlam_dpix_func=lambda w: w / 1000,
            unit=u.AA,
            r_scale=RScale(),
        )
        assert d.has_calibration_params

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            Disperser(u.AA)
