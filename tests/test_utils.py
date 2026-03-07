"""Tests for internal utility functions."""

import pytest
from astropy import units as u

from unite._utils import (
    C_KMS,
    _alpha_name,
    _broadcast,
    _ensure_flux_density,
    _ensure_flux_density_quantity,
    _ensure_velocity,
    _ensure_wavelength,
    _flux_density_conversion_factor,
    _make_register,
    _wavelength_conversion_factor,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    """Tests for physical constants."""

    def test_c_kms(self):
        assert pytest.approx(299792.458, rel=1e-6) == C_KMS


# ---------------------------------------------------------------------------
# Registry decorator
# ---------------------------------------------------------------------------


class TestMakeRegister:
    """Tests for _make_register decorator factory."""

    def test_register_class(self):
        registry = {}
        register = _make_register(registry)

        @register
        class Foo:
            pass

        assert 'Foo' in registry
        assert registry['Foo'] is Foo

    def test_register_multiple(self):
        registry = {}
        register = _make_register(registry)

        @register
        class A:
            pass

        @register
        class B:
            pass

        assert len(registry) == 2


# ---------------------------------------------------------------------------
# Alpha naming
# ---------------------------------------------------------------------------


class TestAlphaName:
    """Tests for Excel-style column naming."""

    def test_first_26(self):
        assert _alpha_name(0) == 'a'
        assert _alpha_name(25) == 'z'

    def test_double_letters(self):
        assert _alpha_name(26) == 'aa'
        assert _alpha_name(27) == 'ab'

    def test_triple_letters(self):
        # 26 + 26*26 = 702
        assert _alpha_name(702) == 'aaa'


# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------


class TestBroadcast:
    """Tests for _broadcast utility."""

    def test_scalar_broadcast(self):
        result = _broadcast(5, 'x', 3)
        assert result == [5, 5, 5]

    def test_list_exact_length(self):
        result = _broadcast([1, 2, 3], 'x', 3)
        assert result == [1, 2, 3]

    def test_list_length_one(self):
        result = _broadcast([5], 'x', 3)
        assert result == [5, 5, 5]

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="'x' has 2 entries"):
            _broadcast([1, 2], 'x', 3)

    def test_tuple_accepted(self):
        result = _broadcast((1, 2, 3), 'x', 3)
        assert result == [1, 2, 3]


# ---------------------------------------------------------------------------
# Wavelength validation
# ---------------------------------------------------------------------------


class TestEnsureWavelength:
    """Tests for _ensure_wavelength."""

    def test_valid_angstrom(self):
        q = 5000.0 * u.AA
        result = _ensure_wavelength(q)
        assert result is q

    def test_valid_micron(self):
        q = 0.5 * u.um
        result = _ensure_wavelength(q)
        assert result is q

    def test_not_quantity_raises(self):
        with pytest.raises(TypeError, match='must be an astropy Quantity'):
            _ensure_wavelength(5000.0)

    def test_wrong_units_raises(self):
        with pytest.raises(ValueError, match=r'wavelength.*length.*units'):
            _ensure_wavelength(100.0 * u.km / u.s)

    def test_ndim_validation(self):
        q = [5000.0, 6000.0] * u.AA
        result = _ensure_wavelength(q, ndim=1)
        assert result is q

    def test_wrong_ndim_raises(self):
        q = 5000.0 * u.AA
        with pytest.raises(ValueError, match='must be 1-D'):
            _ensure_wavelength(q, ndim=1)


# ---------------------------------------------------------------------------
# Wavelength conversion
# ---------------------------------------------------------------------------


class TestWavelengthConversion:
    """Tests for _wavelength_conversion_factor."""

    def test_identity(self):
        f = _wavelength_conversion_factor(u.AA, u.AA)
        assert f == pytest.approx(1.0)

    def test_angstrom_to_micron(self):
        f = _wavelength_conversion_factor(u.AA, u.um)
        assert f == pytest.approx(1e-4)

    def test_micron_to_angstrom(self):
        f = _wavelength_conversion_factor(u.um, u.AA)
        assert f == pytest.approx(1e4)


# ---------------------------------------------------------------------------
# Velocity validation
# ---------------------------------------------------------------------------


class TestEnsureVelocity:
    """Tests for _ensure_velocity."""

    def test_valid(self):
        q = 300 * u.km / u.s
        result = _ensure_velocity(q)
        assert result is q

    def test_not_quantity_raises(self):
        with pytest.raises(TypeError, match='must be an astropy Quantity'):
            _ensure_velocity(300.0)

    def test_wrong_units_raises(self):
        with pytest.raises(ValueError, match='velocity units'):
            _ensure_velocity(300.0 * u.AA)


# ---------------------------------------------------------------------------
# Flux density validation
# ---------------------------------------------------------------------------


class TestEnsureFluxDensity:
    """Tests for flux density validation functions."""

    def test_valid_unit(self):
        _ensure_flux_density(u.erg / u.s / u.cm**2 / u.AA)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match='spectral flux density'):
            _ensure_flux_density(u.Jy)

    def test_quantity_valid(self):
        q = 1.0 * u.erg / u.s / u.cm**2 / u.AA
        result = _ensure_flux_density_quantity(q)
        assert result is q

    def test_quantity_not_quantity_raises(self):
        with pytest.raises(TypeError, match='must be an astropy Quantity'):
            _ensure_flux_density_quantity(1.0)

    def test_quantity_wrong_unit_raises(self):
        with pytest.raises(ValueError, match='spectral flux density'):
            _ensure_flux_density_quantity(1.0 * u.Jy)

    def test_quantity_ndim(self):
        q = [1.0, 2.0] * u.erg / u.s / u.cm**2 / u.AA
        result = _ensure_flux_density_quantity(q, ndim=1)
        assert result is q

    def test_quantity_wrong_ndim_raises(self):
        q = 1.0 * u.erg / u.s / u.cm**2 / u.AA
        with pytest.raises(ValueError, match='must be 1-D'):
            _ensure_flux_density_quantity(q, ndim=1)


# ---------------------------------------------------------------------------
# Flux density conversion
# ---------------------------------------------------------------------------


class TestFluxDensityConversion:
    """Tests for _flux_density_conversion_factor."""

    def test_identity(self):
        unit = u.erg / u.s / u.cm**2 / u.AA
        f = _flux_density_conversion_factor(unit, unit)
        assert f == pytest.approx(1.0)

    def test_cgs_to_scaled(self):
        cgs = u.erg / u.s / u.cm**2 / u.AA
        scaled = 1e-17 * u.erg / u.s / u.cm**2 / u.AA
        f = _flux_density_conversion_factor(cgs, scaled)
        assert f == pytest.approx(1e17, rel=1e-6)
