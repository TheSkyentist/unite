"""Tests for _make_line_labels in unite.model."""

import astropy.units as u

from unite import line, prior
from unite.model import _make_line_labels

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lc():
    return line.LineConfiguration()


def _z(name='z'):
    return line.Redshift(name, prior=prior.Uniform(-0.005, 0.005))


def _fwhm(name):
    return line.FWHM(name, prior=prior.Uniform(100, 3000))


def _flux(name):
    return line.Flux(name, prior=prior.Uniform(0, 5))


# ---------------------------------------------------------------------------
# Unique names — label is just the line name
# ---------------------------------------------------------------------------


class TestUniqueNames:
    def test_single_line(self):
        lc = _lc()
        lc.add_line(
            'Ha',
            6563.0 * u.AA,
            redshift=_z(),
            fwhm_gauss=_fwhm('fwhm'),
            flux=_flux('f'),
        )
        assert _make_line_labels(lc) == ['Ha']

    def test_two_distinct_names(self):
        lc = _lc()
        lc.add_line('Ha', 6563.0 * u.AA)
        lc.add_line('[OIII]', 5007.0 * u.AA)
        assert _make_line_labels(lc) == ['Ha', '[OIII]']

    def test_many_distinct_names_preserve_insertion_order(self):
        lc = _lc()
        for name, wl in [('Ha', 6563.0), ('[NII]a', 6585.0), ('[SII]', 6717.0)]:
            lc.add_line(name, wl * u.AA)
        assert _make_line_labels(lc) == ['Ha', '[NII]a', '[SII]']


# ---------------------------------------------------------------------------
# Multiplet: use add_lines which auto-generates unique names
# ---------------------------------------------------------------------------


class TestMultiplet:
    def test_doublet_names_via_add_lines(self):
        z = _z('z_nlr')
        fwhm = _fwhm('fwhm_nlr')
        flux = _flux('flux_nii')
        lc = _lc()
        lc.add_lines(
            '[NII]', [6585.0, 6550.0] * u.AA, redshift=z, fwhm_gauss=fwhm, flux=flux
        )
        assert _make_line_labels(lc) == ['[NII]_6585', '[NII]_6550']

    def test_triplet_names_via_add_lines(self):
        z = _z('z_oiii')
        fwhm = _fwhm('fwhm_oiii')
        flux = _flux('flux_oiii')
        lc = _lc()
        lc.add_lines(
            '[OIII]',
            [5007.0, 4959.0, 4363.0] * u.AA,
            redshift=z,
            fwhm_gauss=fwhm,
            flux=flux,
        )
        assert _make_line_labels(lc) == ['[OIII]_5007', '[OIII]_4959', '[OIII]_4363']

    def test_wavelength_suffix_preserves_value(self):
        z = _z()
        fwhm = _fwhm('fwhm')
        flux = _flux('f')
        lc = _lc()
        lc.add_lines(
            'X', [1234.56, 1300.49] * u.AA, redshift=z, fwhm_gauss=fwhm, flux=flux
        )
        assert _make_line_labels(lc) == ['X_1234.56', 'X_1300.49']

    def test_multiplet_mixed_with_unique_name(self):
        z = _z('z_nlr')
        fwhm = _fwhm('fwhm_nlr')
        flux = _flux('flux_nii')
        lc = _lc()
        lc.add_line('Ha', 6563.0 * u.AA)
        lc.add_lines(
            '[NII]', [6585.0, 6550.0] * u.AA, redshift=z, fwhm_gauss=fwhm, flux=flux
        )
        assert _make_line_labels(lc) == ['Ha', '[NII]_6585', '[NII]_6550']


# ---------------------------------------------------------------------------
# Multiple components: use unique names directly
# ---------------------------------------------------------------------------


class TestMultipleComponents:
    def test_two_components_different_fwhm(self):
        """Narrow + broad: encoded directly in the unique line names."""
        z = _z('z')
        flux = _flux('f_ha')
        fwhm_n = _fwhm('narrow')
        fwhm_b = _fwhm('broad')
        lc = _lc()
        lc.add_line(
            'Ha_narrow', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm_n, flux=flux
        )
        lc.add_line('Ha_broad', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm_b, flux=flux)
        assert _make_line_labels(lc) == ['Ha_narrow', 'Ha_broad']

    def test_three_components_distinct_names(self):
        z = _z('z')
        flux = _flux('flux')
        fwhm1 = _fwhm('f1')
        fwhm2 = _fwhm('f2')
        fwhm3 = _fwhm('f3')
        lc = _lc()
        lc.add_line('a_f1', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm1, flux=flux)
        lc.add_line('a_f2', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm2, flux=flux)
        lc.add_line('a_f3', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm3, flux=flux)
        labels = _make_line_labels(lc)
        assert labels == ['a_f1', 'a_f2', 'a_f3']
        assert len(set(labels)) == 3

    def test_two_components_different_redshift(self):
        z1 = _z('z_nlr')
        z2 = _z('z_blr')
        fwhm = _fwhm('fwhm')
        flux = _flux('f_ha')
        lc = _lc()
        lc.add_line('Ha_nlr', 6563.0 * u.AA, redshift=z1, fwhm_gauss=fwhm, flux=flux)
        lc.add_line('Ha_blr', 6563.0 * u.AA, redshift=z2, fwhm_gauss=fwhm, flux=flux)
        assert _make_line_labels(lc) == ['Ha_nlr', 'Ha_blr']

    def test_two_components_all_axes_vary(self):
        z1 = _z('z_nlr')
        z2 = _z('z_blr')
        fwhm1 = _fwhm('fwhm_nlr')
        fwhm2 = _fwhm('fwhm_blr')
        flux1 = _flux('flux_nlr')
        flux2 = _flux('flux_blr')
        lc = _lc()
        lc.add_line('Ha_nlr', 6563.0 * u.AA, redshift=z1, fwhm_gauss=fwhm1, flux=flux1)
        lc.add_line('Ha_blr', 6563.0 * u.AA, redshift=z2, fwhm_gauss=fwhm2, flux=flux2)
        assert _make_line_labels(lc) == ['Ha_nlr', 'Ha_blr']

    def test_only_varying_axes_encoded_in_name(self):
        z = _z('z_shared')
        fwhm1 = _fwhm('narrow')
        fwhm2 = _fwhm('broad')
        flux = _flux('flux_shared')
        lc = _lc()
        lc.add_line('Ha_narrow', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm1, flux=flux)
        lc.add_line('Ha_broad', 6563.0 * u.AA, redshift=z, fwhm_gauss=fwhm2, flux=flux)
        labels = _make_line_labels(lc)
        assert labels == ['Ha_narrow', 'Ha_broad']


# ---------------------------------------------------------------------------
# Combined: multiplet + multiple components
# ---------------------------------------------------------------------------


class TestMultipletPlusComponents:
    def test_doublet_two_kinematic_components(self):
        z = _z('z_nlr')
        fwhm_n = _fwhm('narrow')
        fwhm_b = _fwhm('broad')
        flux1 = _flux('flux_n')
        flux2 = _flux('flux_b')
        lc = _lc()
        lc.add_lines(
            '[NII]_narrow',
            [6585.0, 6550.0] * u.AA,
            redshift=z,
            fwhm_gauss=fwhm_n,
            flux=flux1,
        )
        lc.add_lines(
            '[NII]_broad',
            [6585.0, 6550.0] * u.AA,
            redshift=z,
            fwhm_gauss=fwhm_b,
            flux=flux2,
        )
        labels = _make_line_labels(lc)
        assert labels == [
            '[NII]_narrow_6585',
            '[NII]_narrow_6550',
            '[NII]_broad_6585',
            '[NII]_broad_6550',
        ]

    def test_all_labels_unique(self):
        z = _z('z')
        fwhm1 = _fwhm('n')
        fwhm2 = _fwhm('b')
        flux = _flux('f')
        lc = _lc()
        lc.add_lines(
            '[OIII]_narrow',
            [5007.0, 4959.0] * u.AA,
            redshift=z,
            fwhm_gauss=fwhm1,
            flux=flux,
        )
        lc.add_lines(
            '[OIII]_broad',
            [5007.0, 4959.0] * u.AA,
            redshift=z,
            fwhm_gauss=fwhm2,
            flux=flux,
        )
        labels = _make_line_labels(lc)
        assert len(labels) == len(set(labels)), f'Duplicate labels: {labels}'


# ---------------------------------------------------------------------------
# PseudoVoigt: two fwhm params, unique names per component
# ---------------------------------------------------------------------------


class TestMultiFwhmProfile:
    def test_pseudo_voigt_both_fwhm_names_included(self):
        z = _z('z')
        flux = _flux('f')
        fwhm_g1 = _fwhm('fg_narrow')
        fwhm_l1 = _fwhm('fl_narrow')
        fwhm_g2 = _fwhm('fg_broad')
        fwhm_l2 = _fwhm('fl_broad')
        lc = _lc()
        lc.add_line(
            'Ha_narrow',
            6563.0 * u.AA,
            profile='pseudovoigt',
            redshift=z,
            fwhm_gauss=fwhm_g1,
            fwhm_lorentz=fwhm_l1,
            flux=flux,
        )
        lc.add_line(
            'Ha_broad',
            6563.0 * u.AA,
            profile='pseudovoigt',
            redshift=z,
            fwhm_gauss=fwhm_g2,
            fwhm_lorentz=fwhm_l2,
            flux=flux,
        )
        labels = _make_line_labels(lc)
        assert labels == ['Ha_narrow', 'Ha_broad']


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_config_returns_empty_list(self):
        assert _make_line_labels(_lc()) == []

    def test_labels_are_strings(self):
        lc = _lc()
        lc.add_line('Ha', 6563.0 * u.AA)
        lc.add_line('[NII]', 6585.0 * u.AA)
        assert all(isinstance(lb, str) for lb in _make_line_labels(lc))

    def test_output_length_matches_number_of_entries(self):
        z = _z('z')
        fwhm = _fwhm('fwhm')
        flux = _flux('f')
        lc = _lc()
        lc.add_lines(
            'line',
            [5007.0, 4959.0, 4363.0, 6563.0] * u.AA,
            redshift=z,
            fwhm_gauss=fwhm,
            flux=flux,
        )
        assert len(_make_line_labels(lc)) == 4

    def test_no_leading_or_trailing_underscores(self):
        z = _z('z_nlr')
        fwhm = _fwhm('fwhm_nlr')
        flux = _flux('flux_nii')
        lc = _lc()
        lc.add_lines(
            '[NII]', [6585.0, 6550.0] * u.AA, redshift=z, fwhm_gauss=fwhm, flux=flux
        )
        for lb in _make_line_labels(lc):
            assert not lb.startswith('_')
            assert not lb.endswith('_')

    def test_unique_name_never_gets_token_suffixes(self):
        """A name that appears only once must always be returned bare."""
        lc = _lc()
        lc.add_line(
            'Ha',
            6563.0 * u.AA,
            redshift=_z('some_very_long_redshift_name'),
            fwhm_gauss=_fwhm('some_very_long_fwhm_name'),
            flux=_flux('some_very_long_flux_name'),
        )
        assert _make_line_labels(lc) == ['Ha']
