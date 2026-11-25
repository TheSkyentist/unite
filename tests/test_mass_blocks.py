"""Tests for config-derived NUTS mass-matrix blocks and ``fit(mass_matrix=...)``."""

import astropy.units as u
import numpy as np
import pytest

from unite import model
from unite._mass import mass_blocks
from unite.continuum import ContinuumConfiguration, Linear
from unite.instrument.generic import SimpleDisperser
from unite.line import FWHM, LineConfiguration, Redshift
from unite.prior import Fixed, Uniform
from unite.spectrum import Spectra, Spectrum


def _spectrum(lo=6400.0, hi=6900.0, n=400, calib=False):
    wl = np.linspace(lo, hi, n) * u.AA
    kw = {}
    if calib:
        from unite.instrument import FluxScale, RScale

        kw = {
            'r_scale': RScale(prior=Uniform(0.8, 1.2)),
            'flux_scale': FluxScale(prior=Uniform(0.8, 1.2)),
        }
    disp = SimpleDisperser(wavelength=wl, R=3000.0, name='d', **kw)
    dl = np.gradient(wl)
    unit = u.Unit('1e-17 erg / (s cm2 AA)')
    rng = np.random.default_rng(0)
    flux = (1.0 + rng.normal(0, 0.1, n)) * unit
    return Spectrum(
        low=wl - 0.5 * dl,
        high=wl + 0.5 * dl,
        flux=flux,
        error=np.full(n, 0.1) * unit,
        disperser=disp,
        name='s',
    )


def _build(line_config, cont_config=None, spectrum=None):
    spectra = Spectra([spectrum or _spectrum()], redshift=0.0)
    spectra.prepare(line_config, cont_config)
    spectra.compute_scales(spectra.prepared_line_config, spectra.prepared_cont_config)
    builder = model.ModelBuilder(
        spectra.prepared_line_config, spectra.prepared_cont_config, spectra
    )
    return builder, builder.build()[1]


def _config():
    """Narrow Ha/[NII] sharing kinematics, a separate broad Ha, a far [SII]."""
    lc = LineConfiguration()
    z = Redshift(prior=Uniform(-0.005, 0.005))
    fw = FWHM(prior=Uniform(100.0, 600.0))
    lc.add_line('Ha', 6563.0 * u.AA, redshift=z, fwhm_gauss=fw)
    lc.add_line('NII', 6584.0 * u.AA, redshift=z, fwhm_gauss=fw)
    lc.add_line(
        'Ha_broad',
        6563.0 * u.AA,
        redshift=Redshift(prior=Uniform(-0.005, 0.005)),
        fwhm_gauss=FWHM(prior=Uniform(1000.0, 4000.0)),
    )
    lc.add_line(
        'SII',
        6731.0 * u.AA,
        redshift=Redshift(prior=Uniform(-0.005, 0.005)),
        fwhm_gauss=FWHM(prior=Uniform(100.0, 600.0)),
    )
    return lc


def _free(args):
    return {n for n, p in args.all_priors.items() if not isinstance(p, Fixed)}


class TestMassBlocks:
    def test_partition_of_free_sites(self):
        _, args = _build(_config())
        blocks = args.mass_blocks
        names = [n for b in blocks for n in b]
        assert len(names) == len(set(names))
        assert set(names) <= _free(args)
        assert all(len(b) >= 2 for b in blocks)
        assert all(b == tuple(sorted(b)) for b in blocks)

    def test_deterministic(self):
        _, args = _build(_config())
        assert mass_blocks(args) == mass_blocks(args)

    def test_shared_kinematics_share_a_block(self):
        _, args = _build(_config())
        cm = args.matrices
        # Ha and NII share z and fwhm: their params must sit in one block.
        block = next(b for b in args.mass_blocks if cm.z_names[0] in b)
        assert cm.flux_names[0] in block
        assert cm.flux_names[1] in block
        assert cm.p0_names[0] in block

    def test_independent_line_not_merged(self):
        _, args = _build(_config())
        cm = args.matrices
        sii_flux = cm.flux_names[3]
        block = next(b for b in args.mass_blocks if sii_flux in b)
        assert cm.flux_names[0] not in block

    def test_continuum_region_params_together(self):
        cc = ContinuumConfiguration.from_lines(
            [6563.0, 6731.0] * u.AA, width=2_000 * u.km / u.s, form=Linear()
        )
        _, args = _build(_config(), cc)
        for resolved in args.cont_resolved_params:
            names = [t.name for t in resolved.values() if t.name in _free(args)]
            blocks = [b for b in args.mass_blocks if set(names) & set(b)]
            assert len(blocks) == 1
            assert set(names) <= set(blocks[0])

    def test_line_joins_its_continuum_region(self):
        cc = ContinuumConfiguration.from_lines(
            [6563.0, 6731.0] * u.AA, width=2_000 * u.km / u.s, form=Linear()
        )
        _, args = _build(_config(), cc)
        sii_flux = args.matrices.flux_names[3]
        cont_names = {t.name for t in args.cont_resolved_params[-1].values()}
        block = next(b for b in args.mass_blocks if sii_flux in b)
        assert cont_names & set(block)

    def test_dependent_priors_stay_together(self):
        lc = LineConfiguration()
        z = Redshift(prior=Uniform(-0.005, 0.005))
        narrow = FWHM(prior=Uniform(100.0, 600.0))
        broad = FWHM(prior=Uniform(narrow + 150.0, 4000.0))
        lc.add_line('Ha', 6563.0 * u.AA, redshift=z, fwhm_gauss=narrow)
        lc.add_line('Ha_b', 6563.0 * u.AA, redshift=z, fwhm_gauss=broad)
        _, args = _build(lc)
        n, b = args.matrices.p0_names
        block = next(blk for blk in args.mass_blocks if n in blk)
        assert b in block

    @pytest.mark.parametrize('cap', [2, 3, 5])
    def test_cap_respected_without_dependent_priors(self, cap):
        _, args = _build(_config())
        assert all(len(b) <= cap for b in mass_blocks(args, max_block_size=cap))

    def test_cap_one_gives_no_blocks(self):
        _, args = _build(_config())
        assert mass_blocks(args, max_block_size=1) == []

    def test_bad_cap_raises(self):
        _, args = _build(_config())
        with pytest.raises(ValueError, match='max_block_size'):
            mass_blocks(args, max_block_size=0)

    def test_fixed_sites_excluded(self):
        lc = LineConfiguration()
        lc.add_line(
            'Ha',
            6563.0 * u.AA,
            redshift=Redshift(prior=Fixed(0.0)),
            fwhm_gauss=FWHM(prior=Uniform(100.0, 600.0)),
        )
        _, args = _build(lc)
        names = {n for b in args.mass_blocks for n in b}
        assert names <= _free(args)
        assert not any(isinstance(args.all_priors[n], Fixed) for n in names)

    def test_calibration_tokens_grouped(self):
        _, args = _build(_config(), spectrum=_spectrum(calib=True))
        calib = {
            t.name
            for _, t in args.spectra[0].disperser.calibration_tokens()
            if not isinstance(t.prior, Fixed)
        }
        assert len(calib) >= 2
        block = next(b for b in args.mass_blocks if next(iter(calib)) in b)
        assert calib <= set(block)

    def test_numpyro_accepts_blocks(self):
        from numpyro import infer

        model_fn, args = _build(_config())[0].build()[0], _build(_config())[1]
        kernel = infer.NUTS(model_fn, dense_mass=args.mass_blocks)
        assert kernel is not None


class TestResolveMassMatrix:
    def _args(self):
        return _build(_config())[1]

    def test_options(self):
        args = self._args()
        assert model._resolve_mass_matrix('dense', args) is True
        assert model._resolve_mass_matrix('diag', args) is False
        assert model._resolve_mass_matrix('blocks', args) == args.mass_blocks

    def test_unknown_string(self):
        with pytest.raises(ValueError, match='mass_matrix must be'):
            model._resolve_mass_matrix('full', self._args())

    def test_none_rejected(self):
        with pytest.raises(ValueError, match='mass_matrix must be'):
            model._resolve_mass_matrix(None, self._args())  # ty: ignore[invalid-argument-type]

    def test_custom_groups(self):
        args = self._args()
        a, b = args.matrices.flux_names[:2]
        assert model._resolve_mass_matrix([(b, a)], args) == [tuple(sorted((a, b)))]

    def test_custom_unknown_name(self):
        with pytest.raises(ValueError, match='not sampled'):
            model._resolve_mass_matrix([('nope',)], self._args())

    def test_custom_duplicate_name(self):
        args = self._args()
        a, b, c = args.matrices.flux_names[:3]
        with pytest.raises(ValueError, match='more than one'):
            model._resolve_mass_matrix([(a, b), (b, c)], args)

    def test_blocks_failure_falls_back_to_dense(self, monkeypatch):
        import unite._mass as mass

        def boom(*a, **k):
            raise RuntimeError('boom')

        monkeypatch.setattr(mass, 'mass_blocks', boom)
        with pytest.warns(UserWarning, match="mass_matrix='dense'"):
            assert model._resolve_mass_matrix('blocks', self._args()) is True


@pytest.mark.slow
@pytest.mark.parametrize('mass_matrix', ['blocks', 'dense', 'diag'])
def test_fit_runs_with_each_option(mass_matrix):
    builder, _ = _build(_config())
    samples, _ = builder.fit(
        num_warmup=20,
        num_samples=10,
        seed=0,
        progress_bar=False,
        mass_matrix=mass_matrix,
    )
    assert len(next(iter(samples.values()))) == 10
