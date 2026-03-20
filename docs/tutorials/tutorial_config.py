r"""
Combining Configurations
==========================================

Three patterns for assembling ``unite`` configurations in code rather than
by hand.  No fitting is performed — see the NIRSpec and Generic tutorials
for the full inference workflow.
"""

# %%
# Imports
# -------

import astropy.units as u
from unite import prior
from unite.line.config import FWHM, Flux, LineConfiguration, Redshift

# %%
# Pattern 1 — ``+`` (strict merge)
# ---------------------------------
#
# Use ``+`` when every token name across both configs is distinct.
# It calls ``merge(strict=True)`` under the hood and raises if any token
# names collide — the safe default that prevents accidental sharing.
#
# Here we model blue and red arms of a spectrograph with independent
# kinematics: each arm gets its own redshift and FWHM.
#
# See :doc:`../usage/line_configuration` for the full token and merging
# reference, and :doc:`../usage/priors` for available prior types.

z_blue = Redshift('z_blue', prior=prior.Uniform(-0.003, 0.003))
fwhm_blue = FWHM('fwhm_blue', prior=prior.Uniform(50, 500))
lc_blue = LineConfiguration()
for name, wl in {'Ha': 6564.61 * u.AA, 'NII_6585': 6585.27 * u.AA}.items():
    lc_blue.add_line(
        name, wl, profile='Gaussian', redshift=z_blue, fwhm_gauss=fwhm_blue
    )

z_red = Redshift('z_red', prior=prior.Uniform(-0.003, 0.003))
fwhm_red = FWHM('fwhm_red', prior=prior.Uniform(50, 500))
lc_red = LineConfiguration()
for name, wl in {'Hb': 4862.68 * u.AA, 'OIII_5008': 5008.24 * u.AA}.items():
    lc_red.add_line(name, wl, profile='Gaussian', redshift=z_red, fwhm_gauss=fwhm_red)

lc_independent = lc_blue + lc_red
print(lc_independent)

# %%
# Pattern 2 — ``merge(strict=False)`` (shared parameters)
# ---------------------------------------------------------
#
# Use ``merge(strict=False)`` when two independently-built configs should
# share a kinematic axis — for example, narrow and broad components of
# the same galaxy that must share a systemic redshift.
#
# Same-named tokens of the same type are collapsed into one model parameter.
# ``self``'s token wins; ``other``'s token is replaced throughout the merged
# config.  ``+`` (strict=True) would raise here because both configs
# contain a token named ``'shared'``.
#
# See :doc:`../usage/line_configuration` (Merging Configurations section)
# and :doc:`../usage/priors` for dependent priors.

z = Redshift('shared', prior=prior.Uniform(-0.005, 0.005))

fwhm_narrow = FWHM('narrow', prior=prior.Uniform(50, 500))
lc_narrow = LineConfiguration()
for name, wl in {'Ha': 6564.61 * u.AA, 'Hb': 4862.68 * u.AA}.items():
    lc_narrow.add_line(name, wl, profile='Gaussian', redshift=z, fwhm_gauss=fwhm_narrow)

fwhm_broad = FWHM('broad', prior=prior.Uniform(fwhm_narrow + 300, 8000))
lc_broad = LineConfiguration()
for name, wl in {'Ha': 6564.61 * u.AA, 'Hb': 4862.68 * u.AA}.items():
    lc_broad.add_line(
        f'{name}_broad',
        wl,
        profile='Gaussian',
        redshift=z,
        fwhm_gauss=fwhm_broad,
        flux=Flux(prior=prior.Uniform(0, 10)),
    )

# Demonstrate that + raises, then use strict=False to succeed.
try:
    _ = lc_narrow + lc_broad
except ValueError as e:
    print(f'strict=True raises: {e}')

lc_combined = lc_narrow.merge(lc_broad, strict=False)
print(lc_combined)

# %%
# Pattern 3 — loops for survey targets
# --------------------------------------
#
# Build the configuration once, then reuse it across all targets in a loop.
# The redshift prior is a *relative offset* from the systemic redshift passed
# to ``Spectra`` at fit time — the config itself is target-independent.
#
# See :doc:`../usage/build_model` for how ``Spectra`` consumes the config and
# filters lines by wavelength coverage.

TARGETS = {'Example-001': 0.12, 'Example-002': 1.87, 'Example-003': 6.78}
LINES = {
    'Ha': 6564.61 * u.AA,
    'Hb': 4862.68 * u.AA,
    'OIII_5008': 5008.24 * u.AA,
    'NII_6585': 6585.27 * u.AA,
}

z_s = Redshift('z', prior=prior.Uniform(-0.005, 0.005))
fwhm_s = FWHM('fwhm', prior=prior.Uniform(50, 500))
survey_config = LineConfiguration()
for name, wl in LINES.items():
    survey_config.add_line(
        name, wl, profile='Gaussian', redshift=z_s, fwhm_gauss=fwhm_s
    )

for target, z_sys in TARGETS.items():
    n = len(list(survey_config))
    print(f'{target}  z={z_sys:.2f}  →  {n} lines configured')

# At fit time:
#
#   for target, z_sys in TARGETS.items():
#       spectra = Spectra([load_spectrum(target)], redshift=z_sys)
#       filtered_lines, filtered_cont = spectra.prepare(survey_config, cc)
#       samples, model_args = ModelBuilder(filtered_lines, filtered_cont, spectra).fit()
