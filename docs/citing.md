# Citing

If you use `unite` in your research, please cite it. This helps support the project and makes your work reproducible.

## Zenodo (software release)

All versioned releases of `unite` are archived on Zenodo:

> **<https://doi.org/10.5281/zenodo.15585034>**

This DOI always resolves to the full release record where you can select a specific version. Zenodo generates BibTeX, APA, and other citation formats automatically on that page.

Example BibTeX entry (visit the link above to get the entry for a specific version):

```bibtex
@software{hviding_unite,
  author       = {Hviding, Raphael Erik},
  title        = {{unite}: Unified liNe Integration Turbo Engine},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15585034},
  url          = {https://doi.org/10.5281/zenodo.15585034}
}
```

## NIRSpec Line Spread Function (LSF) data

If you use the NIRSpec LSF data provided by `unite`, please also cite the relevant source.

For the `point` source LSF, cite the paper that developed the forward-modeling methodology
for generating resolution curves for NIRSpec MSA spectroscopy:

```bibtex
@ARTICLE{2024A&A...684A..87D,
       author = {{de Graaff}, Anna and {Rix}, Hans-Walter and {Carniani}, Stefano and
                 {Suess}, Katherine A. and {Charlot}, St{\'e}phane and
                 {Curtis-Lake}, Emma and {Arribas}, Santiago and {Baker}, William M. and
                 {Boyett}, Kristan and {Bunker}, Andrew J. and {Cameron}, Alex J. and
                 {Chevallard}, Jacopo and {Curti}, Mirko and {Eisenstein}, Daniel J. and
                 {Franx}, Marijn and {Hainline}, Kevin and {Hausen}, Ryan and
                 {Ji}, Zhiyuan and {Johnson}, Benjamin D. and {Jones}, Gareth C. and
                 {Maiolino}, Roberto and {Maseda}, Michael V. and {Nelson}, Erica and
                 {Parlanti}, Eleonora and {Rawle}, Tim and {Robertson}, Brant and
                 {Tacchella}, Sandro and {{\"U}bler}, Hannah and {Williams}, Christina C. and
                 {Willmer}, Christopher N.~A. and {Willott}, Chris},
        title = "{Ionised gas kinematics and dynamical masses of z $\gtrsim$ 6 galaxies
                  from JADES/NIRSpec high-resolution spectroscopy}",
      journal = {A\&A},
         year = 2024,
        month = apr,
       volume = {684},
          eid = {A87},
        pages = {A87},
          doi = {10.1051/0004-6361/202347755},
archivePrefix = {arXiv},
       eprint = {2308.09742},
 primaryClass = {astro-ph.GA},
}
```

For the `uniform` LSF, cite the NIRSpec instrument paper:

```bibtex
@ARTICLE{2022A&A...661A..80J,
       author = {{Jakobsen}, P. and {Ferruit}, P. and {Alves de Oliveira}, C. and
                 {Arribas}, S. and {Bagnasco}, G. and {Barho}, R. and {Beck}, T.~L. and
                 {Birkmann}, S. and {B{\"o}ker}, T. and {Bunker}, A.~J. and
                 {Charlot}, S. and {de Jong}, P. and {de Marchi}, G. and
                 {Ehrenwinkler}, R. and {Falcolini}, M. and {Fels}, R. and {Franx}, M. and
                 {Franz}, D. and {Funke}, M. and {Giardino}, G. and {Gnata}, X. and
                 {Holota}, W. and {Honnen}, K. and {Jensen}, P.~L. and {Jentsch}, M. and
                 {Johnson}, T. and {Jollet}, D. and {Karl}, H. and {Kling}, G. and
                 {K{\"o}hler}, J. and {Kolm}, M.-G. and {Kumari}, N. and
                 {Lander}, M.~E. and {Lemke}, R. and {L{\'o}pez-Caniego}, M. and
                 {L{\"u}tzgendorf}, N. and {Maiolino}, R. and {Manjavacas}, E. and
                 {Marston}, A. and {Maschmann}, M. and {Maurer}, R. and
                 {Messerschmidt}, B. and {Moseley}, S.~H. and {Mosner}, P. and
                 {Mott}, D.~B. and {Muzerolle}, J. and {Pirzkal}, N. and
                 {Pittet}, J.-F. and {Plitzke}, A. and {Posselt}, W. and {Rapp}, B. and
                 {Rauscher}, B.~J. and {Rawle}, T. and {Rix}, H.-W. and {R{\"o}del}, A. and
                 {Rumler}, P. and {Sabbi}, E. and {Salvignol}, J.-C. and {Schmid}, T. and
                 {Sirianni}, M. and {Smith}, C. and {Strada}, P. and {te Plate}, M. and
                 {Valenti}, J. and {Wettemann}, T. and {Wiehe}, T. and {Wiesmayer}, M. and
                 {Willott}, C.~J. and {Wright}, R. and {Zeidler}, P. and {Zincke}, C.},
        title = "{The Near-Infrared Spectrograph (NIRSpec) on the James Webb Space
                  Telescope. I. Overview of the instrument and its capabilities}",
      journal = {A\&A},
         year = 2022,
        month = may,
       volume = {661},
          eid = {A80},
        pages = {A80},
          doi = {10.1051/0004-6361/202142663},
archivePrefix = {arXiv},
       eprint = {2202.03305},
 primaryClass = {astro-ph.IM},
}
```
