# Citing unite

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

<!--## JOSS (journal paper)

A [Journal of Open Source Software (JOSS)](https://joss.theoj.org/) paper is planned. Once published, please also cite:

```bibtex
@article{hviding_unite_joss,
  author  = {Hviding, Raphael Erik},
  title   = {{unite}: Unified liNe Integration Turbo Engine},
  journal = {Journal of Open Source Software},
  year    = {},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {}
}
```

*(This section will be updated upon JOSS acceptance.)*-->

## How Zenodo integration works

When a new GitHub release is created, Zenodo automatically archives a snapshot of the repository and mints a DOI. The `CITATION.cff` file in this repository is read by Zenodo (and GitHub's "Cite this repository" button) to populate author and metadata fields correctly.

For reproducibility, prefer citing the DOI for the specific version you used, which you can find by visiting the Zenodo record above.
