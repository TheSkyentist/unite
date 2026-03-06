"""Continuum configuration and functional forms."""

from unite.continuum.config import (
    ContinuumConfiguration,
    ContinuumNormalizationWavelength,
    ContinuumRegion,
    ContinuumScale,
)
from unite.continuum.fit import ContinuumFitResult, fit_continuum_form
from unite.continuum.library import (
    AttenuatedBlackbody,
    Bernstein,
    Blackbody,
    BSpline,
    Chebyshev,
    ContinuumForm,
    Linear,
    ModifiedBlackbody,
    Polynomial,
    PowerLaw,
    form_from_dict,
    get_form,
)

__all__ = [
    'AttenuatedBlackbody',
    'BSpline',
    'Bernstein',
    'Blackbody',
    'Chebyshev',
    'ContinuumConfiguration',
    'ContinuumFitResult',
    'ContinuumForm',
    'ContinuumNormalizationWavelength',
    'ContinuumRegion',
    'ContinuumScale',
    'Linear',
    'ModifiedBlackbody',
    'Polynomial',
    'PowerLaw',
    'fit_continuum_form',
    'form_from_dict',
    'get_form',
]
