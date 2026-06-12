"""Continuum configuration and functional forms."""

from unite.continuum.config import (
    ContinuumConfiguration,
    ContinuumRegion,
    ContShape,
    NormWavelength,
    Scale,
)
from unite.continuum.fit import ContinuumFitResult, fit_continuum_form
from unite.continuum.library import (
    AttenuatedBlackbody,
    Bernstein,
    Blackbody,
    BSpline,
    Chebyshev,
    ContinuumForm,
    Legendre,
    Linear,
    ModifiedBlackbody,
    Polynomial,
    PowerLaw,
    Template,
    form_from_dict,
    get_form,
)

__all__ = [
    'AttenuatedBlackbody',
    'BSpline',
    'Bernstein',
    'Blackbody',
    'Chebyshev',
    'ContShape',
    'ContinuumConfiguration',
    'ContinuumFitResult',
    'ContinuumForm',
    'ContinuumRegion',
    'Legendre',
    'Linear',
    'ModifiedBlackbody',
    'NormWavelength',
    'Polynomial',
    'PowerLaw',
    'Scale',
    'Template',
    'fit_continuum_form',
    'form_from_dict',
    'get_form',
]
