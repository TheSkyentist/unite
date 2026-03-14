"""Top-level configuration container for unite models.

:class:`Configuration` bundles a :class:`~unite.line.LineConfiguration`,
an optional :class:`~unite.continuum.ContinuumConfiguration`, and an optional
:class:`~unite.instrument.config.InstrumentConfig` into a single
serializable object.

All sub-configs can still be serialized individually via their own
``to_dict`` / ``from_dict`` methods.  :class:`Configuration` adds
convenience for round-tripping the combined object to YAML.

Examples
--------
Build and save:

>>> from unite.line import LineConfiguration, Redshift, FWHM
>>> from unite.continuum import ContinuumConfiguration
>>> from unite.instrument import InstrumentConfig, RScale
>>> from unite.instrument.nirspec import G235H, G395H
>>> from unite.prior import TruncatedNormal
>>> import astropy.units as u
>>> z, w = Redshift('nlr'), FWHM('nlr')
>>> lines = LineConfiguration()
>>> lines.add_line('Ha', 6564.61 * u.AA, redshift=z, fwhm=w)
>>> cont = ContinuumConfiguration.from_lines(lines.wavelengths)
>>> r = RScale(prior=TruncatedNormal(1.0, 0.05, 0.8, 1.2))
>>> dispersers = InstrumentConfig([G235H(r_scale=r), G395H(r_scale=r)])
>>> cfg = Configuration(lines, cont, dispersers=dispersers)
>>> cfg.save('config.yaml')

Load and inspect:

>>> cfg2 = Configuration.load('config.yaml')
>>> cfg2.lines
LineConfiguration: ...
>>> cfg2.dispersers
InstrumentConfig: 2 disperser(s) ...
"""

from __future__ import annotations

from pathlib import Path

import yaml

from unite.continuum.config import ContinuumConfiguration
from unite.line.config import LineConfiguration


class Configuration:
    """Container for a complete unite model configuration.

    Parameters
    ----------
    lines : LineConfiguration
        Emission line configuration.
    continuum : ContinuumConfiguration, optional
        Continuum configuration.  ``None`` if not used.
    dispersers : InstrumentConfig, optional
        Instrument configuration describing disperser models and calibration
        tokens.  When provided,
        :meth:`~unite.instrument.config.InstrumentConfig.validate`
        is called immediately and :class:`UserWarning` is issued if any
        calibration axis lacks a fixed anchor.

    Attributes
    ----------
    lines : LineConfiguration
    continuum : ContinuumConfiguration or None
    dispersers : InstrumentConfig or None
    """

    def __init__(
        self,
        lines: LineConfiguration,
        continuum: ContinuumConfiguration | None = None,
        dispersers=None,
    ) -> None:
        self.lines = lines
        self.continuum = continuum
        self.dispersers = dispersers
        if dispersers is not None:
            dispersers.validate()

    # ------------------------------------------------------------------
    # Dict serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a YAML-safe dictionary.

        Returns
        -------
        dict
            Contains ``'lines'`` and, if present, ``'continuum'`` and
            ``'dispersers'`` keys.
        """
        d: dict = {'lines': self.lines.to_dict()}
        if self.continuum is not None:
            d['continuum'] = self.continuum.to_dict()
        if self.dispersers is not None:
            d['dispersers'] = self.dispersers.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Configuration:
        """Deserialize from a dictionary.

        Parameters
        ----------
        d : dict
            As produced by :meth:`to_dict`.

        Returns
        -------
        Configuration
        """
        from unite.instrument.config import InstrumentConfig

        lines = LineConfiguration.from_dict(d['lines'])
        continuum = None
        if 'continuum' in d:
            continuum = ContinuumConfiguration.from_dict(d['continuum'])
        dispersers = None
        if 'dispersers' in d:
            dispersers = InstrumentConfig.from_dict(d['dispersers'])
        # Bypass validate() on load — the config was already validated when saved.
        obj = cls.__new__(cls)
        obj.lines = lines
        obj.continuum = continuum
        obj.dispersers = dispersers
        return obj

    # ------------------------------------------------------------------
    # YAML serialization
    # ------------------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialize to a YAML string.

        Returns
        -------
        str
        """
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> Configuration:
        """Deserialize from a YAML string.

        Parameters
        ----------
        text : str
            YAML string as produced by :meth:`to_yaml`.

        Returns
        -------
        Configuration
        """
        return cls.from_dict(yaml.safe_load(text))

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a YAML file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        """
        Path(path).write_text(self.to_yaml())

    @classmethod
    def load(cls, path: str | Path) -> Configuration:
        """Load from a YAML file.

        Parameters
        ----------
        path : str or Path
            Path to a YAML file written by :meth:`save`.

        Returns
        -------
        Configuration
        """
        return cls.from_yaml(Path(path).read_text())

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __add__(self, other: Configuration) -> Configuration:
        """Combine two configurations (strict mode — raises on name collisions).

        Merges lines, continuum regions, and dispersers from both configurations.
        Each sub-config combination follows strict mode — raises if any parameter
        or disperser names collide.

        Parameters
        ----------
        other : Configuration

        Returns
        -------
        Configuration
            New configuration combining lines, continuum, and dispersers from both
            *self* and *other*.

        Raises
        ------
        ValueError
            If any parameter or disperser names appear in both configurations.
        TypeError
            If *other* is not a :class:`Configuration`.
        """
        if not isinstance(other, Configuration):
            return NotImplemented

        # Combine lines (required, always present)
        lines = self.lines + other.lines

        # Combine continuum (optional)
        continuum = None
        if self.continuum is not None and other.continuum is not None:
            continuum = self.continuum + other.continuum
        elif self.continuum is not None:
            continuum = self.continuum
        elif other.continuum is not None:
            continuum = other.continuum

        # Combine dispersers (optional)
        dispersers = None
        if self.dispersers is not None and other.dispersers is not None:
            dispersers = self.dispersers + other.dispersers
        elif self.dispersers is not None:
            dispersers = self.dispersers
        elif other.dispersers is not None:
            dispersers = other.dispersers

        return Configuration(lines, continuum=continuum, dispersers=dispersers)

    def __repr__(self) -> str:
        cont_repr = repr(self.continuum) if self.continuum is not None else 'None'
        dispersers_repr = (
            repr(self.dispersers) if self.dispersers is not None else 'None'
        )
        return f'Configuration(\n  lines={self.lines!r},\n  continuum={cont_repr},\n  dispersers={dispersers_repr}\n)'
