from __future__ import annotations

from attrs import define, field, validators


@define
class Calliope:
    """Module parameters for Calliope.

    Attributes
    ----------
    T_floor: float
        Temperature floor applied to chemistry calculation [K].
    include_H2O: bool
        If True, include H2O.
    include_CO2: bool
        If True, include CO2.
    include_N2: bool
        If True, include N2.
    include_S2: bool
        If True, include S2.
    include_SO2: bool
        If True, include SO2.
    include_H2S: bool
        If True, include H2S.
    include_NH3: bool
        If True, include NH3.
    include_H2: bool
        If True, include H2.
    include_CH4: bool
        If True, include CH4.
    include_CO: bool
        If True, include CO.
    rtol: float
        Relative tolerance on solver for mass conservation.
    xtol: float
        Absolute tolerance on solver for mass conservation.
    solubility: bool
        Enable solubility of volatiles into melt.
    """
    T_floor: float      = field(default=700.0, validator=validators.gt(0.0))
    include_H2O: bool   = True
    include_CO2: bool   = True
    include_N2: bool    = True
    include_S2: bool    = True
    include_SO2: bool   = True
    include_H2S: bool   = True
    include_NH3: bool   = True
    include_H2: bool    = True
    include_CH4: bool   = True
    include_CO: bool    = True
    rtol: float         = field(default=1e-4, validator=validators.gt(0.0))
    xtol: float         = field(default=1e-6, validator=validators.gt(0.0))
    solubility: bool    = True

    def is_included(self, vol: str) -> bool:
        """Helper method for getting flag if `vol` is included in outgassing."""
        return getattr(self, f'include_{vol}')

@define
class Atmodeller:
    """Module parameters for Atmodeller.

    Attributes
    ----------
    some_parameter: str
        Not used currently.
    """
    some_parameter: str = field(default="some_value")

@define
class Outgas:
    """Outgassing parameters (fO2) and included volatiles.

    Attributes
    ----------
    fO2_shift_IW: float
        Homogeneous oxygen fugacity in the magma ocean used to represent redox state (log10 units relative to Iron-Wustite).
    module: str
        Outgassing module to be used. Choices: 'calliope' only.
    mass_thresh: float
        Minimum threshold for element mass [kg]. Inventories below this are set to zero.
    calliope: Calliope
        Parameters for CALLIOPE module.
    atmodeller: Atmodeller
        Parameters for atmodeller module.
    """
    fO2_shift_IW: float

    module: str = field(validator=validators.in_(('calliope',)))

    mass_thresh: float = field(default=1e16, validator=validators.gt(0.0))

    calliope: Calliope      = field(factory=Calliope)
    atmodeller: Atmodeller  = field(factory=Atmodeller)
