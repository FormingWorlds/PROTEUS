from __future__ import annotations

from attrs import define, field, validators


@define
class Outgas:
    """Outgassing parameters (fO2) and included volatiles.

    Attributes
    ----------
    fO2_shift_IW: float
        log10(ΔIW), atmosphere/interior boundary oxidation state.
    module: str
        Which outgassing module to use, choices: 'calliope', 'atmodeller'.
    calliope: Calliope
        Parameters for calliope module.
    atmodeller: Atmodeller
        Parameters for atmodeller module.
    """
    fO2_shift_IW: float

    module: str = field(validator=validators.in_(('calliope', 'atmodeller')))

    calliope: Calliope
    atmodeller: Atmodeller


@define
class Calliope:
    """Module parameters for Calliope.

    Attributes
    ----------
    include_H2O: bool
        If True, include H2O compound.
    include_CO2: bool
        If True, include CO2 compound.
    include_N2: bool
        If True, include N2 compound.
    include_S2: bool
        If True, include S2 compound.
    include_SO2: bool
        If True, include SO2 compound.
    include_H2: bool
        If True, include H2 compound.
    include_CH4: bool
        If True, include CH4 compound.
    include_CO: bool
        If True, include CO compound.
    """
    include_H2O: bool
    include_CO2: bool
    include_N2: bool
    include_S2: bool
    include_SO2: bool
    include_H2: bool
    include_CH4: bool
    include_CO: bool


@define
class Atmodeller:
    """Module parameters for Atmodeller.

    Attributes
    ----------
    some_parameter: str
        Not used currently.
    """
    some_parameter: str
