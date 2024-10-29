from __future__ import annotations

from attrs import define, field, validators


@define
class Outgas:
    """Outgassing parameters (fO2) and included volatiles.

    Attributes
    ----------
    fO2_shift_IW: float
        Homogeneous oxygen fugacity in the magma ocean used to represent redox state (log10 units relative to Iron-Wustite).
    module: str
        Outgassing module to be used. Choices: 'calliope', 'atmodeller'.
    calliope: Calliope
        Parameters for CALLIOPE module.
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
        If True, include H2O outgassing.
    include_CO2: bool
        If True, include CO2 outgassing.
    include_N2: bool
        If True, include N2 outgassing.
    include_S2: bool
        If True, include S2 outgassing.
    include_SO2: bool
        If True, include SO2 outgassing.
    include_H2: bool
        If True, include H2 outgassing.
    include_CH4: bool
        If True, include CH4 outgassing.
    include_CO: bool
        If True, include CO outgassing.
    T_floor: float
        Temperature floor applied to outgassing calculation [K].
    """
    include_H2O: bool
    include_CO2: bool
    include_N2: bool
    include_S2: bool
    include_SO2: bool
    include_H2: bool
    include_CH4: bool
    include_CO: bool
    T_floor: float  = field(validator=validators.gt(0.0))

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
    some_parameter: str
