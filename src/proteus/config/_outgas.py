from __future__ import annotations

from attrs import define, field, validators

from ._converters import none_if_none


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

    T_floor: float = field(default=700.0, validator=validators.gt(0.0))
    include_H2O: bool = True
    include_CO2: bool = True
    include_N2: bool = True
    include_S2: bool = True
    include_SO2: bool = True
    include_H2S: bool = True
    include_NH3: bool = True
    include_H2: bool = True
    include_CH4: bool = True
    include_CO: bool = True
    rtol: float = field(default=1e-4, validator=validators.gt(0.0))
    xtol: float = field(default=1e-6, validator=validators.gt(0.0))
    solubility: bool = True

    def is_included(self, vol: str) -> bool:
        """Helper method for getting flag if `vol` is included in outgassing."""
        return getattr(self, f'include_{vol}')


@define
class Atmodeller:
    """Module parameters for Atmodeller (Bower+2025, ApJ 995:59).

    JAX-based volatile partitioning with real gas EOS, non-ideal
    solubility laws, and condensation. Replaces CALLIOPE for
    thermodynamically consistent magma-atmosphere equilibrium.

    Attributes
    ----------
    solver_mode : str
        Root-finding mode: 'robust' (slower compile, better convergence)
        or 'basic' (faster compile, less robust).
    solver_atol : float
        Absolute tolerance for the root-finder.
    solver_max_steps : int
        Maximum iterations for the root-finder.
    solver_multistart : int
        Number of random restarts for the root-finder.
    include_condensates : bool
        Enable condensate phases (graphite, etc.) in the equilibrium.
    T_floor : float
        Temperature floor [K]. Outgassing skipped below this temperature.
    solubility_H2O : str
        Solubility law for H2O. See atmodeller.solubility.library.
    solubility_CO2 : str
        Solubility law for CO2.
    solubility_H2 : str
        Solubility law for H2.
    solubility_N2 : str
        Solubility law for N2.
    solubility_S2 : str
        Solubility law for S2.
    solubility_CO : str
        Solubility law for CO. 'none' = no solubility.
    solubility_CH4 : str
        Solubility law for CH4. 'none' = no solubility.
    """

    solver_mode: str = field(
        default='robust',
        validator=validators.in_(('robust', 'basic')),
    )
    solver_atol: float = field(default=1e-6, validator=validators.gt(0))
    solver_max_steps: int = field(default=256, validator=validators.gt(0))
    solver_multistart: int = field(default=10, validator=validators.gt(0))
    include_condensates: bool = True
    T_floor: float = field(default=700.0, validator=validators.ge(0))
    solubility_H2O: str = 'H2O_peridotite_sossi23'
    solubility_CO2: str = 'CO2_basalt_dixon95'
    solubility_H2: str = 'H2_basalt_hirschmann12'
    solubility_N2: str = 'N2_basalt_dasgupta22'
    solubility_S2: str = 'S2_sulfide_basalt_boulliung23'
    solubility_CO: str = field(default='CO_basalt_yoshioka19', converter=none_if_none)
    solubility_CH4: str = field(default='CH4_basalt_ardia13', converter=none_if_none)


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
    h2_binodal: bool
        Enable binodal-controlled H2 partitioning between atmosphere and
        magma ocean using the Rogers+2025 H2-MgSiO3 miscibility model.
        When enabled, H2 dissolves into the melt above the binodal
        temperature and remains in the atmosphere below it.
    calliope: Calliope
        Parameters for CALLIOPE module.
    atmodeller: Atmodeller
        Parameters for atmodeller module.
    """

    fO2_shift_IW: float

    module: str = field(validator=validators.in_(('calliope', 'atmodeller')))

    mass_thresh: float = field(default=1e16, validator=validators.gt(0.0))

    h2_binodal: bool = False

    calliope: Calliope = field(factory=Calliope)
    atmodeller: Atmodeller = field(factory=Atmodeller)
