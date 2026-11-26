# Default plotting metadata (label, unit scale, log axis) for grid quantities.
from __future__ import annotations

from typing import NamedTuple

from proteus.utils.constants import AU, M_earth, M_sun, R_earth, R_sun, secs_per_day


class QuantityInfo(NamedTuple):
    """Default plotting metadata for one grid input parameter or output column.

    Attributes
    ----------
    label : str
        Display-ready axis or colourbar label, with its own LaTeX ``$...$`` and
        unit. Used directly by matplotlib (no further wrapping needed).
    scale : float
        Factor applied to a raw value before plotting (unit conversion).
    log : bool
        Whether the quantity is shown on a logarithmic axis, and, for a grid
        parameter, whether its colour normalisation is logarithmic.
    """

    label: str
    scale: float
    log: bool


# Default plotting metadata, keyed by grid config path (input parameters) or by
# runtime_helpfile.csv column name (output variables). proteus.utils.plot.get_label,
# get_scale and get_log_scale read this; an absent name falls back to the defaults there.
# A log flag is True wherever utils.coupler.variable_is_logarithmic is (tested).
# Grouped by module, one entry per line, so independent additions merge cleanly.
PRESET_QUANTITIES = {
    ## Grid input parameters (config paths)
    # Orbit module
    'orbit.semimajoraxis': QuantityInfo(r'$a$ [AU]', 1.0, True),
    'orbit.eccentricity': QuantityInfo(r'$e$', 1.0, False),
    # Star module
    'star.mass': QuantityInfo(r'$M_\star$ [$M_\odot$]', 1.0, False),
    'star.age_ini': QuantityInfo('Start age [Gyr]', 1.0, False),
    'star.mors.age_now': QuantityInfo('Stellar age [Gyr]', 1.0, False),
    # Planet structure
    'planet.mass_tot': QuantityInfo(r'$M_{\rm tot}$ [$M_\oplus$]', 1.0, False),
    'planet.volatile_reservoir': QuantityInfo('Volatile reservoir', 1.0, False),
    # core_frac is a mass or radius fraction depending on interior_struct.core_frac_mode.
    'interior_struct.core_frac': QuantityInfo('Core fraction', 1.0, False),
    # Module choices
    'interior_struct.module': QuantityInfo('Structure model', 1.0, False),
    'interior_energetics.module': QuantityInfo('Interior model', 1.0, False),
    'outgas.module': QuantityInfo('Outgassing model', 1.0, False),
    # Atmosphere module
    'atmos_clim.module': QuantityInfo('Atmosphere model', 1.0, False),
    'atmos_clim.surf_state': QuantityInfo('Surface scheme', 1.0, False),
    # Escape module
    'escape.module': QuantityInfo('Escape model', 1.0, False),
    'escape.reservoir': QuantityInfo('Escape reservoir', 1.0, False),
    'escape.zephyrus.efficiency': QuantityInfo(r'$\epsilon$', 1.0, False),
    'escape.zephyrus.Pxuv': QuantityInfo(r'$P_{\rm XUV}$ [bar]', 1.0, True),
    # Outgassing module
    'outgas.fO2_shift_IW': QuantityInfo(r'$\Delta$IW', 1.0, False),
    # Volatile element budgets (planet.elements.<X>_budget, unit set by <X>_mode)
    'planet.elements.H_budget': QuantityInfo('H budget', 1.0, True),
    'planet.elements.C_budget': QuantityInfo('C budget', 1.0, True),
    'planet.elements.N_budget': QuantityInfo('N budget', 1.0, True),
    'planet.elements.S_budget': QuantityInfo('S budget', 1.0, True),
    'planet.elements.O_budget': QuantityInfo('O budget', 1.0, True),
    'planet.elements.metallicity': QuantityInfo(r'Metallicity [$\times$ solar]', 1.0, True),
    ## Output variables, ordered as they appear in runtime_helpfile.csv (GetHelpfileKeys)
    # Model time (solidification_time is computed in post-processing; kept next to Time)
    'Time': QuantityInfo('Time [yr]', 1.0, True),
    'solidification_time': QuantityInfo('Solidification time [yr]', 1.0, True),
    # Orbit (semimajorax is in m, orbital_period in s)
    'semimajorax': QuantityInfo(r'$a$ [AU]', 1.0 / AU, True),
    'separation': QuantityInfo(r'Orbital separation [AU]', 1.0 / AU, True),
    'orbital_period': QuantityInfo('Orbital period [days]', 1.0 / secs_per_day, True),
    'eccentricity': QuantityInfo(r'$e$', 1.0, True),
    # Planet structure
    'R_int': QuantityInfo(r'$R_{\rm int}$ [$R_\oplus$]', 1.0 / R_earth, False),
    'M_int': QuantityInfo(r'$M_{\rm int}$ [$M_\oplus$]', 1.0 / M_earth, False),
    'M_planet': QuantityInfo(r'$M_{\rm planet}$ [$M_\oplus$]', 1.0 / M_earth, False),
    'R_core': QuantityInfo(r'$R_{\rm core}$ [$R_\oplus$]', 1.0 / R_earth, False),
    'P_center': QuantityInfo(r'$P_{\rm center}$ [GPa]', 1.0e-9, False),
    'P_cmb': QuantityInfo(r'$P_{\rm CMB}$ [GPa]', 1.0e-9, False),
    # Temperatures
    'T_surf': QuantityInfo(r'$T_{\rm surf}$ [K]', 1.0, False),
    'T_magma': QuantityInfo(r'$T_{\rm magma}$ [K]', 1.0, False),
    'T_cmb': QuantityInfo(r'$T_{\rm CMB}$ [K]', 1.0, False),
    'T_eqm': QuantityInfo(r'$T_{\rm eq}$ [K]', 1.0, False),
    'T_skin': QuantityInfo(r'$T_{\rm skin}$ [K]', 1.0, False),
    # Energy fluxes (W m-2). F_int, F_atm and F_net can be negative and F_tidal is zero
    # without tides, so those are linear; the always-positive fluxes are logarithmic.
    'F_int': QuantityInfo(r'$F_{\rm int}$ [W m$^{-2}$]', 1.0, False),
    'F_atm': QuantityInfo(r'$F_{\rm atm}$ [W m$^{-2}$]', 1.0, False),
    'F_net': QuantityInfo(r'$F_{\rm net}$ [W m$^{-2}$]', 1.0, False),
    'F_olr': QuantityInfo(r'$F_{\rm OLR}$ [W m$^{-2}$]', 1.0, True),
    'F_ins': QuantityInfo(r'$F_{\rm ins}$ [W m$^{-2}$]', 1.0, True),
    'F_xuv': QuantityInfo(r'$F_{\rm XUV}$ [W m$^{-2}$]', 1.0, True),
    'F_tidal': QuantityInfo(r'$F_{\rm tidal}$ [W m$^{-2}$]', 1.0, False),
    # Interior (melt and mantle); gravity sits in this block in the helpfile
    'gravity': QuantityInfo(r'$g_{\rm surf}$ [m s$^{-2}$]', 1.0, False),
    'Phi_global': QuantityInfo('Melt fraction (mass) [%]', 100.0, False),
    'Phi_global_vol': QuantityInfo('Melt fraction (volume) [%]', 100.0, False),
    'RF_depth': QuantityInfo('Rheological front depth', 1.0, False),
    'M_core': QuantityInfo(r'$M_{\rm core}$ [$M_\oplus$]', 1.0 / M_earth, False),
    'M_mantle': QuantityInfo(r'$M_{\rm mantle}$ [$M_\oplus$]', 1.0 / M_earth, False),
    'T_pot': QuantityInfo(r'$T_{\rm pot}$ [K]', 1.0, False),
    # Star
    'M_star': QuantityInfo(r'$M_\star$ [$M_\odot$]', 1.0 / M_sun, False),
    'R_star': QuantityInfo(r'$R_\star$ [$R_\odot$]', 1.0 / R_sun, False),
    'age_star': QuantityInfo('Stellar age [Gyr]', 1.0e-9, True),
    'T_star': QuantityInfo(r'$T_\star$ [K]', 1.0, False),
    # Observational
    'p_obs': QuantityInfo(r'$P_{\rm obs}$ [bar]', 1.0, True),
    'R_obs': QuantityInfo(r'$R_{\rm obs}$ [$R_\oplus$]', 1.0 / R_earth, False),
    'T_obs': QuantityInfo(r'$T_{\rm obs}$ [K]', 1.0, False),
    'rho_obs': QuantityInfo(r'$\rho_{\rm obs}$ [g cm$^{-3}$]', 1.0e-3, True),
    'transit_depth': QuantityInfo('Transit depth [ppm]', 1.0e6, False),
    'eclipse_depth': QuantityInfo('Eclipse depth [ppm]', 1.0e6, False),
    'bond_albedo': QuantityInfo('Bond albedo', 1.0, False),
    # Atmosphere
    'M_atm': QuantityInfo(r'$M_{\rm atm}$ [kg]', 1.0, True),
    'P_surf': QuantityInfo(r'$P_{\rm surf}$ [bar]', 1.0, True),
    'atm_kg_per_mol': QuantityInfo(r'MMW [g mol$^{-1}$]', 1.0e3, True),
    # Volatile species (per species: whole-planet mass, atmospheric mixing ratio,
    # partial surface pressure), in runtime_helpfile species order
    'H2O_kg_total': QuantityInfo(r'$M_{\rm H_2O}$ [kg]', 1.0, True),
    'H2O_vmr': QuantityInfo(r'$VMR_{\rm H_2O}$', 1.0, True),
    'H2O_bar': QuantityInfo(r'$p_{\rm H_2O}$ [bar]', 1.0, True),
    'CO2_kg_total': QuantityInfo(r'$M_{\rm CO_2}$ [kg]', 1.0, True),
    'CO2_vmr': QuantityInfo(r'$VMR_{\rm CO_2}$', 1.0, True),
    'CO2_bar': QuantityInfo(r'$p_{\rm CO_2}$ [bar]', 1.0, True),
    'O2_kg_total': QuantityInfo(r'$M_{\rm O_2}$ [kg]', 1.0, True),
    'O2_vmr': QuantityInfo(r'$VMR_{\rm O_2}$', 1.0, True),
    'O2_bar': QuantityInfo(r'$p_{\rm O_2}$ [bar]', 1.0, True),
    'H2_kg_total': QuantityInfo(r'$M_{\rm H_2}$ [kg]', 1.0, True),
    'H2_vmr': QuantityInfo(r'$VMR_{\rm H_2}$', 1.0, True),
    'H2_bar': QuantityInfo(r'$p_{\rm H_2}$ [bar]', 1.0, True),
    'CH4_kg_total': QuantityInfo(r'$M_{\rm CH_4}$ [kg]', 1.0, True),
    'CH4_vmr': QuantityInfo(r'$VMR_{\rm CH_4}$', 1.0, True),
    'CH4_bar': QuantityInfo(r'$p_{\rm CH_4}$ [bar]', 1.0, True),
    'CO_kg_total': QuantityInfo(r'$M_{\rm CO}$ [kg]', 1.0, True),
    'CO_vmr': QuantityInfo(r'$VMR_{\rm CO}$', 1.0, True),
    'CO_bar': QuantityInfo(r'$p_{\rm CO}$ [bar]', 1.0, True),
    'N2_kg_total': QuantityInfo(r'$M_{\rm N_2}$ [kg]', 1.0, True),
    'N2_vmr': QuantityInfo(r'$VMR_{\rm N_2}$', 1.0, True),
    'N2_bar': QuantityInfo(r'$p_{\rm N_2}$ [bar]', 1.0, True),
    'NH3_kg_total': QuantityInfo(r'$M_{\rm NH_3}$ [kg]', 1.0, True),
    'NH3_vmr': QuantityInfo(r'$VMR_{\rm NH_3}$', 1.0, True),
    'NH3_bar': QuantityInfo(r'$p_{\rm NH_3}$ [bar]', 1.0, True),
    'S2_kg_total': QuantityInfo(r'$M_{\rm S_2}$ [kg]', 1.0, True),
    'S2_vmr': QuantityInfo(r'$VMR_{\rm S_2}$', 1.0, True),
    'S2_bar': QuantityInfo(r'$p_{\rm S_2}$ [bar]', 1.0, True),
    'SO2_kg_total': QuantityInfo(r'$M_{\rm SO_2}$ [kg]', 1.0, True),
    'SO2_vmr': QuantityInfo(r'$VMR_{\rm SO_2}$', 1.0, True),
    'SO2_bar': QuantityInfo(r'$p_{\rm SO_2}$ [bar]', 1.0, True),
    'H2S_kg_total': QuantityInfo(r'$M_{\rm H_2S}$ [kg]', 1.0, True),
    'H2S_vmr': QuantityInfo(r'$VMR_{\rm H_2S}$', 1.0, True),
    'H2S_bar': QuantityInfo(r'$p_{\rm H_2S}$ [bar]', 1.0, True),
    # Element inventories (whole planet)
    'H_kg_total': QuantityInfo(r'$M_{\rm H}$ [kg]', 1.0, True),
    'O_kg_total': QuantityInfo(r'$M_{\rm O}$ [kg]', 1.0, True),
    'C_kg_total': QuantityInfo(r'$M_{\rm C}$ [kg]', 1.0, True),
    'N_kg_total': QuantityInfo(r'$M_{\rm N}$ [kg]', 1.0, True),
    'S_kg_total': QuantityInfo(r'$M_{\rm S}$ [kg]', 1.0, True),
    # Atmospheric escape
    'esc_rate_total': QuantityInfo(r'$\dot{M}_{\rm esc}$ [g s$^{-1}$]', 1.0e3, True),
}
