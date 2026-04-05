"""Dummy interior structure module using Noack & Lasbleis (2020) scaling laws.

Analytical parameterizations for rocky planet interior structure, calibrated
against full interior structure models for 0.8-2 M_Earth with variable iron
content. Provides all radial profiles (P, rho, g, T) needed by SPIDER/Aragog
without EOS tables or iterative solvers.

Reference: Noack & Lasbleis (2020), A&A 638, A129.
    "Parameterisations of interior properties of rocky planets"
    https://doi.org/10.1051/0004-6361/202037723
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import M_earth, const_G

if TYPE_CHECKING:
    from proteus.config import Config

logger = logging.getLogger('fwl.' + __name__)

# Earth mass in kg (for scaling law normalization)
M_EARTH_KG = M_earth


def _iron_fractions(core_frac: float, core_frac_mode: str, fe_mantle: float = 0.1):
    """Compute iron fractions from core fraction and mantle iron number.

    Parameters
    ----------
    core_frac : float
        Core fraction (mass or radius, per core_frac_mode).
    core_frac_mode : str
        'mass' or 'radius'.
    fe_mantle : float
        Mantle iron number #Fe_M (fraction of iron-bearing minerals). Default 0.1.

    Returns
    -------
    x_cmf : float
        Core mass fraction.
    x_fe : float
        Total planet iron weight fraction.
    x_fem : float
        Iron mass fraction in the mantle.
    """
    # Molar masses [g/mol] (Noack & Lasbleis 2020, below Eq. 6)
    m_fe = 55.845
    m_mg = 24.305
    m_si = 28.0855
    m_o = 15.999

    # Iron mass fraction in mantle minerals (Eq. 6)
    x_fem = (2 * fe_mantle * m_fe) / (
        2 * ((1 - fe_mantle) * m_mg + fe_mantle * m_fe) + m_si + 4 * m_o
    )

    if core_frac_mode == 'mass':
        x_cmf = core_frac
    else:
        # For radius mode, approximate X_CMF from radius fraction.
        # Invert Eq. 9 approximately: R_c ~ 4850 * X_CMF^0.328 * (M/M_E)^0.266
        # For a rough estimate, use X_CMF ~ (R_c/R_p)^3 * rho_ratio
        # Simple approximation: X_CMF ~ core_frac^2.5 (empirical fit)
        x_cmf = core_frac ** 2.5
        x_cmf = max(0.01, min(x_cmf, 0.80))

    # Total iron fraction (Eq. 8 rearranged): X_Fe = X_FeM + X_CMF * (1 - X_FeM)
    x_fe = x_fem + x_cmf * (1 - x_fem)

    return x_cmf, x_fe, x_fem


def solve_dummy_structure(
    config: Config,
    hf_row: dict,
    outdir: str,
    num_spider_nodes: int = 0,
):
    """Solve interior structure using Noack & Lasbleis (2020) scaling laws.

    Parameters
    ----------
    config : Config
        PROTEUS configuration.
    hf_row : dict
        Helpfile row (updated in place).
    outdir : str
        Output directory.
    num_spider_nodes : int
        Number of SPIDER mesh nodes (0 = no SPIDER mesh file).

    Returns
    -------
    spider_mesh_file : str or None
        Path to the SPIDER mesh file, if generated.
    """
    M_p = config.planet.mass_tot * M_EARTH_KG  # planet mass [kg]
    m_ratio = config.planet.mass_tot  # M_p / M_Earth

    # Iron fractions
    x_cmf, x_fe, x_fem = _iron_fractions(
        config.interior_struct.core_frac,
        config.interior_struct.core_frac_mode,
    )

    # --- Noack & Lasbleis (2020) scaling laws ---

    # Eq. 5: Planet radius [km]
    R_p_km = (7030.0 - 1840.0 * x_fe) * m_ratio ** 0.282
    R_p = R_p_km * 1e3  # [m]

    # Eq. 9: Core radius (hot profile) [km]
    R_c_km = 4850.0 * x_cmf ** 0.328 * m_ratio ** 0.266
    R_c = R_c_km * 1e3  # [m]

    # Clamp core radius to be smaller than planet radius
    if R_c >= R_p:
        R_c = 0.9 * R_p
        logger.warning('Core radius clamped to 0.9 * R_p (%.0f km)', R_c / 1e3)

    # Eq. 11: Average core density [kg/m^3]
    rho_c = x_cmf * M_p / (4.0 / 3.0 * np.pi * R_c**3)

    # Eq. 12: Average mantle density [kg/m^3]
    rho_m = (1.0 - x_cmf) * M_p / (4.0 / 3.0 * np.pi * (R_p**3 - R_c**3))

    # Eq. 13: Surface gravitational acceleration [m/s^2]
    g_surf = const_G * M_p / R_p**2

    # Eq. 14: CMB gravitational acceleration [m/s^2]
    g_cmb = const_G * x_cmf * M_p / R_c**2

    # Eq. 15: Average mantle gravity
    g_m_av = 0.5 * (g_surf + g_cmb)

    # Eq. 16: CMB pressure [Pa]
    P_cmb = g_m_av * rho_m * (R_p - R_c)  # [Pa]

    # Eq. 19: Average mantle heat capacity [J/kg/K]
    Cp_m = 1275.0 - 585.0 * x_fem**1.06

    # Eq. 18: Average mantle thermal expansivity [1/K]
    alpha_m = (13.0 + 0.738 * x_cmf - 11.0 * m_ratio**0.04) * 1e-5

    # Core mass and mantle mass
    M_core = x_cmf * M_p

    # Core heat capacity (Dulong-Petit for iron, ~450 J/kg/K)
    cfg_heatcap = config.interior_struct.core_heatcap
    core_heatcap = 450.0 if cfg_heatcap == 'self' else float(cfg_heatcap)

    logger.info(
        'Dummy structure (Noack & Lasbleis 2020): '
        'R_p=%.0f km, R_c=%.0f km, M_p=%.2e kg, X_CMF=%.3f, X_Fe=%.3f',
        R_p_km, R_c_km, M_p, x_cmf, x_fe,
    )
    logger.info(
        '  rho_core=%.0f kg/m^3, rho_mantle=%.0f kg/m^3, '
        'g_surf=%.2f m/s^2, P_cmb=%.1f GPa',
        rho_c, rho_m, g_surf, P_cmb * 1e-9,
    )

    # --- Build radial profiles on a mesh ---

    num_levels = config.interior_energetics.num_levels
    N = num_levels  # staggered nodes (mantle only, CMB to surface)

    # Radii: mantle from R_c to R_p (ascending)
    r_stag = np.linspace(R_c, R_p, N)

    # Pressure: linear hydrostatic (Eq. 16 generalized)
    P_stag = P_cmb * (1.0 - (r_stag - R_c) / (R_p - R_c))
    P_stag = np.maximum(P_stag, 1e5)  # floor at 1 bar

    # Density: linear interpolation between rho_c at CMB and rho_m at surface
    # (simplified; real profile has a jump at CMB)
    rho_stag = rho_m * np.ones(N)

    # Gravity: linear interpolation
    g_stag = g_cmb + (g_surf - g_cmb) * (r_stag - R_c) / (R_p - R_c)

    # Temperature profile from planet.temperature_mode
    T_stag = _build_temperature_profile(config, r_stag, P_stag, R_c, R_p, alpha_m, Cp_m, rho_m, g_m_av)

    # --- Update hf_row ---

    hf_row['R_int'] = R_p
    hf_row['R_core'] = R_c
    hf_row['M_int'] = M_p
    hf_row['M_core'] = M_core
    hf_row['gravity'] = g_surf
    hf_row['core_density'] = rho_c
    hf_row['core_heatcap'] = core_heatcap

    # --- Write output files ---

    data_dir = os.path.join(outdir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # zalmoxis_output.dat: mantle profiles (CMB to surface, ascending r)
    output_file = os.path.join(data_dir, 'zalmoxis_output.dat')
    data = np.column_stack([r_stag, P_stag, rho_stag, g_stag, T_stag])
    np.savetxt(output_file, data, fmt='%.17e',
               header='r[m] P[Pa] rho[kg/m3] g[m/s2] T[K]')
    logger.info('Dummy structure output: %s', output_file)

    # zalmoxis_output_temp.txt: temperature on Aragog mesh
    temp_file = os.path.join(data_dir, 'zalmoxis_output_temp.txt')
    np.savetxt(temp_file, T_stag)

    # SPIDER mesh file (if requested)
    spider_mesh_file = None
    if num_spider_nodes > 0:
        spider_mesh_file = _write_spider_mesh(
            data_dir, r_stag, P_stag, rho_stag, g_stag,
            R_c, R_p, num_spider_nodes,
        )

    return spider_mesh_file


def _build_temperature_profile(config, r_stag, P_stag, R_c, R_p, alpha_m, Cp_m, rho_m, g_m_av):
    """Build temperature profile based on planet.temperature_mode.

    Parameters
    ----------
    config : Config
    r_stag : ndarray
        Radii [m], ascending from CMB to surface.
    P_stag : ndarray
        Pressure [Pa].
    R_c, R_p : float
        Core and planet radii [m].
    alpha_m, Cp_m : float
        Mantle thermal expansivity [1/K] and heat capacity [J/kg/K].
    rho_m : float
        Mantle density [kg/m^3].
    g_m_av : float
        Average mantle gravity [m/s^2].

    Returns
    -------
    T_stag : ndarray
        Temperature at each radius [K].
    """
    mode = config.planet.temperature_mode
    T_surf = config.planet.tsurf_init
    N = len(r_stag)

    if mode == 'isothermal':
        return np.full(N, T_surf)

    elif mode == 'linear':
        T_center = config.planet.tcenter_init
        return T_center + (T_surf - T_center) * (r_stag - R_c) / (R_p - R_c)

    elif mode == 'adiabatic':
        # Adiabatic gradient: dT/dr = -alpha * T * g / Cp (negative = T increases inward)
        # Integrate from surface (r_stag[-1]) inward
        # Use Noack & Lasbleis Eq. 22 approach with nabla_ad ~ alpha * g * T / (rho * Cp)
        T = np.zeros(N)
        T[-1] = T_surf  # surface
        for i in range(N - 2, -1, -1):
            dr = r_stag[i + 1] - r_stag[i]  # positive (ascending mesh)
            # dT/dr < 0 (T increases inward), so T[i] > T[i+1]
            dTdr = -alpha_m * T[i + 1] * g_m_av / Cp_m
            T[i] = T[i + 1] - dTdr * dr  # subtract because dr > 0 and dT/dr < 0
        return T

    elif mode == 'accretion':
        # White & Li (2025) computes T_surf from accretion energy.
        # For the dummy module, use a simplified version:
        # T_cmb from Noack & Lasbleis Eq. 20 (hot silicate melting)
        fe_m = 0.1  # Earth-like mantle iron number
        P_cmb_GPa = P_stag[0] * 1e-9
        T_cmb = 5400.0 * (P_cmb_GPa / 140.0)**0.48 / (1.0 - np.log(1.0 - fe_m))
        # Adiabat from CMB to surface
        T = np.zeros(N)
        T[0] = T_cmb
        for i in range(1, N):
            dr = r_stag[i] - r_stag[i - 1]
            dTdr = -alpha_m * T[i - 1] * g_m_av / Cp_m
            T[i] = T[i - 1] + dTdr * dr  # T decreases outward
        # Store computed T_surf for downstream use
        logger.info(
            'Dummy accretion mode: T_cmb=%.0f K, T_surf=%.0f K (from CMB melting curve)',
            T_cmb, T[-1],
        )
        return T

    else:
        raise ValueError(f"Unknown temperature_mode: '{mode}'")


def _write_spider_mesh(data_dir, r_stag, P_stag, rho_stag, g_stag, R_c, R_p, num_nodes):
    """Write SPIDER-format mesh file.

    SPIDER convention: radii from surface to CMB, gravity negated.
    """
    # Basic nodes (N): surface to CMB.
    # SPIDER convention: -n N means N basic nodes, N-1 staggered nodes.
    # This matches the Zalmoxis mesh convention.
    N = num_nodes
    r_basic = np.linspace(R_p, R_c, N)
    P_basic = np.interp(r_basic, r_stag, P_stag)
    rho_basic = np.interp(r_basic, r_stag, rho_stag)
    g_basic = -np.interp(r_basic, r_stag, g_stag)  # SPIDER: negative = inward

    # Staggered nodes (N-1): midpoints
    r_stag_spider = 0.5 * (r_basic[:-1] + r_basic[1:])
    P_stag_spider = np.interp(r_stag_spider, r_stag, P_stag)
    rho_stag_spider = np.interp(r_stag_spider, r_stag, rho_stag)
    g_stag_spider = -np.interp(r_stag_spider, r_stag, g_stag)

    mesh_file = os.path.join(data_dir, 'spider_mesh.dat')
    with open(mesh_file, 'w') as f:
        f.write(f'# {N} {N - 1}\n')
        for i in range(N):
            f.write(f'{r_basic[i]:.17e} {P_basic[i]:.17e} '
                    f'{rho_basic[i]:.17e} {g_basic[i]:.17e}\n')
        for i in range(N - 1):
            f.write(f'{r_stag_spider[i]:.17e} {P_stag_spider[i]:.17e} '
                    f'{rho_stag_spider[i]:.17e} {g_stag_spider[i]:.17e}\n')

    logger.info('Dummy SPIDER mesh: %s (%d basic + %d staggered nodes)', mesh_file, N, N - 1)
    return mesh_file
