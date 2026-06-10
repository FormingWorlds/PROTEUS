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
from proteus.utils.structure_estimate import (
    iron_fractions as _iron_fractions,
)
from proteus.utils.structure_estimate import (
    nl20_core_radius_km,
    nl20_planet_radius_km,
)

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Earth mass in kg (for scaling law normalization)
M_EARTH_KG = M_earth

# Fei et al. (2021, Nat. Commun. 12, 876) MgSiO3 melting temperature is
# calibrated to ~500 GPa; above this the liquidus_super anchor extrapolates.
FEI2021_LIQUIDUS_P_CALIB_PA = 500e9


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
        mass_tot_M_earth=m_ratio,
    )
    log.debug(
        'Dummy structure: M_p=%.4f M_earth, core_frac=%.3f (%s), x_cmf=%.3f',
        m_ratio,
        config.interior_struct.core_frac,
        config.interior_struct.core_frac_mode,
        x_cmf,
    )

    # --- Noack & Lasbleis (2020) scaling laws ---

    # NL20 Eq. 5 / Eq. 9, shared with structure_estimate so the radius-mode
    # core-fraction inversion round-trips to the same R_c/R_p here.
    R_p_km = nl20_planet_radius_km(x_fe, m_ratio)
    R_p = R_p_km * 1e3  # [m]
    R_c_km = nl20_core_radius_km(x_cmf, m_ratio)
    R_c = R_c_km * 1e3  # [m]

    # Clamp core radius to be smaller than planet radius
    if R_c >= R_p:
        R_c = 0.9 * R_p
        log.warning('Core radius clamped to 0.9 * R_p (%.0f km)', R_c / 1e3)

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

    log.info(
        'Dummy structure (Noack & Lasbleis 2020): '
        'R_p=%.0f km, R_c=%.0f km, M_p=%.2e kg, X_CMF=%.3f, X_Fe=%.3f',
        R_p_km,
        R_c_km,
        M_p,
        x_cmf,
        x_fe,
    )
    log.info(
        '  rho_core=%.0f kg/m^3, rho_mantle=%.0f kg/m^3, g_surf=%.2f m/s^2, P_cmb=%.1f GPa',
        rho_c,
        rho_m,
        g_surf,
        P_cmb * 1e-9,
    )

    # The dummy (Noack & Lasbleis) structure derives core density from the core
    # mass fraction and planet mass, so a numeric core_density is not applied
    # here (unlike core_heatcap above). Warn rather than silently diverge from
    # the configured value.
    cfg_density = getattr(config.interior_struct, 'core_density', 'self')
    if cfg_density != 'self' and not np.isclose(float(cfg_density), rho_c, rtol=1e-3):
        log.warning(
            'Configured core_density=%.0f kg/m^3 is not applied by the dummy '
            '(Noack & Lasbleis) structure, which derives rho_core=%.0f kg/m^3 '
            "from the core mass fraction and planet mass. Set core_density='self' "
            "or use interior_struct.module='zalmoxis' to control it.",
            float(cfg_density),
            rho_c,
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
    T_stag = _build_temperature_profile(
        config, r_stag, P_stag, R_c, R_p, alpha_m, Cp_m, rho_m, g_m_av
    )

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
    np.savetxt(output_file, data, fmt='%.17e', header='r[m] P[Pa] rho[kg/m3] g[m/s2] T[K]')
    log.info('Dummy structure output: %s', output_file)

    # zalmoxis_output_temp.txt: temperature on Aragog mesh
    temp_file = os.path.join(data_dir, 'zalmoxis_output_temp.txt')
    np.savetxt(temp_file, T_stag)

    # SPIDER mesh file (if requested)
    spider_mesh_file = None
    if num_spider_nodes > 0:
        spider_mesh_file = _write_spider_mesh(
            data_dir,
            r_stag,
            P_stag,
            rho_stag,
            g_stag,
            R_c,
            R_p,
            num_spider_nodes,
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

    elif mode == 'isentropic':
        # Entropy-based IC: T from the analytical formula T = T_ref * exp((S-S_ref)/Cp).
        # When paired with const_properties, the interior solver uses this same formula.
        # The structure module just needs a reasonable T profile for hf_row initialization.
        # Use the adiabatic profile from T_surf as a proxy.
        T = np.zeros(N)
        T[-1] = T_surf
        for i in range(N - 2, -1, -1):
            dr = r_stag[i + 1] - r_stag[i]
            dTdr = -alpha_m * T[i + 1] * g_m_av / Cp_m
            T[i] = T[i + 1] - dTdr * dr
        return T

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

    elif mode == 'liquidus_super':
        # CMB-anchored adiabat using the Fei+2021 MgSiO3 liquidus.
        # T_cmb = T_liq_Fei2021(P_cmb) + delta_T_super, then integrate
        # the adiabat upward to the surface. This is the EoS-agnostic
        # IC anchor: the third-party Fei+2021 calibration is shared
        # across PALEOS-internal use and external references, so
        # neither the WB17 (S_0=0) nor the PALEOS (Stebbins-anchored)
        # entropy convention biases the IC. The dummy module provides
        # only a coarse adiabat from constant alpha/Cp/g; the
        # production path (zalmoxis + Aragog) computes the same anchor
        # against the converged structure-solve P_cmb.
        try:
            from zalmoxis.melting_curves import paleos_liquidus
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                'liquidus_super mode requires Zalmoxis '
                '(zalmoxis.melting_curves.paleos_liquidus); import failed: '
                f"{e}. Use temperature_mode='adiabatic_from_cmb' for a "
                'structure-solver-free initial condition.'
            )

        P_cmb = float(P_stag[0])
        if P_cmb > FEI2021_LIQUIDUS_P_CALIB_PA:
            log.warning(
                'Dummy liquidus_super: P_cmb=%.0f GPa exceeds the Fei+2021 '
                'MgSiO3 melting-curve calibration (~%.0f GPa); the CMB anchor '
                'is an extrapolation at this planet mass.',
                P_cmb / 1e9,
                FEI2021_LIQUIDUS_P_CALIB_PA / 1e9,
            )
        T_liq = float(paleos_liquidus(P_cmb))
        delta = float(config.planet.delta_T_super)
        T_cmb = T_liq + delta
        log.info(
            'Dummy liquidus_super: P_cmb=%.2e Pa -> T_liq_Fei2021=%.0f K '
            '+ delta_T_super=%.0f K = T_cmb=%.0f K',
            P_cmb,
            T_liq,
            delta,
            T_cmb,
        )
        T = np.zeros(N)
        T[0] = T_cmb
        for i in range(1, N):
            dr = r_stag[i] - r_stag[i - 1]
            dTdr = -alpha_m * T[i - 1] * g_m_av / Cp_m
            T[i] = T[i - 1] + dTdr * dr  # T decreases outward
        # The constant-coefficient linearized adiabat can overshoot downward
        # over a deep, high-gravity mantle and dip below the liquidus toward
        # the surface. Clamp to the local liquidus so the profile stays molten,
        # consistent with the superliquidus intent of this mode.
        T_liq_profile = paleos_liquidus(P_stag)
        undershoot = T < T_liq_profile
        if np.any(undershoot):
            log.warning(
                'Dummy liquidus_super: linearized adiabat dipped below the '
                'liquidus at %d of %d levels; clamping those to the local '
                'liquidus to keep the initial profile molten.',
                int(np.count_nonzero(undershoot)),
                N,
            )
            T = np.maximum(T, T_liq_profile)
        return T

    elif mode == 'accretion':
        # White & Li (2025) computes T_surf from accretion energy.
        # For the dummy module, use a simplified version:
        # T_cmb from Noack & Lasbleis Eq. 20 (hot silicate melting)
        fe_m = 0.1  # Earth-like mantle iron number
        P_cmb_GPa = P_stag[0] * 1e-9
        T_cmb = 5400.0 * (P_cmb_GPa / 140.0) ** 0.48 / (1.0 - np.log(1.0 - fe_m))
        # Adiabat from CMB to surface
        T = np.zeros(N)
        T[0] = T_cmb
        for i in range(1, N):
            dr = r_stag[i] - r_stag[i - 1]
            dTdr = -alpha_m * T[i - 1] * g_m_av / Cp_m
            T[i] = T[i - 1] + dTdr * dr  # T decreases outward
        # Store computed T_surf for downstream use
        log.info(
            'Dummy accretion mode: T_cmb=%.0f K, T_surf=%.0f K (from CMB melting curve)',
            T_cmb,
            T[-1],
        )
        return T

    elif mode == 'adiabatic_from_cmb':
        # Adiabat anchored at the user-specified CMB temperature, integrated
        # upward from the CMB to the surface. Used when the planet IC is
        # parameterised by a known T_cmb rather than by T_surf (adiabatic)
        # or by an accretion / liquidus calibration. Same gradient form as
        # the other CMB-anchored modes; only the anchor differs.
        T_cmb = config.planet.tcmb_init
        T = np.zeros(N)
        T[0] = T_cmb
        for i in range(1, N):
            dr = r_stag[i] - r_stag[i - 1]
            dTdr = -alpha_m * T[i - 1] * g_m_av / Cp_m
            T[i] = T[i - 1] + dTdr * dr  # T decreases outward
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
            f.write(
                f'{r_basic[i]:.17e} {P_basic[i]:.17e} {rho_basic[i]:.17e} {g_basic[i]:.17e}\n'
            )
        for i in range(N - 1):
            f.write(
                f'{r_stag_spider[i]:.17e} {P_stag_spider[i]:.17e} '
                f'{rho_stag_spider[i]:.17e} {g_stag_spider[i]:.17e}\n'
            )

    log.info('Dummy SPIDER mesh: %s (%d basic + %d staggered nodes)', mesh_file, N, N - 1)
    return mesh_file
