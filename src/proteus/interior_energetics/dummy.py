# Dummy interior module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from proteus.interior_energetics.common import Interior_t
from proteus.interior_energetics.timestep import next_step
from proteus.interior_energetics.wrapper import get_core_heatcap
from proteus.utils.constants import secs_per_year

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def _solidus_liquidus(config: Config) -> tuple[float, float]:
    """Return the (solidus, liquidus) the active scalar backend uses [K].

    The dummy and boundary backends carry separate melting curves; the melt
    fraction of a re-melt must use the one the running backend evolves against,
    not the dummy defaults for both.
    """
    if config.interior_energetics.module == 'boundary':
        b = config.interior_energetics.boundary
        return b.T_solidus, b.T_liquidus
    d = config.interior_energetics.dummy
    return d.mantle_tsol, d.mantle_tliq


def melt_fraction(config: Config, temperature: float) -> float:
    """Global melt fraction of the scalar-backend mantle at a surface temperature.

    Linear between the active backend's solidus and liquidus, saturating at
    fully solid below the solidus and fully molten above the liquidus.

    Parameters
    ----------
    config : Config
        Model configuration.
    temperature : float
        Surface magma temperature [K].

    Returns
    -------
    float
        Melt fraction in [0, 1].
    """
    tsol, tliq = _solidus_liquidus(config)
    if temperature >= tliq:
        return 1.0
    if temperature <= tsol:
        return 0.0
    return (temperature - tsol) / (tliq - tsol)


def melt_state_from_temperature(config: Config, hf_row: dict, temperature: float) -> dict:
    """Mantle melt quantities implied by a surface magma temperature.

    Derives every temperature-dependent mantle quantity the dummy backend
    exposes, so a caller that changes the magma temperature outside the
    normal solve (a giant-impact re-melt) can rewrite a fully consistent
    state rather than leaving the melt fraction and reservoir masses stale.

    Parameters
    ----------
    config : Config
        Model configuration.
    hf_row : dict
        Current helpfile row, read for the structure (``M_int``, ``M_core``,
        ``R_int``, ``R_core``).
    temperature : float
        Surface magma temperature [K].

    Returns
    -------
    dict
        ``T_magma``, ``T_pot``, ``Phi_global``, ``Phi_global_vol``,
        ``M_mantle_liquid``, ``M_mantle_solid`` and ``RF_depth``.
    """
    phi = melt_fraction(config, temperature)
    m_mantle = hf_row['M_int'] - hf_row['M_core']
    r_core = hf_row.get('R_core', config.interior_struct.core_frac * hf_row['R_int'])
    core_radius_frac = r_core / hf_row['R_int']
    return {
        'T_magma': float(temperature),
        'T_pot': float(temperature),
        'Phi_global': phi,
        'Phi_global_vol': phi,
        'M_mantle_liquid': m_mantle * phi,
        'M_mantle_solid': m_mantle * (1.0 - phi),
        'RF_depth': phi * (1.0 - core_radius_frac),
    }


def calculate_simple_mantle_mass(radius: float, core_frac: float, density: float) -> float:
    """
    A very simple interior structure model.

    This calculates mantle mass given planetary mass, radius, and core fraction. This
    assumes a core density equal to that of Earth's, and that the planet mass is simply
    the sum of mantle and core.
    """

    # Volume of mantle shell
    mantle_volume = (4 * np.pi / 3) * (radius**3 - (radius * core_frac) ** 3)

    # Get mass [in SI units]
    mantle_mass = mantle_volume * density

    log.debug('Total mantle mass = %.2e kg' % mantle_mass)
    if mantle_mass <= 0.0:
        raise Exception('Something has gone wrong (mantle mass is negative)')

    return mantle_mass


# Run the dummy interior module
def run_dummy_int(
    config: Config, dirs: dict, hf_row: dict, hf_all: pd.DataFrame, interior_o: Interior_t
):
    # Output dictionary
    output = {}
    output['F_int'] = hf_row['F_atm']

    # Core radius from the structure solve, consistent with the boundary
    # backend. config.core_frac is a mass fraction in 'mass' mode, so it must
    # not be reused as a radius fraction here; the structure's R_core already
    # encodes the realized core radius for either mode.
    R_core = hf_row.get('R_core', config.interior_struct.core_frac * hf_row['R_int'])
    core_radius_frac = R_core / hf_row['R_int']

    # Mantle mass from the structure (M_int - M_core), consistent with the
    # boundary backend and with the structure's own mass budget. Fall back to a
    # density times shell-volume estimate only when the structure did not
    # provide both masses.
    if 'M_int' in hf_row and 'M_core' in hf_row:
        output['M_mantle'] = hf_row['M_int'] - hf_row['M_core']
    else:
        output['M_mantle'] = calculate_simple_mantle_mass(
            hf_row['R_int'],
            core_radius_frac,
            config.interior_energetics.dummy.mantle_rho,
        )

    # Physical parameters
    tmp_init = config.planet.tsurf_init  # Initial magma temperature
    area = 4 * np.pi * hf_row['R_int'] ** 2

    # Interior heat capacity [J K-1]
    cp_int = (
        config.interior_energetics.dummy.mantle_cp * output['M_mantle']
        + get_core_heatcap(config, hf_row) * hf_row['M_core']
    )

    # Subtract tidal contribution to the total heat flux.
    #    This heat energy is generated only in the mantle, not in the core.
    tidal_flux = 0.0
    if config.interior_energetics.heat_tidal:
        tidal_flux = interior_o.tides[0] * output['M_mantle'] / area
    output['F_tidal'] = tidal_flux

    # Radiogenic heating constant with time
    output['F_radio'] = (
        config.interior_energetics.dummy.heat_internal * output['M_mantle'] / area
    )

    # Total flux loss
    F_loss = output['F_int'] - output['F_tidal'] - output['F_radio']

    # Rate of surface temperature change (this will be negative) [K/s]
    dTdt = -F_loss * area / cp_int

    # Timestepping
    if interior_o.ic == 1:
        output['T_magma'] = tmp_init
        dt = 0.0
    else:
        # calculate new time-step [years]
        dt = next_step(config, dirs, hf_row, hf_all, 1.0, interior_o=interior_o)

        # limit time-step based on max change to T_magma
        dtmp_max = (
            hf_row['T_magma'] * config.interior_energetics.tmagma_rtol
            + config.interior_energetics.tmagma_atol
        )
        dt = min(dt, abs(dtmp_max / dTdt) / secs_per_year)  # years

        # update T_magma
        output['T_magma'] = hf_row['T_magma'] + dTdt * dt * secs_per_year

    # Store scalars
    output['T_pot'] = float(output['T_magma'])
    output['Phi_global'] = melt_fraction(config, output['T_magma'])
    output['Phi_global_vol'] = output['Phi_global']
    output['M_mantle_liquid'] = output['M_mantle'] * output['Phi_global']
    output['M_mantle_solid'] = output['M_mantle'] - output['M_mantle_liquid']
    output['RF_depth'] = output['Phi_global'] * (1 - core_radius_frac)
    output['boundary_layer_thickness'] = config.atmos_clim.surface_d

    # Store arrays
    interior_o.phi = np.array([output['Phi_global']])
    interior_o.mass = np.array([output['M_mantle']])
    interior_o.visc = np.array([1.0])  # placeholder to be updated elsewhere
    interior_o.density = np.array([config.interior_energetics.dummy.mantle_rho])
    interior_o.temp = np.array([output['T_magma']])
    interior_o.pres = np.array([hf_row['P_surf']])
    interior_o.radius = np.array([R_core, hf_row['R_int']])

    sim_time = hf_row['Time'] + dt
    return sim_time, output
