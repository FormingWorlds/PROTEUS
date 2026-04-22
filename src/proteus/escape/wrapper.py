# Functions used to handle escape
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.escape.common import calc_unfract_fluxes
from proteus.utils.constants import M_sun, element_list, secs_per_year

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def run_escape(
    config: Config,
    hf_row: dict,
    dirs: dict | None = None,
    dt: float = 0.0,
    stellar_track=None,
) -> None:
    """Run Escape submodule.

    Generic function to run escape calculation using ZEPHYRUS or dummy.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        dirs: dict
            Dictionary of directories
        dt : float
            Time interval over which escape is occuring [yr]
    """
    dirs = dirs or {}

    if not config.escape.module:
        # Keep a minimal, dependency-free disabled path for unit tests.
        hf_row['esc_rate_total'] = 0.0
        for e in element_list:
            hf_row[f'esc_rate_{e}'] = 0.0
        log.info(f'Escape is disabled, bulk rate = {hf_row["esc_rate_total"]:.2e} kg s-1')
        return

    # Snapshot the initial bulk volatile inventory on the first escape call.
    # This baseline is used by `outgas.wrapper.check_desiccation` to verify
    # that any subsequent collapse of `*_kg_total` is actually accounted for
    # by cumulative escape, rather than by an upstream AGNI/outgas failure
    # zeroing the atmosphere as a side effect (CHILI sweep R7/R21 cascade).
    m_init_prev = hf_row.get('M_vol_initial', None)
    try:
        m_init_prev_f = float(m_init_prev) if m_init_prev is not None else 0.0
    except (TypeError, ValueError):
        m_init_prev_f = 0.0
    if not np.isfinite(m_init_prev_f) or m_init_prev_f <= 0.0:
        m_vol_baseline = sum(
            float(hf_row.get(f'{e}_kg_total', 0.0))
            for e in element_list
            if e != 'O'
        )
        hf_row['M_vol_initial'] = m_vol_baseline
        # Reset the cumulative escape counter alongside the baseline so the
        # ratio (lost vs escaped) starts from a consistent zero.
        hf_row['esc_kg_cumulative'] = 0.0

    if config.escape.module == 'dummy':
        run_dummy(config, hf_row)

    elif config.escape.module == 'zephyrus':
        run_zephyrus(config, hf_row, stellar_track)

    elif config.escape.module == 'boreas':
        from proteus.escape.boreas import run_boreas

        run_boreas(config, hf_row, dirs)

    else:
        raise ValueError(f'Invalid escape model: {config.escape.module}')

    log.info(f'Bulk escape rate = {hf_row["esc_rate_total"]:.2e} kg s-1')

    log.info('Elemental escape fluxes:')
    for e in element_list:
        esc_e = float(hf_row.get(f'esc_rate_{e}', 0.0))
        if esc_e > 0:
            log.info('    %2s = %.2e kg s-1' % (e, esc_e))

    # Accumulate cumulative escaped mass [kg] for the desiccation gate. This
    # is the integral of `esc_rate_total * dt` over all escape calls and is
    # persisted to the helpfile so it survives resume.
    esc_step_kg = float(hf_row.get('esc_rate_total', 0.0)) * secs_per_year * float(dt)
    if np.isfinite(esc_step_kg) and esc_step_kg > 0.0:
        hf_row['esc_kg_cumulative'] = (
            float(hf_row.get('esc_kg_cumulative', 0.0)) + esc_step_kg
        )

    # calculate new elemental inventories from loss over duration `dt`
    solvevol_target = calc_new_elements(
        hf_row,
        dt,
        config.escape.reservoir,
        min_thresh=config.outgas.mass_thresh,
    )

    # store new elemental inventories
    for e, mass in solvevol_target.items():
        hf_row[f'{e}_kg_total'] = mass


def atm_escape_fluxes_species_kg(
    hf_row: dict, dt: float,
) -> dict[str, float]:
    """
    Decompose the bulk ZEPHYRUS mass-loss rate into per-species kg
    outflow over `dt` years, weighted by species **mass fraction**
    (#57 Commit C.1; round-5 review fixed the unit error in C).

    Valid for hydrodynamic bulk escape where atmospheric mixing is
    efficient (ZEPHYRUS's target regime). Under that assumption, the
    escaping flux composition matches the bulk atmosphere by mass,
    NOT by mole fraction. For a 50/50 H₂+CO₂ VMR atmosphere, the
    correct mass decomposition gives ~96% CO₂ by mass (since
    MW_CO₂ ≈ 22 × MW_H₂); the previous VMR-only weighting gave 50%
    CO₂ by mass and would have driven R_atm debits wrong by the same
    factor.

    Fractionating escape (Jeans / diffusion-limited) is a follow-on
    correction tracked as its own issue.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row. Consumes `esc_rate_total` [kg/s] and
        `{species}_vmr`.
    dt : float
        Step duration in years.

    Returns
    -------
    dict[species, kg_escaped]
        Zero for species with VMR = 0 or missing. Keys cover
        `proteus.utils.constants.gas_list`.
    """
    from proteus.utils.constants import element_mmw, gas_list

    rate_kg_s = float(hf_row.get('esc_rate_total', 0.0))
    dt_s = float(dt) * secs_per_year
    if rate_kg_s <= 0.0 or dt_s <= 0.0 or not np.isfinite(rate_kg_s * dt_s):
        return {s: 0.0 for s in gas_list}

    m_total_lost = rate_kg_s * dt_s

    vmrs = {
        s: max(0.0, float(hf_row.get(f'{s}_vmr', 0.0)))
        for s in gas_list
    }
    # Species molar masses [kg/mol].
    mws = {s: _species_mass(s, element_mmw) for s in gas_list}

    # Mass-fraction weights: w_s = VMR_s · MW_s / Σ(VMR_i · MW_i).
    weighted = {s: vmrs[s] * mws[s] for s in gas_list}
    total_weight = sum(
        v for v in weighted.values() if np.isfinite(v) and v > 0.0
    )
    if total_weight <= 0.0:
        return {s: 0.0 for s in gas_list}

    return {s: m_total_lost * (weighted[s] / total_weight) for s in gas_list}


def _species_mass(species: str, element_mmw: dict) -> float:
    """Parse a molecular formula (e.g. 'H2O', 'CO2') → kg/mol."""
    import re

    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    total = 0.0
    for elem, count_str in pattern.findall(species):
        if not elem:
            continue
        count = int(count_str) if count_str else 1
        if elem not in element_mmw:
            return 0.0  # unknown element → drop species from the budget
        total += element_mmw[elem] * count
    return total


def run_dummy(config: Config, hf_row: dict):
    """Run dummy escape model.

    Uses a fixed mass loss rate and does not fractionate.

    Parameters
    ----------
        config : Config
            Configuration options for the escape module
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    """

    # Set sound speed to zero
    hf_row['cs_xuv'] = 0.0

    # Set Pxuv to Psurf (if available)
    if 'P_surf' in hf_row:
        hf_row['p_xuv'] = hf_row['P_surf']
    if 'R_int' in hf_row:
        hf_row['R_xuv'] = hf_row['R_int']

    # Set bulk escape rate based on value from user
    hf_row['esc_rate_total'] = float(
        getattr(getattr(config.escape, 'dummy', None), 'rate', 0.0)
    )

    # Always unfractionating (best-effort: unit tests may not populate all keys)
    try:
        reservoir = getattr(config.escape, 'reservoir', None)
        if isinstance(reservoir, str):
            calc_unfract_fluxes(
                hf_row,
                reservoir=reservoir,
                min_thresh=config.outgas.mass_thresh,
            )
    except (KeyError, ValueError, TypeError):
        for e in element_list:
            hf_row[f'esc_rate_{e}'] = 0.0


def run_zephyrus(config: Config, hf_row: dict, stellar_track=None) -> float:
    """Run ZEPHYRUS escape model.

    Calculates the bulk mass loss rate of all elements.

    Parameters
    ----------
        config : dict
            Dictionary of configuration options
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    """

    from zephyrus.escape import EL_escape

    log.info('Running EL escape (ZEPHYRUS) ...')

    # Compute energy-limited escape
    mlr = EL_escape(
        config.escape.zephyrus.tidal,  # tidal contribution (True/False)
        hf_row['semimajorax'],  # planetary semi-major axis [m]
        hf_row['eccentricity'],  # eccentricity
        hf_row['M_planet'],  # planetary mass [kg]
        config.star.mass * M_sun,  # stellar mass [kg] (config is in M_sun)
        config.escape.zephyrus.efficiency,  # efficiency factor
        hf_row['R_int'],  # planetary radius [m]
        hf_row['R_xuv'],  # XUV optically thick planetary radius [m]
        hf_row['F_xuv'],  # [W m-2]
        scaling=3,
    )

    hf_row['esc_rate_total'] = mlr
    hf_row['p_xuv'] = config.escape.zephyrus.Pxuv
    hf_row['R_xuv'] = 0.0  # to be calc'd by atmosphere module

    # Always unfractionating - escaping in bulk
    try:
        reservoir = getattr(config.escape, 'reservoir', None)
        if isinstance(reservoir, str):
            calc_unfract_fluxes(
                hf_row,
                reservoir=reservoir,
                min_thresh=config.outgas.mass_thresh,
            )
    except (KeyError, ValueError, TypeError):
        for e in element_list:
            hf_row[f'esc_rate_{e}'] = 0.0

    return float(mlr)


def calc_new_elements(
    hf_row: dict,
    dt: float,
    reservoir: str,
    min_thresh: float = 1e10,
):
    """Calculate new elemental inventory based on escape rate.

    Parameters
    ----------
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        dt : float
            Time-step length [years]
        min_thresh: float
            Minimum threshold for element mass [kg]. Inventories below this are set to zero.

    Returns
    -------
        tgt : dict
            Volatile element whole-planet inventories [kg]
    """
    # which reservoir?
    match reservoir:
        case 'bulk':
            key = '_kg_total'
        case 'outgas':
            key = '_kg_atm'
        case 'pxuv':
            raise ValueError('Fractionation at p_xuv is not yet supported')
        case _:
            raise ValueError(f"Invalid escape reservoir '{reservoir}'")

    # calculate mass of elements in the reservoir
    res: dict[str, float] = {}
    for e in element_list:
        if e == 'O':  # Oxygen is set by fO2, so we skip it here (const_fO2)
            continue
        res[e] = float(hf_row.get(f'{e}{key}', 0.0))
    M_vols = float(sum(res.values()))

    # check if we just desiccated the planet...
    if M_vols < min_thresh:
        log.debug('    Total mass of volatiles below threshold in escape calculation')
        return res

    # compute mass ratios in escaping reservoir
    emr = {e: (res[e] / M_vols if M_vols > 0 else 0.0) for e in res}

    # total escaped mass over dt [kg]
    esc_mass = float(hf_row.get('esc_rate_total', 0.0)) * secs_per_year * float(dt)

    # compute new TOTAL inventories
    tgt: dict[str, float] = {}
    for e in res:
        lost = esc_mass * emr[e]
        old_total = float(hf_row.get(f'{e}_kg_total', 0.0))
        new_total = old_total - lost
        if new_total < min_thresh:
            new_total = 0.0
        tgt[e] = max(0.0, new_total)

    return tgt
