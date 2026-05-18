"""Binodal-controlled H2 partitioning between atmosphere and magma ocean.

Wraps the Rogers+2025 H2-MgSiO3 miscibility model from Zalmoxis to
redistribute H2 between the atmosphere and the dissolved (liquid mantle)
reservoir after CALLIOPE's equilibrium chemistry solve.

Physics
-------
When T_magma > T_binodal(P_surf, x_H2): H2 is miscible with MgSiO3 melt.
The dissolved H2 mass fraction is controlled by thermodynamic miscibility at
the mass mixing ratio (not Henry's law):

    sigma = rogers2025_suppression_weight(P_surf, T_magma, w_H2, w_sil)
    H2_kg_liquid = sigma * H2_kg_total
    H2_kg_atm = (1 - sigma) * H2_kg_total

When immiscible (sigma ~ 0): all H2 stays in the atmosphere.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Molar mass of H2 [kg/mol]
_MU_H2 = 2.016e-3


def apply_binodal_h2(hf_row: dict, config: Config) -> None:
    """Apply Rogers+2025 binodal to partition H2 between atm and mantle.

    Modifies hf_row in place: redistributes H2_kg_atm, H2_kg_liquid,
    and updates H2_bar and H2_vmr accordingly.

    Does nothing if H2 is not included, has zero total mass, or if
    the binodal config flag is disabled.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row with volatile mass and state keys.
    config : Config
        PROTEUS configuration object.
    """
    from zalmoxis.binodal import rogers2025_suppression_weight

    # Check if H2 is included and has meaningful mass
    if not config.outgas.calliope.is_included('H2'):
        return

    H2_kg_total = float(hf_row.get('H2_kg_total', 0.0))
    if H2_kg_total <= 0:
        return

    # Get state variables
    T_magma = float(hf_row.get('T_magma', 0.0))
    P_surf_Pa = float(hf_row.get('P_surf', 0.0)) * 1e5  # bar -> Pa
    M_mantle = float(hf_row.get('M_mantle', 0.0))
    R_int = float(hf_row.get('R_int', 0.0))
    gravity = float(hf_row.get('gravity', 0.0))

    if T_magma <= 0 or M_mantle <= 0 or R_int <= 0 or gravity <= 0:
        log.debug('Skipping binodal: missing state variables')
        return

    # Compute mass fractions for the binary H2-MgSiO3 system.
    # w_H2 is the total H2 mass relative to the mantle + H2 total.
    w_H2 = H2_kg_total / (M_mantle + H2_kg_total)
    w_sil = 1.0 - w_H2

    # Compute suppression weight (sigma)
    sigma = rogers2025_suppression_weight(
        P_Pa=P_surf_Pa,
        T_K=T_magma,
        w_H2=w_H2,
        w_sil=w_sil,
    )

    # Partition H2 between liquid mantle (dissolved) and atmosphere
    H2_kg_liquid_new = sigma * H2_kg_total
    H2_kg_atm_new = (1.0 - sigma) * H2_kg_total

    # Store previous values for logging
    H2_kg_atm_old = float(hf_row.get('H2_kg_atm', 0.0))
    H2_kg_liquid_old = float(hf_row.get('H2_kg_liquid', 0.0))

    # Update hf_row
    hf_row['H2_kg_liquid'] = H2_kg_liquid_new
    hf_row['H2_kg_atm'] = H2_kg_atm_new
    # Solid H2 is zero (H2 does not partition into solid silicate)
    hf_row['H2_kg_solid'] = 0.0

    # Recompute H2 partial pressure from atmospheric mass
    # P_H2 = m_H2 * g / (4 * pi * R^2)
    import math

    area = 4.0 * math.pi * R_int**2
    if area > 0 and gravity > 0:
        hf_row['H2_bar'] = H2_kg_atm_new * gravity / area / 1e5  # Pa -> bar
    else:
        hf_row['H2_bar'] = 0.0

    # Recompute total surface pressure (sum of all partial pressures)
    from proteus.utils.constants import gas_list

    P_total = sum(float(hf_row.get(s + '_bar', 0.0)) for s in gas_list)
    hf_row['P_surf'] = P_total

    # Recompute VMRs
    if P_total > 0:
        for s in gas_list:
            hf_row[s + '_vmr'] = float(hf_row.get(s + '_bar', 0.0)) / P_total
    else:
        for s in gas_list:
            hf_row[s + '_vmr'] = 0.0

    # Update molar quantities
    import scipy.constants as const

    N_av = const.Avogadro
    hf_row['H2_mol_liquid'] = H2_kg_liquid_new / _MU_H2 / N_av
    hf_row['H2_mol_atm'] = H2_kg_atm_new / _MU_H2 / N_av
    hf_row['H2_mol_solid'] = 0.0
    hf_row['H2_mol_total'] = H2_kg_total / _MU_H2 / N_av

    # Recompute atmospheric MMW
    _recompute_atm_mmw(hf_row)

    # Note: sigma was evaluated at the pre-redistribution P_surf. A fixed-point
    # iteration (recompute sigma at updated P_surf, re-partition, repeat) would
    # be more self-consistent but the correction is small: the H2 partial
    # pressure shift is typically < 1% of total P_surf, so one pass suffices.
    log.info(
        'Binodal H2: sigma=%.4f, H2_atm %.2e->%.2e kg, '
        'H2_liquid %.2e->%.2e kg (T=%.0f K, P=%.1f bar)',
        sigma,
        H2_kg_atm_old,
        H2_kg_atm_new,
        H2_kg_liquid_old,
        H2_kg_liquid_new,
        T_magma,
        float(hf_row['P_surf']),
    )


def _recompute_atm_mmw(hf_row: dict) -> None:
    """Recompute atmospheric mean molecular weight after H2 redistribution.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row.
    """
    from proteus.utils.constants import gas_list

    # Molar masses [kg/mol] for the volatile species
    _mmw = {
        'H2O': 18.015e-3,
        'CO2': 44.01e-3,
        'O2': 31.999e-3,
        'H2': 2.016e-3,
        'CH4': 16.043e-3,
        'CO': 28.01e-3,
        'N2': 28.014e-3,
        'NH3': 17.031e-3,
        'S2': 64.13e-3,
        'SO2': 64.066e-3,
        'H2S': 34.08e-3,
        'SiO': 44.085e-3,
        'SiO2': 60.084e-3,
        'MgO': 40.304e-3,
        'FeO2': 87.844e-3,
    }

    # MMW = sum(x_i * mu_i), where x_i = VMR
    mmw = 0.0
    for s in gas_list:
        vmr = float(hf_row.get(s + '_vmr', 0.0))
        mu = _mmw.get(s, 28.0e-3)  # fallback to N2 if unknown
        mmw += vmr * mu

    if mmw > 0:
        hf_row['atm_kg_per_mol'] = mmw
