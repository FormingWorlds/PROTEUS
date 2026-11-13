"""Per-element escape debit of an atmosphere that no longer exchanges with the mantle.

Once the mantle has solidified no outgassing solve follows the escape step, so
the escaping elements have to be taken out of the atmospheric gases directly.
Fractionating escape removes the elements in proportions that differ from the
gases that hold them (hydrogen leaves water faster than oxygen does), so the
gases cannot all shrink by one factor.

Each gas shrinks by the smallest retained fraction among its elements, which
leaves no element above its target. What remains of an element then forms its
own molecule: O2, H2, N2, S2. Carbon has no gas of its own in the chemistry,
so leftover carbon combines with oxygen into CO and then with hydrogen into
CH4, taking the partner from the leftover first and then from free O2 or H2.
Carbon that still has no partner, as when a methane atmosphere loses its
hydrogen, condenses as graphite.

Every element then holds exactly its mass less the requested loss, between the
gases and the graphite. Rock vapours keep their masses, and rock-forming
elements are not debited: the escaping share the dispatcher gives them is not
removed.

The escape step plans the debit and the outgassing step applies it; the plan
travels on the helpfile row under ``FROZEN_DEBIT_KEY`` with the time of its step.
"""

from __future__ import annotations

import logging

import numpy as np

from proteus.outgas.common import (
    VOLATILE_ELEMENT_STOICH,
    element_masses_from_species,
    set_element_ratios,
)
from proteus.utils.constants import (
    element_list,
    element_mmw,
    gas_list,
    noble_gases,
    vol_element_list,
    vol_gas_list,
)
from proteus.utils.helper import eval_gas_mmw

log = logging.getLogger('fwl.' + __name__)

# Element counts of each gas this debit handles: the volatiles and the noble gases.
STOICH: dict[str, dict[str, int]] = {
    **VOLATILE_ELEMENT_STOICH,
    **{g: {g: 1} for g in noble_gases},
}

# The gas each element forms on its own, and its atoms per molecule.
OWN_GAS: dict[str, tuple[str, int]] = {
    'H': ('H2', 2),
    'O': ('O2', 2),
    'N': ('N2', 2),
    'S': ('S2', 2),
    **{g: (g, 1) for g in noble_gases},
}

# Relative tolerance of the element balance check.
CLOSURE_RTOL = 1e-9

# Transient row key of a planned debit, written by escape and read by outgassing.
FROZEN_DEBIT_KEY = '_esc_frozen_atm_kg'


def _element_moles(mol: dict) -> dict:
    """Moles of each element held by a set of gases [mol]."""
    out: dict[str, float] = {}
    for gas, n in mol.items():
        for e, k in STOICH[gas].items():
            out[e] = out.get(e, 0.0) + k * n
    return out


def _place_carbon(mol: dict, left: dict) -> float:
    """Put leftover carbon into CO, then CH4; return the carbon left over [mol].

    The partner atoms come from the leftover first and then from the free O2
    or H2 of the atmosphere.
    """
    for gas, partner, k in (('CO', 'O', 1), ('CH4', 'H', 4)):
        own, per = OWN_GAS[partner]
        free = left.get(partner, 0.0) + per * mol[own]
        take = min(left['C'], free / k)
        if take <= 0.0:
            continue
        mol[gas] += take
        left['C'] -= take
        need = k * take
        from_left = min(left.get(partner, 0.0), need)
        left[partner] = left.get(partner, 0.0) - from_left
        mol[own] = max(0.0, mol[own] - (need - from_left) / per)
    stranded, left['C'] = left['C'], 0.0
    return stranded


def plan_debit(species_kg: dict, losses_kg: dict) -> tuple[dict, dict, float]:
    """New gas masses after an element-wise loss, the loss applied, and the graphite.

    Parameters
    ----------
    species_kg : dict
        Mass of each gas in the atmosphere [kg]. Gases outside ``STOICH`` are
        ignored and absent from the result.
    losses_kg : dict
        Mass of each element to remove [kg]. An element absent from the
        atmosphere, or a negative loss, removes nothing; a loss larger than
        the element holds removes all of it.

    Returns
    -------
    new_kg : dict
        Mass of each gas in ``STOICH`` after the loss [kg].
    applied_kg : dict
        Mass of each element removed [kg]: the requested loss, capped at what
        the atmosphere holds.
    graphite_kg : float
        Carbon that condensed as graphite for want of a partner [kg]. It stays
        in the planet's carbon total.

    Raises
    ------
    RuntimeError
        If the element balance fails to close.
    """
    mol = {g: max(0.0, float(species_kg.get(g, 0.0))) / eval_gas_mmw(g) for g in STOICH}
    held = _element_moles(mol)
    target = {}
    for e, n in held.items():
        loss = max(0.0, float(losses_kg.get(e, 0.0))) / element_mmw[e]
        target[e] = max(0.0, n - loss)
    kept = {e: (target[e] / n if n > 0.0 else 1.0) for e, n in held.items()}

    new = {g: n * min(kept[e] for e in STOICH[g]) for g, n in mol.items()}
    placed = _element_moles(new)
    left = {e: max(0.0, target[e] - placed.get(e, 0.0)) for e in held}

    stranded = _place_carbon(new, left) if left.get('C', 0.0) > 0.0 else 0.0
    for e, amount in left.items():
        if amount > 0.0 and e in OWN_GAS:
            gas, k = OWN_GAS[e]
            new[gas] = new.get(gas, 0.0) + amount / k

    final = _element_moles(new)
    for e, n in held.items():
        expected = target[e] - (stranded if e == 'C' else 0.0)
        if abs(final.get(e, 0.0) - expected) > CLOSURE_RTOL * max(n, 1e-300):
            raise RuntimeError(f'element balance of the atmospheric debit fails for {e}')
    graphite_kg = stranded * element_mmw['C']
    if graphite_kg > 0.0:
        log.info(
            'Carbon left without a partner gas condenses as graphite: %.3e kg', graphite_kg
        )
    new_kg = {g: n * eval_gas_mmw(g) for g, n in new.items()}
    applied_kg = {e: (n - target[e]) * element_mmw[e] for e, n in held.items()}
    return new_kg, applied_kg, graphite_kg


def clear_frozen_debit(hf_row: dict) -> None:
    """Drop a debit planned on an earlier step, so it cannot be applied late."""
    hf_row.pop('_esc_frozen_atm_kg', None)


def plan_frozen_debit(hf_row: dict, target: dict) -> dict:
    """Plan the debit of a frozen atmosphere and return the element totals after it.

    The per-element losses the targets imply are taken out of the atmospheric
    gases by :func:`plan_debit`. The new gas masses travel on the row, with the
    time of this step, for :func:`apply_frozen_debit` to apply.

    Parameters
    ----------
    hf_row : dict
        Helpfile row of this iteration.
    target : dict
        Element whole-planet inventories after the requested loss [kg].

    Returns
    -------
    totals : dict
        Inventories after the loss the gases can give up [kg]. Elements the
        debit does not place (the rock-forming ones) keep their inventory.
    """
    totals = {e: float(hf_row.get(f'{e}_kg_total', 0.0)) for e in element_list}
    losses = {
        e: max(0.0, totals[e] - target[e])
        for e in element_list
        if e in target and np.isfinite(totals[e]) and np.isfinite(target[e])
    }
    species = {g: float(hf_row.get(f'{g}_kg_atm', 0.0)) for g in gas_list if g in STOICH}
    new_kg, applied, graphite = plan_debit(species, losses)
    out = dict(target)
    for e in element_list:
        if e in losses:
            out[e] = totals[e] - applied.get(e, 0.0)
    hf_row['_esc_frozen_atm_kg'] = {
        'Time': float(hf_row.get('Time', 0.0)),
        'kg': new_kg,
        'graphite_kg': graphite,
    }
    return out


def apply_frozen_debit(hf_row: dict) -> None:
    """Set the atmosphere of a frozen mantle to the gas masses this step's escape left.

    Mixing ratios, partial pressures and the mean molecular weight follow the
    new composition, and the volatile pressure scales with the volatile column
    mass. Carbon left without a partner gas is added to the condensed graphite
    and to ``esc_graphite_kg``, which escape cannot reach. Rock vapours keep
    their masses. A debit planned on another step is discarded with a warning.
    """
    debit = hf_row.pop('_esc_frozen_atm_kg', None)
    if debit is None:
        log.info('Crystallized mantle: volatile exchange frozen, reservoirs preserved')
        return
    if debit['Time'] != float(hf_row.get('Time', 0.0)):
        log.warning('Discarded an escape debit planned at t = %.6e yr', debit['Time'])
        return
    m_old = float(hf_row.get('M_atm', 0.0))
    kg = {g: float(debit['kg'].get(g, hf_row.get(f'{g}_kg_atm', 0.0))) for g in gas_list}
    mol = {g: kg[g] / eval_gas_mmw(g) for g in gas_list}
    m_atm, n_atm = sum(kg.values()), sum(mol.values())
    m_vol_old = float(hf_row.get('M_vol_atm', 0.0))
    m_vol = sum(kg[s] for s in vol_gas_list)
    hf_row['P_vol'] = float(hf_row.get('P_vol', 0.0)) * (
        m_vol / m_vol_old if m_vol_old > 0.0 else 0.0
    )
    hf_row['P_surf'] = hf_row['P_vol'] + float(hf_row.get('P_vap', 0.0))
    hf_row['M_atm'] = m_atm
    hf_row['M_vol_atm'] = m_vol
    hf_row['atm_kg_per_mol'] = m_atm / n_atm if n_atm > 0.0 else 0.0
    for g in gas_list:
        x = mol[g] / n_atm if n_atm > 0.0 else 0.0
        hf_row[f'{g}_kg_atm'] = kg[g]
        hf_row[f'{g}_mol_atm'] = mol[g]
        hf_row[f'{g}_vmr'] = x
        hf_row[f'{g}_bar'] = x * hf_row['P_surf']
    masses = element_masses_from_species({g: kg[g] for g in vol_gas_list})
    for e in vol_element_list:
        hf_row[f'{e}_kg_atm'] = masses.get(e, 0.0)
    graphite = float(debit.get('graphite_kg', 0.0))
    if graphite > 0.0:
        hf_row['C_kg_condensed'] = float(hf_row.get('C_kg_condensed', 0.0)) + graphite
        hf_row['C_cd_kg'] = float(hf_row.get('C_cd_kg', 0.0)) + graphite
        hf_row['esc_graphite_kg'] = float(hf_row.get('esc_graphite_kg', 0.0)) + graphite
    set_element_ratios(hf_row)
    log.info(
        'Crystallized mantle: escape removed %.3e kg from the atmosphere element by element',
        m_old - m_atm,
    )
