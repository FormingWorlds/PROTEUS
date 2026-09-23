# Shared code for outgassing wrapper
from __future__ import annotations

from proteus.utils.constants import element_list, gas_list, noble_gases, vol_list

# Atoms per formula unit for every volatile gas species. Used to split a
# per-species mass into the per-element masses the whole-planet accounting is
# kept in. Shared so the outgassing backends and the trapping step cannot drift
# apart on the stoichiometry.
VOLATILE_ELEMENT_STOICH: dict[str, dict[str, int]] = {
    'H2O': {'H': 2, 'O': 1},
    'CO2': {'C': 1, 'O': 2},
    'O2': {'O': 2},
    'H2': {'H': 2},
    'CH4': {'C': 1, 'H': 4},
    'CO': {'C': 1, 'O': 1},
    'N2': {'N': 2},
    'NH3': {'N': 1, 'H': 3},
    'S2': {'S': 2},
    'SO2': {'S': 1, 'O': 2},
    'H2S': {'H': 2, 'S': 1},
}

assert set(VOLATILE_ELEMENT_STOICH) == set(vol_list), (
    'VOLATILE_ELEMENT_STOICH must cover exactly constants.vol_list; '
    f'missing={set(vol_list) - set(VOLATILE_ELEMENT_STOICH)} '
    f'extra={set(VOLATILE_ELEMENT_STOICH) - set(vol_list)}'
)


def element_masses_from_species(species_kg: dict[str, float]) -> dict[str, float]:
    """Split per-species masses [kg] into per-element masses [kg].

    Each species mass is apportioned across its constituent elements by their
    share of the formula mass, so the element masses sum back to the species
    mass. A species absent from ``VOLATILE_ELEMENT_STOICH`` (a noble gas, a rock
    vapour) is carried through under its own symbol, which is how those species
    already appear in ``element_list``.

    Parameters
    ----------
    species_kg : dict
        Mass per gas species [kg].

    Returns
    -------
    dict
        Mass per element [kg], covering only the elements the input touches.
    """
    from proteus.utils.constants import element_mmw

    out: dict[str, float] = {}
    for species, mass in species_kg.items():
        if not mass:
            continue
        stoich = VOLATILE_ELEMENT_STOICH.get(species)
        if stoich is None:
            # Monatomic species (noble gases, rock vapours) are their own element.
            out[species] = out.get(species, 0.0) + float(mass)
            continue
        formula_mmw = sum(n * element_mmw[el] for el, n in stoich.items())
        if formula_mmw <= 0.0:
            continue
        for el, n in stoich.items():
            out[el] = out.get(el, 0.0) + float(mass) * (n * element_mmw[el] / formula_mmw)
    return out


def expected_keys():
    copy_keys = [
        'P_surf',
        'P_vol',
        'P_vap',
        'M_atm',
        'M_vol_atm',
        'M_vaps',
        'atm_kg_per_mol',
        'fO2_shift_IW_derived',
        'O_res',
    ]

    # reservoirs
    res_list = ('atm', 'liquid', 'solid', 'total')

    # gases
    for s in gas_list:
        copy_keys.append(f'{s}_bar')
        copy_keys.append(f'{s}_vmr')
        for r in res_list:
            # A noble gas is also an element; its _kg_total is escape-owned
            # (like every non-O element total), so it is not copied from the
            # backend here. Its atm/liquid/solid reservoirs are.
            if not (r == 'total' and s in noble_gases):
                copy_keys.append(f'{s}_kg_{r}')
            copy_keys.append(f'{s}_mol_{r}')

    # elements. The `_kg_total` slot is owned by escape (which debits
    # the running budget after the wrapper writes) for every element
    # EXCEPT oxygen. For O, the chemistry solver's output partitions
    # atm+liquid+solid into a fresh total each iteration; the calliope
    # wrapper restores hf_row['O_kg_total'] to the authoritative input
    # immediately after this copy when fO2_source =
    # "from_O_budget", so the escape debit chain is preserved across
    # iterations. Under user_constant the solver's O_kg_total IS the
    # authoritative value, so the copy is the correct write.
    # Some vap_list species are monatomic (e.g. 'Si', 'Na', 'Fe') and share
    # their string with the matching element_list symbol, so the gas loop
    # above already appended f'{e}_kg_{r}' for those elements. Skip keys
    # already present instead of appending a duplicate for the same
    # hf_row slot.
    for e in element_list:
        if e in noble_gases:
            continue  # noble reservoirs are handled in the gas-species loop
        for r in res_list:
            if (r != 'total') or (e == 'O'):
                key = f'{e}_kg_{r}'
                if key not in copy_keys:
                    copy_keys.append(key)

    # element mass ratios in atmosphere (must mirror the unordered-pair
    # registration in coupler.GetHelpfileKeys so run_desiccated zeros
    # them on desiccation instead of leaving stale ratios in hf_row).
    for e1 in element_list:
        for e2 in element_list:
            if (e1 == e2) or (f'{e1}/{e2}_atm' in copy_keys):
                continue
            copy_keys.append(f'{e2}/{e1}_atm')

    return copy_keys
