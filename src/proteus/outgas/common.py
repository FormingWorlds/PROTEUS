# Shared code for outgassing wrapper
from __future__ import annotations

from proteus.utils.constants import element_list, gas_list


def expected_keys():
    copy_keys = ['P_surf', 'M_atm', 'atm_kg_per_mol']

    # reservoirs
    res_list = ('atm', 'liquid', 'solid', 'total')

    # gases
    for s in gas_list:
        copy_keys.append(f'{s}_bar')
        copy_keys.append(f'{s}_vmr')
        for r in res_list:
            copy_keys.append(f'{s}_kg_{r}')
            copy_keys.append(f'{s}_mol_{r}')

    # elements. The `_kg_total` slot is owned by escape (which debits
    # the running budget after the wrapper writes) for every element
    # EXCEPT oxygen. For O, the chemistry solver's output partitions
    # atm+liquid+solid into a fresh total each iteration; the calliope
    # wrapper restores hf_row['O_kg_total'] to the authoritative input
    # immediately after this copy under Path C (fO2_source =
    # "from_O_budget"), so the escape debit chain is preserved across
    # iterations. Under user_constant the solver's O_kg_total IS the
    # authoritative value, so the copy is the correct write.
    for e in element_list:
        for r in res_list:
            if (r != 'total') or (e == 'O'):
                copy_keys.append(f'{e}_kg_{r}')

    # element mass ratios in atmosphere (must mirror the unordered-pair
    # registration in coupler.GetHelpfileKeys so run_desiccated zeros
    # them on desiccation instead of leaving stale ratios in hf_row).
    for e1 in element_list:
        for e2 in element_list:
            if (e1 == e2) or (f'{e1}/{e2}_atm' in copy_keys):
                continue
            copy_keys.append(f'{e2}/{e1}_atm')

    return copy_keys
