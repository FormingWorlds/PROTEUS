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

    # elements
    for e in element_list:
        for r in res_list:
            # do not overwrite the total inventory, since this will be
            # modified by escape. #57 Commit D: O is now a first-class
            # element and O_kg_total IS persisted (set by
            # `populate_O_kg` after each outgas step), so the
            # pre-plan `(e == 'O')` carve-out has been removed.
            if r != 'total':
                copy_keys.append(f'{e}_kg_{r}')

    return copy_keys
