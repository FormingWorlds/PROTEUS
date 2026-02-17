# Shared code for outgassing wrapper
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config

from proteus.utils.constants import element_list, vap_list, vol_list

# from proteus.outgas.wrapper import get_gaslist


def expected_keys(config: Config):
    copy_keys = ['P_surf', 'M_atm', 'atm_kg_per_mol']

    # reservoirs
    res_list = ('atm', 'liquid', 'solid', 'total')

    if config.outgas.silicates:
        gas_list = vol_list + config.outgas.vaplist
    else:
        gas_list = vol_list + vap_list

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
            # do not overwrite total inventory, since this will be modified by escape
            # except oxygen, since we assume it's set by redox buffer (const_fO2)
            if (r != 'total') or (e == 'O'):
                copy_keys.append(f'{e}_kg_{r}')

    return copy_keys
