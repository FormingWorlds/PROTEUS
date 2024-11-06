from __future__ import annotations

import logging
from proteus.utils.constants import const_G, element_list

log = logging.getLogger("fwl."+__name__)

def update_interior_mass(hf_row:dict):
    """
    Update total interior mass.

    M_int = M_core + M_mantle + M_{interior-volatiles}
    """

    # Get mass of all volatiles partitioned into the interior
    M_vols = 0.0
    for e in element_list:
        M_vols += hf_row[e+"_kg_solid"]
        M_vols += hf_row[e+"_kg_liquid"]

    # Add parts together
    hf_row["M_int"] = hf_row["M_mantle"] + hf_row["M_core"] + M_vols


def update_surface_gravity(hf_row:dict):
    """
    Update gravity at atmosphere-interior boundary.
    """

    hf_row["gravity"] = const_G * hf_row["M_int"] / (hf_row["R_int"] ** 2.0)

