from __future__ import annotations

import logging
from proteus.utils.constants import const_G, vol_list, vap_list

log = logging.getLogger("fwl."+__name__)

def update_interior_mass(hf_row:dict):
    """
    Update total interior mass.

    M_int = M_core + M_mantle + M_{interior-volatiles}
    """

    # Get mass of all volatiles partitioned into the INTERIOR
    M_vols = 0.0
    for vol in vol_list:
        M_vols += hf_row[vol+"_kg_solid"]
        M_vols += hf_row[vol+"_kg_liquid"]

    # Add parts together
    hf_row["M_int"] = hf_row["M_mantle"] + hf_row["M_core"] + M_vols


def update_surface_gravity(hf_row:dict):
    """
    Update gravity at atmosphere-interior boundary.
    """

    hf_row["gravity"] = const_G * hf_row["M_int"] / (hf_row["R_int"] ** 2.0)

def simple_core_mass(radius:float, corefrac:float):
    earth_fr = 0.55     # earth core radius fraction
    earth_fm = 0.325    # earth core mass fraction  (https://arxiv.org/pdf/1708.08718.pdf)

    core_rho = (3.0 * earth_fm * M_earth) / (4.0 * np.pi * ( earth_fr * R_earth )**3.0 )  # core density [kg m-3]
    log.debug("Core density = %.2f kg m-3" % core_rho)

    # Calculate core mass
    core_mass = core_rho * 4.0/3.0 * np.pi * (radius * corefrac )**3.0

    return core_mass

def simple_mantle_mass(radius:float, mass:float, corefrac:float)->float:
    '''
    A very simple interior structure model.

    This calculates mantle mass given interior mass, radius, and core fraction. This
    assumes a core density equal to that of Earth's, and that the interior mass is simply
    the sum of mantle and core.

    Duplicated from CALLIOPE.
    '''

    # Get core mass
    core_mass = simple_core_mass(radius, corefrac)

    # Get mantle mass as remainder
    mantle_mass = mass - core_mass

    log.debug("Total mantle mass = %.2e kg" % mantle_mass)
    if (mantle_mass <= 0.0):
        raise Exception("Something has gone wrong (mantle mass is negative)")

    return mantle_mass
