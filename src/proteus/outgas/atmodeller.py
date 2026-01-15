# Function and classes used to run CALLIOPE
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteus.config import Config


from proteus.outgas.common import expected_keys
from proteus.utils.constants import C_solar, N_solar, S_solar, element_list, vol_list
from proteus.utils.helper import UpdateStatusfile

log = logging.getLogger("fwl."+__name__)


from proteus.utils.constants import element_list, vol_list

def calc_surface_pressures(dirs: dict, config: Config, hf_row: dict):
    """
    Fake Atmodeller: reads the same keys as Calliope
    and writes dummy outputs to hf_row
    """

    log.info("Running Atmodeller (dummy version)...")

    # DEBUG: print inputs
    log.info("INPUT hf_row:")
    for k in hf_row:
        if k in ["Time", "T_magma", "M_mantle"] + vol_list:
            log.info(f"  {k}: {hf_row[k]}")

    # --- Dummy calculations ---
    # Just copy previous pressures or set dummy values
    for s in vol_list:
        if s != "O2":
            hf_row[s + "_bar"] = hf_row.get(s + "_bar", 0.1) + 0.1
            hf_row[s + "_kg_atm"] = 1e18  # dummy mass
            hf_row[s + "_vmr"] = 0.5      # dummy VMR

    # Total surface pressure (sum of dummy bars)
    hf_row["P_surf"] = sum(hf_row[s + "_bar"] for s in vol_list if s != "O2")
    hf_row["atm_kg_per_mol"] = 0.029  # dummy mean molar mass

    # DEBUG: print outputs
    log.info("OUTPUT hf_row:")
    for k in hf_row:
        if k in ["P_surf", "atm_kg_per_mol"] + [s + "_bar" for s in vol_list]:
            log.info(f"  {k}: {hf_row[k]}")
