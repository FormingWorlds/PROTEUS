# Generic stellar evolution wrapper
from __future__ import annotations

import logging

from proteus.utils.constants import AU

log = logging.getLogger("fwl."+__name__)

def spectrum_scale_to_toa(fl_arr, sep:float):
    '''
    Scale stellar fluxes from 1 AU to top of atmosphere
    '''
    return np.array(fl_arr) * ( (AU / sep)**2 )

def spectrum_write(wl_arr, fl_arr, hf_row:dict, output_dir:str):
    '''
    Write stellar spectrum to file.
    '''

    # Header information
    header = (
        "# WL(nm)\t Flux(ergs/cm**2/s/nm)   Stellar flux at t_star = %.2e yr"
        % hf_row["age_star"]
    )

    # Write to TSV file
    np.savetxt(
        os.path.join(output_dir, "data", "%d.sflux" % hf_row["Time"]),
        np.array([wl_arr, fl_arr]).T,
        header=header,
        comments="",
        fmt="%.8e",
        delimiter="\t",
    )
