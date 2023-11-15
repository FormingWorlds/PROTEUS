#!/usr/bin/env python3

# Python script to test stellar evolution implementation in PROTEUS

from utils.coupler import *
from utils.stellar_mors import *
from utils.stellar_baraffe import *
from utils.stellar_common import *

import shutil

Myr = 1.0e6

def evolve(cfg_file: str, tf: float, nsamps: int = 500):

    # Read in PROTEUS config file
    COUPLER_options, time_dict = ReadInitFile( cfg_file )

    # Set directories
    dirs = SetDirectories(COUPLER_options)

    # Check if output directory exists, otherwise create
    CleanDir( dirs["output"] )
    CleanDir( dirs["output"]+"/data/" )

    # Check which model we are using
    model = COUPLER_options['star_model']

    # Store copy of modern spectrum in memory (1 AU)
    StellarFlux_wl, StellarFlux_fl = ModernSpectrumLoad(dirs, COUPLER_options)

    # Solve for UV-PL band edge
    MorsSolveUV(dirs,COUPLER_options,StellarFlux_wl,StellarFlux_fl)

    # Prep evolution data
    match model:
        case 1:
            # Calculate band-integrated fluxes for modern stellar spectrum (1 AU)
            COUPLER_options = MorsCalculateFband(dirs, COUPLER_options)
        case 2:
            # Load track
            track = BaraffeLoadtrack(COUPLER_options)
        case _:
            raise Exception("Unsupported stellar model")

    # Parameters
    ti = time_dict['star']  # Start time, yr
    dt = COUPLER_options['sspec_dt_update']

    # Print info
    print("")
    print("Parameters:")
    print("\t star_model = %d" % COUPLER_options['star_model'])
    print("\t star_spectrum = '%s'" % COUPLER_options['star_spectrum'])
    print("\t ti = %1.3e Myr" % (time_dict['star'] * 1.e-6))
    print("\t tf = %1.3e Myr" % (tf * 1.e-6))
    print("\t dt = %1.3e Myr" % (dt * 1.e-6))

    # Calculate historical spectrum (1 AU) over time, saving it to files
    print("Running evolution code...")
    t_arr = np.logspace(np.log10(ti),np.log10(tf),nsamps)
    for i,t in enumerate(t_arr):

        print("Age = %1.2e yr, Progress = %3.1f%%" % (t,(i+1)/len(t_arr)*100.0))

        time_dict["star"] = t
        time_dict["planet"] = t

        match model:
            case 1:
                fl,fls = MorsSpectrumCalc(time_dict['star'], StellarFlux_wl, StellarFlux_fl,COUPLER_options)
            case 2:
                fl,fls = BaraffeSpectrumCalc(time_dict['star'], StellarFlux_fl,COUPLER_options, track)

        SpectrumWrite(time_dict,StellarFlux_wl,fl,fls,dirs['output']+"/data/")


if __name__ == "__main__":
    print("Evolving stellar spectrum.\n")

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg' 
    
    # Parameters
    t_final =       3.0e3 * Myr     # Final time for evolution
    samples =       20

    print("NOTE: File convention differs with this script, compared to PROTEUS!")
    print("      Files of the format t.sflux* refer to a STAR AGE of t years.")
    print(" ")

    # Evolve star over time
    evolve(cfg, t_final, samples)

    print("Done!")


# End of file
