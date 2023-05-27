#!/usr/bin/env python3

# Python script to test stellar evolution implementation in PROTEUS

from utils.coupler import *
from utils.mors import *

Myr = 1.0e6

def run(tf: float):
    # Read in COUPLER input file
    COUPLER_options, time_dict = ReadInitFile( dirs, "init_coupler.cfg" )

    # Calculate band-integrated fluxes for modern stellar spectrum (1 AU)
    COUPLER_options = ModernSpectrumFband(dirs, COUPLER_options)
    
    # Store copy of modern spectrum in memory (1 AU)
    StellarFlux_wl, StellarFlux_fl = ModernSpectrumLoad(dirs, COUPLER_options)

    # Parameters
    ti = time_dict['star']  # Start time, yr
    dt = COUPLER_options['sflux_dt_update']

    # Print info
    print("Parameters:")
    print("\t star_spectrum = '%s'" % COUPLER_options['star_spectrum'])
    print("\t ti = %1.3e Myr" % (time_dict['star'] * 1.e-6))
    print("\t tf = %1.3e Myr" % (tf * 1.e-6))
    print("\t dt = %1.3e  yr" % dt)

    # Calculate historical spectrum (1 AU) over time, saving it to files
    print("Running evolution code...")
    t = ti
    while t < tf:
        time_dict["star"] += dt
        time_dict["planet"] += dt

        HistoricalSpectrumWrite(time_dict, StellarFlux_wl, StellarFlux_fl,dirs,COUPLER_options)

        t += dt

if __name__ == "__main__":
    print("Evolve stellar spectrum with Mors.\n")
    
    t_final = 200.0 * Myr

    # Check if output directory exists, otherwise create
    if not os.path.exists(dirs["output"]):
        os.makedirs(dirs["output"])
        print("Create output directory:", dirs["output"])

    # If it does exist, empty the folder
    else:
        print("Empty output directory:", dirs["output"])
        CleanOutputDir( dirs["output"] )

    # Evolve star over time
    run(t_final)

    print("Done!")


# End of file
