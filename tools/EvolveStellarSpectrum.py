#!/usr/bin/env python3

# Python script to test stellar evolution implementation in PROTEUS

from utils.coupler import *
from utils.stellar import *

Myr = 1.0e6

def run(tf: float):
    # Read in PROTEUS config file
    COUPLER_options, time_dict = ReadInitFile( "init_coupler.cfg" )

    # Set directories
    dirs = SetDirectories(COUPLER_options)

    # Check if output directory exists, otherwise create
    CleanDir( dirs["output"] )

    # Check which model we are using
    model = COUPLER_options['star_model']

    # Store copy of modern spectrum in memory (1 AU)
    StellarFlux_wl, StellarFlux_fl = ModernSpectrumLoad(dirs, COUPLER_options)

    # Prep evolution data
    match model:
        case 1:
            # Calculate band-integrated fluxes for modern stellar spectrum (1 AU)
            COUPLER_options = MorsCalculateFband(dirs, COUPLER_options)
        case 2:
            # Load track
            track = BaraffeLoadtrack(COUPLER_options)

    # Parameters
    ti = time_dict['star']  # Start time, yr
    dt = COUPLER_options['sflux_dt_update']

    # Print info
    print("Parameters:")
    print("\t star_model = %d" % COUPLER_options['star_model'])
    print("\t star_spectrum = '%s'" % COUPLER_options['star_spectrum'])
    print("\t ti = %1.3e Myr" % (time_dict['star'] * 1.e-6))
    print("\t tf = %1.3e Myr" % (tf * 1.e-6))
    print("\t dt = %1.3e Myr" % (dt * 1.e-6))

    # Calculate historical spectrum (1 AU) over time, saving it to files
    print("Running evolution code...")
    t_prev = 0
    for t in np.logspace(np.log10(ti),np.log10(tf),1000):

        if (t-t_prev) >= dt:

            t_prev = t

            print("Age = %1.2e yr, Progress = %3.1f%%" % (t,t/tf*100.0))

            time_dict["star"] = t
            time_dict["planet"] = t

            match model:
                case 1:
                    fl,fls = MorsSpectrumCalc(time_dict['star'], StellarFlux_wl, StellarFlux_fl,COUPLER_options)
                case 2:
                    fl,fls = BaraffeSpectrumCalc(time_dict['star'], StellarFlux_fl,COUPLER_options, track)

            SpectrumWrite(time_dict,StellarFlux_wl,fl,fls,dirs)


if __name__ == "__main__":
    print("Evolve stellar spectrum with Mors.\n")
    
    # Parameters
    t_final =       200.0 * Myr     # Final time for evolution

    print("NOTE: File convention differs with this script, compared to PROTEUS!")
    print("      Files of the format t.sflux* refer to a STAR AGE of t years.")

    # Evolve star over time
    run(t_final)

    print("Done!")


# End of file