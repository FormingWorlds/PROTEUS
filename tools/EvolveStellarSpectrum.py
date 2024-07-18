#!/usr/bin/env python3

# Python script to test stellar evolution implementation in PROTEUS

from utils.coupler import *
from utils.baraffe import *
from plot.cpl_sflux import plot_sflux
from plot.cpl_sflux_cross import plot_sflux_cross

import mors
import shutil

Myr = 1.0e6

def evolve(cfg_file: str, tf: float, nsamps: int = 500):

    # Read in PROTEUS config file
    COUPLER_options, time_dict = ReadInitFile( cfg_file )

    # Set directories
    dirs = SetDirectories(COUPLER_options)

    print(dirs["output"])

    # Check if output directory exists, otherwise create
    CleanDir( dirs["output"] )
    CleanDir( dirs["output"]+"/data/" )

    # Check which model we are using
    model = int(COUPLER_options['star_model'])
    Mstar = float(COUPLER_options['star_mass'])
    age   = float(COUPLER_options['star_age_modern'])

    # Modern spectrum file 
    modern_path = os.path.join(dirs["fwl"],COUPLER_options["star_spectrum"])
    shutil.copyfile(modern_path, os.path.join(dirs["output"],"-1.sflux"))

    # Prep evolution data
    match model:
        case 1:
            # WITH MORS

            # load modern spectrum 
            modern = mors.spec.Spectrum()
            modern.LoadTSV(modern_path)
            modern.CalcBandFluxes()

            # get best rotation percentile 
            best_pctle, _ = mors.synthesis.FitModernProperties(modern, Mstar, age/1e6)

            # modern properties 
            props = mors.synthesis.GetProperties(Mstar, best_pctle, age/1e6)
        case 2:
            # WITH BARAFFE

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
    print("\t star_spectrum = '%s'" % modern_path)
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
                # WITH MORS 
                synthetic = mors.synthesis.CalcScaledSpectrumFromProps(modern, props, t/1e6)
                synthetic.WriteTSV(dirs['output']+"/data/%d.sflux"%time_dict['star'])
            case 2:
                # WITH BARAFFE
                fl,fls = BaraffeSpectrumCalc(time_dict['star'], StellarFlux_fl,COUPLER_options, track)


    # Plot 
    plot_sflux_cross(dirs["output"], modern_age=age)
    plot_sflux(dirs["output"])


if __name__ == "__main__":
    print("Evolving stellar spectrum.\n")

    if len(sys.argv) == 2:
        cfg = sys.argv[1]
    else:
        cfg = 'init_coupler.cfg' 
    
    # Parameters
    t_final =       9.0e3 * Myr     # Final time for evolution
    samples =       20

    print("NOTE: File convention differs with this script, compared to PROTEUS!")
    print("      Files of the format t.sflux* refer to a STAR AGE of t years.")
    print(" ")

    # Evolve star over time
    evolve(cfg, t_final, samples)

    print("Done!")


# End of file
