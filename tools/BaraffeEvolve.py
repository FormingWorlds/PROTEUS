#!/usr/bin/env python3

# Read and write spectra using baraffe evolution tracks

import matplotlib.pyplot as plt
import numpy as np

def main():

    file = 'BHAC15-M0p090000.txt'
    data = np.loadtxt(file,skiprows=1).T

    track = {
        'name':     file,
        'Mstar':    data[0],            # M_sun
        't':        10.0**data[1]*1.e-6,# Myr
        'Teff':     data[2],            # K
        'Lstar':    10.0**data[3],      # L_sun
        'g':        10.0**data[4],      # ms-2 ????
        'Rstar':    data[5]             # R_sun
    }

    fig,ax = plt.subplots(1,1)

    ax.plot(track['t'],track['Lstar'])

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Time [Myr]")

    # fig.savefig('baraffe.pdf')
    plt.show()


if __name__ == '__main__':

    print("Baraffe evolve")
    main()
    print("Done!")

# End of file
