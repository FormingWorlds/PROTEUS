#!/usr/bin/env python3

# Plots stellar flux from `output/`

import glob 
import numpy as np
import matplotlib.pyplot as plt
from utils.modules_coupler import dirs
from utils.utils_coupler import natural_sort

def plot_sflux(output_dir):

    # Find and sort files
    files = glob.glob(output_dir+"/*.sflux")
    files = natural_sort(files)

    # Arrays for storing data over time
    time_t = []
    wave_t = []
    flux_t = []
    for f in files:
        # Load data
        X = np.loadtxt(f,skiprows=2,delimiter='\t').T
        
        # Parse data
        time = int(f.split('/')[-1].split('.')[0])
        wave = X[0]
        flux = X[1]

        # Save data
        time_t.append(time)
        wave_t.append(wave)
        flux_t.append(flux)

    # Plot data
    N = len(time_t)

    fig,ax = plt.subplots(1,1)

    # ax.set_yscale("log")
    ax.set_ylabel("Stellar flux [erg s-1 cm-2 nm-1]")

    # ax.set_xscale("log")
    ax.set_xlabel("Wavelength [nm]")

    for i in range(N):
        t = time_t[i] * 1.e-6
        ax.plot(wave_t[i],flux_t[i],label="%1.3e"%t,alpha=0.6)

    ax.set_xlim([0,1100.0])

    ax.legend()

    fig.savefig(output_dir+"/plot_sflux.pdf")


# Run directly
if __name__ == '__main__':

    print("Plotting stellar flux over time...")

    plot_sflux(dirs['output'])

    print("Done!")


# End of file
