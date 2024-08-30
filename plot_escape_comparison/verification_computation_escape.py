from proteus.utils.constants import *
import pandas as pd
import mors
import numpy as np
from proteus.utils.zephyrus import EL_escape
import matplotlib.pyplot as plt


# Escape parameters
tidal_contribution = 0             # tidal correction factor (0 < K_tide < 1)        [dimensionless]
a                  = a_earth*AU    # planetary semi-major axis                       [m]
e                  = e_earth       # planetary eccentricty                           [dimensionless]
Mp                 = M_earth       # planetary mass                                  [kg]
Ms                 = Ms            # Stellar mass                                    [kg]
epsilon            = 0.50          # efficiency factor                               [dimensionless]
Rp                 = R_earth       # planetary radius                                [m]
Rxuv               = Rp            # XUV planetary radius                            [m]
G                  = const_G       # gravitational constant                          [m3 kg-1 s-2]
Omega_star         = 1.0           # Rotational rate of the star                     [Omega_sun]

# List of start times for different simulations
start_times = [100, 200, 400, 600, 800, 1000]

Fxuv    = []
escape  = []

# Initialize a figure for plotting
plt.figure(figsize=(12, 8))

# Loop through each start time and process the corresponding file
for start_time in start_times:
    print('Start time = ', start_time, 'Myr')
    # Get Fxuv as in RunZEPHYRUS() (= with Mors)
    path_to_output      = '/Users/emmapostolec/Documents/PHD/SCIENCE/CODES/PROTEUS/output/'
    sim_start_time      = f'escape_start_{start_time}Myr/'
    csv_file            = 'runtime_helpfile.csv'
    output_file         = path_to_output + sim_start_time + csv_file
    df                  = pd.read_csv(output_file, delimiter='\t')
    age_star            = df["age_star"]   # [years]

    for time in age_star : 
        # Initialize the Star object and global variables at time[0]
        print(age_star[time])
        star                = mors.Star(Mstar=Ms/Ms, Age=age_star[time]/1e6, Omega=Omega_star)
        age_star_mors       = star.Tracks['Age']
        Fxuv_mors           = ((star.Tracks['Lx'] + star.Tracks['Leuv']) / (4 * np.pi * (a * 1e2)**2)) * ergcm2stoWm2
        Fxuv_interp         = np.interp(age_star[time], age_star_mors * 1e6, Fxuv_mors)
        Fxuv.append(Fxuv_interp)   

        mlr                 = EL_escape(tidal_contribution, a, e, Mp, Ms, epsilon, Rp, Rxuv, Fxuv_interp)
        escape.append(mlr)


        # Plotting the results
        plt.plot(age_star, mlr, lw=2, label=f'Time after star formation = {start_time} Myr')

# Configure the plot
plt.xlabel('Time [years]')
plt.ylabel(r'Total Escape Rate [kg s$^{-1}$]')
plt.yscale('log')
plt.legend(loc='best')
plt.grid(alpha=0.5)
plt.savefig('verification_plot_escape_start_times.png', dpi=180)
plt.show()
