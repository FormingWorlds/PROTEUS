import errno
import os
from datetime import datetime
# from natsort import natsorted #https://pypi.python.org/pypi/natsort
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy import interpolate
import spider_coupler_utils

## Constants:
L_sun           = 3.828e+26             # W, IAU definition
AU              = 1.495978707e+11       # m
R_gas           = 8.31446261815324      # J K−1 mol−1
M_earth         = 5.972E24              # kg
R_core_earth    = 3485000.0             # m
M_core_earth    = 1.94E24               # kg

# https://stackoverflow.com/questions/14115254/creating-a-folder-with-timestamp
def make_output_dir():
    output_dir = os.getcwd()+"/output/"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+"/"
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..

    return output_dir

# https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str


def InterpolateStellarLuminosity(star_mass, time_current, time_offset, mean_distance):

    luminosity_df = pd.read_csv("luminosity_tracks/Lum_m"+str(star_mass)+".txt")

    time_current    = time_current/1e+6         # Myr
    ages            = luminosity_df["age"]*1e+3 # Myr
    luminosities    = luminosity_df["lum"]      # L_sol

    # Interpolate luminosity for current time
    interpolate_luminosity  = interpolate.interp1d(ages, luminosities)
    interpolated_luminosity = interpolate_luminosity([time_current+time_offset])*L_sun

    stellar_toa_heating = interpolated_luminosity / ( 4. * np.pi * (mean_distance*AU)**2. )
    
    return stellar_toa_heating[0], interpolated_luminosity/L_sun

def AtmosphericHeight(T_profile, P_profile, m_planet, r_planet):

    z_profile       = np.zeros(len(P_profile))
    P_s             = np.max(P_profile)
    grav_s          = spider_coupler_utils.gravity( m_planet, r_planet )

    # print(np.max(T_profile), np.min(T_profile))

    for n in range(0, len(z_profile)):

        T_mean_below    = np.mean(T_profile[n:])
        P_z             = P_profile[n]
        # print(T_mean_below, P_z)

        z_profile[n] = - R_gas * T_mean_below * np.log(P_z/P_s) / grav_s

    return z_profile

