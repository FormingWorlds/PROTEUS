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

h2_mol_mass     = 0.00201588            # kg/mol
h2o_mol_mass    = 0.01801528            # kg/mol
co2_mol_mass    = 0.04401               # kg/mol
ch4_mol_mass    = 0.01604               # kg/mol
co_mol_mass     = 0.02801               # kg/mol
n2_mol_mass     = 0.028014              # kg/mol
o2_mol_mass     = 0.031999              # kg/mol
he_mol_mass     = 0.0040026             # kg/mol


def CalcMolRatios(h2_kg, h2o_kg, co2_kg, ch4_kg, co_kg, n2_kg, o2_kg, he_kg):

    h2_mol  = h2_kg/h2_mol_mass     # mol
    h2o_mol = h2o_kg/h2o_mol_mass   # mol
    co2_mol = co2_kg/co2_mol_mass   # mol
    ch4_mol = ch4_kg/ch4_mol_mass   # mol
    co_mol  = co_kg/co_mol_mass     # mol
    n2_mol  = n2_kg/_mol_mass       # mol
    o2_mol  = o2_kg/_mol_mass       # mol
    he_mol  = he_kg/he_mol_mass     # mol

    # Radiative gases + He
    total_mol = h2_mol + h2o_mol + co2_mol + ch4_mol + co_mol + n2_mol + o2_mol + he_mol

    h2_ratio  = h2_mol / total_mol
    h2o_ratio = h2o_mol / total_mol 
    co2_ratio = co2_mol / total_mol 
    ch4_ratio = ch4_mol / total_mol 
    co_ratio  = co_mol / total_mol 
    n2_ratio  = n2_mol / total_mol 
    o2_ratio  = o2_mol / total_mol 
    he_ratio  = he_mol / total_mol 

    return h2_ratio, h2o_ratio, co2_ratio, ch4_ratio, co_ratio, n2_ratio, o2_ratio, he_ratio


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

