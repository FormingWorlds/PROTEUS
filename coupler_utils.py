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

### Constants ###

# Astronomical constants
L_sun           = 3.828e+26             # W, IAU definition
AU              = 1.495978707e+11       # m
R_gas           = 8.31446261815324      # J K−1 mol−1
M_earth         = 5.972E24              # kg
R_core_earth    = 3485000.0             # m
M_core_earth    = 1.94E24               # kg
mol             = 6.02214076e+23        # mol definition

# Elements
h_mol_mass      = 0.001008              # kg mol−1
c_mol_mass      = 0.012011              # kg mol−1
o_mol_mass      = 0.015999              # kg mol−1
n_mol_mass      = 0.014007              # kg mol−1
he_mol_mass     = 0.0040026             # kg mol−1
ar_mol_mass     = 0.039948              # kg mol−1
ne_mol_mass     = 0.020180              # kg mol−1
kr_mol_mass     = 0.083798              # kg mol−1
xe_mol_mass     = 0.131293              # kg mol−1

# Compounds
h2o_mol_mass    = 0.01801528            # kg mol−1
co2_mol_mass    = 0.04401               # kg mol−1
h2_mol_mass     = 0.00201588            # kg mol−1
ch4_mol_mass    = 0.01604               # kg mol−1
co_mol_mass     = 0.02801               # kg mol−1
n2_mol_mass     = 0.028014              # kg mol−1
o2_mol_mass     = 0.031999              # kg mol−1
so2_mol_mass    = 0.064066              # kg mol−1
h2s_mol_mass    = 0.0341                # kg mol−1

def Calc_XH_Ratios(mantle_mass, h2o_ppm, co2_ppm, h2_ppm, ch4_ppm, co_ppm, n2_ppm, o2_ppm, he_ppm):

    h2o_mol = float(h2o_ppm)*1e6*mantle_mass/h2o_mol_mass   # mol
    co2_mol = float(co2_ppm)*1e6*mantle_mass/co2_mol_mass   # mol
    h2_mol  = float(h2_ppm)*1e6*mantle_mass/h2_mol_mass     # mol
    ch4_mol = float(ch4_ppm)*1e6*mantle_mass/ch4_mol_mass   # mol
    co_mol  = float(co_ppm)*1e6*mantle_mass/co_mol_mass     # mol
    n2_mol  = float(n2_ppm)*1e6*mantle_mass/n2_mol_mass     # mol
    o2_mol  = float(o2_ppm)*1e6*mantle_mass/o2_mol_mass     # mol
    he_mol  = float(he_ppm)*1e6*mantle_mass/he_mol_mass     # mol

    # Radiative species + He
    total_mol = h2o_mol + co2_mol + h2_mol + ch4_mol + co_mol + n2_mol + o2_mol + he_mol

    h_mol_total  = h2o_mol*2. + h2_mol*2. + ch4_mol*4. 
    o_mol_total  = h2o_mol*1. + co2_mol*2. + co_mol*1. + o2_mol*2.
    c_mol_total  = co2_mol*1. + ch4_mol*1. + co_mol*1.
    n_mol_total  = n2_mol*2.
    s_mol_total  = 0.
    he_mol_total = he_mol*1.

    O_H_mol_ratio = o_mol_total / h_mol_total       # mol/mol
    C_H_mol_ratio = c_mol_total / h_mol_total       # mol/mol
    N_H_mol_ratio  = n_mol_total / h_mol_total      # mol/mol
    S_H_mol_ratio = s_mol_total / h_mol_total       # mol/mol 
    He_H_mol_ratio  = he_mol_total / h_mol_total    # mol/mol 

    # Counter VULCAN bug with zero abundances
    if N_H_mol_ratio == 0.: 
        N_H_mol_ratio = 1e-99
    if S_H_mol_ratio == 0.: 
        S_H_mol_ratio = 1e-99

    return O_H_mol_ratio, C_H_mol_ratio, N_H_mol_ratio, S_H_mol_ratio, He_H_mol_ratio

def CalcMassMolRatios(h2o_kg, co2_kg, h2_kg, ch4_kg, co_kg, n2_kg, o2_kg, he_kg):

    h2o_mol = h2o_kg/h2o_mol_mass   # mol
    co2_mol = co2_kg/co2_mol_mass   # mol
    h2_mol  = h2_kg/h2_mol_mass     # mol
    ch4_mol = ch4_kg/ch4_mol_mass   # mol
    co_mol  = co_kg/co_mol_mass     # mol
    n2_mol  = n2_kg/n2_mol_mass     # mol
    o2_mol  = o2_kg/o2_mol_mass     # mol
    he_mol  = he_kg/he_mol_mass     # mol

    # Radiative species + He
    total_mol = h2o_mol + co2_mol + h2_mol + ch4_mol + co_mol + n2_mol + o2_mol + he_mol

    h2o_mass_mol_ratio = h2o_mol / total_mol  # mol/mol
    co2_mass_mol_ratio = co2_mol / total_mol  # mol/mol
    h2_mass_mol_ratio  = h2_mol / total_mol   # mol/mol
    ch4_mass_mol_ratio = ch4_mol / total_mol  # mol/mol 
    co_mass_mol_ratio  = co_mol / total_mol   # mol/mol 
    n2_mass_mol_ratio  = n2_mol / total_mol   # mol/mol 
    o2_mass_mol_ratio  = o2_mol / total_mol   # mol/mol 
    he_mass_mol_ratio  = he_mol / total_mol   # mol/mol 

    return h2o_mass_mol_ratio, co2_mass_mol_ratio, h2_mass_mol_ratio, ch4_mass_mol_ratio, co_mass_mol_ratio, n2_mass_mol_ratio, o2_mass_mol_ratio, he_mass_mol_ratio

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

def PrintCurrentState(time_current, surfaceT_current, h2o_kg, co2_kg, h2_kg, ch4_kg, co_kg, n2_kg, o2_kg, he_kg, h2o_mass_mol_ratio, co2_mass_mol_ratio, h2_mass_mol_ratio, ch4_mass_mol_ratio, co_mass_mol_ratio, n2_mass_mol_ratio, o2_mass_mol_ratio, he_mass_mol_ratio, p_s, heat_flux, ic_filename, stellar_toa_heating, solar_lum):

    # Print final statement
    print("--------------------------------------------------------")
    print("      ==> RUNTIME INFO <==")
    print(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print("Time [Myr]:", str(float(time_current)/1e6))
    print ("T_s [K]:", surfaceT_current)
    print ("H2O:", h2o_kg, " [kg]", h2o_mass_mol_ratio, " [mass mol ratio]")
    print ("CO2:", co2_kg, " [kg]", co2_mass_mol_ratio, " [mass mol ratio]")
    print ("H2:", h2_kg, " [kg]", h2_mass_mol_ratio, " [mass mol ratio]")
    print ("CH4:", ch4_kg, " [kg]", ch4_mass_mol_ratio, " [mass mol ratio]")
    print ("CO:", co_kg, " [kg]", co_mass_mol_ratio, " [mass mol ratio]")
    print ("N2:", n2_kg, " [kg]", n2_mass_mol_ratio, " [mass mol ratio]")
    print ("O2:", o2_kg, " [kg]", o2_mass_mol_ratio, " [mass mol ratio]")
    print ("p_s:", p_s*1e-3, " [bar]")
    print ("TOA heating:", stellar_toa_heating)
    print ("L_star:", solar_lum)
    print ("Total heat flux [W/m^2]:", heat_flux)
    print ("Last file name:", ic_filename)
    print("--------------------------------------------------------")

def write_surface_quantitites(output_dir, file_name):

    # logger.info( 'building atmosphere' )

    sim_times = spider_coupler_utils.get_all_output_times(output_dir)  # yr

    keys_t = ( ('atmosphere','mass_liquid'),
               ('atmosphere','mass_solid'),
               ('atmosphere','mass_mantle'),
               ('atmosphere','mass_core'),
               ('atmosphere','CO2','liquid_kg'),
               ('atmosphere','CO2','solid_kg'),
               ('atmosphere','CO2','initial_kg'),
               ('atmosphere','CO2','atmosphere_kg'),
               ('atmosphere','CO2','atmosphere_bar'),
               ('atmosphere','H2O','liquid_kg'),
               ('atmosphere','H2O','solid_kg'),
               ('atmosphere','H2O','initial_kg'),
               ('atmosphere','H2O','atmosphere_kg'),
               ('atmosphere','H2O','atmosphere_bar'),
               ('atmosphere','temperature_surface'),
               ('atmosphere','emissivity'),
               ('rheological_front_phi','phi_global'),
               ('atmosphere','Fatm'))

    data_a = spider_coupler_utils.get_dict_surface_values_for_times( keys_t, sim_times )

    # Mass properties
    mass_liquid_a = data_a[0,:]
    mass_solid_a = data_a[1,:]
    mass_mantle_a = data_a[2,:]
    mass_mantle = mass_mantle_a[0]          # time independent
    mass_core_a = data_a[3,:]
    mass_core = mass_core_a[0]              # time independent
    planet_mass_a = mass_core_a + mass_mantle_a
    planet_mass = mass_core + mass_mantle   # time independent

    # compute total mass (kg) in each reservoir
    CO2_liquid_kg_a = data_a[4,:]
    CO2_solid_kg_a = data_a[5,:]
    CO2_total_kg_a = data_a[6,:]
    CO2_total_kg = CO2_total_kg_a[0]        # time-independent
    CO2_atmos_kg_a = data_a[7,:]
    CO2_atmos_a = data_a[8,:]
    CO2_escape_kg_a = CO2_total_kg - CO2_liquid_kg_a - CO2_solid_kg_a - CO2_atmos_kg_a

    H2O_liquid_kg_a = data_a[9,:]
    H2O_solid_kg_a = data_a[10,:]
    H2O_total_kg_a = data_a[11,:]
    H2O_total_kg = H2O_total_kg_a[0]        # time-independent
    H2O_atmos_kg_a = data_a[12,:]
    H2O_atmos_a = data_a[13,:]
    H2O_escape_kg_a = H2O_total_kg - H2O_liquid_kg_a - H2O_solid_kg_a - H2O_atmos_kg_a

    temperature_surface_a = data_a[14,:]
    emissivity_a = data_a[15,:]             # internally computed emissivity
    phi_global = data_a[16,:]               # global melt fraction
    Fatm = data_a[17,:]

    # output surface + atmosphere quantities
    out_a = np.column_stack( (sim_times, temperature_surface_a, mass_core_a, mass_mantle_a, H2O_atmos_kg_a, CO2_atmos_kg_a ) )
    np.savetxt( output_dir+file_name, out_a )

