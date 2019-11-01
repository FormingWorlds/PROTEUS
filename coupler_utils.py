import errno
import sys, os, shutil
from datetime import datetime
# from natsort import natsorted #https://pypi.python.org/pypi/natsort
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from scipy import interpolate
import spider_coupler_utils as su
import pandas as pd
import json
import subprocess
import fileinput # https://kaijento.github.io/2017/05/28/python-replacing-lines-in-file/
import SocRadConv
import SocRadModel
from natsort import natsorted #https://pypi.python.org/pypi/natsort
import glob
import math
import plot_interior
import plot_global
import plot_stacked
import plot_atmosphere
from atmosphere_column import atmos

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
H_mol_mass      = 0.001008              # kg mol−1
C_mol_mass      = 0.012011              # kg mol−1
O_mol_mass      = 0.015999              # kg mol−1
N_mol_mass      = 0.014007              # kg mol−1
S_mol_mass      = 0.03206               # kg mol−1
He_mol_mass     = 0.0040026             # kg mol−1
Ar_mol_mass     = 0.039948              # kg mol−1
Ne_mol_mass     = 0.020180              # kg mol−1
Kr_mol_mass     = 0.083798              # kg mol−1
Xe_mol_mass     = 0.131293              # kg mol−1

# Volatile molar masses
H2O_mol_mass    = 0.01801528            # kg mol−1
CO2_mol_mass    = 0.04401               # kg mol−1
H2_mol_mass     = 0.00201588            # kg mol−1
CH4_mol_mass    = 0.01604               # kg mol−1
CO_mol_mass     = 0.02801               # kg mol−1
N2_mol_mass     = 0.028014              # kg mol−1
O2_mol_mass     = 0.031999              # kg mol−1
SO2_mol_mass    = 0.064066              # kg mol−1
H2S_mol_mass    = 0.0341                # kg mol−1

molar_mass      = {
        "H2O" : 0.01801528,            # kg mol−1
        "CO2" : 0.04401,               # kg mol−1
        "H2"  : 0.00201588,            # kg mol−1
        "CH4" : 0.01604,               # kg mol−1
        "CO"  : 0.02801,               # kg mol−1
        "N2"  : 0.028014,              # kg mol−1
        "O2"  : 0.031999,              # kg mol−1
        "SO2" : 0.064066,              # kg mol−1
        "H2S" : 0.0341,                # kg mol−1 
    }

volatile_species = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2", "S", "He" ]


# Henry's law coefficients
# Add to dataframe + save to disk
volatile_distribution_coefficients = {          # X_henry -> ppm/Pa
    'H2O_henry':     8.086e-01,                
    'H2O_henry_pow': 1.709e+00,                
    # 'CO2_henry':     1.937e-09,            
    # 'CO2_henry_pow': 7.140e-01,                
    # 'H2O_henry':     6.800e-02,                   # Lebrun+13
    # 'H2O_henry_pow': 1.4285714285714286,          # Lebrun+13
    'CO2_henry':     4.4E-6,                      # Lebrun+13
    'CO2_henry_pow': 1.0,                         # Lebrun+13
    'H2_henry':      0.00000257, 
    'H2_henry_pow':  1.0, 
    'CH4_henry':     0.00000010, 
    'CH4_henry_pow': 1.0, 
    'CO_henry':      0.00000016, 
    'CO_henry_pow':  1.0, 
    'N2_henry':      0.00007000, 
    'N2_henry_pow':  1.8,
    'N2_henry_reduced':      74.15775114, 
    'N2_henry_pow_reduced':  4.58195381, 
    'O2_henry':      0.001E-9, 
    'O2_henry_pow':  1.0, 
    'S_henry':       0.005, 
    'S_henry_pow':   1.0, 
    'He_henry':      0.001E-9, 
    'He_henry_pow':  1.0,
    'H2O_kdist':     1.0E-4,                    # distribution coefficients
    'H2O_kabs':      0.01,                      # absorption (m^2/kg)
    'CO2_kdist':     5.0E-4, 
    'CO2_kabs':      0.05,    
    'H2_kdist':      0.0E-0,  
    'H2_kabs':       0.00,     
    'N2_kdist':      0.0E-0,  
    'N2_kabs':       0.00,     
    'CH4_kdist':     0.0E-0, 
    'CH4_kabs':      0.00,    
    'CO_kdist':      0.0E-0,  
    'CO_kabs':       0.00,     
    'O2_kdist':      0.0E-0,  
    'O2_kabs':       0.00,     
    'S_kdist':       0.0E-0,   
    'S_kabs':        0.00,      
    'He_kdist':      0.0E-0,  
    'He_kabs':       0.00
    }


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
    grav_s          = su.gravity( m_planet, r_planet )

    # print(np.max(T_profile), np.min(T_profile))

    for n in range(0, len(z_profile)):

        T_mean_below    = np.mean(T_profile[n:])
        P_z             = P_profile[n]
        # print(T_mean_below, P_z)

        z_profile[n] = - R_gas * T_mean_below * np.log(P_z/P_s) / grav_s

    return z_profile

def PrintCurrentState(time_current, runtime_helpfile, p_s, SPIDER_options, stellar_toa_heating, solar_lum, loop_counter, output_dir):

    # Print final statement
    print("---------------------------------------------------------")
    print("==> RUNTIME INFO <==")
    print(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print("LOOP:", loop_counter)
    print("Time [Myr]:", str(float(time_current)/1e6))
    print("T_s [K]:", runtime_helpfile.iloc[-1]["T_surf"])
    print("Helpfile properties:")
    print(runtime_helpfile.tail(1))
    print("p_s:", p_s*1e-3, " [bar]")
    print("TOA heating:", stellar_toa_heating)
    print("L_star:", solar_lum)
    print("Total heat flux [W/m^2]:", SPIDER_options["heat_flux"])
    print("Last file name:", SPIDER_options["ic_interior_filename"])
    print("---------------------------------------------------------")

def UpdateHelpfile(loop_counter, output_dir, vulcan_dir, file_name, runtime_helpfile, input_flag, atm_chemistry, SPIDER_options):

    # If runtime_helpfle not existent, create it + write to disk
    if not os.path.isfile(output_dir+file_name):
        runtime_helpfile = pd.DataFrame(columns=['Time', 'Input', 'T_surf', 'Heat_flux', 'P_surf', 'M_atm', 'Phi_global', 'M_core', 'M_mantle', 'M_mantle_liquid', 'M_mantle_solid', 'H_mol', 'O/H', 'C/H', 'N/H', 'S/H', 'He/H'])
        runtime_helpfile.to_csv( output_dir+file_name, index=False, sep=" ") 
        time_current = 0
        #, 'H2O_atm_bar', 'CO2_atm_bar', 'H2_atm_bar', 'CH4_atm_bar', 'CO_atm_bar', 'N2_atm_bar', 'O2_atm_bar', 'S_atm_bar', 'He_atm_bar'run

        # Save SPIDER options file
        SPIDER_options_save = pd.DataFrame(SPIDER_options, index=[0])
        SPIDER_options_save.to_csv( output_dir+"spider_options.csv", index=False, sep=" ")

    # Data dict
    runtime_helpfile_new = {}

    ### Read in last SPIDER input as a baseline
    sim_times = su.get_all_output_times(output_dir)  # yr
    sim_time  = sim_times[-1]

    # SPIDER keys from JSON file that are read in
    keys_t = ( ('atmosphere','mass_liquid'),
               ('atmosphere','mass_solid'),
               ('atmosphere','mass_mantle'),
               ('atmosphere','mass_core'),

               ('atmosphere','temperature_surface'),
               ('rheological_front_phi','phi_global'),
               ('atmosphere','Fatm'),
               ('atmosphere','pressure_surface'),
               )

    data_a = su.get_dict_surface_values_for_specific_time( keys_t, sim_time )

    # Fill the new dict
    runtime_helpfile_new["Time"]  = sim_time
    runtime_helpfile_new["Input"] = input_flag

    # Mass properties
    runtime_helpfile_new["M_mantle_liquid"] = data_a[0,:]
    runtime_helpfile_new["M_mantle_solid"]  = data_a[1,:]
    runtime_helpfile_new["M_mantle"]        = data_a[2,:]        
    runtime_helpfile_new["M_core"]          = data_a[3,:]         

    # Surface properties
    runtime_helpfile_new["T_surf"]          = data_a[4,:]
    runtime_helpfile_new["Phi_global"]      = data_a[5,:]  # global melt fraction
    runtime_helpfile_new["Heat_flux"]       = data_a[6,:]
    runtime_helpfile_new["P_surf"]          = data_a[7,:]  # total surface pressure

    # Total atmospheric mass
    runtime_helpfile_new["M_atm"] = 0

    # For all considered volatiles
    for vol in volatile_species:
        if SPIDER_options[vol+"_initial_total_abundance"] > 0.:
            keys_t = ( 
                        # ('atmosphere',vol,'liquid_kg'),
                        # ('atmosphere',vol,'solid_kg'),
                        ('atmosphere',vol,'atmosphere_kg'),
                        ('atmosphere',vol,'atmosphere_bar') 
                     )
            
            data_a = su.get_dict_surface_values_for_specific_time( keys_t, sim_time )

            # runtime_helpfile_new[vol+"_liquid_kg"] = data_a[0,:]
            # runtime_helpfile_new[vol+"_solid_kg"]  = data_a[1,:]
            runtime_helpfile_new[vol+"_atm_kg"]  = data_a[0,:]
            runtime_helpfile_new[vol+"_atm_bar"] = data_a[1,:]

            runtime_helpfile_new["M_atm"] += runtime_helpfile_new[vol+"_atm_kg"]

    # # SPIDER keys from JSON file that are read in
    # keys_t = ( ('atmosphere','mass_liquid'),
    #            ('atmosphere','mass_solid'),
    #            ('atmosphere','mass_mantle'),
    #            ('atmosphere','mass_core'),

    #            ('atmosphere','temperature_surface'),
    #            ('atmosphere','emissivity'),
    #            ('rheological_front_phi','phi_global'),
    #            ('atmosphere','Fatm'),
    #            ('atmosphere','pressure_surface'),

    #            ('atmosphere','H2O','liquid_kg'),
    #            ('atmosphere','H2O','solid_kg'),
    #            ('atmosphere','H2O','initial_kg'),
    #            ('atmosphere','H2O','atmosphere_kg'),
    #            ('atmosphere','H2O','atmosphere_bar'),

    #            ('atmosphere','CO2','liquid_kg'),
    #            ('atmosphere','CO2','solid_kg'),
    #            ('atmosphere','CO2','initial_kg'),
    #            ('atmosphere','CO2','atmosphere_kg'),
    #            ('atmosphere','CO2','atmosphere_bar'),

    #            ('atmosphere','H2','liquid_kg'),
    #            ('atmosphere','H2','solid_kg'),
    #            ('atmosphere','H2','initial_kg'),
    #            ('atmosphere','H2','atmosphere_kg'),
    #            ('atmosphere','H2','atmosphere_bar'),

    #            ('atmosphere','CH4','liquid_kg'),
    #            ('atmosphere','CH4','solid_kg'),
    #            ('atmosphere','CH4','initial_kg'),
    #            ('atmosphere','CH4','atmosphere_kg'),
    #            ('atmosphere','CH4','atmosphere_bar'),

    #            ('atmosphere','CO','liquid_kg'),
    #            ('atmosphere','CO','solid_kg'),
    #            ('atmosphere','CO','initial_kg'),
    #            ('atmosphere','CO','atmosphere_kg'),
    #            ('atmosphere','CO','atmosphere_bar'),

    #            ('atmosphere','N2','liquid_kg'),
    #            ('atmosphere','N2','solid_kg'),
    #            ('atmosphere','N2','initial_kg'),
    #            ('atmosphere','N2','atmosphere_kg'),
    #            ('atmosphere','N2','atmosphere_bar'),

    #            ('atmosphere','O2','liquid_kg'),
    #            ('atmosphere','O2','solid_kg'),
    #            ('atmosphere','O2','initial_kg'),
    #            ('atmosphere','O2','atmosphere_kg'),
    #            ('atmosphere','O2','atmosphere_bar'),

    #            ('atmosphere','S','liquid_kg'),
    #            ('atmosphere','S','solid_kg'),
    #            ('atmosphere','S','initial_kg'),
    #            ('atmosphere','S','atmosphere_kg'),
    #            ('atmosphere','S','atmosphere_bar'),

    #            ('atmosphere','He','liquid_kg'),
    #            ('atmosphere','He','solid_kg'),
    #            ('atmosphere','He','initial_kg'),
    #            ('atmosphere','He','atmosphere_kg'),
    #            ('atmosphere','He','atmosphere_bar'),

    #            )

    # # data_a = su.get_dict_surface_values_for_times( keys_t, sim_times )
    # data_a = su.get_dict_surface_values_for_specific_time( keys_t, sim_time )

    

    # # Volatile properties
    # H2O_liquid_kg_a     = data_a[4,:]
    # H2O_solid_kg_a      = data_a[5,:]
    # H2O_init_kg_a       = data_a[6,:]
    # H2O_atmos_kg_a      = data_a[7,:]
    # H2O_atmos_pressure  = data_a[8,:]
    # H2O_total_kg        = H2O_liquid_kg_a + H2O_solid_kg_a + H2O_atmos_kg_a
    # H2O_escape_kg_a     = H2O_total_kg - H2O_liquid_kg_a - H2O_solid_kg_a - H2O_atmos_kg_a

    # CO2_liquid_kg_a     = data_a[9,:]
    # CO2_solid_kg_a      = data_a[10,:]
    # CO2_init_kg_a       = data_a[11,:]
    # CO2_atmos_kg_a      = data_a[12,:]
    # CO2_atmos_pressure  = data_a[13,:]
    # CO2_total_kg        = CO2_liquid_kg_a + CO2_solid_kg_a + CO2_atmos_kg_a
    # CO2_escape_kg_a     = CO2_total_kg - CO2_liquid_kg_a - CO2_solid_kg_a - CO2_atmos_kg_a

    # H2_liquid_kg_a      = data_a[14,:]
    # H2_solid_kg_a       = data_a[15,:]
    # H2_init_kg_a        = data_a[16,:]
    # H2_atmos_kg_a       = data_a[17,:]
    # H2_atmos_pressure   = data_a[18,:]
    # H2_total_kg         = H2_liquid_kg_a + H2_solid_kg_a + H2_atmos_kg_a
    # H2_escape_kg_a      = H2_total_kg - H2_liquid_kg_a - H2_solid_kg_a - H2_atmos_kg_a

    # CH4_liquid_kg_a     = data_a[19,:]
    # CH4_solid_kg_a      = data_a[20,:]
    # CH4_init_kg_a       = data_a[21,:]
    # CH4_atmos_kg_a      = data_a[22,:]
    # CH4_atmos_pressure  = data_a[23,:]
    # CH4_total_kg        = CH4_liquid_kg_a + CH4_solid_kg_a + CH4_atmos_kg_a
    # CH4_escape_kg_a     = CH4_total_kg - CH4_liquid_kg_a - CH4_solid_kg_a - CH4_atmos_kg_a

    # CO_liquid_kg_a      = data_a[24,:]
    # CO_solid_kg_a       = data_a[25,:]
    # CO_init_kg_a        = data_a[26,:]
    # CO_atmos_kg_a       = data_a[27,:]
    # CO_atmos_pressure   = data_a[28,:]
    # CO_total_kg         = CO_liquid_kg_a + CO_solid_kg_a + CO_atmos_kg_a
    # CO_escape_kg_a      = CO_total_kg - CO_liquid_kg_a - CO_solid_kg_a - CO_atmos_kg_a

    # N2_liquid_kg_a      = data_a[29,:]
    # N2_solid_kg_a       = data_a[30,:]
    # N2_init_kg_a        = data_a[31,:]
    # N2_atmos_kg_a       = data_a[32,:]
    # N2_atmos_pressure   = data_a[33,:]
    # N2_total_kg         = N2_liquid_kg_a + N2_solid_kg_a + N2_atmos_kg_a
    # N2_escape_kg_a      = N2_total_kg - N2_liquid_kg_a - N2_solid_kg_a - N2_atmos_kg_a

    # O2_liquid_kg_a      = data_a[34,:]
    # O2_solid_kg_a       = data_a[35,:]
    # O2_init_kg_a        = data_a[36,:]
    # O2_atmos_kg_a       = data_a[37,:]
    # O2_atmos_pressure   = data_a[38,:]
    # O2_total_kg         = O2_liquid_kg_a + O2_solid_kg_a + O2_atmos_kg_a
    # O2_escape_kg_a      = O2_total_kg - O2_liquid_kg_a - O2_solid_kg_a - O2_atmos_kg_a

    # S_liquid_kg_a       = data_a[39,:]
    # S_solid_kg_a        = data_a[40,:]
    # S_init_kg_a         = data_a[41,:]
    # S_atmos_kg_a        = data_a[42,:]
    # S_atmos_pressure    = data_a[43,:]
    # S_total_kg          = S_liquid_kg_a + S_solid_kg_a + S_atmos_kg_a
    # S_escape_kg_a       = S_total_kg - S_liquid_kg_a - S_solid_kg_a - S_atmos_kg_a

    # He_liquid_kg_a      = data_a[44,:]
    # He_solid_kg_a       = data_a[45,:]
    # He_init_kg_a        = data_a[46,:]
    # He_atmos_kg_a       = data_a[47,:]
    # He_atmos_pressure   = data_a[48,:]
    # He_total_kg         = He_liquid_kg_a + He_solid_kg_a + He_atmos_kg_a
    # He_escape_kg_a      = He_total_kg - He_liquid_kg_a - He_solid_kg_a - He_atmos_kg_a

    

    # # For "Interior" sub-loop (SPIDER)
    # if input_flag == "Interior":

    #     # # 1st timestep: apply chemical equilibrium to whole volatile mass
    #     # if loop_counter["init"] == 0:
    #     #     H2O_atmos_kg   = H2O_total_kg
    #     #     CO2_atmos_kg   = CO2_total_kg
    #     #     H2_atmos_kg    = H2_total_kg
    #     #     CH4_atmos_kg   = CH4_total_kg
    #     #     CO_atmos_kg    = CO_total_kg
    #     #     N2_atmos_kg    = N2_total_kg
    #     #     O2_atmos_kg    = O2_total_kg
    #     #     S_atmos_kg     = S_total_kg
    #     #     He_atmos_kg    = He_total_kg
    #     # # 2nd timestep+: outgassed atmosphere follows partitioning behavior
    #     # else:
    #     #     H2O_atmos_kg   = H2O_atmos_kg_a
    #     #     CO2_atmos_kg   = CO2_atmos_kg_a
    #     #     H2_atmos_kg    = H2_atmos_kg_a
    #     #     CH4_atmos_kg   = CH4_atmos_kg_a
    #     #     CO_atmos_kg    = CO_atmos_kg_a
    #     #     N2_atmos_kg    = N2_atmos_kg_a
    #     #     O2_atmos_kg    = O2_atmos_kg_a
    #     #     S_atmos_kg     = S_atmos_kg_a
    #     #     He_atmos_kg    = He_atmos_kg_a 
    #     H2O_atmos_kg   = H2O_atmos_kg_a
    #     CO2_atmos_kg   = CO2_atmos_kg_a
    #     H2_atmos_kg    = H2_atmos_kg_a
    #     CH4_atmos_kg   = CH4_atmos_kg_a
    #     CO_atmos_kg    = CO_atmos_kg_a
    #     N2_atmos_kg    = N2_atmos_kg_a
    #     O2_atmos_kg    = O2_atmos_kg_a
    #     S_atmos_kg     = S_atmos_kg_a
    #     He_atmos_kg    = He_atmos_kg_a        

    #     # Atmospheric mass from SPIDER output
    #     atm_kg = H2O_atmos_kg + CO2_atmos_kg + H2_atmos_kg + CH4_atmos_kg + CO_atmos_kg + N2_atmos_kg + O2_atmos_kg + S_atmos_kg + He_atmos_kg

    # # For "Atmosphere" sub-loop (VULCAN+SOCRATES)
    # if input_flag == "Atmosphere":

    #     # # Total surface pressure
    #     # P_surf  = float(atm_chemistry.iloc[0]["Pressure"])                 # bar

    #     # Recalculate partial pressures using Dalton's law
    #     H2O_atmos_pressure = float(atm_chemistry.iloc[0]["H2O"]) * P_surf  # bar
    #     CO2_atmos_pressure = float(atm_chemistry.iloc[0]["CO2"]) * P_surf  # bar
    #     H2_atmos_pressure  = float(atm_chemistry.iloc[0]["H2"])  * P_surf  # bar
    #     CH4_atmos_pressure = float(atm_chemistry.iloc[0]["CH4"]) * P_surf  # bar
    #     CO_atmos_pressure  = float(atm_chemistry.iloc[0]["CO"])  * P_surf  # bar
    #     N2_atmos_pressure  = float(atm_chemistry.iloc[0]["N2"])  * P_surf  # bar
    #     O2_atmos_pressure  = float(atm_chemistry.iloc[0]["O2"])  * P_surf  # bar
    #     S_atmos_pressure   = float(atm_chemistry.iloc[0]["S"])   * P_surf  # bar
    #     He_atmos_pressure  = float(atm_chemistry.iloc[0]["He"])  * P_surf  # bar

    #     ## !! TO DO: READ IN TOTAL MASSES FROM VULCAN !!

    #     # h_mol_total = 
    #     # O_H_ratio   = 
    #     # C_H_ratio   = 
    #     # N_H_ratio   = 
    #     # S_H_ratio   = 
    #     # He_H_ratio  = 

    #     # H2O_atmos_kg   = 
    #     # CO2_atmos_kg   = 
    #     # H2_atmos_kg    = 
    #     # CH4_atmos_kg   = 
    #     # CO_atmos_kg    = 
    #     # N2_atmos_kg    = 
    #     # O2_atmos_kg    = 
    #     # S_atmos_kg     = 
    #     # He_atmos_kg    = 

    #     # !! DERIVE TOTAL MASS FROM VULCAN OUTPUT
    #     H2O_atmos_kg   = H2O_atmos_kg_a
    #     CO2_atmos_kg   = CO2_atmos_kg_a
    #     H2_atmos_kg    = H2_atmos_kg_a
    #     CH4_atmos_kg   = CH4_atmos_kg_a
    #     CO_atmos_kg    = CO_atmos_kg_a
    #     N2_atmos_kg    = N2_atmos_kg_a
    #     O2_atmos_kg    = O2_atmos_kg_a
    #     S_atmos_kg     = S_atmos_kg_a
    #     He_atmos_kg    = He_atmos_kg_a

    #     # Atmospheric mass from ATMOS output --> CHANGE TO MAS OVER ALL SPECIES
    #     atm_kg = H2O_atmos_kg + CO2_atmos_kg + H2_atmos_kg + CH4_atmos_kg + CO_atmos_kg + N2_atmos_kg + O2_atmos_kg + S_atmos_kg + He_atmos_kg

    #     # / !! 

    # For "Atmosphere" sub-loop (VULCAN+SOCRATES) update heat flux from SOCRATES
    if input_flag == "Atmosphere":

        # Update heat flux from latest SOCRATES output
        runtime_helpfile_new["Heat_flux"] = SPIDER_options["heat_flux"]

        # Fatm_table                        = np.genfromtxt(output_dir+"OLRFlux.dat", names=['Time', 'Fatm'], dtype=None, skip_header=0)
        # time_list, Fatm_list  = list(Fatm_table['Time']), list(Fatm_table['Fatm'])
        # Fatm_newest    = Fatm_list[-1]
        # if loop_counter["init"] >= 1: 
        #     Fatm_newest    = Fatm_list[-1]
        # else: Fatm_newest                 = Fatm_list
        # runtime_helpfile_new["Heat_flux"] = Fatm_table['Fatm'][-1]


    # # Calculate element X/H ratios for ATMOS/VULCAN input
    # H2O_mol        = H2O_atmos_kg / H2O_mol_mass  # mol
    # CO2_mol        = CO2_atmos_kg / CO2_mol_mass  # mol
    # H2_mol         = H2_atmos_kg  / H2_mol_mass   # mol
    # CH4_mol        = CH4_atmos_kg / CH4_mol_mass  # mol
    # CO_mol         = CO_atmos_kg  / CO_mol_mass   # mol
    # N2_mol         = N2_atmos_kg  / N2_mol_mass   # mol
    # O2_mol         = O2_atmos_kg  / O2_mol_mass   # mol
    # S_mol          = S_atmos_kg   / S_mol_mass    # mol
    # He_mol         = He_atmos_kg  / He_mol_mass   # mol
  
    # # Radiative species + S + He
    # total_mol      = H2O_mol + CO2_mol + H2_mol + CH4_mol + CO_mol + N2_mol + O2_mol+ S_mol + He_mol

    # # Total numbers of elements
    # H_mol_total    = H2O_mol * 2. + H2_mol  * 2. + CH4_mol * 4. 
    # O_mol_total    = H2O_mol * 1. + CO2_mol * 2. + CO_mol  * 1. + O2_mol * 2.
    # C_mol_total    = CO2_mol * 1. + CH4_mol * 1. + CO_mol  * 1.
    # N_mol_total    = N2_mol  * 2.
    # S_mol_total    = S_mol   * 1. ## TO DO: Take into account major species?
    # He_mol_total   = He_mol  * 1.

    # # Relative to H
    # O_H_ratio      = O_mol_total  / H_mol_total      # mol/mol
    # C_H_ratio      = C_mol_total  / H_mol_total      # mol/mol
    # N_H_ratio      = N_mol_total  / H_mol_total      # mol/mol
    # S_H_ratio      = S_mol_total  / H_mol_total      # mol/mol 
    # He_H_ratio     = He_mol_total / H_mol_total      # mol/mol 

    # # Counter VULCAN bug with zero abundances
    # if N_H_ratio == 0.: 
    #     N_H_ratio = 1e-99
    # if S_H_ratio == 0.: 
    #     S_H_ratio = 1e-99
    # if He_H_ratio == 0.: 
    #     He_H_ratio = 1e-99

    # Baseline
    H2O_mol        = 0.
    CO2_mol        = 0.
    H2_mol         = 0.
    CH4_mol        = 0.
    CO_mol         = 0.
    N2_mol         = 0.
    O2_mol         = 0.
    S_mol          = 0.
    He_mol         = 0.

    # Else
    if SPIDER_options["H2O_initial_total_abundance"] > 0.:
        H2O_mol        = runtime_helpfile_new["H2O_atm_kg"] / H2O_mol_mass  # mol
    if SPIDER_options["CO2_initial_total_abundance"] > 0.:
        CO2_mol        = runtime_helpfile_new["CO2_atm_kg"] / CO2_mol_mass  # mol
    if SPIDER_options["H2_initial_total_abundance"] > 0.:
        H2_mol         = runtime_helpfile_new["H2_atm_kg"]  / H2_mol_mass   # mol
    if SPIDER_options["CH4_initial_total_abundance"] > 0.:
        CH4_mol        = runtime_helpfile_new["CH4_atm_kg"] / CH4_mol_mass  # mol
    if SPIDER_options["CO_initial_total_abundance"] > 0.:
        CO_mol         = runtime_helpfile_new["CO_atm_kg"]  / CO_mol_mass   # mol
    if SPIDER_options["N2_initial_total_abundance"] > 0.:
        N2_mol         = runtime_helpfile_new["N2_atm_kg"]  / N2_mol_mass   # mol
    if SPIDER_options["O2_initial_total_abundance"] > 0.:
        O2_mol         = runtime_helpfile_new["O2_atm_kg"]  / O2_mol_mass   # mol
    if SPIDER_options["S_initial_total_abundance"] > 0.:
        S_mol          = runtime_helpfile_new["S_atm_kg"]   / S_mol_mass    # mol
    if SPIDER_options["He_initial_total_abundance"] > 0.:
        He_mol         = runtime_helpfile_new["He_atm_kg"]  / He_mol_mass   # mol

    # # Calculate element X/H ratios for ATMOS/VULCAN input
    # H2O_mol        = H2O_atmos_kg / H2O_mol_mass  # mol
    # CO2_mol        = CO2_atmos_kg / CO2_mol_mass  # mol
    # H2_mol         = H2_atmos_kg  / H2_mol_mass   # mol
    # CH4_mol        = CH4_atmos_kg / CH4_mol_mass  # mol
    # CO_mol         = CO_atmos_kg  / CO_mol_mass   # mol
    # N2_mol         = N2_atmos_kg  / N2_mol_mass   # mol
    # O2_mol         = O2_atmos_kg  / O2_mol_mass   # mol
    # S_mol          = S_atmos_kg   / S_mol_mass    # mol
    # He_mol         = He_atmos_kg  / He_mol_mass   # mol
  
    # Radiative species + S + He
    total_mol      = H2O_mol + CO2_mol + H2_mol + CH4_mol + CO_mol + N2_mol + O2_mol+ S_mol + He_mol

    # Total numbers of elements
    H_mol_total    = H2O_mol * 2. + H2_mol  * 2. + CH4_mol * 4. 
    O_mol_total    = H2O_mol * 1. + CO2_mol * 2. + CO_mol  * 1. + O2_mol * 2.
    C_mol_total    = CO2_mol * 1. + CH4_mol * 1. + CO_mol  * 1.
    N_mol_total    = N2_mol  * 2.
    S_mol_total    = S_mol   * 1. ## TO DO: Take into account major species?
    He_mol_total   = He_mol  * 1.

    runtime_helpfile_new["H_mol"] = H_mol_total

    # Relative to H
    O_H_ratio      = O_mol_total  / runtime_helpfile_new["H_mol"] # mol/mol
    C_H_ratio      = C_mol_total  / runtime_helpfile_new["H_mol"] # mol/mol
    N_H_ratio      = N_mol_total  / runtime_helpfile_new["H_mol"] # mol/mol
    S_H_ratio      = S_mol_total  / runtime_helpfile_new["H_mol"] # mol/mol 
    He_H_ratio     = He_mol_total / runtime_helpfile_new["H_mol"] # mol/mol 

    # Counter VULCAN bug with zero abundances
    min_val     = 1e-99
    runtime_helpfile_new["O/H"]   = np.max([O_H_ratio, min_val])
    runtime_helpfile_new["C/H"]   = np.max([C_H_ratio, min_val])
    runtime_helpfile_new["N/H"]   = np.max([N_H_ratio, min_val])
    runtime_helpfile_new["S/H"]   = np.max([S_H_ratio, min_val])
    runtime_helpfile_new["He/H"]  = np.max([He_H_ratio, min_val])

    ## / Read in data

    # # Add all parameters to dataframe + update file on disk
    # runtime_helpfile_new = pd.DataFrame({
    #     'Time':             sim_time, 
    #     'Input':            input_flag, 
    #     'T_surf':           temperature_surface_a,
    #     'Phi_global':       phi_global,
    #     'Heat_flux':        Fatm,
    #     'P_surf':           P_surf,    
    #     'M_core':           mass_core_a, 
    #     'M_mantle':         mass_mantle_a, 
    #     'M_mantle_liquid':  mass_liquid_a,
    #     'M_mantle_solid':   mass_solid_a,
    #     'M_atm':            atm_kg,
    #     'H2O_liquid_kg':    H2O_liquid_kg_a, 
    #     'CO2_liquid_kg':    CO2_liquid_kg_a, 
    #     'H2_liquid_kg':     H2_liquid_kg_a, 
    #     'CH4_liquid_kg':    CH4_liquid_kg_a, 
    #     'CO_liquid_kg':     CO_liquid_kg_a, 
    #     'N2_liquid_kg':     N2_liquid_kg_a, 
    #     'O2_liquid_kg':     O2_liquid_kg_a, 
    #     'S_liquid_kg':      S_liquid_kg_a, 
    #     'He_liquid_kg':     He_liquid_kg_a,
    #     'H2O_solid_kg':     H2O_solid_kg_a, 
    #     'CO2_solid_kg':     CO2_solid_kg_a, 
    #     'H2_solid_kg':      H2_solid_kg_a, 
    #     'CH4_solid_kg':     CH4_solid_kg_a, 
    #     'CO_solid_kg':      CO_solid_kg_a, 
    #     'N2_solid_kg':      N2_solid_kg_a, 
    #     'O2_solid_kg':      O2_solid_kg_a, 
    #     'S_solid_kg':       S_solid_kg_a, 
    #     'He_solid_kg':      He_solid_kg_a,
    #     'H2O_atm_kg':       H2O_atmos_kg, 
    #     'CO2_atm_kg':       CO2_atmos_kg, 
    #     'H2_atm_kg':        H2_atmos_kg, 
    #     'CH4_atm_kg':       CH4_atmos_kg, 
    #     'CO_atm_kg':        CO_atmos_kg, 
    #     'N2_atm_kg':        N2_atmos_kg, 
    #     'O2_atm_kg':        O2_atmos_kg, 
    #     'S_atm_kg':         S_atmos_kg, 
    #     'He_atm_kg':        He_atmos_kg,
    #     'H2O_atm_bar':      H2O_atmos_pressure, 
    #     'CO2_atm_bar':      CO2_atmos_pressure, 
    #     'H2_atm_bar':       H2_atmos_pressure, 
    #     'CH4_atm_bar':      CH4_atmos_pressure, 
    #     'CO_atm_bar':       CO_atmos_pressure, 
    #     'N2_atm_bar':       N2_atmos_pressure, 
    #     'O2_atm_bar':       O2_atmos_pressure, 
    #     'S_atm_bar':        S_atmos_pressure, 
    #     'He_atm_bar':       He_atmos_pressure, 
    #     'H_mol':            H_mol_total, 
    #     'O/H':              O_H_ratio, 
    #     'C/H':              C_H_ratio, 
    #     'N/H':              N_H_ratio, 
    #     'S/H':              S_H_ratio, 
    #     'He/H':             He_H_ratio
    #     }, index=[0])
    # runtime_helpfile = runtime_helpfile.append(runtime_helpfile_new) 
    # runtime_helpfile.to_csv( output_dir+file_name, index=False, sep=" ")

    # Add all parameters to dataframe + update file on disk
    runtime_helpfile_new = pd.DataFrame({
        'Time':             runtime_helpfile_new["Time"],
        'Input':            runtime_helpfile_new["Input"],
        'T_surf':           runtime_helpfile_new["T_surf"],
        'Heat_flux':        runtime_helpfile_new["Heat_flux"],
        'P_surf':           runtime_helpfile_new["P_surf"],
        'M_atm':            runtime_helpfile_new["M_atm"],
        'Phi_global':       runtime_helpfile_new["Phi_global"],
        'M_core':           runtime_helpfile_new["M_core"],
        'M_mantle':         runtime_helpfile_new["M_mantle"],
        'M_mantle_liquid':  runtime_helpfile_new["M_mantle_liquid"],
        'M_mantle_solid':   runtime_helpfile_new["M_mantle_solid"],
        # 'H2O_atm_bar':      runtime_helpfile_new["H2O_atm_bar"],
        # 'CO2_atm_bar':      runtime_helpfile_new["CO2_atm_bar"],
        # 'H2_atm_bar':       runtime_helpfile_new["H2_atm_bar"],
        # 'CH4_atm_bar':      runtime_helpfile_new["CH4_atm_bar"],
        # 'CO_atm_bar':       runtime_helpfile_new["CO_atm_bar"],
        # 'N2_atm_bar':       runtime_helpfile_new["N2_atm_bar"],
        # 'O2_atm_bar':       runtime_helpfile_new["O2_atm_bar"],
        # 'S_atm_bar':        runtime_helpfile_new["S_atm_bar"],
        # 'He_atm_bar':       runtime_helpfile_new["He_atm_bar"],
        'H_mol':            runtime_helpfile_new["H_mol"],
        'O/H':              runtime_helpfile_new["O/H"],
        'C/H':              runtime_helpfile_new["C/H"],
        'N/H':              runtime_helpfile_new["N/H"],
        'S/H':              runtime_helpfile_new["S/H"],
        'He/H':             runtime_helpfile_new["He/H"],
        }, index=[0])
    runtime_helpfile = runtime_helpfile.append(runtime_helpfile_new) 
    runtime_helpfile.to_csv( output_dir+file_name, index=False, sep=" ")

    # Save SPIDER_options to disk
    SPIDER_options_save = pd.read_csv(output_dir+"spider_options.csv", sep=" ")
    SPIDER_options_save = SPIDER_options_save.append(SPIDER_options, ignore_index=True) 
    SPIDER_options_save.to_csv( output_dir+"spider_options.csv", index=False, sep=" ")

    # Advance time_current in main loop
    time_current = runtime_helpfile.iloc[-1]["Time"]

    return runtime_helpfile, time_current

def PrintSeparator():
    print("-------------------------------------------------------------------------------------------------------")
    pass

def PrintHalfSeparator():
    print("-----------------------------------------------")
    pass

# Generate/adapt VULCAN input files
def UpdateVulcanInputFiles( time_current, loop_counter, vulcan_dir, output_dir, runtime_helpfile, R_solid_planet ):

    # Initialize VULCAN input files at very first instance
    if loop_counter["init"] == 0 and loop_counter["atm"] == 0:
        with open(output_dir+'vulcan_XH_ratios.dat', 'w') as file:
            file.write('time             O                C                N                S                He\n')
        with open(output_dir+'vulcan_XH_ratios.dat', 'a') as file:
            file.write('0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00\n')

        # Change surface gravity in VULCAN init file
        M_solid_planet = runtime_helpfile.iloc[-1]["M_core"] + runtime_helpfile.iloc[-1]["M_mantle"]
        grav_s         = su.gravity( M_solid_planet, float(R_solid_planet) ) * 100. # cm s^-2

        # Generate init TP structure
        atm             = atmos()
        atm.ps          = runtime_helpfile.iloc[-1]["P_surf"]*1e3 # mbar
        pstart          = atm.ps#*.995
        rat             = (atm.ptop/pstart)**(1./atm.nlev)
        logLevels       = [pstart*rat**i for i in range(atm.nlev+1)]
        logLevels.reverse()
        levels          = [atm.ptop+i*(pstart-atm.ptop)/(atm.nlev-1) for i in range(atm.nlev+1)]
        atm.pl          = np.array(logLevels)
        atm.p           = (atm.pl[1:] + atm.pl[:-1]) / 2
        pressure_grid   = atm.p*1e-3 # bar
        temp_grid       = np.ones(len(pressure_grid))*runtime_helpfile.iloc[-1]["T_surf"]

        # Min/max P for VULCAN config file
        P_b = np.max(pressure_grid)*1e6*0.9999 # pressure at the bottom, (bar)->(dyne/cm^2)
        P_t = np.min(pressure_grid)*1e6*1.0001 # pressure at the top, (bar)->(dyne/cm^2)
        
        # Generate initial TP structure file
        out_a       = np.column_stack( ( temp_grid, pressure_grid ) ) # K, bar
        init_TP     = str(int(time_current))+"_atm_TP_profile_init.dat"
        np.savetxt( output_dir+init_TP, out_a )

        # Adjust copied file in output_dir
        for line in fileinput.input(vulcan_dir+'vulcan_cfg.py', inplace=True):
            if line.strip().startswith('g = '):
                line = "g = "+str(grav_s)+' # surface gravity (cm/s^2)\n'
            if line.strip().startswith('spider_file = '):
                line = "spider_file = '"+str(output_dir)+"vulcan_XH_ratios.dat'\n"
            if line.strip().startswith('atm_file = '):
                line = "atm_file = '"+str(output_dir)+str(init_TP)+"'\n"
            if line.strip().startswith('P_b = '):
                line = "P_b = "+str(P_b)+" # pressure at the bottom (dyne/cm^2)\n"
            if line.strip().startswith('P_t = '):
                line = "P_t = "+str(P_t)+" # pressure at the top (dyne/cm^2)\n"
            sys.stdout.write(line)
           

    # After first loop use updated TP profiles from SOCRATES
    else:
        # Modify the SOCRATES TP structure input
        run_TP_file = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*_atm_TP_profile.dat")])[-1]
        
        # Min/max P for VULCAN config file
        atm_table = np.genfromtxt(output_dir+run_TP_file, names=['T', 'Pbar'], dtype=None, skip_header=0)
        pressure_grid, temp_grid = atm_table['Pbar']*1e6, atm_table['T'] # (bar)->(dyne/cm^2), K
        P_b = np.max(pressure_grid)*0.9999 # pressure at the bottom, (dyne/cm^2)
        P_t = np.min(pressure_grid)*1.0001 # pressure at the top, (dyne/cm^2)

        # Adjust copied file in output_dir
        for line in fileinput.input(vulcan_dir+'vulcan_cfg.py', inplace=True):
            if line.strip().startswith('atm_file = '):
                line = "atm_file = '"+str(output_dir)+str(run_TP_file)+"'\n"
            if line.strip().startswith('P_b = '):
                line = "P_b = "+str(P_b)+" # pressure at the bottom (dyne/cm^2)\n"
            if line.strip().startswith('P_t = '):
                line = "P_t = "+str(P_t)+" # pressure at the top (dyne/cm^2)\n"
            sys.stdout.write(line)

    # Write elemental X/H ratios
    with open(output_dir+'vulcan_XH_ratios.dat', 'a') as file:
        file.write(str('{:.10e}'.format(time_current))+" "+str('{:.10e}'.format(runtime_helpfile.iloc[-1]["O/H"]))+" "+str('{:.10e}'.format(runtime_helpfile.iloc[-1]["C/H"]))+" "+str('{:.10e}'.format(runtime_helpfile.iloc[-1]["N/H"]))+" "+str('{:.10e}'.format(runtime_helpfile.iloc[-1]["S/H"]))+" "+str('{:.10e}'.format(runtime_helpfile.iloc[-1]["He/H"]))+"\n")

    # Last .json file -> print time
    last_file = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*.json")])[-1]
    time_print = last_file[:-5]

    # Update VULCAN output file names
    volume_mixing_ratios = str(time_print)+"_atm_chemistry_volume.dat"
    mass_mixing_ratios   = str(time_print)+"_atm_chemistry_mass.dat"
    for line in fileinput.input(vulcan_dir+'vulcan_cfg.py', inplace=True):
        if line.strip().startswith('EQ_outfile = '):
            line = "EQ_outfile = '"+str(output_dir)+str(volume_mixing_ratios)+"'"+' # volume mixing ratio\n'
        if line.strip().startswith('mass_outfile = '):
            line = "mass_outfile = '"+str(output_dir)+str(mass_mixing_ratios)+"'"+' # mass mixing ratio\n'
        sys.stdout.write(line)

    # Save modified temporary config file to compare versions
    shutil.copy(vulcan_dir+'vulcan_cfg.py', output_dir+str(time_print)+'_vulcan_cfg.py')

    return volume_mixing_ratios, mass_mixing_ratios

# Calulcate partial pressures from 
def ModifiedHenrysLaw( atm_chemistry, output_dir, file_name ):

    PrintSeparator()
    print("HACK –– apply partitioning directly on JSON files")
    PrintSeparator()

    # Total pressure
    P_surf = atm_chemistry.iloc[0]["Pressure"]*1e5 # Pa

    # Open the .json data
    with open(output_dir+file_name) as f:
        data = json.load(f)

    # Move unaltered JSON --> .txt
    shutil.move(output_dir+file_name, output_dir+file_name[:-5]+".txt")

    mantle_melt_kg = float(data["atmosphere"]["mass_liquid"]["values"][0])*float(data["atmosphere"]["mass_liquid"]["scaling"])
    print("M_mantle_liquid: ", mantle_melt_kg, "kg")
    PrintHalfSeparator()

    # Loop over all radiative species considered
    for volatile in volatile_species:

        # Dalton's law for partial pressures
        p_vol       = float(atm_chemistry.iloc[0][volatile])*P_surf # Pa

        # Modified Henry's law for obtaining melt abundances
        henry_alpha = volatile_distribution_coefficients[volatile+"_alpha"]
        henry_beta  = volatile_distribution_coefficients[volatile+"_beta"]

        # Find melt abundance
        X_vol_ppm   = henry_alpha * (p_vol**(1/henry_beta)) # ppm wt
        X_vol_kg    = X_vol_ppm*1e-6*mantle_melt_kg       # kg

        # Read in former scaled values
        liquid_ppm_scaled     = float(data["atmosphere"][volatile]["liquid_ppm"]["values"][0])
        liquid_kg_scaled      = float(data["atmosphere"][volatile]["liquid_kg"]["values"][0])
        atmosphere_kg_scaled  = float(data["atmosphere"][volatile]["atmosphere_kg"]["values"][0])
        atmosphere_bar_scaled = float(data["atmosphere"][volatile]["atmosphere_bar"]["values"][0])

        # Read in scalings
        liquid_ppm_scaling     = float(data["atmosphere"][volatile]["liquid_ppm"]["scaling"])
        liquid_kg_scaling      = float(data["atmosphere"][volatile]["liquid_kg"]["scaling"])
        atmosphere_kg_scaling  = float(data["atmosphere"][volatile]["atmosphere_kg"]["scaling"])
        atmosphere_bar_scaling = float(data["atmosphere"][volatile]["atmosphere_bar"]["scaling"])

        # Calculate former physical values
        liquid_ppm     = liquid_ppm_scaled*liquid_ppm_scaling
        liquid_kg      = liquid_kg_scaled*liquid_kg_scaling
        atmosphere_kg  = atmosphere_kg_scaled*atmosphere_kg_scaling
        atmosphere_bar = atmosphere_bar_scaled*atmosphere_bar_scaling

        # Calculate new physical values, ensure mass conservation
        liquid_ppm_new      = X_vol_ppm
        liquid_kg_new       = X_vol_kg
        atmosphere_kg_new   = (liquid_kg+atmosphere_kg)-liquid_kg_new
        atmosphere_bar_new  = p_vol

        # Calculate new scaled values
        liquid_ppm_new_scaled      = liquid_ppm_new/liquid_ppm_scaling
        liquid_kg_new_scaled       = liquid_kg_new/liquid_kg_scaling
        atmosphere_kg_new_scaled   = atmosphere_kg_new/atmosphere_kg_scaling
        atmosphere_bar_new_scaled  = atmosphere_bar_new/atmosphere_bar_scaling

        # Print the changes
        print(volatile, "liquid_ppm (scaled): ", liquid_ppm, "->", liquid_ppm_new, 
            "(", liquid_ppm_scaled, "->", liquid_ppm_new_scaled, ") ppm wt")
        print(volatile, "liquid_kg (scaled): ", liquid_kg, "->", liquid_kg_new, 
            "(", liquid_kg_scaled, "->", liquid_kg_new_scaled, ") kg")
        print(volatile, "atmosphere_kg (scaled): ", atmosphere_kg, "->", atmosphere_kg_new, 
            "(", atmosphere_kg_scaled, "->", atmosphere_kg_new_scaled, ") kg")
        print(volatile, "atmosphere_bar (scaled): ", atmosphere_bar, "->", atmosphere_bar_new, 
            "(", atmosphere_bar_scaled, "->", atmosphere_bar_new_scaled, ") bar")

        # Replace old with recalculated values
        data["atmosphere"][volatile]["liquid_ppm"]["values"]    = [str(liquid_ppm_new_scaled)]
        data["atmosphere"][volatile]["liquid_kg"]["values"]     = [str(liquid_kg_new_scaled)]
        data["atmosphere"][volatile]["atmosphere_kg"]["values"] = [str(atmosphere_kg_new_scaled)]
        data["atmosphere"][volatile]["atmosphere_bar"]["values"] = [str(atmosphere_bar_new_scaled)]
        # print(data["atmosphere"][volatile])
        PrintHalfSeparator()

        ## THERE IS ANOTHER ENTRY IN JSON:
        ## "Magma ocean volatile content"
        ## ---> CHECK IF THAT ALSO NEEDS TO BE REPLACED

    # Save the changed JSON file for read-in by SPIDER
    with open(output_dir+file_name, 'w') as f:
        json.dump(data, f, indent=4)

# run VULCAN
def RunVULCAN( time_current, loop_counter, vulcan_dir, coupler_dir, output_dir, runtime_helpfile, SPIDER_options ):

    # Generate/adapt VULCAN input files
    volume_mixing_ratios_name, mass_mixing_ratios_name = UpdateVulcanInputFiles( time_current, loop_counter, vulcan_dir, output_dir, runtime_helpfile, SPIDER_options["R_solid_planet"] )

    # Runtime info
    PrintSeparator()
    print("VULCAN run, loop ", loop_counter, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    PrintSeparator()

    # Switch to VULCAN directory; run VULCAN, switch back to main directory
    os.chdir(vulcan_dir)
    subprocess.run(["python", "vulcan.py", "-n"], shell=False)
    os.chdir(coupler_dir)

    # # Copy VULCAN dumps to output folder
    # shutil.copy(vulcan_dir+'output/vulcan_EQ.txt', output_dir+str(int(time_current))+"_atm_chemistry.dat")

    # Read in data from VULCAN output
    atm_chemistry = pd.read_csv(output_dir+volume_mixing_ratios_name, skiprows=1, delim_whitespace=True)
    print(atm_chemistry.iloc[:, 0:5])

    return atm_chemistry

def RunSOCRATES( time_current, time_offset, star_mass, mean_distance, output_dir, runtime_helpfile, atm_chemistry, loop_counter ):

    # Interpolate TOA heating from Baraffe models and distance from star
    stellar_toa_heating, solar_lum = InterpolateStellarLuminosity(star_mass, time_current, time_offset, mean_distance)

    # Runtime info
    PrintSeparator()
    print("SOCRATES run, loop ", loop_counter, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    PrintSeparator()
    # print(volatiles_mixing_ratio)

    # Calculate OLR flux for a given surface temperature w/ SOCRATES
    ## TO DO: Adapt to read in atmospheric profile from VULCAN
    heat_flux = SocRadConv.RadConvEqm(output_dir, time_current, runtime_helpfile, stellar_toa_heating, atm_chemistry, loop_counter) # W/m^2

    # Save OLR flux to be fed to SPIDER
    with open(output_dir+"OLRFlux.dat", "a") as f:
        f.write(str(float(time_current))+" "+str(float(heat_flux))+"\n")
        f.close()

    return heat_flux, stellar_toa_heating, solar_lum

def RunSPIDER( time_current, time_target, output_dir, SPIDER_options, loop_counter, runtime_helpfile, atm_chemistry ):

    # Define which volatiles to track in SPIDER
    species_call = ""
    for vol in volatile_species: 
        if SPIDER_options[vol+"_initial_total_abundance"] > 0.:
            species_call = species_call + "," + vol
    species_call = species_call[1:] # Remove "," in front

    # Recalculate time stepping
    if SPIDER_options["IC_INTERIOR"] == 2:      
        dtmacro     = SPIDER_options["dtmacro"]
        dtime       = time_target - time_current
        SPIDER_options["nstepsmacro"] =  math.ceil( dtime / float(SPIDER_options["dtmacro"]) )
    # For init loop and start from beginning
    else:
        dtmacro     = 0

    ### SPIDER base call sequence 
    call_sequence = [   
                        "spider", 
                        "-options_file",          "spider_input.opts", 
                        "-IC_INTERIOR",           str(SPIDER_options["IC_INTERIOR"]),
                        "-IC_ATMOSPHERE",         str(SPIDER_options["IC_ATMOSPHERE"]),
                        "-SURFACE_BC",            str(SPIDER_options["SURFACE_BC"]), 
                        "-surface_bc_value",      str(SPIDER_options["heat_flux"]), 
                        "-tsurf_poststep_change", str(SPIDER_options["tsurf_poststep_change"]),
                        "-nstepsmacro",           str(SPIDER_options["nstepsmacro"]), 
                        "-dtmacro",               str(dtmacro), 
                        "-radius",                str(SPIDER_options["R_solid_planet"]), 
                        "-coresize",              str(SPIDER_options["planet_coresize"]),
                        "-volatile_names",        str(species_call)
                    ]

    ### Conditional additions to call sequence

    # Define distribution coefficients and total mass/surface pressure for volatiles > 0
    for vol in volatile_species:
        if SPIDER_options[vol+"_initial_total_abundance"] > 0.:

            # Very first timestep: feed volatile initial abundance [ppm]
            if loop_counter["init"] == 0:
                    call_sequence.extend(["-"+vol+"_initial_total_abundance", str(SPIDER_options[vol+"_initial_total_abundance"])])

            # After very first timestep, starting w/ 2nd init loop
            if loop_counter["init"] >= 1:

                ## KLUDGE: Read in the same abundances every time -> no feedback from ATMOS
                call_sequence.extend(["-"+vol+"_initial_total_abundance", str(SPIDER_options[vol+"_initial_total_abundance"])])
                # # TO DO: MAKE THIS WORK
                # # Load partial pressures from VULCAN
                # volume_mixing_ratio     = atm_chemistry.iloc[0][vol]
                # surface_pressure_total  = atm_chemistry.iloc[0]["Pressure"]*1e5 # bar -> Pa
                # partial_pressure_vol    = surface_pressure_total*volume_mixing_ratio
                # SPIDER_options[vol+"_initial_atmos_pressure"] = partial_pressure_vol
                # call_sequence.extend(["-"+vol+"_initial_atmos_pressure", str(SPIDER_options[vol+"_initial_atmos_pressure"])])

            call_sequence.extend(["-"+vol+"_henry", str(volatile_distribution_coefficients[vol+"_henry"])])
            call_sequence.extend(["-"+vol+"_henry_pow", str(volatile_distribution_coefficients[vol+"_henry_pow"])])
            call_sequence.extend(["-"+vol+"_kdist", str(volatile_distribution_coefficients[vol+"_kdist"])])
            call_sequence.extend(["-"+vol+"_kabs", str(volatile_distribution_coefficients[vol+"_kabs"])])
            call_sequence.extend(["-"+vol+"_molar_mass", str(molar_mass[vol])])

    # With start of the main loop only:
    # Volatile specific options: post step settings, restart filename
    if SPIDER_options["IC_INTERIOR"] == 2:
        call_sequence.extend([ 
                                "-ic_interior_filename",  str(output_dir+SPIDER_options["ic_interior_filename"]),
                                "-activate_poststep", 
                                "-activate_rollback"
                             ])
        for vol in volatile_species:
            if SPIDER_options[vol+"_initial_total_abundance"] > 0.:
                call_sequence.extend(["-"+vol+"_poststep_change", str(SPIDER_options[vol+"_poststep_change"])])

    # Runtime info
    PrintSeparator()
    print("SPIDER run, loop ", loop_counter, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), "| flags:")
    for flag in call_sequence:
        print(flag, end =" ")
    print()
    PrintSeparator()

    # Call SPIDER
    subprocess.call(call_sequence)

    # Update restart filename for next SPIDER run
    SPIDER_options["ic_interior_filename"] = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*.json")])[-1]

    return SPIDER_options

def CleanOutputDir( output_dir ):

    PrintSeparator()
    print("Remove old output files:", end =" ")
    for file in natsorted(glob.glob(output_dir+"*.*")):
        os.remove(file)
        print(os.path.basename(file), end =" ")
    print("==> Done.")

# Plot conditions throughout run for on-the-fly analysis
def UpdatePlots( output_dir ):

    PrintSeparator()
    print("Plot current evolution")
    PrintSeparator()
    output_times = su.get_all_output_times()
    if len(output_times) <= 8:
        plot_times = output_times
    else:
        plot_times = [ output_times[0]]         # first snapshot
        for i in [ 2, 15, 22, 30, 45, 66 ]:     # distinct timestamps
            plot_times.append(output_times[int(round(len(output_times)*(i/100.)))])
        plot_times.append(output_times[-1])     # last snapshot
    print("snapshots:", plot_times)

    # # Global properties for all timesteps
    # if len(output_times) > 1:
    #     plot_global.plot_global(output_dir)   

    # Specific timesteps for paper plots
    plot_interior.plot_interior(plot_times)     
    plot_atmosphere.plot_atmosphere(output_dir, plot_times)
    plot_stacked.plot_stacked(output_dir, plot_times)
    
    # One plot per timestep for video files
    plot_atmosphere.plot_current_mixing_ratio(output_dir, plot_times[-1]) 

    # Close all figures
    plt.close()

# https://stackoverflow.com/questions/14115254/creating-a-folder-with-timestamp
# https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def make_output_dir( output_dir ):
    save_dir = output_dir+"save/"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+"/"
    try:
       os.makedirs(save_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(save_dir):
            pass
        else:
            raise
    return save_dir

def SaveOutput( output_dir ):

    # Copy old files to separate folder
    save_dir = make_output_dir( output_dir ) #
    print("===> Copy files to separate dir for this run to:", save_dir)
    # shutil.copy(output_dir+"spider_input.opts", save_dir+"spider_input.opts")
    for file in natsorted(glob.glob(output_dir+"*.*")):
        shutil.copy(file, save_dir+os.path.basename(file))
        print(os.path.basename(file), end =" ")

