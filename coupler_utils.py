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
s_mol_mass      = 0.03206               # kg mol−1
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

# Henry's law coefficients
# Add to dataframe + save to disk
henry_coefficients = pd.DataFrame({
    'H2O_alpha': 5.0637663E-2,  # ppm/Pa
    'H2O_beta': 3.22290237,     
    'CO2_alpha': 1.95E-7, 
    'CO2_beta': 0.71396905, 
    'H2_alpha': 2.572E-6, 
    'H2_beta': 1.0, 
    'CH4_alpha': 9.9E-8, 
    'CH4_beta': 1.0, 
    'CO_alpha': 1.6E-7, 
    'CO_beta': 1.0, 
    'N2_alpha': 5.0E-7, 
    'N2_beta': 2.0, 
    'O2_alpha': 0.001E-9, 
    'O2_beta': 1.0, 
    'S_alpha': 0.001E-9, 
    'S_beta': 1.0, 
    'He_alpha': 0.001E-9, 
    'He_beta': 1.0}, index=[0])


# https://stackoverflow.com/questions/14115254/creating-a-folder-with-timestamp
def make_output_dir( output_dir ):
    output_dir = output_dir+"save/"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+"/"
    try:
        os.makedirs(output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..

    return save_dir

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

def PrintCurrentState(time_current, runtime_helpfile, p_s, heat_flux, stellar_toa_heating, solar_lum, loop_counter, output_dir):

    restart_file = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*.json")])[-1]

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
    print("Total heat flux [W/m^2]:", heat_flux)
    print("Last file name:", restart_file)
    print("---------------------------------------------------------")

    return restart_file

def UpdateHelpfile(loop_counter, output_dir, file_name, runtime_helpfile=[]):

    # # Check if file is already existent, if not, generate and save
    # if os.path.isfile(output_dir+file_name):
    #     runtime_helpfile = pd.read_csv(output_dir+file_name)
    # else:
    #     runtime_helpfile = pd.DataFrame(columns=['Time', 'Input', 'T_surf', 'M_core', 'M_mantle', 'M_mantle_liquid', 'M_mantle_solid', 'M_atm', 'H2O_atm_kg', 'CO2_atm_kg', 'H2_atm_kg', 'CH4_atm_kg', 'CO_atm_kg', 'N2_atm_kg', 'O2_atm_kg', 'S_atm_kg', 'He_atm_kg', 'H_mol', 'O/H', 'C/H', 'N/H', 'S/H', 'He/H'])
    #     runtime_helpfile.to_csv( output_dir+file_name, index=False, sep=" ") 

    # # Check if SPIDER was previously initialized
    # if loop_counter["total"] == 0 and loop_counter["init"] == 0 and loop_counter["atm"] == 0:

    # # Check if file is already existent, if not, generate and save
    # if loop_counter["total"] == 0 and loop_counter["init"] == 0:
    #     runtime_helpfile = pd.DataFrame(columns=['Time', 'Input', 'T_surf', 'M_core', 'M_mantle', 'M_mantle_liquid', 'M_mantle_solid', 'M_atm', 'H2O_atm_kg', 'CO2_atm_kg', 'H2_atm_kg', 'CH4_atm_kg', 'CO_atm_kg', 'N2_atm_kg', 'O2_atm_kg', 'S_atm_kg', 'He_atm_kg', 'H_mol', 'O/H', 'C/H', 'N/H', 'S/H', 'He/H'])
    #     runtime_helpfile.to_csv( output_dir+file_name, index=False, sep=" ") 

    if loop_counter["total"] == 0:
        input_flag = "INIT"
    else:
        input_flag = "RUN"

    if os.path.isfile(output_dir+file_name):

        # Get only last sim time
        sim_times = su.get_all_output_times(output_dir)  # yr
        sim_time  = sim_times[-1]

        keys_t = ( ('atmosphere','mass_liquid'),
                   ('atmosphere','mass_solid'),
                   ('atmosphere','mass_mantle'),
                   ('atmosphere','mass_core'),

                   ('atmosphere','H2O','liquid_kg'),
                   ('atmosphere','H2O','solid_kg'),
                   ('atmosphere','H2O','initial_kg'),
                   ('atmosphere','H2O','atmosphere_kg'),
                   ('atmosphere','H2O','atmosphere_bar'),

                   ('atmosphere','CO2','liquid_kg'),
                   ('atmosphere','CO2','solid_kg'),
                   ('atmosphere','CO2','initial_kg'),
                   ('atmosphere','CO2','atmosphere_kg'),
                   ('atmosphere','CO2','atmosphere_bar'),

                   ('atmosphere','H2','liquid_kg'),
                   ('atmosphere','H2','solid_kg'),
                   ('atmosphere','H2','initial_kg'),
                   ('atmosphere','H2','atmosphere_kg'),
                   ('atmosphere','H2','atmosphere_bar'),

                   ('atmosphere','CH4','liquid_kg'),
                   ('atmosphere','CH4','solid_kg'),
                   ('atmosphere','CH4','initial_kg'),
                   ('atmosphere','CH4','atmosphere_kg'),
                   ('atmosphere','CH4','atmosphere_bar'),

                   ('atmosphere','CO','liquid_kg'),
                   ('atmosphere','CO','solid_kg'),
                   ('atmosphere','CO','initial_kg'),
                   ('atmosphere','CO','atmosphere_kg'),
                   ('atmosphere','CO','atmosphere_bar'),

                   ('atmosphere','N2','liquid_kg'),
                   ('atmosphere','N2','solid_kg'),
                   ('atmosphere','N2','initial_kg'),
                   ('atmosphere','N2','atmosphere_kg'),
                   ('atmosphere','N2','atmosphere_bar'),

                   ('atmosphere','O2','liquid_kg'),
                   ('atmosphere','O2','solid_kg'),
                   ('atmosphere','O2','initial_kg'),
                   ('atmosphere','O2','atmosphere_kg'),
                   ('atmosphere','O2','atmosphere_bar'),

                   ('atmosphere','S','liquid_kg'),
                   ('atmosphere','S','solid_kg'),
                   ('atmosphere','S','initial_kg'),
                   ('atmosphere','S','atmosphere_kg'),
                   ('atmosphere','S','atmosphere_bar'),

                   ('atmosphere','He','liquid_kg'),
                   ('atmosphere','He','solid_kg'),
                   ('atmosphere','He','initial_kg'),
                   ('atmosphere','He','atmosphere_kg'),
                   ('atmosphere','He','atmosphere_bar'),

                   ('atmosphere','temperature_surface'),
                   ('atmosphere','emissivity'),
                   ('rheological_front_phi','phi_global'),
                   ('atmosphere','Fatm') )

        # data_a = su.get_dict_surface_values_for_times( keys_t, sim_times )
        data_a = su.get_dict_surface_values_for_specific_time( keys_t, sim_time )

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
        H2O_liquid_kg_a = data_a[4,:]
        H2O_solid_kg_a = data_a[5,:]
        H2O_total_kg_a = data_a[6,:]
        H2O_total_kg = H2O_total_kg_a[0]        # time-independent
        H2O_atmos_kg_a = data_a[7,:]
        H2O_atmos_a = data_a[8,:]
        H2O_escape_kg_a = H2O_total_kg - H2O_liquid_kg_a - H2O_solid_kg_a - H2O_atmos_kg_a

        CO2_liquid_kg_a = data_a[9,:]
        CO2_solid_kg_a = data_a[10,:]
        CO2_total_kg_a = data_a[11,:]
        CO2_total_kg = CO2_total_kg_a[0]        # time-independent
        CO2_atmos_kg_a = data_a[12,:]
        CO2_atmos_a = data_a[13,:]
        CO2_escape_kg_a = CO2_total_kg - CO2_liquid_kg_a - CO2_solid_kg_a - CO2_atmos_kg_a

        H2_liquid_kg_a = data_a[14,:]
        H2_solid_kg_a = data_a[15,:]
        H2_total_kg_a = data_a[16,:]
        H2_total_kg = H2_total_kg_a[0]        # time-independent
        H2_atmos_kg_a = data_a[17,:]
        H2_atmos_a = data_a[18,:]
        H2_escape_kg_a = H2_total_kg - H2_liquid_kg_a - H2_solid_kg_a - H2_atmos_kg_a

        CH4_liquid_kg_a = data_a[19,:]
        CH4_solid_kg_a = data_a[20,:]
        CH4_total_kg_a = data_a[21,:]
        CH4_total_kg = CH4_total_kg_a[0]        # time-independent
        CH4_atmos_kg_a = data_a[22,:]
        CH4_atmos_a = data_a[23,:]
        CH4_escape_kg_a = CH4_total_kg - CH4_liquid_kg_a - CH4_solid_kg_a - CH4_atmos_kg_a

        CO_liquid_kg_a = data_a[24,:]
        CO_solid_kg_a = data_a[25,:]
        CO_total_kg_a = data_a[26,:]
        CO_total_kg = CO_total_kg_a[0]        # time-independent
        CO_atmos_kg_a = data_a[27,:]
        CO_atmos_a = data_a[28,:]
        CO_escape_kg_a = CO_total_kg - CO_liquid_kg_a - CO_solid_kg_a - CO_atmos_kg_a

        N2_liquid_kg_a = data_a[29,:]
        N2_solid_kg_a = data_a[30,:]
        N2_total_kg_a = data_a[31,:]
        N2_total_kg = N2_total_kg_a[0]        # time-independent
        N2_atmos_kg_a = data_a[32,:]
        N2_atmos_a = data_a[33,:]
        N2_escape_kg_a = N2_total_kg - N2_liquid_kg_a - N2_solid_kg_a - N2_atmos_kg_a

        O2_liquid_kg_a = data_a[34,:]
        O2_solid_kg_a = data_a[35,:]
        O2_total_kg_a = data_a[36,:]
        O2_total_kg = O2_total_kg_a[0]        # time-independent
        O2_atmos_kg_a = data_a[37,:]
        O2_atmos_a = data_a[38,:]
        O2_escape_kg_a = O2_total_kg - O2_liquid_kg_a - O2_solid_kg_a - O2_atmos_kg_a

        S_liquid_kg_a = data_a[39,:]
        S_solid_kg_a = data_a[40,:]
        S_total_kg_a = data_a[41,:]
        S_total_kg = S_total_kg_a[0]        # time-independent
        S_atmos_kg_a = data_a[42,:]
        S_atmos_a = data_a[43,:]
        S_escape_kg_a = S_total_kg - S_liquid_kg_a - S_solid_kg_a - S_atmos_kg_a

        He_liquid_kg_a = data_a[44,:]
        He_solid_kg_a = data_a[45,:]
        He_total_kg_a = data_a[46,:]
        He_total_kg = He_total_kg_a[0]        # time-independent
        He_atmos_kg_a = data_a[47,:]
        He_atmos_a = data_a[48,:]
        He_escape_kg_a = He_total_kg - He_liquid_kg_a - He_solid_kg_a - He_atmos_kg_a

        temperature_surface_a = data_a[49,:]
        emissivity_a = data_a[50,:]             # internally computed emissivity
        phi_global = data_a[51,:]               # global melt fraction
        Fatm = data_a[52,:]

        if input_flag == "INIT":

            ## For redistribution of total atmosphere mass in VULCAN:
            H2O_atmos_kg   = H2O_total_kg
            CO2_atmos_kg   = CO2_total_kg
            H2_atmos_kg    = H2_total_kg
            CH4_atmos_kg   = CH4_total_kg
            CO_atmos_kg    = CO_total_kg
            N2_atmos_kg    = N2_total_kg
            O2_atmos_kg    = O2_total_kg
            S_atmos_kg     = S_total_kg
            He_atmos_kg    = He_total_kg

            ## --> Element X/H ratios for VULCAN input
            h2o_mol        = H2O_atmos_kg/h2o_mol_mass # mol
            co2_mol        = CO2_atmos_kg/co2_mol_mass # mol
            h2_mol         = H2_atmos_kg/h2_mol_mass   # mol
            ch4_mol        = CH4_atmos_kg/ch4_mol_mass # mol
            co_mol         = CO_atmos_kg/co_mol_mass   # mol
            n2_mol         = N2_atmos_kg/n2_mol_mass   # mol
            o2_mol         = O2_atmos_kg/o2_mol_mass   # mol
            s_mol          = S_atmos_kg/he_mol_mass    # mol
            he_mol         = He_atmos_kg/he_mol_mass   # mol
          
            # Radiative species + He
            total_mol      = h2o_mol + co2_mol + h2_mol + ch4_mol + co_mol + n2_mol + o2_mol+ s_mol + he_mol

            h_mol_total    = h2o_mol * 2. + h2_mol  * 2. + ch4_mol * 4. 
            o_mol_total    = h2o_mol * 1. + co2_mol * 2. + co_mol  * 1. + o2_mol * 2.
            c_mol_total    = co2_mol * 1. + ch4_mol * 1. + co_mol  * 1.
            n_mol_total    = n2_mol  * 2.
            s_mol_total    = s_mol   * 1. ## TO DO: Take into account major species?
            he_mol_total   = he_mol  * 1.

            O_H_ratio  = o_mol_total / h_mol_total       # mol/mol
            C_H_ratio  = c_mol_total / h_mol_total       # mol/mol
            N_H_ratio  = n_mol_total / h_mol_total      # mol/mol
            S_H_ratio  = s_mol_total / h_mol_total       # mol/mol 
            He_H_ratio = he_mol_total / h_mol_total      # mol/mol 

            # Counter VULCAN bug with zero abundances
            if N_H_ratio == 0.: 
                N_H_ratio = 1e-99
            if S_H_ratio == 0.: 
                S_H_ratio = 1e-99

            ### TO DO: ADD RECALC OF VOLATILES_PPM FOR INIT LOOP!!!

        # during main loop
        else:

            h_mol_total = runtime_helpfile.iloc[-1]['H_mol'] # total mol of H
            O_H_ratio   = runtime_helpfile.iloc[-1]['O/H']   # elemental ratio (N_x/N_y)
            C_H_ratio   = runtime_helpfile.iloc[-1]['C/H']
            N_H_ratio   = runtime_helpfile.iloc[-1]['N/H']
            S_H_ratio   = runtime_helpfile.iloc[-1]['S/H']
            He_H_ratio  = runtime_helpfile.iloc[-1]['He/H']

            ## For redistribution of total atmosphere mass in VULCAN:
            H2O_atmos_kg   = H2O_atmos_kg_a
            CO2_atmos_kg   = CO2_atmos_kg_a
            H2_atmos_kg    = H2_atmos_kg_a
            CH4_atmos_kg   = CH4_atmos_kg_a
            CO_atmos_kg    = CO_atmos_kg_a
            N2_atmos_kg    = N2_atmos_kg_a
            O2_atmos_kg    = O2_atmos_kg_a
            S_atmos_kg     = S_atmos_kg_a
            He_atmos_kg    = He_atmos_kg_a

        atm_kg = H2O_atmos_kg + CO2_atmos_kg + H2_atmos_kg + CH4_atmos_kg + CO_atmos_kg + N2_atmos_kg + O2_atmos_kg + S_atmos_kg + He_atmos_kg

        # Add to dataframe + save to disk
        runtime_helpfile_new = pd.DataFrame({
            'Time': sim_time, 
            'Input': input_flag, 
            'T_surf': temperature_surface_a, 
            'M_core': mass_core_a, 
            'M_mantle': mass_mantle_a, 
            'M_mantle_liquid': mass_liquid_a,
            'M_mantle_solid': mass_solid_a,
            'M_atm': atm_kg, 
            'H2O_atm_kg': H2O_atmos_kg, 
            'CO2_atm_kg': CO2_atmos_kg, 
            'H2_atm_kg': H2_atmos_kg, 
            'CH4_atm_kg': CH4_atmos_kg, 
            'CO_atm_kg': CO_atmos_kg, 
            'N2_atm_kg': N2_atmos_kg, 
            'O2_atm_kg': O2_atmos_kg, 
            'S_atm_kg': S_atmos_kg, 
            'He_atm_kg': He_atmos_kg, 
            'H_mol': h_mol_total, 
            'O/H': O_H_ratio, 
            'C/H': C_H_ratio, 
            'N/H': N_H_ratio, 
            'S/H': S_H_ratio, 
            'He/H': He_H_ratio}, index=[0])
        runtime_helpfile = runtime_helpfile.append(runtime_helpfile_new) 
        runtime_helpfile.to_csv( output_dir+file_name, index=False, sep=" ")

    else:
        runtime_helpfile = pd.DataFrame(columns=['Time', 'Input', 'T_surf', 'M_core', 'M_mantle', 'M_mantle_liquid', 'M_mantle_solid', 'M_atm', 'H2O_atm_kg', 'CO2_atm_kg', 'H2_atm_kg', 'CH4_atm_kg', 'CO_atm_kg', 'N2_atm_kg', 'O2_atm_kg', 'S_atm_kg', 'He_atm_kg', 'H_mol', 'O/H', 'C/H', 'N/H', 'S/H', 'He/H'])
        runtime_helpfile.to_csv( output_dir+file_name, index=False, sep=" ") 

    return runtime_helpfile

def PrintSeparator():
    print("---------------------------------------------------------")
    pass

def PrintHalfSeparator():
    print("---------------------")
    pass

# Generate/adapt VULCAN input files
def UpdateVulcanInputFiles( time_current, loop_counter, vulcan_dir, output_dir, runtime_helpfile, R_solid_planet ):

    # Initialize VULCAN input files
    if loop_counter["init"] == 0 and loop_counter["atm"] == 0:
        with open(output_dir+'vulcan_XH_ratios.dat', 'w') as file:
            file.write('time             O                C                N                S                He\n')
        with open(output_dir+'vulcan_XH_ratios.dat', 'a') as file:
            file.write('0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00\n')

        # Change surface gravity in VULCAN init file
        M_solid_planet = runtime_helpfile.iloc[-1]["M_core"] + runtime_helpfile.iloc[-1]["M_mantle"]
        grav_s         = su.gravity( M_solid_planet, float(R_solid_planet) ) * 100. # cm s^-2

        # Adjust copied file in output_dir
        for line in fileinput.input(vulcan_dir+'vulcan_cfg.py', inplace=True):
            if line.strip().startswith('g = '):
                line = "g = "+str(grav_s)+' # surface gravity (cm/s^2)\n'
            if line.strip().startswith('spider_file = '):
                line = "spider_file = '"+str(output_dir)+"vulcan_XH_ratios.dat'\n"
            if line.strip().startswith('atm_file = '):
                line = "atm_file = 'socrates_input/TP.dat'\n"
            sys.stdout.write(line)
    # After first loop use updated TP profiles from SOCRATES
    else:
        # Modify the SOCRATES TP structure input
        SOCRATES_TP = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*_atm_TP_profile.dat")])[-1]

        # Adjust copied file in output_dir
        for line in fileinput.input(vulcan_dir+'vulcan_cfg.py', inplace=True):
            if line.strip().startswith('atm_file = '):
                line = "atm_file = '"+str(output_dir)+str(SOCRATES_TP)+"'\n"
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


def ModifiedHenrysLaw( atm_chemistry, output_dir, file_name ):

    PrintSeparator()
    print("HACK –– apply partitioning directly on JSON files")
    PrintSeparator()

    # Total pressure
    P_surf = atm_chemistry.iloc[0]["Pressure"]*1e5 # Pa

    volatile_species = [ "H2O", "CO2", "H2", "CH4", "CO", "N2", "O2", "S", "He" ]

    # Make copy of latest JSON
    shutil.copy(output_dir+file_name, output_dir+file_name[:-5]+"_orig.json")

    with open(output_dir+file_name) as f:
        data = json.load(f)

    mantle_melt_kg = float(data["atmosphere"]["mass_liquid"]["values"][0])*float(data["atmosphere"]["mass_liquid"]["scaling"])
    print("M_mantle_liquid: ", mantle_melt_kg, "kg")

    # Loop over all radiative species considered
    for volatile in volatile_species:

        # Dalton's law for partial pressures
        p_vol = atm_chemistry.iloc[0][volatile]*P_surf

        # Modified Henry's law for obtaining melt abundances
        henry_alpha = henry_coefficients.iloc[-1][volatile+"_alpha"]
        henry_beta  = henry_coefficients.iloc[-1][volatile+"_beta"]

        # Find melt abundance
        X_vol_ppm = henry_alpha * p_vol**(1/henry_beta) # ppm wt
        X_vol_kg  = X_vol_ppm*1e-6*mantle_melt_kg       # kg

        # Read in scalings
        scaling_ppm   = float(data["atmosphere"][volatile]["liquid_ppm"]["scaling"])
        scaling_kg    = float(data["atmosphere"][volatile]["liquid_kg"]["scaling"])

        # Print the changes
        print(volatile, "liquid_ppm: ", float(data["atmosphere"][volatile]["liquid_ppm"]["values"][0])*scaling_ppm, "->", X_vol_ppm)
        print(volatile, "liquid_kg: ", float(data["atmosphere"][volatile]["liquid_kg"]["values"][0])*scaling_kg, "->", X_vol_kg)
        print(volatile, "liquid_ppm/scaled: ", float(data["atmosphere"][volatile]["liquid_ppm"]["values"][0]), "->", X_vol_ppm/scaling_ppm)
        print(volatile, "liquid_kg/scaled: ", float(data["atmosphere"][volatile]["liquid_kg"]["values"][0]), "->", X_vol_kg/scaling_kg)

        data["atmosphere"][volatile]["liquid_ppm"]["values"] = [str(X_vol_ppm/scaling_ppm)]
        data["atmosphere"][volatile]["liquid_kg"]["values"] = [str(X_vol_kg/scaling_kg)]
        PrintHalfSeparator()
        
        # for item in data["atmosphere"]:

    # Save the changed JSON file for read-in by SPIDER
    with open(output_dir+file_name, 'w') as f:
        json.dump(data, f)


    # run VULCAN
def RunVULCAN( time_current, loop_counter, vulcan_dir, coupler_dir, output_dir, runtime_helpfile, SPIDER_options ):

    R_solid_planet = SPIDER_options["R_solid_planet"]

    # Generate/adapt VULCAN input files
    volume_mixing_ratios_name, mass_mixing_ratios_name = UpdateVulcanInputFiles( time_current, loop_counter, vulcan_dir, output_dir, runtime_helpfile, R_solid_planet )

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

    # plot_atmosphere.plot_mixing_ratios(output_dir, atm_chemistry, int(time_current)) # specific time steps

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
    heat_flux = str(SocRadConv.RadConvEqm(output_dir, time_current, runtime_helpfile.iloc[-1]["T_surf"], stellar_toa_heating, atm_chemistry)) # W/m^2

    # Save OLR flux to be fed to SPIDER
    with open(output_dir+"OLRFlux.dat", "a") as f:
        f.write(str(float(time_current))+" "+str(float(heat_flux))+"\n")
        f.close()

    return heat_flux, stellar_toa_heating, solar_lum

def RunSPIDER( time_current, time_target, output_dir, SPIDER_options, loop_counter, runtime_helpfile, runtime_helpfile_name, restart_filename ):

    # SPIDER start input options
    SURFACE_BC            = str(SPIDER_options["SURFACE_BC"])
    SOLVE_FOR_VOLATILES   = str(SPIDER_options["SOLVE_FOR_VOLATILES"])
    H2O_poststep_change   = str(SPIDER_options["H2O_poststep_change"])
    CO2_poststep_change   = str(SPIDER_options["CO2_poststep_change"])
    H2_poststep_change    = str(SPIDER_options["H2_poststep_change"])
    CH4_poststep_change   = str(SPIDER_options["CH4_poststep_change"])
    CO_poststep_change    = str(SPIDER_options["CO_poststep_change"])
    N2_poststep_change    = str(SPIDER_options["N2_poststep_change"])
    O2_poststep_change    = str(SPIDER_options["O2_poststep_change"])
    S_poststep_change     = str(SPIDER_options["S_poststep_change"])
    He_poststep_change    = str(SPIDER_options["He_poststep_change"])
    tsurf_poststep_change = str(SPIDER_options["tsurf_poststep_change"])
    R_solid_planet        = str(SPIDER_options["R_solid_planet"])
    planet_coresize       = str(SPIDER_options["planet_coresize"])
    heat_flux             = str(SPIDER_options["heat_flux"])

    # Restart flag
    start_condition       = str(SPIDER_options["start_condition"])

    # Time stepping
    if start_condition == "1":
        dtmacro               = str(SPIDER_options["dtmacro_init"])
        nstepsmacro           = str(SPIDER_options["nstepsmacro_init"])
    elif start_condition == "2":
        dtmacro               = str(SPIDER_options["dtmacro"])
        # Recalculate time stepping
        dtime                 = time_target - time_current
        nstepsmacro           = str( math.ceil( dtime / float(dtmacro) ) )

    # Restart SPIDER with (progressively more) self-consistent atmospheric composition
    if loop_counter["init"] >= 1 or loop_counter["atm"] >= 1:
        M_mantle_liquid           = runtime_helpfile.iloc[-1]["M_mantle_liquid"] # kg
        SPIDER_options["H2O_ppm"] = str(runtime_helpfile.iloc[-1]["H2O_atm_kg"]/M_mantle_liquid)
        SPIDER_options["CO2_ppm"] = str(runtime_helpfile.iloc[-1]["CO2_atm_kg"]/M_mantle_liquid)
        SPIDER_options["H2_ppm"]  = str(runtime_helpfile.iloc[-1]["H2_atm_kg"]/M_mantle_liquid)
        SPIDER_options["CH4_ppm"] = str(runtime_helpfile.iloc[-1]["CH4_atm_kg"]/M_mantle_liquid)
        SPIDER_options["CO_ppm"]  = str(runtime_helpfile.iloc[-1]["CO_atm_kg"]/M_mantle_liquid)
        SPIDER_options["N2_ppm"]  = str(runtime_helpfile.iloc[-1]["N2_atm_kg"]/M_mantle_liquid)
        SPIDER_options["O2_ppm"]  = str(runtime_helpfile.iloc[-1]["O2_atm_kg"]/M_mantle_liquid)
        SPIDER_options["S_ppm"]   = str(runtime_helpfile.iloc[-1]["S_atm_kg"]/M_mantle_liquid)
        SPIDER_options["He_ppm"]  = str(runtime_helpfile.iloc[-1]["He_atm_kg"]/M_mantle_liquid)

    # Initial volatile budget, in ppm relative to molten mantle mass
    H2O_ppm           = str(SPIDER_options["H2O_ppm"])
    CO2_ppm           = str(SPIDER_options["CO2_ppm"])
    H2_ppm            = str(SPIDER_options["H2_ppm"]) 
    CH4_ppm           = str(SPIDER_options["CH4_ppm"])
    CO_ppm            = str(SPIDER_options["CO_ppm"]) 
    N2_ppm            = str(SPIDER_options["N2_ppm"]) 
    O2_ppm            = str(SPIDER_options["O2_ppm"]) 
    S_ppm             = str(SPIDER_options["S_ppm"])  
    He_ppm            = str(SPIDER_options["He_ppm"]) 

    # SPIDER call sequence 
    call_sequence = [   
                        "spider", 
                        "-options_file", "spider_input.opts", 
                        "-initial_condition",     start_condition, 
                        "-SURFACE_BC",            SURFACE_BC, 
                        "-surface_bc_value",      heat_flux, 
                        "-SOLVE_FOR_VOLATILES",   SOLVE_FOR_VOLATILES, 
                        "-tsurf_poststep_change", tsurf_poststep_change, 
                        "-nstepsmacro",           nstepsmacro, 
                        "-dtmacro",               dtmacro, 
                        "-radius",                R_solid_planet, 
                        "-coresize",              planet_coresize
                    ]

    # Define ppm volatiles only for init loop
    if start_condition == "1":
        call_sequence.extend([
                        "-H2O_initial",           H2O_ppm, 
                        "-CO2_initial",           CO2_ppm, 
                        "-H2_initial",            H2_ppm, 
                        "-N2_initial",            N2_ppm, 
                        "-CH4_initial",           CH4_ppm, 
                        "-O2_initial",            O2_ppm, 
                        "-CO_initial",            CO_ppm, 
                        "-S_initial",             S_ppm, 
                        "-He_initial",            He_ppm 
                        ])
    else:
        call_sequence.extend([ 
                        "-activate_poststep", 
                        "-activate_rollback",
                        "-ic_filename",           output_dir+restart_filename,
                        "-H2O_poststep_change",   H2O_poststep_change, 
                        "-CO2_poststep_change",   CO2_poststep_change,
                        "-H2_poststep_change",    H2_poststep_change,
                        "-CH4_poststep_change",   CH4_poststep_change, 
                        "-CO_poststep_change",    CO_poststep_change, 
                        "-N2_poststep_change",    N2_poststep_change, 
                        "-O2_poststep_change",    O2_poststep_change, 
                        "-S_poststep_change",     S_poststep_change, 
                        "-He_poststep_change",    He_poststep_change 
                        ])

    # Runtime info
    PrintSeparator()
    print("SPIDER run, loop ", loop_counter, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), "| flags:")
    for flag in call_sequence:
        print(flag, end =" ")
    print()
    PrintSeparator()

    # Call SPIDER
    subprocess.call(call_sequence)

    # Update helpfile
    runtime_helpfile = UpdateHelpfile(loop_counter, output_dir, runtime_helpfile_name, runtime_helpfile)

    return SPIDER_options, runtime_helpfile, runtime_helpfile.iloc[-1]["Time"]
    # / RunSPIDER

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
    print("Plot current evolution, ", end=" ")
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

    # Globale properties for all timesteps
    plot_global.plot_global(output_dir)   

    # Specific timesteps for paper plots
    plot_interior.plot_interior(plot_times)     
    plot_atmosphere.plot_atmosphere(output_dir)
    plot_stacked.plot_stacked(output_dir)
    
    # One plot per timestep for video files
    plot_atmosphere.plot_current_mixing_ratio(output_dir) 

def SaveOutput( output_dir ):

    # Copy old files to separate folder
    save_dir = coupler_utils.make_output_dir( output_dir ) #
    print("===> Copy files to separate dir for this run to:", save_dir)
    shutil.copy(output_dir+"spider_input.opts", save_dir+"spider_input.opts")
    for file in natsorted(glob.glob(output_dir+"*.*")):
        shutil.copy(file, save_dir+os.path.basename(file))
        print(os.path.basename(file), end =" ")

