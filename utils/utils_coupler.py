#!/usr/bin/env python3

# Import utils-specific modules
from utils.modules_utils import *

import utils.cpl_atmosphere as cpl_atmosphere
import utils.cpl_global as cpl_global
import utils.cpl_stacked as cpl_stacked
import utils.cpl_interior as cpl_interior

# Henry's law coefficients
# Add to dataframe + save to disk
volatile_distribution_coefficients = {               # X_henry -> ppm/Pa
    # 'H2O_henry':           6.800e-02,              # Lebrun+13
    # 'H2O_henry_pow':       1.4285714285714286,     # Lebrun+13         
    # 'CO2_henry':           4.4E-6,                 # Lebrun+13
    # 'CO2_henry_pow':       1.0,                    # Lebrun+13
    'H2O_henry':             1.033e+00,                
    'H2O_henry_pow':         1.747e+00,       
    'CO2_henry':             1.937e-09,            
    'CO2_henry_pow':         7.140e-01,                
    'H2_henry':              2.572e-06, 
    'H2_henry_pow':          1.0, 
    'CH4_henry':             9.937e-08, 
    'CH4_henry_pow':         1.0, 
    'CO_henry':              1.600e-07, 
    'CO_henry_pow':          1.0, 
    'N2_henry':              7.000e-05, 
    'N2_henry_pow':          1.8,
    'N2_henry_reduced':      7.416e+01,
    'N2_henry_pow_reduced':  4.582e+00,
    'O2_henry':              0.001E-10, 
    'O2_henry_pow':          1.0, 
    'S_henry':               5.000e-03, 
    'S_henry_pow':           1.0, 
    'He_henry':              0.001E-9, 
    'He_henry_pow':          1.0,
    # 'H2O_kdist':             1.0E-4,                 # distribution coefficients
    # 'H2O_kabs':              0.01,                   # absorption (m^2/kg)
    # 'CO2_kdist':             5.0E-4, 
    # 'CO2_kabs':              0.05,
    'H2O_kdist':             0.0E-0,
    'H2O_kabs':              0.00,
    'CO2_kdist':             0.0E-0,  
    'CO2_kabs':              0.00,
    'H2_kdist':              0.0E-0,  
    'H2_kabs':               0.00,     
    'N2_kdist':              0.0E-0,  
    'N2_kabs':               0.00,     
    'CH4_kdist':             0.0E-0, 
    'CH4_kabs':              0.00,    
    'CO_kdist':              0.0E-0,  
    'CO_kabs':               0.00,     
    'O2_kdist':              0.0E-0,  
    'O2_kabs':               0.00,     
    'S_kdist':               0.0E-0,   
    'S_kabs':                0.00,      
    'He_kdist':              0.0E-0,  
    'He_kabs':               0.00
    }

# Spectral bands for stellar fluxes, in nm
star_bands = {
    "xr" : [1.e-3 , 10.0],  # X-ray,  defined by mors
    "e1" : [10.0  , 32.0],  # EUV1,   defined by mors
    "e2" : [32.0  , 92.0],  # EUV2,   defined by mors
    "uv" : [92.0  , 400.0], # UV,     defined by me
    "pl" : [400.0 , 1.e9],  # planck, defined by me
    'bo' : [1.e-3 , 1.e9]   # bolo,   defined by me
}

# https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

def PrintCurrentState(time_dict, runtime_helpfile, COUPLER_options, atm, loop_counter, dirs):

    # Print final statement
    print("---------------------------------------------------------")
    print("==> RUNTIME INFO <==")
    print(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print("LOOP:", loop_counter)
    print("Time [Myr]:", str(float(time_dict["planet"])/1e6))
    print("T_s [K]:", runtime_helpfile.iloc[-1]["T_surf"])
    print("Phi_global:", runtime_helpfile.iloc[-1]["Phi_global"])
    print("Helpfile properties:")
    print(runtime_helpfile.tail(10))
    print("P_surf [bar]:", runtime_helpfile.iloc[-1]["P_surf"], " ")
    print("TOA heating [W/m^2]:", atm.toa_heating)
    print("F_int [W/m^2]:", COUPLER_options["F_int"])
    print("F_atm [W/m^2]:", COUPLER_options["F_atm"])
    print("F_net [W/m^2]:", COUPLER_options["F_net"])
    print("Last file name:", COUPLER_options["ic_interior_filename"])
    print("---------------------------------------------------------")

    # Save atm object to disk
    with open(dirs["output"]+"/"+str(int(time_dict["planet"]))+"_atm.pkl", "wb") as atm_file: pkl.dump(atm, atm_file)


def UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, input_flag, COUPLER_options):

    runtime_helpfile_name = "runtime_helpfile.csv"
    COUPLER_options_name  = "COUPLER_options.csv"

    # If runtime_helpfle not existent, create it + write to disk
    if not os.path.isfile(dirs["output"]+"/"+runtime_helpfile_name):
        runtime_helpfile = pd.DataFrame(columns=['Time', 'Input', 'T_surf', 'F_int', 'F_atm', 'F_net', 'P_surf', 'M_atm', 'M_atm_kgmol', 'Phi_global', 'RF_depth', 'M_mantle', 'M_core', 'M_mantle_liquid', 'M_mantle_solid', 'H_mol_atm', 'H_mol_solid', 'H_mol_liquid', 'H_mol_total', 'O_mol_total', 'C_mol_total', 'N_mol_total', 'S_mol_total', 'He_mol_total', 'O/H_atm', 'C/H_atm', 'N/H_atm', 'S/H_atm', 'He/H_atm', 'H2O_mr', 'CO2_mr', 'H2_mr', 'CO_mr', 'CH4_mr', 'N2_mr', 'O2_mr', 'S_mr', 'He_mr'])
        runtime_helpfile.to_csv( dirs["output"]+"/"+runtime_helpfile_name, index=False, sep=" ") 
        time_dict["planet"] = 0
        #, 'H2O_atm_bar', 'CO2_atm_bar', 'H2_atm_bar', 'CH4_atm_bar', 'CO_atm_bar', 'N2_atm_bar', 'O2_atm_bar', 'S_atm_bar', 'He_atm_bar'run

        # Save coupler options to file
        COUPLER_options_save = pd.DataFrame(COUPLER_options, index=[0])
        COUPLER_options_save.to_csv( dirs["output"]+"/"+COUPLER_options_name, index=False, sep=" ")

    # Data dict
    runtime_helpfile_new = {}

    # print(runtime_helpfile)

    # For "Interior" sub-loop (SPIDER)
    if input_flag == "Interior":

        ### Read in last SPIDER base parameters
        sim_times = su.get_all_output_times(dirs["output"])  # yr
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
                   ('rheological_front_dynamic','depth'),
                   )

        data_a = su.get_dict_surface_values_for_specific_time( keys_t, sim_time, indir=dirs["output"] )

        # Fill the new dict
        runtime_helpfile_new["Time"]  = sim_time
        runtime_helpfile_new["Input"] = input_flag

        # Mass properties
        runtime_helpfile_new["M_mantle_liquid"] = float(data_a[0])
        runtime_helpfile_new["M_mantle_solid"]  = float(data_a[1])
        runtime_helpfile_new["M_mantle"]        = float(data_a[2]        )
        runtime_helpfile_new["M_core"]          = float(data_a[3]         )

        # Surface properties
        runtime_helpfile_new["T_surf"]          = float(data_a[4])
        runtime_helpfile_new["Phi_global"]      = float(data_a[5])  # global melt fraction
        runtime_helpfile_new["F_int"]           = float(data_a[6])  # Heat flux from interior
        runtime_helpfile_new["P_surf"]          = float(data_a[7])  # total surface pressure
        runtime_helpfile_new["RF_depth"]        = float(data_a[8])/COUPLER_options["radius"]  # depth of rheological front

        # Manually calculate heat flux at near-surface from energy gradient
        json_file   = su.MyJSON( dirs["output"]+'/{}.json'.format(sim_time) )
        Etot        = json_file.get_dict_values(['data','Etot_b'])
        rad         = json_file.get_dict_values(['data','radius_b'])
        area        = json_file.get_dict_values(['data','area_b'])
        E0          = Etot[1] - (Etot[2]-Etot[1]) * (rad[2]-rad[1]) / (rad[1]-rad[0])
        F_int2      = E0/area[0]
        print(">>>>>>> F_int2:", F_int2, "F_int:", runtime_helpfile_new["F_int"] )
        # Limit F_int to positive values
        runtime_helpfile_new["F_int"] = np.amax([F_int2, 0.])


        # Check and replace NaNs
        if np.isnan(runtime_helpfile_new["T_surf"]):
            json_file_time = su.MyJSON( dirs["output"]+'/{}.json'.format(sim_time) )
            int_tmp   = json_file_time.get_dict_values(['data','temp_b'])
            print("Replace T_surf NaN:", runtime_helpfile_new["T_surf"], "-->", int_tmp[0], "K")
            runtime_helpfile_new["T_surf"] = int_tmp[0]

        # Total atmospheric mass
        runtime_helpfile_new["M_atm"] = 0

        # Now volatile data
        for vol in volatile_species:

            # Instantiate empty
            runtime_helpfile_new[vol+"_mr"]     = 0.

            if COUPLER_options[vol+"_included"]:

                keys_t = ( 
                            ('atmosphere',vol,'liquid_kg'),
                            ('atmosphere',vol,'solid_kg'),
                            ('atmosphere',vol,'atmosphere_kg'),
                            ('atmosphere',vol,'atmosphere_bar'),
                            ('atmosphere',vol,'mixing_ratio')  
                         )
                
                data_a = su.get_dict_surface_values_for_specific_time( keys_t, sim_time, indir=dirs["output"] )

                runtime_helpfile_new[vol+"_liquid_kg"]  = float(data_a[0])
                runtime_helpfile_new[vol+"_solid_kg"]   = float(data_a[1])
                runtime_helpfile_new[vol+"_atm_kg"]     = float(data_a[2])
                runtime_helpfile_new[vol+"_atm_bar"]    = float(data_a[3])
                runtime_helpfile_new[vol+"_mr"]         = float(data_a[4])

                # Total mass of atmosphere
                runtime_helpfile_new["M_atm"] += runtime_helpfile_new[vol+"_atm_kg"]
                # print(vol, runtime_helpfile_new[vol+"_atm_kg"])

        ## Derive X/H ratios for atmosphere from interior outgassing

        # Number of mols per species and reservoir
        for vol in volatile_species:

            # Total and baseline
            
            runtime_helpfile_new[vol+"_mol_atm"]    = 0.
            runtime_helpfile_new[vol+"_mol_solid"]  = 0.
            runtime_helpfile_new[vol+"_mol_liquid"] = 0.
            runtime_helpfile_new[vol+"_mol_total"]  = 0.
            
            # Only for the ones tracked in SPIDER
            if COUPLER_options[vol+"_included"]:
                runtime_helpfile_new[vol+"_mol_atm"]    = runtime_helpfile_new[vol+"_atm_kg"] / molar_mass[vol]
                runtime_helpfile_new[vol+"_mol_solid"]  = runtime_helpfile_new[vol+"_solid_kg"] / molar_mass[vol]
                runtime_helpfile_new[vol+"_mol_liquid"] = runtime_helpfile_new[vol+"_liquid_kg"] / molar_mass[vol]
                runtime_helpfile_new[vol+"_mol_total"] = runtime_helpfile_new[vol+"_mol_atm"]      \
                                                         + runtime_helpfile_new[vol+"_mol_solid"]  \
                                                         + runtime_helpfile_new[vol+"_mol_liquid"]

        runtime_helpfile_new["M_atm_kgmol"] = 0.

        # Number of mols per element and reservoir
        for res in [ "total", "solid", "liquid", "atm" ]: 
            runtime_helpfile_new["H_mol_"+res]  = runtime_helpfile_new["H2O_mol_"+res] * 2. \
                                                + runtime_helpfile_new["H2_mol_"+res]  * 2. \
                                                + runtime_helpfile_new["CH4_mol_"+res] * 4. 
            if ('O' in element_list):
                runtime_helpfile_new["O_mol_"+res]  = runtime_helpfile_new["H2O_mol_"+res] * 1. \
                                                    + runtime_helpfile_new["CO2_mol_"+res] * 2. \
                                                    + runtime_helpfile_new["CO_mol_"+res]  * 1. \
                                                    + runtime_helpfile_new["O2_mol_"+res]  * 2.
            if ('C' in element_list):
                runtime_helpfile_new["C_mol_"+res]  = runtime_helpfile_new["CO2_mol_"+res] * 1. \
                                                    + runtime_helpfile_new["CH4_mol_"+res] * 1. \
                                                    + runtime_helpfile_new["CO_mol_"+res]  * 1.
            if ('N' in element_list):
                runtime_helpfile_new["N_mol_"+res]  = runtime_helpfile_new["N2_mol_"+res]  * 2.
            if ('S' in element_list):
                runtime_helpfile_new["S_mol_"+res]  = runtime_helpfile_new["S_mol_"+res]   * 1.
            if ('He' in element_list):
                runtime_helpfile_new["He_mol_"+res] = runtime_helpfile_new["He_mol_"+res]  * 1.

            if res == "atm":
                runtime_helpfile_new["M_atm_kgmol"] = 0.0
                for elem in element_list:
                    runtime_helpfile_new["M_atm_kgmol"] += runtime_helpfile_new[elem+"_mol_"+res] * molar_mass[elem]

        # Avoid division by 0
        min_val     = 1e-99
        for elem in element_list:
            runtime_helpfile_new[elem+"_mol_atm"] = np.max([runtime_helpfile_new[elem+"_mol_atm"], min_val])
            break

        # Calculate X/H ratios
        for element in [n for n in element_list if n != 'H']:
            runtime_helpfile_new[element+"/H_atm"] = runtime_helpfile_new[element+"_mol_atm"]  / runtime_helpfile_new["H_mol_atm"]
            runtime_helpfile_new[element+"/H_atm"] = np.max([runtime_helpfile_new[element+"/H_atm"], min_val])

        COUPLER_options["F_int"]      = runtime_helpfile_new["F_int"]

        # F_atm from before
        if loop_counter["total"] >= loop_counter["init_loops"]:
            run_atm = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere']
            COUPLER_options["F_atm"] = run_atm["F_atm"].iloc[-1]
        else:
            COUPLER_options["F_atm"]      = 0.
        
        COUPLER_options["F_net"]      = COUPLER_options["F_atm"]-COUPLER_options["F_int"]
        runtime_helpfile_new["F_atm"] = COUPLER_options["F_atm"]
        runtime_helpfile_new["F_net"] = COUPLER_options["F_net"]

    # For "Atmosphere" sub-loop (VULCAN+SOCRATES) update heat flux from SOCRATES
    elif input_flag == "Atmosphere":

        # Define input flag
        runtime_helpfile_new["Input"]           = input_flag   

        # Infos from latest interior loop
        run_int = runtime_helpfile.loc[runtime_helpfile['Input']=='Interior']
        runtime_helpfile_new["Phi_global"]      = run_int.iloc[-1]["Phi_global"]
        runtime_helpfile_new["RF_depth"]        = run_int.iloc[-1]["RF_depth"]     
        runtime_helpfile_new["M_mantle"]        = run_int.iloc[-1]["M_mantle"]       
        runtime_helpfile_new["M_core"]          = run_int.iloc[-1]["M_core"]         
        runtime_helpfile_new["M_mantle_liquid"] = run_int.iloc[-1]["M_mantle_liquid"]
        runtime_helpfile_new["M_mantle_solid"]  = run_int.iloc[-1]["M_mantle_solid"]
        runtime_helpfile_new["Time"]            = run_int.iloc[-1]["Time"]
        runtime_helpfile_new["F_int"]           = run_int.iloc[-1]["F_int"]

        # From latest atmosphere iteration
        runtime_helpfile_new["T_surf"]          = COUPLER_options["T_surf"] 
        runtime_helpfile_new["F_atm"]           = COUPLER_options["F_atm"]

        COUPLER_options["F_int"] = run_int.iloc[-1]["F_int"]
        COUPLER_options["F_net"] = COUPLER_options["F_atm"] - COUPLER_options["F_int"]

        ### Adjust F_net to break atm main loop:
        t_curr          = run_int.iloc[-1]["Time"]
        run_atm         = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere']
        run_atm_last    = run_atm.loc[run_atm['Time'] != t_curr]
        # IF in early MO phase and RF is deep in mantle
        if runtime_helpfile_new["RF_depth"] >= COUPLER_options["RF_crit"]:
            COUPLER_options["F_net"] = -COUPLER_options["F_eps"]
            print("Early MO phase and RF is deep in mantle. RF_depth = ", runtime_helpfile.iloc[-1]["RF_depth"])
        if loop_counter["init"] >= loop_counter["init_loops"]:
            
            if loop_counter["init"] == loop_counter["init_loops"]:
                Ts_last         = runtime_helpfile.iloc[-1]["T_surf"]

            else:
                Ts_last         = run_atm_last.iloc[-1]["T_surf"]

            # IF T_surf change too high
            if abs(Ts_last-COUPLER_options["T_surf"]) >= COUPLER_options["dTs_atm"]: 
                COUPLER_options["F_net"] = -COUPLER_options["F_eps"]   
                print("T_surf change too high. dT =", Ts_last-COUPLER_options["T_surf"])
            # OR IF negligible change in F_atm in the last two entries
            if round(COUPLER_options["F_atm"],2) == round(run_atm.iloc[-1]["F_atm"],2):
                COUPLER_options["F_net"] = -COUPLER_options["F_eps"]
                print("Negligible change in F_atm in the last two entries. F_atm(curr/-1) = ", round(COUPLER_options["F_atm"],2), round(run_atm.iloc[-1]["F_atm"],2))

        # Write F_net to next file
        runtime_helpfile_new["F_net"]           = COUPLER_options["F_net"]

        # Other info from latest iteration run (X/H ratios stay fixed w/o loss)
        runtime_helpfile_new["P_surf"]          = runtime_helpfile.iloc[-1]["P_surf"]         
        runtime_helpfile_new["M_atm"]           = runtime_helpfile.iloc[-1]["M_atm"]
        runtime_helpfile_new["M_atm_kgmol"]     = runtime_helpfile.iloc[-1]["M_atm_kgmol"]
        for res in [ "total", "solid", "liquid", "atm" ]: 
            for elem in element_list:
                runtime_helpfile_new[elem+"_mol_"+res]       = runtime_helpfile.iloc[-1][elem+"_mol_"+res]
        # runtime_helpfile_new["H_mol_atm"]       = runtime_helpfile.iloc[-1]["H_mol_atm"]
        # runtime_helpfile_new["H_mol_solid"]     = runtime_helpfile.iloc[-1]["H_mol_solid"]
        # runtime_helpfile_new["H_mol_liquid"]    = runtime_helpfile.iloc[-1]["H_mol_liquid"] 
        # runtime_helpfile_new["H_mol_total"]     = runtime_helpfile.iloc[-1]["H_mol_total"] 
        # runtime_helpfile_new["O_mol_total"]     = runtime_helpfile.iloc[-1]["O_mol_total"]
        # runtime_helpfile_new["C_mol_total"]     = runtime_helpfile.iloc[-1]["C_mol_total"]   
        # runtime_helpfile_new["N_mol_total"]     = runtime_helpfile.iloc[-1]["N_mol_total"]   
        # runtime_helpfile_new["S_mol_total"]     = runtime_helpfile.iloc[-1]["S_mol_total"]   
        # runtime_helpfile_new["He_mol_total"]    = runtime_helpfile.iloc[-1]["He_mol_total"]
        for elem in [n for n in element_list if n != 'H']:
            runtime_helpfile_new[elem+"/H_atm"]         = runtime_helpfile.iloc[-1][elem+"/H_atm"]

        for vol in volatile_species:
            runtime_helpfile_new[vol+"_mr"]          = runtime_helpfile.iloc[-1][vol+"_mr"]
        # runtime_helpfile_new["H2O_mr"]          = runtime_helpfile.iloc[-1]["H2O_mr"]
        # runtime_helpfile_new["CO2_mr"]          = runtime_helpfile.iloc[-1]["CO2_mr"]
        # runtime_helpfile_new["H2_mr"]           = runtime_helpfile.iloc[-1]["H2_mr"]
        # runtime_helpfile_new["CO_mr"]           = runtime_helpfile.iloc[-1]["CO_mr"]
        # runtime_helpfile_new["CH4_mr"]          = runtime_helpfile.iloc[-1]["CH4_mr"]
        # runtime_helpfile_new["N2_mr"]           = runtime_helpfile.iloc[-1]["N2_mr"]
        # runtime_helpfile_new["O2_mr"]           = runtime_helpfile.iloc[-1]["O2_mr"]
        # runtime_helpfile_new["S_mr"]            = runtime_helpfile.iloc[-1]["S_mr"]
        # runtime_helpfile_new["He_mr"]           = runtime_helpfile.iloc[-1]["He_mr"]    
    
    # Add all parameters to dataframe + update file on disk
    # runtime_helpfile_new = pd.DataFrame({
    #     'Time':             runtime_helpfile_new["Time"],
    #     'Input':            runtime_helpfile_new["Input"],
    #     'T_surf':           runtime_helpfile_new["T_surf"],
    #     'F_int':            runtime_helpfile_new["F_int"],
    #     'F_atm':            runtime_helpfile_new["F_atm"],
    #     'F_net':            runtime_helpfile_new["F_net"],
    #     'P_surf':           runtime_helpfile_new["P_surf"],
    #     'M_atm':            runtime_helpfile_new["M_atm"],
    #     'M_atm_kgmol':      runtime_helpfile_new["M_atm_kgmol"],
    #     'Phi_global':       runtime_helpfile_new["Phi_global"],
    #     'RF_depth':         runtime_helpfile_new["RF_depth"],
    #     'M_mantle':         runtime_helpfile_new["M_mantle"],
    #     'M_core':           runtime_helpfile_new["M_core"],
    #     'M_mantle_liquid':  runtime_helpfile_new["M_mantle_liquid"],
    #     'M_mantle_solid':   runtime_helpfile_new["M_mantle_solid"],
    #     'H_mol_atm':        runtime_helpfile_new["H_mol_atm"],
    #     'H_mol_solid':      runtime_helpfile_new["H_mol_solid"],
    #     'H_mol_liquid':     runtime_helpfile_new["H_mol_liquid"],
    #     'H_mol_total':      runtime_helpfile_new["H_mol_total"],
    #     'O_mol_total':      runtime_helpfile_new["O_mol_total"],
    #     'C_mol_total':      runtime_helpfile_new["C_mol_total"],
    #     'N_mol_total':      runtime_helpfile_new["N_mol_total"],
    #     'S_mol_total':      runtime_helpfile_new["S_mol_total"],
    #     'He_mol_total':     runtime_helpfile_new["He_mol_total"],
    #     'O/H_atm':          runtime_helpfile_new["O/H_atm"],
    #     'C/H_atm':          runtime_helpfile_new["C/H_atm"],
    #     'N/H_atm':          runtime_helpfile_new["N/H_atm"],
    #     'S/H_atm':          runtime_helpfile_new["S/H_atm"],
    #     'He/H_atm':         runtime_helpfile_new["He/H_atm"],
    #     'H2O_mr':           runtime_helpfile_new["H2O_mr"],
    #     'CO2_mr':           runtime_helpfile_new["CO2_mr"],
    #     'H2_mr':            runtime_helpfile_new["H2_mr"],
    #     'CO_mr':            runtime_helpfile_new["CO_mr"],
    #     'CH4_mr':           runtime_helpfile_new["CH4_mr"],
    #     'N2_mr':            runtime_helpfile_new["N2_mr"],
    #     'O2_mr':            runtime_helpfile_new["O2_mr"],
    #     'S_mr':             runtime_helpfile_new["S_mr"],
    #     'He_mr':            runtime_helpfile_new["He_mr"],
    #     }, index=[0])

    runtime_helpfile_new = pd.DataFrame(runtime_helpfile_new,index=[0])
    runtime_helpfile = pd.concat([runtime_helpfile, runtime_helpfile_new])

    print(dirs["output"]+"/"+runtime_helpfile_name)
    runtime_helpfile.to_csv( dirs["output"]+"/"+runtime_helpfile_name, index=False, sep=" ")

    # Save COUPLER_options to disk
    COUPLER_options_save = pd.read_csv(dirs["output"]+"/"+COUPLER_options_name, sep=" ")
    COUPLER_options_df = pd.DataFrame.from_dict([COUPLER_options])
    COUPLER_options_save = pd.concat([ COUPLER_options_save, COUPLER_options_df],ignore_index=True)
    COUPLER_options_save.to_csv( dirs["output"]+"/"+COUPLER_options_name, index=False, sep=" ")

    # Advance time_current in main loop
    time_dict["planet"] = runtime_helpfile.iloc[-1]["Time"]
    time_dict["star"]   = time_dict["planet"] + time_dict["offset"]

    return runtime_helpfile, time_dict, COUPLER_options

def PrintSeparator():
    print("-------------------------------------------------------------------------------------------------------------")
    pass

def PrintHalfSeparator():
    print("--------------------------------------------------")
    pass

# Calculate eqm temperature given stellar flux and bond albedo
def calc_eqm_temperature(I_0, A_B):
    return (I_0 * (1.0 - A_B) / (4.0 * phys.sigma))**(1.0/4.0)

# Generate/adapt atmosphere chemistry/radiation input files
def StructAtm( loop_counter, dirs, runtime_helpfile, COUPLER_options ):

    # In the beginning: standard surface temperature from last entry
    if loop_counter["total"] < loop_counter["init_loops"]:
        COUPLER_options["T_surf"] = runtime_helpfile.iloc[-1]["T_surf"]
    
    # Check for flux_convergence scheme criteria
    elif (COUPLER_options["flux_convergence"] == 1 \
    and runtime_helpfile.iloc[-1]["RF_depth"] < COUPLER_options["RF_crit"] \
    and COUPLER_options["F_net"] > COUPLER_options["F_diff"]*COUPLER_options["F_int"]) \
    or  COUPLER_options["flux_convergence"] == 2:

        PrintSeparator()
        print(">>>>>>>>>> Flux convergence scheme <<<<<<<<<<<")

        COUPLER_options["flux_convergence"] = 2

        # In case last atm T_surf from flux convergence scheme was smaller(!) than threshold 
        if abs(COUPLER_options["F_net"]) < COUPLER_options["F_eps"]:
            
            COUPLER_options["T_surf"] = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere'].iloc[-1]["T_surf"]
            print("Use previous T_surf =", COUPLER_options["T_surf"])

        else:

            # Last T_surf and time from atmosphere, K
            t_curr          = runtime_helpfile.iloc[-1]["Time"]
            run_atm         = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere']
            run_atm_prev    = run_atm.loc[run_atm['Time'] != t_curr]
            run_atm_curr    = run_atm.loc[run_atm['Time'] == t_curr]
            t_previous_atm  = run_atm_prev.iloc[-1]["Time"]
            Ts_previous_atm = run_atm_prev.iloc[-1]["T_surf"]
            Ts_last_atm     = run_atm.iloc[-1]["T_surf"]

            print("F_net", str(COUPLER_options["F_net"]), "Ts_previous_atm:", Ts_previous_atm, "Ts_last_atm", Ts_last_atm, "dTs_atm", str(COUPLER_options["dTs_atm"]), "t_curr", t_curr, "t_previous_atm", t_previous_atm)

            # Apply flux convergence via shallow layer function
            COUPLER_options["T_surf"] = shallow_mixed_ocean_layer(COUPLER_options["F_net"], Ts_previous_atm, COUPLER_options["dTs_atm"], t_curr, t_previous_atm)

            # Prevent atmospheric oscillations
            if len(run_atm_curr) > 2 and (np.sign(run_atm_curr["F_net"].iloc[-1]) != np.sign(run_atm_curr["F_net"].iloc[-2])) and (np.sign(run_atm_curr["F_net"].iloc[-2]) != np.sign(run_atm_curr["F_net"].iloc[-3])):
                COUPLER_options["T_surf"] = np.mean([run_atm.iloc[-1]["T_surf"], run_atm.iloc[-2]["T_surf"]])
                print("Prevent oscillations, new T_surf =", COUPLER_options["T_surf"])

            print("dTs_atm (K):", COUPLER_options["dTs_atm"], "t_previous_atm:", t_previous_atm, "Ts_previous_atm:", Ts_previous_atm, "Ts_last_atm:", Ts_last_atm, "t_curr:", t_curr, "Ts_curr:", COUPLER_options["T_surf"])

        PrintSeparator()

    # Use Ts_int
    else:
        # Standard surface temperature from last entry
        COUPLER_options["T_surf"] = runtime_helpfile.iloc[-1]["T_surf"]

    # Create atmosphere object and set parameters
    pl_radius = COUPLER_options["radius"]
    pl_mass = COUPLER_options["gravity"] * pl_radius * pl_radius / phys.G

    vol_list = { 
                  "H2O" : runtime_helpfile.iloc[-1]["H2O_mr"], 
                  "CO2" : runtime_helpfile.iloc[-1]["CO2_mr"],
                  "H2"  : runtime_helpfile.iloc[-1]["H2_mr"], 
                  "N2"  : runtime_helpfile.iloc[-1]["N2_mr"],  
                  "CH4" : runtime_helpfile.iloc[-1]["CH4_mr"], 
                  "O2"  : runtime_helpfile.iloc[-1]["O2_mr"], 
                  "CO"  : runtime_helpfile.iloc[-1]["CO_mr"], 
                  "He"  : 0.,
                  "NH3" : 0., 
                }

    atm = atmos(COUPLER_options["T_surf"], runtime_helpfile.iloc[-1]["P_surf"]*1e5, 
                COUPLER_options["P_top"]*1e5, pl_radius, pl_mass,
                vol_mixing=vol_list
                )

    atm.zenith_angle    = COUPLER_options["zenith_angle"]
    atm.albedo_pl       = COUPLER_options["albedo_pl"]
    atm.albedo_s        = COUPLER_options["albedo_s"]
        

    return atm, COUPLER_options

# run VULCAN/atmosphere chemistry
def RunVULCAN( atm, time_dict, loop_counter, dirs, runtime_helpfile, COUPLER_options ):

    # Runtime info
    PrintSeparator()
    print("VULCAN run... (loop =", loop_counter, ")")
    PrintSeparator()

    # Copy template file
    vul_cfg = dirs["vulcan"]+"vulcan_cfg.py"                    # Template configuration file for VULCAN
    if os.path.exists(vul_cfg):
        os.remove(vul_cfg)
    shutil.copyfile(dirs["utils"]+"init_vulcan_cfg.py",vul_cfg)

    # Delete old PT profile
    vul_ptp = dirs["vulcan"]+"output/PROTEUS_PT_input.txt"      # Path to input PT profile
    if os.path.exists(vul_ptp):
        os.remove(vul_ptp)


    hf_recent = runtime_helpfile.iloc[-1]

    # Write missing parameters to cfg file
    with open(vul_cfg, 'a') as vcf:

        vcf.write("# < PROTEUS INSERT > \n")

        # System/planet parameters
        vcf.write("Rp = %1.5e \n"           % (atm.planet_radius*100.0))        # Radius [cm]
        vcf.write("gs = %g \n"              % (atm.grav_s*100.0))               # Surface gravity [cm/s^2]
        vcf.write("sl_angle = %g \n"        % (atm.zenith_angle*3.141/180.0))   # Solar zenith angle [rad]
        vcf.write("orbit_radius = %1.5e \n" % COUPLER_options["mean_distance"]) # Semi major axis [AU]
        vcf.write("r_star = %1.5e \n"       % COUPLER_options["star_radius"])   # Star's radius [R_sun]

        # Set background gas based on gas with highest mixing ratio from list of options
        bg_gas = np.array([ 'H2', 'N2', 'O2', 'CO2' ])
        bg_val = np.array([ hf_recent["%s_mr"%gas] for gas in bg_gas ])  # Get values
        bg_mask = np.argsort(bg_val)  # Get sorting mask
        bg_val = bg_val[bg_mask]   # Sort gas values
        bg_gas = bg_gas[bg_mask]   # Sort gas names
        if (bg_val[-1] < 2.0*bg_val[-2]) or (bg_val[-1] < 0.5):
            print("Warning: Background gas '%s' is not significantly abundant!"%bg_gas[-1])
            print("         Mixing ratio of '%s' is %g" % (bg_gas[-1],bg_val[-1]))
            print("         Mixing ratio of '%s' is %g" % (bg_gas[-2],bg_val[-2]))
        vcf.write("atm_base = '%s' \n"      % str(bg_gas[-1]))

        # Pressure grid limits
        vcf.write("P_b = %1.5e \n"          % float(hf_recent["P_surf"]*1.0e6))        # pressure at the bottom (dyne/cm^2)
        vcf.write("P_t = %1.5e \n"          % float(COUPLER_options["P_top"]*1.0e6))   # pressure at the top (dyne/cm^2)

        # Plotting behaviour
        vcf.write("use_live_plot  = %s \n"  % str(bool(COUPLER_options["plot_onthefly"] == 1)))

        # Make copy of element_list as a set, since it'll be used a lot in the code below
        set_elem_list = set(element_list)  

        # Rayleigh scattering gases
        rayleigh_candidates = ['N2','O2', 'H2']
        rayleigh_str = ""
        for rc in rayleigh_candidates:  
            if set(re.sub('[1-9]', '', rc)).issubset(set_elem_list):  # Remove candidates which aren't supported by elem_list
                rayleigh_str += "'%s',"%rc
        rayleigh_str = rayleigh_str[:-1]
        vcf.write("scat_sp = [%s] \n" % rayleigh_str)

        # Gases for diffusion-limit escape at TOA      
        # escape_candidates = ['H2','H']
        # escape_str = ""
        # for ec in escape_candidates:  
        #     if set(re.sub('[1-9]', '', ec)).issubset(set_elem_list):  # Remove candidates which aren't supported by elem_list
        #         escape_str += "'%s',"%ec
        # escape_str = escape_str[:-1]
        # vcf.write("diff_esc = [%s] \n" % escape_str)

        # Atom list     
        atom_str = ""
        for elem in element_list:  
            atom_str += "'%s',"%elem
        atom_str = atom_str[:-1]
        vcf.write("atom_list  = [%s] \n" % atom_str)

        # Choose most appropriate chemical network and species-to-plot
        oxidising = False

        for inert in ['He','Xe','Ar','Ne','Kr']:  # Remove inert elements from list, since they don't matter for reactions
            set_elem_list.discard(inert) 

        if set_elem_list == {'H','C','O'}:
            net_str = 'thermo/CHO_photo_network.txt'
            plt_spe = ['H2', 'H', 'H2O', 'C2H2', 'CH4', 'CO2'] 

        elif set_elem_list == {'N','H','C','O'}:
            if oxidising:
                net_str = 'thermo/NCHO_full_photo_network.txt'
                plt_spe = ['N2', 'O2', 'H2', 'H2O', 'NH3', 'CH4', 'CO', 'O3'] 
            else:
                net_str = 'thermo/NCHO_photo_network.txt'
                plt_spe = ['H2', 'H', 'H2O', 'OH', 'CH4', 'HCN', 'N2', 'NH3'] 
            
        elif set_elem_list == {'S','N','H','C','O'}:
            if oxidising:
                net_str = 'thermo/SNCHO_full_photo_network.txt'
                plt_spe = ['O2', 'N2', 'O3', 'H2', 'H2O', 'NH3', 'CH4', 'SO2', 'S'] 
            else:
                net_str = 'thermo/SNCHO_photo_network.txt'
                plt_spe = ['N2', 'H2', 'S', 'H', 'OH', 'NH3', 'CH4', 'HCN'] 

        vcf.write("network  = '%s' \n" % net_str)
        plt_str = ""
        for spe in plt_spe:
            plt_str += "'%s'," % spe
        vcf.write("plot_spec = [%s] \n" % plt_str[:-1])
        
        # Bottom boundary mixing ratios are fixed according to SPIDER (??)
        # fix_bb_mr = "{"
        # for v in volatile_species:
        #     fix_bb_mr += " '%s' : %1.5e ," % (v,hf_recent["%s_mr"%v])
        # fix_bb_mr = fix_bb_mr[:-1]+" }"
        # vcf.write("use_fix_sp_bot = %s \n"   % str(fix_bb_mr))
        vcf.write("use_fix_sp_bot = {} \n")


        # Has NOT run atmosphere before
        if (loop_counter["atm"] == 0):

            # PT profile
            # vcf.write("atm_type = 'isothermal' \n")
            # vcf.write("Tiso = %g \n"        % float(COUPLER_options["T_eqm"]))

            # Abundances
            vcf.write("ini_mix = 'const_mix' \n")  # other options: 'EQ', 'table', 'vulcan_ini'
            const_mix = "{"
            for v in volatile_species:
                const_mix += " '%s' : %1.5e ," % (v,hf_recent["%s_mr"%v])
            const_mix = const_mix[:-1]+" }"
            vcf.write("const_mix = %s \n"   % str(const_mix))

        # Has run atmosphere before
        # else:

            # PT Profile
            # vcf.write("atm_type = 'file' \n")
            # vcf.write("atm_file = '%s' \n" % vul_ptp)

            # Abundances
            # vcf.write("ini_mix = 'vulcan_ini' \n")
            # vcf.write("vul_ini = 'output/PROTEUS_MX_input.vul' \n")
            # ! WRITE ABUNDANCES HERE
            

        vcf.write("# </ PROTEUS INSERT > \n")
        vcf.write(" ")

    # Write PT profile (to VULCAN, from AEOLUS)
    vul_PT = np.array(
        [np.array(atm.pl)  [::-1] * 10.0,
         np.array(atm.tmpl)[::-1]
        ]
    ).T
    header = "#(dyne/cm2)\t (K) \n Pressure\t Temp"
    np.savetxt(vul_ptp,vul_PT,delimiter="\t",header=header,comments='',fmt="%1.5e")


    # Switch to VULCAN directory, run VULCAN, switch back to main directory
    vulcan_run_cmd = "python vulcan.py"  
    if (loop_counter["atm"] > 0):      # If not first run, skip building chem_funcs
        vulcan_run_cmd += " -n"

    if debug:
        vulcan_print = sys.stdout
    else:
        vulcan_print = open(dirs["output"]+"vulcan_recent.log",'w')

    os.chdir(dirs["vulcan"])
    subprocess.run([vulcan_run_cmd], shell=True, check=True, stdout=vulcan_print)
    os.chdir(dirs["coupler"])

    if not debug:
        vulcan_print.close()

    # Copy VULCAN output data file to output folder
    vulcan_recent = dirs["output"]+str(int(time_dict["planet"]))+"_atm_chemistry.vul"
    shutil.copyfile(dirs["vulcan"]+'output/PROTEUS_MX_output.vul', vulcan_recent )

    # Read in data from VULCAN output
    with (open(vulcan_recent, "rb")) as vof:
        vul_data = pkl.load(vof)
        
    print(vul_data)

    # < LEGACY CODE >
    # # Update SPIDER restart options w/ surface partial pressures
    # for vol in volatile_species:
        
    #     # Calculate partial pressure from VULCAN output
    #     volume_mixing_ratio     = atm_chemistry.iloc[0][vol]
    #     surface_pressure_total  = atm_chemistry.iloc[0]["Pressure"]*1e5 # bar -> Pa
    #     partial_pressure_vol    = surface_pressure_total*volume_mixing_ratio

    #     # Only for major atmospheric species
    #     if partial_pressure_vol > 1.: # Pa
    #         COUPLER_options[vol+"_initial_atmos_pressure"] = partial_pressure_vol
    # </ LEGACY CODE >

    return atm

def RunAEOLUS( atm, time_dict, dirs, runtime_helpfile, loop_counter, COUPLER_options ):

    # Runtime info
    PrintSeparator()
    print("SOCRATES run... (loop =", loop_counter, ")")
    PrintSeparator()

    # Calculate temperature structure and heat flux w/ SOCRATES
    _, atm = SocRadConv.RadConvEqm(dirs, time_dict, atm, standalone=False, cp_dry=False, trppD=True, rscatter=True,calc_cf=False) # W/m^2
    
    # Atmosphere net flux from topmost atmosphere node; do not allow heating
    COUPLER_options["F_atm"] = np.max( [ 0., atm.net_flux[0] ] )

    # Clean up run directory
    PrintSeparator()
    print("Remove SOCRATES auxiliary files:", end =" ")
    for file in natural_sort(glob.glob(dirs["output"]+"/current??.????")):
        os.remove(file)
        print(os.path.basename(file), end =" ")
    for file in natural_sort(glob.glob(dirs["output"]+"/profile.*")):
        os.remove(file)
        print(os.path.basename(file), end =" ")
    print(">>> Done.")

    return atm, COUPLER_options

def RunSPIDER( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile ):

    SPIDER_options_file = dirs["output"]+"/init_spider.opts"
    SPIDER_options_file_orig = dirs["utils"]+"/init_spider.opts"

    print("IC_INTERIOR =",COUPLER_options["IC_INTERIOR"])

    # First run
    if (loop_counter["init"] == 0):
        if os.path.isfile(SPIDER_options_file):
            os.remove(SPIDER_options_file)
        shutil.copy(SPIDER_options_file_orig,SPIDER_options_file)

    # Define which volatiles to track in SPIDER
    species_call = ""
    for vol in volatile_species: 
        if COUPLER_options[vol+"_included"]:
            species_call = species_call + "," + vol
    species_call = species_call[1:] # Remove "," in front

    # Recalculate time stepping
    if COUPLER_options["IC_INTERIOR"] == 2:  

        # Current step
        json_file   = su.MyJSON( dirs["output"]+'/{}.json'.format(int(time_dict["planet"])) )
        step        = json_file.get_dict(['step'])

        dtmacro     = float(COUPLER_options["dtmacro"])
        dtswitch    = float(COUPLER_options["dtswitch"])

        # Time resolution adjustment in the beginning
        if time_dict["planet"] < 1000:
            dtmacro = 10
            dtswitch = 50
        if time_dict["planet"] < 100:
            dtmacro = 2
            dtswitch = 5
        if time_dict["planet"] < 10:
            dtmacro = 1
            dtswitch = 1

        # Runtime left
        dtime_max   = time_dict["target"] - time_dict["planet"]

        # Limit Atm-Int switch
        dtime       = np.min([ dtime_max, dtswitch ])

        # Number of total steps until currently desired switch/end time
        COUPLER_options["nstepsmacro"] =  step + math.ceil( dtime / dtmacro )

        print("TIME OPTIONS IN RUNSPIDER:")
        print(dtmacro, dtswitch, dtime_max, dtime, COUPLER_options["nstepsmacro"])


    # For init loop
    else:
        dtmacro     = 0

    # Prevent interior oscillations during last-stage freeze-out
    net_loss = COUPLER_options["F_atm"]
    if len(runtime_helpfile) > 100 and runtime_helpfile.iloc[-1]["Phi_global"] <= COUPLER_options["phi_crit"]:
        net_loss = np.amax([abs(COUPLER_options["F_atm"]), COUPLER_options["F_eps"]])
        if debug:
            print("Prevent interior oscillations during last-stage freeze-out: F_atm =", COUPLER_options["F_atm"], "->", net_loss)

    ### SPIDER base call sequence 
    call_sequence = [   
                        dirs["spider"]+"/spider", 
                        "-options_file",          SPIDER_options_file, 
                        "-outputDirectory",       dirs["output"],
                        "-IC_INTERIOR",           str(COUPLER_options["IC_INTERIOR"]),
                        "-IC_ATMOSPHERE",         str(COUPLER_options["IC_ATMOSPHERE"]),
                        "-SURFACE_BC",            str(COUPLER_options["SURFACE_BC"]), 
                        "-surface_bc_value",      str(net_loss), 
                        "-teqm",                  str(COUPLER_options["T_eqm"]), 
                        "-nstepsmacro",           str(COUPLER_options["nstepsmacro"]), 
                        "-dtmacro",               str(dtmacro), 
                        "-radius",                str(COUPLER_options["radius"]), 
                        "-gravity",               "-"+str(COUPLER_options["gravity"]), 
                        "-coresize",              str(COUPLER_options["planet_coresize"]),
                        "-volatile_names",        str(species_call)
                    ]

    # Min of fractional and absolute Ts poststep change
    if time_dict["planet"] > 0:
        dTs_frac = float(COUPLER_options["tsurf_poststep_change_frac"]) * float(runtime_helpfile["T_surf"].iloc[-1])
        dT_int_max = np.min([ float(COUPLER_options["tsurf_poststep_change"]), float(dTs_frac) ])
        call_sequence.extend(["-tsurf_poststep_change", str(dT_int_max)])
    else:
        call_sequence.extend(["-tsurf_poststep_change", str(COUPLER_options["tsurf_poststep_change"])])

    # Define distribution coefficients and total mass/surface pressure for volatiles > 0
    for vol in volatile_species:
        if COUPLER_options[vol+"_included"]:

            # Set atmospheric pressure based on helpfile output
            if loop_counter["total"] > loop_counter["init_loops"]:
                key = vol+"_initial_atmos_pressure"
                val = float(runtime_helpfile[vol+"_mr"].iloc[-1]) * float(runtime_helpfile["P_surf"].iloc[-1]) * 1.0e5   # convert bar to Pa
                COUPLER_options[key] = val

            # Load volatiles
            if COUPLER_options["IC_ATMOSPHERE"] == 1:
                call_sequence.extend(["-"+vol+"_initial_total_abundance", str(COUPLER_options[vol+"_initial_total_abundance"])])
            elif COUPLER_options["IC_ATMOSPHERE"] == 3:
                call_sequence.extend(["-"+vol+"_initial_atmos_pressure", str(COUPLER_options[vol+"_initial_atmos_pressure"])])

            # Exception for N2 case: reduced vs. oxidized
            if vol == "N2" and COUPLER_options["N2_partitioning"] == 1:
                volatile_distribution_coefficients["N2_henry"] = volatile_distribution_coefficients["N2_henry_reduced"]
                volatile_distribution_coefficients["N2_henry_pow"] = volatile_distribution_coefficients["N2_henry_pow_reduced"]

            call_sequence.extend(["-"+vol+"_henry", str(volatile_distribution_coefficients[vol+"_henry"])])
            call_sequence.extend(["-"+vol+"_henry_pow", str(volatile_distribution_coefficients[vol+"_henry_pow"])])
            call_sequence.extend(["-"+vol+"_kdist", str(volatile_distribution_coefficients[vol+"_kdist"])])
            call_sequence.extend(["-"+vol+"_kabs", str(volatile_distribution_coefficients[vol+"_kabs"])])
            call_sequence.extend(["-"+vol+"_molar_mass", str(molar_mass[vol])])
            call_sequence.extend(["-"+vol+"_SOLUBILITY 1"])

    # With start of the main loop only:
    # Volatile specific options: post step settings, restart filename
    if COUPLER_options["IC_INTERIOR"] == 2:
        call_sequence.extend([ 
                                "-ic_interior_filename", 
                                str(dirs["output"]+"/"+COUPLER_options["ic_interior_filename"]),
                                "-activate_poststep", 
                                "-activate_rollback"
                             ])
        for vol in volatile_species:
            if COUPLER_options[vol+"_included"]:
                call_sequence.extend(["-"+vol+"_poststep_change", str(COUPLER_options[vol+"_poststep_change"])])

    # Gravitational separation of solid and melt phase, 0: off | 1: on
    if COUPLER_options["SEPARATION"] == 1:
        call_sequence.extend(["-SEPARATION", str(1)])

    # Mixing length parameterization: 1: variable | 2: constant
    if COUPLER_options["mixing_length"] == 1:
        call_sequence.extend(["-mixing_length", str(1)])

    # Ultra-thin thermal boundary layer at top, 0: off | 1: on
    if COUPLER_options["PARAM_UTBL"] == 1:
        call_sequence.extend(["-PARAM_UTBL", str(1)])
        call_sequence.extend(["-param_utbl_const", str(COUPLER_options["param_utbl_const"])])

    # Check for convergence, if not converging, adjust tolerances iteratively
    if len(runtime_helpfile) > 30 and loop_counter["total"] > loop_counter["init_loops"] :

        # Check convergence for interior cycles
        run_int = runtime_helpfile.loc[runtime_helpfile['Input']=='Interior']

        # First, relax too restrictive dTs
        if run_int["Time"].iloc[-1] == run_int["Time"].iloc[-3]:
            if COUPLER_options["tsurf_poststep_change"] <= 300:
                COUPLER_options["tsurf_poststep_change"] += 10
                print(">>> Raise dT poststep_changes:", COUPLER_options["tsurf_poststep_change"], COUPLER_options["tsurf_poststep_change_frac"])
            else:
                print(">> dTs_int too high! >>", COUPLER_options["tsurf_poststep_change"], "K")
        # Slowly limit again if time advances smoothly
        if (run_int["Time"].iloc[-1] != run_int["Time"].iloc[-3]) and COUPLER_options["tsurf_poststep_change"] > 30:
            COUPLER_options["tsurf_poststep_change"] -= 10
            print(">>> Lower tsurf_poststep_change poststep changes:", COUPLER_options["tsurf_poststep_change"], COUPLER_options["tsurf_poststep_change_frac"])

        if run_int["Time"].iloc[-1] == run_int["Time"].iloc[-7]:
            if "solver_tolerance" not in COUPLER_options:
                COUPLER_options["solver_tolerance"] = 1.0e-10
            if COUPLER_options["solver_tolerance"] < 1.0e-2:
                COUPLER_options["solver_tolerance"] = float(COUPLER_options["solver_tolerance"])*2.
                print(">>> ADJUST tolerances:", COUPLER_options["solver_tolerance"])
            COUPLER_options["adjust_tolerance"] = 1
            print(">>> CURRENT TOLERANCES:", COUPLER_options["solver_tolerance"])

        # If tolerance was adjusted, restart SPIDER w/ new tolerances
        if "adjust_tolerance" in COUPLER_options:
            print(">>>>> >>>>> RESTART W/ ADJUSTED TOLERANCES")
            call_sequence.extend(["-ts_sundials_atol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-ts_sundials_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_snes_atol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_snes_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_ksp_atol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosts_ksp_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosic_ksp_rtol", str(COUPLER_options["solver_tolerance"])])
            call_sequence.extend(["-atmosic_ksp_atol", str(COUPLER_options["solver_tolerance"])])

    # Runtime info
    PrintSeparator()
    print("Running SPIDER... (loop counter = ", loop_counter, ")")
    if debug:
        print("   Flags:")
        for flag in call_sequence:
            print("   ",flag)
        print()

    call_string = " ".join(call_sequence)

    # Run SPIDER
    if debug:
        spider_print = sys.stdout
    else:
        spider_print = open(dirs["output"]+"spider_recent.log",'w')

    subprocess.run([call_string],shell=True,check=True,stdout=spider_print)

    if not debug:
        spider_print.close()

    # Update restart filename for next SPIDER run
    COUPLER_options["ic_interior_filename"] = natural_sort([os.path.basename(x) for x in glob.glob(dirs["output"]+"/*.json")])[-1]

    return COUPLER_options

# String sorting not based on natsorted package
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def CleanOutputDir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

# Plot conditions throughout run for on-the-fly analysis
def UpdatePlots( output_dir, COUPLER_options, time_dict ):

    if COUPLER_options["plot_onthefly"] == 1 or time_dict["planet"] > time_dict["target"]:

        PrintSeparator()
        print("Updating plots...")
        PrintSeparator()
        output_times = su.get_all_output_times( output_dir )
        if len(output_times) <= 8:
            plot_times = output_times
        else:
            plot_times = [ output_times[0]]         # first snapshot
            for i in np.logspace(-1,0,10,base=10):     # distinct timestamps
                j = int(math.floor( (len(output_times)-1)*(i)))
                plot_times.append(output_times[j])
        print("snapshots:", plot_times)

        # Global properties for all timesteps
        if len(output_times) > 1:
            cpl_global.plot_global(output_dir)   

        # Specific timesteps for paper plots
        cpl_interior.plot_interior(output_dir, plot_times)     
        cpl_atmosphere.plot_atmosphere(output_dir, plot_times)
        cpl_stacked.plot_stacked(output_dir, plot_times)
        
        # # One plot per timestep for video files
        # plot_atmosphere.plot_current_mixing_ratio(output_dir, plot_times[-1], use_vulcan) 

        # Close all figures
        plt.close()

# https://stackoverflow.com/questions/14115254/creating-a-folder-with-timestamp
# https://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def make_output_dir( output_dir ):
    save_dir = output_dir+"/"+"output_save/"+datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+"/"
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
    # for file in natsorted(glob.glob(output_dir+"/"+"*.*")):
    for file in natural_sort(glob.glob(output_dir+"/"+"*.*")):
        shutil.copy(file, save_dir+os.path.basename(file))
        print(os.path.basename(file), end =" ")


def ReadInitFile( dirs, init_file_passed ):

    # Read in input file as dictionary
    COUPLER_options  = {}
    time_dict       = {}
    print("Read in init file:", end=" ")

    # Output directory
    if os.path.isfile(init_file_passed):
        init_file = init_file_passed
    # Coupler directory
    else: 
        init_file = dirs["coupler"]+"/init_coupler.cfg"
        shutil.copy(dirs["coupler"]+"/init_coupler.cfg", init_file_passed)

    print(init_file)   
    print("Settings:")

    # Open file and fill dict
    with open(init_file) as f:
        
        # Filter blank lines
        lines = list(filter(None, (line.rstrip() for line in f)))
        
        for line in lines:
            
            # Skip comments
            if not line.startswith("#"):
                
                # Skip inline comments
                line = line.split("#")[0]
                line = line.split(",")[0]

                print(line)

                # Assign key and value
                (key, val) = line.split("=")
                key = key.strip()
                val = val.strip()
                
                # Standard options
                if not line.startswith("time_"):

                    # Some parameters are int
                    if key in [ "IC_INTERIOR", "IC_ATMOSPHERE", "SURFACE_BC", "nstepsmacro", "use_vulcan", "ic_interior_filename", "plot_onthefly"]:
                        val = int(val)
                    # Some are str
                    elif key in [ 'star_spectrum' ]:
                        val = str(val)
                    # Most are float
                    else:
                        val = float(val)

                    COUPLER_options[key] = val


                # Time options
                elif line.startswith("time_"):

                        line = line.split("_")[1]

                        (key, val) = line.split("=")
                    
                        time_dict[str(key.strip())] = float(val.strip())

                        if line.startswith("star"):
                            time_dict["offset"] = float(val.strip())

    return COUPLER_options, time_dict

def shallow_mixed_ocean_layer(F_eff, Ts_last, dT_max, t_curr, t_last):

    # Properties of the shallow mixed ocean layer
    c_p_layer   = 1000          # J kg-1 K-1
    rho_layer   = 3000          # kg m-3
    depth_layer = 1000          # m

    def ocean_evolution(t, y): 

        # Specific heat of mixed ocean layer
        mu      = c_p_layer * rho_layer * depth_layer # J K-1 m-2

        # RHS of ODE
        RHS     = - F_eff / mu

        return RHS

    # For SI conversion
    yr          = 3.154e+7      # s

    ### Compute Ts_curr at current time t_curr from previous time t_last
    t_last      = t_last*yr     # yr
    t_curr      = t_curr*yr     # yr
    Ts_last     = Ts_last       # K

    # Solve ODE
    sol_curr    = solve_ivp(ocean_evolution, [t_last, t_curr], [Ts_last])

    # New current surface temperature from shallow mixed layer
    Ts_curr     = sol_curr.y[0][-1] # K

    # Slow change IF dT too high
    if abs(Ts_last-Ts_curr) > dT_max:
        dT_sgn  = np.sign(Ts_last-Ts_curr)
        print("Limit max dT:", Ts_curr, "->", Ts_last-dT_sgn*dT_max)
        Ts_curr = Ts_last-dT_sgn*dT_max
    if abs(Ts_last-Ts_curr) > 0.05*Ts_last:
        dT_sgn  = np.sign(Ts_last-Ts_curr)
        print("Limit max dT:", Ts_curr, "->", Ts_last-dT_sgn*0.01*Ts_last)
        Ts_curr = Ts_last-dT_sgn*0.05*Ts_last

    print("t_last:", t_last/yr, "Ts_last:", Ts_last)
    print("t_curr:", t_curr/yr, "Ts_curr:", Ts_curr)

    return Ts_curr


def SolarConstant(time_dict: dict, COUPLER_options: dict):
    """Calculates the bolometric flux of the star at a previous time t. 

    Uses the Mors module, which reads stellar evolution tracks from 
    Spada et al. (2013). Flux is scaled to the star-planet distance.

    Parameters
    ----------
        time_dict : dict
            Time dictionary, including star's age
        COUPLER_options : dict
            Dictionary of coupler options variables

    Returns
    ----------
        flux : float
            Flux at planet's orbital separation (solar constant) in W/m^2
        heat : float
            Heating rate at TOA in W/m^2

    """ 

    Mstar = COUPLER_options["star_mass"]
    Tstar = time_dict['star'] * 1.e-6  # Convert from yr to Myr

    Lstar = mors.Value(Mstar, Tstar, 'Lbol')  # Units of L_sun
    Lstar *= 382.8e24 # Convert to W, https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html

    mean_distance = COUPLER_options["mean_distance"] * AU

    flux = Lstar /  ( 4. * np.pi * mean_distance * mean_distance )
    heat = flux * ( 1. - COUPLER_options["albedo_pl"] )

    return flux, heat


def ModernSpectrumLoad(dirs: dict, COUPLER_options: dict):
    """Load modern spectrum into memory.

    Scaled to 1 AU from the star. Generate these spectra using the python script
    'GetStellarSpectrum.py' in the 'tools' directory.

    Parameters
    ----------
        dirs : dict
            Directories dictionary
        COUPLER_options : dict
            Dictionary of coupler options variables

    Returns
    ----------
        spec_wl : np.array[float]
            Wavelength [nm]
        spec_fl : np.array[float]
            Flux [erg s-1 cm-2 nm-1]
    """

    spec_file = dirs["coupler"]+"/"+COUPLER_options["star_spectrum"]
    if os.path.isfile(spec_file):
        spec_data = np.loadtxt(spec_file, skiprows=2,delimiter='\t').T
        spec_wl = spec_data[0]
        spec_fl = spec_data[1]
    else:
        raise Exception("Cannot find stellar spectrum!")

    
    return spec_wl, spec_fl

def ModernSpectrumFband(dirs: dict, COUPLER_options: dict):
    """Calculates the integrated fluxes in each stellar band for the modern spectrum.

    These integrated fluxes have units of [erg s-1 cm-2] and are scaled to 
    1 AU from the star. Uses Numpy's trapz() to integrate the spectrum.

    Parameters
    ----------
        dirs : dict
            Directories dictionary
        COUPLER_options : dict
            Dictionary of coupler options variables

    Returns
    ----------
        COUPLER_options : dict
            Dictionary of coupler options variables, now containing integrated fluxes
    """

    # Load spectrum
    spec_wl, spec_fl = ModernSpectrumLoad(dirs, COUPLER_options)

    # Upper limit of wavelength range
    star_bands['bo'][1] = np.amax(spec_wl)
    star_bands['pl'][1] = np.amax(spec_wl)

    print("Modern spectrum F_band values:")

    # Integrate fluxes across wl, for each band
    for band in star_bands.keys():

        wl_min = star_bands[band][0]
        i_min = (np.abs(spec_wl - wl_min)).argmin()

        wl_max = star_bands[band][1]
        i_max = (np.abs(spec_wl - wl_max)).argmin()

        band_wl = spec_wl[i_min:i_max] 
        band_fl = spec_fl[i_min:i_max]

        fl_integ = np.trapz(band_fl,band_wl)

        COUPLER_options["Fband_modern_"+band] = fl_integ 

        print('Band %s [%d,%d] = %g' % (band,i_min,i_max,fl_integ))

    return COUPLER_options

def HistoricalSpectrumWrite(time_dict: dict, spec_wl: list, spec_fl: list, dirs : dict, COUPLER_options: dict):
    """Write historical spectrum to disk, for a time t.

    Uses the Mors evolution model. Spectrum scaled to 1 AU from the star.

    Parameters
    ----------
        time_dict : dict
            Time dictionary, including stellar age
        spec_wl : list
            Modern spectrum wavelength array [nm]
        spec_fl : list
            Modern spectrum flux array at 1 AU [erg s-1 cm-2 nm-1]
        dirs : dict
            Directories dictionary
        COUPLER_options : dict
            Dictionary of coupler options variables

    Returns
    ----------
        historical_spectrum : str
            Path to historical spectrum file written by this function.
    """

    # 1 AU in cm
    AU_cm = 1.496e+13

    # Get historical flux in each band provided by Mors
    Mstar = COUPLER_options["star_mass"]
    pctle = COUPLER_options["star_rot_percentile"]
    tstar = time_dict["star"] * 1.e-6

    # Rstar = COUPLER_options["star_radius"]
    Rstar = mors.Value(Mstar, tstar, 'Rstar') # radius in solar radii
    COUPLER_options['star_radius'] = Rstar
    Rstar_cm = Rstar * 6.957e+10  # radius in cm

    # Get temperature from Mors
    Tstar = mors.Value(Mstar, tstar, 'Teff')

    # Use Wien's law to get the temperature corresponding to the planckian region (FROM MODERN SPECTRUM!!)
    # imax = np.argmax(spec_fl[2:-2])
    # wl_max = spec_wl[imax] * 1.e-9
    # Tstar = 2.897771955e-3 / wl_max   # https://physics.nist.gov/cgi-bin/cuu/Value?bwien
    
    COUPLER_options['star_temperature'] = Tstar

    Omega = mors.Percentile(Mstar=Mstar, percentile=pctle)

    Ldict = mors.Lxuv(Mstar=Mstar, Age=tstar, Omega=Omega)

    # Fluxes scaled to 1 AU [erg s-1 cm-2]
    sf = (Rstar_cm / AU_cm) ** 2
    F_band = {
        'xr' : Ldict["Fx"] * sf,
        'e1' : Ldict["Feuv1"] * sf,
        'e2' : Ldict["Feuv2"] * sf,
        'pl' : 0.0  # Calc below
    }   

    # Find (total) bolometric flux
    # Lstar = mors.Value(Mstar, tstar, 'Lbol') * 382.8e24 # Units of [W]
    # F_band['bo'] = Lstar / ( 4. * np.pi * AU_cm * AU_cm ) * 1.e7 # Convert to [erg s-1 cm-2]

    # Find flux in planckian region
    hc_by_kT = phys.h * phys.c / (phys.k * Tstar)
    planck_func = lambda lam : 1.0/( (lam ** 5.0) * ( np.exp( hc_by_kT/ lam) - 1.0 ) )  

    planck_wl = np.linspace(star_bands['pl'][0] * 1e-9, star_bands['pl'][1] * 1e-9, 10000)
    planck_fl = planck_func(planck_wl)
    I_planck = np.trapz(planck_fl, planck_wl)  # Integrate planck function over wavelength

    I_planck *= 2 * phys.h * phys.c * phys.c   # W m-2 sr-1, at stellar surface
    I_planck *= 4 * np.pi # W m-2, integrate over solid angle
    I_planck *= sf  # Scale to 1 AU
    I_planck *= 1.0e3  # erg s-1 cm-2, convert units

    F_band['pl'] = I_planck

    # Find flux in UV region based on what's missing
    # F_remainder = F_band['bo'] - F_band['xr'] - F_band['e1'] - F_band['e2'] - F_band['pl']
    # F_band['uv'] = F_remainder

    # Get dimensionless ratios of past flux to modern flux
    # It's important that they have the same units
    Q_band = {}
    for band in ['xr','e1','e2','pl']:
        F_modern_band = COUPLER_options["Fband_modern_"+band]
        Q_band[band] = F_band[band] / F_modern_band
    Q_band['uv'] = Q_band['e2']  # Assume that UV scales like EUV2 (Is this reasonable??)

    # Calculate historical spectrum...
    if len(spec_wl) != len(spec_fl):
        raise Exception("Stellar spectrum wl and fl arrays are of different lengths!")
    
    print(F_band)
    print(Q_band)

    hspec_fl = np.zeros((len(spec_wl)))
    # Loop over each wl bin
    for i in range(len(spec_wl)):
        wl = spec_wl[i]
        fl = spec_fl[i]

        # Work out which band we are in
        for band in star_bands.keys():
            if star_bands[band][0] <= wl <= star_bands[band][1]:
                # Apply scale factor for this band
                hspec_fl[i] = fl * Q_band[band]
                break

    
    # Save historical spectrum
    X = np.array([spec_wl,hspec_fl]).T
    outname = dirs['output'] + "/%d.sflux" % time_dict['planet']
    header = '# Historical stellar flux (1 AU) at t_star = %d Myr \n# WL(nm)\t Flux(ergs/cm**2/s/nm)' % tstar
    np.savetxt(outname, X, header=header,comments='',fmt='%1.5e',delimiter='\t')

    return outname

# End of file
