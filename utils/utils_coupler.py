#!/usr/bin/env python3

# Import utils-specific modules
from utils.modules_utils import *

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

    # Plot conditions throughout run for on-the-fly analysis
    #UpdatePlots( dirs["output"], COUPLER_options["use_vulcan"] )

def UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, input_flag, COUPLER_options):

    runtime_helpfile_name = "runtime_helpfile.csv"
    COUPLER_options_name  = "COUPLER_options.csv"

    # If runtime_helpfle not existent, create it + write to disk
    if not os.path.isfile(dirs["output"]+"/"+runtime_helpfile_name):
        runtime_helpfile = pd.DataFrame(columns=['Time', 'Input', 'T_surf', 'F_int', 'F_atm', 'F_net', 'P_surf', 'M_atm', 'M_atm_kgmol', 'Phi_global', 'RF_depth', 'M_mantle', 'M_core', 'M_mantle_liquid', 'M_mantle_solid', 'H_mol_atm', 'H_mol_solid', 'H_mol_liquid', 'H_mol_total', 'O_mol_total', 'C_mol_total', 'N_mol_total', 'S_mol_total', 'He_mol_total', 'O/H_atm', 'C/H_atm', 'N/H_atm', 'S/H_atm', 'He/H_atm', 'H2O_mr', 'CO2_mr', 'H2_mr', 'CO_mr', 'CH4_mr', 'N2_mr', 'O2_mr', 'S_mr', 'He_mr'])
        runtime_helpfile.to_csv( dirs["output"]+"/"+runtime_helpfile_name, index=False, sep=" ") 
        time_dict["planet"] = 0
        #, 'H2O_atm_bar', 'CO2_atm_bar', 'H2_atm_bar', 'CH4_atm_bar', 'CO_atm_bar', 'N2_atm_bar', 'O2_atm_bar', 'S_atm_bar', 'He_atm_bar'run

        # Save SPIDER options file
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
        runtime_helpfile_new["M_mantle_liquid"] = data_a[0,:]
        runtime_helpfile_new["M_mantle_solid"]  = data_a[1,:]
        runtime_helpfile_new["M_mantle"]        = data_a[2,:]        
        runtime_helpfile_new["M_core"]          = data_a[3,:]         

        # Surface properties
        runtime_helpfile_new["T_surf"]          = data_a[4,:]
        runtime_helpfile_new["Phi_global"]      = data_a[5,:]  # global melt fraction
        runtime_helpfile_new["F_int"]           = data_a[6,:]  # Heat flux from interior
        runtime_helpfile_new["P_surf"]          = data_a[7,:]  # total surface pressure
        runtime_helpfile_new["RF_depth"]        = data_a[8,:]/COUPLER_options["R_solid_planet"]  # depth of rheological front

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

            if COUPLER_options[vol+"_initial_total_abundance"] > 0. or COUPLER_options[vol+"_initial_atmos_pressure"] > 0.:

                keys_t = ( 
                            ('atmosphere',vol,'liquid_kg'),
                            ('atmosphere',vol,'solid_kg'),
                            ('atmosphere',vol,'atmosphere_kg'),
                            ('atmosphere',vol,'atmosphere_bar'),
                            ('atmosphere',vol,'mixing_ratio')  
                         )
                
                data_a = su.get_dict_surface_values_for_specific_time( keys_t, sim_time, indir=dirs["output"] )

                runtime_helpfile_new[vol+"_liquid_kg"]  = data_a[0,:]
                runtime_helpfile_new[vol+"_solid_kg"]   = data_a[1,:]
                runtime_helpfile_new[vol+"_atm_kg"]     = data_a[2,:]
                runtime_helpfile_new[vol+"_atm_bar"]    = data_a[3,:]
                runtime_helpfile_new[vol+"_mr"]         = data_a[4,:]

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
            # runtime_helpfile_new[vol+"_mol_total"]  = 1e-6*COUPLER_options[vol+"_initial_total_abundance"]*runtime_helpfile_new["M_mantle"] / molar_mass[vol]
            
            # Only for the ones tracked in SPIDER
            if COUPLER_options[vol+"_initial_total_abundance"] > 0. or COUPLER_options[vol+"_initial_atmos_pressure"] > 0.:
                runtime_helpfile_new[vol+"_mol_atm"]    = runtime_helpfile_new[vol+"_atm_kg"] / molar_mass[vol]
                runtime_helpfile_new[vol+"_mol_solid"]  = runtime_helpfile_new[vol+"_solid_kg"] / molar_mass[vol]
                runtime_helpfile_new[vol+"_mol_liquid"] = runtime_helpfile_new[vol+"_liquid_kg"] / molar_mass[vol]
                runtime_helpfile_new[vol+"_mol_total"] = runtime_helpfile_new[vol+"_mol_atm"]      \
                                                         + runtime_helpfile_new[vol+"_mol_solid"]  \
                                                         + runtime_helpfile_new[vol+"_mol_liquid"]

        runtime_helpfile_new["M_atm_kgmol"] = 0.

        # Number of mols per element and reservoir
        for res in [ "total", "solid", "liquid", "atm" ]: # , "atm"
            runtime_helpfile_new["H_mol_"+res]  = runtime_helpfile_new["H2O_mol_"+res] * 2. \
                                                + runtime_helpfile_new["H2_mol_"+res]  * 2. \
                                                + runtime_helpfile_new["CH4_mol_"+res] * 4. 
            runtime_helpfile_new["O_mol_"+res]  = runtime_helpfile_new["H2O_mol_"+res] * 1. \
                                                + runtime_helpfile_new["CO2_mol_"+res] * 2. \
                                                + runtime_helpfile_new["CO_mol_"+res]  * 1. \
                                                + runtime_helpfile_new["O2_mol_"+res]  * 2.
            runtime_helpfile_new["C_mol_"+res]  = runtime_helpfile_new["CO2_mol_"+res] * 1. \
                                                + runtime_helpfile_new["CH4_mol_"+res] * 1. \
                                                + runtime_helpfile_new["CO_mol_"+res]  * 1.
            runtime_helpfile_new["N_mol_"+res]  = runtime_helpfile_new["N2_mol_"+res]  * 2.
            runtime_helpfile_new["S_mol_"+res]  = runtime_helpfile_new["S_mol_"+res]   * 1.
            runtime_helpfile_new["He_mol_"+res] = runtime_helpfile_new["He_mol_"+res]  * 1.

            if res == "atm":
                # for element in element_list:
                #     Dmol = runtime_helpfile_new[element+"_mol_total"] - runtime_helpfile_new[element+"_mol_solid"] - runtime_helpfile_new[element+"_mol_liquid"]
                #     if Dmol > 0.:
                #         runtime_helpfile_new[element+"_mol_atm"] = Dmol
                runtime_helpfile_new["M_atm_kgmol"] = runtime_helpfile_new["H_mol_"+res]  * molar_mass["H"] \
                                                    + runtime_helpfile_new["O_mol_"+res]  * molar_mass["O"] \
                                                    + runtime_helpfile_new["C_mol_"+res]  * molar_mass["C"] \
                                                    + runtime_helpfile_new["N_mol_"+res]  * molar_mass["N"] \
                                                    + runtime_helpfile_new["S_mol_"+res]  * molar_mass["S"] \
                                                    + runtime_helpfile_new["He_mol_"+res] * molar_mass["He"]

        # Avoid division by 0
        min_val     = 1e-99
        runtime_helpfile_new["H_mol_atm"] = np.max([runtime_helpfile_new["H_mol_atm"], min_val])

        # Calculate X/H ratios
        for element in [ "O", "C", "N", "S", "He" ]:
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
        # FIXME:VULCAN
        runtime_helpfile_new["P_surf"]          = runtime_helpfile.iloc[-1]["P_surf"]         
        runtime_helpfile_new["M_atm"]           = runtime_helpfile.iloc[-1]["M_atm"]
        runtime_helpfile_new["M_atm_kgmol"]     = runtime_helpfile.iloc[-1]["M_atm_kgmol"]
        runtime_helpfile_new["H_mol_atm"]       = runtime_helpfile.iloc[-1]["H_mol_atm"]
        runtime_helpfile_new["H_mol_solid"]     = runtime_helpfile.iloc[-1]["H_mol_solid"]
        runtime_helpfile_new["H_mol_liquid"]    = runtime_helpfile.iloc[-1]["H_mol_liquid"] 
        runtime_helpfile_new["H_mol_total"]     = runtime_helpfile.iloc[-1]["H_mol_total"] 
        runtime_helpfile_new["O_mol_total"]     = runtime_helpfile.iloc[-1]["O_mol_total"]
        runtime_helpfile_new["C_mol_total"]     = runtime_helpfile.iloc[-1]["C_mol_total"]   
        runtime_helpfile_new["N_mol_total"]     = runtime_helpfile.iloc[-1]["N_mol_total"]   
        runtime_helpfile_new["S_mol_total"]     = runtime_helpfile.iloc[-1]["S_mol_total"]   
        runtime_helpfile_new["He_mol_total"]    = runtime_helpfile.iloc[-1]["He_mol_total"]
        runtime_helpfile_new["O/H_atm"]         = runtime_helpfile.iloc[-1]["O/H_atm"]
        runtime_helpfile_new["C/H_atm"]         = runtime_helpfile.iloc[-1]["C/H_atm"]
        runtime_helpfile_new["N/H_atm"]         = runtime_helpfile.iloc[-1]["N/H_atm"]
        runtime_helpfile_new["S/H_atm"]         = runtime_helpfile.iloc[-1]["S/H_atm"]
        runtime_helpfile_new["He/H_atm"]        = runtime_helpfile.iloc[-1]["He/H_atm"]
        runtime_helpfile_new["H2O_mr"]          = runtime_helpfile.iloc[-1]["H2O_mr"]
        runtime_helpfile_new["CO2_mr"]          = runtime_helpfile.iloc[-1]["CO2_mr"]
        runtime_helpfile_new["H2_mr"]           = runtime_helpfile.iloc[-1]["H2_mr"]
        runtime_helpfile_new["CO_mr"]           = runtime_helpfile.iloc[-1]["CO_mr"]
        runtime_helpfile_new["CH4_mr"]          = runtime_helpfile.iloc[-1]["CH4_mr"]
        runtime_helpfile_new["N2_mr"]           = runtime_helpfile.iloc[-1]["N2_mr"]
        runtime_helpfile_new["O2_mr"]           = runtime_helpfile.iloc[-1]["O2_mr"]
        runtime_helpfile_new["S_mr"]            = runtime_helpfile.iloc[-1]["S_mr"]
        runtime_helpfile_new["He_mr"]           = runtime_helpfile.iloc[-1]["He_mr"]    
    
    ## / Read in data

    # Add all parameters to dataframe + update file on disk
    runtime_helpfile_new = pd.DataFrame({
        'Time':             runtime_helpfile_new["Time"],
        'Input':            runtime_helpfile_new["Input"],
        'T_surf':           runtime_helpfile_new["T_surf"],
        'F_int':            runtime_helpfile_new["F_int"],
        'F_atm':            runtime_helpfile_new["F_atm"],
        'F_net':            runtime_helpfile_new["F_net"],
        'P_surf':           runtime_helpfile_new["P_surf"],
        'M_atm':            runtime_helpfile_new["M_atm"],
        'M_atm_kgmol':      runtime_helpfile_new["M_atm_kgmol"],
        'Phi_global':       runtime_helpfile_new["Phi_global"],
        'RF_depth':         runtime_helpfile_new["RF_depth"],
        'M_mantle':         runtime_helpfile_new["M_mantle"],
        'M_core':           runtime_helpfile_new["M_core"],
        'M_mantle_liquid':  runtime_helpfile_new["M_mantle_liquid"],
        'M_mantle_solid':   runtime_helpfile_new["M_mantle_solid"],
        'H_mol_atm':        runtime_helpfile_new["H_mol_atm"],
        'H_mol_solid':      runtime_helpfile_new["H_mol_solid"],
        'H_mol_liquid':     runtime_helpfile_new["H_mol_liquid"],
        'H_mol_total':      runtime_helpfile_new["H_mol_total"],
        'O_mol_total':      runtime_helpfile_new["O_mol_total"],
        'C_mol_total':      runtime_helpfile_new["C_mol_total"],
        'N_mol_total':      runtime_helpfile_new["N_mol_total"],
        'S_mol_total':      runtime_helpfile_new["S_mol_total"],
        'He_mol_total':     runtime_helpfile_new["He_mol_total"],
        'O/H_atm':          runtime_helpfile_new["O/H_atm"],
        'C/H_atm':          runtime_helpfile_new["C/H_atm"],
        'N/H_atm':          runtime_helpfile_new["N/H_atm"],
        'S/H_atm':          runtime_helpfile_new["S/H_atm"],
        'He/H_atm':         runtime_helpfile_new["He/H_atm"],
        'H2O_mr':           runtime_helpfile_new["H2O_mr"],
        'CO2_mr':           runtime_helpfile_new["CO2_mr"],
        'H2_mr':            runtime_helpfile_new["H2_mr"],
        'CO_mr':            runtime_helpfile_new["CO_mr"],
        'CH4_mr':           runtime_helpfile_new["CH4_mr"],
        'N2_mr':            runtime_helpfile_new["N2_mr"],
        'O2_mr':            runtime_helpfile_new["O2_mr"],
        'S_mr':             runtime_helpfile_new["S_mr"],
        'He_mr':            runtime_helpfile_new["He_mr"],
        }, index=[0])
    runtime_helpfile = runtime_helpfile.append(runtime_helpfile_new) 
    # print(runtime_helpfile)
    print(dirs["output"]+"/"+runtime_helpfile_name)
    runtime_helpfile.to_csv( dirs["output"]+"/"+runtime_helpfile_name, index=False, sep=" ")

    # Save COUPLER_options to disk
    COUPLER_options_save = pd.read_csv(dirs["output"]+"/"+COUPLER_options_name, sep=" ")
    COUPLER_options_save = COUPLER_options_save.append(COUPLER_options, ignore_index=True) 
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

# Generate/adapt atmosphere chemistry/radiation input files
def StructAtm( loop_counter, dirs, runtime_helpfile, COUPLER_options ):

    # Volatile molar concentrations: must sum to ~1 !
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

    atm = atmos(COUPLER_options["T_surf"], runtime_helpfile.iloc[-1]["P_surf"]*1e5, vol_list)
        

    # if COUPLER_options["use_vulcan"] != 0:
                
        # with open(output_dir+'vulcan_XH_ratios.dat', 'w') as file:
        #     file.write('time             O                C                N                S                He\n')
        # with open(output_dir+'vulcan_XH_ratios.dat', 'a') as file:
        #     file.write('0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00 0.0000000000e+00\n')

        # # Change surface gravity in VULCAN init file
        # M_solid_planet = runtime_helpfile.iloc[-1]["M_core"] + runtime_helpfile.iloc[-1]["M_mantle"]
        # grav_s         = su.gravity( M_solid_planet, float(R_solid_planet) ) * 100. # cm s^-2

        # # Generate init TP structure
        # atm             = atmos()
        # atm.ps          = runtime_helpfile.iloc[-1]["P_surf"]*1e5 # bar->Pa
        # pstart          = atm.ps#*.995
        # rat             = (atm.ptop/pstart)**(1./atm.nlev)
        # logLevels       = [pstart*rat**i for i in range(atm.nlev+1)]
        # logLevels.reverse()
        # levels          = [atm.ptop+i*(pstart-atm.ptop)/(atm.nlev-1) for i in range(atm.nlev+1)]
        # atm.pl          = np.array(logLevels)
        # atm.p           = (atm.pl[1:] + atm.pl[:-1]) / 2
        # pressure_grid   = atm.p*1e-5 # Pa->bar
        # temp_grid       = np.ones(len(pressure_grid))*runtime_helpfile.iloc[-1]["T_surf"]

        # # # Min/max P for VULCAN config file
        # # P_b = np.max(pressure_grid)*1e6*0.9999 # pressure at the bottom, (bar)->(dyne/cm^2)
        # # P_t = np.min(pressure_grid)*1e6*1.0001 # pressure at the top, (bar)->(dyne/cm^2)
        
        # # # Generate initial TP structure file
        # # out_a       = np.column_stack( ( temp_grid, pressure_grid ) ) # K, bar
        # # init_TP     = str(int(time_current))+"_atm_TP_profile_init.dat"
        # # np.savetxt( output_dir+init_TP, out_a )

        # # # Adjust copied file in output_dir
        # # for line in fileinput.input(vulcan_dir+'vulcan_cfg.py', inplace=True):
        # #     if line.strip().startswith('g = '):
        # #         line = "g = "+str(grav_s)+' # surface gravity (cm/s^2)\n'
        # #     if line.strip().startswith('spider_file = '):
        # #         line = "spider_file = '"+str(output_dir)+"vulcan_XH_ratios.dat'\n"
        # #     if line.strip().startswith('atm_file = '):
        # #         line = "atm_file = '"+str(output_dir)+str(init_TP)+"'\n"
        # #     if line.strip().startswith('P_b = '):
        # #         line = "P_b = "+str(P_b)+" # pressure at the bottom (dyne/cm^2)\n"
        # #     if line.strip().startswith('P_t = '):
        # #         line = "P_t = "+str(P_t)+" # pressure at the top (dyne/cm^2)\n"
        # #     sys.stdout.write(line)
           

    # # After first loop use updated TP profiles from SOCRATES
    # else:
    #     # Modify the SOCRATES TP structure input
    #     run_TP_file = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*_atm_TP_profile.dat")])[-1]
        
    #     # Min/max P for VULCAN config file
    #     atm_table = np.genfromtxt(output_dir+run_TP_file, names=['T', 'Pbar'], dtype=None, skip_header=0)
    #     pressure_grid, temp_grid = atm_table['Pbar']*1e6, atm_table['T'] # (bar)->(dyne/cm^2), K
    #     P_b = np.max(pressure_grid)*0.9999 # pressure at the bottom, (dyne/cm^2)
    #     P_t = np.min(pressure_grid)*1.0001 # pressure at the top, (dyne/cm^2)

    #     # Adjust copied file in output_dir
    #     for line in fileinput.input(vulcan_dir+'vulcan_cfg.py', inplace=True):
    #         if line.strip().startswith('atm_file = '):
    #             line = "atm_file = '"+str(output_dir)+str(run_TP_file)+"'\n"
    #         if line.strip().startswith('P_b = '):
    #             line = "P_b = "+str(P_b)+" # pressure at the bottom (dyne/cm^2)\n"
    #         if line.strip().startswith('P_t = '):
    #             line = "P_t = "+str(P_t)+" # pressure at the top (dyne/cm^2)\n"
    #         sys.stdout.write(line)

    # # Write elemental X/H ratios
    # with open(output_dir+'vulcan_XH_ratios.dat', 'a') as file:
    #     file.write(str('{:.10e}'.format(time_current))+" "+str('{:.10e}'.format(runtime_helpfile.iloc[-1]["O/H_atm"]))+" "+str('{:.10e}'.format(runtime_helpfile.iloc[-1]["C/H_atm"]))+" "+str('{:.10e}'.format(runtime_helpfile.iloc[-1]["N/H_atm"]))+" "+str('{:.10e}'.format(runtime_helpfile.iloc[-1]["S/H_atm"]))+" "+str('{:.10e}'.format(runtime_helpfile.iloc[-1]["He/H_atm"]))+"\n")

    # # Last .json file -> print time
    # last_file = natsorted([os.path.basename(x) for x in glob.glob(output_dir+"*.json")])[-1]
    # time_print = last_file[:-5]

    # # Update VULCAN output file names
    # volume_mixing_ratios = str(time_print)+"_atm_chemistry_volume.dat"
    # mass_mixing_ratios   = str(time_print)+"_atm_chemistry_mass.dat"
    # for line in fileinput.input(vulcan_dir+'vulcan_cfg.py', inplace=True):
    #     if line.strip().startswith('EQ_outfile = '):
    #         line = "EQ_outfile = '"+str(output_dir)+str(volume_mixing_ratios)+"'"+' # volume mixing ratio\n'
    #     if line.strip().startswith('mass_outfile = '):
    #         line = "mass_outfile = '"+str(output_dir)+str(mass_mixing_ratios)+"'"+' # mass mixing ratio\n'
    #     sys.stdout.write(line)

    # # Save modified temporary config file to compare versions
    # shutil.copy(vulcan_dir+'vulcan_cfg.py', output_dir+str(time_print)+'_vulcan_cfg.py')

    # return volume_mixing_ratios, mass_mixing_ratios

    return atm, COUPLER_options

# run VULCAN/atmosphere chemistry
def RunAtmChemistry( atm, time_dict, loop_counter, dirs, runtime_helpfile, COUPLER_options ):

    # # Generate/adapt atm structure
    # atm = StructAtm( time_current, loop_counter, vulcan_dir, output_dir, runtime_helpfile )
    # # volume_mixing_ratios_name, mass_mixing_ratios_name = AtmStruct( time_current, loop_counter, vulcan_dir, output_dir, runtime_helpfile, COUPLER_options["R_solid_planet"] )

    if COUPLER_options["use_vulcan"] != 0:

        # Runtime info
        PrintSeparator()
        print("VULCAN run, loop ", loop_counter, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        PrintSeparator()

        # Switch to VULCAN directory; run VULCAN, switch back to main directory
        os.chdir(dirs["vulcan"])
        subprocess.run(["python", "vulcan.py", "-n"], shell=False)
        os.chdir(dirs["coupler"])

        # # Copy VULCAN dumps to output folder
        # shutil.copy(vulcan_dir+'output/vulcan_EQ.txt', output_dir+str(int(time_current))+"_atm_chemistry.dat")

        # Read in data from VULCAN output
        atm_chemistry = pd.read_csv(dirs["output"]+"/"+volume_mixing_ratios_name, skiprows=1, delim_whitespace=True)
        print(atm_chemistry.iloc[:, 0:5])

        # # Update SPIDER restart options w/ surface partial pressures
        # for vol in volatile_species:
            
        #     # Calculate partial pressure from VULCAN output
        #     volume_mixing_ratio     = atm_chemistry.iloc[0][vol]
        #     surface_pressure_total  = atm_chemistry.iloc[0]["Pressure"]*1e5 # bar -> Pa
        #     partial_pressure_vol    = surface_pressure_total*volume_mixing_ratio

        #     # Only for major atmospheric species
        #     if partial_pressure_vol > 1.: # Pa
        #         COUPLER_options[vol+"_initial_atmos_pressure"] = partial_pressure_vol

    return atm

def RunSOCRATES( atm, time_dict, dirs, runtime_helpfile, loop_counter, COUPLER_options ):

    # Interpolate TOA heating from Baraffe models and distance from star
    atm.toa_heating = SocRadConv.InterpolateStellarLuminosity(COUPLER_options["star_mass"], time_dict, COUPLER_options["mean_distance"], atm.albedo_pl, COUPLER_options["Sfrac"])

    # Runtime info
    PrintSeparator()
    print("SOCRATES run, loop ", loop_counter, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    PrintSeparator()

    # Calculate temperature structure and heat flux w/ SOCRATES
    atm_dry, atm = SocRadConv.RadConvEqm(dirs, time_dict, atm, loop_counter, COUPLER_options, standalone=False, cp_dry=False, trpp=True, rscatter=True) # W/m^2
    
    # Atmosphere net flux from topmost atmosphere node; do not allow heating
    COUPLER_options["F_atm"] = np.max( [ 0., atm.net_flux[0] ] )

    # Clean up run directory
    PrintSeparator()
    print("Remove SOCRATES auxiliary files:", end =" ")
    # for file in natsorted(glob.glob(dirs["output"]+"/current??.????")):
    for file in natural_sort(glob.glob(dirs["output"]+"/current??.????")):
        os.remove(file)
        print(os.path.basename(file), end =" ")
    # for file in natsorted(glob.glob(dirs["output"]+"/profile.*")):
    for file in natural_sort(glob.glob(dirs["output"]+"/profile.*")):
        os.remove(file)
        print(os.path.basename(file), end =" ")
    print(">>> Done.")

    return atm, COUPLER_options

def RunSPIDER( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile ):

    # Check if input file present in current dir, if not copy standard from SPIDER repo
    SPIDER_options_file = dirs["output"]+"/init_spider.opts"

    # Standard spider .opts file
    SPIDER_options_file_vanilla = dirs["utils"]+"/init_spider_vanilla.opts"

    if not os.path.isfile(SPIDER_options_file):
        shutil.copy(SPIDER_options_file_vanilla, dirs["output"]+"/init_spider.opts")
        # SPIDER_options_file = SPIDER_options_file_vanilla

    # Define which volatiles to track in SPIDER
    species_call = ""
    for vol in volatile_species: 
        if COUPLER_options[vol+"_initial_total_abundance"] > 0. or COUPLER_options[vol+"_initial_atmos_pressure"] > 0.:
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

        PrintSeparator()
        PrintSeparator()
        PrintSeparator()
        print("TIME OPTIONS IN RUNSPIDER:")
        print(dtmacro, dtswitch, dtime_max, dtime, COUPLER_options["nstepsmacro"])
        PrintSeparator()
        PrintSeparator()
        PrintSeparator()

    # For init loop
    else:
        dtmacro     = 0

    # Prevent interior oscillations during last-stage freeze-out
    net_loss = COUPLER_options["F_atm"]
    if len(runtime_helpfile) > 100 and runtime_helpfile.iloc[-1]["Phi_global"] <= COUPLER_options["phi_crit"]:
        net_loss = np.amax([abs(COUPLER_options["F_atm"]), COUPLER_options["F_eps"]])
        print("Prevent interior oscillations during last-stage freeze-out: F_atm =", COUPLER_options["F_atm"], "->", net_loss)

    # net_loss = np.amin([net_loss, 1e4])
    # print("------>>>> HERE", net_loss, 1e4)

    ### SPIDER base call sequence 
    call_sequence = [   
                        dirs["spider"]+"/spider", 
                        "-options_file",          SPIDER_options_file, 
                        "-outputDirectory",       dirs["output"],
                        "-IC_INTERIOR",           str(COUPLER_options["IC_INTERIOR"]),
                        "-IC_ATMOSPHERE",         str(COUPLER_options["IC_ATMOSPHERE"]),
                        "-SURFACE_BC",            str(COUPLER_options["SURFACE_BC"]), 
                        "-surface_bc_value",      str(net_loss), 
                        "-nstepsmacro",           str(COUPLER_options["nstepsmacro"]), 
                        "-dtmacro",               str(dtmacro), 
                        "-radius",                str(COUPLER_options["R_solid_planet"]), 
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

    ## Conditional additions to call sequence

    # Define distribution coefficients and total mass/surface pressure for volatiles > 0
    for vol in volatile_species:
        if COUPLER_options[vol+"_initial_total_abundance"] > 0. or COUPLER_options[vol+"_initial_atmos_pressure"] > 0.:

            # # Very first timestep: feed volatile initial abundance [ppm]
            # if loop_counter["init"] == 0:
            
            # if COUPLER_options["IC_ATMOSPHERE"] == 1:
            call_sequence.extend(["-"+vol+"_initial_total_abundance", str(COUPLER_options[vol+"_initial_total_abundance"])])

            # # After very first timestep, starting w/ 2nd init loop
            # if loop_counter["init"] >= 1:
            # # if COUPLER_options["IC_ATMOSPHERE"] == 3:

            #     # # Load partial pressures from VULCAN
            #     # call_sequence.extend(["-"+vol+"_initial_atmos_pressure", str(COUPLER_options[vol+"_initial_atmos_pressure"])])

            # ## KLUDGE: Read in the same abundances every time -> no feedback from ATMOS
            # if COUPLER_options["use_vulcan"] == 0 or COUPLER_options["use_vulcan"] == 1:
            #     call_sequence.extend(["-"+vol+"_initial_total_abundance", str(COUPLER_options[vol+"_initial_total_abundance"])])

            # Exception for N2 case: reduced vs. oxidized
            if vol == "N2" and COUPLER_options["N2_partitioning"] == 1:
                volatile_distribution_coefficients["N2_henry"] = volatile_distribution_coefficients["N2_henry_reduced"]
                volatile_distribution_coefficients["N2_henry_pow"] = volatile_distribution_coefficients["N2_henry_pow_reduced"]

            call_sequence.extend(["-"+vol+"_henry", str(volatile_distribution_coefficients[vol+"_henry"])])
            call_sequence.extend(["-"+vol+"_henry_pow", str(volatile_distribution_coefficients[vol+"_henry_pow"])])
            call_sequence.extend(["-"+vol+"_kdist", str(volatile_distribution_coefficients[vol+"_kdist"])])
            call_sequence.extend(["-"+vol+"_kabs", str(volatile_distribution_coefficients[vol+"_kabs"])])
            call_sequence.extend(["-"+vol+"_molar_mass", str(molar_mass[vol])])

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
            if COUPLER_options[vol+"_initial_total_abundance"] > 0. or COUPLER_options[vol+"_initial_atmos_pressure"] > 0.:
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
    print("SPIDER run, loop ", loop_counter, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), "| flags:")
    for flag in call_sequence:
        print(flag, end =" ")
    print()
    PrintSeparator()

    # Call SPIDER
    subprocess.call(call_sequence)

    # Update restart filename for next SPIDER run
    # COUPLER_options["ic_interior_filename"] = natsorted([os.path.basename(x) for x in glob.glob(dirs["output"]+"/*.json")])[-1]
    COUPLER_options["ic_interior_filename"] = natural_sort([os.path.basename(x) for x in glob.glob(dirs["output"]+"/*.json")])[-1]

    return COUPLER_options

# String sorting not based on natsorted package
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def CleanOutputDir( output_dir ):

    types = ("*.json", "*.log", "*.csv", "*.pkl", "current??.????", "profile.*", "*.pdf", "*.png", "radiation_code.lock") 
    files_to_delete = []
    for files in types:
        files_to_delete.extend(glob.glob(output_dir+"/"+files))

    PrintSeparator()
    print("Remove old output files:")

    for file in natural_sort(files_to_delete): 
        os.remove(file)
        print(os.path.basename(file), end=" ")
    print("\n==> Done.")

    # print("SORT W/ NATSORT:")
    # for file in natsorted(files_to_delete):
    #     os.remove(file)
    #     print(os.path.basename(file), end =" ")
    # print("\n==> Done.")

# Plot conditions throughout run for on-the-fly analysis
def UpdatePlots( output_dir, COUPLER_options, time_dict ):

    if COUPLER_options["plot_onthefly"] == 1 or time_dict["planet"] > time_dict["target"]:

        PrintSeparator()
        print("Plot current evolution")
        PrintSeparator()
        output_times = su.get_all_output_times( output_dir )
        if len(output_times) <= 8:
            plot_times = output_times
        else:
            plot_times = [ output_times[0]]         # first snapshot
            for i in [ 2, 15, 22, 30, 45, 66 ]:     # distinct timestamps
                plot_times.append(output_times[int(round(len(output_times)*(i/100.)))])
            plot_times.append(output_times[-1])     # last snapshot
        print("snapshots:", plot_times)

        # Global properties for all timesteps
        if len(output_times) > 1:
            utils.cpl_global.plot_global(output_dir)   

        # Specific timesteps for paper plots
        utils.cpl_interior.plot_interior(output_dir, plot_times)     
        utils.cpl_atmosphere.plot_atmosphere(output_dir, plot_times)
        utils.cpl_stacked.plot_stacked(output_dir, plot_times)
        
        # # One plot per timestep for video files
        # utils.plot_atmosphere.plot_current_mixing_ratio(output_dir, plot_times[-1], use_vulcan) 

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
        init_file = dirs["coupler"]+"/init_coupler.opts"
        shutil.copy(dirs["coupler"]+"/init_coupler.opts", init_file_passed)

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
                    if key in [ "IC_INTERIOR", "IC_ATMOSPHERE", "SURFACE_BC", "nstepsmacro", "use_vulcan", "ic_interior_filename" ]:
                        val = int(val)
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

