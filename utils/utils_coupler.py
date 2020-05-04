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

def PrintCurrentState(time_dict, runtime_helpfile, COUPLER_options, atm, loop_counter, dirs):

    # Print final statement
    print("---------------------------------------------------------")
    print("==> RUNTIME INFO <==")
    print(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    print("LOOP:", loop_counter)
    print("Time [Myr]:", str(float(time_dict["planet"])/1e6))
    print("T_s [K]:", runtime_helpfile.iloc[-1]["T_surf"])
    print("Helpfile properties:")
    # print(runtime_helpfile[["Time", "Input", "M_atm", "M_atm_kgmol", "H_mol_atm", "H_mol_solid", "H_mol_liquid", "H_mol_total", "O_mol_total", "O/H_atm", "P_surf"]])
    print(runtime_helpfile.tail(2))
    print("P_surf [bar]:", runtime_helpfile.iloc[-1]["P_surf"], " ")
    print("TOA heating [W/m^2]:", atm.toa_heating)
    print("MO heat flux [W/m^2]:", COUPLER_options["heat_flux"])
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
        runtime_helpfile = pd.DataFrame(columns=['Time', 'Input', 'T_surf', 'Heat_flux', 'P_surf', 'M_atm', 'M_atm_kgmol', 'Phi_global', 'M_mantle', 'M_core', 'M_mantle_liquid', 'M_mantle_solid', 'H_mol_atm', 'H_mol_solid', 'H_mol_liquid', 'H_mol_total', 'O_mol_total', 'C_mol_total', 'N_mol_total', 'S_mol_total', 'He_mol_total', 'O/H_atm', 'C/H_atm', 'N/H_atm', 'S/H_atm', 'He/H_atm', 'H2O_mr', 'CO2_mr', 'H2_mr', 'CO_mr', 'CH4_mr', 'N2_mr', 'O2_mr', 'S_mr', 'He_mr'])
        runtime_helpfile.to_csv( dirs["output"]+"/"+runtime_helpfile_name, index=False, sep=" ") 
        time_dict["planet"] = 0
        #, 'H2O_atm_bar', 'CO2_atm_bar', 'H2_atm_bar', 'CH4_atm_bar', 'CO_atm_bar', 'N2_atm_bar', 'O2_atm_bar', 'S_atm_bar', 'He_atm_bar'run

        # Save SPIDER options file
        COUPLER_options_save = pd.DataFrame(COUPLER_options, index=[0])
        COUPLER_options_save.to_csv( dirs["output"]+"/"+COUPLER_options_name, index=False, sep=" ")

    # Data dict
    runtime_helpfile_new = {}

    print(runtime_helpfile)

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
        runtime_helpfile_new["Heat_flux"]       = data_a[6,:]
        runtime_helpfile_new["P_surf"]          = data_a[7,:]  # total surface pressure

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
                print(vol, runtime_helpfile_new[vol+"_atm_kg"])

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


    # For "Atmosphere" sub-loop (VULCAN+SOCRATES) update heat flux from SOCRATES
    elif input_flag == "Atmosphere":

        # Fatm_table                        = np.genfromtxt(output_dir+"OLRFlux.dat", names=['Time', 'Fatm'], dtype=None, skip_header=0)
        # time_list, Fatm_list  = list(Fatm_table['Time']), list(Fatm_table['Fatm'])
        # Fatm_newest    = Fatm_list[-1]
        # if loop_counter["init"] >= 1: 
        #     Fatm_newest    = Fatm_list[-1]
        # else: Fatm_newest                 = Fatm_list
        # runtime_helpfile_new["Heat_flux"] = Fatm_table['Fatm'][-1]

        # Define input flag
        runtime_helpfile_new["Input"]           = input_flag   

        # Update heat flux from latest SOCRATES output
        runtime_helpfile_new["Heat_flux"]       = COUPLER_options["heat_flux"]

        # Other info from latest SPIDER run (X/H ratios stay fixed w/o loss)
        runtime_helpfile_new["Time"]            = runtime_helpfile.iloc[-1]["Time"]       
        runtime_helpfile_new["T_surf"]          = runtime_helpfile.iloc[-1]["T_surf"]     
        runtime_helpfile_new["P_surf"]          = runtime_helpfile.iloc[-1]["P_surf"]         
        runtime_helpfile_new["M_atm"]           = runtime_helpfile.iloc[-1]["M_atm"]
        runtime_helpfile_new["M_atm_kgmol"]     = runtime_helpfile.iloc[-1]["M_atm_kgmol"]
        runtime_helpfile_new["Phi_global"]      = runtime_helpfile.iloc[-1]["Phi_global"]     
        runtime_helpfile_new["M_mantle"]        = runtime_helpfile.iloc[-1]["M_mantle"]       
        runtime_helpfile_new["M_core"]          = runtime_helpfile.iloc[-1]["M_core"]         
        runtime_helpfile_new["M_mantle_liquid"] = runtime_helpfile.iloc[-1]["M_mantle_liquid"]
        runtime_helpfile_new["M_mantle_solid"]  = runtime_helpfile.iloc[-1]["M_mantle_solid"]
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
        'Heat_flux':        runtime_helpfile_new["Heat_flux"],
        'P_surf':           runtime_helpfile_new["P_surf"],
        'M_atm':            runtime_helpfile_new["M_atm"],
        'M_atm_kgmol':      runtime_helpfile_new["M_atm_kgmol"],
        'Phi_global':       runtime_helpfile_new["Phi_global"],
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
    print(runtime_helpfile)
    print(dirs["output"]+"/"+runtime_helpfile_name)
    runtime_helpfile.to_csv( dirs["output"]+"/"+runtime_helpfile_name, index=False, sep=" ")

    # Save COUPLER_options to disk
    COUPLER_options_save = pd.read_csv(dirs["output"]+"/"+COUPLER_options_name, sep=" ")
    COUPLER_options_save = COUPLER_options_save.append(COUPLER_options, ignore_index=True) 
    COUPLER_options_save.to_csv( dirs["output"]+"/"+COUPLER_options_name, index=False, sep=" ")

    # Advance time_current in main loop
    time_dict["planet"] = runtime_helpfile.iloc[-1]["Time"]
    time_dict["star"]   = time_dict["planet"] + time_dict["offset"]

    return runtime_helpfile, time_dict

def PrintSeparator():
    print("-------------------------------------------------------------------------------------------------------------")
    pass

def PrintHalfSeparator():
    print("--------------------------------------------------")
    pass

# Generate/adapt atmosphere chemistry/radiation input files
def StructAtm( time_dict, loop_counter, dirs, runtime_helpfile ):

    # Initialize atmosphere structure
    # if loop_counter["init"] == 0 and loop_counter["atm"] == 0:

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

    atm = atmos(runtime_helpfile.iloc[-1]["T_surf"], runtime_helpfile.iloc[-1]["P_surf"]*1e5, vol_list)
        

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

    return atm

# # Calulcate partial pressures from 
# def ModifiedHenrysLaw( atm_chemistry, output_dir, file_name ):

#     PrintSeparator()
#     print("HACK –– apply partitioning directly on JSON files")
#     PrintSeparator()

#     # Total pressure
#     P_surf = atm_chemistry.iloc[0]["Pressure"]*1e5 # Pa

#     # Open the .json data
#     with open(output_dir+file_name) as f:
#         data = json.load(f)

#     # Move unaltered JSON --> .txt
#     shutil.move(output_dir+file_name, output_dir+file_name[:-5]+".txt")

#     mantle_melt_kg = float(data["atmosphere"]["mass_liquid"]["values"][0])*float(data["atmosphere"]["mass_liquid"]["scaling"])
#     print("M_mantle_liquid: ", mantle_melt_kg, "kg")
#     PrintHalfSeparator()

#     # Loop over all radiative species considered
#     for volatile in volatile_species:

#         # Dalton's law for partial pressures
#         p_vol       = float(atm_chemistry.iloc[0][volatile])*P_surf # Pa

#         # Modified Henry's law for obtaining melt abundances
#         henry_alpha = volatile_distribution_coefficients[volatile+"_alpha"]
#         henry_beta  = volatile_distribution_coefficients[volatile+"_beta"]

#         # Find melt abundance
#         X_vol_ppm   = henry_alpha * (p_vol**(1/henry_beta)) # ppm wt
#         X_vol_kg    = X_vol_ppm*1e-6*mantle_melt_kg       # kg

#         # Read in former scaled values
#         liquid_ppm_scaled     = float(data["atmosphere"][volatile]["liquid_ppm"]["values"][0])
#         liquid_kg_scaled      = float(data["atmosphere"][volatile]["liquid_kg"]["values"][0])
#         atmosphere_kg_scaled  = float(data["atmosphere"][volatile]["atmosphere_kg"]["values"][0])
#         atmosphere_bar_scaled = float(data["atmosphere"][volatile]["atmosphere_bar"]["values"][0])

#         # Read in scalings
#         liquid_ppm_scaling     = float(data["atmosphere"][volatile]["liquid_ppm"]["scaling"])
#         liquid_kg_scaling      = float(data["atmosphere"][volatile]["liquid_kg"]["scaling"])
#         atmosphere_kg_scaling  = float(data["atmosphere"][volatile]["atmosphere_kg"]["scaling"])
#         atmosphere_bar_scaling = float(data["atmosphere"][volatile]["atmosphere_bar"]["scaling"])

#         # Calculate former physical values
#         liquid_ppm     = liquid_ppm_scaled*liquid_ppm_scaling
#         liquid_kg      = liquid_kg_scaled*liquid_kg_scaling
#         atmosphere_kg  = atmosphere_kg_scaled*atmosphere_kg_scaling
#         atmosphere_bar = atmosphere_bar_scaled*atmosphere_bar_scaling

#         # Calculate new physical values, ensure mass conservation
#         liquid_ppm_new      = X_vol_ppm
#         liquid_kg_new       = X_vol_kg
#         atmosphere_kg_new   = (liquid_kg+atmosphere_kg)-liquid_kg_new
#         atmosphere_bar_new  = p_vol

#         # Calculate new scaled values
#         liquid_ppm_new_scaled      = liquid_ppm_new/liquid_ppm_scaling
#         liquid_kg_new_scaled       = liquid_kg_new/liquid_kg_scaling
#         atmosphere_kg_new_scaled   = atmosphere_kg_new/atmosphere_kg_scaling
#         atmosphere_bar_new_scaled  = atmosphere_bar_new/atmosphere_bar_scaling

#         # Print the changes
#         print(volatile, "liquid_ppm (scaled): ", liquid_ppm, "->", liquid_ppm_new, 
#             "(", liquid_ppm_scaled, "->", liquid_ppm_new_scaled, ") ppm wt")
#         print(volatile, "liquid_kg (scaled): ", liquid_kg, "->", liquid_kg_new, 
#             "(", liquid_kg_scaled, "->", liquid_kg_new_scaled, ") kg")
#         print(volatile, "atmosphere_kg (scaled): ", atmosphere_kg, "->", atmosphere_kg_new, 
#             "(", atmosphere_kg_scaled, "->", atmosphere_kg_new_scaled, ") kg")
#         print(volatile, "atmosphere_bar (scaled): ", atmosphere_bar, "->", atmosphere_bar_new, 
#             "(", atmosphere_bar_scaled, "->", atmosphere_bar_new_scaled, ") bar")

#         # Replace old with recalculated values
#         data["atmosphere"][volatile]["liquid_ppm"]["values"]    = [str(liquid_ppm_new_scaled)]
#         data["atmosphere"][volatile]["liquid_kg"]["values"]     = [str(liquid_kg_new_scaled)]
#         data["atmosphere"][volatile]["atmosphere_kg"]["values"] = [str(atmosphere_kg_new_scaled)]
#         data["atmosphere"][volatile]["atmosphere_bar"]["values"] = [str(atmosphere_bar_new_scaled)]
#         # print(data["atmosphere"][volatile])
#         PrintHalfSeparator()

#         ## THERE IS ANOTHER ENTRY IN JSON:
#         ## "Magma ocean volatile content"
#         ## ---> CHECK IF THAT ALSO NEEDS TO BE REPLACED

#     # Save the changed JSON file for read-in by SPIDER
#     with open(output_dir+file_name, 'w') as f:
#         json.dump(data, f, indent=4)

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
    atm.toa_heating = atm_rad_conv.SocRadConv.InterpolateStellarLuminosity(COUPLER_options["star_mass"], time_dict, COUPLER_options["mean_distance"], atm.albedo_pl)

    # Runtime info
    PrintSeparator()
    print("SOCRATES run, loop ", loop_counter, "|", datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    PrintSeparator()

    # Calculate temperature structure and heat flux w/ SOCRATES
    atm_dry, atm = atm_rad_conv.SocRadConv.RadConvEqm(dirs, time_dict, atm, loop_counter, COUPLER_options, standalone=False, cp_dry=False, trpp=True) # W/m^2
    
    # MO heat flux from topmost atmosphere node; do not allow heating
    COUPLER_options["heat_flux"] = np.max( [ 0., atm.net_flux[0] ] )

    # Clean up run directory
    PrintSeparator()
    print("Remove SOCRATES auxiliary files:")
    # for file in natsorted(glob.glob(dirs["output"]+"/current??.????")):
    for file in natural_sort(glob.glob(dirs["output"]+"/current??.????")):
        os.remove(file)
        print(os.path.basename(file), end =" ")
    # for file in natsorted(glob.glob(dirs["output"]+"/profile.*")):
    for file in natural_sort(glob.glob(dirs["output"]+"/profile.*")):
        os.remove(file)
        print(os.path.basename(file), end =" ")
    print("==> Done.")

    return atm, COUPLER_options

def RunSPIDER( time_dict, dirs, COUPLER_options, loop_counter, runtime_helpfile ):

    # Check if input file present in current dir, if not copy from SPIDER repo
    SPIDER_options_file = dirs["output"]+"/bu_input.opts"
    if not os.path.isfile(SPIDER_options_file):
        SPIDER_options_file_vanilla = dirs["spider"]+"/examples/lichtenberg_2019/bu_input_standard.opts"
        shutil.copy(SPIDER_options_file_vanilla, SPIDER_options_file)

    # Define which volatiles to track in SPIDER
    species_call = ""
    for vol in volatile_species: 
        if COUPLER_options[vol+"_initial_total_abundance"] > 0. or COUPLER_options[vol+"_initial_atmos_pressure"] > 0.:
            species_call = species_call + "," + vol
    species_call = species_call[1:] # Remove "," in front

    # Recalculate time stepping
    if COUPLER_options["IC_INTERIOR"] == 2:  

        dtmacro     = float(COUPLER_options["dtmacro"])
        dtswitch    = float(COUPLER_options["dtswitch"])

        # Runtime left
        dtime_max   = time_dict["target"] - time_dict["planet"]

        # Limit Atm-Int switch
        dtime       = np.min([ dtime_max, dtswitch ])


        COUPLER_options["nstepsmacro"] =  math.ceil( dtime / dtmacro )

    # For init loop
    else:
        dtmacro     = 0

    ### SPIDER base call sequence 
    call_sequence = [   
                        dirs["spider"]+"/spider", 
                        "-options_file",          SPIDER_options_file, 
                        "-outputDirectory",       dirs["output"],
                        "-IC_INTERIOR",           str(COUPLER_options["IC_INTERIOR"]),
                        "-IC_ATMOSPHERE",         str(COUPLER_options["IC_ATMOSPHERE"]),
                        "-SURFACE_BC",            str(COUPLER_options["SURFACE_BC"]), 
                        "-surface_bc_value",      str(COUPLER_options["heat_flux"]), 
                        "-tsurf_poststep_change", str(COUPLER_options["tsurf_poststep_change"]),
                        "-nstepsmacro",           str(COUPLER_options["nstepsmacro"]), 
                        "-dtmacro",               str(dtmacro), 
                        "-radius",                str(COUPLER_options["R_solid_planet"]), 
                        "-coresize",              str(COUPLER_options["planet_coresize"]),
                        "-volatile_names",        str(species_call)
                    ]

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

    # Check for convergence, if not converging, adjust tolerances iteratively
    if runtime_helpfile["Time"].iloc[-1] == runtime_helpfile["Time"].iloc[-10]:
        if "solver_tolerance" not in COUPLER_options:
            COUPLER_options["solver_tolerance"] = 1.0e-10
        COUPLER_options["solver_tolerance"] = float(COUPLER_options["solver_tolerance"])*2.
        COUPLER_options["adjust_tolerance"] = 1
        print(">>>>> ADJUST TOLERANCES:", COUPLER_options["solver_tolerance"])

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
def UpdatePlots( output_dir, use_vulcan=0 ):

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
                
                # Skipe inline comments
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

