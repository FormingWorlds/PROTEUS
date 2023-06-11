# Functions used to help run PROTEUS which are mostly submodule agnostic.

# Import utils-specific modules
from utils.modules_ext import *
from utils.constants import *
from utils.spider import *
from utils.helper import *

import plot.cpl_atmosphere as cpl_atmosphere
import plot.cpl_global as cpl_global
import plot.cpl_stacked as cpl_stacked
import plot.cpl_interior as cpl_interior

# Handle optional command line arguments for volatiles
# Optional arguments: https://towardsdatascience.com/learn-enough-python-to-be-useful-argparse-e482e1764e05
def parse_console_arguments():
    
    parser = argparse.ArgumentParser(description='PROTEUS optional command line arguments')
    parser.add_argument('-cfg_file', type=str, default="init_coupler.cfg", help='Specify cfg filename')
    parser.add_argument('-restart_file', type=str, default="0", help='Restart from specific .json file in folder. Specify only the number of the file.')
    parser.add_argument('-H2O_ppm', type=float, help='H2O initial abundance (ppm wt).')
    parser.add_argument('-CO2_ppm', type=float, help='CO2 initial abundance (ppm wt).')
    parser.add_argument('-H2_ppm', type=float, help='H2 initial abundance (ppm wt).')
    parser.add_argument('-CO_ppm', type=float, help='CO initial abundance (ppm wt).')
    parser.add_argument('-N2_ppm', type=float, help='N2 initial abundance (ppm wt).')
    parser.add_argument('-CH4_ppm', type=float, help='CH4 initial abundance (ppm wt).')
    parser.add_argument('-O2_ppm', type=float, help='O2 initial abundance (ppm wt).')
    parser.add_argument('-H2O_bar', type=float, help='H2O initial abundance (bar).')
    parser.add_argument('-CO2_bar', type=float, help='CO2 initial abundance (bar).')
    parser.add_argument('-H2_bar', type=float, help='H2 initial abundance (bar).')
    parser.add_argument('-CO_bar', type=float, help='CO initial abundance (bar).')
    parser.add_argument('-N2_bar', type=float, help='N2 initial abundance (bar).')
    parser.add_argument('-CH4_bar', type=float, help='CH4 initial abundance (bar).')
    parser.add_argument('-O2_bar', type=float, help='O2 initial abundance (bar).')
    # parser.add_argument('-rf', '--restart_file', type=str, help='Restart from specific .json file in folder. Specify only the number of the file.')
    parser.add_argument('-r', '--restart', action='store_true', help='Restart from last file in folder.')
    args = parser.parse_args()

    return args

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
        runtime_helpfile.to_csv( dirs["output"]+"/"+runtime_helpfile_name, index=False, sep="\t") 
        time_dict["planet"] = 0
        #, 'H2O_atm_bar', 'CO2_atm_bar', 'H2_atm_bar', 'CH4_atm_bar', 'CO_atm_bar', 'N2_atm_bar', 'O2_atm_bar', 'S_atm_bar', 'He_atm_bar'run

        # Save coupler options to file
        COUPLER_options_save = pd.DataFrame(COUPLER_options, index=[0])
        COUPLER_options_save.to_csv( dirs["output"]+"/"+COUPLER_options_name, index=False, sep="\t")

    # Data dict
    runtime_helpfile_new = {}

    # print(runtime_helpfile)

    # For "Interior" sub-loop (SPIDER)
    if input_flag == "Interior":

        ### Read in last SPIDER base parameters
        sim_times = get_all_output_times(dirs["output"])  # yr
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

        data_a = get_dict_surface_values_for_specific_time( keys_t, sim_time, indir=dirs["output"] )

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
        json_file   = MyJSON( dirs["output"]+'/{}.json'.format(sim_time) )
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
            json_file_time = MyJSON( dirs["output"]+'/{}.json'.format(sim_time) )
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
                
                data_a = get_dict_surface_values_for_specific_time( keys_t, sim_time, indir=dirs["output"] )

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
        for elem in [n for n in element_list if n != 'H']:
            runtime_helpfile_new[elem+"/H_atm"]         = runtime_helpfile.iloc[-1][elem+"/H_atm"]

        for vol in volatile_species:
            runtime_helpfile_new[vol+"_mr"]          = runtime_helpfile.iloc[-1][vol+"_mr"]

    runtime_helpfile_new = pd.DataFrame(runtime_helpfile_new,index=[0])
    runtime_helpfile = pd.concat([runtime_helpfile, runtime_helpfile_new])

    print(dirs["output"]+"/"+runtime_helpfile_name)
    runtime_helpfile.to_csv( dirs["output"]+"/"+runtime_helpfile_name, index=False, sep="\t")

    # Save COUPLER_options to disk
    COUPLER_options_save = pd.read_csv(dirs["output"]+"/"+COUPLER_options_name, sep="\t")
    COUPLER_options_df = pd.DataFrame.from_dict([COUPLER_options])
    COUPLER_options_save = pd.concat([ COUPLER_options_save, COUPLER_options_df],ignore_index=True)
    COUPLER_options_save.to_csv( dirs["output"]+"/"+COUPLER_options_name, index=False, sep="\t")

    # Advance time_current in main loop
    time_dict["planet"] = runtime_helpfile.iloc[-1]["Time"]
    time_dict["star"]   = time_dict["planet"] + time_dict["offset"]

    return runtime_helpfile, time_dict, COUPLER_options

# Calculate eqm temperature given stellar flux and bond albedo
def calc_eqm_temperature(I_0, A_B):
    return (I_0 * (1.0 - A_B) / (4.0 * phys.sigma))**(1.0/4.0)


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


def ReadInitFile( init_file_passed ):

    # Read in input file as dictionary
    COUPLER_options  = {}
    time_dict       = {}
    print("Read in init file:", end=" ")

    if os.path.isfile(init_file_passed):
        init_file = init_file_passed
    else: 
        raise Exception("Init file provided is not a file or does not exist!")

    print(init_file)   
    if debug: print("Settings:")

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

                if debug: print(line)

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
                    elif key in [ 'star_spectrum', 'star_btrack', 'dir_output' ]:
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


# Plot conditions throughout run for on-the-fly analysis
def UpdatePlots( output_dir, COUPLER_options, time_dict ):

    if COUPLER_options["plot_onthefly"] == 1 or time_dict["planet"] > time_dict["target"]:

        PrintSeparator()
        print("Updating plots...")
        PrintSeparator()
        output_times = get_all_output_times( output_dir )
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

def SetDirectories(COUPLER_options: dict):
    # Set directories dictionary

    coupler_dir = os.getenv('COUPLER_DIR')

    dirs = {
            "output": coupler_dir+"/"+COUPLER_options['dir_output']+"/", 
            "coupler": coupler_dir, 
            "rad_conv": coupler_dir+"/AEOLUS/", 
            "vulcan": coupler_dir+"/VULCAN/", 
            "spider": coupler_dir+"/SPIDER/", 
            "utils": coupler_dir+"/utils/"
            }
    
    return dirs


# End of file