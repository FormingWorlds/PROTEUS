# Functions used to help run PROTEUS which are mostly submodule agnostic.

# Import utils-specific modules
from utils.modules_ext import *
from utils.constants import *
from utils.spider import *
from utils.helper import *

log = logging.getLogger("PROTEUS")

import plot.cpl_atmosphere as cpl_atmosphere
import plot.cpl_global as cpl_global
import plot.cpl_stacked as cpl_stacked
import plot.cpl_interior as cpl_interior
import plot.cpl_sflux as cpl_sflux
import plot.cpl_sflux_cross as cpl_sflux_cross
import plot.cpl_fluxes as cpl_fluxes
import plot.cpl_interior_cmesh as cpl_interior_cmesh

# Handle optional command line arguments for PROTEUS
def parse_console_arguments()->dict:
    parser = argparse.ArgumentParser(description='PROTEUS command line arguments')
    parser.add_argument('--cfg', type=str, default="init_coupler.cfg", help='Path to configuration file')
    args = vars(parser.parse_args())
    return args

# https://stackoverflow.com/questions/13490292/format-number-using-latex-notation-in-python
def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

def PrintCurrentState(time_dict, runtime_helpfile, COUPLER_options):
    PrintHalfSeparator()
    log.info("Runtime info...")
    log.info("    System time  :   %s  "         % str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    log.info("    Model time   :   %.3e   yr"    % float(time_dict["planet"]))
    log.info("    T_surf       :   %.3e   K"     % float(runtime_helpfile.iloc[-1]["T_surf"]))
    log.info("    P_surf       :   %.3e   bar"   % float(runtime_helpfile.iloc[-1]["P_surf"]))
    log.info("    Phi_global   :   %.3e   "      % float(runtime_helpfile.iloc[-1]["Phi_global"]))
    log.info("    Instellation :   %.3e   W/m^2" % float(runtime_helpfile.iloc[-1]["F_ins"]))
    log.info("    F_int        :   %.3e   W/m^2" % float(runtime_helpfile.iloc[-1]["F_int"]))
    log.info("    F_atm        :   %.3e   W/m^2" % float(runtime_helpfile.iloc[-1]["F_atm"])) 
    log.info("    |F_net|      :   %.3e   W/m^2" % abs(float(runtime_helpfile.iloc[-1]["F_net"])))


def GetHelpfileKeys():
    '''
    Variables to be held in the helpfile
    '''

    # Basic keys
    keys = [
            # Model tracking 
            "Time", 

            # Stellar 
            "R_star", 

            # Temperatures 
            "T_surf", "T_mantle", "T_eqm", 

            # Fluxes 
            "F_int", "F_atm", "F_net", "F_olr", "F_ins", "F_sct", 

            # Surface composition
            "P_surf", "atm_kg_per_mol", # gases will be added below
            
            # Interior properties
            "Phi_global", "RF_depth", "M_core", "M_mantle", "M_mantle_solid", "M_mantle_liquid"
            ]

    # gases
    for s in volatile_species:
        keys.append[s+"_mol_atm"]   
        keys.append[s+"_mol_solid"] 
        keys.append[s+"_mol_liquid"]
        keys.append[s+"_mol_total"] 
        keys.append[s+"_kg_atm"]   
        keys.append[s+"_kg_solid"] 
        keys.append[s+"_kg_liquid"]
        keys.append[s+"_kg_total"] 
        keys.append[s+"_vmr"] 
        keys.append[s+"_bar"] 

    # element masses
    for e in element_list:
        keys.append[e+"_kg_atm"]   
        keys.append[e+"_kg_solid"] 
        keys.append[e+"_kg_liquid"]
        keys.append[e+"_kg_total"] 

    # elemental ratios
    for e1 in element_list:
        for e2 in element_list:
            if e1==e2:
                continue 
            k = "%s/%s"%(e1,e2)
            if k in keys:
                continue 
            keys.append(k)
        
    return keys 

def CreateHelpfile():
    '''
    Create helpfile to hold output variables.
    '''
    
    return pd.DataFrame( columns=GetHelpfileKeys())

def ZeroHelpfileRow():
    '''
    Get a dictionary with same keys as helpfile but with values of zero
    '''
    out = {}
    for k in GetHelpfileKeys():
        out[k] = 0.0
    return out

def ExtendHelpfile(current_hf:pd.DataFrame, new_row:dict):
    '''
    Extend helpfile with new row of variables
    '''
    
    # validate keys 
    missing_keys = set(GetHelpfileKeys()) - set(new_row.keys())
    if len(missing_keys)>0:
        raise Exception("Cannot add row to helpfile dataframe because it is missing keys: %s"%missing_keys)
    
    # convert row to df 
    new_row = pd.DataFrame([new_row])

    # concatenate and return
    return pd.concat([current_hf, new_row], ignore_index=True) 


def WriteHelpfileToCSV(current_hf):
    '''
    Write helpfile to a CSV file 
    '''
    fpath = os.path.join(dirs["output"] , "runtime_helpfile.csv")
    if os.path.exists(fpath):
        os.remove(fpath)

    current_hf.to_csv(fpath , index=False, sep="\t")
    return 


def GetLastRowFromHelpfile(current_hf:pd.DataFrame):
    '''
    Return the last row of the helpfile as a dictionary
    '''
    return current_hf.loc[-1].to_dict()


def UpdateHelpfile(loop_counter, dirs, time_dict, runtime_helpfile, input_flag, COUPLER_options, solvevol_dict=None):

    runtime_helpfile_name = "runtime_helpfile.csv"
    COUPLER_options_name  = "COUPLER_options.csv"



    # For "Interior" sub-loop (SPIDER)
    if input_flag == "Interior":



    # For "Atmosphere" sub-loop (VULCAN+SOCRATES) update heat flux from SOCRATES
    elif input_flag == "Atmosphere":

        # Define input flag
        runtime_helpfile_new["Input"]           = input_flag   

        # Infos from latest interior loop
        runtime_helpfile_new["R_star"]          = run_int.iloc[-1]["R_star"] 
        runtime_helpfile_new["Phi_global"]      = run_int.iloc[-1]["Phi_global"]
        runtime_helpfile_new["RF_depth"]        = run_int.iloc[-1]["RF_depth"]     
        runtime_helpfile_new["M_mantle"]        = run_int.iloc[-1]["M_mantle"]       
        runtime_helpfile_new["M_core"]          = run_int.iloc[-1]["M_core"]         
        runtime_helpfile_new["M_mantle_liquid"] = run_int.iloc[-1]["M_mantle_liquid"]
        runtime_helpfile_new["M_mantle_solid"]  = run_int.iloc[-1]["M_mantle_solid"]
        runtime_helpfile_new["Time"]            = run_int.iloc[-1]["Time"]
        runtime_helpfile_new["F_int"]           = run_int.iloc[-1]["F_int"]

        # From latest atmosphere iteration
        runtime_helpfile_new["T_eqm"]           = COUPLER_options["T_eqm"]
        runtime_helpfile_new["T_surf"]          = COUPLER_options["T_surf"] 
        runtime_helpfile_new["F_atm"]           = COUPLER_options["F_atm"]
        runtime_helpfile_new["F_ins"]           = COUPLER_options["F_ins"]
        runtime_helpfile_new["F_sct"]           = COUPLER_options["F_sct"]

        COUPLER_options["F_int"] = run_int.iloc[-1]["F_int"]
        COUPLER_options["F_net"] = COUPLER_options["F_atm"] - COUPLER_options["F_int"]

        ### Adjust F_net to break atm main loop:
        t_curr          = run_int.iloc[-1]["Time"]
        run_atm         = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere'].drop_duplicates(subset=['Time'], keep='last')
        run_atm_last    = run_atm.loc[run_atm['Time'] != t_curr]

        # Write F_net to next file
        runtime_helpfile_new["F_net"]           = COUPLER_options["F_net"]
        runtime_helpfile_new["F_olr"]           = COUPLER_options["F_olr"]

        # Other info from latest iteration run (X/H ratios stay fixed w/o loss)
        runtime_helpfile_new["P_surf"]          = runtime_helpfile.iloc[-1]["P_surf"]         
        runtime_helpfile_new["M_atm"]           = runtime_helpfile.iloc[-1]["M_atm"]
        runtime_helpfile_new["atm_kg_per_mol"]  = runtime_helpfile.iloc[-1]["atm_kg_per_mol"]

    runtime_helpfile_new = pd.DataFrame(runtime_helpfile_new,index=[0])
    runtime_helpfile = pd.concat([runtime_helpfile, runtime_helpfile_new])

    runtime_helpfile.to_csv( dirs["output"]+"/"+runtime_helpfile_name, index=False, sep="\t")

    # Save COUPLER_options to disk
    COUPLER_options_save = pd.read_csv(dirs["output"]+"/"+COUPLER_options_name, sep="\t")
    COUPLER_options_df = pd.DataFrame.from_dict([COUPLER_options])
    COUPLER_options_save = pd.concat([ COUPLER_options_save, COUPLER_options_df],ignore_index=True)
    COUPLER_options_save.to_csv( dirs["output"]+"/"+COUPLER_options_name, index=False, sep="\t")

    return runtime_helpfile, time_dict, COUPLER_options

# Calculate eqm temperature given stellar flux, ASF scale factor, and bond albedo
def calc_eqm_temperature(I_0, ASF_sf, A_B):
    return (I_0 * ASF_sf * (1.0 - A_B) / const_sigma)**(1.0/4.0)


def ReadInitFile( init_file_passed , verbose=False):

    # Read in input file as dictionary
    COUPLER_options  = {}
    time_dict       = {}
    if verbose: 
        log.info("Read in init file:" + init_file)

    if os.path.isfile(init_file_passed):
        init_file = init_file_passed
    else: 
        raise Exception("Init file provided is not a file or does not exist (%s)" % init_file_passed)

    if verbose: log.info("Settings:")

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

                if verbose: log.info(line)

                # Assign key and value
                (key, val) = line.split("=")
                key = key.strip()
                val = val.strip()
                
                # Standard options
                if not line.startswith("time_"):

                    # Some parameters are int
                    if key in [ "solid_stop", "steady_stop", "iter_max", "emit_stop",
                                "plot_iterfreq", "stellar_heating", "mixing_length", 
                                "atmosphere_chemistry", "solvevol_use_params", "insert_rscatter", "water_cloud",
                                "tropopause", "F_atm_bc", "atmosphere_solve_energy", "atmosphere_surf_state",
                                "dt_dynamic", "prevent_warming", "atmosphere_model", "atmosphere_nlev"]:
                        val = int(val)

                    # Some are str
                    elif key in [ 'star_spectrum', 'dir_output', 
                                  'spectral_file' , 'log_level']:
                        val = str(val)
                        
                    # Most are float
                    else:
                        val = float(val)

                    # Set option
                    COUPLER_options[key] = val


                # Time options
                elif line.startswith("time_"):

                        line = line.split("_")[1]

                        (key, val) = line.split("=")
                    
                        time_dict[str(key.strip())] = float(val.strip())

                        if line.startswith("star"):
                            time_dict["offset"] = float(val.strip())

    # Calculate gravity from mass and radius
    COUPLER_options["gravity"] =  const_G * COUPLER_options["mass"] / (COUPLER_options["radius"] * COUPLER_options["radius"])

    return COUPLER_options, time_dict

def UpdatePlots( output_dir, COUPLER_options, end=False, num_snapshots=7):
    """Update plots during runtime for analysis
    
    Calls various plotting functions which show information about the interior/atmosphere's energy and composition.

    Parameters
    ----------
        output_dir : str
            Output directory containing simulation information
        end : bool
            Is this function being called at the end of the simulation?
    """

    PrintHalfSeparator()
    log.info("Updating plots...")

    # Get all JSON files
    output_times = get_all_output_times( output_dir )

    # Global properties for all timesteps
    if len(output_times) > 1:
        cpl_global.plot_global(output_dir, COUPLER_options)   

    # Check if we are using the dummy atmosphere
    dummy_atm = (COUPLER_options["atmosphere_model"] == 2)
        
    # Filter to JSON files with corresponding NetCDF files
    if not dummy_atm:
        ncs = glob.glob(output_dir + "/data/*_atm.nc")
        nc_times = [int(f.split("/")[-1].split("_atm")[0]) for f in ncs]
        output_times = sorted(list(set(output_times) & set(nc_times)))

    # Work out which times we want to plot
    if len(output_times) <= num_snapshots:
        plot_times = output_times

    else:
        plot_times = []
        tmin = max(1,np.amin(output_times))
        tmax = max(tmin+1, np.amax(output_times))
    
        for s in np.logspace(np.log10(tmin),np.log10(tmax),num_snapshots): # Sample on log-scale

            remaining = list(set(output_times) - set(plot_times)) 
            if len(remaining) == 0:
                break

            v,_ = find_nearest(remaining,s) # Find next new sample
            plot_times.append(int(v))

    plot_times = sorted(set(plot_times)) # Remove any duplicates + resort
    log.debug("Snapshots to plot:" + str(plot_times))

    # Specific timesteps for paper plots
    cpl_interior.plot_interior(output_dir, plot_times)     
    if not dummy_atm:
        cpl_atmosphere.plot_atmosphere(output_dir, plot_times)
        cpl_stacked.plot_stacked(output_dir, plot_times)

        if COUPLER_options["atmosphere_model"] != 1:
            # don't make this plot for AGNI, since it will do it itself
            nc_path = output_dir + "/data/%d_atm.nc"%(int(output_times[-1]))
            cpl_fluxes.plot_fluxes_atmosphere(output_dir, nc_path)


    # Only at the end of the simulation
    if end:
        cpl_global.plot_global(output_dir, COUPLER_options, logt=False)   
        cpl_interior_cmesh.plot_interior_cmesh(output_dir)
        cpl_sflux.plot_sflux(output_dir)
        cpl_sflux_cross.plot_sflux_cross(output_dir)
        cpl_fluxes.plot_fluxes_global(output_dir, COUPLER_options)
 
    # Close all figures
    plt.close()

def SetDirectories(COUPLER_options: dict):
    """Set directories dictionary
    
    Sets paths to the required directories, based on the configuration provided
    by the options dictionary. 

    Parameters
    ----------
        COUPLER_options : dict
            PROTEUS options dictionary

    Returns
    ----------
        dirs : dict
            Dictionary of paths to important directories
    """

    coupler_dir = os.path.abspath(os.getenv('COUPLER_DIR'))

    # PROTEUS folders
    dirs = {
            "output":   os.path.join(coupler_dir,"output",COUPLER_options['dir_output']), 
            "input":    os.path.join(coupler_dir,"input"),
            "coupler":  coupler_dir,
            "janus":    os.path.join(coupler_dir,"JANUS"),
            "agni":     os.path.join(coupler_dir,"AGNI"),
            "vulcan":   os.path.join(coupler_dir,"VULCAN"),
            "spider":   os.path.join(coupler_dir,"SPIDER"),
            "utils":    os.path.join(coupler_dir,"utils")
            }
    
    # FWL data folder
    if os.environ.get('FWL_DATA') == None:
        UpdateStatusfile(dirs, 20)
        raise Exception("The FWL_DATA environment variable where spectral"
                        "and evolution tracks data will be downloaded needs to be set up!"
                        "Did you source PROTEUS.env?")
    else:
        dirs["fwl"] = os.environ.get('FWL_DATA')
    

    # SOCRATES directory
    if COUPLER_options["atmosphere_model"] in [0,1]:
        # needed for atmosphere models 0 and 1
        
        if os.environ.get('RAD_DIR') == None:
            UpdateStatusfile(dirs, 20)
            raise Exception("The RAD_DIR environment variable has not been set")
        else:
            dirs["rad"] = os.environ.get('RAD_DIR')
    
    # Get abspaths
    for key in dirs.keys():
        dirs[key] = os.path.abspath(dirs[key])+"/"

    return dirs


# End of file
