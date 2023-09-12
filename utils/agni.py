# Functions used to handle atmosphere temperature structure (running AGNI, etc.)

from utils.modules_ext import *
from utils.helper import *
from utils.constants import *


def RunAGNI( time_dict, dirs, COUPLER_options, runtime_helpfile ):
    """Run AGNI.
    
    Calculates the temperature structure of the atmosphere and the fluxes, etc.
    Stores the new flux boundary condition to be provided to SPIDER. Limits flux
    change if required.

    Parameters
    ----------
        time_dict : dict
            Dictionary containing simulation time variables
        dirs : dict
            Dictionary containing paths to directories
        COUPLER_options : dict
            Configuration options and other variables
        runtime_helpfile : pd.DataFrame
            Dataframe containing simulation variables (now and historic)

    Returns
    ----------
        COUPLER_options : dict
            Updated configuration options and other variables

    """

    PrintHalfSeparator()
    print("Running AGNI...")

    gravity = phys.G * COUPLER_options["mass"] / (COUPLER_options["radius"])**2

    csv_fpath = dirs["output"]+"/pt.csv"

    mr_str = "\""
    vol_list = { 
                  "H2O" : runtime_helpfile.iloc[-1]["H2O_mr"], 
                  "CO2" : runtime_helpfile.iloc[-1]["CO2_mr"],
                  "H2"  : runtime_helpfile.iloc[-1]["H2_mr"], 
                  "N2"  : runtime_helpfile.iloc[-1]["N2_mr"],  
                  "CH4" : runtime_helpfile.iloc[-1]["CH4_mr"], 
                  "O2"  : runtime_helpfile.iloc[-1]["O2_mr"], 
                  "CO"  : runtime_helpfile.iloc[-1]["CO_mr"], 
                  "He"  : runtime_helpfile.iloc[-1]["He_mr"],
                  "NH3" : 0.0 # not supported by SPIDER 
                }
    for k in vol_list.keys():
        v = vol_list[k]
        mr_str += str("%s=%1.6f," % (k,v) )
    mr_str = mr_str[:-1]
    mr_str += "\""

    # Setup call sequence
    call_sequence = []
    call_sequence.append(dirs["agni"]+"agni_cli.jl")
    call_sequence.append("%1.6e" % COUPLER_options["T_surf"])
    call_sequence.append("%1.6e" % COUPLER_options["TOA_heating"])   # includes albedo_pl already
    call_sequence.append("%1.6e" % gravity)
    call_sequence.append("%1.6e" % COUPLER_options["radius"])
    call_sequence.append("%1.6e" % runtime_helpfile.iloc[-1]["P_surf"])
    call_sequence.append("%1.6e" % COUPLER_options["P_top"])
    call_sequence.append(mr_str)
    call_sequence.append("--spf \"%s\""             % str(dirs["output"]+"runtime_spectral_file"))  # already includes star
    call_sequence.append("--albedo_s %1.6e"         % COUPLER_options["albedo_s"])
    call_sequence.append("--zenith_degrees %1.6e"   % COUPLER_options["zenith_angle"])
    call_sequence.append("--output %s"              % dirs["output"])

    if COUPLER_options["insert_rscatter"] == 1:
        call_sequence.append("--rscatter")
    
    if COUPLER_options["atmosphere_solve_energy"] == 0:
        call_sequence.append("--once")
        call_sequence.append("--ini_dry")
        call_sequence.append("--ini_sat")
    else:
        # Solving for RCE...

        # Start from previous CSV file?
        if (time_dict["planet"] > 3.0) and os.path.exists(csv_fpath):
            call_sequence.append("--load_csv %s" % csv_fpath)

        # If not, the model will start from an isothermal state at T=tstar

    trppt = COUPLER_options["min_temperature"] # use tropopause functionality to introduce temperature floor
    match COUPLER_options["tropopause"]:
        case 0:
            pass
        case 1:
            trppt = max(COUPLER_options["T_skin"], trppt)
        case _:
            raise Exception("Tropopause type not supported by AGNI")
    call_sequence.append("--trppt %1.6e" % trppt)  

    if COUPLER_options["atmosphere_surf_state"] == 0:
        call_sequence.append("--surface 0")
    else:
        call_sequence.append("--surface 2")

    if debug:
        call_sequence.append("--verbose")

    call_string = " ".join(call_sequence)

    print(call_string)

    # Run AGNI
    if debug:
        agni_print = sys.stdout
    else:
        agni_print = open(dirs["output"]+"agni_recent.log",'w')
    subprocess.run([call_string],shell=True,check=True,stdout=agni_print)

    if not debug:
        agni_print.close()

    # Read result
    nc_fpath = dirs["output"]+"/data/"+str(int(time_dict["planet"]))+"_atm.nc"
    os.remove(dirs["output"]+"/fl.csv")
    shutil.move(dirs["output"]+"/atm.nc", nc_fpath)

    ds = nc.Dataset(nc_fpath)
    net_flux =      np.array(ds.variables["fl_N"][:])
    LW_flux_up =    np.array(ds.variables["fl_U_LW"][:])
    ds.close()

    # New flux from SOCRATES
    if (COUPLER_options["F_atm_bc"] == 0):
        F_atm_new = net_flux[0] 
    else:
        F_atm_new = net_flux[-1]  

    # Require that the net flux must be upward (positive)
    if (COUPLER_options["prevent_warming"] == 1):
        F_atm_new = max( 1e-5 , F_atm_new )
            
    COUPLER_options["F_atm"] = F_atm_new
    COUPLER_options["F_olr"] = LW_flux_up[0]
    
    print("SOCRATES fluxes (net@BOA, net@TOA, OLR): %.3f, %.3f, %.3f W/m^2" % (net_flux[-1], net_flux[0] ,LW_flux_up[0]))

    return COUPLER_options

