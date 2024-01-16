# Functions used to handle atmosphere temperature structure (running AGNI, etc.)

from utils.modules_ext import *
from utils.helper import *
from utils.constants import *


def RunAGNI(loop_counter, time_dict, dirs, COUPLER_options, runtime_helpfile ):
    """Run AGNI.
    
    Calculates the temperature structure of the atmosphere and the fluxes, etc.
    Stores the new flux boundary condition to be provided to SPIDER. Limits flux
    change if required.

    Parameters
    ----------
        loop_counter : dict 
            Model loop counter values.
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

    # Setup values to be provided by CLI
    gravity = const_G * COUPLER_options["mass"] / (COUPLER_options["radius"])**2
    
    csv_fpath = dirs["output"]+"pt.csv"

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
    mr_str = "\""
    mrzero = True
    for k in vol_list.keys():
        v = vol_list[k]
        if v < 1.0e-20:
            continue
        mrzero = False
        mr_str += str("%s=%1.6f," % (k,v) )
    mr_str = mr_str[:-1]
    mr_str += "\""
    if mrzero:
        raise Exception("All volatiles have a volume mixing ratio of zero")

    # Setup call sequence with base flags
    call_sequence = []
    call_sequence.append(dirs["agni"]+"agni_cli.jl")
    call_sequence.append("%1.5e" % COUPLER_options["T_surf"])
    call_sequence.append("%1.5e" % COUPLER_options["F_ins"]) 
    call_sequence.append("%1.5e" % COUPLER_options["zenith_angle"]) 
    call_sequence.append("%1.5e" % COUPLER_options["asf_scalefactor"]) 
    call_sequence.append("%1.5e" % COUPLER_options["albedo_pl"]) 
    call_sequence.append("%1.5e" % gravity)
    call_sequence.append("%1.5e" % COUPLER_options["radius"])
    call_sequence.append("%1.5e" % runtime_helpfile.iloc[-1]["P_surf"])
    call_sequence.append("%1.5e" % COUPLER_options["P_top"])
    call_sequence.append("--x_dict %s"              % mr_str)
    call_sequence.append("--sp_file \"%s\""         % str(dirs["output"]+"runtime_spectral_file"))  # already includes star
    call_sequence.append("--albedo_s %1.5e"         % COUPLER_options["albedo_s"])
    call_sequence.append("--output %s"              % dirs["output"])
    call_sequence.append("--tmp_magma %1.5e"        % COUPLER_options["T_surf"])  # magma temperature given by SPIDER output
    call_sequence.append("--tmp_floor %1.5e"        % COUPLER_options["min_temperature"])

    # Rayleigh scattering
    if COUPLER_options["insert_rscatter"] == 1:
        call_sequence.append("--rscatter")

    # Convection scheme
    call_sequence.append("--convect_mlt")

    # Solution type
    load_prev = False
    if COUPLER_options["atmosphere_solve_energy"] == 0:
        # Prescribed dry adiabat + condensation curve
        call_sequence.append("--ini_dry")
        call_sequence.append("--ini_sat")
    else:
        # Solving for RCE
        call_sequence.append("--nlsolve")

        # Start from previous CSV file?
        # If not, the model will start from an isothermal state at T=tstar
        # if os.path.exists(csv_fpath) and (time_dict["planet"] > 0):
        #     load_prev = True  
        #     call_sequence.append("--pt_path %s" % csv_fpath)
        # else :
        #     call_sequence.append("--dtsolve")
        if (time_dict["planet"] < 1):
            call_sequence.append("--dtsolve")

    # Tropopause
    if not load_prev:
        match COUPLER_options["tropopause"]:
            case 0:
                pass
            case 1:
                call_sequence.append("--trppt %1.5e" % COUPLER_options["T_skin"])
            case _:
                raise Exception("Tropopause type not supported by AGNI")
            
    # Surface condition
    surf_state = int(COUPLER_options["atmosphere_surf_state"])
    if (0 <= surf_state <= 2):

        # Don't allow free T_surf on first call (for stability)
        if (loop_counter["total"] <= 1) and (surf_state == 2):
            surf_state = 0
        
        call_sequence.append("--surface %d" % surf_state)

        # Conductive skin case
        if surf_state == 2:
            
            if COUPLER_options["atmosphere_solve_energy"] == 0:
                raise Exception("It is necessary to an energy-conserving solver alongside the conductive lid scheme! Turn them both on or both off.")
            
            call_sequence.append("--skin_k %1.6e" % COUPLER_options["skin_k"])
            call_sequence.append("--skin_d %1.6e" % COUPLER_options["skin_d"])
            
        else:
            call_sequence.append("--tstar_enforce")  # do not allow pt_path to overwrite tstar

    else:
        raise Exception("Invalid surface state %d" % surf_state)

    # Misc flags
    if debug:
        call_sequence.append("--verbose")
    call_sequence.append("--plot")
    call_sequence.append("--nsteps 500") 
    call_sequence.append("--nlevels %d" % int(COUPLER_options["atmosphere_nlev"]))

    call_sequence.append("--convcrit_tmprel   %1.4e" % 2.5 )
    call_sequence.append("--convcrit_fradrel  %1.4e" % 0.1 )
    call_sequence.append("--convcrit_flosspct %1.4e" % 2.0 )

    # Run AGNI
    call_string = " ".join(call_sequence)
    agni_print = open(dirs["output"]+"agni_recent.log",'w')
    subprocess.run([call_string], shell=True, check=True, stdout=sys.stdout, stderr=agni_print)
    agni_print.close()

    # Read result
    nc_fpath = dirs["output"]+"data/"+str(int(time_dict["planet"]))+"_atm.nc"
    shutil.move(dirs["output"]+"atm.nc", nc_fpath)  # read data

    # Move fluxes plot file
    fl_path = dirs["output"]+"plot_fluxes_atmosphere.pdf"   
    if os.path.exists(fl_path):
        os.remove(fl_path)
    shutil.move(dirs["output"]+"fl.pdf", fl_path)

    files_remove = ["pt.pdf", "mf.pdf"]  # remove files
    files_remove.extend(glob.glob("zzframe_*.png"))
    files_remove.extend(glob.glob("zyframe_*.png"))
    for frem in files_remove:
        frem_path = dirs["output"]+"/"+frem 
        if os.path.exists(frem_path):
            os.remove(frem_path)

    # Read AGNI output NetCDF file
    ds = nc.Dataset(nc_fpath)
    net_flux =      np.array(ds.variables["fl_N"][:])
    LW_flux_up =    np.array(ds.variables["fl_U_LW"][:])
    T_surf =        float(ds.variables["tmpl"][-1])
    ds.close()

    # New flux from SOCRATES
    if (COUPLER_options["F_atm_bc"] == 0):
        F_atm_new = net_flux[0] 
    else:
        F_atm_new = net_flux[-1]  

    # Require that the net flux must be upward (positive)
    if (COUPLER_options["prevent_warming"] == 1):
        F_atm_new = max( 1e-8 , F_atm_new )
            
    COUPLER_options["F_atm"]  = F_atm_new
    COUPLER_options["F_olr"]  = LW_flux_up[0]
    COUPLER_options["T_surf"] = T_surf
    
    print("SOCRATES fluxes (net@BOA, net@TOA, OLR): %.3f, %.3f, %.3f W/m^2" % (net_flux[-1], net_flux[0] ,LW_flux_up[0]))

    return COUPLER_options

