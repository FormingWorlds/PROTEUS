# Functions used to handle atmosphere temperature structure (running AGNI, etc.)

from utils.modules_ext import *
from utils.helper import *
from utils.constants import *

import tomlkit as toml

log = logging.getLogger(__name__)

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
    log.info("Running AGNI...")

    # ---------------------------
    # Setup values to be provided to AGNI
    # ---------------------------
    gravity  = const_G * COUPLER_options["mass"] / (COUPLER_options["radius"])**2
    time_str = "%d"%time_dict["planet"]
    agni_debug = bool(log.getEffectiveLevel() == logging.DEBUG)
    make_plots = (COUPLER_options["plot_iterfreq"] > 0) and (loop_counter["total"] % COUPLER_options["plot_iterfreq"] == 0)

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
    mrzero = True
    for k in vol_list.keys():
        v = vol_list[k]
        if v < 1.0e-20:
            continue
        mrzero = False
    if mrzero:
        UpdateStatusfile(dirs, 20)
        raise Exception("All volatiles have a volume mixing ratio of zero")
    
    if COUPLER_options["tropopause"] not in [0,1]:
        UpdateStatusfile(dirs, 20)
        raise Exception("Tropopause type not supported by AGNI")

    # ---------------------------
    # Make configuration file
    # ---------------------------

    # Read base AGNI configuration file
    cfg_base = os.path.join(dirs["utils"] , "init_agni.toml")
    with open(cfg_base, 'r') as hdl:
        cfg_toml = toml.load(hdl) 

    # Setup new AGNI configuration file for this run
    cfg_this = os.path.join(dirs["output"] , "agni_recent.toml")

    # Set title 
    cfg_toml["title"] = "PROTEUS configured"

    # Set planet 
    cfg_toml["planet"]["tmp_surf"] =        COUPLER_options["T_surf"]
    cfg_toml["planet"]["instellation"] =    COUPLER_options["F_ins"]
    cfg_toml["planet"]["s0_fact"] =         COUPLER_options["asf_scalefactor"]
    cfg_toml["planet"]["albedo_b"] =        COUPLER_options["albedo_pl"]
    cfg_toml["planet"]["zenith_angle"] =    COUPLER_options["zenith_angle"]
    cfg_toml["planet"]["albedo_s"] =        COUPLER_options["albedo_s"]
    cfg_toml["planet"]["gravity"] =         gravity
    cfg_toml["planet"]["radius"] =          COUPLER_options["radius"]
    cfg_toml["planet"]["p_surf"] =          runtime_helpfile.iloc[-1]["P_surf"]
    cfg_toml["planet"]["p_top"] =           COUPLER_options["P_top"]
    cfg_toml["planet"]["vmr"] =             vol_list

    # Set files
    cfg_toml["files"]["input_sf"] =         os.path.join(dirs["output"] , "star.sf")
    cfg_toml["files"]["output_dir"] =       os.path.join(dirs["output"])
    
    # Set execution
    cfg_toml["execution"]["num_levels"] =   COUPLER_options["atmosphere_nlev"]
    cfg_toml["execution"]["rayleigh"] =     bool(COUPLER_options["insert_rscatter"] == 1)
    cfg_toml["execution"]["cloud"] =        bool(COUPLER_options["water_cloud"] == 1)
    if COUPLER_options["atmosphere_solve_energy"] == 0:
        # The default cfg assumes solving for energy balance.
        # If we don't want to do that, set the configuration to a prescribed
        # profile of: T(p) = dry + condensing + stratosphere
        initial_arr = ["sat", "dry"]
        for vol in vol_list.keys():
            initial_arr.extend(["con", vol])
        if COUPLER_options["tropopause"] == 1:
            initial_arr.extend(["str", "%.3e"%COUPLER_options["T_skin"]])
        cfg_toml["execution"]["initial_state"] = initial_arr
        
        # Tell AGNI not to solve for RCE
        cfg_toml["execution"]["solvers"] = []
        cfg_toml["execution"]["dry_convection"] = ""
    
    # Solution stuff 
    surf_state = int(COUPLER_options["atmosphere_surf_state"])
    if (0 <= surf_state <= 3):

        # Stability on first call
        if loop_counter["total"] <= 1:
            surf_state = 0
            cfg_toml["execution"]["stabilise"] = True

        # CBL case
        if surf_state == 2:
            if COUPLER_options["atmosphere_solve_energy"] == 0:
                UpdateStatusfile(dirs, 20)
                raise Exception("With AGNI it is necessary to an energy-conserving solver alongside the conductive lid scheme. Turn them both on or both off.")
            cfg_toml["planet"]["skin_k"] =          COUPLER_options["skin_k"]
            cfg_toml["planet"]["skin_d"] =          COUPLER_options["skin_d"]
            cfg_toml["planet"]["tmp_magma"] =       COUPLER_options["T_surf"]

        cfg_toml["execution"]["solution_type"] = surf_state

    else:
        UpdateStatusfile(dirs, 20)
        raise Exception("Invalid surface state %d" % surf_state)

    # Set plots 
    cfg_toml["plots"]["at_runtime"]     = agni_debug and make_plots
    cfg_toml["plots"]["temperature"]    = make_plots
    cfg_toml["plots"]["fluxes"]         = make_plots
    cfg_toml["plots"]["contribution"]   = make_plots
    cfg_toml["plots"]["emission"]       = make_plots
    cfg_toml["plots"]["albedo"]         = make_plots
    cfg_toml["plots"]["mixing_ratios"]  = make_plots

    # Write new configuration file 
    with open(cfg_this, 'w') as hdl:
        toml.dump(cfg_toml, hdl)

    # ---------------------------
    # Run AGNI
    # ---------------------------

    # Setup output stream
    if agni_debug:
        agni_stdout = sys.stdout 
    else:
        log.info("Terminal output suppressed")
        agni_stdout = subprocess.DEVNULL

    # Call the module
    call_sequence = [ os.path.join(dirs["agni"],"agni.jl"), cfg_this]
    proc = subprocess.run(call_sequence, stdout=agni_stdout, stderr=sys.stdout) 
    if proc.returncode != 0:
        UpdateStatusfile(dirs, 22)
        raise Exception("An error occurred when executing AGNI")
    
    # Copy AGNI log into PROTEUS log
    # There are probably better ways to do this, but it works well enough. We don't use agni_debug much anyway
    if agni_debug:
        with open(os.path.join(dirs["output"], "std.log"), "a") as outfile:
            with open(os.path.join(dirs["output"], "agni.log"), "r") as infile:
                outfile.write(infile.read())
    
    # Move files
    files_move = [  
                    ("plot_fluxes.png", "plot_fluxes_atmosphere.png"),
                    ("atm.nc", "data/"+time_str+"_atm.nc"),
                    ("agni.log", "agni_recent.log")
                 ]
    for pair in files_move:
        p_inp = os.path.join(dirs["output"], pair[0])
        p_out = os.path.join(dirs["output"], pair[1])
        if os.path.exists(p_out):
            os.remove(p_out)
        shutil.move(p_inp, p_out)

    # Remove files
    files_remove = ["plot_ptprofile.png", "plot_vmrs.png", "fl.csv", "pt_ini.csv", "pt.csv", "agni.cfg"] 
    for frem in files_remove:
        frem_path = dirs["output"]+"/"+frem 
        if os.path.exists(frem_path):
            os.remove(frem_path)

    # ---------------------------
    # Read results
    # ---------------------------
    
    ds = nc.Dataset(os.path.join(dirs["output"],"data",time_str+"_atm.nc"))
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
    
    log.info("SOCRATES fluxes (net@BOA, net@TOA, OLR): %.3f, %.3f, %.3f W/m^2" % (net_flux[-1], net_flux[0] ,LW_flux_up[0]))

    return COUPLER_options

