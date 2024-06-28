# Functions used to handle atmosphere temperature structure (running AGNI, etc.)

from utils.modules_ext import *
from utils.helper import *
from utils.constants import *

import tomlkit as toml

log = logging.getLogger("PROTEUS")

def _try_agni(loop_counter:dict, dirs:dict, COUPLER_options:dict, 
              runtime_helpfile, make_plots:bool, initial_offset:float, 
              linesearch:bool)->bool:

    # ---------------------------
    # Setup values to be provided to AGNI
    # ---------------------------
    gravity  = const_G * COUPLER_options["mass"] / (COUPLER_options["radius"])**2
    agni_debug = bool(log.getEffectiveLevel() == logging.DEBUG)
    try_spfile = os.path.join(dirs["output"] , "runtime.sf")
    
    # Get stellar spectrum at TOA
    sflux_files = glob.glob(dirs["output"]+"/data/*.sflux")
    sflux_times = [ int(s.split("/")[-1].split(".")[0]) for s in sflux_files]
    sflux_tlast = sorted(sflux_times)[-1]
    sflux_path  = dirs["output"]+"/data/%d.sflux"%sflux_tlast

    # store VMRs
    vol_dict = {}
    for vol in volatile_species:
        if COUPLER_options[vol+"_included"]:
            vmr = runtime_helpfile.iloc[-1][vol+"_mr"]
            if vmr > 1e-40:
                vol_dict[vol] = vmr 
    
    if len(vol_dict) == 0:
        UpdateStatusfile(dirs, 20)
        raise Exception("All volatiles have a volume mixing ratio of zero")
    
    if COUPLER_options["tropopause"] not in [0,1]:
        UpdateStatusfile(dirs, 20)
        raise Exception("Tropopause type not supported by AGNI")

    # ---------------------------
    # Make configuration file
    # ---------------------------

    log.debug("Write cfg file")

    # Read base AGNI configuration file
    cfg_base = os.path.join(dirs["utils"] , "init_agni.toml")
    with open(cfg_base, 'r') as hdl:
        cfg_toml = toml.load(hdl) 

    # Setup new AGNI configuration file for this run
    cfg_this = os.path.join(dirs["output"] , "agni_recent.toml")

    # Set title 
    cfg_toml["title"] = "PROTEUS runtime step %d"%loop_counter["total"]

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
    cfg_toml["planet"]["vmr"] =             vol_dict
    cfg_toml["planet"]["condensates"] =     list(vol_dict.keys())

    # Set files
    cfg_toml["files"]["output_dir"] =       os.path.join(dirs["output"])
    if os.path.exists(try_spfile):
        # exists => don't modify it
        cfg_toml["files"]["input_sf"] =     try_spfile
        cfg_toml["files"]["input_star"] =   ""   
    else:
        # doesn't exist => AGNI will copy it + modify as required
        cfg_toml["files"]["input_sf"] =     COUPLER_options["spectral_file"]
        cfg_toml["files"]["input_star"] =   sflux_path
    
    # Set execution
    cfg_toml["execution"]["num_levels"] =   COUPLER_options["atmosphere_nlev"]
    cfg_toml["execution"]["rayleigh"] =     bool(COUPLER_options["insert_rscatter"] == 1)
    cfg_toml["execution"]["cloud"] =        bool(COUPLER_options["water_cloud"] == 1)
    cfg_toml["execution"]["linesearch"] =   linesearch
    if COUPLER_options["atmosphere_solve_energy"] == 0:
        # The default cfg assumes solving for energy balance.
        # If we don't want to do that, set the configuration to a prescribed
        # profile of: T(p) = dry + condensing + stratosphere
        initial_arr = ["sat", "dry"]
        if COUPLER_options["tropopause"] == 1:
            initial_arr.extend(["str", "%.3e"%COUPLER_options["T_skin"]])
        cfg_toml["execution"]["initial_state"] = initial_arr
        
        # Tell AGNI not to solve for RCE
        cfg_toml["execution"]["solvers"] = []
        cfg_toml["execution"]["convection_type"] = ""

    elif loop_counter["total"] > 1:
        # If solving for RCE and are current inside the init stage, use old T(p)
        # as initial guess for solver.
        ncdfs = glob.glob(os.path.join(dirs["output"], "data","*_atm.nc"))
        ncdf_times = [float(f.split("/")[-1].split("_")[0]) for f in ncdfs]
        nc_path = ncdfs[np.argmax(ncdf_times)]

        log.debug("Initialise from last T(p)")
        cfg_toml["execution"]["initial_state"] = ["ncdf", nc_path, "add", "%.6f"%initial_offset]

    # Solution stuff 
    surf_state = int(COUPLER_options["atmosphere_surf_state"])
    if not (0 <= surf_state <= 3):
        UpdateStatusfile(dirs, 20)
        raise Exception("Invalid surface state %d" % surf_state)

    # CBL case
    if surf_state == 2:
        if COUPLER_options["atmosphere_solve_energy"] == 0:
            UpdateStatusfile(dirs, 20)
            raise Exception("With AGNI it is necessary to an energy-conserving solver alongside the conductive lid scheme. Turn them both on or both off.")
        cfg_toml["planet"]["skin_k"] =    COUPLER_options["skin_k"]
        cfg_toml["planet"]["skin_d"] =    COUPLER_options["skin_d"]
        cfg_toml["planet"]["tmp_magma"] = COUPLER_options["T_surf"]

    # Solution type ~ surface state
    cfg_toml["execution"]["solution_type"] = surf_state

    # Small steps after first iters, since it will be *near* the solution
    # Tighter tolerances during first iters, to ensure consistent coupling
    if loop_counter["total"] > loop_counter["init_loops"]+1:
        cfg_toml["execution"]["dx_max"] = 50.0
    else:
        cfg_toml["execution"]["converge_rtol"] = 1.0e-3
        
    # Set plots 
    cfg_toml["plots"]["at_runtime"]     = agni_debug and make_plots
    cfg_toml["plots"]["temperature"]    = make_plots
    cfg_toml["plots"]["fluxes"]         = make_plots


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
        log.info("AGNI output suppressed (see agni.log)")
        agni_stdout = subprocess.DEVNULL

    # Call the module
    log.debug("Call AGNI subprocess - output below...")
    call_sequence = [ os.path.join(dirs["agni"],"agni.jl"), cfg_this]
    proc = subprocess.run(call_sequence, stdout=agni_stdout, stderr=sys.stdout) 
    success = (proc.returncode == 0)
    
    # Copy AGNI log into PROTEUS log
    # There are probably better ways to do this, but it works well enough. We don't use agni_debug much anyway
    if agni_debug:
        with open(os.path.join(dirs["output"], "std.log"), "a") as outfile:
            with open(os.path.join(dirs["output"], "agni.log"), "r") as infile:
                outfile.write(infile.read())
    
    return success 

def RunAGNI(loop_counter, time_dict, dirs, COUPLER_options, runtime_helpfile ):
    """Run AGNI atmosphere model.
    
    Calculates the temperature structure of the atmosphere and the fluxes, etc.
    Stores the new flux boundary condition to be provided to SPIDER. 

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

    # Inform
    PrintHalfSeparator()
    log.info("Running AGNI...")
    time_str = "%d"%time_dict["planet"]
    make_plots = (COUPLER_options["plot_iterfreq"] > 0) and (loop_counter["total"] % COUPLER_options["plot_iterfreq"] == 0)

    # tracking
    agni_success = False  # success?
    attempts = 0          # number of attempts so far
    max_attempts = 4      # max attempts
    linesearch = True
    offset = 0.0

    # make attempts
    while not agni_success:
        attempts += 1
        log.info("Attempt %d" % attempts)

        # Try the module
        agni_success = _try_agni(loop_counter, dirs, COUPLER_options, runtime_helpfile, make_plots, offset, linesearch)

        if agni_success:
            # success
            log.info("Attempt %d succeeded" % attempts)
            break
        else:
            # failure
            log.warning("Attempt %d failed" % attempts)
            if attempts >= max_attempts:
                UpdateStatusfile(dirs, 22)
                raise Exception("Max attempts when executing AGNI")
            else:
                # try again with offset to initial T(p)
                offset = attempts * 0.2
                if attempts%2 == 0:
                    offset *= -1
                # enable LS
                linesearch = True

    # Move files
    log.debug("Tidy files")
    files_move = [  
                    ("atm.nc", "data/"+time_str+"_atm.nc"),
                    ("agni.log", "agni_recent.log")
                 ]
    if make_plots:
        files_move.append(("plot_fluxes.png", "plot_fluxes_atmosphere.png"))
    for pair in files_move:
        p_inp = os.path.join(dirs["output"], pair[0])
        p_out = os.path.join(dirs["output"], pair[1])
        safe_rm(p_out)
        shutil.move(p_inp, p_out)

    # Remove files
    files_remove = ["plot_ptprofile.png", "plot_vmrs.png", "fl.csv", "pt_ini.csv", "pt.csv", "agni.cfg", "solver_flx.png", "solver_prf.png", "solver_mon.png"] 
    for frem in files_remove:
        frem_path = os.path.join(dirs["output"],frem)
        safe_rm(frem_path)

    # ---------------------------
    # Read results
    # ---------------------------
    
    log.debug("Read results")
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

