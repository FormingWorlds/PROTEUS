# Functions used to handle atmosphere temperature structure (running AGNI, etc.)

from utils.modules_ext import *
from utils.helper import *
from utils.constants import *
from utils.logs import GetCurrentLogfilePath

import tomlkit as toml

log = logging.getLogger("PROTEUS")

def _try_agni(loops_total:int, dirs:dict, COUPLER_options:dict, 
              hf_row:dict, make_plots:bool, initial_offset:float, easy_start:bool,
              linesearch:bool, dx_max:float, resume_prev:bool)->bool:

    # ---------------------------
    # Setup values to be provided to AGNI
    # ---------------------------
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
            vmr = hf_row[vol+"_vmr"]
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
    cfg_base = os.path.join(dirs["utils"] , "templates", "init_agni.toml")
    with open(cfg_base, 'r') as hdl:
        cfg_toml = toml.load(hdl) 

    # Setup new AGNI configuration file for this run
    cfg_this = os.path.join(dirs["output"] , "agni_recent.toml")

    # Set title 
    cfg_toml["title"] = "PROTEUS runtime step %d"%loops_total

    # Set planet 
    cfg_toml["planet"]["tmp_surf"] =        hf_row["T_surf"]
    cfg_toml["planet"]["instellation"] =    hf_row["F_ins"]
    cfg_toml["planet"]["s0_fact"] =         COUPLER_options["asf_scalefactor"]
    cfg_toml["planet"]["albedo_b"] =        COUPLER_options["albedo_pl"]
    cfg_toml["planet"]["zenith_angle"] =    COUPLER_options["zenith_angle"]
    cfg_toml["planet"]["albedo_s"] =        COUPLER_options["albedo_s"]
    cfg_toml["planet"]["gravity"] =         hf_row["gravity"]
    cfg_toml["planet"]["radius"] =          COUPLER_options["radius"]

    # set composition
    cfg_toml["composition"]["p_surf"] =     hf_row["P_surf"]
    cfg_toml["composition"]["p_top"] =      COUPLER_options["P_top"]
    cfg_toml["composition"]["vmr_dict"] =   vol_dict

    chem_type = COUPLER_options["atmosphere_chemistry"]
    if chem_type > 0:
        # any chemistry
        cfg_toml["plots"]["mixing_ratios"] = True

        if chem_type == 1:
            # equilibrium
            cfg_toml["composition"]["chemistry"]   = chem_type
            cfg_toml["composition"]["include_all"] = True

        elif chem_type >= 2:
            # kinetics 
            raise Exception("Chemistry type %d unsupported by AGNI"%chem_type)
    
    # set condensation
    # condensates = []
    # if len(vol_dict) == 1:
    #     # single-gas case
    #     condensates = [list(vol_dict.keys())[0]]
    # else:
    #     # get gas with lowest mixing ratio 
    #     vmr_min = 2.0
    #     gas_min = ""
    #     for k in vol_dict.keys():
    #         if vol_dict[k] < vmr_min:
    #             vmr_min = vol_dict[k]
    #             gas_min = k
    #     # add all gases as condensates, except the least abundant gas 
    #     for k in vol_dict.keys():
    #         if k == gas_min:
    #             continue 
    #         condensates.append(k)
    condensates = ["H2O"]
    cfg_toml["composition"]["condensates"] = condensates

    if len(condensates) > 0:
        cfg_toml["plots"]["mixing_ratios"] = make_plots

    # Set files
    cfg_toml["files"]["output_dir"] =       os.path.join(dirs["output"])
    if os.path.exists(try_spfile):
        # exists => don't modify it
        cfg_toml["files"]["input_sf"] =     try_spfile
        cfg_toml["files"]["input_star"] =   ""   
    else:
        # doesn't exist => AGNI will copy it + modify as required
        cfg_toml["files"]["input_sf"] =     os.path.join(dirs["fwl"],
                                                         COUPLER_options["spectral_file"])
        cfg_toml["files"]["input_star"] =   sflux_path
    
    # Set execution
    cfg_toml["execution"]["num_levels"] =   COUPLER_options["atmosphere_nlev"]
    cfg_toml["execution"]["rayleigh"] =     bool(COUPLER_options["insert_rscatter"] == 1)
    cfg_toml["execution"]["cloud"] =        bool(COUPLER_options["water_cloud"] == 1)
    cfg_toml["execution"]["linesearch"] =   linesearch
    cfg_toml["execution"]["easy_start"] =   easy_start
    cfg_toml["execution"]["dx_max"] =       dx_max

    if COUPLER_options["atmosphere_solve_energy"] == 0:
        # The default cfg assumes solving for energy balance.
        # If we don't want to do that, set the configuration to a prescribed
        # profile of: T(p) = dry + condensing + stratosphere
        initial_arr = ["dry"]
        if COUPLER_options["tropopause"] == 1:
            initial_arr.extend(["str", "%.3e"%COUPLER_options["T_skin"]])
        cfg_toml["execution"]["initial_state"] = initial_arr
        
        # Tell AGNI not to solve for RCE
        cfg_toml["execution"]["solvers"] = []


    elif (loops_total > 1) and resume_prev:
        # If solving for RCE and are current inside the init stage, use old T(p)
        # as initial guess for solver.
        ncdfs = glob.glob(os.path.join(dirs["output"], "data","*_atm.nc"))
        ncdf_times = [float(f.split("/")[-1].split("_")[0]) for f in ncdfs]
        nc_path = ncdfs[np.argmax(ncdf_times)]

        log.debug("Initialise from last T(p)")
        cfg_toml["execution"]["initial_state"] = ["ncdf", nc_path, 
                                                  "add", "%.6f"%initial_offset]

    else:
        log.debug("Initialise isothermal")
        cfg_toml["execution"]["initial_state"] = ["iso", "%.2f"%(hf_row["T_surf"]-1.0)]
        
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
        cfg_toml["planet"]["tmp_magma"] = hf_row["T_magma"]

    # Solution type ~ surface state
    cfg_toml["execution"]["solution_type"] = surf_state

    # Tighter tolerances during first iters, to ensure consistent coupling
    if loops_total < 3:
        cfg_toml["execution"]["converge_rtol"] = 1.0e-3
        
    # Set plots 
    cfg_toml["plots"]["at_runtime"]     = agni_debug and make_plots
    cfg_toml["plots"]["temperature"]    = make_plots
    cfg_toml["plots"]["fluxes"]         = make_plots

    # AGNI log level
    cfg_toml["execution"]["verbosity"] = 1
    # if agni_debug:
    #     cfg_toml["execution"]["verbosity"] = 2

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
    
    # Copy AGNI log into PROTEUS log. There are probably better ways to do this, but it 
    #     works well enough. We don't use agni_debug much anyway
    if agni_debug:
        with open(GetCurrentLogfilePath(dirs["output"]), "a") as outfile:
            with open(os.path.join(dirs["output"], "agni.log"), "r") as infile:
                outfile.write(infile.read())
    
    return success 

def RunAGNI(loops_total:int, dirs:dict, COUPLER_options:dict, hf_row:dict):
    """Run AGNI atmosphere model.
    
    Calculates the temperature structure of the atmosphere and the fluxes, etc.
    Stores the new flux boundary condition to be provided to SPIDER. 

    Parameters
    ----------
        loops_total : int 
            Model total loops counter.
        dirs : dict
            Dictionary containing paths to directories
        COUPLER_options : dict
            Configuration options and other variables
        hf_row : dict
            Dictionary containing simulation variables for current iteration

    Returns
    ----------
        output : dict
            Output variables, as a dictionary

    """

    # Inform
    log.info("Running AGNI...")
    time_str = "%d"%hf_row["Time"]
    make_plots = (COUPLER_options["plot_iterfreq"] > 0) \
                        and (loops_total % COUPLER_options["plot_iterfreq"] == 0)

    # tracking
    agni_success = False  # success?
    attempts = 1          # number of attempts so far

    # default run parameters
    linesearch = True
    easy_start = False
    resume_prev= True
    offset = 0.0
    dx_max = 100.0

    # bootstrapping run parameters
    if loops_total < 2:
        linesearch = True
        easy_start = True
        resume_prev= False
        dx_max = 300.0

    # make attempts
    while not agni_success:
        log.info("Attempt %d" % attempts)

        # Try the module
        agni_success = _try_agni(loops_total, dirs, COUPLER_options, hf_row, make_plots, 
                                 offset, easy_start, linesearch, dx_max, resume_prev)

        if agni_success:
            # success
            log.info("Attempt %d succeeded" % attempts)
            break
        else:
            # failure
            log.warning("Attempt %d failed" % attempts)
            attempts += 1

            if attempts == 2:
                # Try offsetting the temperature profile and decreasing the step size
                linesearch = False
                offset     = 0.2
                dx_max     = 50.0
                resume_prev= True
            elif attempts == 3:
                # Try starting over
                linesearch  = True 
                dx_max      = 200.0
                resume_prev = False 
                easy_start  = True 
            else:
                log.error("Maximum attempts when executing AGNI")
                break
   
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
    files_remove = ["plot_ptprofile.png", "fl.csv", "ptz_ini.csv", "ptz.csv", "agni.toml", 
                    "solver.png", "jacobian.png"] 
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
    SW_flux_up =    np.array(ds.variables["fl_U_SW"][:])
    T_surf =        float(ds.variables["tmp_surf"][:])
    ds.close()

    # New flux from SOCRATES
    if (COUPLER_options["F_atm_bc"] == 0):
        F_atm_new = net_flux[0] 
    else:
        F_atm_new = net_flux[-1]  

    # Require that the net flux must be upward (positive)
    if (COUPLER_options["prevent_warming"] == 1):
        F_atm_new = max( 1e-8 , F_atm_new )
        
    log.info("SOCRATES fluxes (net@BOA, net@TOA, OLR): %.3f, %.3f, %.3f W/m^2" % 
                                        (net_flux[-1], net_flux[0] ,LW_flux_up[0]))

    output = {}
    output["F_atm"]  = F_atm_new
    output["F_olr"]  = LW_flux_up[0]
    output["F_sct"]  = SW_flux_up[0]
    output["T_surf"] = T_surf
    
    return output

