# Functions used to handle atmosphere temperature structure (running AGNI, etc.)

from utils.modules_ext import *
from utils.helper import *
from utils.constants import *
from utils.logs import GetLogfilePath, GetCurrentLogfileIndex

from juliacall import Main as jl

log = logging.getLogger("PROTEUS")

# Constant
AGNI_LOGFILE_NAME="agni_recent.log"

def SyncLogfiles(outdir:str):
    # Logfile paths 
    agni_logpath = os.path.join(outdir, AGNI_LOGFILE_NAME)
    logpath = GetLogfilePath(outdir, GetCurrentLogfileIndex(outdir))

    # Copy logfile content
    with open(agni_logpath, "r") as infile:
        inlines = infile.readlines()

        with open(logpath, "a") as outfile:
            for i,line in enumerate(inlines):
                # First line of agni logfile has NULL chars at the start, for some reason
                if i == 0:
                    line = "[" + line.split("[", 1)[1]
                # copy the line
                outfile.write(line)
            
    # Remove logfile content
    with open(agni_logpath, "w") as hdl:
        hdl.write("")

def ActivateEnv(dirs:dict):

    log.debug("Activating Julia environment")
    jl.seval("using Pkg")
    jl.Pkg.activate(dirs["agni"])

    # Plotting configuration
    jl.seval('ENV["GKSwstype"] = "100"')
    jl.seval("using Plots")
    jl.seval('default(label=nothing)')

    # Import AGNI
    jl.seval("using AGNI")

    # Setup logging from AGNI
    #    This handle will be kept open throughout the PROTEUS simulation, so the file 
    #    should not be deleted at runtime. However, it will be emptied when appropriate.
    verbosity = 1
    logpath = os.path.join(dirs["output"], AGNI_LOGFILE_NAME)
    jl.AGNI.setup_logging(logpath, verbosity)

    log.debug("AGNI will log to '%s'"%logpath)


def ConstructVolDict(hf_row:dict, OPTIONS:dict):
    vol_dict = {}
    for vol in volatile_species:
        if OPTIONS[vol+"_included"]:
            vmr = hf_row[vol+"_vmr"]
            if vmr > 1e-40:
                vol_dict[vol] = vmr 
    
    if len(vol_dict) == 0:
        UpdateStatusfile(dirs, 20)
        raise Exception("All volatiles have a volume mixing ratio of zero")
    
    return vol_dict


def InitAtmos(dirs:dict, OPTIONS:dict, hf_row:dict):
    """Initialise atmosphere struct for use by AGNI.
    
    Does not set the temperature profile.

    Parameters
    ----------
        dirs : dict
            Dictionary containing paths to directories
        OPTIONS : dict
            Configuration options and other variables
        hf_row : dict
            Dictionary containing simulation variables for current iteration

    Returns
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct 

    """

    log.debug("New AGNI atmosphere")

    atmos = jl.AGNI.atmosphere.Atmos_t()

    # Stellar spectrum path
    sflux_files = glob.glob(os.path.join(dirs["output"], "data", "*.sflux"))
    sflux_times = [ int(s.split("/")[-1].split(".")[0]) for s in sflux_files]
    sflux_path  = os.path.join(dirs["output"], "data", "%d.sflux"%int(sorted(sflux_times)[-1]))

    # Spectral file path
    try_spfile = os.path.join(dirs["output"] , "runtime.sf")
    if os.path.exists(try_spfile):
        # exists => don't modify it
        input_sf =      try_spfile
        input_star =    ""   
    else:
        # doesn't exist => AGNI will copy it + modify as required
        input_sf =      os.path.join(dirs["fwl"], OPTIONS["spectral_file"])
        input_star =    sflux_path

    # composition
    vol_dict = ConstructVolDict(hf_row, OPTIONS)
    
    # set condensation
    condensates = []
    if len(vol_dict) == 1:
        # single-gas case
        condensates = list(vol_dict.keys())
    else:
        # get gas with smallest volume mixing ratio 
        vmr_min = 2.0
        gas_min = ""
        for k in vol_dict.keys():
            if vol_dict[k] < vmr_min:
                vmr_min = vol_dict[k]
                gas_min = k
        # set all gases as condensates, except the least abundant gas 
        for k in vol_dict.keys():
            if k == gas_min:
                continue 
            condensates.append(k)

    # Chemistry 
    chem_type = OPTIONS["atmosphere_chemistry"]
    include_all = False
    fc_dir = "_unset"
    if chem_type == 1:
        # equilibrium
        include_all = True
        condensates = []

        # working folder for fastchem coupling
        fc_dir = create_tmp_folder()
        log.debug("Fastchem work folder: '%s'"%fc_dir)

    elif chem_type >= 2:
        # kinetics 
        raise Exception("Chemistry type %d unsupported by AGNI"%chem_type)
    
    # Setup struct 
    jl.AGNI.atmosphere.setup_b(atmos, 
                        dirs["agni"], dirs["output"], input_sf,

                        hf_row["F_ins"], 
                        OPTIONS["asf_scalefactor"], 
                        OPTIONS["albedo_pl"], 
                        OPTIONS["zenith_angle"],

                        hf_row["T_surf"], 
                        hf_row["gravity"], hf_row["R_planet"],
                        
                        int(OPTIONS["atmosphere_nlev"]), 
                        hf_row["P_surf"], 
                        OPTIONS["P_top"],

                        vol_dict, "",

                        flag_rayleigh=bool(OPTIONS["rayleigh"] == 1),
                        flag_cloud=bool(OPTIONS["water_cloud"] == 1),
                        
                        albedo_s=OPTIONS["albedo_s"],
                        condensates=condensates,
                        use_all_gases=include_all,
                        fastchem_work = fc_dir,

                        skin_d=OPTIONS["skin_d"], skin_k=OPTIONS["skin_k"],
                        tmp_magma=hf_row["T_surf"]
                        )

    # Allocate arrays 
    jl.AGNI.atmosphere.allocate_b(atmos,input_star)

    # Set temperature profile from old NetCDF if it exists
    nc_files = glob.glob(os.path.join(dirs["output"],"data","*.nc"))
    if len(nc_files) > 0:
        log.debug("Load NetCDF profile")

        nc_times = [ int(s.split("/")[-1].split("_")[0]) for s in nc_files]
        nc_path  = os.path.join(dirs["output"], "data", "%d_atm.nc"%int(sorted(nc_times)[-1]))
        jl.AGNI.setpt.fromncdf_b(atmos, nc_path)    

    # Otherwise, set isothermal
    else:
        jl.AGNI.setpt.isothermal_b(atmos, hf_row["T_surf"])
        
    # Logging
    SyncLogfiles(dirs["output"])

    return atmos


def DeallocAtmos(atmos):
    """
    Deallocate atmosphere struct
    """                  
    jl.AGNI.atmosphere.deallocate_b(atmos)
    safe_rm(str(atmos.fastchem_work))


def UpdateProfile(atmos, hf_row:dict, OPTIONS:dict):
    """Update atmosphere struct.
    
    Sets the new surface boundary conditions and composition.

    Parameters
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct 
        hf_row : dict
            Dictionary containing simulation variables for current iteration
        OPTIONS : dict
            Configuration options and other variables

    Returns
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct 

    """

    # Update compositions
    vol_dict = ConstructVolDict(hf_row, OPTIONS)
    for g in vol_dict.keys():
        atmos.gas_vmr[g][:] = vol_dict[g]
        atmos.gas_ovmr[g][:] = vol_dict[g]

    # Update pressure grid 
    atmos.p_boa = 1.0e5 * hf_row["P_surf"]
    jl.AGNI.atmosphere.generate_pgrid_b(atmos)

    # Update surface temperature(s)
    atmos.tmp_surf  = hf_row["T_surf"]
    atmos.tmp_magma = hf_row["T_magma"]

    return atmos



def RunAGNI(atmos, loops_total:int, dirs:dict, OPTIONS:dict, hf_row:dict):
    """Run AGNI atmosphere model.
    
    Calculates the temperature structure of the atmosphere and the fluxes, etc.
    Stores the new flux boundary condition to be provided to SPIDER. 

    Parameters
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct
        loops_total : int 
            Model total loops counter.
        dirs : dict
            Dictionary containing paths to directories
        OPTIONS : dict
            Configuration options and other variables
        hf_row : dict
            Dictionary containing simulation variables for current iteration

    Returns
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct
        output : dict
            Output variables, as a dictionary

    """

    # Inform
    log.info("Running AGNI...")
    time_str = "%d"%hf_row["Time"]

    # tracking
    agni_success = False  # success?
    attempts = 0          # number of attempts so far

    # make attempts
    while not agni_success:
        attempts += 1
        log.info("Attempt %d" % attempts)

        # default parameters
        linesearch = 1
        easy_start = False
        dx_max = OPTIONS["tsurf_poststep_change"]+1.0
        ls_increase = 0.1

        # try different solver parameters if struggling
        if attempts == 2:
            linesearch  = 2
            dx_max     *= 2.0
            ls_increase = 1.1

        # first iteration parameters
        if loops_total == 0:
            linesearch  = 2
            easy_start  = True
            dx_max      = 200.0

        log.debug("Solver parameters:")
        log.debug("    ls_method=%d, easy_start=%s, dx_max=%.1f, ls_increase=%.2f"%(
            linesearch, str(easy_start), dx_max, ls_increase
        ))

        # Try solving temperature profile
        agni_success = jl.AGNI.solver.solve_energy_b(atmos, 
                            sol_type=OPTIONS["atmosphere_surf_state"],
                            chem_type=OPTIONS["atmosphere_chemistry"], 

                            conduct=False, convect=True, latent=True, sens_heat=True, 

                            max_steps=130, max_runtime=900.0, 
                            conv_atol=1e-3, conv_rtol=1e-2, 

                            method=1, ls_increase=ls_increase,
                            dx_max=dx_max, ls_method=linesearch, easy_start=easy_start,
                            
                            save_frames=False, modplot=0
                            )

        # Move AGNI logfile content into PROTEUS logfile
        SyncLogfiles(dirs["output"])

        # Model status check
        if agni_success:
            # success
            log.info("Attempt %d succeeded" % attempts)
            break
        else:
            # failure
            log.warning("Attempt %d failed" % attempts)

            # Max attempts
            if attempts >= 2:
                log.error("Maximum attempts when executing AGNI")
                break
        
    # Write output data 
    ncdf_path = os.path.join(dirs["output"],"data",time_str+"_atm.nc")
    jl.AGNI.dump.write_ncdf(atmos, ncdf_path)

    # Make plots 
    if (OPTIONS["plot_iterfreq"] > 0) \
            and (loops_total % OPTIONS["plot_iterfreq"] == 0):

        fmt = OPTIONS["plot_format"]
        jl.AGNI.plotting.plot_fluxes(atmos, os.path.join(dirs["output"],
                                                  "plot_fluxes_atmosphere.%s"%fmt))
        jl.AGNI.plotting.plot_vmr(atmos, os.path.join(dirs["output"], "plot_vmr.%s"%fmt))

    # ---------------------------
    # Parse results
    # ---------------------------
    
    log.debug("Parse results")
    net_flux =      np.array(atmos.flux_n)
    LW_flux_up =    np.array(atmos.flux_u_lw)
    SW_flux_up =    np.array(atmos.flux_u_sw)
    arr_p =         np.array(atmos.p)
    arr_z =         np.array(atmos.z)
    radius =        float(atmos.rp)
    T_surf =        float(atmos.tmp_surf)

    # New flux from SOCRATES
    if (OPTIONS["F_atm_bc"] == 0):
        F_atm_new = net_flux[0] 
    else:
        F_atm_new = net_flux[-1]  

    # Require that the net flux must be upward (positive)
    if (OPTIONS["prevent_warming"] == 1):
        F_atm_new = max( 1e-8 , F_atm_new )
        
    log.info("SOCRATES fluxes (net@BOA, net@TOA, OLR): %.2e, %.2e, %.2e  W m-2" % 
                                        (net_flux[-1], net_flux[0] ,LW_flux_up[0]))

    # find 1 mbar (=100 Pa) level 
    idx = find_nearest(arr_p, 1e2)[1]
    z_obs = arr_z[idx]

    output = {}
    output["F_atm"]  = F_atm_new
    output["F_olr"]  = LW_flux_up[0]
    output["F_sct"]  = SW_flux_up[0]
    output["T_surf"] = T_surf
    output["z_obs"]  = z_obs + radius
    
    return atmos, output

