# Functions used to handle atmosphere thermodynamics (running JANUS, etc.)

from utils.modules_ext import *
from utils.helper import *

log = logging.getLogger("PROTEUS")


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
        log.warning("Limit max dT:", Ts_curr, "->", Ts_last-dT_sgn*dT_max)
        Ts_curr = Ts_last-dT_sgn*dT_max
    if abs(Ts_last-Ts_curr) > 0.05*Ts_last:
        dT_sgn  = np.sign(Ts_last-Ts_curr)
        log.warning("Limit max dT:", Ts_curr, "->", Ts_last-dT_sgn*0.01*Ts_last)
        Ts_curr = Ts_last-dT_sgn*0.05*Ts_last

    log.info("t_last:", t_last/yr, "Ts_last:", Ts_last)
    log.info("t_curr:", t_curr/yr, "Ts_curr:", Ts_curr)

    return Ts_curr


# Prepare surface BC for new atmosphere calculation
def PrepAtm( loop_counter, runtime_helpfile, COUPLER_options ):

    # In the beginning: standard surface temperature from last entry
    if loop_counter["total"] < loop_counter["init_loops"]:
        COUPLER_options["T_surf"] = runtime_helpfile.iloc[-1]["T_surf"]
    
    # Check for flux_convergence scheme criteria
    else:
        if (COUPLER_options["flux_convergence"] == 2) \
            or ( (COUPLER_options["flux_convergence"] == 1) and \
                 (runtime_helpfile.iloc[-1]["RF_depth"] < COUPLER_options["RF_crit"]) #and \
                 #(COUPLER_options["F_net"] > COUPLER_options["F_diff"]*COUPLER_options["F_int"]) \
               ):

            PrintHalfSeparator()
            log.info(">>>>>>>>>> Flux convergence scheme <<<<<<<<<<<")

            COUPLER_options["flux_convergence"] = 2

            # Last T_surf and time from atmosphere, K
            t_curr          = runtime_helpfile.iloc[-1]["Time"]
            run_atm         = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere']
            run_atm_prev    = run_atm.loc[run_atm['Time'] != t_curr]
            run_atm_curr    = run_atm.loc[run_atm['Time'] == t_curr]
            t_previous_atm  = run_atm_prev.iloc[-1]["Time"]
            Ts_previous_atm = run_atm_prev.iloc[-1]["T_surf"]
            Ts_last_atm     = run_atm.iloc[-1]["T_surf"]

            log.info("F_net", str(COUPLER_options["F_net"]), "Ts_previous_atm:", Ts_previous_atm, "Ts_last_atm", Ts_last_atm, "dTs_atm", str(COUPLER_options["dTs_atm"]), "t_curr", t_curr, "t_previous_atm", t_previous_atm)

            # Apply flux convergence via shallow layer function
            COUPLER_options["T_surf"] = shallow_mixed_ocean_layer(COUPLER_options["F_net"], Ts_previous_atm, COUPLER_options["dTs_atm"], t_curr, t_previous_atm)

            # Prevent atmospheric oscillations
            if len(run_atm_curr) > 2 and (np.sign(run_atm_curr["F_net"].iloc[-1]) != np.sign(run_atm_curr["F_net"].iloc[-2])) and (np.sign(run_atm_curr["F_net"].iloc[-2]) != np.sign(run_atm_curr["F_net"].iloc[-3])):
                COUPLER_options["T_surf"] = np.mean([run_atm.iloc[-1]["T_surf"], run_atm.iloc[-2]["T_surf"]])
                log.warning("Prevent oscillations, new T_surf =", COUPLER_options["T_surf"])

            log.info("dTs_atm (K):", COUPLER_options["dTs_atm"], "t_previous_atm:", t_previous_atm, "Ts_previous_atm:", Ts_previous_atm, "Ts_last_atm:", Ts_last_atm, "t_curr:", t_curr, "Ts_curr:", COUPLER_options["T_surf"])

            PrintHalfSeparator()

        # Standard surface temperature from last entry
        else:
            COUPLER_options["T_surf"] = runtime_helpfile.iloc[-1]["T_surf"]

    return COUPLER_options


# Generate atmosphere from input files
def StructAtm( dirs, runtime_helpfile, COUPLER_options ):

    from JANUS.utils.atmosphere_column import atmos
    from JANUS.utils.ReadSpectralFile import ReadBandEdges

    # Create atmosphere object and set parameters
    pl_radius = COUPLER_options["radius"]
    pl_mass = COUPLER_options["mass"]
    
    vol_list = {}
    for vol in volatile_species:
        vol_list[vol] = runtime_helpfile.iloc[-1][vol+"_mr"]

    match COUPLER_options["tropopause"]:
        case 0:
            trppT = 0.0  # none
        case 1:
            trppT = COUPLER_options["T_skin"]  # skin temperature (grey stratosphere)
        case 2:
            trppT = COUPLER_options["min_temperature"]  # dynamically, based on heating rate
        case _:
            UpdateStatusfile(dirs, 20)
            raise Exception("Invalid tropopause option '%d'" % COUPLER_options["tropopause"])

    # Number of levels   
    nlev = int(COUPLER_options["atmosphere_nlev"])
            
    # Spectral bands
    band_edges = ReadBandEdges(dirs["output"]+"star.sf")

    # Cloud properties 
    re   = 1.0e-5 # Effective radius of the droplets [m] (drizzle forms above 20 microns)
    lwm  = 0.8    # Liquid water mass fraction [kg/kg] - how much liquid vs. gas is there upon cloud formation? 0 : saturated water vapor does not turn liquid ; 1 : the entire mass of the cell contributes to the cloud
    clfr = 0.8    # Water cloud fraction - how much of the current cell turns into cloud? 0 : clear sky cell ; 1 : the cloud takes over the entire area of the cell (just leave at 1 for 1D runs)
    do_cloud = bool(COUPLER_options["water_cloud"] == 1)
    alpha_cloud = float(COUPLER_options["alpha_cloud"])

    # Make object 
    atm = atmos(COUPLER_options["T_surf"], runtime_helpfile.iloc[-1]["P_surf"]*1e5, 
                COUPLER_options["P_top"]*1e5, pl_radius, pl_mass,
                band_edges,
                vol_mixing=vol_list, 
                minT = COUPLER_options["min_temperature"],
                maxT = COUPLER_options["max_temperature"],
                trppT=trppT,
                water_lookup=False,
                req_levels=nlev, alpha_cloud=alpha_cloud,
                re=re, lwm=lwm, clfr=clfr, do_cloud=do_cloud
                )

    atm.zenith_angle    = COUPLER_options["zenith_angle"]
    atm.albedo_pl       = COUPLER_options["albedo_pl"]
    atm.inst_sf         = COUPLER_options["asf_scalefactor"]
    atm.albedo_s        = COUPLER_options["albedo_s"]
    atm.instellation    = COUPLER_options["F_ins"]
    atm.skin_d          = COUPLER_options["skin_d"]
    atm.skin_k          = COUPLER_options["skin_k"]

    run_atm = runtime_helpfile.loc[runtime_helpfile['Input']=='Interior'].drop_duplicates(subset=['Time'], keep='last')
    atm.tmp_magma = run_atm.iloc[-1]["T_surf"]

    return atm

def RunJANUS( atm, time_dict, dirs, COUPLER_options, runtime_helpfile, write_in_tmp_dir=True, search_method=0, rtol=1.0e-4):
    """Run JANUS.
    
    Calculates the temperature structure of the atmosphere and the fluxes, etc.
    Stores the new flux boundary condition to be provided to SPIDER. Limits flux
    change if required.

    Parameters
    ----------
        atm : atmos
            Atmosphere object
        time_dict : dict
            Dictionary containing simulation time variables
        dirs : dict
            Dictionary containing paths to directories
        COUPLER_options : dict
            Configuration options and other variables
        runtime_helpfile : pd.DataFrame
            Dataframe containing simulation variables (now and historic)

        write_in_tmp_dir : bool
            Write temporary files in a local folder within /tmp, rather than in the output folder
        search_method : int
            Root finding method used by JANUS
        rtol : float
            Relative tolerance on solution for root finding method
    Returns
    ----------
        atm : atmos
            Updated atmos object
        COUPLER_options : dict
            Updated configuration options and other variables

    """

    # Runtime info
    PrintHalfSeparator()
    log.info("Running JANUS...")

    # Update stdout
    old_stdout , old_stderr = sys.stdout , sys.stderr
    sys.stdout = StreamToLogger(log, logging.INFO)
    sys.stderr = StreamToLogger(log, logging.ERROR)

    # Change dir
    cwd = os.getcwd()
    tmp_dir = dirs["output"]
    if write_in_tmp_dir:
        tmp_dir = "/tmp/socrates_%d/" % np.random.randint(int(100),int(1e13))
        os.makedirs(tmp_dir)
    os.chdir(tmp_dir)

    # Prepare to calculate temperature structure w/ General Adiabat 
    trppD = bool(COUPLER_options["tropopause"] == 2 )
    rscatter = bool(COUPLER_options["insert_rscatter"] == 1)

    # Run JANUS
    if COUPLER_options["atmosphere_solve_energy"] == 0:

        if COUPLER_options["atmosphere_surf_state"] == 1:  # fixed T_Surf
            from JANUS.modules.solve_pt import MCPA
            atm = MCPA(dirs, atm, False, trppD, rscatter)

        elif COUPLER_options["atmosphere_surf_state"] == 2: # conductive lid
            from JANUS.modules.solve_pt import MCPA_CBL

            T_surf_max = -1
            T_surf_old = -1 
            atol       = 1.0e-5

            # Done with initial loops
            if (time_dict["planet"] > 0):

                # Get previous temperature as initial guess
                run_atm = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere'].drop_duplicates(subset=['Time'], keep='last')
                T_surf_old = run_atm.iloc[-1]["T_surf"]

                # Prevent heating of the interior
                if (COUPLER_options["prevent_warming"] == 1):
                    run_atm = runtime_helpfile.loc[runtime_helpfile['Input']=='Interior'].drop_duplicates(subset=['Time'], keep='last')
                    T_surf_max = run_atm.iloc[-1]["T_surf"]

                # calculate tolerance
                tol = rtol * abs(run_atm.iloc[-1]["F_atm"]) + atol
            else:
                tol = 0.1

            # run JANUS
            atm = MCPA_CBL(dirs, atm, trppD, rscatter, method=search_method, atol=tol,
                          atm_bc=int(COUPLER_options["F_atm_bc"]), T_surf_guess=float(T_surf_old)-0.5, T_surf_max=float(T_surf_max))
            
            COUPLER_options["T_surf"] = atm.ts

        else:
            UpdateStatusfile(dirs, 20)
            raise Exception("Invalid surface state chosen for JANUS")
    else:
        UpdateStatusfile(dirs, 20)
        raise Exception("Cannot solve for RCE with JANUS")
    
    # Clean up run directory
    for file in glob.glob(tmp_dir+"/current??.????"):
        os.remove(file)
    for file in glob.glob(tmp_dir+"/profile.*"):
        os.remove(file)
    os.chdir(cwd)
    if write_in_tmp_dir:
        shutil.rmtree(tmp_dir,ignore_errors=True)

    any_cloud = np.any(np.array(atm.clfr) > 1.0e-20)
    log.info("Water clouds have formed = %s"%(str(any_cloud)))
    log.info("SOCRATES fluxes (net@surf, net@TOA, OLR): %.5e, %.5e, %.5e W m-2" % (atm.net_flux[-1], atm.net_flux[0] , atm.LW_flux_up[0]))

    # Save atm data to disk
    nc_fpath = dirs["output"]+"/data/"+str(int(time_dict["planet"]))+"_atm.nc"
    atm.write_ncdf(nc_fpath)

    # Check for NaNs
    if not np.isfinite(atm.net_flux).all():
        UpdateStatusfile(dirs, 22)
        raise Exception("JANUS output array contains NaN or Inf values")

    # Store new flux
    if (COUPLER_options["F_atm_bc"] == 0):
        F_atm_new = atm.net_flux[0]  
    else:
        F_atm_new = atm.net_flux[-1]  

    # Require that the net flux must be upward
    F_atm_lim = F_atm_new
    if (COUPLER_options["prevent_warming"] == 1):
        F_atm_lim = max( 1.0e-8 , F_atm_new )

    # Print if a limit was applied
    if not np.isclose(F_atm_lim , F_atm_new ):
        log.warning("Change in F_atm [W m-2] limited in this step!")
        log.warning("    %g  ->  %g" % (F_atm_new , F_atm_lim))

    # Restore stdout
    sys.stdout , sys.stderr = old_stdout , old_stderr
            
    COUPLER_options["F_atm"] = F_atm_lim         # Net flux at TOA
    COUPLER_options["F_olr"] = atm.LW_flux_up[0] # OLR
    COUPLER_options["F_sct"] = atm.SW_flux_up[0] # Scattered SW flux

    return COUPLER_options

