#Generic atmosphere wrapper
from proteus.utils.modules_ext import *
from proteus.utils.helper import *

#We should make the import dependent of the chosen atmospheric submodule
from proteus.atmos_clim.janus import RunJANUS, StructAtm
from janus.utils.StellarSpectrum import PrepareStellarSpectrum,InsertStellarSpectrum
from proteus.atmos_clim.agni import RunAGNI, InitAtmos, UpdateProfile, ActivateEnv, DeallocAtmos
atm = None
from proteus.atmos_clim.dummy_atmosphere import RunDummyAtm

log = logging.getLogger("PROTEUS")

def RunAtmosphere(OPTIONS:dict, dirs:dict, loop_counter:dict,
                  spfile_path:str, hf_all:pd.DataFrame, hf_row:dict):
    """Run Atmosphere submodule.

    Generic function to run an atmospheric simulation with either JANUS, AGNI or dummy.
    Writes into the hf_row generic variable passed as an arguement.

    Parameters
    ----------
        OPTIONS : dict
            Configuration options and other variables
        dirs : dict
            Dictionary containing paths to directories
        loop_counter : dict
            Dictionary containing iteration information
        spfile_path : str
            Spectral file path
        hf_all : pd.DataFrame
            Dataframe containing simulation variables (now and historic)
        hf_row : dict
            Dictionary containing simulation variables for current iteration

    """

    #Warning! Find a way to store atm object for AGNI
    global atm

    PrintHalfSeparator()
    if OPTIONS["shallow_ocean_layer"] == 1:
        hf_row["T_surf"] = ShallowMixedOceanLayer(hf_all.iloc[-1].to_dict(), hf_row)

    if OPTIONS["atmosphere_model"] == 0:
        # Run JANUS: 
        hf_row["T_surf"] = hf_row["T_magma"]
        atm = StructAtm( dirs, hf_row, OPTIONS )
        atm_output = RunJANUS( atm, hf_row["Time"], dirs, OPTIONS, hf_all)

    elif OPTIONS["atmosphere_model"] == 1:
        # Run AGNI 

        # Initialise atmosphere struct
        no_spfile = not os.path.exists(spfile_path)
        no_atm    = bool(atm == None)
        if no_atm or no_spfile:
            log.debug("Initialise new atmosphere struct")

            # first run?
            if no_atm:
                ActivateEnv(dirs)
                # surface temperature guess
                hf_row["T_surf"] = hf_row["T_magma"]
            else:
                # deallocate old atmosphere 
                DeallocAtmos(atm)

            # allocate new 
            atm = InitAtmos(dirs, OPTIONS, hf_row)

        # Update profile 
        atm = UpdateProfile(atm, hf_row, OPTIONS)

        # Run solver
        atm, atm_output = RunAGNI(atm, loop_counter["total"], dirs, OPTIONS, hf_row)

    elif OPTIONS["atmosphere_model"] == 2:
        # Run dummy atmosphere model 
        atm_output = RunDummyAtm(dirs, OPTIONS, 
                                 hf_row["T_magma"], hf_row["F_ins"], hf_row["R_planet"])

    # Store atmosphere module output variables
    hf_row["z_obs"]  = atm_output["z_obs"] 
    hf_row["F_atm"]  = atm_output["F_atm"] 
    hf_row["F_olr"]  = atm_output["F_olr"] 
    hf_row["F_sct"]  = atm_output["F_sct"] 
    hf_row["T_surf"] = atm_output["T_surf"]
    hf_row["F_net"]  = hf_row["F_int"] - hf_row["F_atm"]

    # Calculate observables (measured at infinite distance)
    hf_row["transit_depth"] =  (hf_row["z_obs"] / hf_row["R_star"])**2.0
    hf_row["contrast_ratio"] = ((hf_row["F_olr"]+hf_row["F_sct"])/hf_row["F_ins"]) * \
                                 (hf_row["z_obs"] / (OPTIONS["mean_distance"]*AU))**2.0

    return

def ShallowMixedOceanLayer(hf_cur:dict, hf_pre:dict):

    # This scheme is not typically used, but it maintained here from legacy code
    # We could consider removing it in the future.

    log.info(">>>>>>>>>> Flux convergence scheme <<<<<<<<<<<")

    # For SI conversion
    yr          = 3.154e+7      # s

    # Last T_surf and time from atmosphere, K
    t_cur  = hf_cur["Time"]*yr
    t_pre  = hf_pre["Time"]*yr
    Ts_pre = hf_pre["T_surf"]

    # Properties of the shallow mixed ocean layer
    c_p_layer   = 1000          # J kg-1 K-1
    rho_layer   = 3000          # kg m-3
    depth_layer = 1000          # m

    def ocean_evolution(t, y): 
        # Specific heat of mixed ocean layer
        mu      = c_p_layer * rho_layer * depth_layer # J K-1 m-2
        # RHS of ODE
        RHS     = - hf_cur["F_net"] / mu
        return RHS

    # Solve ODE
    sol_curr  = solve_ivp(ocean_evolution, [t_pre, t_cur], [Ts_pre])

    # New current surface temperature from shallow mixed layer
    Ts_cur = sol_curr.y[0][-1] # K

    return Ts_cur
