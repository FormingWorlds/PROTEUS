# Functions used to handle atmosphere thermodynamics (running AEOLUS, etc.)

from utils.modules_ext import *
from utils.helper import *

# Debugging
# from AEOLUS.modules.plot_flux_balance import plot_fluxes

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
        print("Limit max dT:", Ts_curr, "->", Ts_last-dT_sgn*dT_max)
        Ts_curr = Ts_last-dT_sgn*dT_max
    if abs(Ts_last-Ts_curr) > 0.05*Ts_last:
        dT_sgn  = np.sign(Ts_last-Ts_curr)
        print("Limit max dT:", Ts_curr, "->", Ts_last-dT_sgn*0.01*Ts_last)
        Ts_curr = Ts_last-dT_sgn*0.05*Ts_last

    print("t_last:", t_last/yr, "Ts_last:", Ts_last)
    print("t_curr:", t_curr/yr, "Ts_curr:", Ts_curr)

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
            print(">>>>>>>>>> Flux convergence scheme <<<<<<<<<<<")

            COUPLER_options["flux_convergence"] = 2

            # In case last atm T_surf from flux convergence scheme was smaller(!) than threshold 
            if abs(COUPLER_options["F_net"]) < COUPLER_options["F_eps"]:
                
                COUPLER_options["T_surf"] = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere'].iloc[-1]["T_surf"]
                print("Use previous T_surf =", COUPLER_options["T_surf"])

            else:

                # Last T_surf and time from atmosphere, K
                t_curr          = runtime_helpfile.iloc[-1]["Time"]
                run_atm         = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere']
                run_atm_prev    = run_atm.loc[run_atm['Time'] != t_curr]
                run_atm_curr    = run_atm.loc[run_atm['Time'] == t_curr]
                t_previous_atm  = run_atm_prev.iloc[-1]["Time"]
                Ts_previous_atm = run_atm_prev.iloc[-1]["T_surf"]
                Ts_last_atm     = run_atm.iloc[-1]["T_surf"]

                print("F_net", str(COUPLER_options["F_net"]), "Ts_previous_atm:", Ts_previous_atm, "Ts_last_atm", Ts_last_atm, "dTs_atm", str(COUPLER_options["dTs_atm"]), "t_curr", t_curr, "t_previous_atm", t_previous_atm)

                # Apply flux convergence via shallow layer function
                COUPLER_options["T_surf"] = shallow_mixed_ocean_layer(COUPLER_options["F_net"], Ts_previous_atm, COUPLER_options["dTs_atm"], t_curr, t_previous_atm)

                # Prevent atmospheric oscillations
                if len(run_atm_curr) > 2 and (np.sign(run_atm_curr["F_net"].iloc[-1]) != np.sign(run_atm_curr["F_net"].iloc[-2])) and (np.sign(run_atm_curr["F_net"].iloc[-2]) != np.sign(run_atm_curr["F_net"].iloc[-3])):
                    COUPLER_options["T_surf"] = np.mean([run_atm.iloc[-1]["T_surf"], run_atm.iloc[-2]["T_surf"]])
                    print("Prevent oscillations, new T_surf =", COUPLER_options["T_surf"])

                print("dTs_atm (K):", COUPLER_options["dTs_atm"], "t_previous_atm:", t_previous_atm, "Ts_previous_atm:", Ts_previous_atm, "Ts_last_atm:", Ts_last_atm, "t_curr:", t_curr, "Ts_curr:", COUPLER_options["T_surf"])

            PrintHalfSeparator()

        # Standard surface temperature from last entry
        else:
            COUPLER_options["T_surf"] = runtime_helpfile.iloc[-1]["T_surf"]

    return COUPLER_options


# Generate atmosphere from input files
def StructAtm( runtime_helpfile, COUPLER_options ):

    from AEOLUS.utils.atmosphere_column import atmos

    # Create atmosphere object and set parameters
    pl_radius = COUPLER_options["radius"]
    pl_mass = COUPLER_options["mass"]
    
    vol_list = { 
                  "H2O" : runtime_helpfile.iloc[-1]["H2O_mr"], 
                  "CO2" : runtime_helpfile.iloc[-1]["CO2_mr"],
                  "H2"  : runtime_helpfile.iloc[-1]["H2_mr"], 
                  "N2"  : runtime_helpfile.iloc[-1]["N2_mr"],  
                  "CH4" : runtime_helpfile.iloc[-1]["CH4_mr"], 
                  "O2"  : runtime_helpfile.iloc[-1]["O2_mr"], 
                  "CO"  : runtime_helpfile.iloc[-1]["CO_mr"], 
                  "He"  : 0.0,  # broken
                  "NH3" : 0.0,  # broken
                }

    match COUPLER_options["tropopause"]:
        case 0:
            trppT = 0
        case 1:
            trppT = COUPLER_options["T_skin"]
        case 2:
            trppT = None
        case _:
            raise ValueError("Invalid tropopause option '%d'" % COUPLER_options["tropopause"])
            
    atm = atmos(COUPLER_options["T_surf"], runtime_helpfile.iloc[-1]["P_surf"]*1e5, 
                COUPLER_options["P_top"]*1e5, pl_radius, pl_mass,
                vol_mixing=vol_list, 
                minT = COUPLER_options["min_temperature"],
                trppT=trppT,
                water_lookup=False
                )

    atm.zenith_angle    = COUPLER_options["zenith_angle"]
    atm.albedo_pl       = COUPLER_options["albedo_pl"]
    atm.albedo_s        = COUPLER_options["albedo_s"]
    atm.toa_heating     = COUPLER_options["TOA_heating"]
    atm.tmp_magma       = COUPLER_options["T_surf"]
    atm.skin_d          = COUPLER_options["skin_d"]
    atm.skin_k          = COUPLER_options["skin_k"]

    return atm

def RunAEOLUS( atm, time_dict, dirs, COUPLER_options, runtime_helpfile):
    """Run AEOLUS.
    
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
    Returns
    ----------
        atm : atmos
            Updated atmos object
        COUPLER_options : dict
            Updated configuration options and other variables

    """

    # Runtime info
    PrintHalfSeparator()
    print("Running AEOLUS...")

    # Change dir
    cwd = os.getcwd()
    os.chdir(dirs["output"])

    # Prepare to calculate temperature structure w/ General Adiabat 
    trppD = bool(COUPLER_options["tropopause"] == 2 )
    rscatter = bool(COUPLER_options["insert_rscatter"] == 1)

    # Run AEOLUS
    if COUPLER_options["atmosphere_solve_energy"] == 0:

        if COUPLER_options["atmosphere_surf_state"] == 1:  # fixed T_Surf
            from AEOLUS.modules.solve_pt import MCPA
            atm = MCPA(dirs, atm, False, trppD, rscatter)

        elif COUPLER_options["atmosphere_surf_state"] == 2: # conductive lid
            from AEOLUS.modules.solve_pt import MCPA_CL

            atm = MCPA_CL(dirs, atm, trppD, rscatter)
            COUPLER_options["T_surf"] = atm.ts
            print(atm.net_flux)

        else:
            raise Exception("Free surface state is not a valid option for AEOLUS")
    else:
        raise Exception("Cannot solve for RCE with AEOLUS")
    
    # Clean up run directory
    os.chdir(cwd)
    for file in glob.glob(dirs["output"]+"/current??.????"):
        os.remove(file)
    for file in glob.glob(dirs["output"]+"/profile.*"):
        os.remove(file)

    print("SOCRATES fluxes (net@surf, net@TOA, OLR): %.5e, %.5e, %.5e W m-2" % (atm.net_flux[-1], atm.net_flux[0] , atm.LW_flux_up[0]))

    # plot_fluxes(atm, dirs["output"]+"/fluxes.pdf")

    # Save atm data to disk
    nc_fpath = dirs["output"]+"/data/"+str(int(time_dict["planet"]))+"_atm.nc"
    atm.write_ncdf(nc_fpath)

    # Store new flux
    if (COUPLER_options["F_atm_bc"] == 0):
        F_atm_new = atm.net_flux[0]  
    else:
        F_atm_new = atm.net_flux[-1]  

    # Flux change limiters
    F_atm_lim = F_atm_new
    if (time_dict["planet"] > 3):

        run_atm = runtime_helpfile.loc[runtime_helpfile['Input']=='Atmosphere'].drop_duplicates(subset=['Time'], keep='last')
        F_atm_old = run_atm.loc[run_atm['Time'] != time_dict["planet"]].iloc[-1]["F_atm"]

        if (F_atm_old < COUPLER_options["F_crit"]):
            rel_max = abs(COUPLER_options["limit_pos_flux_change"])/100.0
            F_atm_lim = min(F_atm_lim, (1+rel_max) * F_atm_old)

            rel_max = abs(COUPLER_options["limit_neg_flux_change"])/100.0
            F_atm_lim = max(F_atm_lim, (1-rel_max) * F_atm_old)

    # Require that the net flux must be upward
    if (COUPLER_options["prevent_warming"] == 1):
        F_atm_lim = max( 1.0e-8 , F_atm_lim )

    # Print if a limit was applied
    if (F_atm_lim != F_atm_new ):
        print("Change in F_atm [W m-2] limited in this step!")
        print("    %g  ->  %g" % (F_atm_new , F_atm_lim))
            
    COUPLER_options["F_atm"] = F_atm_lim
    COUPLER_options["F_olr"] = atm.LW_flux_up[0]

    return COUPLER_options

