# Function used to handle atmosphere thermodynamics (running AEOLUS, etc.)


from utils.modules_ext import *
from utils.helper import *
from AEOLUS.utils.atmosphere_column import atmos
from AEOLUS.modules.radcoupler import RadConvEqm

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


# Generate/adapt atmosphere chemistry/radiation input files
def StructAtm( loop_counter, dirs, runtime_helpfile, COUPLER_options ):

    # In the beginning: standard surface temperature from last entry
    if loop_counter["total"] < loop_counter["init_loops"]:
        COUPLER_options["T_surf"] = runtime_helpfile.iloc[-1]["T_surf"]
    
    # Check for flux_convergence scheme criteria
    elif (COUPLER_options["flux_convergence"] == 1 \
    and runtime_helpfile.iloc[-1]["RF_depth"] < COUPLER_options["RF_crit"] \
    and COUPLER_options["F_net"] > COUPLER_options["F_diff"]*COUPLER_options["F_int"]) \
    or  COUPLER_options["flux_convergence"] == 2:

        PrintSeparator()
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

        PrintSeparator()

    # Use Ts_int
    else:
        # Standard surface temperature from last entry
        COUPLER_options["T_surf"] = runtime_helpfile.iloc[-1]["T_surf"]

    # Create atmosphere object and set parameters
    pl_radius = COUPLER_options["radius"]
    pl_mass = COUPLER_options["gravity"] * pl_radius * pl_radius / phys.G

    vol_list = { 
                  "H2O" : runtime_helpfile.iloc[-1]["H2O_mr"], 
                  "CO2" : runtime_helpfile.iloc[-1]["CO2_mr"],
                  "H2"  : runtime_helpfile.iloc[-1]["H2_mr"], 
                  "N2"  : runtime_helpfile.iloc[-1]["N2_mr"],  
                  "CH4" : runtime_helpfile.iloc[-1]["CH4_mr"], 
                  "O2"  : runtime_helpfile.iloc[-1]["O2_mr"], 
                  "CO"  : runtime_helpfile.iloc[-1]["CO_mr"], 
                  "He"  : 0.,
                  "NH3" : 0., 
                }

    atm = atmos(COUPLER_options["T_surf"], runtime_helpfile.iloc[-1]["P_surf"]*1e5, 
                COUPLER_options["P_top"]*1e5, pl_radius, pl_mass,
                vol_mixing=vol_list
                )

    atm.zenith_angle    = COUPLER_options["zenith_angle"]
    atm.albedo_pl       = COUPLER_options["albedo_pl"]
    atm.albedo_s        = COUPLER_options["albedo_s"]
        

    return atm, COUPLER_options


def RunAEOLUS( atm, time_dict, dirs, runtime_helpfile, loop_counter, COUPLER_options ):

    # Runtime info
    PrintSeparator()
    print("SOCRATES run... (loop =", loop_counter, ")")
    PrintSeparator()

    # Calculate temperature structure and heat flux w/ SOCRATES
    _, atm = RadConvEqm(dirs, time_dict, atm, standalone=False, cp_dry=False, trppD=True, rscatter=True,calc_cf=False) # W/m^2
    
    # Atmosphere net flux from topmost atmosphere node; do not allow heating
    COUPLER_options["F_atm"] = np.max( [ 0., atm.net_flux[0] ] )

    # Clean up run directory
    PrintSeparator()
    print("Remove SOCRATES auxiliary files:", end =" ")
    for file in natural_sort(glob.glob(dirs["output"]+"/current??.????")):
        os.remove(file)
        print(os.path.basename(file), end =" ")
    for file in natural_sort(glob.glob(dirs["output"]+"/profile.*")):
        os.remove(file)
        print(os.path.basename(file), end =" ")
    print(">>> Done.")

    return atm, COUPLER_options
