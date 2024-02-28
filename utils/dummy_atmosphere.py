# Dummy atmosphere module 

from utils.modules_ext import *
from utils.helper import *
from utils.constants import *

log = logging.getLogger(__name__)

# Run the dummy atmosphere module (calculate fluxes)
def RunDummyAtm( time_dict, dirs, COUPLER_options, runtime_helpfile ):

    PrintHalfSeparator()
    log.info("Running dummy_atmosphere...")

    # Parameters
    T_surf_int = COUPLER_options["T_surf"]
    zenith_angle    = COUPLER_options["zenith_angle"]
    albedo_pl       = COUPLER_options["albedo_pl"]
    inst_sf         = COUPLER_options["asf_scalefactor"]
    albedo_s        = COUPLER_options["albedo_s"]
    instellation    = COUPLER_options["F_ins"]
    skin_d          = COUPLER_options["skin_d"]
    skin_k          = COUPLER_options["skin_k"]

    # Check configuration
    if COUPLER_options["atmosphere_solve_energy"] == 1:
        UpdateStatusfile(dirs, 20)
        raise Exception("Cannot solve for RCE with dummy_atmosphere")
    
    if COUPLER_options["insert_rscatter"] == 1:
        log.warning("Rayleigh scattering is enabled but it will be neglected")

    # Simple rad trans
    def _calc_fluxes(x):
        fl_U_LW = const_sigma * (1.0 - albedo_s) * x**4.0
        fl_D_SW = instellation * (1.0 - albedo_pl) * inst_sf * np.cos(zenith_angle * np.pi / 180.0)
        fl_U_SW = 0.0
        fl_N = fl_U_LW + fl_U_SW - fl_D_SW
        return {"fl_U_LW":fl_U_LW, "fl_D_SW":fl_D_SW, "fl_U_SW":fl_U_SW, "fl_N":fl_N}

    # fixed T_Surf
    if COUPLER_options["atmosphere_surf_state"] == 1:  
        T_surf_atm = T_surf_int
        fluxes = _calc_fluxes(T_surf_atm)
        
    # conductive lid
    elif COUPLER_options["atmosphere_surf_state"] == 2: 
        import scipy.optimize as optimise

        # We need to solve for the state where fl_N = f_skn
        # This function takes T_surf_atm as the input value, and returns fl_N - f_skn
        def _resid(x):
            F_skn = skin_k / skin_d * (T_surf_int - x)
            _f = _calc_fluxes(x)
            return _f["fl_N"] - F_skn

        r = optimise.root_scalar(_resid, method='secant', x0=T_surf_int, x1=T_surf_int-10.0, xtol=1.0e-6, maxiter=30)
        T_surf_atm = float(r.root)
        fluxes = _calc_fluxes(T_surf_atm)

    else:
        UpdateStatusfile(dirs, 20)
        raise Exception("Invalid surface state chosen for dummy_atmosphere")
    
    # Require that the net flux must be upward
    F_atm_lim = fluxes["fl_N"]
    if (COUPLER_options["prevent_warming"] == 1):
        F_atm_lim = max( 1.0e-8 , F_atm_lim )

    # Print if a limit was applied
    if not np.isclose(F_atm_lim , fluxes["fl_N"] ):
        log.warning("Change in F_atm [W m-2] limited in this step!")
        log.warning("    %g  ->  %g" % (fluxes["fl_N"] , F_atm_lim))

    # Return result
    log.info("Resultant values:")
    log.info("    T_surf =  %.3e  K"     % T_surf_atm)
    log.info("    F_atm  =  %.3e  W m-2" % F_atm_lim)
    log.info("    F_olr  =  %.3e  W m-2" % fluxes["fl_U_LW"])
    log.info("    F_sct  =  %.3e  W m-2" % fluxes["fl_U_SW"])

    COUPLER_options["T_surf"] = T_surf_atm
    COUPLER_options["F_atm"] =  F_atm_lim             # Net flux at TOA
    COUPLER_options["F_olr"] =  fluxes["fl_U_LW"]     # OLR
    COUPLER_options["F_sct"] =  fluxes["fl_U_SW"]     # Scattered SW flux
    return COUPLER_options
