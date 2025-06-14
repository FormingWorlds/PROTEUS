# Dummy atmosphere module
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import const_sigma
from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger("fwl."+__name__)

# Run the dummy atmosphere module
def RunDummyAtm( dirs:dict, config:Config, T_magma:float, F_ins:float, R_int:float, M_int:float, P_surf:float):
    log.debug("Running dummy atmosphere...")

    # Gamma factor: VERY simple parameterisation for the radiative properties of the atmosphere.
    # It represents a measure of the radiating temperature of the atmosphere above the
    #    surface, relative to the surface temperature itself
    # Setting this to 0 will result in an entirely transparent atmosphere
    # Setting this to 1 will result in an OLR of zero

    # Parameters
    gamma           = config.atmos_clim.dummy.gamma
    zenith_angle    = config.orbit.zenith_angle
    albedo_pl       = config.atmos_clim.albedo_pl
    inst_sf         = config.orbit.s0_factor
    albedo_s        = config.atmos_clim.surf_greyalbedo
    skin_d          = config.atmos_clim.surface_d
    skin_k          = config.atmos_clim.surface_k

    log.debug("Gamma = %.4f" % gamma)

    # Simple rad trans
    def _calc_fluxes(x):
        # surface emission and stellar flux
        fl_U_LW = const_sigma * (x - gamma * x)**4.0
        fl_D_SW = F_ins * (1.0 - albedo_pl) * inst_sf * np.cos(zenith_angle * np.pi / 180.0)

        # surface reflection
        fl_U_SW = fl_D_SW * albedo_s
        fl_D_SW = fl_D_SW * (1.0-albedo_s)

        # net flux at surface
        fl_N = fl_U_LW + fl_U_SW - fl_D_SW

        return {"fl_U_LW":fl_U_LW, "fl_D_SW":fl_D_SW, "fl_U_SW":fl_U_SW, "fl_N":fl_N}

    # fixed T_Surf
    if config.atmos_clim.surf_state == 'fixed':
        log.info("Calculating fluxes with dummy atmosphere")
        T_surf_atm = T_magma
        fluxes = _calc_fluxes(T_surf_atm)

    # conductive lid
    elif config.atmos_clim.surf_state == 'skin':
        log.info("Calculating fluxes with dummy atmosphere and CBL")
        import scipy.optimize as optimise

        # We need to solve for the state where fl_N = f_skn
        # This function takes T_surf_atm as the input value, and returns fl_N - f_skn
        def _resid(x):
            F_skn = skin_k / skin_d * (T_magma - x)
            _f = _calc_fluxes(x)
            return _f["fl_N"] - F_skn

        r = optimise.root_scalar(_resid, method='secant', x0=T_magma, x1=T_magma-10.0,
                                        xtol=1.0e-7, maxiter=40)
        T_surf_atm = float(r.root)
        fluxes = _calc_fluxes(T_surf_atm)

        if r.converged:
            log.info("Found solution after %d iterations" % int(r.iterations))
        else:
            UpdateStatusfile(dirs, 22)
            raise RuntimeError("Could not find solution for T_surf with dummy_atmosphere")

    else:
        UpdateStatusfile(dirs, 20)
        raise ValueError("Invalid surface state chosen for dummy_atmosphere")

    # Require that the net flux must be upward
    F_atm_lim = fluxes["fl_N"]
    if config.atmos_clim.prevent_warming:
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

    # Escape level always at surface for dummy atmosphere
    if config.escape.module == 'zephyrus':
        log.warning("Setting escape level to surface because dummy atmosphere is used")

    output = {}
    output["T_surf"]  = T_surf_atm
    output["F_atm"]   = F_atm_lim             # Net flux at TOA
    output["F_olr"]   = fluxes["fl_U_LW"]     # OLR
    output["F_sct"]   = fluxes["fl_U_SW"]     # Scattered SW flux
    output["R_obs"]   = R_int
    output["rho_obs"] = 3 * M_int / (4*np.pi*R_int**3)
    output["albedo"]  = fluxes["fl_U_SW"]/fluxes["fl_D_SW"]
    output["p_xuv"]   = P_surf
    output["R_xuv"]   = R_int
    output["p_obs"]   = P_surf

    return output
