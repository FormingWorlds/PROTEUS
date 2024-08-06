# Functions used to handle escape

from utils.modules_ext import *
from utils.constants import *
from utils.helper import *

import mors
from zephyrus import EL_escape

log = logging.getLogger("PROTEUS")


def RunDummyEsc(hf_row:dict, dt:float, phi_bulk:float):
    """Run dummy escape model.

    Parameters
    ----------
        hf_row : dict 
            Dictionary of helpfile variables, at this iteration only
        dt : float 
            Time interval over which escape is occuring [yr]
        phi_bulk : float 
            Bulk escape rate [kg s-1]

    Returns
    ----------
        esc_result : dict 
            Dictionary of updated total elemental mass inventories [kg]

    """
    log.info("Running dummy escape...")

    # store value
    out = {}
    out["rate_bulk"] = phi_bulk

    # calculate total mass of volatiles (except oxygen, which is set by fO2)
    M_vols = 0.0
    for e in element_list:
        if e=='O': continue 
        M_vols += hf_row[e+"_kg_total"]


    # for each elem, calculate new total inventory while
    # maintaining a constant mass mixing ratio
    for e in element_list:
        if e=='O': continue

        # current elemental mass ratio in total 
        emr = hf_row[e+"_kg_total"]/M_vols

        log.debug("    %s mass ratio = %.2e "%(e,emr))

        # new total mass of element e, keeping a constant mixing ratio of that element 
        out[e+"_kg_total"] = emr * (M_vols - phi_bulk * dt * secs_per_year)

    return out


def RunZEPHYRUS(M_star,Omega_star,tidal_contribution, semi_major_axis, eccentricity, M_planet, epsilon, R_earth, Rxuv):
    """Run energy-limited escape (for now) model. 

    Parameters
    ----------
        M_star : float  
            Stellar mass in solar mass                  [M_sun, kg]
        Omega_star : float
            Stellar rotation rate                       [rad s-1]
        tidal_contribution  : string 
            Tidal correction factor ('yes' or 'no')     [dimensionless]
        semi_major_axis : float
            Planetary semi-major axis                   [m]
        eccentricity : float
            Planetary eccentricity                      [dimensionless]
        M_planet : float
            Planetary mass                              [kg]
        epsilon : float
            Escape efficiency factor                    [dimensionless]  
        R_earth : float
            Planetary radius                            [m]
        Rxuv : float
            XUV planetary radius                        [m]

    Returns
    ----------
        mlr : float                          
            Total mass loss rate for energy-limited escape    [kg s-1]
    """

    log.info("Running EL escape (ZEPHYRUS) ...")

    ## Step 1 : Load stellar evolution track + compute EL escape 
    star            = mors.Star(Mstar=M_star, Omega=Omega_star)                                                                              # Load the stellar evolution track from MORS
    age_star        = star.Tracks['Age']                                                                                                     # Stellar age                          [Myr]
    Fxuv_star_SI    = ((star.Tracks['Lx']+star.Tracks['Leuv'])/(4*np.pi*(semi_major_axis*1e2)**2)) * ergcm2stoWm2                            # XUV flux                             [erg s-1]
    mlr             = EL_escape(tidal_contribution, semi_major_axis, eccentricity, M_planet, M_star, epsilon, R_earth, Rxuv, Fxuv_star_SI)   # Compute EL escape                    [kg s-1]
    
    # Plot to verify the output 
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax1.loglog(age_star, mlr, '-', color='orange', label='MORS stellar evolution tracks')
    ax1.set_xlabel('Time [Myr]', fontsize=15)
    ax1.set_ylabel(r'Mass loss rate [kg $s^{-1}$]', fontsize=15)
    ax1.set_title('Zephyrus : EL escape for Sun-Earth system', fontsize=15)
    ax1.grid(alpha=0.4)
    ax1.legend()
    ax1.set_yscale('log')
    ax2 = ax1.twinx()
    ylims = ax1.get_ylim()
    ax2.set_ylim((ylims[0]/ s2yr) / M_earth,(ylims[1] / s2yr) / M_earth)
    ax2.set_yscale('log')
    ax2.set_ylabel(r'Mass loss rate [$M_{\oplus}$ $yr^{-1}$]', fontsize=15)
    plt.savefig('test_escape.png', dpi=180)

    ## Step 2 : Updated total elemental mass inventories
    # store value

    out = {}
    out["rate_bulk"] = mlr

    # calculate total mass of volatiles (except oxygen, which is set by fO2)
    M_vols = 0.0
    for e in element_list:
        if e=='O': continue 
        M_vols += hf_row[e+"_kg_total"]


    # for each elem, calculate new total inventory while
    # maintaining a constant mass mixing ratio
    for e in element_list:
        if e=='O': continue

        # current elemental mass ratio in total 
        emr = hf_row[e+"_kg_total"]/M_vols

        log.debug("    %s mass ratio = %.2e "%(e,emr))

        # new total mass of element e, keeping a constant mixing ratio of that element 
        out[e+"_kg_total"] = emr * (M_vols - phi_bulk * dt * secs_per_year)

    return mlr,out


