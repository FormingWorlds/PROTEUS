from __future__ import annotations

import mors
from mors.baraffe import BaraffeSolarConstant, BaraffeStellarRadius
from proteus.star.dummy import calc_instellation
from proteus.utils.constants import R_sun, L_sun

def write_spectrum(fl:list, wl:list, sep:float, time:float):
    """Write stellar spectrum to a file.

    Parameters
    ----------
        fl : list
            Fluxes [erg s-1 cm-2 nm-1]
        wl : list
            Wavelengths [nm]
        sep : float
            Planet-star separation [AU]
        time : float
            Model time [yr]

    """

    # Scale fluxes from 1 AU to TOA
    fl *= (AU / sep) ** 2.0

    # Save spectrum to file
    header = (
        "# WL(nm)\t Flux(ergs/cm**2/s/nm)"
        % hf_row["age_star"]
    )
    np.savetxt(
        os.path.join(self.directories["output"], "data", "%d.sflux" % time),
        np.array([wl, fl]).T,
        header=header,
        comments="",
        fmt="%.8e",
        delimiter="\t",
    )


def calc_eqm_temperature(I_0:float, ASF:float, A_B:float):
    """Calculate planetary equilibrium temperature.

    Parameters
    ----------
        I_0 : float
            Stellar flux [W m-2]
        ASF : float
            Scale factor to stellar flux
        A_B : float
            Enforced bond (or planetary) albedo

    """
    return (I_0 * ASF * (1.0 - A_B) / const_sigma)**(1.0/4.0)


def calc_stellar_radius(OPTIONS:dict, age_star:float):
    """Calculate stellar radius.

    Parameters
    ----------
        OPTIONS : dict
            Model configuration
        age_star : float
            Stellar age [yr]

    Returns
    ----------
        Rstar : float
            Stellar radius [m]
    """

    match OPTIONS["star_model"]:
        case 0:
            # MORS Spada
            Rstar = mors.Value(OPTIONS["star_mass"],age_star / 1e6,"Rstar")

        case 1:
            # MORS Baraffe
            Rstar = BaraffeStellarRadius(age_star)

        case 2:
            # Dummy
            # Get radius from config file

    # Convert to metres and return
    Rstar *= R_sun
    return Rstar

def calc_instellation(OPTIONS:dict, age_star:float, sep:float, Rstar:float):
    """Calculate instellation (TOA downward stellar flux)

    Parameters
    ----------
        OPTIONS : dict
            Model configuration
        age_star : float
            Stellar age [yr]
        sep : float
            Planet-star separation [m]
        Rstar : float
            Stellar radius [m]

    Returns
    ----------
        S_0 : float
            Bolometric flux [W m-2]
    """

     match OPTIONS["star_model"]:
        case 0:
            # MORS Spada
            L_bol = mors.Value(OPTIONS["star_mass"],hf_row["age_star"] / 1e6,"Lbol")
            S_0   = L_bol * L_sun  / (4.0 * np.pi * sep*sep )

        case 1:
            # MORS Baraffe
            S_0 = BaraffeSolarConstant(hf_row["age_star"], sep/AU)

        case 2:
            # Dummy
            S_0 = calc_instellation(OPTIONS["star_teff"], sep, Rstar)

    return S_0

