from __future__ import annotations

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


def calc_eqm_temperature(I_0, ASF, A_B):
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
