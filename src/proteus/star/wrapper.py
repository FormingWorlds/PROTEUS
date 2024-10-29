# Generic stellar evolution wrapper
from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import AU, R_sun, const_sigma

log = logging.getLogger("fwl."+__name__)

if TYPE_CHECKING:
    from proteus import Proteus
    from proteus.config import Config

def init_star(handler:Proteus):
    '''
    Star-related things to be done when the simulation begins.
    '''

    log.debug("Prepare stellar models")

    # Path to the modern spectrum
    #   i.e. that observed by a telescope, or derived from various observations.
    #   This is what we download from OSF.
    star_modern_path = os.path.join(handler.directories["fwl"],
                                    handler.config.star.mors.spec)

    # Copy modern spectrum to output folder, for posterity.
    star_backup_path = os.path.join(handler.directories["output"], "-1.sflux")
    shutil.copyfile(star_modern_path, star_backup_path)

    # Dummy star modules does not require preparation

    # Prepare MORS
    if handler.config.star.module == 'mors':
        import mors

        match handler.config.star.mors.tracks:

            case 'spada':
                # creates track data
                handler.stellar_track = mors.Star(Mstar = handler.config.star.mass,
                                                  percentile = handler.config.star.mors.rot_pctle)

                # load modern spectrum
                # calculate band-integrated fluxes
                handler.star_struct = mors.spec.Spectrum()
                handler.star_struct.LoadTSV(star_modern_path)
                handler.star_struct.CalcBandFluxes()

                # calculate other properties from modern spectrum
                handler.star_props = get_spada_synthesis_properties(
                    handler.stellar_track,
                    handler.config.star.mors.age_now * 1000, # convert Gyr to Myr
                )

            case 'baraffe':
                # creates track data
                handler.stellar_track = mors.BaraffeTrack(handler.config.star.mass)

                # load modern spectrum
                handler.star_modern_wl, handler.star_modern_fl = mors.ModernSpectrumLoad(
                    star_modern_path, star_backup_path
                )

                # calculate other properties from modern spectrum (here bolometric luminosity only)
                handler.star_props = handler.stellar_track.BaraffeLuminosity(handler.config.star.mors.age_now * 1e9)

def get_spada_synthesis_properties(spada_track, age: float):
    """Calculate properties of star for spectrum synthesis
       Mimic the GetProperties function of mors synthesis module using whole track data.

    Parameters
    ----------
        spada_track : star object
            Star object containing track data.
        age : float
            Stellar age  [Myr]

    Returns
    ----------
        out : dict
            Dictionary of radius [m], Teff [K], and band fluxes at 1 AU [erg s-1 cm-2]
    """

    import mors.spectrum as spec

    out = {}
    out["age"] = age
    out["Rstar"] = spada_track.Value(age, "Rstar") * R_sun #[m]
    out["Teff"] = spada_track.Value(age, "Teff") #[K]

    # Luminosities (erg s-1)
    out["L_bo"] = spada_track.Value(age, "Lbol")
    out["L_xr"] = spada_track.Value(age, "Lx")
    out["L_e1"] = spada_track.Value(age, "Leuv1")
    out["L_e2"] = spada_track.Value(age, "Leuv2")

    # Fluxes at 1 AU
    area = (4.0 * np.pi * AU * AU)
    for k in ["bo","xr","e1","e2"]:
        out["F_"+k] = out["L_"+k]/area

    # Get flux from Planckian band
    wl_pl = np.logspace(np.log10(spec.bands_limits["pl"][0]), np.log10(spec.bands_limits["pl"][1]), 1000)
    fl_pl = spec.PlanckFunction_surf(wl_pl, out["Teff"])
    fl_pl = spec.ScaleTo1AU(fl_pl, out["Rstar"])
    out["F_pl"] = np.trapz(fl_pl, wl_pl)
    out["L_pl"] = out["F_pl"] * area

    # Get flux of UV band from remainder
    out["F_uv"] = out["F_bo"] - out["F_xr"] - out["F_e1"] - out["F_e2"] - out["F_pl"]
    out["L_uv"] = out["F_uv"] * area

    return out

def get_new_spectrum(t_star:float, config:Config,
                     star_struct_modern=None, star_props_modern=None,
                     stellar_track=None, modern_wl=None, modern_fl=None):
    '''
    Get new stellar spectrum at 1 AU.
    '''

    log.debug("Get new stellar spectrum (star age = %g Gyr)"%(t_star/1e9))

    # Dummy case
    if config.star.module == 'dummy':
        from proteus.star.dummy import generate_spectrum
        wl, fl = generate_spectrum(config.star.dummy.Teff, config.star.dummy.radius * R_sun)

    # Mors cases
    elif config.star.module == 'mors':

        import mors

        match config.star.mors.tracks:
            case 'spada':
                star_props_hist = get_spada_synthesis_properties(stellar_track, t_star/1e6)
                assert star_struct_modern
                assert star_props_modern
                synthetic = mors.synthesis.CalcScaledSpectrumFromProps(
                    modern_spec=star_struct_modern, modern_dict=star_props_modern, historical_dict=star_props_hist)
                fl = synthetic.fl
                wl = synthetic.wl
            case 'baraffe':
                fl = stellar_track.BaraffeSpectrumCalc(
                        t_star, star_props_modern, modern_fl)
                wl = modern_wl

    return wl, fl

def scale_spectrum_to_toa(fl_arr, sep:float):
    '''
    Scale stellar fluxes from 1 AU to top of atmosphere
    '''
    return np.array(fl_arr) * ( (AU / sep)**2 )

def write_spectrum(wl_arr, fl_arr, hf_row:dict, output_dir:str):
    '''
    Write stellar spectrum to file.
    '''

    log.debug("Writing stellar spectrum to file")

    # Header information
    header = (
        "# WL(nm)\t Flux(ergs/cm**2/s/nm)   Stellar flux at t_star = %.2e yr"
        % hf_row["age_star"]
    )

    # Write to TSV file
    np.savetxt(
        os.path.join(output_dir, "data", "%d.sflux" % hf_row["Time"]),
        np.array([wl_arr, fl_arr]).T,
        header=header,
        comments="",
        fmt="%.8e",
        delimiter="\t",
    )

def update_stellar_quantities(hf_row:dict, config:Config, stellar_track=None):

    # Update value for star's radius
    log.info("Update stellar radius")
    update_stellar_radius(hf_row, config, stellar_track)

    # Update value for instellation flux
    log.info("Update instellation")
    update_instellation(hf_row, config, stellar_track)

    # Calculate new eqm temperature
    log.info("Update equilibrium temperature")
    update_equilibrium_temperature(hf_row, config)

    # Calculate new skin temperature
    # Assuming a grey stratosphere in radiative eqm (https://doi.org/10.5194/esd-7-697-2016)
    hf_row["T_skin"] = hf_row["T_eqm"] * (0.5**0.25)

def update_stellar_radius(hf_row:dict, config:Config, stellar_track=None):
    '''
    Update stellar radius in hf_row, stored in SI units.
    '''

    # Dummy case
    if config.star.module == 'dummy':
        R_star = config.star.dummy.radius

    # Mors cases
    elif config.star.module == 'mors':

        # which track?
        match config.star.mors.tracks:
            case 'spada':
                R_star = stellar_track.Value(hf_row["age_star"] / 1e6, "Rstar")
            case 'baraffe':
                R_star = stellar_track.BaraffeStellarRadius(hf_row["age_star"])

    # Dimensionalise and store in dictionary
    hf_row["R_star"] = R_star * R_sun

def update_instellation(hf_row:dict, config:Config, stellar_track=None):
    '''
    Update hf_row value of bolometric stellar flux impinging upon the planet.
    '''

    # Dummy case
    if config.star.module == 'dummy':
        from proteus.star.dummy import calc_instellation
        S_0 = calc_instellation(config.star.dummy.Teff, hf_row["R_star"], hf_row["separation"])

    # Mors cases
    elif config.star.module == 'mors':

        # which track?
        match config.star.mors.tracks:
            case 'spada':
                S_0 = stellar_track.Value(hf_row["age_star"] / 1e6, "Lbol") * 1e-7 \
                        / (4.0 * np.pi * hf_row["separation"]**2.0 )

            case 'baraffe':
                S_0 = stellar_track.BaraffeSolarConstant(hf_row["age_star"],
                                                    hf_row["separation"]/AU)

    # Update hf_row dictionary
    hf_row["F_ins"] = S_0

def update_equilibrium_temperature(hf_row:dict, config:Config):
    '''
    Calculate planetary equilibrium temperature.
    '''

    # Absorbed stellar flux
    F_asf = hf_row["F_ins"] * config.orbit.s0_factor * (1-config.atmos_clim.albedo_pl)

    # Planetary equilibrium temperature
    hf_row["T_eqm"] = (F_asf / const_sigma)**(1.0/4.0)
