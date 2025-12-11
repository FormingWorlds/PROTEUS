# Generic stellar evolution wrapper
from __future__ import annotations

import logging
import os
import shutil
from typing import TYPE_CHECKING

import numpy as np

from proteus.star.phoenix import get_phoenix_modern_spectrum
from proteus.utils.constants import AU, M_sun, R_sun, const_sigma, ergcm2stoWm2
from proteus.utils.helper import UpdateStatusfile

log = logging.getLogger("fwl."+__name__)

if TYPE_CHECKING:
    from proteus import Proteus
    from proteus.config import Config

# Mass limits on stellar tracks [Msun]
MASS_LIM = {
    "spada":   (0.10, 1.25),
    "baraffe": (0.01, 1.40)
}

def init_star(handler:Proteus):
    '''
    Star-related things to be done when the simulation begins.

    This includes:
        - Preparing the stellar model
        - Reading the present-day stellar spectrum

    Parameters
    ----------
        handler : Proteus
            Proteus object instance
    '''

    log.info("Preparing stellar model")
    log.info("")

    # Dummy star module does not require preparation

    # Prepare MORS
    if handler.config.star.module == 'mors':
        import mors
        mors_cfg = handler.config.star.mors
        fwl_dir  = handler.directories["fwl"]
        starname_input = mors_cfg.star_name
        starname_proper = starname_input.strip().lower().replace(" ", "-").replace("gj-", "gj") + ".txt"

        solar_path = os.path.join(fwl_dir, "stellar_spectra/solar", starname_proper)
        muscles_path = os.path.join(fwl_dir, "stellar_spectra/MUSCLES", starname_proper)

        src = mors_cfg.spectrum_source

        # spectrum_source = None -> try solar, then MUSCLES
        if src is None:
            if os.path.exists(solar_path):
                star_modern_path = solar_path
            elif os.path.exists(muscles_path):
                star_modern_path = muscles_path
            else:
                log.error(f"No stellar spectrum found for '{mors_cfg.star_name}' in reference data.")
                log.error("Check the available spectra at https://fwl-proteus.readthedocs.io/en/latest/data.html.")
                UpdateStatusfile(handler.directories, 23)
                raise FileNotFoundError(
                    f"No solar or MUSCLES spectrum found in reference data for '{mors_cfg.star_name}'.")

        # spectrum_source = 'solar'
        elif src == 'solar':
            if os.path.exists(solar_path):
                star_modern_path = solar_path
            elif os.path.exists(muscles_path):
                log.warning(f"Requested solar spectrum for '{mors_cfg.star_name}', but file cannot be found in solar reference data. ")
                log.warning("Using available MUSCLES spectrum.")
                log.warning("To use MUSCLES by default, set star.mors.spectrum_source = 'muscles'.")
                star_modern_path = muscles_path
            else:
                log.error(f"Requested solar spectrum for '{mors_cfg.star_name}', but the file cannot be found in reference data.")
                log.error("If you would like to use the NREL modern solar spectrum, please set star.mors.star_name = 'sun'.")
                UpdateStatusfile(handler.directories, 23)
                raise FileNotFoundError(f"No solar or MUSCLES spectrum for '{mors_cfg.star_name}'.")

        # spectrum_source = 'muscles'
        elif src == 'muscles':
            if os.path.exists(muscles_path):
                star_modern_path = muscles_path
            elif os.path.exists(solar_path):
                log.warning(f"Requested MUSCLES spectrum for '{mors_cfg.star_name}', but the file cannot be found in MUSCLES reference data.")
                log.warning("The file is available in solar reference data. Using that instead.")
                log.warning("To use solar by default, set star.mors.spectrum_source = 'solar'.")
                star_modern_path = solar_path
            else:
                log.error(f"Requested MUSCLES spectrum for '{mors_cfg.star_name}', but the file cannot be found in MUSCLES reference data.")
                log.error("Check the available MUSCLES spectra at https://fwl-proteus.readthedocs.io/en/latest/data.html.")
                log.error(
                    "If available, MUSCLES spectra can be downloaded via the command line: "
                    f"python -c \"from proteus.utils.data import download_muscles; download_muscles('{mors_cfg.star_name}')\"."
                )
                log.error("If no observed spectrum is available, consider using a PHOENIX synthetic spectrum by setting star.mors.spectrum_source = 'phoenix'.")
                UpdateStatusfile(handler.directories, 23)
                raise FileNotFoundError(f"No MUSCLES spectrum for '{mors_cfg.star_name}'.")

        # spectrum_source = 'phoenix'
        elif src == 'phoenix':
            # Build a temporary stellar track so PHOENIX can infer Teff, radius, log g
            Mstar_tmp = float(handler.config.star.mass)
            Mstar_tmp = max(Mstar_tmp, MASS_LIM[mors_cfg.tracks][0])
            Mstar_tmp = min(Mstar_tmp, MASS_LIM[mors_cfg.tracks][1])

            if mors_cfg.tracks == 'spada':
                age_now_Myr = mors_cfg.age_now * 1000  # convert Gyr to Myr
                pcntle = mors_cfg.rot_pcntle
                period = mors_cfg.rot_period

                if pcntle is not None:
                    age_rot_Myr = 1.0
                    period = None
                else:
                    age_rot_Myr = age_now_Myr

                track_for_phoenix = mors.Star(
                    Mstar=Mstar_tmp,
                    Age=age_rot_Myr,
                    percentile=pcntle,
                    Prot=period
                )
            else:  # 'baraffe'
                track_for_phoenix = mors.BaraffeTrack(Mstar_tmp)

            star_modern_path = str(
                get_phoenix_modern_spectrum(handler, stellar_track=track_for_phoenix)
            )

        else:
            log.error(f"Unknown star.mors.spectrum_source = {src}")
            UpdateStatusfile(handler.directories, 23)
            raise ValueError(f"Unknown spectrum_source: {src}")

        log.info(f"Using stellar spectrum file: {star_modern_path}")
        log.info("")

        # Copy modern spectrum to output folder, for posterity.
        star_backup_path = os.path.join(handler.directories["output/data"], "-1.sflux")
        shutil.copyfile(star_modern_path, star_backup_path)

        # Clip mass if required
        Mstar = float(handler.config.star.mass)
        Mstar = max(Mstar, MASS_LIM[handler.config.star.mors.tracks][0])
        Mstar = min(Mstar, MASS_LIM[handler.config.star.mors.tracks][1])
        if Mstar != handler.config.star.mass:
            log.warning(f"Star mass out of range. Clipped to {Mstar} Msun")

        match handler.config.star.mors.tracks:

            case 'spada':

                age_now_Myr = handler.config.star.mors.age_now * 1000 # convert Gyr to Myr

                # Get rotation period or percentile (one of these is None)
                pcntle = handler.config.star.mors.rot_pcntle
                period = handler.config.star.mors.rot_period

                if pcntle is not None:
                    # Rotation set by percentile.
                    #  The reference age must be 1 Myr for consistency with the assumptions
                    #  withing `mors.star.Percentile()`.
                    age_rot_Myr = 1.0
                    period = None
                else:
                    # Rotation set by rotation period.
                    #  The reference age is set by the current age, for which we have
                    #  measurements of the envelope's rotation period.
                    age_rot_Myr = age_now_Myr

                # load and fit track data
                try:
                    # clip mass
                    handler.stellar_track = mors.Star(Mstar = Mstar,
                                                        Age = age_rot_Myr,
                                                        percentile = pcntle, Prot=period)
                except Exception as e:
                    UpdateStatusfile(handler.directories, 23)
                    raise e

                # load modern spectrum
                # calculate band-integrated fluxes
                handler.star_struct = mors.spec.Spectrum()
                handler.star_struct.LoadTSV(star_modern_path)
                handler.star_struct.CalcBandFluxes()

                # calculate other properties from modern spectrum
                handler.star_props = get_spada_synthesis_properties(
                                                handler.stellar_track, age_now_Myr )

            case 'baraffe':
                # creates track data
                handler.stellar_track = mors.BaraffeTrack(Mstar)

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
    out["F_pl"] = np.trapezoid(fl_pl, wl_pl)
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
        from proteus.star.dummy import generate_spectrum, get_star_radius
        wl, fl = generate_spectrum(config.star.dummy.Teff, get_star_radius(config))

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
    Scale stellar fluxes from 1 AU to top of the planet's atmosphere.

    Parameters
    ----------
        fl_arr : iterable
            Stellar fluxes at 1 AU
        sep : float
            Planet-star distance, in units of AU
    Returns
    ----------
        fl_arr : np.ndarray
            Incoming stellar radiation scaled to the correct distance.
    '''
    return np.array(fl_arr) * ( (AU / sep)**2 )

def write_spectrum(wl_arr, fl_arr, hf_row:dict, output_dir:str):
    '''
    Write stellar spectrum to file.

    Parameters
    ----------
        wl_arr : np.ndarray
            Wavelength array [nm]
        fl_arr : np.ndarray
            Stellar fluxes at 1 AU [erg s-1 cm-2 nm-1]
        hf_row : dict
            Current helpfile row
        output_dir : str
            Proteus output directory
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
    """
    Wrapper function to update stellar quantities, such as luminosity and radius.

    Modifies hf_row in-place. This function is called during the PROTEUS simulation loop.

    Parameters
    ----------
        hf_row : dict
            Current helpfile row
        config : Config
            Proteus configuration object
        stellar_track
            Mors stellar track object, if applicable.
    """

    # Update value for star's radius and mass
    log.debug("Update stellar radius and mass")
    update_stellar_radius(hf_row, config, stellar_track)
    update_stellar_mass(hf_row, config)

    # Update value for instellation flux
    log.debug("Update stellar fluxes and temperature")
    update_instellation(hf_row, config, stellar_track)
    update_stellar_temperature(hf_row, config, stellar_track)
    log.info("    F_ins      = %.3e   W m-2"%hf_row["F_ins"])
    log.info("    F_xuv      = %.3e   W m-2"%hf_row["F_xuv"])
    log.info("    T_star     = %.3f    K"%hf_row["T_star"])

    # Calculate new eqm temperature
    log.debug("Update equilibrium temperature")
    update_equilibrium_temperature(hf_row, config)

    # Calculate new skin temperature
    # Assuming a grey stratosphere in radiative eqm (https://doi.org/10.5194/esd-7-697-2016)
    hf_row["T_skin"] = hf_row["T_eqm"] * (0.5**0.25)

def update_stellar_mass(hf_row:dict, config:Config):
    '''
    Update stellar mass in hf_row, stored in SI units.
    '''
    hf_row["M_star"] = config.star.mass * M_sun

def update_stellar_radius(hf_row:dict, config:Config, stellar_track=None):
    '''
    Update stellar radius in hf_row, stored in SI units.
    '''

    # Dummy case
    if config.star.module == 'dummy':
        from proteus.star.dummy import get_star_radius
        R_star = get_star_radius(config)

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

def update_stellar_temperature(hf_row:dict, config:Config, stellar_track=None):
    '''
    Update stellar temperature in hf_row, stored in SI units.
    '''

    # Dummy case
    if config.star.module == 'dummy':
        hf_row["T_star"] = config.star.dummy.Teff

    # Mors cases
    elif config.star.module == 'mors':

        # which track?
        match config.star.mors.tracks:
            case 'spada':
                hf_row["T_star"] = stellar_track.Value(hf_row["age_star"] / 1e6, "Teff")
            case 'baraffe':
                hf_row["T_star"] = stellar_track.BaraffeStellarTeff(hf_row["age_star"])

def update_instellation(hf_row:dict, config:Config, stellar_track=None):
    '''
    Update hf_row value of bolometric stellar flux impinging upon the planet.
    '''

    # Dummy case
    if config.star.module == 'dummy':
        from proteus.star.dummy import calc_instellation
        S_0 = calc_instellation(config.star.dummy.Teff, hf_row["R_star"], hf_row["separation"])
        Fxuv_SI = 0.0

    # Mors cases
    elif config.star.module == 'mors':

        # which track?
        match config.star.mors.tracks:
            case 'spada':

                age_star = hf_row["age_star"] / 1e6

                # Bolometric flux
                S_0 = stellar_track.Value(age_star, "Lbol") * 1e-7 \
                        / (4.0 * np.pi * hf_row["separation"]**2.0 )

                # Interpolating the XUV flux at the age of the star
                Lxuv_cgs = stellar_track.Value(age_star, 'Lx') + \
                                stellar_track.Value(age_star, 'Leuv')
                Fxuv_SI = Lxuv_cgs/(4*np.pi * (hf_row["separation"]*1e2)**2) * ergcm2stoWm2

            case 'baraffe':

                # Bolometric flux
                S_0 = stellar_track.BaraffeSolarConstant(hf_row["age_star"],
                                                    hf_row["separation"]/AU)

                # XUV flux not provided by Baraffe tracks
                Fxuv_SI = 0.0

    # Update hf_row dictionary
    hf_row["F_ins"] = S_0 * config.star.bol_scale
    hf_row["F_xuv"] = Fxuv_SI * config.star.bol_scale

def update_equilibrium_temperature(hf_row:dict, config:Config):
    '''
    Calculate planetary equilibrium temperature.
    '''

    # Absorbed stellar flux
    F_asf = hf_row["F_ins"] * config.orbit.s0_factor * (1-hf_row["albedo_pl"])

    # Planetary equilibrium temperature
    hf_row["T_eqm"] = (F_asf / const_sigma)**(1.0/4.0)
