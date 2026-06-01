# Function and classes used to run petitRADTRANS
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from petitRADTRANS import physical_constants as cst
from scipy.interpolate import PchipInterpolator

from proteus.utils.helper import eval_gas_mmw

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

petitRADTRANS_TLIMS = (100.0 + 0.5, 4000.0 - 0.5)
petitRADTRANS_LINE_OPACITY_MODE = 'c-k'
petitRADTRANS_GASES = (
    'H2',
    'H',
    'He',
    'H2O',
    'CH4',
    'CO',
    'CO2',
    'O',
    'C',
    'N',
    'NH3',
    'N2',
    'O2',
    'O3',
    'H2S',
    'HCN',
    'NO',
    'NO2',
    'OH',
    'PH3',
    'SiO',
    'SO2',
    'TiO',
    'VO',
    'Na',
    'K',
    'Ca',
    'Ti',
    'Fe',
    'Ni',
    'C2H2',
    'FeH',
)
petitRADTRANS_RAYLEIGH_SPECIES = ('H2', 'He', 'H')
petitRADTRANS_CIA_SPECIES = (
    'H-',
    'H2-H2',
    'H2-He',
    'CO2-CO2',
    'N2-H2',
    'N2-He',
    'N2-N2',
    'N2-O2',
    'O2-O2',
)
petitRADTRANS_IGNORED_GASES = {'e-', 'MMW', 'nabla_ad'}


def _get_input_data_path() -> str:
    """Return a local petitRADTRANS input-data directory."""

    import petitRADTRANS

    package_dir = Path(petitRADTRANS.__file__).resolve().parent
    candidate = package_dir / 'input_data'
    if candidate.is_dir():
        return str(candidate)

    raise FileNotFoundError(
        'Could not locate a petitRADTRANS input_data directory next to the installed '
        'petitRADTRANS package.'
    )


def _get_supported_line_species(gases: list[str]) -> list[str]:
    return [
        gas
        for gas in gases
        if gas not in petitRADTRANS_IGNORED_GASES and gas not in petitRADTRANS_RAYLEIGH_SPECIES
    ]


def _get_supported_rayleigh_species(gases: list[str]) -> list[str]:
    return [gas for gas in gases if gas in petitRADTRANS_RAYLEIGH_SPECIES]


def _get_supported_cia_species(gases: list[str]) -> list[str]:
    contributors = [gas for gas in petitRADTRANS_CIA_SPECIES if gas != 'H-' or 'H-' in gases]
    return contributors


def _vmrs_to_mass_fractions(
    gases: list[str], vmrs: list[np.ndarray]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Convert number fractions to pRT mass fractions and mean molar masses."""

    if len(vmrs) == 0:
        return {}, np.array([])

    vmr_arr = np.array([np.array(vmr, dtype=float) for vmr in vmrs], dtype=float)
    molar_masses = np.array([eval_gas_mmw(gas) for gas in gases], dtype=float)  # kg mol-1
    mass_contrib = vmr_arr * molar_masses[:, None]
    total_mass = np.sum(mass_contrib, axis=0)

    if np.any(total_mass <= 0.0):
        raise ValueError('Atmospheric composition has zero total mass fraction at one or more layers')

    mass_fractions = {
        gas: mass_contrib[i] / total_mass
        for i, gas in enumerate(gases)
        if gas not in petitRADTRANS_IGNORED_GASES
    }
    mean_molar_masses = total_mass / 1.0e-3  # pRT expects amu (g mol-1)
    return mass_fractions, mean_molar_masses


def _build_prt_composition(hf_row: dict, atm: dict, source: str, clip_vmr: float):
    gases, vmrs = _get_mix(hf_row, atm, source, clip_vmr)
    mass_fractions, mean_molar_masses = _vmrs_to_mass_fractions(gases, vmrs)
    line_species = _get_supported_line_species(gases)
    rayleigh_species = _get_supported_rayleigh_species(gases)
    gas_continuum_contributors = _get_supported_cia_species(gases)
    return (
        gases,
        mass_fractions,
        mean_molar_masses,
        line_species,
        rayleigh_species,
        gas_continuum_contributors,
    )


def _get_atm_profile(outdir: str, hf_row: dict) -> dict:
    """
    Reads the atmosphere data from the NetCDF file produced by the 'atmos_clim' module.
    """
    from proteus.atmos_clim.common import read_atmosphere_data

    atm_arr = read_atmosphere_data(
        outdir, [hf_row['Time']], extra_keys=['tmpl', 'pl', 'rl', 'x_gas']
    )

    if (len(atm_arr) == 0) or (atm_arr[-1] is None):
        log.warning(f"Could not read atmosphere data from '{outdir}'")
        return None
    return atm_arr[-1]


def _get_atm_offchem(outdir: str, hf_row: dict, chem_module: str) -> dict:
    """
    Reads the atmosphere data from the csv file produced by the 'atmos_chem' module.
    """

    # Read file
    from proteus.atmos_chem.common import read_result

    df = read_result(outdir, chem_module)

    # Check file exists
    if df is None:
        log.warning(f"Could not read offchem file from '{outdir}'")
        return None

    df.rename(columns={'tmp': 'tmpl', 'p': 'pl', 'z': 'rl'}, inplace=True)
    df['rl'] = df['rl'] + hf_row['R_int']  # convert height to radius (~ check that this is valid with prt)
    return df


def _get_mix(hf_row: dict, atm: dict, source: str, clip_vmr: float) -> tuple:
    """
    Get the gas abundance profiles and names of included gases.

    Parameters
    ---------
    hf_row : dict
        The row of the helpfile for the current time-step.
    atm : dict
        The atmosphere data as a dictionary.
    source : str
        Method for setting the mixing ratios: "outgas", "profile", or "offchem".
    clip_vmr : float
        Minimum VMR for a species to be included in the radiative transfer.
    """

    # Output arrays
    vmr_incl = []  # list of gas VMR arrays
    gas_incl = []  # list of gas names

    nlev_l = len(atm['pl'])

    # For all potentially supported gases...
    for gas in petitRADTRANS_GASES:
        key = gas + '_vmr'
        vmr = np.zeros(nlev_l)  # default to zero abundance

        # Read VMR from the source requested
        if source == 'outgas':
            # from helpfile -> constant VMR with height
            if key in hf_row:
                vmr = np.ones(nlev_l) * float(hf_row[key])

        elif source == 'profile':
            # from NetCDF file
            if key in atm.keys():
                vmr = np.array(atm[key])
                vmr = np.append(vmr[0], vmr)

        elif source == 'offchem':
            # from atmos_chem output file
            if gas in atm.keys():
                vmr = np.array(atm[gas])

        # neglect trace gases
        if np.amax(vmr) >= clip_vmr:
            vmr_incl.append(vmr)
            gas_incl.append(gas)

    return gas_incl, vmr_incl


# ~ adapt this for prt
def _construct_abundances(
    atm: dict, gas_incl: list, vmr_incl: list, T_grid: list, P_grid: list
) -> dict:
    """
    Constructs the abundance dictionary for petitRADTRANS from the gas names and VMR arrays.

    Parameters
    ----------
    atm : dict
        The atmosphere data as a dictionary.
    gas_incl : list
        The list of gas names.
    vmr_incl : list
        The list of VMR profiles.
    T_grid : list
        The temperature grid from petitRADTRANS [K]
    P_grid : list
        The pressure grid from petitRADTRANS [Pa].
    """

    abundances = {}

    for i, gas in enumerate(gas_incl):
        parr = [1e-6] + list(atm['pl']) + [1e13]
        varr = [vmr_incl[i][0]] + list(vmr_incl[i]) + [vmr_incl[i][-1]]
        itp_1d = PchipInterpolator(np.log10(parr), np.log10(varr))

        arr_2d = np.zeros((len(T_grid), len(P_grid)))
        for j, p in enumerate(P_grid):
            arr_2d[:, j] = 10 ** itp_1d(np.log10(p))
        abundances[gas] = arr_2d

    return abundances


def _get_ptr(atm: dict):
    """
    Returns the pressure, temperature and radius from the atmosphere data."""
    prs = np.array(atm['pl'])  # Pa
    tmp = np.array(atm['tmpl'])  # K
    rad = np.array(atm['rl'])  # m
    if prs[1] < prs[0]:
        prs = prs[::-1]
        tmp = np.clip(tmp[::-1], a_min=petitRADTRANS_TLIMS[0], a_max=petitRADTRANS_TLIMS[1])
        rad = rad[::-1]
    return prs, tmp, rad


def _copy_mass_fractions(
    mass_fractions: dict[str, np.ndarray], gases_to_zero: list[str]
) -> dict[str, np.ndarray]:
    copied = {gas: np.array(values, copy=True) for gas, values in mass_fractions.items()}
    for gas in gases_to_zero:
        if gas in copied:
            copied[gas] = np.zeros_like(copied[gas])
    return copied

# ~ adapt this for prt
def transit_depth(hf_row: dict, outdir: str, config: Config, source: str):
    """
    Computes the transit depth spectrum using petitRADTRANS.

    Parameters
    ----------
    hf_row : dict
        The row of the helpfile for the current time-step.
    outdir : str
        The output directory for the PROTEUS run.
    config : Config
        PROTEUS config object.
    source : str
        Method for setting the mixing ratios: "outgas", "profile", or "offchem".
    """
    from petitRADTRANS.radtrans import Radtrans

    from proteus.observe.common import get_transit_fpath

    # All planet quantities in SI
    Rs = hf_row['R_star']  # Radius of star [m]
    Mp = hf_row['M_planet']  # Mass of planet [kg]
    Rp = hf_row['R_int']  # Radius of planet [m]
    Rs_cm = Rs * 100.0
    Rp_cm = Rp * 100.0

    # Get profile from the required source
    if source == 'offchem':
        atm = _get_atm_offchem(outdir, hf_row, config.atmos_chem.module)
    elif source in ('outgas', 'profile'):
        atm = _get_atm_profile(outdir, hf_row)

    # Parse
    if atm is None:
        log.warning(f"Could not read atmosphere data for source '{source}'")
        return None

    # Get composition from requested source
    prs, tmp, _ = _get_ptr(atm)
    (
        gases,
        mass_fractions,
        mean_molar_masses,
        line_species,
        rayleigh_species,
        gas_continuum_contributors,
    ) = _build_prt_composition(hf_row, atm, source, config.observe.platon.clip_vmr)

    # create a Radtrans object
    log.debug('Compute transit depth spectra')
    radtrans = Radtrans(
        pressures=np.array(prs, dtype=float) / 1e5,
        line_species=line_species,
        gas_continuum_contributors=gas_continuum_contributors,
        rayleigh_species=rayleigh_species,
        line_opacity_mode=petitRADTRANS_LINE_OPACITY_MODE,
        path_input_data=_get_input_data_path(),
    )

    # compute full spectrum
    wl_cm, transit_radii, _ = radtrans.calculate_transit_radii(
        temperatures=np.array(tmp, dtype=float),
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses,
        reference_gravity=100.0 * cst.G * Mp / (Rp * Rp),
        reference_pressure=float(prs[-1]) / 1e5,
        planet_radius=Rp_cm,
        frequencies_to_wavelengths=True,
    )
    wl = np.array(wl_cm, dtype=float) * 1e4  # cm -> um
    de = (np.array(transit_radii, dtype=float) / Rs_cm) ** 2 * 1e6
    X = [wl, de]
    header = ''
    header += str('Wavelength/um').ljust(14, ' ') + '\t'
    header += str('None/ppm').ljust(14, ' ') + '\t'

    # loop over removing different gases
    for gas in gases:
        _, transit_radii, _ = radtrans.calculate_transit_radii(
            temperatures=np.array(tmp, dtype=float),
            mass_fractions=_copy_mass_fractions(mass_fractions, [gas]),
            mean_molar_masses=mean_molar_masses,
            reference_gravity=100.0 * cst.G * Mp / (Rp * Rp),
            reference_pressure=float(prs[-1]) / 1e5,
            planet_radius=Rp_cm,
            frequencies_to_wavelengths=True,
        )
        X.append((np.array(transit_radii, dtype=float) / Rs_cm) ** 2 * 1e6)
        header += str(f'{gas}/ppm').ljust(14, ' ') + '\t'  # compose header

    # write file
    X = np.array(X).T
    log.debug('Writing transit depth spectrum')
    np.savetxt(
        get_transit_fpath(outdir, source, 'synthesis'),
        X,
        delimiter='\t',
        fmt='%.8e',
        header=header,
        comments='',
    )

    return X


def eclipse_depth(hf_row: dict, outdir: str, config: Config, source: str):
    """
    Computes the eclipse depth spectrum using PLATON.

    Parameters
    ----------
    hf_row : dict
        The row of the helpfile for the current time-step.
    outdir : str
        The output directory for the PROTEUS run.
    config : Config
        PROTEUS config object.
    source : str
        Method for setting the mixing ratios: "outgas", "profile", or "offchem".
    """
    from petitRADTRANS.radtrans import Radtrans

    from proteus.observe.common import get_eclipse_fpath

    # All planet quantities in SI
    Rs = hf_row['R_star']  # Radius of star [m]
    Mp = hf_row['M_planet']  # Mass of planet [kg]
    Rp = hf_row['R_int']  # Radius of planet [m]
    Ts = hf_row['T_star']  # Stellar temperature
    sep = hf_row['separation']
    Rs_cm = Rs * 100.0
    Rp_cm = Rp * 100.0
    sep_cm = sep * 100.0

    # Get profile from the required source
    if source == 'offchem':
        atm = _get_atm_offchem(outdir, hf_row, config.atmos_chem.module)
    elif source in ('outgas', 'profile'):
        atm = _get_atm_profile(outdir, hf_row)

    # Parse
    if atm is None:
        log.warning(f"Could not read atmosphere data for source '{source}'")
        return None

    # Get composition from requested source
    prs, tmp, _ = _get_ptr(atm)
    (
        gases,
        mass_fractions,
        mean_molar_masses,
        line_species,
        rayleigh_species,
        gas_continuum_contributors,
    ) = _build_prt_composition(hf_row, atm, source, config.observe.platon.clip_vmr)

    # create a Radtrans object
    log.debug('Compute eclipse depth spectrum')
    radtrans = Radtrans(
        pressures=np.array(prs, dtype=float) / 1e5,
        line_species=line_species,
        gas_continuum_contributors=gas_continuum_contributors,
        rayleigh_species=rayleigh_species,
        line_opacity_mode=petitRADTRANS_LINE_OPACITY_MODE,
        path_input_data=_get_input_data_path(),
    )

    # compute full spectrum
    wl_hz, planet_flux, _ = radtrans.calculate_flux(
        temperatures=np.array(tmp, dtype=float),
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses,
        reference_gravity=100.0 * cst.G * Mp / (Rp * Rp),
        planet_radius=Rp_cm,
        star_effective_temperature=Ts,
        star_radius=Rs_cm,
        orbit_semi_major_axis=sep_cm,
        frequencies_to_wavelengths=False,
    )
    stellar_flux = Radtrans.compute_star_spectrum(
        star_effective_temperature=Ts,
        orbit_semi_major_axis=sep_cm,
        frequencies=np.array(wl_hz, dtype=float),
        star_radius=Rs_cm,
    )
    stellar_surface_flux = np.array(stellar_flux, dtype=float) * np.pi * (sep_cm / Rs_cm) ** 2
    de = np.array(planet_flux, dtype=float) / stellar_surface_flux * (Rp_cm / Rs_cm) ** 2 * 1e6
    wl = (cst.c / np.array(wl_hz, dtype=float)) * 1e4
    X = [wl, de]
    header = ''
    header += str('Wavelength/um').ljust(14, ' ') + '\t'
    header += str('None/ppm').ljust(14, ' ') + '\t'

    # loop over removing different gases
    for gas in gases:
        _, planet_flux, _ = radtrans.calculate_flux(
            temperatures=np.array(tmp, dtype=float),
            mass_fractions=_copy_mass_fractions(mass_fractions, [gas]),
            mean_molar_masses=mean_molar_masses,
            reference_gravity=100.0 * cst.G * Mp / (Rp * Rp),
            planet_radius=Rp_cm,
            star_effective_temperature=Ts,
            star_radius=Rs_cm,
            orbit_semi_major_axis=sep_cm,
            frequencies_to_wavelengths=False,
        )
        de = np.array(planet_flux, dtype=float) / stellar_surface_flux * (Rp_cm / Rs_cm) ** 2 * 1e6
        X.append(de)
        header += str(f'{gas}/ppm').ljust(14, ' ') + '\t'  # compose header

    # write file
    X = np.array(X).T
    log.debug('Writing eclipse depth spectra')
    np.savetxt(
        get_eclipse_fpath(outdir, source, 'synthesis'),
        X,
        delimiter='\t',
        fmt='%.8e',
        header=header,
        comments='',
    )

    return X
