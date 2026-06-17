# Function and classes used to run petitRADTRANS
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator

from proteus.utils.constants import (
    prt_cia_species,
    prt_gases,
    prt_ignored_gases,
    prt_rayleigh_species,
)
from proteus.utils.helper import eval_gas_mmw

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

petitRADTRANS_TLIMS = (100.0 + 0.5, 4000.0 - 0.5)
petitRADTRANS_GASES = tuple(prt_gases)
petitRADTRANS_RAYLEIGH_SPECIES = tuple(prt_rayleigh_species)
petitRADTRANS_CIA_SPECIES = tuple(prt_cia_species)
petitRADTRANS_IGNORED_GASES = set(prt_ignored_gases)


def _get_input_data_path(input_data_path: str | None) -> str:
    """Return a local petitRADTRANS input-data directory."""

    if input_data_path is not None:
        return input_data_path

    import petitRADTRANS

    package_dir = Path(petitRADTRANS.__file__).resolve().parent.parent
    candidate = package_dir / 'input_data'
    if candidate.is_dir():
        return str(candidate)

    raise FileNotFoundError(
        'Could not locate a petitRADTRANS input_data directory next to the installed '
        'petitRADTRANS package.'
    )


def _get_supported_line_species(gases: list[str], input_data_path: str | None) -> list[str]:
    """Return gases that have line-opacity tables in the configured pRT data tree."""

    if input_data_path is None:
        input_data_path = _get_input_data_path(None)

    lines_root = Path(input_data_path) / 'opacities' / 'lines' / 'correlated_k'
    supported = []
    for gas in gases:
        if gas in petitRADTRANS_IGNORED_GASES or gas in petitRADTRANS_RAYLEIGH_SPECIES:
            continue
        if (lines_root / gas).is_dir():
            supported.append(gas)
        else:
            log.debug("Skipping gas '%s' because no line-opacity directory exists", gas)
    return supported


def _get_supported_rayleigh_species(gases: list[str], include_rayleigh: bool) -> list[str]:
    if not include_rayleigh:
        return []
    return [gas for gas in gases if gas in petitRADTRANS_RAYLEIGH_SPECIES]


def _get_supported_cia_species(gases: list[str], include_cia: bool) -> list[str]:
    if not include_cia:
        return []

    available = []
    for cia_species in petitRADTRANS_CIA_SPECIES:
        components = [c for c in cia_species.split('--') if c]
        if all(component in gases for component in components):
            available.append(cia_species)
        else:
            log.debug(
                "Skipping CIA species '%s' because not all components are present in the gas mix",
                cia_species,
            )
    return available


def _vmrs_to_mass_fractions(
    gases: list[str], vmrs: list[np.ndarray]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Convert number fractions to pRT mass fractions and mean molar masses."""

    if len(vmrs) == 0:
        return {}, np.array([])

    vmr_arr = np.array([np.array(vmr, dtype=float) for vmr in vmrs], dtype=float)

    # Normalize VMRs to sum to 1 (after clip_vmr they may not sum to 1)
    vmr_sum = np.sum(vmr_arr, axis=0)
    if np.any(vmr_sum <= 0.0):
        raise ValueError(
            'Atmospheric composition has zero total VMR at one or more layers'
        )
    vmr_arr = vmr_arr / vmr_sum

    molar_masses = np.array([eval_gas_mmw(gas) for gas in gases], dtype=float)  # kg mol-1
    mass_contrib = vmr_arr * molar_masses[:, None]
    total_mass = np.sum(mass_contrib, axis=0)

    if np.any(total_mass <= 0.0):
        raise ValueError(
            'Atmospheric composition has zero total mass fraction at one or more layers'
        )

    mass_fractions = {
        gas: mass_contrib[i] / total_mass
        for i, gas in enumerate(gases)
        if gas not in petitRADTRANS_IGNORED_GASES
    }
    mean_molar_masses = total_mass / 1.0e-3  # pRT expects amu (g mol-1)
    return mass_fractions, mean_molar_masses


def _build_prt_composition(
    hf_row: dict,
    atm: dict,
    source: str,
    clip_vmr: float,
    include_rayleigh: bool,
    include_cia: bool,
    input_data_path: str | None,
) -> tuple:
    gases, vmrs = _get_mix(hf_row, atm, source, clip_vmr)
    line_species = _get_supported_line_species(gases, input_data_path)
    rayleigh_species = _get_supported_rayleigh_species(gases, include_rayleigh)
    gas_continuum_contributors = _get_supported_cia_species(gases, include_cia)
    return (
        gases,
        vmrs,
        line_species,
        rayleigh_species,
        gas_continuum_contributors,
    )


def _interpolate_prt_profiles(
    prs: np.ndarray, tmp: np.ndarray, vmrs: list[np.ndarray], n_points: int = 100
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Interpolate temperature and mixing ratios onto a log-spaced pressure grid."""

    log_prs = np.log10(np.asarray(prs, dtype=float))
    pressure_grid = np.logspace(log_prs.min(), log_prs.max(), n_points)
    log_pressure_grid = np.log10(pressure_grid)

    temp_grid = PchipInterpolator(log_prs, np.asarray(tmp, dtype=float))(log_pressure_grid)
    temp_grid = np.clip(
        temp_grid,
        a_min=petitRADTRANS_TLIMS[0],
        a_max=petitRADTRANS_TLIMS[1],
    )

    vmr_grid = []
    for vmr in vmrs:
        interp = PchipInterpolator(log_prs, np.asarray(vmr, dtype=float))(log_pressure_grid)
        vmr_grid.append(np.clip(interp, a_min=0.0, a_max=None))

    return pressure_grid, temp_grid, vmr_grid


def _get_atm_profile(outdir: str, hf_row: dict) -> dict:
    """
    Reads the atmosphere data from the NetCDF file produced by the 'atmos_clim' module.
    """
    from proteus.atmos_clim.common import read_atmosphere_data

    atm_arr = read_atmosphere_data(
        outdir, [hf_row['Time']], extra_keys=['tmpl', 'pl', 'rl', 'g', 'x_gas']
    )

    if (len(atm_arr) == 0) or (atm_arr[-1] is None):
        log.warning(f"Could not read atmosphere data from '{outdir}'")
        return None
    return atm_arr[-1]


def _get_reference_prt_values(atm: dict, config: Config) -> tuple[float, float, float]:
    """Return the reference pressure [bar], radius [cm], and gravity [cgs] from the profile.

    Uses the reference pressure from the config to find the closest matching layer.
    """
    prs = np.asarray(atm['p'], dtype=float)  # Pa
    config_ref_pressure_pa = config.observe.reference_pressure * 1e5  # Convert bar to Pa

    # Find the index with closest pressure to config reference
    ref_idx = np.argmin(np.abs(prs - config_ref_pressure_pa))

    reference_pressure = float(prs[ref_idx] / 1e5)
    reference_radius = float(np.asarray(atm['r'], dtype=float)[ref_idx] * 100.0)
    reference_gravity = float(np.asarray(atm['g'], dtype=float)[ref_idx] * 100.0)
    return reference_pressure, reference_radius, reference_gravity


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
    df['rl'] = df['rl'] + hf_row['R_int']  # convert height to radius
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


def _get_ptr(
    atm: dict, vmrs: list[np.ndarray] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray] | None]:
    """
    Returns the pressure, temperature and radius from the atmosphere data."""
    prs = np.array(atm['pl'])  # Pa
    tmp = np.array(atm['tmpl'])  # K
    rad = np.array(atm['rl'])  # m
    vmrs_sorted = vmrs
    if prs.size > 1 and prs[1] < prs[0]:
        order = slice(None, None, -1)
        prs = prs[::-1]
        tmp = np.clip(tmp[order], a_min=petitRADTRANS_TLIMS[0], a_max=petitRADTRANS_TLIMS[1])
        rad = rad[::-1]
        if vmrs is not None:
            vmrs_sorted = [np.array(vmr, dtype=float)[order] for vmr in vmrs]
    return prs, tmp, rad, vmrs_sorted


def _load_stellar_toa_flux(
    outdir: str, hf_row: dict, target_wavelength_nm: np.ndarray
) -> np.ndarray:
    """Load the PROTEUS-written stellar spectrum from ``data/<Time>.sflux``.

    The spectrum is already scaled to the top of the planet atmosphere, so it
    can be used directly in the eclipse-depth denominator.
    """

    spectrum_path = Path(outdir) / 'data' / f"{int(hf_row['Time'])}.sflux"
    if not spectrum_path.is_file():
        raise FileNotFoundError(f"Stellar spectrum file '{spectrum_path}' not found.")

    star_data = np.loadtxt(spectrum_path, skiprows=1).T
    stellar_wavelength_nm = np.array(star_data[0], dtype=float)
    stellar_flux = np.array(star_data[1], dtype=float)
    stellar_flux = np.interp(target_wavelength_nm, stellar_wavelength_nm, stellar_flux)
    return stellar_flux * 1.0e7  # erg cm-2 s-1 nm-1 -> erg cm-2 s-1 cm-1


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
    Rs_cm = Rs * 100.0

    # Get profile from the required source
    if source == 'offchem':
        atm = _get_atm_offchem(outdir, hf_row, config.atmos_chem.module)
    elif source in ('outgas', 'profile'):
        atm = _get_atm_profile(outdir, hf_row)

    atm_reference = _get_atm_profile(outdir, hf_row)

    include_rayleigh = config.observe.petitRADTRANS.include_rayleigh
    include_cia = config.observe.petitRADTRANS.include_cia
    include_vmr_clipping = config.observe.clip_vmr
    input_data_path = _get_input_data_path(config.observe.petitRADTRANS.input_data_path)

    # Parse
    if (atm is None) or (atm_reference is None):
        log.warning(f"Could not read atmosphere data for source '{source}'")
        return None

    reference_pressure, reference_radius, reference_gravity = _get_reference_prt_values(
        atm_reference, config
    )

    # Get composition from requested source
    (
        gases,
        vmrs,
        line_species,
        rayleigh_species,
        gas_continuum_contributors,
    ) = _build_prt_composition(
        hf_row,
        atm,
        source,
        include_vmr_clipping,
        include_rayleigh,
        include_cia,
        input_data_path,
    )

    prs, tmp, _, vmrs = _get_ptr(atm, vmrs)
    prs, tmp, vmrs = _interpolate_prt_profiles(prs, tmp, vmrs, n_points=100)
    mass_fractions, mean_molar_masses = _vmrs_to_mass_fractions(gases, vmrs)

    # create a Radtrans object
    log.debug('Compute transit depth spectra')
    radtrans = Radtrans(
        pressures=np.array(prs, dtype=float) / 1e5,
        line_species=line_species,
        gas_continuum_contributors=gas_continuum_contributors,
        rayleigh_species=rayleigh_species,
        line_opacity_mode=config.observe.petitRADTRANS.line_opacity_mode,
        path_input_data=input_data_path,
    )

    # compute full spectrum
    wl_cm, transit_radii, _ = radtrans.calculate_transit_radii(
        temperatures=np.array(tmp, dtype=float),
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses,
        reference_gravity=reference_gravity,
        reference_pressure=reference_pressure,
        planet_radius=reference_radius,
        frequencies_to_wavelengths=True,
    )

    wl = np.array(wl_cm, dtype=float) * 1e4  # cm -> um
    de = (np.array(transit_radii, dtype=float) / Rs_cm) ** 2 * 1e6
    X = [wl, de]
    header = ''
    header += str('Wavelength/um').ljust(14, ' ') + '\t'
    header += str('None/ppm').ljust(14, ' ') + '\t'

    # write file
    X = np.array(X).T
    log.debug('Writing transit depth spectrum')
    np.savetxt(
        get_transit_fpath(outdir, source, 'observe'),
        X,
        delimiter='\t',
        fmt='%.8e',
        header=header,
        comments='',
    )

    return X


def eclipse_depth(hf_row: dict, outdir: str, config: Config, source: str):
    """
    Computes the eclipse depth spectrum using petitRADTRANS.

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
    Ts = hf_row['T_star']  # Stellar temperature
    sep = hf_row['separation']
    Rs_cm = Rs * 100.0
    sep_cm = sep * 100.0

    include_rayleigh = config.observe.petitRADTRANS.include_rayleigh
    include_cia = config.observe.petitRADTRANS.include_cia
    include_vmr_clipping = config.observe.clip_vmr
    input_data_path = _get_input_data_path(config.observe.petitRADTRANS.input_data_path)

    # Get profile from the required source
    if source == 'offchem':
        atm = _get_atm_offchem(outdir, hf_row, config.atmos_chem.module)
    elif source in ('outgas', 'profile'):
        atm = _get_atm_profile(outdir, hf_row)

    atm_reference = _get_atm_profile(outdir, hf_row)

    # Parse
    if (atm is None) or (atm_reference is None):
        log.warning(f"Could not read atmosphere data for source '{source}'")
        return None

    reference_pressure, reference_radius, reference_gravity = _get_reference_prt_values(
        atm_reference, config
    )

    # Get composition from requested source
    (
        gases,
        vmrs,
        line_species,
        rayleigh_species,
        gas_continuum_contributors,
    ) = _build_prt_composition(
        hf_row,
        atm,
        source,
        include_vmr_clipping,
        include_rayleigh,
        include_cia,
        input_data_path,
    )

    prs, tmp, _, vmrs = _get_ptr(atm, vmrs)
    prs, tmp, vmrs = _interpolate_prt_profiles(prs, tmp, vmrs, n_points=100)
    mass_fractions, mean_molar_masses = _vmrs_to_mass_fractions(gases, vmrs)

    # create a Radtrans object
    log.debug('Compute eclipse depth spectrum')
    radtrans = Radtrans(
        pressures=np.array(prs, dtype=float) / 1e5,
        line_species=line_species,
        gas_continuum_contributors=gas_continuum_contributors,
        rayleigh_species=rayleigh_species,
        line_opacity_mode=config.observe.petitRADTRANS.line_opacity_mode,
        path_input_data=input_data_path,
    )

    # compute full spectrum
    wl_cm, planet_flux, _ = radtrans.calculate_flux(
        temperatures=np.array(tmp, dtype=float),
        mass_fractions=mass_fractions,
        mean_molar_masses=mean_molar_masses,
        reference_gravity=reference_gravity,
        planet_radius=reference_radius,
        star_effective_temperature=Ts,
        star_radius=Rs_cm,
        orbit_semi_major_axis=sep_cm,
        frequencies_to_wavelengths=True,
    )
    stellar_wavelength_nm = wl_cm * 1.0e7
    stellar_surface_flux = _load_stellar_toa_flux(outdir, hf_row, stellar_wavelength_nm)
    de = (
        np.array(planet_flux, dtype=float)
        / stellar_surface_flux
        * (reference_radius / Rs_cm) ** 2
        * 1e6
    )
    wl = (wl_cm * 1e4)
    X = [wl, de]
    header = ''
    header += str('Wavelength/um').ljust(14, ' ') + '\t'
    header += str('None/ppm').ljust(14, ' ') + '\t'

    # write file
    X = np.array(X).T
    log.debug('Writing eclipse depth spectra')
    np.savetxt(
        get_eclipse_fpath(outdir, source, 'observe'),
        X,
        delimiter='\t',
        fmt='%.8e',
        header=header,
        comments='',
    )

    return X
