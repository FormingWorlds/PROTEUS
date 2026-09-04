# Obliqua tidal heating module
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import juliacall
import netCDF4 as nc
import numpy as np
from juliacall import Main as jl
from scipy.interpolate import interp1d

from proteus.interior_energetics.common import Interior_t
from proteus.orbit.common import Tides_t
from proteus.utils.helper import UpdateStatusfile
from proteus.utils.logs import GetCurrentLogfileIndex, GetLogfilePath

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

Obliqua_LOGFILE_NAME = 'obliqua_recent.log'


def import_obliqua():
    log.debug('Import Obliqua...')
    jl.seval('using Obliqua')


def to_julia_dict(obj):
    """Recursively convert Python dict/list to native Julia Dict/Vector."""
    if isinstance(obj, dict):
        jd = jl.Dict()
        for k, v in obj.items():
            jd[k] = to_julia_dict(v)
        return jd
    elif isinstance(obj, list):
        return [to_julia_dict(v) for v in obj]
    else:
        return obj


def _jlarr(arr: np.ndarray):
    # Make copy of array, and convert to Julia type
    cop = np.array(arr, copy=True, dtype=float).flatten()
    return juliacall.convert(jl.Array[jl.Obliqua.prec, 1], cop)


def _jlsca_float(sca: float):
    # Make a copy of a scalar, and convert to Julia type
    return juliacall.convert(jl.Obliqua.Float64, sca)


def _jlsca_prec(sca: float):
    # Make a copy of a scalar, and convert to Julia type
    return juliacall.convert(jl.Obliqua.prec, sca)


def run_obliqua(
    hf_row: dict, dirs: dict, interior_o: Interior_t, tides_o: Tides_t, config: Config
) -> float:
    """Run the Obliqua tidal heating module.

    Sets the interior tidal heating and returns k-love number. All tidal love-numbers
    are stored in the tides_o object for the specific perturber.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        dirs: dict
            Dictionary of directories.
        interior_o: Interior_t
            Struct containing interior arrays at current time.
        tides_o: Tides_t
            Struct containing tidal arrays at current time.
        config: Config
            PROTEUS config object
    Returns
    ----------
        Imk: float
            Averaged imaginary part of the k love numbers.
    """

    # Calculate axial frequency of rotation
    axial = _jlsca_prec(2 * np.pi / hf_row['axial_period'])

    if config.orbit.perturber == 'star':
        log.debug('Running Obliqua for star-planet tides...')

        # Calculate orbital frequency of rotation
        omega = _jlsca_prec(2 * np.pi / hf_row['orbital_period'])

        # Convert planet-star orbital eccentricity, semi-major axis, and mass
        ecc = _jlsca_float(hf_row['eccentricity'])
        sma = _jlsca_float(hf_row['semimajorax'])
        M_pert = _jlsca_float(hf_row['M_star'])

    elif config.orbit.perturber == 'satellite':
        log.debug('Running Obliqua for satellite-planet tides...')

        # Calculate orbital frequency of rotation
        omega = _jlsca_prec(2 * np.pi / hf_row['orbital_period_sat'])

        # Convert planet-satellite orbital eccentricity, semi-major axis, and mass
        ecc = _jlsca_float(hf_row['eccentricity_sat'])
        sma = _jlsca_float(hf_row['semimajorax_sat'])
        M_pert = _jlsca_float(hf_row['M_sat'])

    # Copy arrays
    arr_keys = ('density', 'visc', 'shear', 'bulk', 'phi', 'mass', 'radius')
    lov = {k: np.array(getattr(interior_o, k), copy=True, dtype=float) for k in arr_keys}

    # Reverse arrays if using SPIDER
    #  Such that i=0 is at the CMB
    if config.interior_energetics.module == 'spider':
        for k in arr_keys:
            lov[k] = lov[k][::-1]

    if config.interior_energetics.module == 'dummy':
        # Construct arrays for obliqua (we need two cells, three edges here)
        for k in arr_keys:
            if k == 'radius':
                rmid = np.median(lov['radius'])
                arr = [lov['radius'][0], rmid, lov['radius'][1]]
                lov[k] = _jlarr(arr[:])
            else:
                arr = [lov[k][0], lov[k][0]]
                lov[k] = _jlarr(arr[:])

    else:
        # for spider/aragog
        # Construct arrays for obliqua using the full interior profile.
        # Obliqua's solid/mushy/fluid multi-phase model resolves the
        # liquid/solid transition internally, so no viscosity-based
        # cutoff is needed here (unlike LovePy's single-phase model).
        n_lev = interior_o.nlev_s
        for k in arr_keys:
            if k == 'radius':
                i = n_lev + 1
            else:
                i = n_lev
            lov[k] = _jlarr(lov[k][:i])

    # Create configuration dictionary for Obliqua
    cfg = {
        'title': 'PROTEUS_run_' + str(round(hf_row['Time'])),
        'params': {
            'out': {
                'path': dirs['output/data'],
                'time': round(hf_row['Time']),
            },
        },
        'orbit': {
            'obliqua': {
                'store_3D': config.orbit.obliqua.store_3D,
                'enforce_ec': config.orbit.obliqua.enforce_ec,
                'optimize_scales': config.orbit.obliqua.optimize_scales,
                'solid_shell': config.orbit.obliqua.solid_shell,
                'min_frac': config.orbit.obliqua.min_frac,
                'visc_l': config.orbit.obliqua.visc_l,
                'visc_lus': config.orbit.obliqua.visc_lus,
                'visc_s': config.orbit.obliqua.visc_s,
                'visc_sus': config.orbit.obliqua.visc_sus,
                'n': config.orbit.obliqua.n,
                'm': config.orbit.obliqua.m,
                'spectrum': 'adaptive',  # Note that this is fixed, since spectrum = full is not useful for the current implementation of Obliqua in PROTEUS.
                's_min': config.orbit.obliqua.k_min,
                's_max': config.orbit.obliqua.k_max,
                'material_mu': config.orbit.obliqua.material_mu,
                'material_k': config.orbit.obliqua.material_k,
                'alpha': config.orbit.obliqua.alpha,
                'module_solid': config.orbit.obliqua.module_solid,
                'module_mushy': config.orbit.obliqua.module_mushy,
                'module_fluid': config.orbit.obliqua.module_fluid,
                'solid': {
                    'ncalc': config.orbit.obliqua.solid.ncalc,
                    'dr_min': config.orbit.obliqua.solid.dr_min,
                    'dr_max': config.orbit.obliqua.solid.dr_max,
                    'core': config.orbit.obliqua.solid.core,
                    'core_props': config.orbit.obliqua.solid.core_props,
                    'inertial_terms': config.orbit.obliqua.solid.inertial_terms,
                    'bulk_l': config.orbit.obliqua.solid.bulk_l,
                    'porosity_thresh': config.orbit.obliqua.solid.porosity_thresh,
                    'dbulk_power': config.orbit.obliqua.solid.dbulk_power,
                },
                'mushy': {
                    'b_width': config.orbit.obliqua.mushy.b_width,
                    't_width': config.orbit.obliqua.mushy.t_width,
                },
                'fluid': {
                    'sigma_R': config.orbit.obliqua.fluid.sigma_R,
                    'sigma_R_inf': config.orbit.obliqua.fluid.sigma_R_inf
                    * config.orbit.obliqua.fluid.sigma_R,
                    'sigma_R_prf': config.orbit.obliqua.fluid.sigma_R_prf,
                    'H_R': config.orbit.obliqua.fluid.H_R,
                    'efficiency': config.orbit.obliqua.fluid.efficiency,
                },
            }
        },
        'interior_energetics': {
            'grain_size': config.interior_energetics.grain_size,
        },
        'struct': {
            'core_density': config.interior_energetics.boundary.core_density,
            'core_shear': config.interior_energetics.boundary.core_shear,
            'core_bulk': config.interior_energetics.boundary.core_bulk,
        },
    }

    cfg = to_julia_dict(cfg)

    # Calculate heating using obliqua
    try:
        # Extract arrays
        rho = lov['density']
        radius = lov['radius']
        visc = lov['visc']
        shear = lov['shear']
        bulk = lov['bulk']
        phi = lov['phi']

        # Add permeability and drained bulk modulus and limit porosity
        perm = jl.Obliqua.interior.get_permeability(phi, cfg)
        perm, phi = jl.Obliqua.interior.limit_porosity(perm, phi, cfg)
        bulkd = jl.Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

        # Run Obliqua to get tidal heating profile and love number
        power_prf, power_blk, nmk, sigma, LNk = jl.Obliqua.run_tides(
            omega,
            axial,
            ecc,
            sma,
            M_pert,
            rho,
            radius,
            visc,
            shear,
            bulk,
            bulkd,
            phi,
            perm,
            cfg,
        )

    except juliacall.JuliaError as e:
        UpdateStatusfile(dirs, 26)
        log.error(e)
        raise RuntimeError('Encountered problem when running Obliqua module')

    if config.interior_energetics.module == 'dummy':
        interior_o.tides[0] = power_prf[1]

    else:
        # Store result, flipping for SPIDER
        if config.interior_energetics.module == 'spider':
            interior_o.tides[:] = power_prf[::-1]
        else:
            interior_o.tides[:] = power_prf[:]

        # Verify result against bulk calculation
        power_blk /= np.sum(lov['mass'])
        log.debug('    power from bulk calc: %.3e W kg-1' % power_blk)

    # Store results in tides_o structure
    storage = tides_o.add(primary='planet', perturber=config.orbit.perturber)
    storage.nmk = np.vstack(nmk).astype(int)
    storage.sigma = sigma
    storage.LNk = LNk

    # Logging
    sync_log_files(dirs['output'])

    return np.mean(np.imag(LNk))


def lookup_from_interior(dirs: dict, config: Config):
    """Run the Obliqua tidal heating module for a simple 0-D interior model.

    Constructs the full k-love number spectrum lookup table.

    Parameters
    ----------
        dirs: dict
            Dictionary of directories.
        config: Config
            PROTEUS config object
    """

    log.info('Running Obliqua for lookup table generation...')

    # Read interior arrays from json file
    file_path = config.orbit.satellite.love_number_sat

    if not file_path:
        raise ValueError(
            'Satellite tidal data file path (`config.orbit.satellite.love_number_sat`) is not specified.'
        )

    # read interior arrays from json file
    with open(file_path) as f:
        d = json.load(f)

        # Convert orbital parameters to Julia types (not used for lookup table generation, but required by Obliqua)
        omega = _jlsca_prec(d['omega'])
        axial = _jlsca_prec(d['axial'])
        ecc = _jlsca_float(d['ecc'])
        sma = _jlsca_float(d['sma'])
        M_pert = _jlsca_float(d['S_mass'])

        # Extract the iron core density from interior density profile, and convert the rest to a Julia array
        density_full = np.array(d['density'], copy=True, dtype=float)
        core_density = density_full[0]
        rho = _jlarr(density_full[1:])

        radius = _jlarr(np.array(d['radius'], copy=True, dtype=float))
        visc = _jlarr(np.array(d['visc'], copy=True, dtype=float))
        shear = _jlarr(np.array(d['shear'], copy=True, dtype=float))
        bulk = _jlarr(np.array(d['bulk'], copy=True, dtype=float))
        phi = _jlarr(np.array(d['phi'], copy=True, dtype=float))

    # Create configuration dictionary for Obliqua
    cfg = {
        'title': 'Lookup_table',
        'params': {
            'out': {
                'path': dirs['output/data'],
                'time': round(0.0),  # store the Obliqua output file at t=0 to avoid
            },  # overwriting the main PROTEUS+Obliqua output file
        },
        'orbit': {
            'obliqua': {
                'store_3D': False,
                'enforce_ec': False,
                'optimize_scales': False,
                'solid_shell': False,
                'min_frac': config.orbit.obliqua.min_frac,
                'visc_l': config.orbit.obliqua.visc_l,
                'visc_lus': config.orbit.obliqua.visc_lus,
                'visc_s': config.orbit.obliqua.visc_s,
                'visc_sus': config.orbit.obliqua.visc_sus,
                'n': config.orbit.obliqua.n,
                'm': config.orbit.obliqua.m,
                'spectrum': 'full',  # full for lookup table generation
                'N_sigma': 100,  # number of frequency points for the lookup table
                'p_min': -8,  # Minimum period for orbital and axial frequencies [log(kyr)]
                'p_max': 4,  # Maximum period for orbital and axial frequencies [log(kyr)]
                'k_min': 'none',  # not used for lookup table generation, k = 1 for all samples
                'k_max': 'none',  # not used for lookup table generation, k = 1 for all samples
                'material_mu': config.orbit.obliqua.material_mu,
                'material_k': config.orbit.obliqua.material_k,
                'alpha': config.orbit.obliqua.alpha,
                # use 0-D modules for lookup table generation, since we are only interested in
                # the love number spectrum for an unresolved interior structure.
                'module_solid': 'solid0d',
                'module_mushy': 'none',
                'module_fluid': 'fluid0d',
                # Not used:
                'solid': {
                    'ncalc': config.orbit.obliqua.solid.ncalc,
                    'dr_min': config.orbit.obliqua.solid.dr_min,
                    'dr_max': config.orbit.obliqua.solid.dr_max,
                    'core': config.orbit.obliqua.solid.core,
                    'core_props': config.orbit.obliqua.solid.core_props,
                    'inertial_terms': config.orbit.obliqua.solid.inertial_terms,
                    'bulk_l': config.orbit.obliqua.solid.bulk_l,
                    'porosity_thresh': config.orbit.obliqua.solid.porosity_thresh,
                    'dbulk_power': config.orbit.obliqua.solid.dbulk_power,
                },
                # Not used:
                'mushy': {
                    'b_width': config.orbit.obliqua.mushy.b_width,
                    't_width': config.orbit.obliqua.mushy.t_width,
                },
                # Used: fix Rayleigh drag to the same value as in the main PROTEUS run.
                'fluid': {
                    'sigma_R': config.orbit.obliqua.fluid.sigma_R,  # used
                    'sigma_R_inf': config.orbit.obliqua.fluid.sigma_R_inf
                    * config.orbit.obliqua.fluid.sigma_R,  # used
                    'sigma_R_prf': config.orbit.obliqua.fluid.sigma_R_prf,  # not used
                    'H_R': config.orbit.obliqua.fluid.H_R,  # not used
                    'efficiency': config.orbit.obliqua.fluid.efficiency,  # used
                },
            }
        },
        # Not used:
        'interior_energetics': {
            'grain_size': config.interior_energetics.grain_size,
        },
        # Used:
        'struct': {
            'core_density': core_density,  # used for density contrast in the fluid0d module
            'core_shear': config.interior_energetics.boundary.core_shear,  # not used
            'core_bulk': config.interior_energetics.boundary.core_bulk,  # not used
        },
    }

    cfg = to_julia_dict(cfg)

    # Calculate heating using obliqua
    try:
        # Add permeability and drained bulk modulus and limit porosity
        perm = jl.Obliqua.interior.get_permeability(phi, cfg)
        perm, phi = jl.Obliqua.interior.limit_porosity(perm, phi, cfg)
        bulkd = jl.Obliqua.interior.get_drained_bulk(bulk, phi, cfg)

        # Run Obliqua to get tidal heating profile and love number
        power_prf, power_blk, nmk, sigma, LNk = jl.Obliqua.run_tides(
            omega,
            axial,
            ecc,
            sma,
            M_pert,
            rho,
            radius,
            visc,
            shear,
            bulk,
            bulkd,
            phi,
            perm,
            cfg,
        )

    except juliacall.JuliaError as e:
        UpdateStatusfile(dirs, 26)
        log.error(e)
        raise RuntimeError('Encountered problem when running Obliqua module')

    # Store lookup table in netcdf file
    nc_path = os.path.join(dirs['output/data'], 'moon_tides.nc')
    with nc.Dataset(nc_path, 'w', format='NETCDF4') as ds:
        N = len(nmk)

        ds.createDimension('mode', N)

        # Ensure elements are extracted cleanly from Julia objects to standard Python/NumPy types
        nmk_rows = [[int(mode[0]), int(mode[1]), int(mode[2])] for mode in nmk]
        nmk_np = np.array(nmk_rows, dtype=np.int64)

        n_vals = nmk_np[:, 0]
        m_vals = nmk_np[:, 1]
        k_vals = nmk_np[:, 2]

        v_n = ds.createVariable('n', 'i4', ('mode',))
        v_m = ds.createVariable('m', 'i4', ('mode',))
        v_k = ds.createVariable('k', 'i4', ('mode',))
        v_sigma = ds.createVariable('sigma', 'f8', ('mode',))
        v_LNk_r = ds.createVariable('LNk_real', 'f8', ('mode',))
        v_LNk_i = ds.createVariable('LNk_imag', 'f8', ('mode',))

        v_n[:] = n_vals
        v_m[:] = m_vals
        v_k[:] = k_vals
        v_sigma[:] = np.array(sigma, dtype=np.float64)
        v_LNk_r[:] = np.real(np.array(LNk, dtype=np.complex128))
        v_LNk_i[:] = np.imag(np.array(LNk, dtype=np.complex128))


def LN_from_lookup(hf_row: dict, dirs: dict, tides_o: Tides_t, config: Config):
    """Extract and populate Love numbers for a satellite from a pre-generated lookup table.

    Since Hansen coefficients dictate which modes are relevant for tidal forcing and depend
    only on orbital eccentricity, this function interpolates/matches the Love numbers (LNk)
    for relevant modes from an Obliqua lookup file spanning a broad frequency range.

    Parameters
    ----------
        hf_row : dict
            Dictionary of current runtime variables
        dirs: dict
            Dictionary of directories.
        tides_o: Tides_t
            Struct containing tidal arrays at current time.
        config: Config
            PROTEUS config object
    """
    # Fetch relevant modes from planet-side forcing
    nmk_p = np.asarray(tides_o.get(primary='planet', perturber='satellite').nmk)

    # Vectorized calculation of forcing frequencies (sigma_s = m * omega_rot - k * omega_orb)
    axial_freq_s = 2.0 * np.pi / hf_row['axial_period_sat']
    orbit_freq_s = 2.0 * np.pi / hf_row['orbital_period_sat']

    # nmk_p[:, 1] is 'm', nmk_p[:, 2] is 'k'
    sigma_s = nmk_p[:, 1] * axial_freq_s - nmk_p[:, 2] * orbit_freq_s

    # Retrieve or load satellite lookup data
    try:
        lookup = tides_o.get(primary='satellite_dict', perturber='planet')
    except KeyError:
        file_path = config.orbit.satellite.love_number_sat

        # Check if the file path is specified
        if not file_path:
            raise ValueError(
                'Satellite tidal data file path (`config.orbit.satellite.love_number_sat`) is not specified.'
            )

        # check the file extension to determine how to read the lookup data or generate it if it doesn't exist
        if file_path.endswith('.nc'):
            # Read the netcdf file
            pass
        elif file_path.endswith('.json'):
            # Generate the lookup data
            lookup_from_interior(dirs, config)
            # Update the file path to point to the generated netcdf file
            file_path = os.path.join(dirs['output/data'], 'moon_tides.nc')
            pass

        tides_o.add_from_file(primary='satellite_dict', perturber='planet', file_path=file_path)

        lookup = tides_o.get(primary='satellite_dict', perturber='planet')

    # Extract lookup arrays
    sigma_lookup = np.asarray(lookup.sigma)
    nmk_lookup = np.asarray(lookup.nmk)
    LNk_lookup = np.asarray(lookup.LNk)

    # Interpolate Love numbers degree-by-degree
    LNk_s = np.zeros(len(nmk_p), dtype=complex)

    for n in np.unique(nmk_p[:, 0]):
        # Lookup entries for this degree
        lookup_mask = nmk_lookup[:, 0] == n

        if not np.any(lookup_mask):
            raise ValueError(f'Lookup table does not contain degree n = {int(n)}.')

        sigma_n = sigma_lookup[lookup_mask]
        LNk_n = LNk_lookup[lookup_mask]

        # Create symmetric lookup
        sigma_sym = np.concatenate((-sigma_n[::-1], sigma_n))
        LNk_sym = np.concatenate((np.conjugate(LNk_n[::-1]), LNk_n))

        # Sort for interpolation
        order = np.argsort(sigma_sym)
        sigma_sym = sigma_sym[order]
        LNk_sym = LNk_sym[order]

        interp_real = interp1d(
            sigma_sym,
            LNk_sym.real,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate',
        )

        interp_imag = interp1d(
            sigma_sym,
            LNk_sym.imag,
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate',
        )

        # Modes requiring this degree
        mode_mask = nmk_p[:, 0] == n

        sigma_eval = sigma_s[mode_mask]

        LNk_s[mode_mask] = interp_real(sigma_eval) + 1j * interp_imag(sigma_eval)

    # Enforce normalization
    zero_mask = (nmk_p[:, 1] < 0) & (nmk_p[:, 2] < 0)
    LNk_s[zero_mask] = 0.0 + 0.0j

    # Store results
    storage = tides_o.add(primary='satellite', perturber='planet')
    storage.nmk = nmk_p
    storage.sigma = sigma_s
    storage.LNk = LNk_s


def read_ncdf(fpath: str):
    out = {}
    ds = nc.Dataset(fpath)

    for key in ds.variables.keys():
        out[key] = ds.variables[key][:]

    ds.close()
    return out


def read_ncdfs(output_dir: str, times: list):
    return [read_ncdf(os.path.join(output_dir, 'data', '%d_obliqua.nc' % t)) for t in times]


def setup_logging(dirs: dict, verbosity: int):
    # Setup logging from Obliqua
    #    This handle will be kept open throughout the PROTEUS simulation, so the file
    #    should not be deleted at runtime. However, it will be emptied when appropriate.
    logpath = os.path.join(dirs['output'], Obliqua_LOGFILE_NAME)
    jl.Obliqua.setup_logging(logpath, verbosity)

    log.debug("Obliqua will log to '%s'" % logpath)


def sync_log_files(outdir: str) -> list[str]:
    """Move Obliqua logfile content into the PROTEUS logfile and clear it.

    Returns the list of lines that were copied, so that callers can scan
    them for failure-mode markers.
    Returns an empty list if the Obliqua logfile cannot be read.
    """
    # Logfile paths
    obliqua_logpath = os.path.join(outdir, Obliqua_LOGFILE_NAME)
    logpath = GetLogfilePath(outdir, GetCurrentLogfileIndex(outdir))

    # Copy logfile content
    try:
        with open(obliqua_logpath, 'r') as infile:
            inlines = infile.readlines()
    except OSError:
        return []

    with open(logpath, 'a') as outfile:
        for i, line in enumerate(inlines):
            # First line of obliqua logfile has NULL chars at the start, for some reason
            if i == 0 and '[' in line:
                line = '[' + line.split('[', 1)[1]
            # copy the line
            outfile.write(line)

    # Remove logfile content
    with open(obliqua_logpath, 'w') as hdl:
        hdl.write('')

    return inlines
