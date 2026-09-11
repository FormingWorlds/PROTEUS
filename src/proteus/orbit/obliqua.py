# Obliqua tidal heating module
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

import juliacall
import netCDF4 as nc
import numpy as np
from attrs import asdict
from juliacall import Main as jl
from scipy.interpolate import interp1d

from proteus.interior_energetics.common import Interior_t
from proteus.orbit.common import Tides_t
from proteus.orbit.hansen import padded_k_range_for_evection
from proteus.utils.helper import UpdateStatusfile
from proteus.utils.julia_common import make_julia_converters, make_log_syncer, to_julia_dict

# Obliqua-precision-bound converters
_jlarr, _jlsca_float, _jlsca_prec = make_julia_converters('Obliqua')

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

Obliqua_LOGFILE_NAME = 'obliqua_recent.log'


def import_obliqua():
    log.debug('Import Obliqua...')
    jl.seval('using Obliqua')


def _padded_obliqua_k_range(
    hf_row: dict, interior_o: Interior_t, tides_o: Tides_t, config: Config
) -> tuple:
    """(s_min, s_max) to pass to Obliqua's 'adaptive' spectrum for the
    satellite-perturber (evection) case.

    An explicit user-supplied ``k_min``/``k_max`` (an int, not ``'none'``)
    is only ever WIDENED by the padding, never narrowed past what the user
    configured.
    """
    k_min_cfg = config.orbit.obliqua.k_min
    k_max_cfg = config.orbit.obliqua.k_max

    # In/near evection band, as judged by evolve_orbit_satellite at the end
    # of its own last call -- internal-only orbit state (see Tides_t's own
    # docstring), not exported to hf_row.
    if not tides_o.evection_zone_active:
        return k_min_cfg, k_max_cfg

    # Padding factor for the look-ahead window
    padding_factor = float(config.orbit.obliqua.evection_padding_factor)
    if padding_factor <= 0.0:
        return k_min_cfg, k_max_cfg

    # Compute the rate of change of eccentricity (de/dt) over the last macro-step
    e_now = float(hf_row['eccentricity_sat'])
    t_now = float(hf_row['Time'])
    e_prev = hf_row.get('_obliqua_prev_ecc')
    t_prev = hf_row.get('_obliqua_prev_time')

    de_dt_yr = 0.0
    if e_prev is not None and t_prev is not None and t_now > t_prev:
        de_dt_yr = (e_now - float(e_prev)) / (t_now - float(t_prev))

    # Persist the cursor for the next call
    hf_row['_obliqua_prev_ecc'] = e_now
    hf_row['_obliqua_prev_time'] = t_now

    dt_next_yr = float(getattr(interior_o, 'dt', 0.0))

    # Compute the padded k-range for the next macro-step
    k_min_pad, k_max_pad = padded_k_range_for_evection(
        e_now, de_dt_yr, dt_next_yr, padding_factor=padding_factor
    )

    # Enforce user-supplied k_min/k_max
    if isinstance(k_min_cfg, int):
        k_min_pad = min(k_min_pad, k_min_cfg)
    if isinstance(k_max_cfg, int):
        k_max_pad = max(k_max_pad, k_max_cfg)

    return int(k_min_pad), int(k_max_pad)


# Config fields that must NOT pass straight through into Obliqua's cfg dict:
# k_min/k_max are PROTEUS-side inputs to the adaptive s_min/s_max window
# computed per call (see run_obliqua/lookup_from_interior, each of which sets
# its own s_min/s_max or k_min/k_max explicitly); evection_padding_factor and
# verbosity are PROTEUS-side bookkeeping Obliqua itself never reads.
_OBLIQUA_CFG_EXCLUDE = ('k_min', 'k_max', 'evection_padding_factor', 'verbosity')


def _obliqua_module_cfg(config: Config) -> dict:
    """Build the ``cfg['orbit']['obliqua']`` sub-dict from
    ``config.orbit.obliqua`` dynamically (via ``attrs.asdict``), so a new
    config field flows through without this function needing an update.
    Excludes PROTEUS-only bookkeeping (``_OBLIQUA_CFG_EXCLUDE``) and patches
    in ``visc_l``/``visc_s`` (from the interior's own log10-viscosity) and
    ``fluid.sigma_R_inf`` (Obliqua's name for ``sigma_R_factor * sigma_R``).
    Callers still set their own ``s_min``/``s_max`` (or ``k_min``/``k_max``)
    and any call-specific keys (``spectrum``, ``store_3D``, ...).
    """
    obliqua_cfg = asdict(config.orbit.obliqua)
    for key in _OBLIQUA_CFG_EXCLUDE:
        obliqua_cfg.pop(key, None)

    obliqua_cfg['visc_l'] = 10**config.interior_energetics.melt_log10visc
    obliqua_cfg['visc_s'] = 10**config.interior_energetics.solid_log10visc

    fluid = obliqua_cfg['fluid']
    fluid['sigma_R_inf'] = fluid.pop('sigma_R_factor') * fluid['sigma_R']

    return obliqua_cfg


def run_obliqua(
    hf_row: dict, dirs: dict, interior_o: Interior_t, tides_o: Tides_t, config: Config
) -> float:
    """Run the Obliqua tidal heating module.

    Sets the interior tidal heating and returns k-love number. All tidal love-numbers
    are stored in the tides_o object for the specific perturber.

    For the satellite perturber, the adaptive k-range window handed to
    Obliqua is padded ahead of where eccentricity is headed over the next
    macro-step while ``tides_o.evection_zone_active`` is set -- see
    ``_padded_obliqua_k_range``.

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

    # Adaptive k-range window passed to Obliqua below
    s_min_eff = config.orbit.obliqua.k_min
    s_max_eff = config.orbit.obliqua.k_max

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

        # Compute the padded k-range for the next macro-step if in/near evection band
        s_min_eff, s_max_eff = _padded_obliqua_k_range(hf_row, interior_o, tides_o, config)

    else:
        UpdateStatusfile(dirs, 26)
        raise ValueError(
            f"run_obliqua requires config.orbit.perturber to be 'star' or 'satellite', "
            f'got {config.orbit.perturber!r}'
        )

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
                **_obliqua_module_cfg(config),
                'spectrum': 'adaptive',
                's_min': s_min_eff,
                's_max': s_max_eff,
            },
        },
        'interior_energetics': {
            'grain_size': config.interior_energetics.grain_size,
        },
        'struct': {
            'core_density': hf_row.get('core_density', config.interior_struct.core_density),
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
        UpdateStatusfile(dirs, 26)
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
                **_obliqua_module_cfg(config),
                'store_3D': False,
                'enforce_ec': False,
                'optimize_scales': False,
                'solid_shell': False,
                'spectrum': 'full',  # full for lookup table generation
                'N_sigma': 100,  # number of frequency points for the lookup table
                'p_min': -8,  # Minimum period for orbital and axial frequencies [log(kyr)]
                'p_max': 4,  # Maximum period for orbital and axial frequencies [log(kyr)]
                'k_min': 'none',  # not used for lookup table generation, k = 1 for all samples
                'k_max': 'none',  # not used for lookup table generation, k = 1 for all samples
                # use 0-D modules for lookup table generation, since we are only interested in
                # the love number spectrum for an unresolved interior structure.
                'module_solid': 'solid0d',
                'module_mushy': 'none',
                'module_fluid': 'fluid0d',
            },
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
    nc_path = os.path.join(dirs['output/data'], 'sat_tides.nc')
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
            UpdateStatusfile(dirs, 26)
            raise ValueError(
                'Satellite tidal data file path (`config.orbit.satellite.love_number_sat`) is not specified.'
            )

        # check the file extension to determine how to read the lookup data or generate it if it doesn't exist
        if file_path.endswith('.nc'):
            # Read the netcdf file
            pass
        elif file_path.endswith('.json'):
            # Generate the lookup data (sat_tides.nc) from the interior json file
            lookup_from_interior(dirs, config)
            # Update the file path to point to the newly generated netcdf file
            file_path = os.path.join(dirs['output/data'], 'sat_tides.nc')

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
            UpdateStatusfile(dirs, 26)
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


# Bound to Obliqua's own recent-run logfile name -- see make_log_syncer's
# docstring; agni.py binds the same factory to its own AGNI_LOGFILE_NAME.
sync_log_files = make_log_syncer(Obliqua_LOGFILE_NAME)
