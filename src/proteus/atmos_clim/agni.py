# Functions used to handle atmosphere temperature structure (running AGNI, etc.)
from __future__ import annotations

import glob
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from juliacall import Main as jl
from juliacall import convert
from scipy.interpolate import PchipInterpolator

from proteus.atmos_clim.common import get_oarr_from_parr, get_spfile_path
from proteus.utils.constants import gas_list
from proteus.utils.helper import (
    UpdateStatusfile,
    create_tmp_folder,
    multiple,
    safe_rm,
)
from proteus.utils.logs import GetCurrentLogfileIndex, GetLogfilePath

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Constant
AGNI_LOGFILE_NAME = 'agni_recent.log'
ALWAYS_DRY = ('CO', 'N2', 'H2')

# Fields PROTEUS expects to find on the Julia Atmos_t struct after
# `atmosphere.allocate_b` succeeds. The list mirrors what `agni.py` and
# `atmos_clim/common.py` actually read at runtime. A missing entry here
# fires AgniSchemaMismatch at IC rather than surfacing as a silent
# AttributeError once the main coupling loop is running.
_REQUIRED_ATMOS_FIELDS = (
    # Pressure-temperature state
    'tmp',
    'tmpl',
    'pl',
    'p_boa',
    'p_oboa',
    'tmp_surf',
    'tmp_magma',
    # Solver flags
    'is_converged',
    'transparent',
    # Radiative fluxes
    'flux_d_sw',
    'flux_u_lw',
    'flux_u_sw',
    'flux_tot',
    # Per-band optical depth [nlev_c x nbands], longwave all-sky
    'tau_band',
    # Diagnostics computed by AGNI's prescribed-T solver path and
    # available on the struct after the radiative calc. Energy-solver
    # path leaves the arrays at their zero-initialised state.
    'diagnostic_Ra',
    'timescale_conv',
    'timescale_rad',
    'mask_c',
    # Gas composition
    'gas_names',
    'gas_vmr',
    'gas_ovmr',
    # Ocean diagnostics
    'ocean_areacov',
    'ocean_maxdepth',
    'ocean_tot',
    # Stellar / transit
    'instellation',
    'transspec_p',
    'transspec_r',
    'transspec_tmp',
    # Chemistry workspace
    'fastchem_work',
)


class AgniSchemaMismatch(RuntimeError):
    """AGNI's Atmos_t is missing a field PROTEUS expects.

    Raised once at first allocate_b, so a future AGNI rename or removal
    surfaces at IC with a clear list of the missing names instead of
    propagating into the coupling loop as a generic AttributeError.
    """


def _check_agni_schema(atmos) -> None:
    """Verify the live Atmos_t carries every field PROTEUS reads.

    Runs after a successful `atmosphere.allocate_b`; both `setup_b` and
    `allocate_b` must have allocated their backing arrays before this
    is called, because several fields (e.g. ``tau_band``,
    ``flux_*``) only exist after the SOCRATES init block runs.
    """
    missing = [name for name in _REQUIRED_ATMOS_FIELDS if not hasattr(atmos, name)]
    if not missing:
        return
    try:
        version = str(jl.AGNI.consts.AGNI_VERSION)
    except Exception:
        version = 'unknown'
    raise AgniSchemaMismatch(
        f'AGNI {version} Atmos_t is missing PROTEUS-required field(s): '
        f'{", ".join(missing)}. The AGNI pin in pyproject.toml may have '
        'moved past a PROTEUS-known schema; update _REQUIRED_ATMOS_FIELDS '
        'and the matching reads in atmos_clim/agni.py.'
    )


def _summarise_tau_band(atmos) -> tuple[float, float]:
    """Reduce the per-band optical-depth array to TOA and surface scalars.

    AGNI stores ``tau_band`` as a ``(nlev_c, nbands)`` Julia array (level
    index outermost). After juliacall conversion the numpy view may
    appear as either ``(nlev_c, nbands)`` or ``(nbands, nlev_c)``; the
    aggregator inspects ``atmos.nlev_c`` and ``atmos.nbands`` to align.

    Returns
    -------
    tuple of (tau_atm_TOA, tau_atm_surface), each the band-mean
    optical depth at that level. NaN on shape or read errors so the
    helpfile column is still well-formed.
    """
    try:
        tau_arr = np.asarray(atmos.tau_band)
    except Exception:
        return float('nan'), float('nan')
    if tau_arr.size == 0:
        return float('nan'), float('nan')
    nlev_c = int(atmos.nlev_c) if hasattr(atmos, 'nlev_c') else tau_arr.shape[0]
    nbands = int(atmos.nbands) if hasattr(atmos, 'nbands') else tau_arr.shape[-1]
    if tau_arr.shape == (nlev_c, nbands):
        toa = float(tau_arr[0, :].mean())
        surf = float(tau_arr[-1, :].mean())
    elif tau_arr.shape == (nbands, nlev_c):
        toa = float(tau_arr[:, 0].mean())
        surf = float(tau_arr[:, -1].mean())
    else:
        log.warning(
            'tau_band has unexpected shape %s for nlev_c=%d, nbands=%d',
            tau_arr.shape,
            nlev_c,
            nbands,
        )
        return float('nan'), float('nan')
    return toa, surf


def _summarise_diagnostics(atmos) -> tuple[float, float]:
    """Reduce the convection / radiation diagnostic arrays to scalars.

    Returns the maximum Rayleigh number across levels and the ratio
    timescale_conv / timescale_rad evaluated at the topmost convective
    level (the radiative-convective boundary). NaN when no level is
    convective or when the diagnostics were not populated (energy
    solver path skips them).
    """
    try:
        ra_arr = np.asarray(atmos.diagnostic_Ra)
        t_conv_arr = np.asarray(atmos.timescale_conv)
        t_rad_arr = np.asarray(atmos.timescale_rad)
        mask_c = np.asarray(atmos.mask_c).astype(bool)
    except Exception:
        return float('nan'), float('nan')
    Ra_max = float(np.nanmax(ra_arr)) if ra_arr.size else float('nan')
    if not mask_c.any() or t_conv_arr.size == 0 or t_rad_arr.size == 0:
        return Ra_max, float('nan')
    rcb_idx = int(np.argmax(mask_c))  # first convective level from TOA downwards
    denom = max(float(t_rad_arr[rcb_idx]), 1e-300)
    ratio = float(t_conv_arr[rcb_idx]) / denom
    return Ra_max, ratio


def _agni_setup_accepts_aerosol_species() -> bool:
    """Detect whether the installed AGNI version's ``setup!`` accepts the
    ``aerosol_species`` kwarg. Older AGNI installs do not, and passing the
    kwarg trips a Julia MethodError. Resolve at module load by scanning
    every ``atmosphere.jl`` under AGNI/src for a kwarg-list reference of
    the form ``aerosol_species ::`` or ``aerosol_species =``.

    ``atmosphere.jl`` lives under ``src/`` or ``src/state/`` depending
    on the AGNI version; this helper tolerates both paths so PROTEUS
    does not have to be kept in lockstep with AGNI's directory structure.

    Returns ``False`` when AGNI is not on the conventional sibling path
    or when no ``atmosphere.jl`` can be located.
    """
    import re

    from proteus.utils.helper import get_proteus_dir

    agni_root = os.path.join(get_proteus_dir(), 'AGNI', 'src')
    if not os.path.isdir(agni_root):
        return False
    pattern = re.compile(r'\baerosol_species\s*(?:::|=)')
    for root, _dirs, files in os.walk(agni_root):
        if 'atmosphere.jl' in files:
            try:
                with open(os.path.join(root, 'atmosphere.jl'), 'r') as fh:
                    if pattern.search(fh.read()):
                        return True
            except OSError:
                continue
    return False


_AGNI_HAS_AEROSOL_SPECIES = _agni_setup_accepts_aerosol_species()


def sync_log_files(outdir: str) -> list[str]:
    """Move AGNI logfile content into the PROTEUS logfile and clear it.

    Returns the list of lines that were copied, so that callers can scan
    them for failure-mode markers (see `_extract_agni_failure_reason`).
    Returns an empty list if the AGNI logfile cannot be read.
    """
    # Logfile paths
    agni_logpath = os.path.join(outdir, AGNI_LOGFILE_NAME)
    logpath = GetLogfilePath(outdir, GetCurrentLogfileIndex(outdir))

    # Copy logfile content
    try:
        with open(agni_logpath, 'r') as infile:
            inlines = infile.readlines()
    except OSError:
        return []

    with open(logpath, 'a') as outfile:
        for i, line in enumerate(inlines):
            # First line of agni logfile has NULL chars at the start, for some reason
            if i == 0 and '[' in line:
                line = '[' + line.split('[', 1)[1]
            # copy the line
            outfile.write(line)

    # Remove logfile content
    with open(agni_logpath, 'w') as hdl:
        hdl.write('')

    return inlines


# AGNI failure-mode markers emitted by AGNI/src/solver.jl lines 967-993.
# Each `failure (X)` substring corresponds to a CODE_* constant in solver.jl.
# When `_solve_energy` detects a non-convergence, we scan the just-synced AGNI
# log lines for the most recent matching marker so the deadlock detector and
# user can distinguish NaN-flux from singular-jacobian from line-search etc.
_AGNI_FAILURE_MARKERS = (
    ('failure (NaN values)', 'nan_flux'),
    ('failure (singular jacobian)', 'singular_jacobian'),
    ('failure (maximum iterations)', 'max_iterations'),
    ('failure (maximum time)', 'max_time'),
    ('failure (configuration)', 'configuration'),
    ('failure (objective function)', 'objective_function'),
    ('failure (other; last step not ok)', 'last_step_failed'),
    ('failure (hydrostatic integration)', 'hydrostatic_integration'),
    ('failure (other)', 'unknown'),
)


def _extract_agni_failure_reason(loglines: list[str]) -> str:
    """Scan AGNI log lines for the most-recent failure-mode marker.

    Parameters
    ----------
        loglines : list[str]
            Lines just emitted by AGNI's solver, as returned by `sync_log_files`.

    Returns
    -------
        str
            Short tag identifying the failure mode (e.g. 'nan_flux',
            'singular_jacobian'), or 'unparsed' if no marker matched.
    """
    # Iterate from the end so the LAST attempt's failure wins when multiple
    # AGNI attempts have run within one PROTEUS iteration.
    for line in reversed(loglines):
        for marker, tag in _AGNI_FAILURE_MARKERS:
            if marker in line:
                return tag
    return 'unparsed'


def _validate_agni_state(atmos) -> tuple[bool, str]:
    """Validate that an AGNI atmosphere struct holds physically sane values.

    Even when `solve_energy_b` returns success, the post-processing path can
    leave non-finite or unphysical state on the struct (CHILI sweep R12, R17
    sometimes returned T_surf=NaN with success=True). This guard catches that
    before the values poison hf_row and propagate downstream.

    Parameters
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct returned by AGNI's solver.

    Returns
    -------
        ok : bool
            True if all checked fields are finite and physically valid.
        reason : str
            Empty string if ok, otherwise a short description of the failure.
    """
    # AGNI marks `is_converged=True` only on CODE_SUC (solver.jl:962-964).
    # If solve_energy_b returned True but is_converged is False we have a
    # contradictory state and must reject it.
    try:
        is_converged = bool(atmos.is_converged)
    except (AttributeError, Exception):  # noqa: BLE001
        is_converged = True  # missing flag = trust the boolean return
    if not is_converged:
        return False, 'atmos.is_converged is False despite solver success'

    # Surface temperature must be finite and positive.
    try:
        t_surf = float(atmos.tmp_surf)
    except (AttributeError, ValueError, Exception):  # noqa: BLE001
        return False, 'atmos.tmp_surf could not be read'
    if not np.isfinite(t_surf) or t_surf <= 0.0:
        return False, f'atmos.tmp_surf = {t_surf} (non-finite or <= 0)'

    # Total flux profile must be entirely finite.
    try:
        tot_flux = np.array(atmos.flux_tot, dtype=float)
    except (AttributeError, ValueError, Exception):  # noqa: BLE001
        return False, 'atmos.flux_tot could not be read'
    if tot_flux.size == 0:
        return False, 'atmos.flux_tot is empty'
    if not np.all(np.isfinite(tot_flux)):
        n_bad = int(np.sum(~np.isfinite(tot_flux)))
        return False, f'atmos.flux_tot has {n_bad} non-finite element(s)'

    return True, ''


def activate_julia(dirs: dict, verbosity: int):
    log.info('Activating Julia environment')
    jl.seval('using Pkg')
    jl.Pkg.activate(dirs['agni'])

    # Plotting configuration
    jl.seval('ENV["GKSwstype"] = "100"')
    jl.seval('using Plots')
    jl.seval('default(label=nothing, dpi=250)')

    # Import AGNI
    jl.seval('import AGNI')

    # Setup logging from AGNI
    #    This handle will be kept open throughout the PROTEUS simulation, so the file
    #    should not be deleted at runtime. However, it will be emptied when appropriate.
    logpath = os.path.join(dirs['output'], AGNI_LOGFILE_NAME)
    jl.AGNI.setup_logging(logpath, verbosity)

    log.debug("AGNI will log to '%s'" % logpath)


def _construct_voldict(hf_row: dict, dirs: dict):
    # get from hf_row
    vol_dict = {}
    vol_sum = 0.0
    for vol in gas_list:
        vol_dict[vol] = hf_row[vol + '_vmr']
        vol_sum += vol_dict[vol]

    # Check that the total VMR is not zero
    if vol_sum < 1e-4:
        UpdateStatusfile(dirs, 20)
        raise ValueError('All volatiles have a volume mixing ratio of zero')

    return vol_dict


def _determine_condensates(vol_list: list):
    """Determine which gases will be condensable.

    Need to ensure that there's at least one 'dry' gas with non-zero opacity.

    Parameters
    -----------
        config : Config
            Configuration options and other variables
        vol_list: list
            List of included gases.

    Returns
    ----------
        condensates : list
            List of allowed-condensable gases
    """

    # single-gas case must be dry
    if len(vol_list) == 1:
        log.warning('Cannot include rainout condensation with only one gas!')
        return []

    # all dry gases...
    return [v for v in vol_list if v not in ALWAYS_DRY]


def _determine_aerosols(dirs: dict) -> list:
    """
    Determine which aerosols are available.

    Parameters
    ----------
        dirs : dict
            Dictionary containing paths to directories

    Returns
    ----------
        aerosols : list
            List of available aerosols
    """

    scattering_dir = os.path.join(dirs['fwl'], 'scattering', 'scattering')
    if not os.path.isdir(scattering_dir):
        log.warning(f'Scattering data directory not found: {scattering_dir}')
        return []

    aerosols = []
    for f in os.listdir(scattering_dir):
        if f.endswith('.mon'):
            aerosols.append(f.replace('.mon', ''))
    aerosols = sorted(aerosols)

    log.debug(f'Available aerosols: {aerosols}')
    return aerosols


def init_agni_atmos(dirs: dict, config: Config, hf_row: dict):
    """Initialise atmosphere struct for use by AGNI.

    Does not set the temperature profile.

    Parameters
    ----------
        dirs : dict
            Dictionary containing paths to directories
        config : Config
            Configuration options and other variables
        hf_row : dict
            Dictionary containing simulation variables for current iteration

    Returns
    ----------
        atmos : atmosphere.Atmos_t
            Atmosphere struct

    """

    log.debug('New AGNI atmosphere')

    atmos = jl.AGNI.atmosphere.Atmos_t()

    # Decide the spectral-file path first; the stellar-flux glob only runs
    # when we actually need a stellar spectrum (i.e. AGNI will copy + modify
    # the spectral file from FWL_DATA). Grey-gas and user-provided paths
    # bypass the glob entirely so a missing or empty `data/*.sflux` directory
    # is not a precondition for those modes.

    # Spectral file path provided?
    if config.atmos_clim.agni.spectral_file is not None:
        # Grey gas?
        if str(config.atmos_clim.agni.spectral_file).lower() == 'greygas':
            try_spfile = 'greygas'
        else:
            try_spfile = os.path.abspath(config.atmos_clim.agni.spectral_file)
            if not os.path.isfile(try_spfile):
                UpdateStatusfile(dirs, 20)
                raise FileNotFoundError(
                    f'AGNI spectral file not found at specified path: {try_spfile}'
                )
    else:
        # No spectral file provided: use existing runtime.sf in output, or
        # let AGNI copy from FWL_DATA + modify as required.
        try_spfile = os.path.join(dirs['output'], 'runtime.sf')

    # Obtain spectral file
    if try_spfile == 'greygas':
        log.info('Requested grey-gas radiative transfer scheme')
        input_sf = 'greygas'
        input_star = ''
    elif os.path.exists(try_spfile):
        # exists in output folder => don't modify it
        input_sf = try_spfile
        input_star = ''
    else:
        # doesn't exist in output folder => AGNI will copy from FWL_DATA + modify.
        # Resolve the stellar spectrum path here, where it is actually needed.
        sflux_files = glob.glob(os.path.join(dirs['output'], 'data', '*.sflux'))
        if not sflux_files:
            UpdateStatusfile(dirs, 20)
            raise FileNotFoundError(
                f'No stellar spectrum (*.sflux) found in {dirs["output"]}/data; '
                'AGNI cannot construct a fresh spectral file without it'
            )
        sflux_times = [int(s.split('/')[-1].split('.')[0]) for s in sflux_files]
        sflux_path = os.path.join(
            dirs['output'], 'data', '%d.sflux' % int(sorted(sflux_times)[-1])
        )
        input_sf = get_spfile_path(dirs['fwl'], config)
        input_star = sflux_path

    # Fast I/O folder
    if (config.atmos_clim.agni.verbosity >= 2) or (config.params.out.logging == 'DEBUG'):
        io_dir = dirs['output']
    else:
        io_dir = create_tmp_folder()
    log.info(f'Temporary-file working dir: {io_dir}')

    # composition
    vol_dict = _construct_voldict(hf_row, dirs)

    # set condensation
    condensates = []
    if config.atmos_clim.agni.oceans or config.atmos_clim.agni.rainout:
        condensates = _determine_condensates(vol_dict.keys())

    # Chemistry
    include_all = bool(config.atmos_clim.agni.chemistry == 'eq')

    # Surface single-scattering albedo
    surface_material = config.atmos_clim.agni.surf_material
    if 'greybody' in str(surface_material).lower():
        # Grey value
        surface_material = 'greybody'
        log.debug('Using grey single-scattering surface properties')

    else:
        # Empirical values
        log.debug(f"Using '{surface_material}' single-scattering surface properties")
        surface_material = os.path.join(dirs['fwl'], surface_material)
        if not os.path.isfile(surface_material):
            UpdateStatusfile(dirs, 20)
            raise FileNotFoundError(surface_material)

    # Boundary pressures
    p_surf = hf_row['P_surf']
    p_top = config.atmos_clim.p_top
    p_surf = max(p_surf, p_top * 1.1)  # this will happen if the atmosphere is stripped

    # Aerosol species dictionary (set MMR to zero initially)
    aerosol_species = {}
    if config.atmos_clim.aerosols_enabled:
        aerosol_species = {a: 0.0 for a in _determine_aerosols(dirs)}
        if len(aerosol_species) == 0:
            log.warning('No data found for aerosol species')

    # Build the AGNI setup! kwargs. The ``aerosol_species`` parameter is
    # only present on newer AGNI installs; if the installed AGNI predates
    # that addition, sending the kwarg raises a Julia MethodError. Detect
    # the kwarg at module load and only pass it when AGNI accepts it.
    setup_kwargs = dict(
        IO_DIR=io_dir,
        flag_rayleigh=config.atmos_clim.rayleigh,
        flag_cloud=config.atmos_clim.cloud_enabled,
        flag_aerosol=config.atmos_clim.aerosols_enabled,
        overlap_method=config.atmos_clim.overlap_method,
        albedo_s=config.atmos_clim.surf_greyalbedo,
        surface_material=surface_material,
        surf_roughness=config.atmos_clim.agni.surf_roughness,
        surf_windspeed=config.atmos_clim.agni.surf_windspeed,
        condensates=condensates,
        phs_timescale=config.atmos_clim.agni.phs_timescale,
        evap_efficiency=config.atmos_clim.agni.evap_efficiency,
        use_all_gases=include_all,
        fastchem_floor=config.atmos_clim.agni.fastchem_floor,
        fastchem_maxiter_chem=config.atmos_clim.agni.fastchem_maxiter_chem,
        fastchem_maxiter_solv=config.atmos_clim.agni.fastchem_maxiter_solv,
        fastchem_xtol_chem=config.atmos_clim.agni.fastchem_xtol_chem,
        fastchem_xtol_elem=config.atmos_clim.agni.fastchem_xtol_elem,
        real_gas=config.atmos_clim.agni.real_gas,
        check_integrity=False,  # don't check thermo files every time
        mlt_criterion=convert(jl.Char, config.atmos_clim.agni.mlt_criterion),
        skin_d=config.atmos_clim.surface_d,
        skin_k=config.atmos_clim.surface_k,
        tmp_magma=hf_row['T_surf'],
        tmp_floor=config.atmos_clim.tmp_minimum,
        κ_grey_lw=config.atmos_clim.agni.grey_opacity_lw,
        κ_grey_sw=config.atmos_clim.agni.grey_opacity_sw,
    )
    if _AGNI_HAS_AEROSOL_SPECIES:
        setup_kwargs['aerosol_species'] = convert(jl.Dict, aerosol_species)

    succ = jl.AGNI.atmosphere.setup_b(
        atmos,
        dirs['agni'],
        dirs['output'],
        input_sf,
        hf_row['F_ins'],
        config.orbit.s0_factor,
        float(hf_row['albedo_pl']),
        config.orbit.zenith_angle,
        hf_row['T_surf'],
        hf_row['gravity'],
        hf_row['R_int'],
        int(config.atmos_clim.num_levels),
        p_surf,
        p_top,
        vol_dict,
        '',
        **setup_kwargs,
    )

    # Check setup! success
    if not bool(succ):
        UpdateStatusfile(dirs, 22)
        raise RuntimeError('Could not setup atmosphere object')

    # Allocate arrays (check_safe_gas: require at least one dry gas with opacity/thermo)
    check_safe = bool(getattr(config.atmos_clim.agni, 'check_safe_gas', True))
    succ = jl.AGNI.atmosphere.allocate_b(atmos, input_star, check_safe_gas=check_safe)

    # Check allocate! success
    if not bool(succ):
        UpdateStatusfile(dirs, 22)
        raise RuntimeError('Could not allocate atmosphere object')

    # Confirm the live Atmos_t carries every field PROTEUS reads
    _check_agni_schema(atmos)

    # Set temperature profile from old NetCDF if it exists
    nc_files = glob.glob(os.path.join(dirs['output'], 'data', '*_atm.nc'))
    if len(nc_files) > 0:
        log.debug('Load NetCDF profile')

        nc_times = [int(s.split('/')[-1].split('_')[0]) for s in nc_files]
        nc_path = os.path.join(dirs['output'], 'data', f'{sorted(nc_times)[-1]:.0f}_atm.nc')
        jl.AGNI.setpt.fromncdf_b(atmos, nc_path)

    # Otherwise, set profile initial guess
    else:
        # do as requested by user in the config
        log.info(f'Initialising T(p) as {config.atmos_clim.agni.ini_profile}')
        match config.atmos_clim.agni.ini_profile:
            case 'loglinear':
                jl.AGNI.setpt.loglinear_b(atmos, -0.5 * hf_row['T_surf'])
            case 'isothermal':
                jl.AGNI.setpt.isothermal_b(atmos, hf_row['T_surf'])
            case 'dry_adiabat':
                jl.AGNI.setpt.dry_adiabat_b(atmos)
            case 'analytic':
                jl.AGNI.setpt.analytic_b(atmos)
            case _:
                UpdateStatusfile(dirs, 20)
                raise ValueError('Invalid initial T(p) profile selected')

        # lower-limit on initial profile
        jl.AGNI.setpt.stratosphere_b(atmos, min(400.0, hf_row['T_surf']))

    # Logging
    sync_log_files(dirs['output'])

    return atmos


def deallocate_atmos(atmos):
    """
    Deallocate atmosphere struct
    """
    jl.AGNI.atmosphere.deallocate_b(atmos)
    safe_rm(str(atmos.fastchem_work))


def update_agni_atmos(atmos, hf_row: dict, dirs: dict, config: Config):
    """Update atmosphere struct.

    Sets the new boundary conditions and composition.

    Parameters
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            AGNI atmosphere struct
        hf_row : dict
            Dictionary containing simulation variables for current iteration
        dirs : dict
            Directories dictionary
        config: Config
            PROTEUS config object

    Returns
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
    """

    # ---------------------
    # Update instellation flux
    atmos.instellation = float(hf_row['F_ins'])

    # ---------------------
    # Update compositions
    vol_dict = _construct_voldict(hf_row, dirs)
    for g in vol_dict.keys():
        atmos.gas_vmr[g][:] = vol_dict[g]
        atmos.gas_ovmr[g][:] = vol_dict[g]

    # ---------------------
    # Update surface temperature(s)
    atmos.tmp_surf = float(hf_row['T_surf'])
    atmos.tmp_magma = float(hf_row['T_magma'])

    # ---------------------
    # Transparent mode?
    if hf_row['P_surf'] < config.atmos_clim.agni.psurf_thresh:
        # set p_boa to threshold value [bar -> Pa]
        atmos.p_boa = float(config.atmos_clim.agni.psurf_thresh) * 1.0e5

        # update struct to handle this mode of operation
        jl.AGNI.atmosphere.make_transparent_b(atmos)
        jl.AGNI.setpt.isothermal_b(atmos, hf_row['T_surf'])

        # return here - don't do anything else to the `atmos` struct
        return atmos

    # ---------------------
    # Store old/current log-pressure vs temperature arrays
    p_old = list(atmos.p)
    t_old = list(atmos.tmp)
    nlev_c = len(p_old)

    #    extend to lower pressures
    p_old = [p_old[0] / 10] + p_old
    t_old = [t_old[0]] + t_old

    #    extend to higher pressures
    p_old = p_old + [p_old[-1] * 10]
    t_old = t_old + [t_old[-1]]

    #    create interpolator
    itp = PchipInterpolator(np.log10(p_old), t_old)

    # ---------------------
    # Update surface pressure [Pa] and generate new grid
    atmos.p_oboa = 1.0e5 * float(hf_row['P_surf'])
    atmos.p_boa = atmos.p_oboa
    jl.AGNI.atmosphere.generate_pgrid_b(atmos)

    # ---------------------
    # Set temperatures at all levels
    for i in range(nlev_c):
        atmos.tmp[i] = float(itp(np.log10(atmos.p[i])))
        atmos.tmpl[i] = float(itp(np.log10(atmos.pl[i])))
    atmos.tmpl[-1] = float(itp(np.log10(atmos.pl[-1])))

    return atmos


def _solve_energy(atmos, loops_total: int, dirs: dict, config: Config):
    """Use AGNI to solve for energy-conserving solution.

    Parameters
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
        loops_total : int
            Model total loops counter.
        dirs : dict
            Dictionary containing paths to directories
        config : Config
            Configuration options and other variables

    Returns
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
        agni_success : bool
            True if AGNI's Newton solver converged on at least one attempt.
            False if all attempts exhausted with the "Maximum attempts" path.
    """

    # atmosphere solver plotting frequency
    modplot = 0
    plot_jacobian = False
    if config.params.out.logging == 'DEBUG':
        modplot = 1
        plot_jacobian = True

    # tracking
    agni_success = False  # success?
    attempts = 0  # number of attempts so far

    # make attempts
    while not agni_success:
        attempts += 1
        log.info('Attempt %d' % attempts)

        # default parameters
        linesearch = int(config.atmos_clim.agni.ls_default)
        easy_start = False
        grey_start = False
        dx_max = float(config.atmos_clim.agni.dx_max)
        ls_increase = 0.7
        perturb_all = bool(config.atmos_clim.agni.perturb_all)
        max_steps = int(config.atmos_clim.agni.max_steps)
        chemistry = bool(config.atmos_clim.agni.chemistry == 'eq')

        # parameters during initial few iterations
        if loops_total < 3:
            dx_max = float(config.atmos_clim.agni.dx_max_ini)
            ls_increase = 1.1
            max_steps = 200

        # parameters for the first iteration
        if loops_total == 0:
            easy_start = True

        # try different solver parameters if struggling
        if attempts == 1:
            pass

        elif attempts == 2:
            linesearch = 1
            dx_max *= 2.0
            ls_increase = 1.1
            perturb_all = True

            if loops_total == 0:
                grey_start = True

        elif attempts == 3:
            linesearch = 1
            dx_max *= 2.0
            ls_increase = 1.1
            perturb_all = True
            easy_start = True

        else:
            # Max attempts
            log.error('Maximum attempts when executing AGNI')
            break

        log.debug('Solver parameters:')
        log.debug(
            '    ls_method=%d, easy_start=%s, dx_max=%.1f, ls_increase=%.2f'
            % (linesearch, str(easy_start), dx_max, ls_increase)
        )

        # Update solver
        jl.AGNI.solver.solve_energy.ls_increase = float(ls_increase)

        # Try solving temperature profile.
        #
        # We wrap the call in a try/except because AGNI unconditionally
        # invokes `plot_step()` (solver.jl lines 969, 973, 978, 983, 986,
        # 989) whenever its Newton solver fails. When the failure is due to
        # NaN fluxes (CODE_NAN, CODE_OBJ), plot_fluxes passes the NaN to
        # Julia's `range()` which rounds NaN to Int64 and throws
        # InexactError. That exception propagates up the pyjulia boundary
        # and would kill the whole Python process, bypassing the
        # atmosphere-interior deadlock detector in proteus.py::start.
        # Catching it here lets us report the failure cleanly and lets the
        # main loop either retry with different solver params or abort via
        # the deadlock counter.
        try:
            agni_success = jl.AGNI.solver.solve_energy_b(
                atmos,
                sol_type=int(config.atmos_clim.surf_state_int),
                method=int(1),
                chem=chemistry,
                conduct=config.atmos_clim.agni.conduction,
                convect=config.atmos_clim.agni.convection,
                sens_heat=config.atmos_clim.agni.sens_heat,
                latent=config.atmos_clim.agni.latent_heat,
                rainout=config.atmos_clim.agni.rainout,
                oceans=config.atmos_clim.agni.oceans,
                max_steps=int(max_steps),
                max_runtime=900.0,
                conv_atol=float(config.atmos_clim.agni.solution_atol),
                conv_rtol=float(config.atmos_clim.agni.solution_rtol),
                fdo=int(config.atmos_clim.agni.fdo),
                ls_method=int(linesearch),
                dx_max=float(dx_max),
                easy_start=easy_start,
                grey_start=grey_start,
                perturb_all=perturb_all,
                save_frames=False,
                modplot=int(modplot),
                plot_jacobian=plot_jacobian,
            )
        except Exception as e:
            # Any Julia-side exception (InexactError on NaN, SingularException,
            # etc.) is treated as an AGNI non-convergence. We intentionally
            # use a bare `Exception` here because juliacall raises its own
            # exception hierarchy that may not be importable at module load
            # time (chicken-and-egg with juliacall init). The message is
            # logged with traceback for post-mortem, and agni_success stays
            # False so the main loop's deadlock counter handles it.
            log.warning(
                'AGNI solve_energy_b raised a Julia-side exception; '
                'treating as a non-converged attempt. Exception: %s',
                e,
            )
            agni_success = False

        # Move AGNI logfile content into PROTEUS logfile (and capture lines
        # for failure-mode parsing).
        log_lines = sync_log_files(dirs['output'])

        # Defensive: even when AGNI reports success, validate that the
        # returned struct holds finite, physically valid state. AGNI's
        # post-solve processing has been observed to leave NaN tmp_surf
        # or non-finite flux_tot on the struct after `is_converged=True`
        # in rare line-search collapse paths (CHILI sweep R12/R17). If we
        # let those values propagate to hf_row, the deadlock detector
        # never fires and the run silently produces garbage.
        if agni_success:
            ok, reason = _validate_agni_state(atmos)
            if not ok:
                log.error(
                    'AGNI reported success but post-solve validation failed '
                    '(%s). Forcing this attempt to be treated as a failure.',
                    reason,
                )
                agni_success = False

        # Model status check
        if agni_success:
            # success
            log.info('Attempt %d succeeded' % attempts)
            break
        else:
            # failure, loop again...
            reason = _extract_agni_failure_reason(log_lines)
            log.warning('Attempt %d failed (reason: %s)', attempts, reason)

    return atmos, bool(agni_success)


def _solve_once(atmos, config: Config):
    """Use AGNI to solve radiative transfer with prescribed T(p) profile

    Parameters
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
        config : Config
            PROTEUS config object

    Returns
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
    """

    # set temperature profile
    #    rainout volatiles at surface
    rained = jl.AGNI.chemistry.calc_composition_b(
        atmos, config.atmos_clim.agni.oceans, False, False
    )
    rained = bool(rained)
    if rained:
        log.info('    gases are condensing at the surface')
    #    dry convection
    jl.AGNI.setpt.dry_adiabat_b(atmos)
    #    condensation above
    if config.atmos_clim.agni.rainout:
        for gas in gas_list:
            jl.AGNI.setpt.saturation_b(atmos, str(gas))
    #    temperature floor in stratosphere
    jl.AGNI.setpt.stratosphere_b(atmos, 0.5)

    # do chemistry
    jl.AGNI.chemistry.calc_composition_b(
        atmos,
        config.atmos_clim.agni.oceans,
        config.atmos_clim.agni.chemistry == 'eq',
        config.atmos_clim.agni.rainout,
    )

    # solve fluxes
    jl.AGNI.energy.calc_fluxes_b(atmos, radiative=True, convective=True)

    # fill kzz values
    jl.AGNI.energy.fill_Kzz_b(atmos)

    return atmos


def _solve_transparent(atmos, config: Config):
    """
    Use AGNI to solve for the surface temperature under a transparent atmosphere

    Parameters
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
        config : Config
            PROTEUS config object

    Returns
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
    """

    atol = float(config.atmos_clim.agni.solution_atol)
    rtol = float(config.atmos_clim.agni.solution_rtol)
    max_steps = 120

    jl.AGNI.solver.solve_transparent_b(
        atmos,
        sol_type=int(config.atmos_clim.surf_state_int),
        conv_atol=atol,
        conv_rtol=rtol,
        max_steps=int(max_steps),
    )
    return atmos


def run_agni(atmos, loops_total: int, dirs: dict, config: Config, hf_row: dict):
    """Run AGNI atmosphere model.

    Calculates the temperature structure of the atmosphere and the fluxes, etc.
    Stores the new flux boundary condition to be provided to SPIDER.

    Parameters
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
        loops_total : int
            Model total loops counter.
        dirs : dict
            Dictionary containing paths to directories
        config : Config
            Configuration options and other variables
        hf_row : dict
            Dictionary containing simulation variables for current iteration

    Returns
    ----------
        atmos : AGNI.atmosphere.Atmos_t
            Atmosphere struct
        output : dict
            Output variables, as a dictionary
    """

    # Inform
    log.debug('Running AGNI...')

    # ---------------------------
    # Solve atmosphere
    # ---------------------------

    # Track whether AGNI's Newton solver actually converged. The transparent
    # and prescribed-T branches do not run a Newton solver, so they cannot
    # "fail" in the deadlock sense; only `_solve_energy` can. We default to
    # True and override only in the energy branch.
    agni_converged = True

    # Transparent case
    if bool(atmos.transparent):
        # no opacity
        log.info('Using transparent solver')
        atmos.transspec_p = float(atmos.p_boa)
        atmos = _solve_transparent(atmos, config)

    # Opaque case
    else:
        # Set observed pressure
        atmos.transspec_p = float(config.atmos_clim.p_obs * 1e5)  # converted to Pa

        # full solver
        if config.atmos_clim.agni.solve_energy:
            log.info('Using nonlinear solver to conserve fluxes')
            atmos, agni_converged = _solve_energy(atmos, loops_total, dirs, config)

        # simplified T(p)
        else:
            log.info('Using prescribed temperature profile')
            atmos = _solve_once(atmos, config)

    # Calculate planet transit radius (to be stored in NetCDF)
    jl.AGNI.atmosphere.calc_observed_rho_b(atmos)

    # Write output data
    log.debug('AGNI write to NetCDF file')
    ncdf_path = os.path.join(dirs['output'], 'data', '%.0f_atm.nc' % hf_row['Time'])
    jl.AGNI.save.write_ncdf(atmos, ncdf_path)

    # Make plots
    if multiple(loops_total, config.params.out.plot_mod):
        cff = os.path.join(dirs['output/plots'], f'plot_cff.{config.params.out.plot_fmt}')
        jl.AGNI.plotting.plot_contfunc1(atmos, cff)

    # ---------------------------
    # Parse results
    # ---------------------------

    log.debug('Parse results')
    tot_flux = np.array(atmos.flux_tot)
    LW_flux_up = np.array(atmos.flux_u_lw)
    SW_flux_up = np.array(atmos.flux_u_sw)
    SW_flux_down = np.array(atmos.flux_d_sw)
    # Guard against zero instellation (nightside or F_ins=0), where there is no
    # downward shortwave flux to reflect and the albedo ratio is undefined.
    albedo = SW_flux_up[0] / SW_flux_down[0] if SW_flux_down[0] > 0.0 else 0.0
    if bool(atmos.transparent):
        R_obs = float(hf_row['R_int'])
        T_obs = float(atmos.tmp_surf)
    else:
        R_obs = float(atmos.transspec_r)
        T_obs = float(atmos.transspec_tmp)

    # Print info to user
    if config.atmos_clim.agni.oceans:
        log.info('    oceans area frac   = %6.3f %%' % float(atmos.ocean_areacov * 100))
        log.info('    oceans max depth   = %6.3f km' % float(atmos.ocean_maxdepth / 1e3))
    log.info('    R_obs photosphere  = %6.1f km' % float(R_obs / 1e3))
    log.info('    Planet Bond albedo = %6.3f %%' % float(albedo * 100))

    # New flux from SOCRATES
    F_atm_new = tot_flux[0]

    # Enforce positive limit on F_atm, if enabled
    if config.planet.prevent_warming:
        F_atm_lim = max(1e-8, F_atm_new)
    else:
        F_atm_lim = F_atm_new
    if not np.isclose(F_atm_lim, F_atm_new):
        log.warning('Change in F_atm [W m-2] limited in this step!')
        log.warning('    %g  ->  %g' % (F_atm_new, F_atm_lim))

    # p_xuv from R_xuv
    if config.escape.xuv_defined_by_radius:
        r_xuv = hf_row['R_xuv']  # m
        p_xuv = get_oarr_from_parr(atmos.r, atmos.p, r_xuv)[1] * 1e-5  # bar

    # R_xuv from p_xuv
    else:
        p_xuv = hf_row['p_xuv']  # bar
        r_xuv = get_oarr_from_parr(atmos.p, atmos.r, p_xuv * 1e5)[1]  # m

    # Diagnostics surfaced into hf_row: band-mean optical depth at TOA
    # and at the surface, plus the Rayleigh number maximum and the
    # convective vs radiative timescale ratio at the RCB.
    tau_TOA, tau_surface = _summarise_tau_band(atmos)
    Ra_max, t_conv_over_t_rad = _summarise_diagnostics(atmos)

    # final things to store
    output = {}
    output['F_atm'] = F_atm_lim
    output['F_olr'] = LW_flux_up[0]
    output['F_sct'] = SW_flux_up[0]
    output['T_surf'] = float(atmos.tmp_surf)
    output['p_obs'] = float(atmos.transspec_p) / 1e5  # convert [Pa] to [bar]
    output['T_obs'] = T_obs
    output['R_obs'] = R_obs
    output['albedo'] = albedo
    output['tau_atm_TOA'] = tau_TOA
    output['tau_atm_surface'] = tau_surface
    output['agni_Ra_max'] = Ra_max
    output['agni_t_conv_over_t_rad'] = t_conv_over_t_rad
    # Transient-only flag (not persisted to helpfile). True if AGNI's Newton
    # solver converged on at least one attempt; False if all attempts were
    # exhausted via the "Maximum attempts" path. The main coupling loop uses
    # this to detect AGNI deadlocks (see proteus.py).
    output['agni_converged'] = bool(agni_converged)
    output['p_xuv'] = p_xuv  # Pressure at Rxuv   [bars]
    output['R_xuv'] = r_xuv  # Radius at Pxuv     [m]
    output['ocean_areacov'] = float(atmos.ocean_areacov)
    output['ocean_maxdepth'] = float(atmos.ocean_maxdepth)
    output['P_surf_clim'] = float(atmos.p_boa) / 1e5  # Calculated Psurf [bar]

    for g in gas_list:
        if g in list(atmos.gas_names):
            output[g + '_ocean'] = float(atmos.ocean_tot[g])
        else:
            output[g + '_ocean'] = 0.0

    # set composition at xuv
    for g in gas_list:
        if g in atmos.gas_vmr:
            _, x_xuv = get_oarr_from_parr(atmos.p, atmos.gas_vmr[g], p_xuv * 1e5)
            hf_row[g + '_vmr_xuv'] = x_xuv
        else:
            hf_row[g + '_vmr_xuv'] = 0.0

    return atmos, output
