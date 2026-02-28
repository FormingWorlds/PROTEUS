# Generic interior wrapper
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.optimize as optimise

from proteus.interior.common import Interior_t
from proteus.outgas.wrapper import calc_target_elemental_inventories
from proteus.utils.constants import M_earth, R_earth, const_G, element_list
from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def update_gravity(hf_row: dict):
    """
    Update surface gravity.
    """
    hf_row['gravity'] = const_G * hf_row['M_int'] / (hf_row['R_int'] * hf_row['R_int'])


def calculate_core_mass(hf_row: dict, config: Config):
    """
    Calculate the core mass of the planet.
    """
    hf_row['M_core'] = (
        config.struct.core_density
        * 4.0
        / 3.0
        * np.pi
        * (hf_row['R_int'] * config.struct.corefrac) ** 3.0
    )


def update_planet_mass(hf_row: dict):
    """
    Calculate total planet mass, as sum of dry+wet parts.
    """

    # Update total element mass
    hf_row['M_ele'] = 0.0
    for e in element_list:
        if e == 'O':  # Oxygen is set by fO2, so we skip it here (const_fO2)
            continue
        hf_row['M_ele'] += hf_row[e + '_kg_total']

    # Add to total planet mass
    hf_row['M_planet'] = hf_row['M_int'] + hf_row['M_ele']


def get_nlevb(config: Config):
    """
    Get number of interior basic-nodes (level edges) from config.
    """
    match config.interior.module:
        case 'spider':
            return int(config.interior.spider.num_levels)
        case 'aragog':
            return int(config.interior.aragog.num_levels)
        case 'dummy':
            return 2
    raise ValueError(f"Invalid interior module selected '{config.interior.module}'")


def determine_interior_radius(dirs: dict, config: Config, hf_all: pd.DataFrame, hf_row: dict):
    """
    Determine the interior radius (R_int) of the planet.

    This uses the interior model's hydrostatic integration to estimate the planet's
    interior mass from a given radius. The radius is then adjusted until the interior mass
    achieves the target mass provided by the user in the config file.
    """

    log.info('Using %s interior module to solve structure' % config.interior.module)

    # Initial guess for interior radius and gravity
    if config.interior.module == 'spider':
        spider_dir = dirs['spider']
    else:
        spider_dir = None
    int_o = Interior_t(
        get_nlevb(config), spider_dir=spider_dir, eos_dir=config.interior.eos_dir
    )
    int_o.ic = 1
    hf_row['R_int'] = R_earth
    calculate_core_mass(hf_row, config)
    hf_row['gravity'] = 9.81

    # Target mass
    M_target = config.struct.mass_tot * M_earth

    # We need to solve for the state hf_row[M_planet] = config.struct.mass_tot
    # This function takes R_int as the input value, and returns the mass residual
    def _resid(x):
        hf_row['R_int'] = x

        log.debug('Try R = %.2e m = %.3f R_earth' % (x, x / R_earth))

        # Use interior model to get dry mass from radius
        calculate_core_mass(hf_row, config)
        run_interior(dirs, config, hf_all, hf_row, int_o, verbose=False)
        update_gravity(hf_row)

        # Get wet mass
        calc_target_elemental_inventories(dirs, config, hf_row)

        # Get total planet mass
        update_planet_mass(hf_row)

        # Calculate residual
        res = hf_row['M_planet'] - M_target
        log.debug('    yields M = %.5e kg , resid = %.3e kg' % (hf_row['M_planet'], res))

        return res

    # Set tolerance
    match config.interior.module:
        case 'aragog':
            rtol = config.interior.aragog.tolerance
        case 'spider':
            rtol = config.interior.spider.tolerance
        case _:
            rtol = 1e-7

    # Find the radius
    r = optimise.root_scalar(
        _resid,
        method='secant',
        xtol=1e3,
        rtol=rtol,
        maxiter=10,
        x0=hf_row['R_int'],
        x1=hf_row['R_int'] * 1.02,
    )
    hf_row['R_int'] = float(r.root)
    calculate_core_mass(hf_row, config)
    run_interior(dirs, config, hf_all, hf_row, int_o)
    update_gravity(hf_row)

    # Result
    log.info('Found solution for interior structure')
    log.info(
        'M_planet: %.1e kg = %.3f M_earth' % (hf_row['M_planet'], hf_row['M_planet'] / M_earth)
    )
    log.info('R_int: %.1e m  = %.3f R_earth' % (hf_row['R_int'], hf_row['R_int'] / R_earth))
    log.info(' ')


def determine_interior_radius_with_zalmoxis(
    dirs: dict, config: Config, hf_all: pd.DataFrame, hf_row: dict, outdir: str
):
    """
    Determine the interior radius (R_int) of the planet using Zalmoxis.

    When the interior module is SPIDER, also writes a SPIDER-format mesh
    file from the Zalmoxis structure solution and stores the path in
    ``dirs['spider_mesh']`` for subsequent calls.
    """

    log.info('Using Zalmoxis to solve for interior structure')
    from proteus.interior.zalmoxis import zalmoxis_solver

    nlev_b = get_nlevb(config)
    spider_dir = dirs.get('spider') if config.interior.module == 'spider' else None
    int_o = Interior_t(nlev_b, spider_dir=spider_dir, eos_dir=config.interior.eos_dir)
    int_o.ic = 1

    # Default to adiabatic temperature mode for SPIDER + T-dependent mantle EOS.
    # This produces a self-consistent adiabat that matches SPIDER's initial
    # entropy profile, avoiding large mesh shifts on the first Phase 2 update.
    _TDEP_PREFIXES = ('WolfBower2018', 'RTPress100TPa')
    if (
        config.interior.module == 'spider'
        and config.struct.zalmoxis.temperature_mode == 'isothermal'
        and config.struct.zalmoxis.mantle_eos.startswith(_TDEP_PREFIXES)
    ):
        log.info(
            'Switching Zalmoxis temperature_mode from isothermal to adiabatic '
            'for SPIDER coupling with T-dependent mantle EOS'
        )
        config.struct.zalmoxis.temperature_mode = 'adiabatic'

    # Request SPIDER mesh file if interior module is SPIDER
    num_spider_nodes = nlev_b if config.interior.module == 'spider' else 0
    _cmb_radius, spider_mesh_file = zalmoxis_solver(
        config, outdir, hf_row, num_spider_nodes=num_spider_nodes
    )

    # Store mesh file path for subsequent SPIDER calls
    if spider_mesh_file:
        dirs['spider_mesh'] = spider_mesh_file

    run_interior(dirs, config, hf_all, hf_row, int_o)


def solve_structure(
    dirs: dict, config: Config, hf_all: pd.DataFrame, hf_row: dict, outdir: str
):
    """
    Solve for the planet structure based on the method set in the configuration file.

    If the structure is set by the radius, then this is trivial because the radius is used
    as an input to the interior modules anyway. If the structure is set by mass, then it is
    solved as an inverse problem for now.
    """

    # Set total mass by radius
    # We might need here to setup a determine_interior_mass function as mass calculation depends on gravity
    if config.struct.set_by == 'radius_int':
        # radius defines interior structure
        hf_row['R_int'] = config.struct.radius_int * R_earth
        calculate_core_mass(hf_row, config)
        # initial guess for mass, which will be updated by the interior model
        hf_row['M_int'] = 1.2 * M_earth
        update_gravity(hf_row)

    # Set by total mass (mantle + core + volatiles)
    elif config.struct.set_by == 'mass_tot':
        # Choose the method to determine the interior radius
        match config.struct.module:
            case 'self':
                return determine_interior_radius(dirs, config, hf_all, hf_row)
            case 'zalmoxis':
                config.orbit.module = (
                    'dummy'  # Switch to dummy orbit module when using Zalmoxis for now
                )
                if config.params.stop.solid.phi_crit < 0.01:
                    log.warning(
                        'phi_crit=%.4f is below 0.01. Zalmoxis cases may plateau '
                        'at ~0.9%% melt fraction, so phi_crit < 0.01 can prevent '
                        'the simulation from terminating. Consider phi_crit >= 0.01.',
                        config.params.stop.solid.phi_crit,
                    )
                return determine_interior_radius_with_zalmoxis(
                    dirs, config, hf_all, hf_row, outdir
                )
        raise ValueError(
            f"Invalid structure interior module selected '{config.interior.module}'"
        )

    # Otherwise, error
    else:
        log.error('Invalid constraint on interior structure: %s' % config.struct.set_by)


def run_interior(
    dirs: dict,
    config: Config,
    hf_all: pd.DataFrame,
    hf_row: dict,
    interior_o: Interior_t,
    verbose: bool = True,
):
    """Run interior mantle evolution model.

    Parameters
    ----------
        dirs : dict
            Dictionary of directories.
        config : Config
            Model configuration
        hf_all : pd.DataFrame
            Dataframe of historical runtime variables
        hf_row : dict
            Dictionary of current runtime variables
        interior_o : Interior_t
            Interior struct.
        verbose : bool
            Verbose printing enabled.
    """

    # Use the appropriate interior model
    if verbose:
        log.info('Evolve interior...')
    log.debug('Using %s module to evolve interior' % config.interior.module)

    # Write tidal heating file
    if config.interior.tidal_heat:
        interior_o.write_tides(dirs['output'])

    if config.interior.module == 'spider':
        # Import
        from proteus.interior.spider import ReadSPIDER, RunSPIDER

        # Run SPIDER (pass external mesh file if available from Zalmoxis)
        mesh_file = dirs.get('spider_mesh')
        RunSPIDER(dirs, config, hf_all, hf_row, interior_o, mesh_file=mesh_file)
        sim_time, output = ReadSPIDER(dirs, config, hf_row['R_int'], interior_o)

    elif config.interior.module == 'aragog':
        from proteus.interior.aragog import AragogRunner

        AragogRunnerInstance = AragogRunner(config, dirs, hf_row, hf_all, interior_o)
        # Run Aragog
        sim_time, output = AragogRunnerInstance.run_solver(hf_row, interior_o, dirs)

    elif config.interior.module == 'dummy':
        # Import
        from proteus.interior.dummy import run_dummy_int

        # Run dummy interior
        sim_time, output = run_dummy_int(config, dirs, hf_row, hf_all, interior_o)

    # Read output
    for k in output.keys():
        if k in hf_row.keys():
            val = output[k]
            # Convert numpy arrays and scalars to Python scalars for NumPy 2.0 compatibility
            if isinstance(val, np.generic):
                hf_row[k] = val.item()
            elif np.isscalar(val):
                hf_row[k] = val
            else:
                try:
                    arr = np.asarray(val)
                    if arr.size == 1 and hasattr(arr, 'item'):
                        hf_row[k] = arr.item()
                    else:
                        hf_row[k] = val
                except Exception as exc:
                    log.error(
                        'Failed to convert output value for key %r (%r) to a NumPy array/scalar (%s: %s)',
                        k,
                        val,
                        type(exc).__name__,
                        exc,
                    )
                    raise

    # Update rheological parameters
    #    Only calculate viscosity here if using dummy module
    calc_visc = bool(config.interior.module == 'dummy')
    interior_o.update_rheology(visc=calc_visc)

    # Ensure values are >= 0
    for k in ('M_mantle', 'M_mantle_liquid', 'M_mantle_solid', 'M_core', 'Phi_global'):
        hf_row[k] = max(hf_row[k], 0.0)

    # Check that the new temperature is remotely reasonable
    if not (0 < hf_row['T_magma'] < 1e6):
        UpdateStatusfile(dirs, 21)
        raise ValueError('T_magma is out of range: %g K' % float(hf_row['T_magma']))

    # Update dry interior mass
    hf_row['M_int'] = hf_row['M_mantle'] + hf_row['M_core']

    # Update planet mass
    update_planet_mass(hf_row)

    # Apply step limiters
    if hf_row['Time'] > 0:
        # Prevent increasing melt fraction, if enabled
        T_magma_prev = float(hf_all.iloc[-1]['T_magma'])
        Phi_global_prev = float(hf_all.iloc[-1]['Phi_global'])
        if config.atmos_clim.prevent_warming and (interior_o.ic == 2):
            hf_row['Phi_global'] = min(hf_row['Phi_global'], Phi_global_prev)
            hf_row['T_magma'] = min(hf_row['T_magma'], T_magma_prev)

        # Do not allow massive increases to T_surf, always
        dT_delta = config.interior.spider.tsurf_atol
        dT_delta += config.interior.spider.tsurf_rtol * T_magma_prev
        if hf_row['T_magma'] > T_magma_prev + dT_delta:
            log.warning('Prevented large increase to T_magma!')
            log.warning('   Clipped from %.2f K' % hf_row['T_magma'])
            hf_row['T_magma'] = T_magma_prev + dT_delta
            hf_row['Phi_global'] = Phi_global_prev

    # Print result of interior module
    if verbose:
        log.info('    T_magma    = %.3f K' % float(hf_row['T_magma']))
        log.info('    Phi_global = %.3f  ' % float(hf_row['Phi_global']))
        log.info('    RF_depth   = %.3f  ' % float(hf_row['RF_depth']))
        log.info('    F_int      = %.2e W m-2' % float(hf_row['F_int']))
        if config.interior.tidal_heat:
            log.info('    F_tidal    = %.2e W m-2' % float(hf_row['F_tidal']))
        if config.interior.radiogenic_heat:
            log.info('    F_radio    = %.2e W m-2' % float(hf_row['F_radio']))

    # Actual time step size
    interior_o.dt = float(sim_time) - hf_row['Time']


def update_structure_from_interior(
    dirs: dict,
    config: Config,
    hf_row: dict,
    interior_o: Interior_t,
    last_struct_time: float,
) -> float:
    """Re-run Zalmoxis with SPIDER's current T(r) to update structure.

    Writes SPIDER's temperature profile to a file, runs Zalmoxis in
    prescribed-temperature mode, and writes an updated mesh file for the
    next SPIDER call.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories.
    config : Config
        Model configuration.
    hf_row : dict
        Current runtime variables.
    interior_o : Interior_t
        Interior state with current T(r) on staggered nodes.
    last_struct_time : float
        Simulation time [yr] of the last structure update.

    Returns
    -------
    float
        Updated ``last_struct_time`` (set to current time if update occurred,
        otherwise returned unchanged).
    """
    # Check trigger conditions
    current_time = hf_row['Time']
    if config.struct.update_interval <= 0:
        return last_struct_time
    if (current_time - last_struct_time) < config.struct.update_interval:
        return last_struct_time

    log.info('Updating structure from interior T(r) via Zalmoxis')

    outdir = dirs['output']
    num_layers = config.struct.zalmoxis.num_levels

    # Build SPIDER's mantle T(r) in ascending radius (CMB to surface)
    # interior_o.radius is basic nodes (surface to CMB), temp is staggered nodes
    r_stag = 0.5 * (interior_o.radius[:-1] + interior_o.radius[1:])
    r_ascending = r_stag[::-1]
    T_ascending = interior_o.temp[::-1]

    # Zalmoxis prescribed mode expects a 1D array of num_layers temperatures
    # sampled from center (r=0) to surface (r=R_planet), matching its internal
    # radii. SPIDER only covers the mantle (CMB to surface), so we hold T
    # constant at the CMB value for the core region.
    R_cmb = float(r_ascending[0])
    R_surf = float(r_ascending[-1])
    T_cmb = float(T_ascending[0])

    r_full = np.linspace(0.0, R_surf, num_layers)
    T_full = np.empty(num_layers)
    for i, r in enumerate(r_full):
        if r <= R_cmb:
            T_full[i] = T_cmb
        else:
            T_full[i] = np.interp(r, r_ascending, T_ascending)

    temp_profile_path = os.path.join(outdir, 'data', 'spider_temp_profile.dat')
    np.savetxt(temp_profile_path, T_full)

    # Temporarily override Zalmoxis config for prescribed temperature mode
    from proteus.interior.zalmoxis import zalmoxis_solver

    orig_temp_mode = config.struct.zalmoxis.temperature_mode
    orig_temp_file = config.struct.zalmoxis.temperature_profile_file

    config.struct.zalmoxis.temperature_mode = 'prescribed'
    config.struct.zalmoxis.temperature_profile_file = temp_profile_path

    try:
        nlev_b = get_nlevb(config)
        num_spider_nodes = nlev_b if config.interior.module == 'spider' else 0
        _cmb_radius, spider_mesh_file = zalmoxis_solver(
            config, outdir, hf_row, num_spider_nodes=num_spider_nodes
        )
    finally:
        # Restore original config
        config.struct.zalmoxis.temperature_mode = orig_temp_mode
        config.struct.zalmoxis.temperature_profile_file = orig_temp_file

    if spider_mesh_file:
        dirs['spider_mesh'] = spider_mesh_file

        # Remap entropy in the latest SPIDER JSON to match the new mesh.
        # Without this, the old dS/dxi applied on the new xi grid produces
        # incorrect absolute entropy, causing CVode failures at high mass.
        if config.interior.module == 'spider':
            from proteus.interior.spider import (
                get_all_output_times,
                remap_entropy_for_new_mesh,
            )

            sim_times = get_all_output_times(dirs['output'])
            if len(sim_times) > 0:
                latest_json = os.path.join(dirs['output'], 'data', '%.0f.json' % sim_times[-1])
                remap_entropy_for_new_mesh(
                    json_path=latest_json,
                    new_mesh_file=spider_mesh_file,
                    radius_phys=hf_row['R_int'],
                )

    log.info(
        'Structure updated: R_int=%.3e m, gravity=%.3f m/s^2',
        hf_row['R_int'],
        hf_row['gravity'],
    )
    return current_time


def get_all_output_times(output_dir: str, model: str):
    if model == 'spider':
        from proteus.interior.spider import get_all_output_times as _get_output_times
    elif model == 'aragog':
        from proteus.interior.aragog import get_all_output_times as _get_output_times
    else:
        return []

    return _get_output_times(output_dir)


def read_interior_data(output_dir: str, model: str, times: list):
    if len(times) == 0:
        return []

    if model == 'spider':
        from proteus.interior.spider import read_jsons

        return read_jsons(output_dir, times)

    elif model == 'aragog':
        from proteus.interior.aragog import read_ncdfs

        return read_ncdfs(output_dir, times)

    else:
        return []
