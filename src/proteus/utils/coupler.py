# Functions used to help run PROTEUS which are mostly module agnostic.

# Import utils-specific modules
from __future__ import annotations

import glob
import logging
import os
import subprocess
from datetime import datetime
from typing import TYPE_CHECKING

import matplotlib as mpl  # noqa

mpl.use('Agg')  # noqa
from string import ascii_letters

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from proteus.utils.constants import (
    element_list,
    secs_per_hour,
    secs_per_minute,
    vap_list,
    vol_list,
)
from proteus.utils.helper import UpdateStatusfile, create_tmp_folder, get_proteus_dir, safe_rm
from proteus.utils.plot import sample_times

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

LOCKFILE_NAME = 'keepalive'
AGNI_MIN_VERSION = '1.8.0'


def _get_current_time():
    """
    Get the current system time as a formatted string.
    """
    return str(datetime.now().strftime('%Y-%m-%d %H:%M:%S %Z'))


def _get_git_revision(dir: str) -> str:
    """
    Get git hash for repository in `dir`.
    """
    # change dir
    cwd = os.getcwd()
    try:
        os.chdir(dir)
    except Exception:
        # If we can't change to the directory, return unknown
        return 'unknown'

    # get hash (https://stackoverflow.com/a/21901260)
    try:
        hash = (
            subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                timeout=5,  # Add timeout to prevent hanging
            )
            .decode('ascii')
            .strip()
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
        Exception,
    ):
        # Handle: git not found, not a git repo, git command failed, or any other error
        hash = 'unknown'
    finally:
        # Always change dir back, even if an exception occurred
        try:
            os.chdir(cwd)
        except Exception:
            pass  # If we can't change back, continue anyway

    return hash


def _get_socrates_version():
    """
    Get the installed SOCRATES version.
    """
    RAD_DIR = os.environ.get('RAD_DIR')
    if RAD_DIR is None:
        raise EnvironmentError('RAD_DIR environment variable is not set.')

    verpath = os.path.join(RAD_DIR, 'version')
    with open(verpath, 'r') as hdl:
        ver = hdl.read().replace('\n', '')
    return str(ver)


def _get_spider_version():
    """
    Get the installed SPIDER version.
    """

    # This is the only one that we use, and it won't be updated in the future.
    return '0.2.0'


def _get_petsc_version():
    """
    Get the installed PETSc version.
    """

    # Like SPIDER, this is the only one that we use
    return '1.3.19.0'


def _get_agni_version(dirs: dict):
    """
    Get the installed AGNI version
    """
    from tomllib import load as tomlload

    with open(os.path.join(dirs['agni'], 'Project.toml'), 'rb') as hdl:
        agni_meta = tomlload(hdl)
    return agni_meta['version']


def _get_julia_version():
    """
    Get the installed Julia version
    """
    return subprocess.check_output(['julia', '--version']).decode('utf-8').split()[-1]


def validate_module_versions(dirs: dict, config: Config):
    """Raise if module versions are incompatible."""

    log.info('Validating module versions')

    # Read PROTEUS dependencies
    from importlib.metadata import requires

    deps_raw = requires('fwl-proteus')

    # Get minimum required version for a module (return None if no version specified)
    def _get_expver(mod):
        for v in deps_raw:
            if '=' not in v:
                continue
            vmod = v.split('=')[0].strip('>')  # get name of module
            vver = v.split('=')[1]  # get req version of module
            if mod == vmod:
                return vver
        return None

    # Split version string into major/minor/patch components
    def _split_ver(vver):
        # ignore 'alpha' part and remove leading letters
        vver = vver.split('-')[0].strip(ascii_letters)

        # split into version parts
        s = vver.split('.')
        major = int(s[0])
        minor = int(s[1])
        try:
            patch = int(s[2])
        except Exception:
            patch = 0
        return major, minor, patch

    def _valid_ver(act_str: str, exp_str: str, name: str) -> bool:
        """Check if found version is compatible with expected version.

        Parameters
        -----------
        - act_str:str
            Actual module version found installed
        - exp_str:str
            Expected or required version of the module
        - name:str
            Module name

        Returns
        ----------
        - bool
            Version is compatible
        """

        # return True if expected is None
        if exp_str is None:
            return True

        # convert from string to m/m/p format (or y/m/d format)
        vact = _split_ver(act_str)
        vexp = _split_ver(exp_str)

        log.debug(f'Parsed {name:10s} version as {vact}. Requires>={vexp}')

        # Lexicographic tuple comparison (correct semver ordering)
        if vact >= vexp:
            return True

        log.error(f'{name} module is out of date: installed {act_str} < expected {exp_str}')
        return False

    # Loop through required modules...
    valid = True

    # Interior module
    match config.interior_energetics.module:
        case 'spider':
            # do not validate SPIDER version
            pass
        case 'aragog':
            from aragog import __version__ as aragog_version

            valid &= _valid_ver(aragog_version, _get_expver('fwl-aragog'), 'Aragog')

    # Struct module
    if config.interior_struct.module == 'zalmoxis':
        from zalmoxis import __version__ as zalmoxis_version

        valid &= _valid_ver(zalmoxis_version, _get_expver('fwl-zalmoxis'), 'Zalmoxis')

    # Atmosphere module
    match config.atmos_clim.module:
        case 'janus':
            from janus import __version__ as janus_version

            valid &= _valid_ver(janus_version, _get_expver('fwl-janus'), 'JANUS')
        case 'agni':
            valid &= _valid_ver(_get_agni_version(dirs), AGNI_MIN_VERSION, 'AGNI')

    # Outgassing module
    if config.outgas.module == 'calliope':
        from calliope import __version__ as calliope_version

        valid &= _valid_ver(calliope_version, _get_expver('fwl-calliope'), 'CALLIOPE')

    # Escape module
    match config.escape.module:
        case 'zephyrus':
            from zephyrus import __version__ as zephyrus_version

            valid &= _valid_ver(zephyrus_version, _get_expver('fwl-zephyrus'), 'ZEPHYRUS')
        case 'boreas':
            from boreas import __version__ as boreas_version

            valid &= _valid_ver(boreas_version, _get_expver('boreas'), 'BOREAS')

    # Star module
    if config.star.module == 'mors':
        from mors import __version__ as mors_version

        valid &= _valid_ver(mors_version, _get_expver('fwl-mors'), 'MORS')

    # Exit
    if not valid:
        UpdateStatusfile(dirs, 20)
        raise EnvironmentError(
            'Out-of-date modules detected. Refer to the Troubleshooting guide:\n'
            'https://proteus-framework.org/proteus/troubleshooting.html'
        )
    log.info(' ')


def print_system_configuration(dirs: dict):
    """
    Print the current system configuration.
    """
    import platform
    import pwd
    import sys

    # Try to get the login name using os.getlogin()
    try:
        username = os.getlogin()
    except OSError:
        username = pwd.getpwuid(os.getuid()).pw_name

    log.info('Current time      ' + _get_current_time())
    log.info('Python version    ' + sys.version.split(' ')[0])
    log.info('System hostname   ' + str(os.uname()[1]))
    log.info('System username   ' + str(username))
    log.info('Platform type     ' + str(platform.system()))
    log.info('FWL data path     ' + dirs['fwl'])
    log.info(' ')


def print_module_configuration(dirs: dict, config: Config, config_path: str):
    """
    Print the current module configuration, with versions.
    """

    # PROTEUS
    from proteus import __version__ as proteus_version

    log.info('PROTEUS version   ' + proteus_version)
    log.info('PROTEUS location  ' + dirs['proteus'])
    log.info('PROTEUS git hash  ' + _get_git_revision(dirs['proteus']))
    log.info('Config file       ' + str(config_path))
    log.info('Output path       ' + dirs['output'])
    log.info(' ')

    # Interior module
    write = 'Interior module   %s' % config.interior_energetics.module
    match config.interior_energetics.module:
        case 'spider':
            write += ' version ' + _get_spider_version()
        case 'aragog':
            from aragog import __version__ as aragog_version

            write += ' version ' + aragog_version
    log.info(write)
    if config.interior_energetics.module == 'spider':
        log.info('  - PETSc         version ' + _get_petsc_version())

    # Atmosphere module
    write = 'Atmos_clim module %s' % config.atmos_clim.module
    match config.atmos_clim.module:
        case 'janus':
            from janus import __version__ as janus_version

            write += ' version ' + janus_version
        case 'agni':
            write += ' version ' + _get_agni_version(dirs)
    log.info(write)
    if config.atmos_clim.module in ['janus', 'agni']:
        log.info('  - SOCRATES      version %s at %s' % (_get_socrates_version(), dirs['rad']))
        if config.atmos_clim.module == 'agni':
            log.info('  - Julia         version ' + _get_julia_version())

    # Outgassing module
    write = 'Outgas module     %s' % config.outgas.module
    if config.outgas.module == 'calliope':
        from calliope import __version__ as calliope_version

        write += ' version ' + calliope_version
    log.info(write)

    # Escape module
    write = 'Escape module     %s' % config.escape.module
    match config.escape.module:
        case 'zephyrus':
            from zephyrus import __version__ as zephyrus_version

            write += ' version ' + zephyrus_version
        case 'boreas':
            from boreas import __version__ as boreas_version

            write += ' version ' + boreas_version
    log.info(write)

    # Star module
    write = 'Star module       %s' % config.star.module
    if config.star.module == 'mors':
        from mors import __version__ as mors_version

        write += ' version ' + mors_version
    log.info(write)

    # Orbit module
    log.info('Orbit module      %s' % config.orbit.module)
    if config.orbit.module == 'lovepy':
        log.info('  - Julia         version ' + _get_julia_version())

    # Accretion module
    log.info('Accretion module  %s' % config.accretion.module)

    # Atmospheric chemistry module
    log.info('Atmos_chem module %s' % config.atmos_chem.module)

    # Observations synthesis module
    write = 'Observe module    %s' % config.observe.synthesis
    if config.observe.synthesis == 'platon':
        from platon import __version__ as platon_version

        write += ' version ' + platon_version
    log.info(write)

    # End spacer
    log.info(' ')


def print_citation(config: Config):
    """
    Print information on which papers should be cited.
    """

    log.info('If you use these results in a publication, please cite:')

    def _cite(key: str, url: str):
        log.info('  - ' + key + ', ' + url)

    # Core PROTEUS papers
    _cite('Lichtenberg et al. (2021)', 'https://doi.org/10.1029/2020JE006711')
    _cite('Nicholls et al. (2024)', 'https://doi.org/10.1029/2024JE008576')

    # Atmosphere module
    match config.atmos_clim.module:
        case 'janus':
            _cite('Graham et al. (2021)', 'https://doi.org/10.3847/PSJ/ac214c')
        case 'agni':
            _cite('Nicholls et al. (2025)', 'https://doi.org/10.1093/mnras/stae2772')
        case _:
            pass

    # Interior module
    match config.interior_energetics.module:
        case 'spider':
            _cite('Bower et al. (2021)', 'https://doi.org/10.3847/PSJ/ac5fb1')
        case 'aragog':
            # _cite("Bower et al. (2025)", "in prep")
            pass
        case _:
            pass

    # Outgassing module
    match config.outgas.module:
        case 'calliope':
            # Covered by Nicholls et al. (2024, 2025)
            pass
        case 'atmodeller':
            # _cite("Bower et al. (2025)", "in prep")
            pass
        case _:
            pass

    # Escape module
    match config.outgas.module:
        case 'zephyrus':
            # _cite("Postolec et al. (2025)", "in prep")
            pass
        case _:
            pass

    # Star module
    match config.star.module:
        case 'mors':
            _cite('Johnstone et al. (2021)', 'https://doi.org/10.1051/0004-6361/202038407')
        case _:
            pass

    # Orbit module
    match config.orbit.module:
        case 'lovepy':
            _cite('Hay & Matsuyama (2019)', 'https://doi.org/10.3847/1538-4357/ab0c21')
        case _:
            pass

    # Delivery module
    match config.accretion.module:
        case _:
            pass

    # Observations synthesis module
    match config.observe.synthesis:
        case 'platon':
            _cite('Zhang et al. (2024)', 'https://doi.org/10.48550/arXiv.2410.22398')
        case _:
            pass

    # Atmospheric chemistry module
    if config.atmos_chem.when != 'manually':
        match config.atmos_chem.module:
            case 'vulcan':
                _cite('Tsai et al. (2021)', 'https://doi.org/10.3847/1538-4357/ac29bc')
            case _:
                pass


def print_header():
    log.info(':::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    log.info('                   PROTEUS framework                   ')
    log.info('            Copyright (C) %4d Forming Worlds          ' % (datetime.now().year))
    log.info(':::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    log.info(' ')


def print_stoptime(start_time):
    end_time = datetime.now()
    log.info('Simulation stopped at: ' + _get_current_time())

    run_time = end_time - start_time
    run_time = run_time.total_seconds()
    if run_time > secs_per_hour:
        log.info('Total runtime: %.2f hours' % (run_time / secs_per_hour))
    elif run_time > secs_per_minute:
        log.info('Total runtime: %.2f minutes' % (run_time / secs_per_minute))
    else:
        log.info('Total runtime: %.2f seconds' % run_time)

    log.info(' ')


def assert_mass_conservation(config: Config, hf_row: dict, atol_frac: float = 1e-6) -> None:
    """Runtime invariant: M_atm <= M_planet and sum of per-species kg_atm
    matches M_atm.

    Issue #677 invariant. M_atm sums atmospheric oxygen (over gas_list of
    *_kg_atm, including the O atoms in H2O / CO2 / SO2), and M_planet =
    M_int + M_ele counts the same oxygen in M_ele, so whole-planet O
    accounting keeps the two sides symmetric and M_atm <= M_planet holds
    by construction. This assertion catches any regression that
    re-introduces an asymmetry by dropping O from one side.

    Parameters
    ----------
    hf_row : dict
        Helpfile row at the end of an iteration, after run_outgassing
        and update_planet_mass have written M_atm and M_planet.
    atol_frac : float
        Relative tolerance for the two invariants. Default 1e-6 admits
        accumulated float-rounding from the per-species sum but not
        any physically meaningful drift.

    Raises
    ------
    RuntimeError
        If M_atm > M_planet (by more than ``atol_frac`` of M_planet) or
        the per-species sum disagrees with M_atm by more than that.
    """

    if config.outgas.silicates:
        gas_list = vol_list + config.outgas.vaplist
    else:
        gas_list = vol_list + vap_list

    M_atm = float(hf_row.get('M_atm', 0.0))
    M_planet = float(hf_row.get('M_planet', 0.0))

    # Pre-IC short-circuit: M_planet == 0 means the structure solve has
    # not yet populated the hf_row. The invariants are not meaningful
    # before update_planet_mass has run, so we skip silently. The runtime
    # call site fires this AFTER update_planet_mass so M_planet > 0 in
    # normal operation; this branch only protects against direct invocation
    # in tests or in odd resume paths.
    if M_planet <= 0.0:
        return

    # Invariant 1: atmosphere mass <= total planet mass.
    if M_atm > M_planet * (1.0 + atol_frac):
        raise RuntimeError(
            f'Mass conservation violation (issue #677 regression?): '
            f'M_atm={M_atm:.3e} kg exceeds M_planet={M_planet:.3e} kg '
            f'(relative excess {(M_atm / M_planet - 1) * 100:.3f}%). '
            f'Likely cause: an aggregation site re-introduced the '
            f'"if e == \'O\': continue" skip. Check update_planet_mass, '
            f'calc_target_elemental_inventories, and load_zalmoxis_configuration.'
        )

    # Invariant 2: M_atm stays in sync with the per-species kg_atm fields it
    # is summed from. This guards against a future reordering that mutates a
    # species kg_atm after M_atm is computed without refreshing M_atm.
    summed = sum(float(hf_row.get(s + '_kg_atm', 0.0)) for s in gas_list)
    if M_atm > 0.0:
        rel = abs(summed - M_atm) / M_atm
        if rel > atol_frac:
            raise RuntimeError(
                f'M_atm bookkeeping inconsistency: M_atm={M_atm:.3e} kg but '
                f'sum_s(s_kg_atm)={summed:.3e} kg (relative difference '
                f'{rel * 100:.3f}%). One of the gas-species kg_atm fields '
                f'is stale or the M_atm sum loop is missing a species.'
            )


def PrintCurrentState(hf_row: dict):
    """
    Print the current state of the model to the logger
    """
    log.info('Runtime info...')
    log.info('    Wall time  = %s  ' % _get_current_time())
    log.info('    Model time = %.2e   yr' % float(hf_row['Time']))
    log.info('    T_surf     = %8.3f   K' % float(hf_row['T_surf']))
    log.info('    T_magma    = %8.3f   K' % float(hf_row['T_magma']))
    log.info('    P_surf     = %.2e   bar' % float(hf_row['P_surf']))
    log.info('    Phi_global = %.2e   ' % float(hf_row['Phi_global']))
    log.info('    F_atm      = %.2e   W m-2' % float(hf_row['F_atm']))
    log.info('    F_int      = %.2e   W m-2' % float(hf_row['F_int']))


def CreateLockFile(output_dir: str):
    """
    Create a lock file which, if removed, will signal for the simulation to stop.
    """
    keepalive_file = os.path.join(output_dir, LOCKFILE_NAME)
    safe_rm(keepalive_file)
    with open(keepalive_file, 'w') as fp:
        fp.write(
            'Removing this file will be interpreted by PROTEUS as a request to stop the simulation loop\n'
        )
    return keepalive_file


def GetHelpfileKeys(config: Config):
    """
    Variables to be held in the helpfile.

    All dimensional quantites should be stored in SI units, except those noted below.
    * Pressure is in units of [bar].
    * Time is in units of [years].

    """

    # fmt: off
    # Basic keys
    keys = [
        # Model tracking
        'Time',  # [yr]

        # Orbital and spin parameters of planet
        'semimajorax',      # semi-major axis [m]
        'separation',       # time-averaged separation [m]
        'perihelion',       # lowest point in orbit [m]
        'orbital_period',   # orbital duration [s]
        'eccentricity',     # orbital eccentricity [1]
        'Imk2',             # Imaginary part of k2 Love Number [1]
        'axial_period',     # day length of planet around its axis [s]

        # Satellite system
        'perigee',          # lowest point in orbit [m]
        'semimajorax_sat',  # semi-major axis [m]
        'M_sat',            # mass of satellite [kg]
        'plan_sat_am',      # angular momentum of sat+pla [kg m2 s-1],

        # Planet structure
        'R_int',            # interior radius [m]
        'M_int',            # interior mass [kg]
        'M_planet',         # total planet wet+dry mass [kg]
        'M_silicates',      # outgassed rock vapour mass , w/o oxygen [kg]
        'R_core',           # core radius [m]
        'R_solvus',         # solvus radius for global_miscibility mode [m]
        'P_solvus',         # solvus pressure for global_miscibility mode [Pa]
        'T_solvus',         # solvus temperature for global_miscibility mode [K]
        'P_center',         # central pressure from Zalmoxis structure [Pa]
        'P_cmb',            # core-mantle boundary pressure from Zalmoxis structure [Pa]
        'core_density',     # core density from structure solver [kg m-3]
        'core_heatcap',     # core heat capacity [J kg-1 K-1]
        'X_H2_int',         # H2 mass fraction in interior (sub-Neptune mode) [1]

        # Temperatures
        'T_surf',           # global surface temperature [K]
        'T_magma',          # global outgassing temperature [K]
        'T_cmb',           # core temperature [K]
        'T_eqm',            # grey radiative equilibrium temperature [K]
        'T_skin',           # grey radiative skin temperature [K]
        'T_surface_initial',  # self-consistent T_surf from accretion mode [K]
        'T_surf_accr',      # surface temperature from accretion energy balance [K]
        'T_cmb_initial',    # initial CMB temperature from White+Li thermal state [K]
        'DeltaT_accretion',  # accretion-energy DeltaT contribution [K]
        'DeltaT_adiabat',   # adiabatic DeltaT contribution [K]
        'DeltaT_differentiation',  # core-mantle differentiation DeltaT contribution [K]
        'U_grav_diff',      # gravitational binding energy (differentiated) [J]
        'U_grav_undiff',    # gravitational binding energy (undifferentiated) [J]

        # Planet energy fluxes
        'F_int',            # flux from top of interior [W m-2]
        'F_atm',            # flux from top of atmosphere [W m-2]
        'F_net',            # flux difference F_int-F_atm [W m-2]
        'F_olr',            # outgoing longwave radiation [W m-2]
        'F_sct',            # outgoing shortwave radiation [W m-2]
        'F_ins',            # incoming instellation flux [W m-2]
        'F_xuv',            # incoming XUV radiation flux [W m-2]
        'tau_atm_TOA',      # optical depth at TOA, at ref wavelength [1]
        'tau_atm_surface',  # optical depth at surface, at ref wavelength [1]
        'atm_Ra_max',      # maximum Rayleigh number across levels [1]
        'atm_t_conv_over_t_rad',  # convective vs radiative timescale ratio [1]
        'F_tidal',          # tidal heat flux arising at surface [W m-2]
        'F_radio',          # radiogenic heat flux arising at surface [W m-2]
        'F_cmb',             # heat flux at the CMB (signed, +out-of-core) [W m-2]

        # Planet interior properties
        'gravity',          # surface gravity [m s-2]
        'Phi_global',       # mantle melt mass-fraction [1]
        'Phi_global_vol',   # mantle melt volume-fraction [1]
        'RF_depth',         # depth of rheological front [1]
        'M_core',           # dry mass of core [kg]
        'M_mantle',         # dry mass of mantle [kg]
        'M_mantle_solid',   # dry mass of solid-phase mantle [kg]
        'M_mantle_liquid',  # dry mass of liquid-phase mantle [kg]
        'T_pot',            # characteristic mantle potential temperature [K]
        'boundary_layer_thickness',  # thermal boundary layer thickness [m]

        # Energy-conservation columns (Aragog A1+A2 + per-call integrals).
        # ``E_state_cons_J`` is the canonical conservation-grade integrated
        # mantle enthalpy: ``Σ h(P,S) × ρ_struct × V`` with the FROZEN
        # structural mass weighting from ``mesh.staggered_effective_density
        # × volume``. Pairs with ``dE_predicted_cons_J`` (cumulative sum
        # of ``step_dE_F_int_J + step_dE_F_cmb_J + step_dE_Q_radio_cons_J
        # + step_dE_Q_tidal_cons_J``) for the running residual:
        #   E_residual_cons_J  = (E_state_cons - E_state_cons[0]) - dE_predicted_cons_J
        #   E_residual_cons_frac = E_residual_cons_J / max(|ΔE_state_cons|, 1 J)
        # Closes to ~5 % of total cooling and ~2 % of initial reservoir
        # over multi-Myr trajectories. The state-mass enthalpy
        # ``E_state_J`` is reported as a diagnostic snapshot only; do
        # NOT use it for residual checks. State-dependent ``ρ(P,S) × V``
        # mass weighting introduces a non-conservation cross term that
        # grows with mantle cooling, so a residual built on
        # ``E_state_J`` would conflate that frame artefact with real
        # numerical drift.
        # ``solver_residual_J`` is the cumulative entropy-ODE LHS-RHS
        # residual over the trajectory and closes to machine precision
        # (~1e-7 of total cooling); it is the rigorous solver-correctness
        # check. ``E_th_mantle`` is the legacy ``m × Cp_apparent × T``
        # proxy with phase-dependent jumps in the mushy zone -- not for
        # conservation use. ``Q_radio_W`` / ``Q_tidal_W`` are instantaneous
        # mantle-integrated source powers in watts (do NOT integrate
        # trapezoidally; spike-prone at CVODE phase-boundary moments).
        # ``F_cmb`` is the analogous instantaneous CMB heat flux. The
        # conservation primitive is the per-call integral set computed by
        # Aragog over its CVODE sub-step trajectory:
        #   step_dE_F_int_J        = -∫ F_int * A_int dt   [J]
        #   step_dE_F_cmb_J        = +∫ F_cmb * A_cmb dt   [J]
        #   step_dE_Q_*_J          = +∫ Q_* dt             [J] (state-mass)
        #   step_dE_Q_*_cons_J     = +∫ Q_* dt             [J] (frozen-mass)
        #   step_solver_residual_J = ∫ (LHS - RHS) dt      [J]
        'E_th_mantle',      # legacy thermal-energy proxy [J] (do not use for conservation)
        'E_state_J',         # state-mass integrated mantle enthalpy [J] (diagnostic only)
        'E_state_cons_J',    # frozen-mass conservation-grade enthalpy [J]
        'Q_radio_W',         # instantaneous mantle-integrated radiogenic power [W]
        'Q_tidal_W',         # instantaneous mantle-integrated tidal power [W]
        'step_dE_F_int_J',   # per-call ∫ -F_int*A_int dt [J]
        'step_dE_F_cmb_J',   # per-call ∫ +F_cmb*A_cmb dt [J]
        'step_dE_Q_radio_J', # per-call ∫ +Q_radio dt [J] (state-mass, instrumentation)
        'step_dE_Q_tidal_J', # per-call ∫ +Q_tidal dt [J] (state-mass, instrumentation)
        'step_dE_Q_radio_cons_J',  # per-call ∫ +Q_radio dt [J] (frozen-mass)
        'step_dE_Q_tidal_cons_J',  # per-call ∫ +Q_tidal dt [J] (frozen-mass)
        'step_solver_residual_J',  # per-call entropy-ODE LHS-RHS [J]
        'dE_predicted_cons_J',  # cumulative sum of step_dE_*_cons_J across rows [J]
        'E_residual_cons_J',    # (E_state_cons - E_state_cons[0]) - dE_predicted_cons_J [J]
        'E_residual_cons_frac', # E_residual_cons_J / max(|ΔE_state_cons|, 1 J) [1]
        'solver_residual_J',    # cumulative entropy-ODE LHS-RHS residual [J]
        'Cp_eff',           # effective mantle heat capacity [J kg-1 K-1]

        # Host star properties
        'M_star',           # mass of star [kg]
        'R_star',           # photospheric radius [m]
        'age_star',         # age relative to deuterium fusion 'stellar birthline' [yr]
        'T_star',           # photospheric temperature [K]

        # Planet observational properties
        'p_obs',            # transit pressure level [bar]
        'R_obs',            # transit radius [m]
        'T_obs',            # transit temperature [K]
        'rho_obs',          # transit bulk density [kg m-3]
        'transit_depth',    # primary transit light curve depth [1]
        'eclipse_depth',    # secondary eclipse light curve depth [1]
        'albedo_pl',        # INPUT bond albedo from config: constant value or interpolated from table [1]
        'bond_albedo',      # OUTPUT calculated bond albedo from radtrans: SW_UP/SW_DN, zero if no scattering [1]

        # Atmospheric composition from outgassing
        'M_ele',            # total mass of tracked elements (utils.constants.element_list) rock vapour and volatile
        'M_atm',            # total mass of atmosphere [kg]
        'P_surf',           # volatile surface pressure [bar]
        'P_silicates',      #outgassed surface pressure [bar]
        'atm_kg_per_mol',   # outgassed atmosphere MMW [kg mol-1]

        # Iron-wustite buffer offset that the chemistry solver actually
        # equilibrated to, and the O mass-balance residual of that
        # equilibrium. Under planet.fO2_source = "user_constant" the
        # offset echoes the configured outgas.fO2_shift_IW (so the column
        # is single-source-of-truth for downstream analysis) and the
        # residual is zero (O is an output, not a constraint). Under
        # planet.fO2_source = "from_O_budget" the offset is the solver
        # output and the residual is the 5th element-mass residual paired
        # with the H/C/N/S residuals reported by CALLIOPE. The IW buffer
        # convention is backend-specific: CALLIOPE uses O'Neill & Eggins
        # (2002), atmodeller uses the Hirschmann combined buffer. The
        # two disagree by roughly 0.95 dex at 3000 K, so direct
        # cross-backend comparison of this column requires converting
        # one of the conventions; an independent comparison harness will
        # eventually pick a single canonical convention.
        'fO2_shift_IW_derived',  # equilibrated IW-buffer offset [log10]
        'O_res',                 # O mass-balance residual [kg]

        # Desiccation escape-balance gate. M_vol_initial is the sum over
        # all elements (oxygen included) of *_kg_total captured on the
        # first escape call, used
        # as the reference point for `outgas.wrapper.check_desiccation`'s
        # "is the loss accounted for by escape?" sanity check.
        # esc_kg_cumulative is the running sum of esc_rate_total * dt
        # over the whole run. Both must be persisted to the CSV so
        # resume preserves the gate's state.
        'M_vol_initial',    # bulk volatile inventory baseline [kg]
        'esc_kg_cumulative', # cumulative escaped mass [kg]
        ]

    if config.outgas.silicates:
        gas_list = vol_list + config.outgas.vaplist
    else:
        gas_list = vol_list + vap_list

    # gases from outgassing
    for s in gas_list:
        keys.append(s + '_mol_atm')     # number outgassed to atmosphere [mol]
        keys.append(s + '_mol_solid')   # number in solid mantle [mol]
        keys.append(s + '_mol_liquid')  # number in liquid mantle [mol]
        keys.append(s + '_mol_total')   # number in whole planet [mol]
        keys.append(s + '_kg_atm')      # mass outgassed to atmosphere [kg]
        keys.append(s + '_kg_solid')    # mass in solid mantle [kg]
        keys.append(s + '_kg_liquid')   # mass in liquid mantle [kg]
        keys.append(s + '_kg_total')    # mass in whole planet [kg]
        keys.append(s + '_vmr')         # outgassed volume mixing ratio [1]
        keys.append(s + '_bar')         # partial surface pressure [bar]
        keys.append(s + '_vmr_xuv')     # volume mixing ratio at XUV level [1]

    # quantities for each element
    for e in element_list:
        if e not in gas_list:
            keys.append(e + '_kg_atm')      # mass outgassed to atmosphere [kg]
            keys.append(e + '_kg_solid')    # mass in solid mantle [kg]
            keys.append(e + '_kg_liquid')   # mass in liquid mantle [kg]
            keys.append(e + '_kg_total')    # mass in whole planet [kg]

    # element mass ratios in atmosphere
    for e1 in element_list:
        for e2 in element_list:
            # do not add reversed ratios
            if (e1 == e2) or (f'{e1}/{e2}_atm' in keys):
                continue
            # add ratio of e2 to e1 (e.g. C/O, but not O/C)
            keys.append(f'{e2}/{e1}_atm')

    # Atmospheric escape
    keys.append('p_xuv')                # pressure of XUV absorption [bar]
    keys.append('R_xuv')                # radius of XUV absorption [m]
    keys.append('cs_xuv')               # sound speed, at R_xuv [m s-1]
    keys.append('esc_rate_total')       # bulk escape rate [kg s-1]
    for e in element_list:
        keys.append('esc_rate_' + e)    # escape rate of each element [kg s-1]

    # Climate quantities
    keys.append('P_surf_clim')          # total surface pressure, after rainout [bar]
    keys.append('ocean_areacov')        # ocean surface-area fraction [1]
    keys.append('ocean_maxdepth')       # maximum depth of oceans [m]
    for s in gas_list:
        keys.append(s + '_ocean')       # ocean surface density [kg m-2]

    # Diagnostic variables
    keys.append('wtg_surf')         # Weak temperature gradient parameter at the surface [1]
    keys.append('roche_limit')      # Roche limit, orbital distance  [m]
    keys.append('breakup_period')   # Critical day length [s]
    keys.append('hill_radius')      # Hill radius, radial distance [m]

    # Simulation's computational variables
    keys.append('runtime')          # Simulation wall-clock runtime [s]
    keys.append("fO2_shift_LavAtmos") #relative to IW buffer

    return keys


def CreateHelpfileFromDict(d: dict, config: Config):
    """
    Create helpfile to hold output variables.
    """
    log.debug('Creating new helpfile from dict')
    return pd.DataFrame([d], columns=GetHelpfileKeys(config), dtype=float)


def ZeroHelpfileRow(config: Config):
    """
    Get a dictionary with same keys as helpfile but with values of zero
    """
    out = {}
    for k in GetHelpfileKeys(config):
        out[k] = 0.0
    return out


def _populate_energy_residual(current_hf: pd.DataFrame, new_row: dict) -> None:
    """Fill the cumulative energy-conservation columns of ``new_row`` in place.

    The conservation primitive is the per-call energy integral set
    computed by Aragog over its CVODE sub-step trajectory:

        step_dE_F_int_J        = -∫ F_int * A_int dt   over the call [J]
        step_dE_F_cmb_J        = +∫ F_cmb * A_cmb dt   over the call [J]
        step_dE_Q_*_cons_J     = +∫ Q_* dt             [J] (frozen-mass)
        step_solver_residual_J = ∫ (LHS - RHS) dt      [J]

    The cumulative ``dE_predicted_cons_J`` is the running sum of the
    flux+source integrals across all helpfile rows. This eliminates the
    previous helpfile-side trapezoidal interpolation between
    end-of-step F_cmb snapshots, which was prone to phase-boundary
    spikes: a single CVODE sub-step transient could blow up the
    integral by orders of magnitude when used as a trapezoid endpoint
    over a PROTEUS iteration's worth of time.

    Row 0 sets all cumulative columns to zero by definition.
    ``E_residual_cons_frac`` normalises by ``max(|ΔE_state_cons|, 1 J)``
    so the residual stays bounded when both numerator and denominator
    are tiny (quiescent steady state). Closes to ~5 % of total cooling
    over multi-Myr trajectories.

    ``solver_residual_J`` is the running entropy-ODE LHS-RHS check;
    closes to machine precision (~1e-7 of total cooling) and flags
    real CVODE step rejection or atol/rtol issues if it drifts.

    Active only when ``E_state_cons_J`` is finite and non-zero,
    signalling that an EOS-aware interior module populated it. Other
    modules leave the column at 0.0 (from ZeroHelpfileRow) and the
    residual columns stay at 0.0 too.

    The frozen-mass framing is required for the residual to close.
    A state-mass alternative (``ρ(P,S) × V`` re-evaluated each step)
    would carry a non-conservation cross term that grows with mantle
    cooling and masks real numerical drift.
    """
    e_state_cons_now = float(new_row.get('E_state_cons_J', 0.0))
    if not np.isfinite(e_state_cons_now) or e_state_cons_now == 0.0:
        for k in (
            'dE_predicted_cons_J',
            'E_residual_cons_J',
            'E_residual_cons_frac',
            'solver_residual_J',
        ):
            new_row.setdefault(k, 0.0)
        return

    # Per-call energy increment from Aragog [J]. Sign is already baked
    # into each step delta (positive = energy added to mantle). Uses
    # frozen-mass Q_*_cons to pair with the frozen-mass E_state_cons_J.
    dE_inc_cons = (
        float(new_row.get('step_dE_F_int_J', 0.0))
        + float(new_row.get('step_dE_F_cmb_J', 0.0))
        + float(new_row.get('step_dE_Q_radio_cons_J', 0.0))
        + float(new_row.get('step_dE_Q_tidal_cons_J', 0.0))
    )
    solver_inc = float(new_row.get('step_solver_residual_J', 0.0))

    n_prior = len(current_hf)
    if n_prior == 0:
        # Anchor row: cumulative integrals start at zero by definition.
        new_row['dE_predicted_cons_J'] = 0.0
        new_row['E_residual_cons_J'] = 0.0
        new_row['E_residual_cons_frac'] = 0.0
        new_row['solver_residual_J'] = 0.0
        return

    prev = current_hf.iloc[-1]
    dE_pred_cons_prev = float(prev.get('dE_predicted_cons_J', 0.0))
    dE_pred_cons_now = dE_pred_cons_prev + dE_inc_cons

    # Anchor the cumulative actual-energy change on the first populated
    # E_state_cons_J. A 0.0 entry marks a row written before this column
    # existed; anchoring on it would fold the entire absolute mantle enthalpy
    # into the residual.
    e_state_series = current_hf['E_state_cons_J']
    valid_anchor = e_state_series[(e_state_series != 0.0) & np.isfinite(e_state_series)]
    e_state_cons_anchor = float(
        valid_anchor.iloc[0] if len(valid_anchor) > 0 else e_state_series.iloc[0]
    )
    dE_actual_cons_now = e_state_cons_now - e_state_cons_anchor
    residual_cons_now = dE_actual_cons_now - dE_pred_cons_now

    new_row['dE_predicted_cons_J'] = dE_pred_cons_now
    new_row['E_residual_cons_J'] = residual_cons_now
    new_row['E_residual_cons_frac'] = residual_cons_now / max(abs(dE_actual_cons_now), 1.0)

    # Cumulative entropy-ODE solver residual.
    solver_resid_prev = float(prev.get('solver_residual_J', 0.0))
    new_row['solver_residual_J'] = solver_resid_prev + solver_inc


def ExtendHelpfile(current_hf: pd.DataFrame, new_row: dict, config: Config):
    """
    Extend helpfile with new row of variables

    """
    log.debug('Extending helpfile with new row')

    # ── Energy-conservation cumulative bookkeeping ─────────────────────
    # Compute dE_predicted_cons_J / E_residual_cons_J / E_residual_cons_frac
    # / solver_residual_J for the new row from the prior-row state stored
    # in current_hf and the per-call integrals populated by the active
    # interior module. Active only when E_state_cons_J is finite and
    # non-zero (Aragog with the entropy EOS path); other interior modules
    # leave the column at 0.0 via ZeroHelpfileRow and the residual
    # columns stay at 0.0 too, signalling "diagnostic not available for
    # this run" to downstream plotting.
    _populate_energy_residual(current_hf, new_row)

    # Validate keys. We guard in both directions:
    # - Missing keys (schema expects but new_row lacks) are a real bug (a
    #   module forgot to set a value) and must raise.
    # - Unknown keys (new_row has but schema doesn't) are silently dropped
    #   by `columns=GetHelpfileKeys()` in the DataFrame construction, which
    #   means resume would lose those values. We WARN here rather than raise
    #   so existing hf_row private/transient fields (_structure_stale, etc.)
    #   and string-valued fields (core_state_initial) don't break runs.
    # Private (underscore-prefixed) keys are intentionally transient and
    # are excluded from both checks.

    schema = set(GetHelpfileKeys(config))

    row_keys = {k for k in new_row.keys() if not k.startswith('_')}
    # Known non-numeric / non-persistent keys that are written into hf_row
    # but deliberately not tracked in the helpfile CSV schema.
    _ALLOWED_NON_SCHEMA_KEYS = frozenset(
        {
            'core_state_initial',  # string: 'liquid'/'mixed'/'solid'
            # IC consistency sentinel for the issue #677 oxygen-budget
            # check. Set by calc_target_elemental_inventories, consumed
            # and reset to -1.0 by check_ic_oxygen_budget on the first
            # outgas call. Intentionally not persisted to the CSV
            # because subsequent runs (or resumed runs) re-derive it
            # from the config on the next IC pass.
            'O_kg_user_ic',
        }
    )
    missing_keys = schema - row_keys
    unknown_keys = row_keys - schema - _ALLOWED_NON_SCHEMA_KEYS
    if missing_keys:
        raise Exception('Helpfile row is missing expected keys: %s' % missing_keys)
    if unknown_keys:
        log.warning(
            'Helpfile row contains keys not declared in GetHelpfileKeys(config) '
            '(they will be silently dropped from the CSV): %s. '
            'Either add them to the schema or explicitly allowlist them in '
            '_ALLOWED_NON_SCHEMA_KEYS.',
            sorted(unknown_keys),
        )

    # convert row to df, only including keys in the schema
    # which is defined by GetHelpfileKeys(config)
    new_row = pd.DataFrame([new_row], columns=GetHelpfileKeys(config), dtype=float)
    # Check for NaN values. Print warning if any are found and convert to zero.
    for col in new_row.columns:
        if new_row[col].isna().any():
            log.warning(
                'hf_row[%s] is NaN at t=%.2e years; setting to zero.',
                col,
                new_row['Time'].iloc[0],
            )
            new_row[col] = new_row[col].fillna(0.0)

    # concatenate and return
    return pd.concat([current_hf, new_row], ignore_index=True)


def WriteHelpfileToCSV(output_dir: str, current_hf: pd.DataFrame, config: Config):
    """
    Write helpfile to a CSV file
    """
    log.debug('Writing helpfile to CSV file')

    # check for invalid or missing keys
    difference = set(GetHelpfileKeys(config)) - set(current_hf.keys())
    if len(difference) > 0:
        raise Exception('There are mismatched keys in helpfile: ' + str(difference))

    # remove old file
    fpath = os.path.join(output_dir, 'runtime_helpfile.csv')
    if os.path.exists(fpath):
        os.remove(fpath)

    # write new file
    current_hf.to_csv(fpath, index=False, sep='\t', float_format='%.10e')
    return fpath


def ReadHelpfileFromCSV(output_dir: str):
    """
    Read helpfile from disk CSV file to DataFrame
    """
    fpath = os.path.join(output_dir, 'runtime_helpfile.csv')
    if not os.path.exists(fpath):
        raise Exception("Cannot find helpfile at '%s'" % fpath)
    return pd.read_csv(fpath, sep=r'\s+')


def variable_is_logarithmic(varname: str) -> bool:
    """Does this variable naturally vary across orders several of magnitude?

    This variable should also be positive-valued.

    Parameters
    ----------
    - varname (str): Name of variable.

    Returns
    ----------
    - out (bool): True if scales logarithmically.
    """

    # Linear-scaling is default behaviour
    out = False

    # Check specific variables
    if varname in (
        'P_surf',
        'P_surf_clim',
        'rho_obs',
        'p_obs',
        'p_xuv',
        'Time',
        'semimajorax',
        'eccentricity',
        'params.stop.time.maximum',
        'orbit.semimajoraxis',
    ):
        out = True

    # Check compositional variables
    elif '_vmr' in varname:
        out = True
    elif '_bar' in varname:
        out = True

    return out


def UpdatePlots(hf_all: pd.DataFrame, dirs: dict, config: Config, end=False, num_snapshots=7):
    """Update plots during runtime for analysis

    Calls various plotting functions which show information about the
    interior/atmosphere's energy and composition.

    Parameters
    ----------
    hf_all : pd.DataFrame
        Dataframe containing all the output data
    dirs : dict
        Dictionary of directories
    config : Config
        PROTEUS configuration object
    end : bool
        Is this function being called at the end of the simulation?
    num_snapshots : int
        Number of snapshots to include in each plot.
    """

    # Import utilities
    from proteus.atmos_clim.common import read_atmosphere_data
    from proteus.interior_energetics.wrapper import read_interior_data

    # Import plotting functions
    from proteus.plot.cpl_atmosphere import plot_atmosphere
    from proteus.plot.cpl_bolometry import plot_bolometry
    from proteus.plot.cpl_chem_atmosphere import plot_chem_atmosphere
    from proteus.plot.cpl_emission import plot_emission
    from proteus.plot.cpl_escape import plot_escape
    from proteus.plot.cpl_fluxes_atmosphere import plot_fluxes_atmosphere
    from proteus.plot.cpl_fluxes_global import plot_fluxes_global
    from proteus.plot.cpl_global import plot_global
    from proteus.plot.cpl_interior import plot_interior
    from proteus.plot.cpl_interior_cmesh import plot_interior_cmesh
    from proteus.plot.cpl_orbit import plot_orbit
    from proteus.plot.cpl_population import (
        plot_population_mass_radius,
        plot_population_time_density,
    )
    from proteus.plot.cpl_sflux import plot_sflux
    from proteus.plot.cpl_sflux_cross import plot_sflux_cross
    from proteus.plot.cpl_spectra import plot_spectra
    from proteus.plot.cpl_structure import plot_structure
    # from proteus.plot.cpl_visual import plot_visual

    # Directories
    output_dir = dirs['output']
    fwl_dir = dirs['fwl']

    # Check model configuration
    dummy_atm = config.atmos_clim.module == 'dummy'
    dummy_int = config.interior_energetics.module == 'dummy'
    # agni = config.atmos_clim.module == 'agni'
    spider = config.interior_energetics.module == 'spider'
    aragog = config.interior_energetics.module == 'aragog'
    observed = bool(config.observe.synthesis is not None)

    # Get all output times
    output_times = []
    plot_times = []
    if spider:
        from proteus.interior_energetics.spider import get_all_output_times

        output_times = get_all_output_times(output_dir)
    if aragog:
        from proteus.interior_energetics.aragog import get_all_output_times

        output_times = get_all_output_times(output_dir)

    # Global properties for all timesteps
    plot_global(hf_all, output_dir, config)

    # Elemental mass inventory
    plot_escape(hf_all, output_dir, plot_format=config.params.out.plot_fmt)

    # Planet and satellite orbit parameters
    if config.orbit.evolve or config.orbit.satellite:
        plot_orbit(hf_all, output_dir, config.params.out.plot_fmt)

    # Which times do we have atmosphere data for?
    if not dummy_atm:
        ncs = glob.glob(os.path.join(output_dir, 'data', '*_atm.nc'))
        nc_times = [int(f.split('/')[-1].split('_atm')[0]) for f in ncs]

        # Check intersection of atmosphere and interior data
        if dummy_int:
            output_times = nc_times
        else:
            output_times = sorted(list(set(output_times) & set(nc_times)))

    # Samples for plotting profiles
    if len(output_times) > 0:
        tmin = 1.0
        if np.amax(output_times) > 1e3:
            tmin = 1e3
        plot_times, _ = sample_times(output_times, num_snapshots, tmin=tmin)
        log.debug('Snapshots to plot:' + str(plot_times))

        # Interior profiles
        if not dummy_int:
            int_data = read_interior_data(
                output_dir, config.interior_energetics.module, plot_times
            )
            plot_interior(
                output_dir,
                plot_times,
                int_data,
                config.interior_energetics.module,
                config.params.out.plot_fmt,
            )

        # Atmosphere profiles
        if not dummy_atm:
            atm_data = read_atmosphere_data(output_dir, plot_times)

            # Atmosphere temperature/height profiles
            plot_atmosphere(output_dir, plot_times, atm_data, config.params.out.plot_fmt)

            # Atmospheric chemistry
            plot_chem_atmosphere(
                output_dir,
                config.atmos_chem.module,
                plot_format=config.params.out.plot_fmt,
                plot_offchem=False,
            )

            # Atmosphere and interior, stacked radially
            if not dummy_int:
                plot_structure(
                    hf_all,
                    output_dir,
                    plot_times,
                    int_data,
                    atm_data,
                    config.interior_energetics.module,
                    config.params.out.plot_fmt,
                )

            # Energy flux profiles
            plot_fluxes_atmosphere(output_dir, config.params.out.plot_fmt)

    # Only at the end of the simulation
    if end:
        # Global plot with linear-time axis
        plot_global(hf_all, output_dir, config, logt=False)

        # Energy flux balance
        plot_fluxes_global(hf_all, output_dir, config)

        # Bolometric observables
        plot_bolometry(hf_all, output_dir, plot_format=config.params.out.plot_fmt)

        # Spectral observables
        if observed:
            plot_spectra(output_dir, plot_format=config.params.out.plot_fmt)

        # Chemical profiles
        if not dummy_atm:
            plot_chem_atmosphere(
                output_dir, config.atmos_chem.module, plot_format=config.params.out.plot_fmt
            )
            atm_data = read_atmosphere_data(output_dir, plot_times)
            plot_fluxes_atmosphere(output_dir, config.params.out.plot_fmt)
            plot_atmosphere(output_dir, plot_times, atm_data, config.params.out.plot_fmt)

        # Visualise planet and star
        # if agni:
        # plot_visual(hf_all, output_dir, idx=-1, plot_format=config.params.out.plot_fmt)

        # Check that the simulation ran for long enough to make useful plots
        if len(hf_all['Time']) >= 3:
            plot_population_mass_radius(hf_all, output_dir, fwl_dir, config.params.out.plot_fmt)
            plot_population_time_density(
                hf_all, output_dir, fwl_dir, config.params.out.plot_fmt
            )

            plt_modern = bool(config.star.module == 'mors')
            if plt_modern:
                modern_age = config.star.mors.age_now * 1e9
            else:
                modern_age = -1
            plot_sflux(
                output_dir, plt_modern=plt_modern, plot_format=config.params.out.plot_fmt
            )
            plot_sflux_cross(
                output_dir, modern_age=modern_age, plot_format=config.params.out.plot_fmt
            )

            if plot_times and not dummy_int:
                plot_interior_cmesh(
                    output_dir,
                    plot_times,
                    int_data,
                    config.interior_energetics.module,
                    plot_format=config.params.out.plot_fmt,
                )

            if plot_times and not dummy_atm:
                plot_emission(output_dir, plot_times, plot_format=config.params.out.plot_fmt)

    # Close all figures
    plt.close('all')


def remove_excess_files(outdir: str, rm_spectralfiles: bool = False):
    """Remove excess files from the output directory

    Parameters
    ----------
    outdir: str
        Path to the simulation's output directory
    rm_spectralfiles: bool
        Whether to remove spectral files
    """

    # Files to remove, relative to outdir
    rm_paths = ['agni_recent.log', 'data/.spider_tmp', LOCKFILE_NAME]

    # Remove spectral files if requested
    if rm_spectralfiles:
        rm_paths.append('runtime.sf')
        rm_paths.append('runtime.sf_k')

    # Loop over files
    for f in rm_paths:
        f = os.path.join(outdir, f)

        # Remove the file
        log.debug(f'Removing {f}')
        safe_rm(f)


def get_proteus_directories(outdir='_unset') -> dict[str, str]:
    """Create dict of proteus directories from root dir.

    Parameters
    ----------
    outdir : str
        Name of the simulation's output directory

    Returns
    -------
    dirs : dict[str, str]
        Proteus directories dict
    """
    root_dir = get_proteus_dir()

    return {
        'proteus': root_dir,
        'agni': os.path.join(root_dir, 'AGNI'),
        'lovepy': os.path.join(root_dir, 'lovepy'),
        'input': os.path.join(root_dir, 'input'),
        'spider': os.path.join(root_dir, 'SPIDER'),
        'aragog': os.path.join(root_dir, 'aragog'),
        'zalmoxis': os.path.join(root_dir, 'Zalmoxis'),
        'vulcan': os.path.join(root_dir, 'VULCAN'),
        'tools': os.path.join(root_dir, 'tools'),
        'utils': os.path.join(root_dir, 'src', 'proteus', 'utils'),
        'output': os.path.join(root_dir, 'output', outdir),
        'output/data': os.path.join(root_dir, 'output', outdir, 'data'),
        'output/observe': os.path.join(root_dir, 'output', outdir, 'observe'),
        'output/offchem': os.path.join(root_dir, 'output', outdir, 'offchem'),
        'output/plots': os.path.join(root_dir, 'output', outdir, 'plots'),
    }


def set_directories(config: Config) -> dict[str, str]:
    """Set directories dictionary

    Sets paths to the required directories, based on the configuration provided
    by the options dictionary.

    Parameters
    ----------
    config : Config
        PROTEUS options dictionary

    Returns
    ----------
    dirs : dict
        Dictionary of paths to important directories
    """
    # Resolve 'auto' path to a timestamped unique name.
    # Note: this mutates config.params.out.path so that Config.write()
    # records the resolved name in init_coupler.toml (intentional).
    outdir = config.params.out.path
    if outdir == 'auto':
        import secrets
        from datetime import datetime

        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = secrets.token_hex(2)  # 4 hex chars
        outdir = f'run_{stamp}_{suffix}'
        config.params.out.path = outdir
        log.info('Auto output path: %s', outdir)

    dirs = get_proteus_directories(outdir=outdir)

    # FWL data folder
    if os.environ.get('FWL_DATA') is None:
        UpdateStatusfile(dirs, 20)
        raise EnvironmentError(
            'The FWL_DATA environment variable has not been set. '
            'See https://proteus-framework.org/PROTEUS/How-to/installation.html'
        )
    else:
        dirs['fwl'] = os.environ.get('FWL_DATA')

    # SOCRATES directory
    if config.atmos_clim.module in ('janus', 'agni'):
        # needed for atmosphere models 0 and 1

        if os.environ.get('RAD_DIR') is None:
            UpdateStatusfile(dirs, 20)
            raise EnvironmentError(
                'The RAD_DIR environment variable has not been set (required by AGNI/JANUS). '
                'See https://proteus-framework.org/PROTEUS/How-to/installation.html'
            )
        else:
            dirs['rad'] = os.environ.get('RAD_DIR')

    # Temporary directory
    if config.params.out.logging == 'DEBUG':
        dirs['temp'] = dirs['output']
    else:
        dirs['temp'] = create_tmp_folder()
    log.info(f'Temporary-file working dir: {dirs["temp"]}')

    # Get abspaths
    for key in dirs.keys():
        dirs[key] = os.path.abspath(dirs[key]) + '/'

    return dirs
