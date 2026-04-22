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

from proteus.utils.constants import element_list, gas_list, secs_per_hour, secs_per_minute
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
    verpath = os.path.join(os.environ.get('RAD_DIR'), 'version')
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


def GetHelpfileKeys():
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
        'R_core',           # core radius [m]
        'R_solvus',         # solvus radius for global_miscibility mode [m]
        'P_solvus',         # solvus pressure for global_miscibility mode [Pa]
        'T_solvus',         # solvus temperature for global_miscibility mode [K]
        'P_center',         # central pressure from Zalmoxis structure [Pa]
        'core_density',     # core density from structure solver [kg m-3]
        'core_heatcap',     # core heat capacity [J kg-1 K-1]
        'X_H2_int',         # H2 mass fraction in interior (sub-Neptune mode) [1]

        # Temperatures
        'T_surf',           # global surface temperature [K]
        'T_magma',          # global outgassing temperature [K]
        'T_core',           # core temperature [K]
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
        'F_tidal',          # tidal heat flux arising at surface [W m-2]
        'F_radio',          # radiogenic heat flux arising at surface [W m-2]

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
        'E_th_mantle',      # total mantle thermal energy [J]
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
        'M_ele',            # total mass of tracked elements (utils.constants.element_list)
        'M_atm',            # total mass of atmosphere [kg]
        'P_surf',           # total surface pressure [bar]
        'atm_kg_per_mol',   # outgassed atmosphere MMW [kg mol-1]

        # Desiccation escape-balance gate (commit ea22c07b, 2026-04-08).
        # M_vol_initial is the sum of non-oxygen *_kg_total captured on the
        # first escape call, used as the reference point for
        # `outgas.wrapper.check_desiccation`'s "is the loss accounted for by
        # escape?" sanity check. esc_kg_cumulative is the running sum of
        # esc_rate_total * dt over the whole run. Both must be persisted to
        # the CSV so resume preserves the gate's state.
        'M_vol_initial',    # bulk volatile inventory baseline [kg]
        'esc_kg_cumulative', # cumulative escaped mass [kg]

        # Redox scaffolding (#57; src/proteus/redox). All zero in
        # `redox.mode = 'static'` (default); populated by the redox
        # solver in `fO2_init` / `composition` modes.
        'fO2_shift_IW',     # ΔIW ≡ log10(fO2) − log10(fO2_IW) at surface
                            # [log10 units]. Written by redox.solver
                            # (or pinned to config.outgas.fO2_shift_IW in
                            # static mode); read by CALLIOPE / atmodeller
                            # / SPIDER / Zalmoxis consumers unchanged.
        'fO2_dIW',          # diagnostic log10 fO2 relative to IW buffer
        'fO2_dQFM',         # diagnostic log10 fO2 relative to QFM buffer
        'fO2_dNNO',         # diagnostic log10 fO2 relative to NNO buffer
        'R_budget_atm',     # atmosphere redox budget [mol e-] (Evans 2012)
        'R_budget_mantle',  # mantle redox budget [mol e-]
        'R_budget_core',    # core redox budget [mol e-]
        'R_budget_total',   # sum over reservoirs; conserved quantity
        'R_escaped_cum',    # cumulative signed ΔR_atm impulse from
                            # escape over the whole run [mol e-].
                            # NOT the signed "mol e- removed" — the
                            # sign follows the debit rule (+ for
                            # reducing-species escape, - for
                            # oxidising-species escape). See
                            # redox/conservation.py::debit_escape.
        'redox_conservation_residual',  # |ΔR_total − ΔR_escape − ΔR_dispro|
                            # per step; soft-fail ceiling at redox.soft_tol
        'redox_solver_fallback_count',  # cumulative Brent failures
        'redox_mode_active',  # enum (static=0 / fO2_init=1 / composition=2)
        # Fe speciation state (global bulk, mantle-melt reservoir)
        'Fe3_frac',         # Fe³⁺/ΣFe in the bulk melt [1]
        'FeO_total_wt',     # total Fe as FeO-equivalent in mantle [wt%]
        'Fe2O3_wt',         # Fe₂O₃ in mantle [wt%]; derived from Fe3_frac
        'n_Fe3_melt',       # moles Fe³⁺ in the global MO melt reservoir
        'n_Fe2_melt',       # moles Fe²⁺
        'n_Fe3_solid_total',  # Σ per-cell; full profile in interior NetCDF
        'n_Fe2_solid_total',
        'n_Fe0_solid_total',  # disproportionation metal still in mantle
        'dm_Fe0_to_core_cum',  # cumulative Fe⁰ drained to core [kg]
                              # (# 653 Mariana metal-saturation hook)
        'redox_delta_IW_suggested_by_mariana',  # ΔIW warm-start from
                              # Mariana's advance_fe_reservoirs (NaN when
                              # Fe3_frac=0). Persisted for resume so the
                              # Brent solver has a warm start after
                              # `proteus start -r`. NOT log10 fO2 — the
                              # coupling layer converts via log10_fO2_IW
                              # at the melt surface before writing.
    ]

    # quantities for each gas, from outgassing
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
        keys.append(e + '_kg_atm')      # mass outgassed to atmosphere [kg]
        keys.append(e + '_kg_solid')    # mass in solid mantle [kg]
        keys.append(e + '_kg_liquid')   # mass in liquid mantle [kg]
        keys.append(e + '_kg_total')    # mass in whole planet [kg]
        keys.append(e + '_kg_core')     # mass in metallic core [kg] (#57 redox;
                                        # zero in static mode; populated by
                                        # #526 metal-silicate partitioning hook)

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

    # fmt: on
    return keys


def CreateHelpfileFromDict(d: dict):
    """
    Create helpfile to hold output variables.
    """
    log.debug('Creating new helpfile from dict')
    return pd.DataFrame([d], columns=GetHelpfileKeys(), dtype=float)


def ZeroHelpfileRow():
    """
    Get a dictionary with same keys as helpfile but with values of zero
    """
    out = {}
    for k in GetHelpfileKeys():
        out[k] = 0.0
    return out


def ExtendHelpfile(current_hf: pd.DataFrame, new_row: dict):
    """
    Extend helpfile with new row of variables
    """
    log.debug('Extending helpfile with new row')

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
    schema = set(GetHelpfileKeys())
    row_keys = {k for k in new_row.keys() if not k.startswith('_')}
    # Known non-numeric / non-persistent keys that are written into hf_row
    # but deliberately not tracked in the helpfile CSV schema.
    _ALLOWED_NON_SCHEMA_KEYS = frozenset(
        {
            'core_state_initial',  # string: 'liquid'/'mixed'/'solid'
        }
    )
    missing_keys = schema - row_keys
    unknown_keys = row_keys - schema - _ALLOWED_NON_SCHEMA_KEYS
    if missing_keys:
        raise Exception('Helpfile row is missing expected keys: %s' % missing_keys)
    if unknown_keys:
        log.warning(
            'Helpfile row contains keys not declared in GetHelpfileKeys() '
            '(they will be silently dropped from the CSV): %s. '
            'Either add them to the schema or explicitly allowlist them in '
            '_ALLOWED_NON_SCHEMA_KEYS.',
            sorted(unknown_keys),
        )

    # convert row to df
    new_row = pd.DataFrame([new_row], columns=GetHelpfileKeys(), dtype=float)

    # concatenate and return
    return pd.concat([current_hf, new_row], ignore_index=True)


def WriteHelpfileToCSV(output_dir: str, current_hf: pd.DataFrame):
    """
    Write helpfile to a CSV file
    """
    log.debug('Writing helpfile to CSV file')

    # check for invalid or missing keys
    difference = set(GetHelpfileKeys()) - set(current_hf.keys())
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
    Read helpfile from disk CSV file to DataFrame.

    Legacy CSVs written before the redox scaffolding (#57) lack the
    new `fO2_shift_IW`, `R_budget_*`, `Fe*`, `n_Fe*`, and `*_kg_core`
    columns. On resume we seed any missing column to zero so downstream
    code receives a DataFrame with the full current schema. The seeded
    zeros are harmless in `redox.mode = 'static'` (the default);
    evolving modes bootstrap from these zeros just like a fresh run.
    """
    fpath = os.path.join(output_dir, 'runtime_helpfile.csv')
    if not os.path.exists(fpath):
        raise Exception("Cannot find helpfile at '%s'" % fpath)
    df = pd.read_csv(fpath, sep=r'\s+')
    expected = set(GetHelpfileKeys())
    missing = expected - set(df.columns)
    if missing:
        log.info(
            'ReadHelpfileFromCSV: seeding %d missing columns with zero '
            '(legacy CSV pre-#57): %s',
            len(missing),
            sorted(missing),
        )
        # Single concat avoids PerformanceWarning about fragmentation.
        zero_block = pd.DataFrame(
            {col: 0.0 for col in sorted(missing)},
            index=df.index,
        )
        df = pd.concat([df, zero_block], axis=1)
    return df


def SeedRedoxDefaultsOnResume(hf_row: dict, config: 'Config') -> None:
    """
    Fill in sane defaults for redox-scaffolding keys on resume from a
    legacy CSV (#57).

    `ReadHelpfileFromCSV` seeds missing columns to zero. For `fO2_shift_IW`
    specifically, zero is a valid physical value (IW buffer with no
    offset) so we cannot use zero as a "missing" sentinel. Instead we
    check whether the column existed on disk (tracked by the caller)
    and, when it did not, fill from `config.outgas.fO2_shift_IW`.

    Call this after `ReadHelpfileFromCSV` when resuming, before the
    first step.
    """
    # In practice the simple rule "if fO2_shift_IW is exactly 0.0 after
    # migration, override from config" is too aggressive because 0.0 is
    # a valid value. The safer approach is to check whether the CSV
    # file on disk had the column; we do this in the caller. For now
    # this function documents the intended API; the caller supplies
    # an explicit list of keys to reset.
    # TODO(#57 Commit C): refine once the redox solver is wired in.
    default_fO2 = float(getattr(config.outgas, 'fO2_shift_IW', 0.0))
    if hf_row.get('fO2_shift_IW', 0.0) == 0.0 and default_fO2 != 0.0:
        hf_row['fO2_shift_IW'] = default_fO2


# O-atom count per supported gas species. Used by
# :func:`compute_O_kg_from_species` to surface the atmospheric /
# melt / solid O inventory that CALLIOPE & atmodeller compute
# implicitly via the fO2 buffer (#57 Commit D).
#
# SCOPING (Commit D.1, round-7 physics review):
# - `vol_list` species (H2O, CO2, ...) are atmospheric volatiles.
#   Their O inventory is 'mobile O' — moves between atmosphere,
#   melt, and solid via outgassing/dissolution. We count these in
#   every reservoir.
# - `vap_list` species (SiO, SiO2, MgO, FeO2) are silicate vapours.
#   Their `_mol_atm` entry represents genuine mobile O (silicate
#   atoms in the atmosphere after vaporisation). Their `_mol_liquid`
#   and `_mol_solid` entries, if a backend ever writes them, are
#   silicate-bound O ALREADY INSIDE `M_int` (the interior mass from
#   Aragog/SPIDER/Zalmoxis). Counting them there would double-count
#   silicate O.
# - We therefore count vap_list species in `atm` only.
_SPECIES_O_COUNT_VOL: dict[str, int] = {
    'H2O': 1,
    'CO2': 2,
    'CO': 1,
    'SO2': 2,
    'O2': 2,
    # H2, CH4, N2, NH3, S2, H2S all contain zero oxygen atoms.
}

_SPECIES_O_COUNT_VAP_ATM_ONLY: dict[str, int] = {
    'SiO': 1,
    'SiO2': 2,
    'MgO': 1,
    'FeO2': 2,
}

# Legacy alias kept for test backward-compat; the union dict still
# includes vap_list, but `compute_O_kg_from_species` restricts their
# inclusion to the `atm` reservoir.
_SPECIES_O_COUNT = {**_SPECIES_O_COUNT_VOL, **_SPECIES_O_COUNT_VAP_ATM_ONLY}


def compute_O_kg_from_species(hf_row: dict, reservoir: str) -> float:
    """
    Sum O-atom mass across supported O-bearing species in the named
    reservoir (#57 Commit D / D.1).

    Uses `{species}_mol_{reservoir}` when available; falls back to
    `{species}_kg_{reservoir}` / MW_species when the mol column is
    absent or zero. The fallback is needed because atmodeller's
    `calc_surface_pressures_atmodeller` writes only `_kg_*` columns,
    not `_mol_*` (round-7 review showstopper).

    Vap-list species (SiO, SiO2, MgO, FeO2) contribute ONLY to the
    `atm` reservoir. Their liquid/solid forms represent silicate-
    bound O that is already inside `M_int`; counting them here
    would double-count silicate O (round-7 review showstopper).

    Parameters
    ----------
    hf_row : dict
        Live helpfile row with `{species}_mol_{reservoir}` OR
        `{species}_kg_{reservoir}` populated by the most recent
        outgas / climate step.
    reservoir : str
        One of 'atm', 'liquid', 'solid'.

    Returns
    -------
    float
        Mass of oxygen atoms [kg] in the reservoir. NaN inputs are
        skipped with a warning.
    """
    import math

    from proteus.utils.constants import element_mmw

    O_mw = element_mmw['O']

    # Pick the species scope for this reservoir.
    if reservoir == 'atm':
        species_counts = {**_SPECIES_O_COUNT_VOL, **_SPECIES_O_COUNT_VAP_ATM_ONLY}
    elif reservoir in ('liquid', 'solid'):
        species_counts = dict(_SPECIES_O_COUNT_VOL)
    else:
        raise ValueError(
            f"compute_O_kg_from_species: reservoir must be one of "
            f"'atm'|'liquid'|'solid'; got {reservoir!r}"
        )

    total_mol_O = 0.0
    for species, n_O in species_counts.items():
        mol_key = f'{species}_mol_{reservoir}'
        kg_key = f'{species}_kg_{reservoir}'

        n_mol = 0.0
        if mol_key in hf_row:
            v = float(hf_row[mol_key])
            if math.isfinite(v):
                n_mol = v
        if n_mol == 0.0 and kg_key in hf_row:
            # Fallback: derive mol from kg via the species' molar mass.
            # Required for the atmodeller backend which writes only kg_*.
            kg = float(hf_row[kg_key])
            if math.isfinite(kg):
                mw = _species_mmw(species)
                if mw > 0.0:
                    n_mol = kg / mw

        total_mol_O += n_mol * n_O

    return total_mol_O * O_mw


def _species_mmw(species: str) -> float:
    """Molar mass of a species formula in kg/mol (parsing helper)."""
    import re

    from proteus.utils.constants import element_mmw
    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    total = 0.0
    for elem, count_str in pattern.findall(species):
        if not elem or elem not in element_mmw:
            return 0.0
        count = int(count_str) if count_str else 1
        total += element_mmw[elem] * count
    return total


def populate_core_composition(hf_row: dict, config: 'Config') -> None:
    """
    Populate `{element}_kg_core` from `config.interior_struct.core_comp`
    and `hf_row['M_core']` (#57 Commit E.3).

    Call once at init after M_core has been computed, and on every
    step the core composition could change (today, only #526 metal-
    silicate partitioning can change it, and #526 is stubbed to
    zero, so a one-shot init call is sufficient). Re-calling is
    idempotent and cheap; safe to call per-step if a future hook
    adds core evolution.

    CoreComp wt% fields map to element_list names. Missing fields
    (e.g. `Si_wt` when CoreComp defines only Fe/H/O) default to 0.

    Parameters
    ----------
    hf_row
        Helpfile row; `{element}_kg_core` columns are overwritten
        for every element in `element_list`.
    config
        Config instance; reads `interior_struct.core_comp` (a
        CoreComp dataclass with Fe_wt / H_wt / O_wt / Si_wt fields).
    """
    core_comp = getattr(
        getattr(config, 'interior_struct', None), 'core_comp', None,
    )
    if core_comp is None:
        return    # No CoreComp configured; leave *_kg_core at 0.

    M_core = float(hf_row.get('M_core', 0.0))
    if M_core <= 0.0:
        return    # Structure not solved yet; skip.

    # element_list is ['H', 'O', 'C', 'N', 'S', 'Si', 'Mg', 'Fe', 'Na'].
    # CoreComp today defines Fe_wt / H_wt / O_wt / Si_wt. Others 0.
    for e in element_list:
        wt_field = f'{e}_wt'
        wt = float(getattr(core_comp, wt_field, 0.0))
        hf_row[f'{e}_kg_core'] = M_core * (wt / 100.0)


def populate_O_kg(hf_row: dict) -> None:
    """
    Populate `O_kg_atm`, `O_kg_liquid`, `O_kg_solid`, `O_kg_total` in
    hf_row from the current species inventory (#57 Commit D / D.1).

    Call after any outgas step that updates `{species}_{mol|kg}_*`
    columns; idempotent on repeat calls. Also call from
    `run_desiccated` / `run_crystallized` so the O_kg_* columns are
    not left stale when a step takes the non-outgas branch.
    """
    for reservoir in ('atm', 'liquid', 'solid'):
        hf_row[f'O_kg_{reservoir}'] = compute_O_kg_from_species(
            hf_row, reservoir,
        )
    hf_row['O_kg_total'] = (
        hf_row['O_kg_atm']
        + hf_row['O_kg_liquid']
        + hf_row['O_kg_solid']
    )


def M_ele_excl_O(hf_row: dict) -> float:
    """
    Sum of `{element}_kg_total` over every tracked element EXCEPT oxygen.

    Regression-compatibility helper introduced with #57. Pre-redox
    PROTEUS computed `M_ele = Σ_{e != O} hf_row[e+"_kg_total"]` because
    oxygen was skipped as "set by fO2". After the redox scaffolding
    lands, `M_ele` becomes `Σ_e hf_row[e+"_kg_total"]` for ALL elements.

    Tests (`tests/test_redox_scaffolding.py`, `tests/test_static_mode.py`)
    use this helper to compare non-O totals bit-for-bit against a
    pre-#57 baseline run.
    """
    return sum(
        float(hf_row[f'{e}_kg_total'])
        for e in element_list
        if e != 'O'
    )


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
    from proteus.plot.cpl_visual import plot_visual

    # Directories
    output_dir = dirs['output']
    fwl_dir = dirs['fwl']

    # Check model configuration
    dummy_atm = config.atmos_clim.module == 'dummy'
    dummy_int = config.interior_energetics.module == 'dummy'
    agni = config.atmos_clim.module == 'agni'
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

        # Visualise planet and star
        if agni:
            plot_visual(hf_all, output_dir, idx=-1, plot_format=config.params.out.plot_fmt)

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
        'tools': os.path.join(root_dir, 'tools'),
        'vulcan': os.path.join(root_dir, 'VULCAN'),
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
