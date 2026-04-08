# Generic interior wrapper
from __future__ import annotations

import gc
import logging
import os
import shutil
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.optimize as optimise

from proteus.interior_energetics.common import Interior_t
from proteus.outgas.wrapper import calc_target_elemental_inventories
from proteus.utils.constants import M_earth, R_earth, const_G, element_list
from proteus.utils.helper import UpdateStatusfile

if TYPE_CHECKING:
    from proteus.config import Config

# Counter for consecutive Zalmoxis convergence failures during time evolution.
# Reset on each successful structure update. Crash after max_consecutive.
_zalmoxis_fail_count = 0
_ZALMOXIS_MAX_CONSECUTIVE_FAILS = 5

# Counter for consecutive SPIDER CVode failures during time evolution.
# Reset on each successful SPIDER call. Crash after max_consecutive.
_spider_fail_count = 0
_SPIDER_MAX_CONSECUTIVE_FAILS = 3

log = logging.getLogger('fwl.' + __name__)


def get_core_density(config: Config, hf_row: dict) -> float:
    """Resolve core density: numerical value from config, or from hf_row if 'self'."""
    val = config.interior_struct.core_density
    if val == 'self':
        return float(hf_row.get('core_density', 10738.0))
    return float(val)


def get_core_heatcap(config: Config, hf_row: dict) -> float:
    """Resolve core heat capacity: numerical value from config, or from hf_row if 'self'."""
    val = config.interior_struct.core_heatcap
    if val == 'self':
        return float(hf_row.get('core_heatcap', 450.0))
    return float(val)


def update_gravity(hf_row: dict):
    """
    Update surface gravity.
    """
    hf_row['gravity'] = const_G * hf_row['M_int'] / (hf_row['R_int'] * hf_row['R_int'])


def calculate_core_mass(hf_row: dict, config: Config):
    """
    Calculate the core mass of the planet.
    """
    rho_core = get_core_density(config, hf_row)
    hf_row['M_core'] = (
        rho_core
        * 4.0
        / 3.0
        * np.pi
        * (hf_row['R_int'] * config.interior_struct.core_frac) ** 3.0
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
    match config.interior_energetics.module:
        case 'spider':
            return int(config.interior_energetics.num_levels)
        case 'aragog':
            return int(config.interior_energetics.num_levels)
        case 'dummy':
            return 2
    raise ValueError(f"Invalid interior module selected '{config.interior_energetics.module}'")


# The 10 phase-property files Aragog's EntropyEOS and SPIDER's lookup
# loader both expect, in SPIDER's canonical P-S header format.
_SPIDER_EOS_PHASE_FILES = (
    'temperature_melt.dat',
    'temperature_solid.dat',
    'density_melt.dat',
    'density_solid.dat',
    'heat_capacity_melt.dat',
    'heat_capacity_solid.dat',
    'adiabat_temp_grad_melt.dat',
    'adiabat_temp_grad_solid.dat',
    'thermal_exp_melt.dat',
    'thermal_exp_solid.dat',
)

# P-S melting curves. Aragog's `_load_spider_phase_boundary` hardcodes
# these filenames. SPIDER's bundled lookup_data ships them under the
# legacy `{solidus,liquidus}_A11_H13.dat` names; we rename on copy so a
# single canonical layout satisfies both solvers.
_SPIDER_EOS_MELTING_CURVES = ('solidus_P-S.dat', 'liquidus_P-S.dat')


def _is_spider_ps_format(path: str) -> bool:
    """Cheap first-line sniff to distinguish P-S tables from P-T tables.

    SPIDER's canonical P-S format starts with ``# 5 <n_S> <n_P>`` (5 is
    the number of header lines the loader expects). The legacy P-T
    format (shipped in Zenodo 17417017) starts with
    ``#pressure temperature density``. Both formats use the same
    filenames, so a content sniff is the only way to distinguish them
    when populating spider_eos_dir from FWL_DATA.
    """
    try:
        with open(path) as f:
            first = f.readline().strip()
    except OSError:
        return False
    # P-S header is exactly "# 5 <int> <int>".
    parts = first.split()
    return (
        len(parts) >= 2
        and parts[0] == '#'
        and parts[1] == '5'
    )


def _provide_spider_eos_tables(config: Config, outdir: str, dirs: dict) -> None:
    """Ensure that Aragog and SPIDER can find a complete P-S lookup set.

    Populates ``output/<case>/data/spider_eos/`` with the 12 files both
    solvers need at runtime (10 phase-property files + 2 P-S melting
    curves). This is the PROTEUS-side data-resolution layer that lets
    ``interior_energetics.module = "aragog"`` work with
    ``interior_struct.module = "spider"`` — the historical pattern used
    by the R8 CHILI baseline, which previously hard-failed under Aragog
    because no caller produced the P-S tables.

    Resolution order (first available wins):

    1. **Already populated** — if ``dirs['spider_eos_dir']`` is set and
       the target directory already contains the 12 expected files, do
       nothing. This keeps Zalmoxis's ``generate_spider_tables()``
       output path and cache semantics untouched.

    2. **FWL_DATA (Zenodo 19473625)** — if the canonical Zenodo download
       target exists and is complete, copy the 12 files into the output
       directory. This is the self-sufficient path: once the user runs
       ``proteus get all`` (or any non-offline start), the Zenodo record
       populates FWL_DATA and subsequent runs find the complete set
       here.

    3. **SPIDER submodule fallback** — if FWL_DATA is incomplete but the
       SPIDER git submodule is cloned at ``dirs['spider']/lookup_data/``,
       copy the 10 phase files verbatim and rename the legacy
       ``{solidus,liquidus}_A11_H13.dat`` melting curves to
       ``{solidus,liquidus}_P-S.dat``. This keeps the pre-Zenodo
       workflow alive for users who have the submodule but haven't
       refreshed their FWL_DATA tree.

    4. **Hard failure** — if neither source yields a complete set, raise
       ``FileNotFoundError`` with a clear message pointing the user at
       ``proteus get all`` or the Zenodo record.

    Side effects: sets ``dirs['spider_eos_dir']``,
    ``dirs['spider_solidus_ps']``, ``dirs['spider_liquidus_ps']``.
    """
    target_dir = os.path.join(outdir, 'data', 'spider_eos')

    # Case 1: already populated (e.g. by an earlier call this session or
    # by Zalmoxis's generate_spider_tables in a prior structure solve).
    existing = dirs.get('spider_eos_dir')
    if existing and os.path.isdir(existing):
        missing = [
            f for f in (_SPIDER_EOS_PHASE_FILES + _SPIDER_EOS_MELTING_CURVES)
            if not os.path.isfile(os.path.join(existing, f))
        ]
        if not missing:
            log.debug('spider_eos_dir already populated at %s, reusing', existing)
            dirs['spider_solidus_ps'] = os.path.join(existing, 'solidus_P-S.dat')
            dirs['spider_liquidus_ps'] = os.path.join(existing, 'liquidus_P-S.dat')
            return
        log.debug(
            'spider_eos_dir=%s exists but is missing %d file(s); repopulating',
            existing,
            len(missing),
        )

    os.makedirs(target_dir, exist_ok=True)

    # Import lazily so the helper is usable outside of a full PROTEUS
    # install (e.g. unit tests that stub out FWL_DATA).
    from proteus.utils.data import GetFWLData

    fwl_data = GetFWLData()
    zenodo_root = (
        fwl_data
        / 'interior_lookup_tables'
        / '1TPa-dK09-elec-free'
        / 'MgSiO3_Wolf_Bower_2018_1TPa'
    )

    # Case 2: FWL_DATA (Zenodo 19473625) complete set.
    # We check BOTH that all 12 files exist AND that density_melt.dat
    # is in the canonical SPIDER P-S format (not the legacy P-T format
    # from the older Zenodo 17417017 record, which shipped under the
    # same filenames). Without this content check a stale FWL_DATA tree
    # from before the Zenodo 19473625 release would be silently accepted
    # and Aragog's EntropyEOS would then crash trying to parse the P-T
    # file as a P-S table. See chili_earth_spider_sweep.md migration note.
    zenodo_files = (
        list(_SPIDER_EOS_PHASE_FILES) + list(_SPIDER_EOS_MELTING_CURVES)
    )
    zenodo_missing = [
        f for f in zenodo_files if not (zenodo_root / f).is_file()
    ]
    zenodo_format_ok = (
        not zenodo_missing
        and _is_spider_ps_format(str(zenodo_root / 'density_melt.dat'))
    )
    if not zenodo_format_ok and not zenodo_missing:
        log.warning(
            'FWL_DATA EOS tables at %s exist but density_melt.dat is not '
            'in SPIDER P-S format. This usually means the directory was '
            'populated by the legacy Zenodo 17417017 record (P-T format). '
            'Falling through to the SPIDER submodule. Refresh FWL_DATA '
            'with `proteus get all` to fetch Zenodo 19473625.',
            zenodo_root,
        )
    if zenodo_format_ok:
        log.info(
            'Providing P-S EOS tables to spider_eos_dir from FWL_DATA (Zenodo 19473625)'
        )
        for f in zenodo_files:
            src = zenodo_root / f
            dst = os.path.join(target_dir, f)
            if not os.path.isfile(dst):
                shutil.copy2(src, dst)
        dirs['spider_eos_dir'] = target_dir
        dirs['spider_solidus_ps'] = os.path.join(target_dir, 'solidus_P-S.dat')
        dirs['spider_liquidus_ps'] = os.path.join(target_dir, 'liquidus_P-S.dat')
        return

    # Case 3: SPIDER submodule fallback. The 10 phase files have
    # canonical names; the 2 melting curves need the _A11_H13 -> _P-S
    # rename on copy.
    spider_bundle = None
    spider_root = dirs.get('spider')
    if spider_root:
        candidate = os.path.join(spider_root, 'lookup_data', '1TPa-dK09-elec-free')
        if os.path.isdir(candidate):
            spider_bundle = candidate

    if spider_bundle is not None:
        melt_map = {
            'solidus_P-S.dat': 'solidus_A11_H13.dat',
            'liquidus_P-S.dat': 'liquidus_A11_H13.dat',
        }
        phase_missing = [
            f for f in _SPIDER_EOS_PHASE_FILES
            if not os.path.isfile(os.path.join(spider_bundle, f))
        ]
        melt_missing = [
            canonical for canonical, legacy in melt_map.items()
            if not os.path.isfile(os.path.join(spider_bundle, legacy))
        ]
        if not phase_missing and not melt_missing:
            log.info(
                'Providing P-S EOS tables to spider_eos_dir from SPIDER submodule '
                '(bundled lookup_data). FWL_DATA was incomplete: missing %d file(s).',
                len(zenodo_missing),
            )
            for f in _SPIDER_EOS_PHASE_FILES:
                src = os.path.join(spider_bundle, f)
                dst = os.path.join(target_dir, f)
                if not os.path.isfile(dst):
                    shutil.copy2(src, dst)
            for canonical, legacy in melt_map.items():
                src = os.path.join(spider_bundle, legacy)
                dst = os.path.join(target_dir, canonical)
                if not os.path.isfile(dst):
                    shutil.copy2(src, dst)
            dirs['spider_eos_dir'] = target_dir
            dirs['spider_solidus_ps'] = os.path.join(target_dir, 'solidus_P-S.dat')
            dirs['spider_liquidus_ps'] = os.path.join(target_dir, 'liquidus_P-S.dat')
            return
        log.warning(
            'SPIDER submodule lookup_data present at %s but incomplete '
            '(phase missing=%d, melting curves missing=%d). Falling through.',
            spider_bundle,
            len(phase_missing),
            len(melt_missing),
        )

    # Case 4: neither source yielded a complete set.
    raise FileNotFoundError(
        'Could not provide SPIDER/Aragog P-S EOS tables at '
        f'{target_dir}. FWL_DATA source '
        f'{zenodo_root} is missing {len(zenodo_missing)} of '
        f'{len(zenodo_files)} required files '
        f'({zenodo_missing[:3]}...), and the SPIDER submodule fallback '
        f'was unavailable at {dirs.get("spider", "<no spider dir set>")}'
        '/lookup_data/1TPa-dK09-elec-free/. Run `proteus get all` to '
        'fetch Zenodo record 19473625, or ensure the SPIDER submodule '
        'is cloned.'
    )


def determine_interior_radius(
    dirs: dict, config: Config, hf_all: pd.DataFrame, hf_row: dict, outdir: str
):
    """
    Determine the interior radius (R_int) of the planet.

    This uses the interior model's hydrostatic integration to estimate the planet's
    interior mass from a given radius. The radius is then adjusted until the interior mass
    achieves the target mass provided by the user in the config file.
    """

    log.info('Using %s interior module to solve structure' % config.interior_energetics.module)

    # Provide P-S lookup tables for Aragog's entropy solver (and SPIDER
    # when it runs under this structure path). Mirrors the
    # generate_spider_tables() call at the top of the zalmoxis and dummy
    # structure paths. The helper resolves from FWL_DATA/Zenodo first,
    # then the SPIDER submodule as a fallback.
    if config.interior_energetics.module in ('spider', 'aragog'):
        _provide_spider_eos_tables(config, outdir, dirs)

    # Initial guess for interior radius and gravity
    if config.interior_energetics.module == 'spider':
        spider_dir = dirs['spider']
    else:
        spider_dir = None
    int_o = Interior_t(
        get_nlevb(config), spider_dir=spider_dir, eos_dir=config.interior_struct.eos_dir
    )
    int_o.ic = 1
    hf_row['R_int'] = R_earth
    calculate_core_mass(hf_row, config)
    hf_row['gravity'] = 9.81

    # Target mass
    M_target = config.planet.mass_tot * M_earth

    # We need to solve for the state hf_row[M_planet] = config.planet.mass_tot
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
    match config.interior_energetics.module:
        case 'aragog':
            rtol = config.interior_energetics.num_tolerance
        case 'spider':
            rtol = config.interior_energetics.num_tolerance
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


def determine_interior_radius_with_dummy(
    dirs: dict, config: Config, hf_all: pd.DataFrame, hf_row: dict, outdir: str
):
    """Determine interior structure using Noack & Lasbleis (2020) scaling laws.

    Ultra-fast analytical parameterization replacing Zalmoxis. Fills all
    hf_row keys and writes output files needed by SPIDER/Aragog.
    """
    from proteus.interior_struct.dummy import solve_dummy_structure

    nlev_b = get_nlevb(config)
    num_spider_nodes = nlev_b if config.interior_energetics.module == 'spider' else 0

    spider_mesh_file = solve_dummy_structure(
        config,
        hf_row,
        outdir,
        num_spider_nodes=num_spider_nodes,
    )

    if spider_mesh_file:
        dirs['spider_mesh'] = spider_mesh_file
        dirs['spider_mesh_prev'] = spider_mesh_file + '.prev'

    # Generate P-S EOS tables for SPIDER/Aragog (if PALEOS)
    if config.interior_energetics.module in ('spider', 'aragog'):
        from proteus.interior_struct.zalmoxis import generate_spider_tables

        spider_tables = generate_spider_tables(config, outdir)
        if spider_tables is not None:
            dirs['spider_eos_dir'] = spider_tables['eos_dir']
            dirs['spider_solidus_ps'] = spider_tables['solidus_path']
            dirs['spider_liquidus_ps'] = spider_tables['liquidus_path']

    # Derived quantities
    hf_row['M_mantle'] = hf_row['M_int'] - hf_row['M_core']

    # Run first interior step
    int_o = Interior_t(
        nlev_b, spider_dir=dirs.get('spider'), eos_dir=config.interior_struct.eos_dir
    )
    int_o.ic = 1
    run_interior(dirs, config, hf_all, hf_row, int_o, verbose=False)
    update_gravity(hf_row)

    calc_target_elemental_inventories(dirs, config, hf_row)
    update_planet_mass(hf_row)

    log.info('Dummy structure solve complete')
    log.info(
        'R_int: %.1e m = %.3f R_earth, M_int: %.1e kg = %.3f M_earth',
        hf_row['R_int'],
        hf_row['R_int'] / R_earth,
        hf_row['M_int'],
        hf_row['M_int'] / M_earth,
    )


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
    from proteus.interior_struct.zalmoxis import zalmoxis_solver

    nlev_b = get_nlevb(config)
    spider_dir = dirs.get('spider') if config.interior_energetics.module == 'spider' else None
    int_o = Interior_t(nlev_b, spider_dir=spider_dir, eos_dir=config.interior_struct.eos_dir)
    int_o.ic = 1

    # Set Zalmoxis to 'adiabatic' mode for T-dependent mantle EOS.
    # NOTE: In practice, Zalmoxis converges the structure using a linear T
    # guess and breaks on mass convergence BEFORE the adiabat gate activates.
    # The adiabat flag is still set so that (a) the correct EOS code paths
    # are selected inside Zalmoxis, and (b) standalone Zalmoxis can use the
    # adiabat if the gate is ever fixed.  SPIDER provides its own T(r)
    # through entropy evolution, so the linear T initial guess is fine.
    _TDEP_PREFIXES = ('WolfBower2018', 'RTPress100TPa')
    _temp_mode_override: str | None = None
    if (
        config.interior_energetics.module == 'spider'
        and config.planet.temperature_mode == 'isothermal'
        and config.interior_struct.zalmoxis.mantle_eos.startswith(_TDEP_PREFIXES)
    ):
        log.info(
            'Overriding Zalmoxis temperature_mode from isothermal to adiabatic '
            'for SPIDER coupling with T-dependent mantle EOS (local override, '
            'does not mutate config)',
        )
        _temp_mode_override = 'adiabatic'

    # Request SPIDER mesh file if interior module is SPIDER.
    # Pass the override as a parameter instead of mutating config.planet.
    num_spider_nodes = nlev_b if config.interior_energetics.module == 'spider' else 0
    _cmb_radius, spider_mesh_file = zalmoxis_solver(
        config,
        outdir,
        hf_row,
        num_spider_nodes=num_spider_nodes,
        temperature_mode_override=_temp_mode_override,
    )

    # Store mesh file path for subsequent SPIDER calls
    if spider_mesh_file:
        dirs['spider_mesh'] = spider_mesh_file
        # Save initial mesh as baseline for blending comparisons
        prev_path = spider_mesh_file + '.prev'
        shutil.copy2(spider_mesh_file, prev_path)
        dirs['spider_mesh_prev'] = prev_path

    # Mesh convergence starts inactive (no blending needed at init)
    dirs['mesh_shift_active'] = False
    dirs['mesh_convergence_steps'] = 0

    # Generate SPIDER P-S EOS tables from PALEOS if applicable.
    # This converts Zalmoxis's P-T EOS data into the P-S format that both
    # SPIDER and Aragog (entropy solver) need.
    if config.interior_energetics.module in ('spider', 'aragog'):
        from proteus.interior_struct.zalmoxis import generate_spider_tables

        spider_tables = generate_spider_tables(config, outdir)
        if spider_tables is not None:
            dirs['spider_eos_dir'] = spider_tables['eos_dir']
            dirs['spider_solidus_ps'] = spider_tables['solidus_path']
            dirs['spider_liquidus_ps'] = spider_tables['liquidus_path']

    # NOTE: run_interior runs with the *original* temperature_mode (restored
    # by the finally block above), not the overridden 'adiabatic'.  This is
    # correct: the Zalmoxis solver already used the adiabatic mode to compute
    # the structure, and run_interior (SPIDER/ARAGOG) manages its own T(r).
    run_interior(dirs, config, hf_all, hf_row, int_o)


def equilibrate_initial_state(dirs: dict, config: Config, hf_row: dict, outdir: str):
    """Iterate CALLIOPE + Zalmoxis until structure and composition converge.

    Runs volatile partitioning (CALLIOPE with optional binodal H2) followed
    by structure re-computation (Zalmoxis) in a loop. No SPIDER call.
    Convergence is checked on relative changes in R_int and P_surf.

    Called from proteus.py after the initial structure solve and volatile
    inventory setup. The converged state provides a self-consistent starting
    point for SPIDER's first time step.

    Parameters
    ----------
    dirs : dict
        Directory paths (including spider_mesh, spider_eos_dir).
    config : Config
        PROTEUS configuration.
    hf_row : dict
        Current helpfile row with volatile masses, R_int, P_surf, T_magma, etc.
    outdir : str
        Output directory for Zalmoxis files.
    """
    from proteus.interior_struct.zalmoxis import generate_spider_tables, zalmoxis_solver
    from proteus.outgas.wrapper import calc_target_elemental_inventories, run_outgassing

    max_iter = config.interior_struct.zalmoxis.equilibrate_max_iter
    tol = config.interior_struct.zalmoxis.equilibrate_tol
    nlev_b = get_nlevb(config)
    num_spider_nodes = nlev_b if config.interior_energetics.module == 'spider' else 0

    log.info(
        'Starting init equilibration loop (max %d iter, tol %.1f%%)',
        max_iter,
        tol * 100,
    )

    # Initialize convergence metrics (used in the warning if max_iter reached)
    delta_R = 1.0
    delta_P = 1.0

    for i in range(max_iter):
        R_old = float(hf_row.get('R_int', 0.0))
        P_old = float(hf_row.get('P_surf', 0.0))

        # 1. Volatile partitioning: recompute elemental targets and
        #    run CALLIOPE to get atmosphere/melt distribution
        calc_target_elemental_inventories(dirs, config, hf_row)
        run_outgassing(dirs, config, hf_row)

        # 2. Re-compute structure with updated composition
        #    (volatile_profile is built inside zalmoxis_solver from hf_row)
        _cmb_radius, spider_mesh_file = zalmoxis_solver(
            config, outdir, hf_row, num_spider_nodes=num_spider_nodes
        )

        # Update M_mantle from Zalmoxis results (M_int and M_core are set
        # by zalmoxis_solver, but M_mantle is not). run_outgassing needs
        # an up-to-date M_mantle for dissolved fraction calculations.
        hf_row['M_mantle'] = float(hf_row.get('M_int', 0.0)) - float(hf_row.get('M_core', 0.0))

        # Update mesh path if written
        if spider_mesh_file:
            dirs['spider_mesh'] = spider_mesh_file
            prev_path = spider_mesh_file + '.prev'
            shutil.copy2(spider_mesh_file, prev_path)
            dirs['spider_mesh_prev'] = prev_path

        # 3. Check convergence
        R_new = float(hf_row.get('R_int', 0.0))
        P_new = float(hf_row.get('P_surf', 0.0))

        delta_R = abs(R_new - R_old) / R_old if R_old > 0 else 1.0
        delta_P = abs(P_new - P_old) / P_old if P_old > 0 else 1.0

        log.info(
            'Equilibration iter %d/%d: dR/R=%.4f, dP/P=%.4f (R=%.3e m, P=%.2f bar)',
            i + 1,
            max_iter,
            delta_R,
            delta_P,
            R_new,
            P_new,
        )

        if delta_R < tol and delta_P < tol:
            log.info('Equilibration converged after %d iterations', i + 1)
            break
    else:
        log.warning(
            'Equilibration did not converge after %d iterations (dR/R=%.4f, dP/P=%.4f)',
            max_iter,
            delta_R,
            delta_P,
        )

    # 4. Regenerate SPIDER EOS tables with final composition
    if config.interior_energetics.module in ('spider', 'aragog'):
        spider_tables = generate_spider_tables(config, outdir)
        if spider_tables is not None:
            dirs['spider_eos_dir'] = spider_tables['eos_dir']
            dirs['spider_solidus_ps'] = spider_tables['solidus_path']
            dirs['spider_liquidus_ps'] = spider_tables['liquidus_path']


def solve_structure(
    dirs: dict, config: Config, hf_all: pd.DataFrame, hf_row: dict, outdir: str
):
    """
    Solve for the planet structure based on the method set in the configuration file.

    If the structure is set by the radius, then this is trivial because the radius is used
    as an input to the interior modules anyway. If the structure is set by mass, then it is
    solved as an inverse problem for now.
    """

    # Set by total mass (mantle + core + volatiles)
    if config.planet.mass_tot is not None:
        # Choose the method to determine the interior radius
        match config.interior_struct.module:
            case 'dummy':
                return determine_interior_radius_with_dummy(
                    dirs, config, hf_all, hf_row, outdir
                )
            case 'spider':
                return determine_interior_radius(dirs, config, hf_all, hf_row, outdir)
            case 'zalmoxis':
                # Zalmoxis computes its own radius; temporarily disable orbital
                # feedback during structure solve (restored after return).
                _orig_orbit_module = config.orbit.module
                config.orbit.module = 'dummy'
                try:
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
                finally:
                    config.orbit.module = _orig_orbit_module
        raise ValueError(
            f"Invalid structure interior module selected '{config.interior_struct.module}'"
        )

    else:
        raise ValueError('planet.mass_tot must be set to solve for the interior structure')


def run_interior(
    dirs: dict,
    config: Config,
    hf_all: pd.DataFrame,
    hf_row: dict,
    interior_o: Interior_t,
    verbose: bool = True,
    write_data: bool = True,
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
        write_data : bool
            Write per-timestep data files (NetCDF/JSON) to disk. When False,
            the solver still runs but skips the data file write. Used by the
            dt_write time guard to prevent excessive output during rapid
            early evolution. SPIDER JSON writes are unaffected (managed by
            the C binary).
    """

    # Use the appropriate interior model
    if verbose:
        log.info('Evolve interior...')
    log.debug('Using %s module to evolve interior' % config.interior_energetics.module)

    # Write tidal heating file
    if config.interior_energetics.heat_tidal:
        interior_o.write_tides(dirs['output'])

    if config.interior_energetics.module == 'spider':
        global _spider_fail_count
        # Import
        from proteus.interior_energetics.spider import ReadSPIDER, RunSPIDER

        # Run SPIDER (pass external mesh file if available from Zalmoxis).
        # Note: write_data is not forwarded here. SPIDER JSON output is
        # controlled by the C binary; Python cannot suppress it per-timestep.
        mesh_file = dirs.get('spider_mesh')
        try:
            RunSPIDER(dirs, config, hf_all, hf_row, interior_o, mesh_file=mesh_file)
            _spider_fail_count = 0
        except RuntimeError as e:
            _spider_fail_count += 1
            log.warning(
                'SPIDER CVode failure #%d/%d. '
                'Keeping previous interior state for this step. Error: %s',
                _spider_fail_count,
                _SPIDER_MAX_CONSECUTIVE_FAILS,
                str(e)[:200],
            )
            if _spider_fail_count >= _SPIDER_MAX_CONSECUTIVE_FAILS:
                log.error(
                    'SPIDER failed %d consecutive times. Aborting.',
                    _spider_fail_count,
                )
                raise
            # Skip ReadSPIDER; keep hf_row values from previous step.
            # The interior state is stale by one coupling step, but the
            # atmosphere + outgassing still advance. The stale interior
            # pushes the planet past the stiff rheological transition.
            #
            # IMPORTANT: still advance interior_o.dt and _spider_cumulative_time
            # so that main-loop bookkeeping stays consistent. Without this,
            # the main loop at proteus.py:496 increments hf_row['Time'] by the
            # previous successful dt (stale) while _spider_cumulative_time is
            # frozen, leaving the two counters desynchronised by one dtswitch
            # per fallback event. That in turn confuses the retry ladder on
            # the next attempt.
            from proteus.interior_energetics.timestep import next_step

            dtswitch = next_step(config, dirs, hf_row, hf_all, 1.0)
            interior_o._spider_cumulative_time += dtswitch
            interior_o.dt = dtswitch
            return
        sim_time, output = ReadSPIDER(dirs, config, hf_row['R_int'], interior_o)

    elif config.interior_energetics.module == 'aragog':
        from proteus.interior_energetics.aragog import AragogRunner

        runner = AragogRunner(config, dirs, hf_row, hf_all, interior_o)
        sim_time, output = runner.run_solver(hf_row, interior_o, dirs, write_data=write_data)

    elif config.interior_energetics.module == 'dummy':
        # Import
        from proteus.interior_energetics.dummy import run_dummy_int

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
    calc_visc = bool(config.interior_energetics.module == 'dummy')
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
        if config.planet.prevent_warming and (interior_o.ic == 2):
            hf_row['Phi_global'] = min(hf_row['Phi_global'], Phi_global_prev)
            hf_row['T_magma'] = min(hf_row['T_magma'], T_magma_prev)

        # Do not allow massive increases to T_surf
        dT_delta = config.interior_energetics.tmagma_atol
        dT_delta += config.interior_energetics.tmagma_rtol * T_magma_prev
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
        if config.interior_energetics.heat_tidal:
            log.info('    F_tidal    = %.2e W m-2' % float(hf_row['F_tidal']))
        if config.interior_energetics.heat_radiogenic:
            log.info('    F_radio    = %.2e W m-2' % float(hf_row['F_radio']))

    # Actual time step size.
    # For SPIDER: use the coupling timestepper's dtswitch directly,
    # tracked via interior_o._spider_cumulative_time. The old approach
    # (sim_time - hf_row['Time']) fails because SPIDER's JSON filenames
    # alias to 0 when tsurf_poststep_change terminates the BDF early
    # and llround(time_years) rounds to 0.
    # For Aragog/dummy: sim_time is returned directly from the solver
    # and is reliable.
    if config.interior_energetics.module == 'spider':
        from proteus.interior_energetics.timestep import next_step

        dtswitch = next_step(config, dirs, hf_row, hf_all, 1.0)
        interior_o._spider_cumulative_time += dtswitch
        interior_o.dt = dtswitch
    else:
        interior_o.dt = float(sim_time) - hf_row['Time']

    # TODO: When config.interior_struct.module == 'zalmoxis', the Aragog mesh
    # is set up once during setup_solver and never refreshed during
    # equilibration iterations. If Zalmoxis re-runs and produces a
    # new zalmoxis_output.dat, Aragog uses the stale initial mesh.
    # Fix: pass the refreshed mesh to Aragog after each Zalmoxis call
    # during equilibration, or regenerate the Aragog mesh here.


def update_structure_from_interior(
    dirs: dict,
    config: Config,
    hf_row: dict,
    interior_o: Interior_t,
    last_struct_time: float,
    last_Tmagma: float,
    last_Phi: float,
) -> tuple[float, float, float]:
    """Re-run Zalmoxis with SPIDER's current T(r) to update structure.

    Uses a hybrid trigger: fires when either the relative change in
    T_magma or the absolute change in Phi_global exceeds configured
    thresholds, subject to a minimum interval (floor) and maximum
    interval (ceiling).  When ``update_interval == 0``, no dynamic
    updates are performed (structure is computed only at init).

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
    last_Tmagma : float
        T_magma [K] at the last structure update.
    last_Phi : float
        Phi_global at the last structure update.

    Returns
    -------
    tuple[float, float, float]
        (last_struct_time, last_Tmagma, last_Phi) — updated to current
        values if an update occurred, otherwise returned unchanged.
    """
    no_update = (last_struct_time, last_Tmagma, last_Phi)

    # Dynamic updates disabled
    if config.interior_struct.zalmoxis.update_interval <= 0:
        return no_update

    current_time = hf_row['Time']
    elapsed = current_time - last_struct_time

    # Evaluate triggers
    triggered = False
    reason = ''

    # Mesh convergence trigger: bypasses normal floor when mesh is still
    # converging toward the true Zalmoxis solution after blending
    mesh_converging = dirs.get('mesh_shift_active', False)
    if mesh_converging and elapsed >= config.interior_struct.zalmoxis.mesh_convergence_interval:
        triggered = True
        reason = (
            f'mesh convergence (elapsed {elapsed:.1f} yr '
            f'>= {config.interior_struct.zalmoxis.mesh_convergence_interval:.1f} yr)'
        )

    # Floor: don't update too frequently (only for non-convergence triggers)
    if not triggered and elapsed < config.interior_struct.zalmoxis.update_min_interval:
        return no_update

    # Ceiling: guaranteed update after max interval
    if not triggered and elapsed >= config.interior_struct.zalmoxis.update_interval:
        triggered = True
        reason = f'ceiling ({elapsed:.1f} yr >= {config.interior_struct.zalmoxis.update_interval:.1f} yr)'

    # Phi_global absolute change (primary trigger: directly reflects rheological state)
    if not triggered:
        dPhi = abs(hf_row['Phi_global'] - last_Phi)
        if dPhi >= config.interior_struct.zalmoxis.update_dphi_abs:
            triggered = True
            reason = f'dPhi={dPhi:.3f} >= {config.interior_struct.zalmoxis.update_dphi_abs}'

    # T_magma relative change (secondary: catches cases where Phi is constant but T changes)
    if not triggered and last_Tmagma > 0:
        dT_frac = abs(hf_row['T_magma'] - last_Tmagma) / last_Tmagma
        if dT_frac >= config.interior_struct.zalmoxis.update_dtmagma_frac:
            triggered = True
            reason = (
                f'dT/T={dT_frac:.3f} >= {config.interior_struct.zalmoxis.update_dtmagma_frac}'
            )

    # Composition change: check dissolved volatile fractions.
    # When the binodal or CALLIOPE changes how much H2/H2O is dissolved,
    # the mantle density profile shifts, requiring a structure update.
    comp_changed = False
    if not triggered:
        M_mantle = float(hf_row.get('M_mantle', 0.0))
        if M_mantle > 0:
            for species in ('H2O', 'H2'):
                w_new = float(hf_row.get(f'{species}_kg_liquid', 0.0)) / M_mantle
                w_old = dirs.get(f'_last_w_{species}_liquid', w_new)
                if w_old > 1e-6:
                    dw = abs(w_new - w_old) / w_old
                    if dw >= 0.05:
                        triggered = True
                        comp_changed = True
                        reason = f'd_w_{species}={dw:.3f} >= 0.05'
                        break

    if not triggered:
        return no_update

    log.info('Updating structure from interior T(r) via Zalmoxis (trigger: %s)', reason)

    outdir = dirs['output']

    # Build SPIDER's mantle T(r) in ascending radius (CMB to surface)
    # interior_o.radius is basic nodes (surface to CMB), temp is staggered nodes
    r_stag = 0.5 * (interior_o.radius[:-1] + interior_o.radius[1:])
    r_ascending = r_stag[::-1]
    T_ascending = interior_o.temp[::-1]

    # Build T(r,P) interpolator from SPIDER/Aragog output in memory.
    # SPIDER only covers the mantle (CMB to surface), so hold T constant
    # at the CMB value for the core region.
    R_cmb = float(np.squeeze(r_ascending[0]))
    T_cmb = float(np.squeeze(T_ascending[0]))

    # Capture arrays in closure for the temperature function
    _r_asc = np.asarray(r_ascending, dtype=float)
    _T_asc = np.asarray(T_ascending, dtype=float)
    _R_cmb = R_cmb
    _T_cmb = T_cmb

    def temperature_function(r, P):
        if r <= _R_cmb:
            return _T_cmb
        return float(np.interp(r, _r_asc, _T_asc))

    from proteus.interior_struct.zalmoxis import zalmoxis_solver

    # Save current mesh as baseline for blending
    prev_path = dirs.get('spider_mesh_prev')
    current_mesh = dirs.get('spider_mesh')
    if current_mesh and os.path.isfile(current_mesh):
        if not prev_path:
            prev_path = current_mesh + '.prev'
            dirs['spider_mesh_prev'] = prev_path
        shutil.copy2(current_mesh, prev_path)

    global _zalmoxis_fail_count

    nlev_b = get_nlevb(config)
    num_spider_nodes = nlev_b if config.interior_energetics.module == 'spider' else 0

    # Save current structure values for fallback on convergence failure
    _saved_structure = {
        k: hf_row[k]
        for k in (
            'R_int',
            'M_int',
            'M_core',
            'M_mantle',
            'P_surf',
            'R_core',
            'P_center',
            'rho_avg',
        )
        if k in hf_row
    }

    try:
        _cmb_radius, spider_mesh_file = zalmoxis_solver(
            config,
            outdir,
            hf_row,
            num_spider_nodes=num_spider_nodes,
            temperature_function=temperature_function,
        )
        _zalmoxis_fail_count = 0  # Reset on success
    except RuntimeError as e:
        _zalmoxis_fail_count += 1
        log.warning(
            'Zalmoxis convergence failure #%d/%d during time evolution. '
            'Falling back to previous structure. Error: %s',
            _zalmoxis_fail_count,
            _ZALMOXIS_MAX_CONSECUTIVE_FAILS,
            str(e)[:200],
        )
        if _zalmoxis_fail_count >= _ZALMOXIS_MAX_CONSECUTIVE_FAILS:
            log.error(
                'Zalmoxis failed %d consecutive times. Aborting.',
                _zalmoxis_fail_count,
            )
            raise
        # Restore previous structure values
        hf_row.update(_saved_structure)
        hf_row['_structure_stale'] = True
        # Keep the previous mesh file and CMB radius
        spider_mesh_file = dirs.get('spider_mesh')
        _cmb_radius = float(hf_row.get('R_core', 0.0))

    if spider_mesh_file:
        dirs['spider_mesh'] = spider_mesh_file

        # Blend mesh to limit per-update radius shift
        from proteus.interior_energetics.spider import blend_mesh_files

        actual_shift = blend_mesh_files(
            prev_path or '',
            spider_mesh_file,
            max_shift=config.interior_struct.zalmoxis.mesh_max_shift,
        )
        still_converging = actual_shift > config.interior_struct.zalmoxis.mesh_max_shift

        # Track convergence steps; give up after 20 consecutive blends
        # to avoid infinite rapid-update loops when Zalmoxis and SPIDER
        # persistently disagree (e.g. extreme mass / low CMF)
        max_convergence_steps = 20
        n_conv = dirs.get('mesh_convergence_steps', 0)
        if still_converging:
            n_conv += 1
            if n_conv > max_convergence_steps:
                log.warning(
                    'Mesh convergence did not complete after %d steps '
                    '(shift still %.1f%%), reverting to normal triggers',
                    max_convergence_steps,
                    actual_shift * 100,
                )
                still_converging = False
                n_conv = 0
            else:
                log.info(
                    'Mesh convergence step %d/%d: shift %.1f%% clamped to %.1f%%',
                    n_conv,
                    max_convergence_steps,
                    actual_shift * 100,
                    config.interior_struct.zalmoxis.mesh_max_shift * 100,
                )
        else:
            n_conv = 0

        dirs['mesh_shift_active'] = still_converging
        dirs['mesh_convergence_steps'] = n_conv

        # Update .prev for next iteration
        if not prev_path:
            prev_path = spider_mesh_file + '.prev'
            dirs['spider_mesh_prev'] = prev_path
        shutil.copy2(spider_mesh_file, prev_path)

        # Remap entropy in the latest SPIDER JSON to match the new mesh.
        # Without this, the old dS/dxi applied on the new xi grid produces
        # incorrect absolute entropy, causing CVode failures at high mass.
        if config.interior_energetics.module == 'spider':
            from proteus.interior_energetics.spider import (
                get_all_output_times,
                remap_entropy_for_new_mesh,
            )

            try:
                sim_times = get_all_output_times(dirs['output'])
            except Exception as exc:
                log.warning(
                    'Could not retrieve SPIDER output times from %s; '
                    'skipping entropy remap: %s',
                    dirs['output'],
                    exc,
                )
                sim_times = []
            if len(sim_times) > 0:
                latest_json = os.path.join(dirs['output'], 'data', '%.0f.json' % sim_times[-1])
                # When global_miscibility is enabled, SPIDER's domain
                # extends to R_solvus, not R_int. Use the appropriate
                # radius for entropy remapping.
                if config.interior_struct.zalmoxis.global_miscibility and 'R_solvus' in hf_row:
                    remap_radius = hf_row['R_solvus']
                else:
                    remap_radius = hf_row['R_int']
                remap_entropy_for_new_mesh(
                    json_path=latest_json,
                    new_mesh_file=spider_mesh_file,
                    radius_phys=remap_radius,
                )
    else:
        # No mesh file produced — reset convergence state
        dirs['mesh_shift_active'] = False
        dirs['mesh_convergence_steps'] = 0

    # Clean up temporary arrays
    del r_stag, r_ascending, T_ascending
    gc.collect()

    # Regenerate SPIDER EOS tables when composition changed substantially.
    # This ensures SPIDER's lookup tables reflect the updated volatile
    # fractions in the mantle (e.g. after binodal H2 redistribution).
    # TODO: Aragog P-T tables are not regenerated on composition change.
    # Currently only SPIDER tables are refreshed. If Aragog runs with
    # composition-dependent melting curves, stale tables may be used.
    if comp_changed and config.interior_energetics.module in ('spider', 'aragog'):
        from proteus.interior_struct.zalmoxis import generate_spider_tables

        spider_tables = generate_spider_tables(config, outdir)
        if spider_tables is not None:
            dirs['spider_eos_dir'] = spider_tables['eos_dir']
            dirs['spider_solidus_ps'] = spider_tables['solidus_path']
            dirs['spider_liquidus_ps'] = spider_tables['liquidus_path']
            log.info('Regenerated SPIDER EOS tables (composition change)')

    # Update composition sentinels for next trigger check
    M_mantle = float(hf_row.get('M_mantle', 0.0))
    if M_mantle > 0:
        for species in ('H2O', 'H2'):
            dirs[f'_last_w_{species}_liquid'] = (
                float(hf_row.get(f'{species}_kg_liquid', 0.0)) / M_mantle
            )

    log.info(
        'Structure updated: R_int=%.3e m, gravity=%.3f m/s^2',
        hf_row['R_int'],
        hf_row['gravity'],
    )
    return (current_time, float(hf_row['T_magma']), float(hf_row['Phi_global']))


def get_all_output_times(output_dir: str, model: str):
    if model == 'spider':
        from proteus.interior_energetics.spider import get_all_output_times as _get_output_times
    elif model == 'aragog':
        from proteus.interior_energetics.aragog import get_all_output_times as _get_output_times
    else:
        return []

    return _get_output_times(output_dir)


def read_interior_data(output_dir: str, model: str, times: list):
    if len(times) == 0:
        return []

    if model == 'spider':
        from proteus.interior_energetics.spider import read_jsons

        return read_jsons(output_dir, times)

    elif model == 'aragog':
        from proteus.interior_energetics.aragog import read_ncdfs

        return read_ncdfs(output_dir, times)

    else:
        return []
