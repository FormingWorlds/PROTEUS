from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np

# ensure juliacall is imported before torch
# see issue here: https://github.com/pytorch/pytorch/issues/78829
from juliacall import Main as jl  # noqa

import proteus.utils.archive as archive
from proteus.config import read_config_object
from proteus.utils.constants import vap_list, vol_list
from proteus.utils.helper import (
    CleanDir,
    PrintHalfSeparator,
    PrintSeparator,
    UpdateStatusfile,
    multiple,
)
from proteus.utils.logs import (
    GetCurrentLogfileIndex,
    GetLogfilePath,
    setup_logger,
)

# Set number of OpenMP threads used by the SciPy matrix solver
#     This primarily affects VULCAN, but also Aragog.
#     Not setting this variable will allow SciPy to use all available CPU cores,
#     which can actually slow down performance. Choosing 4 is safe, as this is the limit
#     on GitHub runners, and is reasonable for desktop PCs and interactive servers.
# os.environ["OMP_NUM_THREADS"] = "4"


class Proteus:
    def __init__(self, *, config_path: Path | str) -> None:
        # Read and parse configuration file
        self.config_path = config_path
        self.config = read_config_object(config_path)

        # Setup directories dictionary
        self.directories: dict = None  # Directories dictionary
        self.init_directories()

        # Helpfile variables for the current iteration
        self.hf_row = None

        # Helpfile variables from all previous iterations
        self.hf_all = None

        # Loop counters
        self.init_stage = False
        self.loops = None

        # Interior
        self.interior_o = None  # Interior object from interior/common.py

        # Atmosphere
        self.atmos_o = None  # Atmosphere object from atmos_clim/common.py

        # Model has finished?
        self.finished_prev = False  # Satisfied termination in prev iteration
        self.finished_both = False  # Satisfied termination in current and previous
        self.desiccated = False  # Entire volatile inventory has been lost
        self.crystallized = (
            False  # Mantle solidified, outgassing stopped but evolution continues
        )
        self.lockfile = '/tmp/none'  # Path to keepalive file

        # Default values for mors.spada cases
        self.star_props = None
        self.star_struct = None

        # Default values for mors.baraffe cases
        self.stellar_track = None
        self.star_modern_fl = None
        self.star_modern_wl = None

        # Stellar spectrum (wavelengths, fluxes)
        self.star_wl = None
        self.star_fl = None

        # Time at which star was last updated
        self.sspec_prev = -np.inf  # spectrum
        self.sinst_prev = -np.inf  # instellation and radius

        # State at which structure was last re-computed via Zalmoxis
        self.last_struct_time = -np.inf
        self.last_struct_Tmagma = np.inf
        self.last_struct_Phi = np.inf

    def _get_initial_tmagma(self) -> float:
        """Get the initial surface temperature from the solver config.

        Returns
        -------
        float
            Initial magma ocean surface temperature [K].
        """
        return float(self.config.planet.tsurf_init)

    def init_directories(self):
        """Initialize directories dictionary"""
        from proteus.utils.coupler import set_directories

        self.directories = set_directories(self.config)

    def start(self, *, resume: bool = False, offline: bool = False):
        """Start PROTEUS simulation.

        Parameters
        ----------
        resume : bool
            If True, continue from previous simulation.
        offline : bool
            Run in offline mode; do not try to connect to the internet.
        """

        # Import things needed to run PROTEUS
        #    atmospheric chemistry
        from proteus.atmos_chem.wrapper import run_chemistry

        #    atmosphere solver
        from proteus.atmos_clim import run_atmosphere
        from proteus.atmos_clim.common import Albedo_t, Atmos_t

        #    escape and outgas
        from proteus.escape.wrapper import run_escape

        #    interior
        from proteus.interior_energetics.common import Interior_t
        from proteus.interior_energetics.wrapper import (
            get_nlevb,
            reset_run_state,
            run_interior,
            solve_structure,
            update_planet_mass,
        )

        # Clear module-level consecutive-failure counters so a prior run
        # in the same Python process (pytest session, `proteus grid`
        # ensemble, Jupyter kernel) cannot leave a stale counter that
        # trips this run's abort threshold on the first solver failure.
        reset_run_state()

        #    synthetic observations
        from proteus.observe.wrapper import run_observe

        #    orbit
        from proteus.orbit.wrapper import init_orbit, run_orbit

        #    outgassing
        from proteus.outgas.wrapper import (
            calc_target_elemental_inventories,
            check_desiccation,
            run_crystallized,
            run_desiccated,
            run_outgassing,
        )

        #   stellar spectrum and evolution
        from proteus.star.wrapper import (
            get_new_spectrum,
            init_star,
            scale_spectrum_to_toa,
            update_stellar_mass,
            update_stellar_quantities,
            write_spectrum,
        )

        #   other utilities
        from proteus.utils.coupler import (
            CreateHelpfileFromDict,
            CreateLockFile,
            ExtendHelpfile,
            PrintCurrentState,
            ReadHelpfileFromCSV,
            UpdatePlots,
            WriteHelpfileToCSV,
            ZeroHelpfileRow,
            print_citation,
            print_header,
            print_module_configuration,
            print_stoptime,
            print_system_configuration,
            remove_excess_files,
            validate_module_versions,
        )

        #    lookup and reference data
        from proteus.utils.data import download_sufficient_data

        # termination criteria
        from proteus.utils.terminate import check_termination, print_termination_criteria

        # First things
        start_time = datetime.now()
        self.config.params.resume = resume
        self.config.params.offline = offline
        UpdateStatusfile(self.directories, 0)

        # Clean output directory if starting fresh
        if not self.config.params.resume:
            CleanDir(self.directories['output'])
            CleanDir(self.directories['output/data'])
            CleanDir(self.directories['output/observe'])
            CleanDir(self.directories['output/offchem'])
            CleanDir(self.directories['output/plots'])

        # Get next logfile path
        logindex = 1 + GetCurrentLogfileIndex(self.directories['output'])
        logpath = GetLogfilePath(self.directories['output'], logindex)

        # Switch to logger
        setup_logger(logpath=logpath, logterm=True, level=self.config.params.out.logging)
        log = logging.getLogger('fwl.' + __name__)

        # Print header
        print_header()

        # Print system configuration
        print_system_configuration(self.directories)

        # Print module configuration
        print_module_configuration(self.directories, self.config, self.config_path)

        # Ensure that submodules are on the correct versions
        validate_module_versions(self.directories, self.config)

        # Print termination criteria
        print_termination_criteria(self.config)
        PrintHalfSeparator()

        # Count iterations
        self.loops = {
            'total': 0,  # Total number of iters performed
            'total_min': self.config.params.stop.iters.minimum,
            'total_loops': self.config.params.stop.iters.maximum,
            'init_loops': 3,  # Maximum number of init iters
        }
        self.init_stage = True

        # Write config to output directory, for future reference
        self.config.write(os.path.join(self.directories['output'], 'init_coupler.toml'))

        # Create lockfile for keeping simulation running
        self.lockfile = CreateLockFile(self.directories['output'])

        # Download basic data
        download_sufficient_data(self.config)

        # Initialise interior object
        if self.config.interior_energetics.module == 'spider':
            spider_dir = self.directories['spider']
        else:
            spider_dir = None
        self.interior_o = Interior_t(
            get_nlevb(self.config),
            spider_dir=spider_dir,
            eos_dir=self.config.interior_struct.eos_dir,
        )

        # Initialise atmosphere object
        self.atmos_o = Atmos_t()
        if self.config.atmos_clim.albedo_from_file:
            log.debug('Reading albedo data from file')
            self.atmos_o.albedo_o = Albedo_t(self.config.atmos_clim.albedo_pl)
            if not self.atmos_o.albedo_o.ok:
                UpdateStatusfile(self.directories, 22)
                raise RuntimeError('Problem when loading albedo data file')

        # Is the model resuming from a previous state?
        if not self.config.params.resume:
            # New simulation

            # SPIDER initial condition
            self.interior_o.ic = 1

            # Create an empty initial row for helpfile
            self.hf_row = ZeroHelpfileRow()

            # Stellar mass
            update_stellar_mass(self.hf_row, self.config)

            # Initial time
            self.hf_row['Time'] = 0.0
            self.hf_row['age_star'] = self.config.star.age_ini * 1e9

            # Initial guess for flux.
            # When flux_guess < 0 (sentinel), compute from Stefan-Boltzmann:
            # sigma * T_magma^4. This adapts to any initial temperature and
            # ensures parity between SPIDER and Aragog. flux_guess=0 is valid
            # (zero flux) and will NOT trigger the automatic computation.
            flux_guess = self.config.interior_energetics.flux_guess
            if flux_guess < 0:
                from scipy.constants import Stefan_Boltzmann

                T_ini = self._get_initial_tmagma()
                flux_guess = Stefan_Boltzmann * T_ini**4
                log.info(
                    'flux_guess from sigma*T^4: T_magma=%.0f K -> F=%.2e W/m^2',
                    T_ini,
                    flux_guess,
                )
            self.hf_row['F_atm'] = flux_guess
            self.hf_row['F_int'] = self.hf_row['F_atm']
            self.hf_row['T_eqm'] = 2000.0

            # Validate cross-config constraints
            if (
                self.config.planet.temperature_mode == 'accretion'
                and self.config.interior_struct.module == 'spider'
            ):
                raise ValueError(
                    "temperature_mode='accretion' requires "
                    "interior_struct.module='zalmoxis' (needs Zalmoxis "
                    'structure for gravitational energy computation)'
                )

            # Solve interior structure
            solve_structure(
                self.directories,
                self.config,
                self.hf_all,
                self.hf_row,
                self.directories['output'],
            )

            # Initialize structure-update sentinels so the first dynamic
            # update uses the post-init state as its baseline
            self.last_struct_time = 0.0
            self.last_struct_Tmagma = self.hf_row.get('T_magma', np.inf)
            self.last_struct_Phi = self.hf_row.get('Phi_global', np.inf)

            # Store partial pressures and list of included volatiles
            inc_gases = []
            for s in vol_list:
                if s != 'O2':
                    pp_val = self.config.planet.gas_prs.get_pressure(s)
                    include = self.config.outgas.calliope.is_included(s)
                else:
                    pp_val = 0.0
                    include = True

                if include:
                    inc_gases.append(s)
                    self.hf_row[s + '_bar'] = max(1.0e-30, float(pp_val))
                else:
                    self.hf_row[s + '_bar'] = 0.0
            for s in vap_list:
                inc_gases.append(s)
                self.hf_row[s + '_bar'] = 0.0

            # Inform user
            log.info("Initial inventory set by '%s'" % self.config.planet.volatile_mode)
            log.info('Included gases:')
            for s in inc_gases:
                write = '    '
                write += 'vapour  ' if s in vap_list else 'volatile'
                write += '  %-8s' % s
                if self.config.planet.volatile_mode == 'gas_prs':
                    write += ' : %6.2f bar' % self.hf_row[s + '_bar']
                log.info(write)

            # Equilibrate structure + composition before main loop.
            # Iterates CALLIOPE + Zalmoxis (no SPIDER) until R_int and
            # P_surf converge. Only active when config flag is set.
            if (
                self.config.interior_struct.zalmoxis.equilibrate_init
                and self.config.interior_struct.module == 'zalmoxis'
            ):
                from proteus.interior_energetics.wrapper import equilibrate_initial_state

                equilibrate_initial_state(
                    self.directories,
                    self.config,
                    self.hf_row,
                    self.directories['output'],
                )
                # Update sentinels with post-equilibration state
                self.last_struct_Tmagma = self.hf_row.get('T_magma', np.inf)
                self.last_struct_Phi = self.hf_row.get('Phi_global', np.inf)

        else:
            # Resuming from disk
            log.info('Resuming the simulation from the disk')

            # Read helpfile from disk
            self.hf_all = ReadHelpfileFromCSV(self.directories['output'])

            # Check length
            if len(self.hf_all) <= self.loops['init_loops'] + 1:
                UpdateStatusfile(self.directories, 20)
                raise RuntimeError('Simulation is too short to be resumed')

            # Get last row from helpfile dataframe
            self.hf_row = self.hf_all.iloc[-1].to_dict()

            # Resume banner: since proteus_00.log is opened in append mode on
            # resume, every prior session's banner + output stays in the file
            # with no visible marker of where the new session picks up. A
            # self-contained three-line resume banner makes log triage
            # (grep, tail -f, monitor cron filters) tractable. This is
            # cosmetic only; no state is changed.
            log.info('=' * 60)
            log.info(
                '=== RESUME at helpfile row %d, t = %.3e yr, Phi = %.4f',
                len(self.hf_all), float(self.hf_row.get('Time', 0.0)),
                float(self.hf_row.get('Phi_global', float('nan'))),
            )
            log.info('=' * 60)

            # Check if the planet is desiccated
            self.desiccated = check_desiccation(self.config, self.hf_row)

            # Extract all archived data files
            log.debug('Extracting archived data files')
            self.extract_archives()

            # Interior initial condition
            self.interior_o.ic = 2

            # Restore tides data
            if self.config.orbit.module is not None:
                self.interior_o.resume_tides(self.directories['output'])

            # Set loop counters
            self.loops['total'] = len(self.hf_all)
            self.init_stage = False

            # Restore Zalmoxis mesh path for resumed SPIDER runs
            if (
                self.config.interior_struct.module == 'zalmoxis'
                and self.config.interior_energetics.module == 'spider'
            ):
                mesh_path = os.path.join(self.directories['output'], 'data', 'spider_mesh.dat')
                if os.path.isfile(mesh_path):
                    self.directories['spider_mesh'] = mesh_path
                    prev_path = mesh_path + '.prev'
                    if os.path.isfile(prev_path):
                        self.directories['spider_mesh_prev'] = prev_path
                    self.directories['mesh_shift_active'] = False
                    self.directories['mesh_convergence_steps'] = 0
                    log.info('Restored Zalmoxis mesh file: %s', mesh_path)

            # Restore spider_eos tables pointer for resumed SPIDER / Aragog
            # runs. The initial-structure path (solve_structure +
            # determine_interior_radius_with_zalmoxis -> generate_spider_tables)
            # populates dirs['spider_eos_dir'] on a fresh run, but that path
            # is skipped on resume. Without the rehydration, SPIDER's
            # _try_spider raises
            # `FileNotFoundError: interior_struct.eos_dir must be set when
            # no Zalmoxis-generated EOS tables are available`. Same issue
            # bites Aragog when it needs the P-S tables at re-init.
            # Observed 2026-04-21 on SPIDER Run C resume.
            eos_dir_restored = os.path.join(
                self.directories['output'], 'data', 'spider_eos'
            )
            if os.path.isdir(eos_dir_restored):
                self.directories['spider_eos_dir'] = eos_dir_restored
                solidus_ps = os.path.join(eos_dir_restored, 'solidus_P-S.dat')
                liquidus_ps = os.path.join(eos_dir_restored, 'liquidus_P-S.dat')
                if os.path.isfile(solidus_ps):
                    self.directories['spider_solidus_ps'] = solidus_ps
                if os.path.isfile(liquidus_ps):
                    self.directories['spider_liquidus_ps'] = liquidus_ps
                log.info('Restored spider_eos_dir: %s', eos_dir_restored)

            # Initialize structure-update sentinels from last helpfile row
            self.last_struct_time = self.hf_row.get('Time', 0.0)
            self.last_struct_Tmagma = self.hf_row.get('T_magma', np.inf)
            self.last_struct_Phi = self.hf_row.get('Phi_global', np.inf)

        log.info(' ')

        # Prepare star stuff
        init_star(self)

        # Prepare orbit stuff
        init_orbit(self)

        # Track the last simulation time at which data was written to disk,
        # so that dt_write_rel can suppress high-frequency writes during
        # rapid early evolution. Initialised to -inf so the first eligible
        # iteration always writes.
        self.last_write_time = -np.inf

        # Deadlock detector for the atmosphere-interior coupling.
        # Counts consecutive iterations in which the atmosphere solver did
        # NOT converge AND the interior state (T_magma, Phi_global, F_atm)
        # is bit-exactly identical to the previous iteration. When the
        # counter reaches `agni_deadlock_max`, the run aborts with status 22.
        # This catches the failure mode where AGNI returns "Maximum attempts"
        # and PROTEUS would otherwise silently accept a frozen state and
        # advance Time indefinitely. See output_files/analysis_1EO_M1_profiles
        # for the diagnosis.
        self.agni_deadlock_count = 0
        self.agni_deadlock_max = 3

        # Main loop
        # Collects the index of the snapshots that already underwent a VULCAN calculation to avoid repeating:
        vulcan_completed_loops = set()
        UpdateStatusfile(self.directories, 1)
        while not self.finished_both:
            # Determine whether this iteration is a data-write snapshot.
            # Two conditions must both be satisfied:
            #   1. iteration count matches write_mod (existing behaviour)
            #   2. enough simulation time has elapsed since the last write
            #      (relative guard: min interval = dt_write_rel * Time)
            iter_ok = multiple(self.loops['total'], self.config.params.out.write_mod)
            dt_write_rel = self.config.params.out.dt_write_rel
            cur_time = self.hf_row.get('Time', 0.0)
            time_ok = dt_write_rel <= 0 or (
                cur_time - self.last_write_time >= dt_write_rel * max(cur_time, 1.0)
            )
            is_snapshot = iter_ok and time_ok
            # New rows
            if self.loops['total'] > 0:
                # Create new row to hold the updated variables. This will be
                #    overwritten by the routines below.
                self.hf_row = self.hf_all.iloc[-1].to_dict()
            log.info(' ')
            PrintSeparator()
            log.info('Loop counters')
            log.info('current    init    maximum')
            log.info(
                ' %6d    %4d     %6d '
                % (
                    self.loops['total'],
                    self.loops['init_loops'],
                    self.loops['total_loops'],
                )
            )

            ############### INTERIOR
            PrintHalfSeparator()

            # Evolve interior
            run_interior(
                self.directories,
                self.config,
                self.hf_all,
                self.hf_row,
                self.interior_o,
                write_data=is_snapshot,
            )

            # Advance current time in main loop according to interior step
            self.hf_row['Time'] += self.interior_o.dt  # in years
            self.hf_row['age_star'] += self.interior_o.dt  # in years

            # Re-compute structure if Zalmoxis feedback is active
            if (
                not self.init_stage
                and self.config.interior_struct.module == 'zalmoxis'
                and self.config.interior_struct.zalmoxis.update_interval > 0
            ):
                from proteus.interior_energetics.wrapper import update_structure_from_interior

                (
                    self.last_struct_time,
                    self.last_struct_Tmagma,
                    self.last_struct_Phi,
                ) = update_structure_from_interior(
                    self.directories,
                    self.config,
                    self.hf_row,
                    self.interior_o,
                    self.last_struct_time,
                    self.last_struct_Tmagma,
                    self.last_struct_Phi,
                )
                # gc.collect() already called inside update_structure_from_interior()

            ############### / INTERIOR AND STRUCTURE

            ############### ORBIT AND TIDES
            PrintHalfSeparator()
            run_orbit(self.hf_row, self.config, self.directories, self.interior_o)

            ############### / ORBIT AND TIDES

            ############### STELLAR FLUX MANAGEMENT
            PrintHalfSeparator()
            log.info('Stellar flux management...')
            update_stellar_spectrum = False

            # Calculate new instellation and radius
            if (
                abs(self.hf_row['Time'] - self.sinst_prev) > self.config.params.dt.starinst
            ) or (self.loops['total'] == 0):
                self.sinst_prev = self.hf_row['Time']

                update_stellar_quantities(
                    self.hf_row, self.config, stellar_track=self.stellar_track
                )

            # Calculate a new (historical) stellar spectrum
            if (
                abs(self.hf_row['Time'] - self.sspec_prev) > self.config.params.dt.starspec
            ) or (self.loops['total'] == 0):
                self.sspec_prev = self.hf_row['Time']
                update_stellar_spectrum = True

                # Get the new spectrum using the appropriate module
                log.info('Updating stellar spectrum')
                self.star_wl, self.star_fl = get_new_spectrum(
                    # Required variables
                    self.hf_row['age_star'],
                    self.config,
                    # Variables needed for mors.spada
                    star_struct_modern=self.star_struct,
                    star_props_modern=self.star_props,
                    # Variables needed for mors.baraffe
                    stellar_track=self.stellar_track,
                    modern_wl=self.star_modern_wl,
                    modern_fl=self.star_modern_fl,
                )

                # Scale fluxes from 1 AU to TOA
                self.star_fl = scale_spectrum_to_toa(self.star_fl, self.hf_row['separation'])

                # Save spectrum to file
                write_spectrum(
                    self.star_wl, self.star_fl, self.hf_row, self.directories['output']
                )

            else:
                log.info('Updated spectrum not required')

            ############### / STELLAR FLUX MANAGEMENT

            ############### ESCAPE
            if (self.loops['total'] > self.loops['init_loops'] + 2) and (not self.desiccated):
                PrintHalfSeparator()
                run_escape(self.config, self.hf_row, self.directories, self.interior_o.dt)

            ############### / ESCAPE

            ############### REDOX (#57)
            # Static mode (default): pins hf_row['fO2_shift_IW'] to
            # config.outgas.fO2_shift_IW AND writes passive
            # R_budget_* diagnostics. No-op on physics; the downstream
            # run_outgassing call below picks up the pinned scalar
            # exactly as pre-#57.
            # Evolving modes ('fO2_init' or 'composition'): the
            # transactional Brent solver (plan v6 §3.3) finds ΔIW
            # that closes the Evans/Hirschmann redox budget using
            # deep-copied probes of hf_row per inner outgas call.
            # Only the solved fO2_shift_IW scalar is committed to
            # hf_row; the authoritative outgas commit remains the
            # run_outgassing call below.
            # Init-stage guard (Commit C.1): during init
            # (loops['total'] <= init_loops + 2) evolving modes
            # short-circuit to static. calc_target_elemental_inventories
            # has not yet converged, so a Brent solve at that stage
            # would commit a meaningless ΔIW.
            try:
                from proteus.redox.coupling import (
                    _write_passive_diagnostics,
                    run_redox_step,
                )
                from proteus.utils.coupler import populate_core_composition
                # Populate {element}_kg_core from CoreComp × M_core
                # (#57 plan v6 §3.5). Cheap; idempotent. Core mass
                # and composition are both static in stub mode, so a
                # re-call is a no-op, but a future #526 hook that
                # mutates CoreComp will see consistent state.
                populate_core_composition(self.hf_row, self.config)
                redox_mode = getattr(
                    getattr(self.config, 'redox', None), 'mode', 'static',
                )
                in_init = self.loops['total'] <= (
                    self.loops['init_loops'] + 2
                )
                if redox_mode != 'static' and in_init:
                    # Early-iteration short-circuit.
                    # In `composition` mode, try to seed fO2 from the
                    # mantle composition via the chosen oxybarometer
                    # (plan v6 §3.6). If the seed returns NaN (oxybaro
                    # failure, missing MantleComp), fall back to the
                    # pinned config.outgas.fO2_shift_IW.
                    _mc = getattr(
                        self.config.interior_struct, 'mantle_comp', None,
                    )
                    if redox_mode == 'composition':
                        from proteus.interior_energetics.aragog import (
                            get_mesh_state_for_redox,
                        )
                        from proteus.redox.coupling import (
                            seed_composition_fO2,
                        )
                        _ms = None
                        if getattr(
                            self.config.interior_energetics, 'module', None,
                        ) == 'aragog':
                            _ms = get_mesh_state_for_redox(self.interior_o)
                        delta_iw = seed_composition_fO2(
                            self.hf_row, self.config, mesh_state=_ms,
                        )
                        if not (delta_iw == delta_iw):    # NaN check
                            self.hf_row['fO2_shift_IW'] = float(
                                self.config.outgas.fO2_shift_IW
                            )
                    else:
                        self.hf_row['fO2_shift_IW'] = float(
                            self.config.outgas.fO2_shift_IW
                        )
                    _write_passive_diagnostics(self.hf_row, _mc)
                else:
                    from proteus.escape.wrapper import (
                        atm_escape_fluxes_species_kg,
                    )
                    from proteus.interior_energetics.aragog import (
                        get_mesh_state_for_redox,
                    )
                    mesh_state = None
                    if getattr(
                        self.config.interior_energetics, 'module', None,
                    ) == 'aragog':
                        mesh_state = get_mesh_state_for_redox(
                            self.interior_o
                        )
                    escape_fluxes = None
                    if self.config.escape.module:
                        escape_fluxes = atm_escape_fluxes_species_kg(
                            self.hf_row, self.interior_o.dt,
                        )
                    hf_row_prev_dict = (
                        self.hf_all.iloc[-1].to_dict()
                        if (self.hf_all is not None and len(self.hf_all) > 0)
                        else dict(self.hf_row)
                    )
                    run_redox_step(
                        self.hf_row, hf_row_prev_dict,
                        self.directories, self.config,
                        mesh_state=mesh_state,
                        escape_fluxes_species_kg=escape_fluxes,
                    )
            except (ValueError, KeyError, RuntimeError,
                    FloatingPointError) as redox_exc:
                # Domain-specific failures fall back to static.
                # AttributeError / TypeError / ImportError propagate
                # intentionally — they indicate a real bug and should
                # not be silently swallowed.
                log.warning(
                    'Redox step raised %s (%s); falling back to static '
                    'fO2_shift_IW for this step.',
                    type(redox_exc).__name__, redox_exc,
                )
                self.hf_row['fO2_shift_IW'] = float(
                    self.config.outgas.fO2_shift_IW
                )
            ############### / REDOX

            ############### OUTGASSING
            PrintHalfSeparator()

            # Recalculate mass targets during init phase, since these will be adjusted
            #    depending on the true melt fraction and T_magma found by SPIDER at runtime.
            if self.init_stage:
                calc_target_elemental_inventories(self.directories, self.config, self.hf_row)

            else:
                # Check crystallization: outgassing stops but simulation continues
                # TODO (future development): Disequilibrium crystallization.
                # The current framework assumes local thermodynamic equilibrium: melt
                # fraction is determined by the local P-T via the melting curves.
                # Fractional crystallization with compositional zonation requires
                # explicit tracking of the solid composition field, which is beyond
                # the current solver capabilities. See Boujibar+2020 for discussion.
                if self.config.params.stop.solid.freeze_volatiles and not self.crystallized:
                    if (
                        self.hf_row.get('Phi_global', 1.0)
                        <= self.config.params.stop.solid.phi_crit
                    ):
                        self.crystallized = True
                        log.info(
                            'Mantle crystallized (Phi_global <= %.3f). '
                            'Outgassing stopped. Dissolved volatiles trapped in solid mantle.',
                            self.config.params.stop.solid.phi_crit,
                        )

                # Check desiccation (can happen even if crystallized, via escape)
                if not self.desiccated:
                    self.desiccated = check_desiccation(self.config, self.hf_row)

            # Handle volatile exchange
            if self.desiccated:
                run_desiccated(self.config, self.hf_row)
            elif self.crystallized:
                run_crystallized(self.config, self.hf_row)
            else:
                run_outgassing(self.directories, self.config, self.hf_row)

            # Add mass of total volatile element mass (M_ele) to total mass of mantle+core
            update_planet_mass(self.hf_row)

            ############### / OUTGASSING

            ############### ATMOSPHERE CLIMATE
            PrintHalfSeparator()

            # When global_miscibility is enabled, the atmosphere lower
            # boundary is the solvus (binodal surface), not the magma
            # ocean surface. Override the hf_row values that AGNI reads
            # so the atmosphere is computed from the solvus outward.
            # Save originals to restore after the atmosphere step.
            _saved_atm_bc = {}
            if (
                self.config.interior_struct.zalmoxis.global_miscibility
                and 'R_solvus' in self.hf_row
            ):
                R_sol = self.hf_row.get('R_solvus')
                if R_sol is not None and R_sol < self.hf_row['R_int']:
                    _saved_atm_bc = {
                        'T_surf': self.hf_row['T_surf'],
                        'P_surf': self.hf_row['P_surf'],
                        'R_int': self.hf_row['R_int'],
                        'T_magma': self.hf_row['T_magma'],
                    }
                    self.hf_row['T_surf'] = self.hf_row['T_solvus']
                    self.hf_row['T_magma'] = self.hf_row['T_solvus']
                    self.hf_row['P_surf'] = self.hf_row['P_solvus'] * 1e-5  # Pa -> bar
                    self.hf_row['R_int'] = R_sol

            run_atmosphere(
                self.atmos_o,
                self.config,
                self.directories,
                self.loops,
                self.star_wl,
                self.star_fl,
                update_stellar_spectrum,
                self.hf_all,
                self.hf_row,
            )

            # Restore original boundary values if they were overridden
            if _saved_atm_bc:
                for key, val in _saved_atm_bc.items():
                    self.hf_row[key] = val

            # Atmosphere-interior coupling deadlock detection.
            # If the atmosphere solver failed AND the interior state has
            # not moved since the previous committed row, increment the
            # deadlock counter. After `agni_deadlock_max` consecutive such
            # iterations, abort the run with status 22 ("Atmosphere model
            # error"). This catches the case where AGNI returns "Maximum
            # attempts" with no converged solution and the coupling layer
            # would otherwise silently freeze indefinitely. See deep
            # diagnosis at
            #   output_files/analysis_1EO_M1_profiles/DEEP_ANALYSIS_FINDINGS.md
            #
            # The "frozen" criterion uses bit-exact equality for T_magma and
            # Phi_global (which truly do not move in a deadlocked interior)
            # but a small relative tolerance for F_atm (1e-6) so that jittery
            # AGNI non-convergence noise on the same physical state still
            # registers as frozen. Without the F_atm tolerance, the detector
            # would silently miss deadlocks where AGNI returns slightly
            # different non-converged values on each retry.
            #
            # Guard: hf_all is None until the first row is appended at the
            # bottom of the loop body. On a fresh run's first iteration
            # there is no "previous" row to compare against, so the
            # deadlock detector cannot fire yet — count the failure but do
            # not abort.
            if not self.atmos_o.converged:
                if self.hf_all is not None and len(self.hf_all) >= 1:
                    prev = self.hf_all.iloc[-1]
                    cur_F = float(self.hf_row.get('F_atm', 0.0))
                    prev_F = float(prev.get('F_atm', 0.0))
                    F_rel_change = abs(cur_F - prev_F) / max(abs(prev_F), 1.0)
                    interior_frozen = (
                        float(prev.get('T_magma', 0.0))
                            == float(self.hf_row.get('T_magma', 0.0))
                        and float(prev.get('Phi_global', 0.0))
                            == float(self.hf_row.get('Phi_global', 0.0))
                        and F_rel_change < 1.0e-6
                    )
                else:
                    interior_frozen = False

                if interior_frozen:
                    self.agni_deadlock_count += 1
                    log.warning(
                        'AGNI did not converge AND interior state is frozen '
                        '(consecutive deadlock count = %d / %d)',
                        self.agni_deadlock_count,
                        self.agni_deadlock_max,
                    )
                    if self.agni_deadlock_count >= self.agni_deadlock_max:
                        log.error(
                            'Atmosphere-interior coupling deadlock detected: '
                            'AGNI failed to converge for %d consecutive '
                            'iterations with no change in (T_magma, Phi_global, '
                            'F_atm). Aborting to prevent an indefinite '
                            'stuck-loop. Try (a) reducing the interior dt, '
                            '(b) switching AGNI to a more robust solver mode, '
                            'or (c) checking that the surface boundary '
                            'condition has not entered a regime AGNI cannot '
                            'represent (e.g. very thick H2-rich atmospheres '
                            'at the rheological transition). See '
                            'output_files/analysis_1EO_M1_profiles/'
                            'DEEP_ANALYSIS_FINDINGS.md for the diagnosis.',
                            self.agni_deadlock_count,
                        )
                        UpdateStatusfile(self.directories, 22)
                        raise RuntimeError(
                            'Atmosphere-interior coupling deadlock: '
                            f'{self.agni_deadlock_count} consecutive AGNI '
                            'failures with frozen interior state.'
                        )
                else:
                    # AGNI failed but interior is still moving — reset.
                    self.agni_deadlock_count = 0
            else:
                # AGNI converged — clear the counter.
                self.agni_deadlock_count = 0

            ############### / ATMOSPHERE CLIMATE

            ############### ONLINE ATMOSPHERIC CHEMISTRY
            if (
                self.config.atmos_chem.when == 'online'
            ):  # checking if the toml file says online at atmos_chem and then when
                if (
                    is_snapshot and not self.desiccated
                ):  # checking if the loop is a snapshot and runs VULCAN
                    if self.loops['total'] not in vulcan_completed_loops:
                        run_chemistry(self.directories, self.config, self.hf_row)
                        vulcan_completed_loops.add(
                            self.loops['total']
                        )  # adds it to the completed loops/snapshots
            ############### / ONLINE ATMOSPHERIC CHEMISTRY

            ############### HOUSEKEEPING AND CONVERGENCE CHECK

            PrintHalfSeparator()

            # Update model wall-clock runtime
            run_time = datetime.now() - start_time
            self.hf_row['runtime'] = float(run_time.total_seconds())

            # Adjust total iteration counters
            self.loops['total'] += 1

            # Init stage?
            if self.loops['total'] > self.loops['init_loops']:
                self.init_stage = False

            # Keep time at zero during init stage
            if self.init_stage:
                self.hf_row['Time'] = 0.0
            else:
                self.interior_o.ic = 2

            # Update full helpfile
            if self.loops['total'] > 1:
                # append row
                self.hf_all = ExtendHelpfile(self.hf_all, self.hf_row)
            else:
                # first iter => generate new HF from dict
                self.hf_all = CreateHelpfileFromDict(self.hf_row)

            # Write helpfile to disk (gated by is_snapshot, which
            # combines write_mod iteration check and dt_write time check)
            if is_snapshot:
                WriteHelpfileToCSV(self.directories['output'], self.hf_all)
                self.last_write_time = self.hf_row.get('Time', 0.0)

            # Print info to terminal and log file
            PrintCurrentState(self.hf_row)

            # Check for convergence
            if not self.init_stage:
                log.info('Checking convergence criteria')
                check_termination(self)

            # Make plots
            if (
                is_snapshot
                and multiple(self.loops['total'], self.config.params.out.plot_mod)
                and not self.finished_both
            ):
                log.info('Making plots')
                UpdatePlots(self.hf_all, self.directories, self.config)

            # Update or create data archive
            if (
                is_snapshot
                and multiple(self.loops['total'], self.config.params.out.archive_mod)
                and not self.finished_both
            ):
                log.info('Updating archive of model output data')
                # do not remove ALL files
                archive.update(self.directories['output/data'], remove_files=False)
                # remove all files EXCEPT the latest ones
                archive.remove_old(self.directories['output/data'], self.hf_row['Time'] * 0.99)

            ############### / HOUSEKEEPING AND CONVERGENCE CHECK

        # Write conditions at the end of simulation
        log.info('Writing data')
        WriteHelpfileToCSV(self.directories['output'], self.hf_all)

        # Ensure the final interior state is on disk so resume can find it.
        # dt_write_rel may have suppressed the write on the last iteration.
        if (
            self.config.interior_energetics.module == 'aragog'
            and self.interior_o.aragog_solver is not None
        ):
            from proteus.interior_energetics.aragog import AragogRunner

            out = self.interior_o.aragog_solver.get_state()
            AragogRunner._write_output_ncdf(
                self.directories['output'], self.hf_row['Time'], out
            )

        # Run offline chemistry
        if self.config.atmos_chem.when == 'offline':
            log.info(' ')
            PrintSeparator()
            if self.desiccated:
                log.warning('Cannot calculate atmospheric chemistry after desiccation')
            else:
                run_chemistry(self.directories, self.config, self.hf_row)

        # Synthetic observations
        if self.config.observe.synthesis is not None:
            log.info(' ')
            PrintSeparator()
            if self.desiccated:
                log.warning('Cannot observe planet after desiccation')
            else:
                run_observe(self.hf_row, self.directories['output'], self.config)

        # Make final plots
        if self.config.params.out.plot_mod is not None:
            log.info('Making final plots')
            UpdatePlots(self.hf_all, self.directories, self.config, end=True)

        # Tidy up
        log.info(' ')
        log.debug('Tidy up before exit')
        remove_excess_files(
            self.directories['output'], rm_spectralfiles=self.config.params.out.remove_sf
        )

        # Archive the folder ./output/data/, and remove files
        if self.config.params.out.archive_mod is not None:
            log.info('Archiving output data into tar files')
            archive.update(self.directories['output/data'], remove_files=True)

        # Stop time and model duration
        print_stoptime(start_time)

        # Print citation
        print_citation(self.config)

    def extract_archives(self):
        """
        Extract archived data files in subfolders of the output directory
        """
        archive.extract(self.directories['output/data'], remove_tar=True)

    def create_archives(self):
        """
        Pack data files in subfolders of the output directory into archival tar files
        """
        archive.create(self.directories['output/data'], remove_files=True)

    def observe(self):
        # Extract archived data
        self.extract_archives()

        # Load data from helpfile
        from proteus.utils.coupler import ReadHelpfileFromCSV

        hf_all = ReadHelpfileFromCSV(self.directories['output'])

        # Check length
        if len(hf_all) < 1:
            raise Exception('Simulation is too short to be postprocessed')

        # Get last row
        hf_row = hf_all.iloc[-1].to_dict()

        # Run observations pipeline, typically invoked via CLI
        from proteus.observe.wrapper import run_observe

        run_observe(hf_row, self.directories['output'], self.config)

    def offline_chemistry(self):
        # Extract archived data
        self.extract_archives()

        # Load data from helpfile
        from proteus.utils.coupler import ReadHelpfileFromCSV

        hf_all = ReadHelpfileFromCSV(self.directories['output'])

        # Check length
        if len(hf_all) < 1:
            raise Exception('Simulation is too short to be postprocessed')

        # Get last row
        hf_row = hf_all.iloc[-1].to_dict()

        # Run offline chemistry, typically invoked via CLI
        from proteus.atmos_chem.wrapper import run_chemistry

        result = run_chemistry(self.directories, self.config, hf_row)

        # return the dataframe
        return result
