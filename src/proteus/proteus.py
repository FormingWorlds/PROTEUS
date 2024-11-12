from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
import numpy as np

from proteus.config import read_config_object
from proteus.atmos_clim import RunAtmosphere
from proteus.escape.wrapper import RunEscape
from proteus.interior import run_interior
from proteus.outgas.wrapper import calc_target_elemental_inventories, run_outgassing
from proteus.orbit.wrapper import update_separation, update_period
from proteus.star.wrapper import (
    get_new_spectrum,
    init_star,
    scale_spectrum_to_toa,
    update_stellar_quantities,
    write_spectrum,
)
from proteus.utils.constants import (
    AU,
    M_earth,
    R_earth,
    const_G,
    gas_list,
    vap_list,
    vol_list,
)
from proteus.utils.coupler import (
    CreateHelpfileFromDict,
    CreateLockFile,
    ExtendHelpfile,
    PrintCurrentState,
    ReadHelpfileFromCSV,
    SetDirectories,
    UpdatePlots,
    WriteHelpfileToCSV,
    ZeroHelpfileRow,
    print_header,
    print_module_configuration,
    print_system_configuration,
)
from proteus.utils.data import download_sufficient_data
from proteus.utils.helper import (
    CleanDir,
    PrintHalfSeparator,
    PrintSeparator,
    UpdateStatusfile,
    safe_rm,
)
from proteus.utils.logs import (
    GetCurrentLogfileIndex,
    GetLogfilePath,
    setup_logger,
)
from calliope.structure import calculate_mantle_mass


class Proteus:
    def __init__(self, *, config_path: Path | str) -> None:

        # Read and parse configuration file
        self.config_path = config_path
        self.config = read_config_object(config_path)

        # Setup directories dictionary
        self.init_directories()

        # Helpfile variables for the current iteration
        self.hf_row = None

        # Helpfile variables from all previous iterations
        self.hf_all = None

        # Loop counters
        self.loops = None

        # Interior
        self.IC_INTERIOR = -1   # initial condition
        self.dt = 1.0           # Time step variable [yr]

        # Model has finished?
        self.finished = False
        self.lockfile = "/tmp/none"

        # Default values for mors.spada cases
        self.star_props  = None
        self.star_struct = None

        # Default values for mors.baraffe cases
        self.stellar_track  = None
        self.star_modern_fl = None
        self.star_modern_wl = None

        # Stellar spectrum (wavelengths, fluxes)
        self.star_wl = None
        self.star_fl = None

        # Time at which star was last updated
        self.sspec_prev = -np.inf   # spectrum
        self.sinst_prev = -np.inf   # instellation and radius

    def init_directories(self):
        """Initialize directories dictionary"""
        self.directories = SetDirectories(self.config)

    def start(self, *, resume: bool = False):
        """Start PROTEUS simulation.

        Parameters
        ----------
        resume : bool
            If True, continue from previous simulation
        """

        # First things
        start_time = datetime.now()
        UpdateStatusfile(self.directories, 0)

        # Clean output directory
        if not resume:
            CleanDir(self.directories["output"])
            CleanDir(os.path.join(self.directories["output"], "data"))

        # Get next logfile path
        logindex = 1 + GetCurrentLogfileIndex(self.directories["output"])
        logpath = GetLogfilePath(self.directories["output"], logindex)

        # Switch to logger
        setup_logger(logpath=logpath, logterm=True, level=self.config.params.out.logging)
        log = logging.getLogger("fwl."+__name__)

        # Print header
        print_header()

        # Print system configuration
        print_system_configuration(self.directories)

        # Print module configuration
        print_module_configuration(self.directories, self.config, self.config_path)
        PrintHalfSeparator()

        # Count iterations
        self.loops = {
            "total": 0,  # Total number of iters performed
            "total_min": 5,  # Minimum number of total loops
            "total_loops": self.config.params.stop.iters.maximum,  # Maximum number of total loops
            "init": 0,  # Number of init iters performed
            "init_loops": 2,  # Maximum number of init iters
            "steady": -1,  # Number of iterations passed since steady-state declared
            "steady_loops": 3,  # Number of iterations to perform post-steady state
            "steady_check": 15,  # Number of iterations to look backwards when checking steady state
        }

        # Write config to output directory, for future reference
        self.config.write(os.path.join(self.directories["output"], "init_coupler.toml"))

        # Is the model resuming from a previous state?
        if not resume:
            # New simulation

            # SPIDER initial condition
            self.IC_INTERIOR = 1

            # Create an empty initial row for helpfile
            self.hf_row = ZeroHelpfileRow()

            # Initial time
            self.hf_row["Time"] = 0.0
            self.hf_row["age_star"] = self.config.star.age_ini * 1e9

            # Initial guess for flux
            self.hf_row["F_atm"] = self.config.interior.F_initial
            self.hf_row["F_int"] = self.hf_row["F_atm"]
            self.hf_row["T_eqm"] = 2000.0

            # Planet size conversion, and calculate mantle mass (= liquid + solid)
            self.hf_row["M_int"] = self.config.struct.mass * M_earth
            self.hf_row["R_int"] = self.config.struct.radius * R_earth
            self.hf_row["gravity"] = const_G * self.hf_row["M_int"] / (self.hf_row["R_int"] ** 2.0)
            self.hf_row["M_mantle"] = calculate_mantle_mass(
                self.hf_row["R_int"], self.hf_row["M_int"], self.config.struct.corefrac
            )

            # Store partial pressures and list of included volatiles
            inc_gases = []
            for s in vol_list:
                pp_val = self.config.delivery.volatiles.get_pressure(s)
                include = self.config.outgas.calliope.is_included(s)

                if include:
                    inc_gases.append(s)
                    self.hf_row[s + "_bar"] = max(1.0e-30, float(pp_val))
                else:
                    self.hf_row[s + "_bar"] = 0.0
            for s in vap_list:
                inc_gases.append(s)
                self.hf_row[s + "_bar"] = 0.0

            # Inform user
            log.info("Initial inventory set by '%s'"%self.config.delivery.initial)
            log.info("Included gases")
            for s in inc_gases:
                log.info("    %s  %-8s : %6.2f bar" %
                            ("vapour  " if s in vap_list else "volatile", s,
                            self.hf_row[s + "_bar"])
                        )
        else:
            # Resuming from disk
            log.info("Resuming the simulation from the disk")

            # SPIDER initial condition
            self.IC_INTERIOR = 2

            # Read helpfile from disk
            self.hf_all = ReadHelpfileFromCSV(self.directories["output"])

            # Check length
            if len(self.hf_all) <= self.loops["init_loops"] + 1:
                raise Exception("Simulation is too short to be resumed")

            # Get last row
            self.hf_row = self.hf_all.iloc[-1].to_dict()

            # Set loop counters
            self.loops["total"] = len(self.hf_all)
            self.loops["init"] = self.loops["init_loops"] + 1
        log.info(" ")

        # Create lockfile for keeping simulation running
        self.lockfile = CreateLockFile(self.directories["output"])

        # Download basic data
        download_sufficient_data(self.config)

        # Prepare star stuff
        init_star(self)

        # Main loop
        UpdateStatusfile(self.directories, 1)
        while not self.finished:
            # New rows
            if self.loops["total"] > 0:
                # Create new row to hold the updated variables. This will be
                #    overwritten by the routines below.
                self.hf_row = self.hf_all.iloc[-1].to_dict()

            log.info(" ")
            PrintSeparator()
            log.info("Loop counters")
            log.info("init    total     steady")
            log.info(
                "%1d/%1d   %04d/%04d   %1d/%1d"
                % (
                    self.loops["init"],
                    self.loops["init_loops"],
                    self.loops["total"],
                    self.loops["total_loops"],
                    self.loops["steady"],
                    self.loops["steady_loops"],
                )
            )

            ############### INTERIOR

            # Run interior model
            self.dt = run_interior(self.directories, self.config,
                                self.loops, self.IC_INTERIOR, self.hf_all,  self.hf_row)

            # Advance current time in main loop according to interior step
            self.hf_row["Time"]     += self.dt    # in years
            self.hf_row["age_star"] += self.dt    # in years

            ############### / INTERIOR

            ############### ORBIT AND TIDES

            # Update orbital separation
            update_separation(self.hf_row,
                                self.config.orbit.semimajoraxis,
                                self.config.orbit.eccentricity)

            # Update orbital period
            update_period(self.hf_row, self.config.orbit.semimajoraxis)

            ############### / ORBIT AND TIDES

            ############### STELLAR FLUX MANAGEMENT
            PrintHalfSeparator()
            log.info("Stellar flux management...")
            update_stellar_spectrum = False

            # Calculate new instellation and radius
            if (abs(self.hf_row["Time"] - self.sinst_prev) > self.config.params.dt.starinst) or (
                self.loops["total"] == 0
            ):
                self.sinst_prev = self.hf_row["Time"]

                update_stellar_quantities(self.hf_row, self.config, stellar_track=self.stellar_track)

            # Calculate a new (historical) stellar spectrum
            if (abs(self.hf_row["Time"] - self.sspec_prev) > self.config.params.dt.starspec) or (
                self.loops["total"] == 0
            ):
                self.sspec_prev = self.hf_row["Time"]
                update_stellar_spectrum = True

                # Get the new spectrum using the appropriate module
                log.info("Updating stellar spectrum")
                self.star_wl, self.star_fl = get_new_spectrum(

                                        # Required variables
                                        self.hf_row["age_star"], self.config,

                                        # Variables needed for mors.spada
                                        star_struct_modern=self.star_struct,
                                        star_props_modern=self.star_props,

                                        # Variables needed for mors.baraffe
                                        stellar_track=self.stellar_track,
                                        modern_wl=self.star_modern_wl,
                                        modern_fl=self.star_modern_fl,
                                        )

                # Scale fluxes from 1 AU to TOA
                self.star_fl = scale_spectrum_to_toa(self.star_fl, self.hf_row["separation"])

                # Save spectrum to file
                write_spectrum(self.star_wl, self.star_fl,
                                self.hf_row, self.directories["output"])

            else:
                log.info("New spectrum not required at this time")

            ############### / STELLAR FLUX MANAGEMENT

            ############### ESCAPE
            if (self.loops["total"] >= self.loops["init_loops"]):
                RunEscape(self.config, self.hf_row, self.dt, self.stellar_track)

            ############### / ESCAPE

            ############### OUTGASSING
            PrintHalfSeparator()

            #    recalculate mass targets during init phase, since these will be adjusted
            #    depending on the true melt fraction and T_magma found by SPIDER at runtime.
            if self.loops["init"] < self.loops["init_loops"]:
                calc_target_elemental_inventories(self.directories, self.config, self.hf_row)

            # solve for atmosphere composition
            run_outgassing(self.directories, self.config, self.hf_row)

            # Add atmosphere mass to interior mass, to get total planet mass
            self.hf_row["M_planet"] = self.hf_row["M_int"] + self.hf_row["M_atm"]

            # Check for when atmosphere has escaped.
            #    This will mean that the mixing ratios become undefined, so use value of 0.
            if self.hf_row["P_surf"] < 1.0e-10:
                for gas in gas_list:
                    self.hf_row[gas+"_vmr"] = 0.0
                self.hf_row["atm_kg_per_mol"] = 0.0

            ############### / OUTGASSING

            ############### ATMOSPHERE CLIMATE
            RunAtmosphere(self.config, self.directories, self.loops,
                                self.star_wl, self.star_fl, update_stellar_spectrum,
                                self.hf_all, self.hf_row)

            ############### HOUSEKEEPING AND CONVERGENCE CHECK

            PrintHalfSeparator()

            # Update init loop counter
            # Next init iter
            if self.loops["init"] < self.loops["init_loops"]:
                self.loops["init"] += 1
                self.hf_row["Time"] = 0.0
            # Reset restart flag once SPIDER has correct heat flux
            if self.loops["total"] >= self.loops["init_loops"]:
                self.IC_INTERIOR = 2

            # Adjust total iteration counters
            self.loops["total"] += 1

            # Update full helpfile
            if self.loops["total"] > 1:
                # append row
                self.hf_all = ExtendHelpfile(self.hf_all, self.hf_row)
            else:
                # first iter => generate new HF from dict
                self.hf_all = CreateHelpfileFromDict(self.hf_row)

            # Write helpfile to disk
            WriteHelpfileToCSV(self.directories["output"], self.hf_all)

            # Print info to terminal and log file
            PrintCurrentState(self.hf_row)

            log.info("Check convergence criteria")

            # Stop simulation when planet is completely solidified
            if self.config.params.stop.solid.enabled and (self.hf_row["Phi_global"] <= self.config.params.stop.solid.phi_crit):
                UpdateStatusfile(self.directories, 10)
                log.info("")
                log.info("===> Planet solidified! <===")
                log.info("")
                self.finished = True

            # Stop simulation when flux is small
            if (
                self.config.params.stop.radeqm.enabled
                and (self.loops["total"] > self.loops["init_loops"] + 1)
                and (abs(self.hf_row["F_atm"]) <= self.config.params.stop.radeqm.F_crit)
            ):
                UpdateStatusfile(self.directories, 14)
                log.info("")
                log.info("===> Planet no longer cooling! <===")
                log.info("")
                self.finished = True

            # Determine when the simulation enters a steady state
            if (
                self.config.params.stop.steady.enabled
                and (self.loops["total"] > self.loops["steady_check"] * 2 + 5)
                and (self.loops["steady"] < 0)
            ):
                # How many iterations to look backwards
                lb1 = -int(self.loops["steady_check"])
                lb2 = -1

                # Get data
                arr_t = np.array(self.hf_all["Time"])
                arr_f = np.array(self.hf_all["F_atm"])
                arr_p = np.array(self.hf_all["Phi_global"])

                # Time samples
                t1 = arr_t[lb1]
                t2 = arr_t[lb2]

                # Flux samples
                flx_m = max(abs(arr_f[lb1]), abs(arr_f[lb2]))

                # Check melt fraction rate
                phi_1 = arr_p[lb1]
                phi_2 = arr_p[lb2]
                phi_r = abs(phi_2 - phi_1) / (t2 - t1)

                # Stop when flux is small and melt fraction is unchanging
                if (flx_m < self.config.params.stop.steady.F_crit) \
                    and (phi_r < self.config.params.stop.steady.dprel):
                    log.debug("Steady state declared")
                    self.loops["steady"] = 0

            # Steady-state handling
            if self.loops["steady"] >= 0:
                # Force the model to do N=steady_loops more iterations before stopping
                # This is to make absolutely sure that it has reached a steady state.
                self.loops["steady"] += 1

                # Stop now?
                if self.loops["steady"] >= self.loops["steady_loops"]:
                    UpdateStatusfile(self.directories, 11)
                    log.info("")
                    log.info("===> Planet entered a steady state! <===")
                    log.info("")
                    self.finished = True

            # Atmosphere has escaped
            if self.hf_row["M_atm"] <= self.config.params.stop.escape.mass_frac * self.hf_all.iloc[0]["M_atm"]:
                UpdateStatusfile(self.directories, 15)
                log.info("")
                log.info("===> Atmosphere has escaped! <===")
                log.info("")
                self.finished = True

            # Maximum time reached
            if self.hf_row["Time"] >= self.config.params.stop.time.maximum:
                UpdateStatusfile(self.directories, 13)
                log.info("")
                log.info("===> Target time reached! <===")
                log.info("")
                self.finished = True

            # Maximum loops reached
            if self.loops["total"] > self.loops["total_loops"]:
                UpdateStatusfile(self.directories, 12)
                log.info("")
                log.info("===> Maximum number of iterations reached! <===")
                log.info("")
                self.finished = True

            # Check if the minimum number of loops have been performed
            if self.finished and (self.loops["total"] < self.loops["total_min"]):
                log.info("Minimum number of iterations not yet attained; continuing...")
                self.finished = False
                UpdateStatusfile(self.directories, 1)

            # Check if keepalive file has been removed - this means that the model should exit ASAP
            if not os.path.exists(self.lockfile):
                UpdateStatusfile(self.directories, 25)
                log.info("")
                log.info("===> Model exit was requested! <===")
                log.info("")
                self.finished = True

            # Make plots if required and go to next iteration
            if (
                (self.config.params.out.plot_mod > 0)
                and (self.loops["total"] % self.config.params.out.plot_mod == 0)
                and (not self.finished)
            ):
                PrintHalfSeparator()
                log.info("Making plots")
                UpdatePlots(self.hf_all, self.directories, self.config)

            ############### / HOUSEKEEPING AND CONVERGENCE CHECK

        # FINAL THINGS BEFORE EXIT
        PrintHalfSeparator()

        # Clean up files
        safe_rm(self.lockfile)

        # Plot conditions at the end
        UpdatePlots(self.hf_all, self.directories, self.config, end=True)
        end_time = datetime.now()
        log.info("Simulation stopped at: " + end_time.strftime("%Y-%m-%d_%H:%M:%S"))

        run_time = end_time - start_time
        run_time = run_time.total_seconds() / 60  # minutes
        if run_time > 60:
            log.info("Total runtime: %.2f hours" % (run_time / 60))
        else:
            log.info("Total runtime: %.2f minutes" % run_time)
