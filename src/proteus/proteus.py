from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np

from proteus.config import read_config_object
from proteus.star.wrapper import (
    get_new_spectrum,
    init_star,
    scale_spectrum_to_toa,
    update_stellar_mass,
    update_stellar_quantities,
    write_spectrum,
)
from proteus.utils.constants import (
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
    print_citation,
    print_header,
    print_module_configuration,
    print_stoptime,
    print_system_configuration,
)
from proteus.utils.helper import (
    CleanDir,
    PrintHalfSeparator,
    PrintSeparator,
    UpdateStatusfile,
    multiple,
    safe_rm,
)
from proteus.utils.logs import (
    GetCurrentLogfileIndex,
    GetLogfilePath,
    setup_logger,
)
from proteus.utils.terminate import check_termination, print_termination_criteria


class Proteus:
    def __init__(self, *, config_path: Path | str) -> None:

        # Read and parse configuration file
        self.config_path = config_path
        self.config = read_config_object(config_path)

        # Setup directories dictionary
        self.directories:dict = None # Directories dictionary
        self.init_directories()

        # Helpfile variables for the current iteration
        self.hf_row = None

        # Helpfile variables from all previous iterations
        self.hf_all = None

        # Loop counters
        self.loops = None

        # Interior
        self.interior_o = None      # Interior object from common.py

        # Model has finished?
        self.finished1 = False          # Satisfied finishing criteria once
        self.finished2 = False          # Satisfied finishing criteria twice
        self.has_escaped = False        # Atmosphere has escaped
        self.lockfile = "/tmp/none"     # Path to keepalive file

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

    def start(self, *, resume: bool = False, offline: bool = False):
        """Start PROTEUS simulation.

        Parameters
        ----------
        resume : bool
            If True, continue from previous simulation.
        offline : bool
            Run in offline mode; do not try to connect to the internet.
        """

        # Import
        from proteus.atmos_clim import RunAtmosphere
        from proteus.escape.wrapper import RunEscape
        from proteus.interior.common import Interior_t
        from proteus.interior.wrapper import run_interior, solve_structure
        from proteus.orbit.wrapper import init_orbit, run_orbit
        from proteus.outgas.wrapper import calc_target_elemental_inventories, run_outgassing
        from proteus.utils.data import download_sufficient_data

        # First things
        start_time = datetime.now()
        self.config.params.resume = resume
        self.config.params.offline = offline
        UpdateStatusfile(self.directories, 0)

        # Clean output directory
        if not self.config.params.resume:
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

        # Print termination criteria
        print_termination_criteria(self.config)

        PrintHalfSeparator()

        # Count iterations
        self.loops = {
            "total": 0,  # Total number of iters performed
            "total_min": self.config.params.stop.iters.minimum,
            "total_loops": self.config.params.stop.iters.maximum,
            "init": 0,  # Number of init iters performed
            "init_loops": 3,  # Maximum number of init iters
        }

        # Write config to output directory, for future reference
        self.config.write(os.path.join(self.directories["output"], "init_coupler.toml"))

        # Create lockfile for keeping simulation running
        self.lockfile = CreateLockFile(self.directories["output"])

        # Download basic data
        download_sufficient_data(self.config)

        # Initialise interior struct object.
        self.interior_o = Interior_t(self.directories["output"], self.config)

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
            self.hf_row["Time"] = 0.0
            self.hf_row["age_star"] = self.config.star.age_ini * 1e9

            # Initial guess for flux
            self.hf_row["F_atm"] = self.config.interior.F_initial
            self.hf_row["F_int"] = self.hf_row["F_atm"]
            self.hf_row["T_eqm"] = 2000.0

            # Solve interior structure
            solve_structure(self.directories, self.config, self.hf_all, self.hf_row)

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

            # Interior initial condition
            self.interior_o.ic = 2

            # Restore tides data
            if self.config.orbit.module is not None:
                self.interior_o.resume_tides()

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

        # Prepare star stuff
        init_star(self)

        # Prepare orbit stuff
        init_orbit(self)

        # Main loop
        UpdateStatusfile(self.directories, 1)
        while not self.finished2:
            # New rows
            if self.loops["total"] > 0:
                # Create new row to hold the updated variables. This will be
                #    overwritten by the routines below.
                self.hf_row = self.hf_all.iloc[-1].to_dict()

            log.info(" ")
            PrintSeparator()
            log.info("Loop counters")
            log.info("init      total")
            log.info(
                "%1d/%1d     %04d/%04d "
                % (
                    self.loops["init"],
                    self.loops["init_loops"],
                    self.loops["total"],
                    self.loops["total_loops"],
                )
            )

            ############### INTERIOR
            PrintHalfSeparator()
            run_interior(self.directories, self.config,
                            self.hf_all, self.hf_row, self.interior_o)


            # Advance current time in main loop according to interior step
            self.hf_row["Time"]     += self.interior_o.dt    # in years
            self.hf_row["age_star"] += self.interior_o.dt    # in years

            ############### / INTERIOR AND STRUCTURE

            ############### ORBIT AND TIDES
            PrintHalfSeparator()
            run_orbit(self.hf_row, self.config, self.directories, self.interior_o)

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
                PrintHalfSeparator()
                RunEscape(self.config, self.hf_row, self.interior_o.dt, self.stellar_track)

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
                self.has_escaped = True
                for gas in gas_list:
                    self.hf_row[gas+"_vmr"] = 0.0
                self.hf_row["atm_kg_per_mol"] = 0.0

            ############### / OUTGASSING

            ############### ATMOSPHERE CLIMATE
            if not self.has_escaped:
                PrintHalfSeparator()
                RunAtmosphere(self.config, self.directories, self.loops,
                                self.star_wl, self.star_fl, update_stellar_spectrum,
                                self.hf_all, self.hf_row)

            ############### / ATMOSPHERE CLIMATE

            ############### HOUSEKEEPING AND CONVERGENCE CHECK

            PrintHalfSeparator()

            # Update init loop counter
            # Next init iter
            if self.loops["init"] < self.loops["init_loops"]:
                self.loops["init"] += 1
                self.hf_row["Time"] = 0.0
            # Reset restart flag once SPIDER has correct heat flux
            if self.loops["total"] >= self.loops["init_loops"]:
                self.interior_o.ic = 2

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
            if multiple(self.loops["total"], self.config.params.out.write_mod):
                    WriteHelpfileToCSV(self.directories["output"], self.hf_all)

            # Print info to terminal and log file
            PrintCurrentState(self.hf_row)

            # Check for convergence
            if self.loops["total"] >= self.loops["init_loops"]:
                log.info("Checking convergence criteria")
                check_termination(self)

            # Make plots
            if multiple(self.loops["total"], self.config.params.out.plot_mod) \
                and not self.finished2:

                log.info("Making plots")
                UpdatePlots(self.hf_all, self.directories, self.config)

            ############### / HOUSEKEEPING AND CONVERGENCE CHECK

        # FINAL THINGS BEFORE EXIT
        PrintSeparator()

        # Clean up files
        safe_rm(self.lockfile)

        # Plot and write conditions at the end of simulation
        WriteHelpfileToCSV(self.directories["output"], self.hf_all)
        UpdatePlots(self.hf_all, self.directories, self.config, end=True)

        # Stop time and model duration
        print_stoptime(start_time)

        # Print citation
        print_citation(self.config)
