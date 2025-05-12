from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np

import proteus.utils.archive as archive
from proteus.config import read_config_object
from proteus.utils.constants import (
    vap_list,
    vol_list,
)
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
os.environ["OMP_NUM_THREADS"] = "4"

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
        self.init_stage = False
        self.loops = None

        # Interior
        self.interior_o = None      # Interior object from interior/common.py

        # Atmosphere
        self.atmos_o = None     # Atmosphere object from atmos_clim/common.py

        # Model has finished?
        self.finished_prev = False          # Satisfied termination in prev iteration
        self.finished_both = False          # Satisfied termination in current and previous
        self.desiccated = False             # Entire volatile inventory has been lost
        self.lockfile = "/tmp/none"         # Path to keepalive file

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
        from proteus.atmos_clim.common import Atmos_t

        #    escape and outgas
        from proteus.escape.wrapper import run_escape

        #    interior
        from proteus.interior.common import Interior_t
        from proteus.interior.wrapper import get_nlevb, run_interior, solve_structure

        #    synthetic observations
        from proteus.observe.wrapper import run_observe

        #    orbit
        from proteus.orbit.wrapper import init_orbit, run_orbit

        #    outgassing
        from proteus.outgas.wrapper import (
            calc_target_elemental_inventories,
            check_desiccation,
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
            CleanDir(self.directories["output"])
            CleanDir(self.directories["output/data"])
            CleanDir(self.directories["output/observe"])
            CleanDir(self.directories["output/offchem"])
            CleanDir(self.directories["output/plots"])

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
            "init_loops": 3,  # Maximum number of init iters
        }
        self.init_stage = True

        # Write config to output directory, for future reference
        self.config.write(os.path.join(self.directories["output"], "init_coupler.toml"))

        # Create lockfile for keeping simulation running
        self.lockfile = CreateLockFile(self.directories["output"])

        # Download basic data
        download_sufficient_data(self.config)

        # Initialise interior and atmosphere objects
        self.interior_o = Interior_t(get_nlevb(self.config))
        self.atmos_o    = Atmos_t()

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
                if s != "O2":
                    pp_val = self.config.delivery.volatiles.get_pressure(s)
                    include = self.config.outgas.calliope.is_included(s)
                else:
                    pp_val = 0.0
                    include = True

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
            log.info("Included gases:")
            for s in inc_gases:
                write = "    "
                write += "vapour  " if s in vap_list else "volatile"
                write += "  %-8s" % s
                if self.config.delivery.initial == "volatiles":
                    write += " : %6.2f bar"%self.hf_row[s + "_bar"]
                log.info(write)

        else:
            # Resuming from disk
            log.info("Resuming the simulation from the disk")

            # Read helpfile from disk
            self.hf_all = ReadHelpfileFromCSV(self.directories["output"])

            # Check length
            if len(self.hf_all) <= self.loops["init_loops"] + 1:
                UpdateStatusfile(self.directories, 20)
                raise RuntimeError("Simulation is too short to be resumed")

            # Get last row from helpfile dataframe
            self.hf_row = self.hf_all.iloc[-1].to_dict()

            # Check if the planet is desiccated
            self.desiccated = check_desiccation(self.config, self.hf_row)

            # Extract all archived data files
            log.debug("Extracting archived data files")
            self.extract_archives()

            # Interior initial condition
            self.interior_o.ic = 2

            # Restore tides data
            if self.config.orbit.module is not None:
                self.interior_o.resume_tides(self.directories["output"])

            # Set loop counters
            self.loops["total"] = len(self.hf_all)
            self.init_stage = False

        log.info(" ")

        # Prepare star stuff
        init_star(self)

        # Prepare orbit stuff
        init_orbit(self)

        # Main loop
        UpdateStatusfile(self.directories, 1)
        while not self.finished_both:
            # New rows
            if self.loops["total"] > 0:
                # Create new row to hold the updated variables. This will be
                #    overwritten by the routines below.
                self.hf_row = self.hf_all.iloc[-1].to_dict()

            log.info(" ")
            PrintSeparator()
            log.info("Loop counters")
            log.info("current    init    maximum")
            log.info(
                " %6d    %4d     %6d "
                % (
                    self.loops["total"],
                    self.loops["init_loops"],
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
                log.info("Updated spectrum not required")

            ############### / STELLAR FLUX MANAGEMENT

            ############### ESCAPE
            if (self.loops["total"] > self.loops["init_loops"]+2) and (not self.desiccated):
                PrintHalfSeparator()
                run_escape(self.config, self.hf_row, self.interior_o.dt, self.stellar_track)

            ############### / ESCAPE

            ############### OUTGASSING
            PrintHalfSeparator()

            # Recalculate mass targets during init phase, since these will be adjusted
            #    depending on the true melt fraction and T_magma found by SPIDER at runtime.
            if self.init_stage:
                calc_target_elemental_inventories(self.directories, self.config, self.hf_row)

            else:
                # Check if desiccation has occurred
                self.desiccated = check_desiccation(self.config, self.hf_row)

            # handle desiccated planet
            if self.desiccated:
                run_desiccated(self.config, self.hf_row)

            # solve for atmosphere composition
            else:
                run_outgassing(self.directories, self.config, self.hf_row)

            # Add atmosphere mass to interior mass, to get total planet mass
            self.hf_row["M_planet"] = self.hf_row["M_int"] + self.hf_row["M_atm"]

            ############### / OUTGASSING

            ############### ATMOSPHERE CLIMATE
            PrintHalfSeparator()
            run_atmosphere(self.atmos_o, self.config, self.directories, self.loops,
                                self.star_wl, self.star_fl, update_stellar_spectrum,
                                self.hf_all, self.hf_row)

            ############### / ATMOSPHERE CLIMATE

            ############### HOUSEKEEPING AND CONVERGENCE CHECK

            PrintHalfSeparator()

            # Update model wall-clock runtime
            run_time = datetime.now() - start_time
            self.hf_row["runtime"] = float(run_time.total_seconds())

            # Adjust total iteration counters
            self.loops["total"] += 1

            # Init stage?
            if self.loops["total"] > self.loops["init_loops"]:
                self.init_stage = False

            # Keep time at zero during init stage
            if self.init_stage:
                self.hf_row["Time"] = 0.0
            else:
                self.interior_o.ic = 2

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
            if not self.init_stage:
                log.info("Checking convergence criteria")
                check_termination(self)

            # Make plots
            if multiple(self.loops["total"], self.config.params.out.plot_mod) \
                and not self.finished_both:

                log.info("Making plots")
                UpdatePlots(self.hf_all, self.directories, self.config)

            # Update or create data archive
            if multiple(self.loops["total"], self.config.params.out.archive_mod) \
                and not self.finished_both:

                log.info("Updating archive of model output data")
                # do not remove ALL files
                archive.update(self.directories["output/data"], remove_files=False)
                # remove all files EXCEPT the latest ones
                archive.remove_old(self.directories["output/data"],self.hf_row["Time"]*0.99)

            ############### / HOUSEKEEPING AND CONVERGENCE CHECK

        # Write conditions at the end of simulation
        log.info("Writing data")
        WriteHelpfileToCSV(self.directories["output"], self.hf_all)

        # Run offline chemistry
        if self.config.atmos_chem.when == "offline":
            log.info(" ")
            PrintSeparator()
            if self.desiccated:
                log.warning("Cannot calculate atmospheric chemistry after desiccation")
            else:
                run_chemistry(self.directories, self.config, self.hf_row)

        # Synthetic observations
        if self.config.observe.synthesis is not None:
            log.info(" ")
            PrintSeparator()
            if self.desiccated:
                log.warning("Cannot observe planet after desiccation")
            else:
                run_observe(self.hf_row, self.directories["output"], self.config)

        # Make final plots
        if self.config.params.out.plot_mod is not None:
            log.info("Making final plots")
            UpdatePlots(self.hf_all, self.directories, self.config, end=True)

        # Tidy up
        log.info(" ")
        log.debug("Tidy up before exit")
        remove_excess_files(self.directories["output"],
                            rm_spectralfiles=self.config.params.out.remove_sf)

        # Archive the folder ./output/data/, and remove files
        if self.config.params.out.archive_mod is not None:
            log.info("Archiving output data into tar files")
            archive.update(self.directories["output/data"], remove_files=True)

        # Stop time and model duration
        print_stoptime(start_time)

        # Print citation
        print_citation(self.config)

    def extract_archives(self):
        """
        Extract archived data files in subfolders of the output directory
        """
        archive.extract(self.directories["output/data"], remove_tar=True)

    def create_archives(self):
        """
        Pack data files in subfolders of the output directory into archival tar files
        """
        archive.create(self.directories["output/data"], remove_files=True)

    def observe(self):
        # Extract archived data
        self.extract_archives()

        # Load data from helpfile
        from proteus.utils.coupler import ReadHelpfileFromCSV
        hf_all = ReadHelpfileFromCSV(self.directories["output"])

        # Check length
        if len(hf_all) < 1:
            raise Exception("Simulation is too short to be postprocessed")

        # Get last row
        hf_row = hf_all.iloc[-1].to_dict()

        # Run observations pipeline, typically invoked via CLI
        from proteus.observe.wrapper import run_observe
        run_observe(hf_row, self.directories["output"], self.config)


    def offline_chemistry(self):
        # Extract archived data
        self.extract_archives()

        # Load data from helpfile
        from proteus.utils.coupler import ReadHelpfileFromCSV
        hf_all = ReadHelpfileFromCSV(self.directories["output"])

        # Check length
        if len(hf_all) < 1:
            raise Exception("Simulation is too short to be postprocessed")

        # Get last row
        hf_row = hf_all.iloc[-1].to_dict()

        # Run offline chemistry, typically invoked via CLI
        from proteus.atmos_chem.wrapper import run_chemistry
        result = run_chemistry(self.directories, self.config, hf_row)

        # return the dataframe
        return result
