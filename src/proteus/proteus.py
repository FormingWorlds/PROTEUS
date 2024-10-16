from __future__ import annotations

import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import toml
from attrs import asdict
from calliope.structure import calculate_mantle_mass

import proteus.utils.constants
from proteus.atmos_clim import RunAtmosphere
from proteus.config import read_config_object
from proteus.escape.wrapper import RunEscape
from proteus.interior import run_interior
from proteus.outgas.wrapper import calc_target_elemental_inventories, run_outgassing
from proteus.utils.constants import (
    AU,
    L_sun,
    M_earth,
    R_earth,
    const_G,
    element_list,
    volatile_species,
)
from proteus.utils.coupler import (
    CalculateEqmTemperature,
    CreateHelpfileFromDict,
    CreateLockFile,
    ExtendHelpfile,
    GitRevision,
    PrintCurrentState,
    ReadHelpfileFromCSV,
    SetDirectories,
    UpdatePlots,
    WriteHelpfileToCSV,
    ZeroHelpfileRow,
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


class Proteus:
    def __init__(self, *, config_path: Path | str) -> None:
        self.config_path = config_path
        self.config = read_config_object(config_path)

        self.init_directories()


    def init_directories(self):
        """Initialize directories dictionary"""
        self.directories = SetDirectories(self.config)

        # Keep `constants.dirs` around for compatibility with other parts of the code
        # At some point, they should all reference `Proteus.directories`
        proteus.utils.constants.dirs = self.directories

    def start(self, *, resume: bool = False):
        """Start PROTEUS simulation.

        Parameters
        ----------
        resume : bool
            If True, continue from previous simulation
        """
        import mors

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

        # Print information to logger
        log.info(":::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        log.info("            PROTEUS framework (version 0.1)            ")
        log.info(":::::::::::::::::::::::::::::::::::::::::::::::::::::::")
        log.info(" ")
        start_time = datetime.now()
        log.info("Current time: " + start_time.strftime("%Y-%m-%d_%H:%M:%S"))
        log.info("Hostname    : " + str(os.uname()[1]))
        log.info("PROTEUS hash: " + GitRevision(self.directories["proteus"]))
        log.info("Py version  : " + sys.version.split(" ")[0])
        log.info("Config file : " + str(self.config_path))
        log.info("Output dir  : " + self.directories["output"])
        log.info("FWL data dir: " + self.directories["fwl"])
        if self.config.atmos_clim.module in ['janus', 'agni']:
            log.info("SOCRATES dir: " + self.directories["rad"])
        log.info(" ")

        # Count iterations
        loop_counter = {
            "total": 0,  # Total number of iters performed
            "total_min": 5,  # Minimum number of total loops
            "total_loops": self.config.params.stop.iters.maximum,  # Maximum number of total loops
            "init": 0,  # Number of init iters performed
            "init_loops": 2,  # Maximum number of init iters
            "steady": -1,  # Number of iterations passed since steady-state declared
            "steady_loops": 3,  # Number of iterations to perform post-steady state
            "steady_check": 15,  # Number of iterations to look backwards when checking steady state
        }

        # Model has completed?
        finished = False

        # Config file paths
        config_path_backup = os.path.join(self.directories["output"], "init_coupler.toml")

        # Is the model resuming from a previous state?
        if not resume:
            # New simulation

            # SPIDER initial condition
            IC_INTERIOR = 1

            # Write aggregate config to output directory, for future reference
            with open(config_path_backup, "w") as toml_file:
                toml.dump(asdict(self.config), toml_file)

            # No previous iterations to be stored. It is therefore important that the
            #    submodules do not try to read data from the 'past' iterations, since they do
            #    not yet exist.
            hf_all = None

            # Create an empty initial row for helpfile
            hf_row = ZeroHelpfileRow()

            # Initial time
            hf_row["Time"] = 0.0
            hf_row["age_star"] = self.config.star.age_ini * 1e9

            # Initial guess for flux
            hf_row["F_atm"] = self.config.interior.F_initial
            hf_row["F_int"] = hf_row["F_atm"]
            hf_row["T_eqm"] = 2000.0

            # Planet size conversion, and calculate mantle mass (= liquid + solid)
            hf_row["M_planet"] = self.config.struct.mass * M_earth
            hf_row["R_planet"] = self.config.struct.radius * R_earth
            hf_row["gravity"] = const_G * hf_row["M_planet"] / (hf_row["R_planet"] ** 2.0)
            hf_row["M_mantle"] = calculate_mantle_mass(
                hf_row["R_planet"], hf_row["M_planet"], self.config.struct.corefrac
            )

            # Store partial pressures and list of included volatiles
            log.info("Input partial pressures:")
            inc_vols = []
            for s in volatile_species:
                pp_val = getattr(self.config.delivery.volatiles, s)
                include = getattr(self.config.outgas.calliope, f'include_{s}')

                log.info(
                    "    %-6s : %-5.2f bar (included = %s)"
                    % (s, pp_val, include)
                )

                if include:
                    inc_vols.append(s)
                    hf_row[s + "_bar"] = max(1.0e-30, float(pp_val))
                else:
                    hf_row[s + "_bar"] = 0.0
            log.info("Included volatiles: " + str(inc_vols))

        else:
            # Resuming from disk
            log.info("Resuming the simulation from the disk")

            # Copy cfg file
            with open(config_path_backup, "w") as toml_file:
                toml.dump(self.config, toml_file)

            # SPIDER initial condition
            IC_INTERIOR = 2

            # Read helpfile from disk
            hf_all = ReadHelpfileFromCSV(self.directories["output"])

            # Check length
            if len(hf_all) <= loop_counter["init_loops"] + 1:
                raise Exception("Simulation is too short to be resumed")

            # Get last row
            hf_row = hf_all.iloc[-1].to_dict()

            # Set loop counters
            loop_counter["total"] = len(hf_all)
            loop_counter["init"] = loop_counter["init_loops"] + 1

            # Set instellation
            S_0 = hf_row["F_ins"]
            F_inst_prev = S_0

            # Set volatile mass targets f
            solvevol_target = {}
            for e in element_list:
                if e == "O":
                    continue
                solvevol_target[e] = hf_row[e + "_kg_total"]

        # Download basic data
        download_sufficient_data(self.config)

        # Handle stellar spectrum...

        # Store copy of modern spectrum in memory (1 AU)
        sspec_prev = -np.inf
        sinst_prev = -np.inf
        star_modern_path = os.path.join(self.directories["fwl"], self.config.star.mors.spec)
        shutil.copyfile(star_modern_path, os.path.join(self.directories["output"], "-1.sflux"))

        # Prepare stellar models
        match self.config.star.mors.tracks:
            case 'spada':
                # load modern spectrum
                star_struct_modern = mors.spec.Spectrum()
                star_struct_modern.LoadTSV(star_modern_path)
                star_struct_modern.CalcBandFluxes()

                # modern properties
                star_props_modern = mors.synthesis.GetProperties(
                    self.config.star.mass,
                    self.config.star.omega,
                    self.config.star.age_now * 1000,
                )

            case 'baraffe':
                modern_wl, modern_fl = mors.ModernSpectrumLoad(
                    star_modern_path, self.directories["output"] + "/-1.sflux"
                )

                baraffe = mors.BaraffeTrack(self.config.star.mass)

        # Create lockfile
        keepalive_file = CreateLockFile(self.directories["output"])

        # Main loop
        UpdateStatusfile(self.directories, 1)
        while not finished:
            # New rows
            if loop_counter["total"] > 0:
                # Create new row to hold the updated variables. This will be
                #    overwritten by the routines below.
                hf_row = hf_all.iloc[-1].to_dict()

            log.info(" ")
            PrintSeparator()
            log.info("Loop counters")
            log.info("init    total     steady")
            log.info(
                "%1d/%1d   %04d/%04d   %1d/%1d"
                % (
                    loop_counter["init"],
                    loop_counter["init_loops"],
                    loop_counter["total"],
                    loop_counter["total_loops"],
                    loop_counter["steady"],
                    loop_counter["steady_loops"],
                )
            )

            ############### ORBIT AND TIDES

            # Calculate time-averaged orbital separation (and convert from AU to metres)
            # https://physics.stackexchange.com/a/715749
            hf_row["separation"] = self.config.orbit.semimajoraxis * AU * \
                                        (1 + 0.5 * self.config.orbit.eccentricity**2.0)


            ############### / ORBIT AND TIDES

            ############### INTERIOR

            # Run interior model
            dt = run_interior(self.directories, self.config,
                                loop_counter, IC_INTERIOR, hf_all,  hf_row)

            # Advance current time in main loop according to interior step
            hf_row["Time"] += dt        # in years
            hf_row["age_star"] += dt    # in years

            ############### / INTERIOR

            ############### STELLAR FLUX MANAGEMENT
            PrintHalfSeparator()
            log.info("Stellar flux management...")
            update_stellar_spectrum = False

            # Calculate new instellation and radius
            if (abs(hf_row["Time"] - sinst_prev) > self.config.params.dt.starinst) or (
                loop_counter["total"] == 0
            ):
                sinst_prev = hf_row["Time"]

                # Get previous instellation
                if loop_counter["total"] > 0:
                    F_inst_prev = S_0
                else:
                    F_inst_prev = 0.0

                log.info("Updating instellation and radius")

                match self.config.star.mors.tracks:
                    case 'spada':
                        hf_row["R_star"] = (
                            mors.Value(
                                self.config.star.mass,
                                hf_row["age_star"] / 1e6,
                                "Rstar"
                            )
                            * mors.const.Rsun
                            * 1.0e-2
                        )
                        S_0 = (
                            mors.Value(
                                self.config.star.mass,
                                hf_row["age_star"] / 1e6,
                                "Lbol"
                            ) * L_sun  / (4.0 * np.pi * hf_row["separation"]**2.0 )
                        )

                    case 'baraffe':
                        hf_row["R_star"] = (
                            baraffe.BaraffeStellarRadius(hf_row["age_star"])
                            * mors.const.Rsun
                            * 1.0e-2
                        )
                        S_0 = baraffe.BaraffeSolarConstant(
                            hf_row["age_star"], hf_row["separation"]/AU
                        )

                # Calculate new eqm temperature
                T_eqm_new = CalculateEqmTemperature(
                    S_0, self.config.orbit.s0_factor, self.config.atmos_clim.albedo_pl
                )

                hf_row["F_ins"] = S_0  # instellation
                hf_row["T_eqm"] = T_eqm_new
                hf_row["T_skin"] = (
                    T_eqm_new * (0.5**0.25)
                )  # Assuming a grey stratosphere in radiative eqm (https://doi.org/10.5194/esd-7-697-2016)

                log.debug("Instellation change: %+.4e W m-2 (to 4dp)" % abs(S_0 - F_inst_prev))

            # Calculate a new (historical) stellar spectrum
            if (abs(hf_row["Time"] - sspec_prev) > self.config.params.dt.starspec) or (
                loop_counter["total"] == 0
            ):
                sspec_prev = hf_row["Time"]
                update_stellar_spectrum = True

                log.info("Updating stellar spectrum")
                match self.config.star.mors.tracks:
                    case 'spada':
                        synthetic = mors.synthesis.CalcScaledSpectrumFromProps(
                            star_struct_modern, star_props_modern, hf_row["age_star"] / 1e6
                        )
                        fl = synthetic.fl  # at 1 AU
                        wl = synthetic.wl
                    case 'baraffe':
                        fl = baraffe.BaraffeSpectrumCalc(
                            hf_row["age_star"], self.config["star_luminosity_modern"], modern_fl
                        )
                        wl = modern_wl

                # Scale fluxes from 1 AU to TOA
                fl *= (AU / hf_row["separation"]) ** 2.0

                # Save spectrum to file
                header = (
                    "# WL(nm)\t Flux(ergs/cm**2/s/nm)   Stellar flux at t_star = %.2e yr"
                    % hf_row["age_star"]
                )
                np.savetxt(
                    os.path.join(self.directories["output"], "data", "%d.sflux" % hf_row["Time"]),
                    np.array([wl, fl]).T,
                    header=header,
                    comments="",
                    fmt="%.8e",
                    delimiter="\t",
                )

            else:
                log.info("New spectrum not required at this time")

            ############### / STELLAR FLUX MANAGEMENT

            ############### ESCAPE
            if (loop_counter["total"] >= loop_counter["init_loops"]):
                RunEscape(self.config, hf_row, dt)
            ############### / ESCAPE

            ############### OUTGASSING
            PrintHalfSeparator()

            #    recalculate mass targets during init phase, since these will be adjusted
            #    depending on the true melt fraction and T_magma found by SPIDER at runtime.
            if loop_counter["init"] < loop_counter["init_loops"]:
                calc_target_elemental_inventories(self.directories, self.config, hf_row)

            # solve for atmosphere composition
            run_outgassing(self.directories, self.config, hf_row)

            ############### / OUTGASSING

            ############### ATMOSPHERE SUB-LOOP
            RunAtmosphere(self.config, self.directories, loop_counter, wl, fl, update_stellar_spectrum, hf_all, hf_row)

            ############### HOUSEKEEPING AND CONVERGENCE CHECK

            PrintHalfSeparator()

            # Update init loop counter
            # Next init iter
            if loop_counter["init"] < loop_counter["init_loops"]:
                loop_counter["init"] += 1
                hf_row["Time"] = 0.0
            # Reset restart flag once SPIDER has correct heat flux
            if loop_counter["total"] >= loop_counter["init_loops"]:
                IC_INTERIOR = 2

            # Adjust total iteration counters
            loop_counter["total"] += 1

            # Update full helpfile
            if loop_counter["total"] > 1:
                # append row
                hf_all = ExtendHelpfile(hf_all, hf_row)
            else:
                # first iter => generate new HF from dict
                hf_all = CreateHelpfileFromDict(hf_row)

            # Write helpfile to disk
            WriteHelpfileToCSV(self.directories["output"], hf_all)

            # Print info to terminal and log file
            PrintCurrentState(hf_row)

            log.info("Check convergence criteria")

            # Stop simulation when planet is completely solidified
            if self.config.params.stop.solid.enabled and (hf_row["Phi_global"] <= self.config.params.stop.solid.phi_crit):
                UpdateStatusfile(self.directories, 10)
                log.info("")
                log.info("===> Planet solidified! <===")
                log.info("")
                finished = True

            # Stop simulation when flux is small
            if (
                self.config.params.stop.radeqm.enabled
                and (loop_counter["total"] > loop_counter["init_loops"] + 1)
                and (abs(hf_row["F_atm"]) <= self.config.params.stop.radeqm.F_crit)
            ):
                UpdateStatusfile(self.directories, 14)
                log.info("")
                log.info("===> Planet no longer cooling! <===")
                log.info("")
                finished = True

            # Determine when the simulation enters a steady state
            if (
                self.config.params.stop.steady.enabled
                and (loop_counter["total"] > loop_counter["steady_check"] * 2 + 5)
                and (loop_counter["steady"] < 0)
            ):
                # How many iterations to look backwards
                lb1 = -int(loop_counter["steady_check"])
                lb2 = -1

                # Get data
                arr_t = np.array(hf_all["Time"])
                arr_f = np.array(hf_all["F_atm"])
                arr_p = np.array(hf_all["Phi_global"])

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
                if (flx_m < self.config["steady_flux"]) and (phi_r < self.config["steady_dprel"]):
                    log.debug("Steady state declared")
                    loop_counter["steady"] = 0

            # Steady-state handling
            if loop_counter["steady"] >= 0:
                # Force the model to do N=steady_loops more iterations before stopping
                # This is to make absolutely sure that it has reached a steady state.
                loop_counter["steady"] += 1

                # Stop now?
                if loop_counter["steady"] >= loop_counter["steady_loops"]:
                    UpdateStatusfile(self.directories, 11)
                    log.info("")
                    log.info("===> Planet entered a steady state! <===")
                    log.info("")
                    finished = True

            # Atmosphere has escaped
            if hf_row["M_atm"] <= self.config.params.stop.escape.mass_frac * hf_all.iloc[0]["M_atm"]:
                UpdateStatusfile(self.directories, 15)
                log.info("")
                log.info("===> Atmosphere has escaped! <===")
                log.info("")
                finished = True

            # Maximum time reached
            if hf_row["Time"] >= self.config.params.stop.time.maximum * 1e9:
                UpdateStatusfile(self.directories, 13)
                log.info("")
                log.info("===> Target time reached! <===")
                log.info("")
                finished = True

            # Maximum loops reached
            if loop_counter["total"] > loop_counter["total_loops"]:
                UpdateStatusfile(self.directories, 12)
                log.info("")
                log.info("===> Maximum number of iterations reached! <===")
                log.info("")
                finished = True

            # Check if the minimum number of loops have been performed
            if finished and (loop_counter["total"] < loop_counter["total_min"]):
                log.info("Minimum number of iterations not yet attained; continuing...")
                finished = False
                UpdateStatusfile(self.directories, 1)

            # Check if keepalive file has been removed - this means that the model should exit ASAP
            if not os.path.exists(keepalive_file):
                UpdateStatusfile(self.directories, 25)
                log.info("")
                log.info("===> Model exit was requested! <===")
                log.info("")
                finished = True

            # Make plots if required and go to next iteration
            if (
                (self.config.params.out.plot_mod > 0)
                and (loop_counter["total"] % self.config.params.out.plot_mod == 0)
                and (not finished)
            ):
                PrintHalfSeparator()
                log.info("Making plots")
                UpdatePlots(hf_all, self.directories["output"], self.config)

            ############### / HOUSEKEEPING AND CONVERGENCE CHECK

        # FINAL THINGS BEFORE EXIT
        PrintHalfSeparator()

        # Clean up files
        safe_rm(keepalive_file)

        # Plot conditions at the end
        UpdatePlots(hf_all, self.directories["output"], self.config, end=True)
        end_time = datetime.now()
        log.info("Simulation stopped at: " + end_time.strftime("%Y-%m-%d_%H:%M:%S"))

        run_time = end_time - start_time
        run_time = run_time.total_seconds() / 60  # minutes
        if run_time > 60:
            log.info("Total runtime: %.2f hours" % (run_time / 60))
        else:
            log.info("Total runtime: %.2f minutes" % run_time)
