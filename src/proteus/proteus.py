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
from calliope.solve import (
    equilibrium_atmosphere,
    get_target_from_params,
    get_target_from_pressures,
)
from calliope.structure import calculate_mantle_mass

import proteus.utils.constants
from proteus.atmos_clim import RunAtmosphere
from proteus.config import read_config_object
from proteus.interior import run_interior
from proteus.utils.constants import (
    AU,
    L_sun,
    M_earth,
    R_earth,
    const_G,
    element_list,
    secs_per_year,
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
    ValidateInitFile,
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

        # Validate options
        ValidateInitFile(self.directories, self.config)

        # Clean output directory
        if not resume:
            CleanDir(self.directories["output"])
            CleanDir(os.path.join(self.directories["output"], "data"))

        # Get next logfile path
        logindex = 1 + GetCurrentLogfileIndex(self.directories["output"])
        logpath = GetLogfilePath(self.directories["output"], logindex)

        # Switch to logger
        setup_logger(logpath=logpath, logterm=True, level=self.config["log_level"])
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
        if self.config["atmosphere_model"] in [0, 1]:
            log.info("SOCRATES dir: " + self.directories["rad"])
        log.info(" ")

        # Count iterations
        loop_counter = {
            "total": 0,  # Total number of iters performed
            "total_min": 5,  # Minimum number of total loops
            "total_loops": self.config["iter_max"],  # Maximum number of total loops
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

        # Import the appropriate escape module
        if self.config["escape_model"] == 0:
            pass
        elif self.config["escape_model"] == 1:
            from proteus.utils.escape import RunZEPHYRUS
        elif self.config["escape_model"] == 2:
            from proteus.utils.escape import RunDummyEsc

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
            hf_row["age_star"] = self.config["time_star"]

            # Initial guess for flux
            hf_row["F_atm"] = self.config["F_atm"]
            hf_row["F_int"] = hf_row["F_atm"]
            hf_row["T_eqm"] = 2000.0

            # Planet size conversion, and calculate mantle mass (= liquid + solid)
            hf_row["M_planet"] = self.config["mass"] * M_earth
            hf_row["R_planet"] = self.config["radius"] * R_earth
            hf_row["gravity"] = const_G * hf_row["M_planet"] / (hf_row["R_planet"] ** 2.0)
            hf_row["M_mantle"] = calculate_mantle_mass(
                hf_row["R_planet"], hf_row["M_planet"], self.config["planet_coresize"]
            )

            # Store partial pressures and list of included volatiles
            log.info("Input partial pressures:")
            inc_vols = []
            for s in volatile_species:
                key_pp = str(s + "_initial_bar")
                key_in = str(s + "_included")

                log.info(
                    "    %-6s : %-5.2f bar (included = %s)"
                    % (s, self.config[key_pp], str(self.config[key_in] > 0))
                )

                if self.config[key_in] > 0:
                    inc_vols.append(s)
                    hf_row[s + "_bar"] = max(1.0e-30, float(self.config[key_pp]))
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
        star_modern_path = os.path.join(self.directories["fwl"], self.config["star_spectrum"])
        shutil.copyfile(star_modern_path, os.path.join(self.directories["output"], "-1.sflux"))

        # Prepare stellar models
        match self.config["star_model"]:
            case 'spada':
                # load modern spectrum
                star_struct_modern = mors.spec.Spectrum()
                star_struct_modern.LoadTSV(star_modern_path)
                star_struct_modern.CalcBandFluxes()

                # modern properties
                star_props_modern = mors.synthesis.GetProperties(
                    self.config["star_mass"],
                    self.config["star_rot_pctle"],
                    self.config["star_age_modern"] * 1000,
                )

            case 'baraffe':
                modern_wl, modern_fl = mors.ModernSpectrumLoad(
                    star_modern_path, self.directories["output"] + "/-1.sflux"
                )

                baraffe = mors.BaraffeTrack(self.config["star_mass"])

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
            hf_row["separation"] = self.config["semimajoraxis"] * AU * \
                                        (1 + 0.5 * self.config["eccentricity"]**2.0)


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
            if (abs(hf_row["Time"] - sinst_prev) > self.config["sinst_dt_update"]) or (
                loop_counter["total"] == 0
            ):
                sinst_prev = hf_row["Time"]

                # Get previous instellation
                if loop_counter["total"] > 0:
                    F_inst_prev = S_0
                else:
                    F_inst_prev = 0.0

                log.info("Updating instellation and radius")

                match self.config["star_model"]:
                    case 'spada':
                        hf_row["R_star"] = (
                            mors.Value(
                                self.config["star_mass"],
                                hf_row["age_star"] * 1000,
                                "Rstar"
                            )
                            * mors.const.Rsun
                            * 1.0e-2
                        )
                        S_0 = (
                            mors.Value(
                                self.config["star_mass"],
                                hf_row["age_star"] * 1000,
                                "Lbol"
                            ) * L_sun  / (4.0 * np.pi * hf_row["separation"]**2.0 )
                        )
                    case 'baraffe':
                        hf_row["R_star"] = (
                            baraffe.BaraffeStellarRadius(hf_row["age_star"]) * 1e9
                            * mors.const.Rsun
                            * 1.0e-2
                        )
                        S_0 = baraffe.BaraffeSolarConstant(
                            hf_row["age_star"] * 1e9, hf_row["separation"]/AU
                        )

                # Calculate new eqm temperature
                T_eqm_new = CalculateEqmTemperature(
                    S_0, self.config["asf_scalefactor"], self.config["albedo_pl"]
                )

                hf_row["F_ins"] = S_0  # instellation
                hf_row["T_eqm"] = T_eqm_new
                hf_row["T_skin"] = (
                    T_eqm_new * (0.5**0.25)
                )  # Assuming a grey stratosphere in radiative eqm (https://doi.org/10.5194/esd-7-697-2016)

                log.debug("Instellation change: %+.4e W m-2 (to 4dp)" % abs(S_0 - F_inst_prev))

            # Calculate a new (historical) stellar spectrum
            if (abs(hf_row["Time"] - sspec_prev) > self.config["sspec_dt_update"]) or (
                loop_counter["total"] == 0
            ):
                sspec_prev = hf_row["Time"]
                update_stellar_spectrum = True

                log.info("Updating stellar spectrum")
                match self.config["star_model"]:
                    case 'spada':
                        synthetic = mors.synthesis.CalcScaledSpectrumFromProps(
                            star_struct_modern, star_props_modern, hf_row["age_star"] * 1000
                        )
                        fl = synthetic.fl  # at 1 AU
                        wl = synthetic.wl
                    case 'baraffe':
                        fl = baraffe.BaraffeSpectrumCalc(
                            hf_row["age_star"] * 1e9, self.config["star_luminosity_modern"], modern_fl
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

            if (loop_counter["total"] >= loop_counter["init_loops"]) and self.config["escape_model"]:
                PrintHalfSeparator()

                if self.config["escape_model"] == 'zephyrus':
                    esc_result = RunZEPHYRUS(
                        hf_row,
                        dt,
                        self.config["star_mass"],
                        self.config["star_omega"],
                        self.config["escape_el_tidal_correction"],
                        self.config["semimajoraxis"] * AU,
                        self.config["eccentricity"],
                        hf_row["M_planet"],
                        self.config["efficiency_factor"],
                        hf_row["R_planet"],
                        hf_row["R_planet"],
                    )

                elif self.config["escape_model"] == 'dummy':
                    esc_result = RunDummyEsc(hf_row, dt, self.config["escape_dummy_rate"])

                # store total escape rate
                hf_row["esc_rate_total"] = esc_result["rate_bulk"]
                log.info(
                    "Bulk escape rate: %.2e kg yr-1 = %.2e kg s-1"
                    % (hf_row["esc_rate_total"] * secs_per_year, hf_row["esc_rate_total"])
                )
                # update elemental mass targets
                for e in element_list:
                    if e == "O":
                        continue

                    # store change, for statistics
                    esc_m = esc_result[e + "_kg_total"]

                    # update total elemental inventory
                    solvevol_target[e] = esc_m

                    # do not allow negative masses
                    solvevol_target[e] = max(0.0, solvevol_target[e])

            ############### / ESCAPE

            ############### OUTGASSING
            PrintHalfSeparator()
            solvevol_inp = {}
            solvevol_inp["M_mantle"] = hf_row["M_mantle"]
            solvevol_inp["T_magma"] = hf_row["T_magma"]
            solvevol_inp["Phi_global"] = hf_row["Phi_global"]
            solvevol_inp["gravity"] = hf_row["gravity"]
            solvevol_inp["mass"] = hf_row["M_planet"]
            solvevol_inp["radius"] = hf_row["R_planet"]
            solvevol_inp['radius'] = self.config['radius']
            solvevol_inp['fO2_shift_IW'] = self.config['fO2_shift_IW']
            solvevol_inp['hydrogen_earth_oceans'] = self.config['hydrogen_earth_oceans']
            solvevol_inp['CH_ratio'] = self.config['CH_ratio']
            solvevol_inp['nitrogen_ppmw'] = self.config['nitrogen_ppmw']
            solvevol_inp['sulfur_ppmw'] = self.config['sulfur_ppmw']
            for s in ('H2O', 'CO2', 'N2', 'S2', 'SO2', 'H2', 'CH4', 'CO'):
                solvevol_inp[f'{s}_initial_bar'] = self.config[f'{s}_initial_bar']
                solvevol_inp[f'{s}_included'] = self.config[f'{s}_included']

            #    recalculate mass targets during init phase, since these will be adjusted
            #    depending on the true melt fraction and T_magma found by SPIDER at runtime.
            if loop_counter["init"] < loop_counter["init_loops"]:
                # calculate target mass of atoms (except O, which is derived from fO2)
                if self.config.delivery.initial == 'elements':
                    solvevol_target = get_target_from_params(solvevol_inp)
                else:
                    solvevol_target = get_target_from_pressures(solvevol_inp)

                # prevent numerical issues
                for key in solvevol_target.keys():
                    if solvevol_target[key] < 1.0e4:
                        solvevol_target[key] = 0.0

            # solve for atmosphere composition
            solvevol_result = equilibrium_atmosphere(solvevol_target, solvevol_inp)

            #    store results
            for k in solvevol_result.keys():
                if k in hf_row.keys():
                    hf_row[k] = solvevol_result[k]

            #    calculate total atmosphere mass
            hf_row["M_atm"] = 0.0
            for s in volatile_species:
                hf_row["M_atm"] += hf_row[s + "_kg_atm"]
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
            if (self.config["solid_stop"] == 1) and (hf_row["Phi_global"] <= self.config["phi_crit"]):
                UpdateStatusfile(self.directories, 10)
                log.info("")
                log.info("===> Planet solidified! <===")
                log.info("")
                finished = True

            # Stop simulation when flux is small
            if (
                (self.config["emit_stop"] == 1)
                and (loop_counter["total"] > loop_counter["init_loops"] + 1)
                and (abs(hf_row["F_atm"]) <= self.config["F_crit"])
            ):
                UpdateStatusfile(self.directories, 14)
                log.info("")
                log.info("===> Planet no longer cooling! <===")
                log.info("")
                finished = True

            # Determine when the simulation enters a steady state
            if (
                (self.config["steady_stop"] == 1)
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
            if hf_row["M_atm"] <= self.config["escape_stop"] * hf_all.iloc[0]["M_atm"]:
                UpdateStatusfile(self.directories, 15)
                log.info("")
                log.info("===> Atmosphere has escaped! <===")
                log.info("")
                finished = True

            # Maximum time reached
            if hf_row["Time"] >= self.config["time_target"]:
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
                (self.config["plot_iterfreq"] > 0)
                and (loop_counter["total"] % self.config["plot_iterfreq"] == 0)
                and (not finished)
            ):
                PrintHalfSeparator()
                log.info("Making plots")
                UpdatePlots(hf_all, self.directories["output"], self.config)

            ############### / HOUSEKEEPING AND CONVERGENCE CHECK

        # FINAL THINGS BEFORE EXIT
        PrintHalfSeparator()

        # Deallocate atmosphere
        # if self.config["atmosphere_model"] == 1:
        #     deallocate_atmos(atm)

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
