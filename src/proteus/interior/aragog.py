# Aragog interior module
from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import netCDF4 as nc
import numpy as np
import pandas as pd
import platformdirs
from aragog import Output, Solver, aragog_file_logger
from aragog.parser import (
    Parameters,
    _BoundaryConditionsParameters,
    _EnergyParameters,
    _InitialConditionParameters,
    _MeshParameters,
    _PhaseMixedParameters,
    _PhaseParameters,
    _Radionuclide,
    _ScalingsParameters,
    _SolverParameters,
)

from proteus.interior.common import Interior_t
from proteus.interior.timestep import next_step
from proteus.utils.constants import R_earth, radnuc_data, secs_per_year

if TYPE_CHECKING:
    from proteus.config import Config

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA',
                                   platformdirs.user_data_dir('fwl_data')))


class AragogRunner():

    def __init__(self, config: Config, dirs: dict, hf_row: dict, hf_all:
                 pd.DataFrame, interior_o: Interior_t):
        AragogRunner.setup_logger(config, dirs)
        dt = AragogRunner.compute_time_step(config, dirs, hf_row, hf_all,
                                            interior_o)
        self.setup_or_update_solver(config, hf_row, interior_o, dt, dirs)
        interior_o.aragog_solver.initialize()
        self.aragog_solver = interior_o.aragog_solver

    @staticmethod
    def setup_logger(config: Config, dirs: dict):
        file_level = logging.getLevelName(config.interior.aragog.logging)
        aragog_file_logger(console_level=logging.WARNING, file_level=file_level,
                           log_dir=dirs["output"])

    @staticmethod
    def compute_time_step(config: Config, dirs: dict, hf_row: dict,
                          hf_all: pd.DataFrame, interior_o: Interior_t) -> (
            float):
        if interior_o.ic == 1:
            interior_o.aragog_solver = None
            return 0.0
        else:
            step_sf = 1.0  # dt scale factor
            return next_step(config, dirs, hf_row, hf_all, step_sf)

    @staticmethod
    def setup_or_update_solver(config: Config, hf_row: dict,
                               interior_o: Interior_t, dt: float, dirs: dict):
        if interior_o.aragog_solver is None:
            AragogRunner.setup_solver(config, hf_row, interior_o)
            if config.params.resume:
                AragogRunner.update_solver(dt, hf_row, interior_o,
                                   output_dir=dirs["output"])
        else:
            AragogRunner.update_solver(dt, hf_row, interior_o)

    @staticmethod
    def setup_solver(config:Config, hf_row:dict, interior_o:Interior_t):

        scalings = _ScalingsParameters(
            radius = R_earth, # scaling radius [m]
            temperature = 4000, # scaling temperature [K]
            density = 4000, # scaling density
            time = secs_per_year, # scaling time [sec]
            )

        solver = _SolverParameters(
            start_time = 0,
            end_time = 0,
            atol = config.interior.aragog.tolerance,
            rtol = config.interior.aragog.tolerance,
            tsurf_poststep_change = config.interior.aragog.tsurf_poststep_change,
            event_triggering = config.interior.aragog.event_triggering,
            )

        boundary_conditions = _BoundaryConditionsParameters(
            # 4 = prescribed heat flux
            outer_boundary_condition = 4,
            # first guess surface heat flux [W/m2]
            outer_boundary_value = hf_row["F_atm"],
            # 1 = core cooling model
            # 2 = prescribed heat flux
            # 3 = prescribed temperature
            inner_boundary_condition =  (
                config.interior.aragog.inner_boundary_condition),
            # core temperature [K], if inner_boundary_condition = 3
            inner_boundary_value = (
                config.interior.aragog.inner_boundary_value),
            # only used in gray body BC, outer_boundary_condition = 1
            emissivity = 1,
            # only used in gray body BC, outer_boundary_condition = 1
            equilibrium_temperature = hf_row["T_eqm"],
            # used if inner_boundary_condition = 1
            core_density = config.struct.core_density,
            # used if inner_boundary_condition = 1
            core_heat_capacity = 880,
            )

        mesh = _MeshParameters(
            # planet radius [m]
            outer_radius = hf_row["R_int"],
            # core radius [m]
            inner_radius = config.struct.corefrac * hf_row["R_int"],
            # basic nodes
            number_of_nodes = config.interior.aragog.num_levels,
            mixing_length_profile = "constant",
            # AdamsWilliamsonEOS parameter [kg/m3]
            surface_density = 4090,
            # [m/s-2]
            gravitational_acceleration = hf_row["gravity"],
            # AW-EOS parameter [Pa]
            adiabatic_bulk_modulus = config.interior.bulk_modulus,
            )

        energy = _EnergyParameters(
            conduction = config.interior.aragog.conduction,
            convection = config.interior.aragog.convection,
            gravitational_separation = (
                config.interior.aragog.gravitational_separation),
            mixing = config.interior.aragog.mixing,
            radionuclides = config.interior.radiogenic_heat,
            tidal = config.interior.tidal_heat,
            tidal_array = interior_o.tides
            )

        initial_condition = _InitialConditionParameters(
            # 1 = linear profile
            # 2 = user-defined profile
            # 3 = adiabatic profile
            initial_condition = 3,
            # initial top temperature (K)
            surface_temperature = config.interior.aragog.ini_tmagma,
            )

        # Get look up data directory, will be configurable in the future
        LOOK_UP_DIR = (
            FWL_DATA_DIR /
            "interior_lookup_tables/1TPa-dK09-elec-free/"
            "MgSiO3_Wolf_Bower_2018/"
        )
        MELTING_DIR = (
            FWL_DATA_DIR /
            "interior_lookup_tables/Melting_curves/"
        )

        phase_liquid = _PhaseParameters(
            density = LOOK_UP_DIR / "density_melt.dat",
            viscosity = 1E2,
            heat_capacity = LOOK_UP_DIR / "heat_capacity_melt.dat",
            melt_fraction = 1,
            thermal_conductivity = 4,
            #thermal_expansivity = 1.0E-5,
            thermal_expansivity = LOOK_UP_DIR / "thermal_exp_melt.dat",
            )

        phase_solid = _PhaseParameters(
            density = LOOK_UP_DIR / "density_solid.dat",
            viscosity = 1E21,
            heat_capacity = LOOK_UP_DIR / "heat_capacity_solid.dat",
            melt_fraction = 0,
            thermal_conductivity = 4,
            #thermal_expansivity = 1.0E-5,
            thermal_expansivity = LOOK_UP_DIR / "thermal_exp_solid.dat",
            )

        phase_mixed = _PhaseMixedParameters(
            latent_heat_of_fusion = 4e6,
            rheological_transition_melt_fraction = config.interior.rheo_phi_loc,
            rheological_transition_width = config.interior.rheo_phi_wid,
            solidus = MELTING_DIR / config.interior.melting_dir / "solidus.dat",
            liquidus = (MELTING_DIR / config.interior.melting_dir /
                       "liquidus.dat"),
            phase = "mixed",
            phase_transition_width = 0.1,
            grain_size = config.interior.grain_size,
            )

        radionuclides = []
        if config.interior.radiogenic_heat:
            # offset by age_ini, which converts model simulation time to the
            # actual age
            radio_t0 = config.delivery.radio_tref - config.star.age_ini
            radio_t0 *= 1e9 # Convert Gyr to yr

            def _append_radnuc(_iso, _cnc):
                radionuclides.append(
                    _Radionuclide(
                        name = _iso,
                        t0_years = radio_t0,
                        abundance = radnuc_data[_iso]["abundance"],
                        concentration = _cnc,
                        heat_production = radnuc_data[_iso]["heatprod"],
                        half_life_years = radnuc_data[_iso]["halflife"],
                    ))

            if config.delivery.radio_K > 0.0:
                _append_radnuc("k40", config.delivery.radio_K)

            if config.delivery.radio_Th > 0.0:
                _append_radnuc("th232", config.delivery.radio_Th)

            if config.delivery.radio_U > 0.0:
                _append_radnuc("u235", config.delivery.radio_U)
                _append_radnuc("u238", config.delivery.radio_U)

        param = Parameters(
            boundary_conditions = boundary_conditions,
            energy = energy,
            initial_condition = initial_condition,
            mesh = mesh,
            phase_solid = phase_solid,
            phase_liquid = phase_liquid,
            phase_mixed = phase_mixed,
            radionuclides = radionuclides,
            scalings = scalings,
            solver = solver,
            )

        interior_o.aragog_solver = Solver(param)

    @staticmethod
    def update_solver(dt:float, hf_row:dict, interior_o:Interior_t,
                                output_dir:str = None):

        # Set solver time
        # hf_row["Time"] is in yr so do not need to scale as long as scaling
        # time is secs_per_year
        interior_o.aragog_solver.parameters.solver.start_time = hf_row["Time"]
        interior_o.aragog_solver.parameters.solver.end_time = (hf_row["Time"]
                                                               + dt)

        # Get temperature field from previous run
        if output_dir is not None: # read it from output directory
            Tfield = read_last_Tfield(output_dir, hf_row["Time"])
        else: # get it from solver
            Tfield = interior_o.aragog_solver.temperature_staggered[:, -1]

        # Update initial condition
        Tfield = (Tfield /
                  interior_o.aragog_solver.parameters.scalings.temperature)
        # switch to user-defined init
        (interior_o.aragog_solver.parameters.initial_condition.
         initial_condition) = 2
        (interior_o.aragog_solver.parameters.initial_condition.
         init_temperature) = Tfield

        # Update boundary conditions
        (interior_o.aragog_solver.parameters.boundary_conditions.
         outer_boundary_value) = hf_row["F_atm"]

        # Update tidal heating within the mantle
        interior_o.aragog_solver.parameters.energy.tidal_array = (
            interior_o.tides /
            interior_o.aragog_solver.parameters.scalings.power_per_mass)

    def run_solver(self, hf_row, interior_o, dirs):
        # Run Aragog solver
        self.aragog_solver.solve()
        # Get Aragog output
        output = self.get_output(hf_row, interior_o)
        sim_time = self.aragog_solver.parameters.solver.end_time

        # Write output to a file
        self.write_output(dirs["output"], sim_time)

        return sim_time, output

    def write_output(self, output_dir:str, time:float):

        aragog_output: Output = Output(self.aragog_solver)

        fpath = os.path.join(output_dir,"data","%d_int.nc"%time)
        aragog_output.write_at_time(fpath,-1)

    def get_output(self, hf_row:dict, interior_o:Interior_t):

        aragog_output: Output = Output(self.aragog_solver)
        aragog_output.state.update(aragog_output.solution.y,
                                   aragog_output.solution.t)
        output = {"M_mantle": aragog_output.mantle_mass,
                  "T_magma": aragog_output.solution_top_temperature,
                  "Phi_global": aragog_output.melt_fraction_global,
                  "RF_depth": aragog_output.rheological_front,
                  "F_int": aragog_output.convective_heat_flux_basic[-1, -1]}

        if output["Phi_global"] > (1.0 - 1.0e-8):
            output["M_mantle_liquid"] = output["M_mantle"]
            output["M_mantle_solid"] = 0.0
        elif output["Phi_global"] < 1.e-8:
            output["M_mantle_liquid"] = 0.0
            output["M_mantle_solid"] = output["M_mantle"]
        else:
            output["M_mantle_liquid"] = output["M_mantle"] * output["Phi_global"]
            output["M_mantle_solid"] = (output["M_mantle"] *
                                        (1.0 - output["Phi_global"]))

        # Calculate surface area
        radii = aragog_output.radii_km_basic * 1e3 # [m]
        area  = 4 * np.pi * radii[-1]**2 # [m^2]

        # Mass at each mesh layer
        mass_s = aragog_output.mass_staggered[:,-1] # [kg]

        # Radiogenic heating
        Hradio_s = aragog_output.heating_radio[:,-1]  # [W kg-1]
        output["F_radio"] = np.dot(Hradio_s, mass_s) / area  # [W m-2]

        # Tidal heating flux
        Htidal_s = aragog_output.heating_tidal[:,-1]  # [W kg-1]
        output["F_tidal"] = np.dot(Htidal_s, mass_s)/area

        # Store arrays
        # FIX ME - Should extract values from staggered nodes rather than cropping
        # basic nodes.
        interior_o.phi      = np.array(aragog_output.melt_fraction_staggered[:,-1])
        interior_o.visc     = np.power(
            10.0, aragog_output.log10_viscosity_staggered[:,-1])
        interior_o.density  = np.array(
            aragog_output.density_basic[:,-1])[1:]
        interior_o.radius   = radii[:] # length N+1
        interior_o.mass     = mass_s[:]
        interior_o.temp     = np.array(
            aragog_output.temperature_K_staggered[:,-1])
        interior_o.pres     = np.array(
            aragog_output.pressure_GPa_staggered[:,-1] * 1e9)

        return output

def read_last_Tfield(output_dir:str, time:float):

    # Read Aragog output at last run
    fpath = os.path.join(output_dir,"data","%d_int.nc"%time)
    out = read_ncdf(fpath)

    # Get temperature field at basic nodes and interpolate to staggered nodes
    T_basic = out["temp_b"]
    N_basic = len(T_basic)
    T_staggered = (T_basic[0:N_basic-1] + T_basic[1:N_basic] ) / 2.0

    return T_staggered

def get_all_output_times(output_dir:str):
    files = glob.glob(output_dir+"/data/*_int.nc")
    years = [int(f.split("/")[-1].split("_int")[0]) for f in files]
    mask = np.argsort(years)

    return [years[i] for i in mask]

def read_ncdf(fpath:str):
    out = {}
    ds = nc.Dataset(fpath)

    for key in ds.variables.keys():
        out[key] = ds.variables[key][:]

    ds.close()
    return out

def read_ncdfs(output_dir:str, times:list):
    return [read_ncdf(os.path.join(output_dir, "data", "%d_int.nc"%t))
            for t in times]
