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

aragog_solver = None
log = logging.getLogger("fwl."+__name__)

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))

# Run the Aragog interior module
def RunAragog(config:Config, dirs:dict,
                hf_row:dict, hf_all:pd.DataFrame, interior_o:Interior_t):

    global aragog_solver

    # Setup Aragog logger
    file_level = logging.ERROR
    if config.params.out.logging == "DEBUG":
        file_level = logging.DEBUG
    aragog_file_logger(console_level = logging.WARNING,
                       file_level = file_level,
                       log_dir = dirs["output"])

    # Compute time step
    if interior_o.ic==1:
        dt = 0.0
        aragog_solver = None
    else:
        step_sf = 1.0 # dt scale factor
        dt = next_step(config, dirs, hf_row, hf_all, step_sf)

    # Setup Aragog parameters from options at first iteration
    if (aragog_solver is None):
        SetupAragogSolver(config, hf_row, interior_o)
        # Update state from stored data if resuming a simulation
        if config.params.resume:
            UpdateAragogSolver(dt, hf_row, interior_o, output_dir=dirs["output"])
    # Update varying parameters in ongoing simulation
    else:
        UpdateAragogSolver(dt, hf_row, interior_o)

    # Run Aragog solver
    aragog_solver.initialize()
    aragog_solver.solve()

    # Get Aragog output
    output = GetAragogOutput(hf_row, interior_o)
    sim_time = aragog_solver.parameters.solver.end_time

    # Write output to a file
    WriteAragogOutput(dirs["output"],sim_time)

    return sim_time, output


def SetupAragogSolver(config:Config, hf_row:dict, interior_o:Interior_t):

    global aragog_solver

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
            )

    boundary_conditions = _BoundaryConditionsParameters(
            outer_boundary_condition = 4, # 4 = prescribed heat flux
            outer_boundary_value = hf_row["F_atm"], # first guess surface heat flux [W/m2]
            inner_boundary_condition = 1, # 1 = core cooling model, 3 = prescribed temperature
            inner_boundary_value = 4000, # core temperature [K], if inner_boundary_condition = 3
            emissivity = 1, # only used in gray body BC, outer_boundary_condition = 1
            equilibrium_temperature = hf_row["T_eqm"], # only used in gray body BC, outer_boundary_condition = 1
            core_density = config.struct.core_density, # used if inner_boundary_condition = 1
            core_heat_capacity = 880, # used if inner_boundary_condition = 1
            )

    mesh = _MeshParameters(
            outer_radius = hf_row["R_int"], # planet radius [m]
            inner_radius = config.struct.corefrac * hf_row["R_int"], # core radius [m]
            number_of_nodes = config.interior.aragog.num_levels, # basic nodes
            mixing_length_profile = "constant",
            surface_density = 4090, # AdamsWilliamsonEOS parameter [kg/m3]
            gravitational_acceleration = hf_row["gravity"], # [m/s-2]
            adiabatic_bulk_modulus = config.interior.bulk_modulus, # AW-EOS parameter [Pa]
            )

    energy = _EnergyParameters(
            conduction = True,
            convection = True,
            gravitational_separation = False,
            mixing = False,
            radionuclides = config.interior.radiogenic_heat,
            tidal = config.interior.tidal_heat,
            tidal_array = interior_o.tides
            )

    initial_condition = _InitialConditionParameters(
            surface_temperature = config.interior.aragog.ini_tmagma, # initial top temperature (K)
            basal_temperature = config.interior.aragog.ini_tmagma, # initial bottom temperature (K)
            )

    # Get look up data directory, will be configurable in the future
    LOOK_UP_DIR = FWL_DATA_DIR / "interior_lookup_tables/1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018/"

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
            solidus = LOOK_UP_DIR / "solidus.dat",
            liquidus = LOOK_UP_DIR / "liquidus.dat",
            phase = "mixed",
            phase_transition_width = 0.1,
            grain_size = config.interior.grain_size,
            )

    radionuclides = []
    if config.interior.radiogenic_heat:
        # offset by age_ini, which converts model simulation time to the actual age
        radio_t0 = config.delivery.radio_tref - config.star.age_ini
        radio_t0 *= 1e9 # Convert Gyr to yr

        def _append_radnuc(_iso, _cnc):
            radionuclides.append(_Radionuclide(
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

    aragog_solver = Solver(param)

def UpdateAragogSolver(dt:float, hf_row:dict, interior_o:Interior_t,
                            output_dir:str = None):

    # Set solver time
    # hf_row["Time"] is in yr so do not need to scale as long as scaling time is secs_per_year
    aragog_solver.parameters.solver.start_time = hf_row["Time"]
    aragog_solver.parameters.solver.end_time = hf_row["Time"] + dt

    # Get temperature field from previous run
    if output_dir is not None: # read it from output directory
        Tfield = read_last_Tfield(output_dir, hf_row["Time"])
    else: # get it from solver
        Tfield = aragog_solver.temperature_staggered[:, -1]

    # Update initial condition
    Tfield = Tfield / aragog_solver.parameters.scalings.temperature
    aragog_solver.parameters.initial_condition.from_field = True
    aragog_solver.parameters.initial_condition.init_temperature = Tfield

    # Update boundary conditions
    aragog_solver.parameters.boundary_conditions.outer_boundary_value = hf_row["F_atm"]

    # Update tidal heating within the mantle
    aragog_solver.parameters.energy.tidal_array = \
        interior_o.tides / aragog_solver.parameters.scalings.power_per_mass

def WriteAragogOutput(output_dir:str, time:float):

    aragog_output: Output = Output(aragog_solver)

    fpath = os.path.join(output_dir,"data","%d_int.nc"%time)
    aragog_output.write_at_time(fpath,-1)

def GetAragogOutput(hf_row:dict, interior_o:Interior_t):

    aragog_output: Output = Output(aragog_solver)
    aragog_output.state.update(aragog_output.solution.y, aragog_output.solution.t)
    output = {}

    output["M_mantle"] = aragog_output.mantle_mass
    output["T_magma"] = aragog_output.solution_top_temperature
    output["Phi_global"] = float(aragog_output.melt_fraction_global)
    output["RF_depth"] = float(aragog_output.rheological_front)
    output["F_int"] = aragog_output.convective_heat_flux_basic[-1,-1] # Need to be revised for consistency

    output["M_mantle_liquid"] = output["M_mantle"] * output["Phi_global"]
    output["M_mantle_solid"] = output["M_mantle"] - output["M_mantle_liquid"]

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
    # FIX ME - Should extract values from staggered nodes rather than cropping basic nodes.
    interior_o.phi      = np.array(aragog_output.melt_fraction_staggered[:,-1])
    interior_o.visc     = np.power(10.0, aragog_output.log10_viscosity_staggered[:,-1])
    interior_o.density  = np.array(aragog_output.density_basic[:,-1])[1:]
    interior_o.radius   = radii[:] # length N+1
    interior_o.mass     = mass_s[:]
    interior_o.temp     = np.array(aragog_output.temperature_K_staggered[:,-1])
    interior_o.pres     = np.array(aragog_output.pressure_GPa_staggered[:,-1] * 1e9)

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
    return [read_ncdf(os.path.join(output_dir, "data", "%d_int.nc"%t)) for t in times]
