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
from aragog import Output, Solver
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

from proteus.interior.timestep import next_step
from proteus.utils.constants import R_earth, secs_per_year, radnuc_data

if TYPE_CHECKING:
    from proteus.config import Config

aragog_solver = None
log = logging.getLogger("fwl."+__name__)

FWL_DATA_DIR = Path(os.environ.get('FWL_DATA', platformdirs.user_data_dir('fwl_data')))

# Run the Aragog interior module
def RunAragog(config:Config, dirs:dict, IC_INTERIOR:int, hf_row:dict, hf_all:pd.DataFrame):

    global aragog_solver

    # Compute time step
    if IC_INTERIOR==1:
        dt = 0.0
        aragog_solver = None
    else:
        step_sf = 1.0 # dt scale factor
        dt = next_step(config, dirs, hf_row, hf_all, step_sf)

    # Setup Aragog parameters from options at first iteration
    if (aragog_solver is None):
        SetupAragogSolver(config, hf_row)
    # Else only update varying parameters
    else:
        UpdateAragogSolver(dt, hf_row)

    # Run Aragog solver
    aragog_solver.initialize()
    aragog_solver.solve()

    # Get Aragog output
    output = GetAragogOutput(hf_row)
    sim_time = aragog_solver.parameters.solver.end_time

    # Write output to a file
    WriteAragogOutput(dirs["output"],sim_time)

    return sim_time, output

def SetupAragogSolver(config:Config, hf_row:dict):

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
            inner_boundary_condition = 1, # 3 = prescribed temperature
            inner_boundary_value = 4000, # core temperature [K], if inner_boundary_condition = 3
            emissivity = 1, # only used in gray body BC, outer_boundary_condition = 1
            equilibrium_temperature = hf_row["T_eqm"], # only used in gray body BC, outer_boundary_condition = 1
            core_density = 10738.332568062382, # not used now
            core_heat_capacity = 880, # not used now
            )

    mesh = _MeshParameters(
            outer_radius = hf_row["R_int"], # planet radius [m]
            inner_radius = config.struct.corefrac * hf_row["R_int"], # core radius [m]
            number_of_nodes = config.interior.aragog.num_levels, # basic nodes
            mixing_length_profile = "constant",
            surface_density = 4090, # AdamsWilliamsonEOS parameter [kg/m3]
            gravitational_acceleration = hf_row["gravity"], # [m/s-2]
            adiabatic_bulk_modulus = 260E9, # AdamsWilliamsonEOS parameter [Pa]
            )

    energy = _EnergyParameters(
            conduction = True,
            convection = True,
            gravitational_separation = False,
            mixing = False,
            radionuclides = False,
            tidal = False,
            )

    initial_condition = _InitialConditionParameters(
            surface_temperature = config.interior.aragog.ini_tmagma, # initial top temperature (K)
            basal_temperature = config.interior.aragog.ini_tmagma, # initial bottom temperature (K)
            )

    phase_liquid = _PhaseParameters(
            density = 4000,
            viscosity = 1E2,
            heat_capacity = 1000,
            melt_fraction = 1,
            thermal_conductivity = 4,
            thermal_expansivity = 1.0E-5,
            )

    phase_solid = _PhaseParameters(
            density = 4200,
            viscosity = 1E21,
            heat_capacity = 1000,
            melt_fraction = 0,
            thermal_conductivity = 4,
            thermal_expansivity = 1.0E-5,
            )

    phase_mixed = _PhaseMixedParameters(
            latent_heat_of_fusion = 4e6,
            rheological_transition_melt_fraction = 0.4,
            rheological_transition_width = 0.15,
            solidus = FWL_DATA_DIR / "interior_lookup_tables/1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018/solidus.dat",
            liquidus = FWL_DATA_DIR / "interior_lookup_tables/1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018/liquidus.dat",
            phase = "mixed",
            phase_transition_width = 0.1,
            grain_size = config.interior.grain_size,
            )

    radionuclides = []
    if config.interior.radiogenic_heat:
        radio_t0 = config.delivery.radio_tref * 1e9 # Convert Gyr to yr

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

def UpdateAragogSolver(dt:float, hf_row:dict):

    # Set solver time
    # hf_row["Time"] is in yr so do not need to scale as long as scaling time is secs_per_year
    aragog_solver.parameters.solver.start_time = hf_row["Time"]
    aragog_solver.parameters.solver.end_time = hf_row["Time"] + dt

    # Update initial condition (temperature field from previous run)
    Tfield: np.array = aragog_solver.temperature_staggered[:, -1]
    Tfield = Tfield / aragog_solver.parameters.scalings.temperature
    aragog_solver.parameters.initial_condition.from_field = True
    aragog_solver.parameters.initial_condition.init_temperature = Tfield

    # Update boundary conditions
    aragog_solver.parameters.boundary_conditions.outer_boundary_value = hf_row["F_atm"]

    return

def WriteAragogOutput(output_dir:str, time:float):

    aragog_output: Output = Output(aragog_solver)

    fpath = os.path.join(output_dir,"data","%d_int.nc"%time)
    aragog_output.write_at_time(fpath,-1)

def GetAragogOutput(hf_row:dict):

    aragog_output: Output = Output(aragog_solver)
    output = {}

    output["M_mantle"] = aragog_output.mantle_mass
    output["T_magma"] = aragog_output.solution_top_temperature
    output["Phi_global"] = float(aragog_output.melt_fraction_global)
    output["RF_depth"] = float(aragog_output.rheological_front)
    output["F_int"] = aragog_output.convective_heat_flux_basic[-1,-1] # Need to be revised for consistency

    output["M_mantle_liquid"] = output["M_mantle"] * output["Phi_global"]
    output["M_mantle_solid"] = output["M_mantle"] - output["M_mantle_liquid"]
    output["M_core"] = aragog_output.core_mass

    # Tidal heating is not supported by Aragog (yet)
    output["F_tidal"] = 0.0

    # Radiogenic heating
    Hradio_s = aragog_output.heating_radio[:,-1] # [W kg-1]
    mass_s   = aragog_output.mass_staggered[:,-1] # [kg]
    Hradio_total = np.dot(Hradio_s, mass_s)

    radii_s  = aragog_output.radii_km_staggered * 1e3 # [m]
    area = 4 * np.pi * radii_s[0]**2
    output["F_radio"] = Hradio_total / area

    return output


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
