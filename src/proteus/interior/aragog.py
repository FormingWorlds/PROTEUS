# Aragog interior module
from __future__ import annotations

import logging

import pandas as pd

from aragog import Solver
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
from proteus.utils.constants import R_earth

aragog_solver = None
log = logging.getLogger("fwl."+__name__)

# Run the Aragog interior module
def RunAragog(OPTIONS:dict, dirs:dict, IC_INTERIOR:int, hf_row:dict, hf_all:pd.DataFrame):

    global aragog_solver

    # Compute time step
    if IC_INTERIOR==1:
        dt = 0.0
    else:
        step_sf = 1.0 # dt scale factor
        dt = next_step(OPTIONS, dirs, hf_row, hf_all, step_sf)

    # Setup Aragog parameters from options at first iteration
    if (aragog_solver is None):
        SetupAragogSolver(OPTIONS, hf_row)
    # Else only update varying parameters
    else:
        UpdateAragogSolver(dt, hf_row)

    # Run Aragog solver
    aragog_solver.initialize()
    aragog_solver.solve()

    # Get Aragog output
    output = GetAragogOutput(OPTIONS, hf_row)
    sim_time = aragog_solver.parameters.solver.end_time

    return sim_time, output

def SetupAragogSolver(OPTIONS:dict, hf_row:dict):

    global aragog_solver

    scalings = _ScalingsParameters(
            radius = R_earth, # scaling radius [m]
            temperature = 4000, # scaling temperature [K]
            density = 4000, # scaling density
            time = 3155760, # scaling time (unit???)
            )

    solver = _SolverParameters(
            start_time = 0,
            end_time = 0,
            atol = 1e-6,
            rtol = 1e-6,
            )

    boundary_conditions = _BoundaryConditionsParameters(
            outer_boundary_condition = 4, # 4 = prescribed heat flux
            outer_boundary_value = OPTIONS["F_atm"], # first guess surface heat flux [W/m2]
            inner_boundary_condition = 3, # 3 = prescribed temperature
            inner_boundary_value = 4000, # core temperature [K]
            emissivity = 1, # only used in gray body BC, outer_boundary_condition = 1
            equilibrium_temperature = 273, # only used in gray body BC, outer_boundary_condition = 1
            core_radius = OPTIONS["planet_coresize"] * OPTIONS["radius"] * R_earth, # not used now
            core_density = 10738.332568062382, # not used now
            core_heat_capacity = 880, # not used now
            )

    mesh = _MeshParameters(
            outer_radius = OPTIONS["radius"] * R_earth, # planet radius [m]
            inner_radius = OPTIONS["planet_coresize"] * OPTIONS["radius"] * R_earth, # core radius [m]
            number_of_nodes = OPTIONS["interior_nlev"], # 50
            mixing_length_profile = "constant",
            surface_density = 4090,
            gravitational_acceleration = hf_row["gravity"], # [m/s-2]
            adiabatic_bulk_modulus = 260E9, # TO CLARIFY
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
            surface_temperature = 4000, # initial top temperature (K)
            basal_temperature = 4000, # initial bottom temperature (K)
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
            solidus = "aragog/data/test/solidus_1d_lookup.dat",
            liquidus = "aragog/data/test/liquidus_1d_lookup.dat",
            phase = "mixed",
            phase_transition_width = 0.1,
            grain_size = OPTIONS["grain_size"],
            )

    radionuclides = _Radionuclide(
            name = "U235",
            t0_years = 4.55E9,
            abundance = 0.0072045,
            concentration = 0.031,
            heat_production = 5.68402E-4,
            half_life_years = 704E6,
            )

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

def GetAragogOutput(OPTIONS:dict, hf_row:dict):

    output = {}
    output["M_mantle"] = hf_row["M_mantle"]
    output["M_core"] = hf_row["M_core"]
    output["F_int"] = hf_row["F_atm"]
    output["T_magma"] = 3500.0
    output["Phi_global"] = 0.2
    output["M_mantle_liquid"] = output["M_mantle"] * output["Phi_global"]
    output["M_mantle_solid"] = output["M_mantle"] - output["M_mantle_liquid"]
    output["RF_depth"] = output["Phi_global"] * (1- OPTIONS["planet_coresize"])

    return output
