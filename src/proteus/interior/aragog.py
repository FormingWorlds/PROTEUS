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
    # Setup Aragog parameters from options at first iteration
    if (aragog_solver is None):
        SetupAragogSolver(OPTIONS, hf_row)
    # Else only update varying parameters
    else:
        UpdateAragogSolver()

    # Compute time step
    if IC_INTERIOR==1:
        dt = 0.0
    else:
        step_sf = 1.0 # dt scale factor
        dt = next_step(OPTIONS, dirs, hf_row, hf_all, step_sf)

    # Run Aragog solver
    aragog_solver.initialize()
    aragog_solver.solve()

    # Get Aragog output
    output = GetAragogOutput(OPTIONS, hf_row)
    sim_time = hf_row["Time"] + dt

    return sim_time, output

def SetupAragogSolver(OPTIONS:dict, hf_row:dict):

    global aragog_solver

    scalings = _ScalingsParameters(
            radius = OPTIONS["radius"] * R_earth, #6371000
            temperature = 4000,
            density = 4000,
            time = 3155760,
            )

    solver = _SolverParameters(
            start_time = 0,
            end_time = 200,
            atol = 1e-6,
            rtol = 1e-6,
            )

    boundary_conditions = _BoundaryConditionsParameters(
            outer_boundary_condition = 1,
            outer_boundary_value = 1500,
            inner_boundary_condition = 2,
            inner_boundary_value = 0,
            emissivity = 1,
            equilibrium_temperature = 273,
            core_radius = OPTIONS["planet_coresize"] * OPTIONS["radius"] * R_earth, #3504050
            core_density = 10738.332568062382,
            core_heat_capacity = 880,
            )

    mesh = _MeshParameters(
            outer_radius = OPTIONS["radius"] * R_earth, #6371000
            inner_radius = 5371000,
            number_of_nodes = OPTIONS["interior_nlev"], #50
            mixing_length_profile = "constant",
            surface_density = 4090,
            gravitational_acceleration = hf_row["gravity"], #9.81
            adiabatic_bulk_modulus = 260E9,
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
            surface_temperature = 3600,
            basal_temperature = 4000,
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

def UpdateAragogSolver():

    # Get output temperature field from previous run
    Tfield: np.array = aragog_solver.temperature_staggered[:, -1]
    # Make it dimensionless
    Tfield = Tfield / aragog_solver.parameters.scalings.temperature

    # Set the init field to previous result
    aragog_solver.parameters.initial_condition.from_field = True
    aragog_solver.parameters.initial_condition.init_temperature = Tfield

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
