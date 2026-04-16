"""Z.1: Verify JAX dSdt produces same output as numpy _dSdt_single.

Runs a brief PROTEUS startup with the chili_repro_v2 config to
construct both the numpy EntropySolver and the JAX components.
Calls both RHS implementations on the SAME state and compares.

Pre-flight gate for option Z (analytic Jacobian via JAX). If the
JAX RHS doesn't match numpy RHS to within ~1e-8 relative error,
CVODE's Newton iteration with the JAX-derived Jacobian will fail
to converge against the actual (numpy) RHS.

Usage:
    cd /Users/timlichtenberg/git/PROTEUS-Z
    PYTHONPATH=src python scripts/z01_verify_jax_numpy_parity.py

Output:
    Per-component absolute and relative error of dS/dt.
    Overall verdict: PASS (matched within tol) or FAIL (which terms
    diverge).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Use the chili_repro_v2 config as test scenario
PROTEUS_DIR = Path('/Users/timlichtenberg/git/PROTEUS-Z')
CONFIG_PATH = PROTEUS_DIR / 'output_files' / 'chili_repro_v2.toml'

# Prefer to use the main PROTEUS chili_repro_v2 config (in main checkout)
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path('/Users/timlichtenberg/git/PROTEUS/output_files/chili_repro_v2.toml')


def main():
    print('=== Z.1 parity check: JAX dSdt vs numpy _dSdt_single ===')
    print(f'Config: {CONFIG_PATH}')

    # ── PROTEUS startup ──
    from proteus.config import read_config_object
    from proteus.interior_energetics.aragog import AragogRunner
    from proteus.interior_energetics.common import Interior_t

    config = read_config_object(str(CONFIG_PATH))
    print(f'Config loaded; interior_energetics.module = {config.interior_energetics.module}')
    print(f'core_bc = {getattr(config.interior_energetics.aragog, "core_bc", "?")}')
    print(f'atol_temperature_equivalent = {config.interior_energetics.aragog.atol_temperature_equivalent}')

    # Construct a minimal hf_row matching CHILI initial state
    hf_row = {
        'Time': 0.0,
        'R_int': 6371000.0,
        'gravity': 9.81,
        'M_int': 5.972e24,
        'F_atm': 1.40e5,
        'T_eqm': 273.0,
        'tides': 0.0,
        'R_core': 0.55 * 6371000.0,
        'M_core': 0.32 * 5.972e24,
        'T_core': 7199.0,
        'P_surf': 257.66,
        'T_magma': 3820.0,
    }
    interior_o = Interior_t(nlev_b=80)
    interior_o.aragog_solver = None
    interior_o.tides = np.array([0.0])
    interior_o.ic = 1
    import pandas as pd
    hf_all = pd.DataFrame([hf_row])

    # We cannot easily call AragogRunner without all PROTEUS dirs
    # configured. Instead, manually invoke the solver setup pieces.
    print('\n--- Setting up numpy aragog solver ---')
    dirs = {
        'output': '/tmp/z01_test',
        'spider_eos_dir': '/Users/timlichtenberg/work/fwl_data/interior_lookup_tables/1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa',
    }
    Path(dirs['output']).mkdir(exist_ok=True)
    interior_o._spider_eos_dir = dirs['spider_eos_dir']

    AragogRunner.setup_logger(config, dirs)
    AragogRunner.setup_solver(config, hf_row, interior_o, dirs['output'])
    interior_o.aragog_solver.initialize()
    AragogRunner._set_entropy_ic(config, interior_o, dirs['output'], hf_row)

    solver = interior_o.aragog_solver
    print(f'numpy solver initialized; state vector size n = {len(solver._S0)}')
    print(f'  core_bc = {solver._core_bc}')
    print(f'  state_is_extended = {solver._state_is_extended}')

    # ── Build JAX components ──
    print('\n--- Building JAX components ---')
    import jax.numpy as jnp

    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import BoundaryParams, dSdt_energy_balance
    from aragog.jax.solver import dSdt as jax_dsdt

    eos_jax = EntropyEOS_JAX(dirs['spider_eos_dir'])

    params_jax = PhaseParams(
        phi_rheo=config.interior_energetics.rfront_loc,
        phi_width=config.interior_energetics.rfront_wid,
        viscosity_solid=10.0 ** float(config.interior_energetics.solid_log10visc),
        viscosity_liquid=10.0 ** float(config.interior_energetics.melt_log10visc),
        grain_size=config.interior_energetics.grain_size,
        k_solid=float(config.interior_energetics.solid_cond),
        k_liquid=float(config.interior_energetics.melt_cond),
        conduction=config.interior_energetics.trans_conduction,
        convection=config.interior_energetics.trans_convection,
        grav_sep=config.interior_energetics.trans_grav_sep,
        mixing=config.interior_energetics.trans_mixing,
        eddy_diff_thermal=float(config.interior_energetics.eddy_diffusivity_thermal),
        eddy_diff_chemical=float(config.interior_energetics.eddy_diffusivity_chemical),
        kappah_floor=config.interior_energetics.kappah_floor,
        bottom_up_grav_sep=True,
    )
    mesh_jax = MeshArrays.from_numpy_mesh(solver.evaluator.mesh)

    bc_cfg = solver.parameters.boundary_conditions
    # Energy_balance constants from the numpy solver's cached BC values
    cmb_area = float(getattr(solver, '_cmb_area', 0.0))
    core_M = float(getattr(solver, '_core_M', 0.0))
    cmb_dr_cmb = float(getattr(solver, '_cmb_dr_cmb', 0.0))
    print(f'energy_balance constants: cmb_area={cmb_area:.3e}, core_M={core_M:.3e}, cmb_dr_cmb={cmb_dr_cmb:.3e}')

    # If we're in energy_balance mode, set inner_bc_type=5 to signal
    # the new dSdt_energy_balance path. Otherwise use the existing BC type.
    inner_bc_type_for_jax = (
        5 if solver._core_bc == 'energy_balance' else bc_cfg.inner_boundary_condition
    )

    bc_jax = BoundaryParams(
        outer_bc_type=bc_cfg.outer_boundary_condition,
        outer_bc_value=bc_cfg.outer_boundary_value,
        emissivity=bc_cfg.emissivity,
        T_eq=bc_cfg.equilibrium_temperature,
        inner_bc_type=inner_bc_type_for_jax,
        inner_bc_value=bc_cfg.inner_boundary_value,
        core_density=solver.parameters.mesh.core_density,
        core_heat_capacity=bc_cfg.core_heat_capacity,
        tfac_core_avg=getattr(bc_cfg, 'tfac_core_avg', 1.147),
        cmb_area=cmb_area,
        core_M=core_M,
        cmb_dr_cmb=cmb_dr_cmb,
    )

    # Heating: just use zeros for the parity test
    n_stag = solver._n_stag
    heating = jnp.zeros(n_stag)
    args = (eos_jax, params_jax, mesh_jax, bc_jax, heating)
    print(f'JAX components built; mesh has n_stag={n_stag}')

    # ── Compare RHS at the IC ──
    print('\n--- Calling both RHS implementations at t=0, S=initial ---')
    S0 = solver._S0
    f_np = np.asarray(solver._dSdt_single(0.0, S0))
    print(f'numpy state vector: shape={S0.shape}; output shape={f_np.shape}')

    if solver._state_is_extended and solver._core_bc == 'energy_balance':
        # Use the new dSdt_energy_balance for the full N+1 state
        print(f'Using dSdt_energy_balance (N+1 state)')
        S_jax_input = jnp.asarray(S0)
        f_jax_full = np.asarray(dSdt_energy_balance(0.0, S_jax_input, args))
        f_np_entropy = f_np  # full N+1 comparison
    else:
        # Quasi_steady or other N-state mode
        S_jax_input = jnp.asarray(S0[:n_stag] if solver._state_is_extended else S0)
        f_jax_full = np.asarray(jax_dsdt(0.0, S_jax_input, args))
        f_np_entropy = f_np[:n_stag] if f_np.size == n_stag + 1 else f_np

    abs_err = np.abs(f_np_entropy - f_jax_full)
    denom = np.maximum(np.abs(f_np_entropy), 1e-30)
    rel_err = abs_err / denom

    print(f'\n=== Comparison results (n_stag={n_stag} entropy components) ===')
    print(f'numpy dS/dt  range: [{f_np_entropy.min():+.3e}, {f_np_entropy.max():+.3e}]')
    print(f'JAX   dS/dt  range: [{f_jax_full.min():+.3e}, {f_jax_full.max():+.3e}]')
    print(f'max abs err: {abs_err.max():.3e}')
    print(f'max rel err: {rel_err.max():.3e} at index {rel_err.argmax()}')
    print(f'median rel err: {np.median(rel_err):.3e}')

    if rel_err.max() < 1e-8:
        print('\n✅ PASS: JAX RHS matches numpy RHS within 1e-8 rel error')
        print('   → safe to use JAX-derived Jacobian with numpy solver')
        return 0
    else:
        print('\n❌ FAIL: JAX RHS diverges from numpy RHS')
        # Show worst components
        worst = np.argsort(rel_err)[-5:][::-1]
        print('\nWorst components (idx, numpy, JAX, abs_err, rel_err):')
        for i in worst:
            print(f'  [{i:3d}] {f_np_entropy[i]:+.4e}  vs  {f_jax_full[i]:+.4e}  '
                  f'abs={abs_err[i]:.3e}  rel={rel_err[i]:.3e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
