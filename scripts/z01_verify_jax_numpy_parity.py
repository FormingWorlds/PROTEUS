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

# Force the JAX-augmented aragog-Z to take precedence over the installed
# aragog (which is the PROTEUS submodule and lacks the new compute_phase_state
# / dSdt_energy_balance code). aragog-Z must come BEFORE site-packages.
ARAGOG_Z = Path('/Users/timlichtenberg/git/aragog-Z/src')
if str(ARAGOG_Z) not in sys.path:
    sys.path.insert(0, str(ARAGOG_Z))

import numpy as np  # noqa: E402

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
        matprop_smooth_width=float(getattr(
            config.interior_energetics.spider, 'matprop_smooth_width', 0.0)),
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
        print('Using dSdt_energy_balance (N+1 state)')
        S_jax_input = jnp.asarray(S0)
        f_jax_full = np.asarray(dSdt_energy_balance(0.0, S_jax_input, args))
        f_np_entropy = f_np  # full N+1 comparison

        # Diagnostic: compare key intermediates
        print('\n--- Diagnostic intermediates at CMB ---')
        print(f'numpy state._heat_flux[0:3]: {solver.state._heat_flux[0]:.3e}, {solver.state._heat_flux[1]:.3e}, {solver.state._heat_flux[2]:.3e}')
        print(f'numpy entropy_basic[0]:      {solver.state._entropy_basic[0]:.3f}')
        print(f'numpy phase_basic.temp[0]:   {float(solver.state.phase_basic.temperature().flat[0]):.2f} K')
        print(f'numpy phase_basic.cp[0]:     {float(solver.state.phase_basic.heat_capacity().flat[0]):.2f}')
        print(f'numpy capacitance_stag[0]:   {float(solver.state.capacitance_staggered().flat[0]):.3e}')

        # Compare heat_flux at the divergent indices 57-61 (mid-mantle)
        from aragog.jax.phase import compute_fluxes as jax_cf
        flux_jax = jax_cf(jnp.asarray(S0[:n_stag]), 0.0, eos_jax, params_jax, mesh_jax,
                          heating, S_basic_cmb_override=S0[0], dSdr_cmb_override=S0[n_stag])
        hf_jax = np.asarray(flux_jax.heat_flux)
        hf_np = np.asarray(solver.state._heat_flux).ravel()
        print('\n--- CMB-region heat_flux (idx 0-3) ---')
        print(f'{"idx":>4} {"numpy F":>12} {"JAX F":>12} {"diff%":>8}')
        for i in range(0, 4):
            d = (hf_jax[i]-hf_np[i])/abs(hf_np[i])*100 if abs(hf_np[i]) > 1e-10 else 0
            print(f'{i:>4} {hf_np[i]:>12.3e} {hf_jax[i]:>12.3e} {d:>+7.1f}%')
        print('\n--- Mid-mantle heat_flux comparison (basic nodes 55-65) ---')
        print(f'{"idx":>4} {"numpy F":>12} {"JAX F":>12} {"diff%":>8}')
        for i in range(55, 66):
            d = (hf_jax[i]-hf_np[i])/abs(hf_np[i])*100 if abs(hf_np[i]) > 1e-10 else 0
            print(f'{i:>4} {hf_np[i]:>12.3e} {hf_jax[i]:>12.3e} {d:>+7.1f}%')

        # Surface-region heat flux: idx 77 worst residual is at the second-
        # to-last staggered cell (entropy block index 77 of 79). Print
        # F[n_basic-3..n_basic-1] = F[78..80] in 81-basic-node mesh.
        n_basic = hf_np.size
        print('\n--- Surface-region heat_flux (last 4 basic nodes) ---')
        print(f'{"idx":>4} {"numpy F":>12} {"JAX F":>12} {"diff%":>8}')
        for i in range(n_basic - 4, n_basic):
            d = (hf_jax[i]-hf_np[i])/abs(hf_np[i])*100 if abs(hf_np[i]) > 1e-10 else 0
            print(f'{i:>4} {hf_np[i]:>12.3e} {hf_jax[i]:>12.3e} {d:>+7.4f}%')

        # Surface-region inputs to flux: rho, T, k, Cp, alpha, dTdPs,
        # dSdr, kappa_h. Identifies whether the residual is from the
        # phase evaluation (compute_phase_state) or the MLT/conduction
        # downstream of identical inputs.
        print('\n--- Surface-region phase + MLT (last 3 basic nodes) ---')
        from aragog.jax.phase import compute_mlt as _jax_mlt2
        from aragog.jax.phase import evaluate_phase as _jax_eval2
        S_stag_arr_surf = jnp.asarray(S0[:n_stag])
        r_basic_arr_surf = np.asarray(solver.evaluator.mesh.basic.radii).ravel()
        r_stag0_surf = 0.5 * (r_basic_arr_surf[0] + r_basic_arr_surf[1])
        dr_off_surf = r_basic_arr_surf[0] - r_stag0_surf
        S_basic_cmb_surf = float(S0[0]) + float(S0[n_stag]) * dr_off_surf
        S_basic_jax2 = mesh_jax.quantity_matrix @ S_stag_arr_surf
        S_basic_jax2 = S_basic_jax2.at[0].set(S_basic_cmb_surf)
        dSdr_jax2 = mesh_jax.d_dr_matrix @ S_stag_arr_surf
        dSdr_jax2 = dSdr_jax2.at[0].set(S0[n_stag])
        ph_b_jax2 = _jax_eval2(eos_jax, params_jax, mesh_jax.P_basic, S_basic_jax2)
        kh_jax2, _ = _jax_mlt2(dSdr_jax2, ph_b_jax2, mesh_jax, params_jax)
        rho_b_np2 = np.asarray(solver.state.phase_basic.density()).ravel()
        T_b_np2 = np.asarray(solver.state.phase_basic.temperature()).ravel()
        Cp_b_np2 = np.asarray(solver.state.phase_basic.heat_capacity()).ravel()
        alpha_b_np2 = np.asarray(solver.state.phase_basic.thermal_expansivity()).ravel()
        dTdPs_b_np2 = np.asarray(solver.state.phase_basic.dTdPs()).ravel()
        dSdr_np2 = np.asarray(solver.state._dSdr).ravel()
        kh_np2 = np.asarray(solver.state._eddy_diffusivity).ravel()
        Sb_np2 = np.asarray(solver.state._entropy_basic).ravel()
        for i in range(n_basic - 3, n_basic):
            print(f'  idx={i}:')
            print(f'    S_basic:    np={Sb_np2[i]:.4f}  jax={float(S_basic_jax2[i]):.4f}  d={float(S_basic_jax2[i])-Sb_np2[i]:+.2e}')
            print(f'    rho:        np={rho_b_np2[i]:.4f}  jax={float(ph_b_jax2.density[i]):.4f}  d%={(float(ph_b_jax2.density[i])-rho_b_np2[i])/rho_b_np2[i]*100:+.4f}')
            print(f'    T:          np={T_b_np2[i]:.4f}  jax={float(ph_b_jax2.temperature[i]):.4f}  d%={(float(ph_b_jax2.temperature[i])-T_b_np2[i])/T_b_np2[i]*100:+.4f}')
            print(f'    Cp:         np={Cp_b_np2[i]:.4f}  jax={float(ph_b_jax2.heat_capacity[i]):.4f}  d%={(float(ph_b_jax2.heat_capacity[i])-Cp_b_np2[i])/Cp_b_np2[i]*100:+.4f}')
            print(f'    alpha:      np={alpha_b_np2[i]:.3e}  jax={float(ph_b_jax2.thermal_expansivity[i]):.3e}  d%={(float(ph_b_jax2.thermal_expansivity[i])-alpha_b_np2[i])/alpha_b_np2[i]*100:+.4f}')
            print(f'    dTdPs:      np={dTdPs_b_np2[i]:.3e}  jax={float(ph_b_jax2.dTdPs[i]):.3e}  d%={(float(ph_b_jax2.dTdPs[i])-dTdPs_b_np2[i])/dTdPs_b_np2[i]*100:+.4f}')
            print(f'    dSdr:       np={dSdr_np2[i]:.3e}  jax={float(dSdr_jax2[i]):.3e}')
            print(f'    kappa_h:    np={kh_np2[i]:.3e}  jax={float(kh_jax2[i]):.3e}  d%={(float(kh_jax2[i])-kh_np2[i])/kh_np2[i]*100:+.4f}')

        # F[0] decomposition: F_cond[0] = -k * dT/dr; F_conv[0] = rho*T*kappa_h*(-dSdr)
        from aragog.jax.phase import compute_mlt as _jax_mlt
        from aragog.jax.phase import evaluate_phase as _jax_eval
        S_stag_arr = jnp.asarray(S0[:n_stag])
        # Replicate dSdt_energy_balance overrides exactly: S_basic[0] uses
        # the boundary-state dSdr_cmb projected through dr_offset.
        r_basic_arr = np.asarray(solver.evaluator.mesh.basic.radii).ravel()
        r_stag0 = 0.5 * (r_basic_arr[0] + r_basic_arr[1])
        dr_off = r_basic_arr[0] - r_stag0
        S_basic_cmb_val = float(S0[0]) + float(S0[n_stag]) * dr_off
        S_basic_jax_arr = mesh_jax.quantity_matrix @ S_stag_arr
        S_basic_jax_arr = S_basic_jax_arr.at[0].set(S_basic_cmb_val)
        dSdr_jax_arr = mesh_jax.d_dr_matrix @ S_stag_arr
        dSdr_jax_arr = dSdr_jax_arr.at[0].set(S0[n_stag])
        ph_b_jax = _jax_eval(eos_jax, params_jax, mesh_jax.P_basic, S_basic_jax_arr)
        # phase_stag eval is unused in this diagnostic — kept for future
        # dTdr-comparison if needed.
        kh_jax_arr, _ = _jax_mlt(dSdr_jax_arr, ph_b_jax, mesh_jax, params_jax)

        # numpy equivalents
        dSdr_np = np.asarray(solver.state._dSdr).ravel()
        kh_np = np.asarray(solver.state._eddy_diffusivity).ravel() if hasattr(solver.state, '_eddy_diffusivity') else None
        rho_b_np = np.asarray(solver.state.phase_basic.density()).ravel()
        T_b_np = np.asarray(solver.state.phase_basic.temperature()).ravel()
        k_b_np = np.asarray(solver.state.phase_basic.thermal_conductivity()).ravel() if hasattr(solver.state.phase_basic, 'thermal_conductivity') else None

        print('\n--- F[0] component decomposition ---')
        print('                            numpy            JAX           diff%')
        print(f'  S_basic[0]:           {float(solver.state._entropy_basic[0]):>12.4f}   {float(S_basic_jax_arr[0]):>12.4f}   {(float(S_basic_jax_arr[0])-float(solver.state._entropy_basic[0]))/abs(float(solver.state._entropy_basic[0]))*100:+7.4f}%')
        print(f'  dSdr[0]:              {dSdr_np[0]:>12.3e}   {float(dSdr_jax_arr[0]):>12.3e}')
        print(f'  rho_basic[0]:         {rho_b_np[0]:>12.3f}   {float(ph_b_jax.density[0]):>12.3f}')
        print(f'  T_basic[0]:           {T_b_np[0]:>12.3f}   {float(ph_b_jax.temperature[0]):>12.3f}')
        if k_b_np is not None:
            print(f'  k_basic[0]:           {k_b_np[0]:>12.4f}   {float(ph_b_jax.thermal_conductivity[0]):>12.4f}')
        if kh_np is not None:
            print(f'  kappa_h[0]:           {kh_np[0]:>12.3e}   {float(kh_jax_arr[0]):>12.3e}')
        # Cp, alpha, dTdPs at basic[0] (kappa_h diagnostic — MLT uses these)
        Cp_b_np = np.asarray(solver.state.phase_basic.heat_capacity()).ravel()
        alpha_b_np = np.asarray(solver.state.phase_basic.thermal_expansivity()).ravel()
        dTdPs_b_np = np.asarray(solver.state.phase_basic.dTdPs()).ravel()
        print(f'  Cp_basic[0]:          {Cp_b_np[0]:>12.4f}   {float(ph_b_jax.heat_capacity[0]):>12.4f}')
        print(f'  alpha_basic[0]:       {alpha_b_np[0]:>12.3e}   {float(ph_b_jax.thermal_expansivity[0]):>12.3e}')
        print(f'  dTdPs_basic[0]:       {dTdPs_b_np[0]:>12.3e}   {float(ph_b_jax.dTdPs[0]):>12.3e}')
        # super-adiabatic (T/Cp)*dSdr at basic[0] — drives MLT and SPIDER conduction
        sa_np = T_b_np[0]/max(Cp_b_np[0], 100.) * dSdr_np[0]
        sa_jax = float(ph_b_jax.temperature[0])/max(float(ph_b_jax.heat_capacity[0]), 100.) * float(dSdr_jax_arr[0])
        print(f'  superadiab[0]:        {sa_np:>12.3e}   {sa_jax:>12.3e}')

        # CMB-region capacitance (rho * T at staggered nodes)
        # — drives dS/dt at the bottom cells.
        cap_np = np.asarray(solver.state.capacitance_staggered()).ravel()
        # Replicate JAX capacitance_staggered: phase_stag.density * phase_stag.temperature
        from aragog.jax.phase import evaluate_phase as _jax_eval
        S_stag_jax = jnp.asarray(S0[:n_stag])
        phase_stag_jax = _jax_eval(eos_jax, params_jax, mesh_jax.P_stag, S_stag_jax)
        cap_jax = np.asarray(phase_stag_jax.density * phase_stag_jax.temperature)
        print('\n--- CMB-region capacitance_stag (idx 0-3) ---')
        print(f'{"idx":>4} {"numpy cap":>12} {"JAX cap":>12} {"diff%":>8}')
        for i in range(0, 4):
            d = (cap_jax[i]-cap_np[i])/abs(cap_np[i])*100 if abs(cap_np[i]) > 1e-10 else 0
            print(f'{i:>4} {cap_np[i]:>12.3e} {cap_jax[i]:>12.3e} {d:>+7.1f}%')

        # Compare rho, T, kappa_h, dSdr at index 57
        print('\n--- Component breakdown at basic node 57 ---')
        rho_np = float(np.asarray(solver.state.phase_basic.density()).ravel()[57])
        T_np = float(np.asarray(solver.state.phase_basic.temperature()).ravel()[57])
        dSdr_np = float(solver.state._dSdr[57])
        print(f'numpy: rho={rho_np:.2f}  T={T_np:.2f}  dSdr={dSdr_np:+.3e}')

        from aragog.jax.phase import compute_mlt as jax_mlt
        from aragog.jax.phase import evaluate_phase as jax_eval
        S_basic_jax = mesh_jax.quantity_matrix @ jnp.asarray(S0[:n_stag])
        S_basic_jax = S_basic_jax.at[0].set(S0[0])
        phase_b_jax = jax_eval(eos_jax, params_jax, mesh_jax.P_basic, S_basic_jax)
        dSdr_jax = mesh_jax.d_dr_matrix @ jnp.asarray(S0[:n_stag])
        kappa_h_jax, _ = jax_mlt(dSdr_jax, phase_b_jax, mesh_jax, params_jax)
        print(f'JAX:   rho={float(phase_b_jax.density[57]):.2f}  T={float(phase_b_jax.temperature[57]):.2f}  '
              f'dSdr={float(dSdr_jax[57]):+.3e}  kappa_h={float(kappa_h_jax[57]):.3e}')
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
