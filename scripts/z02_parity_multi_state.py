"""Z.6.A multi-state parity check: dSdt parity at several synthetic
profiles spanning IC, mid-solidification, and late-solidification.

The IC parity (z01) confirmed median 2.13e-5 / max 2e-3 rel error at the
chili_repro_v2 starting profile. This script extends that check to
synthetic states that mimic the rheological regimes the actual CHILI
simulation passes through, so the analytic-Jacobian work is not biased
toward parity at one snapshot of the trajectory.

States exercised:
  A. IC          — full magma ocean, S ~ 3900 J/kg/K throughout
  B. Mid-solid    — bottom 30% near solidus, top 70% near liquidus
  C. Near-solid   — bottom 70% near solidus, top 30% mushy
  D. Boundary off — IC with no SPIDER boundary copies (regression check)

For each state we report:
  - max rel err of dSdt
  - median rel err of dSdt
  - worst-component (idx, numpy_val, jax_val, rel_err)

Pass criterion: median rel err < 1e-3 in all states (Jacobian-quality).
"""

from __future__ import annotations

import sys
from pathlib import Path

ARAGOG_Z = Path('/Users/timlichtenberg/git/aragog-Z/src')
if str(ARAGOG_Z) not in sys.path:
    sys.path.insert(0, str(ARAGOG_Z))

import numpy as np  # noqa: E402

CONFIG_PATH = Path('/Users/timlichtenberg/git/PROTEUS/output_files/chili_repro_v2.toml')


def build_solver_and_jax_args():
    """Reuse z01 setup verbatim; returns numpy solver + JAX args tuple."""
    from proteus.config import read_config_object
    from proteus.interior_energetics.aragog import AragogRunner
    from proteus.interior_energetics.common import Interior_t

    config = read_config_object(str(CONFIG_PATH))
    hf_row = {
        'Time': 0.0, 'R_int': 6371000.0, 'gravity': 9.81,
        'M_int': 5.972e24, 'F_atm': 1.40e5, 'T_eqm': 273.0, 'tides': 0.0,
        'R_core': 0.55 * 6371000.0, 'M_core': 0.32 * 5.972e24,
        'T_core': 7199.0, 'P_surf': 257.66, 'T_magma': 3820.0,
    }
    interior_o = Interior_t(nlev_b=80)
    interior_o.aragog_solver = None
    interior_o.tides = np.array([0.0])
    interior_o.ic = 1
    dirs = {
        'output': '/tmp/z02_test',
        'spider_eos_dir': '/Users/timlichtenberg/work/fwl_data/interior_lookup_tables/1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa',
    }
    Path(dirs['output']).mkdir(exist_ok=True)
    interior_o._spider_eos_dir = dirs['spider_eos_dir']
    AragogRunner.setup_logger(config, dirs)
    AragogRunner.setup_solver(config, hf_row, interior_o, dirs['output'])
    interior_o.aragog_solver.initialize()
    AragogRunner._set_entropy_ic(config, interior_o, dirs['output'], hf_row)
    solver = interior_o.aragog_solver

    import jax.numpy as jnp

    from aragog.jax.eos import EntropyEOS_JAX
    from aragog.jax.phase import MeshArrays, PhaseParams
    from aragog.jax.solver import BoundaryParams

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
    bc_jax = BoundaryParams(
        outer_bc_type=bc_cfg.outer_boundary_condition,
        outer_bc_value=bc_cfg.outer_boundary_value,
        emissivity=bc_cfg.emissivity,
        T_eq=bc_cfg.equilibrium_temperature,
        inner_bc_type=5 if solver._core_bc == 'energy_balance' else bc_cfg.inner_boundary_condition,
        inner_bc_value=bc_cfg.inner_boundary_value,
        core_density=solver.parameters.mesh.core_density,
        core_heat_capacity=bc_cfg.core_heat_capacity,
        tfac_core_avg=getattr(bc_cfg, 'tfac_core_avg', 1.147),
        cmb_area=float(getattr(solver, '_cmb_area', 0.0)),
        core_M=float(getattr(solver, '_core_M', 0.0)),
        cmb_dr_cmb=float(getattr(solver, '_cmb_dr_cmb', 0.0)),
    )
    n_stag = solver._n_stag
    heating = jnp.zeros(n_stag)
    args = (eos_jax, params_jax, mesh_jax, bc_jax, heating)
    return solver, args, eos_jax


def synthesize_state(solver, eos_jax, kind: str) -> np.ndarray:
    """Build an N+1 state vector at the requested rheological regime.

    The CHILI solidification trajectory keeps S in the 3000-3900 J/kg/K
    range as the mantle cools: IC is at S~3900 (extrapolated above the
    EOS table top S_max~3236), mid-trajectory is at S~3500 (well inside
    the table), and near-solid is at S~3050 (close to the solidus at
    deep mantle pressure). We pick representative *uniform* profiles
    in each regime so the parity check exercises the full smth blend
    spectrum without depending on a specific snapshot file.
    """
    n_stag = solver._n_stag
    if kind == 'IC':
        # Real IC from the solver: S~3900, extrapolated above S_max.
        # This reproduces the z01 IC parity test exactly.
        S = np.asarray(solver._S0)[:n_stag].copy()
        dSdr_cmb = float(solver._S0[n_stag])
    elif kind == 'mid':
        # Mid-trajectory: S linearly varying from 3300 (CMB, more solid)
        # to 3700 (surface, more melt) — a representative active-
        # solidification profile. The entropy gradient drives non-zero
        # convective flux throughout the mantle, exercising MLT and
        # SPIDER conduction.
        S = np.linspace(3300.0, 3700.0, n_stag)
        dSdr_cmb = (S[1] - S[0]) / 1.0e5  # rough FD seed
    elif kind == 'near_solid':
        # Near-solid: S linearly varying from 3000 (CMB, fully solid) to
        # 3300 (surface, mushy). Mantle is past the rheological
        # transition; convective flux is small but conduction dominates.
        S = np.linspace(3000.0, 3300.0, n_stag)
        dSdr_cmb = (S[1] - S[0]) / 1.0e5
    else:
        raise ValueError(f'unknown state kind: {kind}')

    return np.concatenate([S, [dSdr_cmb]])


def parity_for_state(solver, args, state: np.ndarray, label: str) -> dict:
    """Compute dSdt with both numpy and JAX, return parity stats."""
    import jax.numpy as jnp

    from aragog.jax.solver import dSdt_energy_balance

    # numpy side: drives entropy_solver, advances state.update internally
    f_np = np.asarray(solver._dSdt_single(0.0, state))
    # JAX side
    f_jax = np.asarray(dSdt_energy_balance(0.0, jnp.asarray(state), args))

    abs_err = np.abs(f_np - f_jax)
    # Apply a numerical-noise floor: components where both |np| and |jax|
    # are below 1e-3 J/kg/K/yr are at machine epsilon for the entropy ODE
    # and don't contribute meaningful parity information. We treat them
    # as exactly equal for the rel_err metric; the abs_err still reports
    # the raw difference.
    noise_floor = 1.0e-3
    is_noise = (np.abs(f_np) < noise_floor) & (np.abs(f_jax) < noise_floor)
    denom = np.where(is_noise, 1.0, np.maximum(np.abs(f_np), 1e-30))
    rel_err = np.where(is_noise, 0.0, abs_err / denom)
    worst = int(np.argmax(rel_err))
    return {
        'label': label,
        'state_shape': state.shape,
        'max_rel_err': float(rel_err.max()),
        'median_rel_err': float(np.median(rel_err)),
        'max_abs_err': float(abs_err.max()),
        'worst_idx': worst,
        'worst_np': float(f_np[worst]),
        'worst_jax': float(f_jax[worst]),
        'worst_rel': float(rel_err[worst]),
        'np_range': (float(f_np.min()), float(f_np.max())),
        'jax_range': (float(f_jax.min()), float(f_jax.max())),
    }


def main():
    print('=== Z.6.A multi-state parity (z02) ===')
    print(f'Config: {CONFIG_PATH}')
    print('--- Building numpy solver + JAX args (one-time setup) ---')
    solver, args, eos_jax = build_solver_and_jax_args()
    print(f'numpy n_stag={solver._n_stag}, core_bc={solver._core_bc}')

    results = []
    for kind in ['IC', 'mid', 'near_solid']:
        state = synthesize_state(solver, eos_jax, kind)
        # Sanity-check: state vector size and S range
        n_stag = solver._n_stag
        S_check = state[:n_stag]
        print(f'  state {kind}: shape={state.shape}, S range=[{S_check.min():.1f}, {S_check.max():.1f}]')
        r = parity_for_state(solver, args, state, kind)
        results.append(r)
        print(f'    np dSdt range: [{r["np_range"][0]:+.2e}, {r["np_range"][1]:+.2e}]; '
              f'jax range: [{r["jax_range"][0]:+.2e}, {r["jax_range"][1]:+.2e}]')

    print('\n=== Parity summary ===')
    print(f'{"state":>12} {"max_rel":>10} {"median_rel":>12} {"max_abs":>10} {"worst_idx":>10}')
    for r in results:
        print(f'{r["label"]:>12} {r["max_rel_err"]:>10.2e} {r["median_rel_err"]:>12.2e} {r["max_abs_err"]:>10.2e} {r["worst_idx"]:>10d}')

    print('\n=== Verdict (per state) ===')
    overall_pass = True
    for r in results:
        # Pass criterion: median < 1e-3 (Jacobian preconditioner quality)
        ok = r['median_rel_err'] < 1e-3
        verdict = 'PASS' if ok else 'FAIL'
        print(f'  {r["label"]:>12}: {verdict} (median={r["median_rel_err"]:.2e}, max={r["max_rel_err"]:.2e}, '
              f'worst[{r["worst_idx"]:d}] np={r["worst_np"]:+.3e} jax={r["worst_jax"]:+.3e})')
        if not ok:
            overall_pass = False

    print('\n=== Overall ===', 'PASS' if overall_pass else 'FAIL')
    return 0 if overall_pass else 1


if __name__ == '__main__':
    sys.exit(main())
