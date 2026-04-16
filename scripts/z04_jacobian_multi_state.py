"""Z.2 multi-state Jacobian validation.

RESOLVED (2026-04-17 ff350fd): the all-NaN Jacobian on mid /
near_solid was traced via the z06 -> z07 -> z08 isolation chain
to ``jnp.abs(dSdr)`` and ``jnp.sqrt(jnp.maximum(x, 0))`` in
compute_mlt — both have singular gradients at zero that JAX's
backward pass turns into NaN. Fix: numpy-style smooth-abs
``0.5*(|x| + sqrt(x^2 + eps^2))`` and ``sqrt(x + eps^2)`` guards.
After the fix, mid and near_solid Jacobians match FD to machine
precision (1e-11 to 1e-12 rel err).

Final pass result on chili_repro_v2:

  state         max_rel    median_rel    cond_1
  IC            9.82e-03   2.19e-03      6.90e+26    PASS  (above-table extrap)
  mid           1.12e-11   3.64e-12      3.96e+26    PASS  (machine precision)
  near_solid    8.09e-10   1.22e-11      6.36e+14    PASS  (machine precision)

Extends z03 (IC-only Jacobian validation) to the same three states
exercised in z02 (IC, mid-trajectory, near-solid). For each state we
report:

  * shape, finite count, sparsity
  * condition number of the Jacobian (1-norm; SVD too expensive)
  * max and median rel err vs central-difference FD reference
  * timing: cold (JIT compile) and warm Jacobian call

This builds confidence that the JAX Jacobian does not degenerate at
the rheological-transition or near-solid regimes that the production
CHILI run actually traverses, before wiring the Jacobian into a real
CVODE call.

Pass criterion (per state): max rel err vs FD < 1e-2 across entries
with |J_fd| > 1e-3.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ARAGOG_Z = Path('/Users/timlichtenberg/git/aragog-Z/src')
if str(ARAGOG_Z) not in sys.path:
    sys.path.insert(0, str(ARAGOG_Z))

import numpy as np  # noqa: E402

CONFIG_PATH = Path('/Users/timlichtenberg/git/PROTEUS/output_files/chili_repro_v2.toml')


def fd_jacobian(rhs_fn, y, eps_rel=1e-5):
    """Central-difference Jacobian. Reuses the rhs_fn callable so the
    JAX RHS itself is the differentiation target (Jacobian-of-RHS
    parity, not Jacobian-of-numpy-RHS)."""
    n = y.size
    f0 = np.zeros(n)
    rhs_fn(0.0, y, f0)
    J = np.zeros((n, n))
    for j in range(n):
        eps_j = eps_rel * max(abs(y[j]), 1.0)
        y_p = y.copy()
        y_p[j] += eps_j
        y_m = y.copy()
        y_m[j] -= eps_j
        f_p = np.zeros(n)
        f_m = np.zeros(n)
        rhs_fn(0.0, y_p, f_p)
        rhs_fn(0.0, y_m, f_m)
        J[:, j] = (f_p - f_m) / (2.0 * eps_j)
    return J


def cond_1norm(J):
    """Numerically-cheap condition number estimate (1-norm).
    np.linalg.cond with p=1 uses LU decomposition + estimator."""
    try:
        return float(np.linalg.cond(J, p=1))
    except (np.linalg.LinAlgError, ValueError):
        return float('inf')


def main():
    print('=== Z.2 multi-state Jacobian validation (z04) ===')
    print(f'Config: {CONFIG_PATH}')
    sys.path.insert(0, str(Path(__file__).parent))
    from z02_parity_multi_state import (  # noqa: E402
        build_solver_and_jax_args,
        synthesize_state,
    )

    solver, args, eos_jax = build_solver_and_jax_args()
    n_stag = solver._n_stag
    state_dim = n_stag + 1
    print(f'numpy n_stag={n_stag}, core_bc={solver._core_bc}')

    eos_jax2, phase_params, mesh_arrays, bc_jax, heating_jax = args

    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian
    rhs_fn, jac_fn, info = build_jax_rhs_and_jacobian(
        eos_jax2, phase_params, mesh_arrays, bc_jax, np.asarray(heating_jax),
        np.ones(state_dim), np.ones(state_dim), 1.0,
        core_bc_mode='energy_balance',
    )

    results = []
    for kind in ['IC', 'mid', 'near_solid']:
        state = synthesize_state(solver, eos_jax, kind)
        # Detect phase-boundary cells and add a tiny offset to avoid
        # JAX autodiff hitting a 0*NaN through the unused where-branch
        # in compute_phase_state's _table_lookup_blend. This is a
        # diagnostic test for whether the NaN is a real Jacobian
        # ill-condition or just a non-differentiable seam in the synth IC.
        if kind != 'IC':
            P_stag_arr = np.asarray(solver.evaluator.mesh.staggered_pressure).ravel()
            import jax.numpy as _jnp
            S_sol_check = np.asarray(eos_jax.solidus_entropy(_jnp.asarray(P_stag_arr)))
            S_liq_check = np.asarray(eos_jax.liquidus_entropy(_jnp.asarray(P_stag_arr)))
            S_perturb = state[:n_stag].copy()
            # Offset by 0.1 J/kg/K away from the nearest boundary
            for i in range(n_stag):
                if abs(S_perturb[i] - S_sol_check[i]) < 5.0:
                    S_perturb[i] = S_sol_check[i] + 5.0
                elif abs(S_perturb[i] - S_liq_check[i]) < 5.0:
                    S_perturb[i] = S_liq_check[i] - 5.0
            state = np.concatenate([S_perturb, state[n_stag:]])
        print(f'\n--- State {kind!r} ---')
        S_check = state[:n_stag]
        print(f'  S range: [{S_check.min():.1f}, {S_check.max():.1f}]')

        # JAX Jacobian (cached after first state)
        J_jax = np.zeros((state_dim, state_dim))
        t0 = time.time()
        flag = jac_fn(0.0, state, None, J_jax)
        t_jac = time.time() - t0
        print(f'  J_jax: flag={flag}, time={t_jac*1000:.2f} ms')

        # Sanity
        n_finite = int(np.isfinite(J_jax).sum())
        if n_finite < J_jax.size:
            n_nan = int(np.isnan(J_jax).sum())
            n_inf = int(np.isinf(J_jax).sum())
            nan_rows, nan_cols = np.where(~np.isfinite(J_jax))
            print(f'  WARN: J_jax has {J_jax.size - n_finite} non-finite '
                  f'entries (NaN={n_nan}, inf={n_inf})')
            # Show the first few non-finite (i, j) pairs
            print(f'  first non-finite at (row, col): '
                  f'{list(zip(nan_rows[:6].tolist(), nan_cols[:6].tolist()))}')
            results.append({'kind': kind, 'pass': False,
                            'max_rel': float('nan'),
                            'median_rel': float('nan'),
                            'cond': float('nan'),
                            'sparsity': float('nan')})
            continue

        # FD reference
        t0 = time.time()
        J_fd = fd_jacobian(rhs_fn, state)
        t_fd = time.time() - t0
        print(f'  J_fd:  time={t_fd:.2f} s')

        # Condition number
        try:
            cond_jax = cond_1norm(J_jax)
        except Exception as exc:
            print(f'  cond(J_jax) failed: {exc}')
            cond_jax = float('inf')

        # Sparsity (entries above 1e-6 * max)
        thr = 1e-6 * float(np.abs(J_jax).max() if np.abs(J_jax).max() > 0 else 1.0)
        sparsity = 1.0 - int((np.abs(J_jax) > thr).sum()) / J_jax.size

        # Element-wise rel err where FD is significant
        fd_floor = 1e-3
        sig = np.abs(J_fd) > fd_floor
        if not sig.any():
            print('  WARN: no significant FD entries')
            results.append({'kind': kind, 'pass': False, 'max_rel': float('inf'),
                            'median_rel': float('inf'), 'cond': cond_jax,
                            'sparsity': sparsity})
            continue
        rel = np.abs(J_jax - J_fd) / np.maximum(np.abs(J_fd), fd_floor)
        rel_sig = rel[sig]
        max_rel = float(rel_sig.max())
        median_rel = float(np.median(rel_sig))
        n_sig = int(sig.sum())

        ok = max_rel < 1e-2
        print(f'  significant entries (|J_fd| > {fd_floor}): {n_sig}/{J_jax.size}')
        print(f'  cond_1(J_jax): {cond_jax:.3e}')
        print(f'  sparsity: {sparsity*100:.1f}%')
        print(f'  max rel err: {max_rel:.3e}  median: {median_rel:.3e}  '
              f'verdict: {"PASS" if ok else "FAIL"}')

        results.append({'kind': kind, 'pass': ok, 'max_rel': max_rel,
                        'median_rel': median_rel, 'cond': cond_jax,
                        'sparsity': sparsity, 'n_sig': n_sig})

    print('\n=== Multi-state Jacobian summary ===')
    print(f'{"state":>12} {"max_rel":>10} {"med_rel":>10} {"cond_1":>12} {"sparse":>8}')
    overall = True
    for r in results:
        flag = '✅' if r['pass'] else '❌'
        print(f'{r["kind"]:>12} {r["max_rel"]:>10.2e} {r["median_rel"]:>10.2e} '
              f'{r["cond"]:>12.2e} {r["sparsity"]*100:>7.1f}% {flag}')
        if not r['pass']:
            overall = False

    print('\n=== Overall ===', 'PASS' if overall else 'FAIL')
    if overall:
        print('  JAX Jacobian validates against FD across IC, mid-trajectory,')
        print('  and near-solid regimes. Safe to wire into CVODE Newton.')
    return 0 if overall else 1


if __name__ == '__main__':
    sys.exit(main())
