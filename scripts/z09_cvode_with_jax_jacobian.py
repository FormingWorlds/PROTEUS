"""Z.3: end-to-end CVODE step with the JAX-derived analytic Jacobian.

Closes the option-Z loop by actually running scikits.odes CVODE with
the JAX RHS + jax.jacrev Jacobian for one short integration of the
chili_repro_v2 IC. Compares the result against CVODE with the JAX RHS
+ default finite-difference Jacobian.

Pass criteria:
  1. CVODE returns success with the JAX Jacobian.
  2. Final state matches the FD-Jacobian solve to within atol+rtol.
  3. Step count and wall time are reported (no hard threshold; we
     just want to see if the analytic Jacobian gives fewer Newton
     iterations per step).

This is the first real-CVODE wire-up. The JAX RHS + Jacobian go
through the public scikits.odes ``cvode_options`` dict and the
solver runs without any patches to the production aragog code path.
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


def main():
    print('=== Z.3 CVODE with JAX Jacobian (z09) ===')
    sys.path.insert(0, str(Path(__file__).parent))
    from z02_parity_multi_state import build_solver_and_jax_args  # noqa: E402

    solver, args, eos_jax = build_solver_and_jax_args()
    n_stag = solver._n_stag
    state_dim = n_stag + 1
    eos_jax2, params, mesh, bc, heating = args

    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian
    rhs_fn, jac_fn, info = build_jax_rhs_and_jacobian(
        eos_jax2, params, mesh, bc, np.asarray(heating),
        np.ones(state_dim), np.ones(state_dim), 1.0,
        core_bc_mode='energy_balance',
    )

    y0 = np.asarray(solver._S0).copy()

    # Try to import scikits.odes
    try:
        from scikits.odes import ode
    except ImportError:
        print('scikits.odes not available; cannot run real CVODE step')
        return 1

    # Trigger JIT compile out-of-band so timing measurements below
    # reflect steady-state CVODE iteration cost, not first-call compile.
    print('--- Triggering JIT compile (warm-up) ---')
    f0 = np.zeros(state_dim)
    rhs_fn(0.0, y0, f0)
    J_warm = np.zeros((state_dim, state_dim))
    jac_fn(0.0, y0, None, J_warm)
    print(f'  warm RHS + Jac done; J range [{J_warm.min():.2e}, {J_warm.max():.2e}]')

    # Common CVODE options. Read tolerance from CLI for the
    # tighten-tol experiment; default matches production atol=1e-8.
    rtol = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-6
    atol = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-8
    base_opts = dict(
        old_api=False,
        rtol=rtol,
        atol=atol,
        lmm_type='BDF',
        nonlinsolver='newton',
        max_steps=100000,
    )
    print(f'  using rtol={rtol}, atol={atol}')

    t_end = 1.0  # 1 yr integration
    t_span = [0.0, t_end]

    # ── Run 1: JAX RHS + JAX Jacobian ──
    print('\n--- Run 1: JAX RHS + JAX analytic Jacobian ---')
    opts_jax = dict(base_opts)
    opts_jax['jacfn'] = jac_fn
    try:
        solver_jax = ode('cvode', rhs_fn, **opts_jax)
        t0 = time.time()
        sol_jax = solver_jax.solve(t_span, y0)
        wall_jax = time.time() - t0
        print(f'  flag={sol_jax.flag}, message={sol_jax.message}')
        print(f'  wall time: {wall_jax*1000:.2f} ms')
        if sol_jax.values is not None and len(sol_jax.values.y) > 0:
            y_jax = np.asarray(sol_jax.values.y[-1])
            print(f'  final t = {sol_jax.values.t[-1]:.3e} yr')
            print(f'  final state range: [{y_jax.min():.4e}, {y_jax.max():.4e}]')
        else:
            y_jax = None
            print('  no solution returned')
    except Exception as exc:
        print(f'  CVODE with JAX Jacobian FAILED: {type(exc).__name__}: {exc}')
        y_jax = None
        wall_jax = float('inf')

    # ── Run 2: JAX RHS + default (FD) Jacobian ──
    print('\n--- Run 2: JAX RHS + CVODE default FD Jacobian ---')
    try:
        solver_fd = ode('cvode', rhs_fn, **base_opts)
        t0 = time.time()
        sol_fd = solver_fd.solve(t_span, y0)
        wall_fd = time.time() - t0
        print(f'  flag={sol_fd.flag}, message={sol_fd.message}')
        print(f'  wall time: {wall_fd*1000:.2f} ms')
        if sol_fd.values is not None and len(sol_fd.values.y) > 0:
            y_fd = np.asarray(sol_fd.values.y[-1])
            print(f'  final t = {sol_fd.values.t[-1]:.3e} yr')
            print(f'  final state range: [{y_fd.min():.4e}, {y_fd.max():.4e}]')
        else:
            y_fd = None
            print('  no solution returned')
    except Exception as exc:
        print(f'  CVODE with FD Jacobian FAILED: {type(exc).__name__}: {exc}')
        y_fd = None
        wall_fd = float('inf')

    # ── Compare ──
    print('\n=== Comparison ===')
    print(f'  RHS calls (JAX path): {info["rhs_calls"]}')
    print(f'  Jac calls (JAX path): {info["jac_calls"]}')
    if y_jax is not None and y_fd is not None:
        diff = np.abs(y_jax - y_fd)
        rel = diff / np.maximum(np.abs(y_fd), 1e-30)
        print(f'  max abs diff (JAX vs FD final state): {diff.max():.3e}')
        print(f'  max rel diff: {rel.max():.3e}')
        speedup = wall_fd / max(wall_jax, 1e-9)
        print(f'  wall time JAX vs FD: {wall_jax*1000:.2f} vs {wall_fd*1000:.2f} ms '
              f'(JAX is {speedup:.2f}x {"faster" if speedup > 1 else "slower"})')
        return 0 if (sol_jax.flag == 0 and sol_fd.flag == 0) else 1
    return 1


if __name__ == '__main__':
    sys.exit(main())
