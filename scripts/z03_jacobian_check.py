"""Z.2: Validate JAX-derived analytic Jacobian against finite-difference
Jacobian on the chili_repro_v2 IC.

The analytic-Jacobian path for option Z uses ``jax.jacrev`` to
differentiate ``aragog.jax.solver.dSdt_energy_balance`` w.r.t. the
state vector. CVODE consumes the resulting Jacobian as a Newton
preconditioner. For the Jacobian to be a *useful* preconditioner it
must:

1. Have the right shape (N+1, N+1) for energy_balance mode.
2. Contain only finite values.
3. Have non-trivial sparsity pattern (entropy ODE is locally
   coupled, so most off-diagonal terms beyond the few nearest
   neighbours should be zero or near-zero).
4. Match a finite-difference Jacobian within ~1e-3 relative error
   per entry where the FD Jacobian is itself well-conditioned.

We use a small relative perturbation ε=1e-5 and central differences
for the FD reference. Bigger ε would dominate by truncation, smaller
ε by floating-point cancellation; 1e-5 is the standard choice for
double-precision second-derivative estimation.

Pass criterion: max element-wise rel err < 1e-2 across all entries
where |J_fd| > 1e-3 (entries below the noise floor are skipped).

Usage:
    cd /Users/timlichtenberg/git/PROTEUS-Z
    python scripts/z03_jacobian_check.py
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
    print('=== Z.2 Jacobian validation (z03) ===')
    print(f'Config: {CONFIG_PATH}')
    # Reuse z02 setup
    sys.path.insert(0, str(Path(__file__).parent))
    from z02_parity_multi_state import build_solver_and_jax_args  # noqa: E402

    solver, args, eos_jax = build_solver_and_jax_args()
    n_stag = solver._n_stag
    print(f'numpy n_stag={n_stag}, core_bc={solver._core_bc}')

    # Build the JAX RHS + Jacobian via the option Z infrastructure. We
    # use identity scalings (state_scale=1, rhs_scale=1, t_ref=1) so
    # the Jacobian represents pure d(dSdt_phys)/dS_phys, which is what
    # we compare the FD reference against.
    eos_jax2, phase_params, mesh_arrays, bc_jax, heating_jax = args
    state_dim = n_stag + 1  # energy_balance
    state_scale = np.ones(state_dim)
    rhs_scale = np.ones(state_dim)
    t_ref = 1.0
    heating_arr = np.asarray(heating_jax)

    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian
    print('--- Building JAX RHS + Jacobian (energy_balance mode) ---')
    t0 = time.time()
    rhs_fn, jac_fn, info = build_jax_rhs_and_jacobian(
        eos_jax2, phase_params, mesh_arrays, bc_jax, heating_arr,
        state_scale, rhs_scale, t_ref,
        core_bc_mode='energy_balance',
    )
    print(f'  build time: {time.time()-t0:.2f} s (no JIT trigger yet)')

    # IC state vector
    y0 = np.asarray(solver._S0).copy()
    print(f'  state shape: {y0.shape}; expected ({state_dim},)')
    assert y0.shape == (state_dim,), 'state size mismatch'

    # ── Trigger Jacobian JIT compile + first call ──
    print('--- Computing JAX Jacobian (first call: JIT compile) ---')
    J_jax = np.zeros((state_dim, state_dim))
    t0 = time.time()
    flag = jac_fn(0.0, y0, None, J_jax)
    t_jit = time.time() - t0
    print(f'  jac_fn flag={flag}, JIT-compile + first call: {t_jit:.2f} s')
    print(f'  J shape={J_jax.shape}, dtype={J_jax.dtype}')
    print(f'  J min/max: {J_jax.min():.3e} / {J_jax.max():.3e}')

    # Second call: JIT cached, should be fast
    t0 = time.time()
    flag = jac_fn(0.0, y0, None, J_jax)
    t_warm = time.time() - t0
    print(f'  warm jac_fn call: {t_warm*1000:.2f} ms (~{(t_jit/max(t_warm,1e-9)):.0f}x speedup over cold)')

    # ── Sanity checks ──
    print('\n--- Sanity checks ---')
    n_finite = int(np.isfinite(J_jax).sum())
    n_total = J_jax.size
    print(f'  finite entries: {n_finite}/{n_total}')
    assert n_finite == n_total, 'Jacobian has non-finite entries'

    # Sparsity: count entries with |J| > 1e-6 * |J|.max()
    thresh = 1e-6 * float(np.abs(J_jax).max())
    n_nonzero = int((np.abs(J_jax) > thresh).sum())
    sparsity = 1.0 - n_nonzero / n_total
    print(f'  sparsity (|J| < {thresh:.2e}): {sparsity*100:.1f}% (entries below noise floor)')

    # Bandwidth: distance from main diagonal for non-trivial entries
    rows, cols = np.where(np.abs(J_jax) > thresh)
    if rows.size:
        bw = int(np.abs(rows - cols).max())
        print(f'  effective bandwidth: {bw} (entropy ODE is locally coupled)')

    # ── Finite-difference reference Jacobian ──
    print('\n--- Computing FD reference Jacobian (central diff, eps=1e-5) ---')
    eps_rel = 1e-5
    f0 = np.zeros(state_dim)
    rhs_fn(0.0, y0, f0)
    J_fd = np.zeros((state_dim, state_dim))
    t0 = time.time()
    for j in range(state_dim):
        eps_j = eps_rel * max(abs(y0[j]), 1.0)
        y_plus = y0.copy()
        y_plus[j] += eps_j
        y_minus = y0.copy()
        y_minus[j] -= eps_j
        f_plus = np.zeros(state_dim)
        f_minus = np.zeros(state_dim)
        rhs_fn(0.0, y_plus, f_plus)
        rhs_fn(0.0, y_minus, f_minus)
        J_fd[:, j] = (f_plus - f_minus) / (2.0 * eps_j)
    t_fd = time.time() - t0
    print(f'  FD Jacobian time: {t_fd:.2f} s ({state_dim} columns x 2 RHS calls each)')

    # ── Compare J_jax vs J_fd ──
    print('\n--- JAX vs FD Jacobian comparison ---')
    abs_err = np.abs(J_jax - J_fd)
    fd_floor = 1e-3
    significant = np.abs(J_fd) > fd_floor
    # Only compare where FD value is meaningful (avoid noise-floor div)
    if not significant.any():
        print('  WARN: no entries above FD floor; relax floor or check IC')
        return 1
    rel_err = abs_err / np.maximum(np.abs(J_fd), fd_floor)
    rel_err_significant = rel_err[significant]
    max_rel = float(rel_err_significant.max())
    median_rel = float(np.median(rel_err_significant))
    print(f'  significant entries (|J_fd| > {fd_floor}): {int(significant.sum())}/{n_total}')
    print(f'  max rel err  (significant): {max_rel:.3e}')
    print(f'  median rel err (significant): {median_rel:.3e}')
    print(f'  max abs err  (any): {abs_err.max():.3e}')

    # Worst entry
    worst = np.unravel_index(np.argmax(rel_err * significant), rel_err.shape)
    print(f'  worst entry [{worst[0]:d}, {worst[1]:d}]: '
          f'J_fd={J_fd[worst]:+.3e}, J_jax={J_jax[worst]:+.3e}, '
          f'rel_err={rel_err[worst]:.3e}')

    # Pass criterion
    pass_threshold = 1e-2
    if max_rel < pass_threshold:
        print(f'\n✅ PASS: max rel err {max_rel:.2e} < {pass_threshold:.0e}')
        print('  Jacobian is suitable as a CVODE Newton preconditioner.')
        return 0
    else:
        print(f'\n❌ FAIL: max rel err {max_rel:.2e} >= {pass_threshold:.0e}')
        # Show top-5 worst significant entries
        rel_masked = rel_err * significant
        flat_top = np.argsort(rel_masked.ravel())[-5:][::-1]
        print('  Worst 5 significant entries:')
        for idx in flat_top:
            i, j = np.unravel_index(idx, rel_masked.shape)
            print(f'    [{i:3d}, {j:3d}] J_fd={J_fd[i,j]:+.3e} '
                  f'J_jax={J_jax[i,j]:+.3e} rel={rel_err[i,j]:.3e}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
