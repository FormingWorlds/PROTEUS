"""Z.2 NaN-isolation step 3: compute_mlt + relative_velocity Jacobians.

z07 confirmed compute_fluxes NaNs at mid/near_solid while
compute_phase_state (z06) is finite. The bug must be in code between
them: compute_mlt or relative_velocity (always called regardless of
transport flags).

This script computes the Jacobian of each in isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ARAGOG_Z = Path('/Users/timlichtenberg/git/aragog-Z/src')
if str(ARAGOG_Z) not in sys.path:
    sys.path.insert(0, str(ARAGOG_Z))

import numpy as np  # noqa: E402


def main():
    print('=== Z.2 compute_mlt + relative_velocity Jacobians (z08) ===')
    sys.path.insert(0, str(Path(__file__).parent))
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from z02_parity_multi_state import (  # noqa: E402
        build_solver_and_jax_args,
        synthesize_state,
    )

    from aragog.jax.phase import (  # noqa: E402
        compute_mlt,
        evaluate_phase,
        relative_velocity,
    )

    solver, args, eos_jax = build_solver_and_jax_args()
    n_stag = solver._n_stag
    eos_jax2, params, mesh, bc, heating = args

    for kind in ['IC', 'mid', 'near_solid']:
        state = synthesize_state(solver, eos_jax, kind)
        S_stag = jnp.asarray(state[:n_stag])
        S_basic = mesh.quantity_matrix @ S_stag
        S_basic = S_basic.at[0].set(state[0])  # rough
        dSdr = mesh.d_dr_matrix @ S_stag
        dSdr = dSdr.at[0].set(state[n_stag])
        dSdr = dSdr.at[-1].set(dSdr[-2])

        print(f'\n  state={kind!r}, S_basic range=[{float(S_basic.min()):.1f}, {float(S_basic.max()):.1f}]')

        # ── compute_mlt isolated ──
        def f_mlt(S_basic_arg):
            ph = evaluate_phase(eos_jax2, params, mesh.P_basic, S_basic_arg)
            kh, _ = compute_mlt(dSdr, ph, mesh, params)
            return kh.sum()

        kh_grad = jax.grad(f_mlt)(S_basic)
        kh_np = np.asarray(kh_grad)
        n_nan = int(np.isnan(kh_np).sum())
        print(f'    bwd grad(compute_mlt_kh.sum()) wrt S_basic: '
              f'{"FINITE" if n_nan == 0 else f"{n_nan} NaN"}')

        # ── relative_velocity isolated ──
        ph_b = evaluate_phase(eos_jax2, params, mesh.P_basic, S_basic)
        rho = ph_b.density
        phi_b = ph_b.melt_fraction

        def f_rv(rho_arg):
            v_rel = relative_velocity(
                eos_jax2, params, mesh.P_basic, rho_arg, phi_b, mesh.gravity,
            )
            return v_rel.sum()

        rv_grad = jax.grad(f_rv)(rho)
        rv_np = np.asarray(rv_grad)
        n_nan = int(np.isnan(rv_np).sum())
        print(f'    bwd grad(relative_velocity.sum()) wrt rho: '
              f'{"FINITE" if n_nan == 0 else f"{n_nan} NaN"}')

        # ── relative_velocity wrt phi ──
        def f_rv_phi(phi_arg):
            v_rel = relative_velocity(
                eos_jax2, params, mesh.P_basic, rho, phi_arg, mesh.gravity,
            )
            return v_rel.sum()

        rv_phi_grad = jax.grad(f_rv_phi)(phi_b)
        rv_phi_np = np.asarray(rv_phi_grad)
        n_nan = int(np.isnan(rv_phi_np).sum())
        print(f'    bwd grad(relative_velocity.sum()) wrt phi: '
              f'{"FINITE" if n_nan == 0 else f"{n_nan} NaN"}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
