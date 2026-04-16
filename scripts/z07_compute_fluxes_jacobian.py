"""Z.2 NaN-isolation step 2: Jacobian of compute_fluxes alone.

z06 showed compute_phase_state's Jacobian is finite at all 3 states.
This script tests the Jacobian of compute_fluxes (which calls
evaluate_phase + compute_mlt + the conduction/convection/grav/mix
heat-flux assembly).

If compute_fluxes is finite, the NaN must be in dSdt_energy_balance
itself (closure equation, flux divergence, or surface BC).

If compute_fluxes is NaN, narrow further by toggling MLT and the
mass-flux components.
"""

from __future__ import annotations

import sys
from pathlib import Path

ARAGOG_Z = Path('/Users/timlichtenberg/git/aragog-Z/src')
if str(ARAGOG_Z) not in sys.path:
    sys.path.insert(0, str(ARAGOG_Z))

import numpy as np  # noqa: E402


def main():
    print('=== Z.2 compute_fluxes Jacobian (z07) ===')
    sys.path.insert(0, str(Path(__file__).parent))
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from z02_parity_multi_state import (  # noqa: E402
        build_solver_and_jax_args,
        synthesize_state,
    )

    from aragog.jax.phase import compute_fluxes  # noqa: E402

    solver, args, eos_jax = build_solver_and_jax_args()
    n_stag = solver._n_stag
    eos_jax2, params, mesh, bc, heating = args

    for kind in ['IC', 'mid', 'near_solid']:
        state = synthesize_state(solver, eos_jax, kind)
        S_stag = jnp.asarray(state[:n_stag])
        S_basic_cmb_override = float(state[0])  # rough; just for compatibility
        dSdr_cmb_override = float(state[n_stag])

        print(f'\n  state={kind!r}, S range=[{float(S_stag.min()):.1f}, {float(S_stag.max()):.1f}]')

        def f(S_arg):
            flux = compute_fluxes(
                S_arg, 0.0, eos_jax2, params, mesh, heating,
                S_basic_cmb_override=S_basic_cmb_override,
                dSdr_cmb_override=dSdr_cmb_override,
            )
            return flux.heat_flux.sum()

        # Forward sanity
        flux = compute_fluxes(
            S_stag, 0.0, eos_jax2, params, mesh, heating,
            S_basic_cmb_override=S_basic_cmb_override,
            dSdr_cmb_override=dSdr_cmb_override,
        )
        hf = np.asarray(flux.heat_flux)
        print(f'    fwd heat_flux: finite={int(np.isfinite(hf).sum())}/{hf.size}, '
              f'range=[{hf.min():.3e}, {hf.max():.3e}]')

        # Backward
        grad = jax.grad(f)(S_stag)
        grad_np = np.asarray(grad)
        n_finite = int(np.isfinite(grad_np).sum())
        n_nan = int(np.isnan(grad_np).sum())
        print(f'    bwd grad(heat_flux.sum()): {"FINITE" if n_finite == grad_np.size else f"{n_nan} NaN"}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
