"""Z.2 NaN-isolation: Jacobian of compute_phase_state alone.

If compute_phase_state's autodiff backward produces NaN at the mid
state, the bug is in the EOS layer (compute_phase_state) and not in
the downstream RHS / closure equation. Conversely, if compute_phase_state
gives a finite Jacobian at the mid state, the bug must be in the
flux pipeline or the closure equation.

We compute jacrev of a scalar reduction of compute_phase_state's
density at three states: IC (extrapolated above table), mid (in-table
mushy), near_solid (in-table mostly-solid).
"""

from __future__ import annotations

import sys
from pathlib import Path

ARAGOG_Z = Path('/Users/timlichtenberg/git/aragog-Z/src')
if str(ARAGOG_Z) not in sys.path:
    sys.path.insert(0, str(ARAGOG_Z))

import numpy as np  # noqa: E402


def main():
    print('=== Z.2 phase-state-only Jacobian (z06) ===')
    sys.path.insert(0, str(Path(__file__).parent))

    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402

    from aragog.jax.eos import EntropyEOS_JAX  # noqa: E402

    EOS_DIR = Path(
        '/Users/timlichtenberg/work/fwl_data/interior_lookup_tables/'
        '1TPa-dK09-elec-free/MgSiO3_Wolf_Bower_2018_1TPa'
    )
    eos = EntropyEOS_JAX(EOS_DIR)

    # Three test states (uniform per state)
    P = jnp.linspace(120e9, 1e9, 80)  # Earth-like P profile
    states = {
        'IC':         jnp.full(80, 3900.0),  # extrapolated above table top
        'mid':        jnp.linspace(3300.0, 3700.0, 80),
        'near_solid': jnp.linspace(3000.0, 3300.0, 80),
    }

    print('--- Jacobian of compute_phase_state outputs w.r.t. S ---')
    for kind, S in states.items():
        print(f'\n  state={kind!r}, S range=[{float(S.min()):.1f}, {float(S.max()):.1f}]')

        # Forward sanity
        state = eos.compute_phase_state(
            P, S, k_solid=4.0, k_liquid=2.0, matprop_smooth_width=0.01,
        )
        for name, val in [
            ('temperature', state.temperature),
            ('density', state.density),
            ('heat_capacity', state.heat_capacity),
            ('thermal_expansivity', state.thermal_expansivity),
            ('dTdPs', state.dTdPs),
        ]:
            v = np.asarray(val)
            n_finite = int(np.isfinite(v).sum())
            print(f'    fwd {name:>22}: shape={v.shape}, finite={n_finite}/{v.size}, '
                  f'range=[{v.min():.3e}, {v.max():.3e}]')

        # Jacobian of each property summed (scalar) wrt S
        for prop in ['temperature', 'density', 'heat_capacity',
                     'thermal_expansivity', 'dTdPs']:
            def f(S_arg, p=prop):
                state = eos.compute_phase_state(
                    P, S_arg, k_solid=4.0, k_liquid=2.0, matprop_smooth_width=0.01,
                )
                return getattr(state, p).sum()

            grad = jax.grad(f)(S)
            grad_np = np.asarray(grad)
            n_finite = int(np.isfinite(grad_np).sum())
            n_nan = int(np.isnan(grad_np).sum())
            status = 'FINITE' if n_finite == grad_np.size else f'{n_nan} NaN'
            print(f'    bwd grad({prop:>20}).sum(): {status}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
