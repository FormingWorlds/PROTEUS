"""Z.2 NaN-localisation for the multi-state Jacobian failure.

Iteratively turn off transport channels (conduction, convection,
grav_sep, mixing) in the JAX PhaseParams and re-run the Jacobian at
the 'mid' state from z04. The first combination that produces a
finite Jacobian identifies which physics branch carries the autodiff
NaN.

Pass criterion: report which channels can be ON without producing a
NaN Jacobian on the chili_repro_v2 mid state (S in [3300, 3700]).
"""

from __future__ import annotations

import sys
from pathlib import Path

ARAGOG_Z = Path('/Users/timlichtenberg/git/aragog-Z/src')
if str(ARAGOG_Z) not in sys.path:
    sys.path.insert(0, str(ARAGOG_Z))

import numpy as np  # noqa: E402


def main():
    print('=== Z.2 NaN-localisation (z05) ===')
    sys.path.insert(0, str(Path(__file__).parent))
    from z02_parity_multi_state import (  # noqa: E402
        build_solver_and_jax_args,
        synthesize_state,
    )

    from aragog.jax.phase import PhaseParams  # noqa: E402
    from aragog.solver.cvode_jax import build_jax_rhs_and_jacobian  # noqa: E402

    solver, args, eos_jax = build_solver_and_jax_args()
    n_stag = solver._n_stag
    state_dim = n_stag + 1
    eos_jax2, default_params, mesh_arrays, bc_jax, heating_jax = args

    state_mid = synthesize_state(solver, eos_jax, 'mid')

    # Try combinations: (conduction, convection, grav_sep, mixing)
    # Production has all True. We strip channels one at a time.
    combos = [
        # baseline (production CHILI: all on) — known to fail
        ('all on (production)', True, True, True, True),
        # strip mixing only
        ('mixing off', True, True, True, False),
        # strip grav_sep only
        ('grav_sep off', True, True, False, True),
        # strip both grav_sep and mixing (mass-flux off entirely)
        ('grav_sep+mixing off', True, True, False, False),
        # convection only (no conduction/grav/mix)
        ('convection only', False, True, False, False),
        # conduction only
        ('conduction only', True, False, False, False),
        # everything off (sanity: should give zero RHS, zero Jac)
        ('everything off', False, False, False, False),
    ]

    print(f'  state shape: {state_mid.shape}')
    print(f'  S range: [{state_mid[:n_stag].min():.1f}, {state_mid[:n_stag].max():.1f}]')

    for label, cond, conv, grav, mix in combos:
        # Rebuild PhaseParams with the requested channel mask. Other
        # fields copy from the default_params.
        params = PhaseParams(
            phi_rheo=default_params.phi_rheo,
            phi_width=default_params.phi_width,
            viscosity_solid=10.0 ** default_params.log10_visc_solid,
            viscosity_liquid=10.0 ** default_params.log10_visc_liquid,
            grain_size=default_params.grain_size,
            k_solid=default_params.k_solid,
            k_liquid=default_params.k_liquid,
            matprop_smooth_width=default_params.matprop_smooth_width,
            conduction=cond,
            convection=conv,
            grav_sep=grav,
            mixing=mix,
            eddy_diff_thermal=default_params.eddy_diff_thermal,
            eddy_diff_chemical=default_params.eddy_diff_chemical,
            kappah_floor=default_params.kappah_floor,
            bottom_up_grav_sep=bool(default_params.bottom_up_grav_sep),
        )

        rhs_fn, jac_fn, info = build_jax_rhs_and_jacobian(
            eos_jax2, params, mesh_arrays, bc_jax, np.asarray(heating_jax),
            np.ones(state_dim), np.ones(state_dim), 1.0,
            core_bc_mode='energy_balance',
        )

        # Forward sanity
        f0 = np.zeros(state_dim)
        rhs_fn(0.0, state_mid, f0)
        n_finite_f = int(np.isfinite(f0).sum())

        # Jacobian
        J = np.zeros((state_dim, state_dim))
        jac_fn(0.0, state_mid, None, J)
        n_finite_J = int(np.isfinite(J).sum())

        rhs_status = 'FINITE' if n_finite_f == state_dim else f'{state_dim - n_finite_f} NaN'
        jac_status = 'FINITE' if n_finite_J == J.size else f'{J.size - n_finite_J} NaN'
        print(f'  {label:>22}: rhs={rhs_status:>12} | jac={jac_status:>14}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
