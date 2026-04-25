"""Test whether the numpy inner Picard density-iteration tolerance
floor is the algorithmic source of the +2.3 % JAX-vs-numpy R_outer gap.

After EOS density / Anderson Picard / Tsit5-tol / Newton-tol / dtmax
cap / numpy-rtol/atol / Tsit5-vs-Dopri5-vs-Heun all closed at ≤0.34 %
share, the remaining suspect is the architectural difference between
the two paths:

* numpy ``solve_structure`` runs an INNER Picard iteration where
  the structure ODE uses an INTERPOLATED density profile, then density
  is recomputed from the integrated P, looping until self-consistent
  with ``tolerance_inner=1e-4``. The structure ODE never sees the
  fully self-consistent (P, ρ) state, only a Picard-converged
  approximation.

* JAX ``solve_structure_via_jax`` evaluates density from the EOS in
  real-time inside the diffrax RHS, so the structure ODE is FULLY
  self-consistent (no Picard floor).

This probe sweeps numpy's ``tolerance_inner`` from 1e-4 (default) down
to 1e-9. If R_outer drifts from numpy 6.5056e+06 m toward JAX
6.6555e+06 m as tolerance_inner shrinks, the inner Picard floor is the
algorithmic source.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path('/Users/timlichtenberg/git/PROTEUS')

R_CMB = 3.378e6
R_SURFACE = 6.871e6
T_CMB = 4065.0
T_SURF = 2800.0
N_PROFILE = 79
JAX_REFERENCE_R_OUTER = 6.655470e6  # V2 jax default
NUMPY_REFERENCE_R_OUTER = 6.505582e6  # V1 numpy default

BASE_CONFIG_PARAMS = {
    'planet_mass': 5.971257000000001e24,
    'core_mass_fraction': 0.325,
    'core_frac_mode': 'mass',
    'mantle_mass_fraction': 0,
    'temperature_mode': 'adiabatic',
    'surface_temperature': T_SURF,
    'cmb_temperature': T_CMB,
    'center_temperature': T_CMB + 500.0,
    'temp_profile_file': None,
    'layer_eos_config': {
        'core': 'PALEOS:iron',
        'mantle': 'PALEOS-2phase:MgSiO3',
    },
    'mushy_zone_factor': 0.8,
    'mushy_zone_factors': {
        'PALEOS:iron': 0.8,
        'PALEOS:MgSiO3': 0.8,
        'PALEOS:H2O': 0.8,
    },
    'num_layers': 150,
    'target_surface_pressure': 101325.0,
    'tolerance_outer': 3e-3,
    'max_iterations_outer': 200,
    'max_iterations_inner': 200,
    'wall_timeout': 1800.0,
    'use_jax': False,
    'use_anderson': False,
    'relative_tolerance': 1e-5,
    'absolute_tolerance': 1e-6,
}

# Inner Picard tolerance sweep. 1e-4 = default; smaller = tighter.
SWEEP = [
    ('inner_1e-4',  1e-4),  # default
    ('inner_1e-6',  1e-6),
    ('inner_1e-9',  1e-9),
]


CHILD_TEMPLATE = r'''
from __future__ import annotations
import sys, os, json, hashlib, time

sys.path.insert(0, "{repo_root}/Zalmoxis/src")
sys.path.insert(0, "{repo_root}/src")

import numpy as np

config_params = {config_params!r}
r_arr = np.array({r_arr!r})
T_arr = np.array({T_arr!r})

# numpy path uses temperature_function callable
def temperature_function(r, P, _r=r_arr, _T=T_arr):
    return float(np.interp(float(r), _r, _T))

from proteus.interior_struct.zalmoxis import load_zalmoxis_material_dictionaries
from zalmoxis.melting_curves import get_solidus_liquidus_functions
from zalmoxis.solver import main as zal_main

mzf = 0.8
_, liquidus_func = get_solidus_liquidus_functions(
    solidus_id="Stixrude14-solidus",
    liquidus_id="PALEOS-liquidus",
)
def _solidus_func(P, _l=liquidus_func, _m=mzf):
    return _l(P) * _m
melt_funcs = (_solidus_func, liquidus_func)
mat_dicts = load_zalmoxis_material_dictionaries()
input_dir = os.path.join(
    os.environ.get("FWL_DATA", "/Users/timlichtenberg/git/FWL_DATA"),
    "zalmoxis_eos",
)

t0 = time.time()
result = zal_main(
    config_params,
    material_dictionaries=mat_dicts,
    melting_curves_functions=melt_funcs,
    input_dir=input_dir,
    layer_mixtures=None,
    volatile_profile=None,
    temperature_function=temperature_function,
    temperature_arrays=None,
    p_center_hint=None,
    initial_density=None,
    initial_radii=None,
)
t_solve = time.time() - t0

radii = np.asarray(result["radii"])
mass_encl = np.asarray(result["mass_enclosed"])
density = np.asarray(result["density"])
pressure = np.asarray(result["pressure"])

cmb_index = int(np.argmax(mass_encl >= config_params["core_mass_fraction"] * mass_encl[-1]))

out = dict(
    t_solve=t_solve,
    converged=bool(result["converged"]),
    converged_mass=bool(result["converged_mass"]),
    mass_enclosed_last=float(mass_encl[-1]),
    radius_outer=float(radii[-1]),
    pressure_center=float(pressure[0]),
    cmb_index=cmb_index,
    n_outer_iter=int(result.get("n_outer_iter", -1)),
    n_inner_iter_last=int(result.get("n_inner_iter", -1)),
)
print("__JSON__" + json.dumps(out))
'''


def run_variant(name, tolerance_inner, r_arr, T_arr, timeout=2400):
    cp = dict(BASE_CONFIG_PARAMS)
    cp['tolerance_inner'] = tolerance_inner
    code = CHILD_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        config_params=cp,
        r_arr=list(r_arr),
        T_arr=list(T_arr),
    )
    proc = subprocess.run(
        [sys.executable, '-u', '-c', code],
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=timeout,
    )
    if proc.returncode != 0:
        print(f'[FAIL] {name}: returncode={proc.returncode}')
        print('\n'.join(proc.stderr.splitlines()[-20:]))
        return {}
    for line in proc.stdout.splitlines():
        if line.startswith('__JSON__'):
            return json.loads(line[len('__JSON__'):])
    return {}


def main():
    r_arr = np.linspace(R_CMB, R_SURFACE, N_PROFILE)
    T_arr = np.linspace(T_CMB, T_SURF, N_PROFILE)
    print('Numpy inner-Picard tolerance sweep on smooth T(r)')
    print(f'Numpy reference (default) R_outer = {NUMPY_REFERENCE_R_OUTER:.6e} m')
    print(f'JAX (default tols)        R_outer = {JAX_REFERENCE_R_OUTER:.6e} m')
    print(f'Gap to close: +{(JAX_REFERENCE_R_OUTER - NUMPY_REFERENCE_R_OUTER)/NUMPY_REFERENCE_R_OUTER*100:.4f}%')

    results = {}
    for name, tol in SWEEP:
        print(f'\n[{name}] tolerance_inner={tol:.1e}', flush=True)
        results[name] = run_variant(name, tol, r_arr, T_arr)
        r = results[name]
        if r:
            R = r['radius_outer']
            d_np = (R - NUMPY_REFERENCE_R_OUTER) / NUMPY_REFERENCE_R_OUTER * 100
            d_jx = (R - JAX_REFERENCE_R_OUTER) / JAX_REFERENCE_R_OUTER * 100
            print(f'  R_outer={R:.6e}  Δ vs numpy_default={d_np:+.4f}%  Δ vs JAX={d_jx:+.4f}%  '
                  f'converged={r["converged"]} t={r["t_solve"]:.1f}s', flush=True)

    print('\n=== inner-Picard tolerance sweep summary ===', flush=True)
    print(f'{"variant":<14} {"R_outer":>16} {"Δ vs numpy":>14} {"Δ vs JAX":>13} {"t_solve":>10}',
          flush=True)
    for name, _ in SWEEP:
        r = results.get(name, {})
        if not r:
            continue
        R = r['radius_outer']
        d_np = (R - NUMPY_REFERENCE_R_OUTER) / NUMPY_REFERENCE_R_OUTER * 100
        d_jx = (R - JAX_REFERENCE_R_OUTER) / JAX_REFERENCE_R_OUTER * 100
        print(f'{name:<14} {R:16.6e} {d_np:13.4f}% {d_jx:12.4f}% {r["t_solve"]:10.1f}s',
              flush=True)


if __name__ == '__main__':
    main()
