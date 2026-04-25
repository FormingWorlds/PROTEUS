"""Test whether swapping diffrax.Tsit5 for diffrax.Dopri5 closes the
+2 % residual JAX-vs-numpy R_outer gap.

After dtmax / Tsit5-tol / Newton-tol / numpy-tol sweeps all closed at
≤0.34 % share of the +2.30 % gap, the remaining suspect is the
Runge-Kutta tableau. scipy.solve_ivp(method='RK45') uses Dormand-Prince
5(4) (Dopri5). diffrax defaults Zalmoxis to ``Tsit5()`` (Tsitouras 5(4)).
Both are 5(4) PI-controlled but use different coefficient tableaus and
FSAL ordering.

This probe monkey-patches ``diffrax.Tsit5`` -> ``diffrax.Dopri5`` in a
fresh subprocess so ``_build_diffeqsolve_jit`` builds with Dopri5 without
touching Zalmoxis source. Default tolerances 1e-5/1e-6 held fixed.

If R_outer drifts from JAX 6.6555e+06 toward numpy 6.5056e+06 with Dopri5,
the tableau choice is the algorithmic source of the residual ~2 % gap.
If R_outer stays put (matches Tsit5), the residual lives in the Event
interpolant placement, not the integrator algebra.
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
NUMPY_REFERENCE_R_OUTER = 6.505582e6
JAX_DEFAULT_R_OUTER = 6.655470e6  # uncapped Tsit5 default

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
    'tolerance_inner': 1e-4,
    'max_iterations_outer': 100,
    'max_iterations_inner': 100,
    'wall_timeout': 300.0,
    'use_jax': True,
    'use_anderson': False,
    'relative_tolerance': 1e-5,
    'absolute_tolerance': 1e-6,
}

# Solver variants. None = unmodified Tsit5 baseline.
SWEEP = [
    ('tsit5_baseline', 'Tsit5'),
    ('dopri5',         'Dopri5'),
    # Heun (explicit 2nd-order) is a coarse-tableau sanity probe: if even
    # this gives the same R_outer to within tolerance, the integrator
    # algebra is not the cause.
    ('heun_sanity',    'Heun'),
]


CHILD_TEMPLATE = r'''
from __future__ import annotations
import sys, os, json, hashlib, time

sys.path.insert(0, "{repo_root}/Zalmoxis/src")
sys.path.insert(0, "{repo_root}/src")

import numpy as np
SOLVER_CLASS = "{solver_class}"

# Patch diffrax.Tsit5 BEFORE Zalmoxis JIT-compiles so the swapped solver
# is closed over inside _build_diffeqsolve_jit().
import diffrax as _dx
_orig_Tsit5 = _dx.Tsit5
if SOLVER_CLASS != "Tsit5":
    _replacement = getattr(_dx, SOLVER_CLASS)
    _dx.Tsit5 = _replacement
    sys.stderr.write(f"[child] Patched diffrax.Tsit5 -> {{SOLVER_CLASS}}\n")
else:
    sys.stderr.write(f"[child] Tsit5 baseline (no patch)\n")

config_params = {config_params!r}
r_arr = np.array({r_arr!r})
T_arr = np.array({T_arr!r})

temperature_arrays = (np.ascontiguousarray(r_arr), np.ascontiguousarray(T_arr))

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
    temperature_function=None,
    temperature_arrays=temperature_arrays,
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
)
print("__JSON__" + json.dumps(out))
'''


def run_variant(name, solver_class, r_arr, T_arr, timeout=900):
    code = CHILD_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        config_params=BASE_CONFIG_PARAMS,
        r_arr=list(r_arr),
        T_arr=list(T_arr),
        solver_class=solver_class,
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
    print(f'Solver-tableau sweep on smooth T(r): r=[{r_arr[0]:.3e}, {r_arr[-1]:.3e}] m')
    print(f'Numpy reference R_outer = {NUMPY_REFERENCE_R_OUTER:.6e} m')
    print(f'JAX (Tsit5 default)    R_outer = {JAX_DEFAULT_R_OUTER:.6e} m')

    results = {}
    for name, solver_class in SWEEP:
        print(f'\n[{name}] solver={solver_class}', flush=True)
        results[name] = run_variant(name, solver_class, r_arr, T_arr)
        r = results[name]
        if r:
            R = r['radius_outer']
            d_np = (R - NUMPY_REFERENCE_R_OUTER) / NUMPY_REFERENCE_R_OUTER * 100
            d_jx = (R - JAX_DEFAULT_R_OUTER) / JAX_DEFAULT_R_OUTER * 100
            print(f'  R_outer={R:.6e}  Δ vs numpy={d_np:+.4f}%  Δ vs Tsit5={d_jx:+.4f}%  '
                  f't={r["t_solve"]:.1f}s', flush=True)

    print('\n=== solver-tableau sweep summary ===', flush=True)
    print(f'{"variant":<18} {"R_outer":>16} {"Δ vs numpy":>13} {"Δ vs Tsit5":>13} {"t_solve":>10}',
          flush=True)
    for name, _ in SWEEP:
        r = results.get(name, {})
        if not r:
            continue
        R = r['radius_outer']
        d_np = (R - NUMPY_REFERENCE_R_OUTER) / NUMPY_REFERENCE_R_OUTER * 100
        d_jx = (R - JAX_DEFAULT_R_OUTER) / JAX_DEFAULT_R_OUTER * 100
        print(f'{name:<18} {R:16.6e} {d_np:12.4f}% {d_jx:12.4f}% {r["t_solve"]:10.1f}s',
              flush=True)


if __name__ == '__main__':
    main()
