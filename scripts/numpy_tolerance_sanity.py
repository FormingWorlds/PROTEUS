"""Sanity check: is the numpy reference R_outer = 6.5056e+06 m converged?

The structure-parity probe established:
  V1 numpy (rtol=1e-5, atol=1e-6, default): R_outer = 6.5056e+06 m
  V2 jax  (rtol=1e-5, atol=1e-6, default): R_outer = 6.6555e+06 m  (+2.30%)

Tightening JAX's rtol/atol from 1e-5/1e-6 to 1e-9/1e-12 only saved 0.3%.
What if NUMPY isn't converged either? This probe sweeps numpy's
relative_tolerance / absolute_tolerance over the same range and reports
R_outer drift.

If numpy stays at ~6.5056e+06 across all tols, the +2.3% gap is real
(algorithmic difference between the two paths, not numpy convergence
artifact). If numpy drifts toward 6.66e+06 at tight tols, JAX may be
the more-accurate reference and numpy is mis-converged at default
1e-5/1e-6.
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
    'wall_timeout': 600.0,
    'use_jax': False,
    'use_anderson': False,
}

# Just two tols to see if numpy moves at all. Default vs much tighter.
SWEEP = [
    ('numpy_default',  1e-5, 1e-6),
    ('numpy_tight',    1e-9, 1e-12),
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

def chk(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()[:16]

cmb_index = int(np.argmax(mass_encl >= config_params["core_mass_fraction"] * mass_encl[-1]))

out = dict(
    t_solve=t_solve,
    converged=bool(result["converged"]),
    converged_mass=bool(result["converged_mass"]),
    mass_enclosed_last=float(mass_encl[-1]),
    radius_outer=float(radii[-1]),
    pressure_center=float(pressure[0]),
    density_center=float(density[0]),
    cmb_index=cmb_index,
    radii_chk=chk(radii),
    density_chk=chk(density),
    pressure_chk=chk(pressure),
    mass_encl_chk=chk(mass_encl),
)
print("__JSON__" + json.dumps(out))
'''


def run_variant(name, rtol, atol, r_arr, T_arr, timeout=900):
    cp = dict(BASE_CONFIG_PARAMS)
    cp['relative_tolerance'] = rtol
    cp['absolute_tolerance'] = atol
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
    print(f'Numpy-tol sanity sweep: r=[{r_arr[0]:.3e}, {r_arr[-1]:.3e}] m, '
          f'T=[{T_arr[0]:.0f}, {T_arr[-1]:.0f}] K')
    print(f'JAX (default tols) reference R_outer = {JAX_REFERENCE_R_OUTER:.6e} m')

    results = {}
    for name, rtol, atol in SWEEP:
        print(f'\n[{name}] rtol={rtol:.1e}, atol={atol:.1e}', flush=True)
        results[name] = run_variant(name, rtol, atol, r_arr, T_arr)
        r = results[name]
        if r:
            R = r['radius_outer']
            d_vs_jax = (R - JAX_REFERENCE_R_OUTER) / JAX_REFERENCE_R_OUTER * 100
            print(f'  R_outer={R:.6e}  Δ vs JAX={d_vs_jax:+.4f}%   '
                  f't={r["t_solve"]:.1f}s', flush=True)

    print('\n=== Numpy-tol sanity summary ===', flush=True)
    print(f'{"variant":<18} {"R_outer":>16} {"Δ vs JAX":>12} {"t_solve":>10}',
          flush=True)
    for name, _, _ in SWEEP:
        r = results.get(name, {})
        if not r:
            continue
        R = r['radius_outer']
        d = (R - JAX_REFERENCE_R_OUTER) / JAX_REFERENCE_R_OUTER * 100
        print(f'{name:<18} {R:16.6e} {d:11.4f}% {r["t_solve"]:10.1f}s',
              flush=True)


if __name__ == '__main__':
    main()
