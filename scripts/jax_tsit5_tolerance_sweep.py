"""Localise the +2.3 % JAX-vs-numpy R_outer delta to Tsit5 step accuracy.

Companion to ``scripts/jax_numpy_structure_parity_smooth.py``: with all
other variables held fixed (smooth linear T(r), JAX path with Anderson
disabled, fresh subprocess per variant), sweep the diffrax PIDController
tolerances ``relative_tolerance``/``absolute_tolerance`` from the current
default (1e-5/1e-6) down to (1e-9/1e-12).

If R_outer converges toward the numpy reference (6.5056 Mm) as Tsit5
tolerances tighten, **the +2.3 % delta is in the integrator step size**.
If R_outer is insensitive to Tsit5 tolerances, the gap is in the
``optimistix.Newton`` Event termination (rtol=1e-6, atol=1e-6 hardcoded
in ``zalmoxis/jax_eos/solver.py``) or in a structural choice the probe
can't reach.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path('/Users/timlichtenberg/git/PROTEUS')
OUT_DIR = REPO_ROOT / 'output_files' / 'jax_tsit5_tolerance_sweep'
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_CMB = 3.378e6
R_SURFACE = 6.871e6
T_CMB = 4065.0
T_SURF = 2800.0
N_PROFILE = 79
NUMPY_REFERENCE_R_OUTER = 6.505582e6  # m, from V1_numpy in the smooth probe

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
    'use_anderson': False,   # ruled out as the gap source
}

# (rtol, atol) pairs to sweep
SWEEP = [
    ('rtol1e-3_atol1e-4',  1e-3, 1e-4),   # looser than default
    ('rtol1e-5_atol1e-6',  1e-5, 1e-6),   # current PROTEUS default
    ('rtol1e-7_atol1e-9',  1e-7, 1e-9),
    ('rtol1e-9_atol1e-12', 1e-9, 1e-12),  # very tight
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


def run_variant(name, rtol, atol, r_arr, T_arr, timeout=600):
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
    print(f'[FAIL] {name}: no __JSON__')
    return {}


def main():
    r_arr = np.linspace(R_CMB, R_SURFACE, N_PROFILE)
    T_arr = np.linspace(T_CMB, T_SURF, N_PROFILE)
    print(f'Sweep input: r=[{r_arr[0]:.3e}, {r_arr[-1]:.3e}] m, '
          f'T=[{T_arr[0]:.0f}, {T_arr[-1]:.0f}] K')
    print(f'Numpy reference R_outer = {NUMPY_REFERENCE_R_OUTER:.6e} m')

    results = {}
    for name, rtol, atol in SWEEP:
        print(f'\n[{name}] rtol={rtol:.1e}, atol={atol:.1e}', flush=True)
        results[name] = run_variant(name, rtol, atol, r_arr, T_arr)
        r = results[name]
        if r:
            R = r['radius_outer']
            d_vs_numpy = (R - NUMPY_REFERENCE_R_OUTER) / NUMPY_REFERENCE_R_OUTER * 100
            print(f'  R_outer={R:.6e}  Δ vs numpy={d_vs_numpy:+.4f}%   '
                  f't={r["t_solve"]:.1f}s', flush=True)

    print('\n=== Sweep summary ===', flush=True)
    print(f'{"variant":<22} {"R_outer":>16} {"Δ vs numpy":>14} {"t_solve":>10}',
          flush=True)
    for name, _, _ in SWEEP:
        r = results.get(name, {})
        if not r:
            continue
        R = r['radius_outer']
        d = (R - NUMPY_REFERENCE_R_OUTER) / NUMPY_REFERENCE_R_OUTER * 100
        print(f'{name:<22} {R:16.6e} {d:13.4f}% {r["t_solve"]:10.1f}s',
              flush=True)


if __name__ == '__main__':
    main()
