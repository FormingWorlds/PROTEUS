"""Test whether matching numpy's max_step cap closes the +2.3 % JAX gap.

Numpy ``solve_structure`` splits the radial integration into two
``solve_ivp`` calls. The second portion (upper 2 %, controlled by
``adaptive_radial_fraction = 0.98``) caps step size at
``maximum_step = r_est * 0.004 ≈ 28 km`` for the surface region.

The JAX path (``zalmoxis.jax_eos.solver._build_diffeqsolve_jit``) uses a
single ``diffrax.PIDController(rtol, atol)`` with no ``dtmax`` cap.

This probe monkey-patches ``diffrax.PIDController`` in a fresh subprocess
to inject ``dtmax`` and runs the smooth-T fixture from
``scripts/jax_numpy_structure_parity_smooth.py``. If R_outer drifts
toward numpy 6.5056e+06 m as dtmax shrinks, the split-step cap is the
algorithmic source of the +2 % delta.
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
JAX_DEFAULT_R_OUTER = 6.655470e6  # uncapped JAX

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

# dtmax in metres (Inf ≡ no cap, numpy ≈ 28 km, smaller = tighter)
SWEEP = [
    ('dtmax_inf',   None),       # baseline: no cap (matches default JAX)
    ('dtmax_280km', 280_000.0),
    ('dtmax_28km',  28_000.0),   # matches numpy upper-2% cap
    ('dtmax_2.8km', 2_800.0),
]


CHILD_TEMPLATE = r'''
from __future__ import annotations
import sys, os, json, hashlib, time

sys.path.insert(0, "{repo_root}/Zalmoxis/src")
sys.path.insert(0, "{repo_root}/src")

import numpy as np
DTMAX = {dtmax}

# Patch diffrax.PIDController to inject dtmax BEFORE Zalmoxis JIT-compiles.
import diffrax as _dx
_orig_PIDController = _dx.PIDController
if DTMAX is not None:
    def _patched_PIDController(*args, **kwargs):
        kwargs["dtmax"] = DTMAX
        return _orig_PIDController(*args, **kwargs)
    _dx.PIDController = _patched_PIDController
    sys.stderr.write(f"[child] Patched diffrax.PIDController with dtmax={{DTMAX}}\n")
else:
    sys.stderr.write(f"[child] dtmax=None (no cap)\n")

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


def run_variant(name, dtmax, r_arr, T_arr, timeout=900):
    code = CHILD_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        config_params=BASE_CONFIG_PARAMS,
        r_arr=list(r_arr),
        T_arr=list(T_arr),
        dtmax=repr(dtmax),
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
    print(f'dtmax sweep on smooth T(r): r=[{r_arr[0]:.3e}, {r_arr[-1]:.3e}] m')
    print(f'Numpy reference R_outer = {NUMPY_REFERENCE_R_OUTER:.6e} m')
    print(f'JAX default (no cap)    R_outer = {JAX_DEFAULT_R_OUTER:.6e} m')

    results = {}
    for name, dtmax in SWEEP:
        print(f'\n[{name}] dtmax={dtmax}', flush=True)
        results[name] = run_variant(name, dtmax, r_arr, T_arr)
        r = results[name]
        if r:
            R = r['radius_outer']
            d_np = (R - NUMPY_REFERENCE_R_OUTER) / NUMPY_REFERENCE_R_OUTER * 100
            d_jx = (R - JAX_DEFAULT_R_OUTER) / JAX_DEFAULT_R_OUTER * 100
            print(f'  R_outer={R:.6e}  Δ vs numpy={d_np:+.4f}%  Δ vs JAX={d_jx:+.4f}%  '
                  f't={r["t_solve"]:.1f}s', flush=True)

    print('\n=== dtmax sweep summary ===', flush=True)
    print(f'{"variant":<14} {"R_outer":>16} {"Δ vs numpy":>13} {"Δ vs JAX":>12} {"t_solve":>10}',
          flush=True)
    for name, _ in SWEEP:
        r = results.get(name, {})
        if not r:
            continue
        R = r['radius_outer']
        d_np = (R - NUMPY_REFERENCE_R_OUTER) / NUMPY_REFERENCE_R_OUTER * 100
        d_jx = (R - JAX_DEFAULT_R_OUTER) / JAX_DEFAULT_R_OUTER * 100
        print(f'{name:<14} {R:16.6e} {d_np:12.4f}% {d_jx:11.4f}% {r["t_solve"]:10.1f}s',
              flush=True)


if __name__ == '__main__':
    main()
