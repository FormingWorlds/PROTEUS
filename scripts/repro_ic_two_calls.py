"""Reproducer that mimics the *sequence* of two PROTEUS IC Zalmoxis calls.

Live runs (PROTEUS log lines):
  IC call 1 (no volatiles, planet_mass=5.972e24):   bit-identical between A and B
                                                     -> mass_enclosed[-1] = 5.964640840320699e+24
  IC call 2 (volatiles subtracted, 5.971257e24):   DIVERGES
                                                     -> A: 5.96969819543402e+24
                                                     -> B: 5.96246548502287e+24  (delta -0.121 %)

PROTEUS passes `initial_density`/`initial_radii` from a module-level
`_density_cache` populated after each successful call. Call 2 is seeded with
call 1's result.

This script runs THREE variants in fresh subprocesses, each doing both calls
back-to-back (with cache seeding), and checks whether the live IC delta can
be reproduced standalone:

  V1 'A_post': use_jax=False, use_anderson=False, no wall_timeout key
  V2 'B_post': use_jax=False, use_anderson=False, wall_timeout=3600
  V3 'A_post_jax_import': V1 + force `import jax` BEFORE Zalmoxis runs

If V1 == V2: wall_timeout dict-presence is not the cause.
If V1 == V3: JAX module-import side effects are not the cause.
If V1 == V2 == V3 yet differ from the live A run, the cause is upstream of
Zalmoxis (e.g. CALLIOPE outputs, volatile_profile construction).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path('/Users/timlichtenberg/git/PROTEUS')

CALL1_PLANET_MASS = 5.972e24
CALL2_PLANET_MASS = 5.971257000000001e24

BASE_CONFIG_PARAMS = {
    'core_mass_fraction': 0.325,
    'core_frac_mode': 'mass',
    'mantle_mass_fraction': 0,
    'temperature_mode': 'adiabatic',
    'surface_temperature': 3830.0,
    'cmb_temperature': 7199.0,
    'center_temperature': 6000.0,
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
}

VARIANTS = {
    'A_post': {
        'use_jax': False,
        'use_anderson': False,
    },
    'B_post': {
        'use_jax': False,
        'use_anderson': False,
        'wall_timeout': 3600.0,
    },
    'A_post_jax_import': {
        'use_jax': False,
        'use_anderson': False,
        '_force_import_jax': True,
    },
}

CHILD_TEMPLATE = r'''
from __future__ import annotations
import sys, os, json, hashlib

sys.path.insert(0, "{repo_root}/Zalmoxis/src")
sys.path.insert(0, "{repo_root}/src")

config_params = {config_params!r}
call1_mass = {call1_mass!r}
call2_mass = {call2_mass!r}

_force_import_jax = config_params.pop("_force_import_jax", False)
if _force_import_jax:
    import jax  # noqa: F401
    import jaxlib  # noqa: F401
    sys.stderr.write("[child] JAX imported\n")

from proteus.interior_struct.zalmoxis import load_zalmoxis_material_dictionaries
from zalmoxis.melting_curves import get_solidus_liquidus_functions
from zalmoxis.solver import main as zal_main

mantle_eos = "PALEOS-2phase:MgSiO3"
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

import numpy as np
import time

def chk(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()[:16]

# --- Call 1: cold (no cache seeding) ---
sys.stderr.write("[child] CALL1 starting\n")
t0 = time.time()
config_params1 = dict(config_params)
config_params1["planet_mass"] = call1_mass
result1 = zal_main(
    config_params1,
    material_dictionaries=mat_dicts,
    melting_curves_functions=melt_funcs,
    input_dir=input_dir,
    layer_mixtures=None,
    volatile_profile=None,
    temperature_function=None,
    temperature_arrays=None,
    p_center_hint=None,
    initial_density=None,
    initial_radii=None,
)
t_call1 = time.time() - t0
sys.stderr.write(f"[child] CALL1 done in {{t_call1:.1f}}s\n")
density1 = np.asarray(result1["density"]).copy()
radii1 = np.asarray(result1["radii"]).copy()
mass1 = np.asarray(result1["mass_enclosed"])

# --- Call 2: seeded with call-1 cache ---
sys.stderr.write("[child] CALL2 starting\n")
t0 = time.time()
config_params2 = dict(config_params)
config_params2["planet_mass"] = call2_mass
result2 = zal_main(
    config_params2,
    material_dictionaries=mat_dicts,
    melting_curves_functions=melt_funcs,
    input_dir=input_dir,
    layer_mixtures=None,
    volatile_profile=None,
    temperature_function=None,
    temperature_arrays=None,
    p_center_hint=None,
    initial_density=density1,
    initial_radii=radii1,
)
t_call2 = time.time() - t0
sys.stderr.write(f"[child] CALL2 done in {{t_call2:.1f}}s\n")
density2 = np.asarray(result2["density"])
radii2 = np.asarray(result2["radii"])
mass2 = np.asarray(result2["mass_enclosed"])

out = dict(
    t_call1=t_call1,
    t_call2=t_call2,
    call1_mass_last=float(mass1[-1]),
    call1_radius_outer=float(radii1[-1]),
    call1_density_chk=chk(density1),
    call1_radii_chk=chk(radii1),
    call1_converged=bool(result1["converged"]),
    call1_n_layers=int(len(radii1)),
    call2_mass_last=float(mass2[-1]),
    call2_radius_outer=float(radii2[-1]),
    call2_density_chk=chk(density2),
    call2_radii_chk=chk(radii2),
    call2_mass_encl_chk=chk(mass2),
    call2_converged=bool(result2["converged"]),
    call2_n_layers=int(len(radii2)),
)
print("__JSON__" + json.dumps(out), flush=True)
'''


def run_variant(name: str, overrides: dict, log_path: str) -> dict:
    config_params = dict(BASE_CONFIG_PARAMS)
    config_params.update(overrides)

    code = CHILD_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        config_params=config_params,
        call1_mass=CALL1_PLANET_MASS,
        call2_mass=CALL2_PLANET_MASS,
    )

    with open(log_path, 'a') as f:
        f.write(f'\n=== {name} overrides={overrides} ===\n')
        f.flush()

    proc = subprocess.run(
        [sys.executable, '-u', '-c', code],
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=900,
    )

    with open(log_path, 'a') as f:
        f.write('--- stdout ---\n')
        f.write(proc.stdout)
        f.write('--- stderr ---\n')
        f.write(proc.stderr)
        f.write(f'--- returncode={proc.returncode} ---\n')
        f.flush()

    if proc.returncode != 0:
        return {}
    for line in proc.stdout.splitlines():
        if line.startswith('__JSON__'):
            return json.loads(line[len('__JSON__'):])
    return {}


def fmt(v):
    if isinstance(v, float):
        return f'{v:.10e}'
    if isinstance(v, bool):
        return 'T' if v else 'F'
    return str(v)


def main():
    log_path = '/tmp/repro_ic_two_calls.log'
    open(log_path, 'w').close()

    results = {}
    for name, overrides in VARIANTS.items():
        print(f'[{name}] starting overrides={overrides}', flush=True)
        results[name] = run_variant(name, overrides, log_path)
        r = results[name]
        if r:
            print(
                f'[{name}] done '
                f'(t1={r.get("t_call1"):.1f}s, t2={r.get("t_call2"):.1f}s) '
                f'call2_mass={r.get("call2_mass_last"):.10e}',
                flush=True,
            )
        else:
            print(f'[{name}] FAILED — see {log_path}', flush=True)

    keys = (
        't_call1', 't_call2',
        'call1_mass_last', 'call1_density_chk',
        'call2_mass_last', 'call2_radius_outer',
        'call2_density_chk', 'call2_radii_chk', 'call2_mass_encl_chk',
        'call2_n_layers', 'call2_converged',
    )
    print('\n=== Summary ===', flush=True)
    print(f'{"key":<22}', '  '.join(f'{n:>22}' for n in results), flush=True)
    for k in keys:
        row = '  '.join(f'{fmt(results[n].get(k)):>22}' for n in results)
        print(f'{k:<22}', row, flush=True)

    # Pairwise checksums
    base = 'A_post'
    chks_keys = ('call2_density_chk', 'call2_radii_chk', 'call2_mass_encl_chk')
    print(f'\n=== Pairwise checksum equality vs {base} ===', flush=True)
    for n in results:
        if n == base:
            continue
        same = all(results[n].get(k) == results[base].get(k) for k in chks_keys)
        print(f'  {n:<24}: {"IDENTICAL" if same else "DIFFER"}', flush=True)


if __name__ == '__main__':
    main()
