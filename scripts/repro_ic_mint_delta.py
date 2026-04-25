"""Standalone reproducer for the iter-0 M_int delta between A-side
(stage1b_postrevert_baseline) and B-side (stage2_ab_jaxanderson) CHILI runs.

At IC the PROTEUS gate downgrades use_jax/use_anderson to False on the B-side
but it has already set wall_timeout=3600 on the dict. The A-side config_params
has none of those keys at all. Both end up taking the numpy path, but the
dicts differ. The first IC call (no volatiles, planet_mass=5.972e24) gives
bit-identical Zalmoxis output in both runs; the second call (with volatiles
subtracted, planet_mass=5.971257e24) does NOT match. This script reproduces
the second-call inputs and varies config_params keys.

Each variant runs in a fresh subprocess to clear Zalmoxis module-level state
and JIT caches.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path('/Users/timlichtenberg/git/PROTEUS')

# CHILI Earth IC config_params, second equilibration call:
#   total mass = 5.972e+24 kg, volatile_mass = 7.43e+20 kg
#   planet_mass = 5.971257e+24 kg
BASE_CONFIG_PARAMS = {
    'planet_mass': 5.971257000000001e24,
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
    # PROTEUS Zalmoxis defaults (src/proteus/config/_struct.py)
    'tolerance_outer': 3e-3,
    'tolerance_inner': 1e-4,
    'max_iterations_outer': 100,
    'max_iterations_inner': 100,
}

VARIANTS = {
    # A-side: PROTEUS never adds use_jax/use_anderson/wall_timeout keys
    # (matches stage1b_postrevert_baseline path: config.interior_struct.
    # zalmoxis.use_jax = False default, but PROTEUS load_zalmoxis_configuration
    # ALWAYS injects use_jax / use_anderson into config_params from the
    # config attrs class even on A-side — see _struct.py and
    # interior_struct/zalmoxis.py:296-340. So for "A-side" the keys ARE
    # present, just both False, and wall_timeout is absent.)
    'A_no_walltimeout': {
        'use_jax': False,
        'use_anderson': False,
    },
    # B-side post-gate: gate downgraded use_jax/use_anderson to False but
    # had already set wall_timeout=3600.
    'B_post_gate': {
        'use_jax': False,
        'use_anderson': False,
        'wall_timeout': 3600.0,
    },
    # B-side pre-gate: what PROTEUS would have passed if there were no gate.
    'B_pre_gate': {
        'use_jax': True,
        'use_anderson': True,
        'wall_timeout': 3600.0,
    },
    # Isolations: change ONE key at a time relative to A_no_walltimeout.
    'A_plus_walltimeout': {
        'use_jax': False,
        'use_anderson': False,
        'wall_timeout': 3600.0,  # only key different from A_no_walltimeout
    },
    # Truly minimal: no use_jax/use_anderson/wall_timeout at all
    # (this models what config_params looks like if PROTEUS DIDN'T inject
    # the use_jax/use_anderson keys at all, e.g. older PROTEUS pre-3a6b6b00)
    'minimal_no_keys': {},
}

CHILD_TEMPLATE = r'''
from __future__ import annotations
import sys, os, json, hashlib

# Make sure we use the in-PROTEUS Zalmoxis (matches the live runs)
sys.path.insert(0, "{repo_root}/Zalmoxis/src")
sys.path.insert(0, "{repo_root}/src")  # for proteus

# Import PROTEUS-side helpers (uses FWL_DATA EOS paths, matches live runs)
from proteus.interior_struct.zalmoxis import load_zalmoxis_material_dictionaries
from zalmoxis.melting_curves import get_solidus_liquidus_functions

# Build PALEOS melt funcs the same way PROTEUS does for PALEOS-2phase
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

config_params = {config_params!r}

# Call zalmoxis.solver.main directly (no PROTEUS gate)
from zalmoxis.solver import main as zal_main

result = zal_main(
    config_params,
    material_dictionaries=mat_dicts,
    melting_curves_functions=melt_funcs,
    input_dir=os.path.join(
        os.environ.get("FWL_DATA", "/Users/timlichtenberg/git/FWL_DATA"),
        "zalmoxis_eos",
    ),
    layer_mixtures=None,
    volatile_profile=None,
    temperature_function=None,
    temperature_arrays=None,
    p_center_hint=None,
    initial_density=None,
    initial_radii=None,
)

import numpy as np
radii = np.asarray(result["radii"])
mass_encl = np.asarray(result["mass_enclosed"])
density = np.asarray(result["density"])
pressure = np.asarray(result["pressure"])
gravity = np.asarray(result["gravity"])

def chk(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()[:16]

cmb_index = int(np.argmax(mass_encl >= config_params["core_mass_fraction"] * mass_encl[-1]))

out = dict(
    converged=bool(result["converged"]),
    converged_mass=bool(result["converged_mass"]),
    converged_density=bool(result["converged_density"]),
    converged_pressure=bool(result["converged_pressure"]),
    mass_enclosed_last=float(mass_encl[-1]),
    radius_outer=float(radii[-1]),
    gravity_surf=float(gravity[-1]),
    cmb_index=cmb_index,
    n_layers=int(len(radii)),
    pressure_center=float(pressure[0]),
    density_center=float(density[0]),
    radii_chk=chk(radii),
    density_chk=chk(density),
    pressure_chk=chk(pressure),
    mass_encl_chk=chk(mass_encl),
)
print("__JSON__" + json.dumps(out))
'''


def run_variant(name: str, overrides: dict, label: str = '') -> dict:
    config_params = dict(BASE_CONFIG_PARAMS)
    config_params.update(overrides)

    code = CHILD_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        config_params=config_params,
    )

    proc = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=900,
    )

    if proc.returncode != 0:
        print(f'[FAIL] {name}: returncode={proc.returncode}')
        print('STDERR last lines:')
        print('\n'.join(proc.stderr.splitlines()[-30:]))
        return {}

    for line in proc.stdout.splitlines():
        if line.startswith('__JSON__'):
            return json.loads(line[len('__JSON__'):])

    print(f'[FAIL] {name}: no __JSON__ marker. Stderr tail:')
    print('\n'.join(proc.stderr.splitlines()[-10:]))
    return {}


def fmt(v):
    if isinstance(v, float):
        return f'{v:.10e}'
    if isinstance(v, bool):
        return 'T' if v else 'F'
    return str(v)


def main():
    results = {}
    for name, overrides in VARIANTS.items():
        print(f'\n[run] {name} overrides={overrides}')
        results[name] = run_variant(name, overrides)
        r = results[name]
        if r:
            print(f'  mass_last={r.get("mass_enclosed_last"):.10e}')
            print(f'  radius_outer={r.get("radius_outer"):.6e}')
            print(f'  density_chk={r.get("density_chk")}')

    keys = (
        'mass_enclosed_last', 'radius_outer', 'gravity_surf',
        'cmb_index', 'n_layers', 'pressure_center', 'density_center',
        'radii_chk', 'density_chk', 'pressure_chk', 'mass_encl_chk',
        'converged', 'converged_mass',
    )
    print('\n=== Summary ===')
    print(f'{"key":<22}', '  '.join(f'{n:>22}' for n in results))
    for k in keys:
        row = '  '.join(f'{fmt(results[n].get(k)):>22}' for n in results)
        print(f'{k:<22}', row)

    # Pairwise mass deltas vs first variant
    base_name = 'A_no_walltimeout'
    a_mass = results.get(base_name, {}).get('mass_enclosed_last')
    if a_mass is not None and a_mass > 0:
        print(f'\n=== Pairwise mass_enclosed_last deltas vs {base_name} ===')
        for n, r in results.items():
            m = r.get('mass_enclosed_last')
            if m is not None:
                rel = (m - a_mass) / a_mass * 100
                print(f'  {n:<24}: {m:.10e}   delta={rel:+.6f} %')


if __name__ == '__main__':
    main()
