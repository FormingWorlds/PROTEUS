"""Single-shot structure ODE parity test, bypassing all Picard/brentq.

The full zal_main() comparison goes through:
  - Outer Picard on radius_guess
  - Inner Picard on density profile
  - brentq on p_center to satisfy P(R)=P_target

Each layer adds tolerance noise. This probe strips all of that away
and runs ONE call to ``solve_structure`` (numpy) and ONE call to
``solve_structure_via_jax`` with EXACTLY the same:
  - radii grid (linspace 0 → r_outer)
  - y0 = [0, 0, p_center]
  - temperature_function (np.interp on the same r_arr, T_arr) for numpy
  - temperature_arrays = (r_arr, T_arr)                       for JAX
  - layer_mixtures, material_dictionaries, etc.

Outputs: M(r), g(r), P(r) profiles, plus per-cell density evaluated at
the integrated (P, T). If the trajectories agree to integrator
tolerance, the JAX-vs-numpy +2.3 % R_outer gap is in the brentq/Picard
machinery, NOT the structure ODE. If they disagree, the divergence is
already in the per-call ODE solve.

Uses fixed p_center = 4e11 Pa (typical Earth-mass center pressure) and
fixed r_outer = 6.5e6 m so both paths integrate the SAME 0 → 6.5e6 grid.
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
NUM_LAYERS = 150
P_CENTER = 4.0e11  # 400 GPa, ~Earth center
R_OUTER_FIXED = 6.5e6  # m, between numpy 6.5056 and JAX 6.6555


CHILD_TEMPLATE = r'''
from __future__ import annotations
import sys, os, json, hashlib, time

sys.path.insert(0, "{repo_root}/Zalmoxis/src")
sys.path.insert(0, "{repo_root}/src")

import numpy as np

USE_JAX = {use_jax}

r_arr = np.array({r_arr!r})
T_arr = np.array({T_arr!r})
radii = np.linspace(0.0, {r_outer:.6e}, {num_layers})
y0 = [0.0, 0.0, {p_center:.6e}]

p_solidus = 0.8
P_TARGET_SURF = 101325.0

# Build mixtures and materials via the same path zal_main uses
from proteus.interior_struct.zalmoxis import load_zalmoxis_material_dictionaries
from zalmoxis.melting_curves import get_solidus_liquidus_functions
from zalmoxis.structure_model import solve_structure

mzf = 0.8
solidus_raw, liquidus_func = get_solidus_liquidus_functions(
    solidus_id="Stixrude14-solidus",
    liquidus_id="PALEOS-liquidus",
)
def solidus_func(P, _l=liquidus_func, _m=mzf):
    return _l(P) * _m
mat_dicts = load_zalmoxis_material_dictionaries()
input_dir = os.path.join(
    os.environ.get("FWL_DATA", "/Users/timlichtenberg/git/FWL_DATA"),
    "zalmoxis_eos",
)

# Build layer mixtures from the same config the parity probe uses.
config = {{
    'layer_eos_config': {{'core': 'PALEOS:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}},
}}
# build_layer_mixtures expects more fields; use the registry-driven build
from zalmoxis.mixing import LayerMixture
core_mix = LayerMixture(
    components=['PALEOS:iron'],
    fractions=[1.0],
)
mantle_mix = LayerMixture(
    components=['PALEOS-2phase:MgSiO3'],
    fractions=[1.0],
)
layer_mixtures = {{'core': core_mix, 'mantle': mantle_mix}}

# Mass anchors (for layer assignment in coupled_odes via cmb_mass / core_mantle_mass)
PLANET_MASS = 5.971257e24
core_mass_fraction = 0.325
cmb_mass = core_mass_fraction * PLANET_MASS
core_mantle_mass = PLANET_MASS  # 100 % rocky

interpolation_cache = {{}}

if USE_JAX:
    temperature_function = None
    temperature_arrays = (r_arr, T_arr)
else:
    def temperature_function(r, P, _r=r_arr, _T=T_arr):
        return float(np.interp(float(r), _r, _T))
    temperature_arrays = None

mushy_zone_factors = {{
    'PALEOS:iron': 0.8,
    'PALEOS:MgSiO3': 0.8,
    'PALEOS:H2O': 0.8,
}}

t0 = time.time()
mass_encl, gravity, pressure = solve_structure(
    layer_mixtures=layer_mixtures,
    cmb_mass=cmb_mass,
    core_mantle_mass=core_mantle_mass,
    radii=radii,
    adaptive_radial_fraction=0.98,
    relative_tolerance=1e-5,
    absolute_tolerance=1e-6,
    maximum_step={r_outer:.6e} * 0.004,
    material_dictionaries=mat_dicts,
    interpolation_cache=interpolation_cache,
    y0=y0,
    solidus_func=solidus_func,
    liquidus_func=liquidus_func,
    temperature_function=temperature_function,
    mushy_zone_factors=mushy_zone_factors,
    condensed_rho_min=2000.0,
    condensed_rho_scale=200.0,
    binodal_T_scale=300.0,
    use_jax=USE_JAX,
    temperature_arrays=temperature_arrays,
)
t_solve = time.time() - t0

mass_encl = np.asarray(mass_encl, dtype=float)
gravity = np.asarray(gravity, dtype=float)
pressure = np.asarray(pressure, dtype=float)

# Find R where P first goes <= 0 (post-event padding starts)
p_pos = pressure > 0
n_valid = int(np.sum(p_pos))
r_event = float(radii[n_valid - 1]) if n_valid < len(radii) else float(radii[-1])

def chk(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()[:16]

out = dict(
    use_jax=USE_JAX,
    t_solve=t_solve,
    n_valid=n_valid,
    r_event=r_event,
    radii=list(radii),
    mass=list(mass_encl),
    gravity=list(gravity),
    pressure=list(pressure),
    mass_chk=chk(mass_encl),
    gravity_chk=chk(gravity),
    pressure_chk=chk(pressure),
    mass_at_r_outer=float(mass_encl[-1]),
    gravity_at_r_outer=float(gravity[-1]),
    pressure_at_r_outer=float(pressure[-1]),
)
print("__JSON__" + json.dumps(out))
'''


def run_variant(use_jax, r_arr, T_arr, timeout=600):
    code = CHILD_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        use_jax='True' if use_jax else 'False',
        r_arr=list(r_arr),
        T_arr=list(T_arr),
        r_outer=R_OUTER_FIXED,
        num_layers=NUM_LAYERS,
        p_center=P_CENTER,
    )
    proc = subprocess.run(
        [sys.executable, '-u', '-c', code],
        capture_output=True,
        text=True,
        env={**os.environ},
        timeout=timeout,
    )
    if proc.returncode != 0:
        print(f'[FAIL] use_jax={use_jax}: returncode={proc.returncode}')
        print('\n'.join(proc.stderr.splitlines()[-25:]))
        return {}
    for line in proc.stdout.splitlines():
        if line.startswith('__JSON__'):
            return json.loads(line[len('__JSON__'):])
    return {}


def main():
    r_arr = np.linspace(R_CMB, R_SURFACE, N_PROFILE)
    T_arr = np.linspace(T_CMB, T_SURF, N_PROFILE)
    print(f'Single-shot ODE parity: r=[0, {R_OUTER_FIXED:.3e}], '
          f'p_center={P_CENTER:.3e}, num_layers={NUM_LAYERS}')

    print('\n[numpy]', flush=True)
    np_result = run_variant(False, r_arr, T_arr)
    print('\n[jax]', flush=True)
    jx_result = run_variant(True, r_arr, T_arr)

    if not np_result or not jx_result:
        print('FAILED: missing one of the two results')
        return

    print('\n=== profile comparison ===')
    print(f'{"variant":<10} {"t_solve":>10} {"n_valid":>10} {"r_event":>14} '
          f'{"M[-1]":>14} {"g[-1]":>10} {"P[-1]":>14}')
    for tag, r in [('numpy', np_result), ('jax', jx_result)]:
        print(f'{tag:<10} {r["t_solve"]:10.2f} {r["n_valid"]:10d} {r["r_event"]:14.6e} '
              f'{r["mass_at_r_outer"]:14.6e} {r["gravity_at_r_outer"]:10.4f} '
              f'{r["pressure_at_r_outer"]:14.6e}')

    # Per-cell deltas
    np_M = np.asarray(np_result['mass'])
    jx_M = np.asarray(jx_result['mass'])
    np_g = np.asarray(np_result['gravity'])
    jx_g = np.asarray(jx_result['gravity'])
    np_P = np.asarray(np_result['pressure'])
    jx_P = np.asarray(jx_result['pressure'])

    n_common = min(np_result['n_valid'], jx_result['n_valid'])
    print(f'\n=== per-cell deltas (first {n_common} valid cells) ===')
    if n_common > 0:
        dM_rel = (jx_M[:n_common] - np_M[:n_common]) / np_M[:n_common]
        dg_rel = (jx_g[:n_common] - np_g[:n_common]) / np_g[:n_common]
        dP_rel = (jx_P[:n_common] - np_P[:n_common]) / np_P[:n_common]
        print(f'  M: max |dM/M|  = {np.nanmax(np.abs(dM_rel)):.4e}, mean = {np.nanmean(np.abs(dM_rel)):.4e}')
        print(f'  g: max |dg/g|  = {np.nanmax(np.abs(dg_rel)):.4e}, mean = {np.nanmean(np.abs(dg_rel)):.4e}')
        print(f'  P: max |dP/P|  = {np.nanmax(np.abs(dP_rel)):.4e}, mean = {np.nanmean(np.abs(dP_rel)):.4e}')

    print('\n=== checksums ===')
    print(f'mass    numpy={np_result["mass_chk"]}  jax={jx_result["mass_chk"]}')
    print(f'gravity numpy={np_result["gravity_chk"]}  jax={jx_result["gravity_chk"]}')
    print(f'press   numpy={np_result["pressure_chk"]}  jax={jx_result["pressure_chk"]}')

    # Trajectory snapshots at fixed radial cells
    np_r = np.asarray(np_result['radii'])
    print('\n=== trajectory snapshots (cell index, radius_m, M_kg, g, P_Pa) ===')
    print(f'{"idx":>5} {"r":>14} {"M_np":>14} {"M_jx":>14} {"dM/M_np":>12} '
          f'{"P_np":>14} {"P_jx":>14}')
    for idx in [0, 10, 30, 50, 76, 100, 130, 140, 145, 147, 149]:
        if idx >= len(np_r):
            continue
        r = np_r[idx]
        Mn = np_M[idx]
        Mj = jx_M[idx]
        Pn = np_P[idx]
        Pj = jx_P[idx]
        dM = (Mj - Mn) / Mn if Mn > 0 else float('nan')
        print(f'{idx:>5d} {r:14.4e} {Mn:14.6e} {Mj:14.6e} {dM:12.4e} '
              f'{Pn:14.4e} {Pj:14.4e}')

    # Save full profiles for plotting/analysis
    out_npz = REPO_ROOT / 'output_files' / 'single_shot_ode_parity.npz'
    out_npz.parent.mkdir(exist_ok=True)
    np.savez(
        str(out_npz),
        radii=np_r,
        np_M=np_M, np_g=np_g, np_P=np_P,
        jx_M=jx_M, jx_g=jx_g, jx_P=jx_P,
        np_n_valid=np_result['n_valid'],
        jx_n_valid=jx_result['n_valid'],
    )
    print(f'\nSaved profiles to: {out_npz}')


if __name__ == '__main__':
    main()
