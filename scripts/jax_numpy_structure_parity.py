"""Structure-level parity probe: same (r, T) profile through JAX vs numpy.

Feeds the live B-side iter-50 mantle (r, T) profile (from
``output/chili_dry_coupled_stage2_ab_jaxanderson/data/24801_int.nc``)
into ``zalmoxis.solver.main`` along three configurations:

  V1 numpy            : use_jax=False, use_anderson=False, temperature_function only
  V2 jax_anderson     : use_jax=True,  use_anderson=True,  temperature_arrays only (B-side)
  V3 jax_no_anderson  : use_jax=True,  use_anderson=False, temperature_arrays only

At iter 50 of the live B-side run, R_int is +6.66 % above A-side (the in-loop
JAX-vs-numpy gap peak per ``handover_2026_04_24_to_25.md``). The cell-level
EOS density is bit-identical between JAX and numpy
(``scripts/jax_numpy_eos_parity.py``, commit ``34488eac``), so the gap must
sit in the structure ODE (diffrax Tsit5 vs scipy.solve_ivp), the
diffrax.Event termination, or the Anderson outer Picard.

V1 vs V2 ⇒ total JAX-vs-numpy gap at structure level.
V2 vs V3 ⇒ Anderson Picard share of the gap (rest is Tsit5 + Event).

Each variant runs in a fresh subprocess.

Outputs go to output_files/jax_numpy_structure_parity/ (gitignored).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

REPO_ROOT = Path('/Users/timlichtenberg/git/PROTEUS')
OUT_DIR = REPO_ROOT / 'output_files' / 'jax_numpy_structure_parity'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT = (
    REPO_ROOT
    / 'output'
    / 'chili_dry_coupled_stage2_ab_jaxanderson'
    / 'data'
    / '24801_int.nc'
)

# Live B-side iter-50 helpfile values for context (don't change physics).
LIVE_M_INT_B = 6.0334e24    # kg, from helpfile iter 50
LIVE_R_INT_B = 6.8647e6     # m
LIVE_R_INT_A = 6.4339e6     # m, A-side iter 50 from prior analysis (approx)

# Zalmoxis config_params (matches PROTEUS load_zalmoxis_configuration for CHILI Earth)
BASE_CONFIG_PARAMS = {
    'planet_mass': 5.971257000000001e24,   # dry mass after volatile partitioning
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
    'wall_timeout': 600.0,   # 10 min cap; retry doubles to 20 min
}

VARIANTS = {
    # Stage 1: cheap JAX-side variants. Run these first to isolate the
    # Anderson Picard share before paying for the slow numpy callable path.
    'V2_jax_anderson': {
        'use_jax': True,
        'use_anderson': True,
        '_use_arrays': True,    # JAX path uses temperature_arrays
    },
    'V3_jax_no_anderson': {
        'use_jax': True,
        'use_anderson': False,
        '_use_arrays': True,
    },
    # Stage 2: numpy path with the same r-only T closure that PROTEUS uses.
    # This is the slow path (PALEOS clamp; ~150 s/call without retry; up to
    # ~600 s with one retry). Run last so the cheap variants don't gate on it.
    'V1_numpy': {
        'use_jax': False,
        'use_anderson': False,
        '_use_arrays': False,   # numpy path uses temperature_function (callable)
    },
}


def load_iter50_profile():
    ds = Dataset(str(SNAPSHOT))
    r_km = np.asarray(ds.variables['radius_s'][:], dtype=float)
    T = np.asarray(ds.variables['temp_s'][:], dtype=float)
    ds.close()
    r_m = r_km * 1000.0
    # Ascending sort defensively
    order = np.argsort(r_m)
    r_m = np.ascontiguousarray(r_m[order])
    T = np.ascontiguousarray(T[order])
    return r_m, T


CHILD_TEMPLATE = r'''
from __future__ import annotations
import sys, os, json, hashlib, time

sys.path.insert(0, "{repo_root}/Zalmoxis/src")
sys.path.insert(0, "{repo_root}/src")

import numpy as np

config_params = {config_params!r}
use_arrays = bool(config_params.pop("_use_arrays"))
r_arr = np.array({r_arr!r})
T_arr = np.array({T_arr!r})

# Build arrays/callable per the path under test
if use_arrays:
    temperature_arrays = (np.ascontiguousarray(r_arr), np.ascontiguousarray(T_arr))
    temperature_function = None
else:
    temperature_arrays = None
    def temperature_function(r, P, _r=r_arr, _T=T_arr):
        # Ignore P; r-only T (mirrors PROTEUS update_structure_from_interior).
        return float(np.interp(float(r), _r, _T))

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

t0 = time.time()
sys.stderr.write(f"[child] starting solve, use_jax={{config_params.get('use_jax')}}, use_anderson={{config_params.get('use_anderson')}}, arrays={{use_arrays}}\n")
result = zal_main(
    config_params,
    material_dictionaries=mat_dicts,
    melting_curves_functions=melt_funcs,
    input_dir=input_dir,
    layer_mixtures=None,
    volatile_profile=None,
    temperature_function=temperature_function,
    temperature_arrays=temperature_arrays,
    p_center_hint=None,
    initial_density=None,
    initial_radii=None,
)
t_solve = time.time() - t0
sys.stderr.write(f"[child] solve done in {{t_solve:.1f}}s\n")

radii = np.asarray(result["radii"])
mass_encl = np.asarray(result["mass_enclosed"])
density = np.asarray(result["density"])
pressure = np.asarray(result["pressure"])
gravity = np.asarray(result["gravity"])

def chk(arr):
    return hashlib.sha256(np.ascontiguousarray(arr, dtype=np.float64).tobytes()).hexdigest()[:16]

cmb_index = int(np.argmax(mass_encl >= config_params["core_mass_fraction"] * mass_encl[-1]))

# Save full profile for plotting
np.savez(
    "{out_npz}",
    radii=radii, mass_encl=mass_encl, density=density,
    pressure=pressure, gravity=gravity,
    converged=int(bool(result["converged"])),
    cmb_index=cmb_index,
    t_solve=t_solve,
)

out = dict(
    t_solve=t_solve,
    converged=bool(result["converged"]),
    converged_mass=bool(result["converged_mass"]),
    converged_density=bool(result["converged_density"]),
    converged_pressure=bool(result["converged_pressure"]),
    mass_enclosed_last=float(mass_encl[-1]),
    radius_outer=float(radii[-1]),
    gravity_surf=float(gravity[-1]),
    pressure_center=float(pressure[0]),
    density_center=float(density[0]),
    cmb_index=cmb_index,
    n_layers=int(len(radii)),
    radii_chk=chk(radii),
    density_chk=chk(density),
    pressure_chk=chk(pressure),
    mass_encl_chk=chk(mass_encl),
)
print("__JSON__" + json.dumps(out))
'''


def run_variant(name, overrides, r_arr, T_arr):
    cp = dict(BASE_CONFIG_PARAMS)
    cp.update(overrides)
    out_npz = OUT_DIR / f'profile_{name}.npz'
    code = CHILD_TEMPLATE.format(
        repo_root=str(REPO_ROOT),
        config_params=cp,
        r_arr=list(r_arr),
        T_arr=list(T_arr),
        out_npz=str(out_npz),
    )
    proc = subprocess.run(
        [sys.executable, '-u', '-c', code],
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
    print(f'[FAIL] {name}: no __JSON__')
    print('Stderr tail:')
    print('\n'.join(proc.stderr.splitlines()[-10:]))
    return {}


def fmt(v):
    if isinstance(v, float):
        return f'{v:.10e}'
    if isinstance(v, bool):
        return 'T' if v else 'F'
    return str(v)


def plot_profiles(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    profiles = {}
    for name in results:
        npz = OUT_DIR / f'profile_{name}.npz'
        if npz.exists():
            profiles[name] = dict(np.load(npz))

    if 'V1_numpy' not in profiles:
        print('V1_numpy profile missing, skipping plot')
        return
    base = profiles['V1_numpy']

    colors = {'V1_numpy': 'k', 'V2_jax_anderson': 'r', 'V3_jax_no_anderson': 'b'}
    styles = {'V1_numpy': '-', 'V2_jax_anderson': '--', 'V3_jax_no_anderson': ':'}

    # Density
    ax = axes[0, 0]
    for name, p in profiles.items():
        ax.plot(p['radii'] / 1e6, p['density'], colors[name] + styles[name],
                lw=1.5, label=name)
    ax.set_xlabel('r (Mm)')
    ax.set_ylabel(r'$\rho$ (kg/m$^3$)')
    ax.set_title('(a) Density profile')
    ax.legend()
    ax.grid(alpha=0.3)

    # Pressure
    ax = axes[0, 1]
    for name, p in profiles.items():
        ax.plot(p['radii'] / 1e6, p['pressure'] / 1e9,
                colors[name] + styles[name], lw=1.5, label=name)
    ax.set_xlabel('r (Mm)')
    ax.set_ylabel('P (GPa)')
    ax.set_title('(b) Pressure profile')
    ax.legend()
    ax.grid(alpha=0.3)

    # Mass enclosed
    ax = axes[1, 0]
    for name, p in profiles.items():
        ax.plot(p['radii'] / 1e6, p['mass_encl'] / 5.972e24,
                colors[name] + styles[name], lw=1.5, label=name)
    ax.set_xlabel('r (Mm)')
    ax.set_ylabel(r'$M(r)/M_\oplus$')
    ax.set_title('(c) Cumulative mass')
    ax.legend()
    ax.grid(alpha=0.3)

    # Density delta vs V1 (numpy)
    ax = axes[1, 1]
    r_base = base['radii']
    for name, p in profiles.items():
        if name == 'V1_numpy':
            continue
        # Interp p['density'] onto base radii
        rho_interp = np.interp(r_base, p['radii'], p['density'])
        delta = (rho_interp - base['density']) / base['density'] * 100.0
        ax.plot(r_base / 1e6, delta, colors[name] + styles[name], lw=1.5,
                label=f'{name} - V1_numpy')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('r (Mm)')
    ax.set_ylabel(r'$\Delta\rho / \rho_\mathrm{numpy}$ [\%]')
    ax.set_title('(d) Density delta vs numpy path')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(
        'Structure-level parity: same (r,T) profile through JAX vs numpy '
        f'(iter-50 input from B-side)\n'
        f'Live B-side iter 50 R_int = {LIVE_R_INT_B/1e6:.3f} Mm; '
        f'live A-side R_int ≈ {LIVE_R_INT_A/1e6:.3f} Mm',
        fontsize=11,
    )
    plt.tight_layout()
    out = OUT_DIR / 'structure_parity.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f'Wrote {out}')


def main():
    print(f'Loading iter-50 profile from {SNAPSHOT.name}')
    r_arr, T_arr = load_iter50_profile()
    print(f'  r: {r_arr[0]:.3e} to {r_arr[-1]:.3e} m, T: {T_arr.min():.0f} to {T_arr.max():.0f} K, n={len(r_arr)}')

    results = {}
    for name, overrides in VARIANTS.items():
        print(f'\n[{name}] overrides={overrides}', flush=True)
        results[name] = run_variant(name, overrides, r_arr, T_arr)
        r = results[name]
        if r:
            print(f'  R_outer={r["radius_outer"]:.6e}  M_outer={r["mass_enclosed_last"]:.6e}  '
                  f't_solve={r["t_solve"]:.1f}s  converged={r["converged"]}',
                  flush=True)

    keys = (
        't_solve', 'mass_enclosed_last', 'radius_outer', 'gravity_surf',
        'pressure_center', 'density_center', 'cmb_index', 'n_layers',
        'radii_chk', 'density_chk', 'pressure_chk', 'mass_encl_chk',
        'converged', 'converged_mass',
    )
    print('\n=== Summary ===', flush=True)
    print(f'{"key":<22}', '  '.join(f'{n:>22}' for n in results), flush=True)
    for k in keys:
        row = '  '.join(f'{fmt(results[n].get(k)):>22}' for n in results)
        print(f'{k:<22}', row, flush=True)

    # Pairwise R_outer deltas vs V1
    base = results.get('V1_numpy', {}).get('radius_outer')
    if base:
        print(f'\n=== Pairwise R_outer / M_outer deltas vs V1_numpy (R_outer={base:.6e}) ===',
              flush=True)
        for n, r in results.items():
            R = r.get('radius_outer')
            M = r.get('mass_enclosed_last')
            if R and M:
                dR = (R - base) / base * 100
                dM = (M - results['V1_numpy']['mass_enclosed_last']) / results['V1_numpy']['mass_enclosed_last'] * 100
                print(f'  {n:<22}: R_outer={R:.6e}  Δ={dR:+.3f}%   '
                      f'M_outer={M:.6e}  Δ={dM:+.3f}%',
                      flush=True)

    plot_profiles(results)


if __name__ == '__main__':
    main()
