"""Structure-level parity probe (smoothed T profile).

Companion to ``scripts/jax_numpy_structure_parity.py``: runs the same JAX
vs numpy structure comparison but with a SMOOTH linear T(r) instead of
the live iter-50 mantle profile (which has a thermal-boundary-layer spike
that stalls the numpy outer Picard).

Linear T(r) from T_cmb=4065 K (matches iter-50 cell 0) to T_surf=2800 K
(matches iter-50 cell -1) over r in [3.378e6, 6.871e6] m.

Goal: with a tractable T(r), confirm whether the JAX and numpy structure
solvers converge to the SAME R_outer, M_outer, ρ(r) given identical
boundary conditions. If yes, the +6.66 % R_int peak is fully explained
by the live profile shape (which numpy also struggles with). If no,
some Tsit5/Event/numpy-integrator disagreement at smooth profiles
remains.

Outputs to output_files/jax_numpy_structure_parity_smooth/.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path('/Users/timlichtenberg/git/PROTEUS')
OUT_DIR = REPO_ROOT / 'output_files' / 'jax_numpy_structure_parity_smooth'
OUT_DIR.mkdir(parents=True, exist_ok=True)

R_CMB = 3.378e6     # m, matches iter-50 r_s[0]
R_SURFACE = 6.871e6  # m, matches iter-50 r_s[-1]
T_CMB = 4065.0       # K, matches iter-50 T_s[0]
T_SURF = 2800.0      # K, matches iter-50 T_s[-1]
N_PROFILE = 79       # match Aragog mesh size

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
}

VARIANTS = {
    'V2_jax_anderson': {
        'use_jax': True, 'use_anderson': True, '_use_arrays': True,
    },
    'V3_jax_no_anderson': {
        'use_jax': True, 'use_anderson': False, '_use_arrays': True,
    },
    'V1_numpy': {
        'use_jax': False, 'use_anderson': False, '_use_arrays': False,
    },
}


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

if use_arrays:
    temperature_arrays = (np.ascontiguousarray(r_arr), np.ascontiguousarray(T_arr))
    temperature_function = None
else:
    temperature_arrays = None
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


def run_variant(name, overrides, r_arr, T_arr, timeout=1500):
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
        timeout=timeout,
    )
    if proc.returncode != 0:
        print(f'[FAIL] {name}: returncode={proc.returncode}')
        print('\n'.join(proc.stderr.splitlines()[-30:]))
        return {}
    for line in proc.stdout.splitlines():
        if line.startswith('__JSON__'):
            return json.loads(line[len('__JSON__'):])
    print(f'[FAIL] {name}: no __JSON__')
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

    ax = axes[0, 0]
    for name, p in profiles.items():
        ax.plot(p['radii'] / 1e6, p['density'], colors[name] + styles[name],
                lw=1.5, label=name)
    ax.set_xlabel('r (Mm)')
    ax.set_ylabel(r'$\rho$ (kg/m$^3$)')
    ax.set_title('(a) Density profile')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for name, p in profiles.items():
        ax.plot(p['radii'] / 1e6, p['pressure'] / 1e9,
                colors[name] + styles[name], lw=1.5, label=name)
    ax.set_xlabel('r (Mm)')
    ax.set_ylabel('P (GPa)')
    ax.set_title('(b) Pressure profile')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    for name, p in profiles.items():
        ax.plot(p['radii'] / 1e6, p['mass_encl'] / 5.972e24,
                colors[name] + styles[name], lw=1.5, label=name)
    ax.set_xlabel('r (Mm)')
    ax.set_ylabel(r'$M(r)/M_\oplus$')
    ax.set_title('(c) Cumulative mass')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    r_base = base['radii']
    for name, p in profiles.items():
        if name == 'V1_numpy':
            continue
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
        'Structure parity (linear T(r), CMB→surface): JAX vs numpy paths\n'
        f'T_cmb={T_CMB:.0f} K, T_surf={T_SURF:.0f} K, n_pts={N_PROFILE}',
        fontsize=11,
    )
    plt.tight_layout()
    out = OUT_DIR / 'structure_parity_smooth.pdf'
    plt.savefig(out, bbox_inches='tight')
    print(f'Wrote {out}')


def main():
    r_arr = np.linspace(R_CMB, R_SURFACE, N_PROFILE)
    T_arr = np.linspace(T_CMB, T_SURF, N_PROFILE)  # linear, monotonic decrease
    print(f'Smoothed input: r=[{r_arr[0]:.3e}, {r_arr[-1]:.3e}] m, '
          f'T=[{T_arr[0]:.0f}, {T_arr[-1]:.0f}] K, n={len(r_arr)}')

    results = {}
    for name, overrides in VARIANTS.items():
        print(f'\n[{name}] overrides={overrides}', flush=True)
        results[name] = run_variant(name, overrides, r_arr, T_arr)
        r = results[name]
        if r:
            print(f'  R_outer={r["radius_outer"]:.6e}  '
                  f'M_outer={r["mass_enclosed_last"]:.6e}  '
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

    base = results.get('V1_numpy', {}).get('radius_outer')
    if base:
        print(f'\n=== Pairwise R_outer / M_outer deltas vs V1_numpy '
              f'(R_outer={base:.6e}) ===', flush=True)
        for n, r in results.items():
            R = r.get('radius_outer')
            M = r.get('mass_enclosed_last')
            if R and M:
                dR = (R - base) / base * 100
                dM = (M - results['V1_numpy']['mass_enclosed_last']) / results['V1_numpy']['mass_enclosed_last'] * 100
                print(f'  {n:<22}: R_outer={R:.6e}  Δ={dR:+.4f}%   '
                      f'M_outer={M:.6e}  Δ={dM:+.4f}%',
                      flush=True)

    plot_profiles(results)


if __name__ == '__main__':
    main()
