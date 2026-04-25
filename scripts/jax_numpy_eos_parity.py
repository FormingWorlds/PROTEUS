"""JAX vs numpy PALEOS-2phase EOS parity probe.

Cell-level (P, T) → ρ comparison. The hypothesis is that the +6.66 % R_int
peak at iter ~50 in B-side coupled run comes from per-cell density
mis-evaluation in the JAX EOS path through the PALEOS-2phase mantle
melt regime. This probe localises the (P, T) bands where ``get_tdep_density_jax``
diverges from numpy ``get_Tdep_density``.

Builds:
  - solid + melt PALEOS-2phase bilinear caches via load_paleos_table()
  - solidus + liquidus functions (PALEOS-liquidus + mzf=0.8)
  - Stixrude14 stub solidus (numpy uses solidus_func ≈ T_liq * 0.8)
Evaluates both on a P×T grid and on the live B-side iter-50 (P, T) mantle
profile (from ``output/.../data/24801_int.nc``), then plots:
  - 2-D heatmap |Δρ/ρ| vs (log P, T) with PALEOS solidus/liquidus overlay
  - 1-D delta along the live iter-50 profile

Outputs go to output_files/jax_numpy_eos_parity/ (gitignored).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path('/Users/timlichtenberg/git/PROTEUS')
sys.path.insert(0, str(REPO_ROOT / 'Zalmoxis' / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'src'))

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

# Zalmoxis imports
from zalmoxis.eos.interpolation import load_paleos_table  # noqa: E402
from zalmoxis.eos.tdep import get_Tdep_density  # noqa: E402
from zalmoxis.jax_eos.tdep import get_tdep_density_jax  # noqa: E402
from zalmoxis.jax_eos.wrapper import _extract_sub_args  # noqa: E402
from zalmoxis.melting_curves import get_solidus_liquidus_functions  # noqa: E402

OUT_DIR = REPO_ROOT / 'output_files' / 'jax_numpy_eos_parity'
OUT_DIR.mkdir(parents=True, exist_ok=True)

FWL_EOS = Path(os.environ.get('FWL_DATA', '/Users/timlichtenberg/git/FWL_DATA')) / 'zalmoxis_eos'
SOL_FILE = FWL_EOS / 'EOS_PALEOS_MgSiO3' / 'paleos_mgsio3_tables_pt_proteus_solid.dat'
LIQ_FILE = FWL_EOS / 'EOS_PALEOS_MgSiO3' / 'paleos_mgsio3_tables_pt_proteus_liquid.dat'

MZF = 0.8  # mushy_zone_factor (TOML)


def build_caches():
    """Load solid and melt PALEOS tables; return numpy + JAX kwarg dicts."""
    sol_cached = load_paleos_table(str(SOL_FILE))
    liq_cached = load_paleos_table(str(LIQ_FILE))

    # Numpy material_dictionaries dict (matches eos.tdep.get_tabulated_eos call)
    mat_dicts_proxy = {
        # Two named keys: solid_mantle and melted_mantle (used in tdep.py)
        'solid_mantle': {
            'eos_file': str(SOL_FILE),
            'format': 'paleos',
        },
        'melted_mantle': {
            'eos_file': str(LIQ_FILE),
            'format': 'paleos',
        },
    }

    # JAX kwargs: solid → "sol", melt → "liq"
    jax_kwargs = {}
    jax_kwargs.update(_extract_sub_args(sol_cached, 'sol'))
    jax_kwargs.update(_extract_sub_args(liq_cached, 'liq'))

    # melting curves
    sol_func_dummy, liq_func = get_solidus_liquidus_functions(
        solidus_id='Stixrude14-solidus',
        liquidus_id='PALEOS-liquidus',
    )
    # The PROTEUS path derives solidus = liquidus * mzf, see
    # interior_struct/zalmoxis.py:_make_derived_solidus.
    def sol_func(P):
        return liq_func(P) * MZF

    return mat_dicts_proxy, jax_kwargs, sol_func, liq_func


def numpy_density(P, T, mat_dicts, sol_func, liq_func, interp_cache):
    """Numpy ρ via get_Tdep_density (matches PROTEUS coupled path)."""
    return get_Tdep_density(
        P, T, mat_dicts, sol_func, liq_func,
        interpolation_functions=interp_cache,
    )


def jax_density(P, T, sol_func, liq_func, jax_kwargs):
    """JAX ρ via get_tdep_density_jax (single point, no jit batch yet)."""
    T_sol = float(sol_func(P))
    T_liq = float(liq_func(P))
    rho = get_tdep_density_jax(
        jnp.float64(P), jnp.float64(T),
        jnp.float64(T_sol), jnp.float64(T_liq),
        **jax_kwargs,
    )
    return float(rho)


def grid_scan(mat_dicts, jax_kwargs, sol_func, liq_func, n_p=60, n_t=50):
    """Scan a (P, T) grid spanning the PALEOS-2phase mantle regime."""
    log_P = np.linspace(5.0, 12.0, n_p)   # 1 bar to 1 TPa
    P_arr = 10.0 ** log_P                  # Pa
    T_arr = np.linspace(1500.0, 6000.0, n_t)  # K

    rho_np = np.full((n_p, n_t), np.nan)
    rho_jx = np.full((n_p, n_t), np.nan)
    interp_cache = {}

    print(f'Grid scan: {n_p} P × {n_t} T = {n_p*n_t} cells')
    for i, P in enumerate(P_arr):
        for j, T in enumerate(T_arr):
            try:
                r_n = numpy_density(P, T, mat_dicts, sol_func, liq_func, interp_cache)
                if r_n is None:
                    rho_np[i, j] = np.nan
                else:
                    rho_np[i, j] = float(r_n)
            except Exception:
                rho_np[i, j] = np.nan
            try:
                rho_jx[i, j] = jax_density(P, T, sol_func, liq_func, jax_kwargs)
            except Exception:
                rho_jx[i, j] = np.nan

    # Solidus / liquidus curves at the grid pressures
    T_sol_curve = np.array([float(sol_func(P)) for P in P_arr])
    T_liq_curve = np.array([float(liq_func(P)) for P in P_arr])

    return {
        'log_P': log_P, 'P': P_arr, 'T': T_arr,
        'rho_np': rho_np, 'rho_jx': rho_jx,
        'T_sol': T_sol_curve, 'T_liq': T_liq_curve,
    }


def snapshot_probe(mat_dicts, jax_kwargs, sol_func, liq_func, snapshot_path: Path):
    """Per-cell ρ comparison along a live iter-50 (P, T) profile."""
    ds = Dataset(str(snapshot_path))
    r_s = np.asarray(ds.variables['radius_s'][:], dtype=float)
    P_s = np.asarray(ds.variables['pres_s'][:], dtype=float)
    T_s = np.asarray(ds.variables['temp_s'][:], dtype=float)
    rho_s = np.asarray(ds.variables['density_s'][:], dtype=float)
    phi_s = np.asarray(ds.variables['phi_s'][:], dtype=float)
    ds.close()

    rho_np = np.zeros_like(r_s)
    rho_jx = np.zeros_like(r_s)
    interp_cache = {}
    for i, (P, T) in enumerate(zip(P_s, T_s)):
        try:
            r_n = numpy_density(P, T, mat_dicts, sol_func, liq_func, interp_cache)
            rho_np[i] = float(r_n) if r_n is not None else np.nan
        except Exception:
            rho_np[i] = np.nan
        try:
            rho_jx[i] = jax_density(P, T, sol_func, liq_func, jax_kwargs)
        except Exception:
            rho_jx[i] = np.nan

    return {
        'r': r_s, 'P': P_s, 'T': T_s, 'phi': phi_s,
        'rho_aragog': rho_s,  # what Aragog stored at iter 50
        'rho_np': rho_np, 'rho_jx': rho_jx,
    }


def plot_grid(g, out: Path):
    rho_np = g['rho_np']
    rho_jx = g['rho_jx']
    rel = np.abs(rho_jx - rho_np) / rho_np * 100.0
    finite = np.isfinite(rel)
    if not finite.any():
        print('No finite cells for grid plot')
        return
    vmin, vmax = 0.0, np.nanpercentile(rel[finite], 99)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # JAX rho
    ax = axes[0]
    pcm = ax.pcolormesh(g['log_P'], g['T'], rho_np.T, shading='auto', cmap='viridis')
    ax.plot(g['log_P'], g['T_sol'], 'w-', lw=1.5, label='solidus')
    ax.plot(g['log_P'], g['T_liq'], 'w--', lw=1.5, label='liquidus')
    ax.set_xlabel(r'$\log_{10}(P / \mathrm{Pa})$')
    ax.set_ylabel(r'$T$ (K)')
    ax.set_title(r'(a) numpy $\rho_\mathrm{np}(P,T)$ [kg/m$^3$]')
    plt.colorbar(pcm, ax=ax)
    ax.legend(loc='lower right', fontsize=8)

    # Numpy rho
    ax = axes[1]
    pcm = ax.pcolormesh(g['log_P'], g['T'], rho_jx.T, shading='auto', cmap='viridis')
    ax.plot(g['log_P'], g['T_sol'], 'w-', lw=1.5)
    ax.plot(g['log_P'], g['T_liq'], 'w--', lw=1.5)
    ax.set_xlabel(r'$\log_{10}(P / \mathrm{Pa})$')
    ax.set_title(r'(b) JAX $\rho_\mathrm{jax}(P,T)$ [kg/m$^3$]')
    plt.colorbar(pcm, ax=ax)

    # Relative diff
    ax = axes[2]
    pcm = ax.pcolormesh(
        g['log_P'], g['T'], rel.T, shading='auto', cmap='magma',
        vmin=vmin, vmax=vmax,
    )
    ax.plot(g['log_P'], g['T_sol'], 'c-', lw=1.5, label='solidus')
    ax.plot(g['log_P'], g['T_liq'], 'c--', lw=1.5, label='liquidus')
    ax.set_xlabel(r'$\log_{10}(P / \mathrm{Pa})$')
    ax.set_title(r'(c) $|\rho_\mathrm{jax}-\rho_\mathrm{np}| / \rho_\mathrm{np}$ [\%]')
    plt.colorbar(pcm, ax=ax)
    ax.legend(loc='lower right', fontsize=8)

    fig.suptitle(
        'JAX vs numpy PALEOS-2phase EOS density parity, mzf=0.8',
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    print(f'Wrote {out}')

    # Print stats
    print(f'Grid stats: max |rel|={np.nanmax(rel):.3f}%, '
          f'p99={np.nanpercentile(rel[finite], 99):.3f}%, '
          f'p90={np.nanpercentile(rel[finite], 90):.3f}%, '
          f'median={np.nanmedian(rel[finite]):.3f}%')

    # Find hot-spot
    if np.nanmax(rel) > 0.1:
        idx = np.unravel_index(np.nanargmax(rel), rel.shape)
        i, j = idx
        print(f'  Hot spot: P=10^{g["log_P"][i]:.2f} Pa, T={g["T"][j]:.0f} K, '
              f'rel={rel[i,j]:.3f}%, np={rho_np[i,j]:.1f}, jx={rho_jx[i,j]:.1f}')


def plot_snapshot(s, out: Path):
    rho_np = s['rho_np']
    rho_jx = s['rho_jx']
    rho_a = s['rho_aragog']
    r = s['r']

    rel_np_vs_aragog = (rho_np - rho_a) / rho_a * 100.0
    rel_jx_vs_aragog = (rho_jx - rho_a) / rho_a * 100.0
    rel_jx_vs_np = (rho_jx - rho_np) / rho_np * 100.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Density profiles
    ax = axes[0, 0]
    ax.plot(r / 1e6, rho_a, 'k-', lw=2, label='Aragog stored', zorder=3)
    ax.plot(r / 1e6, rho_np, 'b--', lw=1.5, label='numpy ρ(P,T)')
    ax.plot(r / 1e6, rho_jx, 'r:', lw=1.5, label='JAX ρ(P,T)')
    ax.set_xlabel('r (Mm)')
    ax.set_ylabel(r'$\rho$ (kg/m$^3$)')
    ax.set_title('(a) Density along iter-50 mantle profile')
    ax.legend()
    ax.grid(alpha=0.3)

    # Relative deltas
    ax = axes[0, 1]
    ax.plot(r / 1e6, rel_np_vs_aragog, 'b-', lw=1, label='numpy - Aragog')
    ax.plot(r / 1e6, rel_jx_vs_aragog, 'r-', lw=1, label='JAX - Aragog')
    ax.plot(r / 1e6, rel_jx_vs_np, 'g-', lw=1.5, label='JAX - numpy')
    ax.axhline(0, color='k', lw=0.5, ls=':')
    ax.set_xlabel('r (Mm)')
    ax.set_ylabel(r'$\Delta\rho / \rho$ [\%]')
    ax.set_title('(b) Per-cell relative differences')
    ax.legend()
    ax.grid(alpha=0.3)

    # T(r) and phi(r) for context
    ax = axes[1, 0]
    ax.plot(r / 1e6, s['T'], 'k-', label='T(r)')
    ax.set_xlabel('r (Mm)')
    ax.set_ylabel('T (K)', color='k')
    ax2 = ax.twinx()
    ax2.plot(r / 1e6, s['phi'], 'm-', label=r'$\phi$(r)')
    ax2.set_ylabel(r'$\phi$ (melt frac)', color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    ax.set_title('(c) T(r) and melt fraction at iter 50')
    ax.grid(alpha=0.3)

    # P(r)
    ax = axes[1, 1]
    ax.plot(r / 1e6, s['P'] / 1e9, 'k-')
    ax.set_xlabel('r (Mm)')
    ax.set_ylabel('P (GPa)')
    ax.set_title('(d) P(r) at iter 50')
    ax.grid(alpha=0.3)

    fig.suptitle(
        'JAX vs numpy ρ(P,T) parity along live B-side iter-50 mantle profile\n'
        '(snapshot 24801_int.nc, output/chili_dry_coupled_stage2_ab_jaxanderson)',
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out, bbox_inches='tight')
    print(f'Wrote {out}')

    print('Snapshot stats:')
    print(f'  numpy vs Aragog: max |Δρ/ρ|={np.nanmax(np.abs(rel_np_vs_aragog)):.3f}%, '
          f'mean={np.nanmean(rel_np_vs_aragog):+.3f}%')
    print(f'  JAX   vs Aragog: max |Δρ/ρ|={np.nanmax(np.abs(rel_jx_vs_aragog)):.3f}%, '
          f'mean={np.nanmean(rel_jx_vs_aragog):+.3f}%')
    print(f'  JAX   vs numpy:  max |Δρ/ρ|={np.nanmax(np.abs(rel_jx_vs_np)):.6f}%, '
          f'mean={np.nanmean(rel_jx_vs_np):+.6f}%')


def main():
    print('Loading PALEOS-2phase tables...')
    mat_dicts, jax_kwargs, sol_func, liq_func = build_caches()
    print('  done.')

    # 1. Synthetic (P, T) grid scan
    g = grid_scan(mat_dicts, jax_kwargs, sol_func, liq_func, n_p=60, n_t=60)
    plot_grid(g, OUT_DIR / 'eos_parity_grid.pdf')
    np.savez(OUT_DIR / 'eos_parity_grid.npz', **g)

    # 2. Live iter-50 snapshot probe
    snap = REPO_ROOT / 'output' / 'chili_dry_coupled_stage2_ab_jaxanderson' / 'data' / '24801_int.nc'
    if snap.exists():
        s = snapshot_probe(mat_dicts, jax_kwargs, sol_func, liq_func, snap)
        plot_snapshot(s, OUT_DIR / 'eos_parity_snapshot_iter50.pdf')
        np.savez(OUT_DIR / 'eos_parity_snapshot_iter50.npz', **s)
    else:
        print(f'WARNING: snapshot {snap} not found, skipping snapshot probe')


if __name__ == '__main__':
    main()
