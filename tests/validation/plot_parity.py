#!/usr/bin/env python3
"""Plot SPIDER vs Aragog parity comparison results.

Reads runtime_helpfile.csv from both SPIDER and Aragog runs and
generates comparison plots for T_magma, Phi, heat flux, and
solidification time.

Usage:
    python plot_parity.py
    python plot_parity.py --mass 1.0 --mix-tag nomix
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_BASE = Path(__file__).parent.parent.parent / 'output' / 'parity'
PLOTS_DIR = OUTPUT_BASE / 'plots'


def load_helpfile(case_name):
    """Load runtime_helpfile.csv for a case.

    Parameters
    ----------
    case_name : str
        Case name (e.g., 'parity_M1.0_spider_nomix').

    Returns
    -------
    pd.DataFrame or None
        Helpfile data, or None if not found.
    """
    path = Path('output') / 'parity' / case_name / 'runtime_helpfile.csv'
    if not path.is_file():
        print(f'  Not found: {path}')
        return None
    return pd.read_csv(path)


def plot_comparison(mass=1.0, mix_tag='nomix'):
    """Generate parity comparison plots.

    Parameters
    ----------
    mass : float
        Planet mass.
    mix_tag : str
        'nomix' or 'mix'.
    """
    spider_name = f'parity_M{mass:.1f}_spider_{mix_tag}'
    aragog_name = f'parity_M{mass:.1f}_aragog_{mix_tag}'

    spider = load_helpfile(spider_name)
    aragog = load_helpfile(aragog_name)

    if spider is None or aragog is None:
        print(f'Cannot compare: missing data for {mass} ME {mix_tag}')
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(
        f'SPIDER vs Aragog Parity: {mass:.1f} ME, {mix_tag}',
        fontsize=14, fontweight='bold',
    )

    # 1. T_magma vs time
    ax = axes[0, 0]
    ax.plot(spider['Time'], spider['T_magma'], 'b-', lw=2, label='SPIDER')
    ax.plot(aragog['Time'], aragog['T_magma'], 'r--', lw=2, label='Aragog')
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('T_magma (K)')
    ax.set_xscale('log')
    ax.set_title('Mantle temperature')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Phi vs time
    ax = axes[0, 1]
    phi_key_spider = 'Phi_global' if 'Phi_global' in spider.columns else 'Phi_global_vol'
    phi_key_aragog = 'Phi_global' if 'Phi_global' in aragog.columns else 'Phi_global_vol'
    ax.plot(spider['Time'], spider[phi_key_spider], 'b-', lw=2, label='SPIDER')
    ax.plot(aragog['Time'], aragog[phi_key_aragog], 'r--', lw=2, label='Aragog')
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('Global melt fraction')
    ax.set_xscale('log')
    ax.set_title('Melt fraction')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. F_int vs time
    ax = axes[0, 2]
    if 'F_int' in spider.columns and 'F_int' in aragog.columns:
        ax.plot(spider['Time'], spider['F_int'], 'b-', lw=2, label='SPIDER')
        ax.plot(aragog['Time'], aragog['F_int'], 'r--', lw=2, label='Aragog')
        ax.set_xlabel('Time (yr)')
        ax.set_ylabel('F_int (W/m^2)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Interior heat flux')
        ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. T_magma relative difference
    ax = axes[1, 0]
    # Interpolate onto common time grid
    t_min = max(spider['Time'].iloc[3], aragog['Time'].iloc[3])
    t_max = min(spider['Time'].iloc[-1], aragog['Time'].iloc[-1])
    if t_max > t_min:
        t_common = np.logspace(np.log10(max(t_min, 1)), np.log10(t_max), 200)
        T_s_interp = np.interp(t_common, spider['Time'], spider['T_magma'])
        T_a_interp = np.interp(t_common, aragog['Time'], aragog['T_magma'])
        rel_diff = (T_s_interp - T_a_interp) / T_s_interp * 100
        ax.plot(t_common, rel_diff, 'k-', lw=2)
        ax.axhline(5, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
        ax.axhline(-5, color='orange', linestyle='--', alpha=0.5)
        ax.axhline(0, color='grey', alpha=0.3)
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('(T_SPIDER - T_Aragog) / T_SPIDER (%)')
    ax.set_xscale('log')
    ax.set_title('T_magma relative difference')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Phi difference
    ax = axes[1, 1]
    if t_max > t_min:
        phi_s = np.interp(t_common, spider['Time'], spider[phi_key_spider])
        phi_a = np.interp(t_common, aragog['Time'], aragog[phi_key_aragog])
        ax.plot(t_common, phi_s - phi_a, 'k-', lw=2)
        ax.axhline(0, color='grey', alpha=0.3)
    ax.set_xlabel('Time (yr)')
    ax.set_ylabel('Phi_SPIDER - Phi_Aragog')
    ax.set_xscale('log')
    ax.set_title('Melt fraction difference')
    ax.grid(True, alpha=0.3)

    # 6. Summary stats
    ax = axes[1, 2]
    ax.axis('off')
    t_solid_spider = spider['Time'].iloc[-1]
    t_solid_aragog = aragog['Time'].iloc[-1]

    summary = (
        f'Solidification time:\n'
        f'  SPIDER: {t_solid_spider:.2e} yr\n'
        f'  Aragog: {t_solid_aragog:.2e} yr\n'
        f'  Ratio:  {t_solid_spider / t_solid_aragog:.3f}\n\n'
        f'Initial T_magma:\n'
        f'  SPIDER: {spider["T_magma"].iloc[0]:.1f} K\n'
        f'  Aragog: {aragog["T_magma"].iloc[0]:.1f} K\n\n'
        f'Initial Phi:\n'
        f'  SPIDER: {spider[phi_key_spider].iloc[0]:.4f}\n'
        f'  Aragog: {aragog[phi_key_aragog].iloc[0]:.4f}\n\n'
        f'Iterations:\n'
        f'  SPIDER: {len(spider)}\n'
        f'  Aragog: {len(aragog)}'
    )
    ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    outpath = PLOTS_DIR / f'parity_M{mass:.1f}_{mix_tag}.png'
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f'Saved: {outpath}')


def main():
    parser = argparse.ArgumentParser(description='Plot SPIDER-Aragog parity results')
    parser.add_argument('--mass', type=float, nargs='+', default=[1.0])
    parser.add_argument('--mix-tag', choices=['nomix', 'mix'], default='nomix')
    args = parser.parse_args()

    for mass in args.mass:
        plot_comparison(mass, args.mix_tag)


if __name__ == '__main__':
    main()
