#!/usr/bin/env python3
"""Plot validation results from the coupling matrix.

Reads runtime_helpfile.csv from each case directory and generates
per-case and summary plots.

Usage:
    python plot_coupling_results.py output/validation/
    python plot_coupling_results.py output/validation/ --case M1.0_CMF0.325_spider_calliope_dry
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_helpfile(case_dir):
    """Load runtime_helpfile.csv from a case directory."""
    hf_path = Path(case_dir) / 'runtime_helpfile.csv'
    if not hf_path.is_file():
        return None
    try:
        df = pd.read_csv(hf_path, sep=r'\s+|,', engine='python')
        if len(df) < 2:
            return None
        return df
    except Exception:
        return None


def parse_case_name(name):
    """Parse case name into components."""
    parts = name.split('_')
    result = {}
    for p in parts:
        if p.startswith('M'):
            result['mass'] = float(p[1:])
        elif p.startswith('CMF'):
            result['cmf'] = float(p[3:])
        elif p in ('spider', 'aragog'):
            result['interior'] = p
        elif p in ('calliope', 'atmodeller'):
            result['outgas'] = p
        elif p in ('dry', '1EO', '500ppmw'):
            result['volatiles'] = p
    return result


# ── Per-case plots ──────────────────────────────────────────────────


def plot_case(case_dir, case_name, output_dir=None):
    """Generate diagnostic plots for a single case."""
    df = load_helpfile(case_dir)
    if df is None:
        print(f'  No data for {case_name}')
        return

    if output_dir is None:
        output_dir = Path(case_dir) / 'plots'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    time = df['Time'].values
    time_myr = time / 1e6

    # 1. T_magma evolution
    if 'T_magma' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_myr, df['T_magma'], 'k-', lw=1.5)
        ax.set_xlabel('Time [Myr]')
        ax.set_ylabel('T_magma [K]')
        ax.set_title(f'{case_name}: Magma temperature')
        ax.set_xscale('log')
        fig.savefig(output_dir / 'T_magma_evolution.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 2. Phi_global evolution
    if 'Phi_global' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_myr, df['Phi_global'], 'b-', lw=1.5)
        ax.set_xlabel('Time [Myr]')
        ax.set_ylabel('Melt fraction')
        ax.set_title(f'{case_name}: Global melt fraction')
        ax.set_xscale('log')
        ax.set_ylim(-0.05, 1.05)
        fig.savefig(output_dir / 'Phi_evolution.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 3. R_int evolution
    R_earth = 6.371e6
    if 'R_int' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_myr, df['R_int'] / R_earth, 'r-', lw=1.5)
        ax.set_xlabel('Time [Myr]')
        ax.set_ylabel('R_int [R_Earth]')
        ax.set_title(f'{case_name}: Interior radius')
        ax.set_xscale('log')
        fig.savefig(output_dir / 'R_int_evolution.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 4. Surface pressure
    if 'P_surf' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(time_myr, df['P_surf'], 'g-', lw=1.5)
        ax.set_xlabel('Time [Myr]')
        ax.set_ylabel('P_surf [bar]')
        ax.set_title(f'{case_name}: Surface pressure')
        ax.set_xscale('log')
        ax.set_yscale('log')
        fig.savefig(output_dir / 'P_surf_evolution.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 5. Energy balance
    flux_cols = [c for c in ['F_int', 'F_atm', 'F_olr'] if c in df.columns]
    if flux_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        for col in flux_cols:
            ax.plot(time_myr, np.abs(df[col]), label=col, lw=1.5)
        ax.set_xlabel('Time [Myr]')
        ax.set_ylabel('|Flux| [W/m²]')
        ax.set_title(f'{case_name}: Energy fluxes')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        fig.savefig(output_dir / 'energy_balance.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f'  Plots saved to {output_dir}')


# ── Summary plots ───────────────────────────────────────────────────


def plot_summary(base_dir):
    """Generate cross-case summary plots."""
    base_dir = Path(base_dir)
    summary_dir = base_dir / 'summary_plots'
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Collect final states from all cases
    results = []
    for case_dir in sorted(base_dir.iterdir()):
        if not case_dir.is_dir() or case_dir.name in ('configs', 'summary_plots'):
            continue
        df = load_helpfile(case_dir)
        if df is None or len(df) < 3:
            continue
        info = parse_case_name(case_dir.name)
        info['name'] = case_dir.name
        info['n_rows'] = len(df)
        info['T_final'] = float(df['T_magma'].iloc[-1]) if 'T_magma' in df.columns else np.nan
        info['Phi_final'] = float(df['Phi_global'].iloc[-1]) if 'Phi_global' in df.columns else np.nan
        info['t_final_Myr'] = float(df['Time'].iloc[-1]) / 1e6
        if 'R_int' in df.columns:
            info['R_final'] = float(df['R_int'].iloc[-1]) / 6.371e6
        results.append(info)

    if not results:
        print('No completed cases found.')
        return

    rdf = pd.DataFrame(results)
    print(f'Summary: {len(rdf)} completed cases')

    # Mass-Radius diagram
    if 'mass' in rdf.columns and 'R_final' in rdf.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        for interior in rdf['interior'].unique():
            mask = rdf['interior'] == interior
            ax.scatter(rdf.loc[mask, 'mass'], rdf.loc[mask, 'R_final'],
                       label=interior, s=30, alpha=0.7)
        ax.set_xlabel('Mass [M_Earth]')
        ax.set_ylabel('R_int [R_Earth]')
        ax.set_title('Mass-Radius at final timestep')
        ax.legend()
        fig.savefig(summary_dir / 'mass_radius_diagram.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Solidification time heatmap
    if 'Phi_final' in rdf.columns:
        solidified = rdf[rdf['Phi_final'] < 0.01]
        if len(solidified) > 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(solidified['mass'], solidified['t_final_Myr'], s=40)
            ax.set_xlabel('Mass [M_Earth]')
            ax.set_ylabel('Solidification time [Myr]')
            ax.set_title('Time to solidification (Phi < 0.01)')
            ax.set_yscale('log')
            fig.savefig(summary_dir / 'solidification_time.png', dpi=150, bbox_inches='tight')
            plt.close(fig)

    print(f'Summary plots saved to {summary_dir}')

    return rdf


# ── CLI ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='Plot coupling validation results')
    parser.add_argument('base_dir', help='Base output directory (e.g., output/validation/)')
    parser.add_argument('--case', help='Plot a single case by name')
    parser.add_argument('--summary-only', action='store_true', help='Only summary plots')

    args = parser.parse_args()
    base = Path(args.base_dir)

    if args.case:
        case_dir = base / args.case
        plot_case(str(case_dir), args.case)
    elif args.summary_only:
        plot_summary(str(base))
    else:
        # Plot all cases + summary
        for case_dir in sorted(base.iterdir()):
            if case_dir.is_dir() and case_dir.name not in ('configs', 'summary_plots'):
                plot_case(str(case_dir), case_dir.name)
        plot_summary(str(base))


if __name__ == '__main__':
    main()
