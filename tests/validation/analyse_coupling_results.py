#!/usr/bin/env python3
"""Analyse coupling validation matrix results.

Compares SPIDER vs Aragog, quantifies mass/CMF/volatile scaling,
checks energy conservation, and flags anomalies.

Usage:
    python analyse_coupling_results.py /path/to/validation/
    python analyse_coupling_results.py /tmp/habrok_transfer/validation/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Data loading ────────────────────────────────────────────────────


def load_all_cases(base_dir):
    """Load all helpfiles into a dict of DataFrames."""
    base = Path(base_dir)
    cases = {}
    for hf in sorted(base.glob('M*/runtime_helpfile.csv')):
        name = hf.parent.name
        try:
            df = pd.read_csv(hf, sep='\t')
            if len(df) >= 2:
                cases[name] = df
        except Exception:
            pass
    return cases


def parse_name(name):
    """Extract mass, CMF, interior, outgas, volatiles from case name."""
    parts = name.split('_')
    result = {'name': name}
    i = 0
    while i < len(parts):
        p = parts[i]
        if p.startswith('M'):
            result['mass'] = float(p[1:])
        elif p.startswith('CMF'):
            result['cmf'] = float(p[3:])
        elif p in ('spider', 'aragog'):
            result['interior'] = p
        elif p in ('calliope', 'atmodeller'):
            result['outgas'] = p
        elif p in ('dry', '1EO', '500ppmw'):
            vol_parts = [p]
            while i + 1 < len(parts) and parts[i + 1] not in ('spider', 'aragog', 'calliope', 'atmodeller'):
                i += 1
                vol_parts.append(parts[i])
            result['volatiles'] = '_'.join(vol_parts)
        i += 1
    return result


def build_summary(cases):
    """Build summary DataFrame from all cases."""
    rows = []
    for name, df in cases.items():
        info = parse_name(name)
        info['n_rows'] = len(df)
        info['T_final'] = df['T_magma'].iloc[-1]
        info['T_initial'] = df['T_magma'].iloc[0]
        info['Phi_final'] = df['Phi_global'].iloc[-1]
        info['Phi_initial'] = df['Phi_global'].iloc[0]
        info['t_final_Myr'] = df['Time'].iloc[-1] / 1e6
        if 'R_int' in df.columns:
            info['R_final'] = df['R_int'].iloc[-1] / 6.371e6
        if 'P_surf' in df.columns:
            info['P_final'] = df['P_surf'].iloc[-1]
        if 'F_int' in df.columns and 'F_atm' in df.columns:
            F_int = df['F_int'].values
            F_atm = df['F_atm'].values
            valid = (F_int > 0) & (F_atm > 0)
            if valid.any():
                rel_err = np.abs(F_int[valid] - F_atm[valid]) / np.maximum(F_int[valid], 1)
                info['energy_max_err'] = rel_err.max()
                info['energy_mean_err'] = rel_err.mean()
        rows.append(info)
    return pd.DataFrame(rows)


# ── Analysis functions ──────────────────────────────────────────────


def spider_vs_aragog(cases, summary, output_dir):
    """Compare SPIDER and Aragog for matched (mass, CMF, volatile) pairs."""
    output_dir = Path(output_dir) / 'spider_vs_aragog'
    output_dir.mkdir(parents=True, exist_ok=True)

    spider = summary[summary['interior'] == 'spider'].copy()
    aragog = summary[summary['interior'] == 'aragog'].copy()

    # Build matching key
    spider['key'] = spider.apply(
        lambda r: f"M{r['mass']}_CMF{r['cmf']}_{r.get('volatiles', 'dry')}", axis=1
    )
    aragog['key'] = aragog.apply(
        lambda r: f"M{r['mass']}_CMF{r['cmf']}_{r.get('volatiles', 'dry')}", axis=1
    )

    matched = spider.merge(aragog, on='key', suffixes=('_spider', '_aragog'))
    print(f'\nSPIDER vs Aragog: {len(matched)} matched pairs')

    if len(matched) == 0:
        return

    # Compare final T_magma
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    ax = axes[0]
    ax.scatter(matched['T_final_spider'], matched['T_final_aragog'], s=30, alpha=0.7)
    lims = [0, max(matched['T_final_spider'].max(), matched['T_final_aragog'].max()) * 1.1]
    ax.plot(lims, lims, 'k--', alpha=0.3, label='1:1')
    ax.set_xlabel('T_magma final (SPIDER) [K]')
    ax.set_ylabel('T_magma final (Aragog) [K]')
    ax.set_title('Final T_magma')
    ax.legend()

    ax = axes[1]
    ax.scatter(matched['Phi_final_spider'], matched['Phi_final_aragog'], s=30, alpha=0.7)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('Phi final (SPIDER)')
    ax.set_ylabel('Phi final (Aragog)')
    ax.set_title('Final melt fraction')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax = axes[2]
    if 'R_final_spider' in matched.columns and 'R_final_aragog' in matched.columns:
        ax.scatter(matched['R_final_spider'], matched['R_final_aragog'], s=30, alpha=0.7)
        lims = [
            min(matched['R_final_spider'].min(), matched['R_final_aragog'].min()) * 0.95,
            max(matched['R_final_spider'].max(), matched['R_final_aragog'].max()) * 1.05,
        ]
        ax.plot(lims, lims, 'k--', alpha=0.3)
        ax.set_xlabel('R_int (SPIDER) [R_Earth]')
        ax.set_ylabel('R_int (Aragog) [R_Earth]')
        ax.set_title('Final radius')

    fig.tight_layout()
    fig.savefig(output_dir / 'spider_vs_aragog_scatter.png', dpi=150)
    plt.close(fig)
    print('  Saved spider_vs_aragog_scatter.png')

    # T_magma evolution comparison for key cases
    for vol in ['dry', '1EO_H2O', '500ppmw_H']:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
        fig.suptitle(f'T_magma evolution: SPIDER vs Aragog ({vol})', fontsize=14)

        for i, mass in enumerate([0.5, 1.0, 2.0, 3.0, 5.0]):
            ax = axes[i]
            for interior, color, ls in [('spider', 'C0', '-'), ('aragog', 'C1', '--')]:
                case_name = f'M{mass}_CMF0.325_{interior}_calliope_{vol}'
                if case_name in cases:
                    df = cases[case_name]
                    t_myr = df['Time'].values / 1e6
                    t_myr = np.maximum(t_myr, 1e-7)  # avoid log(0)
                    ax.plot(t_myr, df['T_magma'], color=color, ls=ls,
                            lw=1.5, label=interior)
            ax.set_xscale('log')
            ax.set_xlabel('Time [Myr]')
            ax.set_title(f'{mass} M_Earth')
            if i == 0:
                ax.set_ylabel('T_magma [K]')
                ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / f'Tmagma_comparison_{vol}.png', dpi=150)
        plt.close(fig)
        print(f'  Saved Tmagma_comparison_{vol}.png')


def mass_scaling(summary, output_dir):
    """Analyse mass and CMF scaling of cooling."""
    output_dir = Path(output_dir) / 'scaling'
    output_dir.mkdir(parents=True, exist_ok=True)

    for interior in ['spider', 'aragog']:
        sub = summary[(summary['interior'] == interior) & (summary['outgas'] == 'calliope')]
        if len(sub) == 0:
            continue

        # T_final vs mass, colored by volatiles
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{interior.upper()}: Mass scaling (CMF=0.325)', fontsize=14)

        cmf_sub = sub[np.isclose(sub['cmf'], 0.325)]

        ax = axes[0]
        for vol, marker in [('dry', 'o'), ('1EO_H2O', 's'), ('500ppmw_H', '^')]:
            v = cmf_sub[cmf_sub['volatiles'] == vol]
            if len(v) > 0:
                ax.scatter(v['mass'], v['T_final'], marker=marker, s=60, label=vol)
        ax.set_xlabel('Mass [M_Earth]')
        ax.set_ylabel('T_magma final [K]')
        ax.set_title('Final temperature')
        ax.legend()

        ax = axes[1]
        for vol, marker in [('dry', 'o'), ('1EO_H2O', 's'), ('500ppmw_H', '^')]:
            v = cmf_sub[cmf_sub['volatiles'] == vol]
            if len(v) > 0:
                ax.scatter(v['mass'], v['Phi_final'], marker=marker, s=60, label=vol)
        ax.set_xlabel('Mass [M_Earth]')
        ax.set_ylabel('Phi final')
        ax.set_title('Final melt fraction')

        ax = axes[2]
        if 'R_final' in cmf_sub.columns:
            for vol, marker in [('dry', 'o'), ('1EO_H2O', 's'), ('500ppmw_H', '^')]:
                v = cmf_sub[cmf_sub['volatiles'] == vol]
                if len(v) > 0:
                    ax.scatter(v['mass'], v['R_final'], marker=marker, s=60, label=vol)
            ax.set_xlabel('Mass [M_Earth]')
            ax.set_ylabel('R_int [R_Earth]')
            ax.set_title('Final radius')

        fig.tight_layout()
        fig.savefig(output_dir / f'mass_scaling_{interior}.png', dpi=150)
        plt.close(fig)
        print(f'  Saved mass_scaling_{interior}.png')


def volatile_effect(cases, summary, output_dir):
    """Compare dry vs volatile cooling curves."""
    output_dir = Path(output_dir) / 'volatile_effect'
    output_dir.mkdir(parents=True, exist_ok=True)

    for interior in ['spider', 'aragog']:
        fig, axes = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
        fig.suptitle(f'{interior.upper()}: Volatile effect on cooling (CMF=0.325)', fontsize=14)

        for i, mass in enumerate([0.5, 1.0, 2.0, 3.0, 5.0]):
            ax = axes[i]
            for vol, color in [('dry', 'C0'), ('1EO_H2O', 'C1'), ('500ppmw_H', 'C2')]:
                case_name = f'M{mass}_CMF0.325_{interior}_calliope_{vol}'
                if case_name in cases:
                    df = cases[case_name]
                    t_myr = np.maximum(df['Time'].values / 1e6, 1e-7)
                    ax.plot(t_myr, df['T_magma'], color=color, lw=1.5, label=vol)
            ax.set_xscale('log')
            ax.set_xlabel('Time [Myr]')
            ax.set_title(f'{mass} M_Earth')
            if i == 0:
                ax.set_ylabel('T_magma [K]')
                ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(output_dir / f'volatile_effect_{interior}.png', dpi=150)
        plt.close(fig)
        print(f'  Saved volatile_effect_{interior}.png')


def energy_check(summary, output_dir):
    """Check energy conservation across all cases."""
    output_dir = Path(output_dir) / 'diagnostics'
    output_dir.mkdir(parents=True, exist_ok=True)

    if 'energy_max_err' not in summary.columns:
        print('  No energy data available')
        return

    valid = summary.dropna(subset=['energy_max_err'])
    if len(valid) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    for interior, color in [('spider', 'C0'), ('aragog', 'C1')]:
        sub = valid[valid['interior'] == interior]
        if len(sub) > 0:
            ax.scatter(sub['mass'], sub['energy_max_err'] * 100,
                       label=interior, alpha=0.7, s=30, color=color)
    ax.set_xlabel('Mass [M_Earth]')
    ax.set_ylabel('Max |F_int - F_atm| / F_int [%]')
    ax.set_title('Energy conservation check')
    ax.axhline(1, color='r', ls='--', alpha=0.3, label='1% threshold')
    ax.legend()
    ax.set_yscale('log')
    fig.savefig(output_dir / 'energy_conservation.png', dpi=150)
    plt.close(fig)
    print('  Saved energy_conservation.png')


def anomaly_report(summary, output_dir):
    """Flag anomalous cases."""
    output_dir = Path(output_dir) / 'diagnostics'
    output_dir.mkdir(parents=True, exist_ok=True)

    anomalies = []

    # Warming: T_final > T_initial
    warming = summary[summary['T_final'] > summary['T_initial'] + 10]
    for _, row in warming.iterrows():
        anomalies.append(f"WARMING: {row['name']} T={row['T_initial']:.0f}->{row['T_final']:.0f} K")

    # Phi out of range
    bad_phi = summary[(summary['Phi_final'] < -0.01) | (summary['Phi_final'] > 1.01)]
    for _, row in bad_phi.iterrows():
        anomalies.append(f"PHI_RANGE: {row['name']} Phi={row['Phi_final']:.4f}")

    # Very few rows (< 5)
    few_rows = summary[summary['n_rows'] < 5]
    for _, row in few_rows.iterrows():
        anomalies.append(f"FEW_ROWS: {row['name']} only {row['n_rows']} rows")

    report_path = output_dir / 'anomaly_report.txt'
    with open(report_path, 'w') as f:
        f.write(f'Anomaly Report: {len(anomalies)} issues found\n')
        f.write('=' * 60 + '\n\n')
        for a in anomalies:
            f.write(a + '\n')
        if not anomalies:
            f.write('No anomalies detected.\n')

    print(f'  {len(anomalies)} anomalies -> {report_path}')
    for a in anomalies[:10]:
        print(f'    {a}')


# ── Main ────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='Analyse coupling validation results')
    parser.add_argument('base_dir', help='Directory with validation case subdirs')
    parser.add_argument('--output', default=None, help='Output dir for plots (default: base_dir/analysis)')
    args = parser.parse_args()

    base = Path(args.base_dir)
    output_dir = Path(args.output) if args.output else base / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading cases from {base}...')
    cases = load_all_cases(base)
    print(f'  Loaded {len(cases)} cases')

    summary = build_summary(cases)
    summary.to_csv(output_dir / 'summary.csv', index=False)
    print(f'  Summary saved to {output_dir / "summary.csv"}')

    # Print overview
    for interior in ['spider', 'aragog']:
        sub = summary[summary['interior'] == interior]
        if len(sub) > 0:
            print(f'\n  {interior.upper()}: {len(sub)} cases, '
                  f'T range {sub["T_final"].min():.0f}-{sub["T_final"].max():.0f} K, '
                  f'Phi range {sub["Phi_final"].min():.3f}-{sub["Phi_final"].max():.3f}')

    print('\n--- SPIDER vs Aragog ---')
    spider_vs_aragog(cases, summary, output_dir)

    print('\n--- Mass scaling ---')
    mass_scaling(summary, output_dir)

    print('\n--- Volatile effect ---')
    volatile_effect(cases, summary, output_dir)

    print('\n--- Energy conservation ---')
    energy_check(summary, output_dir)

    print('\n--- Anomaly report ---')
    anomaly_report(summary, output_dir)

    print(f'\nAll analysis outputs in {output_dir}')


if __name__ == '__main__':
    main()
