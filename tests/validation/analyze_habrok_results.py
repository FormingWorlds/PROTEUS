#!/usr/bin/env python
"""Analyze results from the Habrok validation grid.

Reads all case directories, extracts key metrics from status files and
runtime helpfiles, and produces:
  1. A summary CSV with all case metrics
  2. A text summary of completions by block
  3. An AW vs ZAL paired comparison table
  4. A stability frontier map (max mass per CMF)

Usage
-----
    python analyze_habrok_results.py --outdir /scratch/$USER/habrok_validation

    # Include diagnostic plots (requires matplotlib)
    python analyze_habrok_results.py --outdir /scratch/$USER/habrok_validation --plot
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Case name parser ─────────────────────────────────────────────

# Pattern: {block}_M{mass}_CMF{cmf}_{tag}[_Tlin][_P2u{interval}][_n{levels}][_S{entropy}]
CASE_PATTERN = re.compile(
    r'^(?P<block>[A-HJ-Z])_M(?P<mass>[\d.]+)_CMF(?P<cmf>[\d.]+)_(?P<tag>AW|ZAL)'
    r'(?:_(?P<tmode>Tlin))?'
    r'(?:_P2u(?P<update>\d+))?'
    r'(?:_n(?P<nlevels>\d+))?'
    r'(?:_S(?P<entropy>\d+))?$'
)


def parse_case_name(name: str) -> dict:
    """Extract parameters from a case directory name.

    Parameters
    ----------
    name : str
        Case directory name (e.g. 'A_M1.0_CMF0.325_ZAL').

    Returns
    -------
    dict
        Parsed parameters, or empty dict if name doesn't match.
    """
    m = CASE_PATTERN.match(name)
    if not m:
        return {}
    return {
        'block': m.group('block'),
        'mass': float(m.group('mass')),
        'cmf': float(m.group('cmf')),
        'mode': m.group('tag'),
        'temperature_mode': m.group('tmode') or 'auto',
        'update_interval': int(m.group('update')) if m.group('update') else 0,
        'num_levels': int(m.group('nlevels')) if m.group('nlevels') else 60,
        'ini_entropy': int(m.group('entropy')) if m.group('entropy') else 3000,
    }


# ── Result extraction ────────────────────────────────────────────


def read_status(case_dir: Path) -> tuple[int, str]:
    """Read the status file for a case.

    Parameters
    ----------
    case_dir : Path
        Path to the case output directory.

    Returns
    -------
    tuple[int, str]
        (iteration_count, exit_reason) or (-1, "not started") if missing.
    """
    status_file = case_dir / 'status'
    if not status_file.exists():
        return -1, 'not started'

    lines = status_file.read_text().strip().splitlines()
    if len(lines) < 2:
        return -1, 'malformed status'

    try:
        n_iters = int(lines[0].strip())
    except ValueError:
        n_iters = -1

    return n_iters, lines[1].strip()


def read_helpfile_final(case_dir: Path) -> dict:
    """Read the last row of the runtime helpfile.

    Parameters
    ----------
    case_dir : Path
        Path to the case output directory.

    Returns
    -------
    dict
        Final-state metrics, or empty dict if helpfile missing/empty.
    """
    hf_path = case_dir / 'runtime_helpfile.csv'
    if not hf_path.exists():
        return {}

    try:
        df = pd.read_csv(hf_path, sep='\t')
        if df.empty:
            return {}
        row = df.iloc[-1]

        result = {}
        # Map column names to output keys
        col_map = {
            'Time': 't_final_yr',
            'T_magma': 'T_magma_K',
            'Phi_global': 'Phi_global',
            'F_int': 'F_int_Wm2',
            'RF_depth': 'RF_depth',
            'R_int': 'R_int_m',
            'M_int': 'M_int_kg',
        }
        for col, key in col_map.items():
            if col in row.index:
                result[key] = float(row[col])

        result['n_helpfile_rows'] = len(df)
        return result

    except Exception:
        return {}


def count_eos_warnings(case_dir: Path) -> int:
    """Count EOS out-of-range warnings in SPIDER logs.

    Searches for spider_recent.log in the case data directory.

    Parameters
    ----------
    case_dir : Path
        Path to the case output directory.

    Returns
    -------
    int
        Number of EOS warning lines, or -1 if log not found.
    """
    log_path = case_dir / 'spider_recent.log'
    if not log_path.exists():
        # Try alternate location
        log_path = case_dir / 'data' / 'spider_recent.log'
    if not log_path.exists():
        return -1

    count = 0
    try:
        for line in log_path.read_text().splitlines():
            if 'WARNING [EOS' in line or 'WARNING [EOS table]' in line:
                count += 1
    except Exception:
        return -1
    return count


# ── Analysis ─────────────────────────────────────────────────────


def collect_results(outdir: Path) -> pd.DataFrame:
    """Collect results from all case directories.

    Parameters
    ----------
    outdir : Path
        Root output directory containing cases/ subdirectory.

    Returns
    -------
    pd.DataFrame
        One row per case with all metrics.
    """
    cases_dir = outdir / 'cases'
    if not cases_dir.exists():
        print(f'ERROR: No cases directory found at {cases_dir}', file=sys.stderr)
        sys.exit(1)

    rows = []
    for case_dir in sorted(cases_dir.iterdir()):
        if not case_dir.is_dir():
            continue

        name = case_dir.name
        params = parse_case_name(name)
        if not params:
            continue

        n_iters, exit_reason = read_status(case_dir)
        hf = read_helpfile_final(case_dir)
        n_eos_warn = count_eos_warnings(case_dir)

        completed = 'Completed' in exit_reason
        errored = 'Error' in exit_reason
        timed_out = n_iters >= 0 and not completed and not errored

        row = {
            'name': name,
            **params,
            'n_iters': n_iters,
            'exit_reason': exit_reason,
            'completed': completed,
            'errored': errored,
            'timed_out': timed_out,
            'n_eos_warnings': n_eos_warn,
            **hf,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def print_block_summary(df: pd.DataFrame) -> None:
    """Print completion summary grouped by block."""
    block_labels = {
        'A': 'AW vs ZAL baseline',
        'H': 'Adiabatic vs linear T control',
        'D': 'Phase 2 feedback',
        'E': 'Resolution convergence',
        'F': 'Initial entropy sensitivity',
    }

    print('\n' + '=' * 70)
    print('BLOCK SUMMARY')
    print('=' * 70)

    for block_id in sorted(df['block'].unique()):
        bd = df[df['block'] == block_id]
        n_total = len(bd)
        n_done = bd['completed'].sum()
        n_err = bd['errored'].sum()
        n_todo = n_total - n_done - n_err
        label = block_labels.get(block_id, '')

        print(f'\n  Block {block_id} -- {label}')
        print(f'    {n_done}/{n_total} completed, {n_err} errors, {n_todo} pending/timeout')

        # Show failed cases
        failed = bd[bd['errored']]
        if not failed.empty:
            for _, row in failed.iterrows():
                print(f'    FAIL: {row["name"]}  ({row["exit_reason"]})')


def print_aw_vs_zal_comparison(df: pd.DataFrame) -> None:
    """Print paired comparison of AW vs ZAL results."""
    block_a = df[df['block'] == 'A'].copy()
    if block_a.empty:
        return

    print('\n' + '=' * 70)
    print('AW vs ZAL PAIRED COMPARISON (Block A)')
    print('=' * 70)
    print(
        f'  {"Mass":>5s}  {"CMF":>5s}  '
        f'{"AW t_solid":>12s}  {"ZAL t_solid":>12s}  {"Ratio":>7s}  '
        f'{"AW T_mag":>10s}  {"ZAL T_mag":>10s}  {"AW":>4s}  {"ZAL":>4s}'
    )
    print('  ' + '-' * 84)

    for mass in sorted(block_a['mass'].unique()):
        for cmf in sorted(block_a['cmf'].unique()):
            aw = block_a[
                (block_a['mass'] == mass) & (block_a['cmf'] == cmf) & (block_a['mode'] == 'AW')
            ]
            zal = block_a[
                (block_a['mass'] == mass) & (block_a['cmf'] == cmf) & (block_a['mode'] == 'ZAL')
            ]

            if aw.empty or zal.empty:
                continue

            aw = aw.iloc[0]
            zal = zal.iloc[0]

            aw_t = f'{aw.get("t_final_yr", float("nan")):.0f}' if aw['completed'] else '---'
            zal_t = f'{zal.get("t_final_yr", float("nan")):.0f}' if zal['completed'] else '---'

            if aw['completed'] and zal['completed']:
                t_aw = aw.get('t_final_yr', float('nan'))
                t_zal = zal.get('t_final_yr', float('nan'))
                ratio = f'{t_aw / t_zal:.2f}' if t_zal > 0 else '---'
            else:
                ratio = '---'

            aw_tm = f'{aw.get("T_magma_K", float("nan")):.0f}' if aw['completed'] else '---'
            zal_tm = f'{zal.get("T_magma_K", float("nan")):.0f}' if zal['completed'] else '---'

            aw_status = 'OK' if aw['completed'] else ('ERR' if aw['errored'] else '...')
            zal_status = 'OK' if zal['completed'] else ('ERR' if zal['errored'] else '...')

            print(
                f'  {mass:5.1f}  {cmf:5.3f}  '
                f'{aw_t:>12s}  {zal_t:>12s}  {ratio:>7s}  '
                f'{aw_tm:>10s}  {zal_tm:>10s}  {aw_status:>4s}  {zal_status:>4s}'
            )


def print_stability_frontier(df: pd.DataFrame) -> None:
    """Print the maximum stable mass for each CMF and mode."""
    print('\n' + '=' * 70)
    print('STABILITY FRONTIER (max mass that completed)')
    print('=' * 70)

    completed = df[df['completed']]
    if completed.empty:
        print('  No completed cases.')
        return

    print(f'  {"CMF":>5s}  {"AW max mass":>12s}  {"ZAL max mass":>13s}')
    print('  ' + '-' * 34)

    all_cmfs = sorted(df['cmf'].unique())
    for cmf in all_cmfs:
        aw_done = completed[(completed['cmf'] == cmf) & (completed['mode'] == 'AW')]
        zal_done = completed[(completed['cmf'] == cmf) & (completed['mode'] == 'ZAL')]

        aw_max = f'{aw_done["mass"].max():.1f} M_E' if not aw_done.empty else '---'
        zal_max = f'{zal_done["mass"].max():.1f} M_E' if not zal_done.empty else '---'

        print(f'  {cmf:5.3f}  {aw_max:>12s}  {zal_max:>13s}')


def print_phase2_comparison(df: pd.DataFrame) -> None:
    """Compare Phase 1 vs Phase 2 results."""
    block_d = df[df['block'] == 'D']
    if block_d.empty:
        return

    # Find matching Phase 1 cases in Block A
    block_a_zal = df[(df['block'] == 'A') & (df['mode'] == 'ZAL')]

    print('\n' + '=' * 70)
    print('PHASE 2 FEEDBACK COMPARISON (Block D vs Block A ZAL)')
    print('=' * 70)
    print(
        f'  {"Mass":>5s}  '
        f'{"P1 t_solid":>12s}  {"P2 t_solid":>12s}  {"Diff%":>7s}  '
        f'{"P1 T_mag":>10s}  {"P2 T_mag":>10s}'
    )
    print('  ' + '-' * 64)

    for _, p2 in block_d.iterrows():
        p1 = block_a_zal[
            (block_a_zal['mass'] == p2['mass']) & (block_a_zal['cmf'] == p2['cmf'])
        ]
        if p1.empty:
            continue
        p1 = p1.iloc[0]

        p1_t = f'{p1.get("t_final_yr", float("nan")):.0f}' if p1['completed'] else '---'
        p2_t = f'{p2.get("t_final_yr", float("nan")):.0f}' if p2['completed'] else '---'

        if p1['completed'] and p2['completed']:
            t1 = p1.get('t_final_yr', float('nan'))
            t2 = p2.get('t_final_yr', float('nan'))
            diff = f'{100 * (t2 - t1) / t1:+.1f}%' if t1 > 0 else '---'
        else:
            diff = '---'

        p1_tm = f'{p1.get("T_magma_K", float("nan")):.0f}' if p1['completed'] else '---'
        p2_tm = f'{p2.get("T_magma_K", float("nan")):.0f}' if p2['completed'] else '---'

        print(
            f'  {p2["mass"]:5.1f}  '
            f'{p1_t:>12s}  {p2_t:>12s}  {diff:>7s}  '
            f'{p1_tm:>10s}  {p2_tm:>10s}'
        )


def print_adiabat_comparison(df: pd.DataFrame) -> None:
    """Compare adiabatic vs linear T mode results (Block H vs Block A)."""
    block_h = df[df['block'] == 'H']
    if block_h.empty:
        return

    # Find matching adiabatic cases from Block A
    block_a_zal = df[(df['block'] == 'A') & (df['mode'] == 'ZAL')]

    print('\n' + '=' * 70)
    print('ADIABATIC vs LINEAR T MODE (Block H vs Block A ZAL)')
    print('=' * 70)
    print(
        f'  {"Mass":>5s}  '
        f'{"Adiab t_sol":>12s}  {"Linear t_sol":>13s}  {"Diff%":>7s}  '
        f'{"Adiab T_mag":>12s}  {"Linear T_mag":>13s}'
    )
    print('  ' + '-' * 70)

    for _, h_row in block_h.iterrows():
        a_match = block_a_zal[
            (np.isclose(block_a_zal['mass'], h_row['mass']))
            & (np.isclose(block_a_zal['cmf'], h_row['cmf']))
        ]
        if a_match.empty:
            continue
        a_row = a_match.iloc[0]

        a_t = f'{a_row.get("t_final_yr", float("nan")):.0f}' if a_row['completed'] else '---'
        h_t = f'{h_row.get("t_final_yr", float("nan")):.0f}' if h_row['completed'] else '---'

        if a_row['completed'] and h_row['completed']:
            t_a = a_row.get('t_final_yr', float('nan'))
            t_h = h_row.get('t_final_yr', float('nan'))
            diff = f'{100 * (t_h - t_a) / t_a:+.1f}%' if t_a > 0 else '---'
        else:
            diff = '---'

        a_tm = f'{a_row.get("T_magma_K", float("nan")):.0f}' if a_row['completed'] else '---'
        h_tm = f'{h_row.get("T_magma_K", float("nan")):.0f}' if h_row['completed'] else '---'

        print(
            f'  {h_row["mass"]:5.1f}  '
            f'{a_t:>12s}  {h_t:>13s}  {diff:>7s}  '
            f'{a_tm:>12s}  {h_tm:>13s}'
        )


def print_resolution_convergence(df: pd.DataFrame) -> None:
    """Show resolution convergence results."""
    block_e = df[df['block'] == 'E']
    if block_e.empty:
        return

    # Also include the matching default-resolution case from Block A
    ref = df[
        (df['block'] == 'A')
        & (df['mode'] == 'ZAL')
        & (np.isclose(df['mass'], 3.0))
        & (np.isclose(df['cmf'], 0.325))
    ]

    combined = pd.concat([ref, block_e]).sort_values('num_levels')

    print('\n' + '=' * 70)
    print('RESOLUTION CONVERGENCE (Block E, M=3.0, CMF=0.325, ZAL)')
    print('=' * 70)
    print(
        f'  {"num_levels":>10s}  {"t_solid":>10s}  {"T_magma":>10s}  {"Phi_final":>10s}  {"Status":>8s}'
    )
    print('  ' + '-' * 54)

    for _, row in combined.iterrows():
        nl = int(row['num_levels'])
        if row['completed']:
            t = f'{row.get("t_final_yr", float("nan")):.0f}'
            tm = f'{row.get("T_magma_K", float("nan")):.0f}'
            phi = f'{row.get("Phi_global", float("nan")):.4f}'
            status = 'OK'
        else:
            t = tm = phi = '---'
            status = 'ERR' if row['errored'] else '...'

        print(f'  {nl:10d}  {t:>10s}  {tm:>10s}  {phi:>10s}  {status:>8s}')


# ── Plotting ─────────────────────────────────────────────────────


def make_plots(df: pd.DataFrame, outdir: Path) -> None:
    """Generate diagnostic plots.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe.
    outdir : Path
        Directory to save plots.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('WARNING: matplotlib not available, skipping plots')
        return

    plot_dir = outdir / 'plots'
    plot_dir.mkdir(exist_ok=True)

    # 1. Completion map: mass x CMF, panels for AW and ZAL
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, mode in zip(axes, ['AW', 'ZAL']):
        mode_df = df[df['mode'] == mode]
        masses = sorted(mode_df['mass'].unique())
        cmfs = sorted(mode_df['cmf'].unique())

        status_map = np.full((len(cmfs), len(masses)), np.nan)
        for _, row in mode_df.iterrows():
            mi = masses.index(row['mass'])
            ci = cmfs.index(row['cmf'])
            if row['completed']:
                status_map[ci, mi] = 1.0
            elif row['errored']:
                status_map[ci, mi] = 0.0
            else:
                status_map[ci, mi] = 0.5

        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(['#d32f2f', '#ff9800', '#4caf50'])
        ax.imshow(
            status_map,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation='nearest',
        )
        ax.set_xticks(range(len(masses)))
        ax.set_xticklabels([f'{m:.1f}' for m in masses])
        ax.set_yticks(range(len(cmfs)))
        ax.set_yticklabels([f'{c:.3f}' for c in cmfs])
        ax.set_xlabel('Mass [M_E]')
        ax.set_ylabel('CMF')
        ax.set_title(f'{mode} mode')

        # Annotate with t_solid or exit reason
        for _, row in mode_df.iterrows():
            mi = masses.index(row['mass'])
            ci = cmfs.index(row['cmf'])
            if row['completed'] and 't_final_yr' in row and not np.isnan(row['t_final_yr']):
                label = f'{row["t_final_yr"]:.0f}'
            elif row['errored']:
                label = 'ERR'
            elif row.get('n_iters', -1) < 0:
                label = 'N/A'
            else:
                label = '...'
            ax.text(mi, ci, label, ha='center', va='center', fontsize=7, color='white')

    fig.suptitle('Completion status (green=done, orange=timeout, red=error)', y=1.02)
    fig.tight_layout()
    fig.savefig(plot_dir / 'completion_map.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2. t_solid vs mass for each CMF (AW vs ZAL overlay)
    completed = df[df['completed']].copy()
    if not completed.empty and 't_final_yr' in completed.columns:
        fig, ax = plt.subplots(figsize=(10, 6))

        cmfs = sorted(completed['cmf'].unique())
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(cmfs)))

        for cmf, color in zip(cmfs, colors):
            for mode, ls, marker in [('AW', '--', 's'), ('ZAL', '-', 'o')]:
                subset = completed[
                    (completed['cmf'] == cmf) & (completed['mode'] == mode)
                ].sort_values('mass')
                if not subset.empty:
                    ax.plot(
                        subset['mass'],
                        subset['t_final_yr'],
                        ls=ls,
                        marker=marker,
                        color=color,
                        label=f'CMF={cmf:.2f} {mode}',
                        markersize=5,
                    )

        ax.set_xlabel('Planet mass [M_E]')
        ax.set_ylabel('Solidification time [yr]')
        ax.set_yscale('log')
        ax.legend(fontsize=7, ncol=2)
        ax.set_title('Solidification time vs planet mass')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / 't_solid_vs_mass.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f'  Plots saved to {plot_dir}/')


# ── Main ─────────────────────────────────────────────────────────


def main():
    """Analyze Habrok validation grid results."""
    parser = argparse.ArgumentParser(
        description='Analyze Habrok validation grid results',
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        required=True,
        help='Root output directory (same as used for generate_habrok_grid.py)',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate diagnostic plots (requires matplotlib)',
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=None,
        help='Output CSV path (default: <outdir>/results_summary.csv)',
    )
    args = parser.parse_args()

    outdir = args.outdir.resolve()

    # Collect results
    print(f'Collecting results from {outdir}/cases/ ...')
    df = collect_results(outdir)

    if df.empty:
        print('No results found.')
        sys.exit(1)

    # Write CSV
    csv_path = args.csv or (outdir / 'results_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'  Summary CSV: {csv_path} ({len(df)} cases)')

    # Print summaries
    n_total = len(df)
    n_done = df['completed'].sum()
    n_err = df['errored'].sum()
    n_pending = n_total - n_done - n_err

    print(
        f'\n  Overall: {n_done}/{n_total} completed, {n_err} errors, {n_pending} pending/timeout'
    )

    print_block_summary(df)
    print_aw_vs_zal_comparison(df)
    print_stability_frontier(df)
    print_phase2_comparison(df)
    print_adiabat_comparison(df)
    print_resolution_convergence(df)

    # Plots
    if args.plot:
        make_plots(df, outdir)

    print('\n' + '=' * 70)
    print('Done.')


if __name__ == '__main__':
    main()
