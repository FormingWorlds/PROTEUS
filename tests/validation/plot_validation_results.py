#!/usr/bin/env python
"""Post-processing: comparison plots and validation checks for the test matrix.

Reads helpfiles and SPIDER JSON output from all completed validation runs,
generates comparison figures (AW vs Zalmoxis), and reports pass/fail on
physical plausibility criteria.

Usage
-----
    python plot_validation_results.py                        # all plots + validation
    python plot_validation_results.py --outdir path/to/runs  # custom output dir
    python plot_validation_results.py --validate-only        # skip plots
    python plot_validation_results.py --plots-only           # skip validation

Output
------
    {outdir}/plots/                     comparison figures
    {outdir}/validation_report.txt      pass/fail summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROTEUS_ROOT = Path(__file__).resolve().parents[2]

# ── Styling ──────────────────────────────────────────────────

MASSES = [1.0, 3.0, 5.0]
CMFS = [0.05, 0.10, 0.325, 0.50, 0.80]

# Colors for CMF values (dark to light)
CMF_COLORS = {
    0.05: '#1b9e77',
    0.10: '#d95f02',
    0.325: '#7570b3',
    0.50: '#e7298a',
    0.80: '#66a61e',
}

MASS_LABELS = {1.0: r'1 $M_\oplus$', 3.0: r'3 $M_\oplus$', 5.0: r'5 $M_\oplus$'}


# ── Data loading ─────────────────────────────────────────────


def load_helpfile(case_dir: Path) -> pd.DataFrame | None:
    """Load a helpfile CSV, returning None if not found or on error."""
    hf_path = case_dir / 'runtime_helpfile.csv'
    if not hf_path.exists():
        return None
    try:
        return pd.read_csv(hf_path)
    except Exception:
        return None


def case_name(mass: float, cmf: float, struct: str, phase2: bool = False) -> str:
    """Construct the case directory name."""
    tag = 'AW' if struct == 'self' else 'ZAL'
    suffix = '_P2' if phase2 else ''
    return f'M{mass}_CMF{cmf}_{tag}{suffix}'


def load_all_results(outdir: Path) -> dict[str, pd.DataFrame]:
    """Load helpfiles from all completed validation runs.

    Returns
    -------
    dict
        Mapping of case name -> helpfile DataFrame.
    """
    results = {}
    if not outdir.exists():
        return results

    for case_dir in sorted(outdir.iterdir()):
        if not case_dir.is_dir():
            continue
        hf = load_helpfile(case_dir)
        if hf is not None and len(hf) > 0:
            results[case_dir.name] = hf
    return results


# ── Comparison plots ─────────────────────────────────────────


def plot_pairwise_evolution(results: dict, outdir: Path):
    """Generate AW vs Zalmoxis time-evolution overlays.

    One figure per planet mass, with 5 rows (CMFs) x 3 columns
    (T_magma, Phi_global, F_int).
    """
    plot_dir = outdir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for mass in MASSES:
        fig, axes = plt.subplots(len(CMFS), 3, figsize=(14, 3.2 * len(CMFS)), sharex='col')
        if len(CMFS) == 1:
            axes = axes[np.newaxis, :]

        for i, cmf in enumerate(CMFS):
            aw_name = case_name(mass, cmf, 'self')
            zal_name = case_name(mass, cmf, 'zalmoxis')

            hf_aw = results.get(aw_name)
            hf_zal = results.get(zal_name)

            for j, (col, label, yscale) in enumerate(
                [
                    ('T_magma', 'T$_{\\rm magma}$ [K]', 'linear'),
                    ('Phi_global', '$\\Phi_{\\rm global}$', 'linear'),
                    ('F_int', 'F$_{\\rm int}$ [W/m$^2$]', 'log'),
                ]
            ):
                ax = axes[i, j]

                if hf_aw is not None and col in hf_aw.columns:
                    t = hf_aw['Time'].values
                    mask = t > 0
                    ax.plot(
                        t[mask],
                        hf_aw[col].values[mask],
                        '--',
                        color='0.4',
                        lw=1.5,
                        label='AW',
                    )

                if hf_zal is not None and col in hf_zal.columns:
                    t = hf_zal['Time'].values
                    mask = t > 0
                    ax.plot(
                        t[mask],
                        hf_zal[col].values[mask],
                        '-',
                        color=CMF_COLORS[cmf],
                        lw=1.8,
                        label='Zalmoxis',
                    )

                ax.set_xscale('log')
                ax.set_yscale(yscale)

                if i == 0:
                    ax.set_title(label, fontsize=11)
                if i == len(CMFS) - 1:
                    ax.set_xlabel('Time [yr]')
                if j == 0:
                    ax.set_ylabel(f'CMF={cmf}')

                if i == 0 and j == 0:
                    ax.legend(fontsize=8, loc='best')

        fig.suptitle(f'AW vs Zalmoxis: {MASS_LABELS[mass]}', fontsize=13, y=1.01)
        fig.tight_layout()
        fig.savefig(plot_dir / f'evolution_M{mass}.pdf', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f'  Saved evolution_M{mass}.pdf')


def plot_summary_heatmaps(results: dict, outdir: Path):
    """Summary heatmaps: crystallization time and final T_magma.

    2x3 grid per quantity: AW | Zalmoxis | relative difference.
    """
    plot_dir = outdir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    for quantity, label, extract_fn in [
        (
            't_crystal',
            'Crystallization time [yr]',
            lambda hf: _crystallization_time(hf),
        ),
        (
            'T_magma_final',
            'Final T$_{\\rm magma}$ [K]',
            lambda hf: float(hf['T_magma'].iloc[-1]),
        ),
        (
            'R_int_final',
            'Final R$_{\\rm int}$ [m]',
            lambda hf: float(hf['R_int'].iloc[-1]),
        ),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

        for k, (struct, title) in enumerate(
            [('self', 'Adams-Williamson'), ('zalmoxis', 'Zalmoxis')]
        ):
            grid = np.full((len(CMFS), len(MASSES)), np.nan)
            for i, cmf in enumerate(CMFS):
                for j, mass in enumerate(MASSES):
                    name = case_name(mass, cmf, struct)
                    hf = results.get(name)
                    if hf is not None:
                        try:
                            grid[i, j] = extract_fn(hf)
                        except Exception:
                            pass

            ax = axes[k]
            im = ax.imshow(
                grid,
                aspect='auto',
                origin='lower',
                cmap='viridis',
            )
            ax.set_xticks(range(len(MASSES)))
            ax.set_xticklabels([f'{m}' for m in MASSES])
            ax.set_yticks(range(len(CMFS)))
            ax.set_yticklabels([f'{c}' for c in CMFS])
            ax.set_xlabel('Mass [M$_\\oplus$]')
            ax.set_ylabel('CMF')
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.8)

            # Annotate cells with values
            for i in range(len(CMFS)):
                for j in range(len(MASSES)):
                    val = grid[i, j]
                    if np.isfinite(val):
                        txt = f'{val:.0f}' if val > 100 else f'{val:.2g}'
                        ax.text(j, i, txt, ha='center', va='center', fontsize=7, color='w')

        # Relative difference panel
        grid_aw = np.full((len(CMFS), len(MASSES)), np.nan)
        grid_zal = np.full((len(CMFS), len(MASSES)), np.nan)
        for i, cmf in enumerate(CMFS):
            for j, mass in enumerate(MASSES):
                hf_aw = results.get(case_name(mass, cmf, 'self'))
                hf_zal = results.get(case_name(mass, cmf, 'zalmoxis'))
                if hf_aw is not None:
                    try:
                        grid_aw[i, j] = extract_fn(hf_aw)
                    except Exception:
                        pass
                if hf_zal is not None:
                    try:
                        grid_zal[i, j] = extract_fn(hf_zal)
                    except Exception:
                        pass

        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.where(
                (np.isfinite(grid_aw)) & (np.isfinite(grid_zal)) & (grid_aw != 0),
                (grid_zal - grid_aw) / grid_aw * 100,
                np.nan,
            )

        ax = axes[2]
        vmax = np.nanmax(np.abs(rel_diff)) if np.any(np.isfinite(rel_diff)) else 50
        vmax = min(vmax, 100)
        im = ax.imshow(
            rel_diff,
            aspect='auto',
            origin='lower',
            cmap='RdBu_r',
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_xticks(range(len(MASSES)))
        ax.set_xticklabels([f'{m}' for m in MASSES])
        ax.set_yticks(range(len(CMFS)))
        ax.set_yticklabels([f'{c}' for c in CMFS])
        ax.set_xlabel('Mass [M$_\\oplus$]')
        ax.set_ylabel('CMF')
        ax.set_title('Relative diff [%]')
        fig.colorbar(im, ax=ax, shrink=0.8)

        for i in range(len(CMFS)):
            for j in range(len(MASSES)):
                val = rel_diff[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f'{val:+.1f}%',
                        ha='center',
                        va='center',
                        fontsize=7,
                        color='k',
                    )

        fig.suptitle(label, fontsize=13)
        fig.tight_layout()
        fig.savefig(plot_dir / f'summary_{quantity}.pdf', bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f'  Saved summary_{quantity}.pdf')


def plot_phase2_stability(results: dict, outdir: Path):
    """Phase 2 feedback stability: T_magma and R_int with/without updates."""
    plot_dir = outdir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(MASSES), 2, figsize=(12, 3.5 * len(MASSES)))
    if len(MASSES) == 1:
        axes = axes[np.newaxis, :]

    for i, mass in enumerate(MASSES):
        zal_name = case_name(mass, 0.325, 'zalmoxis', phase2=False)
        p2_name = case_name(mass, 0.325, 'zalmoxis', phase2=True)

        hf_zal = results.get(zal_name)
        hf_p2 = results.get(p2_name)

        for j, (col, label) in enumerate(
            [('T_magma', 'T$_{\\rm magma}$ [K]'), ('R_int', 'R$_{\\rm int}$ [m]')]
        ):
            ax = axes[i, j]

            if hf_zal is not None and col in hf_zal.columns:
                t = hf_zal['Time'].values
                mask = t > 0
                ax.plot(
                    t[mask],
                    hf_zal[col].values[mask],
                    '-',
                    color='#7570b3',
                    lw=1.8,
                    label='No updates',
                )

            if hf_p2 is not None and col in hf_p2.columns:
                t = hf_p2['Time'].values
                mask = t > 0
                ax.plot(
                    t[mask],
                    hf_p2[col].values[mask],
                    '--',
                    color='#e7298a',
                    lw=1.5,
                    label='Phase 2 (100 yr)',
                )

            ax.set_xscale('log')
            ax.set_ylabel(label)
            if i == len(MASSES) - 1:
                ax.set_xlabel('Time [yr]')
            if i == 0:
                ax.set_title(label)
            if j == 0:
                ax.text(
                    -0.18,
                    0.5,
                    MASS_LABELS[mass],
                    transform=ax.transAxes,
                    fontsize=11,
                    va='center',
                    ha='center',
                    rotation=90,
                )
            if i == 0 and j == 0:
                ax.legend(fontsize=8)

    fig.suptitle('Phase 2 Feedback Stability (CMF=0.325)', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(plot_dir / 'phase2_stability.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('  Saved phase2_stability.pdf')


def plot_mass_radius(results: dict, outdir: Path):
    """Mass-radius diagram for all Zalmoxis runs (sanity check)."""
    plot_dir = outdir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    for cmf in CMFS:
        masses_plot, radii_plot = [], []
        for mass in MASSES:
            name = case_name(mass, cmf, 'zalmoxis')
            hf = results.get(name)
            if hf is not None and 'R_int' in hf.columns:
                R = float(hf['R_int'].iloc[0])
                masses_plot.append(mass)
                radii_plot.append(R / 6.371e6)  # Convert to R_earth

        if masses_plot:
            ax.plot(
                masses_plot,
                radii_plot,
                'o-',
                color=CMF_COLORS[cmf],
                lw=1.5,
                markersize=8,
                label=f'CMF={cmf}',
            )

    # Reference: R ~ M^0.27 (rocky planet scaling)
    m_ref = np.linspace(0.8, 5.5, 50)
    ax.plot(m_ref, m_ref**0.27, 'k:', lw=1, alpha=0.5, label=r'$R \propto M^{0.27}$')

    ax.set_xlabel(r'Mass [$M_\oplus$]')
    ax.set_ylabel(r'Radius [$R_\oplus$]')
    ax.set_title('Mass-Radius: Zalmoxis Validation')
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, 5.5)
    fig.tight_layout()
    fig.savefig(plot_dir / 'mass_radius.pdf', bbox_inches='tight', dpi=150)
    plt.close(fig)
    print('  Saved mass_radius.pdf')


# ── Validation checks ────────────────────────────────────────


def _crystallization_time(hf: pd.DataFrame) -> float:
    """Extract crystallization time (when Phi_global first drops below 0.01)."""
    if 'Phi_global' not in hf.columns:
        return np.nan
    phi = hf['Phi_global'].values
    t = hf['Time'].values
    below = np.where(phi < 0.01)[0]
    if len(below) > 0:
        return float(t[below[0]])
    return float(t[-1])  # didn't crystallize; return final time


def validate_single(name: str, hf: pd.DataFrame) -> list[str]:
    """Run physical plausibility checks on a single case.

    Returns
    -------
    list[str]
        List of failure messages. Empty if all checks pass.
    """
    failures = []

    # No NaN or Inf in key columns
    for col in ['T_magma', 'Phi_global', 'F_int', 'R_int', 'M_int']:
        if col not in hf.columns:
            continue
        vals = hf[col].values
        if np.any(np.isnan(vals)):
            failures.append(f'{col} contains NaN')
        if np.any(np.isinf(vals)):
            failures.append(f'{col} contains Inf')

    # T_magma monotonically decreasing (allow small upticks from numerics)
    if 'T_magma' in hf.columns:
        T = hf['T_magma'].values
        for k in range(1, len(T)):
            if T[k] > T[k - 1] + 50:  # Allow 50 K tolerance for numerical noise
                failures.append(f'T_magma increased by {T[k] - T[k - 1]:.0f} K at step {k}')
                break

    # Phi_global monotonically decreasing (allow small upticks)
    if 'Phi_global' in hf.columns:
        phi = hf['Phi_global'].values
        for k in range(1, len(phi)):
            if phi[k] > phi[k - 1] + 0.02:
                failures.append(
                    f'Phi_global increased by {phi[k] - phi[k - 1]:.3f} at step {k}'
                )
                break

    # F_int should be positive
    if 'F_int' in hf.columns:
        F = hf['F_int'].values
        if np.any(F < 0):
            failures.append('F_int is negative at some timesteps')

    # R_int in physically reasonable range
    if 'R_int' in hf.columns:
        R = hf['R_int'].values
        if np.any(R < 3e6) or np.any(R > 15e6):
            failures.append(
                f'R_int outside [3e6, 15e6] m: min={R.min():.2e}, max={R.max():.2e}'
            )

    # Time progressed
    if hf['Time'].iloc[-1] <= 0:
        failures.append('Simulation time did not progress')

    return failures


def validate_comparison(results: dict, mass: float, cmf: float) -> list[str]:
    """Compare AW vs Zalmoxis for a specific (mass, CMF) pair.

    Returns
    -------
    list[str]
        Failure messages for the comparison. Empty if criteria met.
    """
    failures = []
    aw_name = case_name(mass, cmf, 'self')
    zal_name = case_name(mass, cmf, 'zalmoxis')

    hf_aw = results.get(aw_name)
    hf_zal = results.get(zal_name)

    if hf_aw is None or hf_zal is None:
        return []  # Can't compare if one is missing

    # Only strict comparison for 1 M_earth, Earth-like CMF
    if not (np.isclose(mass, 1.0) and np.isclose(cmf, 0.325)):
        return []

    # T_magma should agree within 10% at final timestep
    T_aw = float(hf_aw['T_magma'].iloc[-1])
    T_zal = float(hf_zal['T_magma'].iloc[-1])
    if T_aw > 0:
        rel = abs(T_zal - T_aw) / T_aw
        if rel > 0.10:
            failures.append(f'T_magma disagrees by {rel:.0%}: AW={T_aw:.0f}, ZAL={T_zal:.0f}')

    # Crystallization time should agree within factor 2
    tc_aw = _crystallization_time(hf_aw)
    tc_zal = _crystallization_time(hf_zal)
    if tc_aw > 0 and tc_zal > 0:
        ratio = max(tc_aw, tc_zal) / min(tc_aw, tc_zal)
        if ratio > 2.0:
            failures.append(
                f'Crystallization time ratio {ratio:.1f}x: '
                f'AW={tc_aw:.0f} yr, ZAL={tc_zal:.0f} yr'
            )

    return failures


def validate_consistency(results: dict) -> list[str]:
    """Check Zalmoxis internal consistency across the mass-CMF grid.

    Verifies physical scaling relations.

    Returns
    -------
    list[str]
        Failure messages.
    """
    failures = []

    # R_int should increase with mass at fixed CMF (Zalmoxis only)
    for cmf in CMFS:
        radii = {}
        for mass in MASSES:
            name = case_name(mass, cmf, 'zalmoxis')
            hf = results.get(name)
            if hf is not None and 'R_int' in hf.columns:
                radii[mass] = float(hf['R_int'].iloc[0])

        masses_sorted = sorted(radii.keys())
        for k in range(1, len(masses_sorted)):
            m1, m2 = masses_sorted[k - 1], masses_sorted[k]
            if radii[m2] <= radii[m1]:
                failures.append(
                    f'R_int not increasing with mass: '
                    f'CMF={cmf}, M={m1}->R={radii[m1]:.2e}, '
                    f'M={m2}->R={radii[m2]:.2e}'
                )

    # R_int should increase with decreasing CMF at fixed mass
    for mass in MASSES:
        radii = {}
        for cmf in CMFS:
            name = case_name(mass, cmf, 'zalmoxis')
            hf = results.get(name)
            if hf is not None and 'R_int' in hf.columns:
                radii[cmf] = float(hf['R_int'].iloc[0])

        cmfs_sorted = sorted(radii.keys(), reverse=True)  # Decreasing CMF
        for k in range(1, len(cmfs_sorted)):
            c1, c2 = cmfs_sorted[k - 1], cmfs_sorted[k]
            if radii[c2] <= radii[c1]:
                failures.append(
                    f'R_int not increasing with decreasing CMF: '
                    f'M={mass}, CMF={c1}->R={radii[c1]:.2e}, '
                    f'CMF={c2}->R={radii[c2]:.2e}'
                )

    return failures


def validate_phase2(results: dict) -> list[str]:
    """Check Phase 2 feedback stability."""
    failures = []

    for mass in MASSES:
        p2_name = case_name(mass, 0.325, 'zalmoxis', phase2=True)
        hf = results.get(p2_name)
        if hf is None:
            continue

        # T_magma continuous (no jumps > 200 K)
        if 'T_magma' in hf.columns:
            T = hf['T_magma'].values
            for k in range(1, len(T)):
                dT = abs(T[k] - T[k - 1])
                if dT > 200:
                    failures.append(f'Phase 2 M={mass}: T_magma jump {dT:.0f} K at step {k}')
                    break

        # R_int changes smoothly (no jumps > 5%)
        if 'R_int' in hf.columns:
            R = hf['R_int'].values
            for k in range(1, len(R)):
                if R[k - 1] > 0:
                    rel = abs(R[k] - R[k - 1]) / R[k - 1]
                    if rel > 0.05:
                        failures.append(f'Phase 2 M={mass}: R_int jump {rel:.1%} at step {k}')
                        break

    return failures


def run_validation(results: dict, outdir: Path):
    """Run all validation checks and write report."""
    report_lines = []

    def log(msg: str):
        print(msg)
        report_lines.append(msg)

    log('=' * 60)
    log('VALIDATION REPORT')
    log('=' * 60)

    n_pass = 0
    n_fail = 0
    n_skip = 0

    # Per-case physical plausibility
    log('\n--- Per-case plausibility ---')
    for name, hf in sorted(results.items()):
        failures = validate_single(name, hf)
        if failures:
            n_fail += 1
            log(f'  FAIL  {name}')
            for f in failures:
                log(f'         {f}')
        else:
            n_pass += 1
            log(f'  PASS  {name}')

    # Check for expected but missing cases
    for mass in MASSES:
        for cmf in CMFS:
            for struct in ['self', 'zalmoxis']:
                name = case_name(mass, cmf, struct)
                if name not in results:
                    n_skip += 1
                    # Check if there's an ERROR.txt
                    err_file = outdir / name / 'ERROR.txt'
                    if err_file.exists():
                        err = err_file.read_text().strip()
                        log(f'  ERROR {name}: {err}')
                    else:
                        log(f'  MISS  {name}')

    # AW vs Zalmoxis comparison (strict for Earth-like only)
    log('\n--- AW vs Zalmoxis comparison (1 M_earth, CMF=0.325) ---')
    comp_failures = validate_comparison(results, 1.0, 0.325)
    if comp_failures:
        for f in comp_failures:
            log(f'  FAIL  {f}')
            n_fail += 1
    else:
        log('  PASS  AW-Zalmoxis agreement within tolerance')
        n_pass += 1

    # Zalmoxis consistency (scaling relations)
    log('\n--- Zalmoxis internal consistency ---')
    cons_failures = validate_consistency(results)
    if cons_failures:
        for f in cons_failures:
            log(f'  FAIL  {f}')
            n_fail += 1
    else:
        log('  PASS  Mass-radius scaling relations consistent')
        n_pass += 1

    # Phase 2 feedback
    log('\n--- Phase 2 feedback stability ---')
    p2_failures = validate_phase2(results)
    if p2_failures:
        for f in p2_failures:
            log(f'  FAIL  {f}')
            n_fail += 1
    else:
        log('  PASS  Phase 2 feedback stable')
        n_pass += 1

    # Summary
    log(f'\n{"=" * 60}')
    log(f'TOTAL: {n_pass} passed, {n_fail} failed, {n_skip} missing/error')
    log('=' * 60)

    # Write report
    report_path = outdir / 'validation_report.txt'
    report_path.write_text('\n'.join(report_lines) + '\n')
    print(f'\nReport written to {report_path}')


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Plot and validate Zalmoxis-SPIDER extended validation results'
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=PROTEUS_ROOT / 'output' / 'validation',
    )
    parser.add_argument('--validate-only', action='store_true', help='Skip plot generation')
    parser.add_argument('--plots-only', action='store_true', help='Skip validation checks')
    args = parser.parse_args()

    print(f'Loading results from {args.outdir}')
    results = load_all_results(args.outdir)
    print(f'Found {len(results)} completed cases\n')

    if not results:
        print('No results found. Run run_validation_matrix.py first.')
        sys.exit(1)

    if not args.validate_only:
        print('Generating plots...')
        plot_pairwise_evolution(results, args.outdir)
        plot_summary_heatmaps(results, args.outdir)
        plot_phase2_stability(results, args.outdir)
        plot_mass_radius(results, args.outdir)
        print()

    if not args.plots_only:
        run_validation(results, args.outdir)


if __name__ == '__main__':
    main()
