#!/usr/bin/env python
"""Radial profile analysis: T(r), phi(r), solidus/liquidus at multiple timesteps.

Loads SPIDER JSON snapshots from validation runs and generates radial
cross-section plots showing the evolution of temperature and melt fraction
with depth, overlaid on the solidus and liquidus curves.

Usage
-----
    python plot_radial_profiles.py                          # all 1 M_earth cases
    python plot_radial_profiles.py --cases M1.0_CMF0.325_AW M1.0_CMF0.325_ZAL
    python plot_radial_profiles.py --outdir path/to/runs

Output
------
    {outdir}/plots/radial_{case}.pdf         per-case T(r) + phi(r)
    {outdir}/plots/radial_comparison.pdf     AW vs ZAL side-by-side
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize

PROTEUS_ROOT = Path(__file__).resolve().parents[2]

# ── Styling ──────────────────────────────────────────────────

CMFS = [0.05, 0.10, 0.325, 0.50, 0.80]
CMF_COLORS = {
    0.05: '#1b9e77',
    0.10: '#d95f02',
    0.325: '#7570b3',
    0.50: '#e7298a',
    0.80: '#66a61e',
}

# Timestep selection: pick ~6 evenly spaced snapshots in log-time
N_SNAPSHOTS = 6


# ── Data loading ─────────────────────────────────────────────


def load_spider_json(fpath: Path) -> dict:
    """Load a SPIDER JSON snapshot and extract radial profiles.

    Returns
    -------
    dict with keys:
        time_yr, radius_km, temp_K, phi, solidus_K, liquidus_K, pressure_GPa
    """
    with open(fpath) as f:
        d = json.load(f)

    data = d['data']

    def extract(key: str) -> np.ndarray:
        field = data[key]
        scaling = float(field['scaling'])
        return np.array([float(v) for v in field['values']]) * scaling

    return {
        'time_yr': float(d['time_years']),
        'radius_km': extract('radius_b') / 1e3,
        'temp_K': extract('temp_b'),
        'phi': extract('phi_b'),
        'solidus_K': extract('solidus_temp_b'),
        'liquidus_K': extract('liquidus_temp_b'),
        'pressure_GPa': extract('pressure_b') / 1e9,
    }


def load_case_snapshots(case_dir: Path) -> list[dict]:
    """Load all SPIDER JSON snapshots for a case, sorted by time."""
    data_dir = case_dir / 'data'
    if not data_dir.exists():
        return []

    jsons = sorted(
        [f for f in data_dir.iterdir() if f.suffix == '.json'],
        key=lambda p: int(p.stem),
    )

    snapshots = []
    for fpath in jsons:
        try:
            snapshots.append(load_spider_json(fpath))
        except Exception:
            continue

    return sorted(snapshots, key=lambda s: s['time_yr'])


def select_snapshots(snapshots: list[dict], n: int = N_SNAPSHOTS) -> list[dict]:
    """Select n snapshots approximately evenly spaced in log-time.

    Always includes first and last snapshot.
    """
    if len(snapshots) <= n:
        return snapshots

    # Filter to positive times for log spacing
    positive = [s for s in snapshots if s['time_yr'] > 0]
    if len(positive) <= n:
        return [snapshots[0]] + positive

    log_times = np.log10([s['time_yr'] for s in positive])
    targets = np.linspace(log_times[0], log_times[-1], n)

    selected = []
    for target in targets:
        idx = np.argmin(np.abs(log_times - target))
        if positive[idx] not in selected:
            selected.append(positive[idx])

    # Always include last
    if selected[-1] is not positive[-1]:
        selected.append(positive[-1])

    return selected


# ── Per-case radial profile plot ─────────────────────────────


def plot_case_profiles(case_dir: Path, outdir: Path):
    """Plot T(r) with solidus/liquidus and phi(r) for a single case."""
    case_name = case_dir.name
    snapshots = load_case_snapshots(case_dir)
    if not snapshots:
        print(f'  SKIP {case_name}: no JSON snapshots')
        return

    selected = select_snapshots(snapshots)
    if not selected:
        print(f'  SKIP {case_name}: no positive-time snapshots')
        return

    plot_dir = outdir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_t, ax_phi) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    # Colormap for time
    times = [s['time_yr'] for s in selected]
    t_min, t_max = max(times[0], 1), max(times[-1], 10)
    if t_max / t_min > 10:
        norm = LogNorm(vmin=t_min, vmax=t_max)
    else:
        norm = Normalize(vmin=t_min, vmax=t_max)
    cmap = plt.cm.viridis

    for snap in selected:
        t = snap['time_yr']
        r = snap['radius_km']
        color = cmap(norm(max(t, t_min)))

        # Temperature panel
        ax_t.plot(snap['temp_K'], r, '-', color=color, lw=1.5)

        # Melt fraction panel
        ax_phi.plot(snap['phi'], r, '-', color=color, lw=1.5)

    # Solidus and liquidus from last snapshot (pressure-dependent, nearly static)
    last = selected[-1]
    ax_t.plot(last['solidus_K'], last['radius_km'], 'k--', lw=1, alpha=0.7, label='Solidus')
    ax_t.plot(last['liquidus_K'], last['radius_km'], 'k:', lw=1, alpha=0.7, label='Liquidus')

    # Also from first snapshot to show if they differ
    first = selected[0]
    ax_t.plot(
        first['solidus_K'],
        first['radius_km'],
        '--',
        color='0.6',
        lw=0.8,
        alpha=0.5,
    )
    ax_t.plot(
        first['liquidus_K'],
        first['radius_km'],
        ':',
        color='0.6',
        lw=0.8,
        alpha=0.5,
    )

    ax_t.set_xlabel('Temperature [K]', fontsize=11)
    ax_t.set_ylabel('Radius [km]', fontsize=11)
    ax_t.set_title(f'Temperature profile — {case_name}', fontsize=12)
    ax_t.legend(fontsize=9)

    ax_phi.set_xlabel(r'Melt fraction, $\phi$', fontsize=11)
    ax_phi.set_title(f'Melt fraction — {case_name}', fontsize=12)
    ax_phi.set_xlim(-0.02, 1.02)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_t, ax_phi], shrink=0.85, pad=0.02)
    cbar.set_label('Time [yr]', fontsize=10)

    fig.savefig(plot_dir / f'radial_{case_name}.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved radial_{case_name}.pdf')


# ── AW vs Zalmoxis comparison ────────────────────────────────


def plot_comparison(outdir: Path):
    """Side-by-side AW vs Zalmoxis radial profiles for each CMF.

    5 rows (CMFs) x 4 columns: T(r) AW, T(r) ZAL, phi(r) AW, phi(r) ZAL.
    """
    plot_dir = outdir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Check which CMFs have both AW and ZAL data
    available_cmfs = []
    for cmf in CMFS:
        aw_dir = outdir / f'M1.0_CMF{cmf}_AW'
        zal_dir = outdir / f'M1.0_CMF{cmf}_ZAL'
        if aw_dir.exists() and zal_dir.exists():
            available_cmfs.append(cmf)

    if not available_cmfs:
        print('  SKIP comparison: no complete AW+ZAL pairs')
        return

    fig, axes = plt.subplots(
        len(available_cmfs),
        4,
        figsize=(20, 3.5 * len(available_cmfs)),
        layout='constrained',
    )
    if len(available_cmfs) == 1:
        axes = axes[np.newaxis, :]

    for i, cmf in enumerate(available_cmfs):
        for j, (struct, tag) in enumerate([('AW', 'AW'), ('ZAL', 'Zalmoxis')]):
            case_dir = outdir / f'M1.0_CMF{cmf}_{struct}'
            snapshots = load_case_snapshots(case_dir)
            selected = select_snapshots(snapshots)

            if not selected:
                continue

            ax_t = axes[i, j * 2]
            ax_phi = axes[i, j * 2 + 1]

            times = [s['time_yr'] for s in selected]
            t_min, t_max = max(times[0], 1), max(times[-1], 10)
            if t_max / t_min > 10:
                norm = LogNorm(vmin=t_min, vmax=t_max)
            else:
                norm = Normalize(vmin=t_min, vmax=t_max)
            cmap = plt.cm.viridis

            for snap in selected:
                t = snap['time_yr']
                r = snap['radius_km']
                color = cmap(norm(max(t, t_min)))
                if t < 1e3:
                    lbl = f'{t:.0f} yr'
                elif t < 1e6:
                    lbl = f'{t / 1e3:.1f} kyr'
                else:
                    lbl = f'{t / 1e6:.2f} Myr'

                ax_t.plot(snap['temp_K'], r, '-', color=color, lw=1.2, label=lbl)
                ax_phi.plot(snap['phi'], r, '-', color=color, lw=1.2)

            # Solidus/liquidus from last snapshot
            last = selected[-1]
            ax_t.plot(
                last['solidus_K'],
                last['radius_km'],
                'k--',
                lw=1,
                alpha=0.6,
                label='Solidus',
            )
            ax_t.plot(
                last['liquidus_K'],
                last['radius_km'],
                'k:',
                lw=1,
                alpha=0.6,
                label='Liquidus',
            )

            # Time legend on T panel
            ax_t.legend(fontsize=6, loc='lower left', ncol=2)

            # Labels
            if i == 0:
                ax_t.set_title(f'{tag} — Temperature [K]', fontsize=10)
                ax_phi.set_title(f'{tag} — Melt fraction', fontsize=10)
            if i == len(available_cmfs) - 1:
                ax_t.set_xlabel('Temperature [K]')
                ax_phi.set_xlabel(r'Melt fraction, $\phi$')

            ax_phi.set_xlim(-0.02, 1.02)

            if j == 0:
                ax_t.set_ylabel(f'CMF = {cmf}\nRadius [km]', fontsize=10)
            else:
                ax_t.set_ylabel('')
                ax_phi.set_ylabel('')

    fig.suptitle(
        r'Radial profiles: AW vs Zalmoxis — 1 $M_\oplus$',
        fontsize=14,
    )
    fig.savefig(plot_dir / 'radial_comparison.pdf', dpi=150)
    plt.close(fig)
    print('  Saved radial_comparison.pdf')


# ── Final-state comparison ────────────────────────────────────


def plot_final_state(outdir: Path):
    """Compare final T(r) and phi(r) between AW and Zalmoxis for all CMFs.

    2 panels: T(r) and phi(r), with AW dashed and ZAL solid, colored by CMF.
    """
    plot_dir = outdir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax_t, ax_phi) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

    for cmf in CMFS:
        color = CMF_COLORS[cmf]

        for struct, ls, lw in [('AW', '--', 1.2), ('ZAL', '-', 1.8)]:
            case_dir = outdir / f'M1.0_CMF{cmf}_{struct}'
            snapshots = load_case_snapshots(case_dir)
            if not snapshots:
                continue

            last = snapshots[-1]
            label = f'CMF={cmf} {struct}'

            ax_t.plot(last['temp_K'], last['radius_km'], ls, color=color, lw=lw, label=label)
            ax_phi.plot(last['phi'], last['radius_km'], ls, color=color, lw=lw, label=label)

        # Solidus from last AW snapshot (representative)
        aw_dir = outdir / f'M1.0_CMF{cmf}_AW'
        aw_snaps = load_case_snapshots(aw_dir)
        if aw_snaps and cmf == 0.325:
            ax_t.plot(
                aw_snaps[-1]['solidus_K'],
                aw_snaps[-1]['radius_km'],
                'k--',
                lw=0.8,
                alpha=0.5,
                label='Solidus (CMF=0.325)',
            )
            ax_t.plot(
                aw_snaps[-1]['liquidus_K'],
                aw_snaps[-1]['radius_km'],
                'k:',
                lw=0.8,
                alpha=0.5,
                label='Liquidus (CMF=0.325)',
            )

    ax_t.set_xlabel('Temperature [K]', fontsize=11)
    ax_t.set_ylabel('Radius [km]', fontsize=11)
    ax_t.set_title(r'Final temperature profile — 1 $M_\oplus$', fontsize=12)
    ax_t.legend(fontsize=7, loc='lower left', ncol=2)

    ax_phi.set_xlabel(r'Melt fraction, $\phi$', fontsize=11)
    ax_phi.set_title(r'Final melt fraction — 1 $M_\oplus$', fontsize=12)
    ax_phi.set_xlim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(plot_dir / 'radial_final_state.pdf', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  Saved radial_final_state.pdf')


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Plot radial T(r) and phi(r) profiles from validation runs'
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=PROTEUS_ROOT / 'output' / 'validation',
    )
    parser.add_argument(
        '--cases',
        nargs='+',
        help='Specific case names to plot (default: all M1.0 cases)',
    )
    args = parser.parse_args()

    # Determine which cases to plot
    if args.cases:
        case_dirs = [args.outdir / c for c in args.cases]
    else:
        case_dirs = sorted(
            [d for d in args.outdir.iterdir() if d.is_dir() and d.name.startswith('M1.0')]
        )

    print(f'Plotting radial profiles for {len(case_dirs)} cases\n')

    # Per-case profiles
    for case_dir in case_dirs:
        plot_case_profiles(case_dir, args.outdir)

    # Comparison plots
    print()
    plot_comparison(args.outdir)
    plot_final_state(args.outdir)


if __name__ == '__main__':
    main()
