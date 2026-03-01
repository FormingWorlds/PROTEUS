#!/usr/bin/env python
"""Generate self-contained HTML validation report for Habrok grid results.

Loads all results from the Habrok output directory, generates PNG plots,
runs validation checks, and writes:
  - ``validation_report.md``  — Markdown with linked images
  - ``validation_report.html`` — Self-contained HTML with base64-embedded PNGs

Usage
-----
    python generate_validation_report.py --outdir /scratch/$USER/habrok_validation_v3

    # Skip plots (report from cached PNGs)
    python generate_validation_report.py --outdir ... --skip-plots

    # Plots only, no report
    python generate_validation_report.py --outdir ... --plots-only
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize

# ── Constants ────────────────────────────────────────────────────

MASSES = [1.0, 2.0, 3.0]
CMFS = [0.10, 0.325, 0.50]

CMF_COLORS = {0.10: '#d95f02', 0.325: '#7570b3', 0.50: '#e7298a'}
MASS_LABELS = {
    1.0: r'1 $M_\oplus$',
    2.0: r'2 $M_\oplus$',
    3.0: r'3 $M_\oplus$',
}

PHI_CRYSTAL = 0.01
DPI = 150

# Case name regex (matches generate_habrok_grid.py output)
CASE_RE = re.compile(
    r'^(?P<block>[A-Z])_M(?P<mass>[\d.]+)_CMF(?P<cmf>[\d.]+)_(?P<tag>AW|ZAL)'
    r'(?:_(?P<tmode>Tlin))?'
    r'(?:_P2u(?P<update>\d+))?'
    r'(?:_n(?P<nlevels>\d+))?'
    r'(?:_S(?P<entropy>\d+))?$'
)


# ── Data loading ─────────────────────────────────────────────────


def load_helpfile(case_dir: Path) -> pd.DataFrame | None:
    """Load runtime helpfile, returning None if missing or empty."""
    hf_path = case_dir / 'runtime_helpfile.csv'
    if not hf_path.exists():
        return None
    try:
        df = pd.read_csv(hf_path, sep='\t')
        return df if len(df) > 0 else None
    except Exception:
        return None


def load_all_helpfiles(outdir: Path) -> dict[str, pd.DataFrame]:
    """Load helpfiles from all case directories."""
    results = {}
    cases_dir = outdir / 'cases'
    search_dir = cases_dir if cases_dir.exists() else outdir
    if not search_dir.exists():
        return results
    for d in sorted(search_dir.iterdir()):
        if d.is_dir():
            hf = load_helpfile(d)
            if hf is not None:
                results[d.name] = hf
    return results


def load_spider_json(fpath: Path) -> dict | None:
    """Load a SPIDER JSON snapshot and return extracted arrays."""
    try:
        with open(fpath) as f:
            d = json.load(f)
        data = d['data']

        def extract(key):
            field = data[key]
            s = float(field['scaling'])
            return np.array([float(v) for v in field['values']]) * s

        return {
            'time_yr': float(d.get('time_years', 0)),
            'radius_km': extract('radius_b') / 1e3,
            'temp_K': extract('temp_b'),
            'phi': extract('phi_b'),
            'solidus_K': extract('solidus_temp_b'),
            'liquidus_K': extract('liquidus_temp_b'),
            'pressure_GPa': extract('pressure_b') / 1e9,
        }
    except Exception:
        return None


def load_case_snapshots(case_dir: Path, n_select: int = 6) -> list[dict]:
    """Load SPIDER JSON snapshots, select ~n evenly spaced in log-time."""
    data_dir = case_dir / 'data'
    if not data_dir.exists():
        return []
    jsons = sorted(
        [f for f in data_dir.iterdir() if f.suffix == '.json'],
        key=lambda p: int(p.stem),
    )
    snaps = [s for f in jsons if (s := load_spider_json(f)) is not None]
    snaps.sort(key=lambda s: s['time_yr'])
    if len(snaps) <= n_select:
        return snaps
    positive = [s for s in snaps if s['time_yr'] > 0]
    if len(positive) <= n_select:
        return [snaps[0]] + positive
    log_t = np.log10([s['time_yr'] for s in positive])
    targets = np.linspace(log_t[0], log_t[-1], n_select)
    selected = []
    for tgt in targets:
        idx = int(np.argmin(np.abs(log_t - tgt)))
        if positive[idx] not in selected:
            selected.append(positive[idx])
    if selected[-1] is not positive[-1]:
        selected.append(positive[-1])
    return selected


def crystallization_time(hf: pd.DataFrame) -> float:
    """Time when Phi_global first drops below PHI_CRYSTAL."""
    if 'Phi_global' not in hf.columns:
        return np.nan
    phi = hf['Phi_global'].values
    t = hf['Time'].values
    below = np.where(phi < PHI_CRYSTAL)[0]
    return float(t[below[0]]) if len(below) > 0 else float(t[-1])


def case_status(case_dir: Path) -> str:
    """Read case status: 'completed', 'errored', 'timeout', or 'missing'."""
    status_file = case_dir / 'status'
    if not status_file.exists():
        return 'missing'
    text = status_file.read_text().strip()
    if 'Completed' in text:
        return 'completed'
    if 'Error' in text:
        return 'errored'
    return 'timeout'


# ── Plotting helpers ─────────────────────────────────────────────


def _save(fig, plot_dir: Path, name: str):
    """Save figure as PNG and close."""
    fig.savefig(plot_dir / name, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f'  {name}')


def _axis_label(name: str, unit: str) -> str:
    return f'{name} [{unit}]' if unit else name


# ── Plot functions ───────────────────────────────────────────────


def plot_evolution(results: dict, plot_dir: Path):
    """Time-evolution overlay: AW vs ZAL for each mass (3 figures)."""
    cols = [
        ('T_magma', r'$T_{\rm magma}$', 'K', 'linear'),
        ('Phi_global', r'$\Phi_{\rm global}$', '', 'linear'),
        ('F_int', r'$F_{\rm int}$', r'W m$^{-2}$', 'log'),
    ]

    for mass in MASSES:
        fig, axes = plt.subplots(
            len(CMFS),
            len(cols),
            figsize=(14, 3.2 * len(CMFS)),
            layout='constrained',
        )
        if len(CMFS) == 1:
            axes = axes[np.newaxis, :]
        has_data = False

        for i, cmf in enumerate(CMFS):
            aw_name = f'A_M{mass}_CMF{cmf}_AW'
            zal_name = f'B_M{mass}_CMF{cmf}_ZAL'

            for j, (col, label, unit, yscale) in enumerate(cols):
                ax = axes[i, j]
                plotted = False

                for name, ls, color, lbl in [
                    (aw_name, '--', '0.4', 'AW'),
                    (zal_name, '-', CMF_COLORS[cmf], 'Zalmoxis'),
                ]:
                    hf = results.get(name)
                    if hf is not None and col in hf.columns:
                        t = hf['Time'].values
                        mask = t > 0
                        if mask.any():
                            ax.plot(
                                t[mask],
                                hf[col].values[mask],
                                ls,
                                color=color,
                                lw=1.5,
                                label=lbl,
                            )
                            plotted = True
                            has_data = True

                if not plotted:
                    ax.plot([1e2, 1e6], [1, 1e6], alpha=0)
                ax.set_xscale('log')
                ax.set_yscale(yscale)
                if i == 0:
                    ax.set_title(_axis_label(label, unit), fontsize=10)
                if i == len(CMFS) - 1:
                    ax.set_xlabel('Time [yr]')
                else:
                    ax.tick_params(labelbottom=False)
                if j == 0:
                    ax.set_ylabel(f'CMF = {cmf}', fontsize=10)
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)

        fig.suptitle(f'AW vs Zalmoxis — {MASS_LABELS[mass]}', fontsize=13)
        if has_data:
            _save(fig, plot_dir, f'evolution_{mass:.0f}M.png')
        else:
            plt.close(fig)


def plot_radial_profiles(outdir: Path, plot_dir: Path):
    """Per-case radial T(r) and phi(r) profiles at selected timesteps."""
    cases_dir = outdir / 'cases'
    if not cases_dir.exists():
        cases_dir = outdir

    for mass in [1.0, 3.0]:
        for block, struct_tag in [('A', 'AW'), ('B', 'ZAL')]:
            name = f'{block}_M{mass}_CMF0.325_{struct_tag}'
            case_dir = cases_dir / name
            snaps = load_case_snapshots(case_dir)
            if not snaps:
                continue

            fig, (ax_t, ax_phi) = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
            times = [s['time_yr'] for s in snaps]
            t_min, t_max = max(times[0], 1), max(times[-1], 10)
            norm = LogNorm(t_min, t_max) if t_max / t_min > 10 else Normalize(t_min, t_max)
            cmap = plt.cm.viridis

            for snap in snaps:
                t = snap['time_yr']
                r = snap['radius_km']
                c = cmap(norm(max(t, t_min)))
                ax_t.plot(snap['temp_K'], r, '-', color=c, lw=1.5)
                ax_phi.plot(snap['phi'], r, '-', color=c, lw=1.5)

            last = snaps[-1]
            ax_t.plot(
                last['solidus_K'], last['radius_km'], 'k--', lw=1, alpha=0.7, label='Solidus'
            )
            ax_t.plot(
                last['liquidus_K'], last['radius_km'], 'k:', lw=1, alpha=0.7, label='Liquidus'
            )

            ax_t.set_xlabel('Temperature [K]')
            ax_t.set_ylabel('Radius [km]')
            ax_t.set_title(f'T(r) — {name}')
            ax_t.legend(fontsize=9)
            ax_phi.set_xlabel(r'Melt fraction, $\phi$')
            ax_phi.set_title(f'Melt fraction — {name}')
            ax_phi.set_xlim(-0.02, 1.02)

            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=[ax_t, ax_phi], shrink=0.85, pad=0.02, label='Time [yr]')
            _save(fig, plot_dir, f'radial_{name}.png')


def plot_radial_comparison(outdir: Path, plot_dir: Path):
    """Side-by-side AW vs ZAL radial profiles at 1 and 3 M_earth."""
    cases_dir = outdir / 'cases'
    if not cases_dir.exists():
        cases_dir = outdir

    for mass in [1.0, 3.0]:
        avail_cmfs = [
            cmf
            for cmf in CMFS
            if (cases_dir / f'A_M{mass}_CMF{cmf}_AW').exists()
            and (cases_dir / f'B_M{mass}_CMF{cmf}_ZAL').exists()
        ]
        if not avail_cmfs:
            continue

        fig, axes = plt.subplots(
            len(avail_cmfs),
            4,
            figsize=(20, 3.5 * len(avail_cmfs)),
            layout='constrained',
        )
        if len(avail_cmfs) == 1:
            axes = axes[np.newaxis, :]

        for i, cmf in enumerate(avail_cmfs):
            for j, (block, tag) in enumerate([('A', 'AW'), ('B', 'ZAL')]):
                case_dir = cases_dir / f'{block}_M{mass}_CMF{cmf}_{tag}'
                snaps = load_case_snapshots(case_dir)
                if not snaps:
                    continue

                ax_t = axes[i, j * 2]
                ax_phi = axes[i, j * 2 + 1]

                times = [s['time_yr'] for s in snaps]
                t_min, t_max = max(times[0], 1), max(times[-1], 10)
                norm = LogNorm(t_min, t_max) if t_max / t_min > 10 else Normalize(t_min, t_max)
                cmap = plt.cm.viridis

                for snap in snaps:
                    t = snap['time_yr']
                    r = snap['radius_km']
                    c = cmap(norm(max(t, t_min)))
                    lbl = (
                        f'{t:.0f} yr'
                        if t < 1e3
                        else f'{t / 1e3:.1f} kyr'
                        if t < 1e6
                        else f'{t / 1e6:.2f} Myr'
                    )
                    ax_t.plot(snap['temp_K'], r, '-', color=c, lw=1.2, label=lbl)
                    ax_phi.plot(snap['phi'], r, '-', color=c, lw=1.2)

                last = snaps[-1]
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
                ax_t.legend(fontsize=6, loc='lower left', ncol=2)

                if i == 0:
                    ax_t.set_title(f'{tag} — Temperature [K]', fontsize=10)
                    ax_phi.set_title(f'{tag} — Melt fraction', fontsize=10)
                if i == len(avail_cmfs) - 1:
                    ax_t.set_xlabel('Temperature [K]')
                    ax_phi.set_xlabel(r'$\phi$')
                ax_phi.set_xlim(-0.02, 1.02)
                if j == 0:
                    ax_t.set_ylabel(f'CMF={cmf}\nRadius [km]', fontsize=10)

        fig.suptitle(f'Radial profiles: AW vs Zalmoxis — {MASS_LABELS[mass]}', fontsize=14)
        _save(fig, plot_dir, f'radial_comparison_{mass:.0f}M.png')


def plot_adiabat_mesh_shift(results: dict, outdir: Path, plot_dir: Path):
    """Bar chart of mesh shift: adiabatic vs linear T at first Phase 2 update."""
    cases_dir = outdir / 'cases'
    if not cases_dir.exists():
        cases_dir = outdir

    masses_plot = [1.0, 3.0]
    shifts = {'adiabatic': [], 'linear': []}

    for mass in masses_plot:
        for mode, pattern in [
            ('adiabatic', f'B_M{mass}_CMF0.325_ZAL'),
            ('linear', f'C_M{mass}_CMF0.325_ZAL_Tlin'),
        ]:
            # Try to compute mesh shift from first two Zalmoxis mesh files
            # or from R_int change between steps 0 and 1
            hf = results.get(pattern)
            if hf is not None and 'R_int' in hf.columns and len(hf) >= 2:
                r0 = float(hf['R_int'].iloc[0])
                r1 = float(hf['R_int'].iloc[1])
                shift = abs(r1 - r0) / r0 if r0 > 0 else np.nan
            else:
                shift = np.nan
            shifts[mode].append(shift)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(masses_plot))
    w = 0.35
    bars_a = ax.bar(
        x - w / 2,
        [s * 100 for s in shifts['adiabatic']],
        w,
        label='Adiabatic (auto)',
        color='#4caf50',
    )
    bars_l = ax.bar(
        x + w / 2,
        [s * 100 for s in shifts['linear']],
        w,
        label='Linear (forced)',
        color='#d32f2f',
    )

    for bars in [bars_a, bars_l]:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h) and h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.5,
                    f'{h:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                )

    ax.axhline(10, color='k', ls=':', lw=0.8, alpha=0.5, label='10% target')
    ax.set_xticks(x)
    ax.set_xticklabels([MASS_LABELS[m] for m in masses_plot])
    ax.set_ylabel('Relative mesh shift at first update [%]')
    ax.set_title('Adiabatic vs linear T mode — mesh shift')
    ax.legend()
    _save(fig, plot_dir, 'adiabat_mesh_shift.png')


def plot_adiabat_vs_linear_evolution(results: dict, plot_dir: Path):
    """T_magma(t) overlay: adiabatic vs linear T init at 1 and 3 M_earth."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, mass in zip(axes, [1.0, 3.0]):
        for name, label, color, ls in [
            (f'B_M{mass}_CMF0.325_ZAL', 'Adiabatic (auto)', '#4caf50', '-'),
            (f'C_M{mass}_CMF0.325_ZAL_Tlin', 'Linear (forced)', '#d32f2f', '--'),
        ]:
            hf = results.get(name)
            if hf is not None and 'T_magma' in hf.columns:
                t = hf['Time'].values
                mask = t > 0
                if mask.any():
                    ax.plot(
                        t[mask],
                        hf['T_magma'].values[mask],
                        ls,
                        color=color,
                        lw=1.5,
                        label=label,
                    )
        ax.set_xscale('log')
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel(r'$T_{\rm magma}$ [K]')
        ax.set_title(MASS_LABELS[mass])
        ax.legend(fontsize=9)

    fig.suptitle('Adiabatic vs linear T mode — thermal evolution', fontsize=13)
    fig.tight_layout()
    _save(fig, plot_dir, 'adiabat_vs_linear_evolution.png')


def plot_phase2_stability(results: dict, plot_dir: Path):
    """Phase 1 vs P2u100 vs P2u1000: T_magma and R_int evolution."""
    p2_cols = [
        ('T_magma', r'$T_{\rm magma}$', 'K'),
        ('R_int', r'$R_{\rm int}$', 'm'),
    ]
    fig, axes = plt.subplots(len(MASSES), len(p2_cols), figsize=(12, 3.5 * len(MASSES)))
    if len(MASSES) == 1:
        axes = axes[np.newaxis, :]

    for i, mass in enumerate(MASSES):
        configs = [
            (f'B_M{mass}_CMF0.325_ZAL', 'Phase 1', '#7570b3', '-'),
            (f'D_M{mass}_CMF0.325_ZAL_P2u100', r'P2 $\Delta t$=100 yr', '#e7298a', '--'),
        ]
        for j, (col, label, unit) in enumerate(p2_cols):
            ax = axes[i, j]
            for name, cfg_lbl, color, ls in configs:
                hf = results.get(name)
                if hf is not None and col in hf.columns:
                    t = hf['Time'].values
                    mask = t > 0
                    if mask.any():
                        ax.plot(
                            t[mask],
                            hf[col].values[mask],
                            ls,
                            color=color,
                            lw=1.5,
                            label=cfg_lbl,
                        )
            ax.set_xscale('log')
            ax.set_ylabel(_axis_label(label, unit), fontsize=10)
            if i == len(MASSES) - 1:
                ax.set_xlabel('Time [yr]')
            else:
                ax.tick_params(labelbottom=False)
            if i == 0:
                ax.set_title(_axis_label(label, unit), fontsize=11)
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

    fig.suptitle('Phase 2 feedback stability (CMF = 0.325)', fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, plot_dir, 'phase2_stability.png')


def plot_phase2_mesh_shift(results: dict, plot_dir: Path):
    """Mesh shift magnitude over time for P2u100 cases."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for mass, color in zip(MASSES, ['#d95f02', '#7570b3', '#e7298a']):
        p2_name = f'D_M{mass}_CMF0.325_ZAL_P2u100'
        hf = results.get(p2_name)
        if hf is None or 'R_int' not in hf.columns or len(hf) < 3:
            continue

        t = hf['Time'].values
        R = hf['R_int'].values
        shift = np.abs(np.diff(R)) / R[:-1] * 100
        mask = t[1:] > 0
        if mask.any():
            ax.plot(
                t[1:][mask],
                shift[mask],
                'o-',
                color=color,
                markersize=3,
                lw=1,
                label=MASS_LABELS[mass],
            )

    ax.axhline(5, color='k', ls=':', lw=0.8, alpha=0.5, label='5% threshold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Step-to-step R_int shift [%]')
    ax.set_title('Phase 2 mesh shift timeline (P2u100, CMF=0.325)')
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, plot_dir, 'phase2_mesh_shift_timeline.png')


def plot_summary_heatmaps(results: dict, plot_dir: Path):
    """Heatmaps for crystallization time and final T_magma."""

    def _extract_tc(hf):
        return crystallization_time(hf)

    def _extract_tm(hf):
        return float(hf['T_magma'].iloc[-1])

    for quantity, extract_fn, label, unit in [
        ('t_crystal', _extract_tc, r'Crystallization time', 'yr'),
        ('T_magma_final', _extract_tm, r'Final $T_{\rm magma}$', 'K'),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')

        for k, (struct, title) in enumerate([('AW', 'Adams-Williamson'), ('ZAL', 'Zalmoxis')]):
            grid = np.full((len(CMFS), len(MASSES)), np.nan)
            for i, cmf in enumerate(CMFS):
                for j, mass in enumerate(MASSES):
                    hf = results.get(
                        f'{"A" if struct == "AW" else "B"}_M{mass}_CMF{cmf}_{struct}'
                    )
                    if hf is not None:
                        try:
                            grid[i, j] = extract_fn(hf)
                        except Exception:
                            pass

            ax = axes[k]
            im = ax.imshow(grid, aspect='auto', origin='lower', cmap='viridis')
            ax.set_xticks(range(len(MASSES)))
            ax.set_xticklabels([MASS_LABELS[m] for m in MASSES])
            ax.set_yticks(range(len(CMFS)))
            ax.set_yticklabels([f'{c}' for c in CMFS])
            ax.set_xlabel(r'Mass [$M_\oplus$]')
            ax.set_ylabel('CMF')
            ax.set_title(title)
            fig.colorbar(im, ax=ax, shrink=0.8, label=_axis_label(label, unit))

            for i in range(len(CMFS)):
                for j in range(len(MASSES)):
                    v = grid[i, j]
                    if np.isfinite(v):
                        txt = f'{v:.0f}' if v > 100 else f'{v:.2g}'
                        ax.text(j, i, txt, ha='center', va='center', fontsize=8, color='w')

        # Relative difference panel
        grid_aw = np.full((len(CMFS), len(MASSES)), np.nan)
        grid_zal = np.full((len(CMFS), len(MASSES)), np.nan)
        for i, cmf in enumerate(CMFS):
            for j, mass in enumerate(MASSES):
                for struct, grid_ in [('AW', grid_aw), ('ZAL', grid_zal)]:
                    hf = results.get(
                        f'{"A" if struct == "AW" else "B"}_M{mass}_CMF{cmf}_{struct}'
                    )
                    if hf is not None:
                        try:
                            grid_[i, j] = extract_fn(hf)
                        except Exception:
                            pass

        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(
                np.isfinite(grid_aw) & np.isfinite(grid_zal) & (grid_aw != 0),
                (grid_zal - grid_aw) / grid_aw * 100,
                np.nan,
            )

        ax = axes[2]
        vmax = min(np.nanmax(np.abs(rel)), 100) if np.any(np.isfinite(rel)) else 50
        im = ax.imshow(rel, aspect='auto', origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(MASSES)))
        ax.set_xticklabels([MASS_LABELS[m] for m in MASSES])
        ax.set_yticks(range(len(CMFS)))
        ax.set_yticklabels([f'{c}' for c in CMFS])
        ax.set_xlabel(r'Mass [$M_\oplus$]')
        ax.set_ylabel('CMF')
        ax.set_title(r'(ZAL$-$AW)/AW')
        fig.colorbar(im, ax=ax, shrink=0.8, label='Relative difference [%]')

        for i in range(len(CMFS)):
            for j in range(len(MASSES)):
                v = rel[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f'{v:+.1f}%', ha='center', va='center', fontsize=8, color='k')

        fig.suptitle(_axis_label(label, unit), fontsize=13)
        _save(fig, plot_dir, f'summary_{quantity}.png')


def plot_mass_radius(results: dict, plot_dir: Path):
    """Mass-radius diagram from Zalmoxis initial structure."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for cmf in CMFS:
        m_plot, r_plot = [], []
        for mass in MASSES:
            hf = results.get(f'B_M{mass}_CMF{cmf}_ZAL')
            if hf is not None and 'R_int' in hf.columns:
                m_plot.append(mass)
                r_plot.append(float(hf['R_int'].iloc[0]) / 6.371e6)
        if m_plot:
            ax.plot(
                m_plot,
                r_plot,
                'o-',
                color=CMF_COLORS[cmf],
                lw=1.5,
                markersize=8,
                label=f'CMF = {cmf}',
            )

    m_ref = np.linspace(0.8, 3.5, 50)
    ax.plot(m_ref, m_ref**0.27, 'k:', lw=1, alpha=0.5, label=r'$R \propto M^{0.27}$')
    ax.set_xlabel(r'Planet mass [$M_\oplus$]')
    ax.set_ylabel(r'Planet radius [$R_\oplus$]')
    ax.set_title('Mass-radius relation — Zalmoxis validation')
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, 3.5)
    fig.tight_layout()
    _save(fig, plot_dir, 'mass_radius.png')


def plot_resolution_convergence(results: dict, plot_dir: Path):
    """t_crystal and T_magma_final vs num_levels at 3 M_earth, CMF=0.325."""
    nlevels = [40, 60, 90, 120]
    t_vals, tm_vals = [], []

    for nl in nlevels:
        if nl == 60:
            name = 'B_M3.0_CMF0.325_ZAL'
        else:
            name = f'E_M3.0_CMF0.325_ZAL_n{nl}'
        hf = results.get(name)
        if hf is not None:
            t_vals.append(crystallization_time(hf))
            tm_vals.append(float(hf['T_magma'].iloc[-1]))
        else:
            t_vals.append(np.nan)
            tm_vals.append(np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(nlevels, t_vals, 'o-', color='#7570b3', lw=1.5, markersize=8)
    ax1.set_xlabel('Number of mesh levels')
    ax1.set_ylabel('Crystallization time [yr]')
    ax1.set_title('Resolution convergence — t_crystal')
    ax1.set_xticks(nlevels)

    ax2.plot(nlevels, tm_vals, 's-', color='#e7298a', lw=1.5, markersize=8)
    ax2.set_xlabel('Number of mesh levels')
    ax2.set_ylabel(r'Final $T_{\rm magma}$ [K]')
    ax2.set_title(r'Resolution convergence — $T_{\rm magma}$')
    ax2.set_xticks(nlevels)

    fig.suptitle(r'3 $M_\oplus$, CMF = 0.325, Zalmoxis', fontsize=13)
    fig.tight_layout()
    _save(fig, plot_dir, 'resolution_convergence.png')


def plot_entropy_sensitivity(results: dict, plot_dir: Path):
    """T_magma(t) for 3 initial entropies at 3 M_earth, 2 CMFs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    entropy_colors = {2600: '#1b9e77', 2800: '#7570b3', 3200: '#d95f02'}

    for ax, cmf in zip(axes, [0.10, 0.325]):
        for ini_s, color in entropy_colors.items():
            name = f'F_M3.0_CMF{cmf}_ZAL_S{ini_s}'
            hf = results.get(name)
            if hf is not None and 'T_magma' in hf.columns:
                t = hf['Time'].values
                mask = t > 0
                if mask.any():
                    ax.plot(
                        t[mask],
                        hf['T_magma'].values[mask],
                        '-',
                        color=color,
                        lw=1.5,
                        label=f'$S_0$ = {ini_s} J/(kg K)',
                    )
        ax.set_xscale('log')
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel(r'$T_{\rm magma}$ [K]')
        ax.set_title(f'CMF = {cmf}')
        ax.legend(fontsize=9)

    fig.suptitle(r'Initial entropy sensitivity — 3 $M_\oplus$, Zalmoxis', fontsize=13)
    fig.tight_layout()
    _save(fig, plot_dir, 'entropy_sensitivity.png')


# ── Validation checks ────────────────────────────────────────────


def run_checks(results: dict, outdir: Path) -> list[dict]:
    """Run all validation checks and return list of {name, status, detail}."""
    checks = []

    def add(name, passed, detail=''):
        checks.append({'name': name, 'status': 'PASS' if passed else 'FAIL', 'detail': detail})

    # Per-case plausibility
    for case_name, hf in sorted(results.items()):
        fails = []
        for col in ['T_magma', 'Phi_global', 'F_int', 'R_int', 'M_int']:
            if col in hf.columns:
                v = hf[col].values
                if np.any(np.isnan(v)):
                    fails.append(f'{col} contains NaN')
                if np.any(np.isinf(v)):
                    fails.append(f'{col} contains Inf')

        if 'T_magma' in hf.columns:
            T = hf['T_magma'].values
            for k in range(1, len(T)):
                if T[k] > T[k - 1] + 50:
                    fails.append(f'T_magma increase {T[k] - T[k - 1]:.0f} K at step {k}')
                    break

        if 'Phi_global' in hf.columns:
            phi = hf['Phi_global'].values
            for k in range(1, len(phi)):
                if phi[k] > phi[k - 1] + 0.02:
                    fails.append(f'Phi_global increase {phi[k] - phi[k - 1]:.3f} at step {k}')
                    break

        if 'F_int' in hf.columns and np.any(hf['F_int'].values < 0):
            fails.append('F_int negative')

        if 'R_int' in hf.columns:
            R = hf['R_int'].values
            if np.any(R < 3e6) or np.any(R > 15e6):
                fails.append(f'R_int outside [3e6, 15e6]: {R.min():.2e}-{R.max():.2e}')

        if hf['Time'].iloc[-1] <= 0:
            fails.append('No time progression')

        add(f'Plausibility: {case_name}', len(fails) == 0, '; '.join(fails))

    # AW vs ZAL comparison (Block A)
    for mass in MASSES:
        for cmf in CMFS:
            aw_hf = results.get(f'A_M{mass}_CMF{cmf}_AW')
            zal_hf = results.get(f'B_M{mass}_CMF{cmf}_ZAL')
            if aw_hf is None or zal_hf is None:
                continue

            # Tolerance depends on mass
            t_tol = 0.10 if mass <= 1.0 else 0.20
            tc_factor = 2.0 if mass <= 1.0 else 3.0

            T_aw = float(aw_hf['T_magma'].iloc[-1])
            T_zal = float(zal_hf['T_magma'].iloc[-1])
            if T_aw > 0:
                rel = abs(T_zal - T_aw) / T_aw
                add(
                    f'AW~ZAL T_magma M={mass} CMF={cmf}',
                    rel <= t_tol,
                    f'rel={rel:.1%}, AW={T_aw:.0f}, ZAL={T_zal:.0f}',
                )

            tc_aw = crystallization_time(aw_hf)
            tc_zal = crystallization_time(zal_hf)
            if tc_aw > 0 and tc_zal > 0:
                ratio = max(tc_aw, tc_zal) / min(tc_aw, tc_zal)
                add(
                    f'AW~ZAL t_crystal M={mass} CMF={cmf}',
                    ratio <= tc_factor,
                    f'ratio={ratio:.2f}, AW={tc_aw:.0f}, ZAL={tc_zal:.0f}',
                )

    # M-R scaling (Zalmoxis)
    for cmf in CMFS:
        radii = {}
        for mass in MASSES:
            hf = results.get(f'B_M{mass}_CMF{cmf}_ZAL')
            if hf is not None and 'R_int' in hf.columns:
                radii[mass] = float(hf['R_int'].iloc[0])
        ms = sorted(radii.keys())
        ok = all(radii[ms[k]] > radii[ms[k - 1]] for k in range(1, len(ms)))
        add(f'M-R scaling CMF={cmf}', ok or len(ms) < 2, f'R={[radii.get(m) for m in ms]}')

    # R-CMF scaling
    for mass in MASSES:
        radii = {}
        for cmf in CMFS:
            hf = results.get(f'B_M{mass}_CMF{cmf}_ZAL')
            if hf is not None and 'R_int' in hf.columns:
                radii[cmf] = float(hf['R_int'].iloc[0])
        cs = sorted(radii.keys(), reverse=True)
        ok = all(radii[cs[k]] > radii[cs[k - 1]] for k in range(1, len(cs)))
        add(f'R-CMF scaling M={mass}', ok or len(cs) < 2, f'R={[radii.get(c) for c in cs]}')

    # Adiabat mesh shift (Block H) — can only check if Phase 2 data exists
    for mass in [1.0, 3.0]:
        hf_ad = results.get(f'B_M{mass}_CMF0.325_ZAL')
        hf_lin = results.get(f'C_M{mass}_CMF0.325_ZAL_Tlin')
        if hf_ad is not None and hf_lin is not None:
            if 'R_int' in hf_ad.columns and len(hf_ad) >= 2:
                r0 = float(hf_ad['R_int'].iloc[0])
                r1 = float(hf_ad['R_int'].iloc[1])
                shift_ad = abs(r1 - r0) / r0 if r0 > 0 else np.nan
            else:
                shift_ad = np.nan
            if 'R_int' in hf_lin.columns and len(hf_lin) >= 2:
                r0 = float(hf_lin['R_int'].iloc[0])
                r1 = float(hf_lin['R_int'].iloc[1])
                shift_lin = abs(r1 - r0) / r0 if r0 > 0 else np.nan
            else:
                shift_lin = np.nan

            if np.isfinite(shift_ad) and mass >= 3.0:
                add(
                    f'Adiabat mesh shift M={mass}',
                    shift_ad < 0.10,
                    f'adiabat={shift_ad:.1%}, linear={shift_lin:.1%}',
                )

    # Phase 2 stability (Block D)
    for mass in MASSES:
        for cmf in CMFS:
            name = f'D_M{mass}_CMF{cmf}_ZAL_P2u100'
            hf = results.get(name)
            if hf is None:
                continue
            fails = []
            if 'T_magma' in hf.columns:
                T = hf['T_magma'].values
                for k in range(1, len(T)):
                    if abs(T[k] - T[k - 1]) > 200:
                        fails.append(f'T_magma jump {abs(T[k] - T[k - 1]):.0f} K')
                        break
            if 'R_int' in hf.columns:
                R = hf['R_int'].values
                for k in range(1, len(R)):
                    if R[k - 1] > 0 and abs(R[k] - R[k - 1]) / R[k - 1] > 0.05:
                        fails.append(f'R_int jump {abs(R[k] - R[k - 1]) / R[k - 1]:.1%}')
                        break
            add(f'P2 smooth: {name}', len(fails) == 0, '; '.join(fails))

    # Phase 2 vs Phase 1 agreement
    for mass in MASSES:
        ref_hf = results.get(f'B_M{mass}_CMF0.325_ZAL')
        p2_hf = results.get(f'D_M{mass}_CMF0.325_ZAL_P2u100')
        if ref_hf is not None and p2_hf is not None:
            tc_ref = crystallization_time(ref_hf)
            tc_p2 = crystallization_time(p2_hf)
            tol = 0.20 if mass <= 1.0 else 0.50
            if tc_ref > 0 and tc_p2 > 0:
                rel = abs(tc_p2 - tc_ref) / tc_ref
                add(
                    f'P2~P1 t_crystal M={mass}',
                    rel <= tol,
                    f'rel={rel:.1%}, P1={tc_ref:.0f}, P2={tc_p2:.0f}',
                )

    # Resolution convergence (Block E)
    ref_hf = results.get('B_M3.0_CMF0.325_ZAL')
    hi_hf = results.get('E_M3.0_CMF0.325_ZAL_n120')
    if ref_hf is not None and hi_hf is not None:
        tc_ref = crystallization_time(ref_hf)
        tc_hi = crystallization_time(hi_hf)
        if tc_ref > 0 and tc_hi > 0:
            rel = abs(tc_hi - tc_ref) / tc_ref
            add(
                'Resolution convergence n60 vs n120',
                rel <= 0.10,
                f'rel={rel:.0%}, n60={tc_ref:.0f}, n120={tc_hi:.0f}',
            )

    return checks


# ── Report generation ────────────────────────────────────────────


def build_test_matrix_table(outdir: Path, results: dict) -> str:
    """Build markdown table of all cases with status."""
    cases_dir = outdir / 'cases'
    search_dir = cases_dir if cases_dir.exists() else outdir

    rows = []
    for d in sorted(search_dir.iterdir()):
        if not d.is_dir():
            continue
        m = CASE_RE.match(d.name)
        if not m:
            continue
        status = case_status(d)
        hf = results.get(d.name)
        t_final = ''
        if hf is not None:
            t_final = f'{float(hf["Time"].iloc[-1]):.0f}'
        rows.append(
            f'| {d.name} | {m.group("block")} | {m.group("mass")} | '
            f'{m.group("cmf")} | {m.group("tag")} | {status} | {t_final} |'
        )

    header = '| Case | Block | Mass | CMF | Mode | Status | t_final [yr] |\n'
    header += '|------|-------|------|-----|------|--------|--------------|'
    return header + '\n' + '\n'.join(rows) if rows else '*No cases found.*'


def build_checks_table(checks: list[dict]) -> str:
    """Build markdown table of validation checks."""
    lines = ['| Check | Status | Detail |', '|-------|--------|--------|']
    for ch in checks:
        lines.append(f'| {ch["name"]} | **{ch["status"]}** | {ch["detail"]} |')
    return '\n'.join(lines)


def generate_markdown(outdir: Path, results: dict, checks: list[dict], plot_dir: Path) -> str:
    """Generate the validation report as Markdown."""
    n_pass = sum(1 for c in checks if c['status'] == 'PASS')
    n_fail = sum(1 for c in checks if c['status'] == 'FAIL')
    n_total = len(checks)
    now = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    # Collect PNG filenames
    pngs = sorted(f.name for f in plot_dir.glob('*.png')) if plot_dir.exists() else []

    sections = []

    # Executive summary
    sections.append(
        textwrap.dedent(f"""\
    # Zalmoxis-SPIDER Coupling Validation Report

    **Generated**: {now}
    **Cases analyzed**: {len(results)}
    **Validation checks**: {n_pass}/{n_total} passed, {n_fail} failed

    ## Executive Summary

    """)
    )

    if n_fail == 0:
        sections.append('All validation checks passed.\n\n')
    else:
        sections.append(f'{n_fail} check(s) failed. See details below.\n\n')

    # Test matrix
    sections.append('## Test Matrix\n\n')
    sections.append(build_test_matrix_table(outdir, results))
    sections.append('\n\n')

    # AW vs Zalmoxis (Block A)
    sections.append('## AW vs Zalmoxis Comparison (Block A)\n\n')
    for mass in MASSES:
        fname = f'evolution_{mass:.0f}M.png'
        if fname in pngs:
            sections.append(f'![{fname}](plots/{fname})\n\n')
    for fname in pngs:
        if fname.startswith('radial_comparison_'):
            sections.append(f'![{fname}](plots/{fname})\n\n')
    for fname in pngs:
        if fname.startswith('summary_'):
            sections.append(f'![{fname}](plots/{fname})\n\n')

    # Radial profiles
    sections.append('## Radial Profiles\n\n')
    for fname in pngs:
        if fname.startswith('radial_A_'):
            sections.append(f'![{fname}](plots/{fname})\n\n')

    # Adiabatic T mode (Block B vs C)
    sections.append('## Adiabatic vs Linear T Mode (Blocks B, C)\n\n')
    for fname in ['adiabat_mesh_shift.png', 'adiabat_vs_linear_evolution.png']:
        if fname in pngs:
            sections.append(f'![{fname}](plots/{fname})\n\n')

    # Phase 2 feedback (Block D)
    sections.append('## Phase 2 Structural Feedback (Block D)\n\n')
    for fname in ['phase2_stability.png', 'phase2_mesh_shift_timeline.png']:
        if fname in pngs:
            sections.append(f'![{fname}](plots/{fname})\n\n')

    # Resolution convergence
    sections.append('## Resolution Convergence (Block E)\n\n')
    if 'resolution_convergence.png' in pngs:
        sections.append('![resolution](plots/resolution_convergence.png)\n\n')

    # Mass-radius
    sections.append('## Mass-Radius Diagram\n\n')
    if 'mass_radius.png' in pngs:
        sections.append('![M-R](plots/mass_radius.png)\n\n')

    # Validation checklist
    sections.append('## Validation Checklist\n\n')
    sections.append(build_checks_table(checks))
    sections.append('\n')

    return ''.join(sections)


def generate_html(md_content: str, plot_dir: Path) -> str:
    """Convert Markdown report to self-contained HTML with embedded PNGs."""
    # Simple Markdown-to-HTML conversion for the report structure
    html_parts = [
        textwrap.dedent("""\
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Zalmoxis-SPIDER Validation Report</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
             sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px;
             line-height: 1.6; color: #333; }
      h1 { border-bottom: 2px solid #7570b3; padding-bottom: 10px; }
      h2 { border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 40px; }
      h3 { margin-top: 30px; }
      img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #eee; }
      table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
      th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: left; }
      th { background: #f5f5f5; }
      .pass { color: #4caf50; font-weight: bold; }
      .fail { color: #d32f2f; font-weight: bold; }
      code { background: #f5f5f5; padding: 2px 5px; border-radius: 3px; }
    </style>
    </head>
    <body>
    """)
    ]

    # Build base64 image cache
    img_cache = {}
    if plot_dir.exists():
        for png_file in plot_dir.glob('*.png'):
            data = png_file.read_bytes()
            img_cache[png_file.name] = base64.b64encode(data).decode('ascii')

    # Process Markdown line by line
    in_table = False
    for line in md_content.split('\n'):
        stripped = line.strip()

        # Headers
        if stripped.startswith('# ') and not stripped.startswith('## '):
            html_parts.append(f'<h1>{stripped[2:]}</h1>\n')
        elif stripped.startswith('## '):
            html_parts.append(f'<h2>{stripped[3:]}</h2>\n')
        elif stripped.startswith('### '):
            html_parts.append(f'<h3>{stripped[4:]}</h3>\n')

        # Images
        elif stripped.startswith('!['):
            import re as _re

            img_match = _re.match(r'!\[.*?\]\(plots/(.*?)\)', stripped)
            if img_match:
                fname = img_match.group(1)
                if fname in img_cache:
                    html_parts.append(
                        f'<img src="data:image/png;base64,{img_cache[fname]}" alt="{fname}">\n'
                    )

        # Table rows
        elif stripped.startswith('|'):
            if stripped.startswith('|---') or stripped.startswith('| ---'):
                continue  # Skip separator
            cells = [c.strip() for c in stripped.split('|')[1:-1]]
            if not in_table:
                html_parts.append('<table>\n<tr>')
                for c in cells:
                    html_parts.append(f'<th>{c}</th>')
                html_parts.append('</tr>\n')
                in_table = True
            else:
                html_parts.append('<tr>')
                for c in cells:
                    css = ''
                    if '**PASS**' in c:
                        css = ' class="pass"'
                        c = c.replace('**', '')
                    elif '**FAIL**' in c:
                        css = ' class="fail"'
                        c = c.replace('**', '')
                    html_parts.append(f'<td{css}>{c}</td>')
                html_parts.append('</tr>\n')
        else:
            if in_table:
                html_parts.append('</table>\n')
                in_table = False
            if stripped.startswith('**') and '**:' in stripped:
                # Bold label
                html_parts.append(f'<p>{stripped}</p>\n')
            elif stripped:
                html_parts.append(f'<p>{stripped}</p>\n')

    if in_table:
        html_parts.append('</table>\n')

    html_parts.append('</body>\n</html>\n')
    return ''.join(html_parts)


# ── Main ─────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='Generate self-contained validation report')
    parser.add_argument(
        '--outdir',
        type=Path,
        required=True,
        help='Root output directory (same as generate_habrok_grid.py)',
    )
    parser.add_argument(
        '--skip-plots', action='store_true', help='Skip plot generation (use cached PNGs)'
    )
    parser.add_argument(
        '--plots-only', action='store_true', help='Generate plots only, skip report'
    )
    args = parser.parse_args()

    outdir = args.outdir.resolve()
    plot_dir = outdir / 'plots'

    # Load results
    print(f'Loading results from {outdir}')
    results = load_all_helpfiles(outdir)
    print(f'Found {len(results)} completed cases\n')

    if not results:
        print('No results found. Run the Habrok grid first.')
        sys.exit(1)

    # Generate plots
    if not args.skip_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
        print('Generating plots...')
        plot_evolution(results, plot_dir)
        plot_radial_profiles(outdir, plot_dir)
        plot_radial_comparison(outdir, plot_dir)
        plot_adiabat_mesh_shift(results, outdir, plot_dir)
        plot_adiabat_vs_linear_evolution(results, plot_dir)
        plot_phase2_stability(results, plot_dir)
        plot_phase2_mesh_shift(results, plot_dir)
        plot_summary_heatmaps(results, plot_dir)
        plot_mass_radius(results, plot_dir)
        plot_resolution_convergence(results, plot_dir)
        print()

    if args.plots_only:
        return

    # Run validation checks
    print('Running validation checks...')
    checks = run_checks(results, outdir)
    n_pass = sum(1 for c in checks if c['status'] == 'PASS')
    n_fail = sum(1 for c in checks if c['status'] == 'FAIL')
    print(f'  {n_pass} passed, {n_fail} failed\n')

    # Generate Markdown report
    md = generate_markdown(outdir, results, checks, plot_dir)
    md_path = outdir / 'validation_report.md'
    md_path.write_text(md)
    print(f'Markdown report: {md_path}')

    # Generate HTML report
    html = generate_html(md, plot_dir)
    html_path = outdir / 'validation_report.html'
    html_path.write_text(html)
    print(f'HTML report:     {html_path}')

    # Print check summary
    print(f'\nValidation: {n_pass}/{len(checks)} passed', end='')
    if n_fail > 0:
        print(f', {n_fail} FAILED:')
        for ch in checks:
            if ch['status'] == 'FAIL':
                print(f'  FAIL  {ch["name"]}: {ch["detail"]}')
    else:
        print(' — all checks passed.')


if __name__ == '__main__':
    main()
