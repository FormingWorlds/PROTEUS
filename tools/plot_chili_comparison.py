#!/usr/bin/env python3
"""Plot PROTEUS output overlaid on CHILI intercomparison data.

Generates comparison figures mirroring Nicholls et al. (in prep) using
the Wong colorblind-friendly palette. Overlays the current PROTEUS run
on the CHILI v2 submission and all other intercomparison models.

Figures produced:
    Fig 1 - Melt fraction vs time (Earth + Venus)
    Fig 2 - Solidification milestones (Earth grid)
    Fig 3 - Atmospheric composition at Phi=95% and 5% (Earth)
    Fig 4 - H and C mass budgets at solidification (Earth)
    Fig 5 - Atmospheric composition at Phi=5% (Venus, single panel)
    Fig 6 - fO2 vs degassing temperature (Venus, absolute + relative)
    Fig 7 - Volatile retention vs time (Venus)
    Fig 8 - OLR vs melt fraction and surface temperature (Earth)
    Fig 9 - T_surf, R_RF/R_p, viscosity vs Phi (Earth)
    P_surf - Surface pressure vs time (Earth)
    Grid   - Solidification time vs H inventory (requires --grid-dir)

Usage:
    # Nominal cases only
    python tools/plot_chili_comparison.py \\
        --proteus-earth output/tutorial_earth/ \\
        --proteus-venus output/tutorial_venus/ \\
        --output output_files/chili_plots/

    # With the 3x3 Earth volatile grid
    python tools/plot_chili_comparison.py \\
        --proteus-earth output/tutorial_earth/ \\
        --proteus-venus output/tutorial_venus/ \\
        --grid-dir output/ \\
        --output output_files/chili_plots/

The CHILI comparison data is cloned automatically from GitHub if
--chili-repo does not point to an existing checkout.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, ScalarFormatter

CHILI_REPO_URL = 'https://github.com/projectcuisines/chili.git'

WONG = [
    '#000000',
    '#E69F00',
    '#56B4E9',
    '#009E73',
    '#F0E442',
    '#0072B2',
    '#D55E00',
    '#CC79A7',
]

MODELS = {
    'gooey': {'color': WONG[1], 'label': 'GOOEY'},
    'neongooey': {'color': WONG[2], 'label': 'NEONGOOEY'},
    'proteus': {'color': WONG[0], 'label': 'PROTEUS CHILI', 'ls': '--', 'lw': 1.5},
    'pacman': {'color': WONG[3], 'label': 'PACMAN'},
    'lincs': {'color': WONG[5], 'label': 'LINCS'},
    'moai': {'color': WONG[6], 'label': 'MOAI'},
    'planatmo': {'color': WONG[7], 'label': 'PlanAtMO'},
}

GAS_SPECIES = ['H2O', 'CO2', 'CO', 'H2', 'CH4', 'N2', 'S2', 'SO2', 'H2S']
GAS_LABELS = {
    'H2O': r'H$_2$O',
    'CO2': r'CO$_2$',
    'CO': 'CO',
    'H2': r'H$_2$',
    'CH4': r'CH$_4$',
    'N2': r'N$_2$',
    'S2': r'S$_2$',
    'SO2': r'SO$_2$',
    'H2S': r'H$_2$S',
}
GAS_COLORS = {
    'H2O': '#2196F3',
    'CO2': '#FF5722',
    'CO': '#FF9800',
    'H2': '#9C27B0',
    'CH4': '#4CAF50',
    'N2': '#795548',
    'S2': '#FFC107',
    'SO2': '#607D8B',
    'H2S': '#E91E63',
}

_PLOT_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
}

GRID_H = {
    'Hlow': 1.6e20,
    'Hmid': 7.8e20,
    'Hhigh': 16.0e20,
}
GRID_H_LABELS = {'Hlow': '1 EO', 'Hmid': '5 EO', 'Hhigh': '10 EO'}
GRID_C_LABELS = {
    'Clow': r'C$_\mathrm{low}$',
    'Cmid': r'C$_\mathrm{mid}$',
    'Chigh': r'C$_\mathrm{high}$',
}

PHI_RELAX = 0.01


def _get_proteus_sha():
    try:
        r = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        return r.stdout.strip()
    except Exception:
        return '?'


def _new_style(sha):
    return {
        'color': '#D55E00',
        'linewidth': 2.5,
        'label': f'PROTEUS ({sha})',
        'zorder': 10,
    }


def _get_col(df, *candidates):
    """Match column by exact stem (before parenthesised units). No substring fallback."""
    for cand in candidates:
        for c in df.columns:
            if cand.lower() == c.lower().split('(')[0].strip():
                return df[c]
    return None


def _get_phi(df):
    col = _get_col(df, 'phi', 'phi_global', 'melt_fraction', 'melt')
    if col is None:
        return None
    if col.max() > 1:
        col = col / 100
    return col


def _smooth(y, window=9, log=False):
    """Light centered rolling mean to damp per-timestep solver jitter.

    Applied only to the current PROTEUS trajectory; the submitted
    intercomparison data is always plotted unsmoothed. Both coordinates are
    smoothed so the line does not fold back on itself where the solver
    briefly reverses (T_surf has small step-to-step reversals). Quantities
    spanning several decades on a log axis (OLR) are averaged in log space.
    Endpoints are preserved exactly so the solidification value is not
    shifted.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return y
    if log:
        if (y <= 0).any():
            return y
        z = np.log10(y)
    else:
        z = y
    s = pd.Series(z).rolling(window=window, center=True, min_periods=1).mean().to_numpy()
    out = 10**s if log else s
    out[0] = y[0]
    out[-1] = y[-1]
    return out


def _find_row_at_phi(df, phi_target, phi_col='Phi_global', relax=PHI_RELAX):
    """Find the first row where phi <= target, with optional relaxation."""
    if phi_col not in df.columns:
        return None
    idx = df[phi_col] <= phi_target
    if idx.any():
        return df[idx].iloc[0]
    idx = df[phi_col] <= phi_target + relax
    if idx.any():
        print(f'  Warning: relaxed Phi threshold from {phi_target} to {phi_target + relax}')
        return df[idx].iloc[0]
    return None


def _load_chili_csv(path):
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, comment='#')
    except (pd.errors.ParserError, FileNotFoundError, PermissionError):
        print(f'Warning: could not parse {path}')
        return None
    # Drop the t=0 initialization row. Some models report a melt fraction of
    # zero there before the run starts, which otherwise places a spurious
    # point at full solidification and corrupts melt-fraction milestones.
    tcol = _get_col(df, 't', 'time')
    if tcol is not None:
        df = df[tcol > 0].reset_index(drop=True)
    return df


def _load_proteus_helpfile(output_dir):
    hf = output_dir / 'runtime_helpfile.csv'
    if not hf.is_file():
        return None
    try:
        df = pd.read_csv(hf, sep=r'\s+', engine='python')
    except (pd.errors.ParserError, FileNotFoundError, PermissionError):
        print(f'Warning: could not parse {hf}')
        return None
    df = df[df['Time'] > 0]
    if df.empty:
        return None
    df.attrs['_profiles'] = _extract_profiles(output_dir, df)
    return df


def _extract_profiles(output_dir, hf):
    try:
        import netCDF4 as nc
    except ImportError:
        print('Warning: netCDF4 not installed, skipping interior profile extraction (Fig 9)')
        return {'Phi_global': np.array([]), 'R_rheo': np.array([]), 'visc_avg': np.array([])}

    data_dir = output_dir / 'data'
    phi_crit = 0.4
    results = {'Phi_global': [], 'R_rheo': [], 'visc_avg': []}

    nc_index = {}
    if data_dir.is_dir():
        for c in data_dir.glob('*_int.nc'):
            try:
                nc_index[int(c.stem.split('_')[0])] = c
            except ValueError:
                continue

    for _, row in hf.iterrows():
        # Aragog names snapshots with %d (truncation), not rounding.
        t = int(row['Time'])
        ncf = nc_index.get(t)
        if ncf is None:
            for ct, cp in nc_index.items():
                if abs(ct - t) < 10:
                    ncf = cp
                    break
        if ncf is None:
            continue
        try:
            with nc.Dataset(str(ncf)) as ds:
                phi_s = ds.variables['phi_s'][:]
                r_s = ds.variables['radius_s'][:]
                visc_s = ds.variables['log10visc_s'][:]
                phi_g = float(ds.variables['phi_global'][:])
        except (KeyError, IndexError, OSError) as e:
            print(f'Warning: skipping {ncf.name}: {e}')
            continue

        solid = phi_s < phi_crit
        if solid.any() and not solid.all():
            r_rheo = r_s[solid][-1] * 1e3
        elif solid.all():
            r_rheo = r_s[-1] * 1e3
        else:
            r_rheo = r_s[0] * 1e3
        visc_avg = 10 ** np.mean(visc_s)
        results['Phi_global'].append(phi_g)
        results['R_rheo'].append(r_rheo)
        results['visc_avg'].append(visc_avg)

    for k in results:
        results[k] = np.array(results[k])
    if len(results['Phi_global']) == 0:
        print('Warning: no interior profiles loaded')
    return results


def ensure_chili_repo(chili_path):
    outputs = chili_path / 'intercomparison' / 'outputs'
    if chili_path.is_dir() and outputs.is_dir():
        return outputs
    subprocess.run(
        ['git', 'clone', '--depth', '1', CHILI_REPO_URL, str(chili_path)], check=True
    )
    return chili_path / 'intercomparison' / 'outputs'


# ── Fig 1: Melt fraction vs time ──────────────────────────────────────
def plot_fig1(intercomp, pe, pv, NS, out):
    fig, ax = plt.subplots(figsize=(7, 5))

    for mk, st in MODELS.items():
        for planet, ls in [('earth', '-'), ('venus', '--')]:
            df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-{planet}-data.csv')
            if df is None:
                continue
            t = _get_col(df, 't', 'time')
            phi = _get_phi(df)
            if t is None or phi is None:
                continue
            label = st['label'] if planet == 'earth' else None
            ax.plot(
                t / 1e6,
                phi,
                ls,
                color=st['color'],
                linewidth=st.get('lw', 1.2),
                alpha=0.8,
                label=label,
            )

    if pe is not None:
        ax.plot(
            pe['Time'] / 1e6,
            pe['Phi_global'],
            '-',
            color=NS['color'],
            linewidth=NS['linewidth'],
            label=NS['label'] + ' Earth',
            zorder=NS['zorder'],
        )
    if pv is not None:
        ax.plot(
            pv['Time'] / 1e6,
            pv['Phi_global'],
            '--',
            color=NS['color'],
            linewidth=NS['linewidth'],
            label=NS['label'] + ' Venus',
            zorder=NS['zorder'],
        )

    ax.set_xlabel('Time (Myr)')
    ax.set_ylabel('Melt fraction (by volume)')
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 1e1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2, loc='upper right', framealpha=0.9)
    _save(fig, out, 'chili_fig1_melt_fraction')


# ── Fig 2: Solidification milestones (3-panel, Nicholls+ style) ──────
H_LEVELS = ['Nmnl', 'Hlow', 'Hmid', 'Hhigh']
C_LEVELS = ['Clow', 'Cmid', 'Chigh']
C_ALPHAS = {'Clow': 0.45, 'Cmid': 0.7, 'Chigh': 1.0}


def _milestone_time(df, phi_target, fallback_last=False):
    """Return the time (Myr) at which phi first drops below phi_target.

    If fallback_last is True and phi never reaches the target, return the
    time of the final snapshot (the closest the run got to the target)
    instead of NaN. Used for the submitted PROTEUS CHILI runs, whose
    archived output stops just above 5% melt by volume.
    """
    t = _get_col(df, 't', 'time')
    phi = _get_phi(df)
    if t is None or phi is None:
        return np.nan
    idx = phi <= phi_target
    if idx.any():
        return float(t[idx].iloc[0]) / 1e6
    if fallback_last:
        return float(t.iloc[-1]) / 1e6
    return np.nan


def _milestone_time_proteus(df, phi_target):
    """Return time (Myr) from a PROTEUS helpfile dataframe."""
    row = _find_row_at_phi(df, phi_target)
    if row is None:
        return np.nan
    return float(row['Time']) / 1e6


def plot_fig2(intercomp, pe, NS, out, grid_dir=None):
    milestones = [0.95, 0.40, 0.05]
    titles = [
        r'$\mathbf{(a)}$ Time to reach $\phi$ = 95%',
        r'$\mathbf{(b)}$ Time to reach $\phi$ = 40%',
        r'$\mathbf{(c)}$ Time to reach $\phi$ = 5%',
    ]
    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    for ax, phi_tgt, title in zip(axes, milestones, titles):
        for mk, st in MODELS.items():
            # Nominal Earth (x marker at y=Nmnl)
            df_nom = _load_chili_csv(intercomp / mk / f'evolution-{mk}-earth-data.csv')
            use_last = mk == 'proteus'
            t_nom = (
                _milestone_time(df_nom, phi_tgt, fallback_last=use_last)
                if df_nom is not None
                else np.nan
            )

            if not np.isnan(t_nom):
                ax.plot(
                    t_nom,
                    0,
                    'x',
                    color=st['color'],
                    markersize=7,
                    markeredgewidth=1.5,
                    zorder=5,
                )

            # Grid cases: connected lines per C level
            for ci, cl in enumerate(C_LEVELS):
                y_vals = []
                t_vals = []
                for hi, hl in enumerate(['Hlow', 'Hmid', 'Hhigh']):
                    # Try H-C naming first, then H-only (GOOEY/LINCS)
                    df = _load_chili_csv(
                        intercomp / mk / f'evolution-{mk}-earth-grid-{hl}-{cl}-data.csv'
                    )
                    if df is None and ci == 0:
                        df = _load_chili_csv(
                            intercomp / mk / f'evolution-{mk}-earth-grid-{hl}-data.csv'
                        )
                    if df is None:
                        continue
                    t_m = _milestone_time(df, phi_tgt, fallback_last=use_last)
                    if not np.isnan(t_m):
                        y_vals.append(hi + 1)
                        t_vals.append(t_m)

                if t_vals:
                    label = st['label'] if ci == 1 and ax is axes[0] else None
                    ax.plot(
                        t_vals,
                        y_vals,
                        'o-',
                        color=st['color'],
                        alpha=C_ALPHAS[cl],
                        markersize=6,
                        linewidth=st.get('lw', 1.2),
                        label=label,
                        zorder=3,
                    )

        # Current PROTEUS run: nominal
        if pe is not None:
            t_nom = _milestone_time_proteus(pe, phi_tgt)
            if not np.isnan(t_nom):
                ax.plot(
                    t_nom,
                    0,
                    'x',
                    color=NS['color'],
                    markersize=11,
                    markeredgewidth=3.0,
                    zorder=10,
                )

        # Current PROTEUS run: grid cases
        if grid_dir is not None:
            for ci, cl in enumerate(C_LEVELS):
                y_vals = []
                t_vals = []
                for hi, hl in enumerate(['Hlow', 'Hmid', 'Hhigh']):
                    run = grid_dir / f'chili_grid_earth_{hl}_{cl}'
                    df = _load_proteus_helpfile(run) if run.is_dir() else None
                    if df is None:
                        continue
                    t_m = _milestone_time_proteus(df, phi_tgt)
                    if not np.isnan(t_m):
                        y_vals.append(hi + 1)
                        t_vals.append(t_m)

                if t_vals:
                    label = NS['label'] if ci == 1 and ax is axes[0] else None
                    ax.plot(
                        t_vals,
                        y_vals,
                        'o-',
                        color=NS['color'],
                        alpha=C_ALPHAS[cl],
                        markersize=9,
                        linewidth=3.0,
                        markeredgecolor='black',
                        markeredgewidth=0.6,
                        label=label,
                        zorder=10,
                    )

        ax.set_xscale('log')
        ax.set_xlim(1e-3, 500)
        ax.set_yticks(range(len(H_LEVELS)))
        ax.set_yticklabels(
            [r'Nmnl', r'H$_\mathrm{low}$', r'H$_\mathrm{mid}$', r'H$_\mathrm{high}$']
        )
        ax.set_ylim(-0.5, 3.5)
        ax.set_title(title, fontsize=12, loc='left')
        ax.grid(axis='x', alpha=0.15)

    axes[-1].set_xlabel('Earth simulated time [Myr]')
    axes[0].legend(fontsize=7, ncol=3, loc='lower right', framealpha=0.9)
    fig.tight_layout()
    _save(fig, out, 'chili_fig2_milestones')


# ── Figs 3 & 5: Atmospheric composition (stacked bars + T_surf stars) ─
def _collect_atm_panel(intercomp, proteus_df, planet, phi_tgt, NS):
    """Collect bar data and T_surf for one panel of the atmospheric composition plot.

    Returns bars ordered so that PROTEUS CHILI and the current PROTEUS run are adjacent.
    """
    pre_proteus = []
    proteus_pair = []
    post_proteus = []
    seen_chili_proteus = False

    for mk, st in MODELS.items():
        df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-{planet}-data.csv')
        if df is None:
            continue
        phi = _get_phi(df)
        if phi is None:
            continue
        idx = phi <= phi_tgt + 2 * PHI_RELAX
        if idx.any():
            subset = df[idx]
            row = subset.loc[(phi[subset.index] - phi_tgt).abs().idxmin()]
        else:
            # Run never reaches the target melt fraction; use its closest
            # approach (final snapshot) so the model still appears.
            row = df.loc[phi.idxmin()]
        entry = {
            'name': st['label'],
            'is_proteus': False,
            'gas': {},
            't_surf': np.nan,
        }
        for g in GAS_SPECIES:
            col = _get_col(df, f'p_{g}')
            val = float(row[col.name]) if col is not None and col.name in row.index else 0
            entry['gas'][g] = val if not np.isnan(val) else 0
        t_col = _get_col(df, 'T_surf')
        if t_col is not None:
            entry['t_surf'] = float(row[t_col.name])

        if mk == 'proteus':
            seen_chili_proteus = True
            proteus_pair.append(entry)
        elif seen_chili_proteus:
            post_proteus.append(entry)
        else:
            pre_proteus.append(entry)

    if proteus_df is not None:
        row = _find_row_at_phi(proteus_df, phi_tgt)
        if row is not None:
            entry = {
                'name': NS['label'],
                'is_proteus': True,
                'gas': {},
                't_surf': float(row.get('T_surf', np.nan)),
            }
            for g in GAS_SPECIES:
                entry['gas'][g] = float(row.get(f'{g}_bar', 0))
            proteus_pair.append(entry)

    ordered = pre_proteus + proteus_pair + post_proteus
    model_names = [e['name'] for e in ordered]
    gas_data = {g: [e['gas'].get(g, 0) for e in ordered] for g in GAS_SPECIES}
    t_surf_vals = [e['t_surf'] for e in ordered]
    is_proteus = [e['is_proteus'] for e in ordered]

    return model_names, gas_data, t_surf_vals, is_proteus


def _plot_atm_composition(intercomp, proteus_df, planet, NS, out, fig_name):
    """Stacked bar chart of partial pressures at Phi=95% and 5%, matching Nicholls+ layout."""
    phi_targets = [0.95, 0.05]
    planet_cap = planet.capitalize()
    panel_labels = [r'$\mathbf{(a)}$', r'$\mathbf{(b)}$']
    titles = [
        rf'Nominal-{planet_cap} atmosphere at $\phi$ = 95%',
        rf'Nominal-{planet_cap} atmosphere at $\phi$ = 5%',
    ]

    raw_panels = []
    for phi_tgt in phi_targets:
        raw_panels.append(_collect_atm_panel(intercomp, proteus_df, planet, phi_tgt, NS))

    all_names = []
    for names, _, _, _ in raw_panels:
        for n in names:
            if n not in all_names:
                all_names.append(n)
    n_slots = len(all_names)
    is_proteus_global = [
        any(e[3][e[0].index(n)] for e in raw_panels if n in e[0]) for n in all_names
    ]

    panels = []
    all_species = []
    max_bar = 0.0
    max_t = 0.0
    min_t = np.inf
    for names, gas, tsv, isp in raw_panels:
        name_to_idx = {n: i for i, n in enumerate(names)}
        gas_aligned = {g: [0.0] * n_slots for g in GAS_SPECIES}
        tsv_aligned = [np.nan] * n_slots
        isp_aligned = [False] * n_slots
        for j, slot_name in enumerate(all_names):
            if slot_name in name_to_idx:
                src = name_to_idx[slot_name]
                for g in GAS_SPECIES:
                    gas_aligned[g][j] = gas[g][src]
                tsv_aligned[j] = tsv[src]
                isp_aligned[j] = isp[src]

        present = [g for g in GAS_SPECIES if np.array(gas_aligned[g]).sum() > 0]
        for g in present:
            if g not in all_species:
                all_species.append(g)
        totals = np.zeros(n_slots)
        for g in present:
            totals += np.array(gas_aligned[g], dtype=float).clip(min=0)
        if n_slots > 0:
            max_bar = max(max_bar, totals.max())
        valid_t = [t for t in tsv_aligned if not np.isnan(t)]
        if valid_t:
            max_t = max(max_t, max(valid_t))
            min_t = min(min_t, min(valid_t))
        panels.append((gas_aligned, tsv_aligned, isp_aligned, present))

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    bar_ylim = max_bar * 1.15
    t_pad = (max_t - min_t) * 0.15
    t_ylim = (min_t - t_pad, max_t + t_pad)
    x = np.arange(n_slots)
    bar_width = 0.6

    for ax, (gas, tsv, isp, present), title, plabel in zip(axes, panels, titles, panel_labels):
        bottom = np.zeros(n_slots)
        for g in present:
            vals = np.array(gas[g], dtype=float).clip(min=0)
            edgecolors = ['black' if ip else 'white' for ip in isp]
            edgewidths = [2.0 if ip else 0.5 for ip in isp]
            ax.bar(
                x,
                vals,
                bottom=bottom,
                color=GAS_COLORS[g],
                label=GAS_LABELS.get(g, g),
                width=bar_width,
                edgecolor=edgecolors,
                linewidth=edgewidths,
            )
            bottom += vals

        ax.set_xticks(x)
        tick_labels = []
        for n, ip in zip(all_names, is_proteus_global):
            if ip:
                tick_labels.append(rf'$\bf{{{n}}}$')
            else:
                tick_labels.append(n)
        ax.set_xticklabels(tick_labels, fontsize=9, rotation=45, ha='right')
        for tl, ip in zip(ax.get_xticklabels(), is_proteus_global):
            if ip:
                tl.set_color(NS['color'])
        ax.set_ylabel('Partial pressure [bar]')
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, bar_ylim)
        ax.set_xlim(-0.5, n_slots - 0.5)
        ax.text(0.02, 0.93, plabel, transform=ax.transAxes, fontsize=14, va='top')

        ax2 = ax.twinx()
        for i, (t, ip) in enumerate(zip(tsv, isp)):
            if not np.isnan(t):
                ax2.plot(
                    i,
                    t,
                    marker='*',
                    color=NS['color'] if ip else 'grey',
                    markersize=16 if ip else 12,
                    markeredgecolor='white',
                    markeredgewidth=1.0,
                    zorder=10,
                    linestyle='none',
                )
        ax2.set_ylabel(r'$T_\mathrm{surf}$ [K]')
        ax2.set_ylim(*t_ylim)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    unique = [(h, la) for h, la in zip(handles, labels) if la not in seen and not seen.add(la)]
    axes[0].legend(
        [u[0] for u in unique],
        [u[1] for u in unique],
        fontsize=8,
        ncol=3,
        loc='upper right',
        framealpha=0.9,
    )
    fig.tight_layout()
    _save(fig, out, fig_name)


def plot_fig3(intercomp, pe, NS, out):
    _plot_atm_composition(intercomp, pe, 'earth', NS, out, 'chili_fig3_atm_composition')


def plot_fig5(intercomp, pv, NS, out):
    """Venus atmospheric composition at phi=5% only (single panel), matching Nicholls+ Fig 5."""
    phi_tgt = 0.05
    names, gas, tsv, isp = _collect_atm_panel(intercomp, pv, 'venus', phi_tgt, NS)
    if not names:
        print('Warning: no Venus data for Fig 5')
        return

    present = [g for g in GAS_SPECIES if np.array(gas[g]).sum() > 0]
    n = len(names)
    x = np.arange(n)
    is_proteus_global = [ip for ip in isp]

    fig, ax = plt.subplots(figsize=(8, 6))
    bottom = np.zeros(n)
    bar_width = 0.6

    for g in present:
        vals = np.array(gas[g], dtype=float).clip(min=0)
        edgecolors = ['black' if ip else 'white' for ip in isp]
        edgewidths = [2.0 if ip else 0.5 for ip in isp]
        ax.bar(
            x,
            vals,
            bottom=bottom,
            color=GAS_COLORS[g],
            label=GAS_LABELS.get(g, g),
            width=bar_width,
            edgecolor=edgecolors,
            linewidth=edgewidths,
        )
        bottom += vals

    ax.set_xticks(x)
    tick_labels = [rf'$\bf{{{n}}}$' if ip else n for n, ip in zip(names, is_proteus_global)]
    ax.set_xticklabels(tick_labels, fontsize=9, rotation=45, ha='right')
    for tl, ip in zip(ax.get_xticklabels(), is_proteus_global):
        if ip:
            tl.set_color(NS['color'])
    ax.set_ylabel('Partial pressure [bar]')
    ax.set_title(r'Nominal-Venus atmosphere at $\phi$ = 5%', fontsize=12)
    # Adopt the same axis limits as the Nominal Earth panel (Figure 3b).
    ax.set_ylim(0, 650)

    ax2 = ax.twinx()
    for i, (t, ip) in enumerate(zip(tsv, isp)):
        if not np.isnan(t):
            ax2.plot(
                i,
                t,
                marker='*',
                color=NS['color'] if ip else 'grey',
                markersize=16 if ip else 12,
                markeredgecolor='white',
                markeredgewidth=1.0,
                zorder=10,
                linestyle='none',
            )
    ax2.set_ylabel(r'$T_\mathrm{surf}$ [K]')
    ax2.set_ylim(1500, 4000)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, la) for h, la in zip(handles, labels) if la not in seen and not seen.add(la)]
    ax.legend(
        [u[0] for u in unique],
        [u[1] for u in unique],
        fontsize=8,
        ncol=3,
        loc='upper right',
        framealpha=0.9,
    )
    fig.tight_layout()
    _save(fig, out, 'chili_fig5_venus_atm')


# ── Fig 4: H, C mass budgets ─────────────────────────────────────────
def plot_fig4(intercomp, pe, NS, out):
    """H and C mass budgets in 3 reservoirs, matching Nicholls+ Fig 4 layout."""
    reservoirs = ['atm', 'melt', 'solid']
    res_titles = {
        'atm': 'Outgassed to atm.',
        'melt': 'Dissolved in melt',
        'solid': 'Stored in solid',
    }
    res_hatches = {'atm': '', 'melt': '..', 'solid': '//'}
    h_color = '#009E73'
    c_color = '#CC3311'
    panel_labels = [r'$\mathbf{(a)}$', r'$\mathbf{(b)}$', r'$\mathbf{(c)}$']

    model_names = []
    is_proteus = []
    data = {}
    seen_chili_proteus = False
    pre, pair, post = [], [], []

    for mk, st in MODELS.items():
        df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-earth-data.csv')
        if df is None:
            continue
        phi = _get_phi(df)
        if phi is None:
            continue
        idx = phi <= 0.05 + 2 * PHI_RELAX
        if idx.any():
            subset = df[idx]
            row = subset.loc[(phi[subset.index] - 0.05).abs().idxmin()]
        else:
            row = df.iloc[-1]
        entry = {'name': st['label'], 'is_proteus': False}
        for r in reservoirs:
            for el in ['H', 'C']:
                col = _get_col(df, f'mass{el}_{r}')
                entry[f'{el}_{r}'] = float(row[col.name]) if col is not None else 0
        if mk == 'proteus':
            seen_chili_proteus = True
            pair.append(entry)
        elif seen_chili_proteus:
            post.append(entry)
        else:
            pre.append(entry)

    if pe is not None:
        row = _find_row_at_phi(pe, 0.05)
        if row is None:
            row = pe.iloc[-1]
        entry = {'name': NS['label'], 'is_proteus': True}
        for r in reservoirs:
            hf_r = 'liquid' if r == 'melt' else r
            for el in ['H', 'C']:
                entry[f'{el}_{r}'] = float(row.get(f'{el}_kg_{hf_r}', 0))
        pair.append(entry)

    ordered = pre + pair + post
    model_names = [e['name'] for e in ordered]
    is_proteus = [e['is_proteus'] for e in ordered]
    for e in ordered:
        for key in e:
            if key not in ('name', 'is_proteus'):
                data.setdefault(key, []).append(e[key])

    n = len(model_names)
    x = np.arange(n)
    width = 0.35

    max_val = 0.0
    for r in reservoirs:
        for el in ['H', 'C']:
            vals = np.array(data.get(f'{el}_{r}', [0] * n), dtype=float)
            vals[vals < 0] = 0
            if len(vals) > 0:
                max_val = max(max_val, vals.max())

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ylim_top = max_val * 1.15 / 1e20

    for ax, r, plabel in zip(axes, reservoirs, panel_labels):
        h_vals = np.array(data.get(f'H_{r}', [0] * n), dtype=float).clip(min=0) / 1e20
        c_vals = np.array(data.get(f'C_{r}', [0] * n), dtype=float).clip(min=0) / 1e20

        for i in range(n):
            ec_h = 'black' if is_proteus[i] else 'white'
            ew_h = 2.0 if is_proteus[i] else 0.5
            ec_c = 'black' if is_proteus[i] else 'white'
            ew_c = 2.0 if is_proteus[i] else 0.5
            ax.bar(
                x[i] - width / 2,
                h_vals[i],
                color=h_color,
                width=width,
                hatch=res_hatches[r],
                edgecolor=ec_h,
                linewidth=ew_h,
            )
            ax.bar(
                x[i] + width / 2,
                c_vals[i],
                color=c_color,
                width=width,
                hatch=res_hatches[r],
                edgecolor=ec_c,
                linewidth=ew_c,
            )
            if r == 'atm' and h_vals[i] > 0:
                ax.text(
                    x[i] - width / 2,
                    h_vals[i] * 0.35,
                    'H',
                    ha='center',
                    va='center',
                    fontsize=7,
                    fontweight='bold',
                    color='white',
                )
            if r == 'atm' and c_vals[i] > 0:
                ax.text(
                    x[i] + width / 2,
                    c_vals[i] * 0.35,
                    'C',
                    ha='center',
                    va='center',
                    fontsize=7,
                    fontweight='bold',
                    color='white',
                )

        ax.set_ylim(0, ylim_top)
        ax.set_ylabel(r'$10^{20}$ kg')
        ax.text(
            0.02,
            0.92,
            f'{plabel} {res_titles[r]}',
            transform=ax.transAxes,
            fontsize=12,
            va='top',
        )
        ax.grid(axis='y', alpha=0.2)

    axes[-1].set_xticks(x)
    tick_labels = []
    for nm, ip in zip(model_names, is_proteus):
        if ip:
            tick_labels.append(rf'$\bf{{{nm}}}$')
        else:
            tick_labels.append(nm)
    axes[-1].set_xticklabels(tick_labels, fontsize=9, rotation=45, ha='right')
    for tl, ip in zip(axes[-1].get_xticklabels(), is_proteus):
        if ip:
            tl.set_color(NS['color'])

    fig.suptitle(r'Nominal-Earth H,C inventories at $\phi$ = 5%', fontsize=13)
    fig.tight_layout()
    _save(fig, out, 'chili_fig4_mass_budgets')


# ── Fig 6: fO2 vs temperature ────────────────────────────────────────
def _iw_fischer11(T):
    """Fischer et al. (2011) IW buffer: log10(fO2/bar)."""
    return 6.94059 - 28180.8 / T


def _iw_oneill02(T):
    """O'Neill & Eggins (2002) IW buffer: log10(fO2/bar).

    Includes heat-capacity term; matches CALLIOPE's oxygen_fugacity module.
    """
    return 2.0 * (-244118.0 + 115.559 * T - 8.474 * T * np.log(T)) / (np.log(10) * 8.31441 * T)


def plot_fig6(intercomp, pv, NS, out):
    """fO2 vs degassing temperature (Venus): (a) absolute, (b) relative to IW, matching Nicholls+ Fig 6."""
    fig, axes = plt.subplots(2, 1, figsize=(7, 9), sharex=True)
    ax_abs, ax_rel = axes

    T_ref = np.linspace(1400, 4000, 200)
    ax_abs.plot(
        T_ref, _iw_fischer11(T_ref), ':', color='grey', linewidth=1.5, label='Fischer+11'
    )
    ax_abs.plot(
        T_ref, _iw_oneill02(T_ref), '--', color='grey', linewidth=1.5, label="O'Neill+02"
    )
    ax_rel.axhline(0, color='grey', linestyle='--', linewidth=1.0)

    for mk, st in MODELS.items():
        df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-venus-data.csv')
        if df is None:
            continue
        ts = _get_col(df, 'T_surf', 'Tsurf')
        fO2 = _get_col(df, 'fO2_melt', 'fO2')
        phi = _get_phi(df)
        if ts is None or fO2 is None or phi is None:
            continue
        valid = fO2 > 0
        if not valid.any():
            continue
        log_fO2 = np.log10(fO2[valid])
        T_v = ts[valid].values
        kw = dict(color=st['color'], linewidth=st.get('lw', 1.2), alpha=0.8, label=st['label'])
        ax_abs.plot(T_v, log_fO2, linestyle=st.get('ls', '-'), **kw)
        delta_iw = log_fO2.values - _iw_oneill02(T_v)
        ax_rel.plot(T_v, delta_iw, linestyle=st.get('ls', '-'), **kw)

        idx5 = phi <= 0.05 + 2 * PHI_RELAX
        if idx5.any():
            subset = df[idx5]
            r5 = subset.loc[(phi[subset.index] - 0.05).abs().idxmin()]
            fO2_5 = float(fO2.loc[r5.name])
            T_5 = float(ts.loc[r5.name])
            if fO2_5 > 0:
                ax_abs.plot(
                    T_5, np.log10(fO2_5), 'o', color=st['color'], markersize=8, zorder=5
                )
                ax_rel.plot(
                    T_5,
                    np.log10(fO2_5) - _iw_oneill02(T_5),
                    'o',
                    color=st['color'],
                    markersize=8,
                    zorder=5,
                )

    if pv is not None:
        T = pv['T_surf'].values
        log10_fO2_IW = _iw_fischer11(T)
        iw_shift = (
            pv['fO2_shift_IW_derived'].values if 'fO2_shift_IW_derived' in pv.columns else 4.0
        )
        log10_fO2 = log10_fO2_IW + iw_shift
        kw = dict(
            color=NS['color'], linewidth=NS['linewidth'], label=NS['label'], zorder=NS['zorder']
        )
        ax_abs.plot(T, log10_fO2, '-', **kw)
        delta_iw = log10_fO2 - _iw_oneill02(T)
        ax_rel.plot(T, delta_iw, '-', **kw)

        row5 = _find_row_at_phi(pv, 0.05)
        if row5 is not None:
            T5 = float(row5['T_surf'])
            log_f5 = _iw_fischer11(T5) + float(row5.get('fO2_shift_IW_derived', 4.0))
            ax_abs.plot(
                T5,
                log_f5,
                'o',
                color=NS['color'],
                markersize=10,
                markeredgecolor='black',
                markeredgewidth=1.0,
                zorder=15,
            )
            ax_rel.plot(
                T5,
                log_f5 - _iw_oneill02(T5),
                'o',
                color=NS['color'],
                markersize=10,
                markeredgecolor='black',
                markeredgewidth=1.0,
                zorder=15,
            )

    ax_abs.set_ylabel(r'$\log_{10}(f\mathrm{O}_2 / \mathrm{bar})$')
    ax_abs.text(
        0.98,
        0.95,
        r'$\mathbf{(a)}$ Absolute $f$O$_2$',
        transform=ax_abs.transAxes,
        fontsize=12,
        va='top',
        ha='right',
    )
    ax_abs.legend(fontsize=7, ncol=2, loc='lower left', framealpha=0.9)
    ax_abs.invert_xaxis()
    ax_abs.set_xlim(3300, None)
    ax_abs.set_ylim(-10, None)
    ax_abs.grid(alpha=0.15)

    ax_rel.set_xlabel('Degassing temperature [K]')
    ax_rel.set_ylabel(r"$\Delta$IW (O'Neill+02)")
    ax_rel.text(
        0.98,
        0.95,
        r'$\mathbf{(b)}$ Relative $f$O$_2$',
        transform=ax_rel.transAxes,
        fontsize=12,
        va='top',
        ha='right',
    )
    ax_rel.set_xlim(3300, None)
    ax_rel.set_ylim(bottom=0)
    ax_rel.legend(fontsize=7, ncol=2, loc='lower left', framealpha=0.9)
    ax_rel.grid(alpha=0.15)

    fig.tight_layout()
    _save(fig, out, 'chili_fig6_fO2_vs_T')


# ── Fig 7: Volatile retention (Venus) ──────────────────────────────
def plot_fig7(intercomp, pv, NS, out):
    """Volatile retention % vs time for Venus, matching Nicholls+ Fig 7."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    ax_h, ax_c = axes
    panel_labels = [r'$\mathbf{(a)}$', r'$\mathbf{(b)}$']

    for mk, st in MODELS.items():
        df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-venus-data.csv')
        if df is None:
            continue
        t = _get_col(df, 't')
        phi = _get_phi(df)
        if t is None or phi is None:
            continue
        t_myr = t / 1e6
        for ax, element in zip(axes, ['H', 'C']):
            kw = dict(
                color=st['color'],
                linewidth=st.get('lw', 1.2),
                alpha=0.8,
                label=st['label'] if element == 'H' else None,
            )
            total_cols = [f'mass{element}_atm', f'mass{element}_melt', f'mass{element}_solid']
            total = np.zeros(len(df))
            for tc in total_cols:
                col = _get_col(df, tc)
                if col is not None:
                    total += col.values.astype(float)
            if total.max() <= 0:
                continue
            first_nonzero = np.argmax(total > 0)
            retained = total / total[first_nonzero] * 100
            retained = np.clip(retained, 0, 100)
            ax.plot(t_myr, retained, linestyle=st.get('ls', '-'), **kw)

            for phi_tgt, marker in [(0.95, 'x'), (0.05, 'o')]:
                idx = phi <= phi_tgt + 2 * PHI_RELAX
                if not idx.any():
                    continue
                subset = df[idx]
                r = subset.loc[(phi[subset.index] - phi_tgt).abs().idxmin()]
                ri = r.name
                iloc_ri = df.index.get_loc(ri)
                ax.plot(
                    float(t_myr.iloc[iloc_ri]),
                    float(retained[iloc_ri]),
                    marker=marker,
                    color=st['color'],
                    markersize=8,
                    zorder=5,
                    markeredgewidth=1.5 if marker == 'x' else 1.0,
                    linestyle='none',
                )

    if pv is not None:
        t_myr = pv['Time'].values / 1e6
        for ax, element in zip(axes, ['H', 'C']):
            kw = dict(
                color=NS['color'],
                linewidth=NS['linewidth'],
                zorder=NS['zorder'],
                label=NS['label'] if element == 'H' else None,
            )
            total_col = f'{element}_kg_total'
            if total_col not in pv.columns:
                continue
            total = pv[total_col].values
            if total.max() <= 0:
                continue
            first_nonzero = np.argmax(total > 0)
            retained = total / total[first_nonzero] * 100
            ax.plot(t_myr, retained, '-', **kw)
            for phi_tgt, marker in [(0.95, 'x'), (0.05, 'o')]:
                row = _find_row_at_phi(pv, phi_tgt)
                if row is not None:
                    ti = float(row['Time']) / 1e6
                    ri = float(row[total_col]) / total[0] * 100
                    ax.plot(
                        ti,
                        ri,
                        marker=marker,
                        color=NS['color'],
                        markersize=10,
                        markeredgecolor='black',
                        markeredgewidth=1.5 if marker == 'x' else 1.0,
                        zorder=15,
                        linestyle='none',
                    )

    for ax, ylabel, plabel in zip(axes, ['Hydrogen', 'Carbon'], panel_labels):
        ax.set_ylabel(ylabel)
        ax.set_ylim(-2, 105)
        ax.set_xscale('log')
        ax.set_xlim(1e-3, 300)
        ax.grid(alpha=0.15)
        ax.text(
            0.96,
            0.93,
            plabel,
            transform=ax.transAxes,
            fontsize=14,
            ha='right',
            va='top',
            fontweight='bold',
        )

    ax_c.set_xlabel('Simulated time [Myr]')

    # Single combined legend on the carbon panel: every code plus a
    # melt-fraction marker key (x at phi=95%, o at phi=5%).
    handles, labels = ax_h.get_legend_handles_labels()
    marker_key = [
        Line2D(
            [],
            [],
            marker='x',
            color='0.25',
            linestyle='none',
            markersize=8,
            markeredgewidth=1.5,
            label=r'$\phi = 95\%$',
        ),
        Line2D(
            [],
            [],
            marker='o',
            color='0.25',
            linestyle='none',
            markersize=8,
            label=r'$\phi = 5\%$',
        ),
    ]
    ax_c.legend(
        handles + marker_key,
        labels + [h.get_label() for h in marker_key],
        fontsize=7,
        ncol=2,
        loc='lower left',
        framealpha=0.9,
    )

    fig.suptitle('Nominal-Venus volatiles retained [%]', fontsize=13)
    fig.tight_layout()
    _save(fig, out, 'chili_fig7_volatiles')


# ── Fig 8: OLR ───────────────────────────────────────────────────────
def plot_fig8(intercomp, pe, NS, out):
    """OLR vs melt fraction and T_surf, matching Nicholls+ Fig 8."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    asr_val = 208.0  # 50 Myr ASR: L_sun(50Myr)/(4pi*1AU^2) * (1-0.1) * 0.375 * cos(48.19)
    sn_limit = 293.0  # Simpson-Nakajima steam runaway limit (Nakajima+1992)

    for mk, st in MODELS.items():
        df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-earth-data.csv')
        if df is None:
            continue
        phi = _get_phi(df)
        olr = _get_col(df, 'flux_OLR', 'F_olr')
        ts = _get_col(df, 'T_surf', 'Tsurf')
        if phi is None or olr is None:
            continue
        kw = dict(color=st['color'], linewidth=st.get('lw', 1.2), alpha=0.8, label=st['label'])
        axes[0].plot(phi * 100, olr, linestyle=st.get('ls', '-'), **kw)
        if ts is not None:
            axes[1].plot(ts, olr, linestyle=st.get('ls', '-'), **kw)

    if pe is not None:
        kw = dict(
            color=NS['color'], linewidth=NS['linewidth'], label=NS['label'], zorder=NS['zorder']
        )
        olr_smooth = _smooth(pe['F_olr'].values, log=True)
        phi_smooth = _smooth(pe['Phi_global'].values * 100)
        tsurf_smooth = _smooth(pe['T_surf'].values)
        axes[0].plot(phi_smooth, olr_smooth, '-', **kw)
        axes[1].plot(tsurf_smooth, olr_smooth, '-', **kw)

    for ax in axes:
        ax.set_yscale('log')
        ax.set_ylim(1e2, 5e5)
        ax.grid(alpha=0.15)

    axes[0].set_ylabel(r'Outgoing LW radiation [W/m$^2$]')

    # ASR reference on panel (a), Simpson-Nakajima limit on panel (b):
    # one reference line per panel, matching the paper layout.
    axes[0].axhline(asr_val, color='black', linestyle='--', linewidth=1.0)
    axes[0].text(
        0.03,
        asr_val * 1.1,
        f'50 Myr ASR\n({asr_val:.0f} W/m$^2$)',
        transform=axes[0].get_yaxis_transform(),
        fontsize=8,
        fontweight='bold',
        va='bottom',
        ha='left',
    )
    axes[1].axhline(sn_limit, color='black', linestyle='-.', linewidth=1.0)
    axes[1].text(
        0.03,
        sn_limit * 1.1,
        f'SN limit\n({sn_limit:.0f} W/m$^2$)',
        transform=axes[1].get_yaxis_transform(),
        fontsize=8,
        fontweight='bold',
        va='bottom',
        ha='left',
    )

    axes[0].text(
        0.03, 0.96, r'$\mathbf{(a)}$', transform=axes[0].transAxes, fontsize=14, va='top'
    )
    axes[0].set_xlabel('Melt fraction [vol%]')
    axes[0].set_xlim(100, 0)
    axes[0].legend(fontsize=7, ncol=1, loc='upper right', framealpha=0.9)

    axes[1].text(
        0.03, 0.96, r'$\mathbf{(b)}$', transform=axes[1].transAxes, fontsize=14, va='top'
    )
    axes[1].set_xlabel(r'$T_\mathrm{surf}$ [K]')
    axes[1].set_xlim(3400, 1400)

    fig.tight_layout()
    _save(fig, out, 'chili_fig8_olr')


# ── Fig 9: Geodynamics (T_surf, R_RF, viscosity vs phi) ──────────────
def plot_fig9(intercomp, pe, NS, out):
    """3-panel geodynamics: T_surf, R_RF/R_p, viscosity vs phi, matching Nicholls+ Fig 9."""
    # Radii are plotted in absolute units (Mm). The CHILI CSVs report R_solid /
    # R_trans in metres and carry no planet-radius column, so a fractional axis
    # would require a per-model surface radius that is not available; absolute
    # radius is the faithful representation of the shared quantity.
    R_cmb_Mm = 3.385  # PROTEUS Earth-case core radius (core mass fraction 0.325)
    if pe is not None and 'R_core' in getattr(pe, 'columns', []) and len(pe):
        R_cmb_Mm = float(pe['R_core'].iloc[0]) / 1e6
    visc_solid = 5e22  # solid Earth mantle viscosity [Pa s] (Peltier+1981, McKenzie 1967)
    visc_water = 1e-3  # water viscosity at STP [Pa s]

    fig, axes = plt.subplots(3, 1, figsize=(7, 11), sharex=True)
    ax_t, ax_r, ax_v = axes
    panel_labels = [r'$\mathbf{(a)}$', r'$\mathbf{(b)}$', r'$\mathbf{(c)}$']

    for mk, st in MODELS.items():
        df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-earth-data.csv')
        if df is None:
            continue
        phi = _get_phi(df)
        if phi is None:
            continue
        phi_pct = phi * 100
        kw = dict(color=st['color'], linewidth=st.get('lw', 1.2), alpha=0.8, label=st['label'])

        ts = _get_col(df, 'T_surf', 'Tsurf')
        if ts is not None:
            ax_t.plot(phi_pct, ts, linestyle=st.get('ls', '-'), **kw)

        r_sol = _get_col(df, 'R_solid', 'R_trans')
        if r_sol is not None:
            valid = r_sol > 0
            r_Mm = r_sol[valid] / 1e6
            ax_r.plot(phi_pct[valid], r_Mm, linestyle=st.get('ls', '-'), **kw)

        visc = _get_col(df, 'viscosity', 'visc')
        if visc is not None:
            valid = visc > 0
            if valid.any():
                ax_v.plot(phi_pct[valid], visc[valid], linestyle=st.get('ls', '-'), **kw)

    if pe is not None:
        profs = pe.attrs.get('_profiles', {})
        kw = dict(
            color=NS['color'], linewidth=NS['linewidth'], label=NS['label'], zorder=NS['zorder']
        )
        if 'T_surf' in pe.columns:
            ax_t.plot(
                _smooth(pe['Phi_global'].values * 100),
                _smooth(pe['T_surf'].values),
                '-',
                **kw,
            )
        if len(profs.get('R_rheo', [])) > 0:
            ax_r.plot(
                _smooth(profs['Phi_global'] * 100),
                _smooth(np.array(profs['R_rheo']) / 1e6),
                '-',
                **kw,
            )
        if len(profs.get('visc_avg', [])) > 0:
            ax_v.plot(
                _smooth(profs['Phi_global'] * 100),
                _smooth(profs['visc_avg'], log=True),
                '-',
                **kw,
            )

    _anno_box = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85)
    ax_r.axhline(R_cmb_Mm, color='black', linestyle='--', linewidth=1.0)
    ax_r.text(
        0.98,
        R_cmb_Mm,
        'Core-mantle boundary (PROTEUS)',
        transform=ax_r.get_yaxis_transform(),
        fontsize=8,
        va='top',
        ha='right',
        bbox=_anno_box,
    )
    ax_v.axhline(visc_solid, color='black', linestyle='--', linewidth=1.0)
    ax_v.text(
        0.5,
        visc_solid,
        'Solid mantle viscosity',
        transform=ax_v.get_yaxis_transform(),
        fontsize=8,
        va='bottom',
        ha='center',
        bbox=_anno_box,
    )
    ax_v.axhline(visc_water, color='black', linestyle=':', linewidth=1.0)
    ax_v.text(
        0.98,
        visc_water,
        'Water STP viscosity',
        transform=ax_v.get_yaxis_transform(),
        fontsize=8,
        va='bottom',
        ha='right',
        bbox=_anno_box,
    )

    ax_t.set_ylabel(r'$T_\mathrm{surf}$ [K]')
    ax_r.set_ylabel(r'$R_\mathrm{RF}$ [Mm]')
    ax_r.set_ylim(2.9, 7.2)
    ax_v.set_ylabel(r'$\eta$ [Pa s]')
    ax_v.set_yscale('log')
    ax_v.set_xlabel('Melt fraction [vol%]')

    for ax, plabel in zip(axes, panel_labels):
        ax.set_xlim(100, 0)
        ax.grid(alpha=0.15)
        ax.text(0.02, 0.93, plabel, transform=ax.transAxes, fontsize=14, va='top')

    ax_t.legend(fontsize=7, ncol=2, loc='lower left', framealpha=0.9)
    fig.tight_layout()
    _save(fig, out, 'chili_fig9_geodynamics')


# ── Surface pressure vs time ─────────────────────────────────────────
def plot_psurf(intercomp, pe, NS, out):
    fig, ax = plt.subplots(figsize=(7, 5))

    for mk, st in MODELS.items():
        df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-earth-data.csv')
        if df is None:
            continue
        t = _get_col(df, 't', 'time')
        p = _get_col(df, 'p_surf', 'P_surf')
        if t is None or p is None:
            continue
        ax.plot(
            t / 1e6,
            p,
            linestyle=st.get('ls', '-'),
            color=st['color'],
            linewidth=st.get('lw', 1.2),
            alpha=0.8,
            label=st['label'],
        )

    if pe is not None:
        ax.plot(
            pe['Time'] / 1e6,
            pe['P_surf'],
            '-',
            color=NS['color'],
            linewidth=NS['linewidth'],
            label=NS['label'],
            zorder=NS['zorder'],
        )

    ax.set_xlabel('Time (Myr)')
    ax.set_ylabel('Surface pressure (bar)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e-3, 1e1)
    # Bound the y-axis to the physical pressure range; one model reports a
    # spurious near-zero initial point that would otherwise stretch the axis
    # across 20 empty decades.
    ax.set_ylim(1e1, 1e5)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    _save(fig, out, 'chili_psurf_vs_time')


# ── Grid solidification timescales ──────────────────────────────────
def plot_grid_timescales(intercomp, grid_dir, NS, out):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    c_markers = {'Clow': 'o', 'Cmid': 's', 'Chigh': 'D'}
    c_colors = {'Clow': WONG[5], 'Cmid': WONG[6], 'Chigh': WONG[3]}
    h_levels = ['Hlow', 'Hmid', 'Hhigh']
    plotted_chili = False

    for cl in ['Clow', 'Cmid', 'Chigh']:
        # Submitted PROTEUS CHILI grid: dashed line, open markers.
        h_sub = []
        t_sub = []
        for hl in h_levels:
            df = _load_chili_csv(
                intercomp / 'proteus' / f'evolution-proteus-earth-grid-{hl}-{cl}-data.csv'
            )
            if df is None:
                continue
            t_sol = _milestone_time(df, 0.05)
            if np.isnan(t_sol):
                continue
            h_sub.append(GRID_H[hl])
            t_sub.append(t_sol)
        if h_sub:
            plotted_chili = True
            ax.plot(
                np.array(h_sub) / 1e20,
                t_sub,
                marker=c_markers[cl],
                color=c_colors[cl],
                linestyle='--',
                linewidth=1.5,
                markersize=8,
                markerfacecolor='white',
                zorder=4,
            )

        # Current PROTEUS run: solid line, filled markers.
        h_vals = []
        t_vals = []
        for hl in h_levels:
            run = grid_dir / f'chili_grid_earth_{hl}_{cl}'
            df = _load_proteus_helpfile(run) if run.is_dir() else None
            if df is None:
                continue
            row = _find_row_at_phi(df, 0.05)
            if row is None:
                continue
            h_vals.append(GRID_H[hl])
            t_vals.append(float(row['Time']) / 1e6)
        if h_vals:
            ax.plot(
                np.array(h_vals) / 1e20,
                t_vals,
                marker=c_markers[cl],
                color=c_colors[cl],
                linewidth=2,
                markersize=8,
                zorder=6,
            )

    # Legend: carbon level by colour and marker, data source by line style.
    c_handles = [
        Line2D(
            [],
            [],
            color=c_colors[cl],
            marker=c_markers[cl],
            linestyle='-',
            markersize=8,
            label=GRID_C_LABELS[cl],
        )
        for cl in ['Clow', 'Cmid', 'Chigh']
    ]
    src_handles = [
        Line2D(
            [],
            [],
            color='0.4',
            linestyle='-',
            linewidth=2,
            marker='o',
            markersize=7,
            label=NS['label'],
        ),
    ]
    # Only advertise PROTEUS CHILI in the legend if at least one submitted
    # point cleared the solidification threshold and was plotted.
    if plotted_chili:
        src_handles.append(
            Line2D(
                [],
                [],
                color='0.4',
                linestyle='--',
                linewidth=1.5,
                marker='o',
                markersize=7,
                markerfacecolor='white',
                label='PROTEUS CHILI',
            )
        )
    ax.legend(handles=c_handles + src_handles, fontsize=9, framealpha=0.9)

    ax.set_xlabel(r'H inventory ($\times 10^{20}$ kg)')
    ax.set_ylabel('Solidification time (Myr)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks([2, 3, 5, 7, 10, 15])
    ax.set_yticks([0.5, 0.7, 1, 2, 3, 5, 8])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    ax.set_title('CHILI Earth grid: solidification vs H inventory')
    _save(fig, out, 'chili_grid_solidification')


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(out / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {name}')


def main():
    parser = argparse.ArgumentParser(
        description='Plot PROTEUS output overlaid on CHILI intercomparison data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='See docs/Tutorials/chili_intercomparison.md for the full tutorial.',
    )
    parser.add_argument(
        '--proteus-earth',
        type=Path,
        default=None,
        help='Path to PROTEUS Nominal Earth output directory (e.g. output/tutorial_earth/)',
    )
    parser.add_argument(
        '--proteus-venus',
        type=Path,
        default=None,
        help='Path to PROTEUS Nominal Venus output directory (e.g. output/tutorial_venus/)',
    )
    parser.add_argument(
        '--chili-repo',
        type=Path,
        default=Path('/tmp/chili'),
        help='Path to CHILI repository checkout (cloned automatically if missing)',
    )
    parser.add_argument(
        '--grid-dir',
        type=Path,
        default=None,
        help='Parent directory containing chili_grid_earth_* output directories',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('output_files/chili_plots'),
        help='Output directory for plots (default: output_files/chili_plots/)',
    )
    args = parser.parse_args()

    matplotlib.rcParams.update(_PLOT_STYLE)

    intercomp = ensure_chili_repo(args.chili_repo)
    sha = _get_proteus_sha()
    NS = _new_style(sha)
    pe = _load_proteus_helpfile(args.proteus_earth) if args.proteus_earth else None
    pv = _load_proteus_helpfile(args.proteus_venus) if args.proteus_venus else None

    if pe is not None:
        print(
            f'Loaded PROTEUS Earth: {len(pe)} rows, '
            f'T={pe["T_magma"].max():.0f}->{pe["T_magma"].min():.0f} K, '
            f'Phi_min={pe["Phi_global"].min():.3f}'
        )
    elif args.proteus_earth:
        print(
            f'WARNING: --proteus-earth given but no data found in {args.proteus_earth}',
            file=sys.stderr,
        )
        print('Figs 2-4, 8-9 will show CHILI models only')
    if pv is not None:
        print(
            f'Loaded PROTEUS Venus: {len(pv)} rows, '
            f'T={pv["T_magma"].max():.0f}->{pv["T_magma"].min():.0f} K, '
            f'Phi_min={pv["Phi_global"].min():.3f}'
        )
    elif args.proteus_venus:
        print(
            f'WARNING: --proteus-venus given but no data found in {args.proteus_venus}',
            file=sys.stderr,
        )
        print('Figs 1, 5-7 (Venus) will show CHILI models only')

    plot_fig1(intercomp, pe, pv, NS, args.output)
    plot_fig2(intercomp, pe, NS, args.output, grid_dir=args.grid_dir)
    plot_fig3(intercomp, pe, NS, args.output)
    plot_fig4(intercomp, pe, NS, args.output)
    plot_fig5(intercomp, pv, NS, args.output)
    plot_fig6(intercomp, pv, NS, args.output)
    plot_fig7(intercomp, pv, NS, args.output)
    plot_fig8(intercomp, pe, NS, args.output)
    plot_fig9(intercomp, pe, NS, args.output)
    plot_psurf(intercomp, pe, NS, args.output)

    if args.grid_dir is not None:
        plot_grid_timescales(intercomp, args.grid_dir, NS, args.output)
    else:
        print('Skipping grid plot (pass --grid-dir to enable)')

    print(f'\nAll plots saved to {args.output}/')


if __name__ == '__main__':
    main()
