#!/usr/bin/env python3
"""Plot PROTEUS output overlaid on CHILI intercomparison data.

Generates comparison figures mirroring Nicholls et al. (in prep) using
the Wong colorblind-friendly palette. Overlays the current PROTEUS run
on the CHILI v2 submission and all other intercomparison models.

Figures produced:
    Fig 1 - Melt fraction vs time (Earth + Venus)
    Fig 2 - Solidification milestones (Earth)
    Fig 3 - Atmospheric composition at Phi=95% and 5% (Earth)
    Fig 4 - H and C mass budgets at solidification (Earth)
    Fig 5 - Atmospheric composition at Phi=95% and 5% (Venus)
    Fig 6 - fO2 vs surface temperature (Earth)
    Fig 7 - OLR vs melt fraction and surface temperature (Earth)
    Fig 8 - T_surf, rheological front, viscosity vs Phi (Earth)
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


def _find_row_at_phi(df, phi_target, phi_col='Phi_global', relax=PHI_RELAX):
    """Find the first row where phi <= target, with optional relaxation."""
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
        return pd.read_csv(path, comment='#')
    except (pd.errors.ParserError, FileNotFoundError, PermissionError):
        print(f'Warning: could not parse {path}')
        return None


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
        print('Warning: netCDF4 not installed, skipping interior profile extraction (Fig 8)')
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
        t = int(round(row['Time']))
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
            label=NS['label'],
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


def _milestone_time(df, phi_target):
    """Return the time (Myr) at which phi first drops below phi_target."""
    t = _get_col(df, 't', 'time')
    phi = _get_phi(df)
    if t is None or phi is None:
        return np.nan
    idx = phi <= phi_target
    if not idx.any():
        return np.nan
    return float(t[idx].iloc[0]) / 1e6


def _milestone_time_proteus(df, phi_target):
    """Return time (Myr) from a PROTEUS helpfile dataframe."""
    row = _find_row_at_phi(df, phi_target)
    if row is None:
        return np.nan
    return float(row['Time']) / 1e6


def plot_fig2(intercomp, pe, NS, out, grid_dir=None, proteus_grid=None):
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
            t_nom = _milestone_time(df_nom, phi_tgt) if df_nom is not None else np.nan

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
                    t_m = _milestone_time(df, phi_tgt)
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
        ax.set_xlim(1e-3, 300)
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
        if not idx.any():
            continue
        subset = df[idx]
        row = subset.loc[(phi[subset.index] - phi_tgt).abs().idxmin()]
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
                label=g,
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
                    markeredgecolor='black' if ip else 'none',
                    markeredgewidth=0.8 if ip else 0,
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
    _plot_atm_composition(intercomp, pv, 'venus', NS, out, 'chili_fig5_venus_atm')


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
def plot_fig6(intercomp, pe, NS, out):
    fig, ax = plt.subplots(figsize=(7, 5))

    for mk, st in MODELS.items():
        df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-earth-data.csv')
        if df is None:
            continue
        ts = _get_col(df, 'T_surf', 'Tsurf')
        fO2 = _get_col(df, 'fO2_melt', 'fO2')
        if ts is None or fO2 is None:
            continue
        valid = fO2 > 0
        if not valid.any():
            continue
        ax.plot(
            ts[valid],
            np.log10(fO2[valid]),
            linestyle=st.get('ls', '-'),
            color=st['color'],
            linewidth=st.get('lw', 1.2),
            alpha=0.8,
            label=st['label'],
        )

    if pe is not None:
        T = pe['T_surf'].values
        # Fischer et al. (2011) IW buffer: log10(fO2) = a + b/T
        # Coefficients match CALLIOPE's calliope.oxygen_fugacity module
        log10_fO2_IW = 6.94059 - 28180.8 / T
        iw_shift = (
            pe['fO2_shift_IW_derived'].values if 'fO2_shift_IW_derived' in pe.columns else 4.0
        )
        log10_fO2 = log10_fO2_IW + iw_shift
        ax.plot(
            T,
            log10_fO2,
            '-',
            color=NS['color'],
            linewidth=NS['linewidth'],
            label=NS['label'],
            zorder=NS['zorder'],
        )

    ax.set_xlabel('Surface temperature (K)')
    ax.set_ylabel(r'log$_{10}$(fO$_2$) (bar)')
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    _save(fig, out, 'chili_fig6_fO2_vs_T')


# ── Fig 7: OLR ───────────────────────────────────────────────────────
def plot_fig7(intercomp, pe, NS, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for mk, st in MODELS.items():
        df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-earth-data.csv')
        if df is None:
            continue
        phi = _get_phi(df)
        olr = _get_col(df, 'flux_OLR', 'F_olr')
        ts = _get_col(df, 'T_surf', 'Tsurf')
        if phi is None or olr is None:
            continue
        kw = dict(
            linestyle=st.get('ls', '-'),
            color=st['color'],
            linewidth=st.get('lw', 1.2),
            alpha=0.8,
            label=st['label'],
        )
        axes[0].plot(phi * 100, olr, **kw)
        if ts is not None:
            axes[1].plot(ts, olr, **kw)

    if pe is not None:
        kw = dict(
            color=NS['color'], linewidth=NS['linewidth'], label=NS['label'], zorder=NS['zorder']
        )
        axes[0].plot(pe['Phi_global'] * 100, pe['F_olr'], '-', **kw)
        axes[1].plot(pe['T_surf'], pe['F_olr'], '-', **kw)

    for i, (xl, tl) in enumerate(
        [
            ('Melt fraction (vol%)', '(a) OLR vs melt fraction'),
            ('Surface temperature (K)', '(b) OLR vs surface temperature'),
        ]
    ):
        axes[i].set_xlabel(xl)
        axes[i].set_ylabel(r'OLR (W m$^{-2}$)')
        axes[i].set_yscale('log')
        axes[i].set_title(tl, fontsize=12)
        axes[i].legend(fontsize=7, ncol=2, framealpha=0.9)
    axes[0].set_xlim(100, 0)
    fig.tight_layout()
    _save(fig, out, 'chili_fig7_olr')


# ── Fig 8: Geodynamics (T_surf, R_solid, viscosity) ──────────────────
def plot_fig8(intercomp, pe, NS, out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        ('T_surf', 'Tsurf', 'Surface temperature (K)', False),
        ('R_solid', 'R_solid', 'Rheological front (m)', False),
        ('viscosity', 'visc', r'Viscosity (Pa$\cdot$s)', True),
    ]

    for ax, (c1, c2, yl, logsc) in zip(axes, panels):
        for mk, st in MODELS.items():
            df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-earth-data.csv')
            if df is None:
                continue
            phi = _get_phi(df)
            val = _get_col(df, c1, c2)
            if phi is None or val is None:
                continue
            valid = val > 0 if logsc else val.notna()
            if not valid.any():
                continue
            ax.plot(
                phi[valid] * 100,
                val[valid],
                linestyle=st.get('ls', '-'),
                color=st['color'],
                linewidth=st.get('lw', 1.2),
                alpha=0.8,
                label=st['label'],
            )

        if pe is not None:
            profs = pe.attrs.get('_profiles', {})
            if c1 == 'T_surf' and 'T_surf' in pe.columns:
                ax.plot(
                    pe['Phi_global'] * 100,
                    pe['T_surf'],
                    '-',
                    color=NS['color'],
                    linewidth=NS['linewidth'],
                    label=NS['label'],
                    zorder=NS['zorder'],
                )
            elif c1 == 'R_solid' and len(profs.get('R_rheo', [])) > 0:
                ax.plot(
                    profs['Phi_global'] * 100,
                    profs['R_rheo'],
                    '-',
                    color=NS['color'],
                    linewidth=NS['linewidth'],
                    label=NS['label'],
                    zorder=NS['zorder'],
                )
            elif c1 == 'viscosity' and len(profs.get('visc_avg', [])) > 0:
                ax.plot(
                    profs['Phi_global'] * 100,
                    profs['visc_avg'],
                    '-',
                    color=NS['color'],
                    linewidth=NS['linewidth'],
                    label=NS['label'],
                    zorder=NS['zorder'],
                )

        ax.set_xlabel('Melt fraction (vol%)')
        ax.set_ylabel(yl)
        ax.set_xlim(100, 0)
        if logsc:
            ax.set_yscale('log')
        ax.legend(fontsize=6, ncol=2, framealpha=0.9)

    fig.tight_layout()
    _save(fig, out, 'chili_fig8_geodynamics')


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
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    _save(fig, out, 'chili_psurf_vs_time')


# ── Grid solidification timescales ──────────────────────────────────
def plot_grid_timescales(grid_dir, out):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    c_markers = {'Clow': 'o', 'Cmid': 's', 'Chigh': 'D'}
    c_colors = {'Clow': WONG[5], 'Cmid': WONG[6], 'Chigh': WONG[3]}

    for cl in ['Clow', 'Cmid', 'Chigh']:
        h_vals = []
        t_vals = []
        for hl in ['Hlow', 'Hmid', 'Hhigh']:
            run = grid_dir / f'chili_grid_earth_{hl}_{cl}'
            hf = run / 'runtime_helpfile.csv'
            if not hf.is_file():
                continue
            df = _load_proteus_helpfile(run)
            if df is None:
                continue
            row = _find_row_at_phi(df, 0.05)
            if row is None:
                continue
            t_sol = float(row['Time']) / 1e6
            h_vals.append(GRID_H[hl])
            t_vals.append(t_sol)

        if h_vals:
            ax.plot(
                np.array(h_vals) / 1e20,
                t_vals,
                marker=c_markers[cl],
                color=c_colors[cl],
                linewidth=2,
                markersize=8,
                label=GRID_C_LABELS[cl],
            )

    ax.set_xlabel(r'H inventory ($\times 10^{20}$ kg)')
    ax.set_ylabel('Solidification time (Myr)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10, framealpha=0.9)
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
        print('Figs 2-4, 6-8 will show CHILI models only')
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
        print('Fig 1 Venus and Fig 5 will show CHILI models only')

    plot_fig1(intercomp, pe, pv, NS, args.output)
    plot_fig2(intercomp, pe, NS, args.output, grid_dir=args.grid_dir)
    plot_fig3(intercomp, pe, NS, args.output)
    plot_fig4(intercomp, pe, NS, args.output)
    plot_fig5(intercomp, pv, NS, args.output)
    plot_fig6(intercomp, pe, NS, args.output)
    plot_fig7(intercomp, pe, NS, args.output)
    plot_fig8(intercomp, pe, NS, args.output)
    plot_psurf(intercomp, pe, NS, args.output)

    if args.grid_dir is not None:
        plot_grid_timescales(args.grid_dir, args.output)
    else:
        print('Skipping grid plot (pass --grid-dir to enable)')

    print(f'\nAll plots saved to {args.output}/')


if __name__ == '__main__':
    main()
