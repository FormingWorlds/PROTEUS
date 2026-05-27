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
    'proteus': {'color': WONG[0], 'label': 'PROTEUS CHILI', 'ls': '--', 'lw': 1.5},
    'neongooey': {'color': WONG[2], 'label': 'NEONGOOEY'},
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


# ── Fig 2: Solidification milestones ──────────────────────────────────
def plot_fig2(intercomp, pe, NS, out):
    milestones = [0.95, 0.40, 0.05]
    fig, ax = plt.subplots(figsize=(7, 5))

    for mk, st in MODELS.items():
        df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-earth-data.csv')
        if df is None:
            continue
        t = _get_col(df, 't', 'time')
        phi = _get_phi(df)
        if t is None or phi is None:
            continue
        times = []
        for m in milestones:
            idx = phi <= m
            times.append(float(t[idx].iloc[0]) / 1e3 if idx.any() else np.nan)
        ax.plot(
            milestones,
            times,
            'o-',
            color=st['color'],
            label=st['label'],
            markersize=5,
            linewidth=st.get('lw', 1.2),
            alpha=0.8,
        )

    if pe is not None:
        times = []
        for m in milestones:
            idx = pe['Phi_global'] <= m
            times.append(float(pe['Time'][idx].iloc[0]) / 1e3 if idx.any() else np.nan)
        ax.plot(
            milestones,
            times,
            's-',
            color=NS['color'],
            linewidth=NS['linewidth'],
            markersize=7,
            label=NS['label'],
            zorder=NS['zorder'],
        )

    ax.set_xlabel(r'Melt fraction milestone ($\Phi$)')
    ax.set_ylabel('Time to reach milestone (kyr)')
    ax.set_yscale('log')
    ax.set_xlim(1.0, 0.0)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    _save(fig, out, 'chili_fig2_milestones')


# ── Figs 3 & 5: Atmospheric composition (grouped bars) ───────────────
def _plot_atm_composition(intercomp, proteus_df, planet, NS, out, fig_name):
    """Grouped bar chart of partial pressures at Phi=95% and 5%."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    phi_targets = [0.95, 0.05]
    planet_cap = planet.capitalize()
    titles = [rf'{planet_cap} $\Phi$ = 95%', rf'{planet_cap} $\Phi$ = 5%']

    active_species = []

    for ax, phi_tgt, title in zip(axes, phi_targets, titles):
        model_names = []
        gas_data = {g: [] for g in GAS_SPECIES}

        for mk, st in MODELS.items():
            df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-{planet}-data.csv')
            if df is None:
                continue
            phi = _get_phi(df)
            if phi is None:
                continue
            idx = phi <= phi_tgt
            if not idx.any():
                continue
            row = df[idx].iloc[0]
            model_names.append(st['label'])
            for g in GAS_SPECIES:
                col = _get_col(df, f'p_{g}')
                val = float(row[col.name]) if col is not None and col.name in row.index else 0
                gas_data[g].append(val if not np.isnan(val) else 0)

        if proteus_df is not None:
            row = _find_row_at_phi(proteus_df, phi_tgt)
            if row is not None:
                model_names.append(NS['label'])
                for g in GAS_SPECIES:
                    gas_data[g].append(float(row.get(f'{g}_bar', 0)))

        if not model_names:
            continue

        present = [g for g in GAS_SPECIES if np.array(gas_data[g]).sum() > 0]
        for g in present:
            if g not in active_species:
                active_species.append(g)

        n_models = len(model_names)
        n_species = len(present)
        if n_species == 0:
            continue
        bar_width = 0.7 / n_species
        x = np.arange(n_models)
        for i, g in enumerate(present):
            vals = np.array(gas_data[g])
            vals[vals <= 0] = np.nan
            ax.bar(
                x + (i - n_species / 2 + 0.5) * bar_width,
                vals,
                color=GAS_COLORS[g],
                label=g,
                width=bar_width,
                edgecolor='white',
                linewidth=0.3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel('Partial pressure (bar)')
        ax.set_title(title, fontsize=12)
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.01)

    axes[0].legend(fontsize=7, ncol=3, loc='upper left', framealpha=0.9)
    fig.tight_layout()
    _save(fig, out, fig_name)


def plot_fig3(intercomp, pe, NS, out):
    _plot_atm_composition(intercomp, pe, 'earth', NS, out, 'chili_fig3_atm_composition')


def plot_fig5(intercomp, pv, NS, out):
    _plot_atm_composition(intercomp, pv, 'venus', NS, out, 'chili_fig5_venus_atm')


# ── Fig 4: H, C mass budgets ─────────────────────────────────────────
def plot_fig4(intercomp, pe, NS, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    reservoirs = ['atm', 'melt', 'solid']
    res_colors = {'atm': WONG[2], 'melt': WONG[6], 'solid': WONG[3]}

    for ax, element in zip(axes, ['H', 'C']):
        model_names = []
        res_data = {r: [] for r in reservoirs}

        for mk, st in MODELS.items():
            df = _load_chili_csv(intercomp / mk / f'evolution-{mk}-earth-data.csv')
            if df is None:
                continue
            phi = _get_phi(df)
            if phi is None:
                continue
            row = df[phi <= 0.05].iloc[0] if (phi <= 0.05).any() else df.iloc[-1]
            model_names.append(st['label'])
            for r in reservoirs:
                col = _get_col(df, f'mass{element}_{r}')
                res_data[r].append(float(row[col.name]) if col is not None else 0)

        if pe is not None:
            row = _find_row_at_phi(pe, 0.05)
            if row is None:
                row = pe.iloc[-1]
            model_names.append(NS['label'])
            for r in reservoirs:
                hf_r = 'liquid' if r == 'melt' else r
                res_data[r].append(float(row.get(f'{element}_kg_{hf_r}', 0)))

        if not model_names:
            continue
        x = np.arange(len(model_names))
        width = 0.25
        for i, r in enumerate(reservoirs):
            vals = np.array(res_data[r], dtype=float)
            vals[vals <= 0] = np.nan
            ax.bar(
                x + (i - 1) * width,
                vals,
                color=res_colors[r],
                label=r.capitalize(),
                width=width,
                edgecolor='white',
                linewidth=0.3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=7, rotation=45, ha='right')
        ax.set_ylabel(f'{element} mass (kg)')
        ax.set_title(f'{element} budget at solidification', fontsize=12)
        ax.set_yscale('log')

    axes[0].legend(fontsize=8, framealpha=0.9)
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
    fig.savefig(out / f'{name}.pdf', dpi=200, bbox_inches='tight')
    fig.savefig(out / f'{name}.png', dpi=200, bbox_inches='tight')
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
    plot_fig2(intercomp, pe, NS, args.output)
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
