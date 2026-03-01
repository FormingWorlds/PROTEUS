#!/usr/bin/env python3
"""Comprehensive analysis of Habrok validation grid results.

Produces publication-quality plots comparing AW vs ZAL mesh implementations
across the full parameter space, with rigorous assessment of numerical
accuracy, physical robustness, and EOS validity limits.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Load data ──────────────────────────────────────────────────────

DATA_FILE = Path(__file__).parent / 'all_data.json'
OUTDIR = Path(__file__).parent / 'plots'
OUTDIR.mkdir(exist_ok=True)

with open(DATA_FILE) as f:
    raw = json.load(f)


# Parse case metadata from name
def parse_case(name):
    """Extract block, mass, CMF, mode, and extras from case name."""
    parts = name.split('_')
    block = parts[0]
    mass = float(parts[1][1:])  # M5.0 -> 5.0
    cmf = float(parts[2][3:])  # CMF0.325 -> 0.325
    mode = parts[3]  # AW or ZAL
    extras = {}
    for p in parts[4:]:
        if p.startswith('P2u'):
            extras['update_interval'] = float(p[3:])
        elif p.startswith('n'):
            extras['num_levels'] = int(p[1:])
        elif p.startswith('S'):
            extras['ini_entropy'] = float(p[1:])
    return {'block': block, 'mass': mass, 'cmf': cmf, 'mode': mode, **extras}


cases = {}
for name, rows in raw.items():
    meta = parse_case(name)
    meta['name'] = name
    meta['rows'] = rows
    meta['steps'] = len(rows)
    if rows:
        last = rows[-1]
        meta['t_final'] = last['Time']
        meta['T_magma_final'] = last['T_magma']
        meta['Phi_final'] = last['Phi_global']
        meta['solidified'] = last['Phi_global'] < 0.01
        meta['R_int'] = last['R_int']
        meta['M_int'] = last['M_int']
        meta['M_core'] = last['M_core']
        meta['gravity'] = last['gravity']
    cases[name] = meta

# Physical constants
M_EARTH = 5.972e24  # kg
R_EARTH = 6.371e6  # m

# ── Color scheme ───────────────────────────────────────────────────

AW_COLOR = '#2166ac'  # blue
ZAL_COLOR = '#b2182b'  # red
CMF_COLORS = {0.10: '#1b9e77', 0.20: '#d95f02', 0.325: '#7570b3', 0.50: '#e7298a'}
MASS_MARKERS = {0.5: 'o', 1.0: 's', 2.0: 'D', 3.0: '^', 5.0: 'v', 7.0: 'P', 10.0: 'X'}

plt.rcParams.update(
    {
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 150,
    }
)


# ══════════════════════════════════════════════════════════════════
# FIGURE 1: T_magma evolution — AW vs ZAL, Block A comparison
# ══════════════════════════════════════════════════════════════════


def plot_thermal_evolution_comparison():
    """Block A: T_magma(t) for all mass-CMF pairs, AW vs ZAL side-by-side."""
    masses = [0.5, 1.0, 2.0, 3.0, 5.0]
    cmfs = [0.10, 0.20, 0.325, 0.50]

    fig, axes = plt.subplots(
        len(masses), len(cmfs), figsize=(20, 22), sharex=False, sharey=False
    )
    fig.suptitle(
        'Block A: Thermal Evolution — AW (blue) vs ZAL (red)',
        fontsize=16,
        fontweight='bold',
        y=0.98,
    )

    for i, mass in enumerate(masses):
        for j, cmf in enumerate(cmfs):
            ax = axes[i, j]
            aw_name = f'A_M{mass}_CMF{cmf}_AW'
            zal_name = f'A_M{mass}_CMF{cmf}_ZAL'

            for name, color, label in [(aw_name, AW_COLOR, 'AW'), (zal_name, ZAL_COLOR, 'ZAL')]:
                if name in cases:
                    c = cases[name]
                    t = [r['Time'] for r in c['rows']]
                    T = [r['T_magma'] for r in c['rows']]
                    if t and max(t) > 0:
                        ax.plot(t, T, color=color, linewidth=1.5, label=label)

            ax.set_title(f'M={mass} M$_\\oplus$, CMF={cmf}', fontsize=10)
            if i == len(masses) - 1:
                ax.set_xlabel('Time [yr]')
            if j == 0:
                ax.set_ylabel('$T_\\mathrm{magma}$ [K]')
            ax.set_ylim(0, 4000)
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=8)

            # Annotate stuck cases
            for name, color in [(aw_name, AW_COLOR), (zal_name, ZAL_COLOR)]:
                if name in cases:
                    c = cases[name]
                    if c['steps'] < 15 and c['t_final'] < 500:
                        ax.text(
                            0.5,
                            0.5,
                            f'{"AW" if "AW" in name else "ZAL"}\nSTUCK',
                            transform=ax.transAxes,
                            ha='center',
                            va='center',
                            fontsize=8,
                            color=color,
                            alpha=0.6,
                            fontweight='bold',
                        )

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTDIR / 'fig1_thermal_evolution_block_A.png')
    plt.close(fig)
    print('Fig 1: Thermal evolution (Block A)')


# ══════════════════════════════════════════════════════════════════
# FIGURE 2: Melt fraction evolution — AW vs ZAL
# ══════════════════════════════════════════════════════════════════


def plot_melt_fraction_evolution():
    """Block A: Phi_global(t) for all mass-CMF pairs."""
    masses = [0.5, 1.0, 2.0, 3.0, 5.0]
    cmfs = [0.10, 0.20, 0.325, 0.50]

    fig, axes = plt.subplots(
        len(masses), len(cmfs), figsize=(20, 22), sharex=False, sharey=True
    )
    fig.suptitle(
        'Block A: Melt Fraction Evolution — AW (blue) vs ZAL (red)',
        fontsize=16,
        fontweight='bold',
        y=0.98,
    )

    for i, mass in enumerate(masses):
        for j, cmf in enumerate(cmfs):
            ax = axes[i, j]
            aw_name = f'A_M{mass}_CMF{cmf}_AW'
            zal_name = f'A_M{mass}_CMF{cmf}_ZAL'

            for name, color, label in [(aw_name, AW_COLOR, 'AW'), (zal_name, ZAL_COLOR, 'ZAL')]:
                if name in cases:
                    c = cases[name]
                    t = [r['Time'] for r in c['rows']]
                    phi = [r['Phi_global'] for r in c['rows']]
                    if t and max(t) > 0:
                        ax.plot(t, phi, color=color, linewidth=1.5, label=label)

            # Solidification threshold
            ax.axhline(0.01, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

            ax.set_title(f'M={mass} M$_\\oplus$, CMF={cmf}', fontsize=10)
            if i == len(masses) - 1:
                ax.set_xlabel('Time [yr]')
            if j == 0:
                ax.set_ylabel('$\\Phi_\\mathrm{global}$')
            ax.set_ylim(-0.02, 1.05)
            if i == 0 and j == 0:
                ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUTDIR / 'fig2_melt_fraction_block_A.png')
    plt.close(fig)
    print('Fig 2: Melt fraction evolution (Block A)')


# ══════════════════════════════════════════════════════════════════
# FIGURE 3: Solidification time comparison — scatter plot
# ══════════════════════════════════════════════════════════════════


def plot_solidification_times():
    """Parity plot: AW vs ZAL solidification times for cases where both completed."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Solidification time parity
    ax = axes[0]
    masses = [0.5, 1.0, 2.0, 3.0, 5.0]
    cmfs = [0.10, 0.20, 0.325, 0.50]

    for mass in masses:
        for cmf in cmfs:
            aw_name = f'A_M{mass}_CMF{cmf}_AW'
            zal_name = f'A_M{mass}_CMF{cmf}_ZAL'
            if aw_name in cases and zal_name in cases:
                aw = cases[aw_name]
                zal = cases[zal_name]
                if aw['solidified'] and zal['solidified']:
                    ax.scatter(
                        aw['t_final'],
                        zal['t_final'],
                        color=CMF_COLORS[cmf],
                        marker=MASS_MARKERS[mass],
                        s=100,
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=5,
                    )

    # 1:1 line
    lims = [1e3, 1e6]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5, label='1:1')
    # +/- 20% bands
    ax.fill_between(
        lims,
        [l * 0.8 for l in lims],
        [l * 1.2 for l in lims],
        alpha=0.1,
        color='gray',
        label='$\\pm$20%',
    )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('AW solidification time [yr]')
    ax.set_ylabel('ZAL solidification time [yr]')
    ax.set_title('(a) Solidification Time Parity', fontweight='bold')
    ax.set_aspect('equal')

    # Legend for masses and CMFs
    mass_handles = [
        Line2D(
            [0],
            [0],
            marker=MASS_MARKERS[m],
            color='gray',
            markersize=8,
            linestyle='none',
            label=f'{m} M$_\\oplus$',
        )
        for m in masses
    ]
    cmf_handles = [
        Line2D(
            [0],
            [0],
            marker='o',
            color=CMF_COLORS[c],
            markersize=8,
            linestyle='none',
            label=f'CMF={c}',
        )
        for c in cmfs
    ]
    ax.legend(handles=mass_handles + cmf_handles, loc='upper left', fontsize=7, ncol=2)

    # Panel B: Final T_magma parity
    ax = axes[1]
    for mass in masses:
        for cmf in cmfs:
            aw_name = f'A_M{mass}_CMF{cmf}_AW'
            zal_name = f'A_M{mass}_CMF{cmf}_ZAL'
            if aw_name in cases and zal_name in cases:
                aw = cases[aw_name]
                zal = cases[zal_name]
                if aw['solidified'] and zal['solidified']:
                    ax.scatter(
                        aw['T_magma_final'],
                        zal['T_magma_final'],
                        color=CMF_COLORS[cmf],
                        marker=MASS_MARKERS[mass],
                        s=100,
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=5,
                    )

    lims = [1400, 1900]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    ax.fill_between(
        lims,
        [l * 0.95 for l in lims],
        [l * 1.05 for l in lims],
        alpha=0.1,
        color='gray',
        label='$\\pm$5%',
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel('AW final $T_\\mathrm{magma}$ [K]')
    ax.set_ylabel('ZAL final $T_\\mathrm{magma}$ [K]')
    ax.set_title('(b) Final Magma Ocean Temperature Parity', fontweight='bold')
    ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig3_solidification_parity.png')
    plt.close(fig)
    print('Fig 3: Solidification parity')


# ══════════════════════════════════════════════════════════════════
# FIGURE 4: Mass-CMF stability map
# ══════════════════════════════════════════════════════════════════


def plot_stability_map():
    """Which mass-CMF combinations solidify, persist in partial melt, or crash?"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax_idx, (mode, title) in enumerate([('AW', 'Adams-Williamson'), ('ZAL', 'Zalmoxis')]):
        ax = axes[ax_idx]
        masses_all = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        cmfs_all = [0.10, 0.20, 0.325, 0.50]

        for mass in masses_all:
            for cmf in cmfs_all:
                # Find matching case (Block A, B, or C)
                found = None
                for block in ['A', 'B', 'C']:
                    name = f'{block}_M{mass}_CMF{cmf}_{mode}'
                    if name in cases:
                        found = cases[name]
                        break

                if found is None:
                    # EOS limit failure or not in grid
                    ax.scatter(
                        cmf, mass, marker='x', color='black', s=80, linewidths=2, zorder=5
                    )
                elif found['solidified']:
                    ax.scatter(
                        cmf,
                        mass,
                        marker='o',
                        color='green',
                        s=120,
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=5,
                    )
                elif found['steps'] < 15 and found['t_final'] < 500:
                    ax.scatter(
                        cmf,
                        mass,
                        marker='s',
                        color='red',
                        s=120,
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=5,
                    )
                elif found['Phi_final'] > 0.01 and found['t_final'] > 5e5:
                    ax.scatter(
                        cmf,
                        mass,
                        marker='D',
                        color='orange',
                        s=120,
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=5,
                    )
                else:
                    ax.scatter(
                        cmf,
                        mass,
                        marker='^',
                        color='blue',
                        s=120,
                        edgecolors='black',
                        linewidths=0.5,
                        zorder=5,
                    )

        ax.set_xlabel('Core Mass Fraction')
        ax.set_ylabel('Planet Mass [M$_\\oplus$]')
        ax.set_title(f'({chr(97 + ax_idx)}) {title}', fontweight='bold')
        ax.set_yscale('log')
        ax.set_yticks(masses_all)
        ax.set_yticklabels([str(m) for m in masses_all])
        ax.set_xlim(0.05, 0.55)

    # Common legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker='o',
            color='green',
            markersize=10,
            linestyle='none',
            label='Solidified ($\\Phi < 0.01$)',
        ),
        Line2D(
            [0],
            [0],
            marker='D',
            color='orange',
            markersize=10,
            linestyle='none',
            label='Persistent melt ($t > 0.5$ Myr)',
        ),
        Line2D(
            [0],
            [0],
            marker='^',
            color='blue',
            markersize=10,
            linestyle='none',
            label='Still running',
        ),
        Line2D(
            [0],
            [0],
            marker='s',
            color='red',
            markersize=10,
            linestyle='none',
            label='Stuck (EOS stiffness)',
        ),
        Line2D(
            [0],
            [0],
            marker='x',
            color='black',
            markersize=10,
            linestyle='none',
            label='Failed / not in grid',
        ),
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=5,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUTDIR / 'fig4_stability_map.png')
    plt.close(fig)
    print('Fig 4: Stability map')


# ══════════════════════════════════════════════════════════════════
# FIGURE 5: Structural parameters — R, M, g comparison
# ══════════════════════════════════════════════════════════════════


def plot_structural_comparison():
    """Compare R_int, M_int, gravity between AW and ZAL at initial timestep."""
    masses = [0.5, 1.0, 2.0, 3.0, 5.0]
    cmfs = [0.10, 0.20, 0.325, 0.50]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Collect data
    data = {
        'mass': [],
        'cmf': [],
        'R_aw': [],
        'R_zal': [],
        'M_aw': [],
        'M_zal': [],
        'g_aw': [],
        'g_zal': [],
        'Mc_aw': [],
        'Mc_zal': [],
    }

    for mass in masses:
        for cmf in cmfs:
            aw_name = f'A_M{mass}_CMF{cmf}_AW'
            zal_name = f'A_M{mass}_CMF{cmf}_ZAL'
            if aw_name in cases and zal_name in cases:
                aw_r0 = cases[aw_name]['rows'][0]
                zal_r0 = cases[zal_name]['rows'][0]
                data['mass'].append(mass)
                data['cmf'].append(cmf)
                data['R_aw'].append(aw_r0['R_int'] / R_EARTH)
                data['R_zal'].append(zal_r0['R_int'] / R_EARTH)
                data['M_aw'].append(aw_r0['M_int'] / M_EARTH)
                data['M_zal'].append(zal_r0['M_int'] / M_EARTH)
                data['g_aw'].append(aw_r0['gravity'])
                data['g_zal'].append(zal_r0['gravity'])
                data['Mc_aw'].append(aw_r0['M_core'] / M_EARTH)
                data['Mc_zal'].append(zal_r0['M_core'] / M_EARTH)

    # (a) Radius parity
    ax = axes[0, 0]
    for i in range(len(data['mass'])):
        ax.scatter(
            data['R_aw'][i],
            data['R_zal'][i],
            color=CMF_COLORS[data['cmf'][i]],
            marker=MASS_MARKERS[data['mass'][i]],
            s=100,
            edgecolors='black',
            linewidths=0.5,
        )
    lims = [0.5, 2.5]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('AW $R_\\mathrm{int}$ [$R_\\oplus$]')
    ax.set_ylabel('ZAL $R_\\mathrm{int}$ [$R_\\oplus$]')
    ax.set_title('(a) Planet Radius', fontweight='bold')

    # (b) Interior mass parity
    ax = axes[0, 1]
    for i in range(len(data['mass'])):
        ax.scatter(
            data['M_aw'][i],
            data['M_zal'][i],
            color=CMF_COLORS[data['cmf'][i]],
            marker=MASS_MARKERS[data['mass'][i]],
            s=100,
            edgecolors='black',
            linewidths=0.5,
        )
    lims = [0, 6]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('AW $M_\\mathrm{int}$ [$M_\\oplus$]')
    ax.set_ylabel('ZAL $M_\\mathrm{int}$ [$M_\\oplus$]')
    ax.set_title('(b) Interior Mass', fontweight='bold')

    # (c) Surface gravity parity
    ax = axes[1, 0]
    for i in range(len(data['mass'])):
        ax.scatter(
            data['g_aw'][i],
            data['g_zal'][i],
            color=CMF_COLORS[data['cmf'][i]],
            marker=MASS_MARKERS[data['mass'][i]],
            s=100,
            edgecolors='black',
            linewidths=0.5,
        )
    lims = [0, 30]
    ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('AW $g_\\mathrm{surf}$ [m/s$^2$]')
    ax.set_ylabel('ZAL $g_\\mathrm{surf}$ [m/s$^2$]')
    ax.set_title('(c) Surface Gravity', fontweight='bold')

    # (d) Relative differences
    ax = axes[1, 1]
    rel_R = [
        (data['R_zal'][i] - data['R_aw'][i]) / data['R_aw'][i] * 100
        for i in range(len(data['mass']))
    ]
    rel_M = [
        (data['M_zal'][i] - data['M_aw'][i]) / data['M_aw'][i] * 100
        for i in range(len(data['mass']))
    ]
    rel_g = [
        (data['g_zal'][i] - data['g_aw'][i]) / data['g_aw'][i] * 100
        for i in range(len(data['mass']))
    ]

    x = range(len(data['mass']))
    labels = [f'{data["mass"][i]}\n{data["cmf"][i]}' for i in x]
    width = 0.25
    ax.bar([xi - width for xi in x], rel_R, width, color='#66c2a5', label='$\\Delta R/R$')
    ax.bar(list(x), rel_M, width, color='#fc8d62', label='$\\Delta M/M$')
    ax.bar([xi + width for xi in x], rel_g, width, color='#8da0cb', label='$\\Delta g/g$')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Mass / CMF')
    ax.set_ylabel('Relative difference (ZAL−AW)/AW [%]')
    ax.set_title('(d) Relative Structural Differences', fontweight='bold')
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig5_structural_comparison.png')
    plt.close(fig)
    print('Fig 5: Structural comparison')


# ══════════════════════════════════════════════════════════════════
# FIGURE 6: Energy balance — F_int, F_atm, F_net
# ══════════════════════════════════════════════════════════════════


def plot_energy_balance():
    """Interior heat flux evolution for selected mass-CMF pairs."""
    pairs = [(1.0, 0.325), (2.0, 0.325), (3.0, 0.325), (5.0, 0.325)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (mass, cmf) in enumerate(pairs):
        ax = axes[idx // 2, idx % 2]
        aw_name = f'A_M{mass}_CMF{cmf}_AW'
        zal_name = f'A_M{mass}_CMF{cmf}_ZAL'

        for name, color, ls in [(aw_name, AW_COLOR, '-'), (zal_name, ZAL_COLOR, '-')]:
            if name in cases:
                c = cases[name]
                t = [r['Time'] for r in c['rows'] if r['Time'] > 0]
                fint = [r['F_int'] for r in c['rows'] if r['Time'] > 0]
                fatm = [r['F_atm'] for r in c['rows'] if r['Time'] > 0]
                if t:
                    mode = 'AW' if 'AW' in name else 'ZAL'
                    ax.plot(
                        t,
                        fint,
                        color=color,
                        linewidth=1.5,
                        linestyle='-',
                        label=f'{mode} $F_\\mathrm{{int}}$',
                    )
                    ax.plot(
                        t,
                        fatm,
                        color=color,
                        linewidth=1.0,
                        linestyle='--',
                        label=f'{mode} $F_\\mathrm{{atm}}$',
                        alpha=0.7,
                    )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel('Heat flux [W/m$^2$]')
        ax.set_title(f'M={mass} M$_\\oplus$, CMF={cmf}', fontweight='bold')
        ax.set_ylim(1e-1, 1e7)
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig6_energy_balance.png')
    plt.close(fig)
    print('Fig 6: Energy balance')


# ══════════════════════════════════════════════════════════════════
# FIGURE 7: Resolution convergence (Block E)
# ══════════════════════════════════════════════════════════════════


def plot_resolution_convergence():
    """Compare n=60 (default), n=90, n=120 for 3 M_E CMF=0.325 ZAL."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ref_name = 'A_M3.0_CMF0.325_ZAL'  # n=60 default
    e90_name = 'E_M3.0_CMF0.325_ZAL_n90'
    e120_name = 'E_M3.0_CMF0.325_ZAL_n120'

    colors = {ref_name: '#1b9e77', e90_name: '#d95f02', e120_name: '#7570b3'}
    labels = {ref_name: 'n=60 (default)', e90_name: 'n=90', e120_name: 'n=120'}

    # (a) T_magma
    ax = axes[0]
    for name in [ref_name, e90_name, e120_name]:
        if name in cases:
            c = cases[name]
            t = [r['Time'] for r in c['rows']]
            T = [r['T_magma'] for r in c['rows']]
            ax.plot(t, T, color=colors[name], linewidth=1.5, label=labels[name])
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('$T_\\mathrm{magma}$ [K]')
    ax.set_title('(a) Temperature', fontweight='bold')
    ax.legend()

    # (b) Phi
    ax = axes[1]
    for name in [ref_name, e90_name, e120_name]:
        if name in cases:
            c = cases[name]
            t = [r['Time'] for r in c['rows']]
            phi = [r['Phi_global'] for r in c['rows']]
            ax.plot(t, phi, color=colors[name], linewidth=1.5, label=labels[name])
    ax.axhline(0.01, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('$\\Phi_\\mathrm{global}$')
    ax.set_title('(b) Melt Fraction', fontweight='bold')
    ax.legend()

    # (c) Summary table
    ax = axes[2]
    ax.axis('off')
    table_data = []
    for name in [ref_name, e90_name, e120_name]:
        if name in cases:
            c = cases[name]
            table_data.append(
                [
                    labels[name],
                    f'{c["steps"]}',
                    f'{c["t_final"]:.0f}',
                    f'{c["T_magma_final"]:.1f}',
                    f'{c["Phi_final"]:.4f}',
                ]
            )
    table = ax.table(
        cellText=table_data,
        colLabels=['Resolution', 'Steps', 't_final [yr]', 'T_magma [K]', 'Phi_final'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('(c) Convergence Summary', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig7_resolution_convergence.png')
    plt.close(fig)
    print('Fig 7: Resolution convergence')


# ══════════════════════════════════════════════════════════════════
# FIGURE 8: Initial entropy sensitivity (Block F)
# ══════════════════════════════════════════════════════════════════


def plot_entropy_sensitivity():
    """Block F: Effect of initial entropy on solidification."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    cmfs = [0.10, 0.325]
    entropies = [2600, 2800, 3000, 3200]
    ent_colors = {2600: '#1b9e77', 2800: '#d95f02', 3000: '#7570b3', 3200: '#e7298a'}

    for ci, cmf in enumerate(cmfs):
        # T_magma
        ax = axes[0, ci]
        for S in entropies:
            if S == 3000:
                name = f'A_M3.0_CMF{cmf}_ZAL'
            else:
                name = f'F_M3.0_CMF{cmf}_ZAL_S{S}'
            if name in cases:
                c = cases[name]
                t = [r['Time'] for r in c['rows']]
                T = [r['T_magma'] for r in c['rows']]
                ax.plot(t, T, color=ent_colors[S], linewidth=1.5, label=f'$S_0$={S}')
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel('$T_\\mathrm{magma}$ [K]')
        ax.set_title(f'(a{ci + 1}) CMF={cmf} — Temperature', fontweight='bold')
        ax.legend(fontsize=8)

        # Phi
        ax = axes[1, ci]
        for S in entropies:
            if S == 3000:
                name = f'A_M3.0_CMF{cmf}_ZAL'
            else:
                name = f'F_M3.0_CMF{cmf}_ZAL_S{S}'
            if name in cases:
                c = cases[name]
                t = [r['Time'] for r in c['rows']]
                phi = [r['Phi_global'] for r in c['rows']]
                ax.plot(t, phi, color=ent_colors[S], linewidth=1.5, label=f'$S_0$={S}')
        ax.axhline(0.01, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel('$\\Phi_\\mathrm{global}$')
        ax.set_title(f'(b{ci + 1}) CMF={cmf} — Melt Fraction', fontweight='bold')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig8_entropy_sensitivity.png')
    plt.close(fig)
    print('Fig 8: Entropy sensitivity')


# ══════════════════════════════════════════════════════════════════
# FIGURE 9: High-mass frontier (Block B + C)
# ══════════════════════════════════════════════════════════════════


def plot_high_mass_frontier():
    """Block B (AW 7-10 M_E) and Block C (ZAL 7 M_E) comparison."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        'High-Mass Frontier: AW (top, 7–10 M$_\\oplus$) vs ZAL (bottom, 7 M$_\\oplus$)',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    cmfs = [0.10, 0.20, 0.325, 0.50]

    # Top row: Block B (AW at 7 and 10 M_E)
    for j, cmf in enumerate(cmfs):
        ax = axes[0, j]
        for mass, ls in [(7.0, '-'), (10.0, '--')]:
            name = f'B_M{mass}_CMF{cmf}_AW'
            if name in cases:
                c = cases[name]
                t = [r['Time'] for r in c['rows']]
                T = [r['T_magma'] for r in c['rows']]
                if t and max(t) > 0:
                    ax.plot(
                        t,
                        T,
                        color=AW_COLOR,
                        linewidth=1.5,
                        linestyle=ls,
                        label=f'AW {mass} M$_\\oplus$',
                    )
                elif t:
                    ax.text(
                        0.5,
                        0.5,
                        f'AW {mass}\nSTUCK',
                        transform=ax.transAxes,
                        ha='center',
                        va='center',
                        fontsize=9,
                        color='red',
                    )
        ax.set_title(f'CMF={cmf}', fontweight='bold')
        ax.set_ylabel('$T_\\mathrm{magma}$ [K]')
        ax.set_ylim(0, 4000)
        ax.legend(fontsize=8)

    # Bottom row: Block C (ZAL at 7 M_E only, since 10+ failed)
    for j, cmf in enumerate(cmfs):
        ax = axes[1, j]
        name = f'C_M7.0_CMF{cmf}_ZAL'
        if name in cases:
            c = cases[name]
            t = [r['Time'] for r in c['rows']]
            T = [r['T_magma'] for r in c['rows']]
            phi = [r['Phi_global'] for r in c['rows']]
            if t and max(t) > 0:
                ax.plot(t, T, color=ZAL_COLOR, linewidth=1.5, label='ZAL 7 M$_\\oplus$')
                ax2 = ax.twinx()
                ax2.plot(t, phi, color=ZAL_COLOR, linewidth=1.0, linestyle='--', alpha=0.5)
                ax2.set_ylabel('$\\Phi$', color=ZAL_COLOR, alpha=0.5)
                ax2.set_ylim(-0.02, 1.05)
        ax.set_xlabel('Time [yr]')
        ax.set_ylabel('$T_\\mathrm{magma}$ [K]')
        ax.set_ylim(0, 4000)
        ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / 'fig9_high_mass_frontier.png')
    plt.close(fig)
    print('Fig 9: High-mass frontier')


# ══════════════════════════════════════════════════════════════════
# FIGURE 10: Phase 2 feedback (Block D)
# ══════════════════════════════════════════════════════════════════


def plot_phase2_feedback():
    """Block D: Iterative Zalmoxis-SPIDER coupling vs Phase 1 only."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    masses = [1.0, 3.0, 5.0]
    for idx, mass in enumerate(masses):
        ax = axes[idx]
        p1_name = f'A_M{mass}_CMF0.325_ZAL'
        p2_name = f'D_M{mass}_CMF0.325_ZAL_P2u100'

        for name, color, label in [
            (p1_name, '#1b9e77', 'Phase 1 (static mesh)'),
            (p2_name, '#d95f02', 'Phase 2 (update=100yr)'),
        ]:
            if name in cases:
                c = cases[name]
                t = [r['Time'] for r in c['rows']]
                T = [r['T_magma'] for r in c['rows']]
                ax.plot(t, T, color=color, linewidth=1.5, label=label)

        ax.set_xlabel('Time [yr]')
        ax.set_ylabel('$T_\\mathrm{magma}$ [K]')
        ax.set_title(f'M={mass} M$_\\oplus$, CMF=0.325', fontweight='bold')
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig10_phase2_feedback.png')
    plt.close(fig)
    print('Fig 10: Phase 2 feedback')


# ══════════════════════════════════════════════════════════════════
# FIGURE 11: Persistent melt fraction deep-dive
# ══════════════════════════════════════════════════════════════════


def plot_persistent_melt():
    """Cases with T_magma < 700 K but Phi > 0.1 — what's going on?"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Select cases with persistent melt
    persist_cases = []
    for name, c in cases.items():
        if c['steps'] > 30 and c['T_magma_final'] < 800 and c['Phi_final'] > 0.05:
            persist_cases.append((name, c))
    persist_cases.sort(key=lambda x: x[1]['mass'])

    # (a) T_magma vs time
    ax = axes[0, 0]
    for name, c in persist_cases[:8]:
        t = [r['Time'] for r in c['rows'] if r['Time'] > 0]
        T = [r['T_magma'] for r in c['rows'] if r['Time'] > 0]
        if t:
            ax.plot(t, T, linewidth=1.2, label=f'{c["mass"]}M {c["mode"]} CMF{c["cmf"]}')
    ax.set_xscale('log')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('$T_\\mathrm{magma}$ [K]')
    ax.set_title('(a) Temperature Evolution', fontweight='bold')
    ax.legend(fontsize=6, ncol=2)

    # (b) Phi vs time
    ax = axes[0, 1]
    for name, c in persist_cases[:8]:
        t = [r['Time'] for r in c['rows'] if r['Time'] > 0]
        phi = [r['Phi_global'] for r in c['rows'] if r['Time'] > 0]
        if t:
            ax.plot(t, phi, linewidth=1.2, label=f'{c["mass"]}M {c["mode"]} CMF{c["cmf"]}')
    ax.axhline(0.01, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('$\\Phi_\\mathrm{global}$')
    ax.set_title('(b) Melt Fraction Evolution', fontweight='bold')
    ax.legend(fontsize=6, ncol=2)

    # (c) T_magma vs Phi trajectory
    ax = axes[1, 0]
    for name, c in persist_cases[:8]:
        T = [r['T_magma'] for r in c['rows']]
        phi = [r['Phi_global'] for r in c['rows']]
        ax.plot(T, phi, linewidth=1.2, label=f'{c["mass"]}M {c["mode"]} CMF{c["cmf"]}')
    ax.set_xlabel('$T_\\mathrm{magma}$ [K]')
    ax.set_ylabel('$\\Phi_\\mathrm{global}$')
    ax.set_title('(c) T–$\\Phi$ Phase Trajectory', fontweight='bold')
    ax.legend(fontsize=6, ncol=2)

    # (d) Summary table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = []
    for name, c in persist_cases[:10]:
        table_data.append(
            [
                f'{c["mass"]}',
                f'{c["cmf"]}',
                c['mode'],
                f'{c["steps"]}',
                f'{c["t_final"] / 1e6:.2f}',
                f'{c["T_magma_final"]:.0f}',
                f'{c["Phi_final"]:.3f}',
            ]
        )
    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=['Mass', 'CMF', 'Mode', 'Steps', 't [Myr]', 'T_mag [K]', 'Phi'],
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.4)
    ax.set_title('(d) Persistent Melt Cases', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig11_persistent_melt.png')
    plt.close(fig)
    print('Fig 11: Persistent melt')


# ══════════════════════════════════════════════════════════════════
# FIGURE 12: Comprehensive summary — solidification time vs mass
# ══════════════════════════════════════════════════════════════════


def plot_summary():
    """Solidification time and final T_magma as function of planet mass."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cmfs = [0.10, 0.20, 0.325, 0.50]

    # (a) Solidification time vs mass
    ax = axes[0]
    for cmf in cmfs:
        aw_m, aw_t = [], []
        zal_m, zal_t = [], []
        for mass in [0.5, 1.0, 2.0, 3.0, 5.0]:
            aw_name = f'A_M{mass}_CMF{cmf}_AW'
            zal_name = f'A_M{mass}_CMF{cmf}_ZAL'
            if aw_name in cases and cases[aw_name]['solidified']:
                aw_m.append(mass)
                aw_t.append(cases[aw_name]['t_final'])
            if zal_name in cases and cases[zal_name]['solidified']:
                zal_m.append(mass)
                zal_t.append(cases[zal_name]['t_final'])
        if aw_m:
            ax.plot(
                aw_m, aw_t, 'o--', color=CMF_COLORS[cmf], markersize=8, label=f'AW CMF={cmf}'
            )
        if zal_m:
            ax.plot(
                zal_m,
                zal_t,
                's-',
                color=CMF_COLORS[cmf],
                markersize=8,
                markerfacecolor='white',
                markeredgecolor=CMF_COLORS[cmf],
                markeredgewidth=2,
                label=f'ZAL CMF={cmf}',
            )
    ax.set_xlabel('Planet Mass [M$_\\oplus$]')
    ax.set_ylabel('Solidification Time [yr]')
    ax.set_title('(a) Solidification Timescale', fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.set_yscale('log')

    # (b) Final T_magma vs mass (solidified only)
    ax = axes[1]
    for cmf in cmfs:
        aw_m, aw_T = [], []
        zal_m, zal_T = [], []
        for mass in [0.5, 1.0, 2.0, 3.0, 5.0]:
            aw_name = f'A_M{mass}_CMF{cmf}_AW'
            zal_name = f'A_M{mass}_CMF{cmf}_ZAL'
            if aw_name in cases and cases[aw_name]['solidified']:
                aw_m.append(mass)
                aw_T.append(cases[aw_name]['T_magma_final'])
            if zal_name in cases and cases[zal_name]['solidified']:
                zal_m.append(mass)
                zal_T.append(cases[zal_name]['T_magma_final'])
        if aw_m:
            ax.plot(aw_m, aw_T, 'o--', color=CMF_COLORS[cmf], markersize=8)
        if zal_m:
            ax.plot(
                zal_m,
                zal_T,
                's-',
                color=CMF_COLORS[cmf],
                markersize=8,
                markerfacecolor='white',
                markeredgecolor=CMF_COLORS[cmf],
                markeredgewidth=2,
            )
    ax.set_xlabel('Planet Mass [M$_\\oplus$]')
    ax.set_ylabel('Final $T_\\mathrm{magma}$ [K]')
    ax.set_title('(b) Final Magma Ocean Temperature', fontweight='bold')

    # (c) Number of steps vs mass
    ax = axes[2]
    for cmf in cmfs:
        aw_m, aw_s = [], []
        zal_m, zal_s = [], []
        for mass in [0.5, 1.0, 2.0, 3.0, 5.0]:
            aw_name = f'A_M{mass}_CMF{cmf}_AW'
            zal_name = f'A_M{mass}_CMF{cmf}_ZAL'
            if aw_name in cases and cases[aw_name]['solidified']:
                aw_m.append(mass)
                aw_s.append(cases[aw_name]['steps'])
            if zal_name in cases and cases[zal_name]['solidified']:
                zal_m.append(mass)
                zal_s.append(cases[zal_name]['steps'])
        if aw_m:
            ax.plot(aw_m, aw_s, 'o--', color=CMF_COLORS[cmf], markersize=8)
        if zal_m:
            ax.plot(
                zal_m,
                zal_s,
                's-',
                color=CMF_COLORS[cmf],
                markersize=8,
                markerfacecolor='white',
                markeredgecolor=CMF_COLORS[cmf],
                markeredgewidth=2,
            )
    ax.set_xlabel('Planet Mass [M$_\\oplus$]')
    ax.set_ylabel('Number of Steps')
    ax.set_title('(c) Computational Cost', fontweight='bold')

    plt.tight_layout()
    fig.savefig(OUTDIR / 'fig12_summary.png')
    plt.close(fig)
    print('Fig 12: Summary')


# ══════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f'Loaded {len(cases)} cases')
    print(f'Solidified: {sum(1 for c in cases.values() if c.get("solidified", False))}')
    print(f'Output: {OUTDIR}\n')

    plot_thermal_evolution_comparison()
    plot_melt_fraction_evolution()
    plot_solidification_times()
    plot_stability_map()
    plot_structural_comparison()
    plot_energy_balance()
    plot_resolution_convergence()
    plot_entropy_sensitivity()
    plot_high_mass_frontier()
    plot_phase2_feedback()
    plot_persistent_melt()
    plot_summary()

    print(f'\nAll plots saved to {OUTDIR}/')
