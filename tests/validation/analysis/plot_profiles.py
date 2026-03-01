#!/usr/bin/env python3
"""Generate interior and atmosphere T-P profile figures for the validation analysis."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path(__file__).parent / 'plots'
OUTDIR.mkdir(exist_ok=True)

with open(Path(__file__).parent / 'profiles.json') as f:
    profiles_raw = json.load(f)

# Data already has scaling applied (physical units)
profiles = profiles_raw

with open(Path(__file__).parent / 'atm_profiles.json') as f:
    atm_profiles = json.load(f)

R_EARTH = 6.371e6


# Color map for time evolution
def time_colors(n):
    """Return n colors from cool (early) to warm (late)."""
    return [cm.plasma(i / max(n - 1, 1)) for i in range(n)]


# ══════════════════════════════════════════════════════════════════
# FIGURE 13: Interior T-P profiles — AW vs ZAL at 1 and 3 M_E
# ══════════════════════════════════════════════════════════════════


def plot_interior_TP_comparison():
    """Interior T(P) profiles at multiple timesteps, AW vs ZAL."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        'Interior Temperature–Pressure Profiles: Evolution Over Time',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    panels = [
        ('A_M1.0_CMF0.325_AW', 'A_M1.0_CMF0.325_ZAL', '1.0 M$_\\oplus$, CMF=0.325'),
        ('A_M3.0_CMF0.325_AW', 'A_M3.0_CMF0.325_ZAL', '3.0 M$_\\oplus$, CMF=0.325'),
        (
            'A_M5.0_CMF0.5_AW',
            'A_M5.0_CMF0.325_ZAL',
            '5.0 M$_\\oplus$ (AW CMF=0.5 vs ZAL CMF=0.325)',
        ),
        (
            'A_M3.0_CMF0.1_ZAL',
            'A_M5.0_CMF0.1_ZAL',
            'CMF=0.1: 3.0 vs 5.0 M$_\\oplus$ (ZAL only)',
        ),
    ]

    for idx, (case_a, case_b, title) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]

        for case_name, ls, mode_label in [(case_a, '-', ''), (case_b, '--', '')]:
            if case_name not in profiles:
                continue
            snaps = profiles[case_name]
            colors = time_colors(len(snaps))

            for si, snap in enumerate(snaps):
                P = np.array(snap['pressure_b']) / 1e9  # GPa
                T = np.array(snap['temp_b'])
                t_yr = snap['time_yr']

                if t_yr < 1:
                    label = f'{case_name.split("_")[3]} t=0'
                elif t_yr < 1000:
                    label = f'{case_name.split("_")[3]} t={t_yr:.0f} yr'
                elif t_yr < 1e6:
                    label = f'{case_name.split("_")[3]} t={t_yr / 1e3:.1f} kyr'
                else:
                    label = f'{case_name.split("_")[3]} t={t_yr / 1e6:.2f} Myr'

                # Only label first and last
                if si == 0 or si == len(snaps) - 1:
                    ax.plot(P, T, color=colors[si], linewidth=1.5, linestyle=ls, label=label)
                else:
                    ax.plot(P, T, color=colors[si], linewidth=0.8, linestyle=ls, alpha=0.6)

        ax.set_xlabel('Pressure [GPa]')
        ax.set_ylabel('Temperature [K]')
        ax.set_title(f'({chr(97 + idx)}) {title}', fontweight='bold')
        ax.legend(fontsize=7, loc='lower right')
        ax.set_ylim(0, 5000)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / 'fig13_interior_TP_profiles.png', bbox_inches='tight')
    plt.close(fig)
    print('Fig 13: Interior T-P profiles')


# ══════════════════════════════════════════════════════════════════
# FIGURE 14: Melt fraction profiles φ(r) — evolution
# ══════════════════════════════════════════════════════════════════


def plot_melt_profiles():
    """Melt fraction φ(r) radial profiles at multiple timesteps."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Melt Fraction Radial Profiles: Evolution Over Time',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    case_list = [
        'A_M1.0_CMF0.325_ZAL',
        'A_M3.0_CMF0.325_ZAL',
        'A_M5.0_CMF0.325_ZAL',
        'A_M3.0_CMF0.1_ZAL',
        'A_M5.0_CMF0.1_ZAL',
        'C_M7.0_CMF0.325_ZAL',
    ]

    for idx, case_name in enumerate(case_list):
        ax = axes[idx // 3, idx % 3]
        if case_name not in profiles:
            ax.set_visible(False)
            continue

        snaps = profiles[case_name]
        colors = time_colors(len(snaps))

        for si, snap in enumerate(snaps):
            r = np.array(snap['radius_b']) / R_EARTH
            phi = np.array(snap['phi_b'])
            t_yr = snap['time_yr']

            if t_yr < 1:
                label = 't=0'
            elif t_yr < 1000:
                label = f't={t_yr:.0f} yr'
            elif t_yr < 1e6:
                label = f't={t_yr / 1e3:.1f} kyr'
            else:
                label = f't={t_yr / 1e6:.2f} Myr'

            ax.plot(r, phi, color=colors[si], linewidth=1.5, label=label)

        ax.axhline(0.01, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)
        ax.set_xlabel('Radius [$R_\\oplus$]')
        ax.set_ylabel('Melt fraction $\\phi$')
        parts = case_name.split('_')
        mass = parts[1][1:]
        cmf = parts[2][3:]
        mode = parts[3]
        ax.set_title(f'({chr(97 + idx)}) M={mass}, CMF={cmf}, {mode}', fontweight='bold')
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / 'fig14_melt_profiles.png', bbox_inches='tight')
    plt.close(fig)
    print('Fig 14: Melt fraction profiles')


# ══════════════════════════════════════════════════════════════════
# FIGURE 15: T vs solidus/liquidus — interior phase diagram
# ══════════════════════════════════════════════════════════════════


def plot_interior_phase_diagram():
    """T(P) with solidus and liquidus for selected cases."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Interior Phase Diagram: Temperature vs Solidus/Liquidus',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    case_list = [
        'A_M1.0_CMF0.325_ZAL',
        'A_M3.0_CMF0.325_ZAL',
        'A_M5.0_CMF0.325_ZAL',
        'A_M3.0_CMF0.1_ZAL',
        'A_M5.0_CMF0.1_ZAL',
        'C_M7.0_CMF0.325_ZAL',
    ]

    for idx, case_name in enumerate(case_list):
        ax = axes[idx // 3, idx % 3]
        if case_name not in profiles:
            ax.set_visible(False)
            continue

        snaps = profiles[case_name]

        # Plot solidus/liquidus from first snapshot (pressure-dependent, doesn't change)
        P0 = np.array(snaps[0]['pressure_b']) / 1e9
        T_sol = np.array(snaps[0]['solidus_temp_b'])
        T_liq = np.array(snaps[0]['liquidus_temp_b'])
        ax.fill_betweenx(P0, T_sol, T_liq, alpha=0.15, color='orange', label='Mushy zone')
        ax.plot(T_sol, P0, 'k--', linewidth=1.0, alpha=0.7, label='Solidus')
        ax.plot(T_liq, P0, 'k-', linewidth=1.0, alpha=0.7, label='Liquidus')

        # Plot T(P) at selected times
        colors = time_colors(len(snaps))
        for si, snap in enumerate(snaps):
            P = np.array(snap['pressure_b']) / 1e9
            T = np.array(snap['temp_b'])
            t_yr = snap['time_yr']

            if t_yr < 1:
                label = 't=0'
            elif t_yr < 1000:
                label = f't={t_yr:.0f} yr'
            elif t_yr < 1e6:
                label = f't={t_yr / 1e3:.1f} kyr'
            else:
                label = f't={t_yr / 1e6:.2f} Myr'

            ax.plot(T, P, color=colors[si], linewidth=1.8, label=label)

        parts = case_name.split('_')
        mass = parts[1][1:]
        cmf = parts[2][3:]
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Pressure [GPa]')
        ax.set_title(f'({chr(97 + idx)}) M={mass}, CMF={cmf}', fontweight='bold')
        ax.invert_yaxis()
        ax.legend(fontsize=6, loc='lower left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / 'fig15_interior_phase_diagram.png', bbox_inches='tight')
    plt.close(fig)
    print('Fig 15: Interior phase diagram')


# ══════════════════════════════════════════════════════════════════
# FIGURE 16: Coupled interior + atmosphere T-P profiles
# ══════════════════════════════════════════════════════════════════


def plot_coupled_TP():
    """Full interior + atmosphere T-P profile on a single axis."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Coupled Interior + Atmosphere T–P Structure', fontsize=14, fontweight='bold', y=0.98
    )

    case_list = [
        'A_M1.0_CMF0.325_AW',
        'A_M1.0_CMF0.325_ZAL',
        'A_M3.0_CMF0.325_ZAL',
        'A_M5.0_CMF0.325_ZAL',
        'A_M5.0_CMF0.1_ZAL',
        'C_M7.0_CMF0.325_ZAL',
    ]

    for idx, case_name in enumerate(case_list):
        ax = axes[idx // 3, idx % 3]

        has_int = case_name in profiles
        has_atm = case_name in atm_profiles

        if not has_int and not has_atm:
            ax.set_visible(False)
            continue

        n_snaps = len(profiles.get(case_name, atm_profiles.get(case_name, [])))
        colors = time_colors(n_snaps)

        for si in range(n_snaps):
            t_yr = None
            # Interior
            if has_int and si < len(profiles[case_name]):
                snap = profiles[case_name][si]
                P_int = np.array(snap['pressure_b']) / 1e5  # bar
                T_int = np.array(snap['temp_b'])
                t_yr = snap['time_yr']
                ax.plot(T_int, P_int, color=colors[si], linewidth=1.5, linestyle='-')

            # Atmosphere
            if has_atm and si < len(atm_profiles[case_name]):
                asnap = atm_profiles[case_name][si]
                P_atm = np.array(asnap['p'])  # already in bar
                T_atm = np.array(asnap['tmp'])
                if t_yr is None:
                    t_yr = asnap['time_yr']
                ax.plot(T_atm, P_atm, color=colors[si], linewidth=1.5, linestyle='--')

            # Label
            if t_yr is not None and (si == 0 or si == n_snaps - 1):
                if t_yr < 1:
                    label = 't=0'
                elif t_yr < 1000:
                    label = f't={t_yr:.0f} yr'
                elif t_yr < 1e6:
                    label = f't={t_yr / 1e3:.1f} kyr'
                else:
                    label = f't={t_yr / 1e6:.2f} Myr'
                # Invisible scatter for legend
                ax.scatter([], [], color=colors[si], s=20, label=label)

        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_xlabel('Temperature [K]')
        ax.set_ylabel('Pressure [bar]')
        parts = case_name.split('_')
        mass = parts[1][1:]
        cmf = parts[2][3:]
        mode = parts[3]
        ax.set_title(f'({chr(97 + idx)}) M={mass}, CMF={cmf}, {mode}', fontweight='bold')

        # Legend: solid = interior, dashed = atmosphere
        from matplotlib.lines import Line2D

        custom = [
            Line2D([0], [0], color='gray', linewidth=1.5, linestyle='-', label='Interior'),
            Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', label='Atmosphere'),
        ]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=custom + handles, fontsize=6, loc='lower left')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / 'fig16_coupled_TP.png', bbox_inches='tight')
    plt.close(fig)
    print('Fig 16: Coupled interior + atmosphere T-P')


# ══════════════════════════════════════════════════════════════════
# FIGURE 17: Eddy diffusivity κ_h and thermal expansion α profiles
# ══════════════════════════════════════════════════════════════════


def plot_transport_properties():
    """Radial profiles of κ_h (eddy diffusivity) and α (thermal expansion)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        'Transport Properties: Eddy Diffusivity $\\kappa_h$ and Thermal Expansion $\\alpha$',
        fontsize=14,
        fontweight='bold',
        y=0.98,
    )

    case_list = [
        'A_M1.0_CMF0.325_ZAL',
        'A_M3.0_CMF0.325_ZAL',
        'A_M5.0_CMF0.325_ZAL',
        'A_M3.0_CMF0.1_ZAL',
        'A_M5.0_CMF0.1_ZAL',
        'C_M7.0_CMF0.325_ZAL',
    ]

    for idx, case_name in enumerate(case_list):
        ax = axes[idx // 3, idx % 3]
        if case_name not in profiles:
            ax.set_visible(False)
            continue

        snaps = profiles[case_name]
        # Plot last snapshot's kappah and alpha
        snap = snaps[-1]
        r = np.array(snap['radius_b']) / R_EARTH
        kappah = np.array(snap['kappah_b'])
        alpha = np.array(snap['alpha_b'])

        color1 = '#d95f02'
        color2 = '#1b9e77'

        ax.semilogy(
            r,
            np.clip(kappah, 1e-10, None),
            color=color1,
            linewidth=1.5,
            label='$\\kappa_h$ [m²/s]',
        )
        ax2 = ax.twinx()
        ax2.plot(
            r,
            alpha * 1e5,
            color=color2,
            linewidth=1.5,
            label='$\\alpha \\times 10^5$ [K$^{-1}$]',
        )
        ax2.axhline(0, color=color2, linewidth=0.5, linestyle=':', alpha=0.5)

        parts = case_name.split('_')
        mass = parts[1][1:]
        cmf = parts[2][3:]
        t_yr = snap['time_yr']
        if t_yr < 1e6:
            tstr = f't={t_yr / 1e3:.1f} kyr'
        else:
            tstr = f't={t_yr / 1e6:.2f} Myr'

        ax.set_xlabel('Radius [$R_\\oplus$]')
        ax.set_ylabel('$\\kappa_h$ [m²/s]', color=color1)
        ax2.set_ylabel('$\\alpha \\times 10^5$ [K$^{-1}$]', color=color2)
        ax.set_title(f'({chr(97 + idx)}) M={mass}, CMF={cmf} @ {tstr}', fontweight='bold')
        ax.tick_params(axis='y', labelcolor=color1)
        ax2.tick_params(axis='y', labelcolor=color2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTDIR / 'fig17_transport_properties.png', bbox_inches='tight')
    plt.close(fig)
    print('Fig 17: Transport properties')


if __name__ == '__main__':
    plot_interior_TP_comparison()
    plot_melt_profiles()
    plot_interior_phase_diagram()
    plot_coupled_TP()
    plot_transport_properties()
    print('\nAll profile plots saved.')
