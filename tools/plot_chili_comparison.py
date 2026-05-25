#!/usr/bin/env python3
"""Plot PROTEUS output overlaid on CHILI intercomparison data.

Downloads comparison data from the CHILI GitHub repository and creates
multi-model comparison plots mirroring the figures from the CHILI
intercomparison papers.

Usage:
    python tools/plot_chili_comparison.py \\
        --proteus-earth output/tutorial_earth/ \\
        --proteus-venus output/tutorial_venus/ \\
        --chili-repo /tmp/chili \\
        --output output_files/chili_plots/

If --chili-repo is not specified, the script will clone the repo to /tmp/chili.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

CHILI_REPO_URL = 'https://github.com/projectcuisines/chili.git'

MODELS = {
    'gooey': {'color': '#e6194b', 'label': 'GOOEY'},
    'neongooey': {'color': '#f58231', 'label': 'NEONGOOEY'},
    'pacman': {'color': '#3cb44b', 'label': 'PACMAN'},
    'lincs': {'color': '#4363d8', 'label': 'LINCS'},
    'moai': {'color': '#911eb4', 'label': 'MOAI'},
    'planatmo': {'color': '#42d4f4', 'label': 'PlanAtMO'},
}

PROTEUS_STYLE = {'color': '#000000', 'linewidth': 2.5, 'label': 'PROTEUS (this run)'}


def _load_chili_csv(path: Path) -> pd.DataFrame | None:
    """Load a CHILI intercomparison CSV, handling missing files."""
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, comment='#')
    except Exception:
        return None


def _load_proteus_helpfile(output_dir: Path) -> pd.DataFrame | None:
    """Load PROTEUS runtime_helpfile.csv."""
    hf = output_dir / 'runtime_helpfile.csv'
    if not hf.is_file():
        return None
    with open(hf) as f:
        header = f.readline().strip().split()
    df = pd.read_csv(hf, sep=r'\s+', header=None, skiprows=1)
    df.columns = header
    return df[df['Time'] > 0]


def ensure_chili_repo(chili_path: Path) -> Path:
    """Clone the CHILI repo if not already present."""
    if chili_path.is_dir() and (chili_path / 'intercomparison').is_dir():
        return chili_path
    print(f'Cloning CHILI repository to {chili_path}...')
    subprocess.run(
        ['git', 'clone', '--depth', '1', CHILI_REPO_URL, str(chili_path)],
        check=True,
    )
    return chili_path


def plot_melt_fraction_evolution(
    chili_path: Path,
    proteus_earth: pd.DataFrame | None,
    proteus_venus: pd.DataFrame | None,
    output_dir: Path,
):
    """CHILI Fig 1: melt fraction vs time for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))

    intercomp = chili_path / 'intercomparison' / 'outputs'

    for model_key, style in MODELS.items():
        for planet, ls in [('earth', '-'), ('venus', '--')]:
            csv = intercomp / model_key / f'evolution-{model_key}-{planet}-data.csv'
            df = _load_chili_csv(csv)
            if df is None:
                continue
            t_col = [
                c
                for c in df.columns
                if 't' in c.lower() and 'yr' in c.lower() or c.lower() in ('t', 'time', 't_yr')
            ]
            phi_col = [c for c in df.columns if 'phi' in c.lower() or 'melt' in c.lower()]
            if not t_col or not phi_col:
                continue
            t = df[t_col[0]] / 1e6
            phi = df[phi_col[0]]
            if phi.max() > 1:
                phi = phi / 100
            label = style['label'] if planet == 'earth' else None
            ax.plot(t, phi, ls, color=style['color'], alpha=0.7, linewidth=1.2, label=label)

    if proteus_earth is not None:
        t = proteus_earth['Time'] / 1e6
        ax.plot(
            t,
            proteus_earth['Phi_global'],
            '-',
            color=PROTEUS_STYLE['color'],
            linewidth=PROTEUS_STYLE['linewidth'],
            label='PROTEUS Earth (this run)',
        )

    if proteus_venus is not None:
        t = proteus_venus['Time'] / 1e6
        ax.plot(
            t,
            proteus_venus['Phi_global'],
            '--',
            color=PROTEUS_STYLE['color'],
            linewidth=PROTEUS_STYLE['linewidth'],
            label='PROTEUS Venus (this run)',
        )

    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Melt fraction')
    ax.set_xscale('log')
    ax.set_xlim(1e-3, 1e3)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.set_title('Cooling and solidification: Earth (solid) and Venus (dashed)')

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'chili_fig1_melt_fraction.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'chili_fig1_melt_fraction.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_dir / "chili_fig1_melt_fraction.pdf"}')


def plot_olr_vs_phi(
    chili_path: Path,
    proteus_earth: pd.DataFrame | None,
    output_dir: Path,
):
    """CHILI Fig 7a: OLR vs melt fraction for Earth."""
    fig, ax = plt.subplots(figsize=(6, 5))

    intercomp = chili_path / 'intercomparison' / 'outputs'

    for model_key, style in MODELS.items():
        csv = intercomp / model_key / f'evolution-{model_key}-earth-data.csv'
        df = _load_chili_csv(csv)
        if df is None:
            continue
        phi_col = [c for c in df.columns if 'phi' in c.lower() or 'melt' in c.lower()]
        olr_col = [c for c in df.columns if 'olr' in c.lower() or 'f_olr' in c.lower()]
        if not phi_col or not olr_col:
            continue
        phi = df[phi_col[0]]
        if phi.max() > 1:
            phi = phi / 100
        ax.plot(
            phi * 100,
            df[olr_col[0]],
            '-',
            color=style['color'],
            alpha=0.7,
            linewidth=1.2,
            label=style['label'],
        )

    if proteus_earth is not None:
        ax.plot(
            proteus_earth['Phi_global'] * 100,
            proteus_earth['F_olr'],
            '-',
            color=PROTEUS_STYLE['color'],
            linewidth=PROTEUS_STYLE['linewidth'],
            label='PROTEUS (this run)',
        )

    ax.set_xlabel('Melt fraction [vol%]')
    ax.set_ylabel('Outgoing LW radiation [W m$^{-2}$]')
    ax.set_yscale('log')
    ax.set_xlim(100, 0)
    ax.legend(fontsize=8)
    ax.set_title('Outgoing longwave radiation: Nominal Earth')

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'chili_fig7a_olr_vs_phi.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'chili_fig7a_olr_vs_phi.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_dir / "chili_fig7a_olr_vs_phi.pdf"}')


def plot_tsurf_vs_phi(
    chili_path: Path,
    proteus_earth: pd.DataFrame | None,
    output_dir: Path,
):
    """CHILI Fig 8a: T_surf vs melt fraction for Earth."""
    fig, ax = plt.subplots(figsize=(6, 5))

    intercomp = chili_path / 'intercomparison' / 'outputs'

    for model_key, style in MODELS.items():
        csv = intercomp / model_key / f'evolution-{model_key}-earth-data.csv'
        df = _load_chili_csv(csv)
        if df is None:
            continue
        phi_col = [c for c in df.columns if 'phi' in c.lower() or 'melt' in c.lower()]
        ts_col = [
            c
            for c in df.columns
            if 'surf' in c.lower() and 'temp' in c.lower() or c.lower() in ('t_surf', 'tsurf')
        ]
        if not phi_col or not ts_col:
            continue
        phi = df[phi_col[0]]
        if phi.max() > 1:
            phi = phi / 100
        ax.plot(
            phi * 100,
            df[ts_col[0]],
            '-',
            color=style['color'],
            alpha=0.7,
            linewidth=1.2,
            label=style['label'],
        )

    if proteus_earth is not None:
        ax.plot(
            proteus_earth['Phi_global'] * 100,
            proteus_earth['T_surf'],
            '-',
            color=PROTEUS_STYLE['color'],
            linewidth=PROTEUS_STYLE['linewidth'],
            label='PROTEUS (this run)',
        )

    ax.set_xlabel('Melt fraction [vol%]')
    ax.set_ylabel('Surface temperature [K]')
    ax.set_xlim(100, 0)
    ax.legend(fontsize=8)
    ax.set_title('Surface temperature vs melt fraction: Nominal Earth')

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'chili_fig8a_tsurf_vs_phi.pdf', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'chili_fig8a_tsurf_vs_phi.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output_dir / "chili_fig8a_tsurf_vs_phi.pdf"}')


def main():
    parser = argparse.ArgumentParser(description='Plot PROTEUS vs CHILI comparison')
    parser.add_argument(
        '--proteus-earth',
        type=Path,
        default=None,
        help='PROTEUS output directory for Earth case',
    )
    parser.add_argument(
        '--proteus-venus',
        type=Path,
        default=None,
        help='PROTEUS output directory for Venus case',
    )
    parser.add_argument(
        '--chili-repo',
        type=Path,
        default=Path('/tmp/chili'),
        help='Path to CHILI repository (cloned if missing)',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('output_files/chili_plots'),
        help='Output directory for plots',
    )
    args = parser.parse_args()

    chili = ensure_chili_repo(args.chili_repo)

    earth_df = _load_proteus_helpfile(args.proteus_earth) if args.proteus_earth else None
    venus_df = _load_proteus_helpfile(args.proteus_venus) if args.proteus_venus else None

    if earth_df is not None:
        print(
            f'Loaded PROTEUS Earth: {len(earth_df)} rows, '
            f'T={earth_df["T_magma"].max():.0f}->{earth_df["T_magma"].min():.0f} K'
        )
    if venus_df is not None:
        print(
            f'Loaded PROTEUS Venus: {len(venus_df)} rows, '
            f'T={venus_df["T_magma"].max():.0f}->{venus_df["T_magma"].min():.0f} K'
        )

    plot_melt_fraction_evolution(chili, earth_df, venus_df, args.output)
    plot_olr_vs_phi(chili, earth_df, args.output)
    plot_tsurf_vs_phi(chili, earth_df, args.output)

    print(f'\nAll plots saved to {args.output}/')


if __name__ == '__main__':
    main()
