#!/usr/bin/env python3
"""Plot tutorial output for documentation and user verification.

Usage:
    python tools/plot_tutorial.py output/tutorial_earth/
    python tools/plot_tutorial.py output/dummy/ --save docs/assets/tutorials/dummy_global.avif

Reads runtime_helpfile.csv from the given output directory and produces
a multi-panel overview plot showing the key quantities: surface
temperature, melt fraction, surface pressure, and atmospheric fluxes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_helpfile(output_dir: Path) -> pd.DataFrame:
    """Load runtime_helpfile.csv from a PROTEUS output directory."""
    hf_path = output_dir / 'runtime_helpfile.csv'
    if not hf_path.is_file():
        raise FileNotFoundError(f'No runtime_helpfile.csv in {output_dir}')
    return pd.read_csv(hf_path, sep=r'\s+', header=None, skiprows=1)


def read_header(output_dir: Path) -> list[str]:
    """Read column names from the helpfile header."""
    hf_path = output_dir / 'runtime_helpfile.csv'
    with open(hf_path) as f:
        header = f.readline().strip().split()
    return header


def plot_overview(output_dir: Path, save_path: Path | None = None):
    """Create a 2x2 overview plot of the tutorial output."""
    header = read_header(output_dir)
    df = load_helpfile(output_dir)
    df.columns = header

    # Filter to science stage (Time > 0)
    df = df[df['Time'] > 0].copy()
    if len(df) == 0:
        print('No science-stage data (Time > 0). Check the run completed.')
        return

    time_myr = df['Time'] / 1e6

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel (a): Surface temperature
    ax = axes[0, 0]
    ax.plot(time_myr, df['T_magma'], 'C0-', lw=1.5)
    ax.set_ylabel('Surface temperature [K]')
    ax.set_xlabel('Time [Myr]')
    ax.set_xscale('log')
    ax.set_title('(a) Magma ocean cooling')

    # Panel (b): Melt fraction
    ax = axes[0, 1]
    ax.plot(time_myr, df['Phi_global'], 'C1-', lw=1.5)
    ax.set_ylabel('Global melt fraction')
    ax.set_xlabel('Time [Myr]')
    ax.set_xscale('log')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('(b) Solidification')

    # Panel (c): Surface pressure
    ax = axes[1, 0]
    ax.plot(time_myr, df['P_surf'], 'C2-', lw=1.5)
    ax.set_ylabel('Surface pressure [bar]')
    ax.set_xlabel('Time [Myr]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('(c) Atmospheric pressure')

    # Panel (d): Energy fluxes
    ax = axes[1, 1]
    ax.plot(time_myr, df['F_atm'], 'C3-', lw=1.5, label='$F_{atm}$')
    ax.plot(time_myr, df['F_int'], 'C4--', lw=1.5, label='$F_{int}$')
    if 'F_ins' in df.columns:
        ax.plot(time_myr, df['F_ins'], 'C5:', lw=1.0, label='$F_{ins}$')
    ax.set_ylabel('Flux [W m$^{-2}$]')
    ax.set_xlabel('Time [Myr]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.set_title('(d) Energy fluxes')

    fig.suptitle(output_dir.name, fontsize=12, fontweight='bold')
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        print(f'Saved: {save_path}')
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot PROTEUS tutorial output')
    parser.add_argument('output_dir', type=Path, help='Path to PROTEUS output directory')
    parser.add_argument('--save', type=Path, default=None,
                        help='Save plot to file (PNG, PDF, or AVIF via ImageMagick)')
    args = parser.parse_args()

    plot_overview(args.output_dir, args.save)


if __name__ == '__main__':
    main()
