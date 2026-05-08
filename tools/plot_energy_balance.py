#!/usr/bin/env python3
"""
Plot the EOS-consistent energy-conservation diagnostic for a PROTEUS run.

Reads ``runtime_helpfile.csv`` from a single output directory and writes
a 3-panel PDF (plus a per-source-power PDF) summarising:

(a) ``E_state_cons``(t) (frozen-mass integrated mantle enthalpy) and
    ``E_state_cons``(0) + cumulative integral of the RHS power balance.
    A perfectly-conserving run has the two curves overlap.
(b) The fractional residual ``E_residual_cons_frac`` vs time. The 1 J
    floor on the divisor keeps this bounded at quiescent steady state.
(c) Source-power decomposition: -F_int*A_int (top boundary loss),
    +F_cmb*A_cmb (CMB inflow), Q_radio_W, Q_tidal_W in watts.

Usage
-----
    python tools/plot_energy_balance.py <output_dir> [--label LABEL]

Output
------
    output_files/energy_balance/<label>_balance.pdf
    output_files/energy_balance/<label>_powers.pdf

The script never modifies the input directory and never overwrites
files outside ``output_files/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use('Agg')  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REQUIRED = (
    'Time',
    'E_state_cons_J',
    'F_int',
    'F_cmb',
    'Q_radio_W',
    'Q_tidal_W',
    'R_int',
    'R_core',
    'dE_predicted_cons_J',
    'E_residual_cons_J',
    'E_residual_cons_frac',
)


def _load(simdir: str) -> pd.DataFrame:
    """Read the runtime helpfile and validate the diagnostic columns are
    present. Raises with a concrete column-list on schema mismatch so a
    pre-diagnostic legacy run is identifiable at a glance."""
    fpath = os.path.join(simdir, 'runtime_helpfile.csv')
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f'No runtime_helpfile.csv at {fpath}')
    df = pd.read_csv(fpath, sep=r'\s+')
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise KeyError(
            f'Helpfile missing energy-conservation columns: {missing}. '
            'Run was launched before the diagnostic was added.'
        )
    if not np.any(df['E_state_cons_J'] != 0.0):
        raise ValueError(
            f'E_state_cons_J is identically zero across {len(df)} rows. '
            'The active interior module did not populate the diagnostic '
            '(non-Aragog or Aragog without entropy formulation).'
        )
    return df


def _powers(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return the per-row power contributions in watts, with sign such
    that positive = adds energy to the mantle."""
    R_int = df['R_int'].to_numpy()
    R_core = df['R_core'].to_numpy()
    A_int = 4.0 * np.pi * R_int * R_int
    A_core = np.where(R_core > 0, 4.0 * np.pi * R_core * R_core, A_int)
    return {
        '-F_int * A_int (top loss)': -df['F_int'].to_numpy() * A_int,
        '+F_cmb * A_core (CMB inflow)': df['F_cmb'].to_numpy() * A_core,
        'Q_radio': df['Q_radio_W'].to_numpy(),
        'Q_tidal': df['Q_tidal_W'].to_numpy(),
    }


def _plot_balance(df: pd.DataFrame, out_path: str, label: str) -> None:
    """Three-panel balance figure."""
    t = df['Time'].to_numpy()
    E_state = df['E_state_cons_J'].to_numpy()
    E_anchor = E_state[0]
    E_predicted = E_anchor + df['dE_predicted_cons_J'].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.0), sharex=True)

    ax = axes[0]
    ax.plot(t, E_state, label=r'$E_{\rm state\_cons}$ (EOS-integrated, frozen mass)', lw=1.5)
    ax.plot(
        t,
        E_predicted,
        label=r'$E_{\rm anchor} + \int P_{\rm RHS}\, dt$',
        lw=1.0,
        ls='--',
    )
    ax.set_ylabel('Mantle energy [J]')
    ax.set_title(f'(a) energy state vs predicted: {label}')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    frac = df['E_residual_cons_frac'].to_numpy()
    ax.plot(t, frac, lw=1.0, color='C3')
    ax.axhline(0.0, color='k', lw=0.5, alpha=0.5)
    ax.set_ylabel(r'$E_{\rm residual} / \max(|\Delta E_{\rm state}|, 1\,{\rm J})$')
    ax.set_title('(b) fractional residual')
    ax.grid(alpha=0.3)
    if np.all(np.isfinite(frac)) and np.any(np.abs(frac) > 0):
        ymax = max(0.01, 1.1 * float(np.max(np.abs(frac))))
        ax.set_ylim(-ymax, ymax)

    ax = axes[2]
    powers = _powers(df)
    for name, arr in powers.items():
        if np.any(arr != 0.0):
            ax.plot(t, arr, label=name, lw=1.0)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('Power contribution [W]')
    ax.set_title('(c) source decomposition')
    ax.set_xscale('log')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


def _plot_powers(df: pd.DataFrame, out_path: str, label: str) -> None:
    """Standalone log-log absolute-value power plot. Lets a reader see
    decade-scale crossings between the boundary losses and the volumetric
    sources on the same log-log canvas. Uses abs(P) so the plot is robust
    to transient sign flips."""
    t = df['Time'].to_numpy()
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.5))
    powers = _powers(df)
    for name, arr in powers.items():
        if np.any(arr != 0.0):
            ax.plot(t, np.abs(arr), label=name, lw=1.0)
    ax.set_xlabel('Time [yr]')
    ax.set_ylabel('|Power| [W]')
    ax.set_title(f'Per-source absolute power: {label}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(out_path, format='pdf')
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        'simdir',
        help='Path to a PROTEUS output directory containing '
        'runtime_helpfile.csv (e.g. output/verify_dilon_phicap005).',
    )
    parser.add_argument(
        '--label',
        default=None,
        help='Slug for output filenames; defaults to the basename of simdir.',
    )
    parser.add_argument(
        '--outdir',
        default='output_files/energy_balance',
        help='Output directory for PDFs (gitignored). Created if missing.',
    )
    args = parser.parse_args(argv)

    label = args.label or os.path.basename(os.path.normpath(args.simdir))
    df = _load(args.simdir)

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    balance_path = out_dir / f'{label}_balance.pdf'
    powers_path = out_dir / f'{label}_powers.pdf'
    _plot_balance(df, str(balance_path), label)
    _plot_powers(df, str(powers_path), label)

    print(f'Wrote {balance_path}')
    print(f'Wrote {powers_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
