"""Compare a PROTEUS interior_refactor run to the CHILI baseline.

Loads both:
  - the CHILI reference CSV (output_files/chili_reference/outputs/...)
  - the PROTEUS run helpfile from the auto-named output dir

and produces overlay plots of the key intercomparison quantities, plus a
text table of the relative agreement at three benchmark times: 1 kyr,
phi=0.5 crossing, and phi=phi_crit (solidification).

Usage:
    python compare_to_chili.py earth_spider

The argument selects which CHILI case to compare against. The PROTEUS run
output is located via the standard auto-named output dir under output/
(e.g. output/earth_spider for tests/validation/chili/earth_spider.toml).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Map case name -> (CHILI CSV path, PROTEUS output dir name)
CASES = {
    'earth_spider': (
        'output_files/chili_reference/outputs/evolution-proteus-earth-data.csv',
        'output/earth_spider',
    ),
    'earth_aragog': (
        'output_files/chili_reference/outputs/evolution-proteus-earth-data.csv',
        'output/chili_earth_aragog',
    ),
}


def load_chili(path: Path) -> pd.DataFrame:
    """Load the CHILI reference CSV. Column names from CHILI Table 3."""
    df = pd.read_csv(path)
    return df


def load_proteus(output_dir: Path) -> pd.DataFrame:
    """Load the PROTEUS runtime helpfile from the case output directory."""
    hf = output_dir / 'runtime_helpfile.csv'
    if not hf.exists():
        raise FileNotFoundError(
            f'PROTEUS helpfile not found at {hf}. '
            'Has the run started? Check the output directory name matches '
            'the config file path.'
        )
    df = pd.read_csv(hf, sep='\t')
    return df


# Mapping from CHILI column name -> (PROTEUS column name, units note)
# Both are scalars per row.
CHILI_TO_PROTEUS = {
    't(yr)': 'Time',
    'T_surf(K)': 'T_surf',
    'T_pot(K)': 'T_magma',
    'flux_OLR(W/m2)': 'F_olr',
    'flux_ASR(W/m2)': 'F_ins',
    'phi(vol_frac)': 'Phi_global',
    'p_surf(bar)': 'P_surf',
    'p_H2O(bar)': 'H2O_bar',
    'p_CO2(bar)': 'CO2_bar',
    'p_CO(bar)': 'CO_bar',
    'p_H2(bar)': 'H2_bar',
    'p_CH4(bar)': 'CH4_bar',
    'p_O2(bar)': 'O2_bar',
}


def quick_table(chili: pd.DataFrame, prot: pd.DataFrame) -> None:
    """Print a quick agreement table at the iteration-0 state, the phi=0.5
    crossing, and the final state of the CHILI reference."""
    print()
    print('=== Iteration 0 (initial state) ===')
    print(f'{"quantity":<25} {"CHILI":>12} {"PROTEUS":>12} {"ratio":>10}')
    for ck, pk in CHILI_TO_PROTEUS.items():
        if pk not in prot.columns or ck not in chili.columns:
            continue
        c = chili[ck].iloc[0]
        p = prot[pk].iloc[0]
        ratio = p / c if abs(c) > 1e-12 else float('nan')
        print(f'{ck:<25} {c:>12.4g} {p:>12.4g} {ratio:>10.3f}')

    # Find phi=0.5 crossing in each
    def crossing(df, col, val):
        x = df[col].to_numpy()
        idx = int(np.argmin(np.abs(x - val)))
        return df.iloc[idx]
    print()
    print('=== Phi=0.5 crossing (rheological front) ===')
    cr_c = crossing(chili, 'phi(vol_frac)', 0.5)
    cr_p = crossing(prot, 'Phi_global', 0.5)
    print(f'  CHILI:   t={cr_c["t(yr)"]:.3e} yr, T_surf={cr_c["T_surf(K)"]:.0f} K, '
          f'p_H2O={cr_c["p_H2O(bar)"]:.1f} bar')
    print(f'  PROTEUS: t={cr_p["Time"]:.3e} yr, T_surf={cr_p["T_surf"]:.0f} K, '
          f'p_H2O={cr_p["H2O_bar"]:.1f} bar')

    print()
    print('=== Final iteration in PROTEUS run ===')
    print(f'  PROTEUS: t={prot["Time"].iloc[-1]:.3e} yr, '
          f'T_surf={prot["T_surf"].iloc[-1]:.0f} K, '
          f'Phi={prot["Phi_global"].iloc[-1]:.4f}, '
          f'p_H2O={prot["H2O_bar"].iloc[-1]:.1f} bar')
    print(f'  CHILI final: t={chili["t(yr)"].iloc[-1]:.3e} yr, '
          f'T_surf={chili["T_surf(K)"].iloc[-1]:.0f} K, '
          f'Phi={chili["phi(vol_frac)"].iloc[-1]:.4f}, '
          f'p_H2O={chili["p_H2O(bar)"].iloc[-1]:.1f} bar')


def overlay_plot(chili: pd.DataFrame, prot: pd.DataFrame, case: str,
                  out_pdf: Path) -> None:
    """Produce a 3x3 overlay grid of CHILI vs PROTEUS for the key quantities."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    panels = [
        ('T_surf(K)', 'T_surf', r'$T_\mathrm{surf}$ [K]', False),
        ('T_pot(K)', 'T_magma', r'$T_\mathrm{pot}$ [K]', False),
        ('phi(vol_frac)', 'Phi_global', r'$\Phi_\mathrm{global}$', False),
        ('flux_OLR(W/m2)', 'F_olr', r'$F_\mathrm{OLR}$ [W m$^{-2}$]', True),
        ('flux_ASR(W/m2)', 'F_ins', r'$F_\mathrm{ASR}$ [W m$^{-2}$]', True),
        ('p_surf(bar)', 'P_surf', r'$P_\mathrm{surf}$ [bar]', True),
        ('p_H2O(bar)', 'H2O_bar', r'$P_{\mathrm{H_2O}}$ [bar]', True),
        ('p_CO2(bar)', 'CO2_bar', r'$P_{\mathrm{CO_2}}$ [bar]', True),
        ('p_H2(bar)', 'H2_bar', r'$P_{\mathrm{H_2}}$ [bar]', True),
    ]
    for ax, (ck, pk, ylab, log_y) in zip(axes.flat, panels):
        if ck in chili.columns:
            ax.plot(chili['t(yr)'], chili[ck],
                    color='black', linestyle='-', linewidth=2,
                    marker='o', markersize=2.5, alpha=0.85, label='CHILI ref')
        if pk in prot.columns:
            ax.plot(prot['Time'], prot[pk],
                    color='#d62728', linestyle='--', linewidth=1.5,
                    marker='s', markersize=2.5, alpha=0.85, label='PROTEUS (this run)')
        ax.set_xscale('log')
        ax.set_xlim(left=max(1.0, prot['Time'].min() if len(prot) else 1.0))
        if log_y:
            ax.set_yscale('log')
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3, which='both')
    for ax in axes[-1]:
        ax.set_xlabel('Model time [yr]')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2,
                   bbox_to_anchor=(0.5, 0.995), frameon=False)
    fig.suptitle(f'CHILI vs PROTEUS interior_refactor — {case}',
                 y=0.97, fontsize=13, weight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f'wrote {out_pdf}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('case', choices=list(CASES.keys()))
    args = p.parse_args()

    repo = Path(__file__).resolve().parents[3]
    chili_path = repo / CASES[args.case][0]
    prot_path = repo / CASES[args.case][1]

    if not chili_path.exists():
        sys.exit(f'CHILI reference not found: {chili_path}')
    chili = load_chili(chili_path)
    prot = load_proteus(prot_path)

    print(f'Loaded CHILI:   {chili_path}, rows={len(chili)}, '
          f'time range {chili["t(yr)"].iloc[0]:.2e} to {chili["t(yr)"].iloc[-1]:.2e} yr')
    print(f'Loaded PROTEUS: {prot_path}, rows={len(prot)}, '
          f'time range {prot["Time"].iloc[0]:.2e} to {prot["Time"].iloc[-1]:.2e} yr')

    quick_table(chili, prot)

    out_dir = repo / 'output_files' / 'chili_validation'
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_plot(chili, prot, args.case, out_dir / f'{args.case}_overlay.pdf')


if __name__ == '__main__':
    main()
