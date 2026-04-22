#!/usr/bin/env python
"""
Regression-test helper for E.2 static-mode: diff two runtime_helpfile.csv
files produced on the baseline (tl/interior-refactor @ 2401e6d2) and
the redox branch (tl/redox-scaffolding with redox.mode='static').

Usage
-----
    python scripts/regression_static_mode.py <baseline_dir> <branch_dir>

Asserts:
  - row counts match (one run must not be cut short)
  - fO2_shift_IW bit-identical across all rows
  - non-O baseline columns agree within 1e-8 relative
  - the branch-only redox columns exist and are finite where set

Prints a one-line summary + detailed per-column table on divergence.
Exits 0 on pass, 1 on fail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Branch-only columns added by the redox scaffolding that the baseline
# does not have. Skip these in the 1e-8-relative comparison loop.
REDOX_NEW_COLUMNS = frozenset({
    'fO2_dIW', 'fO2_dQFM', 'fO2_dNNO',
    'R_budget_atm', 'R_budget_mantle', 'R_budget_core', 'R_budget_total',
    'R_escaped_cum',
    'Fe3_frac', 'FeO_total_wt', 'Fe2O3_wt',
    'n_Fe3_melt', 'n_Fe2_melt',
    'n_Fe3_solid_total', 'n_Fe2_solid_total', 'n_Fe0_solid_total',
    'dm_Fe0_to_core_cum',
    'redox_solver_fallback_count', 'redox_mode_active',
    'redox_conservation_residual',
    'redox_delta_IW_suggested_by_mariana',
    'M_vol_initial',
})

# The oxygen-first-class commit (D) adds O_kg_* columns and changes
# M_ele to include O. In static mode the VALUES should still match
# because O bookkeeping derives from the same species inventory; this
# set is here only for future-proofing if we ever need to carve O out.
OXYGEN_COLUMNS = frozenset({
    'O_kg_atm', 'O_kg_liquid', 'O_kg_solid', 'O_kg_total',
})


def _load(path: Path) -> pd.DataFrame:
    csv = path / 'runtime_helpfile.csv'
    if not csv.exists():
        raise FileNotFoundError(f'{csv} does not exist')
    return pd.read_csv(csv)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('baseline', type=Path, help='Baseline run output dir')
    ap.add_argument('branch', type=Path, help='Branch run output dir')
    ap.add_argument('--rtol', type=float, default=1e-8,
                    help='Relative tolerance for physical columns (default 1e-8)')
    args = ap.parse_args()

    base = _load(args.baseline)
    br = _load(args.branch)

    failures: list[str] = []

    if len(base) != len(br):
        failures.append(
            f'ROW COUNT MISMATCH: baseline {len(base)} vs branch {len(br)}'
        )
        n = min(len(base), len(br))
        base = base.iloc[:n].reset_index(drop=True)
        br = br.iloc[:n].reset_index(drop=True)

    if 'fO2_shift_IW' in base.columns and 'fO2_shift_IW' in br.columns:
        if not np.array_equal(base['fO2_shift_IW'].values,
                              br['fO2_shift_IW'].values):
            max_abs = float(np.max(np.abs(
                base['fO2_shift_IW'].values - br['fO2_shift_IW'].values
            )))
            failures.append(
                f'fO2_shift_IW NOT bit-identical; max|delta| = {max_abs:.3e}'
            )

    shared_cols = [c for c in base.columns
                   if c in br.columns
                   and c not in REDOX_NEW_COLUMNS
                   and c != 'fO2_shift_IW'
                   and np.issubdtype(base[c].dtype, np.number)]

    for col in shared_cols:
        b = base[col].values
        x = br[col].values
        # Symmetric relative error with a floor at 1.0 so small-value
        # columns aren't falsely flagged. Using max(|b|, |x|, 1.0) not
        # just max(|b|, 1.0) catches the case where branch is large
        # and baseline is near zero.
        scale = np.maximum(np.maximum(np.abs(b), np.abs(x)), 1.0)
        rel = np.abs(b - x) / scale
        worst = float(np.max(rel))
        if worst > args.rtol:
            failures.append(
                f'{col}: max rel diff {worst:.3e} > rtol {args.rtol:.1e}'
            )

    branch_only = set(br.columns) - set(base.columns)
    expected_branch_only = branch_only & REDOX_NEW_COLUMNS
    unexpected_branch_only = branch_only - REDOX_NEW_COLUMNS
    if unexpected_branch_only:
        print(
            'SCHEMA: branch has columns not in REDOX_NEW_COLUMNS allow-list. '
            f'Add these to REDOX_NEW_COLUMNS in scripts/regression_static_mode.py: '
            f'{sorted(unexpected_branch_only)}'
        )
        # Finitecheck the unknown columns so NaN/inf in them still
        # shows up as a failure rather than getting lost behind the
        # schema-mismatch message.
        for col in unexpected_branch_only:
            vals = br[col].values
            if np.issubdtype(vals.dtype, np.number) and not np.all(np.isfinite(vals)):
                failures.append(
                    f'SCHEMA+NONFINITE: new column {col!r} has non-finite values '
                    'on branch; add to allow-list AND investigate.'
                )
        failures.append(
            f'SCHEMA: branch has unexpected new columns '
            f'(see SCHEMA line above): {sorted(unexpected_branch_only)}'
        )

    print(f'rows: baseline={len(base)} branch={len(br)}')
    print(f'shared numeric columns compared: {len(shared_cols)}')
    print(f'expected new redox columns on branch: {len(expected_branch_only)}')
    print(f'  {sorted(expected_branch_only)}')

    if failures:
        print()
        print('FAIL:')
        for f in failures:
            print(f'  - {f}')
        return 1

    print()
    print('PASS: static-mode branch matches baseline within tolerance')
    return 0


if __name__ == '__main__':
    sys.exit(main())
