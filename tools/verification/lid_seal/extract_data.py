"""Write the stored tutorial data of the lid seal tests from the 3 tutorial runs.

Reads the seal run, the open run (seal off) and the hold run (tau_leak = inf) and writes, into
the output folder, ``tutorial_closing_rows.csv`` (the closing row and the 13 rows after it, the
columns that are not 0 on the closing row) and ``tutorial_water.csv`` (water in the atmosphere
and under the crust; every 5th row before closure, every row from it).

Usage::

    python tools/verification/lid_seal/extract_data.py SEAL OPEN HOLD OUTDIR --commit SHA
        --aragog SHA --calliope VERSION
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from _runs import closing_row, read_run


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('seal', type=Path)
    ap.add_argument('open', type=Path)
    ap.add_argument('hold', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--commit', required=True, help='PROTEUS commit of the runs')
    ap.add_argument('--aragog', required=True, help='aragog commit of the runs')
    ap.add_argument('--calliope', required=True, help='CALLIOPE version of the runs')
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    seal, off, hold = read_run(a.seal), read_run(a.open), read_run(a.hold)
    k = closing_row(seal)
    head = (
        '# Tutorial input/tutorials/tutorial_lid_seal.toml (docs/Tutorials/lid_seal.md), '
        f'PROTEUS commit {a.commit}, aragog commit {a.aragog}, CALLIOPE {a.calliope}.\n'
    )
    c = seal.iloc[k]
    keep = ['Time', 'Phi_global', 'M_mantle', 'R_int', 'gravity', 'T_magma']
    keep += ['fO2_shift_IW_derived', 'M_atm']
    for x in seal.columns:
        if x.endswith(('_kg_atm', '_mol_atm', '_kg_sealed', '_mol_sealed')) and c[x] != 0.0:
            keep.append(x)
    rows = seal.iloc[k : k + 14][keep].copy()
    rows.insert(0, 'row', np.arange(k, k + 14))
    with open(a.out / 'tutorial_closing_rows.csv', 'w') as f:
        f.write(head)
        f.write(
            f'# Seal run, rows {k} (the closing row) to {k + 13}; '
            f'only columns that are not 0 on row {k}.\n'
        )
        rows.to_csv(f, index=False, float_format='%.17g')
    idx = sorted(set(range(0, k, 5)) | set(range(k, len(seal))))
    cols = ('Phi_global', 'M_mantle', 'esc_step_kg', 'esc_rate_total')
    water = pd.DataFrame(
        {
            'Time': seal['Time'].values[idx],
            'seal_atm': seal['H2O_kg_atm'].values[idx],
            'seal_held': seal['H2O_kg_sealed'].values[idx],
            'seal_eq': seal['H2O_kg_eq'].values[idx],
            'seal_active': seal['seal_active'].values[idx],
        }
        | {x: seal[x].values[idx] for x in cols}
    )
    for name, run in (('open_atm', off), ('hold_atm', hold)):
        same = np.array_equal(run['Time'].values[: len(seal)], seal['Time'].values[: len(run)])
        assert same, f'{name}: the run times differ from the seal run'
        water[name] = run['H2O_kg_atm'].values[idx]
    with open(a.out / 'tutorial_water.csv', 'w') as f:
        f.write(head)
        f.write(
            '# H2O [kg]: atmosphere of the seal, open and hold (tau_leak inf) runs, '
            'held mass and its floor;\n'
        )
        f.write(
            '# melt fraction, mantle mass [kg] and escape of the step [kg, kg s-1] '
            'of the seal run.\n'
        )
        f.write(f'# Every 5th row before the closing row {k}, every row from it.\n')
        water.to_csv(f, index=False, float_format='%.17g')
    print('closing row', k, 'columns', len(keep), 'water rows', len(water))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
