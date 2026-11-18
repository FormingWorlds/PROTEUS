"""Replay the held span of the front-trapping tutorial run with 3 floors of the held mass.

``front`` is the floor of the code (the solid formed since closure at (1 - F_tl) D + F_tl) and
must reproduce the stored run; ``all`` counts every solid formed since the start of the run at
the same partition, with the layers rebuilt from trap_dM_RM, trap_F_tl and Phi_global of all
rows; ``lattice`` counts the solid since closure at D alone.

Usage::

    python tools/verification/lid_seal/replay_trap.py TRAP_RUN OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _runs import closing_row, read_run

import proteus.outgas.lid_seal as ls
from proteus.config import read_config_object
from proteus.outgas.burial_ledger import build_ledger, ledger_total
from proteus.outgas.compaction import BRANCH_REMELT
from proteus.outgas.wrapper import run_crystallized

ROW_KEYS = ('Time', 'Phi_global', 'M_mantle', 'T_magma', 'esc_step_kg', 'esc_rate_total')
TRAP_KEYS = ('trap_dM_RM', 'trap_F_tl', 'trap_branch')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('run', type=Path, help='output folder of the front-trapping seal run')
    ap.add_argument('out', type=Path)
    a = ap.parse_args(argv)
    on = read_run(a.run)
    cfg = read_config_object(a.run / 'init_coupler.toml')
    k = closing_row(on)
    orig = ls.reservoirs
    partition = {s: float(getattr(cfg.outgas, f'D_const_{s}', 0.0)) for s in ls.SEALED_SPECIES}
    res = {}
    for mode in ('front', 'all', 'lattice'):
        row = on.iloc[k].to_dict()

        def all_solid(config, hf_row, j):
            dm = np.maximum(on.trap_dM_RM.values[: j + 1], 0.0)
            f = on.trap_F_tl.values[: j + 1]
            inc = np.column_stack([(1.0 - f) * dm, f * dm])
            layers = build_ledger(on.Phi_global.values[: j + 1], np.cumsum(inc, axis=0))
            xtal, tl = ledger_total(layers, 2)
            _, phi, m_melt = orig(config, {**hf_row, 'seal_xtal_kg': 0.0, 'seal_tl_kg': 0.0})
            return (
                {s: m_melt + partition[s] * xtal + tl for s in ls.SEALED_SPECIES},
                phi,
                m_melt,
            )

        def lattice(config, hf_row):
            _, phi, m_melt = orig(config, {**hf_row, 'seal_xtal_kg': 0.0, 'seal_tl_kg': 0.0})
            solid = hf_row['seal_xtal_kg'] + hf_row['seal_tl_kg']
            return {s: m_melt + partition[s] * solid for s in ls.SEALED_SPECIES}, phi, m_melt

        try:
            for j in range(k + 1, len(on)):
                r = on.iloc[j]
                dt = r.Time - row['Time']
                for x in ROW_KEYS + TRAP_KEYS:
                    row[x] = r[x]
                if mode == 'all':
                    ls.reservoirs = lambda c, h, j=j: all_solid(c, h, j)
                elif mode == 'lattice':
                    ls.reservoirs = lattice
                ls.add_crystals(row, on.iloc[:j])
                if float(row['trap_branch']) == BRANCH_REMELT:
                    raise SystemExit(f'remelt at row {j}: the replay has no trapped release')
                run_crystallized(cfg, row, dt)
                ls.leak_step(cfg, row, dt)
                if mode == 'front':
                    for s in ('H2O', 'CO2'):
                        for c in (f'{s}_kg_sealed', f'{s}_kg_atm'):
                            same = abs(row[c] / r[c] - 1) < 1e-12 or row[c] == r[c]
                            assert same, (j, c, row[c], r[c])
                    xtal = abs(row['seal_xtal_kg'] - r.seal_xtal_kg)
                    assert xtal <= 1e-12 * max(r.seal_xtal_kg, 1.0), (j, xtal)
        finally:
            ls.reservoirs = orig
        res[mode] = {s: row[f'{s}_kg_atm'] for s in ('H2O', 'CO2')}
        res[mode] |= {'P_surf': row['P_surf'], 'Time': row['Time']}
    last = on.iloc[-1]
    res['stored'] = {'H2O': last.H2O_kg_atm, 'CO2': last.CO2_kg_atm}
    res['stored'] |= {'P_surf': last.P_surf, 'Time': last.Time}
    res['close_row'] = k
    for m in res:
        print(m, res[m])
    a.out.write_text(json.dumps(res))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
