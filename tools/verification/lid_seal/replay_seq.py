"""Replay the held span of the front-trapping tutorial run with a batch and a sequential reservoir.

``batch`` is the code: R_i = M_melt + D_i xtal + tl and the floor X_eq R_i; it must reproduce the
stored run. ``seq`` buries at each freezing step [(1 - F_tl) D_i + F_tl] c_i dM, with c_i the
concentration of the free held mass in the melt before the step; the buried mass B_i stays and
does not leak, and the floor is X_eq M_melt + B_i.

Usage::

    python tools/verification/lid_seal/replay_seq.py TRAP_RUN OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _runs import closing_row, read_run

import proteus.outgas.lid_seal as ls
from proteus.config import read_config_object
from proteus.outgas.compaction import BRANCH_REMELT
from proteus.outgas.trapping import effective_partition
from proteus.outgas.wrapper import run_crystallized

SP = ('H2O', 'CO2', 'CO', 'CH4')
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
    D = {s: float(getattr(cfg.outgas, f'D_const_{s}', 0.0)) for s in ls.SEALED_SPECIES}
    orig_res, orig_fm = ls.reservoirs, ls._floor_model
    res = {}
    for mode in ('batch', 'seq'):
        row = on.iloc[k].to_dict()
        B = dict.fromkeys(ls.SEALED_SPECIES, 0.0)
        leaked = dict.fromkeys(SP, 0.0)
        if mode == 'seq':
            ls.reservoirs = lambda c, h: orig_res(
                c, {**h, 'seal_xtal_kg': 0.0, 'seal_tl_kg': 0.0}
            )

            def floor_model(c, h, p):
                f = orig_fm(c, h, p)
                return lambda leak, s: f(leak, s) + B[s]

            ls._floor_model = floor_model
        try:
            for j in range(k + 1, len(on)):
                r, prev = on.iloc[j], on.iloc[j - 1]
                dt = r.Time - row['Time']
                for x in ROW_KEYS + TRAP_KEYS:
                    row[x] = r[x]
                assert float(r.trap_branch) != BRANCH_REMELT, 'remelt in the held span'
                ls.add_crystals(row, on.iloc[:j])
                if mode == 'seq' and r.trap_dM_RM > 0.0:
                    m_melt_prev = min(max(prev.Phi_global, 0.0), 1.0) * prev.M_mantle
                    for s in ls.SEALED_SPECIES:
                        free = row[f'{s}_kg_sealed'] - B[s]
                        if free > 0.0:
                            buried = float(effective_partition(r.trap_F_tl, D[s])) * free
                            B[s] += min(buried / m_melt_prev * r.trap_dM_RM, free)
                before = {s: row[f'{s}_kg_sealed'] for s in SP}
                run_crystallized(cfg, row, dt)
                ls.leak_step(cfg, row, dt)
                for s in SP:
                    leaked[s] += before[s] - row[f'{s}_kg_sealed']
                if mode == 'batch':
                    for c in ('H2O_kg_atm', 'H2O_kg_sealed', 'CO2_kg_atm'):
                        assert abs(row[c] / r[c] - 1) < 1e-12, (j, c, row[c], r[c])
        finally:
            ls.reservoirs, ls._floor_model = orig_res, orig_fm
        res[mode] = {
            f'{s}_{q}': v
            for s in SP
            for q, v in (
                ('atm', row[f'{s}_kg_atm']),
                ('held', row[f'{s}_kg_sealed']),
                ('leaked', leaked[s]),
            )
        }
        res[mode] |= {'P_surf': row['P_surf'], 'Time': float(row['Time'])}
        res[mode] |= {'B_H2O': B['H2O'], 'B_CO2': B['CO2']}
    b, s = res['batch']['H2O_atm'], res['seq']['H2O_atm']
    print(
        f'H2O in the atmosphere: batch {b:.6e} seq {s:.6e} seq/batch - 1 {100 * (s / b - 1):+.4f} %'
    )
    a.out.write_text(json.dumps(res))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
