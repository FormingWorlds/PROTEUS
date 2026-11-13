"""Replay the held span of stored seal runs with the held atmosphere frozen or re-speciated.

For each seal run and mode, the held span is replayed with the code of this tree. Mode
``frozen`` keeps the species of the held gas, as the code does; ``close`` and ``surf`` re-speciate
it after each step as a closed H-C-O gas (``closed_gas.py``) at the closing magma temperature or
at the surface temperature of the row. The end values are compared with the open run (seal off).

Usage::

    python tools/verification/lid_seal/probe_closed_gas.py OPEN OUT.json NAME=SEAL [NAME=SEAL ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from _runs import closing_row, read_run
from closed_gas import SP, elements, solve

import proteus.outgas.lid_seal as ls
from proteus.config import read_config_object
from proteus.outgas.common import element_masses_from_species
from proteus.outgas.wrapper import run_crystallized

MU = {}


def respeciate(row, T):
    """Re-speciate the atmosphere of ``row`` as a closed gas at ``T``; element moles before and
    after, and log10 fO2."""
    n = {s: float(row[f'{s}_mol_atm']) for s in SP}
    for s in SP:
        if n[s] > 0:
            MU[s] = float(row[f'{s}_kg_atm']) / n[s]
    el = elements(n)
    new, y = solve(el, float(row['P_surf']), T, n)
    for s in SP:
        row[f'{s}_mol_atm'], row[f'{s}_kg_atm'] = new[s], new[s] * MU[s]
    em = element_masses_from_species({s: row[f'{s}_kg_atm'] for s in ls.gas_list})
    for e in ('H', 'C', 'O'):
        row[f'{e}_kg_atm'] = em.get(e, 0.0)
    ls._refresh_atmosphere(row)
    return el, elements(new), y


def replay(seal_dir, open_run, mode):
    """End values, their gap to the open run [%], the element drift and the last log10 fO2."""
    cfg = read_config_object(Path(seal_dir) / 'init_coupler.toml')
    seal = read_run(seal_dir)
    k = closing_row(seal)
    row = seal.iloc[k].to_dict()
    tclose = float(row['seal_Tmagma_close'])
    drift, ys = 0.0, []
    for i in range(k + 1, len(seal)):
        r, o = seal.iloc[i], open_run.iloc[i]
        assert r.Time == o.Time, f'row {i}: the seal and the open run differ in time'
        dt = r.Time - row['Time']
        row.update(Time=r.Time, Phi_global=r.Phi_global, M_mantle=r.M_mantle, T_magma=r.T_magma)
        row.update(esc_step_kg=r.esc_step_kg, esc_rate_total=r.esc_rate_total)
        run_crystallized(cfg, row, dt)
        ls.leak_step(cfg, row, dt)
        if mode != 'frozen':
            p0 = float(row['P_surf'])
            a, b, y = respeciate(row, tclose if mode == 'close' else float(r.T_surf))
            drift = max(drift, float(np.max(np.abs(b / a - 1))), abs(row['P_surf'] / p0 - 1))
            ys.append(y)
    o = open_run.iloc[-1]
    out = {s: float(row[f'{s}_kg_atm']) for s in ('H2O', 'CO2', 'CO', 'H2', 'CH4')}
    out['P_surf'] = float(row['P_surf'])
    gap = {
        s: 100 * (out[s] / float(o[f'{s}_kg_atm' if s != 'P_surf' else 'P_surf']) - 1)
        for s in out
    }
    return {
        'end': out,
        'gap_pct': gap,
        'el_drift': drift,
        'logfO2_last': ys[-1] if ys else None,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('open', type=Path, help='output folder of the open run')
    ap.add_argument('out', type=Path)
    ap.add_argument('runs', nargs='+', help='NAME=SEAL_RUN')
    a = ap.parse_args(argv)
    open_run = read_run(a.open)
    res = {}
    for item in a.runs:
        name, seal_dir = item.split('=', 1)
        for mode in ('frozen', 'close', 'surf'):
            key = f'{name}:{mode}'
            res[key] = replay(seal_dir, open_run, mode)
            print(
                key, json.dumps(res[key]['gap_pct']), 'drift', res[key]['el_drift'], flush=True
            )
    a.out.write_text(json.dumps(res))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
