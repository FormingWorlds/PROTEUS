"""Replay the held span of a stored seal run with one model term changed at a time.

The stored seal run gives the melt fraction, mantle mass, magma temperature and escape of each
row; the open run (seal off) gives the surface pressure and the molar ratios of its CALLIOPE
solve at the same times. A variant is a JSON object of switches:

- ``D0``: ``D_const_H2O = 0`` in the floor reservoir.
- ``NOESC``: no escape on the held atmosphere.
- ``TLIVE``: the floor at the live magma temperature, not the closing one.
- ``POPEN``: the floor partial pressures at the surface pressure of the open run.
- ``RESPEC``: after each step the atmosphere takes the molar ratios H2/H2O, CO/CO2 and CH4/CO2
  of the open run at the same H and C moles.
- ``tau``: the leak time [yr] in place of the configured one.

Writes the end atmosphere of each variant to a JSON file.

Usage::

    python tools/verification/lid_seal/replay_variants.py SEAL OPEN OUT.json VARIANTS_JSON
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _runs import closing_row, read_run

import proteus.outgas.lid_seal as ls
from proteus.config import read_config_object
from proteus.outgas.common import element_masses_from_species
from proteus.outgas.wrapper import run_crystallized

HC = ('H2O', 'H2', 'CO2', 'CO', 'CH4')
STATE = {'factor': None}
_PRESSURES, _REFRESH = ls._pressures, ls._refresh_atmosphere


def _scaled_pressures(kg, mol, gravity, r_int):
    vmr, bar, p = _PRESSURES(kg, mol, gravity, r_int)
    f = STATE['factor']
    if f is None:
        return vmr, bar, p
    return vmr, {s: b * f for s, b in bar.items()}, p * f


def _refresh(hf_row):
    f, STATE['factor'] = STATE['factor'], None
    try:
        _REFRESH(hf_row)
    finally:
        STATE['factor'] = f


def respeciate(row, o):
    """Impose the open row's molar ratios on the atmosphere at fixed H and C moles."""
    n = {s: float(row[f'{s}_mol_atm']) for s in HC}
    mu = {s: float(o[f'{s}_kg_atm']) / float(o[f'{s}_mol_atm']) for s in HC}
    n_h = 2 * n['H2O'] + 2 * n['H2'] + 4 * n['CH4']
    n_c = n['CO2'] + n['CO'] + n['CH4']
    r_h = o.H2_mol_atm / o.H2O_mol_atm
    r_co, r_ch4 = o.CO_mol_atm / o.CO2_mol_atm, o.CH4_mol_atm / o.CO2_mol_atm
    co2 = n_c / (1 + r_co + r_ch4)
    new = {'CO2': co2, 'CO': r_co * co2, 'CH4': r_ch4 * co2}
    h2o = (n_h - 4 * new['CH4']) / (2 * (1 + r_h))
    new.update(H2O=h2o, H2=r_h * h2o)
    for s in HC:
        row[f'{s}_mol_atm'], row[f'{s}_kg_atm'] = new[s], new[s] * mu[s]
    el = element_masses_from_species({s: row[f'{s}_kg_atm'] for s in ls.gas_list})
    for e in ('H', 'C', 'O'):
        row[f'{e}_kg_atm'] = el.get(e, 0.0)
    _refresh(row)


def replay(seal_dir, open_run, D0=False, NOESC=False, TLIVE=False, POPEN=False, RESPEC=False, tau=None):  # fmt: skip
    """End atmosphere [kg] and surface pressure [bar] of one variant."""
    cfg = read_config_object(Path(seal_dir) / 'init_coupler.toml')
    if D0:
        cfg.outgas.D_const_H2O = 0.0
    if tau is not None:
        cfg.outgas.lid_seal.tau_leak = tau
    seal = read_run(seal_dir)
    k = closing_row(seal)
    row = seal.iloc[k].to_dict()
    for i in range(k + 1, len(seal)):
        r, o = seal.iloc[i], open_run.iloc[i]
        assert r.Time == o.Time, f'row {i}: the seal and the open run differ in time'
        dt = r.Time - row['Time']
        row.update(Time=r.Time, Phi_global=r.Phi_global, M_mantle=r.M_mantle, T_magma=r.T_magma)
        row.update(
            esc_step_kg=0.0 if NOESC else r.esc_step_kg,
            esc_rate_total=0.0 if NOESC else r.esc_rate_total,
        )
        if TLIVE:
            row['seal_Tmagma_close'] = max(float(r.T_magma), cfg.outgas.T_floor)
        run_crystallized(cfg, row, dt)
        STATE['factor'] = float(o.P_surf) / float(row['P_surf']) if POPEN else None
        ls.leak_step(cfg, row, dt)
        STATE['factor'] = None
        if RESPEC:
            respeciate(row, o)
    end = {s: float(row[f'{s}_kg_atm']) for s in (*HC, 'H', 'C', 'O')}
    return end | {'P_surf': float(row['P_surf'])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('seal', type=Path, help='output folder of the seal run')
    ap.add_argument('open', type=Path, help='output folder of the open run')
    ap.add_argument('out', type=Path)
    ap.add_argument('variants', help='JSON: {name: {switch: value}}')
    a = ap.parse_args(argv)
    ls._pressures, ls._refresh_atmosphere = _scaled_pressures, _refresh
    open_run = read_run(a.open)
    res = {}
    for name, kw in json.loads(a.variants).items():
        res[name] = replay(a.seal, open_run, **kw)
        print(name, json.dumps(res[name]), flush=True)
    a.out.write_text(json.dumps(res))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
