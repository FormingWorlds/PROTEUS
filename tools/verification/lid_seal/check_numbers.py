"""Check every number that the lid seal pages quote from the tutorial runs.

The values come from the tutorial runs and the outputs of the other tools of this folder:
``replay_variants.py`` (the 3 terms of the seal effect), ``probe_closed_gas.py`` (the frozen
species), ``replay_trap.py`` and ``replay_seq.py`` (front trapping) and ``ensemble.py`` (the
round-off starts). Each quoted sentence is a template filled with these values; the check
passes when every filled template is in its page, every printed block of the tutorial equals the
output of the code above it, and every claim in words holds. ``--render`` prints the filled
templates instead.

Usage::

    python tools/verification/lid_seal/check_numbers.py --seal SEAL --open OPEN --tau1e2 RUN
        --trap RUN --variants V.json --probe P.json --trap-replay T.json --seq-replay S.json
        --ensemble E.csv [--docs docs] [--render]
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _runs import closing_row, read_run

PAGES = {
    'T': 'Tutorials/lid_seal.md',
    'E': 'Explanations/lid_seal.md',
    'V': 'Validation/outgas/lid_seal.md',
}
TEMPLATES = [
    ('T', 'The top cell of this mesh is {crust:.1f} km thick, so the crust jumps from 0 to '
          '{crust:.1f} km in one step, and any `delta_close` up to {crust:.1f} km'),
    ('T', 'On the first held step $\\Phi$ falls from {phi_k:.5f} to {phi_k1:.5f}'),
    ('T', 'The melt is still {melt:.1f} % of the mantle at 140 kyr'),
    ('T', 'With the seal, the atmosphere holds {total:.2f} % more water than without the seal.'),
    ('T', 'split the {total:.2f} % into 3 terms'),
    ('T', '- ${spec:+.2f}$: the held gas keeps the species'),
    ('T', 'lower its surface pressure by {plow:.2f} %'),
    ('T', '- ${leak:+.2f}$: the finite leak time keeps this water held;'),
    ('T', '- ${rest:+.2f}$: the rest.'),
    ('T', 'The output above shows a surface pressure {ptot:.1f} % lower in total; the kept '
          'species alone lower it by {plow:.2f} %.'),
    ('T', 'holds {co:.1f} % more CO and {co2:.1f} % less CO$_2$ than the open run'),
    ('T', 'the closing time varies by {t_spread:.1f} yr and the water in the atmosphere at '
          '140 kyr by {seal_spread_tex} % with the seal and by {open_spread_tex} % without it.'),
    ('T', 'the seal effect is their difference at the same start: {band_printed:.3f} % in each '
          'of the 6 starts.'),
    ('E', 'the top cell is {crust:.1f} km thick'),
    ('E', 'the seal closes at $\\Phi_c = {phi_k:.5f}$'),
    ('E', '{co:.1f} % more CO, {co2:.1f} % less CO$_2$ and {h2o:.1f} % more water'),
    ('E', '{ptot:.1f} % lower surface pressure'),
    ('E', 'holds {band_printed:.3f} % more atmospheric water than the open run of its start in '
          'each of 6 starts that differ by round-off; between them the water varies by less than '
          '0.001 %.'),
    ('E', 'give {seq:.2f} % more water'),
    ('V', 'by {spec:.2f} percentage points'),
    ('V', 'lowers it by {leak_abs:.2f} points'),
    ('V', 'The run ends {total:.2f} % above the open run.'),
    ('V', 'On the first {n_below} held steps, about {span_below} yr, the leak lags'),
    ('V', 'the run lies up to {below_max:.3f} % below the open run'),
    ('V', 'The seal closes at {t_close_kyr:.0f} kyr.'),
    ('V', 'That run gives {trap_h2o} kg of water in the atmosphere at 140 kyr'),
    ('V', 'gives {trap_all} kg, {trap_ratio:.2f} times as much'),
    ('V', 'gives {seq:.2f} % more water in the atmosphere at 140 kyr'),
    ('V', 'by at most {closed_gas:.2f} percentage points'),
    ('V', 'holds {co:.1f} % more CO and {co2:.1f} % less CO$_2$ than the open run'),
]  # fmt: skip


def _sci(x: float) -> str:
    """1.339e15 form: 3 decimals, no plus sign in the exponent."""
    m, e = f'{x:.3e}'.split('e')
    return f'{m}e{int(e)}'


def _tex(x: float) -> str:
    """$9 \\times 10^{-4}$ form, 1 significant digit."""
    m, e = f'{x:.0e}'.split('e')
    return f'${m} \\times 10^{{{int(e)}}}$'


def tutorial_blocks(page: str, seal: Path, open_run: Path):
    """Run each python block of the tutorial on the given runs; pairs of (printed, expected)."""
    blocks = re.findall(r'```(python|text)\n(.*?)```', page, re.S)
    env, prev, pairs = {}, None, []
    for kind, body in blocks:
        if kind == 'python':
            if 'savefig' in body or 'tomlkit' in body:
                prev = None
                continue
            body = body.replace('output/tutorial_lid_seal_off/', f'{open_run}/')
            body = body.replace('output/tutorial_lid_seal/', f'{seal}/')
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                exec(body, env)
            prev = out.getvalue()
        elif prev is not None:
            pairs.append((prev.strip(), body.strip()))
            prev = None
    return pairs


def values(a) -> tuple[dict, list]:
    """The quoted values and the claims in words, as (name, holds) pairs."""
    on, off, tau = read_run(a.seal), read_run(a.open), read_run(a.tau1e2)
    k = closing_row(on)
    var = json.loads(a.variants.read_text())
    probe = json.loads(a.probe.read_text())
    trap_r = json.loads(a.trap_replay.read_text())
    seq_r = json.loads(a.seq_replay.read_text())
    trap = read_run(a.trap)
    ens = pd.read_csv(a.ensemble, comment='#', float_precision='round_trip')
    w = off['H2O_kg_atm'].iloc[-1]
    total = 100 * (on['H2O_kg_atm'].iloc[-1] / w - 1)
    spec = 100 * (var['base']['H2O'] - var['b']['H2O']) / w
    leak = 100 * (var['base']['H2O'] - var['t2']['H2O']) / w
    frozen = probe['seal:frozen']['gap_pct']
    gap = (on['H2O_kg_atm'] / off['H2O_kg_atm'] - 1).values[k:]
    below = np.flatnonzero(gap[1:] < 0) + 1
    span = on['Time'].values[k + below[-1]] - on['Time'].values[k]
    band = 100 * (ens.seal_atm / ens.open_atm - 1)
    starts = ens[ens.run != 'printed']
    v = dict(
        crust=on['seal_delta'].iloc[k] / 1e3,
        phi_k=on['Phi_global'].iloc[k],
        phi_k1=on['Phi_global'].iloc[k + 1],
        melt=100 * off['Phi_global'].iloc[-1],
        total=total,
        spec=spec,
        leak=leak,
        leak_abs=-leak,
        rest=total - spec - leak,
        plow=100 * (1 - var['base']['P_surf'] / var['b']['P_surf']),
        ptot=-frozen['P_surf'],
        co=frozen['CO'],
        co2=-frozen['CO2'],
        h2o=frozen['H2O'],
        t_spread=np.ptp(ens.t_close),
        seal_spread=100 * np.ptp(ens.seal_atm) / ens.seal_atm.mean(),
        open_spread=100 * np.ptp(ens.open_atm) / ens.open_atm.mean(),
        band_printed=band[ens.run == 'printed'].item(),
        n_below=int(below.size),
        span_below=int(50 * round(span / 50)),
        below_max=-100 * gap[below].min(),
        t_close_kyr=on['Time'].iloc[k] / 1e3,
        trap_h2o=_sci(trap['H2O_kg_atm'].iloc[-1]),
        trap_all=_sci(trap_r['all']['H2O']),
        trap_ratio=trap_r['all']['H2O'] / trap['H2O_kg_atm'].iloc[-1],
        seq=100 * (seq_r['seq']['H2O_atm'] / seq_r['batch']['H2O_atm'] - 1),
        band_digits={f'{x:.3f}' for x in band},
        closed_gas=max(
            abs(probe[f'seal:{m}']['gap_pct'][s] - frozen[s])
            for m in ('close', 'surf')
            for s in ('H2O', 'CO2', 'CO')
        ),
    )
    # Wall clock and the seal diagnostics (only filled with the seal on) are no model state.
    cols = [c for c in off.columns if c != 'runtime' and not c.startswith('seal_')]
    v['seal_spread_tex'], v['open_spread_tex'] = _tex(v['seal_spread']), _tex(v['open_spread'])
    before = off.iloc[:k][cols].reset_index(drop=True)
    same = on.iloc[:k][cols].reset_index(drop=True)
    claims = [
        ('the 2 runs agree to the last digit before the closing step', same.equals(before)),
        ('about a fifth of the mantle is molten at closure', 0.15 < v['phi_k'] < 0.25),
        ('the leak starts on the first held step', bool(np.diff(on['H2O_kg_sealed'].values[k : k + 2])[0] < 0)),
        ('the probe water gap is the run gap', abs(frozen['H2O'] - total) < 1e-9),
        ('the front replay gives the trap run', abs(trap_r['front']['H2O'] / trap['H2O_kg_atm'].iloc[-1] - 1) < 1e-12),
        ('the batch replay gives the trap run', abs(seq_r['batch']['H2O_atm'] / trap['H2O_kg_atm'].iloc[-1] - 1) < 1e-12),
        ('the printed start is the seal run', ens.loc[ens.run == 'printed', 'seal_atm'].item() == on['H2O_kg_atm'].iloc[-1]),
        ('5 starts, superheat up to 2.5e-9 K', len(starts) == 5 and starts.dT.max() < 2.5e-9 * (1 + 1e-6)),
        ('tau_leak 1e2 changes the water by less than 0.1 %', 100 * abs(tau['H2O_kg_atm'].iloc[-1] / on['H2O_kg_atm'].iloc[-1] - 1) < 0.1),
        ('the 6 seal effects agree to 3 decimals', len(v['band_digits']) == 1),
        ('the starts differ in end water by less than 0.001 %', max(v['seal_spread'], v['open_spread']) < 1e-3),
        ('the held span below the open run is one run of steps', list(below) == list(range(1, below.size + 1))),
    ]  # fmt: skip
    return v, claims


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for name in ('seal', 'open', 'tau1e2', 'trap', 'variants', 'probe'):
        ap.add_argument(f'--{name}', type=Path, required=True)
    for name in ('trap-replay', 'seq-replay', 'ensemble'):
        ap.add_argument(f'--{name}', type=Path, required=True)
    ap.add_argument('--docs', type=Path, default=Path('docs'))
    ap.add_argument('--render', action='store_true', help='print the filled templates')
    a = ap.parse_args(argv)
    v, claims = values(a)
    pages = {key: (a.docs / path).read_text() for key, path in PAGES.items()}
    filled = [(key, t.format(**v)) for key, t in TEMPLATES]
    if a.render:
        for key, text in filled:
            print(f'{PAGES[key]}: {text}')
        for name, ok in claims:
            print(f'claim: {name}: {ok}')
        return 0
    bad = 0
    for printed, expected in tutorial_blocks(pages['T'], a.seal, a.open):
        ok = printed == expected
        bad += not ok
        print('PASS' if ok else 'FAIL', 'printed block', expected.splitlines()[0])
    for key, text in filled:
        ok = text in pages[key]
        bad += not ok
        print('PASS' if ok else 'FAIL', PAGES[key], repr(text))
    for name, ok in claims:
        bad += not ok
        print('PASS' if ok else 'FAIL', 'claim:', name)
    print('FAILS', bad)
    return int(bad > 0)


if __name__ == '__main__':
    sys.exit(main())
