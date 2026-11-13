"""Write ``tutorial_ensemble.csv``: end values of the printed tutorial run and its round-off starts.

Each start is a seal run and an open run of the tutorial with ``planet.delta_T_super`` changed by
dT; the file holds dT, the closing time of the seal run and the water in the atmosphere at the
end of both runs.

Usage::

    python tools/verification/lid_seal/ensemble.py OUT.csv --commit SHA --aragog SHA --calliope VERSION
        printed=SEAL,OPEN k1=SEAL,OPEN ...
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from _runs import closing_row, read_run

BASE_SUPERHEAT = 500.0  # [K], the default planet.delta_T_super of the tutorial


def superheat(run: Path) -> float:
    cfg = tomllib.load(open(run / 'init_coupler.toml', 'rb'))
    return float(cfg['planet'].get('delta_T_super', BASE_SUPERHEAT))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('out', type=Path)
    ap.add_argument('--commit', required=True)
    ap.add_argument('--aragog', required=True, help='aragog commit of the runs')
    ap.add_argument('--calliope', required=True)
    ap.add_argument('starts', nargs='+', help='NAME=SEAL_RUN,OPEN_RUN')
    a = ap.parse_args(argv)
    rows = []
    for item in a.starts:
        name, pair = item.split('=', 1)
        seal_dir, open_dir = (Path(p) for p in pair.split(','))
        dts = superheat(seal_dir)
        assert superheat(open_dir) == dts, f'{name}: the 2 runs start from other superheats'
        s, o = read_run(seal_dir), read_run(open_dir)
        assert s.Time.iloc[-1] == o.Time.iloc[-1], f'{name}: the 2 runs end at other times'
        k = closing_row(s)
        rows.append(
            (name, repr(dts - BASE_SUPERHEAT), repr(float(s.Time[k])))
            + (repr(float(s.H2O_kg_atm.iloc[-1])), repr(float(o.H2O_kg_atm.iloc[-1])))
        )
    head = (
        '# Tutorial input/tutorials/tutorial_lid_seal.toml (docs/Tutorials/lid_seal.md), '
        f'PROTEUS commit {a.commit}, aragog commit {a.aragog}, CALLIOPE {a.calliope}.\n'
        '# The printed run and 5 starts whose planet.delta_T_super differs by dT [K]; '
        'each with the seal on and off.\n'
        '# Closing time [yr] of the seal run, H2O in the atmosphere at 140 kyr [kg] '
        'of the seal and the open run.\n'
        'run,dT,t_close,seal_atm,open_atm\n'
    )
    a.out.write_text(head + ''.join(','.join(r) + '\n' for r in rows))
    print('wrote', a.out, len(rows), 'starts')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
