"""Write the stored rows of the energy-ledger validation cases from their source runs.

Usage: ``python tests/data/energy_ledger/build_cases.py CASE RUN START END CORE_BC [NOTE]``

Writes ``tests/data/energy_ledger/<CASE>.json``: one line of provenance (source helpfile and its
sha256, the PROTEUS git hash and module versions from the run log, the rows), one line of settings
and one line of the columns that ``proteus.utils.energy_ledger.whole_run_ledger`` reads, rows START
to END, with each float as written in the source helpfile.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

from proteus.utils.coupler import read_helpfile_table
from proteus.utils.energy_ledger import _COMMON, _CORE, _OPTIONAL


def main(case, run, start, end, core_bc, note=''):
    run, start, end = Path(run), int(start), int(end)
    src = run / 'runtime_helpfile.csv'
    hf = read_helpfile_table(src)
    log = (run / 'proteus_00.log').read_text(errors='replace')
    found = lambda pat: (re.search(pat, log) or [None, None])[1]  # noqa: E731
    cfg = tomllib.loads((run / 'init_coupler.toml').read_text())
    names = [c for c in _COMMON + _CORE[core_bc] + _OPTIONAL if c in hf.columns]
    prov = {
        'source': str(src),
        'sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
        'proteus_git_hash': found(r'PROTEUS git hash\s+(\S+)'),
        'aragog_version': found(r'aragog version\s+(\S+)'),
        'rows': [start, end],
        'note': note,
    }
    settings = {
        'core_bc': core_bc,
        'core_tfac': float(cfg.get('interior_energetics', {}).get('core_tfac_avg', 1.147)),
    }
    cols = {c: hf[c].iloc[start : end + 1].tolist() for c in names}
    out = Path(__file__).parent / f'{case}.json'
    lines = [
        f'"{k}": {json.dumps(v)}'
        for k, v in (('provenance', prov), ('settings', settings), ('columns', cols))
    ]
    out.write_text('{\n' + ',\n'.join(lines) + '\n}\n')
    print(out, out.stat().st_size, 'bytes,', len(names), 'columns,', end - start + 1, 'rows')


if __name__ == '__main__':
    main(*sys.argv[1:])
