"""Check two call logs for full agreement and list every differing argument and attribute.

usage: python verify.py calls_a.jsonl calls_b.jsonl
Prints the number of common records and structure solves, then each record whose
arguments, per-key or per-attribute digests, or return differ (first 40).
"""
from __future__ import annotations

import sys

from compare import _n, load

(A, _), (B, _) = load(sys.argv[1]), load(sys.argv[2])
common = sorted(s for s in set(A) & set(B) if 'out' in A[s] and 'out' in B[s])
solves = sum(A[s]['in']['fn'].endswith('.zalmoxis_solver') for s in common)
print(f'common completed records {len(common)}, structure solves {solves}')
ndiff = 0
for s in common:
    for ph in ('in', 'out'):
        a, b = A[s][ph], B[s][ph]
        if a['fn'] != b['fn']:
            print(f'seq {s}: call order differs: {a["fn"]} vs {b["fn"]}')
            sys.exit(1)
        bad = []
        for arg in a['args']:
            ka, kb = a['keys'].get(arg), b['keys'].get(arg)
            if ka is not None and kb is not None:
                bad += [f'{arg}.{k}' for k in ka if _n(ka[k]) != _n(kb.get(k))]
            elif _n(a['args'][arg]) != _n(b['args'].get(arg)):
                bad.append(arg)
        if ph == 'out':
            ra, rb = a.get('ret_keys'), b.get('ret_keys')
            if ra is not None and rb is not None:
                bad += [f'<ret>.{k}' for k in ra if k != 'total_time' and _n(ra[k]) != _n(rb.get(k))]
            elif _n(a['ret']) != _n(b['ret']):
                bad.append('<ret>')
        if bad:
            ndiff += 1
            if ndiff <= 40:
                print(f'seq {s} {ph} depth {a["depth"]} {a["fn"]}: {bad[:12]}')
hf = [s for s in common for ph in ('in', 'out')
      if any(_n(v) != _n(B[s][ph]['keys'].get('hf_row', {}).get(k)) for k, v in A[s][ph]['keys'].get('hf_row', {}).items())]
print(f'records with any difference: {ndiff}; records with an hf_row key difference: {len(set(hf))}')
