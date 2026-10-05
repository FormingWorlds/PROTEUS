"""Align two call logs by sequence and name the first call whose inputs agree and outputs differ.

usage: python compare.py calls_a.jsonl calls_b.jsonl
"""
from __future__ import annotations

import json
import re
import sys


def load(p):
    recs, order = {}, []
    for line in open(p):
        r = json.loads(line)
        if 'seq' in r:
            recs.setdefault(r['seq'], {})[r['phase']] = r
            if r['phase'] == 'out':
                order.append(r['seq'])
    return recs, order


def _n(v):
    if not isinstance(v, str):
        return v
    return re.sub(r'proteus_\d+', 'proteus_TMP', re.sub(r'(repro|probe|phys)935_\w+?([/\'])', r'\g<1>935_RUN\2', v))


def diffargs(ra, rb, field='args'):
    out = []
    for k, va in ra[field].items():
        ka, kb = ra['keys'].get(k), rb['keys'].get(k)
        if ka is not None and kb is not None:
            if {x: _n(y) for x, y in ka.items()} != {x: _n(y) for x, y in kb.items()}:
                out.append(k)
        elif _n(va) != _n(rb[field].get(k)):
            out.append(k)
    return out


def main():
    SKIP_RET = set(sys.argv[3:])
    (A, order), (B, _) = load(sys.argv[1]), load(sys.argv[2])
    first_in = None
    for seq in sorted(set(A) & set(B)):
        if diffargs(A[seq]['in'], B[seq]['in']):
            a, b = A[seq]['in'], B[seq]['in']
            print(f'FIRST INPUT DIFF (call order) seq {seq} depth {a["depth"]} {a["fn"]}:', diffargs(a, b))
            return
    # Completion order: the first finished call with equal inputs and different outputs is the innermost one.
    for seq in [s for s in order if s in B]:
        a, b = A[seq], B[seq]
        if a['in']['fn'] != b['in']['fn']:
            print(f'seq {seq}: call order differs: {a["in"]["fn"]} vs {b["in"]["fn"]}')
            return
        fn = a['in']['fn']
        din = diffargs(a['in'], b['in'])
        if din and first_in is None:
            first_in = seq
            print(f'FIRST INPUT DIFF seq {seq} depth {a["in"]["depth"]} {fn}: args {din}')
            for k in din:
                if k in a['in']['keys']:
                    ka, kb = a['in']['keys'][k], b['in']['keys'][k]
                    print('   keys', [x for x in ka if _n(ka[x]) != _n(kb.get(x))][:20])
        if 'out' not in a or 'out' not in b:
            continue
        oa, ob = a['out'], b['out']
        dout = diffargs(oa, ob)
        ra, rb = oa.get('ret_keys'), ob.get('ret_keys')
        if ra is not None and rb is not None:
            rdiff = [k for k in ra if k != 'total_time' and _n(ra[k]) != _n(rb.get(k))]
            if rdiff:
                dout.append(f'<return keys {rdiff}>')
        elif _n(oa['ret']) != _n(ob['ret']) and fn not in SKIP_RET:
            dout.append('<return>')
        if dout and not din:
            print(f'FIRST SAME-IN/DIFF-OUT seq {seq} depth {a["in"]["depth"]} {fn}: {dout}')
            for k in dout:
                if k in oa['keys']:
                    ka, kb = oa['keys'][k], ob['keys'][k]
                    diff = [x for x in ka if _n(ka[x]) != _n(kb.get(x))]
                    print(f'   {k}: {len(diff)} keys differ:', diff[:30])
                    for x in diff[:12]:
                        print(f'      {x}: {ka[x]}  vs  {kb.get(x)}')
            return
    print('records', len(A), len(B))


if __name__ == "__main__":
    main()
