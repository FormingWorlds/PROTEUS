"""Union of hf_row keys that differ anywhere between two logs, with the first seq each appears at."""
from __future__ import annotations

import sys

from compare import _n, load

(A, _), (B, _) = load(sys.argv[1]), load(sys.argv[2])
first = {}
common = sorted(s for s in set(A) & set(B) if 'out' in A[s] and 'out' in B[s])
for s in common:
    for ph in ('in', 'out'):
        ka, kb = A[s][ph]['keys'].get('hf_row', {}), B[s][ph]['keys'].get('hf_row', {})
        for k in ka:
            if _n(ka[k]) != _n(kb.get(k)) and k not in first:
                first[k] = (s, ph, A[s][ph]['fn'], ka[k], kb.get(k))
print('records', len(common), 'solves', sum(A[s]['in']['fn'].endswith('.zalmoxis_solver') for s in common))
for k, v in sorted(first.items(), key=lambda kv: kv[1][0]):
    print(k, v)
