"""Run `proteus start` and log a digest of the inputs and outputs of every coupling call.

usage: python hashrun.py <config.toml> <calls.jsonl>
env:   STOP_AFTER_STRUCT=<n> exits after the n-th zalmoxis_solver call returns (0: no stop).
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import sys

import numpy as np

cfg, log = sys.argv[1], sys.argv[2]
STOP = int(os.environ.get('STOP_AFTER_STRUCT', '0'))
MODULES = [
    'proteus.interior_energetics.wrapper',
    'proteus.interior_energetics.aragog',
    'proteus.interior_energetics.aragog_jax',
    'proteus.atmos_clim',
    'proteus.interior_struct.zalmoxis',
    'proteus.outgas.wrapper',
    'proteus.outgas.calliope',
    'proteus.escape.wrapper',
    'proteus.atmos_clim.wrapper',
    'proteus.atmos_clim.agni',
    'proteus.star.wrapper',
    'proteus.orbit.wrapper',
]
SKIP = {'get_nlevb', 'update_gravity', 'readable_total'}
state = {'seq': 0, 'depth': 0, 'struct': 0}


def _h(x, depth=0):
    """Digest arrays, scalars and containers exactly; objects through their __dict__."""
    if isinstance(x, np.ndarray):
        if x.dtype.kind in 'fiucb':
            return hashlib.sha1(np.ascontiguousarray(x).tobytes() + str(x.shape).encode()).hexdigest()[:12]
        return _h(x.tolist(), depth)
    if isinstance(x, (bool, int, float, str, bytes, np.floating, np.integer, np.bool_)) or x is None:
        return repr(x)
    if depth >= 4:
        return f'<{type(x).__name__}>'
    if isinstance(x, dict):
        return hashlib.sha1(repr(sorted((str(k), _h(v, depth + 1)) for k, v in x.items())).encode()).hexdigest()[:12]
    if isinstance(x, (list, tuple, set, frozenset)):
        items = sorted(map(repr, x)) if isinstance(x, (set, frozenset)) else x
        return hashlib.sha1(repr([_h(v, depth + 1) for v in items]).encode()).hexdigest()[:12]
    if callable(x) and not hasattr(x, '__dict__'):
        return f'<{type(x).__name__}>'
    d = getattr(x, '__dict__', None)
    if d is not None and type(x).__module__.split('.')[0] in ('proteus', 'aragog', 'zalmoxis', 'calliope'):
        return 'obj:' + _h({k: v for k, v in d.items() if not callable(v)}, depth + 1)
    return f'<{type(x).__name__}>'


def _is_obj(v):
    return hasattr(v, '__dict__') and not isinstance(v, type) and not inspect.ismodule(v) and \
        type(v).__module__.split('.')[0] in ('proteus', 'aragog', 'zalmoxis', 'calliope')


def _snap(args, kw, names):
    """Digest each argument; a dict argument also gets a per-key digest."""
    rec, keys = {}, {}
    for name, v in list(zip(names, args)) + list(kw.items()):
        rec[name] = _h(v)
        if isinstance(v, dict) and len(v) > 0:
            keys[name] = {str(k): _h(val) for k, val in v.items()}
        elif _is_obj(v):
            keys[name] = {str(k): _h(val, 1) for k, val in vars(v).items() if not callable(val)}
    return rec, keys


def _write(rec):
    with open(log, 'a') as f:
        f.write(json.dumps(rec, default=str) + '\n')


def _wrap(fn, qual):
    try:
        names = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        names = []
    names += [f'arg{i}' for i in range(len(names), 64)]

    def inner(*args, **kw):
        state['seq'] += 1
        seq, depth = state['seq'], state['depth']
        a_in, k_in = _snap(args, kw, names)
        _write({'seq': seq, 'depth': depth, 'fn': qual, 'phase': 'in', 'args': a_in, 'keys': k_in})
        state['depth'] += 1
        try:
            r = fn(*args, **kw)
        finally:
            state['depth'] -= 1
        a_out, k_out = _snap(args, kw, names)
        rk = dict(r) if isinstance(r, dict) else dict(enumerate(r)) if isinstance(r, tuple) else None
        rk = {str(k): _h(v) for k, v in rk.items()} if rk is not None else None
        _write({'seq': seq, 'depth': depth, 'fn': qual, 'phase': 'out', 'ret': _h(r), 'ret_keys': rk,
                'args': a_out, 'keys': k_out})
        if qual.endswith('.zalmoxis_solver'):
            state['struct'] += 1
            if STOP and state['struct'] >= STOP:
                _write({'fn': 'stop', 'struct': state['struct']})
                os._exit(0)
        return r

    inner.__wrapped__ = fn
    return inner


import aragog  # noqa: E402

import proteus  # noqa: E402

if os.environ.get('PROBE_LOCAL'):
    # A source clone reports 0.0.0.dev0; aragog origin/main 287403d is tag 26.10.03.
    aragog.__version__ = '26.10.03'
    assert 'aragog-dev2-repro' in aragog.__file__ and 'PROTEUS-dev2-repro' in proteus.__file__

for mname in MODULES:
    mod = importlib.import_module(mname)
    for name, obj in list(vars(mod).items()):
        if inspect.isfunction(obj) and name not in SKIP and not name.startswith('__'):
            setattr(mod, name, _wrap(obj, f'{mname.split(".", 1)[1]}.{name}'))

_write({'fn': 'env', 'proteus': proteus.__file__, 'pid': os.getpid(), **{k: os.environ.get(k) for k in (
    'OMP_NUM_THREADS', 'JULIA_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'PYTHONHASHSEED',
    'PYTHONPATH')}})

if os.environ.get('SEED_CALLIOPE'):
    import proteus.outgas.calliope as _pc

    _ea = _pc.equilibrium_atmosphere

    def _seeded(*args, **kw):
        st = np.random.get_state()
        np.random.seed(int(os.environ['SEED_CALLIOPE']))
        try:
            return _ea(*args, **kw)
        finally:
            np.random.set_state(st)

    _pc.equilibrium_atmosphere = _seeded
    _write({'fn': 'seed_calliope', 'seed': os.environ['SEED_CALLIOPE']})

from proteus.cli import cli  # noqa: E402

sys.argv = ['proteus', 'start', '-c', cfg] + ([] if os.environ.get('PROBE_ONLINE') else ['--offline'])
sys.exit(cli())
