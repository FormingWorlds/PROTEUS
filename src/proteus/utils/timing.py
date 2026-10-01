"""Write ``<output>/timing.jsonl``, a record of where the wall time of a run goes.

Written only while a run started with ``PROTEUS_TIMING`` set is open; every
function here is a no-op otherwise. One JSON object per line, flushed as it is
written; the format is described at
https://github.com/egpbos/proteus-bench/blob/main/docs/interface.md
"""

from __future__ import annotations

import datetime as dt
import functools
import json
import os
from itertools import count
from pathlib import Path
from time import perf_counter

FILENAME = 'timing.jsonl'

# Config section naming the module behind each [IT_TIMING] bucket
_SECTION = {
    'atmos': 'atmos_clim',
    'chem': 'atmos_chem',
    'escape': 'escape',
    'interior': 'interior_energetics',
    'orbit': 'orbit',
    'outgas': 'outgas',
    'stellar': 'star',
    'structure': 'interior_struct',
}

_fh = None
_origin = 0.0
_ids = count(1)
_stack: list[tuple] = []  # open spans: (id, name, start, fields)
_submodule: dict[str, object] = {}


def _emit(**event) -> None:
    _fh.write(json.dumps({'v': 1, **event}) + '\n')
    _fh.flush()


def _open(name: str, fields: dict) -> None:
    _stack.append((next(_ids), name, perf_counter(), fields))


def _write(sid: int, name: str, start: float, fields: dict, ok: bool = True) -> float:
    dur = perf_counter() - start
    event = {
        'ev': 'span',
        'id': sid,
        'parent': _stack[-1][0] if _stack else None,
        'name': name,
        't0': round(start - _origin, 6),
        'dur': round(dur, 6),
        **{k: v for k, v in fields.items() if isinstance(v, (int, str))},
    }
    _emit(**event, **({} if ok else {'ok': False}))
    return dur


def _close(ok: bool = True) -> None:
    sid, name, start, fields = _stack.pop()
    _write(sid, name, start, fields, ok)


def start(output_dir, config) -> None:
    """Open the file, write ``run_start`` and begin the ``setup`` phase."""
    global _fh, _origin
    _fh = open(Path(output_dir) / FILENAME, 'w')
    _origin = perf_counter()
    _submodule.update({c: getattr(config, s).module for c, s in _SECTION.items()})
    wall = dt.datetime.now(dt.UTC).isoformat(timespec='milliseconds')
    _emit(ev='run_start', wall=wall.replace('+00:00', 'Z'), pid=os.getpid())
    mark('setup')


def mark(name: str, **fields) -> None:
    """Close the open phase, or for ``iter`` the open iteration, and open ``name``."""
    if _fh is None:
        return
    while len(_stack) > (name == 'iter'):
        _close()
    _open(name, fields)


def step(component: str, start_time: float) -> float:
    """Record a call to ``component`` begun at ``start_time``; return its duration [s]."""
    if _fh is None:
        return perf_counter() - start_time
    fields = {'component': component, 'submodule': _submodule.get(component)}
    return _write(next(_ids), component, start_time, fields)


def end(status: str, **fields) -> None:
    """Close every open span, write ``run_end`` and close the file."""
    global _fh
    if _fh is None:
        return
    while _stack:
        _close(ok=status == 'ok')
    _emit(ev='run_end', t0=round(perf_counter() - _origin, 6), status=status, **fields)
    _fh.close()
    _fh = None


def record_run(run):
    """End the timing record of ``run`` as ``ok``, ``error`` or ``interrupted``."""

    @functools.wraps(run)
    def wrapper(*args, **kwargs):
        try:
            result = run(*args, **kwargs)
        except KeyboardInterrupt:
            end('interrupted')
            raise
        except BaseException as exc:
            end('error', error=f'{type(exc).__name__}: {exc}')
            raise
        end('ok')
        return result

    return wrapper
