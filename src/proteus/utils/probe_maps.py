"""Probe of the memory maps across structure re-solves (probe branch, not for merge)."""

from __future__ import annotations

import logging
import os
import resource
import time

log = logging.getLogger('fwl.' + __name__)
_LAST = [time.perf_counter()]


def _count_maps() -> int:
    """Lines of /proc/self/maps (Linux), -1 elsewhere."""
    try:
        with open('/proc/self/maps') as fh:
            return sum(1 for _ in fh)
    except OSError:
        return -1


def _rss_kb() -> int:
    """Current resident set size [kB] from /proc/self/status, else the peak from getrusage."""
    try:
        with open('/proc/self/status') as fh:
            for line in fh:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1])
    except OSError:
        pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)


def log_maps(where: str) -> None:
    """Log the map count, the RSS and the wall time since the last probe line."""
    now = time.perf_counter()
    log.info(
        'PROBE maps=%d rss_kB=%d where=%s dwall=%.1f s',
        _count_maps(),
        _rss_kb(),
        where,
        now - _LAST[0],
    )
    _LAST[0] = now


def clear_jax_after_resolve() -> None:
    """With PROTEUS_PROBE_CLEAR_JAX=1, drop every JAX compile cache after a re-solve."""
    if os.environ.get('PROTEUS_PROBE_CLEAR_JAX') != '1':
        return
    import jax

    t0 = time.perf_counter()
    jax.clear_caches()
    log.info(
        'PROBE clear_caches %.3f s; maps after %d', time.perf_counter() - t0, _count_maps()
    )
