# Check the status of a PROTEUS parameter grid's cases
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from proteus.utils.helper import CommentFromStatus, ReadStatus

log = logging.getLogger('fwl.' + __name__)

# Broad status categories of the codes in `CommentFromStatus` (proteus.utils.helper),
# shared by `summarise` and the grid analysis. Code 29 is 'Completed (planet evaporated)'.
STATUS_CATEGORIES = {
    'Running': tuple(range(0, 10)),
    'Completed': (*range(10, 20), 29),
    'Error': tuple(range(20, 29)),
}


def status_category(code: int) -> str:
    """Return the broad category of a status code.

    Parameters
    ----------
    code : int
        A PROTEUS status code (see ``CommentFromStatus``).

    Returns
    -------
    str
        'Running' (0-9), 'Completed' (10-19 and 29), 'Error' (20-28), or
        'Unknown' for any other code.
    """
    for name, codes in STATUS_CATEGORIES.items():
        if code in codes:
            return name
    return 'Unknown'


def summarise(pgrid_dir: str, tgt_status: str = None):
    """
    Summarise current status of grid.

    Parameters
    -------------
    * `pgrid_dir`   path to grid folder.
    * `tgt_status`  optional; print case numbers of all runs which have this status.
    """
    if (not os.path.exists(pgrid_dir)) or (not os.path.isdir(pgrid_dir)):
        raise FileNotFoundError("Invalid path '%s'" % pgrid_dir)

    # Find the case folders, and the cases with a config but no folder (not started yet)
    grid = Path(pgrid_dir).resolve()
    names = {p.name for p in grid.glob('case_*')}
    not_started = {p.stem for p in grid.glob('cfgs/case_*.toml')} - names
    case_dirs = [grid / name for name in sorted(names | not_started)]
    log.info("Found %d cases in '%s'", len(case_dirs), grid)
    if not_started:
        log.info('%d cases have a config in cfgs/ but no folder: not started', len(not_started))

    # Read each case's status code, keyed by the folder index parsed from
    # its name. Keying by the real index means a non-contiguous set of case
    # folders (for example after a failed case has been deleted) is handled
    # correctly, rather than assuming the folders are numbered 0..N-1.
    # Check `utils.helper.CommentFromStatus` for information on error codes. A case
    # without a readable status file counts as unknown (STATUS_MISSING).
    log.info('Checking statuses...')
    case_status = {}
    for case_dir in case_dirs:
        name = case_dir.name
        try:
            idx = int(name.rsplit('_', 1)[-1])
        except ValueError:
            log.warning("Ignoring folder with unexpected name '%s'", name)
            continue
        case_status[idx] = ReadStatus(case_dir)

    # Sorted indices and the matching array of status codes
    indices = sorted(case_status.keys())
    codes = np.array([case_status[i] for i in indices], dtype=int)
    N = len(indices)

    # Statistics
    log.info('Statistics:')
    for i in range(-1, 100):
        count = int(np.count_nonzero(codes == i))
        if count == 0:
            continue
        pct = float(count) / N * 100.0
        log.info('  %-5d (%2d%%) %s', count, pct, CommentFromStatus(i))

    # Check options
    gen_cases = {
        # Broad categories (shared ranges)
        **{name: list(codes) for name, codes in STATUS_CATEGORIES.items()},
        'All': list(range(-1, 100)),
        'Unknown': [c for c in range(-1, 100) if status_category(c) == 'Unknown'],
        # Narrower categories
        'Solidified': [10],
        'Steady': [11, 14],
        'Escaped': [15],
        'Disintegrated': [16],
    }

    # sanitise input
    if not tgt_status:
        return True
    tgt_status = str(tgt_status).strip().lower()
    if tgt_status == 'complete':
        tgt_status = 'completed'

    matched = False

    # general cases
    for g in gen_cases.keys():  # for each general case
        if tgt_status == g.lower():
            matched = True
            log.info('%s cases:', g)
            e_any = False
            for idx in indices:
                s = case_status[idx]
                if s in gen_cases[g]:
                    e_any = True
                    log.info('  Case %-5d : Code %-2d - %s', idx, s, CommentFromStatus(s))
            if not e_any:
                log.info('  (None)')

    # code cases
    tgt_status = tgt_status.replace('status=', 'code=')
    if 'code' in tgt_status:
        matched = True
        code = int(tgt_status.replace(' ', '').split('=')[-1])
        log.info('Code %d cases:', code)
        e_any = False
        for idx in indices:
            if case_status[idx] == code:
                e_any = True
                log.info('  Case %-5d : Code %-2d - %s', idx, code, CommentFromStatus(code))
        if not e_any:
            log.info('  (None)')

    if not matched:
        log.warning("Invalid status category '%s'", tgt_status)
        log.info('Run `proteus grid-summarise --help` for info on using this command')

    return matched
