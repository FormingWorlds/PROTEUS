# Check the status of a PROTEUS parameter grid's cases
from __future__ import annotations

import glob
import logging
import os

import numpy as np

from proteus.utils.helper import CommentFromStatus

log = logging.getLogger('fwl.' + __name__)


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

    # Find folders
    pgrid_dir = os.path.abspath(pgrid_dir)
    case_dirs = sorted(glob.glob(pgrid_dir + '/case_*'))
    log.info("Found %d cases in '%s'", len(case_dirs), pgrid_dir)

    # Read each case's status code, keyed by the folder index parsed from
    # its name. Keying by the real index means a non-contiguous set of case
    # folders (for example after a failed case has been deleted) is handled
    # correctly, rather than assuming the folders are numbered 0..N-1.
    # Check `utils.helper.CommentFromStatus` for information on error codes.
    log.info('Checking statuses...')
    case_status = {}
    for case_dir in case_dirs:
        name = os.path.basename(case_dir)
        try:
            idx = int(name.rsplit('_', 1)[-1])
        except ValueError:
            log.warning("Ignoring folder with unexpected name '%s'", name)
            continue
        status_path = os.path.join(case_dir, 'status')
        if not os.path.exists(status_path):
            raise FileNotFoundError("Cannot find status file at '%s'" % status_path)
        with open(status_path, 'r') as hdl:
            lines = hdl.readlines()
        if not lines:
            raise ValueError("Status file is empty: '%s'" % status_path)
        case_status[idx] = int(lines[0].strip())

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
        if i == -1:
            comment = 'Uncategorised'
        else:
            comment = CommentFromStatus(i)
        pct = float(count) / N * 100.0
        log.info('  %-5d (%2d%%) %s', count, pct, comment)

    # Check options
    gen_cases = {
        # Broad categories
        'Running': list(range(0, 10, 1)),
        'Completed': list(range(10, 20, 1)),
        'Error': list(range(20, 30, 1)),
        'All': list(range(0, 100, 1)),
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
            for idx in indices:  # for each grid point
                for s in gen_cases[g]:  # for each case within this general case
                    if case_status[idx] == s:
                        e_any = True
                        log.info('  Case %-5d : Code %-2d - %s', idx, s, CommentFromStatus(s))
                        break
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
