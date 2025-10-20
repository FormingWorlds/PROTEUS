# Check the status of a PROTEUS parameter grid's cases
from __future__ import annotations

import glob
import os

import numpy as np

from proteus.utils.helper import CommentFromStatus


def summarise(pgrid_dir:str, tgt_status:str=None):
    '''
    Summarise current status of grid.

    Parameters
    -------------
    * `pgrid_dir`   path to grid folder.
    * `tgt_status`  optional; print case numbers of all runs which have this status.
    '''
    if (not os.path.exists(pgrid_dir)) or (not os.path.isdir(pgrid_dir)):
        raise FileNotFoundError("Invalid path '%s'" % pgrid_dir)

    # Find folders
    pgrid_dir = os.path.abspath(pgrid_dir)
    case_dirs = glob.glob(pgrid_dir + "/case_*")
    N = len(case_dirs)
    print("Found %d cases in '%s'" % (N,pgrid_dir))

    # Statuses
    # Check `utils.helper.CommentFromStatus` for information on error codes
    print("Checking statuses...")
    status = np.full(N, -1, dtype=int)
    cmmnts = np.full(N, "", dtype=str)
    for i in range(N):
        status_path = os.path.join(pgrid_dir, "case_%06d"%i, "status")
        if not os.path.exists(status_path):
            raise FileNotFoundError("Cannot find status file at '%s'" % status_path)
        with open(status_path,'r') as hdl:
            lines = hdl.readlines()
        status[i] = int(lines[0])
        cmmnts[i] = str(lines[1])

    # Statistics
    print("Statistics:")
    for i in range(-1,100):
        count = np.count_nonzero(status == i)
        if count == 0:
            continue
        if i == -1:
            comment = "Uncategorised"
        else:
            comment = CommentFromStatus(i)
        pct = float(count)/N*100.0
        print("  %-5d (%2d%%) %s" % (count,pct,comment))

    # Check options
    gen_cases = {
        # Broad categories
        "Running":   list(range(0,  10,  1)),
        "Completed": list(range(10, 20,  1)),
        "Error":     list(range(20, 30,  1)),
        "All":       list(range(0,  100, 1)),
        # Narrower categories
        "Solidified":    [10],
        "Steady":        [11, 14],
        "Escaped":       [15],
        "Disintegrated": [16],
    }

    # sanitise input
    if not tgt_status:
        return True
    tgt_status = str(tgt_status).strip().lower()
    if (tgt_status=="complete"):
            tgt_status = "completed"

    matched = False

    # general cases
    for g in gen_cases.keys():  # for each general case
        if tgt_status == g.lower():
            matched = True
            print("%s cases:" % g)
            e_any = False
            for i in range(N):  # for each grid point
                for s in gen_cases[g]:  # for each case within this general case
                    if status[i] == s:
                        e_any = True
                        print("  Case %-5d : Code %-2d - %s" % (i,s, CommentFromStatus(s)))
                        break
            if not e_any:
                print("  (None)")

    # code cases
    tgt_status = tgt_status.replace("status=", "code=")
    if "code" in tgt_status:
        matched = True
        code = int(tgt_status.replace(" ","").split("=")[-1])
        print("Code %d cases:" % code)
        e_any = False
        for i in range(N):
            if status[i] == code:
                e_any = True
                print("  Case %-5d : Code %-2d - %s" % (i,code, CommentFromStatus(code)))
        if not e_any:
            print("  (None)")

    if not matched:
        print("Invalid status category '%s'" % tgt_status)
        print("Run `proteus grid-summarise --help` for info on using this command")

    return matched
