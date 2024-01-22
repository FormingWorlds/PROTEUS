#!/usr/bin/env python3

# Check the status of a PROTEUS parameter grid's cases

import sys, os, glob
import numpy as np
from utils.helper import CommentFromStatus

def summarise(pgrid_dir:str):
    if (not os.path.exists(pgrid_dir)) or (not os.path.isdir(pgrid_dir)):
        raise Exception("Invalid path '%s'" % pgrid_dir)
    
    pgrid_dir = os.path.abspath(pgrid_dir)
    print("Pgrid folder '%s'" % pgrid_dir)

    # Find folders
    case_dirs = glob.glob(pgrid_dir + "/case_*")
    N = len(case_dirs)
    print("Found %d cases" % N)

    # Statuses
    status = np.full(N, -1, dtype=int)
    cmmnts = np.full(N, "", dtype=str)
    for i in range(N):
        with open(os.path.join(pgrid_dir, "case_%05d"%i, "status")) as hdl:
            lines = hdl.readlines()
        status[i] = int(lines[0])
        cmmnts[i] = str(lines[1])
    
    # Statistics
    print("Statistics:")
    print("  %-4d       Uncategorised" % np.count_nonzero(status == -1))
    for i in range(100):
        count = np.count_nonzero(status == i)
        if count == 0:
            continue 
        comment = CommentFromStatus(i)
        pct = float(count)/N*100.0
        print("  %-4d (%2d%%) %s" % (count,pct,comment))

    # Error cases
    print("Error cases:")
    error_any = False
    for i in range(N):
        for s in range(20, 30, 1):
            if status[i] == s:
                error_any = True
                print("  Case %d - %s" % (i,CommentFromStatus(s)))
    if not error_any:
        print("  (None)")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        fold = sys.argv[1]
    else:
        raise Exception("Pgrid folder path not provided")
    summarise(fold)
