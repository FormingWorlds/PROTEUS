#!/usr/bin/env python3

# Check the status of a PROTEUS parameter grid's cases

import sys, os, glob
import numpy as np
from utils.helper import CommentFromStatus

def summarise(pgrid_dir:str, opts:list):
    if (not os.path.exists(pgrid_dir)) or (not os.path.isdir(pgrid_dir)):
        raise Exception("Invalid path '%s'" % pgrid_dir)
    

    # Find folders
    pgrid_dir = os.path.abspath(pgrid_dir)
    case_dirs = glob.glob(pgrid_dir + "/case_*")
    N = len(case_dirs)
    print("Found %d cases in '%s'" % (N,pgrid_dir))

    # Statuses
    # Check `utils.helper.CommentFromStatus` for information on error codes
    status = np.full(N, -1, dtype=int)
    cmmnts = np.full(N, "", dtype=str)
    for i in range(N):
        status_path = os.path.join(pgrid_dir, "case_%05d"%i, "status")
        if not os.path.exists(status_path):
            raise Exception("Cannot find status file at '%s'" % status_path)
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
        print("  %-4d (%2d%%) %s" % (count,pct,comment))

    # Check options
    gen_cases = {
        # Broad categories
        "Running":   list(range(0,  10, 1)),
        "Completed": list(range(10, 20, 1)),
        "Error":     list(range(20, 30, 1)),
        "All":       list(range(0,100,1)),
        # Narrower categories
        "Solidified": [10],
        "Steady":     [11]
    }
    for o in opts:
        o = str(o).lower()
        matched = False

        # general cases
        for g in gen_cases.keys():  # for each general case
            if o == g.lower():
                matched = True
                print("%s cases:" % g)
                e_any = False
                for i in range(N):  # for each grid point
                    for s in gen_cases[g]:  # for each case within this general case
                        if status[i] == s:
                            e_any = True
                            print("  Case %-3d - %s" % (i,CommentFromStatus(s)))
                if not e_any:
                    print("  (None)")

        # code cases
        if "code" in o:
            matched = True
            code = int(o.replace(" ","").split("=")[-1])
            print("Code %d cases:" % code)
            e_any = False
            for i in range(N):
                if status[i] == code:
                    e_any = True
                    print("  Case %-3d - %s" % (i,CommentFromStatus(code)))
            if not e_any:
                print("  (None)")

        if not matched:
            print("Invalid status category '%s'" % o)

def print_help():
   
    print("Command usage: GridSummarise.py [fold] (opt1) (opt2) (opt3) ...")
    print("    [fold] = path to Pgrid output folder, required")
    print("    [optN] = status categories to print, optional")
    print("             'completed', 'running', 'error', or 'code=[c]' for some error code [c]")
    exit(1)

if __name__ == "__main__":

    # Check syntax
    if (len(sys.argv) < 2):
        print("Invalid number of arguments!")
        print_help()
        exit(1)
    if  (len(sys.argv) == 2) and (sys.argv[1].strip().lower() == "help"):
        print_help()
        exit(0)
    
    # Grid folder 
    fold = sys.argv[1]

    # Extra requested status categories
    opts = []  
    if len(sys.argv) > 2:
        for o in sys.argv[2:]:
            opts.append(str(o))

    # Get the summary
    summarise(fold, opts)
    exit(0)
