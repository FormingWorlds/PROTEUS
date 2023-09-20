#!/usr/bin/env python3

"""
Code for running offline chemistry based on the results of a PROTEUS
grid. Depends on RunOfflineChemistry.py
"""

# Import libraries
import numpy as np
import threading, time, glob, os

# Import PROTEUS stuff
from utils.coupler import *
import tools.RunOfflineChemistry as ROC

# If run directly
if __name__ == '__main__':

    print("Start GridOfflineChemistry")

    # Parameters
    basedir =       "output/pspace_redox/"  # Folder containing GridPROTEUS output cases
    tsamples =      3                       # Number of time samples per grid point
    procs_max =     70                      # Maximum number of processes to run at once
    s_width =       1e9                     # Scale width of sampling distribution [yr]
    s_centre =      1e6                     # Centre of sampling distribution [yr]
    runtime_sleep = 40                      # Sleep seconds per iter (required for VULCAN warm up periods to pass)
    ini_method =    1                       # Method used to init VULCAN abundances  (0: const_mix, 1: eqm)

    # Locate output folders
    folders = glob.glob(basedir+"/case_*")
    configs = [ os.path.abspath(f+"/init_coupler.cfg") for f in folders ]
    npoints = len(configs)

    # Validate inputs
    if tsamples < 1:
        raise Exception("Too few time samples per grid point") 
    tprocs = tsamples
    procs_tot = npoints * tprocs 
    while procs_tot > procs_max:
        tprocs -= 1
        if tprocs == 0:
            raise Exception("Too many grid points to satisfy procs_max")
        procs_tot = npoints * tprocs 
    print("We will be running up to %d simultaneous processes. Check that this is what you want." % procs_tot)
    
    print("Sleeping 5 seconds...")
    time.sleep(5)

    # Prepare processes as thread objects
    # They are 'parents' of VULCAN processes but 'children' of this process
    # If this processes, dies then these will die too. This process MUST
    # stay alive until the end in order to get any results from 
    # the calculations made by VULCAN.
    pool = []
    for i in range(npoints):
        fr = bool(i == 0)
        cfgfile = configs[i]

        pool.append(threading.Thread(target =   ROC.parent, 
                                     args =     (cfgfile, tsamples, tprocs, s_width, s_centre),
                                     kwargs =   {"mkfuncs":fr, "runtime_sleep":runtime_sleep, "ini_method":ini_method, "mkplots":False}
                                     )
                    )
    
    # Spawn each process in a new thread, making them a child of this process.
    # They need to stay alive to watch for (grand)child process completion.
    for p in pool:
        p.start()
        
    # Wait for processes to finish
    for i in range(npoints):
        pool[i].join()

    # When this script ends it means that all of the parent threads are complete.
    # This *should* mean that everything ran smoothly and has completed.
    # Be careful to check for orphaned processes.
    print("Exit GridOfflineChemistry")

