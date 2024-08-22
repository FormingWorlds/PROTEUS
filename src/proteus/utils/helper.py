# Small helper functions that can be used universally
# This file should not depend on too many other files, as this can cause circular import issues
from __future__ import annotations

import glob
import logging
import os
import re
import shutil

import numpy as np

log = logging.getLogger("PROTEUS")

def PrintSeparator():
    log.info("===================================================")
    pass

def PrintHalfSeparator():
    log.info("---------------------------------------------------")
    pass

# String sorting inspired by natsorted
def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

# Create a temporary folder
def create_tmp_folder():
    tmp_dir = "/tmp/proteus_%d/" % np.random.randint(int(1e12),int(1e13-1))
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir,ignore_errors=True)
    os.makedirs(tmp_dir)
    return tmp_dir

def safe_rm(fpath:str):
    '''
    Safely remove a file or folder
    '''

    if fpath=="":
        log.warning("Could not remove file at empty path")
        return

    fpath = os.path.abspath(fpath)
    if os.path.exists(fpath):
        if os.path.isfile(fpath):
            os.remove(fpath)

        elif os.path.isdir(fpath):
            subfolders = [ f.path.split("/")[-1] for f in os.scandir(fpath) if f.is_dir() ]
            if ".git" in subfolders:
                log.warning("Not emptying directory '%s' as it contains a Git repository"%fpath)
                return
            shutil.rmtree(fpath)

        else:
            log.warning("Cannot remove unhandled path '%s'"%fpath)

def CommentFromStatus(status:int):
    '''
    Convert status number into comment string
    '''
    desc = ""
    match status:
        # Running cases
        case 0:  desc = "Started"
        case 1:  desc = "Running"
        # Successful cases
        case 10: desc = "Completed (solidified)"
        case 11: desc = "Completed (steady-state)"
        case 12: desc = "Completed (maximum iterations)"
        case 13: desc = "Completed (target time)"
        case 14: desc = "Completed (net flux is small)"
        case 15: desc = "Completed (atmosphere escaped)"
        # Error cases
        case 20: desc = "Error (generic case, or configuration issue)"
        case 21: desc = "Error (Interior model)"
        case 22: desc = "Error (Atmosphere model)"
        case 23: desc = "Error (Stellar evolution model)"
        case 24: desc = "Error (Kinetics model)"
        case 25: desc = "Error (died, or exit requested by user)"
        # Default case
        case _:
            desc = "UNHANDLED STATUS (%d)" % status
            log.warning("Unhandled model status (%d) selected" % status)
    return desc

def UpdateStatusfile(dirs:dict, status:int):
    '''
    Update the status file with the current state of the program
    '''

    # Path to status file
    stsfold = os.path.abspath(dirs["output"])
    stsfile = os.path.join(stsfold,"status")

    # Does the folder exist?
    if not os.path.exists(stsfold):
        os.makedirs(stsfold)

    # Does the status file exist?
    safe_rm(stsfile)

    # Write status file
    with open(stsfile,'x') as hdl:
        hdl.write("%d\n" % status)
        desc = CommentFromStatus(status)
        hdl.write("%s\n"%desc)

def CleanDir(directory, keep_stdlog=False):
    """Clean a directory.

    Deletes a given directory and its contents, then creates it as empty.

    Parameters
    ----------
        directory : string
            Path to directory

    """

    def _check_safe(d):
        subfolders = [ f.path.split("/")[-1] for f in os.scandir(d) if f.is_dir() ]
        if ".git" in subfolders:
            raise Exception("Not emptying directory - it contains a Git repository!")

    # Simple case...
    if not keep_stdlog:
        if os.path.exists(directory):
            _check_safe(directory)
            shutil.rmtree(directory)
        os.makedirs(directory)
        return

    # Case where we want to keep log file...
    # If exists
    if os.path.exists(directory):
        for p in glob.glob(directory+"/*"):
            p = str(p)
            if os.path.isdir(p):
                # Remove folders
                _check_safe(p)
                shutil.rmtree(p)
            else:
                # Remove all files EXCEPT logfiles in topmost dir
                if ".log" not in p:
                    os.remove(p)
    else:
        os.makedirs(directory)


def find_nearest(array, target):
    """Find the element of an array that has a value closest to the target

    Parameters
    ----------
        array : list
            Array to search
        target : float
            Value to approximate

    Returns
    ----------
        close : float
            Element of array that is closest to target
        idx : int
            Index of closest element of array
    """
    array   = np.asarray(array)
    idx     = (np.abs(array - target)).argmin()
    close   = array[idx]
    return close, idx

def mol_to_ele(mol:str):
    '''
    Return the number of each element within a given molecule, as a dictionary
    '''
    decomp = re.findall(r'([A-Z][a-z]?)(\d*)', mol)   # https://codereview.stackexchange.com/a/232664
    elems = {}
    for ev in decomp:
        if ev[1] == '':
            val = 1
        else:
            val = int(ev[1])
        elems[str(ev[0])] = val
    return elems


def recursive_get(d, keys):
    '''
    Function to access nested dictionaries
    '''
    if len(keys) == 1:
        return d[keys[0]]
    return recursive_get(d[keys[0]], keys[1:])
