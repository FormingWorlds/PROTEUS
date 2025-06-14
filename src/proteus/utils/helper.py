# Small helper functions that can be used universally
# This file should not depend on too many other files, as this can cause circular import issues
from __future__ import annotations

import glob
import logging
import os
import re
import shutil

import numpy as np

log = logging.getLogger("fwl."+__name__)

def get_proteus_dir():
    '''
    Get absolute path to PROTEUS directory.

    This should be a directory containing `pyproject.toml`.
    '''

    # Assuming that this file is in `PROTEUS/src/proteus/utils/`
    utils = os.path.dirname(os.path.abspath(__file__))

    # Work upwards from utils
    root = os.path.abspath(os.path.join(utils,"..","..",".."))

    # Check that this path is reasonable
    if "pyproject.toml" not in os.listdir(root):
        raise EnvironmentError(f"Cannot locate PROTEUS directory. Tried '{root}' ")

    return root

def PrintSeparator():
    log.info("===================================================")
    pass

def PrintHalfSeparator():
    log.info("---------------------------------------------------")
    pass

def multiple(a:int,b:int) -> bool:
    '''
    Return true if a is an integer multiple of b. Otherwise, return false.

    This is a more robust version of the modulo operator, which can fail if a or b are None, or if b is 0.
    '''
    if (a is None) or (b is None) or (b == 0):
        return False
    else:
        return bool(a%b == 0)

def mol_to_ele(mol:str):
    '''
    Return the number of atoms of each element in a given molecule, as a dictionary

    https://codereview.stackexchange.com/a/232664
    '''

    # Validate
    if not str(mol[0]).isupper:
        raise ValueError(f"Molecule name '{mol}' is invalid")

    # Get atoms
    decomp = re.findall(r'([A-Z][a-z]?)(\d*)', mol)
    elems = {}
    for ev in decomp:
        if ev[1] == '':
            val = 1
        else:
            val = int(ev[1])
        elems[str(ev[0])] = val

    # Check that what we got is reasonable
    if not elems:
        raise ValueError(f"Could not decompose molecule '{mol}'")

    return elems

# String sorting inspired by natsorted
def natural_sort(lst):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key):
        return [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(lst, key = alphanum_key)

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
        case 0:
            desc = "Started"
        case 1:
            desc = "Running"
        # Successful cases
        case 10:
            desc = "Completed (solidified)"
        case 11:
            desc = "UNUSED_STATUS_CODE (11)"
        case 12:
            desc = "Completed (maximum iterations)"
        case 13:
            desc = "Completed (target time)"
        case 14:
            desc = "Completed (net flux is small)"
        case 15:
            desc = "Completed (volatiles escaped)"
        # Error cases
        case 20:
            desc = "Error (generic case, or configuration issue)"
        case 21:
            desc = "Error (Interior model)"
        case 22:
            desc = "Error (Atmosphere model)"
        case 23:
            desc = "Error (Stellar evolution model)"
        case 24:
            desc = "Error (Kinetics model)"
        case 25:
            desc = "Error (died, or exit requested by user)"
        case 26:
            desc = "Error (Tides/orbit model)"
        case 27:
            desc = "Error (Outgassing model)"
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

def recursive_get(d, keys):
    '''
    Function to access nested dictionaries
    '''
    if len(keys) == 1:
        return d[keys[0]]
    return recursive_get(d[keys[0]], keys[1:])
