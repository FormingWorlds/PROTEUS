# Small helper functions that can be used universally
# This file should not depend on too many other files, as this can cause circular import issues

import numpy as np
import os, shutil, re, glob, logging
from utils.constants import *

log = logging.getLogger(__name__)

def PrintSeparator():
    log.info("=============================================================================================================")
    pass

def PrintHalfSeparator():
    log.info("--------------------------------------------------")
    pass

# String sorting inspired by natsorted
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

# Get comment from status
def CommentFromStatus(status:int):
    desc = ""
    match status:
        # Running cases
        case 0:  desc = "Started"
        case 1:  desc = "Running"
        # Successful cases
        case 10: desc = "Completed (solidified)"
        case 11: desc = "Completed (steady-state)"
        case 12: desc = "Completed (maximum iterations)"
        # Error cases
        case 20: desc = "Error (generic/configuration)"
        case 21: desc = "Error (SPIDER)"
        case 22: desc = "Error (AGNI)"
        case 23: desc = "Error (AEOLUS)"
        case 24: desc = "Error (VULCAN)"
        # Default case
        case _:
            desc = "UNHANDLED STATUS"
            log.warning("Unhandled model status selected")
    return desc

# Update the status file with the current state of the program
def UpdateStatusfile(dirs:dict, status:int):
    # Path to status file 
    stsfold = os.path.abspath(dirs["output"])
    stsfile = os.path.join(stsfold,"status")

    # Does the folder exist?
    if not os.path.exists(stsfold):
        os.makedirs(stsfold)

    # Does the status file exist?
    if os.path.exists(stsfile):
        os.remove(stsfile)

    # Write status file
    with open(stsfile,'x') as hdl:
        hdl.write("%d\n" % status)
        desc = CommentFromStatus(status)
        hdl.write("%s\n"%desc)
        # hdl.flush()
        # os.fsync(hdl.fileno())

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
                # Remove all files EXCEPT std.log in topmost dir
                if not ("std.log" in p):
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

# Return the number of each element within a given molecule, as a dictionary
def mol_to_ele(mol:str):
    decomp = re.findall(r'([A-Z][a-z]?)(\d*)', mol)   # https://codereview.stackexchange.com/a/232664
    elems = {}
    for ev in decomp:
        if ev[1] == '':
            val = 1
        else:
            val = int(ev[1])
        elems[str(ev[0])] = val
    return elems


#====================================================================
def find_xx_for_yy( xx, yy, yywant ):

    a = yy - yywant

    s = sign_change( a )

    # for ease, just add zero at the beginning to enable us to
    # have the same length array.  Could equally add to the end, or
    # interpolate

    s = np.insert(s,0,0)

    result = xx * s

    return result

#====================================================================
def get_first_non_zero_index( myList ):

    # https://stackoverflow.com/questions/19502378/python-find-first-instance-of-non-zero-number-in-list

    index = next((i for i, x in enumerate(myList) if x), None)

    return index

#====================================================================
def sign_change( a ):

    s = (np.diff(np.sign(a)) != 0)*1

    return s

#====================================================================
def recursive_get(d, keys):

    '''function to access nested dictionaries'''

    if len(keys) == 1:
        return d[keys[0]]
    return recursive_get(d[keys[0]], keys[1:])
