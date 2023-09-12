# Small helper functions that can be used universally

from utils.modules_ext import *
from utils.constants import *

def PrintSeparator():
    print("=============================================================================================================")
    pass

def PrintHalfSeparator():
    print("--------------------------------------------------")
    pass

# String sorting inspired by natsorted
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def CleanDir(dir):
    """Clean a directory.

    Deletes a given directory and its contents, then creates it as empty.

    Parameters
    ----------
        dir : string
            Path to directory

    """
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

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
