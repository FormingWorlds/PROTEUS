# Small helper functions that can be used universally

from utils.modules_ext import *
from utils.constants import *

def PrintSeparator():
    print("-------------------------------------------------------------------------------------------------------------")
    pass

def PrintHalfSeparator():
    print("--------------------------------------------------")
    pass

# String sorting not based on natsorted package
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

def CleanOutputDir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def gravity( m, r ):
    g = phys.G*m/r**2
    return g

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
