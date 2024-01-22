# Util functions for stellar evolution

from utils.modules_ext import *
from utils.constants import *
from utils.helper import find_nearest, UpdateStatusfile

def ModernSpectrumLoad(dirs: dict, COUPLER_options: dict):
    """Copy file and load modern spectrum into memory.

    Scaled to 1 AU from the star. Generate these spectra using the python script
    'GetStellarSpectrum.py' in the 'tools' directory.

    Parameters
    ----------
        dirs : dict
            Directories dictionary
        COUPLER_options : dict
            Dictionary of coupler options variables

    Returns
    ----------
        spec_wl : np.array[float]
            Wavelength [nm]
        spec_fl : np.array[float]
            Flux [erg s-1 cm-2 nm-1]
    """

    spec_file = dirs["coupler"]+"/"+COUPLER_options["star_spectrum"]
    if os.path.isfile(spec_file):

        # Copy modern spectrum file to output directory as -1.sflux.
        copy_file = shutil.copyfile(spec_file,dirs['output']+'/-1.sflux') 

        # Load file
        spec_data = np.loadtxt(copy_file, skiprows=2,delimiter='\t').T
        spec_wl = spec_data[0]
        spec_fl = spec_data[1]
    else:
        UpdateStatusfile(dirs, 20)
        raise Exception("Cannot find stellar spectrum")
    

    binwidth_wl = spec_wl[1:] - spec_wl[0:-1]

    if debug:
        print("Modern spectrum statistics:")
        print("\t Flux \n\t\t (min, max) = (%1.2e, %1.2e) erg s-1 cm-2 nm-1" % (np.amin(spec_fl),np.amax(spec_fl)))
        print("\t Wavelength \n\t\t (min, max) = (%1.2e, %1.2e) nm" % (np.amin(spec_wl),np.amax(spec_wl)))
        print("\t Bin width \n\t\t (min, median, max) = (%1.2e, %1.2e, %1.2e) nm" % (np.amin(binwidth_wl),np.median(binwidth_wl),np.amax(binwidth_wl)))
        
    
    return spec_wl, spec_fl
