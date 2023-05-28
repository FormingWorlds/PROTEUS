# Function to evolve the stellar spectrum over time using Baraffe evolution tracks

from utils.modules_ext import *
from utils.constants import *

def LoadBaraffeTrack(COUPLER_options: dict):
    """Load baraffe track into memory


    Parameters
    ----------
        COUPLER_options : dict
            Dictionary of coupler options variables

    Returns
    ----------
        track : dict
            Dictionary containing track data
    """

    file = COUPLER_options['star_btrack']
    data = np.loadtxt(file,skiprows=1).T

    track = {
        'name':     file,
        'Mstar':    data[0],            # M_sun
        't':        10.0**data[1],      # yr
        'Teff':     data[2],            # K
        'Lstar':    10.0**data[3],      # L_sun
        'Rstar':    data[5]             # R_sun
    }

    return track

def BaraffeSpectrumWrite(time_dict: dict, spec_wl: list, spec_fl: list, dirs : dict, COUPLER_options: dict, track: dict):
    """Write historical spectrum to disk, for a time t.

    Uses a Baraffe evolution track. Saves spectrum both at 1 AU and at the surface.

    Parameters
    ----------
        time_dict : dict
            Time dictionary, including stellar age
        spec_wl : list
            Modern spectrum wavelength array [nm]
        spec_fl : list
            Modern spectrum flux array at 1 AU [erg s-1 cm-2 nm-1]
        dirs : dict
            Directories dictionary
        COUPLER_options : dict
            Dictionary of coupler options variables
        track : dict
            Baraffe evolution track

    Returns
    ----------
        outname1 : str
            Path to historical spectrum file at 1 AU written by this function.
        outname1 : str
            Path to historical spectrum file at stellar surface written by this function.
    """

    # Get time and check that it is in range
    tstar = time_dict["star"] * 1.e-6
    tmin = track['t'][0]*1.e-6
    if (tstar < tmin):
        print("WARNING: Star age too low! Clipping to %.1g Myr" % int(tmin))
        tstar = tmin
    tmax = track['t'][-1]*1.e-6
    if (tstar > tmax):
        print("WARNING: Star age too high! Clipping to %.1g Myr" % int(tmax))
        tstar = tmax

    # Find closest row in track
    iclose = (np.abs(track['t'] - time_dict['star'])).argmin()
    
    # Get data from track
    t =         track['t'][iclose] * 1.e-6
    Rstar_cm =  track['Rstar'][iclose] * R_sun_cm
    Lstar =     track['Lstar'][iclose]

    # Get stellar radius and distance scale factor
    sf = (Rstar_cm / AU_cm) ** 2

    # Get luminosity scale factor
    Q_bol = Lstar / float(COUPLER_options['star_luminosity_modern'])
    print(Q_bol)

    # Calculate scaled spectrum
    hspec_fl = np.array(spec_wl) * Q_bol

    # Save historical spectrum at 1 AU
    X = np.array([spec_wl,hspec_fl]).T
    outname1 = dirs['output'] + "/%d.sflux" % time_dict['planet']
    header = '# MStellar flux (Baraffe, 1 AU) at t_star = %d Myr \n# WL(nm)\t Flux(ergs/cm**2/s/nm)' % t
    np.savetxt(outname1, X, header=header,comments='',fmt='%1.3e',delimiter='\t')

    # Save historical spectrum at stellar surface
    Y = np.array([spec_wl,hspec_fl / sf]).T
    outname2 = dirs['output'] + "/%d.sfluxsurf" % time_dict['planet']
    header = '# Stellar flux (Baraffe, surface) at t_star = %d Myr \n# WL(nm)\t Flux(ergs/cm**2/s/nm)' % t
    np.savetxt(outname2, Y, header=header,comments='',fmt='%1.3e',delimiter='\t')

    return outname1, outname2

# End of file
