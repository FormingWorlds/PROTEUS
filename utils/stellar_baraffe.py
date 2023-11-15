# Util functions for wrapping Baraffe evolution tracks

from utils.modules_ext import *
from utils.constants import *
from utils.helper import find_nearest

def BaraffeStellarRadius(time_dict: dict, COUPLER_options: dict, track: dict):
    """Calculates the star's radius at a time t.

    Uses the Baraffe+15 tracks.

    Parameters
    ----------
        time_dict : dict
            Time dictionary, including star's age
        COUPLER_options : dict
            Dictionary of coupler options variables
        track : dict
            Baraffe evolution track

    Returns
    ----------
        Rstar : float
            Radius of star in units of solar radii

    """ 

    tstar = time_dict['star']

    # Get time and check that it is in range
    tmin = track['t'][0]
    if (tstar < tmin):
        print("WARNING: Star age too low! Clipping to %.1g Myr" % int(tmin*1.e-6))
        tstar = tmin
    tmax = track['t'][-1]
    if (tstar > tmax):
        print("WARNING: Star age too high! Clipping to %.1g Myr" % int(tmax*1.e-6))
        tstar = tmax

    # Find closest row in track
    iclose = (np.abs(track['t'] - tstar)).argmin()

    # Get data from track
    Rstar = track['Rstar'][iclose]

    return Rstar


def BaraffeSolarConstant(time_dict: dict, COUPLER_options: dict, track: dict):
    """Calculates the bolometric flux of the star at a previous time t. 

    Uses the Baraffe+15 tracks. Flux is scaled to the star-planet distance.

    Parameters
    ----------
        time_dict : dict
            Time dictionary, including star's age
        COUPLER_options : dict
            Dictionary of coupler options variables
        track : dict
            Baraffe evolution track

    Returns
    ----------
        inst : float
            Flux at planet's orbital separation (solar constant) in W/m^2
        heat : float
            Absorbed stellar flux (ASF) at TOA [W/m^2]

    """ 

    tstar = time_dict['star']

    # Get time and check that it is in range
    tmin = track['t'][0]
    if (tstar < tmin):
        print("WARNING: Star age too low! Clipping to %.1g Myr" % int(tmin*1.e-6))
        tstar = tmin
    tmax = track['t'][-1]
    if (tstar > tmax):
        print("WARNING: Star age too high! Clipping to %.1g Myr" % int(tmax*1.e-6))
        tstar = tmax

    # Find closest row in track
    iclose = (np.abs(track['t'] - tstar)).argmin()

    # Get data from track
    Lstar = track['Lstar'][iclose]
    Lstar *= L_sun
    mean_distance = COUPLER_options["mean_distance"] * AU

    inst = Lstar /  ( 4. * np.pi * mean_distance * mean_distance )
    heat = inst * ( 1. - COUPLER_options["albedo_pl"] ) * COUPLER_options["asf_scalefactor"]

    return inst, heat



def BaraffeLoadtrack(COUPLER_options: dict, pre_interp = True):
    """Load baraffe track into memory

    You can get other tracks from this file on their website: 
    http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/BHAC15_tracks+structure

    Parameters
    ----------
        COUPLER_options : dict
            Dictionary of coupler options variables
        pre_interp : bool
            Pre-interpolate the tracks onto a higher resolution time-grid

    Returns
    ----------
        track : dict
            Dictionary containing track data
    """

    # Load data
    file = COUPLER_options['star_btrack']
    data = np.loadtxt(file,skiprows=1).T

    # Parse data
    Mstar = data[0]
    t =     data[1]
    Teff =  data[2]
    Lstar = data[3]
    Rstar = data[5]

    # Convert appropriately
    if not pre_interp:

        track = {
            'name':     file,
            'Mstar':    Mstar,        # M_sun
            't':        10.0**t,      # yr
            'Teff':     Teff,         # K
            'Lstar':    10.0**Lstar,  # L_sun
            'Rstar':    Rstar         # R_sun
        }

    else:
        # Params for interpolation
        grid_count = 5e4

        # Do interpolation
        print("Interpolating stellar track onto a grid of size %d" % grid_count)
        interp_Teff =   PchipInterpolator(t,Teff)
        interp_Lstar =  PchipInterpolator(t,Lstar)
        interp_Rstar =  PchipInterpolator(t,Rstar)

        new_t = np.logspace( t[0] , t[-1] , int(grid_count))
        new_t = np.log10(new_t)
        new_Teff =  interp_Teff(new_t)
        new_Lstar = interp_Lstar(new_t)
        new_Rstar = interp_Rstar(new_t)

        track = {
            'name':     file,
            'Mstar':    Mstar,            # M_sun
            't':        10.0**new_t,      # yr
            'Teff':     new_Teff,         # K
            'Lstar':    10.0**new_Lstar,  # L_sun
            'Rstar':    new_Rstar         # R_sun
        }
        
    return track

def BaraffeSpectrumCalc(time_star: float, spec_fl: list, COUPLER_options: dict, track: dict):
    """Determine historical spectrum at time_star, using the baraffe tracks

    Uses a Baraffe evolution track. Calculates the spectrum both at 1 AU and 
    at the surface of the star.

    Parameters
    ----------
        time_star : float
            Stellar age [yr]
        spec_fl : list
            Modern spectrum flux array at 1 AU [erg s-1 cm-2 nm-1]
        COUPLER_options : dict
            Dictionary of coupler options variables
        track : dict
            Baraffe evolution track

    Returns
    ----------
        sflux : np.array(float)
            Numpy array of flux at 1 AU
        sfluxsurf : np.array(float)
            Numpy array of flux at stellar surface
    """

    # Get time and check that it is in range
    tstar = time_star * 1.e-6
    tmin = track['t'][0]*1.e-6
    if (tstar < tmin):
        print("WARNING: Star age too low! Clipping to %.1g Myr" % int(tmin))
        tstar = tmin
    tmax = track['t'][-1]*1.e-6
    if (tstar > tmax):
        print("WARNING: Star age too high! Clipping to %.1g Myr" % int(tmax))
        tstar = tmax

    # Find closest row in track
    iclose = (np.abs(track['t'] - time_star)).argmin()
    
    # Get data from track
    Rstar_cm =  track['Rstar'][iclose] * R_sun_cm
    Lstar =     track['Lstar'][iclose]

    # Get stellar radius and distance scale factor
    sf = (Rstar_cm / AU_cm) ** 2

    # Get luminosity scale factor
    Q_bol = Lstar / float(COUPLER_options['star_luminosity_modern'])
    if debug: print(Q_bol)

    # Calculate scaled spectrum
    hspec_fl = np.array(spec_fl) * Q_bol

    hspec_fl_surf = hspec_fl / sf

    return hspec_fl, hspec_fl_surf

def SpectrumWrite(time_dict, wl, sflux, sfluxsurf, folder, write_surf=True, write_1AU=True):
    """Write historical spectrum to files.

    Parameters
    ----------
        time_dict : dict
            Time dictionary, including stellar age
        wl : np.array(float)
            Numpy array of wavelengths
        sflux : np.array(float)
            Numpy array flux at 1 AU
        sfluxsurf : np.array(float)
            Numpy array of flux at stellar surface
        folder : float
            Path to folder where file is to be written

    """

    tstar = time_dict['star'] * 1.0e-6  # yr -> Myr

    if write_1AU:
        X = np.array([wl,sflux]).T
        outname1 = folder + "/%d.sflux" % time_dict['planet']
        header = '# WL(nm)\t Flux(ergs/cm**2/s/nm)          Stellar flux (1 AU) at t_star = %.3f Myr ' % round(tstar,3)
        np.savetxt(outname1, X, header=header,comments='',fmt='%1.4e',delimiter='\t')

    if write_surf:
        Y = np.array([wl,sfluxsurf]).T
        outname2 = folder + "/%d.sfluxsurf" % time_dict['planet']
        header = '# WL(nm)\t Flux(ergs/cm**2/s/nm)          Stellar flux (surface) at t_star = %.3f Myr ' % round(tstar,3)
        np.savetxt(outname2, Y, header=header,comments='',fmt='%1.4e',delimiter='\t')


# End of file
