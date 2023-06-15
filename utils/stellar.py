# Util functions for wrapping Mors XUV evolution library

from utils.modules_ext import *
from utils.constants import *
import Mors as mors

# Spectral bands for stellar fluxes, in nm
star_bands = {
    "xr" : [1.e-3 , 10.0],  # X-ray,  defined by mors
    "e1" : [10.0  , 32.0],  # EUV1,   defined by mors
    "e2" : [32.0  , 92.0],  # EUV2,   defined by mors
    "uv" : [92.0  , 364.5], # UV,     defined by me
    "pl" : [364.5 , 1.e9],  # planck, defined by me
    'bo' : [1.e-3 , 1.e9]   # bolo,   all wavelengths
}

def MorsSolarConstant(time_dict: dict, COUPLER_options: dict):
    """Calculates the bolometric flux of the star at a previous time t. 

    Uses the Mors module, which reads stellar evolution tracks from 
    Spada et al. (2013). Flux is scaled to the star-planet distance.

    Parameters
    ----------
        time_dict : dict
            Time dictionary, including star's age
        COUPLER_options : dict
            Dictionary of coupler options variables

    Returns
    ----------
        flux : float
            Flux at planet's orbital separation (solar constant) in W/m^2
        heat : float
            Heating rate at TOA in W/m^2

    """ 

    Mstar = COUPLER_options["star_mass"]
    tstar = time_dict['star'] * 1.e-6  # Convert from yr to Myr

    Lstar = mors.Value(Mstar, tstar, 'Lbol')  # Units of L_sun
    Lstar *= L_sun # Convert to W

    mean_distance = COUPLER_options["mean_distance"] * AU

    flux = Lstar /  ( 4. * np.pi * mean_distance * mean_distance )
    heat = flux * ( 1. - COUPLER_options["albedo_pl"] )

    return flux, heat

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
        flux : float
            Flux at planet's orbital separation (solar constant) in W/m^2
        heat : float
            Heating rate at TOA in W/m^2

    """ 

    tstar = time_dict['star'] * 1.e-6  # Convert from yr to Myr

    # Get time and check that it is in range
    tmin = track['t'][0]*1.e-6
    if (tstar < tmin):
        print("WARNING: Star age too low! Clipping to %.1g Myr" % int(tmin))
        tstar = tmin
    tmax = track['t'][-1]*1.e-6
    if (tstar > tmax):
        print("WARNING: Star age too high! Clipping to %.1g Myr" % int(tmax))
        tstar = tmax

    # Find closest row in track
    iclose = (np.abs(track['t'] - tstar)).argmin()
    
    # Get data from track
    Lstar = track['Lstar'][iclose]

    mean_distance = COUPLER_options["mean_distance"] * AU

    flux = Lstar /  ( 4. * np.pi * mean_distance * mean_distance ) # [W m-2]
    heat = flux * ( 1. - COUPLER_options["albedo_pl"] )

    return flux, heat


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
        raise Exception("Cannot find stellar spectrum!")
    

    binwidth_wl = spec_wl[1:] - spec_wl[0:-1]

    print("Modern spectrum statistics:")
    print("\t Flux \n\t\t (min, max) = (%1.2e, %1.2e) erg s-1 cm-2 nm-1" % (np.amin(spec_fl),np.amax(spec_fl)))
    print("\t Wavelength \n\t\t (min, max) = (%1.2e, %1.2e) nm" % (np.amin(spec_wl),np.amax(spec_wl)))
    print("\t Bin width \n\t\t (min, median, max) = (%1.2e, %1.2e, %1.2e) nm" % (np.amin(binwidth_wl),np.median(binwidth_wl),np.amax(binwidth_wl)))
    
    return spec_wl, spec_fl

def IntegratePlanckFunction(lam1, lam2, Teff):
    """Calculates the integrated flux from the planck function

    Returned value has units of [erg s-1 cm-2] and is scaled to 
    the surface of the star. Uses Numpy's trapz() to integrate the spectrum.

    Parameters
    ----------
        lam1 : float
            Lower limit of wavelength [nm]
        lam2 : float
            Upper limit of wavelength [nm]
        Teff : float
            Effective temperature of the object

    Returns
    ----------
        I_planck : float
            Flux at stellar surface
    """
    hc_by_kT = phys.h * phys.c / (phys.k * Teff)
    planck_func = lambda lam : 1.0/( (lam ** 5.0) * ( np.exp( hc_by_kT/ lam) - 1.0 ) ) 

    planck_wl = np.linspace(lam1 * 1e-9, lam2 * 1e-9, 2000)
    planck_fl = planck_func(planck_wl)
    I_planck = np.trapz(planck_fl, planck_wl)  # Integrate planck function over wavelength

    I_planck *= 2 * phys.h * phys.c * phys.c   # W m-2 sr-1, at stellar surface
    I_planck *= np.pi # W m-2, integrate over solid angle
    I_planck *= 1.0e3  # erg s-1 cm-2, convert units

    return I_planck

def MorsCalculateFband(dirs: dict, COUPLER_options: dict):
    """Calculates the integrated fluxes in each stellar band for the modern spectrum.

    These integrated fluxes have units of [erg s-1 cm-2] and are scaled to 
    1 AU from the star. Uses Numpy's trapz() to integrate the spectrum.

    Parameters
    ----------
        dirs : dict
            Directories dictionary
        COUPLER_options : dict
            Dictionary of coupler options variables

    Returns
    ----------
        COUPLER_options : dict
            Dictionary of coupler options variables, now containing integrated fluxes
    """

    # Load spectrum
    spec_wl, spec_fl = ModernSpectrumLoad(dirs, COUPLER_options)

    # Upper limit of wavelength range
    star_bands['bo'][1] = np.amax(spec_wl)
    star_bands['pl'][1] = np.amax(spec_wl)

    # Integrate fluxes across wl, for each band
    for band in star_bands.keys():

        wl_min = star_bands[band][0]
        i_min = (np.abs(spec_wl - wl_min)).argmin()

        wl_max = star_bands[band][1]
        i_max = (np.abs(spec_wl - wl_max)).argmin()

        band_wl = spec_wl[i_min:i_max] 
        band_fl = spec_fl[i_min:i_max]

        fl_integ = np.trapz(band_fl,band_wl)

        COUPLER_options["Fband_modern_"+band] = fl_integ 

    # Predict modern spectrum based on this and recalculate Fband values to correct for integration error. Without 
    # this, the integration calculates values for F_band in the XUV regime to be ~200x too small compared to what Mors 
    # predicts. Since we have to assume that Mors is right (otherwise why even use it?) we have to assume that the 
    # difference is a result of error integrating the observed spectrum error. This can be tested by calculating the 
    # bolometric luminosity by integrating the observed spectrum, and comparing it to the known luminosity.
    predict_fl,_ = MorsSpectrumCalc(COUPLER_options['star_age_modern'], spec_wl, spec_fl,COUPLER_options)

    print("Modern spectrum F_band values:")
    for band in star_bands.keys():

        wl_min = star_bands[band][0]
        i_min = (np.abs(spec_wl - wl_min)).argmin()

        wl_max = star_bands[band][1]
        i_max = (np.abs(spec_wl - wl_max)).argmin()

        band_wl = spec_wl[i_min:i_max] 
        band_fl = predict_fl[i_min:i_max]

        fl_integ = np.trapz(band_fl,band_wl)

        COUPLER_options["Fband_modern_"+band] = fl_integ

        print('F_%s [%d,%d] = %g' % (band,wl_min,wl_max,fl_integ))

    return COUPLER_options

def MorsSpectrumCalc(time_star : float, spec_wl: list, spec_fl: list, COUPLER_options: dict):
    """Calculate historical spectrum for a time 'time_star'.

    Uses the Mors evolution model. Calculates for both surface fluxes and those 
    at 1 AU from the star.

    Parameters
    ----------
        time_star : float
            Stellar age [yr]
        spec_wl : list
            Modern spectrum wavelength array [nm]
        spec_fl : list
            Modern spectrum flux array at 1 AU [erg s-1 cm-2 nm-1]
        COUPLER_options : dict
            Dictionary of coupler options variables

    Returns
    ----------
        sflux : np.array(float)
            Numpy array of flux at 1 AU
        sfluxsurf : np.array(float)
            Numpy array of flux at stellar surface
    """

    # Get rotation rate sample percentile
    pctle = COUPLER_options["star_rot_percentile"]
    pctle = min(max(0.01,pctle),99.99)
    
    # Get time and check that it is in range
    tstar = time_star * 1.e-6
    if (tstar < 0.117):
        print("WARNING: Star age too low! Clipping to 0.117 Myr")
        tstar = 0.117
    if (tstar > 11058.0):
        print("WARNING: Star age too high! Clipping to 11058 Myr")
        tstar = 11058.0

    # Get mass and check that it is in range
    Mstar = COUPLER_options["star_mass"]
    if (Mstar < 0.1):
        print("WARNING: Star mass too low! Clipping to 0.1 M_sun")
        Mstar = 0.1
    if (Mstar > 1.25):
        print("WARNING: Star mass too high! Clipping to 1.25 M_sun")
        Mstar = 1.25

    Rstar = mors.Value(Mstar, tstar, 'Rstar') # radius in solar radii
    COUPLER_options['star_radius'] = Rstar
    Rstar_cm = Rstar * R_sun_cm  # radius in cm

    # Get temperature from Mors
    Tstar = mors.Value(Mstar, tstar, 'Teff')
    COUPLER_options['star_temperature'] = Tstar

    Omega = mors.Percentile(Mstar=Mstar, percentile=pctle)

    Ldict = mors.Lxuv(Mstar=Mstar, Age=tstar, Omega=Omega)

    # Fluxes scaled to 1 AU [erg s-1 cm-2]
    sf = (Rstar_cm / AU_cm) ** 2
    F_band = {
        'xr' : Ldict["Fx"] * sf,
        'e1' : Ldict["Feuv1"] * sf,
        'e2' : Ldict["Feuv2"] * sf,
        'pl' : 0.0  # Calc below
    }  

    # Find flux in planckian region and correct for features
    IPF = IntegratePlanckFunction(star_bands['pl'][0], star_bands['pl'][1], Tstar)
    F_band['pl'] = sf * IPF   # Scale to 1 AU 

    # Get dimensionless ratios of past flux to modern flux
    # It's important that they have the same units
    Q_band = {}
    for band in ['xr','e1','e2','pl']:
        Q_band[band] = F_band[band] / COUPLER_options["Fband_modern_"+band]

    # Calculate historical spectrum...
    if len(spec_wl) != len(spec_fl):
        raise Exception("Stellar spectrum wl and fl arrays are of different lengths!")
    
    if debug:
        print("F_band", F_band)
        print("Q_band", Q_band)

    hspec_fl = np.zeros((len(spec_wl)))
    
    # Loop over each wl bin
    for i in range(len(spec_wl)):
        wl = spec_wl[i]
        fl = spec_fl[i]

        # Work out which band we are in
        for band in F_band.keys():
            if star_bands[band][0] <= wl <= star_bands[band][1]:
                # Apply scale factor for this band
                hspec_fl[i] = fl * Q_band[band]
                break

    # Calculate UV scale factor linearly per-bin, making sure that spectrum is
    # continuous at both ends of the UV bandpass. These boundary conditions have
    # to be true, so the assumption here is the linear scaling behaviour 
    # across the UV regime. UV regime is defined by star_bands dictionary.
    i_uv_wl_low = (np.abs(spec_wl - star_bands['uv'][0])).argmin()
    uv_scale_low = hspec_fl[i_uv_wl_low-1] / spec_fl[i_uv_wl_low]

    i_uv_wl_hgh = (np.abs(spec_wl - star_bands['uv'][1])).argmin()
    uv_scale_hgh = hspec_fl[i_uv_wl_hgh+1] / spec_fl[i_uv_wl_hgh]
    
    irange = i_uv_wl_hgh - i_uv_wl_low
    for i in range(i_uv_wl_low,i_uv_wl_hgh+1,1):
        uv_rel_dist = (i - i_uv_wl_low) / irange
        uv_euv2_scale = (1.0 - uv_rel_dist) * uv_scale_low + uv_rel_dist * uv_scale_hgh
        hspec_fl[i] = spec_fl[i] * uv_euv2_scale
    
    hspec_fl_surf = hspec_fl / sf

    return hspec_fl, hspec_fl_surf

def BaraffeLoadtrack(COUPLER_options: dict):
    """Load baraffe track into memory

    You can download other tracks from this file on their website: 
    http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/BHAC15_tracks+structure

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
    t =         track['t'][iclose] * 1.e-6
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

def SpectrumWrite(time_dict, wl, sflux, sfluxsurf, dirs):
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
        dirs : dict
            Directories dictionary

    """

    tstar = time_dict['star'] * 1.0e-6  # yr -> Myr

    X = np.array([wl,sflux]).T
    outname1 = dirs['output'] + "/%d.sflux" % time_dict['planet']
    header = '# WL(nm)\t Flux(ergs/cm**2/s/nm)          Stellar flux (1 AU) at t_star = %d Myr ' % tstar
    np.savetxt(outname1, X, header=header,comments='',fmt='%1.4e',delimiter='\t')

    Y = np.array([wl,sfluxsurf]).T
    outname2 = dirs['output'] + "/%d.sfluxsurf" % time_dict['planet']
    header = '# WL(nm)\t Flux(ergs/cm**2/s/nm)          Stellar flux (surface) at t_star = %d Myr ' % tstar
    np.savetxt(outname2, Y, header=header,comments='',fmt='%1.4e',delimiter='\t')


# End of file
