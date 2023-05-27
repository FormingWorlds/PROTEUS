# Util functions for wrapping Mors XUV evolution library

from utils.modules_ext import *
from utils.constants import *
import Mors as mors

# Spectral bands for stellar fluxes, in nm
star_bands = {
    "xr" : [1.e-3 , 10.0],  # X-ray,  defined by mors
    "e1" : [10.0  , 32.0],  # EUV1,   defined by mors
    "e2" : [32.0  , 92.0],  # EUV2,   defined by mors
    "uv" : [92.0  , 200.0], # UV,     defined by me
    "pl" : [200.0 , 1.e9],  # planck, defined by me
    'bo' : [1.e-3 , 1.e9]   # bolo,   defined by me
}

def SolarConstant(time_dict: dict, COUPLER_options: dict):
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
    Tstar = time_dict['star'] * 1.e-6  # Convert from yr to Myr

    Lstar = mors.Value(Mstar, Tstar, 'Lbol')  # Units of L_sun
    Lstar *= L_sun # Convert to W

    mean_distance = COUPLER_options["mean_distance"] * AU

    flux = Lstar /  ( 4. * np.pi * mean_distance * mean_distance )
    heat = flux * ( 1. - COUPLER_options["albedo_pl"] )

    return flux, heat


def ModernSpectrumLoad(dirs: dict, COUPLER_options: dict):
    """Load modern spectrum into memory.

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
        spec_data = np.loadtxt(spec_file, skiprows=2,delimiter='\t').T
        spec_wl = spec_data[0]
        spec_fl = spec_data[1]
    else:
        raise Exception("Cannot find stellar spectrum!")

    
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

    planck_wl = np.linspace(lam1 * 1e-9, lam2 * 1e-9, 10000)
    planck_fl = planck_func(planck_wl)
    I_planck = np.trapz(planck_fl, planck_wl)  # Integrate planck function over wavelength

    I_planck *= 2 * phys.h * phys.c * phys.c   # W m-2 sr-1, at stellar surface
    I_planck *= np.pi # W m-2, integrate over solid angle
    I_planck *= 1.0e3  # erg s-1 cm-2, convert units

    return I_planck

def ModernSpectrumFband(dirs: dict, COUPLER_options: dict):
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

    print("Modern spectrum F_band values:")

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

        print('Band %s [%d,%d] = %g' % (band,wl_min,wl_max,fl_integ))

    # Stellar radius (NOW) in cm
    Rstar_cm = COUPLER_options['star_radius_modern'] * R_sun_cm

    # Work out how much the planck function overestimates the integrated flux in the 'pl' by testing vs observed flux.
    # This accounts for the fact that T_eff does not include spectral features when we use it to estimate flux, so it's
    # less accurate in doing so compared to the bands which use the output of Mors. This factor is very close to unity.
    fl_planck = IntegratePlanckFunction(star_bands['pl'][0], star_bands['pl'][1], COUPLER_options['star_temperature_modern'])
    fl_planck *= (Rstar_cm / AU_cm) ** 2
    COUPLER_options['observed_vs_planckian'] = COUPLER_options["Fband_modern_pl"] / fl_planck 

    print("observed_vs_planckian =",COUPLER_options['observed_vs_planckian'])

    return COUPLER_options

def HistoricalSpectrumWrite(time_dict: dict, spec_wl: list, spec_fl: list, dirs : dict, COUPLER_options: dict):
    """Write historical spectrum to disk, for a time t.

    Uses the Mors evolution model. Spectrum scaled to 1 AU from the star.

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

    Returns
    ----------
        historical_spectrum : str
            Path to historical spectrum file written by this function.
    """

    # Get rotation rate sample percentile
    pctle = COUPLER_options["star_rot_percentile"]
    pctle = min(max(0.01,pctle),99.99)
    
    # Get time and check that it is in range
    tstar = time_dict["star"] * 1.e-6
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
    F_band['pl'] = sf * COUPLER_options['observed_vs_planckian'] * IPF

    # Get dimensionless ratios of past flux to modern flux
    # It's important that they have the same units
    Q_band = {}
    for band in ['xr','e1','e2','pl']:
        F_modern_band = COUPLER_options["Fband_modern_"+band]
        Q_band[band] = F_band[band] / F_modern_band

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

    # Calculate UV scale factor linearly per-bin, making sure that it's 
    # continuous at both ends of its bandpass. These boundary conditions have
    # to be true, so the assumption here is the linear scaling behaviour 
    # across the UV regime. UV regime is defined by star_bands dictionary.
    i_uv_wl_low = (np.abs(spec_wl - star_bands['uv'][0])).argmin()
    uv_scale_low = hspec_fl[i_uv_wl_low] / spec_fl[i_uv_wl_low]

    i_uv_wl_hgh = (np.abs(spec_wl - star_bands['uv'][1])).argmin()
    uv_scale_hgh = hspec_fl[i_uv_wl_hgh] / spec_fl[i_uv_wl_hgh]
    
    irange = i_uv_wl_hgh - i_uv_wl_low
    for i in range(i_uv_wl_low,i_uv_wl_hgh+1,1):
        wl = spec_wl[i]
        fl = spec_fl[i]
        if star_bands['uv'][0] <= wl <= star_bands['uv'][1]:  # Are we in the UV range?
            uv_rel_dist = (i - i_uv_wl_low) / irange
            uv_euv2_scale = uv_rel_dist * uv_scale_low + (1.0 - uv_rel_dist) * uv_scale_hgh
            hspec_fl[i] = fl * uv_euv2_scale

    # Save historical spectrum at 1 AU
    X = np.array([spec_wl,hspec_fl]).T
    outname = dirs['output'] + "/%d.sflux" % time_dict['planet']
    header = '# Stellar flux (1 AU) at t_star = %d Myr \n# WL(nm)\t Flux(ergs/cm**2/s/nm)' % tstar
    np.savetxt(outname, X, header=header,comments='',fmt='%1.5e',delimiter='\t')

    # Save historical spectrum at stellar surface
    Y = np.array([spec_wl,hspec_fl / sf]).T
    outname = dirs['output'] + "/%d.sfluxsurf" % time_dict['planet']
    header = '# Stellar flux (surface) at t_star = %d Myr \n# WL(nm)\t Flux(ergs/cm**2/s/nm)' % tstar
    np.savetxt(outname, Y, header=header,comments='',fmt='%1.5e',delimiter='\t')

    return outname

# End of file
