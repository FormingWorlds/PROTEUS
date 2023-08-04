# Util functions for wrapping Mors XUV evolution library

from utils.modules_ext import *
from utils.constants import *
from utils.helper import find_nearest
from utils.stellar_common import *
import Mors as mors

# Spectral bands for stellar fluxes, in nm
star_bands = {
    "xr" : [1.e-3 , 10.0],  # X-ray,  defined by mors
    "e1" : [10.0  , 32.0],  # EUV1,   defined by mors
    "e2" : [32.0  , 92.0],  # EUV2,   defined by mors
    "uv" : [92.0  , 300.0], # UV,     defined by Harrison
    "pl" : [300.0 , 1.e9],  # planck, defined by Harrison
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
            Instellation at TOA in W/m^2

    """ 

    tstar = time_dict['star'] * 1.e-6  # Convert from yr to Myr

    Mstar = COUPLER_options["star_mass"]
    if (Mstar < 0.1):
        print("WARNING: Star mass too low! Clipping to 0.1 M_sun")
        Mstar = 0.1
    if (Mstar > 1.25):
        print("WARNING: Star mass too high! Clipping to 1.25 M_sun")
        Mstar = 1.25
    
    Lstar = mors.Value(Mstar, tstar, 'Lbol')  # Units of L_sun
    Lstar *= L_sun # Convert to W

    mean_distance = COUPLER_options["mean_distance"] * AU

    flux = Lstar /  ( 4. * np.pi * mean_distance * mean_distance )
    heat = flux * ( 1. - COUPLER_options["albedo_pl"] )

    return flux, heat

def MorsStellarRadius(time_dict: dict, COUPLER_options: dict):
    """Calculates the star's radius at a time t.

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
        radius : float
            Radius of star in units of solar radii

    """ 

    tstar = time_dict['star'] * 1.e-6  # Convert from yr to Myr

    Mstar = COUPLER_options["star_mass"]
    if (Mstar < 0.1):
        print("WARNING: Star mass too low! Clipping to 0.1 M_sun")
        Mstar = 0.1
    if (Mstar > 1.25):
        print("WARNING: Star mass too high! Clipping to 1.25 M_sun")
        Mstar = 1.25
    
    Rstar = mors.Value(Mstar, tstar, 'Rstar')  # Units of R_sun

    return Rstar


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
    # predicts. Since we assume that Mors is accurate (otherwise why even use it?) we have to assume that the 
    # difference is a result of error integrating the observed spectrum error. This can be validated by calculating the 
    # bolometric luminosity by integrating the observed spectrum, and comparing it to the known luminosity.
    predict_fl,_ = MorsSpectrumCalc(COUPLER_options['star_age_modern'], spec_wl, spec_fl,COUPLER_options)

    # print("Modern spectrum F_band values:")
    for band in star_bands.keys():

        wl_min = star_bands[band][0]
        i_min = (np.abs(spec_wl - wl_min)).argmin()

        wl_max = star_bands[band][1]
        i_max = (np.abs(spec_wl - wl_max)).argmin()

        band_wl = spec_wl[i_min:i_max] 
        band_fl = predict_fl[i_min:i_max]

        fl_integ = np.trapz(band_fl,band_wl)

        COUPLER_options["Fband_modern_"+band] = fl_integ

        # print('F_%s [%d,%d] = %g' % (band,wl_min,wl_max,fl_integ))

    return COUPLER_options



def MorsSolveUV(dirs: dict, COUPLER_options: dict, spec_wl: list, spec_fl: list, uv_try = [100.0,400.0], iters = 5, eps = 0.1):
    """Solve for best wavelength to use for UV-PL band edge.

    The value used for the UV-PL band edge is not well constrained. This
    function solves for the wavelength at this band edge by using a bisection
    search to optimise the accuracy of the flux at the Lyman-alpha wavelength.

    Parameters
    ----------
        dirs : dict
            Directories dictionary
        COUPLER_options : dict
            Dictionary of coupler options variables
        spec_wl : list
            Modern spectrum wavelength array [nm]
        spec_fl : list
            Modern spectrum flux array at 1 AU [erg s-1 cm-2 nm-1]

        uv_try : list
            List of the initial search boundaries
        iters : int
            Maximum number of bisection iterations to perform
        eps : float
            Relative percentage error for break condition (eps = absolute error in flux / target flux * 100)

    Returns
    ----------
        uvpl_best : float
            Optimal wavelength for the UV-PL boundary
    """

    print("Solving for optimal UV-PL band edge using bisection search...")

    # Check inputs
    if uv_try[0] < 100.0:
        print("ERROR: Cannot perform bisection search below 100 nm ")
        exit(1)
    if uv_try[1] <= uv_try[0]:
        print("ERROR: Cannot perform bisection search on invalid range ", uv_try)
        exit(1)

    # Make copy of options dict
    COPY_options = COUPLER_options

    # Variables
    opt_wl    = 121.567                             # Lyman-alpha
    opt_time  = COPY_options["star_age_modern"]     # Time to optimise at
    sol_wl    = -1                                  # Optimal solution
    sol_fl    = -1                                  # Optimal solution

    # Use modern spectrum to get value to optimise for
    modern_wl, modern_fl = ModernSpectrumLoad(dirs, COPY_options)
    _, i_opt = find_nearest(modern_wl,opt_wl)
    fl_opt = modern_fl[i_opt]

    # Bisection loop
    for i in range(iters):

        # Calculate low case
        star_bands["uv"][1] = uv_try[0]
        star_bands["pl"][0] = uv_try[0]
        COPY_options = MorsCalculateFband(dirs, COPY_options)
        fl_low_arr,_ = MorsSpectrumCalc(opt_time, modern_wl, modern_fl, COPY_options)
        fl_low_val = fl_low_arr[i_opt]
        fl_low_err = abs(fl_low_val-fl_opt)
        
        # Calculate high case
        star_bands["uv"][1] = uv_try[1]
        star_bands["pl"][0] = uv_try[1]
        COPY_options = MorsCalculateFband(dirs, COPY_options)
        fl_hgh_arr,_ = MorsSpectrumCalc(opt_time, modern_wl, modern_fl, COPY_options)
        fl_hgh_val = fl_hgh_arr[i_opt]
        fl_hgh_err = abs(fl_hgh_val-fl_opt)

        # Compare low and high cases
        # Low is best
        if (fl_low_err < fl_hgh_err):
            uv_try = [ uv_try[0], (uv_try[0]+uv_try[1])/2.0 ]
            sol_wl = uv_try[0]
            sol_fl = fl_low_val
            fl_bst_err = fl_low_err

        # High is best
        elif (fl_hgh_err < fl_low_err):
            uv_try = [ (uv_try[0]+uv_try[1])/2.0, uv_try[1] ]
            sol_wl = uv_try[1]
            sol_fl = fl_hgh_val
            fl_bst_err = fl_hgh_err

        # Somehow they are equally good choices
        else:
            print("ERROR: Bisection search cannot decide on new search boundary")
            exit(1)

        # Check break condition
        rel_err = fl_bst_err/fl_opt*100.0
        print("    iter %d, rel. err = %.5e pct"  % (i,rel_err))
        if ( rel_err < eps ):
            break

    # Store final case
    star_bands["uv"][1] = sol_wl
    star_bands["pl"][0] = sol_wl
    
    print("   Complete")
    print("   Target flux: %1.2e erg s-1 cm-2 nm-1" % fl_opt)
    print("   Solved flux: %1.2e erg s-1 cm-2 nm-1" % sol_fl)
    print("   Optimal wavelength for band edge is %.2f nm"   % sol_wl)

    return sol_wl


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

    
    # Smooth over any zeros that happen to be left
    for i in range(1,len(hspec_fl)-1):
        if (hspec_fl[i] == 0):
            hspec_fl[i] = 0.5 * (hspec_fl[i-1] + hspec_fl[i+1])
    
    # Scale to surface
    hspec_fl_surf = hspec_fl / sf

    return hspec_fl, hspec_fl_surf


# End of file
