Downloaded from http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/BHAC15_tracks+structure
Reference is Baraffe et al 2015 doi:10.1051/0004-6361/201425481
Edits made for Principles of Planetary Climate:
Changed header globally from original header to 

>>NEW
  M/Ms    log(t)(yr)  Teff     L/Ls    g   R/Rs Log(Li/Li0) log(Tc)  log(ROc)   Mrad     Rrad       k2conv      k2rad

so as to make it easier to split out masses into separate files, and create easily
readable table headers.
=====================================================================================================================


BHAC15 tracks and internal structure for brown dwarfs and low mass star (0.01 Msun to 1.4 Msun)

 M/Ms: mass of the star in units of solar mass
log t: age of the star (in yr)
Teff: effective temperature (in K)
L/Ls: log luminosity in units of solar luminosity (value used Ls=3.839d+33)
g: log g  (surface gravity)
R/Rs : radius of the star in units of solar radius    (value used Rs=6.96d10)
log(Li/Li0): log of the ratio of surface lithium abundance to initial abundance
log Tc : log of central temprature
log ROc: log of central density (in gr/cc)
Mrad: mass of radiative core (in solar mass)
Rrad: radius of radiatif core (in solar radius)
k2conv: convective gyration radius
k2rad: radiative gyration radius


------------------------------------------------------------------------
NOTE: Adopted convention to calculate the gyration radii:

k2conv=[ I/(R**2 * Mstar) ]**(1/2)

with I the moment of inertia defined here by :

I = 2/3 integral(r**2 * dm)
with the integral over the mass of the convective region

Same for k2rad, corresponding to the radiative zone.

Example: To calculate the moment of inertia with the definition
above for a 1 Msun star at t = 4.6 Gyr, the table gives

k2conv= 8.993E-02
k2rad= 2.519E-01

k2**2 = k2conv**2 + k2rad**2 ~ 0.071 
(which is the typical value found for the Sun)

and the moment of inertia is:
 ===> I = k2**2 * Mstar * R**2

END OF NOTE
-----------------------------------------------------------------------------