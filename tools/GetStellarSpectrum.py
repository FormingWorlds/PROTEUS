#!/usr/bin/env python3

# Python script to download and convert stellar spectra from online databases
import sys

stars_online = {
    "muscles": ["gj1132", "gj1214", "gj15a", "gj163", "gj176", "gj436", "gj551", "gj581", "gj649", "gj667c", "gj674", "gj676a", "gj699", "gj729", "gj832", "gj832_synth", "gj849", "gj876", "hd40307", "hd85512", "hd97658", "l-980-5", "lhs-2686", "trappist-1", "v-eps-eri"],
    "vpl": ["sun", "hd128167", "hd114710", "hd206860", "hd22049"]
}

def DownloadModernSpectrum(name, distance):
    """Get a contemporary stellar spectrum
    
    Scaled to 1 AU from the star.

    Parameters
    ----------
        name : str
            Name of star
        distance : float
            Distance to star [ly]
    Returns
    ----------
        filename : str
            Location where modern spectrum has been saved as a plain-text file
    """

    print("Attempting to obtain spectrum for parameters: [star = %s, distance = %1.2e ly]" % (name, distance))

    import requests, certifi, os
    from astropy.io import fits

    # Convert stellar parameters
    distance = float(distance) * 9.46073047e17 # Convert ly -> cm
    name     = str(name).strip().lower()

    r_scale = 1.496e+13  # 1 AU in cm

    # Get database and name of star
    name = name.strip()
    database = ''
    star = ''
    for k in stars_online.keys():
        if name in stars_online[k]:
            star = name
            database = k
            break 
    if (database == ''):
        raise Exception("Could not find star '%s' in stellar databases!" % name)
    else:
        print("\tFound star in '%s' database" % database)

    # Convert data from database source format to plain text file
    plaintext_spectrum = "spec_%s.txt" % star
    database_spectrum  = "spec_%s.%s" % (star,database)
    print("\tDownloading spectrum and writing file '%s'" % plaintext_spectrum)
    new_str = '# Spectrum of %s (%s) at 1 AU\n# WL(nm)\tFlux(ergs/cm**2/s/nm)\n' % (star,database)
    match database:
        case 'muscles':
            cert = certifi.where()
            source = "https://archive.stsci.edu/missions/hlsp/muscles/%s/hlsp_muscles_multi_multi_%s_broadband_v23_adapt-const-res-sed.fits"%(star, star)
            resp = requests.get(source, verify=cert) # Download file
            
            if resp.status_code == 404:  # Try other possible option (v22 instead of v23)
                source = "https://archive.stsci.edu/missions/hlsp/muscles/%s/hlsp_muscles_multi_multi_%s_broadband_v22_adapt-const-res-sed.fits"%(star, star)
            resp = requests.get(source, verify=cert) # Download file

            if (resp.status_code != 200):
                print("\t WARNING: Request returned with status code '%d' (should be 200/OK)" % resp.status_code)

            with open(database_spectrum, "wb") as f:
                f.write(resp.content) 
            
            # Epsilon Eridani is 10.475 light years away and with 0.735 solar radius
            # GJ876 is 15.2 light years away and has 0.3761 solar radius
            # GJ551 (proxima cen) is 4.246 light years away and has 0.1542 solar radius
            # GJ436 is 31.8 light years away and has 0.42 solar radius
            # GJ1214 is 47.5 light years away and has 0.2064 solar radius
            # TRAPPIST-1 is 40.66209 ly away and has 0.1192 solar radius
            hdulist = fits.open(database_spectrum)
            spec = fits.getdata(database_spectrum, 1)

            # WAVELENGTH : midpoint of the wavelength bin in Angstroms
            # WAVELENGTH0: left (blue) edge of the wavelength bin in Angstroms
            # WAVELENGTH1: right (red) edge of the wavelength bin in Angstroms
            # FLUX : average flux density in the wavelength bin in erg s-1 cm-2 Angstroms-1
            # MUSCLES provides fluxes scaled to earth-star distance, but we need to scale it to the surface of the star.
            # need to convert to ergs/cm**2/s/nm

            for n,w in enumerate(spec['WAVELENGTH']):
                wl = w * 0.1  # Convert å to nm
                fl = float(spec['FLUX'][n])*10.0 * (distance / r_scale )**2  # Convert units and scale flux

                new_str += "%1.7e\t%1.7e \n" % (wl,fl)

            with open(plaintext_spectrum, 'w') as f:
                f.write(new_str)

        case 'vpl':
            cert = False  # This is not good, but it will stay for now.
            source = "https://vpl.astro.washington.edu/spectra/stellar/%sum.txt" % star
            resp = requests.get(source, verify=cert) # Download file

            if (resp.status_code != 200):
                print("\t WARNING: Request returned with status code '%d' (should be 200/OK)" % resp.status_code)

            with open(database_spectrum, "wb") as f:
                f.write(resp.content) 

            with open(database_spectrum) as f:
                for line in f.readlines():
                    if not line.startswith("#") and line.split():
                        li = line.split()
                        
                        wl = float(li[0]) * 1.0e3  # Convert um to nm
                        fl = float(li[1]) * 1.0e4  * (distance / r_scale )**2  # Convert units and scale flux 

                        new_str += "%1.7e\t%1.7e \n" % (wl,fl)
                        
            with open(plaintext_spectrum, 'w') as f: 
                f.write(new_str)   

    os.remove(database_spectrum) 

    print("\tDone!")

    return plaintext_spectrum

def PrintHelp():
    print("""
This script downloads and parses stellar spectra from online databases.

Run pattern: GetStellarSpectrum.py [command] [param1] [param2] 

Commands:
    'help'
        Shows this menu.
        No parameters.
    'list'
        Lists available stars.
        No parameters.
    'get'
        Downloads and converts spectrum for given star.
        'param1' : star name
        'param2' : distance from Earth in units of Ly
            """)


# Run script
if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("Not enough arguments provided.")
        PrintHelp()
        exit(1)

    match sys.argv[1]:
        case "list":
            print("Available stars:")
            for k in stars_online.keys():
                for s in stars_online[k]:
                    print("%12s    (%7s)" % (s,k))

        case "get":
            star = str(sys.argv[2])
            sdst = float(sys.argv[3])
            DownloadModernSpectrum(star,sdst)

        case "help":
            PrintHelp()

        case _:
            print("No command provided.")
            PrintHelp()

# End of file
