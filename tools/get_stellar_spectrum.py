#!/usr/bin/env python3

# Python script to download and convert stellar spectra from online databases
from __future__ import annotations

import sys

import numpy as np

stars_online = {
    "muscles": ["gj1132", "gj1214", "gj15a", "gj163", "gj176", "gj436", "gj551", "gj581", "gj649", "gj667c", "gj674", "gj676a", "gj699", "gj729", "gj832", "gj832_synth", "gj849", "gj876", "hd40307", "hd85512", "hd97658", "l-980-5", "lhs-2686", "trappist-1", "v-eps-eri", "l-98-59"],
    "vpl": ["hd128167", "hd114710", "hd206860", "hd22049"],
    "nrel": ["sun"]
}

star_distance_pc = {
    # From NASA exoplanet archive if not otherwise noted
    "v-eps-eri": 3.20260,
    "gj876":     4.67517,
    "gj551":     1.30119,
    "gj436":     9.75321,
    "gj1214":    14.6427,
    "trappist-1": 12.429888806540756,
    "gj1132":     12.613,
    "gj15a":      3.56228,
    "gj163":      15.1285,
    "gj176":      9.470450,
    "gj581":      6.298100,
    "gj649":      10.37960,
    "gj667c":     7.24396,
    "gj674":      4.548960,
    "gj676a":     16.0272,
    "gj699":      1.826550,
    "gj729":      2.9759,    # from Gaia EDR3, see https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=GJ+729
    "gj832":      4.964350,
    "gj832_synth": 4.964350, # synthetic spectrum, same distance as gj832
    "hd40307":    12.9363,
    "hd85512":   11.2810,
    "hd97658":   21.5618,
    "l-980-5":    13.3731,   # from Gaia EDR3, see https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=L980-5
    "lhs-2686":   12.1893,   # from Gaia EDR3, see https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=LHS+2686
    "l-98-59":    10.6194,
    "hd128167":   15.756,    # from Gaia EDR3, see https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=HD+128167
    "hd114710":    9.1975,   # from Gaia EDR3, see https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=HD+114710
    "hd206860":    18.133,   # from Gaia EDR3, see https://simbad.u-strasbg.fr/simbad/sim-basic?Ident=HD+206860
    "hd22049":     3.20260,  # Same as Epsilon Eridani (v-eps-eri), but different identifier in vpl database
    "sun": 0.0               # For NREL case, distance is irrelevant
}

def DownloadModernSpectrum(name, distance=None):
    """Get a contemporary stellar spectrum

    Scaled to 1 AU from the star. Append "#lowres" to star name to use
    lower resolution spectrum, if that's what you want.

    Parameters
    ----------
        name : str
            Name of star (with '#lowres' if required)
        distance : float
            Distance to star [pc] (optional; if not provided, uses value from star_distance_pc)
    Returns
    ----------
        filename : str
            Location where modern spectrum has been saved as a plain-text file
    """

    print("Attempting to obtain spectrum")

    # Import required libraries
    import os

    import certifi
    import requests
    from astropy.io import fits

    name     = str(name).strip().lower()

     # Check if lowres
    name_split = name.split("#")
    if (len(name_split) == 1):
        lowres = False
    elif (len(name_split) == 2) and (name_split[1] == "lowres"):
        lowres = True
    else:
        raise Exception("Invalid unable to parse star name '%s'!" % name)
    name = name_split[0]

    # Get database and name of star
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

    # Determine distance to star in parsec
    if distance is None:
        try:
            distance = star_distance_pc[star]
        except KeyError:
            raise Exception(
                "Distance to star '%s' not found in star_distance_pc; "
                "please provide an explicit distance override (in pc)." % star
            )
    else:
        distance = float(distance)

    # Convert pc -> cm
    pc_in_cm = 3.0856775814914e18
    distance_cm = distance * pc_in_cm

    print(
        "\tParameters: [star = %s, distance = %1.2e cm, lowres = %s]"
        % (star, distance_cm, lowres)
    )

    r_scale = 1.496e+13  # 1 AU in cm

    # Convert data from database source format to plain text file
    plaintext_spectrum = "spec_%s.txt" % star
    database_spectrum  = "spec_%s.%s" % (star,database)
    print("\tDownloading spectrum and writing file '%s'" % plaintext_spectrum)

    if (os.path.isfile(plaintext_spectrum)):
        print("\t(Overwriting existing file)")

    new_str = '# Spectrum of %s (%s) at 1 AU\n# WL(nm)\tFlux(ergs/cm**2/s/nm)\n' % (star,database)
    match database:
        case 'muscles':
            cert = certifi.where()

            if lowres:
                resstr = "const"
            else:
                resstr = "var"

            ok = False
            for v in ["v22","v23-v24","v25"]:

                source = f"https://archive.stsci.edu/missions/hlsp/muscles/{v}/{star}/hlsp_muscles_multi_multi_{star}_broadband_{v}_adapt-{resstr}-res-sed.fits"
                print(f"looking for: {source}")
                resp = requests.get(source, verify=cert) # Download file

                if resp.status_code != 404:
                    ok = True
                    break

            if not ok:
                raise Exception("Could not find file on MUSCLES website")

            if (resp.status_code != 200):
                print("\t WARNING: Request returned with status code '%d' (should be 200/OK)" % resp.status_code)

            with open(database_spectrum, "wb") as f:
                f.write(resp.content)

            from astropy.io import fits
            # from astropy.table import Table

            # Epsilon Eridani is 10.475 light years away and with 0.735 solar radius
            # GJ876 is 15.2 light years away and has 0.3761 solar radius
            # GJ551 (proxima cen) is 4.246 light years away and has 0.1542 solar radius
            # GJ436 is 31.8 light years away and has 0.42 solar radius
            # GJ1214 is 47.5 light years away and has 0.2064 solar radius
            # TRAPPIST-1 is 40.66209 ly away and has 0.1192 solar radius
            spec = fits.getdata(database_spectrum, 1)

            # WAVELENGTH : midpoint of the wavelength bin in Angstroms
            # WAVELENGTH0: left (blue) edge of the wavelength bin in Angstroms
            # WAVELENGTH1: right (red) edge of the wavelength bin in Angstroms
            # FLUX : average flux density in the wavelength bin in erg s-1 cm-2 Angstroms-1

            negaflux = False

            wl_arr = []
            fl_arr = []
            for n,w in enumerate(spec['WAVELENGTH']):
                wl = w * 0.1  # Convert å to nm
                fl = float(spec['FLUX'][n])*10.0 * (distance_cm / r_scale )**2  # Convert units and scale flux

                negaflux = negaflux or (fl <= 0)
                fl = max(0.0,fl)
                wl_arr.append(wl)
                fl_arr.append(fl)

            # remove duplicate rows
            _, mask = np.unique(wl_arr, return_index=True)
            print(mask)

            # convert to ascii
            for i in mask:
                new_str += "%1.7e\t%1.7e \n" % (wl_arr[i],fl_arr[i])

            # write the file
            with open(plaintext_spectrum, 'w') as f:
                f.write(new_str)

            if negaflux:
                print("\tWARNING: The stellar spectrum contained flux value(s) <= 0.0 ! These were set to zero.")

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
                        fl = float(li[1]) * 1.0e4  * (distance_cm / r_scale )**2  # Convert units and scale flux

                        new_str += "%1.7e\t%1.7e \n" % (wl,fl)

            with open(plaintext_spectrum, 'w') as f:
                f.write(new_str)

        case 'nrel':
            cert = certifi.where()
            source = "https://www.nrel.gov/grid/solar-resource/assets/data/newguey2003.txt"  # Set to Sun only.
            resp = requests.get(source, verify=cert) # Download file

            if (resp.status_code != 200):
                print("\t WARNING: Request returned with status code '%d' (should be 200/OK)" % resp.status_code)

            with open(database_spectrum, "wb") as f:
                f.write(resp.content)

            i = -1
            with open(database_spectrum) as f:
                for line in f.readlines():
                    i += 1
                    if (i < 9):
                        continue  # Skip header

                    li = line.split()

                    wl = float(li[0]) # Already in nm
                    fl = float(li[1]) * 1.0e3 # Convert [W m-2 nm-1] -> [erg s-1 cm-2 nm-1], already at 1 AU

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
        'param1' : star name (append '#lowres' to star name to avoid large files)
        'param2' : distance from Earth in units of pc (optional; if not provided, uses value from star_distance_pc)
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
            if len(sys.argv) >= 4:
                # Override distance in pc
                sdst = float(sys.argv[3])
            else:
                # Use distance from star_distance_pc
                sdst = None
            DownloadModernSpectrum(star, sdst)

        case "help":
            PrintHelp()

        case _:
            print("Invalid command provided.")
            PrintHelp()

# End of file
