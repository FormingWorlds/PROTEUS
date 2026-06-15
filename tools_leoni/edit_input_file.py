from __future__ import annotations

import sys

from tomlkit import dumps, parse


def param_edit(inputfile,fO2,orbdist,mass,Cppmw,silicates:bool):
    ''' function which  writes a new HELIOS parameter file using the newly calculated tp profile from Helios'''

    with open(inputfile, "r") as f:
        data = parse(f.read())

    #modify lines in input file
    data["params"]["out"]["path"] = 'grid_{}Mearth_tmax1e6_tmin1e3_{}IW_{}AU_cppmw{}_{}'.format(mass,fO2,orbdist,Cppmw,silicates)
    #data["params"]["out"]["path"] = 'testrun_element_updating'
    data["orbit"]["semimajoraxis"] = float(orbdist)
    data["outgas"]["fO2_shift_IW"] = float(fO2)
    data["outgas"]["silicates"] = silicates
    data["struct"]["mass_tot"] = float(mass)
    data["delivery"]["elements"]["C_ppmw"] = float(Cppmw)
    # write back
    with open(inputfile, "w") as f:
        f.write(dumps(data))



if __name__ == "__main__":
    inputfile=sys.argv[1]
    fO2=sys.argv[2]
    orbdist=sys.argv[3]
    mass=sys.argv[4]
    Cppmw=sys.argv[5]
    silicates = sys.argv[6].lower() == "true"
    print('inputfile: {}, fO2: {}, orbdist: {}, mass: {}, Cppmw: {}, silicates: {}'.format(inputfile,fO2,orbdist,mass,Cppmw,silicates))

    param_edit(inputfile,fO2,orbdist,mass,Cppmw,silicates)
