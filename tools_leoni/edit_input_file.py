from __future__ import annotations

import sys

from tomlkit import dumps, parse


def param_edit(inputfile,fO2,orbdist,Hocean,Cppmw,silicates:bool):
    ''' function which  writes a new HELIOS parameter file using the newly calculated tp profile from Helios'''

    with open(inputfile, "r") as f:
        data = parse(f.read())

    #modify lines in input file
    data["params"]["out"]["path"] = 'grid_9Mearth_tmax5e5_tmin1e3_{}IW_{}AU_cppmw{}_Earthcomp'.format(fO2,orbdist,Cppmw)
    #data["params"]["out"]["path"] = 'testrun_element_updating'
    data["orbit"]["semimajoraxis"] = float(orbdist)
    data["outgas"]["fO2_shift_IW"] = float(fO2)
    data["outgas"]["silicates"] = silicates
    #data["delivery"]["elements"]["H_oceans"] = float(Hocean)
    data["delivery"]["elements"]["C_ppmw"] = float(Cppmw)
    # write back
    with open(inputfile, "w") as f:
        f.write(dumps(data))



if __name__ == "__main__":
    inputfile=sys.argv[1]
    fO2=sys.argv[2]
    orbdist=sys.argv[3]
    Hocean=sys.argv[4]
    Cppmw=sys.argv[5]
    silicates = sys.argv[6].lower() == "true"

    param_edit(inputfile,fO2,orbdist,Hocean,Cppmw,silicates)
