from __future__ import annotations

import sys

from tomlkit import dumps, parse


def param_edit(inputfile,fO2,orbdist,Hocean,CHratio,silicates:bool):
    ''' function which  writes a new HELIOS parameter file using the newly calculated tp profile from Helios'''

    with open(inputfile, "r") as f:
        data = parse(f.read())

    #modify lines in input file
    data["params"]["out"]["path"] = 'gridrun_{}IW_{}AU_{}Hocean_{}CH_{}'.format(fO2,orbdist,Hocean,CHratio,silicates)
    data["orbit"]["semimajoraxis"] = float(orbdist)
    data["outgas"]["fO2_shift_IW"] = float(fO2)
    data["outgas"]["silicates"] = silicates
    data["delivery"]["elements"]["H_oceans"] = float(Hocean)
    data["delivery"]["elements"]["CH_ratio"] = float(CHratio)
    # write back
    with open(inputfile, "w") as f:
        f.write(dumps(data))



if __name__ == "__main__":
    inputfile=sys.argv[1]
    fO2=sys.argv[2]
    orbdist=sys.argv[3]
    Hocean=sys.argv[4]
    CHratio=sys.argv[5]
    silicates = sys.argv[6].lower() == "true"

    param_edit(inputfile,fO2,orbdist,Hocean,CHratio,silicates)
