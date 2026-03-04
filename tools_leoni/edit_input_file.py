from __future__ import annotations

import sys

from tomlkit import dumps, parse


def param_edit(inputfile,fO2,orbdist,Hocean,CHratio,silicates):
    ''' function which  writes a new HELIOS parameter file using the newly calculated tp profile from Helios'''
    with open(inputfile, "r") as f:
        data = parse(f.read())

    # modify values
    data["params"]["params.out"]["path"] = 'gridrun_{}IW_{}AU_{}Hocean_{}CH_{}'.format(fO2,orbdist,Hocean,CHratio,silicates)
    data["orbit"]["semimajoraxis"] = orbdist
    data["outgas"]["fO2_shift_IW"] = fO2
    data["outgas"]["silicates"] = silicates
    data["delivery"]["delivery.elements"]["H_oceans"] = Hocean
    data["delivery"]["delivery.elements"]["CH_ratio"] = CHratio

    # write back
    with open("input.toml", "w") as f:
        f.write(dumps(data))



if __name__ == "__main__":
    inputfile=sys.argv[1]
    fO2=sys.argv[2]
    orbdist=sys.argv[3]
    Hocean=sys.argv[4]
    CHratio=sys.argv[5]
    silicates=sys.argv[6]
