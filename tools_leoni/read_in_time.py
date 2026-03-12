from __future__ import annotations

import glob
import logging
import os
import sys
from typing import TYPE_CHECKING

import pandas as pd
from compar_atmosphere_models import sample_output

from proteus.atmos_clim.common import read_ncdf_profile
from proteus.utils.constants import vol_list
from proteus.utils.helper import natural_sort

if TYPE_CHECKING:
    pass

log = logging.getLogger('fwl.' + __name__)

GASES_STANDARD = (
    'CO',
    'H2O',
    'N2',
    'O2',
    'O3',
    'OH',
    'H',
    'SO',
    'CH4',
    'CO2',
    'H2',
    'H2S',
    'HCN',
    'NH3',
    'SO2',
    'Al',
    'HAlO2',
    'N2O',
    'NO',
    'NO2',
    'HNO3',
    'PH3',
    'N',
    'O',
    'S',
)

REFRACTORY_GASES = (
    'Fe',
    'Si',
    'Ti',
    'K',
    'Mg',
    'SiO',
    'SiO2',
    'TiO',
    'FeO',
    'MgO',
    'Na',
    'TiO2',
    'SiH',
    'SiH4',
)

GASES_ARCiS=('CO',
    'H2O',
    'N2',
    'O2',
    'SO',
    'CH4',
    'CO2',
    'H2',
    'H2S',
    'SO2',
    'NH3')


def make_vmr_dataframe(file,relevant_gases):

    atm_profile = read_ncdf_profile(file, extra_keys=['x_gas'])

    parr = atm_profile['p'] [1::2]* 1e-5  # convert to bar and select only pressures at center of layers
    df = pd.DataFrame(parr, columns=["Pbar"])

    for i, gas in enumerate(relevant_gases):
        key = gas + '_vmr'
        if key in atm_profile.keys():
            df[gas] = list(atm_profile[key])
    return df


def get_chem_atmosphere(runname: str, nsamp: int, relevant_gases: list = None):

    log.info('Plot atmosphere chemical composition')

    input_dir='/data3/leoni/PROTEUS/output/{}/'.format(runname)
    output_dir='/data3/leoni/evolution_project/VMRs/{}'.format(runname)

    #check if output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if not relevant_gases:
        relevant_gases = list(vol_list) + list(GASES_STANDARD) + list(REFRACTORY_GASES)

    # Remove duplicates, preserving order
    relevant_gases = list(dict.fromkeys(relevant_gases))

    # get sampled times at which to get chemistry for arcis
    if nsamp>1:
        times, sample_times, files = sample_output(input_dir, nsamp=nsamp)
    else:
        allfiles = glob.glob(os.path.join(input_dir, 'data', '*_atm.nc'))
        files=[natural_sort(allfiles)[-1]]


    if len(files) == 0:
        log.warning('No atmosphere NetCDF files found in output folder')
        return

    paths=[]
    for i,file in enumerate(files):
        time = sample_times[i]
        vmrs = make_vmr_dataframe(file,relevant_gases)
        fpath = os.path.join(output_dir,'vmrs_time_%s'%time+'.dat')
        vmrs.to_csv(fpath,sep='\t',index=False)
        paths.append(fpath)

    return paths


def create_tp_df(file):

    atm_profile = read_ncdf_profile(file, extra_keys=[])
    parr = atm_profile['p'][1::2]* 1e-5  # convert to bar and select only pressures at center of layers
    df = pd.DataFrame(parr, columns=["Pbar"])
    df['temperature[K]']=atm_profile['t'][1::2] # again, select only temperatures at center of layers
    return df


def get_tps(runname: str, nsamp: int):

    input_dir='/data3/leoni/PROTEUS/output/{}/'.format(runname)
    output_dir='/data3/leoni/evolution_project/TPss/{}'.format(runname)
    #check if output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if nsamp>1:
        plot_times, sample_times, files = sample_output(input_dir, nsamp=nsamp)
    else:
        allfiles = glob.glob(os.path.join(input_dir, 'data', '*_atm.nc'))
        files=[natural_sort(allfiles)[-1]]

    paths=[]
    for i,file in enumerate(files):
        df= create_tp_df(file)
        time=sample_times[i]
        fpath = os.path.join(output_dir, 'tp_time_%s'%time+'.dat')
        df[::-1].to_csv(fpath,sep='\t',index=False)
        paths.append(fpath)

    return sample_times,paths



def config_ARCiS(input_file, tp_file_path, vmr_file_path, species):

    '''function which modifies the ARCiS input file by updating the elementfile from which the abundances are read
    Input Tsurf is not mandatory since then ARCiS will converge to s surface temperature itself'''


    vmrs=pd.read_csv(vmr_file_path,sep='\t').iloc[0]

    tp=pd.read_csv(tp_file_path,sep='\t')
    Psurf=tp['Pbar'][0]
    Tsurface=tp['temperature[K]'][0]

    output_lines = []

    with open(input_file, "r") as f:
        for line in f:
            stripped_line = line.strip()
            key = stripped_line.split("=")[0]
            replaced = False

            if key in species:
                output_lines.append(f"{key}={vmrs[key]}\n")
                replaced=True

            if replaced:
                continue

            if key=="TPfile":
                output_lines.append(f"TPfile={tp_file_path}\n")

            elif key=="pmax":
                output_lines.append("pmax="+str(Psurf)+"d0\n")

            elif key=="Pp":
                output_lines.append("Pp="+str(Psurf)+"d0\n")

            elif key=="Tsurface":
                output_lines.append("Tsurface="+str(Tsurface)+"d0\n")

            else:
                output_lines.append(line)

    # Write the updated lines back to the file
    with open(input_file, "w") as f:
        f.writelines(output_lines)

if __name__ == "__main__":
    runname=sys.argv[1]
    nsamp=int(sys.argv[2])
    sample=int(sys.argv[3])

    vmrpaths=get_chem_atmosphere(runname,nsamp)
    sample_times,tppaths=get_tps(runname,nsamp)

    tp_file_path=tppaths[sample-1]
    time=sample_times[sample-1]
    vmr_file_path=vmrpaths[sample-1]

    species=list(GASES_ARCiS)

    input_file='/data3/leoni/ARCiS/input_PROTEUS.dat'
    config_ARCiS(input_file, tp_file_path, vmr_file_path, species)

    sys.exit(time)
