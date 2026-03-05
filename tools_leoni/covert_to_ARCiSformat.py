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


def make_vmr_dataframe(file,relevant_gases):

    atm_profile = read_ncdf_profile(file, extra_keys=['pl','x_gas'])

    parr = atm_profile['pl'][:-1]* 1e-5  # convert to bar
    df = pd.DataFrame(parr, columns=["Pbar"])

    for i, gas in enumerate(relevant_gases):
        key = gas + '_vmr'
        if key in atm_profile.keys():
            print(key)
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

    for i,file in enumerate(files):
        time = sample_times[i]
        vmrs = make_vmr_dataframe(file,relevant_gases)
        fpath = os.path.join(output_dir,'vmrs_time_%s'%time+'.dat')
        vmrs.to_csv(fpath,sep='\t',index=False)


def create_tp_df(file):

    atm_profile = read_ncdf_profile(file, extra_keys=['pl','tmpl'])
    parr = atm_profile['pl'][:-1] * 1e-5
    print(len(parr))
    df = pd.DataFrame(parr, columns=["Pbar"])
    df['temperature[K]']=atm_profile['tmpl'][:-1]

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

    for i,file in enumerate(files):
        df= create_tp_df(file)
        time=sample_times[i]
        fpath = os.path.join(output_dir, 'tp_time_%s'%time+'.dat')
        df.to_csv(fpath,sep='\t',index=False)



if __name__ == "__main__":
    runname=sys.argv[1]
    nsamp = 5
    get_chem_atmosphere(runname,nsamp)
    get_tps(runname,nsamp)
