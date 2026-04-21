from __future__ import annotations

import glob
import logging
import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

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
    'CH4',
    'CO2',
    'H2',
    'H2S',
    'SO2',
    'NH3',
    'HCN',
    'N2O',
    'O3',)


def sample_times_ARCiS(times: list, nsamp: int, tmin: float = 1.0):

    '''sample times in a way that last time step is excluded becaus ein the case where
    volatiles escape the last iteration is only an extremely thin TP profile '''

    from proteus.utils.helper import find_nearest

    # check count
    if len(times) <= nsamp:
        out_t, out_i = np.unique(times, return_index=True)
        return list(out_t), list(out_i)

    # lower limit
    tmin = max(tmin, np.amin(times))
    tmin = min(tmin, np.amax(times))
    tmin = max(tmin, 1.0)
    # upper limit
    #selected_times=times.copy()
    #selected_times.remove(np.amax(selected_times))
    tmax = max(tmin + 1, np.amax(times))
    # do not allow times outside range
    allowed_times = [int(x) for x in times if tmin <= x <= tmax]

    # get samples on log-time scale
    sample_t = []
    sample_i = []

    for s in np.logspace(np.log10(tmin), np.log10(tmax), nsamp):  # Sample on log-scale
    #for s in np.linspace(tmin, tmax, nsamp): #linear sampling
        print(s)
        remaining = [int(v) for v in set(allowed_times) - set(sample_t)]
        if len(remaining) == 0:
            break

        # Get next nearest time
        val, _ = find_nearest(remaining, s)
        sample_t.append(int(val))
        # Get the index of this time in the original array
        _, idx = find_nearest(times, val)
        sample_i.append(int(idx))

    # sort output
    mask = np.argsort(sample_t)
    out_t, out_i = [], []
    for i in mask:
        out_t.append(sample_t[i])
        out_i.append(sample_i[i])
    return out_t, out_i



def sample_output_ARCiS(output_dir, nsamp:int ,extension:str = "_atm.nc", tmin:float = 1.0):

    """
    Sample output files from a model run based on their time stamps.

    This function searches the `<output_dir>/data` directory for files whose
    names end with the given extension and whose base name is an integer
    time stamp. It then selects up to `nsamp` representative output times
    greater than or equal to `tmin`, using `sample_times`, and returns the
    corresponding times and file paths.

    If no matching files are found, the function returns empty lists. If an
    archive exists in the data directory, an error is logged indicating that
    the archive should be extracted first.

    Parameters
    ----------
    output_dir : str
        Path to the model output directory containing a `data/` subdirectory.
    extension : str, optional
        File extension used to identify output files (default: "_atm.nc").
    tmin : float, optional
        Minimum time to consider when sampling outputs (default: 1.0).
    nsamp : int, optional
        Number of output times to sample (default: 8).

    Returns
    -------
    out_t : list
        List of sampled output times.
    out_f : list
        List of file paths corresponding to the sampled times.
    """


    files = glob.glob(os.path.join(output_dir+"/data", "*"+extension))

    print(files)
    # No files found?
    if len(files) < 1:
        log.error("No output files found, check if arxiv exists and Extract it.")

        # Return empty
        return [], []

    # get times
    times = [int(f.split("/")[-1].split(extension)[0]) for f in files]

    print(times)
    out_t, out_i = sample_times_ARCiS(times, nsamp, tmin=tmin)
    out_f = [files[i] for i in out_i]

    print(np.array(times))
    print(np.array(out_t))
    print(np.array(out_f))
    # return times and file paths
    return np.array(times), np.array(out_t), np.array(out_f)




def make_vmr_dataframe(file,relevant_gases):

    atm_profile = read_ncdf_profile(file, extra_keys=['x_gas'])

    parr = atm_profile['p'] [1::2]* 1e-5  # convert to bar and select only pressures at center of layers
    df = pd.DataFrame(parr, columns=["Pbar"])

    for i, gas in enumerate(relevant_gases):
        key = gas + '_vmr'
        if key in atm_profile.keys():
            df[gas] = list(atm_profile[key])
    return df


def create_mixrat_df(file):

    tp=create_tp_df(file)
    vmrs=make_vmr_dataframe(file,list(GASES_ARCiS))

    cols_to_use = vmrs.columns.difference(tp.columns)
    dfNew = tp.merge(vmrs[cols_to_use], left_index=True, right_index=True, how='outer')

    return dfNew

def get_chem_atmosphere(runname: str, nsamp: int, relevant_gases: list = None, mixratfile=True):

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
        print('nsamp=',nsamp)
        times, sample_times, files = sample_output_ARCiS(input_dir, nsamp=nsamp)
    else:
        allfiles = glob.glob(os.path.join(input_dir, 'data', '*_atm.nc'))
        files=[natural_sort(allfiles)[-1]]


    if len(files) == 0:
        log.warning('No atmosphere NetCDF files found in output folder')
        return

    paths=[]
    mixratpaths=[]
    for i,file in enumerate(files):
        time = sample_times[i]
        vmrs = make_vmr_dataframe(file,relevant_gases)
        fpath = os.path.join(output_dir,'vmrs_time_%s'%time+'.dat')
        vmrs.to_csv(fpath,sep='\t',index=False)
        paths.append(fpath)

        if mixratfile:
            mixrats = create_mixrat_df(file)
            mixratpath=os.path.join(output_dir,'mixingratios_time_%s'%time+'.dat')
            names=mixrats.columns.tolist()
            indices=[0,1] #pressure and temperature
            molecule_names = [i for j, i in enumerate(names) if j not in indices]
            #write number of species in first row and names in second row
            with open(mixratpath, "w") as f:
                f.write(f"{len(molecule_names)}\n")
                for item in molecule_names:
                    f.write(str(item) + '\t')
                f.write("\n")
            # append dataframe data including pressure and temperature
            mixrats.to_csv(mixratpath, mode="a", sep='\t',index=False,header=False)
            mixratpaths.append(mixratpath)
    return paths,mixratpaths


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
        plot_times, sample_times, files = sample_output_ARCiS(input_dir, nsamp=nsamp)
    else:
        allfiles = glob.glob(os.path.join(input_dir, 'data', '*_atm.nc'))
        files=[natural_sort(allfiles)[-1]]

    paths=[]
    for i,file in enumerate(files):
        df= create_tp_df(file)
        time=sample_times[i]
        fpath = os.path.join(output_dir, 'tptime_%s'%time+'.dat')
        df[::-1].to_csv(fpath,sep='\t',index=False)
        paths.append(fpath)

    return sample_times, paths



if __name__ == "__main__":
    runname=sys.argv[1]
    nsamp=int(sys.argv[2])


    vmrpaths,mixratpaths=get_chem_atmosphere(runname,nsamp)
    sample_times,tppaths=get_tps(runname,nsamp)
