from __future__ import annotations

import glob
import logging
import os
import sys
from typing import TYPE_CHECKING

import pandas as pd
from compar_atmosphere_models import sample_output

from proteus.atmos_clim.common import read_ncdf_profile
from proteus.outgas.lavatmos import species_lib
from proteus.utils.constants import element_list, vol_list
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




def get_element_abun(model,time):

    df=pd.read_csv('/data3/leoni/PROTEUS/output/%s/runtime_helpfile.csv'%model,sep='\t')
    print(df)
    hf_row = df.loc[(df["Time"] - time).abs().idxmin()]
    molfracs={}

    nfrac={'H': 0.0,'He':0.0,'C': 0.0,'N':0.0,
           'O': 0.0,'Na':0.0,'Mg': 0.0,'Si':0.0,
           'Fe': 0.0,'Al':0.0,'Ca': 0.0,'Ti':0.0,
           'S': 0.0,'Cl':0.0,'K': 0.0,'F':0.0,'P': 0.0,'V':0.0}

    total_mols=0.0
    for e in element_list:
       molfracs[e]= hf_row[e + '_kg_atm']/species_lib[e].weight
       total_mols+= molfracs[e]
    for e in element_list:
       nfrac[e]= molfracs[e]/total_mols

    element_folder='/data3/leoni/evolution_project/elements_ARCiS/{}/{}/'.format(model,time)

    print(element_folder)
    os.makedirs(element_folder, exist_ok=True)
    with open(element_folder+'elements.dat', "w") as f:
        for key in nfrac:
            f.write(f"{key} {float(nfrac[key]):.3E} \n")
    print('elements in ARCiS format written to {}elements.dat'.format(element_folder))




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
        times, sample_times, files = sample_output(input_dir, nsamp=nsamp)
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
            print('mixingratio file for ARCiS is created ...')
            mixrats = create_mixrat_df(file)
            mixratpath=os.path.join(output_dir,'mixingratios_time_%s'%time+'.dat')
            names=mixrats.columns.tolist()
            indices=[0,1] #pressure and temperature
            molecule_names = [i for j, i in enumerate(names) if j not in indices]
            print(molecule_names)
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

    print(input_dir)
    if nsamp>1:
        plot_times, sample_times, files = sample_output(input_dir, nsamp=nsamp)
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



def config_ARCiS(input_file, tp_file_path, vmr_file_path, elementfile, species, mixratfile=False, mixrat_file_path='mixingratios.dat'):

    '''function which modifies the ARCiS input file by updating the elementfile from which the abundances are read
    Input Tsurf is not mandatory since then ARCiS will converge to s surface temperature itself'''


    print(input_file)
    vmrs=pd.read_csv(vmr_file_path,sep='\t').iloc[0]

    tp=pd.read_csv(tp_file_path,sep='\t')
    Psurf=tp['Pbar'][0]
    print(tp_file_path)
    print('surface temperature output by PROTEUS:',tp['temperature[K]'][0])
    Tsurface=tp['temperature[K]'][0]

    output_lines = []

    with open(input_file, "r") as f:
        for line in f:
            stripped_line = line.strip()
            key = stripped_line.split("=")[0]
            replaced = False

            if key in species:
                output_lines.append(f"{key}={vmrs[key]}\n")
                print(key,vmrs[key])
                replaced=True

            if replaced:
                continue

            if key=="TPfile":
                if mixratfile:
                    output_lines.append(f"TPfile={mixrat_file_path}\n")
                else:
                    output_lines.append(f"TPfile={tp_file_path}\n")

            elif key=="elementfile":
                output_lines.append(f"elementfile={elementfile}\n")

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


    vmrpaths,mixratpaths=get_chem_atmosphere(runname,nsamp)
    sample_times,tppaths=get_tps(runname,nsamp)
