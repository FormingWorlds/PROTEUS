from __future__ import annotations

import glob
import logging
import os
import sys
from typing import TYPE_CHECKING

import pandas as pd

from proteus.outgas.lavatmos import species_lib
from proteus.utils.constants import element_list

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



def config_ARCiS(input_file, tp_file_path, vmr_file_path, elementfile, species, mixratfile=False, mixrat_file_path='mixingratios.dat'):

    '''function which modifies the ARCiS input file by updating the elementfile from which the abundances are read
    Input Tsurf is not mandatory since then ARCiS will converge to s surface temperature itself'''


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

def read_tps(runname: str):

    '''reads in TP files of tPs folder and times'''

    output_dir='/data3/leoni/evolution_project/TPss/{}/'.format(runname)
    tpfiles= [f for f in glob.glob(output_dir + "*.dat")]
    times=[]
    for filename in tpfiles:
        time = filename.split('_')[2]
        times.append(time)
    return times,tpfiles

def read_vmrs(runname: str, mixratfile: bool):

    '''reads in VMRS files of VMRs folder'''

    output_dir='/data3/leoni/evolution_project/VMRs/{}/'.format(runname)
    files = [f for f in glob.glob(output_dir+'vmrs_time_*.dat')]
    if mixratfile:
        mixratfiles = [f for f in glob.glob(output_dir + 'mixingratios_time_*.dat')]
        return files,mixratfiles
    else:
        return files

if __name__ == "__main__":

    input_path='/data3/leoni/ARCiS/'
    runname=sys.argv[1]
    time=int(sys.argv[2])
    mixratfile = sys.argv[3].lower() == "true"
    input_file = sys.argv[4]

    times,tppaths=read_tps(runname)

    if mixratfile:
        vmrpaths,mixratpaths = read_vmrs(runname, mixratfile=True)
        mixrat_file_path = next(vmr for vmr in mixratpaths if vmr.endswith("_{}.dat".format(time)))
    else:
        vmrpaths = read_vmrs(runname, mixratfile=False)
        mixrat_file_path='mixingratios.dat'

    tp_file_path = next(t for t in tppaths if t.endswith("_{}.dat".format(time)))
    vmr_file_path = next(vmr for vmr in vmrpaths if vmr.endswith("_{}.dat".format(time)))

    species=list(GASES_ARCiS)

    elementpath='/data3/leoni/evolution_project/elements_ARCiS/{}/{}/elements.dat'.format(runname,time)
    get_element_abun(runname,time)
    config_ARCiS(input_path+input_file, tp_file_path, vmr_file_path, elementpath, species,mixratfile=mixratfile,mixrat_file_path=mixrat_file_path)
