from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(1, '/data3/leoni/LavAtmos/')
from lavatmos_goot_runner import container_lavatmos

sys.path.insert(1, '/data3/leoni/LavAtmos/ThermoEngine/LavAtmos/')
from paths import paths_importer

paths = paths_importer()

abundances={'C' : 2.6905e-04,'O': 4.3e-07,'H' : 9.9965e-01,'N' : 6.7585e-05,'S' : 1.3178e-05,'P' : 2.5695e-07}

#grid='/data3/leoni/condensates/grid_lavatmos_comp.csv'
grid='/data3/leoni/condensates/grid_lavatmos_Oabun.csv'
model="comp_5"

def make_abundfile(grid,model):
    compositions=pd.read_csv(grid,sep=',')
    compvals=compositions.loc[compositions['comp'] == model].to_dict('records')
    compdict=compvals[0]
    for i in abundances:
        if i in compdict:
            abundances[i]=float(compdict[i])
        else:
            continue
    return abundances


abundances = make_abundfile(grid,model)
print(abundances)

parameters = {

    # General parameters
    'run_name' : 'test_run1',

    # Melt parameters
    'lava_comp' : 'BSE_palm',
    'silicate_abundances' : 'lavatmos3', # 'lavatmos1', 'lavatmos2', 'manual'

    # Volatile parameters
    'P_volatile' : 10, # bar
    'oxygen_abundance' : 'degassed', # 'degassed', 'manual'
    'volatile_comp' :  abundances, # I used renormalised solar composition here
    'melt_fraction': 1.0

}


if __name__ == "__main__":

    temperatures=np.array([1500,1750,2000,2250,2500,2750,3000,3250,3500])
    vmrfile='/data3/leoni/LavAtmos/FastChem/fastchem3/output/boa_chem.dat'
    chem_df = pd.read_csv(vmrfile, sep=r'\s+')
    print(chem_df)
    for T in temperatures:
        lavatmos_instance = container_lavatmos(parameters)
        lavatmos_instance.run_lavatmos(T) # Tboa to update with HELIOS runs
        vmrfile='/data3/leoni/LavAtmos/FastChem/fastchem3/output/boa_chem.dat'
        df = pd.read_csv(vmrfile, sep=r'\s+')
        chem_df = pd.concat([chem_df, df], ignore_index=True)
        print(chem_df)

    chem_df = chem_df.iloc[1:].reset_index(drop=True) #drop first row since it just contains the setup data (random) and reindex the dataframe
    chem_df.to_csv('/data3/leoni/PROTEUS/output/test_thermo_OG.csv', index=False)
