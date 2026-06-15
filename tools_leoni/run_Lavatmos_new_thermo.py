from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(1, '/data3/leoni/PROTEUS/LavAtmos/')
import lavatmos3


class paths_importer:

    def __init__(self):

        '''

        Change the paths as needed. If you don't change the dir structure,
        it should be enough to only change the wkdir.

        '''

        # General directory structure
        self.wkdir = '/data3/leoni/PROTEUS/LavAtmos/'
        self.output_dir = self.wkdir+'output/'
        self.input_dir = self.wkdir+'input/'

        # Inputs
        self.lava_comps = self.input_dir+'lava_compositions/'

        # FastChem 3
        self.fastchem3_dir = os.environ.get("FC_DIR")
        print('fastchem directory:',self.fastchem3_dir)
        #self.fastchem3_dir = self.wkdir+'FastChem/fastchem3/'
        self.fastchem3_input = self.fastchem3_dir+'/input/'
        self.fastchem3_config_template = self.fastchem3_input+'config_template.input'
        self.element_abundances3 = self.fastchem3_input+'element_abundances/'
        self.janafdata=self.wkdir+'data/'


class set_magmaproperties:
    def __init__(self,parameters):

        '''

        reading in properties from the output file

        '''

        # General directory structure
        paths = paths_importer()
        self.P_volatile = parameters['P_volatile']
        self.melt_comp_name = parameters['lava_comp']
        self.output_dir = paths.output_dir
        self.lavatmos_version = parameters['silicate_abundances']
        self.run_name = parameters['run_name']
        self.melt_fraction = 1.0
        self.elementfile = 'element_abundances_output.dat'
        self.volatile_comp = parameters['volatile_comp']
        # Saving volatile comp to csv for so that LavAtmos can read it later
        #need to find better way to read in volatile composition that from a parameter dictionary maybe ?
        print('volatile composition in set_magmaproperties', self.volatile_comp)

paths = paths_importer()
#sys.path.insert(1, '/data3/leoni/PROTEUS/ThermoEngineLite/')


def get_input(grid,modelname):
    compositions=pd.read_csv(grid,sep=',')
    compvals=compositions.loc[compositions['comp'] == modelname].to_dict('records')
    compdict=compvals[0]
    for i in abundances:
        if i in compdict:
            abundances[i]=float(compdict[i])
    Pvol = 10
    Tsurf = 2000
    return abundances,Tsurf,Pvol


abundances={'C' : 2.6905e-04,'O': 4.3e-07,'H' : 9.9965e-01,'N' : 6.7585e-05,'S' : 1.3178e-05,'P' : 2.5695e-07}
parameters = {

    # General parameters
    'run_name' : 'run_thermoengine_lite',

    # Melt parameters
    'lava_comp' : 'BSE_palm',
    'silicate_abundances' : 'lavatmos2', # 'lavatmos1', 'lavatmos2', 'manual'

    # Volatile parameters
    'P_volatile' : 10, # bar
    'oxygen_abundance' : 'degassed', # 'degassed', 'manual'
    'volatile_comp' :  abundances, # I used renormalised solar composition here
    'elementfile': 'element_abundances_output.dat'
    }



#LavAtmos_params={'lava_comp':'BSE_palm','P_volatile':10,'grid':'evolution_output.csv','model':'model'}

class model:
    def __init__(self, abundances, temperature, pvol):
        self.abundances = abundances
        self.temperature = temperature
        self.pvol = pvol


if __name__ == "__main__":

    modelnames=["comp_5"]
    grid='/data3/leoni/condensates/grid_lavatmos_Oabun.csv'
    output_dir = '/data3/leoni/PROTEUS/output/tests_thermolite/'
    paths = paths_importer()
    melt_comp_path = paths.lava_comps


    for i, modelname in enumerate(modelnames):

        abundances,Tsurf,Pvol=get_input(grid,modelname)
        planet=model(abundances,Tsurf,Pvol)
        temperatures=np.array([1500,1750,2000,2250,2500,2750,3000,3250,3500])
        vmrfile='/data3/leoni/PROTEUS/AGNI/fastchem/output/boa_chem.dat'
        parameters.update({'volatile_comp':abundances,'P_volatile':Pvol})
        Magma = set_magmaproperties(parameters)
        chem_df = pd.read_csv(vmrfile, sep=r'\s+')

        comp_vol = pd.Series(parameters['volatile_comp'])
        comp_vol.to_csv(f'{paths.input_dir}volatile_comp.csv',\
                             header=['mole_fraction'])

        #lavatmos_instance = container_lavatmos(parameters)
        #lavatmos_instance.run_lavatmos(T) # Tboa to update with HELIOS runs

        melt_comp_fname = melt_comp_path + Magma.melt_comp_name +'.csv'
        melt_comp_df = pd.read_csv(melt_comp_fname,names=['spec','abund'])
        melt_comp = {}
        for i in melt_comp_df.index:
            melt_comp[melt_comp_df['spec'].loc[i]] = float(melt_comp_df['abund'].loc[i])
        #print('Magma composition:', melt_comp)

        for temp in temperatures:
            system = lavatmos3.melt_vapor_system(paths)
            lavatmos_output = system.vaporise(temp, Magma.P_volatile, melt_comp, abundances, Magma.elementfile, Magma.melt_fraction)
            df = pd.read_csv(vmrfile, sep=r'\s+')
            chem_df = pd.concat([chem_df, df], ignore_index=True)

        chem_df = chem_df.iloc[1:].reset_index(drop=True) #drop first row since it just contains the setup data (random) and reindex the dataframe
        chem_df.to_csv('/data3/leoni/PROTEUS/output/test_thermo_lite.csv', index=False)
