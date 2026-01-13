# Function used to run LavAtmos 2.0
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from proteus.config import Config


from proteus.utils.constants import element_list, gas_list

log = logging.getLogger("fwl."+__name__)

def run_lavatmos(config:Config,hf_row:dict):


    '''

    This function runs the Thermoengine module Lavatmos. Outgassing of refractory species
    are computed from a melt temperature and atmospheric pressure.

    Parameters:
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only

    '''
    import os
    import sys

    sys.path.insert(1, '/data3/leoni/LavAtmos')
    from lavatmos_goot_runner import container_lavatmos

    print(element_list, gas_list)
    lavatmos_dict={'P':0}

    #set element fractions in atmosphere for lavatmos run
    input_eles=['H','C','N','S']
    for e in input_eles:
        lavatmos_dict[e] = hf_row[e + "_kg_atm"]/hf_row["M_atm"]

    parameters = {

    # General parameters
    'run_name' : config.params.out.path,

    # Melt parameters
    'lava_comp' : 'BSE_palm',
    'silicate_abundances' : 'lavatmos2', # 'lavatmos1', 'lavatmos2', 'manual'

    # Volatile parameters
    'P_volatile' : hf_row['P_surf'], # bar
    'oxygen_abundance' : 'degassed', # 'degassed', 'manual'
    'volatile_comp' :  lavatmos_dict,
    }

    lavatmos_instance = container_lavatmos(parameters)
    lavatmos_instance.run_lavatmos(hf_row["T_magma"])

    #read in boa chemistry from last iteration of fastchem and lavatmos
    fastchempath=config.outgas.fastchempath
    if os.path.exists(fastchempath):
        mmr_path = os.path.join(fastchempath, 'boa_chem.dat')
    else:
        raise RuntimeError('cannot find fastchem output from lavatmos loop!')

    #update abundances in output file for next calliope run
    new_atmos_abundances=pd.read_csv(mmr_path,sep=r'\s+')
    hf_row['M_tot']=0

    #need to check here if this is correct still becuase fastchem outopout is in vmr .... maybe need to convert to mmrs first to scale to atmosphere mass ...
    for vol in gas_list:
        hf_row[vol + "_vmr"] = new_atmos_abundances[vol][0]
        hf_row[vol + "_kg_atm"] = new_atmos_abundances[vol][0] * hf_row['M_atm']
        hf_row[vol + "_kg_total"] = new_atmos_abundances[vol][0]+ hf_row[vol + "_kg_solid"] + hf_row[vol + "_kg_liquid"]

        hf_row['M_tot']+= hf_row[e + "_kg_total"]

    #elements are not considered as atomic species but just as inventory
    for e in element_list:
        hf_row[e + "_kg_atm"]=new_atmos_abundances[e][0] * hf_row['M_atm']
        hf_row[e + "_kg_atm"]=new_atmos_abundances[e][0]+ hf_row[e + "_kg_solid"] + hf_row[e + "_kg_liquid"]
