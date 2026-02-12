# Function used to run LavAtmos 2.0
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from proteus.config import Config


from proteus.utils.constants import element_list, vol_list

log = logging.getLogger("fwl."+__name__)


#species db class comes from HELIOS code Kitzmann+2017
class Species_db(object):

    def __init__(self, name, fc_name, weight):

        self.name = name
        self.fc_name = fc_name  # designation in Fastchem
        self.weight = weight  # weight in AMU or g/mol


species_lib = {}

# neutral molecules
species_lib["CO2"] = Species_db(name="CO2", fc_name="C1O2", weight=44.01)
species_lib["H2O"] = Species_db(name="H2O", fc_name="H2O1", weight=18.0153)
species_lib["CO"] = Species_db(name="CO", fc_name="C1O1", weight=28.01)
species_lib["O2"] = Species_db(name="O2", fc_name="O2", weight=31.9988)
species_lib["CH4"] = Species_db(name="CH4", fc_name="C1H4", weight=16.04)
species_lib["HCN"] = Species_db(name="HCN", fc_name="H1C1N1", weight=27.0253)
species_lib["NH3"] = Species_db(name="NH3", fc_name="N1H3", weight=17.031)
species_lib["H2S"] = Species_db(name="H2S", fc_name="H2S1", weight=34.081)
species_lib["PH3"] = Species_db(name="PH3", fc_name="P1H3", weight=33.99758)
species_lib["O3"] = Species_db(name="O3", fc_name="O3", weight=47.9982)
species_lib["O3_IR"] = Species_db(name="O3_IR", fc_name="O3", weight=47.9982)
species_lib["O3_UV"] = Species_db(name="O3_UV", fc_name="O3", weight=47.9982)
species_lib["NO"] = Species_db(name="NO", fc_name="N1O1", weight=30.01)
species_lib["SO2"] = Species_db(name="SO2", fc_name="S1O2", weight=64.066)
species_lib["HS"] = Species_db(name="HS", fc_name="H1S1", weight=33.073)
species_lib["H2"] = Species_db(name="H2", fc_name="H2", weight=2.01588)
species_lib["N2"] = Species_db(name="N2", fc_name="N2", weight=28.0134)
species_lib["SO"] = Species_db(name="SO", fc_name="S1O1", weight=48.0644)
species_lib["OH"] = Species_db(name="OH", fc_name="O1H1", weight=17.007)
species_lib["COS"] = Species_db(name="COS", fc_name="C1O1S1", weight=60.0751)
species_lib["CS"] = Species_db(name="CS", fc_name="C1S1", weight=44.0757)
species_lib["HCHO"] = Species_db(name="HCHO", fc_name="H1C1H1O1", weight=30.02598)
species_lib["C2H4"] = Species_db(name="C2H4", fc_name="C2H4", weight=28.05316)
species_lib["C2H2"] = Species_db(name="C2H2", fc_name="C2H2", weight=26.04)
species_lib["CH3"] = Species_db(name="CH3", fc_name="C1H3", weight=37.04004)
species_lib["C3H"] = Species_db(name="C3H", fc_name="C3H1", weight=37.04004)
species_lib["C2H"] = Species_db(name="C2H", fc_name="C2H1", weight=25.02934)
species_lib["C2N2"] = Species_db(name="C2N2", fc_name="C2N2", weight=52.0348)
species_lib["C3O2"] = Species_db(name="C3O2", fc_name="C3O2", weight=68.0309)
species_lib["C4N2"] = Species_db(name="C4N2", fc_name="C4N2", weight=76.0562)
species_lib["C3"] = Species_db(name="C3", fc_name="C3", weight=36.0321)
species_lib["S2"] = Species_db(name="S2", fc_name="S2", weight=64.13)
species_lib["S3"] = Species_db(name="S3", fc_name="S3", weight=96.195)
species_lib["S2O"] = Species_db(name="S2O", fc_name="S2O1", weight=80.1294)
species_lib["CS2"] = Species_db(name="CS2", fc_name="C1S2", weight=76.1407)
species_lib["NO2"] = Species_db(name="NO2", fc_name="N1O2", weight=46.0055)
species_lib["N2O"] = Species_db(name="N2O", fc_name="N2O1", weight=44.013)
species_lib["HNO3"] = Species_db(name="HNO3", fc_name="H1N1O3", weight=63.01)
species_lib["HNO2"] = Species_db(name="HNO2", fc_name="H1N1O2", weight=47.01)
species_lib["SO3"] = Species_db(name="SO3", fc_name="S1O3", weight=80.066)
species_lib["H2SO4"] = Species_db(name="H2SO4", fc_name="H2S1O4", weight=98.0785)
species_lib["TiO"] = Species_db(name="TiO", fc_name="Ti1O1", weight=63.866)
species_lib["TiO2"] = Species_db(name="TiO2", fc_name="Ti1O2", weight=79.87)
species_lib["TiS"] = Species_db(name="TiS", fc_name="Ti1S1", weight=79.932)
species_lib["TiH"] = Species_db(name="TiH", fc_name="Ti1H1", weight=48.87)
species_lib["VO"] = Species_db(name="VO", fc_name="V1O1", weight=66.9409)
species_lib["SiO"] = Species_db(name="SiO", fc_name="Si1O1", weight=44.08)
species_lib["AlO"] = Species_db(name="AlO", fc_name="Al1O1", weight=42.98)
species_lib["CaO"] = Species_db(name="CaO", fc_name="Ca1O1", weight=56.0774)
species_lib["PO"] = Species_db(name="PO", fc_name="P1O1", weight=46.97316)
species_lib["PO2"] = Species_db(name="PO2", fc_name="P1O2", weight=62.97256)
species_lib["SiH"] = Species_db(name="SiH", fc_name="Si1H1", weight=29.09344)
species_lib["CaH"] = Species_db(name="CaH", fc_name="Ca1H1", weight=41.085899)
species_lib["AlH"] = Species_db(name="AlH", fc_name="Al1H1", weight=27.9889)
species_lib["MgH"] = Species_db(name="MgH", fc_name="Mg1H1", weight=25.3129)
species_lib["CrH"] = Species_db(name="CrH", fc_name="Cr1H1", weight=53.0040)
species_lib["NaH"] = Species_db(name="NaH", fc_name="Na1H1", weight=23.99771)
species_lib["SiO2"] = Species_db(name="SiO2", fc_name="Si1O2", weight=60.08)
species_lib["SiS"] = Species_db(name="SiS", fc_name="Si1S1", weight=60.15)
species_lib["PS"] = Species_db(name="PS", fc_name="P1S1", weight=63.0388)
species_lib["MgO"] = Species_db(name="MgO", fc_name="Mg1O1", weight=40.30440)
species_lib["CN"] = Species_db(name="CN", fc_name="C1N1", weight=26.0174)
species_lib["H2CO"] = Species_db(name="H2CO", fc_name="H2C1O1", weight=30.027)
species_lib["CH"] = Species_db(name="CH", fc_name="C1H1", weight=13.019)
species_lib["PC"] = Species_db(name="PC", fc_name="P1C1", weight=42.984)
species_lib["H2O2"] = Species_db(name="H2O2", fc_name="H2O2", weight=34.016)
species_lib["NH"] = Species_db(name="NH", fc_name="N1H1", weight=15.015)
species_lib["NS"] = Species_db(name="NS", fc_name="N1S1", weight=46.067)
species_lib["PH"] = Species_db(name="PH", fc_name="P1H1", weight=31.9817)
species_lib["PN"] = Species_db(name="PN", fc_name="P1N1", weight=44.98)
species_lib["HS"] = Species_db(name="HS", fc_name="H1S1", weight=33.068)
species_lib["C2"] = Species_db(name="C2", fc_name="C2", weight=24.022)
species_lib["CaOH"] = Species_db(name="CaOH", fc_name="Ca1O1H1", weight=69.096)
species_lib["FeH"] = Species_db(name="FeH", fc_name="Fe1H1", weight=56.853)
species_lib["FeO"] = Species_db(name="FeO", fc_name="Fe1O1", weight=71.844)
species_lib["KOH"] = Species_db(name="KOH", fc_name="K1O1H1", weight=56.109)
species_lib["SiH2"] = Species_db(name="SiH2", fc_name="Si1H2", weight=30.10138)
species_lib["SiH4"] = Species_db(name="SiH4", fc_name="Si1H4", weight=64.177)
species_lib["N2O"] = Species_db(name="N2O", fc_name="N2O1", weight=44.014)
species_lib["NaOH"] = Species_db(name="NaOH", fc_name="Na1O1H1", weight=54.004)
species_lib["N2"] = Species_db(name="N2", fc_name="N2", weight=28.014)
species_lib["NaO"] = Species_db(name="NaO", fc_name="Na1O1", weight=38.99)
species_lib["SiN"] = Species_db(name="SiN", fc_name="Si1N1", weight=74.152)
species_lib["AlN"] = Species_db(name="AlN", fc_name="Al1N1", weight=40.988)
species_lib["CaS"] = Species_db(name="CaS", fc_name="Ca1S1", weight=72.143)
species_lib["HO2"] = Species_db(name="HO2", fc_name="H1O2", weight=33.007)
species_lib["KO"] = Species_db(name="KO", fc_name="K1O1", weight=55.098)
species_lib["MgS"] = Species_db(name="MgS", fc_name="Mg1S1", weight=56.37)
species_lib["NaO"] = Species_db(name="NaO", fc_name="Na1O1", weight=38.989)
species_lib["FeO2H2"] = Species_db(name="FeO2H2", fc_name="Fe1O2H2", weight=89.86)
species_lib["HAlO2"] = Species_db(name="HAlO2", fc_name="H1Al1O2", weight= 59.99)
species_lib["Al2O"] = Species_db(name="Al2O", fc_name="Al2O1", weight=69.96)
species_lib["AlS"] = Species_db(name="AlS", fc_name="Al1S1", weight= 59.05)
species_lib["AlOH"] = Species_db(name="AlOH", fc_name="H1Al1O1", weight=43.99)
species_lib["MgO2H2"] = Species_db(name="MgO2H2", fc_name="Mg1O2H2", weight=58.32)
species_lib["MgOH"] = Species_db(name="MgOH", fc_name="Mg1O1H1", weight=41.31)
species_lib["CaO2H2"] = Species_db(name="CaO2H2", fc_name="Ca1O2H2", weight=74.09)
species_lib["VO"] = Species_db(name="VO", fc_name="V1O1", weight=66.94)
# neutral atoms
species_lib["H"] = Species_db(name="H", fc_name="H", weight=1.007825)
species_lib["He"] = Species_db(name="He", fc_name="He", weight=4.0026)
species_lib["C"] = Species_db(name="C", fc_name="C", weight=12.0096)
species_lib["N"] = Species_db(name="N", fc_name="N", weight=14.007)
species_lib["O"] = Species_db(name="O", fc_name="O", weight=15.999)
species_lib["F"] = Species_db(name="F", fc_name="F", weight=18.9984)
species_lib["Na"] = Species_db(name="Na", fc_name="Na", weight=22.989769)
species_lib["Ne"] = Species_db(name="Ne", fc_name="Ne", weight=20.1797)
species_lib["Ni"] = Species_db(name="Ni", fc_name="Ni", weight=58.6934)
species_lib["Mg"] = Species_db(name="Mg", fc_name="Mg", weight=24.305)
species_lib["Mn"] = Species_db(name="Mn", fc_name="Mn", weight=54.938044)
species_lib["Al"] = Species_db(name="Al", fc_name="Al", weight=26.9815385)
species_lib["Ar"] = Species_db(name="Ar", fc_name="Ar", weight=39.948)
species_lib["Si"] = Species_db(name="Si", fc_name="Si", weight=28.085)
species_lib["P"] = Species_db(name="P", fc_name="P", weight=30.973761998)
species_lib["S"] = Species_db(name="S", fc_name="S", weight=32.06)
species_lib["Cl"] = Species_db(name="Cl", fc_name="Cl", weight=35.45)
species_lib["K"] = Species_db(name="K", fc_name="K", weight=39.0983)
species_lib["Ca"] = Species_db(name="Ca", fc_name="Ca", weight=40.078)
species_lib["Ti"] = Species_db(name="Ti", fc_name="Ti", weight=47.867)
species_lib["V"] = Species_db(name="V", fc_name="V", weight=50.9415)
species_lib["Co"] = Species_db(name="Co", fc_name="Co", weight=58.933194)
species_lib["Cr"] = Species_db(name="Cr", fc_name="Cr", weight=51.9961)
species_lib["Cu"] = Species_db(name="Cu", fc_name="Cu", weight=63.546)
species_lib["Fe"] = Species_db(name="Fe", fc_name="Fe", weight=55.845)
species_lib["Zn"] = Species_db(name="Zn", fc_name="Zn", weight=65.38)

# don't forget the electrons! (they may be tiny but they are important)
species_lib["e-"] = Species_db(name="e-", fc_name="e-", weight=5.4858e-4)

mp = 1.6726231e-21 #kg
kB = 1.38064e-23 #JK-1
particles_per_mol = 6.02214076e+23

class FO2shift:
    ''' models are taken from caliope. oxygen fugacity pO2 need to be in log10 '''
    def __init__(self, model='oneill'):
        self.callmodel = getattr(self, model)

    def __call__(self, T, log10pO2):
        '''Return log10 fO2'''
        return log10pO2 - self.callmodel(T)

    def fischer(self, T):
        '''Fischer et al. (2011) IW'''
        return 6.94059 -28.1808*1E3/T

    def oneill(self, T):
        '''O'Neill and Eggins (2002) IW'''
        return 2*(-244118+115.559*T-8.474*T*np.log(T))/(np.log(10)*8.31441*T)





def run_lavatmos(config:Config,hf_row:dict):


    '''

    This function runs the Thermoengine module Lavatmos. Outgassing of refractory species
    are computed from a melt temperature and atmospheric pressure.

    Parameters:
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
        obudget: oxygen budget already present in the atmosphere before running the outgassing

    '''
    import os
    import sys

    import numpy as np

    sys.path.insert(1, '/data3/leoni/LavAtmos')
    from lavatmos_goot_runner import container_lavatmos

    gas_list= vol_list + config.outgas.vaplist
    lavatmos_dict={'P':0.0}

    #set element fractions in atmosphere for lavatmos run
    input_eles=['H','C','N','S','O']

    total_weight = species_lib['H'].weight + species_lib['C'].weight + species_lib['N'].weight + species_lib['O'].weight + species_lib['S'].weight
    print('total_weight: ',total_weight)
    #lavatmos takes in the abundnace fractions of element not mass fractions so divide by atomic number
    for e in input_eles:
        lavatmos_dict[e] = (hf_row[e + "_kg_atm"] / hf_row["M_atm"]) * (total_weight / ( 5 * species_lib[e].weight))
        print(e,lavatmos_dict[e])

    parameters = {

    # General parameters
    'run_name' : config.params.out.path,

    # Melt parameters
    'lava_comp' : 'BSE_palm',
    'silicate_abundances' : 'lavatmos3', # 'lavatmos1', 'lavatmos2', 'manual'

    # Volatile parameters
    'P_volatile' : hf_row['P_surf'], # bar
    'oxygen_abundance' : 'degassed', # 'degassed', 'manual',
    'volatile_comp' :  lavatmos_dict,
    'melt_fraction' : 1.0
    }

    #compute density for the previous run with calliope output from hf_row:
    kg_per_particle = hf_row['atm_kg_per_mol']/particles_per_mol

    if hf_row['T_surf'] == 0.0 : #make sure that not zero surface temperature in first iteration
        Tsurf=hf_row["T_magma"]
    else:
        Tsurf=hf_row['T_surf']
    rho_old = kg_per_particle * hf_row['P_surf']/(kB*Tsurf)

    M_atmo_old = hf_row['M_atm']
    #M_total = hf_row['M_core'] + hf_row['M_atm'] + hf_row['M_mantle']
   # M_mantle_old = hf_row['M_mantle']

    #log.info('Mass of the planet mantle before running lavatmos: %s'% M_mantle_old)
    #log.info('Mass of the planet atmosphere before running lavatmos: %s'% M_atmo_old)
    #log.info('Mass of the total planet before running lavatmos: %s'% M_total)


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
    mu_outgassed=new_atmos_abundances['mu'][0]

    #rho of armosphere after lavatmos
    #n=rho/mu*mp
    rho_new =  kg_per_particle * new_atmos_abundances['Pbar'][0]/(kB*Tsurf)
    M_atmo_new = M_atmo_old/rho_old  * rho_new #kg assuming volum edoes not change

    for vol in gas_list:
        new_pp = new_atmos_abundances[vol][0]* new_atmos_abundances['Pbar'][0]
        hf_row[vol + "_bar"] = new_pp
        #here need to update it in terms of pressure as well, since this is input for calliiope
        hf_row[vol + "_vmr"] = new_atmos_abundances[vol][0]
        hf_row[vol + "_kg_atm"] = new_atmos_abundances[vol][0] * M_atmo_new * species_lib[vol].weight/mu_outgassed #kg
        hf_row[vol + "_kg_tot"] = hf_row[vol + "_kg_atm"] + hf_row[vol + "_kg_solid"] + hf_row[vol + "_kg_liquid"]

    #elements are not considered as atomic species but just as inventory
    for e in element_list:
        hf_row[e + "_kg_atm"]=new_atmos_abundances[e][0] * M_atmo_new * species_lib[e].weight/mu_outgassed
        hf_row[e + "_kg_atm"]=new_atmos_abundances[e][0]+ hf_row[e + "_kg_solid"] + hf_row[e + "_kg_liquid"]



    #saving new oxygen fugacity for calliope
    log10_fO2 = np.log10(new_atmos_abundances['O2'][0]) + np.log10(new_atmos_abundances['Pbar'][0])  # is this really partical pressure ? Maybe this is actually abundances

    fO2_shift = FO2shift()
    hf_row['fO2_shift'] = fO2_shift(hf_row["T_magma"], log10_fO2)

    print('shift compared to iron wustite buffer:', hf_row['fO2_shift'])




    #in above, M_atmo should be updated. we can use this to uodate also the mantle mass of the planet
   # M_mantle_new = M_total - hf_row['M_core'] - M_atmo_new

    #M_mantle_liquid_new = M_mantle_new - M_solid_old
    #hf_row['M_mantle_liquid'] =  M_mantle_liquid_new
    #hf_row['M_mantle'] = M_mantle_new

    #log.info('Mass of the planet mantle after running lavatmos: %s'% M_mantle_new)
    #log.info('Mass of the planet atmosphere after running lavatmos: %s'% M_atmo_new)
