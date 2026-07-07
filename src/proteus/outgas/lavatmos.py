# Function used to run LavAtmos 2.0
from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

# Local packages and paths
# sys.path.insert(1,'wkdir')
from proteus.utils.constants import (
    const_k,
    const_mp,
    const_Nav,
    electron_molar_mass,
    element_list,
    element_mmw,
    vap_list,
    vol_list,
)
from proteus.utils.coupler import UpdateStatusfile
from proteus.utils.helper import mol_to_ele

sys.path.append(os.getcwd())
if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)
# species db class comes from HELIOS code Kitzmann+2017


# Custom modules
class paths_importer:
    def __init__(self, dirs):
        """

        Change the paths as needed. If you don't change the dir structure,
        it should be enough to only change the wkdir.

        """

        # General directory structure
        self.lavatmos_dir = os.environ.get('LAVA_DIR')
        self.wkdir = os.environ.get('LAVA_DIR')

        log.info('Work directory set as: %s' % self.wkdir)
        self.input_dir = self.wkdir + 'input/'
        self.lava_comps = self.input_dir + 'lava_compositions/'

        # FastChem 3
        self.fastchem3_dir = os.environ.get('FC_DIR')

        self.fastchem3_input = os.path.join(self.fastchem3_dir, 'input/')
        self.fastchem3_config_template = os.path.join(
            self.wkdir, 'input/fastchem3/config_template.input'
        )
        self.element_abundance_template = os.path.join(
            self.wkdir, 'input/fastchem3/element_abundances/element_abundances_template2.dat'
        )
        self.species_data_file = os.path.join(self.fastchem3_dir, 'input/logK/logK.dat')
        self.species_data_file_cond = os.path.join(
            self.fastchem3_dir, 'input/logK/logK_condensates.dat'
        )

        # create directory for output element abundances created by lavatmos if it does not exist yet
        os.makedirs(dirs['output'] + '/element_abundances/', exist_ok=True)
        self.element_abundance_output = os.path.join(
            dirs['output'], 'element_abundances/element_abundances_output.dat'
        )
        os.makedirs(dirs['output'] + '/fastchem/', exist_ok=True)
        self.fastchem3_output = os.path.join(dirs['output'], 'fastchem/')
        self.output_dir = os.path.join(dirs['output'], 'fastchem/')

        self.janafdata = os.path.join(self.wkdir, 'data/')
        log.info('Output directory set as: %s' % self.output_dir)


class set_magmaproperties:
    def __init__(self, config: Config, hf_row: dict, volatile_comp, dirs: dict):
        """

        reading in properties from the output file

        """

        # General directory structure
        paths = paths_importer(dirs)
        # Import input
        if hf_row['T_magma'] > 1500:
            self.T_surf = hf_row['T_magma']
        else:
            self.T_surf = 1500
        self.P_volatile = hf_row['P_surf']
        self.melt_comp_name = 'BSE_palm'
        self.output_dir = paths.output_dir
        self.lavatmos_version = 'lavatmos3'
        self.run_name = 'proteus_run'
        self.melt_fraction = 1.0
        self.volatile_comp = volatile_comp
        # Saving volatile comp to csv for so that LavAtmos can read it later
        # need to find better way to read in volatile composition that from a parameter dictionary maybe ?
        log.info('volatile composition in set_magmaproperties', self.volatile_comp)


class Species_db(object):
    def __init__(self, name, fc_name, weight):
        self.name = name
        self.fc_name = fc_name  # designation in Fastchem
        self.weight = weight  # weight in AMU or g/mol


# (name, FastChem designation); species_lib is keyed by name (they always
# match, so there is no separate key column). Molecular weights are derived
# below from the FastChem formula and proteus.utils.constants.element_mmw, so
# this table only needs to record composition, not weight.
_SPECIES_TABLE = [
    # neutral molecules
    ('CO2', 'C1O2'),
    ('H2O', 'H2O1'),
    ('CO', 'C1O1'),
    ('O2', 'O2'),
    ('CH4', 'C1H4'),
    ('HCN', 'H1C1N1'),
    ('NH3', 'H3N1'),
    ('H2S', 'H2S1'),
    ('PH3', 'P1H3'),
    ('O3', 'O3'),
    ('O3_IR', 'O3'),
    ('O3_UV', 'O3'),
    ('NO', 'N1O1'),
    ('SO2', 'O2S1'),
    ('HS', 'H1S1'),
    ('H2', 'H2'),
    ('N2', 'N2'),
    ('SO', 'S1O1'),
    ('OH', 'O1H1'),
    ('COS', 'C1O1S1'),
    ('CS', 'C1S1'),
    ('HCHO', 'H1C1O1H1'),
    ('C2H4', 'C2H4'),
    ('C2H2', 'C2H2'),
    ('CH3', 'C1H3'),
    ('C3H', 'C3H1'),
    ('C2H', 'C2H1'),
    ('C2N2', 'C2N2'),
    ('C3O2', 'C3O2'),
    ('C4N2', 'C4N2'),
    ('C3', 'C3'),
    ('S2', 'S2'),
    ('S3', 'S3'),
    ('S2O', 'S2O1'),
    ('CS2', 'C1S2'),
    ('NO2', 'N1O2'),
    ('N2O', 'N2O1'),
    ('HNO3', 'H1N1O3'),
    ('HNO2', 'H1N1O2'),
    ('SO3', 'O3S1'),
    ('H2SO4', 'O4S1H2'),
    ('TiO', 'O1Ti1'),
    ('TiO2', 'O2Ti1'),
    ('TiS', 'S1Ti1'),
    ('TiH', 'H1Ti1'),
    ('VO', 'V1O1'),
    ('SiO', 'O1Si1'),
    ('AlO', 'Al1O1'),
    ('CaO', 'Ca1O1'),
    ('PO', 'P1O1'),
    ('PO2', 'P1O2'),
    ('SiH', 'H1Si1'),
    ('CaH', 'H1Ca1'),
    ('AlH', 'H1Al1'),
    ('MgH', 'H1Mg1'),
    ('CrH', 'H1Cr1'),
    ('NaH', 'H1Na1'),
    ('SiO2', 'O2Si1'),
    ('SiS', 'S1Si1'),
    ('PS', 'S1P1'),
    ('MgO', 'Mg1O1'),
    ('CN', 'N1C1'),
    ('H2CO', 'O1C1H2'),
    ('CH', 'H1C1'),
    ('PC', 'C1P1'),
    ('H2O2', 'O2H2'),
    ('NH', 'H1N1'),
    ('NS', 'N1S1'),
    ('PH', 'P1H1'),
    ('PN', 'P1N1'),
    ('C2', 'C2'),
    ('CaOH', 'Ca1O1H1'),
    ('FeH', 'Fe1H1'),
    ('FeO', 'Fe1O1'),
    ('KOH', 'K1O1H1'),
    ('SiH2', 'H2Si1'),
    ('SiH4', 'H4Si1'),
    ('NaOH', 'Na1O1H1'),
    ('NaO', 'Na1O1'),
    ('SiN', 'Si1N1'),
    ('AlN', 'Al1N1'),
    ('CaS', 'Ca1S1'),
    ('HO2', 'H1O2'),
    ('KO', 'K1O1'),
    ('MgS', 'Mg1S1'),
    ('FeO2H2', 'Fe1O2H2'),
    ('HAlO2', 'Al1H1O2'),
    ('Al2O', 'Al2O1'),
    ('AlS', 'Al1S1'),
    ('AlOH', 'Al1H1O1'),
    ('MgO2H2', 'Mg1O2H2'),
    ('MgOH', 'Mg1O1H1'),
    ('CaO2H2', 'Ca1O2H2'),
    # neutral atoms
    ('H', 'H'),
    ('He', 'He'),
    ('C', 'C'),
    ('N', 'N'),
    ('O', 'O'),
    ('F', 'F'),
    ('Na', 'Na'),
    ('Ne', 'Ne'),
    ('Ni', 'Ni'),
    ('Mg', 'Mg'),
    ('Mn', 'Mn'),
    ('Al', 'Al'),
    ('Ar', 'Ar'),
    ('Si', 'Si'),
    ('P', 'P'),
    ('S', 'S'),
    ('Cl', 'Cl'),
    ('K', 'K'),
    ('Ca', 'Ca'),
    ('Ti', 'Ti'),
    ('V', 'V'),
    ('Co', 'Co'),
    ('Cr', 'Cr'),
    ('Cu', 'Cu'),
    ('Fe', 'Fe'),
    ('Zn', 'Zn'),
]


def _fastchem_weight(fc_name):
    """Molecular weight [g/mol] of a FastChem formula string, from element_mmw."""
    atoms = mol_to_ele(fc_name)
    return sum(count * element_mmw[el] for el, count in atoms.items()) * 1000.0  # kg->g


species_lib = {
    name: Species_db(name=name, fc_name=fc_name, weight=_fastchem_weight(fc_name))
    for name, fc_name in _SPECIES_TABLE
}

# don't forget the electrons! (they may be tiny but they are important)
species_lib['e-'] = Species_db(name='e-', fc_name='e-', weight=electron_molar_mass)


class FO2shift:
    """models are taken from caliope. oxygen fugacity pO2 need to be in log10"""

    def __init__(self, model='oneill'):
        self.callmodel = getattr(self, model)

    def __call__(self, T, log10pO2):
        """Return log10 fO2"""
        return log10pO2 - self.callmodel(T)

    def fischer(self, T):
        """Fischer et al. (2011) IW"""
        return 6.94059 - 28.1808 * 1e3 / T

    def oneill(self, T):
        """O'Neill and Eggins (2002) IW"""
        return 2 * (-244118 + 115.559 * T - 8.474 * T * np.log(T)) / (np.log(10) * 8.31441 * T)


def read_in_element_fracs(input_path, time, parameters):
    # read file (skip comment line, split on whitespace)

    df = pd.read_csv(
        input_path + parameters['elementfile'],
        comment='#',  # ignore lines starting with #
        sep=r'\s+',  # split on any whitespace
        header=None,
    )
    # shutil.copy(input_path + parameters['elementfile'], "/data3/leoni/PROTEUS/elementfiles/element_abundances_{}.dat".format(str(time)))
    # make first column the headers and second column the data row
    df = pd.DataFrame([df[1].values], columns=df[0].values)

    df_frac = (10 ** (df - 12)).where(df != 0, 0)
    # dataframe has been multiplied by 1e20, so need to renormalise to get real fractions
    return df_frac / 1e20


def read_in_element_fracs_normalized(input_path):
    """constructs a dataframe with number element fractions of each element as computed by lavatmos

    input:
    - inputpath: path to the directory where the elemnt_abundance folder which contains the lavatmos output elements is located
        this file contains element abundances in fastchem format where ej=10^(xj-12)
    - time: time of the ucrrent timestep wt ahich the outgassing is computed

    output:
    - dataframe with normalised element fractions ej/etot

    """
    # read file (skip comment line, split on whitespace)

    df = pd.read_csv(
        input_path,
        comment='#',  # ignore lines starting with #
        sep=r'\s+',
        header=None,  # split on any whitespace
    )

    # make first column the headers and second column the data row
    abundance_dict = dict(zip(df[0], df[1]))
    for key in abundance_dict.keys():
        if abundance_dict[key] != 0.0:
            abundance_dict[key] = 10 ** (abundance_dict[key] - 12) / 1e20
        else:
            abundance_dict[key] = 0.0

    # add missing elements
    for e in element_list:
        if e not in abundance_dict.keys():
            abundance_dict[e] = 0.0

    total = sum(abundance_dict.values())

    norm_dict = {k: v / total for k, v in abundance_dict.items()}

    return norm_dict


def run_lavatmos(
    dirs: dict, config: Config, hf_row: dict, volatile_fracs: dict, first_iter: bool
):
    """

    This function imports the LavAtmos package from `LAVA_DIR` in-process and
    runs the melt-vapour equilibrium solve directly (no subprocess or
    container). The result is written to a csv file for diagnostics; the
    caller (`compute_silicate_outgassing`) instead reads the element
    abundances and FastChem output that LavAtmos writes as a side effect.

    """
    paths = paths_importer(dirs)
    sys.path.append(paths.lavatmos_dir)
    import lavatmos3

    melt_comp_path = paths.lava_comps
    Magma = set_magmaproperties(config, hf_row, volatile_fracs, dirs)

    # Import melt composition
    melt_comp_fname = melt_comp_path + Magma.melt_comp_name + '.csv'
    melt_comp_df = pd.read_csv(melt_comp_fname, names=['spec', 'abund'])
    melt_comp = {}
    for i in melt_comp_df.index:
        melt_comp[melt_comp_df['spec'].loc[i]] = float(melt_comp_df['abund'].loc[i])

    # Pressure used for melt activities [bar]
    P_melt = 0.01

    # Guess for fO2 [bar]
    fO2_initial_guess = 10 ** hf_row['log10_fO2_vapourise']

    # Lavatmos tolerance
    xatol = 1e-5

    system = lavatmos3.melt_vapor_system(paths)
    lavatmos_output = system.vaporise(
        Magma.T_surf,
        Magma.P_volatile,
        melt_comp,
        volatile_fracs,
        melt_fraction=Magma.melt_fraction,
        P_melt=P_melt,
        fO2_initial_guess=fO2_initial_guess,
        fO2_tries_from_last=bool(not first_iter),
        xatol=xatol,
    )

    # Save results
    output_name = f'{Magma.run_name}.csv'
    lavatmos_output.to_csv(paths.output_dir + output_name)


def compute_silicate_outgassing(dirs: dict, config: Config, hf_row: dict, first_iter: bool):
    """

    This function runs the Thermoengine module Lavatmos. Outgassing of refractory species
    are computed from a melt temperature and atmospheric pressure.

    Parameters:
        dirs : dict
            Dictionary of directories
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only

    """
    paths = paths_importer(dirs)

    log.debug('Computing rock vapourisation with LavAtmos')
    # set element fractions in atmosphere for lavatmos run
    input_eles = ['H', 'C', 'N', 'S', 'O']

    # lavatmos takes in the abudance fractions of element not mass fractions so divide by atomic number

    molfracs = {}
    nfrac = {'P': 0.0}
    total_mols = 0.0
    for e in input_eles:
        molfracs[e] = hf_row[e + '_kg_atm'] / species_lib[e].weight
        total_mols += molfracs[e]
    for e in input_eles:
        if total_mols > 0:
            if e == 'H':
                nfrac[e] = max(molfracs[e] / total_mols, 1e-9)
            else:
                nfrac[e] = molfracs[e] / total_mols
        else:
            if e == 'H':
                nfrac[e] = 1e-9
            else:
                nfrac[e] = 0.0
    log.debug('volatile element fractions going as input to lavatmos : %s', nfrac)
    log.debug('volatile pressure given to lavatmos : %.4e', hf_row['P_surf'])

    # running lavatmos
    run_lavatmos(dirs, config, hf_row, nfrac, first_iter)

    # convert the element abundances from lavatmos file to element fractions, normalized to unity
    element_fracs = read_in_element_fracs_normalized(paths.element_abundance_output)
    # import shutil
    # shutil.copy(paths.element_abundance_output, dirs['output'] + 'element_abundances/element_abundances_output'+'_'+ str(hf_row['Time']) +'_.dat')
    # elementfile = paths.element_abundance_output

    log.debug('element fraction after running lavatmos: %s' % element_fracs)
    # read in boa chemistry from last iteration of fastchem and lavatmos
    output_fc = paths.fastchem3_output
    if os.path.exists(output_fc):
        mmr_path = os.path.join(output_fc, 'boa_chem.dat')
    else:
        UpdateStatusfile(dirs, 27)
        raise RuntimeError('cannot find fastchem output from lavatmos loop!')

    # update abundances in output file for next calliope run
    new_atmos_abundances = pd.read_csv(mmr_path, sep=r'\s+')
    # make sure that the column names are consistent with the rest of the code
    if '#p(bar)' in new_atmos_abundances.columns:
        new_atmos_abundances.rename(columns={'#p(bar)': 'Pbar'}, inplace=True)
    if 'm(u)' in new_atmos_abundances.columns:
        new_atmos_abundances.rename(columns={'m(u)': 'mu'}, inplace=True)

    mu_outgassed = new_atmos_abundances['mu'][0]
    # compute density for the previous run with calliope output from hf_row:
    kg_per_particle = hf_row['atm_kg_per_mol'] / const_Nav

    # hf_row['P_surf'] is in bar; convert to Pascals for use in the ideal gas law
    # 1bar = 100 kPa
    P_surf_kPa = hf_row['P_surf'] * 100  # convert to kgPa
    rho_old = kg_per_particle * P_surf_kPa / (const_k * hf_row['T_magma'])
    M_atmo_old = hf_row['M_atm']

    # rho of armosphere after lavatmos
    # n=rho/mu*const_mp
    # 1bar = 100 kPa
    kg_pp_new = mu_outgassed * const_mp
    # log.debug('new mass per particle :%.4e'%kg_pp_new)
    P_new_kPa = new_atmos_abundances['Pbar'][0] * 100  # convert pressure to cgs
    # log.info('atmospheric pressure :%.2e'%new_atmos_abundances['Pbar'][0])
    rho_new = (
        kg_pp_new * P_new_kPa / (const_k * hf_row['T_magma'])
    )  # convert pressure in cgs to kg !
    # log.debug('new atmospheric density:%.4f'%rho_new)

    if M_atmo_old > 0.0:
        M_atmo_new = (
            M_atmo_old / rho_old
        ) * rho_new  # kg assuming volume does not change but only pressure
    else:  # compute shell volume and from there the new mass with the new density
        Vshell = (
            (4 / 3) * np.pi * (((hf_row['R_int'] + 1e2) ** 3) - (hf_row['R_int'] ** 3))
        )  # assume 1e2 m shell thickness (small shell)
        M_atmo_new = rho_new * Vshell

    log.info('new atmospheric mass:%.2e' % M_atmo_new)

    gas_list = vol_list + vap_list

    # do not update surface pressure!
    Poutgas = (
        new_atmos_abundances['Pbar'][0] - hf_row['P_surf']
    )  # comput ehow much silicates are outgassed
    log.info('pressure of outgassed species: %.4f' % Poutgas)
    log.info('pressure of volatiles before outgassing: %.4f' % hf_row['P_surf'])

    hf_row['P_vol'] = hf_row['P_surf']
    hf_row['P_vap'] = Poutgas

    for vol in gas_list:
        if vol in species_lib.keys():
            vol_key = species_lib[vol].fc_name
        else:
            vol_key = vol
        new_pp = new_atmos_abundances[vol_key][0] * new_atmos_abundances['Pbar'][0]
        hf_row[vol + '_vmr'] = new_atmos_abundances[vol_key][0]
        hf_row[vol + '_bar'] = new_pp

        hf_row[vol + '_kg_atm'] = (
            new_atmos_abundances[vol_key][0]
            * M_atmo_new
            * species_lib[vol].weight
            / mu_outgassed
        )  # kg
        hf_row[vol + '_kg_total'] = (
            hf_row[vol + '_kg_atm'] + hf_row[vol + '_kg_solid'] + hf_row[vol + '_kg_liquid']
        )

        hf_row[vol + '_mol_atm'] = hf_row[vol + '_kg_atm'] / hf_row['atm_kg_per_mol']
        hf_row[vol + '_mol_total'] = (
            hf_row[vol + '_mol_atm'] + hf_row[vol + '_mol_solid'] + hf_row[vol + '_mol_liquid']
        )

    mmw_elements = 0
    for e in element_fracs.keys():
        mmw_elements += element_fracs[e] * species_lib[e].weight

    for e in element_list:
        log.info('element: %s,  %s', e, element_fracs[e])
        log.debug(
            'total mass of element before updating with lavatmos: %s', hf_row[e + '_kg_atm']
        )
        if (
            e in input_eles
        ):  # oxygen should not be added to M_vaps, since it is not counted in M_eles       #and e != 'O':
            log.debug('volatile species, no need to update from lavatmos')
            continue
        else:
            hf_row[e + '_kg_atm'] = (
                element_fracs[e] * M_atmo_new * species_lib[e].weight / mmw_elements
            )

            # don't update total element mass, other wise mass between mantle and atmosphere not conserved -> planet keeps increasing with time
            hf_row[e + '_kg_total'] = (
                hf_row[e + '_kg_atm'] + hf_row[e + '_kg_solid'] + hf_row[e + '_kg_liquid']
            )
            hf_row['M_vaps'] += hf_row[e + '_kg_total']

    # saving new oxygen fugacity from lavatmos run, which is computed as log10 of the partial pressure of O2, to compare with the iron wustite buffer
    log10_fO2 = np.log10(new_atmos_abundances['O2'][0]) + np.log10(
        new_atmos_abundances['Pbar'][0]
    )  # is this really partical pressure ? Maybe this is actually abundances

    fO2_shift = FO2shift()
    hf_row['log10_fO2_vapourise'] = log10_fO2
    hf_row['log10_fO2_shift_vapourise'] = fO2_shift(hf_row['T_magma'], log10_fO2)
    hf_row['P_surf'] = new_atmos_abundances['Pbar'][0]

    log.debug(
        'log10 fO2 shift compared to IW buffer: %.6f' % hf_row['log10_fO2_shift_vapourise']
    )

    # update  hf_row['atm_kg_per_mol']
