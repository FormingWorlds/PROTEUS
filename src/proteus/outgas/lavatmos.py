# Function used to run LavAtmos 2.0
from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from calliope.oxygen_fugacity import OxygenFugacity

# Local packages and paths
# sys.path.insert(1,'wkdir')
from proteus.utils.constants import (
    const_Nav,
    electron_molar_mass,
    element_list,
    element_mmw,
    gas_list,
    noble_gases,
    vap_list,
)
from proteus.utils.coupler import UpdateStatusfile
from proteus.utils.helper import mol_to_ele

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

# Constants
H_NUMFRAC_FLOOR = 1e-9  # minimum H number fraction in LavAtmos input
SURFACE_MELT_FRAC = 1.0  # melt fraction at the surface (LavAtmos input)


# Custom modules
class paths_importer:
    def __init__(self, dirs):
        """

        Change the paths as needed. If you don't change the dir structure,
        it should be enough to only change the wkdir.

        """

        # General directory structure
        lava_dir = os.environ.get('LAVA_DIR')
        if lava_dir is None:
            raise ValueError('Environment variable "LAVA_DIR" is not set')
        fastchem_dir = os.environ.get('FC_DIR')
        if fastchem_dir is None:
            raise ValueError('Environment variable "FC_DIR" is not set')

        # Normalise so a trailing slash in the env var does not change any
        # derived path (os.path.normpath strips it).
        self.lavatmos_dir = os.path.normpath(lava_dir) + '/'
        self.wkdir = self.lavatmos_dir

        log.debug('LavAtmos Work directory set as: %s' % self.wkdir)
        self.input_dir = os.path.join(self.wkdir, 'input') + '/'
        self.lava_comps = os.path.join(self.input_dir, 'lava_compositions') + '/'

        # Sanity check - does lava_comps exist?
        if not os.path.exists(self.lava_comps):
            raise ValueError(
                f"Lava compositions directory '{self.lava_comps}' does not exist. "
                'Please check your LAVA_DIR environment variable.'
            )

        # FastChem 3
        self.fastchem3_dir = os.path.normpath(fastchem_dir) + '/'

        self.fastchem3_input = os.path.join(self.fastchem3_dir, 'input')
        self.fastchem3_config_template = os.path.join(
            self.input_dir, 'fastchem3', 'config_template.input'
        )
        self.element_abundance_template = os.path.join(
            self.input_dir,
            'fastchem3',
            'element_abundances',
            'element_abundances_template2.dat',
        )
        self.species_data_file = os.path.join(self.fastchem3_dir, 'input', 'logK', 'logK.dat')
        self.species_data_file_cond = os.path.join(
            self.fastchem3_dir, 'input', 'logK', 'logK_condensates.dat'
        )

        # create directory for output element abundances created by lavatmos if it does not exist yet
        os.makedirs(os.path.join(dirs['output'], 'element_abundances'), exist_ok=True)
        self.element_abundance_output = os.path.join(
            dirs['output'], 'element_abundances', 'element_abundances_output.dat'
        )
        os.makedirs(os.path.join(dirs['output'], 'fastchem'), exist_ok=True)
        # LavAtmos concatenates fastchem3_output with filenames (e.g. `+ 'config.input'`),
        # so keep the trailing separator on these two directory paths.
        self.fastchem3_output = os.path.join(dirs['output'], 'fastchem', '')
        self.output_dir = os.path.join(dirs['output'], 'fastchem', '')

        self.janafdata = os.path.join(self.wkdir, 'data')
        log.debug('LavAtmos output directory set as: %s' % self.output_dir)


class set_magmaproperties:
    def __init__(self, config: Config, hf_row: dict, volatile_comp, dirs: dict):
        """

        reading in properties from the output file

        """

        # General directory structure
        paths = paths_importer(dirs)

        # Import input from config
        self.T_surf = max(config.outgas.lavatmos.T_min, hf_row['T_magma'])
        self.P_volatile = hf_row['P_surf']
        self.melt_comp_name = config.outgas.lavatmos.melt_comp_name
        self.output_dir = paths.output_dir
        self.run_name = 'proteus_run'
        self.melt_fraction = SURFACE_MELT_FRAC
        self.volatile_comp = volatile_comp


class Species_db(object):
    def __init__(self, name, fc_name, weight):
        self.name = name
        self.fc_name = fc_name  # designation in Fastchem
        self.weight = weight  # weight in AMU or g/mol


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
    ('TiN', 'N1Ti1'),
    ('VO', 'V1O1'),
    ('SiO', 'O1Si1'),
    ('AlO', 'Al1O1'),
    ('AlN', 'Al1N1'),
    ('AlO2', 'Al1O2'),
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
    ('MgN', 'Mg1N1'),
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
    ('CaOH', 'Ca1H1O1'),
    ('FeH', 'Fe1H1'),
    ('FeO', 'Fe1O1'),
    ('FeS', 'Fe1S1'),
    ('KOH', 'H1K1O1'),
    ('KH', 'H1K1'),
    ('KO', 'K1O1'),
    ('SiH2', 'H2Si1'),
    ('SiH4', 'H4Si1'),
    ('NaOH', 'H1Na1O1'),
    ('NaO', 'Na1O1'),
    ('SiN', 'N1Si1'),
    ('CaS', 'Ca1S1'),
    ('HO2', 'H1O2'),
    ('MgS', 'Mg1S1'),
    ('FeO2H2', 'Fe1H2O2'),
    ('HAlO2', 'Al1H1O2'),
    ('Al2O', 'Al2O1'),
    ('AlS', 'Al1S1'),
    ('AlOH', 'Al1H1O1'),
    ('MgO2H2', 'H2Mg1O2'),
    ('MgOH', 'H1Mg1O1'),
    ('CaO2H2', 'Ca1H2O2'),
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

# Back-fill any element_list/gas_list entry not already covered above (e.g. a
# noble gas added to those lists without a matching curated table row). This
# is a best-effort fallback: it assumes name is itself a valid FastChem
# formula, which only holds for single-symbol elements -- multi-atom species
# should get a real row in _SPECIES_TABLE instead, since FastChem's element
# ordering in fc_name is not generally derivable from the plain formula.
for _name in element_list + gas_list:
    if _name not in species_lib:
        species_lib[_name] = Species_db(
            name=_name, fc_name=_name, weight=_fastchem_weight(_name)
        )


def read_in_element_fracs(input_path, time, parameters):
    # read file (skip comment line, split on whitespace)

    df = pd.read_csv(
        input_path + parameters['elementfile'],
        comment='#',  # ignore lines starting with #
        sep=r'\s+',  # split on any whitespace
        header=None,
    )
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
            # lavatmos abundances are scaled by 1e20
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
    caller (`run_vapourisation`) instead reads the element
    abundances and FastChem output that LavAtmos writes as a side effect.

    """

    # Get paths from environment variables
    try:
        paths = paths_importer(dirs)
    except ValueError as e:
        UpdateStatusfile(dirs, 27)
        raise RuntimeError(str(e))

    # Import lavatmos
    lavatmos_dir = str(paths.lavatmos_dir)
    if lavatmos_dir not in sys.path:
        sys.path.append(lavatmos_dir)
    try:
        import lavatmos3
    except ImportError as e:
        UpdateStatusfile(dirs, 27)
        raise RuntimeError(
            f"Failed to import LavAtmos from '{paths.lavatmos_dir}'. "
            'Please check your LAVA_DIR environment variable.'
        ) from e

    Magma = set_magmaproperties(config, hf_row, volatile_fracs, dirs)

    # Import melt composition
    melt_comp_fname = os.path.join(paths.lava_comps, Magma.melt_comp_name + '.csv')
    melt_comp_df = pd.read_csv(melt_comp_fname, names=['spec', 'abund'])
    melt_comp = {}
    for i in melt_comp_df.index:
        melt_comp[melt_comp_df['spec'].loc[i]] = float(melt_comp_df['abund'].loc[i])

    # Pressure used for melt activities [bar]
    P_melt = config.outgas.lavatmos.P_melt

    # Guess for fO2 [bar]
    fO2_initial_guess = 10 ** hf_row['log10_fO2_vapourise']

    # Lavatmos tolerance
    xatol = config.outgas.lavatmos.xatol

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
    lavatmos_output.to_csv(os.path.join(paths.output_dir, output_name))


def run_vapourisation(dirs: dict, config: Config, hf_row: dict, first_iter: bool):
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

    # set element fractions in atmosphere for lavatmos run.
    input_eles = ['H', 'C', 'N', 'S', 'O'] + noble_gases
    hf_row['M_vaps'] = 0.0

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
                nfrac[e] = max(molfracs[e] / total_mols, H_NUMFRAC_FLOOR)
            else:
                nfrac[e] = molfracs[e] / total_mols
        else:
            if e == 'H':
                nfrac[e] = H_NUMFRAC_FLOOR
            else:
                nfrac[e] = 0.0
    log.debug('volatile element fractions going as input to lavatmos : %s', nfrac)
    log.debug('volatile pressure given to lavatmos : %.4e', hf_row['P_surf'])

    # running lavatmos
    run_lavatmos(dirs, config, hf_row, nfrac, first_iter)

    # convert the element abundances from lavatmos file to element fractions, normalized to unity
    element_fracs = read_in_element_fracs_normalized(paths.element_abundance_output)
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

    # Mean particle mass of the combined (volatile + rock-vapour) atmosphere.
    kg_pp_new = mu_outgassed / const_Nav * 1e-3  # kg per particle
    log.debug('new kg per particle: %.4e' % kg_pp_new)

    # New atmospheric mass from the hydrostatic relation
    area = 4.0 * np.pi * hf_row['R_int'] ** 2
    P_surf_new_Pa = new_atmos_abundances['Pbar'][0] * 1e5  # bar -> Pa
    M_atmo_new = P_surf_new_Pa * area / hf_row['gravity']  # kg

    # Split the new total surface pressure into a volatile part (P_vol) and a
    # rock-vapour part (P_vap). P_vap is the excess of the LavAtmos+FastChem total
    # over the volatile-only pressure that went in; it cannot be negative.
    P_surf_new = new_atmos_abundances['Pbar'][0]
    Poutgas = P_surf_new - hf_row['P_surf']  # rock-vapour pressure contribution
    if Poutgas < 0.0:
        log.warning(
            'rock-vapour pressure computed negative (%.3e bar); clamping to zero', Poutgas
        )
        Poutgas = 0.0
    log.debug('pressure of outgassed species: %.4f' % Poutgas)
    log.debug('pressure of volatiles before outgassing: %.4f' % hf_row['P_surf'])

    hf_row['P_vap'] = Poutgas
    hf_row['P_vol'] = P_surf_new - Poutgas  # == old P_surf when Poutgas >= 0

    # Update the tracked atmospheric MMW = volatile + rock-vapour
    hf_row['atm_kg_per_mol'] = mu_outgassed * 1e-3

    mmw_elements = 0
    for e in element_fracs.keys():
        mmw_elements += element_fracs[e] * species_lib[e].weight

    for vol in gas_list:
        if vol in species_lib.keys():
            vol_key = species_lib[vol].fc_name
        else:
            vol_key = vol
        if vol_key not in new_atmos_abundances.columns:
            # FastChem did not emit this species. This happens for noble gases
            # absent from the FastChem element data (e.g. Kr/Xe are not in the
            # bundled logK set). Keep the species' prior value instead of
            # crashing, and note it.
            log.warning(
                'species %s (%s) not present in FastChem output; keeping prior value',
                vol,
                vol_key,
            )
            continue
        new_pp = new_atmos_abundances[vol_key][0] * new_atmos_abundances['Pbar'][0]
        hf_row[vol + '_vmr'] = new_atmos_abundances[vol_key][0]
        hf_row[vol + '_bar'] = new_pp

        hf_row[vol + '_kg_atm'] = (
            new_atmos_abundances[vol_key][0]
            * M_atmo_new
            * species_lib[vol].weight
            / mu_outgassed
        )  # kg
        hf_row[vol + '_mol_atm'] = hf_row[vol + '_kg_atm'] / hf_row['atm_kg_per_mol']

        if vol in vap_list:
            hf_row[vol + '_kg_total'] = (
                0.0  # ensures that elements Na and K, si, Ti are not added to the planet mass computations in interior model
            )
            # but saved separately later in M_vaps
            hf_row[vol + '_mol_total'] = 0.0
        else:
            hf_row[vol + '_kg_total'] = (
                hf_row[vol + '_kg_atm'] + hf_row[vol + '_kg_solid'] + hf_row[vol + '_kg_liquid']
            )
            hf_row[vol + '_mol_total'] = (
                hf_row[vol + '_mol_atm']
                + hf_row[vol + '_mol_solid']
                + hf_row[vol + '_mol_liquid']
            )

    for e in element_list:
        log.debug('element frac:  %s,  %s', e, element_fracs[e])
        if e in input_eles:
            hf_row[e + '_kg_atm'] = (
                                element_fracs[e] * M_atmo_new * species_lib[e].weight / mmw_elements
                            )
            if e == 'O':
                Omass_after_outgas = (
                    element_fracs[e] * M_atmo_new * species_lib[e].weight / mmw_elements
                )
            continue
        else:
            if e not in gas_list:
                hf_row[e + '_kg_atm'] = (
                    element_fracs[e] * M_atmo_new * species_lib[e].weight / mmw_elements
                )
            hf_row['M_vaps'] += (
                element_fracs[e] * M_atmo_new * species_lib[e].weight / mmw_elements
            )  # don't use hf_row[e + '_kg_atm'] becuase this only acounts for e.g. Si in atomic gas form not total Si

    # saving new oxygen fugacity from lavatmos run, which is computed as log10 of the partial pressure of O2, to compare with the iron wustite buffer

    hf_row['M_vaps'] += (
        Omass_after_outgas - hf_row['O_kg_atm']
    )  # add outgassed oxygen mass to elemental vapour species mass
    log.debug(
        'added oxygen from outagssing to initial O budget in atmosphere [kg]: %.4e ',
        Omass_after_outgas - hf_row['O_kg_atm'],
    )

    pO2 = new_atmos_abundances['O2'][0]
    log.debug('O2 patial pressure  very small: %.3e', pO2)
    if pO2 < 1e-20:
        log.debug('O2 patial pressure smaller than 1e-12')
        pO2 = 1e-20
    if new_atmos_abundances['Pbar'][0] < 1e-10:
        psurf = 1e-10
    else:
        psurf = new_atmos_abundances['Pbar'][0]
    log10_fO2 = np.log10(pO2) + np.log10(psurf)

    # OxygenFugacity(T, fO2_shift=0) returns the absolute buffer value
    # IW(T); the shift relative to that buffer is log10_fO2 - IW(T).
    iw_buffer = OxygenFugacity(model=config.outgas.lavatmos.fO2_buffer_model)
    hf_row['log10_fO2_vapourise'] = log10_fO2
    hf_row['log10_fO2_shift_vapourise'] = log10_fO2 - iw_buffer(hf_row['T_magma'])
    hf_row['P_surf'] = new_atmos_abundances['Pbar'][0]
    hf_row['M_atm'] = M_atmo_new

    log.debug(
        'log10 fO2 shift compared to IW buffer: %.6f' % hf_row['log10_fO2_shift_vapourise']
    )

    mask = [hf_row[s + '_vmr'] for s in vap_list]
    for i in np.argsort(mask)[::-1]:
        s = vap_list[i]
        _p = hf_row[s + '_bar']
        _x = hf_row[s + '_vmr']
        _s = '    %-6s     = %-9.2f bar (%.2e VMR)' % (s, _p, _x)
        if _p > 0.01:
            log.info(_s)
        else:
            # don't spam log with species of negligible abundance
            log.debug(_s)

    # print total pressure and mmw
    log.info('    total      = %-9.2f bar' % hf_row['P_surf'])
    log.info('    mmw        = %-9.5f g mol-1' % (hf_row['atm_kg_per_mol'] * 1e3))
