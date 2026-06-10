# Generic outgassing wrapper
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from proteus.outgas.calliope import calc_surface_pressures, calc_target_masses
from proteus.outgas.common import expected_keys
from proteus.outgas.lavatmos_v2 import compute_silicate_outgassing

#from proteus.outgas.lavatmos import compute_silicate_outgassing
from proteus.utils.constants import element_list, vap_list, vol_list

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)



def calc_target_elemental_inventories(dirs: dict, config: Config, hf_row: dict):
    """
    Calculate total amount of volatile elements in the planet
    """

    # zero by default, in case not included
    for e in element_list:
        hf_row[e + '_kg_total'] = 0.0

    # Calculate target for calliope mass conservation
    if config.outgas.module == 'calliope':
        calc_target_masses(dirs, config, hf_row)

    # Update total mass of tracked elements
    hf_row['M_ele'] = 0.0
    for e in element_list:
        if e == 'O':  # Oxygen is set by fO2, so we skip it here (const_fO2)
            continue
        hf_row['M_ele'] += hf_row[e + '_kg_total']


def check_desiccation(config: Config, hf_row: dict) -> bool:
    """
    Check if the planet has desiccated. This is done by checking if all volatile masses
    are below a threshold.

    Parameters
    ----------
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only

    Returns
    -------
        bool
            True if desiccation occurred, False otherwise
    """

    # check if desiccation has occurred
    for e in element_list:
        if e == 'O':  # Oxygen is set by fO2, so we skip it here (const_fO2)
            continue
        if hf_row[e + '_kg_total'] > config.outgas.mass_thresh:
            log.info('Not desiccated, %s = %.2e kg' % (e, hf_row[e + '_kg_total']))
            return False  # return, and allow run_outgassing to proceed

    return True


def run_outgassing(dirs: dict, config: Config, hf_row: dict):
    """
    Run outgassing model to get new volatile surface pressures

    Parameters
    ----------
        dirs : dict
            Dictionary of directory paths
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    """

    log.info('Solving outgassing...')

    if config.outgas.silicates:
        gas_list = vol_list + config.outgas.vaplist
    else:
        gas_list = vol_list + vap_list

    # Run outgassing calculation
    if config.outgas.module == 'calliope':
        calc_surface_pressures(dirs, config, hf_row)

    log.debug('Outgassing complete, calculating atmospheric composition...')
    log.debug('comparison to H2S output by iterating over gas list')
    log.debug('    %-6s     = %-9.2f bar (%.2e VMR)' % ('H2S', hf_row['H2S_bar'], hf_row['H2S_vmr']))

    # calculate total atmosphere mass from sum of gas species
    hf_row['M_atm'] = 0.0
    for s in gas_list:
        #log.info('species %s'%s)
        #log.info('the mass of this species - if silicate should be zero: %s'%hf_row[s + '_kg_atm'])
    #for s in vol_list:
        hf_row['M_atm'] += hf_row[s + '_kg_atm']

    # print outgassed partial pressures (in order of descending abundance)
    mask = [hf_row[s + '_vmr'] for s in gas_list]
    for i in np.argsort(mask)[::-1]:
        s = gas_list[i]
        _p = hf_row[s + '_bar']
        _x = hf_row[s + '_vmr']
        _s = '    %-6s     = %-9.2f bar (%.2e VMR)' % (s, _p, _x)
        if _p > 0.01:
            log.info(_s)
        else:
            # don't spam log with species of negligible abundance
            log.debug(_s)
        #log.info('mass of this species: %s %4e'%s % hf_row[s + '_kg_atm'])

    # print total pressure and mmw
    log.info('    total      = %-9.2f bar' % hf_row['P_surf'])
    log.info('    mmw        = %-9.5f g mol-1' % (hf_row['atm_kg_per_mol'] * 1e3))


def run_desiccated(config: Config, hf_row: dict):
    """
    Handle desiccation of the planet. This substitutes for run_outgassing when the planet
    has lost its entire volatile inventory.

    Parameters
    ----------
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    """

    # if desiccated, set all gas masses to zero
    log.info('Desiccation has occurred - no volatiles remaining')

    if config.outgas.silicates:
        gas_list = vol_list + config.outgas.vaplist
        log.info('lavatmos should be running')
    else:
        log.info('lavatmos should not be running')
        gas_list = vol_list + vap_list

    # Do not set these to zero - avoid divide by zero elsewhere in the code
    excepted_keys = ['atm_kg_per_mol']
    for g in gas_list:
        excepted_keys.append(f'{g}_vmr')

    # Set most values to zero
    for k in expected_keys(config):
        if k not in excepted_keys:
            hf_row[k] = 0.0

    if config.outgas.silicates:
        compute_silicate_outgassing(config,hf_row)




def lavatmos_calliope_run(dirs: dict, config: Config, hf_row: dict):
    """function which runs lavatmos and calliope in a loop until they have converged.
    This allows for a consistentt computation of melt outgassing and dissolution
    Parameters
    ----------
        dirs : dict
            Dictionary of directory paths
        config : Config
            Configuration object
        hf_row : dict
            Dictionary of helpfile variables, at this iteration only
    """

    gas_list = vol_list + config.outgas.vaplist

    #reset all silicate masses to zero:
    for s in gas_list:
        if s in vol_list:
            continue
        else:
            hf_row[s + '_bar'] =0.0
            hf_row[s + '_vmr']=0.0
            hf_row[s + '_kg_atm']=0.0
            hf_row[s+ '_kg_tot']=0.0

    for e in element_list:
        if e in ['H','C','N','O','S','P']:
            continue
        else:
            hf_row[e + '_kg_atm']=0.0
            hf_row[e+ '_kg_tot']=0.0


    run_outgassing(dirs, config, hf_row)

    if config.outgas.silicates:

        #this needs to be commented out for runninglavatmos with the installation from github
        #lavadir = os.environ.get("LAVATMOS_DIR")
        #if lavadir:
            #log.info('Lavatmos directory found: %s' % lavadir)
        #else:
            #log.warning('Lavatmos directory not found, did you set the LAVATMOS_DIR environment variable?')
        if hf_row['Phi_global'] > 0.00:
            compute_silicate_outgassing(config, hf_row)
        else:
            log.info('planet has solidified, no silicate outgassing occurs')
