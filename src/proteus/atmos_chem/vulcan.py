# Run VULCAN chemical kinetics module
from __future__ import annotations

import glob
import logging
import os
import pickle
import shutil
from typing import TYPE_CHECKING

import numpy as np

# Import VULCAN
import vulcan

# Import PROTEUS
from proteus.atmos_clim.common import find_latest_atmosphere_time, read_atmosphere_data
from proteus.star.wrapper import flux_weighted_distance, scale_spectrum_to_stellar_surface
from proteus.utils.constants import AU, R_sun, element_list, vol_list
from proteus.utils.helper import find_nearest

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

VULCAN_NAME = 'vulcan.pkl'
OUTPUT_NAME = 'vulcan.csv'


def _write_boundary_file(fpath: str, fixed_vmr: dict) -> None:
    """Write a VULCAN boundary-condition file.

    Each row is: species, flux (cm-2 s-1), deposition velocity (cm s-1) and
    fixed volume mixing ratio, the columns VULCAN reads from `bot_BC_flux_file`.
    Fluxes and deposition velocities are zero; only the fixed mixing ratios
    (dimensionless) in `fixed_vmr` are prescribed.
    """
    with open(fpath, 'w') as hdl:
        hdl.write('# species  flux(cm-2 s-1)  vdep(cm s-1)  fixed_vmr\n')
        for gas, vmr in fixed_vmr.items():
            hdl.write(f'{gas}  0.0  0.0  {vmr:.8e}\n')


def run_vulcan(dirs: dict, config: Config, hf_row: dict, *, online: bool = False) -> bool:
    """
    Run VULCAN chemical kinetics on the current atmospheric state.

    In offline mode (default), runs once on the final atmospheric state as a
    post-processing step. The offchem directory is wiped and recreated, and
    output is written to fixed filenames (vulcan.pkl, vulcan.csv).

    In online mode, runs at every snapshot during the main simulation loop.
    The offchem directory is created once and reused. Output uses per-snapshot
    filenames (vulcan_{year}.pkl, vulcan_{year}.csv), and the chemical network
    is compiled only once across all snapshots.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories.
    config : Config
        Configuration object.
    hf_row : dict
        Dictionary of current helpfile row.
    online : bool
        If True, use online mode with per-snapshot output and one-time
        network compilation. If False (default), use offline mode.

    Returns
    ----------
    success : bool
        Did VULCAN run successfully?
    """

    log.debug(f'Using VULCAN from: {vulcan.__file__}')
    log.debug(f'    version {vulcan.__version__}')

    success = True

    # ------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------

    if config.atmos_clim.module != 'agni':
        log.warning('VULCAN chemistry only supported with AGNI')
        return False

    # Prepare output directory
    if online:
        # Online: create once, reuse across snapshots
        os.makedirs(dirs['output/offchem'], exist_ok=True)
    else:
        # Offline: wipe and recreate for a clean post-processing run
        shutil.rmtree(dirs['output/offchem'], ignore_errors=True)
        os.makedirs(dirs['output/offchem'])

    # ------------------------------------------------------------
    # READ DATA FROM PROTEUS RUN
    # ------------------------------------------------------------

    # Get all times for which we have data
    year = hf_row['Time']
    log.debug('Reading data for t=%.2e yr' % year)

    # Determine output filenames
    if online:
        # Per-snapshot files so each timestep's result is preserved
        year_int = int(year)
        vulcan_pkl = f'vulcan_{year_int}.pkl'
        vulcan_csv = f'vulcan_{year_int}.csv'
    else:
        # Single fixed files, overwritten each run
        vulcan_pkl = VULCAN_NAME
        vulcan_csv = OUTPUT_NAME

    # Read atmosphere data. Prefer the exact hf_row['Time'] snapshot.
    extra_keys = ['pl', 'tmpl', 'x_gas', 'Kzz']
    atmos = read_atmosphere_data(dirs['output'], [year], extra_keys=extra_keys)

    # Fall back to the final available snapshot on disk.
    if atmos is None:
        # Try to find the best one
        # This is ideally the latest time available
        latest = find_latest_atmosphere_time(dirs['output'])

        # Could not find NetCDF file for reading
        if latest is None:
            log.warning('No atmosphere NetCDFs available; skipping chemistry')
            return False

        # Try to fall back to the latest available snapshot
        log.warning(
            'Atmosphere NetCDF t=%.0f yr missing; using available (t=%.0f yr)' % (year, latest)
        )
        year = latest
        atmos = read_atmosphere_data(dirs['output'], [year], extra_keys=extra_keys)

        # Just fail if we still can't find anything
        if atmos is None:
            log.warning('Atmosphere NetCDF could not be read; skipping chemistry')
            return False
    atmos = atmos[0]  # Get the single snapshot from the list

    # ------------------------------------------------------------
    # WRITE VULCAN INPUT FILES
    # ------------------------------------------------------------

    # Output folder
    vulcan_out = dirs['output/offchem']
    vulcan_plt = dirs['output/offchem']
    log.debug('Writing VULCAN input data')

    # Find a reasonable file for the stellar flux
    ls = glob.glob(dirs['output'] + '/data/*.sflux')
    years = [int(f.split('/')[-1].split('.')[0]) for f in ls]
    sflux_fpath = dirs['output'] + '/data/%d.sflux' % find_nearest(years, year)[0]

    # Read spectrum and scale to surface of the star
    sflux_data = np.loadtxt(sflux_fpath, skiprows=1).T
    star_wl = np.array(sflux_data[0])
    star_fl = scale_spectrum_to_stellar_surface(
        sflux_data[1], flux_weighted_distance(hf_row), hf_row['R_star']
    )

    # Remove small values
    star_wl = star_wl[star_fl > config.atmos_chem.vulcan.clip_fl]
    star_fl = star_fl[star_fl > config.atmos_chem.vulcan.clip_fl]

    # Write spectrum
    star_write = os.path.join(dirs['output/offchem'], 'star.dat')
    np.savetxt(
        star_write,
        np.array([star_wl, star_fl]).T,
        header='# WL(nm)    Flux(ergs/cm**2/s/nm)',
        fmt=['%.4f', '%.4e'],
    )

    # Get TP profile
    p_arr = np.array(atmos['pl']) * 10.0  # dyne/cm^2
    t_arr = np.clip(atmos['tmpl'], a_min=180.0, a_max=None)

    # Get Kzz profile
    k_arr = np.array(atmos['Kzz']) * 1e4  # cm^2/s

    # Write TPK profile
    header = '#(dyne/cm2)\t(K)\t(cm2/s)\n'
    header += 'Pressure\tTemp\tKzz'
    prof_write = os.path.join(dirs['output/offchem'], 'profile.dat')
    np.savetxt(
        prof_write,
        np.array([p_arr[::-1], t_arr[::-1], k_arr[::-1]]).T,
        delimiter='\t',
        header=header,
        comments='',
        fmt='%1.5e',
    )

    # Write mixing ratios
    vmr_write = os.path.join(dirs['output/offchem'], 'vmrs.dat')
    x_gas = [list(p_arr)]
    header = '#Input composition arrays \nPressure\t'
    for key in atmos.keys():
        if '_vmr' in key:
            gas = key.split('_')[0]
            arr = list(atmos[key])
            arr.append(arr[-1])  # extend by 1 level
            x_gas.append(arr)
            header += gas + ' '
    x_gas = np.array(x_gas).T
    np.savetxt(vmr_write, x_gas, delimiter='\t', comments='', header=header)

    # ------------------------------------------------------------
    # CREATE VULCAN CONFIG
    # ------------------------------------------------------------

    # Base config
    log.debug('Creating VULCAN config')
    vcfg = vulcan.Config()

    # Which elements are included?
    e_incl = []
    for e in element_list:
        if hf_row[e + '_kg_atm'] > 1e10:
            e_incl.append(e)
    log.debug(f'Found elements: {e_incl}')

    # Determine network and species of interest
    match config.atmos_chem.vulcan.network:
        case 'CHO':
            ele_arr = ['H', 'O', 'C']
            plt_arr = ['H2', 'H', 'H2O', 'CH4', 'CO', 'CO2', 'C2H2']
            sct_arr = ['H2', 'O2', 'CO2']
            if config.atmos_chem.photo_on:
                network_file = 'CHO_photo_network.txt'
            else:
                network_file = 'CHO_thermo_network.txt'

        case 'NCHO':
            plt_arr = ['H2', 'H', 'H2O', 'CH4', 'CO', 'CO2', 'C2H2', 'NH3', 'N2']
            sct_arr = ['H2', 'O2', 'N2', 'CO2']
            ele_arr = ['H', 'O', 'C', 'N']
            if config.atmos_chem.photo_on:
                network_file = 'NCHO_photo_network.txt'
            else:
                network_file = 'NCHO_thermo_network.txt'

        case 'SNCHO':
            plt_arr = [
                'H2',
                'H',
                'H2O',
                'CH4',
                'CO',
                'CO2',
                'C2H2',
                'NH3',
                'SO2',
                'H2S',
                'S2',
                'S8',
            ]
            sct_arr = ['H2', 'O2', 'N2', 'CO2']
            ele_arr = ['H', 'O', 'C', 'N', 'S']
            network_file = 'SNCHO_photo_network.txt'

        case _:
            log.error(f"Network not recognised: '{config.atmos_chem.vulcan.network}'")
            return False

    log.debug(f"Using '{network_file}' ")
    vcfg.network = os.path.join(vulcan.paths.THERMO_DIR, network_file)
    vcfg.atom_list = ele_arr
    vcfg.use_photo = config.atmos_chem.photo_on
    vcfg.use_ion = False
    vcfg.use_lowT_limit_rates = False
    vcfg.remove_list = []  # reactions switched off, none

    # Parse atmospheric composition into string, work out background gas
    backs = {'H2': 0.0, 'N2': 0.0, 'O2': 0.0, 'CO2': 0.0}
    vcfg.const_mix = {}
    for gas in vol_list:
        vmr = hf_row[gas + '_vmr']
        if gas in backs.keys():
            backs[gas] = vmr
        if vmr > config.atmos_chem.vulcan.clip_vmr:
            vcfg.const_mix[gas] = vmr

    # Translate initial composition to VULCAN's string
    if config.atmos_chem.vulcan.ini_mix == 'profile':
        vcfg.ini_mix = 'table'
    else:
        vcfg.ini_mix = 'const_mix'
    vcfg.vul_ini = vmr_write
    vcfg.scat_sp = sct_arr
    vcfg.T_cross_sp = []

    # Elemental abundances, only read with ini_mix 'eq' or 'const_lowt' (VULCAN defaults)
    vcfg.use_solar = True
    vcfg.fastchem_met_scale = 1.0
    vcfg.O_H = 5.37e-4
    vcfg.C_H = 2.95e-4
    vcfg.N_H = 7.08e-5
    vcfg.S_H = 1.41e-5
    vcfg.He_H = 0.0838

    # Surface boundary condition; a copy, because VULCAN can add species to this dict
    vcfg.use_botflux = False
    if config.atmos_chem.vulcan.fix_surf:
        vcfg.use_fix_sp_bot = dict(vcfg.const_mix)
    else:
        vcfg.use_fix_sp_bot = {}
    vcfg.bot_BC_flux_file = os.path.join(vulcan_out, 'bc_bot.dat')
    _write_boundary_file(vcfg.bot_BC_flux_file, vcfg.use_fix_sp_bot)

    # TOA boundary condition
    vcfg.use_topflux = False
    vcfg.top_BC_flux_file = os.path.join(vulcan_out, 'bc_top.dat')
    _write_boundary_file(vcfg.top_BC_flux_file, {})
    vcfg.diff_esc = []  # species for diffusion-limit escape at TOA
    vcfg.max_flux = 1e13  # cap on diffusion-limited escape flux (cm-2 s-1)

    # Set background gas
    vcfg.atm_base = max(backs, key=backs.get)
    log.debug(f"Background gas is '{vcfg.atm_base}'")

    # TP profile.
    # P_b / P_t are the boundary pressures (surface and TOA), which live
    # on the level edges (atmos['pl'], length nz+1), not on the cell
    # centres (atmos['p'], length nz). Using 'p' truncated the column by
    # one half-cell at each end. The '* 10' converts Pa to barye (CGS).
    vcfg.nz = len(p_arr)
    vcfg.P_b = float(np.amax(atmos['pl'])) * 10
    vcfg.P_t = float(np.amin(atmos['pl'])) * 10
    vcfg.atm_type = 'file'
    vcfg.atm_file = prof_write
    vcfg.para_anaTP = [120.0, 1500.0, 0.1, 0.02, 1.0, 1.0]  # unused with atm_type 'file'
    vcfg.Tiso = 1000.0  # unused with atm_type 'file'
    vcfg.rocky = True
    vcfg.Rp = hf_row['R_int'] * 100.0  # Planetary radius (cm)
    vcfg.gs = hf_row['gravity'] * 100  # surface gravity (cm/s^2)

    # Spectrum
    vcfg.sflux_file = star_write
    vcfg.r_star = hf_row['R_star'] / R_sun  # stellar radius (R_sun)
    vcfg.orbit_radius = flux_weighted_distance(hf_row) / AU  # same distance as star.dat [AU]
    vcfg.sl_angle = config.orbit.zenith_angle * np.pi / 180.0  # the zenith angle
    vcfg.f_diurnal = config.orbit.s0_factor

    # Photochemistry: Eddington coefficient, wavelength bins (nm), flux update frequency
    vcfg.edd = 0.5
    vcfg.dbin1 = 0.1
    vcfg.dbin2 = 2.0
    vcfg.dbin_12trans = 240.0
    vcfg.ini_update_photo_frq = 100
    vcfg.final_update_photo_frq = 5

    # Mixing processes
    vcfg.update_frq = 50  # frequency for updating dz and dzi due to change of mu
    vcfg.use_moldiff = config.atmos_chem.moldiff_on
    vcfg.use_vm_mol = False
    vcfg.use_vz = True
    vcfg.vz_prof = 'const'  # Options: 'const' or 'file'
    vcfg.const_vz = config.atmos_chem.updraft_const  # (cm/s)
    vcfg.use_Kzz = config.atmos_chem.Kzz_on
    if config.atmos_chem.Kzz_const is not None:
        vcfg.Kzz_prof = 'const'
        vcfg.const_Kzz = config.atmos_chem.Kzz_const
    else:
        # if None, will get Kzz from profile
        vcfg.Kzz_prof = 'file'
        vcfg.const_Kzz = 1e5  # <- dummy value, will be replaced when VULCAN is run
    vcfg.K_max = 1e5  # only read with Kzz_prof 'Pfunc'
    vcfg.K_p_lev = 0.1  # only read with Kzz_prof 'Pfunc'

    # Condensation
    vcfg.use_condense = False
    vcfg.use_settling = False
    vcfg.start_conden_time = 1e10
    vcfg.stop_conden_time = 1e10
    vcfg.condense_sp = []
    vcfg.non_gas_sp = []
    vcfg.r_p = {}
    vcfg.rho_p = {}
    vcfg.fix_species = []
    vcfg.fix_species_from_coldtrap_lev = False
    vcfg.humidity = 1.0

    # Convergence
    vcfg.st_factor = 0.5
    vcfg.conv_step = 100
    vcfg.yconv_cri = config.atmos_chem.vulcan.yconv_cri  # check steady-state
    vcfg.slope_cri = config.atmos_chem.vulcan.slope_cri  # check steady-state
    vcfg.yconv_min = 0.5
    vcfg.conver_ignore = []

    # Time-stepping
    vcfg.trun_min = 1e2
    vcfg.runtime = 1.0e22
    vcfg.use_print_prog = True
    vcfg.use_print_delta = False
    vcfg.print_prog_num = 100  # print the progress every x steps
    vcfg.dttry = 1.0e-8
    vcfg.dt_min = 1.0e-10
    vcfg.dt_max = vcfg.runtime * 1e-4
    vcfg.dt_var_max = 2.0
    vcfg.dt_var_min = 0.5
    vcfg.count_min = 120
    vcfg.count_max = int(3e4)
    vcfg.atol = 1.0e-1  # Decrease this if the solutions are not stable
    vcfg.mtol = 1.0e-22
    vcfg.mtol_conv = 1.0e-20
    vcfg.flux_atol = 1.0  # actinic flux tolerance (photons cm-2 s-1 nm-1)
    vcfg.rtol = 0.9  # relative tolerence for adjusting the stepsize
    vcfg.post_conden_rtol = 0.1
    vcfg.rtol_min = 0.02
    vcfg.rtol_max = 2.5
    vcfg.use_adapt_rtol = True
    vcfg.pos_cut = 0
    vcfg.nega_cut = -1.0
    vcfg.loss_eps = 1e-1
    vcfg.flux_cri = 0.1

    # Self-consistent RCE solver; the AGNI settings below are unused while it is off
    vcfg.agni_call_frq = 0
    vcfg.spectral_file = 'greygas'
    vcfg.use_rayleigh = False
    vcfg.surf_albedo = 0.0
    vcfg.Tsurf_guess = 2000.0
    vcfg.agni_atol = 1e-3
    vcfg.agni_rtol = 1e-3
    vcfg.agni_nlev = 60
    vcfg.solve_rce = False

    # Folders and output; PROTEUS reads back the pickle
    vcfg.log_level = config.params.out.logging
    vcfg.clean_output = False
    vcfg.output_dir = vulcan_out
    vcfg.plot_dir = vulcan_plt
    vcfg.out_name = vulcan_pkl
    vcfg.save_type = ['netcdf', 'pickle']
    vcfg.save_evolution = False
    vcfg.save_evo_frq = 10

    # Plotting
    vcfg.plot_dpi = 150
    vcfg.plot_fmt = 'png'
    vcfg.plot_TP = config.atmos_chem.vulcan.save_frames
    vcfg.use_live_plot = config.atmos_chem.vulcan.save_frames
    vcfg.live_plot_frq = 50
    vcfg.use_plot_end = True
    vcfg.use_plot_evo = True
    vcfg.plot_spec = plt_arr
    vcfg.plot_height = False

    # ------------------------------------------------------------
    # RUN VULCAN
    # ------------------------------------------------------------

    # Make chemical network
    has_made = hasattr(run_vulcan, '_made')
    if config.atmos_chem.vulcan.make_funs:
        if online and has_made:
            # Online: already compiled on a previous snapshot, skip
            pass
        else:
            log.debug('Performing `make_chem_funs` step...')
            vulcan.make_chem_funs.make_all(vcfg)
            if online:
                run_vulcan._made = True
            log.debug('    done')

    # Call the solver
    vulcan.run_vulcan(vcfg, not has_made)

    # ------------------------------------------------------------
    # READ AND PARSE OUTPUT FILES
    # ------------------------------------------------------------

    result_file = os.path.join(vulcan_out, vulcan_pkl)
    if not os.path.exists(result_file):
        log.warning(f'Could not find output file {result_file}')
        return False

    # load data from file
    with open(result_file, 'rb') as hdl:
        result = pickle.load(hdl)
    result_dict = {}

    # read t, p, z, kzz
    result_dict['tmp'] = np.array(result['atm']['Tco'])
    result_dict['p'] = np.array(result['atm']['pco']) / 10  # convert to Pa
    result_dict['z'] = np.array(result['atm']['zco'][:-1]) / 100  # convert to m
    result_dict['Kzz'] = np.array(result['atm']['Kzz'])
    result_dict['Kzz'] = np.insert(result_dict['Kzz'], 0, result_dict['Kzz'][0])

    # read mixing ratios
    result_gas = result['variable']['species']
    for i, gas in enumerate(result_gas):
        if '_' not in gas:
            result_dict[gas] = np.array(result['variable']['ymix'][:, i])

    # write csv file, in a human-readable format
    csv_file = os.path.join(vulcan_out, vulcan_csv)
    log.debug(f'Writing to {csv_file}')
    header = ''
    Xarr = []
    for key in result_dict.keys():
        header += str(key).ljust(14, ' ') + '\t'  # compose header
        Xarr.append(list(result_dict[key]))
    Xarr = np.array(Xarr).T  # transpose such that each column is a single variable
    Xarr = Xarr[::-1]  # flip arrays to match AGNI format
    np.savetxt(csv_file, Xarr, delimiter='\t', fmt='%.8e', header=header, comments='')

    log.info('    done')
    return success
