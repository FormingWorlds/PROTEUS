# Run VULCAN chemical kinetics module
from __future__ import annotations

import glob
import logging
import os
import shutil
from typing import TYPE_CHECKING

import numpy as np

# Import PROTEUS
from proteus.atmos_chem.common import chem_csv_name
from proteus.atmos_clim.common import read_atmosphere_data
from proteus.star.wrapper import flux_weighted_distance, scale_spectrum_to_stellar_surface
from proteus.utils.constants import (
    AU,
    CHEM_CONDENSATE_DENSITY,
    R_sun,
    vol_list,
)
from proteus.utils.helper import (
    find_nearest,
    format_subyear_time,
    list_snapshot_times,
    mol_to_ele,
)

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


def network_condensation(network_path: str) -> dict[str, str]:
    """Condensation reactions of a VULCAN chemical network file.

    Reads the reactions listed after the `# condensation` header, as VULCAN
    does, up to the next reaction-type header.

    Returns
    ----------
    pairs : dict
        Condensate species name, keyed by the name of the gas that condenses.
    """
    pairs = {}
    in_section = False
    with open(network_path) as hdl:
        for line in hdl:
            if line.startswith('#'):
                if line.startswith('# condensation'):
                    in_section = True
                elif line.startswith(('# radiative', '# photo', '# ionisation')):
                    in_section = False
                continue
            if in_section and '[' in line:
                reaction = line.partition('[')[-1].rpartition(']')[0]
                gas, _, product = reaction.partition('->')
                pairs[gas.strip()] = product.strip()
    return pairs


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


def read_vulcan_netcdf(fpath: str) -> dict:
    """Read the result of a VULCAN run from its NetCDF output file.

    Values are in the CGS units VULCAN writes: pressure in dyn cm-2, height in
    cm, Kzz in cm2 s-1. Arrays are ordered from the bottom of the column up.

    Parameters
    ----------
    fpath : str
        Path to a ``*_chem.nc`` file written by VULCAN.

    Returns
    -------
    dict
        'p' and 'tmp' at layer centres [nz], 'pl', 'zl' and 'Kzz' at the
        interfaces [nz+1] (Kzz is defined only on the nz-1 interior ones, and
        NaN at the bottom and top), 'species' (list of str) and 'x_gas',
        the volume mixing ratios [nz, number of species].
    """
    import netCDF4 as nc

    with nc.Dataset(fpath) as ds:
        out = {}
        for key in ('p', 'pl', 'tmp', 'zl', 'Kzz', 'x_gas'):
            out[key] = np.ma.filled(ds.variables[key][:].astype(float), np.nan)
        out['species'] = [str(sp) for sp in ds.variables['species'][:]]
    return out


def network_settings(config: Config) -> tuple[str, list, list, list] | None:
    """VULCAN network file and species lists for `atmos_chem.vulcan.network`.

    Returns
    ----------
    settings : tuple or None
        Network file name, elements of the network, species to plot and
        scattering species. None if the network is not recognised.
    """
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
            return None

    return network_file, ele_arr, plt_arr, sct_arr


def _write_ini_table(fpath: str, pco: np.ndarray, p_src: np.ndarray, vmr_src: dict) -> None:
    """Write initial mixing ratios as a VULCAN `table` file on VULCAN's own grid.

    VULCAN reads the rows by index, from the bottom of the column up, so the
    mixing ratios are interpolated in log pressure onto `pco` and written in
    that order. Outside the source range the end values are held constant.
    Linear interpolation keeps the sum of the mixing ratios on each row.

    Parameters
    ----------
    fpath : str
        Path of the file to write.
    pco : np.ndarray
        VULCAN layer-centre pressures [dyn cm-2], from the bottom up.
    p_src : np.ndarray
        Pressures of the source profile, in the same unit as `pco`, in any order.
    vmr_src : dict
        Volume mixing ratio arrays on `p_src`, keyed by species name.
    """
    order = np.argsort(p_src)
    logp_src = np.log10(np.asarray(p_src, dtype=float)[order])
    logp_tgt = np.log10(pco)
    cols = [np.asarray(pco, dtype=float)]
    header = '#Input composition arrays\nPressure'
    for gas, vmr in vmr_src.items():
        cols.append(np.interp(logp_tgt, logp_src, np.asarray(vmr, dtype=float)[order]))
        header += '\t' + gas
    np.savetxt(fpath, np.array(cols).T, delimiter='\t', comments='', header=header)


def _find_previous_chem(data_dir: str, year: float) -> str | None:
    """Path of the latest ``<time>_chem.nc`` in `data_dir` earlier than `year` [yr], or None."""
    earlier = [t for t in list_snapshot_times(data_dir, '_chem.nc') if t < year]
    if not earlier:
        return None
    return os.path.join(data_dir, format_subyear_time(earlier[-1]) + '_chem.nc')


def _write_result_csv(csv_file: str, result: dict) -> None:
    """Write a VULCAN result read from NetCDF as the PROTEUS chemistry CSV.

    Columns: temperature `tmp` [K], pressure `p` [Pa] and height `z` [m] of the
    layers, `Kzz` [cm2 s-1], then the mixing ratio of each gas. Rows go from the
    top of the column down, as in AGNI.
    """
    cols = {}
    cols['tmp'] = result['tmp']
    cols['p'] = result['p'] / 10  # dyn cm-2 to Pa
    cols['z'] = result['zl'][:-1] / 100  # bottom interface of each layer, cm to m
    kzz = result['Kzz'][1:-1]  # defined on the interior interfaces only
    cols['Kzz'] = np.insert(kzz, 0, kzz[0])
    for i, gas in enumerate(result['species']):
        if '_' not in gas:  # condensates and excited states carry an underscore
            cols[gas] = result['x_gas'][:, i]

    header = ''.join(str(key).ljust(14, ' ') + '\t' for key in cols)
    arr = np.array(list(cols.values())).T[::-1]
    np.savetxt(csv_file, arr, delimiter='\t', fmt='%.8e', header=header, comments='')


def run_vulcan(
    dirs: dict, config: Config, hf_row: dict, *, network_compiled: bool = False
) -> bool:
    """
    Run VULCAN chemical kinetics on the atmosphere snapshot at hf_row['Time'].

    VULCAN writes its result as NetCDF only (no pickle), which is moved to
    ``data/<time>_chem.nc``. PROTEUS reads it back and writes
    ``offchem/vulcan_<time>.csv``. The time [yr] is in the sub-year form of
    `format_subyear_time`. VULCAN also draws ``offchem/mix.*`` and
    ``offchem/evo.*`` for each run, which the next run overwrites.

    Parameters
    ----------
    dirs : dict
        Dictionary of directories.
    config : Config
        Configuration object.
    hf_row : dict
        Helpfile row at the snapshot time.
    network_compiled : bool
        Whether an earlier run of this sequence has compiled the chemical
        network already. If not, it is compiled (with `atmos_chem.vulcan.make_funs`).

    Returns
    ----------
    success : bool
        Did VULCAN run successfully?
    """

    # Imported here, so that reading VULCAN output does not need the package
    import vulcan

    log.debug(f'Using VULCAN from: {vulcan.__file__}')
    log.debug(f'    version {vulcan.__version__}')

    success = True

    # ------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------

    if config.atmos_clim.module != 'agni':
        log.warning('VULCAN chemistry only supported with AGNI')
        return False

    os.makedirs(dirs['output/offchem'], exist_ok=True)

    # ------------------------------------------------------------
    # READ DATA FROM PROTEUS RUN
    # ------------------------------------------------------------

    year = float(hf_row['Time'])
    log.info('Running VULCAN chemistry (target t=%.3e yr)' % year)
    vulcan_stem = f'vulcan_{format_subyear_time(year)}'
    vulcan_csv = chem_csv_name('vulcan', year)
    vulcan_cdf_dst = os.path.join(
        dirs['output'], 'data', format_subyear_time(year) + '_chem.nc'
    )
    log.debug(
        'VULCAN at t=%.3e yr writes %s and offchem/%s'
        % (year, os.path.basename(vulcan_cdf_dst), vulcan_csv)
    )

    # Read the atmosphere snapshot; 'p' replaces the combined grid with the
    # layer-centre pressures, on which the mixing ratios are defined
    extra_keys = ['p', 'pl', 'tmpl', 'x_gas', 'Kzz']
    atmos = read_atmosphere_data(dirs['output'], [year], extra_keys=extra_keys)
    if atmos is None:
        log.warning('Atmosphere NetCDF t=%.3e yr could not be read; skipping chemistry' % year)
        return False
    atmos = atmos[0]  # Get the single snapshot from the list

    # ------------------------------------------------------------
    # WRITE VULCAN INPUT FILES
    # ------------------------------------------------------------

    # Output folder
    vulcan_out = dirs['output/offchem']
    vulcan_plt = os.path.join(dirs['output/offchem'], '')  # VULCAN appends file names
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

    # VULCAN grid: level edges 'pl', not cell centres 'p', so the column is not cut by half a cell
    P_b = float(np.amax(atmos['pl'])) * 10  # convert Pa to dyne/cm^2
    P_t = float(np.amin(atmos['pl'])) * 10
    pco = np.logspace(np.log10(P_b), np.log10(P_t), len(p_arr))  # as VULCAN builds it

    # Write initial mixing ratios on the VULCAN grid
    vmr_write = os.path.join(dirs['output/offchem'], 'vmrs.dat')
    ini_mix = config.atmos_chem.vulcan.ini_mix
    prev_chem = None
    if ini_mix == 'previous':
        prev_chem = _find_previous_chem(os.path.join(dirs['output'], 'data'), year)
        if prev_chem is None:
            log.debug('No earlier chemistry result; initialising from the climate profile')
    if prev_chem is not None:
        log.debug(f'Initialising from the earlier chemistry result {prev_chem}')
        prev = read_vulcan_netcdf(prev_chem)
        vmr_src = {gas: prev['x_gas'][:, i] for i, gas in enumerate(prev['species'])}
        _write_ini_table(vmr_write, pco, prev['p'], vmr_src)
    else:
        vmr_src = {k.split('_')[0]: atmos[k] for k in atmos.keys() if '_vmr' in k}
        _write_ini_table(vmr_write, pco, np.array(atmos['p']) * 10.0, vmr_src)

    # ------------------------------------------------------------
    # CREATE VULCAN CONFIG
    # ------------------------------------------------------------

    # Base config
    log.debug('Creating VULCAN config')
    vcfg = vulcan.Config()

    # Determine network and species of interest
    settings = network_settings(config)
    if settings is None:
        return False
    network_file, ele_arr, plt_arr, sct_arr = settings

    log.info(f"    using '{network_file}' network")
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
        # set potential background gases, also from outside the network
        vmr = hf_row[gas + '_vmr']
        if gas in backs.keys():
            backs[gas] = vmr

        # drop gases not in the network
        if len(set(mol_to_ele(gas).keys()) - set(ele_arr)) > 0:
            continue

        # set constant mixing ratio and bottom BC if above the threshold
        if vmr > config.atmos_chem.vulcan.clip_vmr:
            vcfg.const_mix[gas] = vmr

    # Translate initial composition to VULCAN's string
    if ini_mix == 'outgas':
        vcfg.ini_mix = 'const_mix'
    else:
        vcfg.ini_mix = 'table'
    vcfg.vul_ini = vmr_write
    vcfg.scat_sp = sct_arr
    vcfg.T_cross_sp = []

    # Elemental abundances (VULCAN defaults), read only with ini_mix 'eq' or 'const_lowt';
    # use_solar is off, so those modes take these values instead of VULCAN's solar table.
    vcfg.use_solar = False
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
    _write_boundary_file(vcfg.bot_BC_flux_file, dict(vcfg.const_mix))

    # TOA boundary condition
    vcfg.use_topflux = False
    vcfg.top_BC_flux_file = os.path.join(vulcan_out, 'bc_top.dat')
    _write_boundary_file(vcfg.top_BC_flux_file, dict(vcfg.const_mix))
    vcfg.diff_esc = []  # species for diffusion-limit escape at TOA
    vcfg.max_flux = 1e90  # cap on diffusion-limited escape flux (cm-2 s-1)

    # Set background gas
    vcfg.atm_base = max(backs, key=backs.get)
    log.info(f"    background gas is '{vcfg.atm_base}'")

    # TP profile.
    vcfg.nz = len(p_arr)
    vcfg.P_b = P_b
    vcfg.P_t = P_t
    vcfg.atm_type = 'file'
    vcfg.atm_file = prof_write
    vcfg.para_anaTP = [120.0, 1500.0, 0.1, 0.02, 1.0, 1.0]  # unused with atm_type 'file'
    vcfg.Tiso = hf_row['T_surf']  # unused with atm_type 'file'
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
    vcfg.use_vm_mol = config.atmos_chem.vulcan.use_vm_mol
    vcfg.use_vz = config.atmos_chem.vulcan.use_vz
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

    # Condensation of every gas in the network that VULCAN can condense
    vcfg.use_condense = config.atmos_chem.vulcan.use_condense
    vcfg.use_settling = config.atmos_chem.vulcan.use_settling
    conden = {}
    if vcfg.use_condense:
        conden = {
            gas: product
            for gas, product in network_condensation(vcfg.network).items()
            if gas in CHEM_CONDENSATE_DENSITY
        }
        if not conden:
            log.warning(f"Network '{network_file}' has no condensation VULCAN supports")
        log.debug(f'Condensing species: {list(conden.keys())}')
    elif vcfg.use_settling:
        log.warning('VULCAN settling has no effect without condensation')
    vcfg.start_conden_time = 1e10
    vcfg.stop_conden_time = 1e10
    vcfg.condense_sp = list(conden.keys())
    vcfg.non_gas_sp = list(conden.values())
    vcfg.r_p = {conden[gas]: config.atmos_chem.vulcan.r_p for gas in conden}  # (cm)
    vcfg.rho_p = {conden[gas]: CHEM_CONDENSATE_DENSITY[gas] for gas in conden}  # (g cm-3)
    vcfg.use_relax = []  # VULCAN reads these two with condensation, its Config omits them
    vcfg.use_ini_cold_trap = False
    vcfg.fix_species = []
    vcfg.fix_species_from_coldtrap_lev = False
    vcfg.humidity = 1.0

    # Convergence
    vcfg.st_factor = config.atmos_chem.vulcan.st_factor
    vcfg.conv_step = config.atmos_chem.vulcan.conv_step
    vcfg.yconv_cri = config.atmos_chem.vulcan.yconv_cri  # check steady-state
    vcfg.slope_cri = config.atmos_chem.vulcan.slope_cri  # check steady-state
    vcfg.yconv_min = config.atmos_chem.vulcan.yconv_min
    vcfg.conver_ignore = []

    # Time-stepping
    vcfg.trun_min = 1e2
    vcfg.runtime = config.atmos_chem.vulcan.runtime  # (s)
    vcfg.use_print_prog = True
    vcfg.use_print_delta = False
    vcfg.print_prog_num = 100  # print the progress every x steps
    vcfg.dttry = config.atmos_chem.vulcan.dttry  # (s)
    vcfg.dt_min = config.atmos_chem.vulcan.dt_min  # (s)
    vcfg.dt_max = vcfg.runtime * 1e-4
    vcfg.dt_var_max = 2.0
    vcfg.dt_var_min = 0.5
    vcfg.count_min = 120
    vcfg.count_max = int(3e4)
    vcfg.atol = config.atmos_chem.vulcan.atol  # number density (cm-3)
    vcfg.mtol = config.atmos_chem.vulcan.mtol
    vcfg.mtol_conv = config.atmos_chem.vulcan.mtol_conv
    vcfg.flux_atol = config.atmos_chem.vulcan.flux_atol  # actinic flux (photons cm-2 s-1 nm-1)
    vcfg.rtol = config.atmos_chem.vulcan.rtol  # relative tolerance for adjusting the stepsize
    vcfg.post_conden_rtol = config.atmos_chem.vulcan.post_conden_rtol
    vcfg.rtol_min = 0.02
    vcfg.rtol_max = 2.5
    vcfg.use_adapt_rtol = True
    vcfg.pos_cut = 0
    vcfg.nega_cut = -1.0
    vcfg.loss_eps = 1e-1
    vcfg.flux_cri = 0.1

    # Self-consistent RCE solver; the AGNI settings below are unused while it is off.
    # They follow the PROTEUS AGNI config, so a coupled call matches the climate step.
    vcfg.agni_call_frq = 0
    vcfg.spectral_file = 'greygas'
    vcfg.use_rayleigh = config.atmos_clim.rayleigh
    vcfg.surf_albedo = config.atmos_clim.surf_greyalbedo
    vcfg.Tsurf_guess = float(hf_row['T_surf'])
    vcfg.agni_atol = config.atmos_clim.agni.solution_atol
    vcfg.agni_rtol = config.atmos_clim.agni.solution_rtol
    vcfg.agni_nlev = config.atmos_clim.num_levels
    vcfg.solve_rce = config.atmos_clim.agni.solve_energy

    # Folders and output; PROTEUS reads back the NetCDF file
    vcfg.log_level = config.params.out.logging
    vcfg.clean_output = False
    vcfg.output_dir = vulcan_out
    vcfg.plot_dir = vulcan_plt
    vcfg.out_name = vulcan_stem
    vcfg.save_type = ['netcdf']
    vcfg.save_evolution = False
    vcfg.save_evo_frq = 10

    # Plotting
    vcfg.plot_dpi = 150
    vcfg.plot_fmt = config.params.out.plot_fmt
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

    vulcan.run_vulcan(vcfg, config.atmos_chem.vulcan.make_funs and not network_compiled)

    # ------------------------------------------------------------
    # READ AND PARSE OUTPUT FILES
    # ------------------------------------------------------------

    # Move NetCDF output to the PROTEUS output/data folder
    vulcan_cdf_src = os.path.join(vulcan_out, vulcan_stem + '.nc')
    if not os.path.exists(vulcan_cdf_src):
        log.warning(f'Could not find output file {vulcan_cdf_src}')
        return False
    log.debug("Moving VULCAN NetCDF output to '%s'" % vulcan_cdf_dst)
    shutil.move(vulcan_cdf_src, vulcan_cdf_dst)

    # Write the result as CSV, in a human-readable format
    csv_file = os.path.join(vulcan_out, vulcan_csv)
    log.debug(f'Writing to {csv_file}')
    _write_result_csv(csv_file, read_vulcan_netcdf(vulcan_cdf_dst))

    log.info('    done')
    return success
