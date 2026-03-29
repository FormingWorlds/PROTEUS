# This routine needs to be called via python3 src/proteus/plot/compar_atmosphere_models.py.
# then it needs the folders for outputdir1 and outputdir 2 in the terminal as
# sys.argv[1] and sys.argv[2].

from __future__ import annotations

import glob
import logging
import os
import sys
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmcrameri import cm
from matplotlib.ticker import LogLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from proteus.utils.helper import find_nearest
from proteus.utils.plot import sample_times

if TYPE_CHECKING:
    pass

log = logging.getLogger('fwl.' + __name__)


def read_ncdf_profile(nc_fpath: str, extra_keys: list = []):
    """Read data from atmosphere NetCDF output file.

    Automatically reads pressure (p), temperature (t), radius (z) arrays with
    cell-centre (N) and cell-edge (N+1) values interleaved into a single combined array of
    length (2*N+1).

    Extra keys can be read-in using the extra_keys parameter. These will be stored with
    the same dimensions as in the NetCDF file.

    Parameters
    ----------
        nc_fpath : str
            Path to NetCDF file.

        extra_keys : list
            List of extra keys (strings) to read from the file.

    Returns
    ----------
        out : dict
            Dictionary containing numpy arrays of data from the file.
    """

    import netCDF4 as nc

    # open file
    if not os.path.isfile(nc_fpath):
        log.error(f"Could not find NetCDF file '{nc_fpath}'")
        return None
    ds = nc.Dataset(nc_fpath)

    p = np.array(ds.variables['p'][:])
    pl = np.array(ds.variables['pl'][:])

    t = np.array(ds.variables['tmp'][:])
    tl = np.array(ds.variables['tmpl'][:])

    rp = float(ds.variables['planet_radius'][0])
    if 'z' in ds.variables.keys():
        # probably from JANUS, which stores heights
        z = np.array(ds.variables['z'][:])
        zl = np.array(ds.variables['zl'][:])
        r = np.array(z) + rp
        rl = np.array(zl) + rp
    else:
        # probably from AGNI, which stores radii
        r = np.array(ds.variables['r'][:])
        rl = np.array(ds.variables['rl'][:])
        z = np.array(r) - rp
        zl = np.array(rl) - rp

    nlev_c = len(p)

    # read pressure, temperature, height data into dictionary values
    out = {}
    out['p'] = [pl[0]]
    out['t'] = [tl[0]]
    out['z'] = [zl[0]]
    out['r'] = [rl[0]]
    for i in range(nlev_c):
        out['p'].append(p[i])
        out['p'].append(pl[i + 1])

        out['t'].append(t[i])
        out['t'].append(tl[i + 1])

        out['z'].append(z[i])
        out['z'].append(zl[i + 1])

        out['r'].append(r[i])
        out['r'].append(rl[i + 1])

    # flags
    for fk in ('transparent', 'solved', 'converged'):
        if fk in ds.variables.keys():
            out[fk] = ncdf_flag_to_bool(ds.variables[fk])
        else:
            out[fk] = False  # if not available

    # Read extra keys
    for key in extra_keys:
        # Check that key exists
        if key not in ds.variables.keys():
            log.error(f"Could not read '{key}' from NetCDF file")
            continue

        # Reading composition
        if key == 'x_gas':
            gas_l = ds.variables['gases'][:]  # names (bytes matrix)
            gas_x = ds.variables['x_gas'][:]  # vmrs (float matrix)

            # get data for each gas
            for igas, gas in enumerate(gas_l):
                gas_lbl = ''.join([c.decode(encoding='utf-8') for c in gas]).strip()
                out[gas_lbl + '_vmr'] = np.array(gas_x[:, igas])

        else:
            out[key] = np.array(ds.variables[key][:])

    # close file
    ds.close()

    # convert to np arrays
    for key in out.keys():
        out[key] = np.array(out[key], dtype=float)

    return out


def compare_times(times, plottimes):
    # get samples on log-time scale
    sample_t = []
    sample_i = []
    for s in plottimes:  # Sample on log-scale
        if s in times:
            sample_t.append(int(s))
            print('time in list', s)
        else:
            print('time not in list')
            remaining = [int(t) for t in set(times) - set(plottimes)]
            if len(remaining) == 0:
                break
            # Get next nearest time
            val, _ = find_nearest(remaining, s)
            print('nearest value found', val)
            sample_t.append(int(val))

            # Get the index of this time in the original array
            _, idx = find_nearest(times, val)
            sample_i.append(int(idx))
    print(sample_t, sample_i)

    return sample_t, sample_i


def ncdf_flag_to_bool(var) -> bool:
    """Convert NetCDF flag (y/n) to Python bool (true/false)"""
    v = str(var[0].tobytes().decode()).lower()

    # check against expected
    if v == 'y':
        return True
    elif v == 'n':
        return False
    else:
        raise ValueError(f'Could not parse NetCDF atmos flag variable \n {var}')


def read_2model_data(output_dir1: str, output_dir2: str, extension, tmin, nsamp, extra_keys=[]):
    """
    Read all p,t,z profiles from NetCDF files in a PROTEUS output folder.
    compare times at which to plot the output between the two folders and make sure that they agree
    """

    times1, plot_times1, _1 = sample_output(output_dir1, extension, tmin, nsamp)
    times2, plot_times2, _2 = sample_output(output_dir2, extension, tmin, nsamp)

    print()

    # set new array bound for the time arrays from two runs: same lower bound (higher minimum ) and same upper bound (lower maximum)
    lower_bound = max(
        np.array(plot_times1).min(), np.array(plot_times2).min()
    )  # higher minimum
    upper_bound = min(np.array(plot_times1).max(), np.array(plot_times2).max())  # lower maximum

    # times1 = np.clip(times1, lower_bound, upper_bound)
    # times2 = np.clip(times2, lower_bound, upper_bound)

    # check that values in array are not out of bounds
    for name, arr in {'a': plot_times1, 'b': plot_times2}.items():
        mask_low = arr < lower_bound
        mask_high = arr > upper_bound

        if np.any(mask_low | mask_high):
            if name == 'a':
                plot_times1[mask_low] = plot_times2[mask_low]
                plot_times1[mask_high] = plot_times2[mask_high]
            else:
                plot_times2[mask_low] = plot_times1[mask_low]
                plot_times2[mask_high] = plot_times1[mask_high]

    # replace only the remaining (in-bounds) values in times 2 to have the same array as in times 1
    mask2 = (plot_times2 < lower_bound) | (plot_times2 > upper_bound)
    # replace only the remaining (in-bounds) values
    plot_times2[~mask2] = plot_times1[~mask2]

    # now find nearest value

    final_times1, final_indices1 = compare_times(times1, plot_times1)
    final_times2, final_indices2 = compare_times(times2, plot_times2)

    profiles1 = [
        read_ncdf_profile(
            os.path.join(output_dir1, 'data', '%.0f_atm.nc' % t), extra_keys=extra_keys
        )
        for t in final_times1
    ]

    profiles2 = [
        read_ncdf_profile(
            os.path.join(output_dir2, 'data', '%.0f_atm.nc' % t), extra_keys=extra_keys
        )
        for t in final_times2
    ]

    if None in profiles2:
        log.warning('One or more NetCDF files could not be found')
        if os.path.exists(os.path.join(output_dir2, 'data', 'data.tar')):
            log.warning('You may need to extract archived data files')
        return
    if None in profiles1:
        log.warning('One or more NetCDF files could not be found')
        if os.path.exists(os.path.join(output_dir1, 'data', 'data.tar')):
            log.warning('You may need to extract archived data files')
        return

    return final_times1, final_times2, profiles1, profiles2


def sample_output(output_dir, extension: str = '_atm.nc', tmin: float = 1.0, nsamp: int = 8):
    """
    Sample output files from a model run based on their time stamps.

    This function searches the `<output_dir>/data` directory for files whose
    names end with the given extension and whose base name is an integer
    time stamp. It then selects up to `nsamp` representative output times
    greater than or equal to `tmin`, using `sample_times`, and returns the
    corresponding times and file paths.

    If no matching files are found, the function returns empty lists. If an
    archive exists in the data directory, an error is logged indicating that
    the archive should be extracted first.

    Parameters
    ----------
    output_dir : str
        Path to the model output directory containing a `data/` subdirectory.
    extension : str, optional
        File extension used to identify output files (default: "_atm.nc").
    tmin : float, optional
        Minimum time to consider when sampling outputs (default: 1.0).
    nsamp : int, optional
        Number of output times to sample (default: 8).

    Returns
    -------
    out_t : list
        List of sampled output times.
    out_f : list
        List of file paths corresponding to the sampled times.
    """

    files = glob.glob(os.path.join(output_dir + '/data', '*' + extension))

    # No files found?
    if len(files) < 1:
        log.error('No output files found, check if arxiv exists and Extract it.')

        # Return empty
        return [], []

    # get times
    times = [int(f.split('/')[-1].split(extension)[0]) for f in files]
    # print(times)

    out_t, out_i = sample_times(times, nsamp, tmin=tmin)
    out_f = [files[i] for i in out_i]

    # return times and file paths
    print(np.array(times), np.array(out_t), np.array(out_f))
    return np.array(times), np.array(out_t), np.array(out_f)


def plot_atmosphere_comparison(
    output_dir1, output_dir2, extension='_atm.nc', tmin=1e4, nsamp=5, plot_format='pdf'
):
    """
    Compare atmospheric temperature–pressure profiles from two model runs.

    This function reads atmospheric output files from two model output
    directories, samples representative times from each run, and produces
    a single plot comparing their temperature–pressure profiles. Profiles
    from the first model are plotted with solid lines, while profiles from
    the second model are plotted with dashed lines. Line colour encodes the
    simulation time using a logarithmic colour scale.

    The resulting figure is saved to the `plots/` subdirectory of
    `output_dir1`.

    Parameters
    ----------
    output_dir1 : str
        Path to the first model output directory.
    output_dir2 : str
        Path to the second model output directory.
    extension : str, optional
        File extension used to identify atmospheric output files
        (default: "_atm.nc").
    tmin : float, optional
        Minimum simulation time (in years) to consider when sampling outputs
        (default: 1e4).
    nsamp : int, optional
        Number of time samples to plot from each model (default: 5).
    plot_format : str, optional
        File format for the saved plot (e.g., "pdf", "png")
        (default: "pdf").

    Returns
    -------
    None
        The function produces and saves a plot but does not return a value.
    """

    plottimes1, plottimes2, profiles1, profiles2 = read_2model_data(
        output_dir1, output_dir2, extension, tmin, nsamp
    )
    t1 = int(str(plottimes1[0]))
    t2 = int(str(plottimes1[-1]))

    log.info('Plot atmosphere temperatures colourbar')

    norm = mpl.colors.LogNorm(vmin=max(t1, 1), vmax=t2)
    sm = plt.cm.ScalarMappable(cmap=cm.batlowK_r, norm=norm)
    sm.set_array([])

    # Initialise plot
    scale = 1.1
    alpha = 0.6
    fig, ax = plt.subplots(1, 1, figsize=(5 * scale, 4 * scale))
    ax.set_ylabel('Pressure [bar]')
    ax.set_xlabel('Temperature [K]')
    ax.invert_yaxis()
    ax.set_yscale('log')

    tmp_max = 1000.0
    prs_max = 1.0
    for i, t in enumerate(plottimes1):
        prof1 = profiles1[i]

        color = sm.to_rgba(t)
        tmp1 = prof1['t']
        prs1 = prof1['p'] / 1e5

        tmp_max = max(tmp_max, np.amax(tmp1))
        prs_max = max(prs_max, np.amax(prs1))

        ax.plot(tmp1, prs1, color=color, linestyle='-', alpha=alpha, zorder=3)

    for i, t in enumerate(plottimes2):
        prof2 = profiles2[i]

        color = sm.to_rgba(t)

        tmp2 = prof2['t']
        prs2 = prof2['p'] / 1e5

        tmp_max = max(tmp_max, np.amax(tmp2))
        prs_max = max(prs_max, np.amax(prs2))

        ax.plot(tmp2, prs2, color=color, linestyle='--', alpha=alpha, zorder=3)

    # Grid
    ax.grid(alpha=0.2, zorder=2)
    ax.set_xlim(0, tmp_max + 100)
    ax.xaxis.set_minor_locator(MultipleLocator(base=250))

    ax.set_ylim(bottom=prs_max, top=np.amin(prs1))
    ax.yaxis.set_major_locator(LogLocator())

    # Plot colourbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(sm, cax=cax, orientation='vertical')
    cbar.set_label('Time [yr]')

    # Save plot
    fname = os.path.join(output_dir1, 'plots', 'plot_atmosphere_comparison.%s' % plot_format)
    fig.savefig(fname, bbox_inches='tight', dpi=300)


if __name__ == '__main__':
    output_dir1 = sys.argv[1]
    output_dir2 = sys.argv[2]

    plot_atmosphere_comparison(
        output_dir1, output_dir2, tmin=1e4, extension='_atm.nc', nsamp=5, plot_format='pdf'
    )
