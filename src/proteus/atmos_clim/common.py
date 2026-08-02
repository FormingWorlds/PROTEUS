# Common atmosphere climate model functions
from __future__ import annotations

import glob
import logging
import os
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.helper import find_nearest

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)


class LevelsSource(Enum):
    """Origin of the fallback level properties held on `Atmos_t`.

    CONVERGED_SOLVE: levels recorded from a solve this run accepted.
    COMMITTED_ROW: levels taken from the last helpfile row written before this
    run began, which is what a resumed run has until it converges a solve of
    its own.
    """

    CONVERGED_SOLVE = auto()
    COMMITTED_ROW = auto()


# Atmosphere structure class
class Atmos_t:
    def __init__(self):
        # Atmosphere object internal to JANUS or AGNI
        self._atm = None

        # The column JANUS last solved, and JANUS only: None under every other
        # atmosphere module. JANUS copies `_atm` before it integrates and
        # resamples the copy onto the radiative grid, so only the copy carries
        # a profile at all. `_atm` keeps the surface boundary condition between
        # iterations and nothing else, its profile arrays sitting at their
        # allocated length with only the boundary cell written, so anything
        # that wants the atmosphere itself has to read this instead. AGNI
        # solves its column in place and needs no second reference.
        self._atm_janus_last = None

        # Whether the most recent atmosphere call converged. For AGNI this
        # is True iff the Newton/LM solver converged on at least one attempt;
        # JANUS, dummy, and transparent solvers always set it True. The main
        # coupling loop reads this to detect AGNI deadlocks (consecutive
        # failures with no interior state change). Transient, not persisted.
        self.converged: bool = True

        # Photospheric and XUV level properties from the most recent solve
        # that converged, keyed as in the helpfile row. Empty until this run
        # converges a solve of its own; a run that has to substitute before
        # then falls back on the last committed row. Transient, not persisted.
        self.levels_converged: dict[str, float] = {}

        # Where the levels above came from. Tracked here rather than re-derived
        # per call, since a run that has only ever fallen back on the committed
        # row must not report those levels as converged. Transient.
        self.levels_source: LevelsSource = LevelsSource.CONVERGED_SOLVE

        # Number of consecutive iterations whose level properties were not
        # produced by a converged solve of this run, whether they were
        # substituted from the record or kept from a rejected structure for
        # want of anything better. Reset to zero by the next converged solve.
        # Mirrored to the helpfile column `atm_levels_stale` on every
        # iteration, so it is readable per row after the run.
        self.levels_stale_iters: int = 0

        # Atmosphere solves this run has made. Only the first one may fall back
        # on the rows committed before the run started; after that, an empty
        # record means this run's own rows carry rejected levels too.
        self.solves_seen: int = 0


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


def read_ncdf_profile(nc_fpath: str, extra_keys: list = [], combine_edges: bool = True) -> dict:
    """Read data from atmosphere NetCDF output file.

    All variables in SI units, same as NetCDF file content.

    Automatically reads pressure (p), temperature (t), radius (z) arrays.
    If `combine_edges` is True, cell-centre (N) and cell-edge (N+1) values
    are interleaved into a single combined array of length (2*N+1).

    Extra keys can be read-in using the extra_keys parameter. These will be stored with
    the same dimensions as in the NetCDF file.

    Parameters
    ----------
        nc_fpath : str
            Path to NetCDF file.
        extra_keys : list
            List of extra keys (strings) to read from the file.
        combine_edges : bool
            Whether to combine cell-centre and cell-edge values into a single array.

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

    if 'gravity' not in ds.variables:
        raise KeyError(f"NetCDF file '{nc_fpath}' is missing required variable 'gravity'")
    g = np.array(ds.variables['gravity'][:])

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

    # read pressure, temperature, gravity and height data into dictionary values
    out = {}
    if combine_edges:
        out['p'] = [pl[0]]
        out['t'] = [tl[0]]
        out['g'] = [g[0]]  # Edge 0: use first cell centre value as fallback
        out['z'] = [zl[0]]
        out['r'] = [rl[0]]
        for i in range(nlev_c):
            out['p'].append(p[i])
            out['p'].append(pl[i + 1])

            out['t'].append(t[i])
            out['t'].append(tl[i + 1])

            out['g'].append(g[i])  # Cell centre i
            out['g'].append(g[i])  # Edge i+1: use current cell centre value

            out['z'].append(z[i])
            out['z'].append(zl[i + 1])

            out['r'].append(r[i])
            out['r'].append(rl[i + 1])
    else:
        out['p'] = p
        out['t'] = t
        out['g'] = g
        out['z'] = z
        out['r'] = r
        out['pl'] = pl
        out['tmpl'] = tl
        out['zl'] = zl
        out['rl'] = rl

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
        if key == 'gases':
            gas_l = ds.variables['gases'][:]  # names (bytes matrix)
            gases = []
            for igas, gas in enumerate(gas_l):
                gas_lbl = ''.join([c.decode(encoding='utf-8') for c in gas]).strip()
                gases.append(gas_lbl)
            out['gases'] = gases

        elif key == 'x_gas':
            gas_l = ds.variables['gases'][:]  # names (bytes matrix)
            gas_x = ds.variables['x_gas'][:]  # vmrs (float matrix)

            # get data for each gas
            for igas, gas in enumerate(gas_l):
                gas_lbl = ''.join([c.decode(encoding='utf-8') for c in gas]).strip()
                out[gas_lbl + '_vmr'] = np.array(gas_x[:, igas])

        elif key == 'aerosols':
            if 'aerosols' in ds.variables.keys():
                aer_l = ds.variables['aerosols'][:]  # names (bytes matrix)
                aerosols = []
                for iaer, aer in enumerate(aer_l):
                    aer_lbl = ''.join([c.decode(encoding='utf-8') for c in aer]).strip()
                    if len(aer_lbl) > 0:
                        aerosols.append(aer_lbl)
                out['aerosols'] = aerosols

        elif key == 'aer_mmr':
            if 'aer_mmr' in ds.variables.keys():
                aer_l = ds.variables['aerosols'][:]  # names (bytes matrix)
                aer_x = ds.variables['aer_mmr'][:]  # mmrs (float matrix)

                # get data for each aerosol
                for iaer, aer in enumerate(aer_l):
                    aer_lbl = ''.join([c.decode(encoding='utf-8') for c in aer]).strip()
                    if len(aer_lbl) > 0:
                        out[aer_lbl + '_mmr'] = np.array(aer_x[:, iaer])

        else:
            out[key] = np.array(ds.variables[key][:])

    # close file
    ds.close()

    # convert to np arrays
    for key in out.keys():
        try:
            out[key] = np.array(out[key], dtype=float)
        except (AttributeError, TypeError, ValueError):
            out[key] = np.array(out[key])

    return out


def read_atmosphere_data(output_dir: str, times: list, extra_keys=[]):
    """Return atmosphere profiles from NetCDF in PROTEUS output folder, at the given times.

    Arguments
    ----------
        output_dir : str
            Path to PROTEUS output folder.
        times : list
            List of times (floats) to read in [yr].
        extra_keys : list (optional)
            List of extra keys to read from the NetCDF files.

    Returns
    ----------
        list of dicts, or None.
            Each dict contains atmos data at each time.
            If any of the requested times cannot be read, None is returned.
    """
    profiles = [
        read_ncdf_profile(
            os.path.join(output_dir, 'data', '%.0f_atm.nc' % t), extra_keys=extra_keys
        )
        for t in times
    ]
    if None in profiles:
        log.warning('One or more NetCDF files could not be found')
        if os.path.exists(os.path.join(output_dir, 'data', 'data.tar')):
            log.warning('You may need to extract archived data files')
        return

    return profiles


def find_latest_atmosphere_time(output_dir: str) -> float | None:
    """Return the largest available ``*_atm.nc`` snapshot time on disk.

    Arguments
    ----------
        output_dir : str
            Path to PROTEUS output folder.

    Returns
    ----------
        float or None
            Largest snapshot time [yr]. None if no snapshots found.
    """

    # Find netcdf files and get times from their names
    ncs = glob.glob(os.path.join(output_dir, 'data', '*_atm.nc'))
    times = []
    for f in ncs:
        try:
            times.append(float(os.path.basename(f).split('_atm')[0]))
        except ValueError:
            log.warning(f"Could not parse time from NetCDF file '{f}'")

    # Return None if no files found
    if not times:
        return None

    # Return latest (max) time
    return float(max(times))


def get_spfile_name_and_bands(config: Config):
    """
    Get spectral file name and bands from config
    """

    group = config.atmos_clim.spectral_group
    bands = config.atmos_clim.spectral_bands

    return group, bands


def get_spfile_path(fwl_dir: str, config: Config):
    """
    Get path to spectral file, given name and bands.
    """

    # Get group and bands (strings) from config
    group, bands = get_spfile_name_and_bands(config)

    # Construct file path
    return os.path.join(fwl_dir, 'spectral_files', group, bands, group) + '.sf'


def clip_radius_to_hill(config: Config, hf_row: dict, radius: float) -> float:
    """Limit a level radius to the Hill radius, never below the solid body.

    Gas beyond the Hill radius is not bound to the planet, so an XUV radius
    outside it sizes the escape cross-section with material the planet does
    not hold, and the energy-limited rate grows as the cube of the excess.
    The limit is ``escape.hill_clamp_frac`` of the Hill radius, floored at
    ``R_int`` since the solid body is always bound.

    Parameters
    ----------
        config : Config
            Configuration options for PROTEUS.
        hf_row : dict
            Current helpfile row; provides ``hill_radius`` and ``R_int``.
        radius : float
            Level radius to limit [m].

    Returns
    ----------
        float
            The radius, limited when the clip is enabled and applicable.
    """
    if not getattr(config.escape, 'hill_clamp', False):
        return radius

    # Zero before the first orbit update; nothing to clip against yet.
    r_hill = float(hf_row.get('hill_radius', 0.0))
    if not np.isfinite(r_hill) or r_hill <= 0.0:
        return radius

    frac = float(getattr(config.escape, 'hill_clamp_frac', 1.0))
    r_limit = max(frac * r_hill, float(hf_row.get('R_int', 0.0)))
    if radius <= r_limit:
        return radius

    log.warning(
        'Level radius %.4e m exceeds %.3g of the Hill radius (%.4e m); clipping to %.4e m',
        radius,
        frac,
        r_hill,
        r_limit,
    )
    return r_limit


def get_oarr_from_parr(p_arr: list, o_arr: list, p_tgt: float) -> tuple:
    """
    Get the value of o_array corresponding to the p_tgt level in p_arr.

    Parameters
    ----------------
        p_arr: list
            Pressure array
        o_arr: list
            Other array (e.g. radius)
        p_tgt: float
            Target pressure

    Returns
    ----------------
        p_close: float
            Closest pressure in the array
        o_close: float
            Closest value in the other array
    """

    p_close, idx = find_nearest(p_arr, p_tgt)
    return float(p_close), float(o_arr[idx])


def get_radius_from_pressure(p_arr: list, r_arr: list, p_tgt: float) -> tuple[float, float]:
    """Return the radius at a target pressure.

    Thin wrapper around the generic ``get_oarr_from_parr`` for the
    pressure-to-radius case.
    """
    return get_oarr_from_parr(p_arr, r_arr, p_tgt)
