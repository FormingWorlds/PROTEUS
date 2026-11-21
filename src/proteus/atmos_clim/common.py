# Common atmosphere climate model functions
from __future__ import annotations

import logging
import math
import os
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from proteus.utils.constants import const_sigma
from proteus.utils.helper import (
    UpdateStatusfile,
    find_nearest,
    list_snapshot_times,
    snapshot_path_for_time,
)

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


def ncdf_flag_to_bool(var) -> bool | None:
    """Convert NetCDF flag (y/n) to Python bool (true/false)"""
    v = str(var[0].tobytes().decode()).lower()

    # check against expected
    if v == 'y':
        return True
    elif v == 'n':
        return False
    else:
        log.error(f'Could not parse NetCDF atmos flag variable: {var}')
        return None


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
        log.error(f"NetCDF file '{nc_fpath}' is missing required variable 'gravity'")
        g = np.zeros_like(p)  # fallback to zeros
    else:
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
        out[fk] = False  # default if not found
        if fk in ds.variables.keys():
            out[fk] = ncdf_flag_to_bool(ds.variables[fk])
            if out[fk] is None:
                out[fk] = False

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
            snapshot_path_for_time(os.path.join(output_dir, 'data'), t, '_atm.nc'),
            extra_keys=extra_keys,
        )
        for t in times
    ]
    if None in profiles:
        log.warning('One or more NetCDF files could not be found')
        if os.path.exists(os.path.join(output_dir, 'data', 'data.tar')):
            log.warning('You may need to extract archived data files')
        return

    return profiles


def list_atmosphere_times(output_dir: str) -> list[float]:
    """Return the times of the ``*_atm.nc`` snapshots on disk.

    Arguments
    ----------
        output_dir : str
            Path to PROTEUS output folder.

    Returns
    ----------
        list of float
            Snapshot times [yr] in ascending order. Empty if none are found.
    """
    return list_snapshot_times(os.path.join(output_dir, 'data'), '_atm.nc')


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
    times = list_atmosphere_times(output_dir)
    if not times:
        return None
    return float(times[-1])


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

    The file lies in the version directory of the spectral-file dataset for the
    configured group and band count; the path may not exist before the dataset
    is fetched. A group and band count that no manifest declares is read from
    ``<fwl_dir>/spectral_files/<group>/<bands>/<group>.sf``.
    """
    from proteus.data import dataset_dir, spectral_file_key

    # Get group and bands (strings) from config
    group, bands = get_spfile_name_and_bands(config)

    try:
        version_dir = dataset_dir(spectral_file_key(group, bands), data_root=fwl_dir)
    except KeyError:
        path = os.path.join(fwl_dir, 'spectral_files', group, str(bands), group) + '.sf'
        log.info('Spectral file %s/%s is in no manifest; using %s', group, bands, path)
        return path
    return os.path.join(version_dir, group) + '.sf'


def require_spfile_path(dirs: dict, config: Config) -> str:
    """Return the configured spectral file, stopping the run when it is missing.

    Raises
    ------
    FileNotFoundError
        After writing the error status, naming the download command.
    """
    from proteus.utils.data import RELOCATE_HINT, SPECTRAL_FILE_FOLDERS

    path = get_spfile_path(dirs['fwl'], config)
    if not os.path.isfile(path):
        UpdateStatusfile(dirs, 20)
        group, bands = get_spfile_name_and_bands(config)
        fix = (
            f'Fetch it with `proteus get spectral`. {RELOCATE_HINT}'
            if f'{group}/{bands}' in SPECTRAL_FILE_FOLDERS
            else 'No manifest declares this group and band count, so place the file there.'
        )
        raise FileNotFoundError(f"Spectral file does not exist at '{path}'. {fix}")
    return path


def _lid_skin_conductance(hf_row: dict) -> float:
    """Conductance [W m-2 K-1] of the skin in the helpfile columns, positive under a lid.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row, with ``boundary_layer_thickness`` and
        ``boundary_layer_conductivity``.

    Returns
    -------
    float
        ``k / d`` of the skin.

    Raises
    ------
    ValueError
        If the skin has no resistance: the parallel sum has no open path without a skin, as
        on a resume from a helpfile without ``boundary_layer_conductivity``.
    """
    d, k = (
        float(hf_row['boundary_layer_thickness']),
        float(hf_row['boundary_layer_conductivity']),
    )
    if not (d > 0.0 and k > 0.0):
        raise ValueError(
            f'a live Aragog lid needs a skin with a thickness and k, got {d}, {k}; '
            'a helpfile without boundary_layer_conductivity resumes with k = 0'
        )
    return k / d


def lid_acts(hf_row: dict) -> bool:
    """True where the row has a live lid: a lid weight and a lid conductance above 0."""
    return (
        float(hf_row.get('lid_weight', 0.0)) > 0.0
        and float(hf_row.get('lid_conductance', 0.0)) > 0.0
    )


def lid_boundary(hf_row: dict) -> tuple[float, float, float]:
    """Skin and melt temperature for the surface balance of the atmosphere.

    The lid and the open surface act in parallel on one surface temperature:
    ``F = w G_lid (T_b - T) + (1 - w) G_skin (T_m - T) = k / d (T_melt - T)``, with
    ``G_lid`` at most ``G_skin`` so that a lid never conducts more than the open surface, and
    ``T_m = max(T_node, T_b)``: the open part is melt, so its node is held at least at the lid
    base (a construct of the coupling; the solver mesh keeps its own top node). Without a lid
    the columns and ``T_magma`` pass unchanged.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row, with the skin columns and ``T_magma``. A lid acts with
        ``lid_weight``, ``lid_conductance``, ``lid_T_base`` and ``lid_T_node``.

    Returns
    -------
    tuple of float
        Skin thickness ``d`` [m], skin conductivity ``k`` [W m-1 K-1] and ``T_melt`` [K].
    """
    d, k = (
        float(hf_row['boundary_layer_thickness']),
        float(hf_row['boundary_layer_conductivity']),
    )
    if not lid_acts(hf_row):
        return d, k, float(hf_row['T_magma'])
    w, G_lid = float(hf_row['lid_weight']), float(hf_row['lid_conductance'])
    G_skin = _lid_skin_conductance(hf_row)
    G_lid = min(G_lid, G_skin)
    G = w * G_lid + (1.0 - w) * G_skin
    T_b = float(hf_row['lid_T_base'])
    T_m = max(float(hf_row['lid_T_node']), T_b)
    T_melt = w * G_lid * T_b + (1.0 - w) * G_skin * T_m
    return k / G, k, T_melt / G


# smallest surface step [K] of the 2 rows that give the slope of the lid cap
LID_SLOPE_DT_MIN = 0.01


def _earlier_rows(hf_all, hf_row: dict):
    """The rows of ``hf_all`` before ``hf_row`` (which may be its last row), or None if none."""
    if hf_all is None or len(hf_all) == 0:
        return None
    if float(hf_all['Time'].iloc[-1]) == float(hf_row['Time']):
        hf_all = hf_all.iloc[:-1]
    return hf_all if len(hf_all) else None


def _usable_secant(T0: float, T1: float, F0: float, F1: float) -> float:
    """Secant ``(F1 - F0) / (T1 - T0)`` [W m-2 K-1] of ``F_atm`` with ``T_surf``; 0 for a surface
    step below ``LID_SLOPE_DT_MIN``, a slope of 0 or below, or one that is not finite."""
    dT = T1 - T0
    if not abs(dT) >= LID_SLOPE_DT_MIN:
        return 0.0
    slope = (F1 - F0) / dT
    return float(slope) if math.isfinite(slope) and slope > 0.0 else 0.0


def lid_flux_secant(hf_all, hf_row: dict) -> tuple[float, float | None]:
    """Slope ``F'`` [W m-2 K-1] of ``F_atm`` with ``T_surf`` over the last 2 rows, with the
    surface of the row before, the other end of that secant.

    Parameters
    ----------
    hf_all : pandas.DataFrame or None
        Helpfile rows, with or without the current row as the last one.
    hf_row : dict
        Current helpfile row, with ``Time``, ``T_surf`` and ``F_atm``.

    Returns
    -------
    tuple
        ``(slope, T_other)``: ``(F_atm - F_atm before) / (T_surf - T_surf before)`` and the
        ``T_surf`` before [K]. The slope is 0 for a surface step below ``LID_SLOPE_DT_MIN``,
        for a slope of 0 or below, or a non-finite one; ``(0, None)`` without a row before.
    """
    rows = _earlier_rows(hf_all, hf_row)
    if rows is None:
        return 0.0, None
    prev = rows.iloc[-1]
    T_other = float(prev['T_surf'])
    F_prev, F_now = float(prev['F_atm']), float(hf_row['F_atm'])
    return _usable_secant(T_other, float(hf_row['T_surf']), F_prev, F_now), T_other


def lid_fallback_slope(hf_all, hf_row: dict) -> tuple[float, int, int]:
    """Slope ``F'`` [W m-2 K-1] for a live lid whose last secant is unusable.

    Parameters
    ----------
    hf_all : pandas.DataFrame or None
        Helpfile rows, with or without the current row as the last one.
    hf_row : dict
        Current helpfile row, with ``Time`` and ``T_surf``.

    Returns
    -------
    tuple
        ``(slope, rows_back, code)``: the newest usable secant of 2 consecutive earlier rows
        (the rules of :func:`lid_flux_secant`), the number of rows from its later row to the
        current one, and ``LID_SLOPE_GUARD_FALLBACK``; without one in the run, the grey tangent
        ``4 sigma T_surf^3``, 0 and ``LID_SLOPE_GUARD_GREY``.
    """
    rows = _earlier_rows(hf_all, hf_row)
    if rows is not None and len(rows) >= 2:
        T = rows['T_surf'].to_numpy(dtype=float)
        F = rows['F_atm'].to_numpy(dtype=float)
        for i in range(len(T) - 1, 0, -1):
            slope = _usable_secant(T[i - 1], T[i], F[i - 1], F[i])
            if slope > 0.0:
                return slope, len(T) - i, LID_SLOPE_GUARD_FALLBACK
    return 4.0 * const_sigma * float(hf_row['T_surf']) ** 3, 0, LID_SLOPE_GUARD_GREY


def lid_flux_cap(hf_row: dict) -> float | None:
    """Largest surface flux [W m-2] of the lid in the next interior step, or None.

    It is the lid branch of the last atmosphere balance, ``G (T_b - T_surf)`` with the
    conductance of the call (at most ``G_skin``), so that the lid loses at most what the
    atmosphere gave it.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row after the atmosphere step, with ``T_surf`` and the lid columns.

    Returns
    -------
    float or None
        The cap, or None without a lid or for a surface at or above ``T_b``.
    """
    if not lid_acts(hf_row):
        return None
    G_lid = float(hf_row['lid_conductance'])
    C = min(G_lid, _lid_skin_conductance(hf_row)) * (
        float(hf_row['lid_T_base']) - float(hf_row['T_surf'])
    )
    return C if C > 0.0 else None


# open area 1 - w at or below which the open part takes F_atm
LID_OPEN_AREA_FLOOR = 1e-9
# Size of the residual F_atm - B per total area [W m-2] above which a call is logged; absolute,
# since a bound relative to F_atm would warn wherever F_atm crosses 0
LID_RESID_WARN = 1.0


class LidOpenFluxError(RuntimeError):
    """The open part of a live lid has no start value: an input of it is not finite."""


def check_lid_inputs(hf_row: dict) -> None:
    """Refuse a lid column that is not finite where the coupling reads it.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row, with the lid columns of the interior step.

    Raises
    ------
    LidOpenFluxError
        For a ``lid_weight`` that is not finite, a ``lid_conductance`` that is not finite under
        a weight above 0, or a ``lid_T_node`` or ``lid_T_base`` that is not finite under a live
        lid (:func:`lid_acts`).
    """
    w = float(hf_row.get('lid_weight', 0.0))
    G = float(hf_row.get('lid_conductance', 0.0)) if w > 0.0 else 0.0
    live = w > 0.0 and G > 0.0
    T = float(hf_row['lid_T_node']) if live else 0.0
    T_b = float(hf_row['lid_T_base']) if live else 0.0
    if not all(math.isfinite(x) for x in (w, G, T, T_b)):
        raise LidOpenFluxError(
            f'Lid columns: lid_weight {hf_row.get("lid_weight")}, lid_conductance '
            f'{hf_row.get("lid_conductance")}, lid_T_node {hf_row.get("lid_T_node")} or '
            f'lid_T_base {hf_row.get("lid_T_base")} is not finite'
        )


def lid_open_flux(hf_row: dict) -> float:
    """Open-part flux [W m-2] per open area at the start of the step, the column ``lid_F_open``.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row after the atmosphere step, with ``T_surf`` and ``F_atm``.

    Returns
    -------
    float
        ``F_atm`` without a lid. Where a lid acts, ``(F_atm - w L) / (1 - w)`` per open area,
        with ``L = min(G_lid, G_skin) (T_b - T_surf)`` the signed flux of the lid part in the
        call; ``F_atm`` where the open area ``1 - w`` is at or below ``LID_OPEN_AREA_FLOOR``.

    Raises
    ------
    LidOpenFluxError
        For an ``F_atm``, ``T_surf`` or ``lid_T_base`` that is not finite under a live lid. The
        other lid columns are those that :func:`check_lid_inputs` passed before the solve.

    Notes
    -----
    The lid part at ``L`` and the open part sum to ``F_atm`` for any sign of ``F_atm`` and of
    ``T_b - T_surf``. The value tends to ``F_atm`` as the lid weight goes to 0. It is a
    diagnostic: the interior takes ``F_atm`` and the fields of :func:`lid_open_fields`.
    """
    if not lid_acts(hf_row):
        return float(hf_row['F_atm'])
    G = min(float(hf_row['lid_conductance']), _lid_skin_conductance(hf_row))
    F_atm = float(hf_row['F_atm'])
    L = G * (float(hf_row['lid_T_base']) - float(hf_row['T_surf']))
    if not math.isfinite(F_atm + L):
        raise LidOpenFluxError(
            f'Lid open part: F_atm {F_atm} or lid flux {L} is not finite (T_surf '
            f'{hf_row["T_surf"]}, lid_T_base {hf_row["lid_T_base"]})'
        )
    w = float(hf_row['lid_weight'])
    if not 1.0 - w > LID_OPEN_AREA_FLOOR:
        return F_atm
    return (F_atm - w * L) / (1.0 - w)


def lid_open_residual(hf_row: dict) -> float:
    """Residual [W m-2] of the atmosphere solve against the skin balance of the call, per total area.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row after the atmosphere step, with ``T_surf`` and ``F_atm``.

    Returns
    -------
    float
        ``F_atm - B``, with ``B`` the skin balance of the call (:func:`lid_boundary`); where
        ``1 - w`` is above ``LID_OPEN_AREA_FLOOR``, divided by ``1 - w`` it is the start value
        of :func:`lid_open_flux` less the skin flux ``G_skin (T_m - T_surf)``. 0 without a lid.

    Notes
    -----
    The call is logged where the size of the residual is above ``LID_RESID_WARN`` (1 W m-2),
    an absolute bound: the residual is a diagnostic, and a bound relative to ``F_atm`` would warn
    on every step where ``F_atm`` crosses 0.
    """
    if not lid_acts(hf_row):
        return 0.0
    d, k, T_melt = lid_boundary(hf_row)
    return float(hf_row['F_atm']) - k / d * (T_melt - float(hf_row['T_surf']))


# guard codes: 1 is the return code of lid_open_slope for an unusable slope, which a run does not
# record; lid_slope_guard records a cut (2), the newest usable earlier secant (3), the grey
# tangent without one (4), or that earlier secant cut to the grey tangent (5)
LID_SLOPE_GUARD_SIGN = 1
LID_SLOPE_GUARD_SIZE = 2
LID_SLOPE_GUARD_FALLBACK = 3
LID_SLOPE_GUARD_GREY = 4
LID_SLOPE_GUARD_FALLBACK_CUT = 5


def lid_open_slope(
    hf_row: dict, slope: float, T_other: float | None = None
) -> tuple[float, int]:
    """Slope ``F'`` [W m-2 K-1] of the open part of a lid surface, with its guard code.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row, with ``T_surf``.
    slope : float
        Slope of ``F_atm`` with ``T_surf`` [W m-2 K-1].
    T_other : float, optional
        Other end surface [K] of a secant ``slope`` through ``T_surf``, below or above it; the
        cut is then the grey secant ``sigma (T_other^2 + T_surf^2)(T_other + T_surf)`` over
        the same interval. None, not finite or not above 0: the tangent ``4 sigma T_surf^3``.

    Returns
    -------
    tuple
        ``(F', code)``: ``(0.0, LID_SLOPE_GUARD_SIGN)`` for a slope that is not finite or is 0
        or below; a slope above the grey secant over the same interval is cut to it with
        ``LID_SLOPE_GUARD_SIZE``; otherwise the slope and 0.
    """
    F_p = float(slope)
    if not (math.isfinite(F_p) and F_p > 0.0):
        return 0.0, LID_SLOPE_GUARD_SIGN
    T_s = float(hf_row['T_surf'])
    T_e = float(T_other) if T_other is not None and 0.0 < T_other < math.inf else T_s
    F_grey = const_sigma * (T_e**2 + T_s**2) * (T_e + T_s)
    return (F_grey, LID_SLOPE_GUARD_SIZE) if F_p > F_grey else (F_p, 0)


def lid_open_fields(
    hf_row: dict, slope: float, T_other: float | None = None
) -> tuple[dict, int]:
    """Fields of the call balance for the open part of a live lid, per total area.

    Parameters
    ----------
    hf_row : dict
        Current helpfile row after the atmosphere step, with ``T_surf`` and the lid columns.
    slope, T_other
        As in :func:`lid_open_slope`.

    Returns
    -------
    tuple
        ``(fields, code)``: ``open_lid_flux`` ``L = G_l (T_b - T_surf)`` [W m-2],
        ``open_surface_slope`` ``F'`` (cut as :func:`lid_open_slope`), ``open_skin_conductance``
        ``G`` and ``open_lid_conductance`` ``G_l = min(G_lid, G)`` [W m-2 K-1]; the guard code
        of ``F'``.
    """
    F_p, code = lid_open_slope(hf_row, slope, T_other)
    G = _lid_skin_conductance(hf_row)
    G_l = min(float(hf_row['lid_conductance']), G)
    L = G_l * (float(hf_row['lid_T_base']) - float(hf_row['T_surf']))
    return {
        'open_lid_flux': L,
        'open_surface_slope': F_p,
        'open_skin_conductance': G,
        'open_lid_conductance': G_l,
    }, code


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
