# Function and classes used to run SPIDER
from __future__ import annotations

import glob
import json
import logging
import os
import platform
import subprocess as sp
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from proteus.interior.common import Interior_t, get_file_tides
from proteus.interior.timestep import next_step
from proteus.utils.constants import radnuc_data
from proteus.utils.helper import UpdateStatusfile, natural_sort, recursive_get

if TYPE_CHECKING:
    from proteus.config import Config

log = logging.getLogger('fwl.' + __name__)

FWL_DATA_DIR = os.environ.get('FWL_DATA', '')
MELTING_CURVES_DIR = os.path.join(FWL_DATA_DIR, 'interior_lookup_tables', 'Melting_curves')
EOS_DYNAMIC_DIR = os.path.join(FWL_DATA_DIR, 'interior_lookup_tables', 'EOS', 'dynamic')


def _coresize_from_mesh(mesh_file: str) -> float:
    """Extract fractional core radius from a SPIDER external mesh file.

    Parameters
    ----------
    mesh_file : str
        Path to the SPIDER mesh file (header + basic nodes + staggered nodes).

    Returns
    -------
    float
        Core-to-surface radius ratio (R_cmb / R_surface).
    """
    with open(mesh_file) as f:
        header = f.readline()
        nb = int(header.strip('# \n').split()[0])
        # First basic node = surface, last basic node = CMB
        r_surface = float(f.readline().split()[0])
        for _ in range(nb - 2):
            f.readline()
        r_cmb = float(f.readline().split()[0])
    return r_cmb / r_surface


def _check_eos_table_range(eos_dir: str, mesh_file: str | None, P_cmb: float):
    """Warn if CMB pressure approaches EOS table limits.

    Reads the header of the solid-phase density table to determine the
    entropy range.  When the solid-phase table's entropy coverage is
    narrow relative to the melt-phase table, partially-solid nodes near
    the CMB can trigger out-of-range lookups that produce unphysical
    material properties (e.g. negative thermal expansion).

    Parameters
    ----------
    eos_dir : str
        Path to the P-S EOS lookup directory.
    mesh_file : str or None
        Path to the external mesh file (used for context in the warning).
    P_cmb : float
        Pressure at the core-mantle boundary [Pa].
    """
    solid_file = os.path.join(eos_dir, 'density_solid.dat')
    melt_file = os.path.join(eos_dir, 'density_melt.dat')
    if not os.path.isfile(solid_file) or not os.path.isfile(melt_file):
        return

    def _read_entropy_range(filepath):
        """Read the entropy axis scaling and bounds from a P-S table header."""
        with open(filepath) as f:
            header = f.readline()  # "# HEAD NX NY"
            tokens = header.strip('# \n').split()
            head = int(tokens[0])
            nx = int(tokens[1])
            ny = int(tokens[2])
            # Skip to the scaling-factor line (last header line)
            for _ in range(head - 2):
                f.readline()
            scales = f.readline().strip('# \n').split()
            y_scale = float(scales[1])  # entropy scaling factor

            # First data line gives y_min
            first_data = f.readline().split()
            y_min = float(first_data[1]) * y_scale

            # Last unique y value is at the start of the last block:
            # skip (ny-1)*nx - 1 lines to reach it
            skip = (ny - 1) * nx - 1
            for _ in range(skip):
                f.readline()
            last_data = f.readline().split()
            y_max = float(last_data[1]) * y_scale
        return y_min, y_max

    try:
        s_min_solid, s_max_solid = _read_entropy_range(solid_file)
        s_min_melt, s_max_melt = _read_entropy_range(melt_file)
    except (ValueError, IndexError, IOError):
        return  # cannot parse, skip check

    # The critical scenario: if the initial adiabat entropy exceeds the
    # solid-phase table maximum, partially-solid nodes will get clamped
    # lookups that may produce negative thermal expansion.
    if s_max_solid < s_max_melt:
        log.warning(
            'Solid-phase EOS table entropy range (%.0f–%.0f J/kg/K) is narrower '
            'than the melt-phase range (%.0f–%.0f J/kg/K). '
            'If the initial adiabat entropy exceeds %.0f J/kg/K at partially-solid '
            'nodes (high CMB pressure), the solid-phase lookup will be clamped, '
            'potentially producing unphysical material properties.',
            s_min_solid,
            s_max_solid,
            s_min_melt,
            s_max_melt,
            s_max_solid,
        )



# Nondimensional scaling reference (must match SPIDER call_sequence below)
RADIUS0 = 63710000.0  # m


def _read_mesh_file(mesh_path: str) -> tuple[np.ndarray, ...]:
    """Parse a SPIDER external mesh file.

    Parameters
    ----------
    mesh_path : str
        Path to the mesh file.

    Returns
    -------
    r_b, P_b, rho_b, g_b : np.ndarray
        Basic-node radius [m], pressure [Pa], density [kg/m3],
        gravity [m/s2] (surface to CMB).
    r_s, P_s, rho_s, g_s : np.ndarray
        Staggered-node radius [m], pressure [Pa], density [kg/m3],
        gravity [m/s2] (surface to CMB).
    """
    with open(mesh_path) as f:
        header = f.readline()
        tokens = header.strip('# \n').split()
        nb = int(tokens[0])
        ns = int(tokens[1])

        r_b = np.empty(nb)
        P_b = np.empty(nb)
        rho_b = np.empty(nb)
        g_b = np.empty(nb)
        for i in range(nb):
            cols = f.readline().split()
            r_b[i] = float(cols[0])
            P_b[i] = float(cols[1])
            rho_b[i] = float(cols[2])
            g_b[i] = float(cols[3])

        r_s = np.empty(ns)
        P_s = np.empty(ns)
        rho_s = np.empty(ns)
        g_s = np.empty(ns)
        for i in range(ns):
            cols = f.readline().split()
            r_s[i] = float(cols[0])
            P_s[i] = float(cols[1])
            rho_s[i] = float(cols[2])
            g_s[i] = float(cols[3])

    return r_b, P_b, rho_b, g_b, r_s, P_s, rho_s, g_s


def _write_mesh_file(
    mesh_path: str,
    r_b: np.ndarray,
    P_b: np.ndarray,
    rho_b: np.ndarray,
    g_b: np.ndarray,
    r_s: np.ndarray,
    P_s: np.ndarray,
    rho_s: np.ndarray,
    g_s: np.ndarray,
) -> None:
    """Write a SPIDER external mesh file.

    Parameters
    ----------
    mesh_path : str
        Output file path.
    r_b, P_b, rho_b, g_b : np.ndarray
        Basic-node arrays (surface to CMB).
    r_s, P_s, rho_s, g_s : np.ndarray
        Staggered-node arrays (surface to CMB).
    """
    with open(mesh_path, 'w') as f:
        f.write(f'# {len(r_b)} {len(r_s)}\n')
        for i in range(len(r_b)):
            f.write(f'{r_b[i]:.15e} {P_b[i]:.15e} {rho_b[i]:.15e} {g_b[i]:.15e}\n')
        for i in range(len(r_s)):
            f.write(f'{r_s[i]:.15e} {P_s[i]:.15e} {rho_s[i]:.15e} {g_s[i]:.15e}\n')


def blend_mesh_files(old_path: str, new_path: str, max_shift: float = 0.05) -> float:
    """Clamp the maximum per-update mesh shift by blending old and new meshes.

    Compares radii in old and new mesh files. If the maximum fractional
    radius change exceeds ``max_shift``, linearly blends all columns
    (r, P, rho, g) so that the effective shift equals ``max_shift``, and
    overwrites ``new_path`` with the blended mesh.

    Parameters
    ----------
    old_path : str
        Path to the previous mesh file.
    new_path : str
        Path to the new mesh file (overwritten in-place if blending occurs).
    max_shift : float
        Maximum allowed fractional radius change per update.

    Returns
    -------
    float
        Actual maximum fractional radius shift before blending (0.0 if
        old file is missing or node counts differ).
    """
    if not os.path.isfile(old_path):
        return 0.0

    try:
        r_b_old, P_b_old, rho_b_old, g_b_old, r_s_old, P_s_old, rho_s_old, g_s_old = (
            _read_mesh_file(old_path)
        )
        r_b_new, P_b_new, rho_b_new, g_b_new, r_s_new, P_s_new, rho_s_new, g_s_new = (
            _read_mesh_file(new_path)
        )
    except Exception:
        log.warning('Could not read mesh files for blending')
        return 0.0

    if len(r_b_old) != len(r_b_new) or len(r_s_old) != len(r_s_new):
        log.warning(
            'Mesh node count changed (%d/%d -> %d/%d), skipping blend',
            len(r_b_old),
            len(r_s_old),
            len(r_b_new),
            len(r_s_new),
        )
        return 0.0

    actual_shift = float(np.max(np.abs(r_b_new - r_b_old) / np.abs(r_b_old)))

    if actual_shift <= max_shift:
        log.info(
            'Mesh shift %.3f%% within limit (%.1f%%), no blending needed',
            actual_shift * 100,
            max_shift * 100,
        )
        return actual_shift

    alpha = max_shift / actual_shift
    log.info(
        'Mesh shift %.1f%% exceeds limit %.1f%%, blending with alpha=%.4f',
        actual_shift * 100,
        max_shift * 100,
        alpha,
    )

    def _blend(old, new):
        return old + alpha * (new - old)

    _write_mesh_file(
        new_path,
        _blend(r_b_old, r_b_new),
        _blend(P_b_old, P_b_new),
        _blend(rho_b_old, rho_b_new),
        _blend(g_b_old, g_b_new),
        _blend(r_s_old, r_s_new),
        _blend(P_s_old, P_s_new),
        _blend(rho_s_old, rho_s_new),
        _blend(g_s_old, g_s_new),
    )
    return actual_shift


def _rewrite_json_solution(
    json_path: str,
    dSdxi_nondim: np.ndarray,
    S0_nondim: float,
) -> None:
    """Overwrite the entropy solution vector in a SPIDER JSON file.

    Modifies subdomains 0 (dS/dxi) and 1 (S0) in-place, leaving any
    volatile or reaction subdomains untouched.

    Parameters
    ----------
    json_path : str
        Path to the SPIDER JSON file.
    dSdxi_nondim : np.ndarray
        New nondimensional dS/dxi values at basic nodes.
    S0_nondim : float
        New nondimensional S0 (entropy at first staggered node).
    """
    with open(json_path) as f:
        data = json.load(f)

    sd = data['solution']['subdomain data']
    # Subdomain 0: dS/dxi at basic nodes
    sd[0]['values'] = [f'{v:.17e}' for v in dSdxi_nondim]
    # Subdomain 1: S0 (single value)
    sd[1]['values'] = [f'{S0_nondim:.17e}']

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)


def remap_entropy_for_new_mesh(
    json_path: str,
    new_mesh_file: str,
    radius_phys: float,
) -> bool:
    """Interpolate entropy from the old mesh to a new mesh in a SPIDER JSON.

    When Zalmoxis updates the mesh, physical node positions r[i] change.
    The old JSON stores dS/dxi computed on the old mesh; applying it with
    new xi spacing gives incorrect absolute entropy S(r).  This function
    reads S(r) from the old mesh, interpolates onto the new mesh positions,
    then rewrites the JSON solution vector with corrected dS/dxi and S0.

    Parameters
    ----------
    json_path : str
        Path to the SPIDER restart JSON file (modified in-place).
    new_mesh_file : str
        Path to the new external mesh file from Zalmoxis.
    radius_phys : float
        Physical planet surface radius [m] (for consistency check).

    Returns
    -------
    bool
        True if the JSON was modified, False if the mesh change was
        negligible (max relative radius shift < 1e-10).
    """
    # --- Read old mesh data from JSON ---
    with open(json_path) as f:
        jdata = json.load(f)

    data = jdata['data']
    r_s_old = np.array([float(v) for v in data['radius_s']['values']]) * float(
        data['radius_s']['scaling']
    )
    S_s_old = np.array([float(v) for v in data['S_s']['values']]) * float(
        data['S_s']['scaling']
    )
    r_b_old = np.array([float(v) for v in data['radius_b']['values']]) * float(
        data['radius_b']['scaling']
    )

    # --- Read new mesh ---
    r_b_new, _, _, _, r_s_new, _, _, _ = _read_mesh_file(new_mesh_file)
    N_b = len(r_b_new)
    N_s = N_b - 1

    if len(r_b_old) != N_b:
        log.warning(
            'Cannot remap entropy: node count changed (%d -> %d)',
            len(r_b_old),
            N_b,
        )
        return False

    # --- Check if mesh actually changed ---
    max_rel_diff = np.max(np.abs(r_b_new - r_b_old) / r_b_old)
    if max_rel_diff < 1e-10:
        log.debug('Mesh change negligible (max rel diff %.2e), skipping remap', max_rel_diff)
        return False

    log.info(
        'Remapping entropy for new mesh (max radius shift: %.2e relative)',
        max_rel_diff,
    )

    # --- Interpolate S(r) from old to new staggered positions ---
    # SPIDER ordering is surface-to-CMB (decreasing r), np.interp needs ascending
    r_s_old_asc = r_s_old[::-1]
    S_s_old_asc = S_s_old[::-1]
    r_s_new_asc = r_s_new[::-1]
    S_s_new_asc = np.interp(r_s_new_asc, r_s_old_asc, S_s_old_asc)
    S_s_new = S_s_new_asc[::-1]  # back to surface-to-CMB

    # --- Compute new xi coordinates (matches SetMeshRegular in mesh.c) ---
    R_surf = r_b_new[0]
    R_cmb = r_b_new[-1]
    coresize_new = R_cmb / R_surf
    radius_nondim = R_surf / RADIUS0

    dx_b = -radius_nondim * (1.0 - coresize_new) / (N_b - 1)
    xi_s_new = np.array(
        [radius_nondim * coresize_new - 0.5 * dx_b - (N_s - 1 - i) * dx_b for i in range(N_s)]
    )

    # --- Compute dS/dxi (nondimensional) from interpolated entropy ---
    # Get scalings from solution subdomains
    sd = jdata['solution']['subdomain data']
    S0_scaling = float(sd[1]['scaling'])

    S_s_nondim = S_s_new / S0_scaling
    S0_nondim = S_s_nondim[0]

    dSdxi_nondim = np.zeros(N_b)
    # Interior basic nodes: dS/dxi[i] = (S_s[i] - S_s[i-1]) / (xi_s[i] - xi_s[i-1])
    for i in range(1, N_b - 1):
        dSdxi_nondim[i] = (S_s_nondim[i] - S_s_nondim[i - 1]) / (xi_s_new[i] - xi_s_new[i - 1])
    # Boundary extrapolation
    dSdxi_nondim[0] = dSdxi_nondim[1]
    dSdxi_nondim[N_b - 1] = dSdxi_nondim[N_b - 2]

    # --- Write back to JSON ---
    _rewrite_json_solution(json_path, dSdxi_nondim, S0_nondim)

    log.info(
        'Entropy remapped: S0=%.2f J/kg/K, S range=[%.2f, %.2f] J/kg/K',
        S_s_new[0],
        S_s_new.min(),
        S_s_new.max(),
    )
    return True


class MyJSON(object):
    """load and access json data"""

    def __init__(self, filename):
        self.filename = filename
        self.data_d = None
        self._load()

    def _load(self):
        """
        Load json data from file, and store in the class.

        Returns False if file not found.
        """
        if not os.path.isfile(self.filename):
            return False
        with open(self.filename) as json_data:
            self.data_d = json.load(json_data)
        return True

    # was get_field_data
    def get_dict(self, keys):
        """get all data relating to a particular field"""
        dict_d = recursive_get(self.data_d, keys)
        return dict_d

    # was get_field_units
    def get_dict_units(self, keys):
        """get the units (SI) of a particular field"""
        dict_d = recursive_get(self.data_d, keys)
        units = dict_d['units']
        units = None if units == 'None' else units
        return units

    # was get_scaled_field_values
    def get_dict_values(self, keys, fmt_o=''):
        """get the scaled values for a particular quantity"""
        dict_d = recursive_get(self.data_d, keys)
        scaling = float(dict_d['scaling'])
        if len(dict_d['values']) == 1:
            values_a = float(dict_d['values'][0])
        else:
            values_a = np.array([float(value) for value in dict_d['values']])
        scaled_values_a = scaling * values_a
        if fmt_o:
            scaled_values_a = fmt_o.ascale(scaled_values_a)
        return scaled_values_a

    # was get_scaled_field_value_internal
    def get_dict_values_internal(self, keys, fmt_o=''):
        """get the scaled values for the internal nodes (ignore top
        and bottom nodes)"""
        scaled_values_a = self.get_dict_values(keys, fmt_o)
        return scaled_values_a[1:-1]

    def get_mixed_phase_boolean_array(self, nodes='basic'):
        """this array enables us to plot different linestyles for
        mixed phase versus single phase quantities"""
        if nodes == 'basic':
            phi = self.get_dict_values(['data', 'phi_b'])
        elif nodes == 'basic_internal':
            phi = self.get_dict_values_internal(['data', 'phi_b'])
        elif nodes == 'staggered':
            phi = self.get_dict_values(['data', 'phi_s'])
        # define mixed phase by these threshold values
        MIX = (phi < 0.95) & (phi > 0.05)
        return MIX

    def get_melt_phase_boolean_array(self, nodes='basic'):
        """this array enables us to plot different linestyles for
        melt phase regions"""
        if nodes == 'basic':
            phi = self.get_dict_values(['data', 'phi_b'])
        elif nodes == 'basic_internal':
            phi = self.get_dict_values_internal(['data', 'phi_b'])
        elif nodes == 'staggered':
            phi = self.get_dict_values(['data', 'phi_s'])
        MELT = phi > 0.95
        return MELT

    def get_solid_phase_boolean_array(self, nodes='basic'):
        """this array enables us to plot different linestyles for
        solid phase regions"""
        if nodes == 'basic':
            phi = self.get_dict_values(['data', 'phi_b'])
        elif nodes == 'basic_internal':
            phi = self.get_dict_values_internal(['data', 'phi_b'])
        elif nodes == 'staggered':
            phi = self.get_dict_values(['data', 'phi_s'])
        SOLID = phi < 0.05
        return SOLID


def read_jsons(output_dir: str, times: list) -> list[MyJSON]:
    """
    Read JSON files from the output/data/ directory for the specified times.

    Parameters
    ----------
    output_dir : str
        Path to the output directory.
    times : list
        List of times (in years) for which to read the JSON files.

    Returns
    -------
    jsons : list[MyJSON]
        List of MyJSON objects containing the data from the JSON files.
    """
    jsons = []
    for t in times:
        _f = os.path.join(output_dir, 'data', '%.0f.json' % t)  # path to file
        _j = MyJSON(_f)  # load json file
        if _j.data_d is None:
            _j = None  # set to None if data could not be read
        else:
            jsons.append(_j)  # otherwise, append to list
    return jsons


def get_all_output_times(odir: str):
    """
    Get all times (in yr) from the json files located in the output directory
    """

    odir = odir + '/data/'

    # locate times to process based on files located in odir/
    file_l = [f for f in os.listdir(odir) if os.path.isfile(odir + f)]
    if not file_l:
        raise Exception('Output data directory contains no files')

    time_l = [fname for fname in file_l]
    time_l = list(filter(lambda a: a.endswith('json'), time_l))
    time_l = [int(time.split('.json')[0]) for time in time_l]

    # ascending order
    time_l = sorted(time_l, key=int)
    time_a = np.array(time_l)

    return time_a


def interp_rho_melt(S: float, P: float, lookup: str) -> float:
    """
    Return density of pure melt at given entropy and pressure.

    Parameters
    ---------------
    - entropy: float     entropy of layer [J kg-1 K-1]
    - pressure: float    pressure of layer [Pa]
    - lookup: str    directory of SPIDER installation

    Returns
    -----------------
    - density: float    density of pure melt [kg m-3]
    """

    Pvals, Svals, rho_grid = lookup[0, :, 0], lookup[:, 0, 1], lookup[:, :, 2]
    interp = RegularGridInterpolator(
        (Pvals, Svals), rho_grid.T, bounds_error=False, fill_value=None
    )

    rho = interp(np.column_stack((np.atleast_1d(P), np.atleast_1d(S))))
    return float(np.array(rho).item())


# ====================================================================
def _try_spider(
    dirs: dict,
    config: Config,
    IC_INTERIOR: int,
    hf_all: pd.DataFrame,
    hf_row: dict,
    step_sf: float,
    atol_sf: float,
    dT_max: float,
    timeout: float = 60 * 30,
    mesh_file: str | None = None,
):
    """
    Try to run spider with the current configuration.
    """

    # Check that SPIDER can be found
    spider_exec = os.path.join(dirs['spider'], 'spider')
    if not os.path.isfile(spider_exec):
        raise FileNotFoundError("SPIDER executable could not be found at '%s'" % spider_exec)

    # Scale factors for when SPIDER is failing to converge
    step_sf = max(1.0e-10, step_sf)
    atol_sf = max(1.0e-10, atol_sf)

    # Solver tolerances
    spider_atol = atol_sf * config.interior.tolerance
    spider_rtol = atol_sf * config.interior.tolerance_rel

    # Bounds on tolerances
    spider_rtol = min(spider_rtol, 1e-1)
    spider_atol = max(spider_atol, 1e-11)

    # Recalculate time stepping
    if IC_INTERIOR == 2:
        # Get step number from last JSON file
        json_path = os.path.join(dirs['output/data'], '%.0f.json' % hf_row['Time'])
        json_file = MyJSON(json_path)
        if json_file.data_d is None:
            UpdateStatusfile(dirs, 21)
            raise ValueError("JSON file '%s' could not be loaded" % json_path)
        step = json_file.get_dict(['step'])

        # Get new time-step
        dtswitch = next_step(config, dirs, hf_row, hf_all, step_sf)

        # Number of total steps until currently desired switch/end time
        nsteps = 1
        nstepsmacro = step + nsteps
        dtmacro = dtswitch

        log.debug(
            'SPIDER iteration: dt=%.2e yrs in %d steps (at i=%d)'
            % (dtmacro, nsteps, nstepsmacro)
        )

    # For init loop
    else:
        nstepsmacro = 1
        dtmacro = 0
        dtswitch = 0

    empty_file = os.path.join(dirs['output/data'], '.spider_tmp')
    open(empty_file, 'w').close()

    # Compute coresize: use external mesh radii when available, otherwise config
    coresize = config.struct.corefrac
    rho_core = config.struct.core_density
    if mesh_file and os.path.isfile(mesh_file):
        coresize = _coresize_from_mesh(mesh_file)
        log.debug(
            'coresize from external mesh: %.6f (config: %.6f)',
            coresize,
            config.struct.corefrac,
        )
        # Derive average core density from the self-consistent core mass
        # (set by Zalmoxis) and the CMB radius from the mesh file
        R_cmb = coresize * hf_row['R_int']
        M_core = hf_row.get('M_core', 0)
        if R_cmb > 0 and M_core > 0:
            rho_core = M_core / (4.0 / 3.0 * np.pi * R_cmb**3)
            log.debug(
                'rho_core from Zalmoxis structure: %.2f kg/m^3 (config: %.2f)',
                rho_core,
                config.struct.core_density,
            )

    # Determine SPIDER domain boundaries.
    # When global_miscibility is enabled, SPIDER evolves the miscible
    # interior (center to solvus), not the full mantle (CMB to surface).
    spider_radius = hf_row['R_int']
    spider_gravity = hf_row['gravity']
    spider_coresize = coresize
    if config.struct.global_miscibility and 'R_solvus' in hf_row:
        R_solvus = hf_row['R_solvus']
        if R_solvus is not None and R_solvus < hf_row['R_int']:
            spider_radius = R_solvus
            # Gravity at solvus: interpolate from structure if available,
            # otherwise scale by (R_solvus/R_int)^2 * M_solvus/M_int
            spider_gravity = hf_row['gravity'] * (R_solvus / hf_row['R_int']) ** 2
            # Coresize relative to solvus, not surface
            R_cmb_actual = coresize * hf_row['R_int']
            spider_coresize = R_cmb_actual / R_solvus if R_solvus > 0 else coresize
            log.info(
                'SPIDER domain: [%.2e, %.2e] m (solvus), '
                'coresize=%.4f, gravity=%.2f m/s^2',
                R_cmb_actual,
                R_solvus,
                spider_coresize,
                spider_gravity,
            )

    ### SPIDER base call sequence
    call_sequence = [
        spider_exec,
        '-options_file',
        empty_file,
        '-outputDirectory',
        dirs['output/data'],
        '-IC_INTERIOR',
        '%d' % (IC_INTERIOR),
        '-OXYGEN_FUGACITY_offset',
        '%.6e' % (config.outgas.fO2_shift_IW),  # Relative to the specified buffer
        '-surface_bc_value',
        '%.6e' % (hf_row['F_atm']),
        '-teqm',
        '%.6e' % (hf_row['T_eqm']),
        '-n',
        '%d' % (config.interior.num_levels),
        '-nstepsmacro',
        '%d' % (nstepsmacro),
        '-dtmacro',
        '%.6e' % (dtmacro),
        '-radius',
        '%.6e' % spider_radius,
        '-gravity',
        '%.6e' % (-1.0 * spider_gravity),
        '-coresize',
        '%.6e' % spider_coresize,
        '-grain',
        '%.6e' % (config.interior.grain_size),
    ]

    # Tolerance on the change in T_magma during a single SPIDER call
    if hf_row['Time'] > 0:
        dT_poststep = (
            config.interior.spider.tsurf_rtol * hf_row['T_magma']
            + config.interior.spider.tsurf_atol
        )
    else:
        dT_poststep = float(config.interior.spider.tsurf_atol)
    call_sequence.extend(['-tsurf_poststep_change', str(min(dT_max, dT_poststep))])

    # set surface and core entropy (-1 is a flag to ignore)
    call_sequence.extend(['-ic_surface_entropy', '-1'])
    call_sequence.extend(['-ic_core_entropy', '-1'])

    # Initial condition
    if IC_INTERIOR == 2:
        # get last JSON File
        last_filename = natural_sort(
            [os.path.basename(x) for x in glob.glob(dirs['output/data'] + '/*.json')]
        )[-1]
        last_filename = os.path.join(dirs['output/data'], last_filename)
        call_sequence.extend(
            [
                '-ic_interior_filename',
                str(last_filename),
                '-activate_poststep',
                '-activate_rollback',
            ]
        )
    else:
        # set to adiabat
        call_sequence.extend(
            [
                '-ic_adiabat_entropy',
                str(config.interior.spider.ini_entropy),
                '-ic_dsdr',
                str(config.interior.spider.ini_dsdr),  # initial dS/dr everywhere
            ]
        )

    # Mixing length parameterization: 1: variable | 2: constant
    call_sequence.extend(['-mixing_length', str(config.interior.mixing_length)])

    # Solver tolerances
    call_sequence.extend(['-ts_sundials_atol', str(spider_atol)])
    call_sequence.extend(['-ts_sundials_rtol', str(spider_rtol)])
    call_sequence.extend(['-ts_sundials_type', str(config.interior.spider.solver_type)])

    # Rollback
    call_sequence.extend(['-activate_poststep', '-activate_rollback'])

    # Dimensional scalings
    call_sequence.extend(['-radius0', '63710000.0'])
    call_sequence.extend(['-entropy0', '2993.025100070677'])
    call_sequence.extend(['-time0', '1.0E5'])
    call_sequence.extend(['-pressure0', '10.0E5'])

    # Energy transport physics (true->'1', false->'0')
    call_sequence.extend(['-CONDUCTION', str(int(config.interior.conduction))])
    call_sequence.extend(['-CONVECTION', str(int(config.interior.convection))])
    call_sequence.extend(['-MIXING', str(int(config.interior.mixing))])
    call_sequence.extend(
        ['-SEPARATION', str(int(config.interior.grav_sep))]
    )

    # Tidal heating
    if config.interior.tidal_heat:
        call_sequence.extend(['-HTIDAL', '2'])
        call_sequence.extend(['-htidal_filename', get_file_tides(dirs['output'])])

    # EOS lookup data: prefer per-run generated tables (from Zalmoxis/PALEOS),
    # then FWL_DATA, then SPIDER local as final fallback.
    if dirs.get('spider_eos_dir') and os.path.isdir(dirs['spider_eos_dir']):
        eos_dir = dirs['spider_eos_dir']
        log.info('Using Zalmoxis-generated SPIDER EOS tables from %s', eos_dir)
    else:
        eos_dir = os.path.join(EOS_DYNAMIC_DIR, config.interior.eos_dir, 'P-S')
        if not os.path.isdir(eos_dir):
            # Fall back to SPIDER's local lookup_data (uses legacy directory name)
            eos_dir = os.path.join(dirs['spider'], 'lookup_data', '1TPa-dK09-elec-free')
        if not os.path.isdir(eos_dir):
            raise FileNotFoundError(
                f'SPIDER EOS directory not found: {eos_dir}. '
                f"Check interior.eos_dir='{config.interior.eos_dir}'."
            )

    # Resolve melting curve S(P) files: prefer generated paths, then FWL_DATA
    if dirs.get('spider_liquidus_ps') and os.path.isfile(dirs['spider_liquidus_ps']):
        liquidus_ps = dirs['spider_liquidus_ps']
        solidus_ps = dirs['spider_solidus_ps']
        log.info('Using Zalmoxis-generated phase boundaries')
    else:
        mc_dir = os.path.join(MELTING_CURVES_DIR, config.interior.melting_dir)
        liquidus_ps = os.path.join(mc_dir, 'liquidus_P-S.dat')
        solidus_ps = os.path.join(mc_dir, 'solidus_P-S.dat')
        for fpath in (liquidus_ps, solidus_ps):
            if not os.path.isfile(fpath):
                raise FileNotFoundError(
                    f'SPIDER phase boundary file not found: {fpath}. '
                    f"Run 'python tools/generate_spider_phase_boundaries.py "
                    f"--melting-dir {config.interior.melting_dir}' to generate it."
                )

    call_sequence.extend(['-phase_names', 'melt,solid'])

    call_sequence.extend(['-melt_TYPE', '1'])
    call_sequence.extend(
        ['-melt_alpha_filename', os.path.join(eos_dir, 'thermal_exp_melt.dat')]
    )
    call_sequence.extend(['-melt_cp_filename', os.path.join(eos_dir, 'heat_capacity_melt.dat')])
    call_sequence.extend(
        ['-melt_dTdPs_filename', os.path.join(eos_dir, 'adiabat_temp_grad_melt.dat')]
    )
    call_sequence.extend(['-melt_rho_filename', os.path.join(eos_dir, 'density_melt.dat')])
    call_sequence.extend(['-melt_temp_filename', os.path.join(eos_dir, 'temperature_melt.dat')])
    call_sequence.extend(['-melt_phase_boundary_filename', liquidus_ps])
    call_sequence.extend(['-melt_log10visc', '2.0'])
    call_sequence.extend(['-melt_cond', '4.0'])  # conductivity of melt

    call_sequence.extend(['-solid_TYPE', '1'])
    call_sequence.extend(
        ['-solid_alpha_filename', os.path.join(eos_dir, 'thermal_exp_solid.dat')]
    )
    call_sequence.extend(
        ['-solid_cp_filename', os.path.join(eos_dir, 'heat_capacity_solid.dat')]
    )
    call_sequence.extend(
        ['-solid_dTdPs_filename', os.path.join(eos_dir, 'adiabat_temp_grad_solid.dat')]
    )
    call_sequence.extend(['-solid_rho_filename', os.path.join(eos_dir, 'density_solid.dat')])
    call_sequence.extend(
        ['-solid_temp_filename', os.path.join(eos_dir, 'temperature_solid.dat')]
    )
    call_sequence.extend(['-solid_phase_boundary_filename', solidus_ps])
    call_sequence.extend(['-solid_log10visc', '22.0'])
    call_sequence.extend(['-solid_cond', '4.0'])  # conductivity of solid

    # Static pressure profile: external mesh from Zalmoxis, or Adams-Williamson
    if mesh_file and os.path.isfile(mesh_file):
        call_sequence.extend(['-MESH_SOURCE', '1'])
        call_sequence.extend(['-mesh_external_filename', mesh_file])

        # Check EOS table range against CMB pressure (first run only)
        if IC_INTERIOR != 2:
            try:
                with open(mesh_file) as mf:
                    header = mf.readline()
                    nb = int(header.strip('# \n').split()[0])
                    # Skip to last basic node (CMB)
                    for _ in range(nb - 1):
                        line = mf.readline()
                    P_cmb = float(line.split()[1])
                _check_eos_table_range(eos_dir, mesh_file, P_cmb)
            except (ValueError, IndexError, IOError):
                pass  # non-critical check
    else:
        # Adams-Williamson EOS parameters from fitting PREM lower mantle (Earth)
        call_sequence.extend(['-adams_williamson_rhos', '4078.95095544'])
        call_sequence.extend(['-adams_williamson_beta', '1.1115348931000002e-07'])

    # eddy diffusivity
    # if negative, this value is adopted (units m^2/s)
    # if positive, this value is used to scale the internally calculated eddy diffusivity
    call_sequence.extend(['-eddy_diffusivity_thermal', '1.0'])
    call_sequence.extend(['-eddy_diffusivity_chemical', '1.0'])

    # Phase-dependent eddy diffusivity floor
    if config.interior.kappah_floor > 0:
        call_sequence.extend(['-kappah_floor', str(config.interior.kappah_floor)])

    # smoothing of material properties across liquidus and solidus
    # units of melt fraction (non-dimensional)
    call_sequence.extend(
        ['-matprop_smooth_width', '%.6e' % (config.interior.spider.matprop_smooth_width)]
    )

    # Viscosity behaviour (rheological transition location and width, melt fractions)
    call_sequence.extend(['-phi_critical', '%.6e' % (config.interior.rheo_phi_loc)])
    call_sequence.extend(['-phi_width', '%.6e' % (config.interior.rheo_phi_wid)])

    # Relating to the planet's metallic core
    call_sequence.extend(['-CORE_BC', '1'])  # CMB boundary condition
    call_sequence.extend(['-rho_core', '%.6e' % rho_core])  # density
    call_sequence.extend(['-cp_core', '%.6e' % (config.struct.core_heatcap)])  # heat capacity

    # surface boundary condition
    # [4] heat flux (prescribe value using surface_bc_value)
    call_sequence.extend(['-SURFACE_BC', '4'])

    # Note: SPIDER's upper thermal boundary layer treatment is implicit in its
    # entropy formulation. Unlike Aragog's tunable param_utbl/param_utbl_const,
    # SPIDER does not expose UTBL as a configurable parameter. The boundary
    # layer physics is built into the surface entropy BC (bc.c).
    # parameterise the upper thermal boundary layer
    call_sequence.extend(['-PARAM_UTBL', '0'])  # disabled
    call_sequence.extend(['-param_utbl_const', '1.0E-7'])  # value of parameterisation

    # fO2 buffer chosen to define fO2 (7: Iron-Wustite)
    call_sequence.extend(['-OXYGEN_FUGACITY', '2'])

    # radionuclides
    if config.interior.radiogenic_heat:
        # offset by age_ini, which converts model simulation time to the actual age
        radio_t0 = config.delivery.radio_tref - config.star.age_ini
        radio_t0 *= 1e9  # Convert Gyr to yr
        radnuc_names = []

        def _append_radnuc(_iso, _cnc):
            radnuc_names.append(_iso)
            call_sequence.extend([f'-{_iso}_t0', '%.5e' % radio_t0])
            call_sequence.extend([f'-{_iso}_concentration', '%.5f' % _cnc])
            call_sequence.extend(
                [f'-{_iso}_abundance', '%.5e' % radnuc_data[_iso]['abundance']]
            )
            call_sequence.extend(
                [f'-{_iso}_heat_production', '%.5e' % radnuc_data[_iso]['heatprod']]
            )
            call_sequence.extend([f'-{_iso}_half_life', '%.5e' % radnuc_data[_iso]['halflife']])

        if config.delivery.radio_K > 0.0:
            _append_radnuc('k40', config.delivery.radio_K)

        if config.delivery.radio_Th > 0.0:
            _append_radnuc('th232', config.delivery.radio_Th)

        if config.delivery.radio_U > 0.0:
            _append_radnuc('u235', config.delivery.radio_U)
            _append_radnuc('u238', config.delivery.radio_U)

        call_sequence.extend(['-radionuclide_names', ','.join(radnuc_names)])

    # Runtime info
    flags = ''
    for flag in call_sequence:
        flags += ' ' + flag
    # log.debug("SPIDER call sequence: '%s'" % flags)

    call_string = ' '.join(call_sequence)

    # Environment
    spider_env = os.environ.copy()
    if platform.system() == 'Darwin':
        spider_env['PETSC_ARCH'] = 'arch-darwin-c-opt'
    else:
        spider_env['PETSC_ARCH'] = 'arch-linux-c-opt'
    spider_env['PETSC_DIR'] = os.path.join(dirs['proteus'], 'petsc')

    # Run SPIDER
    log.debug('SPIDER output suppressed')
    spider_print = open(dirs['output'] + 'spider_recent.log', 'w')
    spider_print.write(call_string + '\n')
    spider_print.flush()
    spider_succ = True
    try:
        proc = sp.run(
            call_sequence,
            timeout=timeout,
            text=True,
            stdout=spider_print,
            stderr=spider_print,
            env=spider_env,
        )
    except sp.TimeoutExpired:
        log.error('SPIDER process timed-out')
        spider_succ = False
    except Exception as e:
        log.error('SPIDER encountered an error: ' + str(type(e)))
        spider_succ = False
    else:
        spider_succ = bool(proc.returncode == 0)

    spider_print.close()
    return spider_succ


def RunSPIDER(
    dirs: dict,
    config: Config,
    hf_all: pd.DataFrame,
    hf_row: dict,
    interior_o: Interior_t,
    mesh_file: str | None = None,
):
    """
    Wrapper function for running SPIDER.
    This wrapper handles cases where SPIDER fails to find a solution.
    """

    # parameters
    max_attempts = 5  # maximum number of attempts
    step_sf = 1.0  # step scale factor at attempt 1
    atol_sf = 1.0  # tolerance scale factor at attempt 1

    # tracking
    spider_success = False  # success?
    attempts = 0  # number of attempts so far

    # Maximum dT
    dT_max = 1e99
    if config.interior.tidal_heat and (np.amax(interior_o.tides) > 1e-10):
        dT_max = 4.0
        log.info('Tidal heating active; limiting dT_magma to %.2f K' % dT_max)

    # make attempts
    while not spider_success:
        attempts += 1
        log.debug('Attempt %d' % attempts)

        # run SPIDER
        spider_success = _try_spider(
            dirs,
            config,
            interior_o.ic,
            hf_all,
            hf_row,
            step_sf,
            atol_sf,
            dT_max,
            mesh_file=mesh_file,
        )

        if spider_success:
            # success
            log.debug('Attempt %d succeeded' % attempts)
        else:
            # failure
            log.warning('Attempt %d failed' % attempts)
            if attempts >= max_attempts:
                # give up
                log.error('Giving up')
                break
            else:
                # try again (change tolerance and step size)
                log.warning('Trying again')
                step_sf *= 0.1
                atol_sf *= 10.0

    # check status
    if spider_success:
        # success after some attempts
        return True
    else:
        # failure of all attempts
        UpdateStatusfile(dirs, 21)
        raise RuntimeError(
            'An error occurred when executing SPIDER (made %d attempts)' % attempts
        )


def ReadSPIDER(dirs: dict, config: Config, R_int: float, interior_o: Interior_t):
    """
    Read variables from last SPIDER output JSON file into a dictionary
    """

    # Store variables in this dict
    output = {}

    ### Read in last SPIDER base parameters
    sim_time = get_all_output_times(dirs['output'])[-1]  # yr, as an integer value

    # load data file
    json_path = os.path.join(dirs['output/data'], '%.0f.json' % sim_time)
    json_file = MyJSON(json_path)
    if json_file.data_d is None:
        UpdateStatusfile(dirs, 21)
        raise ValueError("JSON file '%s' could not be loaded" % json_path)

    # read scalars
    json_keys = {
        'M_mantle_liquid': ('atmosphere', 'mass_liquid'),
        'M_mantle_solid': ('atmosphere', 'mass_solid'),
        'M_mantle': ('atmosphere', 'mass_mantle'),
        'M_core': ('atmosphere', 'mass_core'),
        'T_magma': ('atmosphere', 'temperature_surface'),
        'Phi_global': ('rheological_front_phi', 'phi_global'),
        'F_int': ('atmosphere', 'Fatm'),
        'RF_depth': ('rheological_front_dynamic', 'depth'),
    }

    # Fill the new dict with scalars, and scale values as required
    for key in json_keys:
        output[key] = float(json_file.get_dict_values(json_keys[key]))
    output['RF_depth'] /= R_int

    # read arrays
    area_b = json_file.get_dict_values(['data', 'area_b'])
    Hradio_s = json_file.get_dict_values(['data', 'Hradio_s'])
    Htidal_s = json_file.get_dict_values(['data', 'Htidal_s'])
    mass_s = json_file.get_dict_values(['data', 'mass_s'])

    # Tidal heating
    output['F_tidal'] = np.dot(Htidal_s, mass_s) / area_b[0]

    # Radiogenic heating
    output['F_radio'] = np.dot(Hradio_s, mass_s) / area_b[0]

    # Arrays at current time
    interior_o.phi = np.array(json_file.get_dict_values(['data', 'phi_s']))
    interior_o.density = np.array(json_file.get_dict_values(['data', 'rho_s']))
    interior_o.radius = np.array(json_file.get_dict_values(['data', 'radius_b']))
    interior_o.visc = np.array(json_file.get_dict_values(['data', 'visc_b']))[1:]
    interior_o.mass = np.array(json_file.get_dict_values(['data', 'mass_s']))
    interior_o.temp = np.array(json_file.get_dict_values(['data', 'temp_s']))
    interior_o.pres = np.array(json_file.get_dict_values(['data', 'pressure_s']))

    vshell = json_file.get_dict_values(['data', 'volume_s'])
    mshell = interior_o.density * vshell

    # Entropy at each layer [J kg-1 K]
    entropy = np.array(json_file.get_dict_values(['data', 'S_s']))

    # Get density of pure-melt at each layer
    rho_melt_arr = np.ones_like(interior_o.pres)
    for i in range(len(interior_o.pres)):
        rho_melt_arr[i] = interp_rho_melt(
            entropy[i], interior_o.pres[i], interior_o.lookup_rho_melt
        )

    # Determine volume of melt at each layer
    vmelt = interior_o.phi * mshell / rho_melt_arr

    # Total volume of each layer
    volume_mantle = np.sum(vshell)

    # Global melt fraction by volume
    output['Phi_global_vol'] = min(1.0, max(0.0, np.sum(vmelt) / volume_mantle))

    # Manually calculate heat flux at near-surface from energy gradient
    # Etot        = json_file.get_dict_values(['data','Etot_b'])
    # rad         = json_file.get_dict_values(['data','radius_b'])
    # area        = json_file.get_dict_values(['data','area_b'])
    # E0          = Etot[1] - (Etot[2]-Etot[1]) * (rad[2]-rad[1]) / (rad[1]-rad[0])
    # F_int2      = E0/area[0]

    # Get estimate of potential temperature
    Fconv = json_file.get_dict_values(['data', 'Jconv_b'])
    Fcond = json_file.get_dict_values(['data', 'Jcond_b'])
    for i in range(len(Fconv)):
        if Fconv[i] > Fcond[i]:
            break
    i = min(i, len(interior_o.temp) - 1)
    output['T_pot'] = float(interior_o.temp[i])

    # Core (CMB) temperature: last staggered node (SPIDER ordering is surface-to-CMB)
    output['T_core'] = float(interior_o.temp[-1])

    # Total thermal energy: E_th = sum(rho_i * Cp_i * T_i * V_i)
    # SPIDER has Cp in its EOS tables. Use the lookup if available,
    # else fall back to Cp = T * dS/dT estimated from entropy tables.
    try:
        cp_est = np.full_like(interior_o.temp, 1200.0)
        # Try to get Cp from the SPIDER EOS lookup (P-S tables)
        if hasattr(interior_o, 'lookup_cp') and interior_o.lookup_cp is not None:
            for i in range(len(interior_o.temp)):
                try:
                    cp_est[i] = float(interior_o.lookup_cp(
                        entropy[i], interior_o.pres[i]
                    ))
                except Exception:
                    pass
        elif hasattr(interior_o, 'lookup_cp_melt') and interior_o.lookup_cp_melt is not None:
            for i in range(len(interior_o.temp)):
                try:
                    cp_est[i] = float(interior_o.lookup_cp_melt(
                        entropy[i], interior_o.pres[i]
                    ))
                except Exception:
                    pass
        E_th = float(np.sum(interior_o.density * cp_est * interior_o.temp * vshell))
        output['E_th_mantle'] = E_th
        Cp_eff = float(np.sum(interior_o.density * cp_est * vshell)) / max(
            float(np.sum(mshell)), 1.0
        )
        output['Cp_eff'] = Cp_eff
    except Exception:
        output['E_th_mantle'] = 0.0
        output['Cp_eff'] = 1200.0

    # Limit F_int to positive values
    if config.atmos_clim.prevent_warming:
        output['F_int'] = max(1.0e-8, output['F_int'])

    # Check NaNs
    if np.isnan(output['T_magma']):
        raise Exception('Magma ocean temperature is NaN')

    return sim_time, output
