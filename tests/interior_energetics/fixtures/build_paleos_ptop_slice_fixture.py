#!/usr/bin/env python3
"""Build the 1D PALEOS EOS slice fixture at P_top for unit testing _estimate_tau_drain.

Reads the PALEOS two-phase SPIDER-format tables and the t = 531.546 yr NetCDF/helpfile
states from a coupled 1 M_earth reference run, evaluates each phase table's 1D slice at
P_top, records the MD5 digest of every source file, and verifies that _PaleosSliceEOS
reproduces Aragog's 2D EntropyEOS to machine precision across [S_min, S_max].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import aragog.eos.entropy
import netCDF4 as nc
import numpy as np
import pandas as pd
from aragog.eos.entropy import EntropyEOS

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.interior_energetics.test_timestep import _PaleosSliceEOS  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent / 'paleos_ptop_531p546_slice.json'
_GIT_DESCRIBE_RE = re.compile(r'[A-Za-z0-9._-]+-[0-9]+-g[0-9a-f]{40}')


def _md5(path: Path) -> str:
    """Return hex MD5 digest of path."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _resolve_source_table(cfg_path: Path) -> str:
    """Resolve and validate the PALEOS EOS source table name from init_coupler.toml."""
    with cfg_path.open('rb') as f_cfg:
        run_cfg = tomllib.load(f_cfg)
    struct_cfg = run_cfg.get('interior_struct', {})
    raw_eos_dir = str(struct_cfg.get('eos_dir', 'none'))
    if raw_eos_dir.lower() != 'none':
        source_table = raw_eos_dir
    else:
        struct_mod = str(struct_cfg.get('module', 'none'))
        if struct_mod != 'zalmoxis':
            raise RuntimeError(
                f'Cannot resolve zalmoxis.mantle_eos when interior_struct.module={struct_mod!r} '
                f'(expected "zalmoxis") in {cfg_path}'
            )
        source_table = str(struct_cfg.get('zalmoxis', {}).get('mantle_eos', 'none'))
    if (
        not source_table
        or source_table.lower() == 'none'
        or not source_table.startswith('PALEOS')
    ):
        raise RuntimeError(
            f'Invalid EOS source_table={source_table!r} resolved from {cfg_path}'
        )
    return source_table


def _resolve_aragog_commit(
    aragog_file: str | Path | None = None,
    run_cmd=subprocess.run,
) -> str:
    """Return git describe string for the aragog checkout used for the 2D EOS evaluation check."""
    if aragog_file is None:
        aragog_file = getattr(aragog.eos.entropy, '__file__', None)
    if not aragog_file:
        raise RuntimeError('Cannot locate aragog.eos.entropy.__file__ to resolve git commit.')
    aragog_repo = Path(aragog_file).resolve().parent
    try:
        top_proc = run_cmd(
            ['git', '-C', str(aragog_repo), 'rev-parse', '--show-toplevel'],
            check=True,
            capture_output=True,
            text=True,
        )
        top_dir = Path(top_proc.stdout.strip()).resolve()
        if 'aragog' not in top_dir.name.lower():
            raise RuntimeError(
                f'Expected git toplevel for aragog checkout, got {top_dir} from {aragog_repo}'
            )
        proc = run_cmd(
            [
                'git',
                '-C',
                str(aragog_repo),
                'describe',
                '--long',
                '--always',
                '--dirty',
                '--abbrev=40',
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        raise RuntimeError(
            f'Failed to resolve aragog git commit at {aragog_repo}; '
            f'run from a git checkout of aragog: {exc}'
        ) from exc
    commit_desc = proc.stdout.strip()
    if _GIT_DESCRIBE_RE.fullmatch(commit_desc) is None:
        raise RuntimeError(
            f'Expected clean aragog git describe string matching <tag>-<N>-g<40hex>, got {commit_desc!r}'
        )
    return commit_desc


def main() -> None:
    """Build and verify paleos_ptop_531p546_slice.json."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--run-dir',
        type=Path,
        required=True,
        help='Path to the coupled reference run directory containing runtime_helpfile.csv and data/.',
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    eos_dir = (run_dir / 'data' / 'spider_eos').resolve()
    nc_531_path = run_dir / 'data' / '531p546_int.nc'
    hf_path = run_dir / 'runtime_helpfile.csv'
    cfg_path = run_dir / 'init_coupler.toml'

    with cfg_path.open('rb') as f_cfg:
        run_cfg = tomllib.load(f_cfg)
    mass_earth = float(run_cfg['planet']['mass_tot'])
    semimajor_au = float(run_cfg['orbit']['semimajoraxis'])
    source_table = _resolve_source_table(cfg_path)
    interior_mod = str(run_cfg['interior_energetics']['module'])
    atmos_mod = str(run_cfg['atmos_clim']['module'])
    # Aragog checkout used for the 2D EntropyEOS evaluation check, not of the run.
    aragog_commit_eos_eval = _resolve_aragog_commit()

    eos = EntropyEOS(eos_dir)
    hf_df = pd.read_csv(hf_path, sep=r'\s+')

    with nc.Dataset(nc_531_path) as ds:
        p_top_pa = float(ds.variables['pres_s'][-1]) * 1.0e9
        r_b_m = np.asarray(ds.variables['radius_b'][:], dtype=float) * 1.0e3
        r_outer_m = float(r_b_m[-1])
        r_inner_top_m = float(r_b_m[-2])
        vol_top_m3 = (4.0 / 3.0) * math.pi * abs(r_outer_m**3 - r_inner_top_m**3)

    s_sol = float(np.asarray(eos.solidus_entropy(p_top_pa)).ravel()[0])
    s_liq = float(np.asarray(eos.liquidus_entropy(p_top_pa)).ravel()[0])

    phase_slices: dict[str, dict[str, list[float]]] = {}
    for phase in ('solid', 'melt'):
        s_grid = np.asarray(eos._tables[f'temperature_{phase}']['S'], dtype=float)
        p_vec = np.full_like(s_grid, p_top_pa, dtype=float)
        pts = np.column_stack([p_vec, s_grid])
        phase_slices[phase] = {
            'S': [float(x) for x in s_grid],
            'temperature': [
                float(x) for x in eos._tables[f'temperature_{phase}']['interp'](pts).ravel()
            ],
            'density': [
                float(x) for x in eos._tables[f'density_{phase}']['interp'](pts).ravel()
            ],
            'heat_capacity': [
                float(x) for x in eos._tables[f'heat_capacity_{phase}']['interp'](pts).ravel()
            ],
        }

    # Verify that _PaleosSliceEOS reproduces Aragog's 2D EntropyEOS across [S_min, S_max]
    slice_eos = _PaleosSliceEOS(
        {
            'S_min': float(eos.S_min),
            'S_max': float(eos.S_max),
            'P_top_Pa': p_top_pa,
            'S_sol_at_P_top': s_sol,
            'S_liq_at_P_top': s_liq,
            'phase_slices': phase_slices,
        }
    )
    s_check = np.linspace(float(eos.S_min), float(eos.S_max), 4000)
    p_check = np.full_like(s_check, p_top_pa, dtype=float)
    max_rel_diff = 0.0
    for eos_fn, slice_fn in (
        (eos.temperature, slice_eos.temperature),
        (eos.heat_capacity, slice_eos.heat_capacity),
        (eos.density, slice_eos.density),
        (eos.melt_fraction, slice_eos.melt_fraction),
    ):
        ref_2d = np.asarray(eos_fn(p_check, s_check), dtype=float).ravel()
        val_1d = np.asarray(slice_fn(p_check, s_check), dtype=float).ravel()
        denom = np.maximum(np.abs(ref_2d), 1.0e-30)
        rel_diff = float(np.max(np.abs(val_1d - ref_2d) / denom))
        max_rel_diff = max(max_rel_diff, rel_diff)
    if max_rel_diff >= 1.0e-14:
        raise RuntimeError(
            f'_PaleosSliceEOS diverged from 2D EntropyEOS: max_rel_diff={max_rel_diff}'
        )

    # Record 29 reference points across solid, solid plateau, solidus, mush, liquidus, and liquid
    s_ref_pts = np.unique(
        np.array(
            [
                float(eos.S_min),
                406.12,
                900.0,
                1400.0,
                1800.0,
                2200.0,
                2400.0,
                2445.0,
                2475.0,
                s_sol - 5.0,
                s_sol,
                s_sol + 5.0,
                s_sol + 0.15 * (s_liq - s_sol),
                s_sol + 0.35 * (s_liq - s_sol),
                0.5 * (s_sol + s_liq),
                3105.0,
                s_sol + 0.75 * (s_liq - s_sol),
                s_sol + 0.90 * (s_liq - s_sol),
                s_liq - 5.0,
                s_liq,
                s_liq + 5.0,
                3717.0,
                4500.0,
                6000.0,
                8000.0,
                10000.0,
                11283.4,
                16000.0,
                float(eos.S_max),
            ],
            dtype=float,
        )
    )
    p_ref_pts = np.full_like(s_ref_pts, p_top_pa, dtype=float)
    reference_2d_points = {
        'S_J_kg_K': [float(x) for x in s_ref_pts],
        'temperature_K': [
            float(x) for x in np.asarray(eos.temperature(p_ref_pts, s_ref_pts)).ravel()
        ],
        'heat_capacity_J_kg_K': [
            float(x) for x in np.asarray(eos.heat_capacity(p_ref_pts, s_ref_pts)).ravel()
        ],
        'density_kg_m3': [
            float(x) for x in np.asarray(eos.density(p_ref_pts, s_ref_pts)).ravel()
        ],
        'melt_fraction': [
            float(x) for x in np.asarray(eos.melt_fraction(p_ref_pts, s_ref_pts)).ravel()
        ],
    }

    snap_files = sorted((run_dir / 'data').glob('*_int.nc'))

    def _find_nc(t_yr: float) -> Path:
        best_p = None
        best_d = float('inf')
        for p in snap_files:
            stem = p.name[: -len('_int.nc')].replace('p', '.')
            d = abs(float(stem) - t_yr)
            if d < best_d:
                best_d = d
                best_p = p
        if best_p is None or best_d >= 2.0e-3:
            raise RuntimeError(f'No snapshot matching t={t_yr} yr within 2e-3 yr in {run_dir}')
        return best_p

    rows_window = []
    for row_idx in range(380, 393):
        t_yr = float(hf_df.loc[row_idx, 'Time'])
        t_surf = float(hf_df.loc[row_idx, 'T_surf'])
        f_atm = float(hf_df.loc[row_idx, 'F_atm'])
        nc_p = _find_nc(t_yr)
        with nc.Dataset(nc_p) as ds_row:
            s_top = float(ds_row.variables['entropy_s'][-1])
            p_top_row = float(ds_row.variables['pres_s'][-1]) * 1.0e9
        rows_window.append(
            {
                'row': int(row_idx),
                'nc_file': nc_p.name,
                'nc_md5': _md5(nc_p),
                'Time_yr': t_yr,
                'T_surf_K': t_surf,
                'F_atm_W_m2': f_atm,
                'P_top_Pa': p_top_row,
                'S_top_J_kg_K': s_top,
            }
        )

    source_files_md5 = {p.name: _md5(p) for p in sorted(eos_dir.glob('*.dat'))}
    source_files_md5['531p546_int.nc'] = _md5(nc_531_path)
    source_files_md5['runtime_helpfile.csv'] = _md5(hf_path)
    source_files_md5['init_coupler.toml'] = _md5(cfg_path)

    r_start = rows_window[0]['row']
    r_end = rows_window[-1]['row']
    t_start = rows_window[0]['Time_yr']
    t_end = rows_window[-1]['Time_yr']
    description = (
        f'1D {source_table} EOS slice at P_top = {p_top_pa:.15e} Pa '
        f'and top-cell states for rows {r_start}..{r_end} '
        f'(t = {t_start:.3f}..{t_end:.3f} yr) of the {mass_earth:g} M_earth '
        f'{semimajor_au:g} au coupled {interior_mod}-{atmos_mod} reference run.'
    )

    payload = {
        'description': description,
        'source_table': source_table,
        'aragog_commit_eos_eval': aragog_commit_eos_eval,
        'max_2d_vs_slice_rel_diff': max_rel_diff,
        'reference_2d_points': reference_2d_points,
        'source_files_md5': source_files_md5,
        'P_top_Pa': p_top_pa,
        'S_min': float(eos.S_min),
        'S_max': float(eos.S_max),
        'S_sol_at_P_top': s_sol,
        'S_liq_at_P_top': s_liq,
        'r_outer_m': r_outer_m,
        'r_inner_top_m': r_inner_top_m,
        'vol_top_m3': vol_top_m3,
        'phase_slices': phase_slices,
        'snap531_rows_380_to_392': rows_window,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
