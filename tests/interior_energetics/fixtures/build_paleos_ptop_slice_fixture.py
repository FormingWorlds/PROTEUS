#!/usr/bin/env python3
"""Build the 1D PALEOS EOS slice fixture at P_top for unit testing _estimate_tau_drain.

Reads the PALEOS two-phase SPIDER-format tables and the t = 531.546 yr NetCDF/helpfile
states from a coupled 1 M_earth reference run, evaluates each phase table's 1D slice at
P_top, records the MD5 digest of every source file, and verifies that the 1D slice
reproduces Aragog's 2D EntropyEOS to machine precision across [S_min, S_max].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tomllib
from pathlib import Path

import aragog
import netCDF4 as nc
import numpy as np
import pandas as pd
from aragog.eos.entropy import EntropyEOS

OUT_PATH = Path(__file__).resolve().parent / 'paleos_ptop_531p546_slice.json'


def _md5(path: Path) -> str:
    """Return hex MD5 digest of path."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _eval_1d_slice(
    phase_slices: dict[str, dict[str, list[float]]],
    s_sol: float,
    s_liq: float,
    s_query: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate 1D slice properties at fixed P_top matching _PaleosSliceEOS."""
    s = np.asarray(s_query, dtype=float)
    phi = np.clip((s - s_sol) / (s_liq - s_sol), 0.0, 1.0)
    mushy = (phi > 0.0) & (phi < 1.0)
    solid_s = np.asarray(phase_slices['solid']['S'], dtype=float)
    melt_s = np.asarray(phase_slices['melt']['S'], dtype=float)
    s_sol_q = np.where(mushy, s_sol, s)
    s_liq_q = np.where(mushy, s_liq, s)

    out: dict[str, np.ndarray] = {'melt_fraction': phi}
    for prop in ('temperature', 'heat_capacity'):
        v_sol = np.interp(
            np.clip(s_sol_q, solid_s[0], solid_s[-1]),
            solid_s,
            np.asarray(phase_slices['solid'][prop], dtype=float),
        )
        v_melt = np.interp(
            np.clip(s_liq_q, melt_s[0], melt_s[-1]),
            melt_s,
            np.asarray(phase_slices['melt'][prop], dtype=float),
        )
        out[prop] = np.where(phi > 0.0, phi * v_melt, 0.0) + np.where(
            phi < 1.0, (1.0 - phi) * v_sol, 0.0
        )

    rho_sol_arr = np.asarray(phase_slices['solid']['density'], dtype=float)
    rho_melt_arr = np.asarray(phase_slices['melt']['density'], dtype=float)
    rho_sol_b = np.interp(np.clip(s_sol, solid_s[0], solid_s[-1]), solid_s, rho_sol_arr)
    rho_liq_b = np.interp(np.clip(s_liq, melt_s[0], melt_s[-1]), melt_s, rho_melt_arr)
    rho_mushy = 1.0 / (phi / rho_liq_b + (1.0 - phi) / rho_sol_b)
    v_sol_rho = np.interp(np.clip(s, solid_s[0], solid_s[-1]), solid_s, rho_sol_arr)
    v_melt_rho = np.interp(np.clip(s, melt_s[0], melt_s[-1]), melt_s, rho_melt_arr)
    out['density'] = np.where(mushy, rho_mushy, np.where(phi >= 0.5, v_melt_rho, v_sol_rho))
    return out


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
    source_table = str(run_cfg['interior_struct']['eos_dir'])
    interior_mod = str(run_cfg['interior_energetics']['module'])
    atmos_mod = str(run_cfg['atmos_clim']['module'])

    aragog_repo = Path(aragog.__file__).resolve().parent
    aragog_commit = subprocess.check_output(
        ['git', '-C', str(aragog_repo), 'rev-parse', 'HEAD'], text=True
    ).strip()
    assert len(aragog_commit) == 40

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

    # Verify that the 1D slice reproduces Aragog's 2D EntropyEOS across [S_min, S_max]
    s_check = np.linspace(float(eos.S_min), float(eos.S_max), 4000)
    p_check = np.full_like(s_check, p_top_pa, dtype=float)
    slice_eval = _eval_1d_slice(phase_slices, s_sol, s_liq, s_check)
    max_rel_diff = 0.0
    for prop, eos_fn in (
        ('temperature', eos.temperature),
        ('heat_capacity', eos.heat_capacity),
        ('density', eos.density),
        ('melt_fraction', eos.melt_fraction),
    ):
        ref_2d = np.asarray(eos_fn(p_check, s_check), dtype=float).ravel()
        val_1d = np.asarray(slice_eval[prop], dtype=float).ravel()
        denom = np.maximum(np.abs(ref_2d), 1.0e-30)
        rel_diff = float(np.max(np.abs(val_1d - ref_2d) / denom))
        max_rel_diff = max(max_rel_diff, rel_diff)
    assert max_rel_diff < 1.0e-14, f'1D slice diverged from 2D EntropyEOS: {max_rel_diff}'

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
        assert best_p is not None and best_d < 2.0e-3
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
        'aragog_commit': aragog_commit,
        'max_2d_vs_slice_rel_diff': max_rel_diff,
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
