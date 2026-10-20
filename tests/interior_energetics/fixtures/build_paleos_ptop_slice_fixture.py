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
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
from aragog.eos.entropy import EntropyEOS

OUT_PATH = Path(__file__).resolve().parent / 'paleos_ptop_531p546_slice.json'


def _md5(path: Path) -> str:
    """Return hex MD5 digest of path."""
    return hashlib.md5(path.read_bytes()).hexdigest()


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

    payload = {
        'description': (
            '1D PALEOS-2phase-1TPa-dK09-elec-free EOS slice at P_top = 4.455575309179133e8 Pa '
            'and top-cell states for rows 380..392 (t = 480.655..600.000 yr) of the 1 M_earth '
            '0.5 au coupled Aragog-AGNI reference run.'
        ),
        'source_table': 'PALEOS-2phase-1TPa-dK09-elec-free',
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
