#!/usr/bin/env python3
"""Extract final numbers from helpfiles and status files for the CHILI runs.

Reads the output directories of the CHILI benchmark runs (chili_venus,
chili_grid_earth_Hhigh_Chigh, chili_grid_earth_Hhigh_Cmid,
chili_grid_earth_Hhigh_Clow, and chili_tr1b) and prints the status, model time,
global melt fraction, temperatures, pressures, and gas speciation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

CASES = [
    ('chili_venus', 'Nominal Venus'),
    ('chili_grid_earth_Hhigh_Chigh', 'Earth Grid H_high C_high (10 EO, high C)'),
    ('chili_grid_earth_Hhigh_Cmid', 'Earth Grid H_high C_mid (10 EO, mid C)'),
    ('chili_grid_earth_Hhigh_Clow', 'Earth Grid H_high C_low (10 EO, low C)'),
    ('chili_tr1b', 'TRAPPIST-1 b'),
]


def extract_run_data(run_dir: Path, display_name: str) -> dict:
    """Extract metrics from status and runtime_helpfile.csv for one run.

    Parameters
    ----------
    run_dir : Path
        Path to the output folder of the simulation run.
    display_name : str
        Human-readable name of the scenario.

    Returns
    -------
    dict
        Parsed metrics including status, end time, melt fraction, and pressures.
    """
    status_file = run_dir / 'status'
    status_lines = status_file.read_text(encoding='utf-8').strip().split('\n')
    status_code = int(status_lines[0].strip())
    status_text = status_lines[1].strip() if len(status_lines) > 1 else ''

    hf_path = run_dir / 'runtime_helpfile.csv'
    with hf_path.open('r', encoding='utf-8', errors='ignore') as f:
        headers = f.readline().split()

    with hf_path.open('rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 20000))
        lines = f.read().decode('utf-8', errors='ignore').strip().split('\n')
        last_line = lines[-1].split()

    row_dict = dict(zip(headers, last_line))

    data_rows = 0
    with hf_path.open('rb') as f:
        for _ in f:
            data_rows += 1
    data_rows -= 1

    t_yr = float(row_dict['Time'])
    t_myr = t_yr / 1.0e6
    phi_mass = float(row_dict['Phi_global'])
    phi_vol = float(row_dict['Phi_global_vol']) if 'Phi_global_vol' in row_dict else None
    t_magma = float(row_dict['T_magma'])
    t_surf = float(row_dict['T_surf'])
    p_surf = float(row_dict['P_surf'])
    f_atm = float(row_dict['F_atm']) if 'F_atm' in row_dict else None
    f_net = float(row_dict['F_net']) if 'F_net' in row_dict else None

    reached_5pct = phi_mass <= 0.05 or status_code == 10

    res = {
        'run_name': run_dir.name,
        'display_name': display_name,
        'status_code': status_code,
        'status_text': status_text,
        'reached_5pct_melt': reached_5pct,
        'rows': data_rows,
        'end_time_yr': t_yr,
        'end_time_myr': t_myr,
        'phi_global_mass': phi_mass,
        'phi_global_vol': phi_vol,
        'phi_global_pct_str': f'{phi_mass * 100:.1f}%',
        'phi_global_pct_exact': phi_mass * 100,
        't_magma_k': t_magma,
        't_surf_k': t_surf,
        'p_surf_bar': p_surf,
        'f_atm_w_m2': f_atm,
        'f_net_w_m2': f_net,
    }

    for col in ('H2O_bar', 'CO2_bar', 'CO_bar', 'H2_bar', 'CH4_bar', 'O2_bar'):
        if col in row_dict:
            res[col.lower()] = float(row_dict[col])

    return res


def build_markdown_summary(results: list[dict]) -> str:
    """Format extracted run metrics into a markdown summary table.

    Parameters
    ----------
    results : list of dict
        Extracted case metrics.

    Returns
    -------
    str
        Markdown formatted table and summary notes.
    """
    lines = [
        '# CHILI Final Numbers Summary',
        '',
        '| Case | Status | Rows | End Time (Myr) | Phi mass | Phi vol | T_surf (K) | P_surf (bar) | H2O (bar) | CO2 (bar) |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for r in results:
        phi_vol_str = (
            f'{r["phi_global_vol"] * 100:.1f}%' if r['phi_global_vol'] is not None else 'N/A'
        )
        lines.append(
            f'| {r["display_name"]} | {r["status_code"]} ({r["status_text"]}) | '
            f'{r["rows"]} | {r["end_time_myr"]:.4f} | {r["phi_global_pct_str"]} | '
            f'{phi_vol_str} | {r["t_surf_k"]:.1f} | {r["p_surf_bar"]:.1f} | '
            f'{r.get("h2o_bar", 0.0):.1f} | {r.get("co2_bar", 0.0):.1f} |'
        )
    return '\n'.join(lines) + '\n'


def main() -> None:
    """Command line entry point for CHILI numbers extractor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--runs',
        type=Path,
        default=Path(
            os.path.expanduser('~/work/release-joss/install-prod/session/PROTEUS/output')
        ),
        help='Root directory holding the CHILI simulation run folders.',
    )
    parser.add_argument(
        '--json-out',
        type=Path,
        default=None,
        help='Optional path to write full JSON output.',
    )
    args = parser.parse_args()

    results = []
    for rname, disp in CASES:
        rdir = args.runs / rname
        if not rdir.is_dir():
            raise FileNotFoundError(f'Run folder not found: {rdir}')
        results.append(extract_run_data(rdir, disp))

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')

    print(build_markdown_summary(results))


if __name__ == '__main__':
    main()
