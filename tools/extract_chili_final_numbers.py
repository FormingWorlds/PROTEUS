#!/usr/bin/env python3
"""Extract final numbers from helpfiles, status, and log files for CHILI runs.

Reads the output directories of the CHILI benchmark runs (chili_venus,
chili_grid_earth_Hhigh_Chigh, chili_grid_earth_Hhigh_Cmid,
chili_grid_earth_Hhigh_Clow, and chili_tr1b) and prints the status, model time,
global melt fraction, temperatures, pressures, atmospheric fluxes (F_atm),
wall times, and for every case the H and C retention, Delta IW and the melt
fraction range of the last interior snapshot. A quantity whose input is missing
is printed as not available; a snapshot that needs netCDF4 to be read raises.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path

CASES = [
    ('chili_venus', 'Nominal Venus'),
    ('chili_grid_earth_Hhigh_Chigh', 'Earth Grid H_high C_high (10 EO, high C)'),
    ('chili_grid_earth_Hhigh_Cmid', 'Earth Grid H_high C_mid (10 EO, mid C)'),
    ('chili_grid_earth_Hhigh_Clow', 'Earth Grid H_high C_low (10 EO, low C)'),
    ('chili_tr1b', 'TRAPPIST-1 b'),
]


def _iw_oneill02(t_k: float) -> float:
    """O'Neill & Eggins (2002) Iron-Wustite buffer."""
    return (
        2.0
        * (-244118.0 + 115.559 * t_k - 8.474 * t_k * math.log(t_k))
        / (math.log(10) * 8.31441 * t_k)
    )


def _snapshot_time(path: Path) -> float:
    """Model time [yr] in a snapshot name such as '86793456p548_int.nc'."""
    return float(path.name.split('_')[0].replace('p', '.'))


def extract_wall_time(run_dir: Path) -> float | None:
    """Total runtime in hours summed over all log parts, or None when a part has none."""
    logs = sorted(run_dir.glob('proteus_*.log'))
    if not logs:
        return None
    total_hours = 0.0
    for log in logs:
        text = log.read_text(encoding='utf-8', errors='replace')
        m = re.search(r'Total runtime:\s*([\d\.]+)\s*hours', text)
        if m:
            total_hours += float(m.group(1))
        else:
            ts = []
            for line in text.split('\n'):
                m_t = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})', line)
                if m_t:
                    ts.append(datetime.strptime(m_t.group(1), '%Y-%m-%d %H:%M:%S'))
            if len(ts) < 2:
                return None
            total_hours += (ts[-1] - ts[0]).total_seconds() / 3600.0
    return total_hours


def extract_run_data(run_dir: Path, display_name: str) -> dict:
    """Extract metrics from status, runtime_helpfile.csv, and logs for one run.

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

    Raises
    ------
    ValueError
        If the last row of runtime_helpfile.csv is short or truncated.
    """
    status_file = run_dir / 'status'
    status_lines = status_file.read_text(encoding='utf-8').strip().split('\n')
    status_code = int(status_lines[0].strip())
    status_text = status_lines[1].strip() if len(status_lines) > 1 else ''

    hf_path = run_dir / 'runtime_helpfile.csv'
    with hf_path.open('r', encoding='utf-8', errors='ignore') as f:
        header_line = f.readline().rstrip('\r\n')
        headers = header_line.split('\t') if '\t' in header_line else header_line.split()
        first_row_line = f.readline().rstrip('\r\n')
        first_row = (
            first_row_line.split('\t') if '\t' in first_row_line else first_row_line.split()
        )

    with hf_path.open('rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 20000))
        lines = f.read().decode('utf-8', errors='ignore').strip().split('\n')
        last_line_str = lines[-1].rstrip('\r\n')
        last_line = (
            last_line_str.split('\t') if '\t' in last_line_str else last_line_str.split()
        )

    if len(last_line) != len(headers):
        raise ValueError(
            f'Partial or short last line in {hf_path}: '
            f'expected {len(headers)} columns, got {len(last_line)}'
        )

    row_dict = dict(zip(headers, last_line))
    r0_dict = dict(zip(headers, first_row))

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
    wall_time_h = extract_wall_time(run_dir)

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
        'wall_time_hours': wall_time_h,
        'wall_time_days': None if wall_time_h is None else wall_time_h / 24.0,
    }

    for col in ('H2O_bar', 'CO2_bar', 'CO_bar', 'H2_bar', 'CH4_bar', 'O2_bar'):
        res[col.lower()] = float(row_dict[col]) if col in row_dict else None

    for el in ('H', 'C'):
        key = f'{el}_kg_total'
        first = float(r0_dict[key]) if key in r0_dict else 0.0
        res[f'{el.lower()}_retention_pct'] = (
            float(row_dict[key]) / first * 100.0 if first > 0 else None
        )

    o2_bar = res['o2_bar']
    res['delta_iw'] = (
        math.log10(o2_bar) - _iw_oneill02(t_magma)
        if o2_bar and o2_bar > 0 and t_magma > 0
        else None
    )

    res['mush_node_range_pct'] = None
    nc_files = sorted((run_dir / 'data').glob('*_int.nc'), key=_snapshot_time)
    if nc_files:
        try:
            import netCDF4
        except ImportError as exc:
            raise ImportError(f'netCDF4 is needed to read {nc_files[-1]}') from exc
        nc = netCDF4.Dataset(nc_files[-1])
        if 'phi_s' in nc.variables:
            phi_s = nc.variables['phi_s'][:]
            res['mush_node_range_pct'] = (
                float(phi_s.min()) * 100.0,
                float(phi_s.max()) * 100.0,
            )

    return res


def _fmt(value: float | None, spec: str, unit: str = '', missing: str = 'n/a') -> str:
    """Format ``value`` with ``spec`` and ``unit``, or return ``missing`` for None."""
    return missing if value is None else f'{value:{spec}}{unit}'


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
        '| Case | Status | Rows | End Time (Myr) | Phi mass | Phi vol | F_atm (W m-2) | Wall time | T_surf (K) | P_surf (bar) | H2O (bar) | CO2 (bar) |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for r in results:
        phi_vol = r['phi_global_vol']
        wall_str = 'n/a'
        if r['wall_time_hours'] is not None:
            wall_str = f'{r["wall_time_hours"]:.1f} h ({r["wall_time_days"]:.2f} d)'
        lines.append(
            f'| {r["display_name"]} | {r["status_code"]} ({r["status_text"]}) | '
            f'{r["rows"]} | {r["end_time_myr"]:.4f} | {r["phi_global_pct_str"]} | '
            f'{_fmt(None if phi_vol is None else phi_vol * 100, ".1f", "%")} '
            f'| {_fmt(r["f_atm_w_m2"], ".2f")} | '
            f'{wall_str} | {r["t_surf_k"]:.1f} | {r["p_surf_bar"]:.1f} | '
            f'{_fmt(r["h2o_bar"], ".1f")} | {_fmt(r["co2_bar"], ".1f")} |'
        )

    na = 'not available'
    for r in results:
        mush = r['mush_node_range_pct']
        extras = [
            f'H retention: {_fmt(r["h_retention_pct"], ".1f", "%", na)}',
            f'C retention: {_fmt(r["c_retention_pct"], ".1f", "%", na)}',
            f'Delta IW: {_fmt(r["delta_iw"], ".1f", "", na)}',
            'mush range: ' + (na if mush is None else f'{mush[0]:.1f}% to {mush[1]:.1f}%'),
        ]
        lines.append(f'\n**{r["display_name"]} notes**: ' + ', '.join(extras))

    return '\n'.join(lines) + '\n'


def main() -> None:
    """Command line entry point for CHILI numbers extractor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--runs',
        type=Path,
        required=True,
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
