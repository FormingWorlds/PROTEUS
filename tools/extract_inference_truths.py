#!/usr/bin/env python3
"""Extract inference observables from a reference PROTEUS runtime helpfile."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI args"""

    parser = argparse.ArgumentParser(
        description=(
            'Read an inference TOML config, locate its reference simulation output, '
            'and print observable values from runtime_helpfile.csv.'
        )
    )
    parser.add_argument('infer_config', type=Path, help='Path to *.infer.toml file')
    parser.add_argument(
        '--all',
        action='store_true',
        help='Extract all columns from the final helpfile row (not only [observables] keys)',
    )
    return parser.parse_args()


def read_toml(path: Path) -> dict:
    """Generic TOML reader"""
    with path.open('rb') as handle:
        return tomllib.load(handle)


def resolve_ref_config(ref_config: str, infer_path: Path, repo_root: Path) -> Path:
    """Get path to reference PROTEUS config file, from inference config"""
    ref_path = Path(ref_config)
    if ref_path.is_absolute():
        if ref_path.is_file():
            return ref_path
        raise FileNotFoundError(f'Reference config does not exist: {ref_path}')

    candidates = [repo_root / ref_path, infer_path.parent / ref_path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = '\n'.join(f'  - {cand}' for cand in candidates)
    raise FileNotFoundError(f'Cannot resolve ref_config "{ref_config}". Tried:\n{tried}')


def resolve_helpfile(out_path: str, repo_root: Path) -> Path:
    """Get path to reference runtime_helpfile.csv, from reference config's params.out.path"""
    run_path = Path(out_path)
    if run_path.is_absolute():
        helpfile = run_path / 'runtime_helpfile.csv'
        if helpfile.is_file():
            return helpfile
        raise FileNotFoundError(f'Cannot find runtime_helpfile.csv at: {helpfile}')

    candidates = [
        repo_root / 'output' / run_path / 'runtime_helpfile.csv',
        repo_root / run_path / 'runtime_helpfile.csv',
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    tried = '\n'.join(f'  - {cand}' for cand in candidates)
    raise FileNotFoundError(
        f'Cannot locate runtime_helpfile.csv for params.out.path="{out_path}". Tried:\n{tried}'
    )


def extract_values(df: pd.DataFrame, names: list[str], idx: int = -1) -> dict[str, object]:
    """Extract observable values from specific row of the helpfile, given a list of column names"""
    missing = [name for name in names if name not in df.columns]
    if missing:
        msg = ', '.join(missing)
        raise KeyError(f'Observable(s) not present in runtime_helpfile.csv: {msg}')

    row = df.iloc[idx]
    return {name: row[name] for name in names}


def main() -> int:
    args = parse_args()
    infer_path = args.infer_config.expanduser().resolve()
    if not infer_path.is_file():
        raise FileNotFoundError(f'Inference config does not exist: {infer_path}')

    # read inference config
    infer_cfg = read_toml(infer_path)
    repo_root = Path(__file__).parent.parent.resolve()
    print(f'# Repository root: {repo_root}')

    # read reference config
    ref_config_raw = infer_cfg.get('ref_config')
    if not isinstance(ref_config_raw, str) or not ref_config_raw:
        raise KeyError('Inference config is missing a string `ref_config` entry')
    ref_config_path = resolve_ref_config(ref_config_raw, infer_path, repo_root)
    ref_cfg = read_toml(ref_config_path)

    # get target time from config
    target_time = ref_cfg.get('params', {}).get('stop', {}).get('time', {}).get('maximum', {})
    if not isinstance(target_time, (int, float)):
        raise KeyError(
            'Reference config is missing `params.stop.time.maximum` entry, or it is not a number'
        )
    target_time = float(target_time)

    # get helpfile path from reference config and read it
    out_path = ref_cfg.get('params', {}).get('out', {}).get('path')
    if not isinstance(out_path, str) or not out_path:
        raise KeyError('Reference config is missing `params.out.path`')
    helpfile_path = resolve_helpfile(out_path, repo_root)

    helpfile = pd.read_csv(helpfile_path, sep=r'\s+')
    if helpfile.empty:
        raise ValueError(f'Helpfile has no rows: {helpfile_path}')

    # get target index
    time_col = helpfile['Time']
    target_idx = np.argmin(np.abs(time_col - target_time))

    # work out which keys to extract
    if args.all:
        names = list(helpfile.columns)
    else:
        observables = infer_cfg.get('observables', {})
        if not isinstance(observables, dict) or not observables:
            raise KeyError(
                'Inference config has no [observables] keys to extract. Use --all to dump all columns.'
            )
        names = list(observables.keys())
    names.extend(['Time'])

    # extract values and print
    extracted = extract_values(helpfile, names, idx=target_idx)
    print(f'# infer_config: {infer_path}')
    print(f'# ref_config: {ref_config_path}')
    print(f'# helpfile: {helpfile_path}')
    print(f'# extracted at index: {target_idx}, time: {extracted["Time"]:.3e} years')
    print('[observables]')
    for name, value in extracted.items():
        if name == 'Time':
            continue
        print(f'"{name}" = {value:.10e}')

    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
