#!/usr/bin/env python3
"""Run SPIDER vs Aragog parity validation cases.

Generates configs from base_parity.toml for both interior modules,
then runs them either locally or via SLURM on Habrok.

Usage:
    # Dry run: generate configs only
    python run_parity.py --dry-run

    # Run locally
    python run_parity.py --local

    # Generate Habrok SLURM script
    python run_parity.py --habrok

    # Run specific mass
    python run_parity.py --local --mass 1.0

    # Run with mixing enabled
    python run_parity.py --local --with-mixing
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

BASE_TOML = Path(__file__).parent / 'base_parity.toml'
OUTPUT_BASE = Path(__file__).parent.parent.parent / 'output' / 'parity'

MASSES = [1.0]
INTERIOR_MODULES = ['spider', 'aragog']


def generate_config(mass, interior, with_mixing=False):
    """Generate a TOML config for one parity case.

    Parameters
    ----------
    mass : float
        Planet mass in Earth masses.
    interior : str
        Interior module ('spider' or 'aragog').
    with_mixing : bool
        Enable mixing and gravitational separation.

    Returns
    -------
    tuple of (str, str)
        (config_path, case_name)
    """
    import tomllib

    import tomli_w

    mix_tag = 'mix' if with_mixing else 'nomix'
    name = f'parity_M{mass:.1f}_{interior}_{mix_tag}'

    with open(BASE_TOML, 'rb') as f:
        cfg = tomllib.load(f)

    cfg['params']['out']['path'] = f'parity/{name}'
    cfg['planet']['planet_mass_tot'] = mass
    cfg['interior_energetics']['module'] = interior

    if with_mixing:
        cfg['interior_energetics']['spider']['mixing'] = True
        cfg['interior_energetics']['spider']['grav_sep'] = True
        cfg['interior_energetics']['aragog']['mixing'] = True
        cfg['interior_energetics']['aragog']['grav_sep'] = True

    # Adjust for high mass
    if mass > 3.0:
        cfg['interior_struct']['zalmoxis']['max_center_pressure_guess'] = 5e13

    config_dir = OUTPUT_BASE / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f'{name}.toml'

    with open(config_path, 'wb') as f:
        tomli_w.dump(cfg, f)

    return str(config_path), name


def run_case(config_path, name, resume=False):
    """Run a single case."""
    output_dir = Path('output') / 'parity' / name
    helpfile = output_dir / 'runtime_helpfile.csv'

    if resume and helpfile.is_file():
        lines = sum(1 for _ in open(helpfile))
        if lines > 10:
            print(f'  SKIP {name}: {lines} rows')
            return 'skipped'

    print(f'  RUN  {name}')
    cmd = ['proteus', 'start', '-c', config_path]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=14400,  # 4 hours
        )
        if result.returncode == 0:
            print(f'  PASS {name}')
            return 'passed'
        else:
            print(f'  FAIL {name} (exit {result.returncode})')
            err_file = output_dir / 'error.log'
            err_file.parent.mkdir(parents=True, exist_ok=True)
            err_file.write_text(result.stderr[-3000:] if result.stderr else '')
            return 'failed'
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT {name}')
        return 'timeout'


def generate_habrok_script(cases):
    """Generate a SLURM array job script for Habrok."""
    script_path = OUTPUT_BASE / 'submit_parity.sh'
    configs_file = OUTPUT_BASE / 'parity_cases.txt'

    with open(configs_file, 'w') as f:
        for config_path, name in cases:
            f.write(f'{config_path}\t{name}\n')

    script = f"""#!/bin/bash
#SBATCH --job-name=parity
#SBATCH --partition=regular
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --array=1-{len(cases)}
#SBATCH --output={OUTPUT_BASE}/logs/%a_%x.out
#SBATCH --error={OUTPUT_BASE}/logs/%a_%x.err

# Load PROTEUS environment
source ~/.bashrc
conda activate proteus

# Get config for this array task
CONFIG=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {configs_file} | cut -f1)
NAME=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {configs_file} | cut -f2)

echo "Running case: $NAME"
echo "Config: $CONFIG"
echo "Start: $(date)"

cd {Path.cwd()}
proteus start -c "$CONFIG"

echo "End: $(date)"
echo "Exit: $?"
"""
    (OUTPUT_BASE / 'logs').mkdir(parents=True, exist_ok=True)
    with open(script_path, 'w') as f:
        f.write(script)

    print(f'SLURM script: {script_path}')
    print(f'Cases file: {configs_file}')
    print(f'Submit with: sbatch {script_path}')


def main():
    parser = argparse.ArgumentParser(description='Run SPIDER-Aragog parity validation')
    parser.add_argument('--dry-run', action='store_true', help='Generate configs only')
    parser.add_argument('--local', action='store_true', help='Run locally')
    parser.add_argument('--habrok', action='store_true', help='Generate Habrok SLURM script')
    parser.add_argument('--mass', type=float, nargs='+', default=None)
    parser.add_argument('--interior', choices=['spider', 'aragog'], default=None)
    parser.add_argument('--with-mixing', action='store_true', help='Enable mixing + grav sep')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    masses = args.mass or MASSES
    interiors = [args.interior] if args.interior else INTERIOR_MODULES

    cases = []
    for mass in masses:
        for interior in interiors:
            config_path, name = generate_config(mass, interior, args.with_mixing)
            cases.append((config_path, name))
            print(f'Generated: {name}')

    if args.dry_run:
        print(f'\n{len(cases)} configs generated. Use --local or --habrok to run.')
        return

    if args.habrok:
        generate_habrok_script(cases)
        return

    if args.local:
        results = {}
        for config_path, name in cases:
            results[name] = run_case(config_path, name, args.resume)

        print('\n=== Results ===')
        for name, status in results.items():
            print(f'  {status:8s} {name}')


if __name__ == '__main__':
    main()
