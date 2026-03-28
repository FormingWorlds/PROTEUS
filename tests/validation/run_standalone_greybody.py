#!/usr/bin/env python3
"""Generate and run standalone grey-body SPIDER vs Aragog comparison.

No atmosphere model, no UTBL, no radiogenic heating.
Pure interior physics with sigma*T^4 surface BC via dummy atmosphere.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

import toml

BASE_TOML = Path(__file__).parent / 'standalone_greybody.toml'
OUTPUT_ROOT = Path('output/standalone_greybody')

MASSES = [1.0]  # Start with 1 ME only; extend to [0.5, 1.0, 2.0, 3.0, 5.0]
SOLVERS = ['spider', 'aragog']


def generate_configs():
    """Generate TOML configs for each mass x solver combination."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config_dir = OUTPUT_ROOT / 'configs'
    config_dir.mkdir(exist_ok=True)

    base = toml.load(BASE_TOML)
    cases = []

    for mass in MASSES:
        for solver in SOLVERS:
            name = f'M{mass:.1f}_{solver}_greybody'
            cfg = _deep_copy_dict(base)

            # Set output path
            cfg['params']['out']['path'] = f'standalone_greybody/{name}'

            # Set mass
            cfg['struct']['mass_tot'] = mass

            # Set interior module
            cfg['interior']['module'] = solver

            # Write config
            cfg_path = config_dir / f'{name}.toml'
            with open(cfg_path, 'w') as f:
                toml.dump(cfg, f)
            cases.append({'name': name, 'config': str(cfg_path)})
            print(f'  {name}')

    # Write case list
    case_list = OUTPUT_ROOT / 'case_list.txt'
    with open(case_list, 'w') as f:
        for case in cases:
            f.write(f'{case["config"]}\n')

    print(f'\nGenerated {len(cases)} configs in {config_dir}')
    return cases


def generate_slurm_script(cases):
    """Generate SLURM array job script."""
    case_list = OUTPUT_ROOT / 'case_list.txt'
    log_dir = Path('output/standalone_greybody/logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    script = f"""#!/bin/bash
#SBATCH --job-name=greybody_parity
#SBATCH --array=1-{len(cases)}
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --partition=regularshort
#SBATCH --output=/home1/p311056/PROTEUS/output/standalone_greybody/logs/%a_%x.out
#SBATCH --error=/home1/p311056/PROTEUS/output/standalone_greybody/logs/%a_%x.err

# Load environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate proteus
module load netCDF-Fortran/4.6.1-gompi-2023a libarchive 2>/dev/null
export PYTHON_JULIAPKG_EXE=$HOME/.julia/juliaup/julia-1.12.5+0.x64.linux.gnu/bin/julia

# Get config for this array task
CONFIG=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {case_list})

echo "Running case $SLURM_ARRAY_TASK_ID: $CONFIG"
echo "Start: $(date)"

cd /home1/p311056/PROTEUS
proteus start -c "$CONFIG"

EXIT=$?
echo "End: $(date)"
echo "Exit: $EXIT"
exit $EXIT
"""

    script_path = OUTPUT_ROOT / 'submit.sh'
    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
    print(f'SLURM script: {script_path}')
    print(f'Submit with: sbatch {script_path}')


def _deep_copy_dict(d):
    """Deep copy a nested dict (avoids import copy)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


if __name__ == '__main__':
    print('Generating standalone grey-body comparison configs...')
    cases = generate_configs()
    generate_slurm_script(cases)
