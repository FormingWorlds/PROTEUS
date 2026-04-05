#!/usr/bin/env python3
"""Generate and submit isolated SPIDER vs Aragog parity test.

4 cases: 2 interior modules x 2 initial temperatures.
Everything dummy except interior energetics. WolfBower2018 EOS.
1 M_Earth, CMF=0.325, 0.1 Earth Ocean H.

Usage:
    python run_parity_isolated.py --habrok    # Generate SLURM script
    python run_parity_isolated.py --dry-run   # Generate configs only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

INTERIOR_MODULES = ['spider', 'aragog']
TSURF_VALUES = [2500, 3500]


def case_name(interior, tsurf):
    return f'parity_M1.0_{interior}_Tsurf{tsurf}'


def generate_config(base_toml, output_dir, interior, tsurf):
    import tomllib

    name = case_name(interior, tsurf)

    with open(base_toml, 'rb') as f:
        cfg = tomllib.load(f)

    cfg['params']['out']['path'] = f'parity_isolated/{name}'
    cfg['interior_energetics']['module'] = interior
    cfg['planet']['tsurf_init'] = tsurf

    config_path = Path(output_dir) / 'configs' / f'{name}.toml'
    config_path.parent.mkdir(parents=True, exist_ok=True)

    import tomli_w

    with open(config_path, 'wb') as f:
        tomli_w.dump(cfg, f)

    return str(config_path), name


def generate_all(base_toml, output_dir):
    cases = []
    for interior in INTERIOR_MODULES:
        for tsurf in TSURF_VALUES:
            config_path, name = generate_config(
                base_toml, output_dir, interior, tsurf,
            )
            cases.append({'config': config_path, 'name': name})
    return cases


def generate_slurm_script(cases, output_dir):
    script_path = Path(output_dir) / 'submit_parity.sh'

    case_list_path = Path(output_dir) / 'case_list.txt'
    with open(case_list_path, 'w') as f:
        for case in cases:
            f.write(f'{case["config"]}\n')

    abs_case_list = str(Path(case_list_path).resolve())

    script = f"""#!/bin/bash
#SBATCH --job-name=proteus_parity
#SBATCH --array=1-{len(cases)}
#SBATCH --time=7-00:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --partition=regular
#SBATCH --output=/home1/p311056/PROTEUS/output/parity_isolated/logs/%a_%x.out
#SBATCH --error=/home1/p311056/PROTEUS/output/parity_isolated/logs/%a_%x.err

# Load environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate proteus
module load netCDF-Fortran/4.6.1-gompi-2023a libarchive 2>/dev/null

# Pin Julia to 1.11.x
JULIA_BIN=$(find $HOME/.juliaup -name "julia" -path "*/julia-1.11*/bin/julia" 2>/dev/null | head -1)
if [ -z "$JULIA_BIN" ]; then
    JULIA_BIN=$(which julia 2>/dev/null)
fi
export PYTHON_JULIAPKG_EXE=$JULIA_BIN

# Environment variables
export FWL_DATA=$HOME/FWL_DATA
export RAD_DIR=$HOME/PROTEUS/socrates
export FC_DIR=$HOME/PROTEUS/AGNI/fastchem

# Get config for this array task
CONFIG=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {abs_case_list})

echo "Task $SLURM_ARRAY_TASK_ID: $CONFIG"
echo "Start: $(date)"

cd /home1/p311056/PROTEUS
proteus start -c "$CONFIG"

EXIT=$?
echo "End: $(date)"
echo "Exit: $EXIT"
exit $EXIT
"""

    with open(script_path, 'w') as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    print(f'SLURM script: {script_path}')
    print(f'Case list: {case_list_path} ({len(cases)} cases)')
    print(f'Submit with: sbatch {script_path}')
    return str(script_path)


def main():
    parser = argparse.ArgumentParser(
        description='Run isolated SPIDER vs Aragog parity test',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--habrok', action='store_true')
    args = parser.parse_args()

    base_toml = Path(__file__).parent / 'base_parity_isolated.toml'
    output_dir = Path('output') / 'parity_isolated'
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = generate_all(base_toml, output_dir)
    print(f'Generated {len(cases)} configs:')
    for c in cases:
        print(f'  {c["name"]}')

    if args.habrok:
        generate_slurm_script(cases, output_dir)
    elif not args.dry_run:
        print('\nUse --habrok or --dry-run')


if __name__ == '__main__':
    main()
