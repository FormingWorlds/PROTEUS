#!/usr/bin/env python3
"""Focused test: 4 all-dummy parity + 4 Earth-like dry cases.

8 total cases to validate the entropy IC fix before full rerun.

Usage:
    python run_focused_test.py --habrok
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def generate_config(base_toml, output_dir, name, overrides):
    import tomllib

    import tomli_w

    with open(base_toml, 'rb') as f:
        cfg = tomllib.load(f)

    cfg['params']['out']['path'] = f'focused_test/{name}'

    for key_path, value in overrides.items():
        parts = key_path.split('.')
        d = cfg
        for p in parts[:-1]:
            d = d[p]
        d[parts[-1]] = value

    config_path = Path(output_dir) / 'configs' / f'{name}.toml'
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'wb') as f:
        tomli_w.dump(cfg, f)

    return str(config_path), name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--habrok', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    output_dir = Path('output') / 'focused_test'
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = []

    # 4 all-dummy parity cases (WolfBower2018 EOS)
    # Hot case uses 3000 K, not 3500 K: at 3500 K the near-fully-molten
    # interior plus pure blackbody BC (F_atm = sigma*T^4 = 8.5e6 W/m^2) is
    # too stiff for SPIDER's CVode. Retries do reduce dt correctly, but the
    # required top-cell cooling rate exceeds what MLT can propagate from an
    # isentropic IC, so the solver eventually aborts. 3000 K stays within
    # the stable envelope.
    parity_base = base_dir / 'base_parity_isolated.toml'
    for interior in ['spider', 'aragog']:
        for tsurf in [2500, 3000]:
            name = f'dummy_{interior}_Tsurf{tsurf}'
            _, n = generate_config(
                parity_base,
                output_dir,
                name,
                {
                    'interior_energetics.module': interior,
                    'planet.tsurf_init': tsurf,
                },
            )
            cases.append({'config': str(Path(output_dir) / 'configs' / f'{n}.toml'), 'name': n})

    # 4 Earth-like dry cases (PALEOS EOS, AGNI atmosphere)
    earthlike_base = base_dir / 'base_earthlike_dry.toml'
    for interior in ['spider', 'aragog']:
        for outgas in ['calliope', 'atmodeller']:
            name = f'earthlike_{interior}_{outgas}_dry'
            _, n = generate_config(
                earthlike_base,
                output_dir,
                name,
                {
                    'interior_energetics.module': interior,
                    'outgas.module': outgas,
                },
            )
            cases.append({'config': str(Path(output_dir) / 'configs' / f'{n}.toml'), 'name': n})

    print(f'Generated {len(cases)} configs:')
    for c in cases:
        print(f'  {c["name"]}')

    if args.habrok:
        case_list_path = output_dir / 'case_list.txt'
        with open(case_list_path, 'w') as f:
            for case in cases:
                f.write(f'{case["config"]}\n')

        abs_case_list = str(Path(case_list_path).resolve())
        script_path = output_dir / 'submit_focused.sh'

        script = f"""#!/bin/bash
#SBATCH --job-name=proteus_focused
#SBATCH --array=1-{len(cases)}
#SBATCH --time=7-00:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=1
#SBATCH --partition=regular
#SBATCH --output=/home1/p311056/PROTEUS/output/focused_test/logs/%a_%x.out
#SBATCH --error=/home1/p311056/PROTEUS/output/focused_test/logs/%a_%x.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate proteus
module load netCDF-Fortran/4.6.1-gompi-2023a libarchive 2>/dev/null

# Force Python to not use bytecode cache (NFS propagation delays)
export PYTHONDONTWRITEBYTECODE=1
# Clear any stale pycache on the compute node
find $HOME/PROTEUS/src -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null
find $HOME/PROTEUS/aragog/src -name '__pycache__' -exec rm -rf {{}} + 2>/dev/null

JULIA_BIN=$(find $HOME/.juliaup -name "julia" -path "*/julia-1.11*/bin/julia" 2>/dev/null | head -1)
if [ -z "$JULIA_BIN" ]; then JULIA_BIN=$(which julia 2>/dev/null); fi
export PYTHON_JULIAPKG_EXE=$JULIA_BIN
export FWL_DATA=$HOME/FWL_DATA
export RAD_DIR=$HOME/PROTEUS/socrates
export FC_DIR=$HOME/PROTEUS/AGNI/fastchem

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
        print(f'\nSLURM script: {script_path}')
        print(f'Submit with: sbatch {script_path}')


if __name__ == '__main__':
    main()
