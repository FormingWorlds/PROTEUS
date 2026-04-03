#!/usr/bin/env python3
"""Generate and run the coupling validation matrix.

Generates TOML configs from a parameter grid and runs them either
locally (sequential or parallel) or via SLURM on Habrok.

Usage:
    # Generate configs only (dry run)
    python run_coupling_matrix.py --dry-run

    # Run locally with 2 workers
    python run_coupling_matrix.py --local --workers 2

    # Run specific cases
    python run_coupling_matrix.py --local --mass 1.0 --interior spider

    # Generate SLURM submission script for Habrok
    python run_coupling_matrix.py --habrok

    # Resume failed cases
    python run_coupling_matrix.py --local --resume
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
from pathlib import Path

# ── Parameter grid ──────────────────────────────────────────────────

MASSES = [0.5, 1.0, 2.0, 3.0, 5.0]
CMFS = [0.2, 0.325, 0.5]
INTERIOR_MODULES = ['spider', 'aragog']
OUTGAS_MODULES = ['calliope', 'atmodeller']
VOLATILE_CONFIGS = {
    # "dry" uses minimal volatiles (1 ppmw H) to avoid division by zero in CALLIOPE
    'dry': {'H_mode': 'ppmw', 'H_budget': 1.0, 'C_mode': 'C/H', 'C_budget': 0.0, 'N_mode': 'N/H', 'N_budget': 0.0, 'S_mode': 'S/H', 'S_budget': 0.0},
    '1EO_H2O': {'H_mode': 'ppmw', 'H_budget': 709.0, 'C_mode': 'C/H', 'C_budget': 0.0, 'N_mode': 'N/H', 'N_budget': 0.0, 'S_mode': 'S/H', 'S_budget': 0.0},
    '500ppmw_H': {'H_mode': 'ppmw', 'H_budget': 500.0, 'C_mode': 'C/H', 'C_budget': 0.5, 'N_mode': 'N/H', 'N_budget': 0.1, 'S_mode': 'S/H', 'S_budget': 0.5},
}

# ── Config generation ───────────────────────────────────────────────


def case_name(mass, cmf, interior, outgas, volatiles):
    """Generate a unique case name."""
    return f'M{mass:.1f}_CMF{cmf:.3f}_{interior}_{outgas}_{volatiles}'


def generate_config(base_toml, output_dir, mass, cmf, interior, outgas, volatiles):
    """Generate a TOML config for one case by modifying the base template."""
    import tomllib

    name = case_name(mass, cmf, interior, outgas, volatiles)

    with open(base_toml, 'rb') as f:
        cfg = tomllib.load(f)

    # Override parameters
    # PROTEUS prepends 'output/' to the path, so use relative path without it
    cfg['params']['out']['path'] = f'validation/{name}'
    cfg['planet']['mass_tot'] = mass
    cfg['interior_struct']['core_frac'] = cmf
    cfg['interior_energetics']['module'] = interior
    cfg['outgas']['module'] = outgas

    # Volatile inventory
    vol_cfg = VOLATILE_CONFIGS[volatiles]
    for key, val in vol_cfg.items():
        cfg['planet']['elements'][key] = val

    # Write TOML
    config_path = Path(output_dir) / 'configs' / f'{name}.toml'
    config_path.parent.mkdir(parents=True, exist_ok=True)

    import tomli_w

    with open(config_path, 'wb') as f:
        tomli_w.dump(cfg, f)

    return str(config_path), name


def generate_all_configs(base_toml, output_dir, filters=None):
    """Generate configs for the full matrix, applying optional filters."""
    filters = filters or {}
    cases = []

    for mass, cmf, interior, outgas, volatiles in itertools.product(
        MASSES, CMFS, INTERIOR_MODULES, OUTGAS_MODULES, VOLATILE_CONFIGS.keys()
    ):
        # Apply filters
        if 'mass' in filters and mass not in filters['mass']:
            continue
        if 'interior' in filters and interior not in filters['interior']:
            continue
        if 'outgas' in filters and outgas not in filters['outgas']:
            continue
        if 'volatiles' in filters and volatiles not in filters['volatiles']:
            continue

        config_path, name = generate_config(
            base_toml, output_dir, mass, cmf, interior, outgas, volatiles
        )
        cases.append({'config': config_path, 'name': name, 'mass': mass,
                       'cmf': cmf, 'interior': interior, 'outgas': outgas,
                       'volatiles': volatiles})

    return cases


# ── Local runner ────────────────────────────────────────────────────


def run_case_local(case, resume=False):
    """Run a single case locally."""
    config = case['config']
    name = case['name']

    # Check if already completed (PROTEUS writes to output/validation/<name>/)
    output_dir = Path('output') / 'validation' / name
    helpfile = output_dir / 'runtime_helpfile.csv'
    if resume and helpfile.is_file():
        lines = sum(1 for _ in open(helpfile))
        if lines > 10:
            print(f'  SKIP {name}: already has {lines} rows')
            return 'skipped'

    print(f'  RUN  {name}')
    cmd = ['proteus', 'start', '-c', config]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200  # 2 hour timeout
        )
        if result.returncode == 0:
            print(f'  PASS {name}')
            return 'passed'
        else:
            # Check if it ran for a while before failing
            if helpfile.is_file():
                lines = sum(1 for _ in open(helpfile))
                print(f'  FAIL {name} (exit {result.returncode}, {lines} rows)')
            else:
                print(f'  FAIL {name} (exit {result.returncode}, no output)')
                # Save stderr for debugging
                err_file = output_dir / 'error.log'
                err_file.parent.mkdir(parents=True, exist_ok=True)
                err_file.write_text(result.stderr[-2000:] if result.stderr else '')
            return 'failed'
    except subprocess.TimeoutExpired:
        print(f'  TIMEOUT {name}')
        return 'timeout'


def run_matrix_local(cases, workers=1, resume=False):
    """Run cases locally, optionally in parallel."""
    if workers <= 1:
        results = {}
        for case in cases:
            results[case['name']] = run_case_local(case, resume)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(run_case_local, case, resume): case['name']
                for case in cases
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as e:
                    results[name] = f'error: {e}'
                    print(f'  ERROR {name}: {e}')

    # Summary
    passed = sum(1 for v in results.values() if v == 'passed')
    failed = sum(1 for v in results.values() if v == 'failed')
    skipped = sum(1 for v in results.values() if v == 'skipped')
    timeout = sum(1 for v in results.values() if v == 'timeout')
    print(f'\nMatrix complete: {passed} passed, {failed} failed, '
          f'{skipped} skipped, {timeout} timeout, {len(results)} total')

    return results


# ── SLURM submission ────────────────────────────────────────────────


def generate_slurm_script(cases, output_dir):
    """Generate a SLURM array job script for Habrok."""
    script_path = Path(output_dir) / 'submit_matrix.sh'

    # Write case list
    case_list_path = Path(output_dir) / 'case_list.txt'
    with open(case_list_path, 'w') as f:
        for case in cases:
            f.write(f'{case["config"]}\n')

    script = f"""#!/bin/bash
#SBATCH --job-name=proteus_validation
#SBATCH --array=1-{len(cases)}
#SBATCH --time=96:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --partition=regular
#SBATCH --output=/home1/p311056/PROTEUS/output/validation/logs/%a_%x.out
#SBATCH --error=/home1/p311056/PROTEUS/output/validation/logs/%a_%x.err

# Load environment
source ~/miniforge3/etc/profile.d/conda.sh
conda activate proteus
module load netCDF-Fortran/4.6.1-gompi-2023a libarchive 2>/dev/null
export PYTHON_JULIAPKG_EXE=$(which julia)

# Get config for this array task
CONFIG=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {case_list_path})

echo "Running case $SLURM_ARRAY_TASK_ID: $CONFIG"
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


# ── CLI ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='Run coupling validation matrix')
    parser.add_argument('--dry-run', action='store_true', help='Generate configs only')
    parser.add_argument('--local', action='store_true', help='Run locally')
    parser.add_argument('--habrok', action='store_true', help='Generate SLURM script')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers (local)')
    parser.add_argument('--resume', action='store_true', help='Skip completed cases')
    parser.add_argument('--mass', type=float, nargs='+', help='Filter by mass')
    parser.add_argument('--interior', choices=['spider', 'aragog'], help='Filter by interior')
    parser.add_argument('--outgas', choices=['calliope', 'atmodeller'], help='Filter by outgas')
    parser.add_argument('--volatiles', choices=list(VOLATILE_CONFIGS.keys()), help='Filter')
    parser.add_argument('--output-dir', default='output/validation', help='Output directory')
    parser.add_argument('--base-toml', default='tests/validation/base_coupling.toml')

    args = parser.parse_args()

    # Build filters
    filters = {}
    if args.mass:
        filters['mass'] = args.mass
    if args.interior:
        filters['interior'] = [args.interior]
    if args.outgas:
        filters['outgas'] = [args.outgas]
    if args.gas_prs:
        filters['volatiles'] = [args.gas_prs]

    # Generate configs
    cases = generate_all_configs(args.base_toml, args.output_dir, filters)
    print(f'Generated {len(cases)} configs')

    if args.dry_run:
        for case in cases:
            print(f'  {case["name"]}')
        return

    if args.habrok:
        generate_slurm_script(cases, args.output_dir)
        return

    if args.local:
        run_matrix_local(cases, workers=args.workers, resume=args.resume)
        return

    parser.print_help()


if __name__ == '__main__':
    main()
