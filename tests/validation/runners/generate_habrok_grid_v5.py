#!/usr/bin/env python
"""Generate Habrok validation grid v5: test bot-review fixes.

New Block J tests that exercise the code changes from bot review:

J1: Resume test — 1M Phase 1, short run (10 steps), then resume
    with Zalmoxis mesh restoration (tests the resume-mesh fix)
J2: 2M Phase 2 with aggressive blending (max_shift=0.02)
    Tests blending + mesh convergence at intermediate mass
J3: 1M Phase 2 with very frequent updates (u=20yr)
    Stress-tests the update trigger logic and gc.collect
J4: 3M Phase 1 with CMF=0.5 (large core, shallow mantle)
    Tests interpolation accuracy with extreme geometry
J5: 1M Phase 2 resume test — run Phase 2 for 10 steps,
    then resume to verify spider_mesh + blending state restored

Usage
-----
    python generate_habrok_grid_v5.py --outdir /scratch/$USER/habrok_validation_v5
    sbatch /scratch/$USER/habrok_validation_v5/slurm_block_J.sh
    # After J1 initial run completes, submit resume:
    sbatch /scratch/$USER/habrok_validation_v5/slurm_block_J_resume.sh
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tomllib
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import toml

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_CONFIG = SCRIPT_DIR / 'base_validation.toml'

RHO_CORE = 10738.33
RHO_MANTLE = 4000.0


def cmf_to_corefrac(cmf: float) -> float:
    """Convert core mass fraction to core radius fraction."""
    return (cmf * RHO_MANTLE / (RHO_CORE * (1.0 - cmf) + cmf * RHO_MANTLE)) ** (1.0 / 3.0)


def max_pressure_guess(mass: float) -> float:
    """Estimate max central pressure for Zalmoxis convergence."""
    if mass < 3.0:
        return 0.99e12
    elif mass < 7.0:
        return 5.0e12
    else:
        return 15.0e12


@dataclass
class Case:
    """A single validation case configuration."""

    block: str
    name_suffix: str
    mass: float
    cmf: float
    struct: str = 'zalmoxis'
    update_interval: float = 0.0
    num_levels: int = 60
    ini_entropy: float = 3000.0
    mesh_max_shift: float = 0.05
    mesh_convergence_interval: float = 10.0
    max_iters: int = 5000
    max_time: float = 1.0e6
    mem_gb: float = 3.0
    is_resume: bool = False
    description: str = ''

    @property
    def name(self) -> str:
        """Unique case name."""
        return f'{self.block}_{self.name_suffix}'

    @property
    def corefrac(self) -> float:
        """Core radius fraction from CMF."""
        return cmf_to_corefrac(self.cmf)


def build_block_J() -> list[Case]:
    """Block J: Test bot-review fixes (5 initial + 2 resume cases)."""
    cases = []

    # J1: Short Phase 1 run for resume testing
    cases.append(
        Case(
            'J',
            'M1_P1_short',
            1.0,
            0.325,
            max_iters=10,
            max_time=1.0e4,
            description='Short Phase 1 run (10 steps) for resume test',
        )
    )

    # J2: 2M Phase 2 with tight blending
    cases.append(
        Case(
            'J',
            'M2_P2_tight_blend',
            2.0,
            0.325,
            update_interval=50,
            mesh_max_shift=0.02,
            mesh_convergence_interval=5.0,
            mem_gb=5.0,
            description='2M Phase 2, tight blending (max_shift=0.02)',
        )
    )

    # J3: 1M Phase 2 with very frequent updates
    cases.append(
        Case(
            'J',
            'M1_P2_freq_updates',
            1.0,
            0.325,
            update_interval=20,
            mesh_max_shift=0.05,
            mesh_convergence_interval=5.0,
            mem_gb=5.0,
            description='1M Phase 2, very frequent updates (u=20yr)',
        )
    )

    # J4: 3M Phase 1 with CMF=0.5 (extreme geometry)
    cases.append(
        Case(
            'J',
            'M3_CMF05_P1',
            3.0,
            0.5,
            max_iters=30,
            max_time=5.0e4,
            description='3M CMF=0.5 Phase 1 (large core, shallow mantle)',
        )
    )

    # J5: 1M Phase 2 short run for resume testing
    cases.append(
        Case(
            'J',
            'M1_P2_short',
            1.0,
            0.325,
            update_interval=50,
            mesh_max_shift=0.05,
            max_iters=15,
            max_time=2.0e4,
            mem_gb=5.0,
            description='Short Phase 2 run (15 steps) for resume test',
        )
    )

    return cases


def build_resume_cases() -> list[Case]:
    """Resume variants for J1 and J5."""
    cases = []

    # J1-resume: continue the Phase 1 run
    cases.append(
        Case(
            'J',
            'M1_P1_short',  # same output dir as J1
            1.0,
            0.325,
            max_iters=25,
            max_time=5.0e4,  # extended limits
            is_resume=True,
            description='Resume J1 Phase 1 — verify mesh restoration',
        )
    )

    # J5-resume: continue the Phase 2 run
    cases.append(
        Case(
            'J',
            'M1_P2_short',  # same output dir as J5
            1.0,
            0.325,
            update_interval=50,
            mesh_max_shift=0.05,
            max_iters=30,
            max_time=5.0e4,
            mem_gb=5.0,
            is_resume=True,
            description='Resume J5 Phase 2 — verify mesh + blending restoration',
        )
    )

    return cases


def apply_overrides(config: dict, case: Case, case_outdir: str) -> dict:
    """Apply case-specific overrides to the base configuration."""
    cfg = deepcopy(config)

    cfg['params']['out']['path'] = case_outdir
    cfg['params']['stop']['iters']['maximum'] = case.max_iters
    cfg['params']['stop']['time']['maximum'] = case.max_time

    cfg['struct']['module'] = case.struct
    cfg['struct']['mass_tot'] = case.mass
    cfg['struct']['corefrac'] = round(case.corefrac, 6)
    cfg['struct']['update_interval'] = case.update_interval
    cfg['struct']['mesh_max_shift'] = case.mesh_max_shift
    cfg['struct']['mesh_convergence_interval'] = case.mesh_convergence_interval

    cfg['struct']['zalmoxis']['coremassfrac'] = case.cmf
    cfg['struct']['zalmoxis']['max_center_pressure_guess'] = max_pressure_guess(case.mass)

    cfg['interior']['spider']['num_levels'] = case.num_levels
    cfg['interior']['spider']['ini_entropy'] = case.ini_entropy

    return cfg


def generate_slurm_script(
    outdir: Path,
    block: str,
    case_indices: list[int],
    mem_gb: float,
    suffix: str = '',
    resume: bool = False,
) -> Path:
    """Generate SLURM job array script."""
    idx_str = ','.join(str(i) for i in case_indices)
    time_limit = '0-06:00:00'
    resume_flag = '--resume ' if resume else ''

    script = f"""\
#!/bin/bash
#SBATCH -J spider-val-v5-{block}{suffix}
#SBATCH --partition=regular
#SBATCH --time={time_limit}
#SBATCH --mem-per-cpu={int(mem_gb)}G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array={idx_str}
#SBATCH -o {outdir}/logs/job_%A_%a.log
#SBATCH -e {outdir}/logs/job_%A_%a.err

# ── Environment setup ────────────────────────────────────────
module purge
module load netCDF-Fortran libarchive

source ~/miniforge3/etc/profile.d/conda.sh
conda activate proteus

export FWL_DATA=$HOME/FWL_DATA
export RAD_DIR=$HOME/PROTEUS/socrates
export PETSC_DIR=$HOME/PROTEUS/petsc
export PETSC_ARCH=arch-linux-c-opt
export PYTHON_JULIAPKG_EXE=$(which julia)

# ── Run case ─────────────────────────────────────────────────
CFG=$(sed "$((SLURM_ARRAY_TASK_ID + 1))q;d" {outdir}/case_list.txt)

echo "========================================"
echo "SPIDER validation grid v5 — Block {block}{suffix}"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Config:     $CFG"
echo "Memory:     {int(mem_gb)} GB"
echo "Resume:     {resume}"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "========================================"
echo ""

proteus start --offline {resume_flag}-c "$CFG"
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "========================================"
"""

    name = f'slurm_block_{block}{suffix}.sh'
    path = outdir / name
    path.write_text(script)
    path.chmod(0o755)
    return path


def main():
    """Generate Habrok validation grid v5."""
    parser = argparse.ArgumentParser(
        description='Generate Habrok validation grid v5 (Block J — bot-review fixes)',
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        required=True,
        help='Root output directory',
    )
    parser.add_argument(
        '--base-config',
        type=Path,
        default=BASE_CONFIG,
        help=f'Base TOML config (default: {BASE_CONFIG})',
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all cases without generating files',
    )
    args = parser.parse_args()

    cases_J = build_block_J()
    resume_cases = build_resume_cases()

    # Print summary
    print(
        f'Habrok validation grid v5: {len(cases_J)} initial + {len(resume_cases)} resume cases\n'
    )
    print('  Block J — Bot-review fix testing:')
    for c in cases_J:
        extras = []
        if c.update_interval > 0:
            extras.append(f'P2 u={c.update_interval:.0f}yr')
        if abs(c.mesh_max_shift - 0.05) > 0.001:
            extras.append(f'max_shift={c.mesh_max_shift}')
        extra_str = f'  ({", ".join(extras)})' if extras else ''
        print(f'    {c.name:35s} {c.mass:4.1f}M CMF={c.cmf:<5} mem={c.mem_gb:.0f}G{extra_str}')
        print(f'      {c.description}')

    print('\n  Resume cases:')
    for c in resume_cases:
        print(f'    {c.name:35s} (resume)')
        print(f'      {c.description}')
    print()

    if args.list:
        return

    # Read base config
    base_path = args.base_config.resolve()
    if not base_path.exists():
        print(f'ERROR: Base config not found: {base_path}', file=sys.stderr)
        sys.exit(1)

    with open(base_path, 'rb') as f:
        base_config = tomllib.load(f)

    # Create output directories
    outdir = args.outdir.resolve()
    cfg_dir = outdir / 'cfgs'
    log_dir = outdir / 'logs'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(base_path, outdir / 'base_validation.toml')

    # Generate per-case configs (initial runs only; resume uses same config)
    case_paths = []
    j_indices = []

    for idx, case in enumerate(cases_J):
        case_outdir = str(outdir / 'cases' / case.name)
        cfg = apply_overrides(base_config, case, case_outdir)

        cfg_path = cfg_dir / f'{case.name}.toml'
        with open(cfg_path, 'w') as f:
            f.write(
                f'# Auto-generated config: {case.name}\n'
                f'# Block {case.block}: {case.description}\n'
                f'# Mass={case.mass} M_E, CMF={case.cmf}, '
                f'corefrac={case.corefrac:.4f}\n\n'
            )
            toml.dump(cfg, f)

        case_paths.append(str(cfg_path))
        j_indices.append(idx)

    # Write case list
    case_list_path = outdir / 'case_list.txt'
    case_list_path.write_text('\n'.join(case_paths) + '\n')

    # Resume configs: same TOML but with extended limits
    resume_paths = []
    resume_indices = []
    for idx, case in enumerate(resume_cases):
        case_outdir = str(outdir / 'cases' / case.name)
        cfg = apply_overrides(base_config, case, case_outdir)

        cfg_path = cfg_dir / f'{case.name}_resume.toml'
        with open(cfg_path, 'w') as f:
            f.write(f'# Auto-generated RESUME config: {case.name}\n# {case.description}\n\n')
            toml.dump(cfg, f)

        resume_paths.append(str(cfg_path))
        resume_indices.append(idx)

    resume_list_path = outdir / 'case_list_resume.txt'
    resume_list_path.write_text('\n'.join(resume_paths) + '\n')

    # Generate SLURM scripts
    slurm_J = generate_slurm_script(outdir, 'J', j_indices, mem_gb=5.0)
    slurm_J_resume = generate_slurm_script(
        outdir,
        'J',
        resume_indices,
        mem_gb=5.0,
        suffix='_resume',
        resume=True,
    )

    print(
        f'Generated {len(cases_J)} configs + {len(resume_cases)} resume configs in {cfg_dir}/'
    )
    print(f'Case list:         {case_list_path}')
    print(f'Resume list:       {resume_list_path}')
    print(f'SLURM Block J:     {slurm_J}')
    print(f'SLURM J resume:    {slurm_J_resume}')
    print()
    print('Next steps:')
    print(f'  1. Submit initial runs: sbatch {slurm_J}')
    print('  2. Wait for J1 + J5 to complete (~30 min)')
    print(f'  3. Submit resume tests: sbatch {slurm_J_resume}')
    print("  4. Verify resume runs use Zalmoxis mesh (check logs for 'Restored Zalmoxis mesh')")


if __name__ == '__main__':
    main()
