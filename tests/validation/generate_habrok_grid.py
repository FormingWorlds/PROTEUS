#!/usr/bin/env python
"""Generate Habrok validation grid for Zalmoxis-SPIDER coupling.

Produces 80 individual TOML configs and a SLURM job array script for
comprehensive testing of interior physics, melt evolution, thermal
evolution, and EOS validity across planetary masses.

Test blocks
-----------
A : AW vs ZAL core comparison      5 masses x 4 CMFs x 2 modes = 40
B : AW stability frontier          2 masses x 3 CMFs            =  6
C : ZAL high-mass frontier         4 masses x 4 CMFs            = 16
D : Phase 2 structural feedback    3 masses                     =  3
E : Resolution convergence         3 num_levels values          =  3
F : Initial entropy sensitivity    2 CMFs x 3 ini_S values      =  6
G : EOS edge probing               3 masses x 2 ini_S values    =  6
                                                           Total = 80

Usage
-----
    # Generate configs (on Habrok login node)
    python generate_habrok_grid.py --outdir /scratch/$USER/habrok_validation

    # Submit to SLURM
    sbatch /scratch/$USER/habrok_validation/slurm_dispatch.sh

    # Monitor
    squeue -u $USER

Prerequisites
-------------
1. PROTEUS installed with tl/zalmoxis-spider-coupling branch
2. SPIDER compiled from tl/external-mesh branch (with alpha clamp fix)
3. FWL_DATA populated (run `proteus get` on login node first)
4. Conda environment activated with all dependencies
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

# ── Physical constants ────────────────────────────────────────────

RHO_CORE = 10738.33  # kg/m³ (iron core)
RHO_MANTLE = 4000.0  # kg/m³ (MgSiO3 mantle)


def cmf_to_corefrac(cmf: float) -> float:
    """Convert core mass fraction to core radius fraction.

    Uses uniform two-layer approximation (constant density per layer).

    Parameters
    ----------
    cmf : float
        Core mass fraction (0 to 1).

    Returns
    -------
    float
        Core radius fraction (0 to 1).
    """
    return (cmf * RHO_MANTLE / (RHO_CORE * (1.0 - cmf) + cmf * RHO_MANTLE)) ** (1.0 / 3.0)


def max_pressure_guess(mass: float) -> float:
    """Estimate max central pressure guess for Zalmoxis solver.

    Scales with planet mass to ensure the pressure solver converges.

    Parameters
    ----------
    mass : float
        Planet mass in Earth masses.

    Returns
    -------
    float
        Maximum central pressure guess in Pa.
    """
    if mass < 3.0:
        return 0.99e12
    elif mass < 7.0:
        return 5.0e12
    elif mass < 12.0:
        return 15.0e12
    else:
        return 30.0e12


# ── Case definition ──────────────────────────────────────────────


BLOCK_LABELS = {
    'A': 'AW vs ZAL comparison',
    'B': 'AW stability frontier',
    'C': 'ZAL high-mass frontier',
    'D': 'Phase 2 feedback',
    'E': 'Resolution convergence',
    'F': 'Initial entropy sensitivity',
    'G': 'EOS edge probing',
}


@dataclass
class Case:
    """A single validation case configuration.

    Parameters
    ----------
    block : str
        Block identifier (A-G).
    mass : float
        Planet mass in Earth masses.
    cmf : float
        Core mass fraction.
    struct : str
        Structure module: "self" (AW) or "zalmoxis" (ZAL).
    update_interval : float
        Structure update interval in years (0 = Phase 1, >0 = Phase 2).
    num_levels : int
        SPIDER mesh resolution (basic nodes).
    ini_entropy : float
        Initial adiabat entropy in J/(kg*K).
    """

    block: str
    mass: float
    cmf: float
    struct: str
    update_interval: float = 0.0
    num_levels: int = 60
    ini_entropy: float = 3000.0

    @property
    def name(self) -> str:
        """Unique case name for output directory."""
        tag = 'AW' if self.struct == 'self' else 'ZAL'
        name = f'{self.block}_M{self.mass}_CMF{self.cmf}_{tag}'
        if self.update_interval > 0:
            name += f'_P2u{int(self.update_interval)}'
        if self.num_levels != 60:
            name += f'_n{self.num_levels}'
        if abs(self.ini_entropy - 3000.0) > 0.1:
            name += f'_S{int(self.ini_entropy)}'
        return name

    @property
    def corefrac(self) -> float:
        """Core radius fraction derived from CMF."""
        return cmf_to_corefrac(self.cmf)


def build_cases() -> list[Case]:
    """Build the full 80-case test matrix.

    Returns
    -------
    list[Case]
        All validation cases, grouped by block.
    """
    cases = []

    # Block A: Core AW vs ZAL comparison (40 cases)
    # Direct comparison across the parameter range where both should work.
    # AW failures at high mass/low CMF are themselves useful data.
    for mass in [0.5, 1.0, 2.0, 3.0, 5.0]:
        for cmf in [0.10, 0.20, 0.325, 0.50]:
            for struct in ['self', 'zalmoxis']:
                cases.append(Case('A', mass, cmf, struct))

    # Block B: AW stability frontier (6 cases)
    # Probe where AW breaks at high mass. Skip CMF=0.10 (known to fail at 3 M_E).
    for mass in [7.0, 10.0]:
        for cmf in [0.20, 0.325, 0.50]:
            cases.append(Case('B', mass, cmf, 'self'))

    # Block C: ZAL high-mass frontier (16 cases)
    # Find maximum stable mass for each CMF with Zalmoxis mesh.
    for mass in [7.0, 10.0, 12.0, 15.0]:
        for cmf in [0.10, 0.20, 0.325, 0.50]:
            cases.append(Case('C', mass, cmf, 'zalmoxis'))

    # Block D: Phase 2 structural feedback (3 cases)
    # Test iterative Zalmoxis-SPIDER coupling at Earth-like CMF.
    # Compare against Block A ZAL cases at same mass (which use update_interval=0).
    for mass in [1.0, 3.0, 5.0]:
        cases.append(Case('D', mass, 0.325, 'zalmoxis', update_interval=100.0))

    # Block E: Resolution convergence (3 cases)
    # Verify that 60-node default is adequate (reference: 3 M_E, CMF=0.325, ZAL).
    for nl in [30, 90, 120]:
        cases.append(Case('E', 3.0, 0.325, 'zalmoxis', num_levels=nl))

    # Block F: Initial entropy sensitivity (6 cases)
    # Test robustness to starting conditions at two CMFs.
    # ini_S=3200 at CMF=0.10 pushes close to solid EOS table edge.
    for cmf in [0.10, 0.325]:
        for ini_s in [2600.0, 2800.0, 3200.0]:
            cases.append(Case('F', 3.0, cmf, 'zalmoxis', ini_entropy=ini_s))

    # Block G: EOS edge probing (6 cases)
    # Low CMF + high mass = high CMB pressure => maximal EOS stress.
    # Vary ini_S to test whether alpha clamp sustains the solver.
    for mass in [5.0, 7.0, 10.0]:
        for ini_s in [2800.0, 3200.0]:
            cases.append(Case('G', mass, 0.10, 'zalmoxis', ini_entropy=ini_s))

    return cases


# ── Config generation ────────────────────────────────────────────


def apply_overrides(config: dict, case: Case, case_outdir: str) -> dict:
    """Apply case-specific overrides to the base configuration.

    Parameters
    ----------
    config : dict
        Base configuration dict (will not be modified).
    case : Case
        Case specification.
    case_outdir : str
        Absolute path for this case's output directory.

    Returns
    -------
    dict
        New configuration dict with overrides applied.
    """
    cfg = deepcopy(config)

    # Output path
    cfg['params']['out']['path'] = case_outdir

    # Structure module and planet parameters
    cfg['struct']['module'] = case.struct
    cfg['struct']['mass_tot'] = case.mass
    cfg['struct']['corefrac'] = round(case.corefrac, 6)
    cfg['struct']['update_interval'] = case.update_interval

    # Zalmoxis parameters (always set for consistency; ignored in AW mode)
    cfg['struct']['zalmoxis']['coremassfrac'] = case.cmf
    cfg['struct']['zalmoxis']['weight_iron_frac'] = case.cmf
    cfg['struct']['zalmoxis']['max_center_pressure_guess'] = max_pressure_guess(case.mass)

    # SPIDER parameters
    cfg['interior']['spider']['num_levels'] = case.num_levels
    cfg['interior']['spider']['ini_entropy'] = case.ini_entropy

    return cfg


# ── SLURM script ─────────────────────────────────────────────────


def generate_slurm_script(outdir: Path, n_cases: int) -> Path:
    """Generate SLURM job array submission script.

    Parameters
    ----------
    outdir : Path
        Root output directory.
    n_cases : int
        Number of cases in the grid.

    Returns
    -------
    Path
        Path to the generated SLURM script.
    """
    script = f"""\
#!/bin/bash
#SBATCH -J spider-val
#SBATCH --partition=regular
#SBATCH --time=1-00:00:00
#SBATCH --mem-per-cpu=3G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{n_cases - 1}%{min(n_cases, 80)}
#SBATCH -o {outdir}/logs/job_%A_%a.log
#SBATCH -e {outdir}/logs/job_%A_%a.err

# ── Environment setup ────────────────────────────────────────
# Adjust these lines for your Habrok environment.
module purge
module load netCDF-Fortran libarchive

# Activate conda environment (uncomment and adjust):
# source ~/miniforge3/etc/profile.d/conda.sh
# conda activate proteus

# ── Run case ─────────────────────────────────────────────────
CFG=$(sed "$((SLURM_ARRAY_TASK_ID + 1))q;d" {outdir}/case_list.txt)

echo "========================================"
echo "SPIDER validation grid"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Config:     $CFG"
echo "Node:       $(hostname)"
echo "Start:      $(date)"
echo "========================================"
echo ""

proteus start --offline -c "$CFG"
EXIT_CODE=$?

echo ""
echo "========================================"
echo "Finished:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "========================================"
"""

    path = outdir / 'slurm_dispatch.sh'
    path.write_text(script)
    path.chmod(0o755)
    return path


# ── Main ─────────────────────────────────────────────────────────


def main():
    """Generate Habrok validation grid."""
    parser = argparse.ArgumentParser(
        description='Generate Habrok validation grid for Zalmoxis-SPIDER coupling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        required=True,
        help='Root output directory (e.g. /scratch/$USER/habrok_validation)',
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

    # Build cases
    cases = build_cases()

    # Print summary by block
    blocks: dict[str, list[Case]] = {}
    for c in cases:
        blocks.setdefault(c.block, []).append(c)

    print(f'Habrok validation grid: {len(cases)} cases\n')
    for block_id, block_cases in sorted(blocks.items()):
        label = BLOCK_LABELS.get(block_id, '')
        print(f'  Block {block_id} -- {label} ({len(block_cases)} cases)')
        for c in block_cases:
            extras = []
            if c.update_interval > 0:
                extras.append(f'update={c.update_interval:.0f}yr')
            if c.num_levels != 60:
                extras.append(f'n={c.num_levels}')
            if abs(c.ini_entropy - 3000.0) > 0.1:
                extras.append(f'S0={c.ini_entropy:.0f}')
            extra_str = f'  ({", ".join(extras)})' if extras else ''
            print(f'    {c.name:45s} {c.mass:5.1f} M_E  CMF={c.cmf:<5}{extra_str}')
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

    # Copy base config for reference
    shutil.copy2(base_path, outdir / 'base_validation.toml')

    # Generate per-case configs
    case_paths = []
    for case in cases:
        case_outdir = str(outdir / 'cases' / case.name)
        cfg = apply_overrides(base_config, case, case_outdir)

        cfg_path = cfg_dir / f'{case.name}.toml'
        with open(cfg_path, 'w') as f:
            f.write(
                f'# Auto-generated config for {case.name}\n'
                f'# Block {case.block}: {BLOCK_LABELS.get(case.block, "")}\n'
                f'# Mass={case.mass} M_E, CMF={case.cmf}, '
                f'mode={case.struct}, corefrac={case.corefrac:.4f}\n\n'
            )
            toml.dump(cfg, f)

        case_paths.append(str(cfg_path))

    # Write case list (for SLURM array indexing)
    case_list_path = outdir / 'case_list.txt'
    case_list_path.write_text('\n'.join(case_paths) + '\n')

    # Generate SLURM script
    slurm_path = generate_slurm_script(outdir, len(cases))

    # Summary
    print(f'Generated {len(cases)} configs in {cfg_dir}/')
    print(f'Case list:     {case_list_path}')
    print(f'SLURM script:  {slurm_path}')
    print()
    print('Next steps:')
    print('  1. Review SLURM script and adjust conda/module lines:')
    print(f'     {slurm_path}')
    print('  2. Ensure FWL_DATA is populated: proteus get')
    print(f'  3. Submit: sbatch {slurm_path}')
    print('  4. Monitor: squeue -u $USER')


if __name__ == '__main__':
    main()
