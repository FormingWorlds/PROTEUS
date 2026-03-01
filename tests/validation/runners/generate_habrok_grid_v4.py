#!/usr/bin/env python
"""Generate Habrok validation grid v4: mesh interpolation + blending tests.

Extends v3 with two new blocks for the PROTEUS PR:

Block H: Mesh interpolation validation (5 cases)
    Tests write_spider_mesh_file() accuracy at various masses,
    CMFs, and SPIDER node counts.

Block I: Mesh blending + memory profiling (5 cases)
    Tests blend_mesh_files() behaviour under different max_shift
    settings and monitors memory usage during Phase 2.

All cases run in the same PROTEUS environment as v3 (AGNI + SPIDER
+ MORS + CALLIOPE + ZEPHYRUS).

Usage
-----
    python generate_habrok_grid_v4.py --outdir /scratch/$USER/habrok_validation_v4

    # Submit Block H (3G jobs)
    sbatch /scratch/$USER/habrok_validation_v4/slurm_block_H.sh

    # Submit Block I (5G jobs)
    sbatch /scratch/$USER/habrok_validation_v4/slurm_block_I.sh
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tomllib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import toml

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_CONFIG = SCRIPT_DIR / "base_validation.toml"

RHO_CORE = 10738.33
RHO_MANTLE = 4000.0


def cmf_to_corefrac(cmf: float) -> float:
    """Convert core mass fraction to core radius fraction."""
    return (cmf * RHO_MANTLE / (RHO_CORE * (1.0 - cmf) + cmf * RHO_MANTLE)) ** (
        1.0 / 3.0
    )


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
    """A single validation case configuration.

    Parameters
    ----------
    block : str
        Block identifier (H or I).
    mass : float
        Planet mass in Earth masses.
    cmf : float
        Core mass fraction.
    struct : str
        Structure module: "zalmoxis" for all v4 cases.
    update_interval : float
        Structure update interval [yr]. 0 = Phase 1.
    num_levels : int
        SPIDER mesh resolution (basic nodes).
    ini_entropy : float
        Initial adiabat entropy [J/(kg*K)].
    temperature_mode : str or None
        Override for Zalmoxis temperature_mode.
    mesh_max_shift : float
        Max fractional radius shift per mesh update (blending).
    mesh_convergence_interval : float
        Convergence loop interval [yr].
    max_iters : int
        Maximum number of iterations (PROTEUS time steps).
    max_time : float
        Maximum simulation time [yr].
    mem_gb : float
        SLURM memory request in GB.
    description : str
        Human-readable case description.
    """

    block: str
    mass: float
    cmf: float
    struct: str = "zalmoxis"
    update_interval: float = 0.0
    num_levels: int = 60
    ini_entropy: float = 3000.0
    temperature_mode: str | None = field(default=None)
    mesh_max_shift: float = 0.05
    mesh_convergence_interval: float = 10.0
    max_iters: int = 5000
    max_time: float = 1.0e6
    mem_gb: float = 3.0
    description: str = ""

    @property
    def name(self) -> str:
        """Unique case name."""
        tag = "ZAL"
        name = f"{self.block}_M{self.mass}_CMF{self.cmf}_{tag}"
        if self.update_interval > 0:
            name += f"_P2u{int(self.update_interval)}"
        if self.num_levels != 60:
            name += f"_n{self.num_levels}"
        if abs(self.mesh_max_shift - 0.05) > 0.001:
            name += f"_ms{self.mesh_max_shift}"
        return name

    @property
    def corefrac(self) -> float:
        """Core radius fraction from CMF."""
        return cmf_to_corefrac(self.cmf)


def build_block_H() -> list[Case]:
    """Block H: Mesh interpolation validation (5 cases).

    Short Phase 1 runs that validate write_spider_mesh_file() accuracy.
    After each run, validate_mesh_interpolation.py compares the SPIDER
    mesh against the Zalmoxis output.
    """
    cases = []

    # H1: Standard 1M, CMF=0.325 — baseline interpolation test
    cases.append(
        Case(
            "H", 1.0, 0.325,
            max_iters=20, max_time=1.0e4,
            description="Mesh interpolation baseline (1M, CMF=0.325)",
        )
    )

    # H2: 3M, CMF=0.1 — extreme case (low CMF, wide mantle, high P)
    cases.append(
        Case(
            "H", 3.0, 0.10,
            max_iters=20, max_time=1.0e4,
            description="Mesh interpolation extreme (3M, CMF=0.1, wide mantle)",
        )
    )

    # H3: 150 SPIDER nodes — tests high-resolution interpolation
    cases.append(
        Case(
            "H", 1.0, 0.325, num_levels=150,
            max_iters=20, max_time=1.0e4,
            description="High-resolution interpolation (150 nodes)",
        )
    )

    # H4: 40 SPIDER nodes — tests minimum allowed resolution
    cases.append(
        Case(
            "H", 1.0, 0.325, num_levels=40,
            max_iters=20, max_time=1.0e4,
            description="Minimum resolution interpolation (40 nodes)",
        )
    )

    # H5: 2M, CMF=0.5 — validate sign conventions with intermediate mass
    cases.append(
        Case(
            "H", 2.0, 0.50,
            max_iters=20, max_time=1.0e4,
            description="Node ordering and sign conventions (2M, CMF=0.5)",
        )
    )

    return cases


def build_block_I() -> list[Case]:
    """Block I: Mesh blending and memory profiling (5 cases).

    Phase 2 runs with varying mesh_max_shift to test blend_mesh_files()
    and gc.collect() memory cleanup.
    """
    cases = []

    # I1: Tight blending (1M, max_shift=0.01) — should fire frequently
    cases.append(
        Case(
            "I", 1.0, 0.325,
            update_interval=50, mesh_max_shift=0.01,
            mesh_convergence_interval=5.0,
            mem_gb=5.0,
            description="Tight blending (max_shift=0.01, fires frequently)",
        )
    )

    # I2: Standard blending (3M) — large initial shift, 5% cap
    cases.append(
        Case(
            "I", 3.0, 0.325,
            update_interval=100, mesh_max_shift=0.05,
            mem_gb=5.0,
            description="Standard blending (3M, 5% cap on large shift)",
        )
    )

    # I3: No-blend control (1M, max_shift=0.99 = effectively disabled)
    cases.append(
        Case(
            "I", 1.0, 0.325,
            update_interval=50, mesh_max_shift=0.99,
            mem_gb=5.0,
            description="No-blend control (max_shift=0.99, effectively disabled)",
        )
    )

    # I4: Memory profiling (1M, frequent updates)
    cases.append(
        Case(
            "I", 1.0, 0.325,
            update_interval=50, mesh_max_shift=0.05,
            mem_gb=5.0,
            description="Memory profiling (1M, frequent Zalmoxis updates)",
        )
    )

    # I5: Memory profiling (3M, tighter updates than I2)
    cases.append(
        Case(
            "I", 3.0, 0.325,
            update_interval=50, mesh_max_shift=0.05,
            mem_gb=5.0,
            description="Memory profiling (3M, u=50yr, more frequent updates)",
        )
    )

    return cases


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
        Configuration dict with overrides applied.
    """
    cfg = deepcopy(config)

    # Output
    cfg["params"]["out"]["path"] = case_outdir
    cfg["params"]["stop"]["iters"]["maximum"] = case.max_iters
    cfg["params"]["stop"]["time"]["maximum"] = case.max_time

    # Structure
    cfg["struct"]["module"] = case.struct
    cfg["struct"]["mass_tot"] = case.mass
    cfg["struct"]["corefrac"] = round(case.corefrac, 6)
    cfg["struct"]["update_interval"] = case.update_interval
    cfg["struct"]["mesh_max_shift"] = case.mesh_max_shift
    cfg["struct"]["mesh_convergence_interval"] = case.mesh_convergence_interval

    # Zalmoxis
    cfg["struct"]["zalmoxis"]["coremassfrac"] = case.cmf
    cfg["struct"]["zalmoxis"]["max_center_pressure_guess"] = max_pressure_guess(
        case.mass
    )

    if case.temperature_mode is not None:
        cfg["struct"]["zalmoxis"]["temperature_mode"] = case.temperature_mode

    # SPIDER
    cfg["interior"]["spider"]["num_levels"] = case.num_levels
    cfg["interior"]["spider"]["ini_entropy"] = case.ini_entropy

    return cfg


def generate_slurm_script(
    outdir: Path, block: str, case_indices: list[int], mem_gb: float
) -> Path:
    """Generate SLURM job array script for a specific block.

    Parameters
    ----------
    outdir : Path
        Root output directory.
    block : str
        Block identifier (H or I).
    case_indices : list[int]
        0-based indices into the case_list.txt for this block.
    mem_gb : float
        Memory per CPU in GB.

    Returns
    -------
    Path
        Path to the generated SLURM script.
    """
    idx_str = ",".join(str(i) for i in case_indices)
    time_limit = "0-06:00:00" if block == "H" else "0-12:00:00"

    script = f"""\
#!/bin/bash
#SBATCH -J spider-val-v4-{block}
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
echo "SPIDER validation grid v4 — Block {block}"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Config:     $CFG"
echo "Memory:     {int(mem_gb)} GB"
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

    path = outdir / f"slurm_block_{block}.sh"
    path.write_text(script)
    path.chmod(0o755)
    return path


def main():
    """Generate Habrok validation grid v4."""
    parser = argparse.ArgumentParser(
        description="Generate Habrok validation grid v4 (Blocks H + I)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Root output directory",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=BASE_CONFIG,
        help=f"Base TOML config (default: {BASE_CONFIG})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all cases without generating files",
    )
    args = parser.parse_args()

    cases_H = build_block_H()
    cases_I = build_block_I()
    all_cases = cases_H + cases_I

    # Print summary
    print(f"Habrok validation grid v4: {len(all_cases)} cases\n")
    for block_label, cases in [("H — Mesh interpolation", cases_H),
                                ("I — Blending + memory", cases_I)]:
        print(f"  Block {block_label} ({len(cases)} cases)")
        for c in cases:
            extras = []
            if c.update_interval > 0:
                extras.append(f"P2 u={c.update_interval:.0f}yr")
            if c.num_levels != 60:
                extras.append(f"n={c.num_levels}")
            if abs(c.mesh_max_shift - 0.05) > 0.001:
                extras.append(f"max_shift={c.mesh_max_shift}")
            extra_str = f"  ({', '.join(extras)})" if extras else ""
            print(f"    {c.name:45s} {c.mass:4.1f}M CMF={c.cmf:<5}"
                  f" mem={c.mem_gb:.0f}G{extra_str}")
            print(f"      {c.description}")
    print()

    if args.list:
        return

    # Read base config
    base_path = args.base_config.resolve()
    if not base_path.exists():
        print(f"ERROR: Base config not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    with open(base_path, "rb") as f:
        base_config = tomllib.load(f)

    # Create output directories
    outdir = args.outdir.resolve()
    cfg_dir = outdir / "cfgs"
    log_dir = outdir / "logs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(base_path, outdir / "base_validation.toml")

    # Generate per-case configs
    case_paths = []
    h_indices = []
    i_indices = []

    for idx, case in enumerate(all_cases):
        case_outdir = str(outdir / "cases" / case.name)
        cfg = apply_overrides(base_config, case, case_outdir)

        cfg_path = cfg_dir / f"{case.name}.toml"
        with open(cfg_path, "w") as f:
            f.write(
                f"# Auto-generated config: {case.name}\n"
                f"# Block {case.block}: {case.description}\n"
                f"# Mass={case.mass} M_E, CMF={case.cmf}, "
                f"corefrac={case.corefrac:.4f}\n\n"
            )
            toml.dump(cfg, f)

        case_paths.append(str(cfg_path))
        if case.block == "H":
            h_indices.append(idx)
        else:
            i_indices.append(idx)

    # Write case list
    case_list_path = outdir / "case_list.txt"
    case_list_path.write_text("\n".join(case_paths) + "\n")

    # Generate separate SLURM scripts per block
    slurm_H = generate_slurm_script(outdir, "H", h_indices, mem_gb=3.0)
    slurm_I = generate_slurm_script(outdir, "I", i_indices, mem_gb=5.0)

    print(f"Generated {len(all_cases)} configs in {cfg_dir}/")
    print(f"Case list:        {case_list_path}")
    print(f"SLURM Block H:    {slurm_H}  (3G, {len(h_indices)} jobs)")
    print(f"SLURM Block I:    {slurm_I}  (5G, {len(i_indices)} jobs)")
    print()
    print("Next steps:")
    print("  1. Review SLURM scripts and adjust env vars")
    print(f"  2. Submit Block H: sbatch {slurm_H}")
    print(f"  3. Submit Block I: sbatch {slurm_I}")
    print("  4. After completion, run validate_mesh_interpolation.py on Block H outputs")


if __name__ == "__main__":
    main()
