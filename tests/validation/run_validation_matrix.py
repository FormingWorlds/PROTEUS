#!/usr/bin/env python
"""Run the extended validation matrix for Zalmoxis-SPIDER coupling.

Test matrix: 3 masses x 5 CMFs x 2 structure modes = 30 runs,
plus 3 Phase 2 feedback runs = 33 total.

Each run evolves a magma ocean planet for up to 1 Myr (or until
crystallization) using AGNI + SPIDER + MORS + CALLIOPE + ZEPHYRUS.

Usage
-----
    python run_validation_matrix.py                         # run all 33 cases
    python run_validation_matrix.py --mass 1.0              # only 1 M_earth
    python run_validation_matrix.py --mass 1 --cmf 0.325    # single case
    python run_validation_matrix.py --struct zalmoxis       # Zalmoxis only
    python run_validation_matrix.py --phase2                # Phase 2 subset
    python run_validation_matrix.py --list                  # list without running
    python run_validation_matrix.py --resume                # skip completed cases
    python run_validation_matrix.py --workers 4             # run 4 cases in parallel
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROTEUS_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = Path(__file__).resolve().parent / 'base_validation.toml'

# ── Test matrix parameters ──────────────────────────────────

MASSES = [1.0, 3.0, 5.0]  # M_earth
CMFS = [0.05, 0.10, 0.325, 0.50, 0.80]
STRUCT_MODES = ['self', 'zalmoxis']

# Densities for CMF -> corefrac conversion (uniform two-layer approximation)
RHO_CORE = 10738.33  # kg/m^3 (iron core, from PROTEUS default)
RHO_MANTLE = 4000.0  # kg/m^3 (approximate MgSiO3 mantle)


def cmf_to_corefrac(cmf: float) -> float:
    """Convert core mass fraction to core radius fraction.

    Uses a uniform two-layer model: rho_core and rho_mantle are constant.
    This is approximate; the real AW profile has depth-dependent density.

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


# ── Case definition ─────────────────────────────────────────


@dataclass
class Case:
    """A single validation run configuration."""

    mass: float  # Planet mass [M_earth]
    cmf: float  # Core mass fraction
    struct: str  # "self" (AW) or "zalmoxis"
    update_interval: float = 0.0  # Structure update interval [yr]

    @property
    def name(self) -> str:
        """Unique case name for output directory."""
        tag = 'AW' if self.struct == 'self' else 'ZAL'
        suffix = '_P2' if self.update_interval > 0 else ''
        return f'M{self.mass}_CMF{self.cmf}_{tag}{suffix}'

    @property
    def corefrac(self) -> float:
        """Approximate core radius fraction for AW mode."""
        return cmf_to_corefrac(self.cmf)


def build_matrix() -> list[Case]:
    """Build the full test matrix.

    Returns
    -------
    list[Case]
        30 main cases + 3 Phase 2 feedback cases = 33 total.
    """
    cases = []

    # Main matrix: 3 masses x 5 CMFs x 2 structure modes
    for mass in MASSES:
        for cmf in CMFS:
            for struct in STRUCT_MODES:
                cases.append(Case(mass=mass, cmf=cmf, struct=struct))

    # Phase 2 feedback: Earth-like CMF for each mass, Zalmoxis with updates
    for mass in MASSES:
        cases.append(Case(mass=mass, cmf=0.325, struct='zalmoxis', update_interval=100.0))

    return cases


# ── Runner ───────────────────────────────────────────────────


def run_case(case: Case, outdir: Path, resume: bool = False) -> bool:
    """Run a single validation case.

    Parameters
    ----------
    case : Case
        Configuration for this run.
    outdir : Path
        Root output directory.
    resume : bool
        If True, skip cases that already have a helpfile.

    Returns
    -------
    bool
        True if the case completed successfully.
    """
    case_dir = outdir / case.name
    helpfile = case_dir / 'runtime_helpfile.csv'

    if resume and helpfile.exists():
        print(f'  SKIP {case.name}: already completed')
        return True

    print(f'  RUN  {case.name} ...', flush=True)
    t0 = time.time()

    try:
        from proteus import Proteus

        runner = Proteus(config_path=BASE_CONFIG)

        # Per-case overrides
        runner.config.params.out.path = str(case_dir)
        runner.config.struct.module = case.struct
        runner.config.struct.mass_tot = case.mass
        runner.config.struct.corefrac = case.corefrac
        runner.config.struct.update_interval = case.update_interval

        if case.struct == 'zalmoxis':
            runner.config.struct.zalmoxis.coremassfrac = case.cmf
            runner.config.struct.zalmoxis.weight_iron_frac = case.cmf

            # Increase central pressure guess for massive/dense planets
            if case.mass >= 3.0:
                runner.config.struct.zalmoxis.max_center_pressure_guess = 5.0e12

        # Re-init directories with the new output path
        runner.init_directories()

        # Run the simulation
        runner.start(resume=False, offline=False)

        dt = time.time() - t0
        n_steps = len(runner.hf_all) if runner.hf_all is not None else 0
        print(f'  DONE {case.name} ({dt:.0f}s, {n_steps} steps)')
        return True

    except Exception as e:
        dt = time.time() - t0
        print(f'  FAIL {case.name} ({dt:.0f}s): {e}')

        # Record the error for post-mortem
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / 'ERROR.txt').write_text(f'{type(e).__name__}: {e}\n')
        return False


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Run Zalmoxis-SPIDER extended validation matrix'
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=PROTEUS_ROOT / 'output' / 'validation',
        help='Root output directory (default: output/validation)',
    )
    parser.add_argument('--mass', type=float, help='Filter by planet mass [M_earth]')
    parser.add_argument('--cmf', type=float, help='Filter by core mass fraction')
    parser.add_argument(
        '--struct',
        choices=['self', 'zalmoxis'],
        help='Filter by structure mode',
    )
    parser.add_argument(
        '--phase2',
        action='store_true',
        help='Only run Phase 2 feedback cases',
    )
    parser.add_argument('--list', action='store_true', help='List all cases without running')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip cases that already have output',
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='Number of cases to run in parallel (default: 1)',
    )
    args = parser.parse_args()

    # Build and filter the matrix
    cases = build_matrix()

    if args.mass is not None:
        cases = [c for c in cases if np.isclose(c.mass, args.mass)]
    if args.cmf is not None:
        cases = [c for c in cases if np.isclose(c.cmf, args.cmf)]
    if args.struct:
        cases = [c for c in cases if c.struct == args.struct]
    if args.phase2:
        cases = [c for c in cases if c.update_interval > 0]

    if not cases:
        print('No cases match the filter criteria.')
        sys.exit(1)

    # Print the matrix
    print(f'Validation matrix: {len(cases)} cases')
    print(f'Output directory:  {args.outdir}')
    print()
    for c in cases:
        extra = f'corefrac={c.corefrac:.3f}' if c.struct == 'self' else ''
        p2 = ' [Phase 2]' if c.update_interval > 0 else ''
        print(f'  {c.name:30s}  {c.mass} M_earth  CMF={c.cmf:<6}  {c.struct:10s} {extra}{p2}')
    print()

    if args.list:
        return

    # Run all cases
    args.outdir.mkdir(parents=True, exist_ok=True)

    results = {}
    t_start = time.time()
    n_workers = max(1, args.workers)

    if n_workers == 1:
        # Sequential execution
        for i, c in enumerate(cases, 1):
            print(f'[{i}/{len(cases)}]')
            results[c.name] = run_case(c, args.outdir, resume=args.resume)
            print()
    else:
        # Parallel execution
        print(f'Running with {n_workers} parallel workers\n')
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(run_case, c, args.outdir, args.resume): c for c in cases
            }
            for future in as_completed(futures):
                c = futures[future]
                try:
                    results[c.name] = future.result()
                except Exception as e:
                    print(f'  CRASH {c.name}: {e}')
                    results[c.name] = False
        print()

    # Summary
    dt_total = time.time() - t_start
    n_pass = sum(results.values())
    n_total = len(results)
    print(f'{"=" * 60}')
    print(f'Results: {n_pass}/{n_total} passed ({dt_total / 3600:.1f} hours)')
    print()
    for name, ok in sorted(results.items()):
        print(f'  {"PASS" if ok else "FAIL"}  {name}')

    # Write summary to file
    summary_path = args.outdir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f'Validation run: {n_pass}/{n_total} passed\n')
        f.write(f'Total time: {dt_total / 3600:.1f} hours\n\n')
        for name, ok in sorted(results.items()):
            f.write(f'{"PASS" if ok else "FAIL"}  {name}\n')
    print(f'\nSummary written to {summary_path}')


if __name__ == '__main__':
    main()
