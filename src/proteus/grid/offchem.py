# Offline atmospheric chemistry on the cases of a PROTEUS grid (`proteus offchem --grid`)
from __future__ import annotations

import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import toml

from proteus.config import read_config_object
from proteus.grid.manage import _grid_positive_int
from proteus.grid.slurm import array_layout, dispatch_script, max_array_size
from proteus.grid.summarise import status_category
from proteus.utils.coupler import get_proteus_directories, helpfile_path, vulcan_problem
from proteus.utils.helper import ReadStatus

log = logging.getLogger('fwl.' + __name__)


def grid_limits(grid: Path) -> dict:
    """Read `max_jobs`, `max_days` [days], `max_mem` [GB] and `cores_per_job` (default 1)
    from the grid file in ``grid``.

    Raises
    ------
    ValueError
        If ``grid`` has no copy.grid.toml, it lacks one of the first three keys, or
        `cores_per_job` is not a positive integer.
    """
    path = grid / 'copy.grid.toml'
    if not path.is_file():
        raise ValueError(f'{path} not found; the job limits are read from the grid file')
    table = toml.load(path)
    keys = ('max_jobs', 'max_days', 'max_mem')
    if missing := [k for k in keys if k not in table]:
        raise ValueError(f'{path} lacks {", ".join(missing)}')
    limits = {k: int(table[k]) if k == 'max_jobs' else float(table[k]) for k in keys}
    limits['cores_per_job'] = _grid_positive_int(table, 'cores_per_job', str(path))
    return limits


def find_cases(grid: Path) -> list[tuple[str, Path]]:
    """Cases of ``grid`` with a helpfile and a config in cfgs/, as (name, config) in order.

    A case still marked running is kept with a warning: Slurm leaves a case it stops so.
    """
    cases = []
    for case_dir in sorted(grid.glob('case_*')):
        if not os.path.isfile(helpfile_path(case_dir)):
            continue
        cfg = grid / 'cfgs' / f'{case_dir.name}.toml'
        if not cfg.is_file():
            log.warning('Skipping %s: it has a helpfile but no config in cfgs/', case_dir.name)
            continue
        if status_category(ReadStatus(case_dir)) == 'Running':
            log.warning('%s is marked running; its output may still change', case_dir.name)
        cases.append((case_dir.name, cfg))
    return cases


def check_cases(grid: Path, cases: list[tuple[str, Path]]) -> bool:
    """Check every case before any runs, and compile the VULCAN network the cases share.

    Returns whether the network was compiled here, so that the cases skip compiling it.

    Raises
    ------
    ValueError
        Naming each case without a chemistry module or whose config writes to another
        folder, or if the VULCAN cases that compile their network use different networks.
    EnvironmentError
        If VULCAN is missing or out of date.
    """
    configs = [read_config_object(cfg) for _, cfg in cases]
    problems = []
    for (name, cfg), config in zip(cases, configs):
        out = get_proteus_directories(outdir=config.params.out.path)['output']
        if not config.atmos_chem.module:
            problems.append(f'{name}: atmos_chem.module is none in {cfg}')
        elif os.path.realpath(out) != os.path.realpath(grid / name):
            problems.append(f'{name}: its config writes to {out}, not to this grid')
    if problems:
        raise ValueError('Cannot run the chemistry of the grid:\n  ' + '\n  '.join(problems))

    vulcan_cfgs = [c for c in configs if c.atmos_chem.module == 'vulcan']
    if not vulcan_cfgs:
        return False
    if problem := vulcan_problem(vulcan_cfgs[0]):
        raise EnvironmentError(problem)
    if not any(c.atmos_chem.vulcan.make_funs for c in vulcan_cfgs):
        return False  # every case uses the network compiled before, as its config asks
    from proteus.atmos_chem.vulcan import compile_network, network_settings

    networks = sorted({network_settings(c)[0] for c in vulcan_cfgs})
    if len(networks) > 1:
        raise ValueError(
            f'The VULCAN cases use the networks {", ".join(networks)}, but VULCAN holds one '
            'compiled network at a time; run them with `proteus offchem -c` instead'
        )
    log.info('Compiling the VULCAN network %s for all cases', compile_network(vulcan_cfgs[0]))
    return True


def _run_case(cfg: Path, flag: list[str], logfile: Path) -> int:
    """Run `proteus offchem` on one case, with its terminal output in ``logfile``."""
    with open(logfile, 'w') as f:
        cmd = ['proteus', 'offchem', '--config', str(cfg), *flag]
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode


def write_slurm(
    grid: Path, cases: list[tuple[str, Path]], limits: dict, flag: list[str]
) -> Path:
    """Write a Slurm job array that runs `proteus offchem` on each case, packed into jobs
    as the grid run packs its cases, with logs in logs/offchem/; return its path."""
    listing = grid / 'offchem_cases.txt'
    listing.write_text(''.join(f'{name}\n' for name, _ in cases))
    cores = limits['cores_per_job']
    jobs, waves = array_layout(len(cases), cores, max_array_size())
    throttle = min(max(1, limits['max_jobs'] // cores), jobs)
    log.info(
        '%d cases in %d jobs of %d cores, up to %d at a time', len(cases), jobs, cores, throttle
    )
    script = grid / 'slurm_offchem.sh'
    script.write_text(
        dispatch_script(
            cfgdir=str(grid / 'cfgs'),
            logdir=str(grid / 'logs' / 'offchem'),
            n_cases=len(cases),
            cores_per_job=cores,
            jobs=jobs,
            throttle=throttle,
            days=limits['max_days'] * waves,
            mem_per_case_mb=int(limits['max_mem'] * 1024),
            command=' '.join(['proteus', 'offchem', *flag, '--config']),
            case_list=str(listing),
            job_name='proteusoffchem.grid.array',
        )
    )
    return script


def offchem_grid(grid: Path, slurm: bool = False) -> list[str]:
    """Run offline chemistry on every case of ``grid`` that has a helpfile.

    Each case runs `proteus offchem` with its config in cfgs/, so with its own `[atmos_chem]`
    table. `max_jobs` of the grid file caps the cases at once; with ``slurm``, a job array
    script is written instead, packed by `cores_per_job`, with `max_days` and `max_mem`
    per case, as for the grid run.

    Returns the names of the cases whose chemistry failed (none with ``slurm``).
    """
    grid = Path(grid)
    limits = grid_limits(grid)
    cases = find_cases(grid)
    if not cases:
        raise ValueError(f'No case in {grid} has a helpfile and a config in cfgs/')
    flag = ['--network-compiled'] if check_cases(grid, cases) else []
    (grid / 'logs' / 'offchem').mkdir(parents=True, exist_ok=True)

    if slurm:
        log.info('Submit to Slurm with `sbatch %s`', write_slurm(grid, cases, limits, flag))
        if flag:
            log.warning('Compile no other VULCAN network until these jobs have finished')
        return []

    jobs = max(1, min(limits['max_jobs'], len(cases), os.cpu_count() or 1))
    log.info('Running offline chemistry on %d cases, %d at a time', len(cases), jobs)
    failed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        runs = {
            pool.submit(_run_case, cfg, flag, grid / 'logs' / 'offchem' / f'{name}.log'): name
            for name, cfg in cases
        }
        for i, run in enumerate(as_completed(runs)):
            code = run.result()
            if code != 0:
                failed.append(runs[run])
            log.info(
                '%d of %d: %s %s', i + 1, len(cases), runs[run], 'failed' if code else 'done'
            )
    log.info('Chemistry finished on %d of %d cases', len(cases) - len(failed), len(cases))
    return sorted(failed)
