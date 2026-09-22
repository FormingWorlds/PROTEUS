"""Record and compare Aragog interior coupling steps in PROTEUS.

Runs one coupling step of PROTEUS with Aragog interior energetics,
recording all SolverOutput fields, all helpfile columns, and repository
commit hashes to an .npz archive. Provides bitwise array comparison
between recordings to detect numerical regressions across solver changes.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from proteus import Proteus


def get_git_commit(repo_path: Path | str) -> str:
    """Return git commit hash of a repository directory.

    Parameters
    ----------
    repo_path : Path or str
        Path to git working directory.

    Returns
    -------
    str
        HEAD commit hash, or 'unknown' if unavailable.
    """
    try:
        res = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return 'unknown'


def get_aragog_repo_path() -> Path:
    """Return the repository directory for the installed aragog package.

    Returns
    -------
    Path
        Root path of the aragog repository.
    """
    import aragog

    pkg_dir = Path(aragog.__file__).resolve().parent
    return pkg_dir.parent.parent


def record_step(config_path: Path | str, output_npz: Path | str) -> None:
    """Run one coupling step and record all solver outputs to .npz.

    Parameters
    ----------
    config_path : Path or str
        Path to TOML configuration file.
    output_npz : Path or str
        Output path for the .npz archive.
    """
    config_path = Path(config_path).resolve()
    output_npz = Path(output_npz).resolve()
    proteus_root = Path(__file__).resolve().parent.parent
    aragog_root = get_aragog_repo_path()

    proteus_commit = get_git_commit(proteus_root)
    aragog_commit = get_git_commit(aragog_root)

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = Proteus(config_path=config_path)
        runner.config.params.out.path = str(Path(tmpdir) / 'compare_run')
        runner.config.params.out.plot_mod = None
        runner.config.params.out.write_mod = 0
        runner.config.params.out.archive_mod = None
        runner.config.params.stop.time.minimum = 0.0
        runner.config.params.stop.time.maximum = 1e2
        runner.config.params.dt.initial = 1e2
        runner.config.params.stop.iters.enabled = True
        runner.config.params.stop.iters.minimum = 0
        runner.config.params.stop.iters.maximum = 1
        runner.init_directories()

        runner.start(resume=False, offline=True)

        out = getattr(runner.interior_o, 'last_solver_output', None)
        if out is None:
            raise RuntimeError('No last_solver_output found on runner.interior_o')

        data_dict: dict[str, np.ndarray] = {
            'meta_proteus_commit': np.array(proteus_commit),
            'meta_aragog_commit': np.array(aragog_commit),
            'meta_config_path': np.array(str(config_path)),
        }

        for f in dataclasses.fields(out):
            val = getattr(out, f.name)
            data_dict[f'solver_output_{f.name}'] = np.asarray(val)

        if runner.hf_all is not None:
            for col in runner.hf_all.columns:
                data_dict[f'hf_all_{col}'] = np.asarray(runner.hf_all[col].values)

        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_npz, **data_dict)
        print(f'Recorded {len(data_dict)} keys to {output_npz}')
        print(f'  PROTEUS commit: {proteus_commit}')
        print(f'  Aragog commit:  {aragog_commit}')


def compare_recordings(ref_path: Path | str, test_path: Path | str) -> bool:
    """Compare two recorded .npz step archives bitwise.

    Parameters
    ----------
    ref_path : Path or str
        Path to reference .npz file.
    test_path : Path or str
        Path to test .npz file.

    Returns
    -------
    bool
        True if all fields are identical; False otherwise.
    """
    ref = np.load(ref_path)
    test = np.load(test_path)

    print(f'Reference: {ref_path}')
    print(f'  PROTEUS: {ref.get("meta_proteus_commit", "unknown")}')
    print(f'  Aragog:  {ref.get("meta_aragog_commit", "unknown")}')
    print(f'Test:      {test_path}')
    print(f'  PROTEUS: {test.get("meta_proteus_commit", "unknown")}')
    print(f'  Aragog:  {test.get("meta_aragog_commit", "unknown")}')

    all_keys = sorted(set(ref.files) | set(test.files))
    missing_in_test = [k for k in ref.files if k not in test.files]
    missing_in_ref = [k for k in test.files if k not in ref.files]

    if missing_in_test:
        print(f'ERROR: Keys missing in test file: {missing_in_test}')
    if missing_in_ref:
        print(f'WARNING: Additional keys in test file: {missing_in_ref}')

    mismatches = 0
    checked_keys = 0

    IGNORED_KEYS = {'hf_all_runtime'}

    for k in all_keys:
        if k.startswith('meta_') or k in IGNORED_KEYS:
            continue
        if k not in ref.files or k not in test.files:
            mismatches += 1
            continue

        a = ref[k]
        b = test[k]
        checked_keys += 1

        is_float = np.issubdtype(a.dtype, np.inexact)
        equal = np.array_equal(a, b, equal_nan=True) if is_float else np.array_equal(a, b)
        if not equal:
            mismatches += 1
            nan_a = np.isnan(a) if is_float else np.zeros_like(a, dtype=bool)
            nan_b = np.isnan(b) if is_float else np.zeros_like(b, dtype=bool)
            if not np.array_equal(nan_a, nan_b):
                print(f'  MISMATCH {k}: NaN pattern differs')
            else:
                valid = ~nan_a
                diff = np.abs(a[valid] - b[valid])
                denom = np.maximum(np.abs(a[valid]), 1e-30)
                rel_diff = np.max(diff / denom) if len(diff) > 0 else 0.0
                print(f'  MISMATCH {k}: max relative diff = {rel_diff:.4e}')

    if mismatches == 0:
        print(f'PASSED: All {checked_keys} data fields match bitwise.')
        return True

    print(f'FAILED: {mismatches} mismatched fields out of {checked_keys}.')
    return False


def main() -> int:
    """CLI entry point for compare_interior_step."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Config TOML to run and record')
    parser.add_argument('--output', type=Path, help='Output path for recorded .npz')
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('REF', 'TEST'),
        type=Path,
        help='Compare two .npz recording files',
    )
    parser.add_argument(
        '--record-both',
        action='store_true',
        help='Record both aragog_compare_off.toml and aragog_compare_on.toml',
    )
    args = parser.parse_args()

    proteus_root = Path(__file__).resolve().parent.parent

    if args.compare:
        match = compare_recordings(args.compare[0], args.compare[1])
        return 0 if match else 1

    if args.record_both:
        off_toml = proteus_root / 'tests/integration/aragog_compare_off.toml'
        on_toml = proteus_root / 'tests/integration/aragog_compare_on.toml'
        off_npz = proteus_root / 'tests/integration/aragog_compare_off.npz'
        on_npz = proteus_root / 'tests/integration/aragog_compare_on.npz'

        print('Recording rheology-off case...')
        record_step(off_toml, off_npz)
        print('Recording rheology-on case...')
        record_step(on_toml, on_npz)
        return 0

    if args.config and args.output:
        record_step(args.config, args.output)
        return 0

    parser.print_help()
    return 1


if __name__ == '__main__':
    sys.exit(main())
