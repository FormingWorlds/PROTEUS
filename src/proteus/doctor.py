"""PROTEUS installation diagnostics.

Structured check system for ``proteus doctor``. Each check returns a
typed result (pass/warn/fail) with a human-readable message and an
optional fix command. The ``proteus update`` command collects all fix
commands and offers to run them.
"""

from __future__ import annotations

import datetime
import importlib.metadata
import importlib.util
import io
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
import traceback
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from urllib.parse import unquote, urlparse

import click
from packaging.requirements import Requirement
from packaging.version import Version

from proteus.utils.coupler import (
    _get_agni_version,
    _get_socrates_version,
    get_proteus_directories,
)

# ─── Check result types ──────────────────────────────────────────────

PASS = 'pass'
WARN = 'warn'
FAIL = 'fail'

_STYLE = {
    PASS: {'fg': 'green'},
    WARN: {'fg': 'yellow'},
    FAIL: {'fg': 'red'},
}

_ICON = {PASS: 'ok', WARN: 'warn', FAIL: 'FAIL'}


@dataclass
class CheckResult:
    """One diagnostic check result."""

    name: str
    category: str
    status: str
    message: str
    fix_cmd: str | None = None
    # When False, fix_cmd is human advice (e.g. "export VAR=<path>"), not a
    # runnable command, so `proteus update` shows it but does not execute it.
    auto_fixable: bool = True

    def echo(self):
        icon = click.style(f'[{_ICON[self.status]}]', **_STYLE[self.status])
        name = click.style(self.name, bold=True)
        click.echo(f'  {icon} {name}: {self.message}')
        if self.fix_cmd and self.status != PASS:
            click.echo(f'       fix: {click.style(self.fix_cmd, fg="cyan")}')

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'category': self.category,
            'status': self.status,
            'message': self.message,
            'fix_cmd': self.fix_cmd,
            'auto_fixable': self.auto_fixable,
        }


# ─── Helpers ──────────────────────────────────────────────────────────


def _repo_root() -> Path:
    """Find the PROTEUS repo root by walking up from this file."""
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / 'pyproject.toml').is_file():
            return candidate
    return here


def _read_pyproject() -> dict:
    """Parse pyproject.toml from the repo root."""
    path = _repo_root() / 'pyproject.toml'
    if not path.is_file():
        return {}
    return tomllib.loads(path.read_text())


def _module_pins() -> dict[str, dict]:
    """Return [tool.proteus.modules] from pyproject.toml."""
    cfg = _read_pyproject()
    return cfg.get('tool', {}).get('proteus', {}).get('modules', {})


def _dependency_specs() -> dict[str, Requirement | None]:
    """Return [project] dependencies as {name: Requirement|None} for FWL packages.

    PEP 508 URL requirements (``name @ git+https://...``) parse into a
    Requirement with an empty specifier set. We still record them so that
    ``check_python_package`` knows the package is tracked (and can report
    "installed" vs "not installed") even when no version bound applies.
    """
    cfg = _read_pyproject()
    deps = cfg.get('project', {}).get('dependencies', [])
    result: dict[str, Requirement | None] = {}
    for dep_str in deps:
        try:
            req = Requirement(dep_str)
        except Exception:
            # PEP 508 URL form may fail in older packaging versions.
            # Extract the package name from the "name @ url" form.
            name = dep_str.split('@')[0].strip().replace('_', '-').lower()
            if 'fwl-' in name:
                result[name] = None
            continue
        if 'fwl-' in req.name:
            result[req.name] = req
    return result


def _editable_checkout_path(dist_name: str) -> str | None:
    """Return the local path of an editable install, or None."""
    try:
        dist = importlib.metadata.distribution(dist_name)
    except PackageNotFoundError:
        return None
    raw = dist.read_text('direct_url.json')
    if not raw:
        return None
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not info.get('dir_info', {}).get('editable'):
        return None
    url = info.get('url', '')
    if not url.startswith('file://'):
        return None
    return unquote(urlparse(url).path)


def _imported_package_dir(dist_name: str) -> str | None:
    """Return the directory of the package that ``import`` would actually load
    for a distribution, or None if it cannot be resolved.

    Uses ``importlib.util.find_spec`` so the module is located on ``sys.path``
    without being imported (no side effects, no heavy submodule init). This
    lets the editable-install annotation be cross-checked against the package
    that is really on the path: a later ``pip install <pin>`` can shadow an
    editable sibling while leaving its ``direct_url.json`` metadata in place.
    """
    import_name = None
    try:
        top = importlib.metadata.distribution(dist_name).read_text('top_level.txt')
        if top:
            import_name = top.strip().splitlines()[0].strip()
    except (PackageNotFoundError, OSError):
        return None
    if not import_name:
        # Fall back to the reverse import-name -> distribution map.
        try:
            for imp, dists in importlib.metadata.packages_distributions().items():
                if dist_name in dists:
                    import_name = imp
                    break
        except Exception:
            return None
    if not import_name:
        return None
    try:
        spec = importlib.util.find_spec(import_name)
    except (ImportError, ValueError, ModuleNotFoundError):
        return None
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(spec.origin)


def _git_head(path: str) -> str | None:
    """Return the full HEAD commit hash for a git checkout, or None."""
    try:
        return subprocess.check_output(
            ['git', '-C', path, 'rev-parse', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_short_head(path: str) -> str | None:
    """Return the short HEAD commit hash for a git checkout, or None."""
    try:
        return subprocess.check_output(
            ['git', '-C', path, 'rev-parse', '--short', 'HEAD'],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _git_dirty(path: str) -> bool | None:
    """Return True if the git working tree has uncommitted changes, False if it
    is clean, or None if the git status could not be determined (not a git
    repository, git missing, or a git error)."""
    try:
        out = subprocess.check_output(
            ['git', '-C', path, 'status', '--porcelain'],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(out.strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _julia_version() -> str | None:
    """Return the installed Julia version string, or None."""
    try:
        out = subprocess.check_output(
            ['julia', '--version'], text=True, stderr=subprocess.DEVNULL
        )
        return out.strip().split()[-1]
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _julia_version_at(exe: str) -> str | None:
    """Return the Julia version reported by a specific julia binary, or None.

    Used to report the Julia that juliacall is bound to (via
    ``PYTHON_JULIAPKG_EXE``), which can differ from the one on ``PATH``.
    """
    try:
        out = subprocess.check_output([exe, '--version'], text=True, stderr=subprocess.DEVNULL)
        return out.strip().split()[-1]
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return None


def _conda_build_lines() -> list[str]:
    """Return conda build strings for the HDF5/netCDF/MPI packages.

    These builds drive the libmpi symbol clash that breaks AGNI. Returns an
    empty list when conda is unavailable or none of the packages are present.
    """
    try:
        out = subprocess.check_output(['conda', 'list'], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return []
    wanted = ('hdf5', 'libnetcdf', 'netcdf4', 'mpich', 'openmpi', 'libmpi')
    lines = []
    for line in out.splitlines():
        fields = line.split()
        if fields and fields[0] in wanted:
            lines.append('  ' + ' '.join(fields))
    return lines


# ─── Check implementations ───────────────────────────────────────────


def check_env_var(
    name: str, *, validate_path: bool = True, required_file: str | None = None
) -> CheckResult:
    """Check that an environment variable is set and its path is valid."""
    val = os.environ.get(name)
    if not val:
        return CheckResult(
            name=name,
            category='environment',
            status=FAIL,
            message='not set',
            fix_cmd=f'export {name}=<path>  # add to your shell rc file',
            auto_fixable=False,
        )
    if validate_path and not os.path.exists(val):
        return CheckResult(
            name=name,
            category='environment',
            status=WARN,
            message=f'set to {val} but path does not exist',
        )
    if required_file and not os.path.isfile(os.path.join(val, required_file)):
        return CheckResult(
            name=name,
            category='environment',
            status=WARN,
            message=f'path exists but {required_file} is missing',
        )
    return CheckResult(
        name=name,
        category='environment',
        status=PASS,
        message=val,
    )


def check_fwl_data() -> list[CheckResult]:
    """Check FWL_DATA contents for required data sets."""
    results = []
    fwl = os.environ.get('FWL_DATA')
    if not fwl or not os.path.isdir(fwl):
        return results

    expected = {
        'spectral_files': 'proteus get spectral',
        'stellar_spectra': 'proteus get stellar',
    }
    for subdir, fix in expected.items():
        path = os.path.join(fwl, subdir)
        if os.path.isdir(path) and os.listdir(path):
            results.append(
                CheckResult(
                    name=f'FWL_DATA/{subdir}',
                    category='data',
                    status=PASS,
                    message='present',
                )
            )
        else:
            results.append(
                CheckResult(
                    name=f'FWL_DATA/{subdir}',
                    category='data',
                    status=WARN,
                    message='missing or empty',
                    fix_cmd=fix,
                )
            )
    return results


def check_julia() -> CheckResult:
    """Check Julia version."""
    ver = _julia_version()
    if ver is None:
        return CheckResult(
            name='julia',
            category='environment',
            status=FAIL,
            message='not found on PATH',
            fix_cmd='curl -fsSL https://install.julialang.org | sh',
        )
    parts = ver.split('.')
    if len(parts) >= 2 and parts[0] == '1' and parts[1] in ('11', '12'):
        return CheckResult(
            name='julia',
            category='environment',
            status=PASS,
            message=f'{ver}',
        )
    return CheckResult(
        name='julia',
        category='environment',
        status=WARN,
        message=f'{ver} (1.11.x or 1.12.x required)',
        fix_cmd='juliaup add 1.12 && juliaup default 1.12',
    )


def check_python_package(name: str, spec: Requirement | None) -> CheckResult:
    """Check a Python package against the pyproject.toml version spec."""
    try:
        installed = Version(importlib.metadata.version(name))
    except PackageNotFoundError:
        fix = f'pip install {name}'
        return CheckResult(
            name=name,
            category='versions',
            status=FAIL,
            message='not installed',
            fix_cmd=fix,
        )

    # Build status message with editable annotation
    checkout = _editable_checkout_path(name)
    annotation = ''
    if checkout:
        short = _git_short_head(checkout)
        dirty = _git_dirty(checkout)
        base = os.path.basename(checkout.rstrip('/'))
        if dirty is None:
            marker = ' (git status unknown)'
        elif dirty:
            marker = ' (dirty)'
        else:
            marker = ''
        annotation = f' [editable @ {base} -> {short}{marker}]'
        # Cross-check that the editable checkout is the package actually on
        # sys.path. A later `pip install <pin>` can shadow the editable sibling
        # while leaving its direct_url metadata in place, so the checkout
        # annotation would otherwise describe a tree that is not imported.
        imported_dir = _imported_package_dir(name)
        if imported_dir:
            real_imported = os.path.realpath(imported_dir)
            real_checkout = os.path.realpath(checkout)
            under_checkout = real_imported == real_checkout or real_imported.startswith(
                real_checkout + os.sep
            )
            if not under_checkout:
                annotation = (
                    f' [editable record @ {base}{marker}, but importing from {imported_dir}]'
                )

    # Check against pyproject.toml minimum bound. Allow pre-releases: editable
    # installs carry setuptools-scm dev versions (e.g. 26.6.1.post1.dev8) that
    # are pre-releases in PEP 440 terms but still satisfy a >= bound; the default
    # SpecifierSet membership would reject them and report a spurious failure.
    if spec and spec.specifier:
        if not spec.specifier.contains(installed, prereleases=True):
            fix = f'pip install -U "{spec}"'
            if checkout:
                fix = f'cd {shlex.quote(checkout)} && git pull && pip install -e .'
            return CheckResult(
                name=name,
                category='versions',
                status=FAIL,
                message=f'{installed} (requires {spec.specifier}){annotation}',
                fix_cmd=fix,
            )

    # Package satisfies the pin
    return CheckResult(
        name=name,
        category='versions',
        status=PASS,
        message=f'{installed}{annotation}',
    )


def check_git_module(name: str, dirs: dict) -> CheckResult:
    """Check a git-pinned module (AGNI, SOCRATES) against pyproject.toml ref."""
    pins = _module_pins()
    pin = pins.get(name.lower(), {})
    pinned_ref = pin.get('ref')

    # Determine the checkout path
    dir_key = name.lower()
    if dir_key == 'socrates':
        path = os.environ.get('RAD_DIR', '')
    else:
        path = dirs.get(dir_key, '')

    if not path or not os.path.isdir(path):
        setup_script = f'bash tools/get_{name.lower()}.sh'
        return CheckResult(
            name=name,
            category='versions',
            status=FAIL,
            message='not installed',
            fix_cmd=setup_script,
        )

    # Get current version
    try:
        if name == 'AGNI':
            ver = _get_agni_version(dirs)
        elif name == 'SOCRATES':
            ver = _get_socrates_version()
        else:
            ver = '?'
    except Exception:
        ver = '?'

    # Check HEAD against pinned ref
    head = _git_head(path)
    if pinned_ref and head:
        if head == pinned_ref:
            return CheckResult(
                name=name,
                category='versions',
                status=PASS,
                message=f'{ver} ({head[:8]})',
            )
        else:
            return CheckResult(
                name=name,
                category='versions',
                status=WARN,
                message=(f'{ver} ({head[:8]}) differs from pin ({pinned_ref[:8]})'),
                fix_cmd=f'bash tools/get_{name.lower()}.sh',
            )

    return CheckResult(
        name=name,
        category='versions',
        status=PASS,
        message=f'{ver}',
    )


# ─── Main entry points ──────────────────────────────────────────────


ENVIRONMENT_VARS = [
    ('FWL_DATA', True, None),
    ('RAD_DIR', True, 'bin/radlib.a'),
    ('FC_DIR', True, None),
    ('PYTHON_JULIAPKG_EXE', False, None),
]

# Mandatory Python packages checked by `proteus doctor`. The optional
# backends (fwl-vulcan, atmodeller) are deliberately excluded: they are
# not required for a standard run, so doctor must not report them as
# missing when a user has not installed them.
PYTHON_PACKAGES = [
    'fwl-proteus',
    'fwl-aragog',
    'fwl-calliope',
    'fwl-janus',
    'fwl-mors',
    'fwl-zephyrus',
    'fwl-zalmoxis',
]

GIT_MODULES = ['AGNI', 'SOCRATES']


def run_all_checks() -> list[CheckResult]:
    """Run all diagnostic checks. Returns a flat list of results.

    Individual check failures are caught so one broken check does not
    prevent the rest from running.
    """
    results: list[CheckResult] = []

    try:
        dirs = get_proteus_directories()
    except Exception as exc:
        dirs = {}
        results.append(
            CheckResult(
                name='proteus-dirs',
                category='environment',
                status=FAIL,
                message=f'Cannot resolve PROTEUS directories: {exc}',
            )
        )

    specs = _dependency_specs()

    # Environment
    for var, validate_path, req_file in ENVIRONMENT_VARS:
        try:
            results.append(
                check_env_var(var, validate_path=validate_path, required_file=req_file)
            )
        except Exception as exc:
            results.append(
                CheckResult(
                    name=var,
                    category='environment',
                    status=FAIL,
                    message=f'check error: {exc}',
                )
            )
    try:
        results.append(check_julia())
    except Exception as exc:
        results.append(
            CheckResult(
                name='julia',
                category='environment',
                status=FAIL,
                message=f'check error: {exc}',
            )
        )

    # Data
    try:
        results.extend(check_fwl_data())
    except Exception as exc:
        results.append(
            CheckResult(
                name='FWL_DATA',
                category='data',
                status=FAIL,
                message=f'check error: {exc}',
            )
        )

    # Python package versions (against pyproject.toml pins)
    for pkg in PYTHON_PACKAGES:
        try:
            results.append(check_python_package(pkg, specs.get(pkg)))
        except Exception as exc:
            results.append(
                CheckResult(
                    name=pkg,
                    category='versions',
                    status=FAIL,
                    message=f'check error: {exc}',
                )
            )

    # Git module versions (against pyproject.toml commit pins)
    for mod in GIT_MODULES:
        try:
            results.append(check_git_module(mod, dirs))
        except Exception as exc:
            results.append(
                CheckResult(
                    name=mod,
                    category='versions',
                    status=FAIL,
                    message=f'check error: {exc}',
                )
            )

    return results


# ─── Run logging and support prompt ──────────────────────────────────

_SUPPORT_EMAIL = 'proteus_dev@formingworlds.space'
_ISSUES_URL = 'https://github.com/FormingWorlds/PROTEUS/issues'
_DISCUSSIONS_URL = 'https://github.com/orgs/FormingWorlds/discussions'
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


class _Tee(io.TextIOBase):
    """Mirror text writes to two streams.

    ``isatty`` is delegated to the primary stream so click keeps emitting
    colour to the terminal while the mirror captures the same text for the log.
    """

    def __init__(self, primary, mirror):
        self._primary = primary
        self._mirror = mirror

    def write(self, s: str) -> int:
        self._primary.write(s)
        self._mirror.write(s)
        return len(s)

    def flush(self):
        self._primary.flush()
        self._mirror.flush()

    def isatty(self) -> bool:
        return self._primary.isatty()


# Keep this list roughly in step with the package grep in the bash
# ``collect_env_info`` in install.sh. jax/jaxlib/equinox pin the Aragog solver
# stack; netcdf4/h5py surface the conda HDF5/MPI clash that breaks AGNI.
_ENV_PACKAGES = (
    'fwl-proteus',
    'fwl-mors',
    'fwl-janus',
    'fwl-calliope',
    'fwl-zephyrus',
    'fwl-aragog',
    'fwl-zalmoxis',
    'fwl-vulcan',
    'juliacall',
    'juliapkg',
    'jax',
    'jaxlib',
    'equinox',
    'netcdf4',
    'h5py',
)
_ENV_VARS = (
    'FWL_DATA',
    'RAD_DIR',
    'FC_DIR',
    'PETSC_DIR',
    'PYTHON_JULIAPKG_EXE',
    'PYTHON_JULIACALL_BINDIR',
    'CONDA_DEFAULT_ENV',
    'CONDA_PREFIX',
)
# Variables whose value is a filesystem path; the report flags whether it exists.
_ENV_PATH_VARS = frozenset(
    {'FWL_DATA', 'RAD_DIR', 'FC_DIR', 'PETSC_DIR', 'PYTHON_JULIAPKG_EXE'}
)


def _collect_environment_info() -> str:
    """Gather machine and environment details for the failure log.

    The block is written into the log so that the log file alone is enough to
    diagnose an install or environment problem, without a back-and-forth. It
    mirrors the bash ``collect_env_info`` in install.sh; keep the two roughly
    in step when adding a variable or package.

    Returns
    -------
    str
        A plain-text, multi-line environment report. This function never
        raises: any collection error is recorded in the report instead.
    """
    lines = ['=== Environment (auto-collected for debugging) ===']
    try:
        lines.append(f'timestamp: {datetime.datetime.now().isoformat(timespec="seconds")}')
        lines.append(f'platform: {platform.platform()}')
        lines.append(f'machine: {platform.machine()}')
        lines.append(f'python: {platform.python_version()} ({sys.executable})')

        lines.append(f'julia (on PATH): {_julia_version() or "(not found)"}')
        # The Julia juliacall actually uses can differ from the PATH one, and
        # an import failure depends on the bound one, so report it directly.
        bound_exe = os.environ.get('PYTHON_JULIAPKG_EXE')
        if bound_exe:
            bound = _julia_version_at(bound_exe) or '(not runnable)'
            lines.append(f'julia (juliacall-bound): {bound}')

        for label, fn in (('socrates', _get_socrates_version), ('agni', _get_agni_version)):
            try:
                lines.append(f'{label}: {fn() or "(unknown)"}')
            except Exception:  # noqa: BLE001 - a probe failure must not abort
                lines.append(f'{label}: (unknown)')

        lines.append('environment variables:')
        for var in _ENV_VARS:
            val = os.environ.get(var)
            if val is None:
                lines.append(f'  {var}=(unset)')
            elif var in _ENV_PATH_VARS and not os.path.exists(val):
                lines.append(f'  {var}={val}  (MISSING)')
            else:
                lines.append(f'  {var}={val}')

        lines.append('installed package versions:')
        for pkg in _ENV_PACKAGES:
            try:
                ver = importlib.metadata.version(pkg)
            except PackageNotFoundError:
                ver = '(not installed)'
            except Exception as exc:  # noqa: BLE001 - a broken dist must not abort
                ver = f'(version unavailable: {exc!r})'
            lines.append(f'  {pkg}: {ver}')

        conda_lines = _conda_build_lines()
        if conda_lines:
            lines.append('conda HDF5/netCDF/MPI builds:')
            lines.extend(conda_lines)

        root = _repo_root()
        head = _git_short_head(str(root))
        if head:
            mark = ' (dirty)' if _git_dirty(str(root)) else ''
            lines.append(f'proteus checkout: {root} -> {head}{mark}')
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        lines.append(f'(environment collection error: {exc!r})')

    return '\n'.join(lines) + '\n'


def _write_failure_log(command: str, transcript: str) -> Path | None:
    """Write the environment info and captured transcript to a timestamped log.

    Parameters
    ----------
    command : str
        Command name used in the file name (``doctor`` or ``update``).
    transcript : str
        Captured console output; ANSI colour codes are stripped before writing.

    Returns
    -------
    Path or None
        Path of the written log file, or None if neither the working directory
        nor the system temporary directory could be written to.
    """
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    # The PID keeps two runs in the same second from overwriting each other.
    name = f'proteus_{command}_{stamp}_{os.getpid()}.log'
    body = (
        f'{_collect_environment_info()}\n=== proteus {command} output ===\n'
        f'{_ANSI_RE.sub("", transcript)}'
    )
    for directory in (Path.cwd(), Path(tempfile.gettempdir())):
        try:
            log_path = directory / name
            log_path.write_text(body, encoding='utf-8')
            return log_path
        except OSError:
            continue
    return None


def _print_support_prompt(command: str, log_path: Path | None):
    """Tell the user where the log is and how to get help.

    Parameters
    ----------
    command : str
        Command name shown in the message (``doctor`` or ``update``).
    log_path : Path or None
        The saved log, or None when it could not be written to disk.
    """
    click.echo()
    click.secho(f'proteus {command} reported problems.', fg='red', bold=True)
    if log_path is not None:
        click.echo(f'A log of this run was saved to:\n  {log_path}')
        what_to_send = 'send that log file'
    else:
        click.secho('A log file could not be written to disk.', fg='yellow')
        what_to_send = 'copy the output above'
    click.echo(
        f'If you need help, {what_to_send} to '
        f'{click.style(_SUPPORT_EMAIL, bold=True)}, '
        'or open an issue or discussion:\n'
        f'  {_ISSUES_URL}\n'
        f'  {_DISCUSSIONS_URL}'
    )


def _run_with_log(command: str, func) -> bool:
    """Run ``func``, mirroring its output; on failure or a crash, write a log
    and tell the user where it is and how to get help.

    Both stdout and stderr are mirrored, so the log captures click output,
    warnings, and any traceback. An exception is treated as a failure: the
    transcript and traceback are logged, the traceback is shown to the user,
    and the function reports failure instead of propagating, so the support
    prompt is never bypassed.

    Parameters
    ----------
    command : str
        Command name for the log file and prompt.
    func : callable
        Zero-argument callable returning True on success, False on failure.

    Returns
    -------
    bool
        True only if ``func`` returned truthy without raising.
    """
    buffer = io.StringIO()
    real_stdout, real_stderr = sys.stdout, sys.stderr
    sys.stdout = _Tee(real_stdout, buffer)
    sys.stderr = _Tee(real_stderr, buffer)
    crash = None
    try:
        ok = bool(func())
    except Exception:  # noqa: BLE001 - report any crash through the log path
        ok = False
        crash = traceback.format_exc()
    finally:
        sys.stdout, sys.stderr = real_stdout, real_stderr

    if ok:
        return True

    transcript = buffer.getvalue()
    if crash:
        transcript += '\n=== traceback ===\n' + crash
    log_path = _write_failure_log(command, transcript)
    if crash:
        # Surface the crash on screen too, not only in the log file.
        click.echo(crash, err=True, nl=False)
    _print_support_prompt(command, log_path)
    return False


def doctor_entry(output_json: bool = False) -> bool:
    """Run diagnostics and print results.

    Parameters
    ----------
    output_json : bool
        If True, print JSON instead of human-readable output.

    Returns
    -------
    bool
        True if no checks failed.
    """
    results = run_all_checks()

    if output_json:
        import json as _json

        click.echo(_json.dumps([r.to_dict() for r in results], indent=2))
        return all(r.status != FAIL for r in results)

    # Group by category
    categories = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    category_labels = {
        'environment': 'Environment',
        'data': 'Reference data',
        'versions': 'Package versions',
    }

    for cat in ('environment', 'data', 'versions'):
        checks = categories.get(cat, [])
        if not checks:
            continue
        click.secho(
            f'\n{category_labels.get(cat, cat)}', fg='yellow', underline=True, bold=True
        )
        for r in checks:
            r.echo()

    # Summary
    n_fail = sum(1 for r in results if r.status == FAIL)
    n_warn = sum(1 for r in results if r.status == WARN)
    fixable = [r for r in results if r.fix_cmd and r.status != PASS and r.auto_fixable]

    click.echo()
    if n_fail == 0 and n_warn == 0:
        click.secho('All checks passed.', fg='green', bold=True)
    else:
        parts = []
        if n_fail:
            parts.append(click.style(f'{n_fail} failed', fg='red'))
        if n_warn:
            parts.append(click.style(f'{n_warn} warnings', fg='yellow'))
        click.echo(', '.join(parts))
        if fixable:
            click.echo(
                f'\nRun {click.style("proteus update", bold=True)} to fix '
                f'{len(fixable)} issue(s) automatically.'
            )

    # A failing check is what triggers the log file and support prompt.
    return n_fail == 0


def _run_fix_command(fix_cmd: str, cwd: Path) -> int:
    """Run a fix command, streaming its combined output through stdout.

    The child's stdout and stderr are read in the parent and re-emitted line by
    line through ``click`` so the output is both shown live and captured by the
    failure log. A subprocess writing to the inherited file descriptors would
    bypass the log entirely.

    Parameters
    ----------
    fix_cmd : str
        Shell command to run.
    cwd : Path
        Directory to run the command in.

    Returns
    -------
    int
        The command's exit code.
    """
    proc = subprocess.Popen(
        fix_cmd,
        shell=True,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout is not None:
        for line in proc.stdout:
            click.echo(line, nl=False)
    return proc.wait()


def update_entry(dry_run: bool = False) -> bool:
    """Run diagnostics and execute fix commands for failing checks.

    Parameters
    ----------
    dry_run : bool
        If True, show what would be done without executing.

    Returns
    -------
    bool
        True if the installation is healthy after the run (no remaining
        failures and no fix command errored). A dry run reports True.
    """
    results = run_all_checks()
    fixable = [r for r in results if r.fix_cmd and r.status != PASS and r.auto_fixable]
    manual = [r for r in results if r.fix_cmd and r.status != PASS and not r.auto_fixable]
    problems = [r for r in results if r.status != PASS]

    if not fixable:
        if not problems:
            click.secho('Nothing to update. All checks passed.', fg='green')
            return True
        # Problems exist but none of them carry an automatic fix command.
        click.secho(
            'Issues found, but none of them have an automatic fix:',
            fg='yellow',
            bold=True,
        )
        for r in problems:
            r.echo()
        # A remaining failure means the install is still unhealthy; a warning
        # alone does not, so only failures trigger the log and support prompt.
        return all(r.status != FAIL for r in problems)

    click.secho(f'{len(fixable)} issue(s) to fix:\n', bold=True)
    for r in fixable:
        icon = click.style(f'[{_ICON[r.status]}]', **_STYLE[r.status])
        click.echo(f'  {icon} {r.name}: {r.message}')
        click.echo(f'       {click.style(r.fix_cmd, fg="cyan")}')

    if manual:
        click.secho(
            '\nThese need manual action (not run automatically):',
            fg='yellow',
            bold=True,
        )
        for r in manual:
            r.echo()

    if dry_run:
        click.echo(f'\n{click.style("Dry run", bold=True)}: no changes made.')
        return True

    root = _repo_root()
    if not (root / 'tools').is_dir():
        click.secho(
            'Cannot run fix commands: no tools/ directory found.\n'
            'proteus update requires a source install (git clone), '
            'not a wheel install.',
            fg='red',
        )
        return False

    click.echo()
    any_fix_failed = False
    for r in fixable:
        click.secho(f'Fixing: {r.name}', bold=True)
        click.echo(f'  $ {r.fix_cmd}')
        returncode = _run_fix_command(r.fix_cmd, root)
        if returncode == 0:
            click.secho('  done', fg='green')
        else:
            click.secho(f'  command failed (exit {returncode})', fg='red')
            click.echo('  Run the command manually to investigate.')
            any_fix_failed = True

    # Re-run the checks via the unwrapped entry so the whole update produces a
    # single failure log, not a nested one.
    click.echo()
    click.secho('Re-checking...', bold=True)
    final_ok = doctor_entry()
    return final_ok and not any_fix_failed


def run_doctor(output_json: bool = False) -> bool:
    """Run ``proteus doctor``; on a failing check, save a log and prompt for help.

    JSON output is for scripts, so it is neither logged nor prompted.

    Parameters
    ----------
    output_json : bool
        If True, print machine-readable JSON instead of the human report.

    Returns
    -------
    bool
        True if all checks passed.
    """
    if output_json:
        return doctor_entry(output_json=True)
    return _run_with_log('doctor', doctor_entry)


def run_update(dry_run: bool = False) -> bool:
    """Run ``proteus update``; on a failed fix or remaining failure, log and prompt.

    Parameters
    ----------
    dry_run : bool
        If True, only show what would be done.

    Returns
    -------
    bool
        True if the update left the installation healthy.
    """
    return _run_with_log('update', lambda: update_entry(dry_run=dry_run))
