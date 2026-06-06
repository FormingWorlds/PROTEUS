"""PROTEUS installation diagnostics.

Structured check system for ``proteus doctor``. Each check returns a
typed result (pass/warn/fail) with a human-readable message and an
optional fix command. The ``proteus update`` command collects all fix
commands and offers to run them.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import shlex
import subprocess
import tomllib
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
    if len(parts) >= 2 and parts[0] == '1' and parts[1] == '11':
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
        message=f'{ver} (1.11.x required)',
        fix_cmd='juliaup add 1.11 && juliaup default 1.11',
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

    # Check against pyproject.toml minimum bound
    if spec and spec.specifier:
        if installed not in spec.specifier:
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


def doctor_entry(output_json: bool = False):
    """Run diagnostics and print results.

    Parameters
    ----------
    output_json : bool
        If True, print JSON instead of human-readable output.
    """
    results = run_all_checks()

    if output_json:
        import json as _json

        click.echo(_json.dumps([r.to_dict() for r in results], indent=2))
        return

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
    fixable = [r for r in results if r.fix_cmd and r.status != PASS]

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


def update_entry(dry_run: bool = False):
    """Run diagnostics and execute fix commands for failing checks.

    Parameters
    ----------
    dry_run : bool
        If True, show what would be done without executing.
    """
    results = run_all_checks()
    fixable = [r for r in results if r.fix_cmd and r.status != PASS]

    if not fixable:
        click.secho('Nothing to update. All checks passed.', fg='green')
        return

    click.secho(f'{len(fixable)} issue(s) to fix:\n', bold=True)
    for r in fixable:
        icon = click.style(f'[{_ICON[r.status]}]', **_STYLE[r.status])
        click.echo(f'  {icon} {r.name}: {r.message}')
        click.echo(f'       {click.style(r.fix_cmd, fg="cyan")}')

    if dry_run:
        click.echo(f'\n{click.style("Dry run", bold=True)}: no changes made.')
        return

    root = _repo_root()
    if not (root / 'tools').is_dir():
        click.secho(
            'Cannot run fix commands: no tools/ directory found.\n'
            'proteus update requires a source install (git clone), '
            'not a wheel install.',
            fg='red',
        )
        return

    click.echo()
    for r in fixable:
        click.secho(f'Fixing: {r.name}', bold=True)
        click.echo(f'  $ {r.fix_cmd}')
        try:
            subprocess.run(
                r.fix_cmd,
                shell=True,
                check=True,
                cwd=str(root),
            )
            click.secho('  done', fg='green')
        except subprocess.CalledProcessError as exc:
            click.secho(f'  command failed (exit {exc.returncode})', fg='red')
            click.echo('  Run the command manually to investigate.')

    # Re-run checks after fixes
    click.echo()
    click.secho('Re-checking...', bold=True)
    doctor_entry()
