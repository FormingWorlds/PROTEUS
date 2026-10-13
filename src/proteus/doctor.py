"""PROTEUS installation diagnostics.

Structured check system for ``proteus doctor``. Each check returns a
typed result (pass/warn/fail) with a human-readable message and an
optional fix command. The ``proteus update`` command collects all fix
commands and offers to run them.
"""

from __future__ import annotations

import datetime
import functools
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
    _get_obliqua_version,
    _get_socrates_version,
    get_proteus_directories,
)
from proteus.utils.helper import git_env, proteus_julia_exe

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
    """Return dependencies as {name: Requirement|None} for FWL and standard packages.

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
    opt_deps = cfg.get('project', {}).get('optional-dependencies', {})
    for extra_deps in opt_deps.values():
        for dep_str in extra_deps:
            try:
                req = Requirement(dep_str)
            except Exception:
                continue
            if req.name in STANDARD_MODULE_PACKAGES:
                result[req.name] = req
    return result


def _get_script_for_package(dist_name: str) -> str | None:
    """Return the ``tools/get_<name>.sh`` setup script for an FWL package.

    Some FWL packages (aragog, zalmoxis, vulcan, morrigan, atmodeller, boreas) have a
    setup script of their own; MORS, JANUS, CALLIOPE and ZEPHYRUS share
    ``tools/get_python_module.sh`` (``_setup_command``).

    Returns the repo-relative script path when it exists, else None.
    """
    short = dist_name.lower()
    prefix = 'fwl-'
    if short.startswith(prefix):
        short = short[len(prefix) :]
    script_rel = f'tools/get_{short}.sh'
    if (_repo_root() / script_rel).is_file():
        return script_rel
    return None


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


# Seconds that one git call in a checkout may take.
_GIT_NOTE_TIMEOUT_S = 10


def _git_stdout(
    path: str, *args: str, options: tuple[str, ...] = (), timeout: float = _GIT_NOTE_TIMEOUT_S
) -> str | None:
    """Return the standard output of one git command in a checkout.

    The environment is ``git_env()``, so that ``git -C`` reads the repository
    at ``path`` and not one that ``GIT_DIR`` names.

    Parameters
    ----------
    path : str
        Checkout directory, passed to ``git -C``.
    *args : str
        The git subcommand and its arguments, for example ``'rev-parse', 'HEAD'``.
    options : tuple of str
        Options of git itself that go before ``-C``, for example
        ``('--no-optional-locks',)``.
    timeout : float
        Seconds the command may take.

    Returns
    -------
    str or None
        The output, or None when git is missing, fails or does not finish
        within ``timeout`` s.
    """
    try:
        return subprocess.check_output(
            ['git', *options, '-C', path, *args],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            env=git_env(),
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def _git_head(path: str) -> str | None:
    """Return the full HEAD commit hash for a git checkout, or None."""
    out = _git_stdout(path, 'rev-parse', 'HEAD')
    return None if out is None else out.strip()


def _git_short_head(path: str) -> str | None:
    """Return the short HEAD commit hash for a git checkout, or None."""
    out = _git_stdout(path, 'rev-parse', '--short', 'HEAD')
    return None if out is None else out.strip()


def _git_dirty(path: str) -> bool | None:
    """Return True if the git working tree has uncommitted changes, False if it
    is clean, or None if the git status could not be determined (not a git
    repository, git missing, a git error, or a git that does not finish within
    ``_GIT_NOTE_TIMEOUT_S``). ``--no-optional-locks`` keeps git from rewriting
    the index of the checkout."""
    out = _git_stdout(path, 'status', '--porcelain', options=('--no-optional-locks',))
    return None if out is None else bool(out.strip())


# The module repositories that tools/get_python_module.sh clones, by distribution.
_PYTHON_MODULE_REPOS = {
    'fwl-calliope': 'CALLIOPE',
    'fwl-janus': 'JANUS',
    'fwl-mors': 'MORS',
    'fwl-zephyrus': 'ZEPHYRUS',
}

# The folder in the PROTEUS root that a setup script clones each module into.
_MODULE_FOLDERS = {
    **_PYTHON_MODULE_REPOS,
    'fwl-aragog': 'aragog',
    'fwl-zalmoxis': 'Zalmoxis',
    'fwl-morrigan': 'Morrigan',
    'fwl-vulcan': 'VULCAN',
    'atmodeller': 'Atmodeller',
}

# The modules whose setup script gives a new clone the branch named like the PROTEUS feature branch.
_PAIRED_MODULES = (
    'fwl-aragog',
    'fwl-zalmoxis',
)


def _git_note_output(path: str, *args: str) -> tuple[int, str] | None:
    """Run one short git command in a checkout for the release-tag note.

    Parameters
    ----------
    path : str
        Checkout directory.
    *args : str
        The git arguments, for example ``'symbolic-ref', '-q', 'HEAD'``.

    Returns
    -------
    tuple of (int, str) or None
        The exit status and the standard output, or None when git is
        missing or does not finish within ``_GIT_NOTE_TIMEOUT_S``.
    """
    try:
        done = subprocess.run(
            ['git', '-C', path, *args],
            capture_output=True,
            text=True,
            timeout=_GIT_NOTE_TIMEOUT_S,
            env=git_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.returncode, done.stdout


def _is_own_clone(path: str) -> bool:
    """Return whether a directory is the top level of its own git clone.

    An empty or broken ``.git`` makes git answer for the repository around
    the directory, so this is checked before the clone is read.

    Parameters
    ----------
    path : str
        Checkout directory.

    Returns
    -------
    bool
        True when ``git rev-parse --show-toplevel`` names the directory
        itself; False otherwise, also when git fails or times out.
    """
    top = _git_note_output(path, 'rev-parse', '--show-toplevel')
    if top is None or top[0] != 0 or not top[1].strip():
        return False
    try:
        return os.path.samefile(top[1].strip(), path)
    except OSError:
        return False


def _floor_version(spec: Requirement | None) -> str | None:
    """Return the one ``>=`` floor of a requirement.

    Parameters
    ----------
    spec : Requirement or None
        The requirement of ``pyproject.toml``, for example ``fwl-mors>=26.10.6``.

    Returns
    -------
    str or None
        The floor, for example ``'26.10.6'``, or None without exactly one
        ``>=`` specifier.
    """
    floors = [s.version for s in (spec.specifier if spec else []) if s.operator == '>=']
    return floors[0] if len(floors) == 1 else None


def _floor_tag_name(spec: Requirement | None, tags: list[str]) -> str | None:
    """Return the one tag whose release number equals the ``>=`` floor of a requirement.

    The match is the one of ``tools/_module_pins.py``: a leading ``v`` is
    ignored and the numbers compare as versions, so ``26.10.06`` matches
    the floor ``26.10.6``.

    Parameters
    ----------
    spec : Requirement or None
        The requirement of ``pyproject.toml``, for example ``fwl-mors>=26.10.6``.
    tags : list of str
        Tag names of the clone.

    Returns
    -------
    str or None
        The tag, or None when the requirement has no single ``>=`` floor
        or not exactly one tag matches it.
    """
    floor = _floor_version(spec)
    if floor is None:
        return None
    matches = []
    for tag in tags:
        text = tag.removeprefix('v')
        if not re.fullmatch(r'[0-9]+(\.[0-9]+)*', text):
            continue
        if Version(text) == Version(floor):
            matches.append(tag)
    return matches[0] if len(matches) == 1 else None


def _tag_commits(checkout: str) -> dict[str, str]:
    """Return the commit of each tag of a clone, read with one git call.

    The full ref name is listed and ``refs/tags/`` stripped, so a tag that
    shares its name with a branch keeps its own name.

    Parameters
    ----------
    checkout : str
        Directory of the clone.

    Returns
    -------
    dict of str to str
        Tag name to commit hash; empty when git cannot list the tags.
    """
    listing = _git_note_output(
        checkout,
        'for-each-ref',
        '--format=%(refname) %(objectname) %(*objectname)',
        'refs/tags',
    )
    lines = listing[1].splitlines() if listing and listing[0] == 0 else []
    # The last field is the commit, for an annotated tag as for a lightweight one.
    return {f[0].removeprefix('refs/tags/'): f[-1] for f in (line.split() for line in lines)}


def _proteus_feature_branch() -> str | None:
    """Return the branch PROTEUS is on, as the setup scripts read it to pair a module branch.

    In CI the branch is ``GITHUB_HEAD_REF`` or ``GITHUB_REF_NAME``; else it
    is read from the PROTEUS checkout with ``git rev-parse --abbrev-ref
    HEAD``, a read-only call with a timeout.

    Returns
    -------
    str or None
        The branch name; None for ``main``, ``master``, a detached HEAD or
        a branch that cannot be read, which pair with no module branch.
    """
    branch = os.environ.get('GITHUB_HEAD_REF') or os.environ.get('GITHUB_REF_NAME')
    if not branch:
        out = _git_note_output(str(_repo_root()), 'rev-parse', '--abbrev-ref', 'HEAD')
        branch = out[1].strip() if out and out[0] == 0 else ''
    return None if branch in ('', 'HEAD', 'main', 'master') else branch


def _release_tag_note(checkout: str, short: str | None, spec: Requirement | None) -> str:
    """Return the note on an editable clone that is not at the release tag of its floor.

    The note has the parts of the installer's stay line: what the clone is
    on, and what PROTEUS expects, in the words of the installer. It costs up
    to 4 git calls with a timeout: the top level of the clone, the branch,
    the tags with their commits, and ``merge-base --is-ancestor``. HEAD is
    compared with the tag commit first, so a branch at the tag commit gets
    no note, as in the installer; a HEAD that has the tag commit as an
    ancestor is newer than the release and gets none either. An Aragog or
    Zalmoxis clone on the branch that PROTEUS is on is the paired branch
    of the installer and gets no note; that costs one call on the PROTEUS
    checkout, and it compares branch names only, without a network call.

    Parameters
    ----------
    checkout : str
        Directory of the editable checkout.
    short : str or None
        Short HEAD commit of the checkout, as ``_git_short_head`` gives it.
    spec : Requirement or None
        The requirement of ``pyproject.toml``.

    Returns
    -------
    str
        For example ``'; on branch main; PROTEUS expects the release tag
        26.10.06'``; empty at the tag commit, newer than it, on the paired
        branch, without a single ``>=`` floor, or when the checkout is not a
        clone of its own that git can read.
    """
    if (
        _floor_version(spec) is None
        or not _is_own_clone(checkout)
        or spec.name.lower() in _module_pins()
    ):
        return ''
    branch = _git_note_output(checkout, 'symbolic-ref', '-q', '--short', 'HEAD')
    if branch is None or branch[0] not in (0, 1):
        return ''
    if (
        branch[0] == 0
        and spec.name.lower() in _PAIRED_MODULES
        and branch[1].strip() == _proteus_feature_branch()
    ):
        return ''
    commits = _tag_commits(checkout)
    tag = _floor_tag_name(spec, list(commits))
    if short and tag and commits[tag].startswith(short):
        return ''
    if short and tag:
        newer = _git_note_output(checkout, 'merge-base', '--is-ancestor', commits[tag], 'HEAD')
        if newer is not None and newer[0] == 0:
            return ''
    if tag:
        expected = f'PROTEUS expects the release tag {tag}'
    else:
        expected = f'PROTEUS expects release {_floor_version(spec)}'
    if branch[0] == 0:
        return f'; on branch {branch[1].strip()}; {expected}'
    if not short:
        return ''
    at_tags = sorted(name for name, commit in commits.items() if commit.startswith(short))
    found = f'at tag {at_tags[0]} ({short})' if at_tags else f'at commit {short}'
    if tag:
        found += ', not the release tag'
    return f'; {found}; {expected}'


def _setup_command(dist_name: str) -> str | None:
    """Return the setup script, with its argument, that clones a module.

    Parameters
    ----------
    dist_name : str
        Distribution name, for example ``fwl-mors``.

    Returns
    -------
    str or None
        The repo-relative command, for example ``'tools/get_aragog.sh'`` or
        ``'tools/get_python_module.sh MORS'``; None when the source tree
        has no script for the module.
    """
    script = _get_script_for_package(dist_name)
    if script:
        return script
    repo = _PYTHON_MODULE_REPOS.get(dist_name.lower())
    if repo and (_repo_root() / 'tools' / 'get_python_module.sh').is_file():
        return f'tools/get_python_module.sh {repo}'
    return None


def _below_floor_advice(
    dist_name: str, checkout: str, spec: Requirement | None, setup: str
) -> str:
    """Return the advice for an editable module clone below its version floor.

    A fix never changes the clone: the advice names the commands that move
    it to the release tag by hand and reinstall it, the ones the installer
    prints, and, for the clone in the module folder of the PROTEUS root
    only, the setup script that replaces it with ``--force``. The tag is
    read from the clone without a fetch; when the clone does not have it
    yet, the checkout picks it from the tags the fetch brings. A folder
    that is not a git clone of its own gets no git command, which would act
    on the repository around it: only the ``--force`` replacement, or the
    advice to move it away.

    Parameters
    ----------
    dist_name : str
        Distribution name, for example ``fwl-mors``.
    checkout : str
        Directory of the editable checkout.
    spec : Requirement or None
        The requirement of ``pyproject.toml``.
    setup : str
        The setup command of ``_setup_command``.

    Returns
    -------
    str
        One line of shell commands with absolute, shell-quoted paths, and
        a comment that names the replacement when the checkout resolves to
        ``<root>/<module folder>``; for a folder that is not a clone of its
        own, the replacement with a comment, or a comment alone.
        A module with a commit pin in ``[tool.proteus.modules]`` gets the
        checkout of that commit in place of the release tag.
    """
    path = shlex.quote(os.path.abspath(checkout))
    root = _repo_root()
    folder = _MODULE_FOLDERS.get(dist_name.lower())
    in_folder = bool(folder) and os.path.realpath(checkout) == os.path.realpath(root / folder)
    replace = f'cd {shlex.quote(str(root))} && bash {setup}'
    ref = _module_pins().get(dist_name.lower(), {}).get('ref')
    if not _is_own_clone(checkout):
        if in_folder:
            return (
                f'{replace} --force  # {path} is not a git clone of its own; this replaces it'
                ' with a new clone, which deletes everything in it'
            )
        target_name = 'its pinned commit' if ref else 'the release tag'
        return (
            f'# {path} is not a git clone of its own, so no git command moves it to'
            f' {target_name}; move it away, then run: {replace}'
        )
    if ref:
        fetch, target = 'origin', shlex.quote(ref)
    else:
        fetch = '--tags'
        tag = _floor_tag_name(spec, list(_tag_commits(checkout)))
        if tag:
            target = f'refs/tags/{shlex.quote(tag)}'
        else:
            pins = shlex.quote(str(root / 'tools' / '_module_pins.py'))
            target = (
                f'"refs/tags/$(git -C {path} tag -l | python {pins} {dist_name} floor-tag)"'
            )
    advice = (
        f'git -C {path} fetch {fetch} && git -C {path} checkout --detach {target}'
        f' && pip install -e {path}'
    )
    if in_folder:
        advice += f'  # or replace the clone: {replace} --force'
    return advice


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


def _set_var_advice(name: str, value: str) -> str:
    """Return the advice that sets ``name`` in the PROTEUS conda env and reloads it.

    Conda loads env variables at activation, so the env is deactivated and
    activated again; a shell without conda sources the PROTEUS exports file.
    With no env or ``base`` active, the advice names the env as
    ``<your PROTEUS environment>`` and passes it with ``-n``, because PROTEUS
    is not installed into ``base``.

    Parameters
    ----------
    name : str
        Variable name.
    value : str
        Value or placeholder to show, for example ``<path>``.

    Returns
    -------
    str
        Human advice, not a runnable command.
    """
    env = os.environ.get('CONDA_DEFAULT_ENV')
    without = '(without conda: source .proteus-env.sh in the PROTEUS folder)'
    if not env or env == 'base':
        env = '<your PROTEUS environment>'
        return (
            f'conda env config vars set -n {env} {name}={value}, then '
            f'conda activate {env} {without}'
        )
    return (
        f'conda env config vars set {name}={value}, then conda deactivate && '
        f'conda activate {env} {without}'
    )


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
            fix_cmd=_set_var_advice(name, '<path>'),
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
    """Check FWL_DATA contents for required data sets.

    A data set found only in the older layout is fixed with ``fwl-io relocate``,
    which moves it in place, rather than with a new download, but only when a
    relocate dry run would move something there. The dry run hashes the whole
    tree, so it runs only for a missing data set with an older-layout folder.
    """
    results = []
    fwl = os.environ.get('FWL_DATA')
    if not fwl or not os.path.isdir(fwl := os.path.expanduser(fwl)):
        return results

    @functools.cache
    def movable():
        try:
            from fwl_io.relocate import plan_relocations

            return [Path(e.legacy_dir).resolve() for e in plan_relocations(fwl).ready]
        except Exception:
            return []

    expected = {
        'atmos_clim/spectral_files': ('proteus get spectral', 'spectral_files'),
        'star/spectra': ('proteus get stellar', 'stellar_spectra'),
    }
    for subdir, (fix, legacy) in expected.items():
        path = os.path.join(fwl, subdir)
        old = os.path.join(fwl, legacy)
        present = os.path.isdir(path) and os.listdir(path)
        if not present and os.path.isdir(old) and os.listdir(old):
            if any(p.is_relative_to(Path(old).resolve()) for p in movable()):
                fix = 'fwl-io relocate'
        if present:
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
    """Check the version of the Julia PROTEUS uses.

    That is ``PYTHON_JULIAPKG_EXE`` when it is set, and the ``julia`` on
    ``PATH`` otherwise. A set variable that names no executable file fails,
    because juliacall does not fall back to the ``julia`` on ``PATH``. Julia
    1.13.x passes; any other version warns, because PROTEUS is tested on 1.13
    and Obliqua resolves only on 1.11 or 1.13. The advice adds 1.13 beside the
    default Julia and points ``PYTHON_JULIAPKG_EXE`` at it, as ``install.sh``
    does.
    """
    path = '<the path that julia +1.13 -e \'print(joinpath(Sys.BINDIR, "julia"))\' prints>'
    fix = 'juliaup add 1.13, then ' + _set_var_advice('PYTHON_JULIAPKG_EXE', path)
    raw = os.environ.get('PYTHON_JULIAPKG_EXE')
    exe = proteus_julia_exe()
    if raw and exe is None:
        return CheckResult(
            name='julia',
            category='environment',
            status=FAIL,
            message=(
                f'PYTHON_JULIAPKG_EXE={raw} is not an executable file; juliacall does not '
                'fall back to the julia on PATH and fails at import'
            ),
            fix_cmd=fix,
            auto_fixable=False,
        )
    ver = _julia_version_at(exe) if exe else _julia_version()
    where = f' at {exe} (PYTHON_JULIAPKG_EXE)' if exe else ''
    if ver is None:
        return CheckResult(
            name='julia',
            category='environment',
            status=FAIL,
            message=f'not runnable{where}' if exe else 'not found on PATH',
            fix_cmd='curl -fsSL https://install.julialang.org | sh',
        )
    if ver.split('.')[:2] == ['1', '13']:
        return CheckResult(
            name='julia',
            category='environment',
            status=PASS,
            message=f'{ver}{where}',
        )
    # fix_julia_111_for_openssl in install.sh is the one place that binds 1.11.
    if exe and ver.split('.')[:2] == ['1', '11']:
        fix += (
            '. The installer binds Julia 1.11 only when the import of PROTEUS fails on '
            'OpenSSL_jll because Python links an OpenSSL older than 3.5; after '
            "conda install -c conda-forge 'openssl>=3.5', Julia 1.13 can be used"
        )
    # Not auto-fixable: the variable is set in the conda env by hand.
    return CheckResult(
        name='julia',
        category='environment',
        status=WARN,
        message=(
            f'{ver}{where} (PROTEUS is tested on Julia 1.13; '
            'Obliqua, the tides module, installs only on 1.11 or 1.13)'
        ),
        fix_cmd=fix,
        auto_fixable=False,
    )


def check_cvode() -> CheckResult:
    """Check that the SUNDIALS CVODE solver is available to Aragog.

    Aragog integrates the interior with CVODE when
    ``interior_energetics.aragog.solver_method = "cvode"``, the production
    setting and the same solver SPIDER uses. Without the wrapper it falls
    back to scipy Radau, which stops on its own melt-fraction cap at the
    crystallization front and so takes far shorter steps through it. The
    fallback is reported here rather than only in the run log, where it is
    a per-solve warning that is easy to miss until a long coupled run has
    already spent hours on it.
    """
    if importlib.util.find_spec('scikits_odes_sundials') is None:
        return CheckResult(
            name='cvode',
            category='environment',
            status=WARN,
            message='not installed; Aragog integrates with scipy Radau instead',
            fix_cmd='bash tools/get_cvode.sh',
        )
    try:
        importlib.import_module('scikits_odes_sundials.cvode')
    except Exception as exc:
        # Installed but not loadable is its own failure: the wrapper is
        # compiled against the SUNDIALS C library, so a version or ABI
        # mismatch imports the package and then fails on the extension.
        return CheckResult(
            name='cvode',
            category='environment',
            status=WARN,
            message=(
                f'installed but does not load ({type(exc).__name__}); '
                'Aragog integrates with scipy Radau instead'
            ),
            fix_cmd='bash tools/get_cvode.sh',
        )
    return CheckResult(
        name='cvode',
        category='environment',
        status=PASS,
        message='available for the Aragog interior',
    )


def _installed_version(name: str) -> Version | None:
    """Return the installed version of a distribution, or None when it is not installed."""
    try:
        return Version(importlib.metadata.version(name))
    except PackageNotFoundError:
        return None


def check_python_package(name: str, spec: Requirement | None) -> CheckResult:
    """Check a Python package against the pyproject.toml version spec."""
    installed = _installed_version(name)
    if installed is None:
        status, fix = FAIL, f'pip install {name}'
        if name in STANDARD_MODULE_PACKAGES:
            setup = _setup_command(name)
            extra = name.lower().removeprefix('fwl-')
            status = WARN
            fix = (
                f'cd {shlex.quote(str(_repo_root()))} && bash {setup}'
                if setup
                else f'pip install "fwl-proteus[{extra}]"'
            )
        return CheckResult(
            name=name, category='versions', status=status, message='not installed', fix_cmd=fix
        )

    # Build status message with editable annotation
    checkout = _editable_checkout_path(name)
    setup = _setup_command(name) if checkout else None
    annotation = ''
    if checkout:
        base = os.path.basename(checkout.rstrip('/'))
        marker = ''
        # git in a folder that is not a clone of its own reads the repository around it.
        if not _is_own_clone(checkout):
            annotation = f' [editable @ {base}; not a git clone of its own]'
        else:
            short = _git_short_head(checkout)
            dirty = _git_dirty(checkout)
            if dirty is None:
                marker = ' (git status unknown)'
            elif dirty:
                marker = ' (dirty)'
            # Only a module that a setup script clones at its floor has a release tag to expect.
            note = _release_tag_note(checkout, short, spec) if setup else ''
            annotation = f' [editable @ {base} -> {short}{marker}{note}]'
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
            auto_fixable = True
            if checkout:
                # A fix never changes a module clone that a setup script makes, so it is advice.
                if setup:
                    fix = _below_floor_advice(name, checkout, spec, setup)
                    auto_fixable = False
                elif _is_own_clone(checkout):
                    fix = f'cd {shlex.quote(checkout)} && git pull && pip install -e .'
                else:
                    # git pull here would update the repository around the folder.
                    fix = (
                        f'{fix}  # {shlex.quote(checkout)} is not a git clone of its own, so no'
                        ' git command updates it; this installs the release in place of the'
                        ' editable folder'
                    )
                    auto_fixable = False
            return CheckResult(
                name=name,
                category='versions',
                status=FAIL,
                message=f'{installed} (requires {spec.specifier}){annotation}',
                fix_cmd=fix,
                auto_fixable=auto_fixable,
            )

    # Package satisfies the pin
    return CheckResult(
        name=name,
        category='versions',
        status=PASS,
        message=f'{installed}{annotation}',
    )


def _is_git_module_present(path: str, required: bool = True) -> bool:
    """Return whether a git module directory is populated."""
    if not path or not os.path.isdir(path):
        return False
    return required or bool(os.listdir(path))


def check_git_module(name: str, dirs: dict, required: bool = True) -> CheckResult:
    """Check a git-pinned module (AGNI, SOCRATES, Obliqua) against pyproject.toml ref.

    Parameters
    ----------
    name : str
        Module name, matching a `[tool.proteus.modules.<name.lower()>]`
        pyproject.toml table and a `dirs[name.lower()]` entry.
    dirs : dict
        Directory mapping from `get_proteus_directories()`.
    required : bool
        When True (AGNI, SOCRATES), a missing checkout returns a FAIL result.
        When False (Obliqua), a missing checkout returns a WARN result with
        its setup command.
    """
    pins = _module_pins()
    pin = pins.get(name.lower(), {})
    pinned_ref = pin.get('ref')

    # Determine the checkout path
    dir_key = name.lower()
    if dir_key == 'socrates':
        path = os.environ.get('RAD_DIR', '')
    else:
        path = dirs.get(dir_key, '')

    if not _is_git_module_present(path, required=required):
        # Unlike the off-pin case below, this fix is not chained to an AGNI
        # rebuild: a not-installed SOCRATES means RAD_DIR is unset, so the AGNI
        # step would have nowhere to find SOCRATES. The install script prints
        # the AGNI follow-up once SOCRATES is in place.
        return CheckResult(
            name=name,
            category='versions',
            status=FAIL if required else WARN,
            message='not installed',
            fix_cmd=f'bash tools/get_{name.lower()}.sh',
        )

    # Get current version
    try:
        if name == 'AGNI':
            ver = _get_agni_version(dirs)
        elif name == 'SOCRATES':
            ver = _get_socrates_version()
        elif name == 'Obliqua':
            ver = _get_obliqua_version(dirs)
        else:
            ver = '?'
    except Exception:
        ver = '?'

    # Check HEAD against pinned ref
    if not _is_own_clone(path):
        ver = f'{ver} [not a git clone of its own]'
        head = None
    else:
        head = _git_head(path)
    if pinned_ref and head and head != pinned_ref:
        # Re-cloning SOCRATES deletes AGNI build products in that tree;
        # the SOCRATES refresh command chains an AGNI rebuild.
        refresh = f'bash tools/get_{name.lower()}.sh'
        auto_fixable = True
        if name == 'SOCRATES':
            refresh = (
                'bash tools/get_socrates.sh "$RAD_DIR" '
                '&& RAD_DIR="$RAD_DIR" bash tools/get_agni.sh 0'
            )
        elif name == 'Obliqua':
            root = shlex.quote(str(_repo_root()))
            folder = shlex.quote(os.path.abspath(path))
            refresh = (
                f'cd {root} && bash tools/get_obliqua.sh --force {folder}  # replaces the '
                'folder with a new clone at the pin, which deletes everything in it'
            )
            auto_fixable = False
        return CheckResult(
            name=name,
            category='versions',
            status=WARN,
            message=(f'{ver} ({head[:8]}) differs from pin ({pinned_ref[:8]})'),
            fix_cmd=refresh,
            auto_fixable=auto_fixable,
        )

    if name == 'Obliqua' and not (Path(path) / 'Manifest.toml').is_file():
        head_str = f' ({head[:8]})' if head else ''
        return CheckResult(
            name=name,
            category='versions',
            status=WARN,
            message=f'{ver}{head_str}: Julia environment not instantiated',
            fix_cmd='bash tools/get_obliqua.sh',
        )

    if pinned_ref and head:
        return CheckResult(
            name=name,
            category='versions',
            status=PASS,
            message=f'{ver} ({head[:8]})',
        )

    return CheckResult(
        name=name,
        category='versions',
        status=PASS,
        message=f'{ver}',
    )


_STANDARD_DISTRIBUTIONS = {
    'morrigan': 'fwl-morrigan',
    'vulcan': 'fwl-vulcan',
    'atmodeller': 'atmodeller',
}


def is_standard_module_installed(name: str, root: Path | str | None = None) -> bool:
    """Return whether a standard module is installed.

    Delegates to the doctor's installation rules: package discovery for
    Morrigan, VULCAN and atmodeller, and directory population for Obliqua.
    """
    key = name.lower()
    if key in _STANDARD_DISTRIBUTIONS:
        return _installed_version(_STANDARD_DISTRIBUTIONS[key]) is not None
    if key == 'obliqua':
        if root is None:
            root = _repo_root()
        path = str(Path(root) / 'Obliqua')
        return _is_git_module_present(path, required=False)
    return False


# ─── Main entry points ──────────────────────────────────────────────


ENVIRONMENT_VARS = [
    ('FWL_DATA', True, None),
    ('RAD_DIR', True, 'bin/radlib.a'),
    ('FC_DIR', True, None),
    ('PYTHON_JULIAPKG_EXE', True, None),
]

# Mandatory Python packages checked by doctor. An absent package in this list
# is a failure.
PYTHON_PACKAGES = [
    'fwl-proteus',
    'fwl-io',
    'fwl-aragog',
    'fwl-calliope',
    'fwl-janus',
    'fwl-mors',
    'fwl-zephyrus',
    'fwl-zalmoxis',
]

# An absent package in this list gives a warning with its setup script,
# or with the package extra when the tree has no setup script.
STANDARD_MODULE_PACKAGES = [
    'fwl-morrigan',
    'fwl-vulcan',
    'atmodeller',
]

GIT_MODULES = ['AGNI', 'SOCRATES']
STANDARD_GIT_MODULES = ['Obliqua']


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
    try:
        results.append(check_cvode())
    except Exception as exc:
        results.append(
            CheckResult(
                name='cvode',
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
    for pkg in (*PYTHON_PACKAGES, *STANDARD_MODULE_PACKAGES):
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
    for mod in (*GIT_MODULES, *STANDARD_GIT_MODULES):
        try:
            results.append(check_git_module(mod, dirs, mod in GIT_MODULES))
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

_SUPPORT_EMAIL = 'dev@proteus-framework.org'
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
    'fwl-morrigan',
    'atmodeller',
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
