"""Real Julia checks of the per-install depot; they need a Julia 1.13 binary and no network."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from helpers import git

from proteus.utils.install_env import julia_depot_entries

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]

_JULIA = os.environ.get('PYTHON_JULIAPKG_EXE') or shutil.which('julia')
_UUID = '7c1e3a52-4f0b-4d6e-9a11-0f6d2b8c9e01'


def _julia_problem(exe: str | None) -> str | None:
    """Return why ``exe`` cannot run these checks: no binary, a launcher or another version."""
    if exe is None:
        return 'no Julia binary found'
    real = Path(exe).resolve()
    if real.name == 'julialauncher' or (real.parent / 'juliaup').exists():
        return f'{exe} is the juliaup launcher, which does not see a scratch HOME'
    done = subprocess.run([exe, '--version'], capture_output=True, text=True, timeout=30)
    if not done.stdout.startswith('julia version 1.13'):
        return f'{exe} is {done.stdout.strip()}; these checks need 1.13'
    return None


_PROBLEM = _julia_problem(_JULIA)
if _PROBLEM:
    pytest.skip(_PROBLEM, allow_module_level=True)


def _scratch_env(home: Path) -> dict[str, str]:
    """The environment of a Julia call: a scratch HOME, offline, with PATH and TMPDIR kept."""
    env = {'HOME': str(home), 'PATH': os.environ['PATH'], 'JULIA_PKG_OFFLINE': 'true'}
    if 'TMPDIR' in os.environ:
        env['TMPDIR'] = os.environ['TMPDIR']
    return env


def _julia(code: str, depot: str | None, home: Path, project: Path | None = None) -> str:
    """Run Julia offline under a scratch HOME and ``depot``, and return its stdout."""
    env = _scratch_env(home)
    if depot is not None:
        env['JULIA_DEPOT_PATH'] = depot
    args = [_JULIA, '--startup-file=no', '--color=no']
    if project is not None:
        args.append(f'--project={project}')
    done = subprocess.run(
        [*args, '-e', code], capture_output=True, text=True, env=env, timeout=50
    )
    assert done.returncode == 0, done.stderr[:1000] + '\n...\n' + done.stderr[-1000:]
    return done.stdout


def _tiny_package(tmp_path: Path) -> Path:
    """A package with no dependencies, so that a Pkg change needs no registry."""
    pkg = tmp_path / 'Tiny'
    (pkg / 'src').mkdir(parents=True)
    (pkg / 'Project.toml').write_text(f'name = "Tiny"\nuuid = "{_UUID}"\nversion = "0.1.0"\n')
    (pkg / 'src' / 'Tiny.jl').write_text('module Tiny; f() = 1; end\n')
    return pkg


def _pkg_change(pkg: Path, depot: str | None, home: Path, project: Path) -> None:
    """Change the Pkg state of ``project`` in ``depot``: a Manifest and compile files."""
    code = (
        f'using Pkg; Pkg.develop(path="{pkg}"); Pkg.precompile(); using Tiny; print(Tiny.f())'
    )
    assert _julia(code, depot, home, project) == '1'


def _state(folder: Path) -> str:
    """Hash of the name and bytes of each file of ``folder``."""
    digest = hashlib.md5(usedforsecurity=False)
    for path in sorted(p for p in folder.rglob('*') if p.is_file()):
        digest.update(str(path.relative_to(folder)).encode() + path.read_bytes())
    return digest.hexdigest()


@pytest.fixture
def stack(tmp_path):
    """Install B with a filled depot, a default depot in HOME, and a package."""
    home = tmp_path / 'home'
    (home / '.julia' / 'compiled').mkdir(parents=True)
    (home / '.julia' / 'compiled' / 'old').write_text('old')
    pkg = _tiny_package(tmp_path)
    depot_b = tmp_path / 'b' / 'julia_depot'
    project_b = tmp_path / 'b' / 'project'
    project_b.mkdir(parents=True)
    _pkg_change(pkg, f'{depot_b}:', home, project_b)
    return {
        'tmp': tmp_path,
        'home': home,
        'pkg': pkg,
        'depot_b': depot_b,
        'project_b': project_b,
    }


def test_pkg_change_in_a_leaves_b_and_the_default_depot_unchanged(stack):
    """Two depots: the Manifest and compile files of B and the user's depot keep their hash."""
    tmp = stack['tmp']
    depot_a = tmp / 'a' / 'julia_depot'
    project_a = tmp / 'a' / 'project'
    project_a.mkdir(parents=True)
    before = (_state(stack['depot_b']), _state(stack['project_b']), _state(stack['home']))
    _pkg_change(stack['pkg'], f'{depot_a}:', stack['home'], project_a)
    assert (project_a / 'Manifest.toml').is_file()
    assert list((depot_a / 'compiled').rglob('Tiny'))
    assert before == (
        _state(stack['depot_b']),
        _state(stack['project_b']),
        _state(stack['home']),
    )


def test_canary_with_no_variable_the_default_depot_changes(stack):
    """With the variable unset the same change writes to HOME, so the check above can fail."""
    tmp = stack['tmp']
    project_a = tmp / 'a' / 'project'
    project_a.mkdir(parents=True)
    before = _state(stack['home'])
    _pkg_change(stack['pkg'], None, stack['home'], project_a)
    assert before != _state(stack['home'])
    assert list((stack['home'] / '.julia' / 'compiled').rglob('Tiny'))


def test_a_new_depot_breaks_a_project_until_instantiate_which_keeps_the_manifest(tmp_path):
    """The package source lives in the depot: `using` fails, and instantiate repairs it offline."""
    home = tmp_path / 'home'
    pkg = _tiny_package(tmp_path)
    git(pkg, 'init', '-q')
    git(pkg, 'add', '-A')
    git(pkg, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', 'x')
    project = tmp_path / 'project'
    project.mkdir()
    add = f'using Pkg; Pkg.add(url="file://{pkg}"); using Tiny; print(Tiny.f())'
    assert _julia(add, f'{tmp_path / "old"}:', home, project) == '1'
    manifest = (project / 'Manifest.toml').read_bytes()
    new = f'{tmp_path / "new"}:'
    env = {**_scratch_env(home), 'JULIA_DEPOT_PATH': new}
    broken = subprocess.run(
        [_JULIA, '--startup-file=no', f'--project={project}', '-e', 'using Tiny'],
        capture_output=True,
        text=True,
        env=env,
        timeout=50,
    )
    assert broken.returncode != 0
    assert 'does not seem to be installed' in broken.stderr
    repair = 'using Pkg; Pkg.instantiate(); using Tiny; print(Tiny.f())'
    assert _julia(repair, new, home, project) == '1'
    assert (project / 'Manifest.toml').read_bytes() == manifest


def test_trailing_colon_adds_the_bundled_depots_and_not_the_home_depot(stack):
    """The form that install.sh writes reads the stdlib compile files and not ``~/.julia``."""
    value = f'{stack["depot_b"]}:'
    entries = _julia('print(join(DEPOT_PATH, "|"))', value, stack['home']).split('|')
    assert entries[0] == str(stack['depot_b'])
    assert len(entries) > 1
    assert str(stack['home'] / '.julia') not in entries


@pytest.mark.parametrize(
    'value',
    ['/x', '/x:', '/x::', '/x:/y:', ':/x', ':', '/x:/x', '~/d:', '', None],
    ids=lambda v: repr(v),
)
def test_parser_of_the_doctor_equals_julia(tmp_path, monkeypatch, value):
    """julia_depot_entries gives the list of Julia, with the bundled depots left out."""
    monkeypatch.setenv('HOME', str(tmp_path))
    entries = _julia('print(join(DEPOT_PATH, "|"))', value, tmp_path).split('|')
    parsed = julia_depot_entries(value)
    assert [e for e in entries if e and not e.endswith('share/julia')] == parsed
    assert (str(tmp_path / '.julia') in parsed) == (value is None or value.startswith(':'))


def test_a_launcher_or_another_version_is_named_as_the_reason_to_skip(tmp_path):
    """The reason of the skip names the binary, so a CI log shows what ran."""
    stub = tmp_path / 'julia'
    stub.write_text('#!/bin/sh\necho "julia version 1.12.6"\n')
    stub.chmod(0o755)
    assert '1.12.6' in _julia_problem(str(stub))
    assert _julia_problem(None) == 'no Julia binary found'
    (tmp_path / 'juliaup').write_text('')
    assert 'launcher' in _julia_problem(str(stub))
