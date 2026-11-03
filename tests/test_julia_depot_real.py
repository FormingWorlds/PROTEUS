"""Real Julia checks of the per-install depot; needs no network."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from proteus.doctor import julia_depot_entries

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]

_JULIA = os.environ.get('PYTHON_JULIAPKG_EXE') or shutil.which('julia')
_UUID = '7c1e3a52-4f0b-4d6e-9a11-0f6d2b8c9e01'


def _julia(code: str, depot: str | None, home: Path, project: Path | None = None) -> str:
    """Run Julia offline under a scratch HOME and ``depot``, and return its stdout."""
    env = {'HOME': str(home), 'PATH': os.environ['PATH'], 'JULIA_PKG_OFFLINE': 'true'}
    if depot is not None:
        env['JULIA_DEPOT_PATH'] = depot
    args = [_JULIA, '--startup-file=no', '--color=no']
    if project is not None:
        args.append(f'--project={project}')
    done = subprocess.run(
        [*args, '-e', code], capture_output=True, text=True, env=env, timeout=50
    )
    assert done.returncode == 0, done.stderr[-2000:]
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
    if _JULIA is None:
        pytest.fail('No Julia found; this tier runs where CI installs Julia.')
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


def test_a_new_depot_needs_instantiate_and_it_keeps_the_manifest(stack):
    """A new depot breaks a recorded project until an instantiate, which changes no Manifest."""
    project = stack['project_b']
    # A depot with no Pkg state: the develop path needs no download, so offline works.
    new = f'{stack["tmp"] / "new" / "julia_depot"}:'
    home = stack['home']
    manifest = (project / 'Manifest.toml').read_bytes()
    _julia('using Pkg; Pkg.instantiate(); using Tiny; print(Tiny.f())', new, home, project)
    assert (project / 'Manifest.toml').read_bytes() == manifest
    compiled = stack['tmp'] / 'new' / 'julia_depot' / 'compiled'
    assert list(compiled.rglob('Tiny'))


def test_trailing_colon_adds_the_bundled_depots_and_not_the_home_depot(stack):
    """The form that install.sh writes reads the stdlib compile files and not ``~/.julia``."""
    value = f'{stack["depot_b"]}:'
    entries = _julia('print(join(DEPOT_PATH, "|"))', value, stack['home']).split('|')
    assert entries[0] == str(stack['depot_b'])
    assert len(entries) > 1
    assert str(stack['home'] / '.julia') not in entries


@pytest.mark.parametrize(
    'value', ['/x', '/x:', '/x::', '/x:/y:', ':/x', ':', '/x:/x'], ids=lambda v: repr(v)
)
def test_parser_of_the_doctor_equals_julia(stack, value):
    """julia_depot_entries gives the list of Julia, with the bundled depots left out."""
    home = stack['home']
    os.environ['HOME'], old = str(home), os.environ['HOME']
    try:
        parsed = julia_depot_entries(value)
    finally:
        os.environ['HOME'] = old
    entries = _julia('print(join(DEPOT_PATH, "|"))', value, home).split('|')
    assert [e for e in entries if not e.endswith(('share/julia',))] == parsed
