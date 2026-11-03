"""Tests for ``proteus.utils.install_env``.

The exports file holds one ``export NAME='value'`` line per variable, which a
fresh bash sources back to the same values; it is rewritten in place. The
conda env gets the variables without a single quote in their value in one
``conda env config vars set`` call, and only when a conda env other than
``base`` (and not a pixi env) is active and ``conda`` is on PATH.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from proteus.utils import install_env
from proteus.utils.install_env import (
    ENV_FILE_NAME,
    active_conda_prefix,
    bind_to_conda_env,
    depot_of_other_install,
    depot_value_is_unusable,
    export_install_vars,
    julia_depot_entries,
    missing_julia_packages,
    quoted_names,
    reload_instruction,
    write_env_file,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# A space and a single quote are the two values that a naive quoting breaks.
_VALUES = {
    'FWL_DATA': '/data/FWL DATA',
    'RAD_DIR': "/home/o'brien/socrates",
    'PYTHON_JULIAPKG_EXE': '/opt/julia-1.13.1/bin/julia',
    'FC_DIR': '/home/u/AGNI/fastchem',
}


def _sourced(path: Path, names) -> dict[str, str]:
    """The variables ``names`` as a fresh bash sees them after sourcing ``path``."""
    script = 'source "$1"; shift; for n in "$@"; do printf "%s\\0" "${!n}"; done'
    out = subprocess.run(
        ['bash', '--noprofile', '--norc', '-c', script, '_', str(path), *names],
        capture_output=True,
        text=True,
        env={'PATH': '/usr/bin:/bin'},
        timeout=10,
        check=True,
    )
    return dict(zip(names, out.stdout.split('\0')))


class TestWriteEnvFile:
    """write_env_file keeps one quoted line per variable."""

    def test_four_values_source_back_unchanged(self, tmp_path):
        """4 variables give 4 lines, and a fresh bash reads back the exact values."""
        path = write_env_file(tmp_path, _VALUES)
        assert path == tmp_path / ENV_FILE_NAME
        lines = path.read_text().splitlines()
        assert len(lines) == 4
        assert lines[1] == "export RAD_DIR='/home/o'\\''brien/socrates'"
        assert _sourced(path, list(_VALUES)) == _VALUES

    def test_rewrite_replaces_lines_and_keeps_others_in_place(self, tmp_path):
        """A second write replaces the lines of its variables, keeps a foreign line and the inode.

        A rename of a temporary file would give a new inode and drop the mode 600.
        """
        path = tmp_path / ENV_FILE_NAME
        path.write_text("export OTHER='keep'\nexport FWL_DATA='/old'\n")
        path.chmod(0o600)
        inode = path.stat().st_ino
        write_env_file(tmp_path, _VALUES)
        write_env_file(tmp_path, {'FWL_DATA': '/new'})
        lines = path.read_text().splitlines()
        assert len(lines) == 5
        assert lines[0] == "export OTHER='keep'"
        assert [ln for ln in lines if ln.startswith('export FWL_DATA=')] == [
            "export FWL_DATA='/new'"
        ]
        assert path.stat().st_ino == inode
        assert path.stat().st_mode & 0o777 == 0o600


class TestConda:
    """The conda env gets the variables only when a conda env is active."""

    @pytest.mark.parametrize(
        ('env', 'conda', 'expected'),
        [
            ({'CONDA_PREFIX': '/envs/proteus'}, '/bin/conda', '/envs/proteus'),
            ({}, '/bin/conda', None),
            ({'CONDA_PREFIX': '/envs/proteus'}, None, None),
            (
                {'CONDA_PREFIX': '/w/.pixi/envs/default', 'PIXI_ENVIRONMENT_NAME': 'default'},
                '/bin/conda',
                None,
            ),
            ({'CONDA_PREFIX': '/opt/conda', 'CONDA_DEFAULT_ENV': 'base'}, '/bin/conda', None),
            (
                {'CONDA_PREFIX': '/opt/conda/envs/based', 'CONDA_DEFAULT_ENV': 'based'},
                '/bin/conda',
                '/opt/conda/envs/based',
            ),
        ],
        ids=[
            'conda-env',
            'no-env',
            'conda-not-on-path',
            'pixi-env',
            'base-env',
            'env-named-based',
        ],
    )
    def test_active_conda_prefix(self, monkeypatch, env, conda, expected):
        """Only a conda env other than base with conda on PATH counts; a pixi env does not.

        `based` starts with `base` and is an env of the user's, so a prefix match fails it.
        """
        for var in ('CONDA_PREFIX', 'PIXI_ENVIRONMENT_NAME', 'CONDA_DEFAULT_ENV'):
            monkeypatch.delenv(var, raising=False)
        for var, value in env.items():
            monkeypatch.setenv(var, value)
        monkeypatch.setattr(install_env.shutil, 'which', lambda exe: conda)
        assert active_conda_prefix() == expected
        assert (active_conda_prefix() is None) is (expected is None)

    def test_bind_passes_all_pairs_without_a_quote_in_one_call(self, monkeypatch):
        """One conda call carries `-p <prefix>` and the 3 NAME=value pairs without a quote.

        RAD_DIR=/home/o'brien/socrates is left out: conda would write it unescaped
        into its activation script. With only quoted values no call is made.
        """
        calls = []
        monkeypatch.setattr(
            install_env.subprocess, 'run', lambda cmd, **kw: calls.append((cmd, kw))
        )
        assert bind_to_conda_env('/envs/proteus', _VALUES) is True
        ((cmd, kw),) = calls
        assert cmd[:7] == ['conda', 'env', 'config', 'vars', 'set', '-p', '/envs/proteus']
        assert cmd[7:] == [f'{n}={v}' for n, v in _VALUES.items() if n != 'RAD_DIR']
        assert kw['check'] is True
        assert kw['stdin'] is subprocess.DEVNULL
        assert quoted_names(_VALUES) == ['RAD_DIR']
        assert bind_to_conda_env('/envs/proteus', {'RAD_DIR': _VALUES['RAD_DIR']}) is False
        assert len(calls) == 1

    def test_conda_writes_a_value_into_its_activation_script_without_escaping(self):
        """conda's POSIX template is export %s='%s', so a value with a quote breaks the script.

        This reads the template from the conda that is importable here; the line it
        makes for /home/o'brien does not parse in bash, the exports-file line does.
        """
        activate = pytest.importorskip(
            'conda.activate', reason='conda is not importable in this Python environment'
        )
        template = getattr(activate.PosixActivator, 'export_var_tmpl', None)
        if template is None:
            template = activate.PosixActivator().export_var_tmpl
        assert template == "export %s='%s'"
        value = "/home/o'brien/socrates"
        broken = template % ('RAD_DIR', value)
        check = ['bash', '--noprofile', '--norc', '-n', '-c']
        assert subprocess.run([*check, broken], capture_output=True).returncode != 0
        quoted = f'export RAD_DIR={install_env._sh_quote(value)}'
        assert subprocess.run([*check, quoted], capture_output=True).returncode == 0

    @pytest.mark.parametrize(
        'error',
        [subprocess.CalledProcessError(1, 'conda'), FileNotFoundError('conda')],
        ids=['conda-fails', 'conda-missing'],
    )
    def test_failed_bind_returns_false_and_keeps_the_exports_file(
        self, monkeypatch, tmp_path, error
    ):
        """A failing conda call reports False; the exports file is still written."""

        def fail(cmd, **kw):
            raise error

        monkeypatch.setenv('CONDA_PREFIX', '/envs/proteus')
        monkeypatch.delenv('PIXI_ENVIRONMENT_NAME', raising=False)
        monkeypatch.setattr(install_env.shutil, 'which', lambda exe: '/bin/conda')
        monkeypatch.setattr(install_env.subprocess, 'run', fail)
        path, prefix, bound = export_install_vars(tmp_path, _VALUES)
        assert (prefix, bound) == ('/envs/proteus', False)
        monkeypatch.undo()  # the bash that sources the file needs the real subprocess.run
        assert _sourced(path, list(_VALUES)) == _VALUES


class TestReloadInstruction:
    """The command that loads the variables into a shell."""

    def test_bound_env_is_deactivated_and_activated(self, monkeypatch):
        """A bound env is reactivated by name, a run without conda sources the file."""
        monkeypatch.setenv('CONDA_DEFAULT_ENV', 'proteus')
        assert reload_instruction(True) == 'conda deactivate && conda activate proteus'
        assert reload_instruction(False) == 'source .proteus-env.sh'
        assert 'conda' not in reload_instruction(False)

    def test_env_without_name_uses_its_prefix(self, monkeypatch):
        """An env activated by path, without CONDA_DEFAULT_ENV, is reactivated by its prefix.

        With neither variable set the command carries a placeholder.
        """
        monkeypatch.delenv('CONDA_DEFAULT_ENV', raising=False)
        monkeypatch.setenv('CONDA_PREFIX', '/envs/p')
        assert reload_instruction(True) == 'conda deactivate && conda activate /envs/p'
        monkeypatch.delenv('CONDA_PREFIX')
        assert reload_instruction(True) == 'conda deactivate && conda activate <env>'


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (None, ['~/.julia']),
        ('', []),
        (':', ['~/.julia']),
        ('/x', ['/x']),
        ('/x:', ['/x']),
        ('/x::', ['/x']),
        ('/x:/y:', ['/x', '/y']),
        (':/x', ['~/.julia', '/x']),
        ('/x:/x', ['/x']),
        ('~/d:', ['~/d']),
    ],
)
def test_julia_depot_entries_follow_julia(monkeypatch, tmp_path, value, expected):
    """The entries equal what Julia 1.13 reads, the bundled depots left out."""
    monkeypatch.setenv('HOME', str(tmp_path))
    entries = julia_depot_entries(value)
    assert entries == [e.replace('~', str(tmp_path)) for e in expected]
    assert not any('~' in e for e in entries)


def test_julia_depot_entries_hold_for_any_value():
    """Any value gives distinct non-empty folders, and only the empty value gives none."""
    hypothesis = pytest.importorskip('hypothesis')
    strategies = pytest.importorskip('hypothesis.strategies')
    parts = strategies.sampled_from(['/a', '/b', ':', ' ', '~', 'é'])

    @hypothesis.settings(max_examples=200)
    @hypothesis.given(strategies.lists(parts, max_size=12).map(''.join))
    def check(value):
        entries = julia_depot_entries(value)
        assert all(entries)
        assert len(entries) == len(set(entries))
        assert not any(':' in e for e in entries)
        assert (entries == []) == (value == '')

    check()


@pytest.mark.parametrize('value', [None, '', ':', '/x:', '/x/other:'])
def test_no_depot_named_julia_depot_is_no_other_install(tmp_path, value):
    """Only the name of the installer can mark the depot of an install."""
    assert depot_of_other_install(value, tmp_path) is False
    (tmp_path / 'conda-meta').mkdir()
    assert depot_of_other_install(value, tmp_path) is False


def test_other_install_needs_a_marker_beside_the_depot(monkeypatch, tmp_path):
    """The same name on a big disk stays the choice of the user; an env beside it does not."""
    monkeypatch.delenv('CONDA_PREFIX', raising=False)
    big = tmp_path / 'big'
    assert depot_of_other_install(f'{big}/julia_depot:', tmp_path) is False
    big.mkdir()
    assert depot_of_other_install(f'{big}/julia_depot:', tmp_path) is False
    (big / 'conda-meta').mkdir()
    assert depot_of_other_install(f'{big}/julia_depot:', tmp_path) is True
    assert depot_of_other_install(f'{big}/julia_depot:', big) is False


def test_missing_parent_counts_as_gone_only_in_an_envs_folder(monkeypatch, tmp_path):
    """An env removed from an envs folder is gone; any other missing folder can still come."""
    monkeypatch.delenv('CONDA_PREFIX', raising=False)
    gone = tmp_path / 'envs' / 'gone'
    assert depot_of_other_install(f'{gone}/julia_depot:', tmp_path) is False
    gone.parent.mkdir()
    assert depot_of_other_install(f'{gone}/julia_depot:', tmp_path) is True
    assert 'is gone' in install_env.other_depot_words(f'{gone}/julia_depot:')
    gone.mkdir()
    assert depot_of_other_install(f'{gone}/julia_depot:', tmp_path) is False
    (gone / 'conda-meta').mkdir()
    assert 'another PROTEUS install' in install_env.other_depot_words(f'{gone}/julia_depot:')


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (None, False),
        ('', False),
        (':', False),
        (':rel', False),
        ('/x', False),
        ('/x:rel', False),
        ('~', False),
        ('~/d:', False),
        ('rel', True),
        ('rel:/x', True),
        ('./x:', True),
        ('~nobody/x:', True),
        ('~nobody', True),
        ('$CONDA_PREFIX/julia_depot:', True),
        ('/a$b:', True),
        ('~/a$b:', True),
    ],
)
def test_depot_value_is_unusable_reads_the_first_entry(value, expected):
    """A relative entry, another user's home and a literal $ give no fixed folder."""
    assert depot_value_is_unusable(value) is expected
    assert depot_value_is_unusable(f'{value or ""}:/later') is expected


def _write_project(folder, project, manifest):
    """Write the Project.toml and Manifest.toml of a Julia project."""
    folder.mkdir()
    (folder / 'Project.toml').write_text(project)
    (folder / 'Manifest.toml').write_text(manifest)


def test_missing_packages_are_the_direct_dependencies_with_a_tree_hash(tmp_path):
    """A stdlib and a dev'd path have no folder under packages and do not count."""
    project = tmp_path / 'p'
    _write_project(
        project,
        '[deps]\nFoo = "1"\nBar = "2"\nLinearAlgebra = "3"\nDev = "4"\nNotInManifest = "5"\n'
        '[weakdeps]\nWeak = "6"\n',
        'manifest_format = "2.0"\n'
        '[[deps.Foo]]\ngit-tree-sha1 = "a"\n'
        '[[deps.Bar]]\ngit-tree-sha1 = "b"\n'
        '[[deps.LinearAlgebra]]\nuuid = "3"\n'
        '[[deps.Dev]]\npath = "../dev"\n'
        '[[deps.Weak]]\ngit-tree-sha1 = "w"\n'
        '[[deps.Indirect]]\ngit-tree-sha1 = "i"\n',
    )
    (tmp_path / 'd1' / 'packages' / 'Foo').mkdir(parents=True)
    assert missing_julia_packages(project, [str(tmp_path / 'd1')]) == ['Bar']
    (tmp_path / 'd2' / 'packages' / 'Bar').mkdir(parents=True)
    assert missing_julia_packages(project, [str(tmp_path / 'd1'), str(tmp_path / 'd2')]) == []


def test_missing_packages_read_the_old_manifest_format(tmp_path):
    """Before format 2 the packages are top-level tables."""
    project = tmp_path / 'p'
    _write_project(project, '[deps]\nFoo = "1"\n', '[[Foo]]\ngit-tree-sha1 = "a"\n')
    assert missing_julia_packages(project, [str(tmp_path)]) == ['Foo']
    (tmp_path / 'packages' / 'Foo').mkdir(parents=True)
    assert missing_julia_packages(project, [str(tmp_path)]) == []


def test_project_with_no_manifest_needs_nothing(tmp_path):
    """No Manifest: the project is not set up, as for an install with no AGNI."""
    assert missing_julia_packages(tmp_path / 'none', [str(tmp_path)]) == []
    (tmp_path / 'none').mkdir()
    (tmp_path / 'none' / 'Project.toml').write_text('[deps]\nFoo = "1"\n')
    assert missing_julia_packages(tmp_path / 'none', [str(tmp_path)]) == []


@pytest.mark.parametrize(
    ('project', 'manifest'),
    [('[deps', 'manifest_format = "2.0"\n'), ('[deps]\n', 'not toml [')],
    ids=['project', 'manifest'],
)
def test_project_that_cannot_be_read_is_named(tmp_path, project, manifest):
    """A broken file gives a failure that names it, and no pass."""
    folder = tmp_path / 'p'
    _write_project(folder, project, manifest)
    with pytest.raises(ValueError, match='cannot be read'):
        missing_julia_packages(folder, [str(tmp_path)])


def test_bytes_that_are_not_utf8_cannot_be_read(tmp_path):
    """The locale of the shell plays no part: a file is decoded as UTF-8."""
    folder = tmp_path / 'p'
    _write_project(folder, '[deps]\n', 'manifest_format = "2.0"\n')
    (folder / 'Project.toml').write_bytes(b'[deps]\nFoo = "\xff"\n')
    with pytest.raises(ValueError, match='cannot be read'):
        missing_julia_packages(folder, [])


def test_project_without_a_project_file_is_named(tmp_path):
    """A Manifest with no Project.toml has no direct dependencies to read."""
    folder = tmp_path / 'p'
    folder.mkdir()
    (folder / 'Manifest.toml').write_text('manifest_format = "2.0"\n')
    with pytest.raises(ValueError, match='cannot be read'):
        missing_julia_packages(folder, [str(tmp_path)])


def _versioned_manifest(folder, text):
    """Write the Manifest of the required Julia beside the other files of the project."""
    (folder / f'Manifest-v{install_env.REQUIRED_JULIA}.toml').write_text(text)


def test_versioned_manifest_alone_is_read(tmp_path):
    """Julia reads Manifest-v<major>.<minor>.toml, so it is no project with no Manifest."""
    folder = tmp_path / 'p'
    folder.mkdir()
    (folder / 'Project.toml').write_text('[deps]\nFoo = "1"\n')
    _versioned_manifest(folder, '[[deps.Foo]]\ngit-tree-sha1 = "a"\n')
    assert missing_julia_packages(folder, [str(tmp_path)]) == ['Foo']


def test_versioned_manifest_wins_over_the_plain_one(tmp_path):
    """With both files Julia uses the versioned one, whatever the plain one records."""
    folder = tmp_path / 'p'
    _write_project(
        folder, '[deps]\nFoo = "1"\nBar = "2"\n', '[[deps.Foo]]\ngit-tree-sha1 = "a"\n'
    )
    _versioned_manifest(folder, '[[deps.Bar]]\ngit-tree-sha1 = "b"\n')
    assert missing_julia_packages(folder, [str(tmp_path)]) == ['Bar']


def test_required_julia_is_the_one_of_the_install_script():
    """The file name that doctor reads follows the version that the installer requires."""
    text = (Path(__file__).parents[2] / 'install.sh').read_text()
    major = re.search(r'^REQUIRED_JULIA_MAJOR=(\d+)$', text, re.M).group(1)
    minor = re.search(r'^REQUIRED_JULIA_MINOR=(\d+)$', text, re.M).group(1)
    assert install_env.REQUIRED_JULIA == f'{major}.{minor}'
