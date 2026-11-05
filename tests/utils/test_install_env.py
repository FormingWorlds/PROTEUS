"""Tests for ``proteus.utils.install_env``.

The exports file holds one ``export NAME='value'`` line per variable, which a
fresh bash sources back to the same values; it is rewritten in place. The
conda env gets the variables without a single quote in their value in one
``conda env config vars set`` call, and only when a conda env other than
``base`` (and not a pixi env) is active and ``conda`` is on PATH.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from proteus.utils import install_env
from proteus.utils.install_env import (
    ENV_FILE_NAME,
    active_conda_prefix,
    bind_to_conda_env,
    export_install_vars,
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
