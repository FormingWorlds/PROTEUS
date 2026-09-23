"""Subprocess checks that PROTEUS imports aragog cleanly without leaking JAX.

Verifies:
1. Importing proteus or proteus.config does not load jax or equinox into sys.modules.
2. Importing proteus.interior_energetics.aragog_phase loads aragog cleanly
   without importing jax or equinox.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.unit


def test_proteus_import_clean_of_jax():
    """Subprocess check that import proteus leaves jax and equinox out of sys.modules."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
    code = (
        'import sys\n'
        'import proteus\n'
        'import proteus.config\n'
        "assert 'jax' not in sys.modules, f'jax leaked: {sys.modules.get(\"jax\")}'\n"
        "assert 'equinox' not in sys.modules, f'equinox leaked: {sys.modules.get(\"equinox\")}'\n"
    )
    res = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f'Import test failed:\n{res.stderr}'


def test_aragog_phase_import_clean_of_jax():
    """Subprocess check that import aragog_phase leaves jax and equinox out of sys.modules."""
    repo_root = pathlib.Path(__file__).resolve().parent.parent.parent
    code = (
        'import sys\n'
        'import proteus.interior_energetics.aragog_phase\n'
        "assert 'jax' not in sys.modules, f'jax leaked: {sys.modules.get(\"jax\")}'\n"
        "assert 'equinox' not in sys.modules, f'equinox leaked: {sys.modules.get(\"equinox\")}'\n"
    )
    res = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f'Import test failed:\n{res.stderr}'
