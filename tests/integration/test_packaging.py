"""Packaging tests for the dataset manifest PROTEUS ships.

The manifest and its checksum registries are read from the installed package
rather than from the source tree, so they have to travel in the built artifact.
Declaring them as package data is necessary but not sufficient: package
discovery, the build backend, or a file that never got tracked can each drop
them from the wheel while every source-tree check still passes, and the failure
appears only once a user installs it.

This file builds a real wheel, so it sits in the integration tier.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from proteus.data import EXOPLANET_REFERENCE, MASS_RADIUS_ZENG_2019

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_built_wheel_carries_the_manifest_and_registries(tmp_path):
    """A built wheel contains the manifest and both committed registries.

    Without them the package imports cleanly and fails at the first fetch with a
    missing-manifest error, which points the reader at the wrong problem.
    """
    build = subprocess.run(
        [
            sys.executable,
            '-m',
            'pip',
            'wheel',
            '--no-deps',
            '-w',
            str(tmp_path),
            str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, f'wheel build failed: {build.stderr[-2000:]}'

    wheels = list(tmp_path.glob('*.whl'))
    assert len(wheels) == 1, f'expected one wheel, got {[w.name for w in wheels]}'
    names = set(zipfile.ZipFile(wheels[0]).namelist())

    assert 'proteus/data/proteus_manifest.toml' in names
    assert f'proteus/data/{EXOPLANET_REFERENCE}.registry.txt' in names
    assert f'proteus/data/{MASS_RADIUS_ZENG_2019}.registry.txt' in names
    # Discrimination: a wheel that shipped the manifest but no registry would
    # satisfy a looser check while failing every checksum verification.
    assert len([n for n in names if n.endswith('.registry.txt')]) == 2
