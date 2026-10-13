"""
Unit tests for helpers of the CHILI comparison plot script.

These tests cover the run label and the smoothing helper of
``tools/plot_chili_comparison.py`` without drawing a figure or reading
the comparison data.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.fixture(scope='module')
def tool():
    """The co-located ``plot_chili_comparison.py``, loaded as a module."""
    script_path = Path(__file__).resolve().parents[2] / 'tools' / 'plot_chili_comparison.py'
    spec = importlib.util.spec_from_file_location('plot_chili_under_test', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_with_log(path: Path, sha: str) -> Path:
    path.mkdir()
    (path / 'proteus_00.log').write_text(
        f'[ INFO  ] PROTEUS version   26.7.14\n[ INFO  ] PROTEUS git hash  {sha}\n'
    )
    return path


def test_run_label_names_the_commit_of_the_run_log(tool, tmp_path, monkeypatch):
    """The label carries the commit that the run log names, not the commit of the
    checkout that draws the figure; two runs of different commits give both."""
    monkeypatch.setattr(
        tool.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='c0ffee1\n')
    )
    earth = _run_with_log(tmp_path / 'earth', 'd0c7f08062bdf08631b61225bad460c6d4e381eb')
    venus = _run_with_log(tmp_path / 'venus', '81e1514466a590a9fd97acea99150ea5d978a751')

    assert tool._get_proteus_sha(earth, None) == 'd0c7f080'
    assert tool._get_proteus_sha(earth, earth) == 'd0c7f080'
    assert tool._get_proteus_sha(earth, venus) == '81e15144/d0c7f080'


def test_run_label_falls_back_to_the_checkout_without_a_run_log(tool, tmp_path, monkeypatch):
    """A run folder without a log, and no run at all, give the commit of the checkout."""
    monkeypatch.setattr(
        tool.subprocess, 'run', lambda *a, **k: SimpleNamespace(stdout='c0ffee1\n')
    )
    (tmp_path / 'empty').mkdir()

    assert tool._get_proteus_sha(tmp_path / 'empty', None) == 'c0ffee1'
    assert tool._get_proteus_sha() == 'c0ffee1'


@pytest.mark.parametrize('log', [False, True])
def test_smooth_accepts_a_helpfile_column_and_keeps_the_end_points(tool, log):
    """A column of a helpfile (a read-only array in pandas 3) is smoothed without
    an error; the first and last value stay exact and the middle is averaged."""
    column = pd.DataFrame({'y': [1.0, 100.0, 1.0, 100.0, 1.0, 100.0, 8.0]})['y'].values

    out = tool._smooth(column, window=3, log=log)

    assert (out[0], out[-1]) == (column[0], column[-1])
    window = [100.0, 1.0, 100.0]
    middle = 10 ** np.mean(np.log10(window)) if log else np.mean(window)
    assert out[2] == pytest.approx(middle)
    assert out[2] != pytest.approx(column[2])
