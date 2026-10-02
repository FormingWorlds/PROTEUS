"""Tests for ``tools/extract_inference_truths.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _load_tool():
    """Load the co-located ``extract_inference_truths.py`` as a module."""
    script = Path(__file__).resolve().parents[2] / 'tools' / 'extract_inference_truths.py'
    spec = importlib.util.spec_from_file_location('extract_inference_truths_under_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_configs(tmp_path, helpfile_text):
    """Write an inference config, its reference config and the reference helpfile."""
    run = tmp_path / 'run'
    run.mkdir()
    (run / 'runtime_helpfile.csv').write_text(helpfile_text, encoding='utf-8')
    ref = tmp_path / 'ref.toml'
    ref.write_text(f'[params.stop.time]\nmaximum = 200.0\n[params.out]\npath = "{run}"\n')
    infer = tmp_path / 'case.infer.toml'
    infer.write_text(f'ref_config = "{ref}"\n[observables]\nR_obs = 1.0\n')
    return infer


def test_extract_prints_the_observable_at_the_row_nearest_the_target_time(
    tmp_path, monkeypatch, capsys
):
    """The value comes from the row nearest the stop time, not the last row, and from its
    own column although an empty field (an older NaN) precedes it. The tool prints ten
    digits, so this checks column placement and row choice, not float precision."""
    tool = _load_tool()
    infer = _write_configs(
        tmp_path,
        'Time\tR_xuv\tR_obs\n100.0\t\t1.25\n190.0\t\t6.371008437289124e6\n400.0\t\t9.5\n',
    )
    monkeypatch.setattr(sys, 'argv', ['extract_inference_truths.py', str(infer)])

    assert tool.main() == 0

    out = capsys.readouterr().out
    assert '"R_obs" = 6.3710084373e+06' in out
    assert 'time: 1.900e+02 years' in out


def test_extract_refuses_a_header_only_helpfile(tmp_path, monkeypatch):
    """A reference helpfile with no rows raises instead of reporting a value."""
    tool = _load_tool()
    infer = _write_configs(tmp_path, 'Time\tR_obs\n')
    monkeypatch.setattr(sys, 'argv', ['extract_inference_truths.py', str(infer)])

    with pytest.raises(ValueError, match='Helpfile has no rows') as excinfo:
        tool.main()

    assert str(tmp_path / 'run' / 'runtime_helpfile.csv') in str(excinfo.value)
