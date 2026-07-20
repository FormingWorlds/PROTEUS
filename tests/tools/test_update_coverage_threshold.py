"""Unit tests for ``tools/update_coverage_threshold.py``.

The module under test maintains the coverage ``fail_under`` thresholds against
fixed per-gate ceilings (fast 80%, full 90%). These tests exercise:

* the fixed ceilings and their agreement with the committed ``pyproject.toml``
  (a drift guard: the same numbers live in three places and must not diverge),
* the cap behaviour (a threshold can rise to, but never above, its ceiling),
* the no-op-at-ceiling short circuit,
* the validation/error contract (missing coverage file, unknown target).

See ``docs/How-to/testing.md`` and
``docs/Explanations/test_framework.md`` for the test framework.
"""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _load_module():
    """Load ``tools/update_coverage_threshold.py`` directly.

    The ``tools/`` directory is not a Python package, and adding the repo root
    to ``sys.path`` would shadow installed ecosystem packages with the local
    checkout. ``importlib.util`` loads the script in isolation.
    """
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / 'tools' / 'update_coverage_threshold.py'
    spec = importlib.util.spec_from_file_location('update_coverage_threshold_uut', script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_uct = _load_module()


_SYNTHETIC_PYPROJECT = """\
[tool.coverage.report]
fail_under = {full}

[tool.proteus.coverage_fast]
fail_under = {fast}
"""


def _write_pyproject(tmp_path: Path, *, fast: float, full: float) -> Path:
    """Write a minimal pyproject.toml carrying both fail_under thresholds."""
    p = tmp_path / 'pyproject.toml'
    p.write_text(_SYNTHETIC_PYPROJECT.format(fast=fast, full=full))
    return p


def _write_coverage(tmp_path: Path, percent: float, name: str = 'coverage.json') -> Path:
    """Write a coverage JSON whose totals report the given percent_covered."""
    import json

    p = tmp_path / name
    p.write_text(json.dumps({'totals': {'percent_covered': percent}}))
    return p


def test_ceilings_are_the_fixed_policy_values():
    """The two gates carry distinct fixed ceilings: fast at 80%, full at 90%.
    Distinct values guard against a copy-paste regression that collapses both
    gates onto a single number."""
    assert _uct.CEILINGS['fast'] == pytest.approx(80.0)
    assert _uct.CEILINGS['full'] == pytest.approx(90.0)
    # The gates must not be equal: a single shared ceiling would let the fast
    # (unit-only) gate chase the unreachable 90% and break every PR.
    assert _uct.CEILINGS['fast'] < _uct.CEILINGS['full']


def test_committed_pyproject_matches_the_ceilings():
    """The committed pyproject.toml fail_under values equal the tool's ceilings.
    The fast 80 / full 90 policy is duplicated across pyproject.toml, this tool,
    and the PR threshold guard; this test fails loudly if any copy drifts."""
    repo_root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((repo_root / 'pyproject.toml').read_text())

    committed_fast = float(data['tool']['proteus']['coverage_fast']['fail_under'])
    committed_full = float(data['tool']['coverage']['report']['fail_under'])

    assert committed_fast == pytest.approx(_uct.CEILINGS['fast'])
    assert committed_full == pytest.approx(_uct.CEILINGS['full'])


def test_read_current_coverage_parses_totals(tmp_path):
    """read_current_coverage returns the float percent_covered from the JSON,
    and raises FileNotFoundError (the error contract) when the file is absent."""
    cov = _write_coverage(tmp_path, 73.41)
    assert _uct.read_current_coverage(cov) == pytest.approx(73.41)

    with pytest.raises(FileNotFoundError):
        _uct.read_current_coverage(tmp_path / 'does_not_exist.json')


def test_main_caps_increase_at_the_fast_ceiling(tmp_path, monkeypatch):
    """With unit coverage well above the ceiling, the fast threshold is raised
    only to the 80% ceiling, never to the measured 95%. This is the discriminating
    case: a missing cap would write 95.0 instead of 80.0."""
    monkeypatch.chdir(tmp_path)
    _write_pyproject(tmp_path, fast=70.0, full=90.0)
    _write_coverage(tmp_path, 95.0, name='coverage-unit.json')
    monkeypatch.setattr(
        sys, 'argv', ['prog', '--coverage-file', 'coverage-unit.json', '--target', 'fast']
    )

    rc = _uct.main()

    assert rc == 0  # threshold updated
    data = tomllib.loads((tmp_path / 'pyproject.toml').read_text())
    written = float(data['tool']['proteus']['coverage_fast']['fail_under'])
    assert written == pytest.approx(80.0)
    # Discrimination: the cap held, the raw 95% was not written through.
    assert written < 95.0


def test_main_is_a_noop_when_already_at_ceiling(tmp_path, monkeypatch):
    """When the threshold already sits at its ceiling, the tool reports no update
    (exit 1) and leaves pyproject.toml byte-identical, so a manual run can never
    push fail_under away from the fixed value the PR guard enforces."""
    monkeypatch.chdir(tmp_path)
    _write_pyproject(tmp_path, fast=80.0, full=90.0)
    _write_coverage(tmp_path, 99.0, name='coverage-unit.json')
    before = (tmp_path / 'pyproject.toml').read_text()
    monkeypatch.setattr(
        sys, 'argv', ['prog', '--coverage-file', 'coverage-unit.json', '--target', 'fast']
    )

    rc = _uct.main()

    assert rc == 1  # no update needed
    assert (tmp_path / 'pyproject.toml').read_text() == before


def test_update_threshold_only_ratchets_upward(tmp_path, monkeypatch):
    """update_threshold_in_pyproject refuses to lower a threshold: a proposed
    value at or below the current one returns False and writes nothing."""
    monkeypatch.chdir(tmp_path)
    _write_pyproject(tmp_path, fast=78.0, full=90.0)
    before = (tmp_path / 'pyproject.toml').read_text()

    # A lower proposed value is rejected (ratchet is monotonic upward).
    assert _uct.update_threshold_in_pyproject('fast', 70.0) is False
    assert (tmp_path / 'pyproject.toml').read_text() == before

    # A strictly higher value is accepted and written.
    assert _uct.update_threshold_in_pyproject('fast', 79.5) is True
    data = tomllib.loads((tmp_path / 'pyproject.toml').read_text())
    assert float(data['tool']['proteus']['coverage_fast']['fail_under']) == pytest.approx(79.5)


def test_unknown_target_raises_value_error(tmp_path, monkeypatch):
    """An unrecognised target is a configuration error: both the read and the
    update helpers raise ValueError rather than silently picking a gate."""
    monkeypatch.chdir(tmp_path)
    _write_pyproject(tmp_path, fast=80.0, full=90.0)

    with pytest.raises(ValueError):
        _uct.read_threshold_from_pyproject('bogus')
    with pytest.raises(ValueError):
        _uct.update_threshold_in_pyproject('bogus', 81.0)
