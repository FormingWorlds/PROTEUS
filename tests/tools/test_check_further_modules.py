"""Unit tests for ``tools/check_further_modules.py``.

Checks imports of the four further modules, validates the doctor diagnostic
report check, tests the Obliqua loader wrapper, exercises the CLI entrypoint,
and verifies the Linux documentation install workflow configuration.

See ``docs/How-to/testing.md`` for test rules.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _REPO_ROOT / 'tools' / 'check_further_modules.py'

_spec = importlib.util.spec_from_file_location('check_further_modules', _TOOL)
assert _spec is not None and _spec.loader is not None
cfm = importlib.util.module_from_spec(_spec)
sys.modules['check_further_modules'] = cfm
_spec.loader.exec_module(cfm)


def test_check_imports_handles_clean_and_failing_imports():
    """Import check catches failures per module and reports clean status when all import."""
    called_modules: list[str] = []

    def fake_importer(name: str):
        called_modules.append(name)
        if name == 'vulcan':
            raise ImportError('no vulcan')
        return object()

    problems = cfm.check_imports(importer=fake_importer)
    assert problems == ['vulcan does not import: ImportError: no vulcan']
    assert called_modules == list(cfm.MODULES)

    clean_called: list[str] = []

    def clean_importer(name: str):
        clean_called.append(name)
        return object()

    assert cfm.check_imports(importer=clean_importer) == []
    assert clean_called == ['morrigan', 'vulcan', 'atmodeller']


def test_check_doctor_verifies_pass_status_and_reports_anomalies():
    """Doctor check verifies pass status for each module and flags missing or warning checks."""
    clean_results = [
        types.SimpleNamespace(name=name, status='pass', message='ok')
        for name in cfm.DOCTOR_NAMES
    ]
    assert cfm.check_doctor(clean_results) == []

    warn_results = [
        types.SimpleNamespace(name='Obliqua', status='warn', message='not installed')
        if name == 'Obliqua'
        else types.SimpleNamespace(name=name, status='pass', message='ok')
        for name in cfm.DOCTOR_NAMES
    ]
    warn_problems = cfm.check_doctor(warn_results)
    assert warn_problems == ['Obliqua: doctor says warn: not installed']

    missing_atmodeller = [
        types.SimpleNamespace(name=name, status='pass', message='ok')
        for name in cfm.DOCTOR_NAMES
        if name != 'atmodeller'
    ]
    missing_problems = cfm.check_doctor(missing_atmodeller)
    assert missing_problems == ['atmodeller: not in the doctor report']

    message_results = [
        types.SimpleNamespace(name='Obliqua', status='fail', message='broken build')
        if name == 'Obliqua'
        else types.SimpleNamespace(name=name, status='pass', message='ok')
        for name in cfm.DOCTOR_NAMES
    ]
    message_problems = cfm.check_doctor(message_results)
    assert message_problems == ['Obliqua: doctor says fail: broken build']


def test_check_obliqua_loads_through_proteus_mock_and_preserves_sys_modules(
    tmp_path,
):
    """Obliqua loader wrapper passes custom path mapping and catches runtime errors."""
    assert 'proteus.orbit.obliqua' not in sys.modules

    received_dirs: list[dict[str, str]] = []

    def failing_loader(custom_paths: dict[str, str]):
        received_dirs.append(custom_paths)
        raise RuntimeError('no project')

    problems = cfm.check_obliqua(tmp_path, loader=failing_loader)
    assert problems == ['Obliqua does not load through PROTEUS: RuntimeError: no project']
    assert received_dirs == [{'obliqua': str(tmp_path / 'Obliqua')}]

    clean_dirs: list[dict[str, str]] = []

    def clean_loader(custom_paths: dict[str, str]):
        clean_dirs.append(custom_paths)
        return object()

    assert cfm.check_obliqua(tmp_path, loader=clean_loader) == []
    assert clean_dirs == [{'obliqua': str(tmp_path / 'Obliqua')}]
    assert 'proteus.orbit.obliqua' not in sys.modules


def test_main_clean_exit_code_and_output(monkeypatch, tmp_path, capsys):
    """Main CLI exits zero when all checks pass and outputs summary message."""
    monkeypatch.setattr(cfm, 'check_imports', lambda importer=None: [])
    monkeypatch.setattr(cfm, 'run_all_checks', lambda: [])
    monkeypatch.setattr(cfm, 'check_doctor', lambda results: [])
    monkeypatch.setattr(cfm, 'check_obliqua', lambda root, loader=None: [])

    exit_code = cfm.main(['--root', str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert 'All 4 further modules are usable.' in captured.out
    assert captured.err == ''


def test_main_failing_exit_code_and_no_short_circuit(monkeypatch, tmp_path, capsys):
    """Main CLI exits 1 on failure, writes errors to stderr, and never aborts early."""
    check_calls: list[str] = []

    def fake_imports(importer=None):
        check_calls.append('imports')
        return ['vulcan does not import: ImportError: no vulcan']

    def fake_doctor(results):
        check_calls.append('doctor')
        return ['Obliqua: doctor says warn: not installed']

    def fake_obliqua(root, loader=None):
        check_calls.append('obliqua')
        return ['Obliqua does not load through PROTEUS: RuntimeError: failed']

    monkeypatch.setattr(cfm, 'check_imports', fake_imports)
    monkeypatch.setattr(cfm, 'run_all_checks', lambda: [])
    monkeypatch.setattr(cfm, 'check_doctor', fake_doctor)
    monkeypatch.setattr(cfm, 'check_obliqua', fake_obliqua)

    exit_code = cfm.main(['--root', str(tmp_path)])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert check_calls == ['imports', 'doctor', 'obliqua']
    assert 'vulcan does not import: ImportError: no vulcan' in captured.err
    assert 'Obliqua: doctor says warn: not installed' in captured.err
    assert 'Obliqua does not load through PROTEUS: RuntimeError: failed' in captured.err
    assert 'All 4 further modules are usable.' not in captured.out


def test_workflow_further_modules_check_step_configuration():
    """Verify that docs-install-linux.yml has the check step properly configured."""
    workflow_path = _REPO_ROOT / '.github' / 'workflows' / 'docs-install-linux.yml'
    workflow_text = workflow_path.read_text(encoding='utf-8')

    try:
        import yaml

        data = yaml.safe_load(workflow_text)
        steps = data['jobs']['install']['steps']
        matching_steps = [
            s
            for s in steps
            if s.get('name') == 'Check that the four further modules are usable'
        ]
        assert len(matching_steps) == 1
        step = matching_steps[0]
        step_index = steps.index(step)

        run_script_indices = [
            i
            for i, s in enumerate(steps)
            if 'bash -e "$RUNNER_TEMP/install_steps.sh"' in s.get('run', '')
        ]
        assert len(run_script_indices) == 1
        assert step_index > run_script_indices[0]

        step_run = step.get('run', '')
        assert 'tools/check_further_modules.py --root "$HOME/PROTEUS"' in step_run
        assert 'conda activate proteus' in step_run
        assert 'source "$HOME/miniforge3/etc/profile.d/conda.sh"' in step_run
        assert step.get('continue-on-error') is not True
        assert 'if' not in step
    except ImportError:
        # Fallback to text inspection if pyyaml is unavailable
        assert 'Check that the four further modules are usable' in workflow_text

    # Text assertions apply unconditionally
    step_marker = '- name: Check that the four further modules are usable'
    assert step_marker in workflow_text
    after_step = workflow_text.split(step_marker, 1)[1]
    step_block = after_step.split('- name:', 1)[0]

    assert 'continue-on-error' not in step_block
    assert 'if:' not in step_block
    assert 'python tools/check_further_modules.py --root "$HOME/PROTEUS"' in step_block
    assert 'conda activate proteus' in step_block
    assert 'source "$HOME/miniforge3/etc/profile.d/conda.sh"' in step_block
    assert workflow_text.find('bash -e "$RUNNER_TEMP/install_steps.sh"') < workflow_text.find(
        step_marker
    )
