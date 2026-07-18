"""Unit tests for ``proteus.atmos_chem.wrapper``.

Exercises the ``run_chemistry`` dispatch function, which routes to
the correct atmospheric chemistry backend (vulcan, dummy, none) based
on the config and scheduling mode (offline, online, manually).

Invariants tested:
  - Guard: no module or 'none' module returns None without side effects
  - Guard: 'manually' scheduling returns None
  - Error contract: unknown module raises ValueError
  - Error contract: unknown scheduling mode raises ValueError
  - Dispatch: correct backend is called for vulcan/dummy in offline/online mode

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from proteus.atmos_chem.wrapper import run_chemistry

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_config(module: str | None = 'none', when: str = 'offline') -> MagicMock:
    """Build a minimal mock config for run_chemistry."""
    config = MagicMock()
    config.atmos_chem.module = module
    config.atmos_chem.when = when
    return config


# -----------------------------------------------------------------------
# Guard: disabled module
# -----------------------------------------------------------------------


def test_run_chemistry_none_module_returns_none():
    """When atmos_chem.module is 'none', run_chemistry returns None
    without calling any backend.

    Edge case: chemistry is disabled in the config.
    """
    config = _make_config(module='none')
    hf_row = {'Time': 100}
    result = run_chemistry(dirs={}, config=config, hf_row=hf_row)
    assert result is None
    assert hf_row['Time'] == 100  # hf_row untouched


def test_run_chemistry_empty_module_returns_none():
    """When atmos_chem.module is empty string, run_chemistry returns
    None. Guards against misconfigured TOML where module = ''.
    """
    config = _make_config(module='')
    hf_row = {'Time': 200}
    result = run_chemistry(dirs={}, config=config, hf_row=hf_row)
    assert result is None
    assert hf_row['Time'] == 200  # hf_row untouched


# -----------------------------------------------------------------------
# Guard: manual scheduling
# -----------------------------------------------------------------------


def test_run_chemistry_manually_skips():
    """When atmos_chem.when is 'manually', run_chemistry returns None
    even with a valid module configured.

    This mode is for user-triggered post-processing, not automatic
    per-timestep execution.
    """
    config = _make_config(module='vulcan', when='manually')
    hf_row = {'Time': 300}
    result = run_chemistry(dirs={}, config=config, hf_row=hf_row)
    assert result is None
    assert hf_row['Time'] == 300  # no side effect on hf_row


# -----------------------------------------------------------------------
# Error contract: unknown module
# -----------------------------------------------------------------------


def test_run_chemistry_unknown_module_raises(tmp_path):
    """An unrecognised module name raises ValueError after recording status.

    The dispatch uses an if/elif/else chain; the else branch must
    raise, not silently return None, and the run's status file must
    record the error state so monitoring sees the truth instead of a
    stale "Running" entry.
    """
    config = _make_config(module='nonexistent_chem', when='offline')
    hf_row = {'Time': 400}
    with pytest.raises(ValueError, match='Invalid atmos_chem module'):
        run_chemistry(dirs={'output': str(tmp_path)}, config=config, hf_row=hf_row)
    assert hf_row['Time'] == 400  # no hf_row side effect before raise
    status = (tmp_path / 'status').read_text()
    assert status.splitlines()[0].strip() == '20'  # error status recorded


# -----------------------------------------------------------------------
# Error contract: unknown scheduling mode
# -----------------------------------------------------------------------


def test_run_chemistry_unknown_when_raises(tmp_path):
    """An unrecognised scheduling mode raises ValueError after recording status.

    Valid modes are 'offline', 'online', 'manually'. Anything else
    must raise after the backend import succeeds, with the error state
    recorded in the run's status file.
    """
    config = _make_config(module='dummy', when='invalid_schedule')
    hf_row = {'Time': 500}
    with pytest.raises(ValueError, match='Invalid atmos_chem.when'):
        run_chemistry(dirs={'output': str(tmp_path)}, config=config, hf_row=hf_row)
    assert hf_row['Time'] == 500  # no hf_row side effect before raise
    status = (tmp_path / 'status').read_text()
    assert status.splitlines()[0].strip() == '20'  # error status recorded


# -----------------------------------------------------------------------
# Dispatch: offline mode
# -----------------------------------------------------------------------


@patch('proteus.atmos_chem.dummy.run_dummy_chem', return_value=True)
@patch('proteus.atmos_chem.wrapper.read_result')
def test_run_chemistry_dummy_offline_dispatches_correctly(mock_read, mock_run):
    """In offline mode with the dummy backend, run_chemistry calls
    run_dummy_chem without the online flag and reads the default
    '<module>.csv' file.

    Discrimination: online mode would pass online=True and construct
    a timestep-specific filename.
    """
    import pandas as pd

    mock_read.return_value = pd.DataFrame({'species': ['H2O'], 'vmr': [0.01]})
    config = _make_config(module='dummy', when='offline')
    dirs = {'output': '/fake/output'}

    result = run_chemistry(dirs=dirs, config=config, hf_row={})

    mock_run.assert_called_once()
    # In offline mode, filename should be None (default)
    mock_read.assert_called_once_with('/fake/output', 'dummy', filename=None)
    assert result is not None
    assert len(result) == 1


# -----------------------------------------------------------------------
# Dispatch: online mode
# -----------------------------------------------------------------------


@patch('proteus.atmos_chem.dummy.run_dummy_chem', return_value=True)
@patch('proteus.atmos_chem.wrapper.read_result')
def test_run_chemistry_dummy_online_constructs_timestep_filename(mock_read, mock_run):
    """In online mode, run_chemistry passes online=True to the backend
    and constructs a timestep-specific filename 'dummy_<Time>.csv'.

    Discrimination: offline mode would use filename=None.
    """
    import pandas as pd

    mock_read.return_value = pd.DataFrame({'species': ['CO2'], 'vmr': [0.001]})
    config = _make_config(module='dummy', when='online')
    dirs = {'output': '/fake/output'}
    hf_row = {'Time': 5000}

    result = run_chemistry(dirs=dirs, config=config, hf_row=hf_row)

    # online=True passed to the backend
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args
    assert call_kwargs[1].get('online') is True or (
        len(call_kwargs[0]) >= 4 and call_kwargs[0][3] is True
    )
    # Timestep-specific filename
    mock_read.assert_called_once_with('/fake/output', 'dummy', filename='dummy_5000.csv')
    assert result is not None


# -----------------------------------------------------------------------
# Backend failure
# -----------------------------------------------------------------------


@patch('proteus.atmos_chem.dummy.run_dummy_chem', return_value=False)
def test_run_chemistry_backend_failure_returns_none(mock_run):
    """When the backend returns False (solver failure), run_chemistry
    returns None without attempting to read output.

    Edge case: VULCAN fails to converge, or the dummy backend
    encounters an error.
    """
    config = _make_config(module='dummy', when='offline')
    hf_row = {'Time': 600}
    result = run_chemistry(dirs={'output': '/fake'}, config=config, hf_row=hf_row)
    assert result is None
    assert hf_row['Time'] == 600  # no side effect on failure
