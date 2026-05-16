"""Unit tests for ``proteus.plot.cpl_global``.

Covers the early-return guard and the entry-wrapper dispatch for
``plot_global``. The full plot path uses ~25 hf columns (gas_list +
fluxes + temperatures + radii) and is exercised by integration tests
in nightly tier; this unit test scope is the lightweight branches.

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import proteus.plot.cpl_global as global_mod

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _install_mock_plt(monkeypatch):
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, MagicMock())
    monkeypatch.setattr(global_mod, 'plt', mock_plt)
    return mock_plt, mock_fig


def _make_config():
    cfg = MagicMock()
    cfg.orbit.s0_factor = 0.375
    cfg.orbit.zenith_angle = 48.2
    cfg.params.out.plot_fmt = 'pdf'
    return cfg


def test_plot_global_returns_early_when_max_time_below_three(monkeypatch, tmp_path):
    """A helpfile whose maximum Time is < 3 yr triggers the early-return
    log message and the function returns without creating a figure.
    Discrimination: subplots not called.
    """
    mock_plt, _ = _install_mock_plt(monkeypatch)
    hf = pd.DataFrame({'Time': np.linspace(0.1, 2.5, 5)})

    global_mod.plot_global(hf, str(tmp_path), config=_make_config())

    assert mock_plt.subplots.call_count == 0


def test_plot_global_entry_invokes_plot_global_twice_for_log_and_linear_axes(monkeypatch, tmp_path):
    """The entry wrapper reads runtime_helpfile.csv and calls plot_global
    twice, once with logt=True and once with logt=False. A regression
    that dropped the second call would lose the linear-time variant.
    Discrimination: two calls, with logt values [True, False] in order.
    """
    captured_calls = []

    def fake_plot(hf_all, output_dir, config, logt=True, tmin=1e3):
        captured_calls.append({'logt': logt, 'output_dir': output_dir, 'config': config})

    monkeypatch.setattr(global_mod, 'plot_global', fake_plot)
    monkeypatch.setattr(pd, 'read_csv', lambda *_a, **_kw: pd.DataFrame({'Time': np.linspace(1, 1e8, 5)}))

    handler = MagicMock()
    handler.directories = {'output': str(tmp_path)}
    handler.config = _make_config()

    global_mod.plot_global_entry(handler)

    # Discrimination: exactly two calls in [True, False] order
    assert len(captured_calls) == 2
    assert captured_calls[0]['logt'] is True
    assert captured_calls[1]['logt'] is False
    assert captured_calls[0]['output_dir'] == str(tmp_path)
    assert captured_calls[0]['config'] is handler.config
