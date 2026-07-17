"""Unit tests for ``proteus.plot.cpl_global``.

Covers the early-return guard and the entry-wrapper dispatch for
``plot_global``. The full plot path uses ~25 hf columns (gas_list +
fluxes + temperatures + radii) and is exercised by integration tests
in nightly tier; this unit test scope is the lightweight branches.

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
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
    assert mock_plt.savefig.call_count == 0


def test_plot_global_entry_invokes_plot_global_twice_for_log_and_linear_axes(
    monkeypatch, tmp_path
):
    """The entry wrapper reads runtime_helpfile.csv and calls plot_global
    twice, once with logt=True and once with logt=False. A regression
    that dropped the second call would lose the linear-time variant.
    Discrimination: two calls, with logt values [True, False] in order.
    """
    captured_calls = []

    def fake_plot(hf_all, output_dir, config, logt=True, tmin=1e3):
        captured_calls.append({'logt': logt, 'output_dir': output_dir, 'config': config})

    monkeypatch.setattr(global_mod, 'plot_global', fake_plot)
    monkeypatch.setattr(
        pd, 'read_csv', lambda *_a, **_kw: pd.DataFrame({'Time': np.linspace(1, 1e8, 5)})
    )

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


def _full_helpfile(n=8):
    """A synthetic runtime_helpfile.csv with every column ``plot_global``
    reads. One gas (H2O) is present (vmr >= 1e-10) so the per-volatile
    branch is exercised; the others sit below the presence threshold.
    """
    from proteus.utils.constants import gas_list

    time = np.logspace(2, 8, n)
    base = {
        'Time': time,
        'F_ins': np.full(n, 1366.0),
        'albedo_pl': np.full(n, 0.3),
        'F_radio': np.linspace(0.01, 0.1, n),
        'F_tidal': np.linspace(0.0, 0.05, n),
        'F_int': np.linspace(120.0, 50.0, n),
        'F_atm': np.linspace(115.0, 48.0, n),
        'F_olr': np.linspace(250.0, 240.0, n),
        'T_surf': np.linspace(3000.0, 1500.0, n),
        'T_magma': np.linspace(3200.0, 1800.0, n),
        'RF_depth': np.linspace(0.0, 0.4, n),
        'Phi_global': np.linspace(1.0, 0.0, n),
        'P_surf': np.linspace(300.0, 100.0, n),
    }
    for vol in gas_list:
        is_present = vol == 'H2O'
        base[f'{vol}_vmr'] = np.full(n, 0.5 if is_present else 1e-12)
        base[f'{vol}_bar'] = np.full(n, 50.0 if is_present else 0.0)
        base[f'{vol}_mol_atm'] = np.full(n, 1.0e22 if is_present else 0.0)
        base[f'{vol}_mol_total'] = np.full(n, 2.0e22 if is_present else 0.0)
    return pd.DataFrame(base)


def test_plot_global_writes_log_axes_figure(monkeypatch, tmp_path):
    """With a full helpfile, ``plot_global`` reaches savefig. Discrimination:
    the log-axis branch sets xscale='log' on every panel via ax.set_xscale;
    the saved filename includes ``_log`` and the configured plot format.
    """
    mock_plt, fig = _install_mock_plt(monkeypatch)
    # subplots returns (fig, 2D-array-of-axes); build a real ndarray of
    # MagicMock axes that supports [i][j] indexing.
    axes = np.empty((3, 2), dtype=object)
    for i in range(3):
        for j in range(2):
            axes[i, j] = MagicMock()
    mock_plt.subplots.return_value = (fig, axes)
    (tmp_path / 'plots').mkdir()

    config = _make_config()
    config.interior_struct = MagicMock()
    config.interior_struct.core_frac = 0.325
    config.params.out.plot_fmt = 'pdf'

    global_mod.plot_global(_full_helpfile(), str(tmp_path), config, logt=True)

    assert fig.savefig.call_count == 1
    saved_path = fig.savefig.call_args.args[0]
    assert saved_path.endswith('plot_global_log.pdf')
    # Discriminator: every panel had xscale set to 'log' in the logt
    # branch. ax_tl is axes[0,0]; check it received set_xscale('log').
    axes[0, 0].set_xscale.assert_any_call('log')


def test_plot_global_writes_linear_axes_figure(monkeypatch, tmp_path):
    """``logt=False`` skips the log-xscale loop and saves with ``_lin``
    in the filename. Discrimination: set_xscale is not called with
    'log' on the top-left axis.
    """
    mock_plt, fig = _install_mock_plt(monkeypatch)
    axes = np.empty((3, 2), dtype=object)
    for i in range(3):
        for j in range(2):
            axes[i, j] = MagicMock()
    mock_plt.subplots.return_value = (fig, axes)
    (tmp_path / 'plots').mkdir()

    config = _make_config()
    config.interior_struct = MagicMock()
    config.interior_struct.core_frac = 0.325
    config.params.out.plot_fmt = 'png'

    global_mod.plot_global(_full_helpfile(), str(tmp_path), config, logt=False)

    saved_path = fig.savefig.call_args.args[0]
    assert saved_path.endswith('plot_global_lin.png')
    # Discriminator: the top-left axis never receives set_xscale('log');
    # it still receives set_yscale('symlog', ...) which lives outside
    # the logt block.
    log_calls = [
        c for c in axes[0, 0].set_xscale.call_args_list if c.args and c.args[0] == 'log'
    ]
    assert log_calls == []
