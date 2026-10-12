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
from matplotlib.colors import to_rgb

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
    # Must be a real number, not a MagicMock: the x-axis label formats it
    # with ':g', which raises TypeError on a mock.
    cfg.star.age_ini = 4.5
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


def _full_helpfile(n=8, n_gases=1):
    """A synthetic runtime_helpfile.csv with every column ``plot_global``
    reads. The first ``n_gases`` gases are present (vmr >= 1e-10), each at
    its own level, so the per-volatile branch is exercised; the others sit
    below the presence threshold.
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
    for i, vol in enumerate(gas_list):
        is_present = i < n_gases
        base[f'{vol}_vmr'] = np.full(n, 0.5 if is_present else 1e-12)
        base[f'{vol}_bar'] = np.full(n, 50.0 / 2**i if is_present else 0.0)
        base[f'{vol}_mol_atm'] = np.full(n, 1.0e22 if is_present else 0.0)
        base[f'{vol}_mol_total'] = np.full(n, (2.0 + i) * 1.0e22 if is_present else 0.0)
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
    config.star.age_ini = 4.5
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


def test_plot_global_takes_its_neutral_colours_from_the_style(draw_real, tmp_path, style):
    """On real axes the total pressure line has the text colour of the style, the
    net atmosphere flux, surface temperature and melt fraction curves its dark
    grey: black and '#444444' in the default style."""
    (tmp_path / 'plots').mkdir()
    config = _make_config()
    config.interior_struct.core_frac = 0.325

    (fig,) = draw_real(
        global_mod.plot_global, _full_helpfile(), str(tmp_path), config, rc=style.rc
    )

    panel = {ax.get_title(loc='left')[:3]: ax for ax in fig.axes}
    assert sorted(panel) == ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    line = {ln.get_label(): ln for ax in fig.axes for ln in ax.lines}
    assert to_rgb(line['Total'].get_color()) == pytest.approx(to_rgb(style.text))
    curves = [line['Net (atm.)'], line['Melt fraction'], panel['(c)'].lines[1]]
    for curve in curves:
        assert to_rgb(curve.get_color()) == pytest.approx(to_rgb(style.dark_grey))
    for ax in fig.axes:
        assert to_rgb(ax._left_title.get_color()) == pytest.approx(to_rgb(style.text))
    # Module and species colours do not follow the style.
    assert to_rgb(panel['(c)'].lines[0].get_color()) == pytest.approx(to_rgb('#ff7f0e'))


# Font sizes of a presentation style, larger than the matplotlib defaults.
_LARGE_FONTS = {'font.size': 14.5, 'axes.labelsize': 18, 'legend.fontsize': 14.5}


@pytest.mark.parametrize('logt', [True, False], ids=['log_time', 'linear_time'])
@pytest.mark.parametrize('rc', [None, _LARGE_FONTS], ids=['default_fonts', 'large_fonts'])
@pytest.mark.parametrize('n_gases', [1, 6, 7], ids=['one_gas', 'six_gases', 'seven_gases'])
def test_plot_global_text_boxes_do_not_collide(draw_real, tmp_path, logt, rc, n_gases):
    """Text geometry of the drawn figure, for both time axes, two font sizes and
    both small gas legends: tick labels of different panels do not intersect, each
    panel label is outside every data area and clear of every legend, the y label
    of the top row is no taller than its panel, so it cannot leave the figure,
    and each legend lies on no more curves than its best place needs."""
    (tmp_path / 'plots').mkdir()
    config = _make_config()
    config.interior_struct.core_frac = 0.325
    config.params.out.plot_fmt = 'png'

    (fig,) = draw_real(
        global_mod.plot_global,
        _full_helpfile(n_gases=n_gases),
        str(tmp_path),
        config,
        logt=logt,
        rc=rc,
    )
    # Measure the figure as savefig draws it.
    fig.set_dpi(200)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    ticks = []
    for i, ax in enumerate(fig.axes):
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            for tick in axis.get_major_ticks():
                if not min(lim) <= tick.get_loc() <= max(lim):
                    continue
                for label in (tick.label1, tick.label2):
                    if label.get_visible() and label.get_text():
                        ticks.append((i, label.get_text(), label.get_window_extent(renderer)))
    # Guard: the labels were found, so an empty list of collisions means something.
    assert len({i for i, _, _ in ticks}) == 6
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    assert len(legends) == 3
    assert legends[2]._ncols == (1 if n_gases <= 6 else 2)
    titles = [ax._left_title for ax in fig.axes]
    boxes = [t.get_window_extent(renderer) for t in titles]
    top_left = fig.axes[0]
    under = [
        sum(
            line.get_transform()
            .transform_path(line.get_path())
            .intersects_bbox(ax.get_legend().get_window_extent(renderer), filled=False)
            for line in ax.lines
        )
        for ax in (fig.axes[0], fig.axes[4], fig.axes[5])
    ]
    # Curves under each legend at its best place in this helpfile, as measured: lines
    # that cross the whole panel leave the mantle and the one-column gas legend no free place.
    bounds = (0, (1 if logt else 0) + (1 if rc else 0), {1: 0, 6: 2, 7: 0}[n_gases])
    # Alignment of every x tick label in view of the bottom row, left to right.
    aligned = [
        [
            label.get_horizontalalignment()
            for label in ax.get_xticklabels()
            if min(ax.get_xlim()) <= label.get_position()[0] <= max(ax.get_xlim())
        ]
        for ax in (fig.axes[4], fig.axes[5])
    ]
    tick_boxes = [box for _, _, box in ticks]

    found = {
        'panel labels': [t.get_text()[:3] for t in titles],
        'tick labels that intersect': [
            (a[1], b[1])
            for n, a in enumerate(ticks)
            for b in ticks[n + 1 :]
            if a[0] != b[0] and a[2].overlaps(b[2])
        ],
        'x label alignments that differ from the centre': [
            (col, n - len(row), ha)
            for col, row in enumerate(aligned)
            for n, ha in enumerate(row)
            if ha != 'center'
        ],
        'panel labels on a tick label or on each other': sum(
            a.overlaps(b) for n, a in enumerate(boxes) for b in boxes[n + 1 :] + tick_boxes
        ),
        'panel labels in a data area': sum(
            b.overlaps(ax.bbox) for b in boxes for ax in fig.axes
        ),
        'panel labels under a legend': sum(
            b.overlaps(leg.get_window_extent(renderer)) for b in boxes for leg in legends
        ),
        'top y label taller than its panel': bool(
            top_left.yaxis.label.get_window_extent(renderer).height > top_left.bbox.height
        ),
        'legends on more curves than their best place needs': [
            (n, bound) for n, bound in zip(under, bounds) if n > bound
        ],
    }
    assert found == {
        'panel labels': ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)'],
        'tick labels that intersect': [],
        # Last label of the left column; first of the right column in log time only.
        'x label alignments that differ from the centre': [(0, -1, 'right')]
        + ([(1, -len(aligned[1]), 'left')] if logt else []),
        'panel labels on a tick label or on each other': 0,
        'panel labels in a data area': 0,
        'panel labels under a legend': 0,
        'top y label taller than its panel': False,
        'legends on more curves than their best place needs': [],
    }
