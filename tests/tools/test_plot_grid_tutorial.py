"""
Unit tests for ``tools/plot_grid_tutorial.py``.

The grid folders are synthetic: each case holds the three files the tool reads
(the resolved config, the status file and a short helpfile). The figures are
drawn on the Agg backend and their axes are read back; no image is compared.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import proteus_mpl  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_rgb, to_rgba  # noqa: E402
from matplotlib.markers import MarkerStyle  # noqa: E402
from PIL import Image  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_script = Path(__file__).resolve().parents[2] / 'tools' / 'plot_grid_tutorial.py'
_spec = importlib.util.spec_from_file_location('plot_grid_tutorial_under_test', _script)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)


def _cycle() -> list[str]:
    """Series colours of the active matplotlib style."""
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


GRID_CASES = [
    (0, 0.5, 0.03, 1000.0, 14),
    (1, 2.0, 0.03, 9000.0, 14),
    (2, 0.5, 0.1225, 5000.0, 10),
    (3, 0.5, 0.5, 1000.0, 10),
    (4, 2.0, 0.5, 9000.0, 10),
]


def _write_case(grid_dir: Path, index: int, mass, distance, h_budget, status, phi_crit=0.03):
    """Write one case folder; a ``phi_crit`` of None turns the solidification stop off."""
    case_dir = grid_dir / f'case_{index:06d}'
    case_dir.mkdir(parents=True)
    solid = (
        'enabled = false\nphi_crit = 0.01'
        if phi_crit is None
        else f'enabled = true\nphi_crit = {phi_crit}'
    )
    (case_dir / 'init_coupler.toml').write_text(
        f'[planet]\nmass_tot = {mass}\n'
        f'[planet.elements]\nH_mode = "ppmw"\nH_budget = {h_budget}\n'
        f'[orbit]\nsemimajoraxis = {distance}\n'
        f'[params.stop.solid]\n{solid}\n'
    )
    (case_dir / 'status').write_text(f'{status}\nCompleted\n')
    headers = ['Step', 'Time', 'F_atm', 'Phi_global', 'T_surf', 'P_surf', 'M_atm']
    # 2 rows of the initial stage (Time = 0), then 3 rows; P_surf differs distinctly from H_budget
    last_phi = 0.02 if status == tool.STATUS_SOLIDIFIED else 1.0
    last_t_surf = 1700.0 + 100.0 * distance + 50.0 * mass
    last_p_surf = 7.5 * h_budget + 100.0 * mass
    rows = [
        (0, 0.0, 100.0, 1.0, 4000.0, 0.1 * h_budget, 1.0e20),
        (1, 0.0, 100.0, 1.0, 4000.0, 0.1 * h_budget, 1.0e20),
        (2, 10.0, 80.0, 1.0, 3500.0, 0.5 * h_budget, 1.0e20),
        (3, 1.0e3 * mass, 50.0, 0.5, 2500.0, 0.8 * h_budget, 1.0e20),
        (4, 2.0e4 * mass, 10.0, last_phi, last_t_surf, last_p_surf, 1.0e20),
    ]
    text = (
        '\t'.join(headers)
        + '\n'
        + ''.join(
            '\t'.join(f'{v:.8e}' if isinstance(v, float) else str(v) for v in r) + '\n'
            for r in rows
        )
    )
    (case_dir / 'runtime_helpfile.csv').write_text(text)


@pytest.fixture
def grid_dir(tmp_path):
    """A grid of cases at multiple masses and distances with energy balance and solidified stops."""
    grid = tmp_path / 'grid'
    for c in GRID_CASES:
        _write_case(grid, *c)
    return grid


def test_load_cases_reads_inputs_stop_and_science_rows(grid_dir):
    """Each record holds the swept inputs, the status code and the rows after Time = 0."""
    cases = tool.load_cases(grid_dir)
    assert len(cases) == 5
    expected_inputs = [(m, d, h, s) for _, m, d, h, s in GRID_CASES]
    assert [
        (c['mass'], c['distance'], c['H_budget'], c['status']) for c in cases
    ] == expected_inputs
    assert all(c['H_mode'] == 'ppmw' and c['phi_crit'] == pytest.approx(0.03) for c in cases)
    # The 2 rows of the initial stage are dropped
    assert all(len(c['hf']) == 3 and c['hf']['Time'].min() > 0 for c in cases)
    assert cases[4]['hf']['Time'].iloc[-1] == pytest.approx(4.0e4)


def test_load_cases_gives_no_threshold_when_the_stop_is_off(tmp_path):
    """A case with the solidification stop off has no threshold, whatever `phi_crit` holds."""
    _write_case(tmp_path, 0, 1.0, 0.5, 5000.0, 14, phi_crit=None)
    cases = tool.load_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0]['phi_crit'] is None


def test_load_cases_skips_case_with_rows_of_the_initial_stage_only(grid_dir, capsys):
    """A case with rows of the initial stage only is skipped and named on the terminal."""
    _write_case(grid_dir, 5, 1.0, 0.5, 5000.0, 10)
    helpfile = grid_dir / 'case_000005' / 'runtime_helpfile.csv'
    helpfile.write_text(''.join(helpfile.read_text().splitlines(keepends=True)[:3]))
    cases = tool.load_cases(grid_dir)
    assert [c['mass'] for c in cases] == [m for _, m, _, _, _ in GRID_CASES]
    assert 'Skipped case_000005: no row after the initial stage' in capsys.readouterr().out


@pytest.mark.parametrize('filename', ['init_coupler.toml', 'status', 'runtime_helpfile.csv'])
@pytest.mark.parametrize('empty', [False, True])
def test_load_cases_skips_case_with_missing_or_empty_required_file(
    grid_dir, capsys, filename, empty
):
    """A case with one required file missing or empty is skipped and named on the terminal."""
    _write_case(grid_dir, 5, 1.0, 0.5, 5000.0, 10)
    target = grid_dir / 'case_000005' / filename
    if empty:
        target.write_text('')
    else:
        target.unlink()
    cases = tool.load_cases(grid_dir)
    assert len(cases) == 5
    assert [c['mass'] for c in cases] == [m for _, m, _, _, _ in GRID_CASES]
    printed = capsys.readouterr().out
    assert 'Skipped case_000005: an output file is missing or empty' in printed


def test_load_cases_without_a_case_folder_raises(tmp_path):
    """A folder without `case_*` is an error that names the folder."""
    (tmp_path / 'manager.log').write_text('')
    with pytest.raises(FileNotFoundError) as err:
        tool.load_cases(tmp_path)
    assert str(tmp_path) in str(err.value)
    assert 'case_*' in str(err.value)


def test_tracks_draw_the_threshold_of_the_config(grid_dir, tmp_path):
    """The dotted line sits at the `phi_crit` of the resolved configs, and its label names it."""
    out = tmp_path / 'tracks.png'
    cases = tool.load_cases(grid_dir)
    with plt.rc_context({'savefig.dpi': 123}):
        fig = tool.plot_tracks(cases, out)
    ax = fig.axes[0]
    flat = [ln for ln in ax.lines if ln.get_linestyle() == ':']
    assert len(flat) == 1
    assert list(flat[0].get_ydata()) == pytest.approx([0.03, 0.03])
    assert [t.get_text() for t in ax.texts] == [r'solidification ($\Phi$ = 0.03)']
    assert ax.get_xscale() == 'log'
    assert ax.get_xlabel() == 'Time [yr]'
    assert ax.get_ylabel() == r'Global melt fraction $\Phi_{\rm global}$'
    # 5 tracks, 3 colours (one per distance), and the legend names all 3 distances
    tracks = [ln for ln in ax.lines if ln.get_linestyle() == '-']
    assert len(tracks) == 5 and len({ln.get_color() for ln in tracks}) == 3
    dist_colors = dict(zip([0.03, 0.1225, 0.5], _cycle()))
    for c, track in zip(cases, tracks):
        assert list(track.get_xdata()) == pytest.approx(list(c['hf']['Time']))
        assert list(track.get_ydata()) == pytest.approx(list(c['hf']['Phi_global']))
        assert track.get_color() == dist_colors[c['distance']]
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        '0.03 AU',
        '0.122 AU',
        '0.5 AU',
    ]
    assert [ln.get_color() for ln in ax.get_legend().get_lines()] == _cycle()[:3]
    # The resolution is the one of the active style, not a value of the tool
    with Image.open(out) as img:
        dpi = img.info['dpi'][0]
    assert pytest.approx(123, abs=1) == dpi
    assert out.stat().st_size > 0
    plt.close(fig)


def test_tracks_draw_no_threshold_when_the_stop_is_off(tmp_path):
    """Without a solidification stop there is no dotted line and no label."""
    _write_case(tmp_path / 'grid', 0, 1.0, 0.5, 5000.0, 14, phi_crit=None)
    fig = tool.plot_tracks(tool.load_cases(tmp_path / 'grid'), tmp_path / 'tracks.png')
    ax = fig.axes[0]
    assert [ln for ln in ax.lines if ln.get_linestyle() == ':'] == []
    assert len(ax.texts) == 0
    plt.close(fig)


def test_outputs_give_pressure_in_bar_against_the_budget(grid_dir, tmp_path):
    """Panel (d) is in bar on a log axis and holds the final `P_surf` of each case."""
    out = tmp_path / 'outputs.png'
    cases = tool.load_cases(grid_dir)
    with plt.rc_context({'savefig.dpi': 123}):
        fig = tool.plot_outputs(cases, out)
    ax_d = fig.axes[3]
    assert ax_d.get_yscale() == 'log'
    # All 3 distance panels use a log x axis with one labelled tick per swept distance
    assert all(fig.axes[i].get_xscale() == 'log' for i in (0, 1, 2))
    for i in (0, 1, 2):
        axis = fig.axes[i].xaxis
        assert list(axis.get_majorticklocs()) == pytest.approx([0.03, 0.1225, 0.5])
        assert [t.get_text() for t in axis.get_majorticklabels()] == ['0.03', '0.122', '0.5']
        assert len(axis.get_minorticklocs()) == 0
    expected = [
        (
            r'(a) Final melt fraction $\Phi_{\rm global}$',
            'Orbital distance [AU]',
            '',
            sorted((d, 0.02 if s == 10 else 1.0) for _, _, d, _, s in GRID_CASES),
        ),
        (
            r'(b) Final $T_{\rm surf}$ [K]',
            'Orbital distance [AU]',
            '',
            sorted((d, 1700.0 + 100.0 * d + 50.0 * m) for _, m, d, _, _ in GRID_CASES),
        ),
        (
            '(c) Stop time [yr]',
            'Orbital distance [AU]',
            '',
            sorted((d, 2.0e4 * m) for _, m, d, _, _ in GRID_CASES),
        ),
        (
            r'(d) Final $P_{\rm surf}$ [bar]',
            'H budget [ppmw]',
            '',
            sorted((h, 7.5 * h + 100.0 * m) for _, m, _, h, _ in GRID_CASES),
        ),
    ]
    for ax, (title, xlabel, ylabel, exp_pts) in zip(fig.axes, expected):
        assert ax.get_title(loc='left') == title
        assert ax.get_xlabel() == xlabel
        assert ax.get_ylabel() == ylabel
        pts = sorted(tuple(c.get_offsets()[0]) for c in ax.collections)
        assert np.asarray(pts) == pytest.approx(np.asarray(exp_pts), rel=1e-12)
    # Assert point colour of each case against expected mass colour
    for ax in fig.axes:
        for c, coll in zip(cases, ax.collections):
            expected_hex = _cycle()[0] if c['mass'] == pytest.approx(0.5) else _cycle()[1]
            if c['status'] == tool.STATUS_SOLIDIFIED:
                assert np.allclose(coll.get_facecolors()[0], to_rgba(expected_hex))
            else:
                assert np.allclose(coll.get_edgecolors()[0], to_rgba(expected_hex))
    with Image.open(out) as img:
        dpi = img.info['dpi'][0]
    assert pytest.approx(123, abs=1) == dpi
    assert out.stat().st_size > 0
    plt.close(fig)


def test_outputs_mark_the_stop_reason(grid_dir, tmp_path):
    """Solidified cases are filled circles, the others open squares; an unknown status is named by its code."""
    _write_case(grid_dir, 5, 2.0, 0.5, 5000.0, 13)
    cases = tool.load_cases(grid_dir)
    fig = tool.plot_outputs(cases, tmp_path / 'outputs.png')
    filled = [len(c.get_facecolors()) > 0 for c in fig.axes[0].collections]
    assert filled == [c['status'] == tool.STATUS_SOLIDIFIED for c in cases]
    assert filled.count(True) == 3 and filled.count(False) == 3
    # Check marker paths on scatter points
    p_s = MarkerStyle('s').get_path().transformed(MarkerStyle('s').get_transform())
    p_o = MarkerStyle('o').get_path().transformed(MarkerStyle('o').get_transform())
    for c, coll in zip(cases, fig.axes[0].collections):
        target = p_o if c['status'] == tool.STATUS_SOLIDIFIED else p_s
        v = coll.get_paths()[0].vertices
        assert v.shape == target.vertices.shape and np.allclose(v, target.vertices)
    # Stop legend labels and handle markers
    stop_legend = fig.axes[1].get_legend()
    labels = [t.get_text() for t in stop_legend.get_texts()]
    assert labels == ['solidified', 'status 13', 'energy balance']
    assert [h.get_marker() for h in stop_legend.legend_handles] == ['o', 's', 's']
    mass_legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in mass_legend.get_texts()] == [
        r'0.5 $M_\oplus$',
        r'2 $M_\oplus$',
    ]
    assert [ln.get_color() for ln in mass_legend.get_lines()] == _cycle()[:2]
    plt.close(fig)


def test_main_writes_both_figures_in_the_asked_format(grid_dir, tmp_path, capsys):
    """The command writes the 2 files into a new folder and leaves no figure open."""
    out = tmp_path / 'new' / 'plots'
    tool.main([str(grid_dir), '--output', str(out), '--format', 'pdf'])
    assert sorted(p.name for p in out.iterdir()) == ['grid_outputs.pdf', 'grid_tracks.pdf']
    assert all(p.stat().st_size > 0 for p in out.iterdir())
    assert plt.get_fignums() == []
    assert 'grid_tracks.pdf' in capsys.readouterr().out


def test_main_uses_default_png_format_and_output_folder(grid_dir, tmp_path, monkeypatch):
    """Calling main without options writes PNGs into output_files/grid_plots and permits existing folders."""
    monkeypatch.chdir(tmp_path)
    tool.main([str(grid_dir)])
    default_out = tmp_path / 'output_files' / 'grid_plots'
    assert sorted(p.name for p in default_out.iterdir()) == [
        'grid_outputs.png',
        'grid_tracks.png',
    ]
    assert all(p.stat().st_size > 0 for p in default_out.iterdir())
    dpis = []
    for p in sorted(default_out.glob('*.png')):
        with Image.open(p) as img:
            dpis.append(img.info['dpi'][0])
    assert dpis
    assert all(pytest.approx(200, abs=1) == d for d in dpis)
    # A second call into the existing folder succeeds
    tool.main([str(grid_dir)])
    assert sorted(p.name for p in default_out.iterdir()) == [
        'grid_outputs.png',
        'grid_tracks.png',
    ]
    assert plt.get_fignums() == []


@pytest.mark.parametrize('variant', ['light', 'dark'])
def test_figures_take_series_and_neutral_colours_from_the_style(grid_dir, tmp_path, variant):
    """Under each proteus-mpl style the series follow its cycle and no neutral is a fixed colour."""
    cases = tool.load_cases(grid_dir)
    cycle = proteus_mpl.CYCLE if variant == 'light' else proteus_mpl.CYCLE_DARK
    with plt.rc_context():
        proteus_mpl.use(variant)
        text, face = to_rgb(plt.rcParams['text.color']), to_rgb(plt.rcParams['axes.facecolor'])
        tracks = tool.plot_tracks(cases, tmp_path / 'tracks.png')
        outputs = tool.plot_outputs(cases, tmp_path / 'outputs.png')
    ax = tracks.axes[0]
    solid = [ln for ln in ax.lines if ln.get_linestyle() == '-']
    assert {to_rgb(ln.get_color()) for ln in solid} == {to_rgb(c) for c in cycle[:3]}
    assert to_rgb(tracks.get_facecolor()) == face
    # The threshold line and its label lie between the text and the surface colour
    threshold = to_rgb([ln for ln in ax.lines if ln.get_linestyle() == ':'][0].get_color())
    label = to_rgb(ax.texts[0].get_color())
    for neutral in (threshold, label):
        assert neutral not in (text, face, to_rgb('grey'), to_rgb('dimgrey'))
        assert all(min(t, f) <= n <= max(t, f) for n, t, f in zip(neutral, text, face))
    # Solidified markers: face in the series colour, edge in the text colour of the style
    solidified = [
        coll
        for c, coll in zip(cases, outputs.axes[2].collections)
        if c['status'] == tool.STATUS_SOLIDIFIED
    ]
    assert len(solidified) == 3
    for coll in solidified:
        assert to_rgb(coll.get_edgecolors()[0]) == pytest.approx(text)
        assert to_rgb(coll.get_facecolors()[0]) in [to_rgb(c) for c in cycle[:2]]
    handles = outputs.axes[1].get_legend().legend_handles
    assert to_rgb(handles[0].get_markeredgecolor()) == pytest.approx(text)
    if variant == 'dark':
        assert sum(text) > 2.0 and text != to_rgb('black')
    plt.close('all')


def test_makers_draw_from_the_grid_folder_of_the_manifest_entry(grid_dir, tmp_path):
    """The 2 makers read the cases of `run_dirs[0]` and write the file at the given path."""
    tracks = tool.make_tracks([grid_dir], tmp_path / 'a.png')
    outputs = tool.make_outputs([grid_dir], tmp_path / 'b.png')
    assert len([ln for ln in tracks.axes[0].lines if ln.get_linestyle() == '-']) == 5
    assert [len(ax.collections) for ax in outputs.axes] == [5, 5, 5, 5]
    assert all((tmp_path / n).stat().st_size > 0 for n in ('a.png', 'b.png'))
    with pytest.raises(FileNotFoundError, match='No case_'):
        tool.make_tracks([tmp_path / 'empty'], tmp_path / 'c.png')
    assert not (tmp_path / 'c.png').exists()
    plt.close('all')


def test_main_style_option_draws_on_the_dark_surface_and_restores_the_style(grid_dir, tmp_path):
    """`--style dark` gives the dark surface at the resolution of the style, for this call only."""
    before = dict(plt.rcParams)
    tool.main([str(grid_dir), '--output', str(tmp_path), '--style', 'dark'])
    with plt.rc_context():
        proteus_mpl.use('dark')
        surface = [round(255 * c) for c in to_rgb(plt.rcParams['figure.facecolor'])]
        style_dpi = plt.rcParams['savefig.dpi']
    with Image.open(tmp_path / 'grid_tracks.png') as img:
        assert list(img.convert('RGB').getpixel((2, 2))) == surface
        assert img.info['dpi'][0] == pytest.approx(style_dpi, abs=1)
    assert sum(surface) < 60 and style_dpi != 200
    assert dict(plt.rcParams) == before


def test_main_style_option_without_the_package_is_a_usage_error(
    grid_dir, tmp_path, monkeypatch, capsys
):
    """Without proteus-mpl, `--style` stops with a message and writes no figure."""
    monkeypatch.setitem(sys.modules, 'proteus_mpl', None)
    with pytest.raises(SystemExit) as err:
        tool.main([str(grid_dir), '--output', str(tmp_path / 'out'), '--style', 'light'])
    assert err.value.code == 2
    assert '--style needs the proteus-mpl package' in capsys.readouterr().err
    assert list((tmp_path / 'out').iterdir()) == []
