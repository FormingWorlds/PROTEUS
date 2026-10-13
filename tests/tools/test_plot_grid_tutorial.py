"""
Unit tests for ``tools/plot_grid_tutorial.py``.

The grid folders are synthetic: each case holds the three files the tool reads
(the resolved config, the status file and a short helpfile). The figures are
drawn on the Agg backend and their axes are read back; no image is compared.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _load_tool():
    """Load ``tools/plot_grid_tutorial.py`` as a module."""
    script_path = Path(__file__).resolve().parents[2] / 'tools' / 'plot_grid_tutorial.py'
    spec = importlib.util.spec_from_file_location('plot_grid_tutorial_under_test', script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


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
    # 2 rows of the initial stage (Time = 0), then 3 rows; P_surf differs distinctly from H_budget
    last_phi = 0.02 if status == tool.STATUS_SOLIDIFIED else 1.0
    last_t_surf = 1700.0 + 100.0 * distance + 50.0 * mass
    last_p_surf = 7.5 * h_budget + 100.0 * mass
    rows = [
        (0.0, 1.0, 4000.0, 0.1 * h_budget),
        (0.0, 1.0, 4000.0, 0.1 * h_budget),
        (10.0, 1.0, 3500.0, 0.5 * h_budget),
        (1.0e3 * mass, 0.5, 2500.0, 0.8 * h_budget),
        (2.0e4 * mass, last_phi, last_t_surf, last_p_surf),
    ]
    text = 'Time Phi_global T_surf P_surf\n' + ''.join(
        ' '.join(f'{v:.8e}' for v in r) + '\n' for r in rows
    )
    (case_dir / 'runtime_helpfile.csv').write_text(text)


@pytest.fixture
def grid_dir(tmp_path):
    """A grid of 4 cases: 2 masses at 2 distances, the inner pair stopped at energy balance."""
    grid = tmp_path / 'grid'
    _write_case(grid, 0, 0.5, 0.03, 1000.0, 14)
    _write_case(grid, 1, 2.0, 0.03, 9000.0, 14)
    _write_case(grid, 2, 0.5, 0.5, 1000.0, 10)
    _write_case(grid, 3, 2.0, 0.5, 9000.0, 10)
    return grid


def test_load_cases_reads_inputs_stop_and_science_rows(grid_dir):
    """Each record holds the swept inputs, the status code and the rows after Time = 0."""
    cases = tool.load_cases(grid_dir)
    assert len(cases) == 4
    assert [(c['mass'], c['distance'], c['H_budget'], c['status']) for c in cases] == [
        (0.5, 0.03, 1000.0, 14),
        (2.0, 0.03, 9000.0, 14),
        (0.5, 0.5, 1000.0, 10),
        (2.0, 0.5, 9000.0, 10),
    ]
    assert all(c['H_mode'] == 'ppmw' and c['phi_crit'] == pytest.approx(0.03) for c in cases)
    # The 2 rows of the initial stage are dropped
    assert all(len(c['hf']) == 3 and c['hf']['Time'].min() > 0 for c in cases)
    assert cases[3]['hf']['Time'].iloc[-1] == pytest.approx(4.0e4)


def test_load_cases_gives_no_threshold_when_the_stop_is_off(tmp_path):
    """A case with the solidification stop off has no threshold, whatever `phi_crit` holds."""
    _write_case(tmp_path, 0, 1.0, 0.5, 5000.0, 14, phi_crit=None)
    cases = tool.load_cases(tmp_path)
    assert len(cases) == 1
    assert cases[0]['phi_crit'] is None


def test_load_cases_skips_cases_without_output(grid_dir, capsys):
    """A case with a missing or empty file, or with rows of the initial stage only, is skipped and named."""
    (grid_dir / 'case_000004').mkdir()
    _write_case(grid_dir, 5, 1.0, 0.5, 5000.0, 10)
    helpfile = grid_dir / 'case_000005' / 'runtime_helpfile.csv'
    helpfile.write_text(''.join(helpfile.read_text().splitlines(keepends=True)[:3]))
    _write_case(grid_dir, 6, 1.0, 0.5, 5000.0, 10)
    (grid_dir / 'case_000006' / 'status').write_text('')
    cases = tool.load_cases(grid_dir)
    assert [c['mass'] for c in cases] == [0.5, 2.0, 0.5, 2.0]
    printed = capsys.readouterr().out
    assert 'Skipped case_000004: an output file is missing or empty' in printed
    assert 'Skipped case_000005: no row after the initial stage' in printed
    assert 'Skipped case_000006: an output file is missing or empty' in printed
    assert printed.count('Skipped') == 3


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
    fig = tool.plot_tracks(cases, out)
    ax = fig.axes[0]
    flat = [ln for ln in ax.lines if ln.get_linestyle() == ':']
    assert len(flat) == 1
    assert list(flat[0].get_ydata()) == pytest.approx([0.03, 0.03])
    assert [t.get_text() for t in ax.texts] == [r'solidification ($\Phi$ = 0.03)']
    assert ax.get_xscale() == 'log'
    # 4 tracks, 2 colours (one per distance), and the legend names both distances
    tracks = [ln for ln in ax.lines if ln.get_linestyle() == '-']
    assert len(tracks) == 4 and len({ln.get_color() for ln in tracks}) == 2
    for c, track in zip(cases, tracks):
        assert list(track.get_xdata()) == pytest.approx(list(c['hf']['Time']))
        assert list(track.get_ydata()) == pytest.approx(list(c['hf']['Phi_global']))
    legend_lines = ax.get_legend().get_lines()
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['0.03 AU', '0.5 AU']
    expected_dist_colors = [tool._colours([0.03, 0.5])[d] for d in (0.03, 0.5)]
    assert [ln.get_color() for ln in legend_lines] == expected_dist_colors
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
    fig = tool.plot_outputs(cases, out)
    ax_d = fig.axes[3]
    assert ax_d.get_title(loc='left') == r'(d) Final $P_{\rm surf}$ [bar]'
    assert ax_d.get_xlabel() == 'H budget [ppmw]'
    assert ax_d.get_yscale() == 'log'
    # All 3 distance panels use log scale on the x axis
    assert all(fig.axes[i].get_xscale() == 'log' for i in (0, 1, 2))
    # Panel (a) plots final melt fraction against orbital distance
    pts_a = sorted(tuple(c.get_offsets()[0]) for c in fig.axes[0].collections)
    assert pts_a == pytest.approx(
        sorted((c['distance'], c['hf']['Phi_global'].iloc[-1]) for c in cases)
    )
    # Panel (b) plots final surface temperature against orbital distance
    pts_b = sorted(tuple(c.get_offsets()[0]) for c in fig.axes[1].collections)
    assert pts_b == pytest.approx(
        sorted((c['distance'], c['hf']['T_surf'].iloc[-1]) for c in cases)
    )
    # Panel (c) holds the stop time, which rises with the planet mass in this grid
    pts_c = sorted(tuple(c.get_offsets()[0]) for c in fig.axes[2].collections)
    assert pts_c == pytest.approx(
        sorted((c['distance'], c['hf']['Time'].iloc[-1]) for c in cases)
    )
    # Panel (d) plots final surface pressure against hydrogen budget
    pts_d = sorted(tuple(c.get_offsets()[0]) for c in ax_d.collections)
    assert pts_d == pytest.approx(
        sorted((c['H_budget'], c['hf']['P_surf'].iloc[-1]) for c in cases)
    )
    assert out.stat().st_size > 0
    plt.close(fig)


def test_outputs_mark_the_stop_reason(grid_dir, tmp_path):
    """Solidified cases are filled circles, the others open squares; an unknown status is named by its code."""
    _write_case(grid_dir, 4, 2.0, 0.5, 5000.0, 13)
    cases = tool.load_cases(grid_dir)
    fig = tool.plot_outputs(cases, tmp_path / 'outputs.png')
    filled = [len(c.get_facecolors()) > 0 for c in fig.axes[0].collections]
    assert filled == [c['status'] == tool.STATUS_SOLIDIFIED for c in cases]
    assert filled.count(True) == 2 and filled.count(False) == 3
    labels = [t.get_text() for t in fig.axes[1].get_legend().get_texts()]
    assert labels == ['solidified', 'status 13', 'energy balance']
    mass_legend = fig.axes[0].get_legend()
    assert [t.get_text() for t in mass_legend.get_texts()] == [
        r'0.5 $M_\oplus$',
        r'2 $M_\oplus$',
    ]
    expected_mass_colors = [tool._colours([0.5, 2.0])[m] for m in (0.5, 2.0)]
    assert [ln.get_color() for ln in mass_legend.get_lines()] == expected_mass_colors
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
    # A second call into the existing folder succeeds
    tool.main([str(grid_dir)])
    assert sorted(p.name for p in default_out.iterdir()) == [
        'grid_outputs.png',
        'grid_tracks.png',
    ]
    assert plt.get_fignums() == []
