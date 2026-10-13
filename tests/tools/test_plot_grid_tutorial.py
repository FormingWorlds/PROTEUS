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
    # 2 rows of the initial stage (Time = 0), then 3 rows; P_surf scales with the H budget
    last_phi = 0.02 if status == tool.STATUS_SOLIDIFIED else 1.0
    rows = [
        (0.0, 1.0, 4000.0, 0.1 * h_budget),
        (0.0, 1.0, 4000.0, 0.1 * h_budget),
        (10.0, 1.0, 3500.0, 0.5 * h_budget),
        (1.0e3 * mass, 0.5, 2500.0, 0.8 * h_budget),
        (2.0e4 * mass, last_phi, 1700.0 + 100.0 * distance, 1.0 * h_budget),
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
    """A case with a missing file or with rows of the initial stage only is skipped and named."""
    (grid_dir / 'case_000004').mkdir()
    _write_case(grid_dir, 5, 1.0, 0.5, 5000.0, 10)
    helpfile = grid_dir / 'case_000005' / 'runtime_helpfile.csv'
    helpfile.write_text(''.join(helpfile.read_text().splitlines(keepends=True)[:3]))
    cases = tool.load_cases(grid_dir)
    assert [c['mass'] for c in cases] == [0.5, 2.0, 0.5, 2.0]
    printed = capsys.readouterr().out
    assert 'Skipped case_000004: output files are missing' in printed
    assert 'Skipped case_000005: no row after the initial stage' in printed
    assert printed.count('Skipped') == 2


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
    fig = tool.plot_tracks(tool.load_cases(grid_dir), out)
    ax = fig.axes[0]
    flat = [ln for ln in ax.lines if ln.get_linestyle() == ':']
    assert len(flat) == 1
    assert list(flat[0].get_ydata()) == pytest.approx([0.03, 0.03])
    assert [t.get_text() for t in ax.texts] == [r'solidification ($\Phi$ = 0.03)']
    # 4 tracks, 2 colours (one per distance), and the legend names both distances
    tracks = [ln for ln in ax.lines if ln.get_linestyle() == '-']
    assert len(tracks) == 4 and len({ln.get_color() for ln in tracks}) == 2
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ['0.03 AU', '0.5 AU']
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
    fig = tool.plot_outputs(tool.load_cases(grid_dir), out)
    ax_d = fig.axes[3]
    assert ax_d.get_title(loc='left') == r'(d) Final $P_{\rm surf}$ [bar]'
    assert ax_d.get_xlabel() == 'H budget [ppmw]'
    assert ax_d.get_yscale() == 'log'
    points = sorted(tuple(c.get_offsets()[0]) for c in ax_d.collections)
    assert points == pytest.approx(
        [(1000.0, 1000.0), (1000.0, 1000.0), (9000.0, 9000.0), (9000.0, 9000.0)]
    )
    # Panel (c) holds the stop time, which rises with the planet mass in this grid
    stop_times = sorted(c.get_offsets()[0][1] for c in fig.axes[2].collections)
    assert stop_times == pytest.approx([1.0e4, 1.0e4, 4.0e4, 4.0e4])
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
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == [
        r'0.5 $M_\oplus$',
        r'2 $M_\oplus$',
    ]
    plt.close(fig)


def test_main_writes_both_figures_in_the_asked_format(grid_dir, tmp_path, capsys):
    """The command writes the 2 files into a new folder and leaves no figure open."""
    out = tmp_path / 'new' / 'plots'
    tool.main([str(grid_dir), '--output', str(out), '--format', 'pdf'])
    assert sorted(p.name for p in out.iterdir()) == ['grid_outputs.pdf', 'grid_tracks.pdf']
    assert all(p.stat().st_size > 0 for p in out.iterdir())
    assert plt.get_fignums() == []
    assert 'grid_tracks.pdf' in capsys.readouterr().out
