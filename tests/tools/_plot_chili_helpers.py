"""
Data builders, colour tables and fixtures shared by the tests of the CHILI
comparison plot script ``tools/plot_chili_comparison.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd


def _run_with_log(path: Path, sha: str) -> Path:
    path.mkdir()
    (path / 'proteus_00.log').write_text(
        '[ INFO  ] PROTEUS version   26.7.14.post1.dev148+gd0c7f0806\n'
        f'[ INFO  ] PROTEUS git hash  {sha}\n'
    )
    return path


# A dark style: no colour is pure black or white, no right spine, a grid on every axes.
DARK_BY_HAND = {
    'text.color': '#E9EEF2',
    'axes.labelcolor': '#E9EEF2',
    'axes.titlecolor': '#E9EEF2',
    'axes.edgecolor': '#E9EEF2',
    'xtick.color': '#9FB0BE',
    'ytick.color': '#9FB0BE',
    'axes.facecolor': '#05070B',
    'figure.facecolor': '#05070B',
    'hatch.color': '#E9EEF2',
    'axes.spines.right': False,
    'axes.grid': True,
}


# Model colours of the CHILI analysis code and the gas colours of the page.
MODEL_COLOURS = {
    'GOOEY': '#bbbbbb',
    'NEONGOOEY': '#ee6677',
    'PROTEUS CHILI': '#eb7d0f',
    'PACMAN': '#4477aa',
    'LINCS': '#228833',
    'MOAI': '#aa3377',
    'PlanAtMO': '#ccbb44',
}


MODEL_KEYS = ('gooey', 'neongooey', 'proteus', 'pacman', 'lincs', 'moai', 'planatmo')


GAS_COLOURS = {
    'H2O': '#027fb1',
    'CO2': '#d24901',
    'CO': '#d1ac02',
    'H2': '#008c01',
    'CH4': '#c720dd',
    'N2': '#870036',
    'S2': '#ff8fa1',
    'SO2': '#00008b',
    'H2S': '#640deb',
    'O2': '#00dcc6',
}


def _comparison_data(root: Path, models=('moai',)) -> Path:
    """A comparison folder with a file for Earth and for Venus of each model."""
    table = pd.DataFrame(
        {
            't(yr)': [0.0, 1e3, 1e5],
            'T_surf(K)': [3000.0, 2500.0, 1700.0],
            'phi(vol_frac)': [0.99, 0.95, 0.05],
            'fO2_melt(bar)': [1e-2, 1e-4, 1e-8],
            'p_H2O(bar)': [1.0, 5.0, 70.0],
            'p_O2(bar)': [1e-2, 0.25, 1e-8],
            'p_SO2(bar)': [0.0, 0.0, 3.0],
            'p_surf(bar)': [20.0, 30.0, 80.0],
            'flux_OLR(W/m2)': [1e5, 1e4, 300.0],
            'R_solid(m)': [3.5e6, 4.0e6, 6.3e6],
            'viscosity(Pa.s)': [1e2, 1e3, 1e20],
            'massH_atm(kg)': [1e20, 2e20, 3e20],
            'massH_melt(kg)': [6e20, 5e20, 1e20],
            'massH_solid(kg)': [0.0, 0.0, 3e20],
            'massC_atm(kg)': [1e20, 2e20, 4e20],
            'massC_melt(kg)': [4e20, 3e20, 5e19],
            'massC_solid(kg)': [0.0, 0.0, 5e19],
        }
    )
    for model in models:
        (root / model).mkdir(parents=True)
        for planet in ('earth', 'venus'):
            table.to_csv(root / model / f'evolution-{model}-{planet}-data.csv')
    return root


def _run_table() -> pd.DataFrame:
    """Rows of a run in the columns of a helpfile that the figures read."""
    return pd.DataFrame(
        {
            'Time': [1.0, 1e4, 1.13e7],
            'Phi_global': [1.0, 0.95, 0.05],
            'T_surf': [3300.0, 2900.0, 1800.0],
            'T_magma': [4200.0, 3100.0, 1810.0],
            'H2O_bar': [0.5, 4.0, 300.0],
            'O2_bar': [2.0e4, 128.0, 2.0e-5],
            'SO2_bar': [0.0, 0.0, 2.5],
            'P_surf': [25.0, 140.0, 310.0],
            'F_olr': [2e5, 2e4, 280.0],
            'R_core': [3.48e6, 3.48e6, 3.48e6],
            'H_kg_atm': [1e20, 2e20, 3e20],
            'H_kg_liquid': [6e20, 5e20, 1e20],
            'H_kg_solid': [0.0, 0.0, 3e20],
            'H_kg_total': [7e20, 7e20, 7e20],
            'C_kg_atm': [1e20, 2e20, 4e20],
            'C_kg_liquid': [4e20, 3e20, 5e19],
            'C_kg_solid': [0.0, 0.0, 5e19],
            'C_kg_total': [5e20, 5e20, 5e20],
        }
    )


def _run_folder(path: Path, sha: str = 'd0c7f08062bdf08631b61225bad460c6d4e381eb') -> Path:
    """A run folder with a log and a helpfile of ``_run_table``."""
    _run_with_log(path, sha)
    _run_table().to_csv(path / 'runtime_helpfile.csv', sep=' ', index=False)
    return path


def _grid_case(tmp_path: Path) -> tuple[Path, Path]:
    """Comparison data with a grid file of the submitted run and of GOOEY, and a
    grid folder with one run."""
    data = _comparison_data(tmp_path / 'data', models=MODEL_KEYS)
    table = pd.read_csv(data / 'proteus' / 'evolution-proteus-earth-data.csv', index_col=0)
    table.to_csv(data / 'proteus' / 'evolution-proteus-earth-grid-Hlow-Clow-data.csv')
    table.to_csv(data / 'gooey' / 'evolution-gooey-earth-grid-Hlow-data.csv')
    (tmp_path / 'grid').mkdir()
    _run_folder(tmp_path / 'grid' / 'chili_grid_earth_Hlow_Clow')
    return data, tmp_path / 'grid'


def _hex(colour) -> str:
    return matplotlib.colors.to_hex(colour)


# Black, white and the greys 'grey', '0.25' and '0.4' as literal colours.
LITERAL_NEUTRALS = {'#000000', '#ffffff', '#808080', '#404040', '#666666'}


def _colours(fig) -> set[str]:
    """Hex colours of every visible text, text box, line, marker and patch of a figure."""
    found = set()
    for artist in fig.findobj(lambda a: a is not fig.patch and a.get_visible()):
        for getter in ('get_color', 'get_markeredgecolor', 'get_markerfacecolor'):
            if isinstance(artist, (matplotlib.text.Text, matplotlib.lines.Line2D)):
                value = getattr(artist, getter, lambda: 'none')()
                if matplotlib.colors.to_rgba(value)[3] > 0:
                    found.add(_hex(value))
        box = artist.get_bbox_patch() if isinstance(artist, matplotlib.text.Text) else None
        for patch in (artist, box):
            if isinstance(patch, matplotlib.patches.Patch):
                for value in (patch.get_facecolor(), patch.get_edgecolor()):
                    if value[3] > 0:
                        found.add(_hex(value))
    return found


def _all_figures(tool, tmp_path):
    """The 11 figures of the page on small tables, by file stem."""
    data, grid = _grid_case(tmp_path)
    earth = tool._load_proteus_helpfile(_run_folder(tmp_path / 'earth'))
    venus = earth.assign(
        Time=earth['Time'] * 2,
        T_magma=earth['T_magma'] + 100,
        T_surf=earth['T_surf'] + 50,
        H2O_bar=earth['H2O_bar'] * 1.5,
        H_kg_atm=earth['H_kg_atm'] * 2,
    )
    calls = tool._figures(data, earth, venus, tool._new_style('d0c7f080'), grid)
    return {name: draw() for name, draw in calls.items()}


def _main_arguments(tool, tmp_path, monkeypatch, *extra):
    """Arguments for ``main`` on small data, with the figure functions replaced."""
    seen = []
    for name in [n for n in dir(tool) if n.startswith('plot_')]:
        monkeypatch.setattr(tool, name, lambda *args, _name=name, **kwargs: seen.append(_name))
    monkeypatch.setattr(tool, 'ensure_chili_repo', lambda path: path)
    monkeypatch.setattr(tool.sys, 'argv', ['plot', '--output', str(tmp_path / 'out'), *extra])
    return seen


def _png_width(path: Path) -> int:
    """Pixel width in the header of a PNG file."""
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    return int.from_bytes(data[16:20], 'big')


def _repo(tmp_path: Path, models=('moai',)) -> Path:
    """A checkout folder of the comparison data with files of ``models``."""
    _comparison_data(tmp_path / 'repo' / 'intercomparison' / 'outputs', models=models)
    return tmp_path / 'repo'
