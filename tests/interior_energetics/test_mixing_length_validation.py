"""
Validation of the mixing length at the core-mantle boundary (CMB) from stored runs.

Invariant: under the default mixing length, 'constant', the heat flux out of the core stays
coupled to the mantle: after 10 kyr no 3 consecutive helpfile rows have F_cmb below
100 W m-2, on uniform meshes of 80 and 160 nodes, on CMB cells of 1500 and 3000 m, with the
rheology on, and for a 1.5 Earth-mass planet. Under 'nearest' the flux decouples for tens of
kyr on 80 nodes and on the 3000 m cell, but not on 160 nodes. The runs and the tool that
stores them: ``tools/verification/mixing_length/cmb_flux.py``. See
``docs/Validation/interior_energetics/aragog.md``.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

_ROOT = Path(__file__).parents[2]
_DIR = _ROOT / 'tools' / 'verification' / 'mixing_length'
DATA = _ROOT / 'tests' / 'data' / 'mixing_length' / 'cmb_flux.npz'
TUTORIAL_CONSTANT = ('u80_constant', 'u160_constant', 'c3000_constant', 'c1500_constant')
# Single or double dip rows pass; 3 rows in a row count as a lasting loss of the flux.
MAX_DIP_ROWS = 2
# Short legend labels; the title of each plot names the mixing length.
LABELS = {
    'u80': '80 nodes, uniform',
    'u160': '160 nodes, uniform',
    'c3000': '3000 m CMB cell',
    'c1500': '1500 m CMB cell',
    'rheology_constant': 'constant',
    'rheology_nearest': 'nearest',
}


def _label(case):
    return LABELS.get(case) or LABELS[case.split('_')[0]]


def _tool():
    """Load cmb_flux.py of the tools folder."""
    spec = importlib.util.spec_from_file_location('cmb_flux_uut', _DIR / 'cmb_flux.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


def _case(case):
    return TOOL.read_case(DATA, case)


def _plotted_rows(frame, points=180):
    """Rows after 2 kyr to draw: about ``points`` rows spread evenly plus each dip row and its
    2 neighbours, so that every dip the tests count is drawn."""
    t, f = frame.Time.to_numpy(), frame.F_cmb.to_numpy()
    every = max(1, len(t) // points)
    low = (t > TOOL.T_COUNT) & (f < TOOL.LOW_FLUX)
    near = low | np.roll(low, 1) | np.roll(low, -1)
    return (t > 2e3) & ((np.arange(len(t)) % every == 0) | near)


def _flux_figure(cases, title):
    """Return a factory for |F_cmb| on a log axis against time, one line per case; a triangle
    marks each drawn row with F_cmb < 0, which the log axis would otherwise show as positive."""

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        negative = False
        for case in cases:
            frame = _case(case)
            late = _plotted_rows(frame)
            t, f = frame.Time[late] / 1e3, frame.F_cmb[late]
            (line,) = ax.plot(t, np.abs(f), lw=1, label=_label(case))
            neg = f < 0
            ax.plot(t[neg], np.abs(f[neg]), 'v', ms=6, color=line.get_color())
            negative |= bool(neg.any())
        if negative:
            ax.plot([], [], 'v', ms=6, color='0.4', label=r'$F_\mathrm{cmb} < 0$')
        ax.axhline(TOOL.LOW_FLUX, ls='--', lw=1, color='0.5', label='100 W m$^{-2}$')
        ax.set_yscale('log')
        ax.set_xlabel('Time [kyr]')
        ax.set_ylabel(r'$|F_\mathrm{cmb}|$ [W m$^{-2}$]')
        ax.set_title(title)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
        fig.tight_layout()
        return fig

    return figure


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_constant_length_keeps_the_cmb_flux_coupled_on_every_mesh(validation_plot):
    """Under 'constant' no 3 consecutive rows after 10 kyr have F_cmb < 100 W m-2 and no
    T_cmb drop between rows after 2 kyr exceeds 10 K, on 4 meshes of the lid-seal tutorial; T_cmb at
    100 kyr of the 4 meshes lies within 25 K. Single-row dips occur only with a CMB cell."""
    found = {case: TOOL.metrics(_case(case)) for case in TUTORIAL_CONSTANT}
    for case, m in found.items():
        assert m['longest_low'] <= MAX_DIP_ROWS, (case, m['dips'])
        assert m['worst_drop'] >= -10.0, case
        assert m['t_end'] == pytest.approx(1e5, abs=1.0), case
    ends = [m['T_end'] for m in found.values()]
    assert max(ends) - min(ends) <= 25.0
    assert found['u80_constant']['n_low'] == found['u160_constant']['n_low'] == 0
    assert found['c3000_constant']['n_low'] > 0 and found['c1500_constant']['n_low'] > 0
    # A start changed at round-off keeps the dips at the same times: they are not round-off.
    twin = TOOL.metrics(_case('c3000_constant_roundoff'))
    assert twin['longest_low'] <= MAX_DIP_ROWS
    assert [t for t, _ in twin['dips']] == pytest.approx(
        [t for t, _ in found['c3000_constant']['dips']], abs=1.0
    )

    validation_plot(
        _flux_figure(TUTORIAL_CONSTANT, "Mixing length 'constant'"),
        'interior_energetics_aragog_constant_cmb_flux',
    )


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_nearest_length_decouples_on_80_nodes_and_on_the_3000_m_cell_only(validation_plot):
    """Under 'nearest' F_cmb stays below 100 W m-2 for more than 10 kyr on the uniform mesh
    of 80 nodes and on the 3000 m CMB cell, and never after 10 kyr on 160 nodes."""
    for case in ('u80_nearest', 'c3000_nearest'):
        m = TOOL.metrics(_case(case))
        assert m['longest_low_span'] > 1e4, (case, m['longest_low_span'])
    assert TOOL.metrics(_case('u160_nearest'))['n_low'] == 0

    validation_plot(
        _flux_figure(
            ('u80_nearest', 'u160_nearest', 'c3000_nearest'), "Mixing length 'nearest'"
        ),
        'interior_energetics_aragog_nearest_cmb_flux',
    )


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_constant_length_keeps_the_flux_coupled_with_the_rheology_on(validation_plot):
    """With the rheology on and the 3000 m CMB cell, 'constant' keeps F_cmb coupled as above;
    the 'nearest' run of the same setup is shown for comparison."""
    m = TOOL.metrics(_case('rheology_constant'))
    assert m['longest_low'] <= MAX_DIP_ROWS, m['dips']
    assert m['worst_drop'] >= -10.0

    validation_plot(
        _flux_figure(('rheology_constant', 'rheology_nearest'), 'Rheology on'),
        'interior_energetics_aragog_rheology_cmb_flux',
    )


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_a_15_earth_mass_planet_keeps_its_core_temperature_under_the_default(validation_plot):
    """The 1.5 Earth-mass case under the default reaches T_cmb = 5199.9 K at 300 kyr on the
    uniform mesh, as a run that sets 'constant' in its file does; on both meshes no row has
    |F_cmb| < 1 W m-2 and no 3 consecutive rows after 10 kyr have F_cmb < 100 W m-2. The T_cmb steps
    between rows are not bounded here: the early cooling is fast and the rows are far apart."""
    uniform = TOOL.metrics(_case('m1p5_uniform'))
    assert uniform['t_end'] == pytest.approx(3e5, abs=1.0)
    assert uniform['T_end'] == pytest.approx(5199.9, abs=1.0)
    assert uniform['longest_low'] <= MAX_DIP_ROWS, uniform['dips']
    default = TOOL.metrics(_case('m1p5_default'))
    assert default['longest_low'] <= MAX_DIP_ROWS, default['dips']
    for case in ('m1p5_uniform', 'm1p5_default'):
        assert np.abs(_case(case).F_cmb).min() >= 1.0, case

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for case in ('m1p5_uniform', 'm1p5_default'):
            frame = _case(case)
            ax.plot(frame.Time / 1e3, frame.T_cmb, lw=1.2, label=TOOL.CASES[case])
        ax.plot(
            [300.0],
            [5199.9],
            'o',
            ms=5,
            label="5199.9 K, uniform mesh, 'constant' set in the file",
        )
        ax.set_xlabel('Time [kyr]')
        ax.set_ylabel(r'$T_\mathrm{cmb}$ [K]')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'interior_energetics_aragog_15_earth_mass_tcmb')


def test_the_dip_measure_counts_consecutive_rows_after_10_kyr():
    """A single low row, a run of 3 low rows and a negative row are counted as such; rows
    before 10 kyr are not counted, and T_cmb steps before 2 kyr are not measured."""
    t = np.array([0.0, 1e3, 5e3, 1.1e4, 1.2e4, 1.3e4, 1.4e4, 1.5e4, 1.6e4, 1.7e4])
    f = np.array([1.0, 1.0, 500, 50.0, 500, 20.0, -5.0, 0.5, 500, 500])
    temp = np.array([5000.0, 4000.0, 3990, 3989, 3988, 3987, 3988.5, 3986, 3985, 3984])
    m = TOOL.metrics(pd.DataFrame({'Time': t, 'F_cmb': f, 'T_cmb': temp}))
    assert m['n_low'] == 4 and m['longest_low'] == 3
    assert m['longest_low_span'] == pytest.approx(1.5e4 - 1.2e4)
    assert m['n_negative'] == 1 and m['n_below_1'] == 1
    assert (m['F_min'], m['F_min_time']) == (-5.0, 1.4e4)
    assert (m['worst_drop'], m['worst_rise']) == (-10.0, 1.5)
    assert m['dips'][0] == (1.1e4, 50.0)


def test_the_stored_runs_hold_every_case_and_name_their_commits():
    """Each case of the tool is stored with its 3 columns of equal length, and the file names
    the PROTEUS and aragog commits of the runs."""
    for case in TOOL.CASES:
        frame = _case(case)
        # Each run starts with 3 rows at t = 0; time then grows in every row.
        assert len(frame) > 100 and np.all(frame.Time[:3] == 0), case
        assert np.all(np.diff(frame.Time[2:]) > 0), case
    assert re.fullmatch(
        r'PROTEUS commit [0-9a-f]{40}, aragog commit [0-9a-f]{40}', TOOL.read_source(DATA)
    )


def test_the_stored_15_earth_mass_config_leaves_the_mixing_length_to_the_default():
    """planet_1p5_earth.toml does not set the mixing length and holds the settings the page
    names: 1.5 Earth masses, 80 nodes, the uniform mesh, the mixture drag and 300 kyr."""
    import tomllib

    from proteus.config import read_config_object

    path = _DIR / 'planet_1p5_earth.toml'
    assert 'mixing_length' not in tomllib.loads(path.read_text())['interior_energetics']
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)  # no deprecated key in the file
        c = read_config_object(path)
    ie = c.interior_energetics
    assert ie.mixing_length == 'constant'
    assert (c.planet.mass_tot, ie.num_levels, ie.aragog.cmb_cell_thickness) == (1.5, 80, 0.0)
    assert ie.aragog.separation_viscosity == 'mixture'
    assert (c.params.stop.time.maximum, c.params.out.path) == (3e5, 'm1p5_uniform')


def test_the_tool_stores_every_case_and_reads_it_back(tmp_path):
    """main writes the 3 columns of each case folder and the commits; read_case and
    read_source give them back unchanged."""
    rng = np.random.default_rng(1)
    want = {}
    for case in TOOL.CASES:
        (tmp_path / case).mkdir()
        frame = pd.DataFrame(rng.random((4, 3)) * [1e3, 5e3, 1e2], columns=TOOL.COLUMNS)
        frame.to_csv(tmp_path / case / 'runtime_helpfile.csv', sep='\t', index=False)
        want[case] = frame
    npz = tmp_path / 'out' / 'cmb.npz'
    assert TOOL.main([str(tmp_path), str(npz), '--commit', 'a' * 40, '--aragog', 'b' * 40]) == 0
    for case, frame in want.items():
        pd.testing.assert_frame_equal(TOOL.read_case(npz, case), frame, check_exact=True)
    assert TOOL.read_source(npz) == f'PROTEUS commit {"a" * 40}, aragog commit {"b" * 40}'
