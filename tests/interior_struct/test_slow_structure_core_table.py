"""The core temperature table of the Zalmoxis structure solve, on a stored coupled state.

The state is the first structure re-solve (102 yr) of the nightly config
``tests/integration/zalmoxis_aragog_calliope.toml`` with ``temperature_mode = 'liquidus_super'``,
``core_bc = 'energy_balance'`` and an update every 100 yr: the row, the Aragog mantle temperature
arrays and the core table PROTEUS builds for that solve
(``fixtures/structure_core_table/state_eb_102yr.npz``).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

pytest.importorskip('zalmoxis')

pytestmark = [pytest.mark.slow, pytest.mark.timeout(3600)]

STATE = (
    pathlib.Path(__file__).parent / 'fixtures' / 'structure_core_table' / 'state_eb_102yr.npz'
)
CONFIG = pathlib.Path(__file__).parents[1] / 'integration' / 'zalmoxis_aragog_calliope.toml'
KEYS = ('R_core', 'R_int', 'P_center', 'P_cmb')
FIXED_PASSES = 4


def _config(use_jax):
    from proteus.config import read_config_object

    config = read_config_object(str(CONFIG))
    config.planet.temperature_mode = 'liquidus_super'
    config.interior_energetics.aragog.core_bc = 'energy_balance'
    z = config.interior_struct.zalmoxis
    z.update_interval, z.update_min_interval, z.use_jax = 100.0, 0.0, use_jax
    return config


def _solve(tmp_path, monkeypatch, name, arrays, table, use_jax=True, main=None):
    """One zalmoxis_solver call from the stored state with an empty density seed; returns the
    row and the last Zalmoxis result (radii, pressure, temperature)."""
    from zalmoxis.solver import main as zalmoxis_main

    from proteus.interior_struct import zalmoxis as Z

    state = np.load(STATE)
    row = json.loads(str(state['row']))
    out = tmp_path / name
    (out / 'data').mkdir(parents=True)
    monkeypatch.setattr(Z, '_density_cache', dict.fromkeys(Z._density_cache))
    seen = {}
    real = main or zalmoxis_main  # not Z.main, which an earlier call of this test wraps

    def recorded(*a, **k):
        seen['res'] = res = real(*a, **k)
        return res

    monkeypatch.setattr(Z, 'main', recorded)
    r, t = arrays

    def temperature_function(radius, pressure):
        return float(t[0]) if radius <= r[0] else float(np.interp(radius, r, t))

    Z.zalmoxis_solver(
        _config(use_jax),
        str(out),
        row,
        num_spider_nodes=int(state['num_spider_nodes']),
        temperature_function=temperature_function,
        temperature_arrays=arrays,
        core_temperature_arrays=table,
    )
    return row, seen['res']


def _fixed_point(table):
    """Zalmoxis main wrapped in the reference rule: no core table, the core T given by radius,
    from the table at the core node pressures of the previous pass, for FIXED_PASSES passes."""
    from zalmoxis.solver import main
    from zalmoxis.structure_model import core_temperature as table_temperature
    from zalmoxis.structure_model import is_core

    def fixed(*a, **k):
        r_m, t_m = (np.asarray(x, float) for x in k['temperature_arrays'])
        res = main(*a, **k)
        for _ in range(FIXED_PASSES):
            r, p = (np.asarray(res[x], float) for x in ('radii', 'pressure'))
            core = is_core(np.asarray(res['mass_enclosed'], float), res['cmb_mass'])
            rc, pc = float(res['core']['r'][-1]), float(res['core']['P'][-1])
            above = r_m > rc
            arrays = (
                np.concatenate([r[core], [rc], r_m[above]]),
                np.concatenate(
                    [
                        table_temperature(p[core], table),
                        [float(table_temperature(pc, table))],
                        t_m[above],
                    ]
                ),
            )
            res = main(*a, **{**k, 'temperature_arrays': arrays})
        return res

    return fixed


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_core_table_gives_the_structure_of_two_independent_paths(
    tmp_path, monkeypatch, validation_plot
):
    """The structure with the core table on the JAX path (the default) against 2 independent
    solves of the same state: the numpy path with the same table (other integrator, other
    temperature code) and a fixed point without the table (the core T given by radius,
    iterated on the core node pressures). R_core, R_int, P_center and P_cmb agree within 1e-4
    (measured at most 4.8e-7 and 5.5e-7). The table moves the structure against a core at the
    temperature of the deepest mantle node by more than 100 times that tolerance (measured
    4.6 % in P_center)."""
    state = np.load(STATE)
    arrays = (state['r'], state['T'])
    table = (state['table_P'], state['table_T'])
    row_jax, res_jax = _solve(tmp_path, monkeypatch, 'jax', arrays, table)
    row_np, res_np = _solve(tmp_path, monkeypatch, 'np', arrays, table, use_jax=False)
    row_fix, res_fix = _solve(
        tmp_path, monkeypatch, 'fix', arrays, None, main=_fixed_point(table)
    )
    row_iso, res_iso = _solve(tmp_path, monkeypatch, 'iso', arrays, None)
    for row in (row_jax, row_np):
        assert row['struct_core_T_used'] == pytest.approx(1.0, abs=0.0)
    for other in (row_np, row_fix):
        for key in KEYS:
            assert other[key] == pytest.approx(row_jax[key], rel=1e-4), key
    shift = max(abs(row_iso[k] / row_jax[k] - 1.0) for k in KEYS)
    assert shift > 1e-2

    def temperature_figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        for res, style, label in (
            (res_iso, ':', 'core at deepest mantle T'),
            (res_jax, '-', 'table, JAX'),
            (res_np, '--', 'table, numpy'),
            (res_fix, '-.', 'fixed point'),
        ):
            r = np.asarray(res['radii'], float) / 1e3
            n = int(np.count_nonzero(r < 5000.0))
            top = int(np.count_nonzero(np.asarray(res['mass_enclosed']) < res['cmb_mass']))
            # 20 nodes and the 2 at the CMB step keep the SVG small
            keep = np.union1d(np.linspace(0, n - 1, 20).astype(int), [top - 1, top])
            ax.plot(r[keep], np.asarray(res['temperature'], float)[keep], style, label=label)
        ax.set_xticks([0, 2000, 4000])
        ax.set_xlabel('radius [km]')
        ax.set_ylabel('temperature [K]')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize='small')
        fig.tight_layout()
        return fig

    def difference_figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(KEYS))
        for k, (row, mark, label) in enumerate(
            ((row_np, 'o', 'numpy vs JAX'), (row_fix, 's', 'fixed point vs JAX'))
        ):
            diff = [max(abs(row[key] / row_jax[key] - 1.0), 1e-10) for key in KEYS]
            ax.semilogy(x + 0.15 * (2 * k - 1), diff, mark, label=label)
        ax.axhline(1e-4, ls='--', alpha=0.6, label='tolerance 1e-4', gid='tolerance')
        ax.set_xticks(x, ['R_core', 'R_int', 'P_center', 'P_cmb'], fontsize='small')
        ax.set_ylim(1e-10, 1e-2)
        ax.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
        ax.minorticks_off()
        ax.set_ylabel('relative difference')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize='small')
        fig.tight_layout()
        return fig

    import matplotlib.pyplot as plt
    import proteus_mpl

    for factory, name in (
        (temperature_figure, 'structure_core_table_temperature'),
        (difference_figure, 'structure_core_table_paths'),
    ):
        with plt.rc_context():
            proteus_mpl.use('light')  # the theme of the written plot, which sets the text sizes
            fig = factory()
            fig.canvas.draw()
            ax, renderer = fig.axes[0], fig.canvas.get_renderer()
            data = [ln for ln in ax.lines if ln.get_gid() != 'tolerance']
            points = np.vstack(
                [ax.transData.transform(np.column_stack(ln.get_data())) for ln in data]
            )
            for artist in (ax.get_legend(), ax.xaxis.label, ax.yaxis.label):
                box = artist.get_window_extent(renderer)
                assert not any(box.contains(px, py) for px, py in points)
            ticks = [
                t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_text()
            ]
            assert not any(a.overlaps(b) for i, a in enumerate(ticks) for b in ticks[i + 1 :])
            plt.close(fig)
        validation_plot(factory, name)
