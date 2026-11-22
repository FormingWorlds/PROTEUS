"""Tests of the core temperature table that a structure solve hands to Zalmoxis."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from proteus.interior_energetics.wrapper import _structure_core_table
from proteus.interior_struct.core_temperature import (
    core_temperature_table,
    structure_core_table,
)

pytestmark = pytest.mark.unit

ROW = {
    'M_core': 1.94e24,
    'P_center': 3.6e11,
    'R_core': 3.48e6,
    'P_cmb': 1.36e11,
    'T_cmb': 4300.0,
}


def _config(core_bc='energy_balance', alpha=1.35e-5, c_p=840.0):
    core = SimpleNamespace(alpha=alpha, c_p=c_p)
    return SimpleNamespace(
        interior_energetics=SimpleNamespace(
            aragog=SimpleNamespace(core_bc=core_bc, core_module=core)
        )
    )


def _gaussian(alpha=1.35e-5, c_p=840.0, **row):
    from aragog.core.profiles import GaussianCoreProfiles

    r = {**ROW, **row}
    return GaussianCoreProfiles.from_structure(
        m_core=r['M_core'],
        p_cen=r['P_center'],
        r_cmb=r['R_core'],
        p_cmb=r['P_cmb'],
        alpha=alpha,
        c_p=c_p,
    )


def _live(t_core, profiles):
    """Interior object with an Aragog solver holding a core budget at ``t_core``."""
    solver = SimpleNamespace(
        _core_module_budget=SimpleNamespace(profiles=profiles),
        get_current_core_temperature=lambda: t_core,
    )
    return SimpleNamespace(aragog_solver=solver)


@pytest.mark.physics_invariant
def test_energy_balance_fits_a_gaussian_core_to_the_row_through_t_cmb():
    """Without a live Aragog core the table is the adiabat of a Gaussian core fitted to the row
    structure with the config alpha and c_p, through the row T_cmb: it starts at (P_cmb, T_cmb)
    and equals the table of the same fit built directly; another c_p gives another table."""
    p, t = _structure_core_table(_config(), SimpleNamespace(), dict(ROW))
    pr, tr = core_temperature_table(_gaussian(), 4300.0)
    assert np.array_equal(p, pr) and np.array_equal(t, tr)
    assert p[0] == pytest.approx(1.36e11, rel=1e-12) and t[0] == pytest.approx(
        4300.0, rel=1e-12
    )
    _, t2 = _structure_core_table(_config(c_p=420.0), SimpleNamespace(), dict(ROW))
    assert t2[-1] > t[-1] * 1.05


@pytest.mark.physics_invariant
def test_a_live_aragog_core_sets_the_table_through_its_own_temperature():
    """With core_bc = "core_module" and a live budget, the table is the adiabat of the budget
    profiles through the Aragog core temperature, not a fit to the row through T_cmb."""
    profiles = _gaussian(P_cmb=1.30e11)
    p, t = _structure_core_table(_config('core_module'), _live(5100.0, profiles), dict(ROW))
    pr, tr = core_temperature_table(profiles, 5100.0)
    assert np.array_equal(p, pr) and np.array_equal(t, tr)
    assert t[0] == pytest.approx(5100.0, rel=1e-12)
    assert p[0] == pytest.approx(1.30e11, rel=1e-12)


@pytest.mark.parametrize('case', ['anchor given', 'not core_module'])
def test_the_row_fit_is_used_with_an_anchor_or_without_core_module(case):
    """An explicit anchor (the start-up passes) or a core_bc other than core_module gives the
    Gaussian fit to the row, even when the interior object holds an Aragog budget."""
    live = _live(5100.0, _gaussian(P_cmb=1.30e11))
    if case == 'anchor given':
        p, t = _structure_core_table(_config('core_module'), live, dict(ROW), t_anchor=4800.0)
        anchor = 4800.0
    else:
        p, t = _structure_core_table(_config('energy_balance'), live, dict(ROW))
        anchor = 4300.0
    pr, tr = structure_core_table(_config(), dict(ROW), anchor)
    assert np.array_equal(p, pr) and np.array_equal(t, tr)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('row_t_cmb', 'anchor'), [(0.0, 'deepest'), (None, 'deepest'), (4300.0, 4300.0)]
)
def test_a_re_solve_anchors_the_table_at_the_row_t_cmb_or_the_deepest_mantle_t(
    tmp_path, row_t_cmb, anchor
):
    """A structure re-solve anchors the core table at the row T_cmb; a row without a positive
    T_cmb (a dummy interior writes none) takes the deepest temperature of the mantle profile
    it hands to Zalmoxis, as the start-up passes do."""
    from unittest.mock import MagicMock, patch

    from proteus.interior_energetics.common import Interior_t
    from proteus.interior_energetics.wrapper import update_structure_from_interior

    config = MagicMock()
    z = config.interior_struct.zalmoxis
    z.update_interval, z.update_min_interval, z.update_stale_ceiling = 100.0, 0.0, 0.0
    z.update_dtmagma_frac, z.update_dphi_abs, z.num_levels = 0.03, 0.05, 50
    z.mesh_max_shift, z.mesh_convergence_interval = 0.05, 10.0
    config.planet.temperature_mode = 'isothermal'
    config.interior_energetics.module = 'spider'
    config.interior_energetics.aragog = _config().interior_energetics.aragog
    (tmp_path / 'data').mkdir()
    dirs = {'output': str(tmp_path), 'spider': str(tmp_path), 'mesh_shift_active': False}
    interior_o = Interior_t(50)
    interior_o.radius = np.linspace(6.371e6, 3.504e6, 50)
    interior_o.temp = np.linspace(2000.0, 3600.0, 49)  # surface-first, as SPIDER stores it
    row = {
        **ROW,
        'Time': 500.0,
        'T_magma': 2000.0,
        'Phi_global': 0.8,
        'R_int': 6.371e6,
        'gravity': 9.81,
    }
    row['M_int'] = row['M_int_target'] = 5.972e24
    if row_t_cmb is None:
        del row['T_cmb']
    else:
        row['T_cmb'] = row_t_cmb

    def solve(config, outdir, hf_row, **kwargs):
        hf_row['M_int'] = hf_row['M_int_target'] = 5.972e24
        return (3.504e6, 'mesh.dat')

    with (
        patch('proteus.interior_struct.zalmoxis.zalmoxis_solver', side_effect=solve) as solver,
        patch('proteus.interior_energetics.wrapper.np.savetxt'),
        patch('proteus.interior_energetics.wrapper.shutil.copy2'),
        patch('proteus.interior_energetics.spider.blend_mesh_files', return_value=0.0),
    ):
        update_structure_from_interior(dirs, config, row, interior_o, 0.0, 2000.0, 0.8)
    kw = solver.call_args.kwargs
    want = kw['temperature_arrays'][1][0] if anchor == 'deepest' else anchor
    assert want == pytest.approx(3600.0 if anchor == 'deepest' else 4300.0, rel=1e-12)
    p, t = kw['core_temperature_arrays']
    assert t[0] == pytest.approx(want, rel=1e-12)
    assert p[0] == pytest.approx(ROW['P_cmb'], rel=1e-9)
