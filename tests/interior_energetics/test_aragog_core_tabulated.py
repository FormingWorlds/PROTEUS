"""The PALEOS tabulated core of Aragog core_module across a structure update.

Module under test: ``proteus.interior_energetics.aragog_core_tabulated`` (``core_content``,
``carried_temperature``) and its use in ``aragog_core.carry_core_across_update``,
``aragog_core_impact.core_call_heat`` and ``aragog_core.write_core_diagnostics``. Invariants: the
core heat content counts from 0 K, ``T C_s(T)`` less the latent and gravitational energy released
below the onset, so a structure term does not depend on the lowest temperature of the liquid
iron table; a structure update keeps the core mass and its total entropy, read from the table;
the ledger jump across the update equals the booked term; the Gaussian profile keeps its heat
content and its Aragog parameters. The arrays are 2 stored Zalmoxis cores of 1 M_E (same core
mass, CMB pressure 127.7 and 141.3 GPa) on a 3 TPa cut of the PALEOS liquid iron table.
"""

from __future__ import annotations

import logging
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('aragog.core.profiles_tabulated')

from proteus.interior_energetics.aragog_core_tabulated import (  # noqa: E402
    carried_temperature,
    core_content,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(600)]

DATA = pathlib.Path(__file__).parent / 'fixtures' / 'core_tabulated'
CUT = DATA / 'paleos_iron_liquid_cut_3tpa.npz'
A_FILE, B_FILE = 'zalmoxis_core_arrays_1me_3000K.npz', 'zalmoxis_core_arrays_1me_2510K.npz'


def _build(arrays, table=CUT):
    from aragog.core.module import build_core_module_budget, load_core_arrays

    a = load_core_arrays(DATA / arrays)
    params = {
        'profile_source': 'paleos',
        'core_liquid_table': str(table),
        'core_profile_file': str(DATA / arrays),
    }
    return build_core_module_budget(
        params,
        r_cmb=float(a['r'][-1]),
        p_cmb_fallback=float(a['P'][-1]),
        m_core=float(a['m'][-1]),
    )


@pytest.fixture(scope='module')
def pair():
    """The core before (1me_3000K) and after (1me_2510K) the update, on the 3 TPa cut."""
    return _build(A_FILE), _build(B_FILE)


@pytest.fixture(scope='module')
def high_cut(tmp_path_factory):
    """The 3 TPa cut without its lowest temperature column: the same table, a higher floor."""
    c = dict(np.load(CUT))
    for k in [k for k in c if np.ndim(c[k]) == 2]:
        c[k] = c[k][:, 1:]
    c['T_K'] = c['T_K'][1:]
    path = tmp_path_factory.mktemp('cut') / 'cut_high_floor.npz'
    np.savez(path, **c)
    return path


def _s_cmb(budget, t):
    return float(budget.profiles.table.value('s', budget.profiles.p_cmb, t))


def _inner_mass(budget, t):
    from proteus.interior_energetics.aragog_core import _inner_core_mass

    return _inner_core_mass(budget, t) if t < budget.t_onset else 0.0


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_structure_term_does_not_see_the_table_floor(pair, high_cut):
    """The carried temperature and the structure term on 2 cuts of one table that differ only
    in the lowest temperature column agree to 1e-12 (measured 2e-16 and 8.5e-14 at 5000 K),
    while the difference of the floor-referenced heat contents moves by more than 1e27 J
    (measured 4.4e27 J): the content counts from 0 K, not from the table floor."""
    old, new = pair
    old_h, new_h = _build(A_FILE, high_cut), _build(B_FILE, high_cut)
    assert float(old_h.profiles.t_grid[0]) > float(old.profiles.t_grid[0]) + 100.0
    t = 5000.0
    t_new, t_new_h = (
        carried_temperature(old, new, t)[0],
        carried_temperature(old_h, new_h, t)[0],
    )
    assert t_new == pytest.approx(t_new_h, rel=1e-12, abs=0.0)
    term = core_content(new, t_new) - core_content(old, t)
    term_h = core_content(new_h, t_new_h) - core_content(old_h, t)
    assert term == pytest.approx(term_h, rel=1e-12, abs=0.0)
    floor = new.heat_content(t_new) - old.heat_content(t)
    floor_h = new_h.heat_content(t_new_h) - old_h.heat_content(t)
    assert abs(floor - floor_h) > 1e27


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
@pytest.mark.parametrize('t', [3800.0, 4100.0, 4500.0, 5200.0])
def test_the_content_of_a_gaussian_budget_is_its_heat_content(t):
    """On the Gaussian profile the content is the heat content of Aragog to 1e-12, above and
    below the onset (near 4147 K for this alloy core)."""
    from aragog.core import build_core_module_budget

    b = build_core_module_budget(
        {'melting_curve': 'iron', 'light_element_fraction': 0.1, 'depression': 1.2},
        r_cmb=3.48e6,
        p_cmb_fallback=136e9,
    )
    assert core_content(b, t) == pytest.approx(b.heat_content(t), rel=1e-12, abs=0.0)
    secular_only = t * float(b.secular_capacity())
    if t < b.t_onset:
        assert abs(core_content(b, t) - secular_only) > 1e-6 * secular_only


def _parcel_term(old, new, t_old, t_new, c_p=840.0, n=256):
    """``c_p int (T_new - T_old) dm`` over n equal-mass parcels of the 2 isentropes."""

    def temps(b, t):
        r = np.linspace(0.0, b.profiles.r_cmb, 4001)
        m = np.asarray(b.profiles.enclosed_mass(r))
        q = (np.arange(n) + 0.5) / n
        return np.asarray(b.profiles.adiabat(np.interp(q * m[-1], m, r), t)), m[-1] / n

    (ta, dm), (tb, _) = temps(old, t_old), temps(new, t_new)
    return c_p * float(np.sum(tb - ta)) * dm


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_update_keeps_the_entropy_and_books_the_parcel_heat(pair, validation_plot):
    """A liquid core keeps its specific entropy at the CMB (1e-10), the structure term equals
    c_p int (T_new - T_old) dm with the override c_p of 840 J/kg/K over equal-mass parcels
    (1e-5; measured 4.7e-6 at 5000 K), and the term is 1.63 times the same integral with the
    table c_p (506 to 523 J/kg/K): the content uses the secular c_p of the budget. A wrong
    sign of the update or a floor-referenced content falls outside these bounds."""
    old, new = pair
    temps = np.linspace(4600.0, 6400.0, 7)
    terms, parcels = [], []
    for t in temps:
        t_new, frozen = carried_temperature(old, new, t)
        assert not frozen and t_new > t  # the compressed core warms
        assert _s_cmb(new, t_new) == pytest.approx(_s_cmb(old, t), rel=1e-10, abs=0.0)
        terms.append(core_content(new, t_new) - core_content(old, t))
        parcels.append(_parcel_term(old, new, t, t_new))
    np.testing.assert_allclose(terms, parcels, rtol=1e-5, atol=0.0)
    t5 = carried_temperature(old, new, 5000.0)[0]
    table_cp = float(old.profiles.table.value('cp', old.profiles.p_cmb, 5000.0))
    ratio = (core_content(new, t5) - core_content(old, 5000.0)) / _parcel_term(
        old, new, 5000.0, t5, c_p=table_cp
    )
    assert 1.5 < ratio < 1.75

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(temps, np.array(terms) / 1e29, 'o-', label='content change T C$_s$ - R')
        ax.plot(temps, np.array(parcels) / 1e29, 'x--', ms=9, label='840 J/kg/K x parcel dT')
        ax.set_xlabel('CMB temperature before the update [K]')
        ax.set_ylabel('structure term [10$^{29}$ J]')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=8)
        fig.tight_layout()
        return fig

    validation_plot(figure, 'core_tabulated_structure_term')


@pytest.mark.physics_invariant
def test_an_update_with_an_inner_core_keeps_the_total_entropy(pair):
    """Below the onset the update keeps M s_c - ds_fusion M_ic to 1e-9 of M s_c, with the mass
    that freezes or melts at the update."""
    old, new = pair
    t = 4100.0  # below the onset of the old core, 4186.5 K
    assert t < old.t_onset
    t_new, frozen = carried_temperature(old, new, t)
    assert frozen
    m, ds = float(old.profiles.m_core), new.ds_fusion
    before = m * _s_cmb(old, t) - ds * _inner_mass(old, t)
    after = m * _s_cmb(new, t_new) - ds * _inner_mass(new, t_new)
    assert after == pytest.approx(before, rel=1e-9, abs=0.0)


def _solver(budget, t_core):
    from aragog.solver.entropy_solver import EXTRA_STATE_SLOTS

    slots = EXTRA_STATE_SLOTS['core_module']
    s0 = np.zeros(len(slots))
    s0[slots.index('T_core')] = t_core
    solver = SimpleNamespace(
        _core_module_budget=budget,
        _r_basic_flat=np.array([budget.profiles.r_cmb]),
        _P_basic_flat=np.array([budget.profiles.p_cmb]),
        _n_stag=0,
        _S0=s0,
        _core_module_q_radio=0.0,
    )
    solver.set_initial_core_temperature = lambda t: s0.__setitem__(slots.index('T_core'), t)
    return solver


def test_the_ledger_jump_across_the_update_is_the_booked_term(pair):
    """The carry books the content change in step_dE_struct_core_J and the next call measures
    the same change as the jump of its start: the core ledger closes across the update."""
    from proteus.interior_energetics.aragog_core import carry_core_across_update
    from proteus.interior_energetics.aragog_core_impact import core_call_heat

    old, new = pair
    t_old = 5000.0
    solver = _solver(new, t_old)
    interior = SimpleNamespace(aragog_solver=solver, _core_t_end=None, _core_struct_booked=0.0)
    hf_row = {
        'P_center': float(new.profiles.pressure(0.0)),
        'M_core': float(new.profiles.m_core),
    }
    carry_core_across_update(hf_row, interior, solver, (old, t_old))
    t_new = carried_temperature(old, new, t_old)[0]
    booked = interior._core_struct_booked
    assert booked == pytest.approx(
        core_content(new, t_new) - core_content(old, t_old), rel=1e-14
    )
    out = SimpleNamespace(step_dE_core_J=0.0, dt_actual=0.0)
    assert core_call_heat(out, interior, lift=booked) == pytest.approx(booked, rel=1e-14)


def test_the_dynamo_columns_are_zero_on_the_tabulated_profile(pair, caplog):
    """The entropy budget of Aragog takes a constant alpha and c_p, which the tabulated profile
    does not have: the dynamo margin and the field strength are 0.0, the other core columns are
    computed, and the skip is logged once."""
    from proteus.interior_energetics.aragog_core import write_core_diagnostics

    _, new = pair
    cm = SimpleNamespace(k_core=130.0, f_ohm=1.0, flux_geometry='const_flux', q_radio=0.0)
    runner = SimpleNamespace(
        aragog_solver=SimpleNamespace(_core_module_budget=new),
        _config=SimpleNamespace(
            interior_energetics=SimpleNamespace(aragog=SimpleNamespace(core_module=cm))
        ),
    )
    with caplog.at_level(logging.INFO, logger='fwl.proteus'):
        for _ in range(2):
            output = {'T_cmb': 5000.0, 'F_cmb': 0.05, 'step_dE_F_cmb_J': 0.0}
            write_core_diagnostics(runner, output, dt_actual_yr=0.0)
            assert output['core_dynamo_margin'] == 0.0 and output['core_B_rms'] == 0.0
            assert output['core_C_eff'] > 0.0 and output['core_regime'] == 0.0
    assert sum('dynamo' in r.message for r in caplog.records) == 1


def test_the_gaussian_source_gives_aragog_the_parameters_of_main():
    """With the Gaussian default the parameters PROTEUS gives Aragog hold no tabulated key:
    the same set as before the profile source existed."""
    from proteus.config._interior import AragogCoreModule
    from proteus.interior_energetics.aragog_core import core_module_params

    config = SimpleNamespace(
        interior_energetics=SimpleNamespace(
            aragog=SimpleNamespace(core_module=AragogCoreModule())
        ),
        params=SimpleNamespace(resume=False),
        interior_struct=SimpleNamespace(module='zalmoxis'),
    )
    params = core_module_params(
        config, {'M_core': 1.94e24, 'P_center': 3.6e11}, SimpleNamespace(), ''
    )
    main = {
        'alpha', 'c_p', 'melting_curve', 'light_element_fraction', 'depression', 't_m0', 't_m1',
        't_m2', 'ds_fusion', 'alpha_c', 'c_light', 'q_radio', 'stratification', 'k_core',
        'layer_base_fraction', 'layer_k_mix', 'layer_g_mix', 'ra_crit_cmb', 'm_core', 'p_cen',
    }  # fmt: skip
    assert set(params) == main
