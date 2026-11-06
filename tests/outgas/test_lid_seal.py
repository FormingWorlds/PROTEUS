"""Tests of ``proteus.outgas.lid_seal``: the melt inventory held under a closed crust.

Invariants checked: the sealed reservoir and the melt exchange mass without loss
(close and open conserve every species, element and mole column); the leak of
one step equals the drop of the sealed mass, per species and per element, and
splits between relaxation and volcanism in the ratio of their rates; the exact
step solution does not depend on the step length; the melt concentration of the
sealed reservoir satisfies its mass balance with the global melt fraction; the
crust depth is bounded by the lower face of the top solid run.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from proteus.config._outgas import LidSeal
from proteus.outgas.common import element_masses_from_species
from proteus.outgas.compaction import BRANCH_REMELT
from proteus.outgas.lid_seal import (
    SEALED_ELEMENTS,
    SEALED_SPECIES,
    assert_sealed_row,
    close_seal,
    closure_state,
    crust_thickness,
    leak_rates,
    leak_step,
    open_seal,
    restore_seal_state,
    run_lid_seal_step,
    seal_enabled,
    sealed_mass,
)
from proteus.utils.constants import gas_list
from proteus.utils.helper import eval_gas_mmw

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

M_MANTLE = 4.0e24  # [kg], an Earth-like mantle
R_INT = 6.371e6  # [m]
GRAVITY = 9.81  # [m s-2]
# A row holding H2O and CO2 in the melt and the atmosphere; masses chosen apart.
MELT = {'H2O': 3.0e20, 'CO2': 7.0e19}
ATM = {'H2O': 1.1e20, 'CO2': 4.0e19}


def _config(**seal):
    """Config stand-in with a real LidSeal and the outgas partition coefficients."""
    outgas = SimpleNamespace(lid_seal=LidSeal(enabled=True, **seal))
    for s in SEALED_SPECIES:
        setattr(outgas, f'D_const_{s}', 0.0)
    outgas.D_const_H2O = 0.0017
    return SimpleNamespace(outgas=outgas, escape=SimpleNamespace(reservoir='outgas'))


def _row(phi=0.4):
    """A helpfile row with species and element reservoirs filled by stoichiometry."""
    row = {'Time': 1.0e4, 'M_mantle': M_MANTLE, 'Phi_global': phi, 'R_int': R_INT}
    row['gravity'] = GRAVITY
    for s in gas_list:
        for r in ('atm', 'liquid', 'solid', 'sealed'):
            row[f'{s}_kg_{r}'] = 0.0
            row[f'{s}_mol_{r}'] = 0.0
    for s, kg in MELT.items():
        row[f'{s}_kg_liquid'] = kg
        row[f'{s}_mol_liquid'] = kg / eval_gas_mmw(s)
    for s, kg in ATM.items():
        row[f'{s}_kg_atm'] = kg
        row[f'{s}_mol_atm'] = kg / eval_gas_mmw(s)
    for e in SEALED_ELEMENTS:
        row[f'{e}_kg_liquid'] = element_masses_from_species(MELT).get(e, 0.0)
        row[f'{e}_kg_atm'] = element_masses_from_species(ATM).get(e, 0.0)
        row[f'{e}_kg_sealed'] = 0.0
        row[f'{e}_kg_total'] = row[f'{e}_kg_liquid'] + row[f'{e}_kg_atm']
    row['M_atm'] = sum(ATM.values())
    return row


def _sealed_row(phi=0.4):
    row = _row(phi)
    close_seal(row)
    return row


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('phi', 'expected'),
    [
        ([0.5, 0.0, 0.0, 0.0], 3.0),  # three solid cells over a molten base
        ([0.0, 0.5, 0.0, 0.0], 2.0),  # a solid cell under the melt does not count
        ([0.5, 0.5, 0.5, 0.0], 1.0),
        ([0.0, 0.0, 0.0, 0.0], 4.0),  # all solid: down to the core face
        ([0.0, 0.0, 0.0, 0.5], 0.0),  # molten top cell: no crust
        ([0.0, 0.0, 0.0, 1e-3], 0.0),  # phi == phi_solid is not crust
    ],
    ids=['three-cells', 'buried-solid', 'one-cell', 'all-solid', 'molten-top', 'at-threshold'],
)
def test_crust_thickness_is_the_top_solid_run_to_its_lower_face(phi, expected):
    """Crust depth reaches the lower face of the deepest cell of the top solid run.

    Uneven faces (1, 3, 6, 10, 15 m) make the lower face differ from the upper face
    of the same cell, so an off-by-one face index gives another depth.
    """
    radius = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
    depth = crust_thickness(np.array(phi), radius, 1e-3)
    face = {0.0: 15.0, 1.0: 10.0, 2.0: 6.0, 3.0: 3.0, 4.0: 1.0}[expected]
    assert depth == pytest.approx(15.0 - face, abs=0.0)
    assert 0.0 <= depth <= radius[-1] - radius[0]


@pytest.mark.parametrize(
    ('phi', 'radius'),
    [([0.0, 0.0], [1.0, 2.0]), ([0.0], [2.0, 1.0]), ([], [1.0])],
    ids=['faces-short', 'radius-decreasing', 'no-cells'],
)
def test_crust_thickness_refuses_a_malformed_mesh(phi, radius):
    """A mesh with the wrong face count or a falling radius is refused."""
    with pytest.raises(ValueError, match='crust_thickness'):
        crust_thickness(np.array(phi), np.array(radius), 1e-3)
    phi_in, radius_in = np.array(phi), np.array(radius)
    with pytest.raises(ValueError):
        crust_thickness(phi_in, radius_in, 1e-3)
    np.testing.assert_array_equal(radius_in, np.array(radius))  # input left unchanged


def _interior(phi, radius=None):
    phi = np.asarray(phi, dtype=float)
    radius = np.linspace(3.4e6, R_INT, phi.size + 1) if radius is None else radius
    return SimpleNamespace(phi=phi, radius=radius, impact_reset=False)


@pytest.mark.physics_invariant
def test_closure_state_reads_the_sub_grid_lid_only_while_it_acts():
    """With lid_weight > 0 the thickness is lid_delta; at lid_weight 0 a stored
    lid_delta (the hidden thickness of a lid that does not act, 0.487 m) is
    ignored and the mesh crust is read instead."""
    cfg = _config()
    interior = _interior([0.6, 0.25])
    acting = closure_state(cfg, {'lid_weight': 1.0, 'lid_delta': 0.31}, interior)
    hidden = closure_state(cfg, {'lid_weight': 0.0, 'lid_delta': 0.487}, interior)
    assert acting == (0.31, 0.25)
    assert hidden == (0.0, 0.25)
    with pytest.raises(KeyError, match='lid_delta'):
        closure_state(cfg, {'lid_weight': 0.5}, interior)
    with pytest.raises(ValueError, match='not finite'):
        closure_state(cfg, {'lid_weight': 1.0, 'lid_delta': float('nan')}, interior)


@pytest.mark.physics_invariant
def test_close_then_open_returns_every_column_unchanged():
    """Closing moves the whole melt inventory to the sealed reservoir and leaves
    the totals; opening moves it back, so the round trip restores the row."""
    row = _row()
    before = dict(row)
    moved = close_seal(row)
    assert moved == pytest.approx(sum(MELT.values()), rel=1e-15)
    for n in (*SEALED_SPECIES, *SEALED_ELEMENTS):
        assert row[f'{n}_kg_liquid'] == 0.0
        assert row[f'{n}_kg_sealed'] == before[f'{n}_kg_liquid']
    assert row['H2O_mol_sealed'] == before['H2O_mol_liquid']
    assert row['H_kg_total'] == before['H_kg_total']
    assert_sealed_row(row)
    assert open_seal(row) == moved
    assert row['seal_open_kg'] == moved
    assert {k: row[k] for k in before} == before


@pytest.mark.parametrize('bad', [float('nan'), -1.0], ids=['nan', 'negative'])
def test_close_refuses_an_unreadable_melt_column_without_moving_mass(bad):
    """A NaN or negative melt mass stops the close before any column changes."""
    row = _row()
    row['O_kg_liquid'] = bad
    before = dict(row)
    with pytest.raises(RuntimeError, match='O_kg_liquid'):
        close_seal(row)
    assert {k: row[k] for k in before if k != 'O_kg_liquid'} == {
        k: before[k] for k in before if k != 'O_kg_liquid'
    }
    assert row['H2O_kg_sealed'] == 0.0


def test_assert_sealed_row_refuses_melt_mass_and_a_nan_reservoir():
    """A sealed row with melt mass left, or with a NaN sealed mass, fails."""
    row = _sealed_row()
    row['CO2_kg_liquid'] = 1.0
    with pytest.raises(RuntimeError, match='CO2_kg_liquid'):
        assert_sealed_row(row)
    row = _sealed_row()
    row['C_kg_sealed'] = float('nan')
    with pytest.raises(RuntimeError, match='C_kg_sealed'):
        assert_sealed_row(row)
    assert sealed_mass(row, 'C') == 0.0
    assert sealed_mass(_sealed_row(), 'C') > 1e19


@pytest.mark.physics_invariant
def test_hold_keeps_every_reservoir_bit_identical():
    """V2: with tau_leak = inf and crust_rate = 0 (the defaults) nothing leaks,
    and the atmosphere is not recomputed: every column stays equal to the bit."""
    row = _sealed_row()
    before = dict(row)
    for _ in range(5):
        leak_step(_config(), row, 1.0e5)
    assert {k: row[k] for k in before} == before
    assert row['H_kg_leak_tau'] == row['H_kg_leak_volc'] == 0.0


def _assert_legend_clear(fig):
    """No legend box holds a vertex of a plotted line or marker, in display coordinates."""
    fig.canvas.draw()
    for ax in fig.axes:
        boxes = [leg.get_window_extent() for leg in (ax.get_legend(),) if leg is not None]
        for line in ax.get_lines():
            xy = ax.transData.transform(np.column_stack(line.get_data()))
            for b in boxes:
                inside = (
                    (b.x0 <= xy[:, 0])
                    & (xy[:, 0] <= b.x1)
                    & (b.y0 <= xy[:, 1])
                    & (xy[:, 1] <= b.y1)
                )
                assert not inside.any(), line.get_label()


def _relax(dt, t_end, tau):
    row = _sealed_row()
    cfg = _config(tau_leak=tau)
    t, out = 0.0, {0.0: row['H2O_kg_sealed']}
    while t < t_end - 1e-9:
        leak_step(cfg, row, dt)
        t += dt
        out[round(t)] = row['H2O_kg_sealed']
    return out, row


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_relaxation_matches_the_exponential_and_not_the_step_length(validation_plot):
    """V3: with crust_rate = 0 the sealed mass decays as M_c exp(-(t - t_c)/tau),
    to 1e-12 at every row, for steps of 1, 10 and 100 kyr (tau = 1 Myr). The
    step solution is exact, so the step length enters only at round-off; a
    forward Euler step would be off by 5.5 % after 1 Myr at 100 kyr steps."""
    tau, m0 = 1.0e6, MELT['H2O']
    runs = {dt: _relax(dt, 1.0e6, tau)[0] for dt in (1.0e3, 1.0e4, 1.0e5)}
    for series in runs.values():
        for t, m in series.items():
            assert m == pytest.approx(m0 * np.exp(-t / tau), rel=1e-12)
    for t in runs[1.0e5]:
        assert runs[1.0e3][t] == pytest.approx(runs[1.0e5][t], rel=1e-12)
    euler = m0 * (1.0 - 1.0e5 / tau) ** 10
    assert abs(runs[1.0e5][1000000] - euler) / euler > 0.05

    def figure():
        import matplotlib.pyplot as plt

        plt.rcParams['svg.fonttype'] = 'none'  # text as SVG text, not glyph paths
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, height_ratios=[3, 1])
        x = np.linspace(0.0, 1.0, 101)
        ax.plot(x, np.exp(-x), label='closed form')
        for dt, marker in zip(runs, ('o', 's', '^')):
            t = np.array(sorted(runs[dt]))[:: max(1, int(2.0e5 / dt))]
            m = np.array([runs[dt][k] for k in t]) / m0
            ax.plot(t / tau, m, marker, mfc='none', label=f'{dt / 1e3:.0f} kyr steps')
            ax2.plot(t / tau, m / np.exp(-t / tau) - 1.0, marker, mfc='none')
        ax.set_ylabel('Sealed fraction')
        ax.legend(loc='upper right')
        ax2.set_xlabel(r'Time since closure / $\tau$')
        ax2.set_ylabel('Error')
        ax2.set_ylim(-2e-12, 2e-12)
        ax2.set_yticks([-1e-12, 0.0, 1e-12], ['-1e-12', '0', '1e-12'])
        ax2.set_xticks([0.0, 0.5, 1.0])
        ax.set_yticks([0.4, 0.7, 1.0])
        fig.tight_layout()
        _assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'outgas_lid_seal_relaxation')


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_volcanic_rate_closes_the_mass_balance_of_the_sealed_reservoir(validation_plot):
    """V4: the volcanic rate k_i = chi_d Mdot_c / (M_melt + D_i M_solid) of the
    Schaefer & Sasselov (2015, ApJ 801:40) Eq. 14 and Eq. 17 form, with the
    global melt fraction. (a) X_melt (M_melt + D M_solid) == M_i; (b) half the
    mantle solid, half at phi 0.4, D = 0: X_melt = M_i / (0.2 M_mantle), which a
    local melt fraction (0.4) halves; (c) Eq. 14 written out with test inputs."""
    cfg = _config(crust_rate=1.0e12, chi_d=0.02)
    rates, mdot, capped = leak_rates(cfg, {'M_mantle': M_MANTLE, 'Phi_global': 0.2}, 1.0)
    assert (mdot, capped) == (1.0e12, False)
    m_i = 1.0e20
    k_co2 = rates['CO2'][1]  # D = 0
    x_melt = k_co2 * m_i / (0.02 * mdot)
    assert x_melt * 0.2 * M_MANTLE == pytest.approx(m_i, rel=1e-14)
    assert x_melt == pytest.approx(m_i / (0.2 * M_MANTLE), rel=1e-14)
    assert abs(x_melt - m_i / (0.4 * M_MANTLE)) / x_melt > 0.4  # local melt fraction
    k_h2o = rates['H2O'][1]  # D = 0.0017
    x_h2o = k_h2o * m_i / (0.02 * mdot)
    assert x_h2o * (0.2 * M_MANTLE + 0.0017 * 0.8 * M_MANTLE) == pytest.approx(m_i, rel=1e-14)
    # Eq. 14, r = rho_m F D_melt S X chi_d, with rho_m F D_melt S = Mdot_c in kg yr-1
    r_hand = 1.0e12 * (m_i / (M_MANTLE * (0.0017 + 0.2 * (1 - 0.0017)))) * 0.02
    assert k_h2o * m_i == pytest.approx(r_hand, rel=1e-12)
    assert rates['H2O'][0] == 0.0  # tau_leak = inf

    grid = np.geomspace(0.01, 1.0, 5)
    curves = {}
    for d in (0.0, 0.01, 0.1, 1.0):
        cfg_d = _config(crust_rate=1.0e12)
        cfg_d.outgas.D_const_CO2 = d
        k = [
            leak_rates(cfg_d, {'M_mantle': M_MANTLE, 'Phi_global': p}, 1.0)[0]['CO2'][1]
            for p in grid
        ]
        curves[d] = np.array(k) * M_MANTLE / (0.02 * 1.0e12)
        assert curves[d] == pytest.approx(1.0 / (d + grid * (1.0 - d)), rel=1e-13)

    def figure():
        import matplotlib.pyplot as plt

        plt.rcParams['svg.fonttype'] = 'none'  # text as SVG text, not glyph paths
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.linspace(0.01, 1.0, 200)
        for d, marker in zip(curves, ('o', 's', '^', 'D')):
            (line,) = ax.plot(x, 1.0 / (d + x * (1.0 - d)), label=f'D = {d:g}')
            ax.plot(grid, curves[d], marker, mfc='none', color=line.get_color())
        ax.plot([], [], 'o', mfc='none', color=plt.rcParams['text.color'], label='code')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Global melt fraction $\Phi$')
        ax.set_ylabel('Scaled volcanic rate')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
        fig.tight_layout()
        _assert_legend_clear(fig)
        return fig

    validation_plot(figure, 'outgas_lid_seal_volcanic_rate')


@pytest.mark.physics_invariant
def test_volcanic_leak_splits_by_rate_and_closes_per_element():
    """crust_rate > 0 with a finite tau: per element, the 2 leak columns sum to
    the drop of the sealed mass; per species the split is (1/tau) : k_i; the
    atmosphere gains what the sealed reservoir loses."""
    cfg = _config(tau_leak=5.0e5, crust_rate=3.0e20)  # k_i = 5e-6 yr-1 against 1/tau = 2e-6
    row = _sealed_row(phi=0.3)
    before = dict(row)
    dt = 2.0e3
    rates, _, _ = leak_rates(cfg, row, dt)
    leak_step(cfg, row, dt)
    for e in SEALED_ELEMENTS:
        drop = before[f'{e}_kg_sealed'] - row[f'{e}_kg_sealed']
        out = row[f'{e}_kg_leak_tau'] + row[f'{e}_kg_leak_volc']
        assert out == pytest.approx(drop, rel=1e-12, abs=1e-6)
        assert row[f'{e}_kg_atm'] - before[f'{e}_kg_atm'] == pytest.approx(
            out, rel=1e-12, abs=1e-6
        )
    inv_tau, k = rates['CO2']
    assert row['C_kg_leak_volc'] / row['C_kg_leak_tau'] == pytest.approx(k / inv_tau, rel=1e-12)
    assert row['C_kg_leak_volc'] > row['C_kg_leak_tau'] > 0.0
    drop = before['CO2_kg_sealed'] - row['CO2_kg_sealed']
    assert drop == pytest.approx(
        before['CO2_kg_sealed'] * -np.expm1(-dt * (inv_tau + k)), rel=1e-12
    )
    assert row['seal_leak_kg_cum'] > 0.0
    assert row['seal_crust_rate'] == 3.0e20
    assert row['seal_volc_capped'] == 0.0


@pytest.mark.physics_invariant
def test_crust_production_is_capped_at_the_melt_and_stops_below_phi_solid():
    """Mdot_c dt cannot exceed the melt mass: a step that asks for more uses
    M_melt / dt and flags it, so k dt stays at or below chi_d; below phi_solid
    no crust forms."""
    cfg = _config(crust_rate=1.0e24, chi_d=0.5)
    rates, mdot, capped = leak_rates(cfg, {'M_mantle': M_MANTLE, 'Phi_global': 0.01}, 100.0)
    assert capped is True
    assert mdot == pytest.approx(0.01 * M_MANTLE / 100.0, rel=1e-15)
    assert rates['CO2'][1] * 100.0 == pytest.approx(0.5, rel=1e-14)
    assert rates['H2O'][1] * 100.0 < 0.5  # D > 0 spreads the volatile over the crystals
    rates, mdot, capped = leak_rates(cfg, {'M_mantle': M_MANTLE, 'Phi_global': 5e-4}, 100.0)
    assert (mdot, capped, rates['CO2'][1]) == (0.0, False, 0.0)
    row = _sealed_row(phi=0.01)
    leak_step(cfg, row, 100.0)
    assert row['seal_volc_capped'] == 1.0


@pytest.mark.physics_invariant
def test_leak_refreshes_the_atmosphere_from_the_masses():
    """After a leak, P_surf = M_atm g / (4 pi R^2), the mixing ratios sum to 1
    and follow the moles, and the mean molar mass is M_atm over the moles."""
    cfg = _config(tau_leak=1.0e4)
    row = _sealed_row()
    leak_step(cfg, row, 1.0e3)
    m_atm = sum(row[f'{s}_kg_atm'] for s in gas_list)
    assert row['M_atm'] == pytest.approx(m_atm, rel=1e-15)
    assert row['P_surf'] == pytest.approx(
        m_atm * GRAVITY / (4 * np.pi * R_INT**2) / 1e5, rel=1e-14
    )
    assert row['P_vol'] == row['P_surf']
    assert sum(row[f'{s}_vmr'] for s in gas_list) == pytest.approx(1.0, rel=1e-14)
    n = sum(row[f'{s}_mol_atm'] for s in gas_list)
    assert row['H2O_vmr'] == pytest.approx(row['H2O_mol_atm'] / n, rel=1e-14)
    assert row['atm_kg_per_mol'] == pytest.approx(m_atm / n, rel=1e-14)
    assert row['H2O_bar'] == pytest.approx(row['H2O_vmr'] * row['P_surf'], rel=1e-14)


def _step(row, interior, sealed, *, init_stage=False, **seal):
    cfg = _config(**seal)
    with (
        patch('proteus.outgas.wrapper.run_outgassing_and_vapourisation') as solve,
        patch('proteus.outgas.wrapper.check_ic_oxygen_budget'),
    ):
        state = run_lid_seal_step(
            {}, cfg, row, interior, sealed, first_iter=False, init_stage=init_stage, dt=10.0
        )
    return state, solve


def test_step_closes_after_the_solve_and_not_in_the_init_stage():
    """An open seal solves the chemistry, then closes when the crust is thick
    enough and the top cell below phi_close; never in the init stage."""
    interior = _interior([0.6, 0.2])
    row = {**_row(), 'lid_weight': 1.0, 'lid_delta': 0.31}
    state, solve = _step(row, interior, False, delta_close=0.3)
    assert state is True and row['seal_active'] == 1.0
    solve.assert_called_once()
    assert row['H2O_kg_liquid'] == 0.0 and row['H2O_kg_sealed'] == MELT['H2O']
    assert row['seal_t_close'] == row['Time']
    row = {**_row(), 'lid_weight': 1.0, 'lid_delta': 0.31}
    assert _step(row, interior, False, init_stage=True, delta_close=0.3)[0] is False
    row = {**_row(), 'lid_weight': 1.0, 'lid_delta': 0.29}
    assert _step(row, interior, False, delta_close=0.3)[0] is False
    assert row['seal_delta'] == 0.29 and row['seal_phi_top'] == 0.2


@pytest.mark.parametrize(
    ('phi_top', 'delta', 'impact', 'still_sealed'),
    [
        (0.29, 0.31, False, True),
        (0.33, 0.31, False, True),  # inside the band phi_close .. phi_close + 0.05
        (0.36, 0.31, False, False),
        (0.29, 0.16, False, True),  # thinner than delta_close, above 0.5 delta_close
        (0.29, 0.14, False, False),
        (0.29, 0.31, True, False),  # a giant impact always opens
    ],
    ids=['held', 'phi-band', 'phi-opens', 'delta-band', 'delta-opens', 'impact'],
)
def test_sealed_step_opens_only_past_the_band(phi_top, delta, impact, still_sealed):
    """V9: the seal opens at phi_top >= phi_close + 0.05, at a crust thinner than
    0.5 delta_close, or on an impact; on opening the sealed mass returns in full
    and the solve of the same step runs; a held step does not solve."""
    interior = _interior([0.6, phi_top])
    interior.impact_reset = impact
    row = {**_sealed_row(), 'lid_weight': 1.0, 'lid_delta': delta, 'Time': 2.0e4}
    row.update({'esc_rate_total': 0.0, 'M_atm': sum(ATM.values())})
    state, solve = _step(row, interior, True, delta_close=0.3)
    assert state is still_sealed
    assert solve.call_count == (0 if still_sealed else 1)
    if still_sealed:
        assert row['H2O_kg_sealed'] == MELT['H2O'] and row['H2O_kg_liquid'] == 0.0
    else:
        assert row['H2O_kg_liquid'] == MELT['H2O'] and row['H2O_kg_sealed'] == 0.0
        assert row['seal_open_kg'] == pytest.approx(sum(MELT.values()), rel=1e-15)


def test_seal_enabled_reads_only_a_true_switch():
    """The seal is on only for a LidSeal with enabled = True; a config built in
    code without the table, or with a non-bool value, leaves it off."""
    assert seal_enabled(_config()) is True
    assert seal_enabled(SimpleNamespace(outgas=SimpleNamespace())) is False
    assert seal_enabled(SimpleNamespace(outgas=SimpleNamespace(lid_seal=LidSeal()))) is False
    odd = SimpleNamespace(outgas=SimpleNamespace(lid_seal=SimpleNamespace(enabled=1)))
    assert seal_enabled(odd) is False


@pytest.mark.physics_invariant
def test_leak_step_moves_species_masses_and_moles_together():
    """After a leak step each species keeps the same molar mass in the sealed and
    the atmospheric reservoir: sealed moles fall by the factor of the sealed mass,
    and the atmosphere gains exactly the leaked mass and moles."""
    cfg = _config(tau_leak=2.0e4)
    row = _sealed_row()
    before = dict(row)
    leak_step(cfg, row, 5.0e3)
    keep = np.exp(-5.0e3 / 2.0e4)
    for s in MELT:
        mol0, kg0 = before[f'{s}_mol_sealed'], before[f'{s}_kg_sealed']
        assert row[f'{s}_kg_sealed'] == pytest.approx(kg0 * keep, rel=1e-14)
        assert row[f'{s}_mol_sealed'] == pytest.approx(mol0 * keep, rel=1e-14)
        leaked = kg0 - row[f'{s}_kg_sealed']
        assert row[f'{s}_kg_atm'] - before[f'{s}_kg_atm'] == pytest.approx(leaked, rel=1e-12)
        gained = row[f'{s}_mol_atm'] - before[f'{s}_mol_atm']
        assert gained == pytest.approx(mol0 * (1.0 - keep), rel=1e-12)
        assert row[f'{s}_kg_atm'] / row[f'{s}_mol_atm'] == pytest.approx(
            eval_gas_mmw(s), rel=1e-12
        )


@pytest.mark.physics_invariant
def test_a_step_that_drains_the_reservoir_passes_the_ledger_check():
    """A step much longer than tau drains the sealed mass; round-off in the 2
    leak parts must not trip the ledger check, and no sealed mass is negative."""
    cfg = _config(tau_leak=3.0, crust_rate=7.0e21, chi_d=0.7)
    row = _sealed_row(phi=0.37)
    before = dict(row)
    leak_step(cfg, row, 1.0e4)
    for e in SEALED_ELEMENTS:
        out = row[f'{e}_kg_leak_tau'] + row[f'{e}_kg_leak_volc']
        assert out == pytest.approx(before[f'{e}_kg_sealed'], rel=1e-12)
        assert row[f'{e}_kg_sealed'] >= 0.0
    assert row['H2O_kg_sealed'] == 0.0


def test_hold_leaves_the_written_atmosphere_untouched():
    """With nothing leaking, the atmosphere columns the chemistry wrote stay as
    they are, even where they differ from what the leak would rebuild."""
    row = _sealed_row()
    row.update({'H2O_bar': 1.234, 'H2O_vmr': 0.5, 'P_surf': 2.468, 'atm_kg_per_mol': 0.03})
    before = dict(row)
    leak_step(_config(), row, 1.0e3)
    assert {k: row[k] for k in before} == before
    assert row['seal_leak_kg_cum'] == 0.0


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('phi', 'crust', 'dt', 'mdot', 'capped'),
    [
        (1e-3, 1.0e10, 1.0, 1.0e10, False),  # phi == phi_solid still makes crust
        (1.2, 1.0e10, 1.0, 1.0e10, False),  # phi above 1 reads as 1
        (-0.1, 1.0e10, 1.0, 0.0, False),  # phi below 0 reads as 0
        (0.25, 1.0e24, 1.0, 1.0e24, False),  # crust_rate dt == M_melt is not capped
        (0.0, 0.0, 1.0, 0.0, False),  # no melt, no crust, no division by zero
    ],
    ids=['at-phi-solid', 'phi-above-1', 'phi-below-0', 'at-the-cap', 'no-melt'],
)
def test_leak_rates_at_their_boundaries(phi, crust, dt, mdot, capped):
    """Boundaries of the volcanic rate: phi_solid is inclusive, Phi is clamped to
    [0, 1], the cap is strict, and an empty melt with no crust gives k = 0."""
    cfg = _config(crust_rate=crust)
    cfg.outgas.D_const_H2O = 0.0
    rates, got_mdot, got_capped = leak_rates(cfg, {'M_mantle': M_MANTLE, 'Phi_global': phi}, dt)
    assert (got_mdot, got_capped) == (mdot, capped)
    phi_c = min(max(phi, 0.0), 1.0)
    expect = 0.02 * mdot / (phi_c * M_MANTLE) if mdot > 0.0 else 0.0
    assert rates['CO2'][1] == pytest.approx(expect, rel=1e-14, abs=0.0)


@pytest.mark.parametrize(
    ('phi_top', 'delta', 'sealed', 'expected'),
    [
        (0.35, 0.31, True, False),  # phi_top == phi_open opens
        (0.29, 0.15, True, True),  # delta == delta_open holds
        (0.29, 0.30, False, True),  # delta == delta_close closes
        (0.30, 0.31, False, False),  # phi_top == phi_close does not close
    ],
    ids=['at-phi-open', 'at-delta-open', 'at-delta-close', 'at-phi-close'],
)
def test_seal_band_edges(phi_top, delta, sealed, expected):
    """Edges of the closure rule: close at delta >= delta_close and phi_top <
    phi_close; open at phi_top >= phi_open or delta < delta_open."""
    interior = _interior([0.6, phi_top])
    row = _sealed_row() if sealed else _row()
    row.update({'lid_weight': 1.0, 'lid_delta': delta, 'esc_rate_total': 0.0})
    state, _ = _step(row, interior, sealed, delta_close=0.3, phi_open=0.35)
    assert state is expected
    assert row['seal_active'] == float(expected)


def test_open_refuses_an_unreadable_sealed_column_without_moving_mass():
    """A NaN sealed mass stops the opening before any column changes."""
    row = _sealed_row()
    row['C_kg_sealed'] = float('nan')
    before = dict(row)
    with pytest.raises(RuntimeError, match='C_kg_sealed'):
        open_seal(row)
    assert row['H2O_kg_sealed'] == before['H2O_kg_sealed'] and row['H2O_kg_liquid'] == 0.0


def _mesh(phi_top):
    """10 mesh cells of 30 km each to the surface; 2 solid cells under the top cell."""
    phi = [0.6] * 7 + [0.2, 0.0, phi_top]
    return _interior(phi, radius=np.linspace(R_INT - 3.0e5, R_INT, 11))


@pytest.mark.physics_invariant
def test_mesh_crust_seal_holds_while_the_top_cell_wanders_around_phi_solid():
    """On the mesh the crust is 0 or whole cells; a sealed crust must not open
    when the top cell crosses phi_solid, only once it holds phi_open of melt."""
    row = {**_row(), 'esc_rate_total': 0.0}
    sealed = None
    for i, phi_top in enumerate((5e-4, 2e-3, 5e-4, 2e-3, 0.2)):
        state, solve = _step(row, _mesh(phi_top), bool(sealed))
        if i == 0:
            assert state is True and row['seal_delta'] == pytest.approx(6.0e4, rel=1e-12)
        sealed = state
        assert state is True, (i, phi_top)
        if i > 0:  # a closed seal counts the cells below phi_open: 0.2, 0.0 and the top
            assert row['seal_delta'] == pytest.approx(9.0e4, rel=1e-12)
        assert solve.call_count == (1 if i == 0 else 0)
    state, solve = _step(row, _mesh(0.4), True)
    assert state is False and solve.call_count == 1
    assert row['H2O_kg_liquid'] == MELT['H2O']


@pytest.mark.physics_invariant
def test_a_held_step_applies_escape_then_the_leak():
    """While sealed, escape scales the atmosphere down (as on a frozen mantle)
    and then the sealed mass leaks into it, by the exact exponential."""
    row = {**_sealed_row(), 'lid_weight': 1.0, 'lid_delta': 0.5}
    m_el = sum(row[f'{e}_kg_atm'] for e in SEALED_ELEMENTS)
    row.update({'esc_rate_total': 1.0, 'esc_step_kg': 0.1 * m_el})
    atm0, sealed0 = row['H2O_kg_atm'], row['H2O_kg_sealed']
    state, solve = _step(row, _interior([0.6, 0.2]), True, delta_close=0.3, tau_leak=50.0)
    keep = np.exp(-10.0 / 50.0)
    assert state is True and solve.call_count == 0
    assert row['H2O_kg_sealed'] == pytest.approx(sealed0 * keep, rel=1e-13)
    expect = atm0 * 0.9 + sealed0 * (1.0 - keep)
    assert row['H2O_kg_atm'] == pytest.approx(expect, rel=1e-12)
    assert row['P_surf'] == pytest.approx(
        row['M_atm'] * GRAVITY / (4 * np.pi * R_INT**2) / 1e5, rel=1e-14
    )
    n = {s: row[f'{s}_kg_atm'] / eval_gas_mmw(s) for s in ATM}
    assert row['atm_kg_per_mol'] == pytest.approx(row['M_atm'] / sum(n.values()), rel=1e-12)
    assert row['H2O_vmr'] == pytest.approx(n['H2O'] / sum(n.values()), rel=1e-12)


@pytest.mark.physics_invariant
def test_a_release_into_the_melt_while_sealed_joins_the_sealed_reservoir():
    """Trapped mass that a remelt returns to the melt while the seal holds is
    under the crust too: the held step moves it to the sealed reservoir."""
    row = {**_sealed_row(), 'lid_weight': 1.0, 'lid_delta': 0.5, 'esc_rate_total': 0.0}
    row['H2O_kg_liquid'], row['H2O_mol_liquid'] = 1.0e18, 1.0e18 / eval_gas_mmw('H2O')
    row['H_kg_liquid'] = element_masses_from_species({'H2O': 1.0e18})['H']
    row.update({'trap_branch': float(BRANCH_REMELT), 'trap_kg_step': -1.0e18})
    state, _ = _step(row, _interior([0.6, 0.2]), True, delta_close=0.3)
    assert state is True
    assert row['H2O_kg_liquid'] == 0.0 and row['H_kg_liquid'] == 0.0
    assert row['H2O_kg_sealed'] == pytest.approx(MELT['H2O'] + 1.0e18, rel=1e-15)


@pytest.mark.physics_invariant
def test_leak_refreshes_the_element_ratios_of_the_atmosphere():
    """The atmospheric element ratios follow the leaked composition."""
    row = _sealed_row()
    row['C/H_atm'] = row['C_kg_atm'] / row['H_kg_atm']
    before = row['C/H_atm']
    leak_step(_config(tau_leak=1.0e4), row, 1.0e4)
    assert row['C/H_atm'] == pytest.approx(row['C_kg_atm'] / row['H_kg_atm'], rel=1e-14)
    assert abs(row['C/H_atm'] / before - 1.0) > 1e-3


def test_restore_seal_state_reads_the_row_and_refuses_stranded_mass():
    """Resume: the seal state comes from seal_active; a row with sealed mass
    under a config with the seal off is refused, since nothing could open it."""
    row = _sealed_row()
    row['seal_active'] = 1.0
    assert restore_seal_state(_config(), row) is True
    assert restore_seal_state(_config(), dict(row, seal_active=0.0)) is False
    off = SimpleNamespace(outgas=SimpleNamespace(lid_seal=LidSeal()))
    with pytest.raises(ValueError, match='lid_seal.enabled'):
        restore_seal_state(off, row)
    assert restore_seal_state(off, _row()) is False


def test_melt_mass_on_a_held_step_without_a_release_fails_the_ledger_check():
    """Only a trapping release may put melt mass under a closed seal; any other
    melt mass on a held step is a ledger fault and stops the run."""
    row = {**_sealed_row(), 'lid_weight': 1.0, 'lid_delta': 0.5, 'esc_rate_total': 0.0}
    row['CO2_kg_liquid'] = 5.0e18
    with pytest.raises(RuntimeError, match='CO2_kg_liquid'):
        _step(row, _interior([0.6, 0.2]), True, delta_close=0.3)
    assert row['CO2_kg_sealed'] == MELT['CO2']


@pytest.mark.physics_invariant
def test_a_remelt_release_under_the_seal_reaches_the_sealed_reservoir():
    """A real trapping step that remelts buried water while the seal holds hands
    it to the held step, which puts it under the crust: no melt mass is left and
    the water mass is conserved across trapped, sealed and atmosphere."""
    import pandas as pd

    from proteus.outgas.trapping import run_trapping

    row = {**_sealed_row(phi=0.3), 'lid_weight': 1.0, 'lid_delta': 0.5, 'esc_rate_total': 0.0}
    row.update(
        {
            'H2O_kg_solid': 2.0e19,
            'H2O_kg_trapped': 2.0e19,
            'H2O_mol_solid': 2.0e19 / eval_gas_mmw('H2O'),
        }
    )
    row.update({'H_kg_solid': element_masses_from_species({'H2O': 2.0e19})['H']})
    row['H_kg_trapped'] = row['H_kg_solid']
    total0 = row['H2O_kg_sealed'] + row['H2O_kg_solid'] + row['H2O_kg_atm']
    hf_all = pd.DataFrame([{**row, 'Phi_global': 0.2, 'Time': row['Time'] - 10.0}])
    cfg = SimpleNamespace(
        outgas=SimpleNamespace(trap_mode='front', D_const_H2O=0.0017, mass_thresh=1e16),
        interior_energetics=SimpleNamespace(rfront_loc=0.5),
    )
    step = run_trapping(cfg, row, hf_all, None)
    assert step is not None and step.total_trapped < 0.0
    released = -step.total_trapped
    state, _ = _step(row, _interior([0.6, 0.2]), True, delta_close=0.3)
    assert state is True and row['H2O_kg_liquid'] == 0.0
    total1 = row['H2O_kg_sealed'] + row['H2O_kg_solid'] + row['H2O_kg_atm']
    assert total1 == pytest.approx(total0, rel=1e-12)
    assert row['H2O_kg_sealed'] == pytest.approx(MELT['H2O'] + released, rel=1e-12)


def _lid_seq(steps, **seal):
    """Run a sequence of (lid_weight, lid_delta, phi_top) steps; return the states."""
    row, sealed, out = {**_row(), 'esc_rate_total': 0.0}, False, []
    for w, d, phi_top in steps:
        row.update({'lid_weight': w, 'lid_delta': d})
        sealed, _ = _step(row, _mesh(phi_top), sealed, **seal)
        out.append((sealed, row['seal_path']))
    return out


def test_a_seal_keeps_reading_the_path_that_closed_it():
    """A mesh-closed seal ignores a thin sub-grid lid that appears later; a
    lid-closed seal opens when its lid thins below delta_open, and reads the
    mesh band once its lid is handed over."""
    mesh_first = _lid_seq([(0.0, 0.0, 5e-4), (1.0, 2.0e3, 2e-3), (1.0, 2.0e3, 2e-3)])
    assert mesh_first == [(True, 1.0), (True, 1.0), (True, 1.0)]
    lid = _lid_seq([(1.0, 0.35, 0.2), (1.0, 0.2, 0.2), (1.0, 0.14, 0.2)], delta_close=0.3)
    assert lid == [(True, 2.0), (True, 2.0), (False, 0.0)]
    hand = _lid_seq([(1.0, 0.35, 0.2), (0.0, 0.0, 5e-4), (0.0, 0.0, 2e-3)], delta_close=0.3)
    assert hand == [(True, 2.0), (True, 2.0), (True, 2.0)]


def test_mesh_band_counts_cells_between_phi_close_and_phi_open():
    """A sealed top cell holding 0.32 melt (above phi_close, below phi_open) is
    still crust for the open test, so the seal holds; at phi_open it opens."""
    row = {**_row(), 'esc_rate_total': 0.0}
    sealed, _ = _step(row, _mesh(5e-4), False)
    held, _ = _step(row, _mesh(0.32), sealed)
    assert held is True and row['seal_delta'] == pytest.approx(9.0e4, rel=1e-12)
    opened, _ = _step(row, _mesh(0.35), held)
    assert opened is False


@pytest.mark.parametrize(
    ('active', 'sealed_kg', 'match'),
    [
        (0.0, 3.0e20, 'seal_active'),
        (float('nan'), 0.0, 'seal_active'),
        (1.0, float('nan'), 'H2O_kg_sealed'),
    ],
    ids=['open-with-sealed-mass', 'nan-flag', 'nan-mass'],
)
def test_restore_seal_state_refuses_a_row_whose_flag_and_masses_disagree(
    active, sealed_kg, match
):
    """Resume with the seal on: sealed mass under an open flag, a non-finite flag
    or a non-finite sealed mass is refused."""
    row = {**_row(), 'seal_active': active, 'H2O_kg_sealed': sealed_kg}
    with pytest.raises(ValueError, match=match):
        restore_seal_state(_config(), row)
    assert restore_seal_state(_config(), {**_row(), 'seal_active': 0.0}) is False
