"""Tests of ``proteus.outgas.lid_seal``: the melt inventory held under a closed crust.

Invariants checked: the sealed reservoir and the melt exchange mass without loss
(close and open conserve every species, element and mole column); the leak of
one step equals the drop of the sealed mass, per species and per element, and
splits between relaxation and volcanism in the ratio of their rates; a sealed
species leaks only down to its dissolved equilibrium and ends each step on the
relaxation toward the floor of the end state; at a constant floor the step
solution does not depend on the step length; the melt concentration of the sealed reservoir
satisfies its mass balance with the global melt fraction; the crust depth is
bounded by the lower face of the top solid run.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from calliope.solubility import SolubilityS2

from proteus.config._outgas import Calliope, LidSeal
from proteus.outgas.common import element_masses_from_species
from proteus.outgas.compaction import BRANCH_REMELT
from proteus.outgas.lid_seal import (
    _LAW,
    LEAK_ELEMENTS,
    S2_CUTOFF,
    SEALED_ELEMENTS,
    SEALED_SPECIES,
    assert_sealed_row,
    close_seal,
    closure_state,
    crust_thickness,
    equilibrium_fractions,
    floor_leak,
    leak_rates,
    leak_step,
    open_seal,
    reservoirs,
    restore_seal_state,
    run_lid_seal_step,
    seal_enabled,
    sealed_mass,
)
from proteus.utils.constants import gas_list, noble_gases, soluble_gases, vol_gas_list
from proteus.utils.helper import eval_gas_mmw

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

M_MANTLE = 4.0e24  # [kg], an Earth-like mantle
R_INT = 6.371e6  # [m]
GRAVITY = 9.81  # [m s-2]
# A row holding H2O and CO2 in the melt and the atmosphere; masses chosen apart.
MELT = {'H2O': 3.0e20, 'CO2': 7.0e19}
ATM = {'H2O': 1.1e20, 'CO2': 4.0e19}
T_SEAL = 1500.0  # [K], magma temperature of the closing solve
D_H2O = 0.0017  # the stand-in partition coefficient of H2O


def _config(solubility=False, **seal):
    """Config stand-in with a real LidSeal and the outgas partition coefficients.

    ``solubility = False`` gives every species a dissolved equilibrium of 0.
    """
    outgas = SimpleNamespace(lid_seal=LidSeal(enabled=True, **seal), T_floor=700.0)
    outgas.calliope = Calliope(solubility=solubility)
    for s in SEALED_SPECIES:
        setattr(outgas, f'D_const_{s}', 0.0)
    outgas.D_const_H2O, outgas.trap_mode = 0.0017, 'none'
    return SimpleNamespace(outgas=outgas, escape=SimpleNamespace(reservoir='outgas'))


def _row(phi=0.4):
    """A helpfile row with species and element reservoirs filled by stoichiometry."""
    row = {'Time': 1.0e4, 'M_mantle': M_MANTLE, 'Phi_global': phi, 'R_int': R_INT}
    row.update({'gravity': GRAVITY, 'T_magma': 1600.0})
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
    row.update(
        {'seal_Tmagma_close': T_SEAL, 'fO2_shift_IW_derived': 2.0}
        | {'seal_phi_close': phi, 'seal_phi_max': phi, 'seal_t_close': row['Time']}
        | {'seal_xtal_kg': 0.0, 'seal_tl_kg': 0.0}
    )
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
    assert crust_thickness(np.array([0.0]), np.array([1.0, 3.0]), 1e-3) == pytest.approx(2.0)


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
    assert acting == (0.31, 0.25, 2)
    assert hidden == (0.0, 0.25, 1)
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
@pytest.mark.parametrize('solubility', [False, True])
def test_hold_keeps_every_reservoir_bit_identical(solubility):
    """With tau_leak = inf and crust_rate = 0 (the defaults) nothing
    leaks, no dissolved equilibrium is computed (a NaN closing temperature
    passes), and the atmosphere is not recomputed: every column stays equal to
    the bit."""
    row = {**_sealed_row(), 'seal_Tmagma_close': float('nan')}
    before = dict(row)
    for _ in range(5):
        leak_step(_config(solubility=solubility), row, 1.0e5)
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


def _relax_to(floor, dt, t_end, tau, m0):
    """Sealed mass by row from the step kernel at a constant floor [kg]."""
    t, m, out = 0.0, m0, {0.0: m0}
    while t < t_end - 1e-9:
        m -= floor_leak(m, -np.expm1(-dt / tau), lambda _: floor)
        t += dt
        out[round(t)] = m
    return out


@pytest.mark.physics_invariant
def test_with_no_solubility_the_leak_decays_to_zero_whatever_the_step():
    """With solubility off every floor is 0 and the sealed mass decays as
    M_c exp(-(t - t_c)/tau), to 1e-12 at every row, for steps of 1, 10 and
    100 kyr (tau = 1 Myr)."""
    tau, m0 = 1.0e6, MELT['H2O']
    runs = {dt: _relax(dt, 1.0e6, tau) for dt in (1.0e3, 1.0e4, 1.0e5)}
    for series, row in runs.values():
        for t, m in series.items():
            assert m == pytest.approx(m0 * np.exp(-t / tau), rel=1e-12)
        assert 'H2O_kg_eq' not in row  # no floor is computed


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_relaxation_matches_the_exponential_and_not_the_step_length(validation_plot):
    """At a constant floor M_eq = 0.4 M_0 the step kernel gives
    M = M_eq + (M_0 - M_eq) exp(-(t - t_c)/tau) to 1e-12 at every row, for steps
    of 1, 10 and 100 kyr (tau = 1 Myr). The step solution is exact, so the step
    length enters only at round-off; a forward Euler step would miss the exact
    distance to the floor by 5.2 % after 1 Myr at 100 kyr steps."""
    tau, m0 = 1.0e6, MELT['H2O']
    floor = 0.4 * m0
    runs = {dt: _relax_to(floor, dt, 1.0e6, tau, m0) for dt in (1.0e3, 1.0e4, 1.0e5)}
    for series in runs.values():
        for t, m in series.items():
            assert m - floor == pytest.approx((m0 - floor) * np.exp(-t / tau), rel=1e-12)
    for t in runs[1.0e5]:
        assert runs[1.0e3][t] == pytest.approx(runs[1.0e5][t], rel=1e-12)
    euler = (m0 - floor) * (1.0 - 1.0e5 / tau) ** 10
    gap = runs[1.0e5][1000000] - floor
    assert abs(gap - euler) / gap > 0.05

    def figure():
        import matplotlib.pyplot as plt

        plt.rcParams['svg.fonttype'] = 'none'  # text as SVG text, not glyph paths
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True, height_ratios=[3, 1])
        x = np.linspace(0.0, 1.0, 101)
        ax.plot(x, np.exp(-x), label='closed form')
        for dt, marker in zip(runs, ('o', 's', '^')):
            t = np.array(sorted(runs[dt]))[:: max(1, int(2.0e5 / dt))]
            m = (np.array([runs[dt][k] for k in t]) - floor) / (m0 - floor)
            ax.plot(t / tau, m, marker, mfc='none', label=f'{dt / 1e3:.0f} kyr steps')
            # Round-off differs between libm builds; plot it at the 1e-12 tolerance.
            ax2.plot(t / tau, np.round(m / np.exp(-t / tau) - 1.0, 12), marker, mfc='none')
        ax.set_ylabel(r'$(M - M_\mathrm{eq}) / (M_0 - M_\mathrm{eq})$')
        ax.legend(loc='upper right')
        ax2.set_xlabel(r'Time since closure / $\tau$')
        ax2.set_ylabel('Error (to 1e-12)')
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
    """The volcanic rate k_i = chi_d Mdot_c / R_i of the Schaefer & Sasselov
    (2015, ApJ 801:40) Eq. 14 and Eq. 17 form, with the global melt fraction:
    R_i = M_melt without trapping, M_melt + D_i M_solid under front trapping with
    the seal closed on a molten mantle (seal_phi_max 1). (a) X_melt R_i == M_i;
    (b) half the mantle solid, half at phi 0.4: X_melt = M_i / (0.2 M_mantle),
    which a local melt fraction (0.4) halves; (c) Eq. 14 written out."""
    cfg = _config(crust_rate=1.0e12, chi_d=0.02)
    row = {'M_mantle': M_MANTLE, 'Phi_global': 0.2, 'seal_phi_max': 1.0}
    rates, mdot, capped = leak_rates(cfg, row, 1.0)
    assert (mdot, capped) == (1.0e12, False)
    assert rates['H2O'] == rates['CO2']  # without trapping both take the melt
    m_i = 1.0e20
    k_co2 = rates['CO2'][1]  # D = 0
    x_melt = k_co2 * m_i / (0.02 * mdot)
    assert x_melt * 0.2 * M_MANTLE == pytest.approx(m_i, rel=1e-14)
    assert x_melt == pytest.approx(m_i / (0.2 * M_MANTLE), rel=1e-14)
    assert abs(x_melt - m_i / (0.4 * M_MANTLE)) / x_melt > 0.4  # local melt fraction
    cfg.outgas.trap_mode = 'front'
    k_h2o = leak_rates(cfg, row, 1.0)[0]['H2O'][1]  # D = 0.0017
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
        cfg_d.outgas.D_const_CO2, cfg_d.outgas.trap_mode = d, 'front'
        k = [leak_rates(cfg_d, {**row, 'Phi_global': p}, 1.0)[0]['CO2'][1] for p in grid]
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
    assert row['seal_crust_rate'] == cfg.outgas.lid_seal.crust_rate
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
    assert rates['H2O'][1] == rates['CO2'][1]  # without trapping the reservoir is the melt
    cfg.outgas.trap_mode = 'front'
    row = {'M_mantle': M_MANTLE, 'Phi_global': 0.01, 'seal_phi_max': 0.2}
    assert leak_rates(cfg, row, 100.0)[0]['H2O'][1] * 100.0 < 0.5  # D > 0: the crystals too
    cfg.outgas.trap_mode = 'none'
    rates, mdot, capped = leak_rates(cfg, {'M_mantle': M_MANTLE, 'Phi_global': 5e-4}, 100.0)
    assert (mdot, capped, rates['CO2'][1]) == (0.0, False, 0.0)
    row = _sealed_row(phi=0.01)
    leak_step(cfg, row, 100.0)
    assert row['seal_volc_capped'] == 1


@pytest.mark.physics_invariant
def test_leak_refreshes_the_atmosphere_from_the_masses():
    """After a leak, P_surf = M_atm g / (4 pi R^2), the mixing ratios sum to 1
    and follow the moles, the mean molar mass is M_atm over the moles, and
    M_vol_atm counts the leaked Ar but not the rock vapour SiO."""
    cfg = _config(tau_leak=1.0e4)
    row = {**_sealed_row(), 'M_vol_atm': 0.0, 'Ar_kg_sealed': 1.0e18, 'SiO_kg_atm': 5.0e18}
    row['Ar_mol_sealed'] = 1.0e18 / eval_gas_mmw('Ar')
    row['SiO_mol_atm'] = 5.0e18 / eval_gas_mmw('SiO')
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
    m_vol = sum(row[f'{s}_kg_atm'] for s in vol_gas_list)
    assert row['M_vol_atm'] == pytest.approx(m_vol, rel=1e-15) and m_vol > sum(ATM.values())
    assert row['M_vol_atm'] == pytest.approx(m_atm - row['SiO_kg_atm'], rel=1e-15)
    assert row['Ar_kg_atm'] > 0.0


def _step(row, interior, sealed, *, init_stage=False, trap_mode='none', hf_all=None, **seal):
    cfg = _config(**seal)
    cfg.outgas.trap_mode = trap_mode
    kw = {} if hf_all is None else {'hf_all': hf_all}
    with (
        patch('proteus.outgas.wrapper.run_outgassing_and_vapourisation') as solve,
        patch('proteus.outgas.wrapper.check_ic_oxygen_budget'),
    ):
        state = run_lid_seal_step(
            {},
            cfg,
            row,
            interior,
            sealed,
            first_iter=False,
            init_stage=init_stage,
            dt=10.0,
            **kw,
        )
    return state, solve


def test_step_closes_after_the_solve_and_not_in_the_init_stage():
    """An open seal solves the chemistry, then closes when the crust is thick
    enough and the top cell below phi_close; never in the init stage."""
    interior = _interior([0.6, 0.2])
    row = {**_row(), 'lid_weight': 1.0, 'lid_delta': 0.31}
    state, solve = _step(row, interior, False, delta_close=0.3)
    assert state is True and row['seal_active'] == 1
    solve.assert_called_once()
    assert row['H2O_kg_liquid'] == 0.0 and row['H2O_kg_sealed'] == MELT['H2O']
    assert row['seal_t_close'] == row['Time']
    row = {**_row(), 'lid_weight': 1.0, 'lid_delta': 0.31}
    assert _step(row, interior, False, init_stage=True, delta_close=0.3)[0] is False
    row = {**_row(), 'lid_weight': 1.0, 'lid_delta': 0.29}
    assert _step(row, interior, False, delta_close=0.3)[0] is False
    assert row['seal_delta'] == row['lid_delta'] and row['seal_phi_top'] == interior.phi[-1]
    assert row['seal_path'] == 0.0  # an open step names no path


def test_an_open_step_checks_the_oxygen_budget_and_a_held_step_does_not():
    """The open step runs the oxygen budget check after its solve, once; a held
    step has no solve and does not run it."""
    cfg, interior = _config(delta_close=0.3), _interior([0.6, 0.2])
    rows = {False: {**_row(), 'lid_weight': 1.0, 'lid_delta': 0.1}}
    rows[True] = {**_sealed_row(), 'lid_weight': 1.0, 'lid_delta': 0.5, 'esc_rate_total': 0.0}
    for sealed, row in rows.items():
        with (
            patch('proteus.outgas.wrapper.run_outgassing_and_vapourisation'),
            patch('proteus.outgas.wrapper.check_ic_oxygen_budget') as check,
        ):
            run_lid_seal_step(
                {}, cfg, row, interior, sealed, first_iter=False, init_stage=False, dt=10.0
            )
        assert row['seal_active'] == float(sealed)
        assert check.call_count == (0 if sealed else 1), sealed
        if not sealed:
            assert check.call_args.args == (cfg, row)


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
    """The seal opens at phi_top >= phi_close + 0.05, at a crust thinner than
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
def test_a_step_that_drains_the_reservoir_leaks_it_whole():
    """A step much longer than tau drains the sealed mass: each element leaks
    its whole held mass, and round-off leaves no sealed mass below 0."""
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
    """10 mesh cells of 30 km each to the surface; under the top cell a solid cell, then 0.2 melt."""
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


@pytest.mark.filterwarnings('error::RuntimeWarning')
@pytest.mark.parametrize('tau', [1.0e-288, 1.0e-290, 1.0e-300])
def test_a_leak_too_fast_for_the_float_range_stops_the_step(tau):
    """A tau_leak whose rate overflows the split of the leak stops the step
    before it writes a column, so no infinite or NaN leak reaches the row."""
    row = _sealed_row()
    before = dict(row)
    with pytest.raises(RuntimeError, match='not finite'):
        leak_step(_config(tau_leak=tau), row, 1.0e3)
    assert row == before


def test_a_nan_leak_stops_the_step_before_it_writes(monkeypatch):
    """A leak that comes out NaN stops the step before it writes a column."""
    monkeypatch.setattr(
        'proteus.outgas.lid_seal._leaks', lambda c, r, b, f: dict.fromkeys(f, np.nan)
    )
    row = _sealed_row()
    before = dict(row)
    with pytest.raises(RuntimeError, match='not finite'):
        leak_step(_config(tau_leak=1.0e4), row, 1.0e3)
    assert row == before


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_an_element_leak_that_is_not_finite_stops_the_step():
    """An element held next to almost no mass of its species gives an element
    share that overflows; the step stops instead of finishing with a NaN leak."""
    row = _sealed_row()
    row['H2O_kg_sealed'] = 1.0e-320
    with pytest.raises(RuntimeError, match='not finite'):
        leak_step(_config(tau_leak=1.0e3), row, 1.0e3)


def test_restore_seal_state_reads_the_row_and_refuses_stranded_mass():
    """Resume: the seal state comes from seal_active; a row with sealed mass
    under a config with the seal off is refused, since nothing could open it."""
    row = _sealed_row()
    row['seal_active'] = 1.0
    assert restore_seal_state(_config(), row) is True
    assert restore_seal_state(_config(), dict(_row(), seal_active=0.0)) is False
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


def _bury_and_remelt(row):
    """Bury 2e19 kg of water in the solid and remelt it with a real trapping step.

    Returns the released mass [kg] and the water in sealed, solid and atmosphere
    before the release [kg].
    """
    import pandas as pd

    from proteus.outgas.trapping import run_trapping

    buried = 2.0e19
    row.update({'H2O_kg_solid': buried, 'H2O_kg_trapped': buried})
    row['H2O_mol_solid'] = buried / eval_gas_mmw('H2O')
    row['H_kg_solid'] = row['H_kg_trapped'] = element_masses_from_species({'H2O': buried})['H']
    total0 = row['H2O_kg_sealed'] + row['H2O_kg_solid'] + row['H2O_kg_atm']
    hf_all = pd.DataFrame([{**row, 'Phi_global': 0.2, 'Time': row['Time'] - 10.0}])
    cfg = SimpleNamespace(
        outgas=SimpleNamespace(trap_mode='front', D_const_H2O=0.0017, mass_thresh=1e16),
        interior_energetics=SimpleNamespace(rfront_loc=0.5),
    )
    step = run_trapping(cfg, row, hf_all, None)
    assert step is not None and step.total_trapped < 0.0
    return -step.total_trapped, total0


@pytest.mark.physics_invariant
def test_a_remelt_release_under_the_seal_reaches_the_sealed_reservoir():
    """A real trapping step that remelts buried water while the seal holds hands
    it to the held step, which puts it under the crust: no melt mass is left and
    the water mass is conserved across trapped, sealed and atmosphere."""
    row = {**_sealed_row(phi=0.3), 'lid_weight': 1.0, 'lid_delta': 0.5, 'esc_rate_total': 0.0}
    released, total0 = _bury_and_remelt(row)
    state, _ = _step(row, _interior([0.6, 0.2]), True, delta_close=0.3)
    assert state is True and row['H2O_kg_liquid'] == 0.0
    total1 = row['H2O_kg_sealed'] + row['H2O_kg_solid'] + row['H2O_kg_atm']
    assert total1 == pytest.approx(total0, rel=1e-12)
    assert row['H2O_kg_sealed'] == pytest.approx(MELT['H2O'] + released, rel=1e-12)


@pytest.mark.physics_invariant
def test_a_remelt_that_releases_element_mass_alone_reaches_the_sealed_reservoir():
    """A remelt with an empty species ledger releases only the element remainder
    (1e3 kg of H), so trap_kg_step is 0; the held step still puts it under the
    crust, and the H mass is conserved across trapped, sealed and atmosphere."""
    import pandas as pd

    from proteus.outgas.trapping import run_trapping

    row = {**_sealed_row(phi=0.2), 'lid_weight': 1.0, 'lid_delta': 0.5, 'esc_rate_total': 0.0}
    row.update({'H_kg_solid': 1.0e3, 'H_kg_trapped': 1.0e3})
    h0 = row['H_kg_sealed'] + row['H_kg_solid'] + row['H_kg_atm']
    hf_all = pd.DataFrame([{**row, 'Phi_global': 0.2, 'Time': row['Time'] - 10.0}])
    row['Phi_global'] = 0.25
    cfg = SimpleNamespace(
        outgas=SimpleNamespace(trap_mode='front', D_const_H2O=0.0017, mass_thresh=1e16),
        interior_energetics=SimpleNamespace(rfront_loc=0.5),
    )
    run_trapping(cfg, row, hf_all, None)
    assert row['trap_kg_step'] == 0.0 and row['trap_branch'] == float(BRANCH_REMELT)
    released = row['H_kg_liquid']
    assert 0.0 < released <= 1.0e3
    state, _ = _step(row, _interior([0.6, 0.2]), True, delta_close=0.3)
    assert state is True and row['H_kg_liquid'] == 0.0
    assert row['H_kg_sealed'] + row['H_kg_solid'] + row['H_kg_atm'] == pytest.approx(
        h0, rel=1e-15
    )
    assert row['H_kg_solid'] == pytest.approx(1.0e3 - released, rel=1e-12)


def _lid_seq(steps, delta=False, **seal):
    """Run a sequence of (lid_weight, lid_delta, phi_top) steps; return the states,
    with seal_delta as a third entry when ``delta`` is set."""
    row, sealed, out = {**_row(), 'esc_rate_total': 0.0}, False, []
    for w, d, phi_top in steps:
        row.update({'lid_weight': w, 'lid_delta': d})
        sealed, _ = _step(row, _mesh(phi_top), sealed, **seal)
        out.append((sealed, row['seal_path'], row['seal_delta'])[: 3 if delta else 2])
    return out


def test_a_seal_keeps_reading_the_path_that_closed_it():
    """A mesh-closed seal ignores a thin sub-grid lid that appears later; a
    lid-closed seal opens when its lid thins below delta_open, and reads the
    mesh band once its lid is handed over."""
    mesh_first = _lid_seq([(0.0, 0.0, 5e-4), (1.0, 2.0e3, 2e-3), (1.0, 2.0e3, 2e-3)])
    assert mesh_first == [(True, 1.0), (True, 1.0), (True, 1.0)]
    lid = _lid_seq([(1.0, 0.35, 0.2), (1.0, 0.2, 0.2), (1.0, 0.14, 0.2)], delta_close=0.3)
    assert lid == [(True, 2.0), (True, 2.0), (False, 0.0)]
    hand = _lid_seq(
        [(1.0, 0.35, 0.2), (0.0, 0.0, 5e-4), (0.0, 0.0, 2e-3), (1.0, 0.1, 5e-4)],
        delta_close=0.3,
    )
    assert hand == [(True, 2.0), (True, 1.0), (True, 1.0), (True, 1.0)]  # on the mesh for good


W_MIN = float(np.nextafter(0.0, 1.0))  # the smallest lid_weight above 0


@pytest.mark.parametrize('w', [0.0, W_MIN, 1e-9, 0.5, 1.0])
def test_any_lid_weight_above_zero_closes_on_the_lid_with_the_whole_inventory(w):
    """The sub-grid lid acts at any lid_weight > 0, however small: it closes the
    seal on lid_delta with the whole melt inventory, not a share scaled by the
    weight. At 0 the mesh, here without a crust, keeps the seal open."""
    row = {**_row(), 'lid_weight': w, 'lid_delta': 0.31, 'esc_rate_total': 0.0}
    before = dict(row)
    sealed, _ = _step(row, _mesh(0.2), False, delta_close=0.3)
    assert sealed is (w > 0.0)
    if not sealed:
        assert row['seal_delta'] == 0.0 and row['H2O_kg_liquid'] == MELT['H2O']
        return
    assert row['seal_path'] == 2 and row['seal_delta'] == row['lid_delta']
    cols = [f'{s}_{u}' for s in SEALED_SPECIES for u in ('kg', 'mol')]
    for c in cols + [f'{e}_kg' for e in SEALED_ELEMENTS]:
        assert row[f'{c}_sealed'] == before[f'{c}_liquid'] and row[f'{c}_liquid'] == 0.0, c
    assert_sealed_row(row)


@pytest.mark.parametrize(('w', 'phi_top'), [(W_MIN, 0.2), (0.5, 0.2), (0.5, 5e-4)])
def test_an_open_seal_under_a_thin_lid_stays_open_whatever_the_mesh(w, phi_top):
    """At lid_weight > 0 an open seal reads lid_delta alone: a lid thinner than
    delta_close keeps it open at any weight, also over a crust on the mesh."""
    row = {**_row(), 'lid_weight': w, 'lid_delta': 0.29, 'esc_rate_total': 0.0}
    sealed, _ = _step(row, _mesh(phi_top), False, delta_close=0.3)
    assert sealed is False and row['seal_path'] == 0.0
    assert row['seal_delta'] == row['lid_delta']


def test_a_lid_closed_seal_stays_closed_while_its_lid_fades_to_zero():
    """A seal closed by the sub-grid lid holds while lid_weight falls step by step
    to 0; at 0 it reads the mesh band below phi_open, which keeps it closed up to
    the band edge and opens it at phi_open."""
    fade = _lid_seq(
        [
            (1.0, 0.35, 0.2),
            (0.5, 0.35, 0.2),
            (W_MIN, 0.35, 0.2),
            (0.0, 0.35, 0.349),
            (0.0, 0.0, 0.35),
        ],
        delta_close=0.3,
        delta=True,
    )
    assert [f[:2] for f in fade] == [(True, 2.0)] * 3 + [(True, 1.0), (False, 0.0)]
    assert [f[2] for f in fade[:3]] == [0.35] * 3
    assert fade[3][2] == pytest.approx(9.0e4, rel=1e-12)  # 0.349, 0.0 and 0.2 cells


def test_a_sealed_row_without_its_path_reads_the_lid_at_any_weight_above_zero():
    """A sealed row with no seal_path (a resume from a row without the column)
    reads the sub-grid lid at any lid_weight > 0, as an open seal does."""
    row = {**_sealed_row(), 'lid_weight': W_MIN, 'lid_delta': 0.35, 'esc_rate_total': 0.0}
    assert 'seal_path' not in row
    sealed, _ = _step(row, _mesh(0.2), True, delta_close=0.3)
    assert sealed is True and row['seal_path'] == 2
    assert row['seal_delta'] == row['lid_delta']


def test_a_sealed_row_without_its_path_reads_the_mesh_band_at_lid_weight_zero():
    """At lid_weight 0 a sealed row with no seal_path reads the mesh band below
    phi_open, as a mesh-closed seal does; a top cell at 0.2 melt keeps it closed."""
    row = {**_sealed_row(), 'lid_weight': 0.0, 'esc_rate_total': 0.0}
    sealed, _ = _step(row, _mesh(0.2), True)
    assert sealed is True and row['seal_path'] == 1
    assert row['seal_delta'] == pytest.approx(9.0e4, rel=1e-12)  # 0.2, 0.0 and 0.2 cells


def _thin_mesh(solid_cells):
    """3 km cells to the surface: ``solid_cells`` solid cells over 0.6 melt."""
    phi = [0.6] * (20 - solid_cells) + [0.0] * solid_cells
    return _interior(phi, radius=np.linspace(R_INT - 6.0e4, R_INT, 21))


@pytest.mark.parametrize(
    ('lid', 'thickness', 'expected'),
    [(False, 3, False), (False, 4, True), (True, 9.9e3, False), (True, 1.0e4, True)],
    ids=['mesh-9km', 'mesh-12km', 'lid-9.9km', 'lid-10km'],
)
def test_delta_close_decides_the_closure_on_both_paths(lid, thickness, expected):
    """The default delta_close of 10 km closes the seal on the crust it reads: on
    the mesh 3 solid cells of 3 km stay open and 4 close; under the sub-grid lid
    9.9 km stays open and 10 km closes."""
    w, d = (1.0, thickness) if lid else (0.0, 0.0)
    row = {**_row(), 'lid_weight': w, 'lid_delta': d, 'esc_rate_total': 0.0}
    sealed, _ = _step(row, _mesh(0.2) if lid else _thin_mesh(thickness), False)
    assert sealed is expected
    assert row['seal_delta'] == pytest.approx(
        thickness if lid else 3.0e3 * thickness, rel=1e-12
    )
    assert row['seal_path'] == (float(1 + lid) if expected else 0.0)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(('tau', 'dt'), [(1.0e4, 1.0e3), (3.0, 1.0e4)], ids=['part', 'drain'])
def test_each_element_leaks_its_own_sealed_mass_off_the_stoichiometry(tau, dt):
    """Element masses apart from the species stoichiometry leak in the same
    fraction as the species: the leak of each element equals the drop of its
    sealed mass and that fraction of its held mass; a drained reservoir ends at 0."""
    row = _sealed_row()
    for e in SEALED_ELEMENTS:  # element masses off the species stoichiometry
        row[f'{e}_kg_sealed'] *= 1.0 - 2.0e-5
    before = dict(row)
    leak_step(_config(tau_leak=tau), row, dt)
    frac = -np.expm1(-dt / tau)
    for e in SEALED_ELEMENTS:
        out = row[f'{e}_kg_leak_tau'] + row[f'{e}_kg_leak_volc']
        assert out == pytest.approx(before[f'{e}_kg_sealed'] * frac, rel=1e-12), e
        assert before[f'{e}_kg_sealed'] - row[f'{e}_kg_sealed'] == pytest.approx(out, rel=1e-12)
        if frac >= 1.0:
            assert row[f'{e}_kg_sealed'] == 0.0


@pytest.mark.parametrize('bad', [float('nan'), float('inf')], ids=['nan', 'inf'])
@pytest.mark.parametrize('key', ['Phi_global', 'M_mantle'])
def test_leak_rates_refuse_a_non_finite_mantle_state(key, bad):
    """A non-finite melt fraction or mantle mass is refused, not read as no crust."""
    row = {'M_mantle': M_MANTLE, 'Phi_global': 0.3, key: bad}
    with pytest.raises(ValueError, match=key):
        leak_rates(_config(crust_rate=3.0e20), row, 1.0e3)
    finite = {**row, key: {'Phi_global': 0.3, 'M_mantle': M_MANTLE}[key]}
    cfg = _config(crust_rate=3.0e20)
    assert leak_rates(cfg, finite, 1.0e3)[1] == cfg.outgas.lid_seal.crust_rate


@pytest.mark.parametrize('bad', [float('nan'), float('inf')], ids=['nan', 'inf'])
def test_closure_state_refuses_a_non_finite_lid_weight(bad):
    """A non-finite lid_weight is refused, not read as no sub-grid lid."""
    row = {'lid_weight': bad, 'lid_delta': 0.35}
    with pytest.raises(ValueError, match='lid_weight'):
        closure_state(_config(), row, _mesh(0.2))
    assert (
        closure_state(_config(), {**row, 'lid_weight': 0.5}, _mesh(0.2))[0] == row['lid_delta']
    )


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
        (1.0, float('inf'), 'H2O_kg_sealed'),
        (1.0, -1.0, 'H2O_kg_sealed'),
    ],
    ids=['open-with-sealed-mass', 'nan-flag', 'nan-mass', 'inf-mass', 'negative-mass'],
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


@pytest.mark.physics_invariant
def test_a_release_a_leak_and_escape_in_one_held_step_keep_the_water_ledger():
    """A remelt release, the leak and escape in one held step: the water lost is
    exactly the escaped share of the atmosphere, and the sealed water is the
    held plus released mass times the exact leak factor."""
    row = {**_sealed_row(phi=0.3), 'lid_weight': 1.0, 'lid_delta': 0.5}
    m_el = sum(row[f'{e}_kg_atm'] for e in SEALED_ELEMENTS)
    row.update({'esc_rate_total': 1.0, 'esc_step_kg': 0.1 * m_el})
    atm0 = row['H2O_kg_atm']
    released, total0 = _bury_and_remelt(row)
    state, _ = _step(row, _interior([0.6, 0.2]), True, delta_close=0.3, tau_leak=1.0e3)
    keep = np.exp(-10.0 / 1.0e3)
    assert state is True and row['H2O_kg_liquid'] == 0.0
    assert row['H2O_kg_sealed'] == pytest.approx((MELT['H2O'] + released) * keep, rel=1e-12)
    total1 = row['H2O_kg_sealed'] + row['H2O_kg_solid'] + row['H2O_kg_atm']
    assert total1 == pytest.approx(total0 - 0.1 * atm0, rel=1e-12)


@pytest.mark.parametrize('path', [float('nan'), 3.0, -1.0], ids=['nan', 'unknown', 'negative'])
def test_restore_seal_state_refuses_an_unknown_seal_path(path):
    """A sealed row must name a known path (0, 1 or 2)."""
    row = {**_sealed_row(), 'seal_active': 1.0, 'seal_path': path}
    with pytest.raises(ValueError, match='seal_path'):
        restore_seal_state(_config(), row)
    assert restore_seal_state(_config(), dict(row, seal_path=2.0)) is True


def test_restore_with_the_seal_off_clears_stale_seal_flags():
    """With the seal off and nothing held, a resume zeroes a stale seal flag and
    path, so later rows do not report a seal."""
    off = SimpleNamespace(outgas=SimpleNamespace(lid_seal=LidSeal()))
    row = {**_row(), 'seal_active': 1.0, 'seal_path': 2.0}
    assert restore_seal_state(off, row) is False
    assert row['seal_active'] == row['seal_path'] == 0.0


def _eq_row(atm, sealed, phi=0.3):
    """A sealed row with the given atmospheric and sealed species masses [kg]."""
    row = {'Time': 1.0e4, 'M_mantle': M_MANTLE, 'Phi_global': phi, 'R_int': R_INT}
    row.update({'gravity': GRAVITY, 'seal_active': 1.0, 'esc_rate_total': 0.0})
    row.update({'seal_Tmagma_close': T_SEAL, 'fO2_shift_IW_derived': 2.0})
    for s in gas_list:
        for r in ('atm', 'liquid', 'solid', 'sealed'):
            row[f'{s}_kg_{r}'] = row[f'{s}_mol_{r}'] = 0.0
    for r, masses in (('atm', atm), ('sealed', sealed)):
        for s, kg in masses.items():
            row[f'{s}_kg_{r}'], row[f'{s}_mol_{r}'] = kg, kg / eval_gas_mmw(s)
    for e in SEALED_ELEMENTS:
        row[f'{e}_kg_atm'] = element_masses_from_species(atm).get(e, 0.0)
        row[f'{e}_kg_sealed'] = element_masses_from_species(sealed).get(e, 0.0)
        row[f'{e}_kg_liquid'] = 0.0
        row[f'{e}_kg_total'] = row[f'{e}_kg_atm'] + row[f'{e}_kg_sealed']
    row['M_atm'] = sum(atm.values())
    return row


def _floor(row, s):
    """Dissolved equilibrium of H2O or CO2 on the melt reservoir M_melt [kg].

    Sossi et al. (2023) 524 p^0.5 ppmw for H2O and Dixon et al. (1995) for CO2,
    at the partial pressure [bar] of the row's atmospheric masses and moles.
    """
    n = sum(row[f'{g}_mol_atm'] for g in gas_list)
    m = sum(row[f'{g}_kg_atm'] for g in gas_list)
    p = row[f'{s}_mol_atm'] / n * m * GRAVITY / (4.0 * np.pi * R_INT**2) / 1e5
    if s == 'H2O':
        x = 524e-6 * np.sqrt(p)
    else:
        w = 3.8e-7 * p * np.exp(-23.0 * (p - 1.0) / (83.15 * T_SEAL))
        x = 1e-2 * 4400.0 * w / (36.6 - 44.0 * w)
    phi = row['Phi_global']
    return x * M_MANTLE * phi


@pytest.mark.physics_invariant
def test_sealed_mass_below_its_floor_does_not_leak():
    """One-way floor: sealed H2O under its dissolved equilibrium stays sealed at
    any tau and crust rate, and the atmosphere is left as it was; with CO2
    leaking in the same step, the H2O mass and moles stay equal to the bit."""
    m_eq = _floor(_eq_row({'H2O': 1.0e17}, {}), 'H2O')
    row = _eq_row({'H2O': 1.0e17}, {'H2O': 0.9 * m_eq})
    before = dict(row)
    leak_step(_config(solubility=True, tau_leak=1.0e2, crust_rate=1.0e20), row, 1.0e3)
    assert row['H2O_kg_sealed'] == before['H2O_kg_sealed']
    assert row['H2O_kg_atm'] == before['H2O_kg_atm']
    assert row['H_kg_leak_tau'] == row['H_kg_leak_volc'] == 0.0
    assert row.get('H2O_kg_eq') == pytest.approx(m_eq, rel=1e-12)
    row = _eq_row({'H2O': 1.0e17, 'CO2': 1.0e15}, {'H2O': 0.9 * m_eq, 'CO2': 1.0e19})
    before = dict(row)
    leak_step(_config(solubility=True, tau_leak=1.0e2), row, 1.0e3)
    assert row['CO2_kg_sealed'] < before['CO2_kg_sealed']
    for key in ('H2O_kg_sealed', 'H2O_mol_sealed', 'H2O_kg_atm', 'H2O_mol_atm'):
        assert row[key] == before[key], key
    assert row['H2O_kg_eq'] > m_eq  # the heavier CO2 raises the H2O partial pressure


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('tau', 'crust'), [(1.0e3, 0.0), (1.0e5, 0.0), (1.0e4, 1.0e21), (float('inf'), 1.0e21)]
)
def test_leak_relaxes_toward_the_floor_of_the_end_state(tau, crust):
    """Above its floor the sealed mass relaxes toward the floor of the end state,
    M_end - M_eq,end = exp(-(1/tau + k) dt) (M_0 - M_eq,end), with M_eq from the
    atmosphere after the leak; the atmosphere gains what the reservoir loses.
    The volcanic term alone (tau = inf) stops at the same floor."""
    atm = {'H2O': 1.0e17, 'CO2': 5.0e16}
    m0 = {'H2O': 2.0 * _floor(_eq_row(atm, {}), 'H2O'), 'CO2': 1.0e19}
    row = _eq_row(atm, m0)
    cfg = _config(solubility=True, tau_leak=tau, crust_rate=crust)
    rates, _, _ = leak_rates(cfg, row, 1.0e3)
    leak_step(cfg, row, 1.0e3)
    for s in ('H2O', 'CO2'):
        keep = np.exp(-1.0e3 * sum(rates[s]))
        m_eq = _floor(row, s)
        assert row[f'{s}_kg_sealed'] - m_eq == pytest.approx(keep * (m0[s] - m_eq), rel=1e-9)
        assert row[f'{s}_kg_sealed'] >= m_eq
        gained = row[f'{s}_kg_atm'] - atm[s]
        assert gained == pytest.approx(m0[s] - row[f'{s}_kg_sealed'], rel=1e-12)
    assert (row['C_kg_leak_volc'] > 0.0) == (crust > 0.0)
    assert (row['C_kg_leak_tau'] > 0.0) == bool(np.isfinite(tau))


@pytest.mark.physics_invariant
def test_a_short_tau_gives_the_equilibrium_partition():
    """With tau = 1e-3 dt the step ends at the gas-melt equilibrium of the frozen
    speciation: H2O and CO2 each split their total so that the sealed mass equals
    its floor, solved here by nested root finds on the 2 atmospheric masses."""
    from scipy.optimize import brentq

    atm = {'H2O': 1.0e17, 'CO2': 1.0e17}
    m0 = {'H2O': 2.0 * _floor(_eq_row(atm, {}), 'H2O'), 'CO2': 1.0e19}
    row = _eq_row(atm, m0)
    leak_step(_config(solubility=True, tau_leak=1.0), row, 1.0e3)
    tot = {s: atm[s] + m0[s] for s in atm}

    def co2_of(a_h2o):
        def gap(a):
            return tot['CO2'] - a - _floor(_eq_row({'H2O': a_h2o, 'CO2': a}, {}), 'CO2')

        return brentq(gap, atm['CO2'], tot['CO2'], xtol=1e-6, rtol=1e-15)

    def h2o_gap(a):
        state = _eq_row({'H2O': a, 'CO2': co2_of(a)}, {})
        return tot['H2O'] - a - _floor(state, 'H2O')

    a_h2o = brentq(h2o_gap, atm['H2O'], tot['H2O'], xtol=1e-6, rtol=1e-15)
    expected = {'H2O': a_h2o, 'CO2': co2_of(a_h2o)}
    for s in atm:
        assert row[f'{s}_kg_atm'] == pytest.approx(expected[s], rel=1e-9)
        assert row[f'{s}_kg_sealed'] == pytest.approx(tot[s] - expected[s], rel=1e-9)


@pytest.mark.parametrize(('t_magma', 'expected'), [(1600.0, 1600.0), (650.0, 700.0)])
def test_closure_stores_the_magma_temperature_of_the_closing_solve(t_magma, expected):
    """The seal stores T_magma floored at outgas.T_floor, the temperature of the
    closing chemistry solve, for the temperature-dependent laws while sealed."""
    row = {**_row(), 'lid_weight': 1.0, 'lid_delta': 0.31, 'T_magma': t_magma}
    assert _step(row, _interior([0.6, 0.2]), False, delta_close=0.3)[0] is True
    assert row.get('seal_Tmagma_close') == expected
    assert row.get('seal_phi_close') == row.get('seal_phi_max') == row['Phi_global']


@pytest.mark.physics_invariant
def test_finite_tau_lies_between_the_frozen_and_the_open_atmosphere():
    """Escape off, H2O the only dissolving species, melt fraction falling
    from 0.3 to 0.15. At every step the atmospheric H2O of tau = 3 kyr is below
    the open equilibrium on the melt alone (the closure state, D = 0) and at or
    above the frozen seal; once it leaks it stays above the frozen value."""
    from scipy.optimize import brentq

    a0 = 1.0e17
    m0 = _floor(_eq_row({'H2O': a0}, {}), 'H2O')
    row = _eq_row({'H2O': a0}, {'H2O': m0})
    cfg = _config(solubility=True, tau_leak=3.0e3)
    tot, leaked = a0 + m0, False
    for phi in np.linspace(0.3, 0.15, 16)[1:]:
        row['Phi_global'] = phi
        leak_step(cfg, row, 1.0e3)
        a = row['H2O_kg_atm']

        def gap(x, phi=phi):
            return tot - x - _floor(_eq_row({'H2O': x}, {}, phi), 'H2O')

        assert a < brentq(gap, a0, tot, xtol=1e-6, rtol=1e-15)
        leaked = leaked or a > a0
        assert a > a0 if leaked else a == a0
    assert leaked


def test_a_leak_off_its_relaxation_fails_the_floor_check():
    """A step kernel that ignores the floor ends below it, and one that leaks
    too little ends off the relaxation; the check in the step stops both."""
    atm = {'H2O': 1.0e17}
    m0 = {'H2O': 2.0 * _floor(_eq_row(atm, {}), 'H2O')}
    cfg = _config(solubility=True, tau_leak=1.0e3)
    with (
        patch('proteus.outgas.lid_seal.floor_leak', lambda m, f, _: f * m),
        pytest.raises(RuntimeError, match='below its floor'),
    ):
        leak_step(cfg, _eq_row(atm, m0), 1.0e3)
    with (
        patch('proteus.outgas.lid_seal.floor_leak', lambda m, f, fl: 1e-6 * f * (m - fl(0.0))),
        pytest.raises(RuntimeError, match='above its floor, not'),
    ):
        leak_step(cfg, _eq_row(atm, m0), 1.0e3)
    leak_step(cfg, _eq_row(atm, m0), 1.0e3)


def test_the_floor_checks_stop_an_end_state_off_by_more_than_round_off():
    """A leak 1e-8 of the held mass past the solve, at lam dt = 100, ends below the
    floor it raises; one 1e-7 past it, at lam dt = 1, ends off the relaxation. Both
    are past the round-off of the checks (1e-12 and 1e-9 of the held mass) and stop."""
    from proteus.outgas.lid_seal import floor_leak

    atm = {'H2O': 1.0e17}
    m0 = {'H2O': 2.0 * _floor(_eq_row(atm, {}), 'H2O')}
    cfg = _config(solubility=True, tau_leak=1.0e3)
    for dt, extra, match in [
        (1.0e5, 1e-8, 'below its floor'),
        (1.0e3, 1e-7, 'above its floor, not'),
    ]:
        with (
            patch(
                'proteus.outgas.lid_seal.floor_leak',
                lambda m, f, fl, x=extra: floor_leak(m, f, fl) + x * m,
            ),
            pytest.raises(RuntimeError, match=match),
        ):
            leak_step(cfg, _eq_row(atm, m0), dt)


def test_a_step_zeroes_the_floors_of_the_last_sealed_step():
    """An open step writes no floor, so the floors of an earlier sealed step read 0."""
    row = {**_row(), 'lid_weight': 1.0, 'lid_delta': 0.29}
    row.update({f'{s}_kg_eq': 1.0 for s in soluble_gases})
    assert _step(row, _interior([0.6, 0.2]), False, delta_close=0.3)[0] is False
    assert all(row[f'{s}_kg_eq'] == 0.0 for s in soluble_gases)


def test_leak_sweeps_that_do_not_settle_stop_the_step():
    """H2O and CO2 couple through the mean molar mass; one sweep cannot settle both."""
    atm = {'H2O': 1.0e17, 'CO2': 5.0e16}
    row = _eq_row(atm, {'H2O': 2.0 * _floor(_eq_row(atm, {}), 'H2O'), 'CO2': 1.0e19})
    with (
        patch('proteus.outgas.lid_seal.SWEEP_MAX', 1),
        pytest.raises(RuntimeError, match='did not settle'),
    ):
        leak_step(_config(solubility=True, tau_leak=1.0e3), dict(row), 1.0e3)
    leak_step(_config(solubility=True, tau_leak=1.0e3), row, 1.0e3)
    assert row['CO2_kg_sealed'] < 1.0e19 and row['CO2_kg_eq'] > 0.0


@pytest.mark.parametrize(
    ('key', 'bad'),
    [('seal_Tmagma_close', v) for v in (float('nan'), float('inf'), -1.0, 0.0)]
    + [('fO2_shift_IW_derived', v) for v in (float('nan'), float('-inf'))],
    ids=str,
)
def test_a_leak_refuses_an_unreadable_closing_state(key, bad):
    """The floor needs a finite closing temperature above 0 K and a finite IW offset."""
    atm = {'H2O': 1.0e17}
    row = _eq_row(atm, {'H2O': 2.0 * _floor(_eq_row(atm, {}), 'H2O')})
    row[key] = bad
    before = dict(row)
    with pytest.raises(ValueError, match='cannot set the dissolved equilibrium'):
        leak_step(_config(solubility=True, tau_leak=1.0e3), row, 1.0e3)
    assert row['H2O_kg_sealed'] == before['H2O_kg_sealed']


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.0], ids=str)
def test_floor_leak_refuses_a_floor_that_is_not_finite_or_negative(bad):
    """A floor that is NaN, inf or negative is refused; a floor of 0 leaks frac M."""
    with pytest.raises(ValueError, match='dissolved equilibrium'):
        floor_leak(1.0e20, 0.5, lambda _: bad)
    assert floor_leak(1.0e20, 0.5, lambda _: 0.0) == pytest.approx(5.0e19, rel=1e-15)


def test_equilibrium_fractions_are_zero_with_solubility_off():
    """With outgas.calliope.solubility off no species dissolves, at any pressure."""
    bar = dict.fromkeys(gas_list, 50.0)
    on = equilibrium_fractions(_config(solubility=True), bar, 500.0, T_SEAL, 2.0)
    off = equilibrium_fractions(_config(), bar, 500.0, T_SEAL, 2.0)
    assert on['H2O'] > 0.0
    assert set(off.values()) == {0.0}


@pytest.mark.physics_invariant
@pytest.mark.parametrize('p_co2', [5.0e3, 6.0e3])
def test_a_co2_leak_above_the_peak_of_the_dixon_law_finishes_its_step(p_co2):
    """The Dixon et al. (1995) CO2 solubility peaks at 83.15 T/23 bar (5423 bar at
    1500 K) and falls above it, but on an Earth-like planet it falls too slowly to
    give the step a second solution: the step finishes on its relaxation on both
    sides of the peak."""
    a_co2 = p_co2 * 1e5 * 4.0 * np.pi * R_INT**2 / GRAVITY
    m0 = 1.1 * _floor(_eq_row({'CO2': a_co2}, {}), 'CO2')
    row = _eq_row({'CO2': a_co2}, {'CO2': m0})
    leak_step(_config(solubility=True, tau_leak=1.0e3), row, 1.0e3)
    m_eq, keep = _floor(row, 'CO2'), np.exp(-1.0)
    assert row['CO2_kg_sealed'] - m_eq == pytest.approx(keep * (m0 - m_eq), rel=1e-9)
    assert row['CO2_bar'] > p_co2


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('residual', 'stops'),
    [
        (lambda u: -0.01 * np.cos(3.0 * np.pi * u), True),
        (lambda u: (u - 0.5) * ((u - 0.5) ** 2 - (1.5 / 16.0) ** 2), True),
        (lambda u: u - 0.5 if u <= 1.0 else 0.5 * np.cos(4.0 * np.pi * (u - 1.0)), False),
    ],
    ids=['3-roots', 'narrow-fold', 'roots-past-the-bracket'],
)
def test_a_step_with_more_than_one_solution_stops(residual, stops):
    """A floor that falls with the gas the leak adds, faster than the leak itself,
    gives the step residual 3 roots, evenly spaced or 1.5/16 of the bracket apart;
    the step must stop rather than take one. Roots beyond the bracket [0, frac M]
    are no solutions of the step, which then finishes on its one root."""
    m0, a0, frac = 1.0e19, 1.0e18, -np.expm1(-1.0)
    row = _eq_row({'CO2': a0}, {'CO2': m0})
    per, span = 1.0 / eval_gas_mmw('CO2'), frac * m0

    def model(config, hf_row, per_kg):  # the floor reads the leak through the moles
        def floor_of(leak, s):
            x = (hf_row[f'{s}_mol_atm'] + leak.get(s, 0.0) * per_kg.get(s, 0.0)) / per - a0
            return m0 - (x - span * residual(x / span)) / frac

        return floor_of

    with patch('proteus.outgas.lid_seal._floor_model', model):
        if stops:
            with pytest.raises(RuntimeError, match='more than one solution'):
                leak_step(_config(solubility=True, tau_leak=1.0e3), row, 1.0e3)
            assert row['CO2_kg_sealed'] < m0
        else:
            leak_step(_config(solubility=True, tau_leak=1.0e3), row, 1.0e3)
            assert row['CO2_kg_sealed'] == pytest.approx(m0 - 0.5 * frac * m0, rel=1e-12)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('phi', [1e-5, 1e-6])
def test_a_step_to_a_floor_far_below_the_held_mass_passes_the_floor_check(phi):
    """With tau much shorter than the step the held CO2 drops to a floor below
    1e-5 of it; the end state then carries the round-off of the held mass, and
    the floor check must allow that round-off, not 1e-12 of the floor."""
    for m0 in np.geomspace(1.0e19, 1.0e21, 10):
        row = _eq_row({'CO2': 1.0e15}, {'CO2': m0}, phi=phi)
        leak_step(_config(solubility=True, tau_leak=1.0), row, 1.0e3)
        assert row['CO2_kg_eq'] < 1.0e-5 * m0
        assert abs(row['CO2_kg_sealed'] - row['CO2_kg_eq']) <= 1.0e-12 * m0


@pytest.mark.physics_invariant
@pytest.mark.parametrize('s2_atm', [0.0, 1.0e-3])
def test_held_s2_leaks_to_a_continuous_floor_at_a_vanishing_s2_pressure(s2_atm):
    """CALLIOPE returns no S2 solubility below 1e-20 bar, which makes the floor
    jump at that pressure. With no or almost no S2 in the atmosphere, held S2
    must still leak to its floor on the continued square-root law and end on it,
    to the round-off of the floor check (1e-12 of the held mass)."""
    atm = {'H2O': 1.0e18, 'CO2': 1.0e18, 'S2': s2_atm}
    row = _eq_row(atm, {'S2': 1.0e11})
    leak_step(_config(solubility=True, tau_leak=1.0e2), row, 1.0e3)
    assert 0.0 < row['S2_kg_eq'] <= row['S2_kg_sealed'] + 1.0e-12 * 1.0e11
    assert row['S2_kg_sealed'] < 1.0e11
    assert row['S2_kg_atm'] > s2_atm


@pytest.mark.physics_invariant
def test_the_s2_fraction_continues_the_square_root_law_below_the_cutoff():
    """At and above S2_CUTOFF the S2 fraction is the CALLIOPE law; below it the
    same square-root law in the partial pressure, so 100 times less pressure
    gives 10 times less S2, where CALLIOPE gives 0."""
    cfg = _config(solubility=True)
    x = {
        p: equilibrium_fractions(
            cfg, {**dict.fromkeys(gas_list, 1.0), 'S2': p}, 2.0, T_SEAL, 2.0
        )
        for p in (S2_CUTOFF, 1e-2 * S2_CUTOFF)
    }
    assert x[S2_CUTOFF]['S2'] == pytest.approx(
        1e-6 * SolubilityS2()(S2_CUTOFF, T_SEAL, 2.0), rel=1e-15, abs=0.0
    )
    assert x[1e-2 * S2_CUTOFF]['S2'] == pytest.approx(
        0.1 * x[S2_CUTOFF]['S2'], rel=1e-12, abs=0.0
    )
    assert SolubilityS2()(1e-2 * S2_CUTOFF, T_SEAL, 2.0) == 0.0


@pytest.mark.physics_invariant
def test_the_floor_takes_the_closing_iw_offset_and_the_surface_pressure():
    """Held N2, S2, CO and CH4 in one leaking step; each floor written to the row is
    the solubility of the end atmosphere at the IW offset of the closing solve
    (fO2_shift_IW_derived, here IW+2) and the surface pressure. At IW+0 the N2
    and S2 floors differ by more than 1 %, so a wrong offset is visible."""
    atm = {
        'H2O': 1.0e19,
        'CO2': 1.0e18,
        'N2': 1.0e17,
        'S2': 1.0e16,
        'CO': 1.0e17,
        'CH4': 1.0e15,
    }
    held = {'N2': 1.0e19, 'S2': 1.0e19, 'CO': 1.0e19, 'CH4': 1.0e18}
    row = _eq_row(atm, held)
    cfg = _config(solubility=True, tau_leak=1.0e3)
    leak_step(cfg, row, 1.0e3)
    bar = {s: row[f'{s}_bar'] for s in gas_list}
    melt = M_MANTLE * row['Phi_global']
    x2 = equilibrium_fractions(cfg, bar, row['P_surf'], T_SEAL, 2.0)
    x0 = equilibrium_fractions(cfg, bar, row['P_surf'], T_SEAL, 0.0)
    for s in held:
        assert row[f'{s}_kg_eq'] == pytest.approx(x2[s] * melt, rel=1e-10), s
    assert row['N2_kg_sealed'] < held['N2'] and row['CO_kg_sealed'] < held['CO']
    for s in ('N2', 'S2'):
        assert abs(x0[s] / x2[s] - 1.0) > 0.01, s


def test_the_law_table_covers_every_soluble_gas_and_reads_the_include_flags():
    """Every species of soluble_gases has a law, and a species switched off in
    outgas.calliope (here CO and Ar) gets no dissolved fraction at any pressure."""
    assert set(_LAW) == set(soluble_gases)
    cfg = _config(solubility=True)
    cfg.outgas.calliope = Calliope(include_CO=False, include_Ar=True)
    bar = dict.fromkeys(gas_list, 10.0)
    x = equilibrium_fractions(cfg, bar, 500.0, T_SEAL, 2.0)
    assert x['CO'] == 0.0 and x['Ar'] > 0.0 and x['CH4'] > 0.0
    cfg.outgas.calliope = Calliope(include_Ar=False)
    assert equilibrium_fractions(cfg, bar, 500.0, T_SEAL, 2.0)['Ar'] == 0.0


def test_co_ch4_and_noble_gas_fractions_follow_the_calliope_laws_and_arguments():
    """CO and CH4 dissolve at their partial pressure and the total pressure, each
    noble gas by its own Henry's law at its partial pressure, as in the CALLIOPE
    solve; every species has its own pressure so a swapped law or argument fails."""
    from calliope.solubility import SolubilityCH4, SolubilityCO, SolubilityNobleGas

    cfg = _config(solubility=True)
    cfg.outgas.calliope = Calliope(
        **{f'include_{g}': True for g in ('CO', 'CH4', *noble_gases)}
    )
    bar = {g: 3.0 + i for i, g in enumerate(gas_list)}
    p_tot, temp = 700.0, 1650.0
    x = equilibrium_fractions(cfg, bar, p_tot, temp, 2.0)
    want = {'CO': SolubilityCO()(bar['CO'], p_tot), 'CH4': SolubilityCH4()(bar['CH4'], p_tot)}
    want.update({g: SolubilityNobleGas(g)(bar[g]) for g in noble_gases})
    for g, ppmw in want.items():
        assert ppmw > 0.0 and x[g] == pytest.approx(1e-6 * ppmw, rel=1e-14), g


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('trap_mode', 'phi', 'share'),
    [('front', 0.1, 0.2 - 0.1), ('front', 0.3, 0.0), ('none', 0.1, 0.0)],
    ids=['front', 'front-remelted', 'none'],
)
def test_with_front_trapping_the_floor_leaves_out_the_crystals_formed_before_closure(
    trap_mode, phi, share
):
    """With trap_mode = 'front' the crystals that froze before closure keep their
    own trapped volatiles, so the floor reservoir of each species counts only the
    crystals formed after closure: X_eq M_mantle (Phi + D_i max(Phi_c - Phi, 0)),
    the melt alone once Phi rises above Phi_c (Phi_c 0.2, here also the highest melt
    fraction since closure); without trapping it is the melt alone,
    X_eq M_mantle Phi. H2O (D 0.0017) and CO2 (D 0.05) leak in
    the same step with crust production on; the volcanic rate takes the same
    reservoir, so the leak splits (1/tau) : chi_d Mdot_c / that."""
    row = _eq_row({'H2O': 1.0e17, 'CO2': 1.0e15}, {'H2O': 3.0e20, 'CO2': 1.0e19}, phi=phi)
    row['seal_phi_close'] = row['seal_phi_max'] = 0.2
    cfg = _config(solubility=True, tau_leak=1.0e3, crust_rate=1.0e20)
    cfg.outgas.trap_mode, cfg.outgas.D_const_CO2 = trap_mode, 0.05
    rates, _, capped = leak_rates(cfg, row, 1.0e3)
    leak_step(cfg, row, 1.0e3)
    for s, d in (('H2O', D_H2O), ('CO2', 0.05)):
        x_eq = _floor(row, s) / (M_MANTLE * phi)
        assert row[f'{s}_kg_eq'] == pytest.approx(
            x_eq * M_MANTLE * (phi + d * share), rel=1e-10
        )
    assert row['H2O_kg_sealed'] < 3.0e20 and not capped
    inv_tau, k = rates['H2O']
    assert k == pytest.approx(0.02 * 1.0e20 / (M_MANTLE * (phi + D_H2O * share)), rel=1e-14)
    assert row['H_kg_leak_volc'] / row['H_kg_leak_tau'] == pytest.approx(k / inv_tau, rel=1e-12)


@pytest.mark.parametrize('phi_max', [float('nan'), -0.1, 1.5, None], ids=str)
def test_front_trapping_refuses_an_unreadable_highest_melt_fraction(phi_max):
    """Under front trapping the floor needs the highest melt fraction since
    closure in [0, 1]. Without trapping the floor does not read it."""
    row = _eq_row({'H2O': 1.0e17}, {'H2O': 3.0e20}, phi=0.1)
    row.pop('seal_phi_max', None)
    if phi_max is not None:
        row['seal_phi_max'] = phi_max
    cfg = _config(solubility=True, tau_leak=1.0e3)
    cfg.outgas.trap_mode = 'front'
    with pytest.raises(ValueError, match='seal_phi_max'):
        leak_step(cfg, dict(row), 1.0e3)
    cfg.outgas.trap_mode = 'none'
    leak_step(cfg, row, 1.0e3)
    assert row['H2O_kg_sealed'] < 3.0e20


@pytest.mark.physics_invariant
def test_under_front_trapping_the_volcanic_rate_takes_the_floor_reservoir():
    """The held mass sits in the floor reservoir R'_i = M_mantle (Phi + D_i
    (Phi_c - Phi)), so its melt concentration is M_i / R'_i and the volcanic rate
    is chi_d Mdot_c / R'_i: at Phi 0.001, Phi_c 0.19 and D 0.0017 that is 0.757
    times the rate on the melt alone, chi_d Mdot_c / M_melt, without trapping."""
    row = {'M_mantle': M_MANTLE, 'Phi_global': 1.0e-3, 'seal_phi_max': 0.19}
    cfg = _config(crust_rate=1.0e15)
    k_none = leak_rates(cfg, row, 1.0)[0]['H2O'][1]
    cfg.outgas.trap_mode = 'front'
    k_front = leak_rates(cfg, row, 1.0)[0]['H2O'][1]
    r_front = M_MANTLE * (1.0e-3 + D_H2O * (0.19 - 1.0e-3))
    assert k_front == pytest.approx(0.02 * 1.0e15 / r_front, rel=1e-14)
    assert k_none == pytest.approx(0.02 * 1.0e15 / (M_MANTLE * 1.0e-3), rel=1e-14)
    assert k_front / k_none == pytest.approx(0.7568, rel=1e-4)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    'phi', [0.0, -1.0e-12, 1.0 + 1.0e-12], ids=['solid', 'below-0', 'above-1']
)
def test_a_front_trapping_closure_at_either_end_of_the_melt_fraction_holds_and_resumes(phi):
    """A closure stores the global melt fraction clipped to [0, 1], so a seal that
    closes on a fully solid mantle, or with round-off past either end, holds, leaks
    and resumes under front trapping; on a solid mantle the floor reservoir is 0."""
    row = {**_row(phi=phi), 'lid_weight': 1.0, 'lid_delta': 0.31, 'esc_rate_total': 0.0}
    assert _step(row, _interior([0.6, 0.2]), False, delta_close=0.3, trap_mode='front')[0]
    clipped = min(max(phi, 0.0), 1.0)
    assert row['seal_phi_close'] == row['seal_phi_max'] == clipped
    row['fO2_shift_IW_derived'] = 2.0
    sealed0 = row['H2O_kg_sealed']
    kw = {'delta_close': 0.3, 'trap_mode': 'front', 'tau_leak': 1.0e3}
    assert _step(row, _interior([0.6, 0.2]), True, **kw)[0]
    assert row['H2O_kg_sealed'] == pytest.approx(sealed0 * np.exp(-10.0 / 1.0e3), rel=1e-13)
    front = _config()
    front.outgas.trap_mode = 'front'
    assert reservoirs(front, row)[0]['H2O'] == pytest.approx(clipped * M_MANTLE, rel=1e-15)
    assert restore_seal_state(front, row) is True


@pytest.mark.physics_invariant
def test_front_trapping_counts_the_crystals_formed_after_the_highest_melt_fraction():
    """A sealed mantle that remelts past Phi_c (0.2) to 0.3 and freezes to 0.15:
    the burial ledger remelts the youngest solid first, so the crystals formed after
    the highest melt fraction since closure hold sealed volatiles and the floor
    reservoir is M_mantle (Phi + D max(Phi_max - Phi, 0)), Phi_max = 0.3."""
    row = {**_sealed_row(phi=0.2), 'lid_weight': 1.0, 'lid_delta': 0.5, 'esc_rate_total': 0.0}
    row.update({'seal_active': 1.0, 'seal_path': 2.0, 'seal_phi_max': 0.2})
    kw = {'delta_close': 0.3, 'trap_mode': 'front', 'solubility': True, 'tau_leak': 1.0e3}
    for phi in (0.25, 0.3, 0.15):
        row['Phi_global'] = phi
        assert _step(row, _interior([0.6, 0.2]), True, **kw)[0]
    assert row['seal_phi_max'] == 0.3 and row['seal_phi_close'] == 0.2
    x_eq = _floor(row, 'H2O') / (M_MANTLE * 0.15)
    expect = x_eq * M_MANTLE * (0.15 + D_H2O * (0.3 - 0.15))
    assert row['H2O_kg_eq'] == pytest.approx(expect, rel=1e-10)


def _front_held_steps(steps, phi0=0.3):
    """Held steps under front trapping from a closure at ``phi0``; each step is
    (Phi, F_tl) and records trap_dM_RM = M_mantle (Phi_prev - Phi) and trap_F_tl, as
    trapping does before the seal step. Returns the row and the completed rows."""
    import pandas as pd

    row = {**_sealed_row(phi=phi0), 'lid_weight': 1.0, 'lid_delta': 0.5, 'esc_rate_total': 0.0}
    row.update({'seal_active': 1.0, 'seal_path': 2.0})
    done = [dict(row)]
    for phi, f_tl in steps:
        rise = phi > row['Phi_global']
        row['trap_dM_RM'] = M_MANTLE * (row['Phi_global'] - phi)
        row['trap_F_tl'], row['trap_branch'] = (
            (0.0, float(BRANCH_REMELT)) if rise else (f_tl, 2.0)
        )
        row['Phi_global'], row['Time'] = phi, row['Time'] + 10.0
        kw = {'hf_all': pd.DataFrame(done)} if rise else {}
        assert _step(
            row, _interior([0.6, 0.2]), True, delta_close=0.3, trap_mode='front', **kw
        )[0]
        done.append(dict(row))
    front = _config()
    front.outgas.trap_mode = 'front'
    return row, reservoirs(front, row)[0]


@pytest.mark.physics_invariant
def test_a_front_trapping_floor_counts_what_the_open_scheme_buries_over_the_same_steps():
    """Held steps freeze the mantle from Phi 0.3 to 0.12 at a trapped melt fraction
    F_tl of 0.02, 0.1 and 0.35. The floor reservoir of each species is the melt plus
    the mass the open front scheme buries over the same steps at a unit melt
    concentration, sum_k [(1 - F_tl,k) D_i + F_tl,k] dM_RM,k (Sim et al. 2024,
    Eq. 6): H2O at D 0.0017 and CO2 at D 0, which the lattice D_i alone leaves out."""
    from proteus.outgas.trapping import trapped_mass

    steps = ((0.27, 0.02), (0.2, 0.1), (0.12, 0.35))
    row, res = _front_held_steps(steps)
    dm = M_MANTLE * np.diff([0.3, *(p for p, _ in steps)]) * -1.0
    for s, d in (('H2O', D_H2O), ('CO2', 0.0)):
        buried = sum(trapped_mass(d, f, 1.0, m, np.inf)[0] for m, (_, f) in zip(dm, steps))
        assert res[s] == pytest.approx(0.12 * M_MANTLE + buried, rel=1e-14), s


@pytest.mark.physics_invariant
def test_a_held_remelt_returns_the_youngest_crystals_with_their_trapped_melt():
    """Closure at Phi 0.3, freezing to 0.2 at F_tl 0.1 and to 0.1 at F_tl 0.4, then a
    remelt to 0.15: the burial ledger remelts the youngest solid first, so half of
    the second layer and its trapped melt return. Crystals left: 0.9 (0.1) + 0.6
    (0.05) = 0.12, trapped melt 0.1 (0.1) + 0.4 (0.05) = 0.03, in M_mantle; a share
    of the mean F_tl would leave 0.0375. A remelt past closure (0.35) leaves none."""
    row, res = _front_held_steps(((0.2, 0.1), (0.1, 0.4), (0.15, 0.0)))
    assert (row['seal_xtal_kg'], row['seal_tl_kg']) == pytest.approx(
        (0.12 * M_MANTLE, 0.03 * M_MANTLE), rel=1e-13
    )
    assert res['H2O'] == pytest.approx(M_MANTLE * (0.15 + D_H2O * 0.12 + 0.03), rel=1e-13)
    assert res['CO2'] == pytest.approx(M_MANTLE * (0.15 + 0.03), rel=1e-13)
    row, res = _front_held_steps(((0.2, 0.1), (0.1, 0.4), (0.35, 0.0)))
    assert (row['seal_xtal_kg'], row['seal_tl_kg'], res['H2O']) == (0.0, 0.0, 0.35 * M_MANTLE)


@pytest.mark.physics_invariant
def test_noble_gas_leak_columns_hold_the_drop_of_the_sealed_mass_split_by_rate():
    """A noble gas leaks as a species: its leak columns sum to the drop of its
    sealed mass and split between relaxation and volcanism as 1/tau : k."""
    row = _eq_row({'H2O': 1.0e17, 'He': 1.0e15}, {'H2O': 3.0e20, 'He': 1.0e16})
    cfg = _config(tau_leak=1.0e3, crust_rate=1.0e20)
    inv_tau, k = leak_rates(cfg, row, 1.0e3)[0]['He']
    leak_step(cfg, row, 1.0e3)
    tau_part, volc_part = row['He_kg_leak_tau'], row['He_kg_leak_volc']
    assert tau_part + volc_part == pytest.approx(1.0e16 - row['He_kg_sealed'], rel=1e-12)
    assert volc_part > 0.0 and volc_part / tau_part == pytest.approx(k / inv_tau, rel=1e-12)


def test_the_cumulative_leak_adds_the_leak_of_every_step():
    """seal_leak_kg_cum adds each step's element and noble-gas leak to the total of
    the steps before it."""
    row = {**_sealed_row(), 'seal_leak_kg_cum': 0.0, 'esc_rate_total': 0.0}
    cfg, steps = _config(tau_leak=1.0e4), []
    for _ in range(2):
        leak_step(cfg, row, 1.0e3)
        steps.append(
            sum(row[f'{e}_kg_leak_tau'] + row[f'{e}_kg_leak_volc'] for e in LEAK_ELEMENTS)
        )
    assert min(steps) > 0.0
    assert row['seal_leak_kg_cum'] == pytest.approx(sum(steps), rel=1e-14)


@pytest.mark.parametrize(
    ('key', 'bad'),
    [
        *(('seal_Tmagma_close', b) for b in (float('nan'), 0.0, -1.0, None)),
        *(('seal_phi_max', b) for b in (float('nan'), -0.1, 1.5, None)),
        *(('seal_phi_close', b) for b in (float('nan'), -0.1, 0.5, None)),
    ],
    ids=str,
)
def test_a_sealed_front_row_resumes_only_with_its_closing_state(key, bad):
    """A sealed last row needs seal_Tmagma_close > 0 K and, under front trapping,
    0 <= seal_phi_close <= seal_phi_max <= 1 (here 0.4); a missing or unreadable
    value stops the resume."""
    row = {**_sealed_row(), 'seal_active': 1.0, 'seal_phi_max': 0.4}
    front = _config()
    front.outgas.trap_mode = 'front'
    assert restore_seal_state(front, dict(row)) is True
    row.pop(key)
    if bad is not None:
        row[key] = bad
    with pytest.raises(ValueError, match=key):
        restore_seal_state(front, row)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('falls', 'stops'), [(True, True), (False, False)], ids=['falls', 'flat']
)
def test_a_zero_leak_step_with_a_second_solution_stops(falls, stops):
    """A floor at or above the held mass gives no leak; if it falls with the gas a
    leak would add, faster than the leak fills the atmosphere, the step residual
    turns negative inside the bracket and has further roots, so the step stops
    rather than take L = 0. A floor that stays above the held mass passes."""
    m0, a0 = 1.0e19, 1.0e18
    row = _eq_row({'CO2': a0}, {'CO2': m0})
    per = 1.0 / eval_gas_mmw('CO2')

    def model(config, hf_row, per_kg):  # the floor reads the leak through the moles
        def floor_of(leak, s):
            x = (hf_row[f'{s}_mol_atm'] + leak.get(s, 0.0) * per_kg.get(s, 0.0)) / per - a0
            return max(1.001 * m0 - 3.0 * x, 0.2 * m0) if falls else 1.001 * m0

        return floor_of

    with patch('proteus.outgas.lid_seal._floor_model', model):
        if stops:
            with pytest.raises(RuntimeError, match='more than one solution'):
                leak_step(_config(solubility=True, tau_leak=1.0e3), row, 1.0e3)
        else:
            leak_step(_config(solubility=True, tau_leak=1.0e3), row, 1.0e3)
    assert row['CO2_kg_sealed'] == m0
