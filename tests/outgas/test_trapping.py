"""Unit tests for ``proteus.outgas.trapping``.

Solid-phase volatile trapping buries volatiles into the crystallising mantle at
the effective partition coefficient ``D_eff = (1 - F_tl) * D_Z + F_tl``, moving
mass from ``{sp}_kg_liquid`` into ``{sp}_kg_solid`` while the whole-planet total
stays put. These tests exercise:

* the melt-fraction form of the crystallised mass, which is what makes the step
  robust to a structure re-solve that changes the mantle mass,
* an incompatible species at ``D_Z = 0`` still trapping through the
  interstitial-melt term alone,
* per-element closure across the three reservoirs, before and after the step,
* the melt concentration taken as the chemistry solver computed it, on the melt
  it dissolved the volatiles into,
* the ``phi_c`` clamp and the warming branch of the dynamic fraction, with
  ``phi_c`` the interior solver's rheological transition ``rfront_loc``,
* the start condition, so the first step with no previous row traps nothing,
* the solid reservoirs surviving a chemistry solve that writes them as zero,
* escape and the desiccation gate both seeing only the reachable inventory,
* ``DeltaT`` derived from the active melting curves rather than fixed at the
  100 K of Sim et al. (2024),
* the drainage integral run on an interior-solver profile, including a front
  that rests on the core-mantle boundary, and the warning every step on the
  no-drainage upper bound emits with its cause and the mass it buried,
* the front draining under the interior solver's per-node gravity, with the
  structure profile and then the surface value as fallbacks.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import attrs
import numpy as np
import pandas as pd
import pytest

from proteus.config._outgas import Outgas
from proteus.escape.wrapper import calc_new_elements, escapable_mass, reservoir_mass
from proteus.outgas.compaction import BRANCH_DARCY, BRANCH_GUARD, volume_to_mass_fraction
from proteus.outgas.trapping import (
    critical_melt_fraction,
    crystallised_mass_from_phi,
    derive_delta_T,
    derived_total_elements,
    escapable_inventory,
    locked_solid_mass,
    restore_locked_totals,
    restore_solid_reservoirs,
    run_trapping,
    snapshot_solid_reservoirs,
    trapped_fraction,
    trapped_mass,
    withhold_locked_totals,
)
from proteus.outgas.wrapper import check_desiccation
from proteus.utils.coupler import assert_mass_conservation

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Hand-built step. The mantle sheds 5% of its melt fraction over the step, so
# dM_RM = 4e24 * 0.05 = 2e23 kg, and the chemistry last dissolved the volatiles
# into 4e24 * 0.45 = 1.8e24 kg of melt. The dissolved masses are chosen so the
# melt concentrations come out exactly 1e-3 (H2O) and 2e-3 (CO2), which keeps the
# trapped mass a product of round numbers with no interpolation to reason about.
_M_MANTLE = 4.0e24
_PHI_PREV = 0.45
_PHI_NOW = 0.40
_DM_RM = 2.0e23
_MELT_MASS = 1.8e24


def _config(rfront_loc: float = 0.5, **overrides):
    """Config double carrying only the fields the trapping step reads.

    A SimpleNamespace rather than a MagicMock because the partition-coefficient
    lookup is a ``getattr`` with a zero default: a Mock would auto-create every
    ``D_const_*`` attribute as a Mock object and the float conversion would
    fail, hiding the very default the physics relies on. ``rfront_loc`` is the
    interior solver's rheological transition, which is also the disaggregation
    melt fraction the trapped fraction is bounded by.
    """
    outgas = SimpleNamespace(
        trap_mode='constant',
        trap_F_tl=0.02,
        trap_tau=1.0e6,
        trap_delta_T=100.0,
        D_const_H2O=0.0017,
        mass_thresh=1e16,
    )
    for key, value in overrides.items():
        setattr(outgas, key, value)
    return SimpleNamespace(
        outgas=outgas, interior_energetics=SimpleNamespace(rfront_loc=rfront_loc)
    )


def _hf_row(**overrides) -> dict:
    """Current helpfile row at the trapping seam."""
    row = {
        'Time': 2.0e4,
        'M_mantle': _M_MANTLE,
        'Phi_global': _PHI_NOW,
        'T_pot': 1999.99,
        'H2O_kg_liquid': 1.8e21,
        'H2O_kg_atm': 0.0,
        'H2O_kg_solid': 0.0,
        'CO2_kg_liquid': 3.6e21,
        'CO2_kg_atm': 0.0,
        'CO2_kg_solid': 0.0,
        'H_kg_liquid': 2.0141e20,
        'H_kg_atm': 0.0,
        'H_kg_solid': 0.0,
        'O_kg_liquid': 4.2039e21,
        'O_kg_atm': 0.0,
        'O_kg_solid': 0.0,
        'C_kg_liquid': 9.8244e20,
        'C_kg_atm': 0.0,
        'C_kg_solid': 0.0,
    }
    row.update(overrides)
    return row


def _hf_all(**overrides) -> pd.DataFrame:
    """Single previous row, which is all the step differences against."""
    prev = {'Time': 1.0e4, 'M_mantle': _M_MANTLE, 'Phi_global': _PHI_PREV, 'T_pot': 2000.0}
    prev.update(overrides)
    return pd.DataFrame([prev])


# Interior-solver profile for the drainage path: 40 uniform cells from a
# core-mantle boundary at 3500 km to a surface at 6300 km, with end-member
# densities 4000 and 3600 kg/m3 at every pressure.
_N_STAG = 40
_R_CMB = 3.5e6
_R_SURF = 6.3e6
_RHO_SOLID = 4000.0
_RHO_MELT = 3600.0


class _PhaseBoundaryEOS:
    """Equation-of-state double returning fixed end-member densities."""

    def _lookup_at_phase_boundary(self, prop, pressure, phase):
        if prop != 'density':
            raise KeyError(prop)
        rho = _RHO_SOLID if phase == 'solid' else _RHO_MELT
        return np.full(np.shape(pressure), rho)


def _aragog_interior(
    phi_stag: np.ndarray,
    porosity: np.ndarray | None = None,
    gravity: float | np.ndarray | None = 9.8,
    structure: tuple[np.ndarray, np.ndarray] | None = None,
) -> SimpleNamespace:
    """Interior double carrying the profiles the drainage integral reads.

    The mixture density follows the lever rule on ``porosity``, so the
    porosity read back from it is exactly that; it defaults to the solver melt
    fraction. The equation of state sits under ``entropy_eos``, the attribute
    the entropy solver uses; a lookup through the unrelated ``eos`` name finds
    nothing and falls back to the published law. ``gravity`` is the solver's
    per-node value on the staggered nodes (``state.phase_staggered._g``), and
    ``structure`` the radius and gravity columns of the structure profile the
    solver was built from (``parameters.mesh.eos_radius`` and ``eos_gravity``).
    """
    phi_stag = np.asarray(phi_stag, dtype=float)
    por = phi_stag if porosity is None else np.asarray(porosity, dtype=float)
    solver = SimpleNamespace(entropy_eos=_PhaseBoundaryEOS())
    if gravity is not None:
        g_stag = np.broadcast_to(np.asarray(gravity, dtype=float), (_N_STAG,)).copy()
        solver.state = SimpleNamespace(phase_staggered=SimpleNamespace(_g=g_stag))
    if structure is not None:
        eos_radius, eos_gravity = structure
        solver.parameters = SimpleNamespace(
            mesh=SimpleNamespace(eos_radius=eos_radius, eos_gravity=eos_gravity)
        )
    return SimpleNamespace(
        phi=phi_stag,
        density=_RHO_SOLID - por * (_RHO_SOLID - _RHO_MELT),
        pres=np.linspace(1.3e11, 1.0e5, _N_STAG),
        radius=np.linspace(_R_CMB, _R_SURF, _N_STAG + 1),
        aragog_solver=solver,
    )


def _drainage_config():
    """Dynamic trapping with the drainage integral over the resolved front."""
    config = _config(
        trap_mode='dynamic',
        trap_tau_source='aragog',
        trap_phi_min=0.01,
        trap_mush_log10visc=-1.0,
        trap_n_front_min=3,
        trap_max_front_fraction=1.0,
    )
    config.interior_energetics = SimpleNamespace(
        rfront_loc=0.5, grain_size=1.0e-3, melt_log10visc=2.0
    )
    return config


@pytest.mark.physics_invariant
def test_crystallised_mass_follows_melt_fraction_through_a_remesh():
    """dM_RM = M_mantle * [Phi(t-1) - Phi(t)]. Phi is a mass fraction and so
    intensive, which is what makes the increment survive a structure re-solve
    that changes the mantle mass: the same 5% of melt lost gives an increment
    that scales with the mantle it is measured against, with no spurious jump
    from the mesh itself."""
    dm, remelted = crystallised_mass_from_phi(_M_MANTLE, _PHI_PREV, _PHI_NOW)
    assert dm == pytest.approx(_DM_RM, rel=1e-12)
    assert not remelted
    # Factor guard: differencing the solid mass instead would need
    # M_mantle_solid, and a regression that used (1 - Phi) rather than the
    # difference would land at 2.4e24 kg, an order of magnitude out.
    assert abs(dm - _M_MANTLE * (1.0 - _PHI_NOW)) > 1.0e24
    # Scale guard: ~1e23 kg, not ~1e21 (a per-cent/fraction mix-up).
    assert 1.0e23 < dm < 1.0e24

    # A remesh that grows the mantle by 10% at the same melt fractions scales
    # the increment by the same 10% and injects nothing else.
    dm_remeshed, _ = crystallised_mass_from_phi(1.1 * _M_MANTLE, _PHI_PREV, _PHI_NOW)
    assert dm_remeshed == pytest.approx(1.1 * _DM_RM, rel=1e-12)

    # Edge case: an unchanged melt fraction crystallises nothing, which is what
    # the prevent-warming clamp asserts when it pins Phi_global.
    still, _ = crystallised_mass_from_phi(_M_MANTLE, _PHI_NOW, _PHI_NOW)
    assert still == pytest.approx(0.0, abs=1.0e-6)

    # Remelting is clamped and reported rather than buried as a negative mass.
    back, was_remelt = crystallised_mass_from_phi(_M_MANTLE, _PHI_NOW, _PHI_PREV)
    assert was_remelt
    assert back == pytest.approx(0.0, abs=1.0e-6)

    # Error contract: a non-finite or absent mantle mass yields no increment
    # rather than propagating a NaN into every trapped mass downstream.
    for bad in (float('nan'), 0.0, -1.0):
        dm_bad, _ = crystallised_mass_from_phi(bad, _PHI_PREV, _PHI_NOW)
        assert dm_bad == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.physics_invariant
def test_incompatible_species_traps_through_the_interstitial_melt_alone():
    """A species at D_Z = 0 takes no place in the crystal lattice, but the melt
    buried between the crystals carries it down regardless, so it still traps
    F_tl * C_Z * dM_RM. With F_tl = 0.02, C_Z = 2e-3 and dM_RM = 2e23 kg that
    is 8e18 kg of CO2, while H2O at D_Z = 0.0017 traps 4.3332e18 kg from a melt
    half as concentrated."""
    carbon = trapped_mass(0.0, 0.02, 2.0e-3, _DM_RM, 3.6e21)
    assert carbon == pytest.approx(8.0e18, rel=1e-12)
    # This is the whole point: a zero partition coefficient is not zero
    # trapping. A regression that short-circuited D_Z = 0 would give 0.0 here.
    assert carbon > 0.0
    # Scale guard: ~1e19 kg, not ~1e22 (melt mass mistaken for concentration).
    assert 1.0e18 < carbon < 1.0e19

    water = trapped_mass(0.0017, 0.02, 1.0e-3, _DM_RM, 1.8e21)
    assert water == pytest.approx(4.3332e18, rel=1e-12)
    # Weight guard: dropping the (1 - F_tl) factor lands at 4.34e18 kg.
    assert abs(water - 0.0217 * 1.0e-3 * _DM_RM) > 1.0e15

    # The bracket is bounded below by F_tl and above by 1 for any physical D_Z,
    # so at equal concentration water can only ever outrun carbon.
    assert trapped_mass(0.0017, 0.02, 2.0e-3, _DM_RM, 3.6e21) > carbon

    # Edge case: the supply cap binds when a long step asks for more than the
    # melt holds, so no step buries more of a species than exists.
    capped = trapped_mass(0.0, 0.5, 1.0e-2, _DM_RM, 1.0e18)
    assert capped == pytest.approx(1.0e18, rel=1e-12)
    assert capped < 0.5 * 1.0e-2 * _DM_RM

    # Error contract: the no-trapping mode is the absence of the process, not
    # F_tl = 0, which would still bury the species at D_Z.
    assert trapped_mass(0.0017, None, 1.0e-3, _DM_RM, 1.8e21) == pytest.approx(0.0, abs=1e-30)
    with pytest.raises(ValueError, match='D_Z'):
        trapped_mass(-0.1, 0.02, 1.0e-3, _DM_RM, 1.8e21)
    with pytest.raises(ValueError, match='F_tl'):
        trapped_mass(0.0017, 1.5, 1.0e-3, _DM_RM, 1.8e21)


@pytest.mark.physics_invariant
def test_trapping_step_moves_mass_between_reservoirs_and_conserves_each_element():
    """The step buries 4.3332e18 kg of H2O and 8e18 kg of CO2, taking each out
    of the liquid and putting it into the solid. The whole-planet total of every
    element is untouched, so the three reservoirs still close against it."""
    row = _hf_row(M_planet=6.0e24, M_atm=0.0, M_vol_atm=0.0)
    for element in ('H', 'O', 'C'):
        row[f'{element}_kg_total'] = (
            row[f'{element}_kg_atm'] + row[f'{element}_kg_liquid'] + row[f'{element}_kg_solid']
        )
    totals_before = {e: row[f'{e}_kg_total'] for e in ('H', 'O', 'C')}

    step = run_trapping(_config(), row, _hf_all())

    assert step is not None
    assert step.trapped_kg['H2O'] == pytest.approx(4.3332e18, rel=1e-12)
    assert step.trapped_kg['CO2'] == pytest.approx(8.0e18, rel=1e-12)
    assert row['H2O_kg_solid'] == pytest.approx(4.3332e18, rel=1e-12)
    assert row['H2O_kg_liquid'] == pytest.approx(1.8e21 - 4.3332e18, rel=1e-12)

    # Conservation: the per-element closure the runtime invariant asserts.
    for element, total in totals_before.items():
        parts = (
            row[f'{element}_kg_atm'] + row[f'{element}_kg_liquid'] + row[f'{element}_kg_solid']
        )
        assert parts == pytest.approx(total, rel=1e-12)
        assert row[f'{element}_kg_solid'] > 0.0
    assert_mass_conservation(row, require_atm_le_planet=False)

    # The element split carries exactly the species mass it came from.
    element_gain = sum(row[f'{e}_kg_solid'] for e in ('H', 'O', 'C'))
    assert element_gain == pytest.approx(step.total_trapped, rel=1e-9)

    # Edge case: a mantle already solid at the previous solve dissolved nothing,
    # so the step is a no-op rather than a division by zero.
    dry = _hf_row(Phi_global=0.0)
    dry_step = run_trapping(_config(), dry, _hf_all(Phi_global=0.0))
    assert dry_step.total_trapped == pytest.approx(0.0, abs=1e-30)
    assert np.isfinite(dry['H2O_kg_liquid'])


@pytest.mark.physics_invariant
def test_melt_concentration_is_the_one_the_chemistry_last_solved_for():
    """The dissolved masses in the row come from the previous chemistry solve,
    which spread them over the previous step's melt. Trapping uses that
    concentration as it is: dividing by the current, smaller melt would inflate
    it by Phi(t-1) / Phi(t), 12.5% on this step. A structure re-solve between
    the two solves scales the crystallised mass but not the concentration, and
    the step that freezes the last of the melt still buries its share."""
    step = run_trapping(_config(), _hf_row(), _hf_all())
    # CALLIOPE's concentration of CO2 is 3.6e21 / (4e24 * 0.45) = 2e-3, so at
    # D_Z = 0 and F_tl = 0.02 the step buries 8e18 kg.
    assert step.trapped_kg['CO2'] == pytest.approx(0.02 * 2.0e-3 * _DM_RM, rel=1e-12)
    assert step.melt_mass == pytest.approx(_MELT_MASS, rel=1e-12)
    # Discrimination guard: the current melt, 4e24 * 0.40, would give 9e18 kg.
    inflated = 0.02 * 3.6e21 / (_M_MANTLE * _PHI_NOW) * _DM_RM
    assert abs(step.trapped_kg['CO2'] - inflated) > 5.0e17

    # A re-solve grew the mantle 10% since the chemistry ran: dM_RM follows the
    # current mantle, the concentration stays the one the solver computed.
    grown = run_trapping(_config(), _hf_row(M_mantle=1.1 * _M_MANTLE), _hf_all())
    assert grown.trapped_kg['CO2'] == pytest.approx(0.02 * 2.0e-3 * 1.1 * _DM_RM, rel=1e-12)

    # Edge case: the last of the melt freezes in one step, Phi 0.45 -> 0. The
    # crystallised 1.8e24 kg held CO2 at 2e-3, and F_tl of it is buried; a
    # concentration taken on the current melt would see none and bury nothing.
    frozen = run_trapping(_config(), _hf_row(Phi_global=0.0), _hf_all())
    assert frozen.dm_rm == pytest.approx(_MELT_MASS, rel=1e-12)
    assert frozen.trapped_kg['CO2'] == pytest.approx(0.02 * 2.0e-3 * _MELT_MASS, rel=1e-12)
    assert 0.0 < frozen.trapped_kg['CO2'] < 3.6e21


@pytest.mark.physics_invariant
def test_dynamic_fraction_clamps_at_disaggregation_and_zeroes_on_warming():
    """F_tl = -(phi_c * tau / DeltaT) * dT/dt. With phi_c = 0.3, tau = 1e6 yr
    and DeltaT = 100 K the prefactor is 3000 yr K-1, so cooling at 1e-6 K/yr
    gives 0.003, cooling at 1e-3 K/yr overshoots to 3.0 and clamps to 0.3, and
    a warming step goes negative and clamps to zero. In a run, phi_c is the
    interior solver's rheological transition rfront_loc, so the clamp moves
    with it."""
    gentle, high, low = trapped_fraction(-1.0e-6, 0.3, 1.0e6, 100.0)
    assert gentle == pytest.approx(0.003, rel=1e-12)
    assert not (high or low)
    # Sign guard: the leading minus is what makes a cooling mantle trap melt.
    assert gentle > 0.0
    # Prefactor guard: dropping tau/DeltaT leaves phi_c * |dT/dt| = 3e-7.
    assert abs(gentle - 0.3 * 1.0e-6) > 1.0e-3

    clamped, high, low = trapped_fraction(-1.0e-3, 0.3, 1.0e6, 100.0)
    assert clamped == pytest.approx(0.3, rel=1e-12)
    assert high and not low
    # Boundedness: the fraction used never leaves the physical window.
    assert 0.0 <= clamped <= 0.3

    warming, high, low = trapped_fraction(2.0e-6, 0.3, 1.0e6, 100.0)
    assert warming == pytest.approx(0.0, abs=1e-15)
    assert low and not high

    # A warming step still buries the species at D_Z, because crystal
    # partitioning continues even with no interstitial melt retained. That is
    # what separates it from the none mode, where both would be zero.
    row = _hf_row(T_pot=2000.02)
    step = run_trapping(_config(trap_mode='dynamic'), row, _hf_all())
    assert step.f_tl == pytest.approx(0.0, abs=1e-15)
    assert step.trapped_kg['H2O'] == pytest.approx(0.0017 * 1.0e-3 * _DM_RM, rel=1e-12)
    assert 'CO2' not in step.trapped_kg

    # The clamp is the solver's rheological transition, not a separate value:
    # cooling 10 K over the 1e4 yr step drives the raw fraction to 10 * phi_c,
    # so the step sits on the clamp, and the clamp lands on rfront_loc. Neither
    # value is the paper's 0.3, which a hard-coded bound would return.
    for rfront_loc in (0.5, 0.35):
        fast = run_trapping(
            _config(rfront_loc=rfront_loc, trap_mode='dynamic'),
            _hf_row(T_pot=1990.0),
            _hf_all(),
        )
        assert fast.f_tl == pytest.approx(rfront_loc, rel=1e-12)
        assert fast.clamped_high

    # Edge case: an isothermal step has a rate of exactly zero, which yields no
    # trapped melt without tripping the lower clamp.
    flat, high, low = trapped_fraction(0.0, 0.3, 1.0e6, 100.0)
    assert flat == pytest.approx(0.0, abs=1e-15)
    assert not (high or low)

    # Error contract: the three parameters are refused out of range.
    with pytest.raises(ValueError, match='phi_c'):
        trapped_fraction(-1.0e-6, 0.0, 1.0e6, 100.0)
    with pytest.raises(ValueError, match='tau'):
        trapped_fraction(-1.0e-6, 0.3, -1.0, 100.0)
    with pytest.raises(ValueError, match='DeltaT'):
        trapped_fraction(-1.0e-6, 0.3, 1.0e6, 0.0)


def test_trapping_waits_for_a_previous_step_and_for_an_enabled_mode():
    """There is no dM_RM and no dT/dt without a previous row, so the first step
    traps nothing and leaves the solid reservoir untouched. The none mode does
    the same at every step, which is the default so an existing run is
    unchanged until trapping is asked for."""
    first = _hf_row()
    assert run_trapping(_config(), first, None) is None
    assert first['H2O_kg_solid'] == pytest.approx(0.0, abs=1e-30)

    empty = _hf_row()
    assert run_trapping(_config(), empty, pd.DataFrame([])) is None
    assert empty['H2O_kg_liquid'] == pytest.approx(1.8e21, rel=1e-12)

    # Time must have advanced past zero before anything is buried.
    at_zero = _hf_row(Time=0.0)
    assert run_trapping(_config(), at_zero, _hf_all()) is None
    assert at_zero['H2O_kg_solid'] == pytest.approx(0.0, abs=1e-30)

    off = _hf_row()
    assert run_trapping(_config(trap_mode='none'), off, _hf_all()) is None
    assert off['H2O_kg_solid'] == pytest.approx(0.0, abs=1e-30)
    assert off['H2O_kg_liquid'] == pytest.approx(1.8e21, rel=1e-12)

    # A step that does run on the same inputs proves the guards above are the
    # reason nothing moved, not an inert calculation.
    live = _hf_row()
    assert run_trapping(_config(), live, _hf_all()).total_trapped > 0.0


def test_solid_reservoir_survives_a_chemistry_solve_that_writes_it_as_zero():
    """CALLIOPE has no solid phase and writes every _kg_solid field as a hard
    zero, and the binodal H2 override does the same. The snapshot taken before
    the solve puts the trapped inventory back afterwards, so it is overwritten
    rather than lost."""
    row = _hf_row(H2O_kg_solid=4.3332e18, H_kg_solid=4.849e17, CO2_kg_solid=8.0e18)
    snapshot = snapshot_solid_reservoirs(row)
    assert snapshot['H2O'] == pytest.approx(4.3332e18, rel=1e-12)

    # Stand in for the chemistry solve, which flattens the whole set.
    for name in snapshot:
        row[f'{name}_kg_solid'] = 0.0
    assert row['H2O_kg_solid'] == pytest.approx(0.0, abs=1e-30)

    restore_solid_reservoirs(row, snapshot)
    assert row['H2O_kg_solid'] == pytest.approx(4.3332e18, rel=1e-12)
    assert row['CO2_kg_solid'] == pytest.approx(8.0e18, rel=1e-12)
    assert row['H_kg_solid'] == pytest.approx(4.849e17, rel=1e-12)

    # A backend that writes a larger condensate mass into the same column keeps
    # it: atmodeller reports graphite there, and taking the larger avoids both
    # erasing it and counting the two sources twice.
    row['CO2_kg_solid'] = 9.0e18
    restore_solid_reservoirs(row, snapshot)
    assert row['CO2_kg_solid'] == pytest.approx(9.0e18, rel=1e-12)

    # Edge case: a reservoir that was empty before the solve is not resurrected,
    # so the restore cannot invent mass the run never had.
    empty = _hf_row()
    empty_snapshot = snapshot_solid_reservoirs(empty)
    empty['N2_kg_solid'] = 5.0e17
    restore_solid_reservoirs(empty, empty_snapshot)
    assert empty['N2_kg_solid'] == pytest.approx(5.0e17, rel=1e-12)


@pytest.mark.physics_invariant
def test_escape_cannot_reach_mass_locked_in_the_solid_mantle():
    """Escape debits the whole-planet total, so without the exclusion it would
    strip volatiles physically locked in the mantle. The mass it may draw on is
    the total less the solid reservoir, and the debit floors at the solid
    reservoir rather than at zero."""
    row = {f'{e}_kg_total': 0.0 for e in ('H', 'O', 'C', 'N', 'S')}
    row['H_kg_total'] = 1.0e20
    row['H_kg_solid'] = 4.0e19
    row['H_kg_atm'] = 1.0e19

    # Sizing: the bulk reservoir sees only the reachable 6e19 kg.
    assert reservoir_mass(row, 'H', '_kg_total') == pytest.approx(6.0e19, rel=1e-12)
    assert escapable_mass(row, 'bulk') == pytest.approx(6.0e19, rel=1e-12)
    # Discrimination: the unexcluded answer is 1e20 kg, well clear of it.
    assert abs(escapable_mass(row, 'bulk') - 1.0e20) > 1.0e19
    # The atmospheric reservoir never holds solid mass, so it is untouched.
    assert reservoir_mass(row, 'H', '_kg_atm') == pytest.approx(1.0e19, rel=1e-12)
    assert escapable_mass(row, 'outgas') == pytest.approx(1.0e19, rel=1e-12)

    # Debiting: an oversized request under the default outgas reservoir cannot
    # drive the total below the locked mass.
    target = calc_new_elements(row, dt=1.0, reservoir='outgas', esc_mass=9.0e19)
    assert target['H'] == pytest.approx(4.0e19, rel=1e-12)
    assert target['H'] >= locked_solid_mass(row, 'H')
    # Without the floor the total would land at 1e19 kg, or be zeroed outright
    # by the minimum-mass threshold; neither is reachable now.
    assert abs(target['H'] - 1.0e19) > 1.0e19

    # Edge case: with nothing locked, the floor is zero and behaviour is
    # unchanged from before trapping existed.
    free = {f'{e}_kg_total': 0.0 for e in ('H', 'O', 'C', 'N', 'S')}
    free['H_kg_total'] = 1.0e20
    free['H_kg_atm'] = 1.0e20
    unlocked = calc_new_elements(free, dt=1.0, reservoir='outgas', esc_mass=9.0e19)
    assert unlocked['H'] == pytest.approx(1.0e19, rel=1e-12)
    # The whole request lands, because with nothing locked the floor is zero;
    # the same request against the locked row above stopped at 4e19 kg.
    assert unlocked['H'] < free['H_kg_total']
    assert escapable_mass(free, 'bulk') == pytest.approx(1.0e20, rel=1e-12)


def test_desiccation_ignores_the_reservoir_nothing_can_reach():
    """A planet whose atmosphere and melt have both emptied is desiccated even
    with a full solid reservoir, because trapped mass can neither escape nor
    outgas. Testing the whole-planet total instead would hold it above the
    threshold forever and the run would never terminate."""
    config = _config()
    config.outgas.mass_thresh = 1e16
    config.params = SimpleNamespace(stop=SimpleNamespace(escape=SimpleNamespace(enabled=False)))

    locked = {}
    for element in ('H', 'O', 'C', 'N', 'S', 'He', 'Ne', 'Ar', 'Kr', 'Xe'):
        locked[f'{element}_kg_total'] = 0.0
        locked[f'{element}_kg_solid'] = 0.0
    locked['H_kg_total'] = 5.0e19
    locked['H_kg_solid'] = 5.0e19

    assert escapable_inventory(locked, 'H') == pytest.approx(0.0, abs=1e-30)
    assert locked['H_kg_total'] > config.outgas.mass_thresh
    assert check_desiccation(config, locked)

    # The same inventory reachable rather than locked is not desiccated.
    reachable = dict(locked, H_kg_solid=0.0, H_kg_atm=5.0e19)
    assert escapable_inventory(reachable, 'H') == pytest.approx(5.0e19, rel=1e-12)
    assert not check_desiccation(config, reachable)

    # Edge case: a partially locked inventory is judged on the remainder alone.
    partial = dict(locked, H_kg_total=5.0e19, H_kg_solid=4.99e19)
    assert escapable_inventory(partial, 'H') == pytest.approx(1.0e17, rel=1e-9)
    assert not check_desiccation(config, partial)


@pytest.mark.reference_pinned
def test_delta_T_is_derived_from_the_melting_curves_not_fixed_at_100_K():
    """Sim, Hirschmann and Hier-Majumder (2024), JGR Planets 129, e2024JE008346
    fix DeltaT = 100 C for every simulation. Here it is the temperature interval
    between the solidus and the melt fraction phi_c, which under the lever rule
    the structure solver uses is phi_c * (T_liquidus - T_solidus). With the
    PALEOS liquidus and mushy_zone_factor = 0.8 that is
    0.3 * 1831.0 * 0.2 = 109.9 K at the surface, close to the paper's value."""
    pytest.importorskip('zalmoxis')
    from proteus.config import read_config_object

    config = read_config_object('input/all_options.toml')
    derived = derive_delta_T(config, phi_c=0.3)

    mzf = config.interior_struct.zalmoxis.mushy_zone_factor
    assert mzf == pytest.approx(0.8, rel=1e-12)
    assert derived == pytest.approx(109.86, rel=1e-3)
    # It is genuinely derived, not the fallback constant being returned.
    assert abs(derived - 100.0) > 5.0
    # Scale guard: same order as the paper, not 10 K or 1000 K, so a curve read
    # in the wrong unit would not pass.
    assert 50.0 < derived < 400.0

    # The derivation is linear in phi_c, which fixes the lever rule
    # independently of the curves themselves.
    assert derive_delta_T(config, phi_c=0.6) == pytest.approx(2.0 * derived, rel=1e-9)

    # Deeper in the mantle the melting interval widens, so DeltaT is a real
    # function of the reference pressure rather than a constant in disguise.
    deep = derive_delta_T(config, phi_c=0.3, pressure=10.0e9)
    assert deep > 1.5 * derived

    # Error contract: an EOS with no melting curves falls back to the paper's
    # 100 K rather than raising or returning a nonsensical interval.
    broken = read_config_object('input/all_options.toml')
    object.__setattr__(broken.interior_struct.zalmoxis, 'mantle_eos', 'NotAnEOS:Nothing')
    assert derive_delta_T(broken, phi_c=0.3) == pytest.approx(100.0, rel=1e-12)


@pytest.mark.physics_invariant
def test_reservoir_closure_invariant_catches_a_debit_without_its_credit():
    """The per-element closure is the check that would actually catch a
    trapping bug. A step that takes mass out of the liquid without crediting the
    solid breaks total == atm + liquid + solid, and the invariant refuses the
    row by name rather than letting the run drift."""
    row = _hf_row(M_planet=6.0e24, M_atm=0.0, M_vol_atm=0.0)
    for element in ('H', 'O', 'C'):
        row[f'{element}_kg_total'] = (
            row[f'{element}_kg_atm'] + row[f'{element}_kg_liquid'] + row[f'{element}_kg_solid']
        )
    run_trapping(_config(), row, _hf_all())
    assert_mass_conservation(row, require_atm_le_planet=False)
    assert row['H_kg_solid'] > 0.0

    # Drop the credit and the closure fails, naming the element.
    broken = dict(row)
    broken['H_kg_solid'] = 0.0
    with pytest.raises(RuntimeError, match='closure failed for H'):
        assert_mass_conservation(broken, require_atm_le_planet=False)

    # Edge case: an element the run never carried has every field at zero and
    # is skipped rather than dividing by a zero total.
    untouched = dict(row)
    untouched['N_kg_total'] = 0.0
    untouched['N_kg_liquid'] = 0.0
    assert_mass_conservation(untouched, require_atm_le_planet=False)

    # A drift under the tolerance is admitted, so float rounding in the element
    # split does not fire the check on every physically sound step.
    nudged = dict(row)
    nudged['H_kg_liquid'] = row['H_kg_liquid'] * (1.0 + 1.0e-9)
    assert_mass_conservation(nudged, require_atm_le_planet=False)


@pytest.mark.physics_invariant
def test_chemistry_cannot_redissolve_what_the_mantle_has_buried():
    """Every outgassing backend partitions a whole-planet inventory between melt
    and atmosphere and has no solid reservoir of its own. The trapped mass is
    withheld from the total for the duration of the solve, so the chemistry
    shares out only what it can reach, and the full total is put back
    afterwards. Without this the reservoirs over-count by exactly the solid."""
    row = _hf_row(H_kg_total=2.0e20, H_kg_solid=4.0e19, H2O_kg_total=1.8e21)
    row['H2O_kg_solid'] = 3.6e20

    saved = withhold_locked_totals(row)
    # The chemistry now sees the reachable inventory alone.
    assert row['H_kg_total'] == pytest.approx(1.6e20, rel=1e-12)
    assert row['H2O_kg_total'] == pytest.approx(1.44e21, rel=1e-12)
    # Discrimination: the unwithheld totals are 2e20 and 1.8e21 kg.
    assert abs(row['H_kg_total'] - 2.0e20) > 1.0e19

    # Stand in for the solve, which splits the reachable total it was given
    # across melt and atmosphere and writes no solid.
    row['H_kg_atm'] = 0.25 * row['H_kg_total']
    row['H_kg_liquid'] = 0.75 * row['H_kg_total']
    row['H_kg_solid'] = 0.0

    restore_locked_totals(row, saved)
    restore_solid_reservoirs(row, {'H': 4.0e19})
    assert row['H_kg_total'] == pytest.approx(2.0e20, rel=1e-12)

    # Closure: the reachable split plus the restored solid is the whole planet.
    parts = row['H_kg_atm'] + row['H_kg_liquid'] + row['H_kg_solid']
    assert parts == pytest.approx(row['H_kg_total'], rel=1e-12)
    assert row['H_kg_solid'] > 0.0

    # Edge case: with nothing trapped the totals are untouched, so a run with
    # trapping disabled sees byte-identical chemistry input.
    plain = _hf_row(H_kg_total=2.0e20)
    assert withhold_locked_totals(plain) == {}
    assert plain['H_kg_total'] == pytest.approx(2.0e20, rel=1e-12)

    # Error contract: an element whose total is not finite is left alone rather
    # than having a NaN propagated into the chemistry input.
    broken = _hf_row(H_kg_total=float('nan'), H_kg_solid=4.0e19)
    withhold_locked_totals(broken)
    assert np.isnan(broken['H_kg_total'])


def test_oxygen_total_is_left_to_the_chemistry_when_the_buffer_owns_it():
    """Holding the oxygen fugacity at a fixed buffer offset requires the
    chemistry to move oxygen, so under ``fO2_source = 'user_constant'`` the
    oxygen total is an output of the solve rather than a conserved budget.
    Trapping must not withhold or debit it, and the per-element closure cannot
    be asserted for it; the trapped oxygen stays as a diagnostic in
    ``O_kg_solid``. Under ``'from_O_budget'`` the wrapper restores the
    authoritative budget, so oxygen is conserved state and is treated like any
    other element."""
    buffered = SimpleNamespace(planet=SimpleNamespace(fO2_source='user_constant'))
    budgeted = SimpleNamespace(planet=SimpleNamespace(fO2_source='from_O_budget'))
    assert derived_total_elements(buffered) == ('O',)
    assert derived_total_elements(budgeted) == ()
    # Only oxygen is ever chemistry-owned; hydrogen and carbon are escape-owned
    # in both modes, so a regression widening this would be caught here.
    assert 'H' not in derived_total_elements(buffered)
    assert 'C' not in derived_total_elements(buffered)

    row = _hf_row(H_kg_total=2.0e20, H_kg_solid=4.0e19, O_kg_total=1.6e21, O_kg_solid=3.2e20)
    saved = withhold_locked_totals(row, derived_total_elements(buffered))
    # Hydrogen is withheld so the chemistry partitions only what it can reach.
    assert 'H' in saved
    assert row['H_kg_total'] == pytest.approx(1.6e20, rel=1e-12)
    # Oxygen is left exactly as it was, for the chemistry to overwrite.
    assert 'O' not in saved
    assert row['O_kg_total'] == pytest.approx(1.6e21, rel=1e-12)

    # With the budget authoritative, oxygen is withheld like everything else.
    row2 = _hf_row(O_kg_total=1.6e21, O_kg_solid=3.2e20)
    saved2 = withhold_locked_totals(row2, derived_total_elements(budgeted))
    assert 'O' in saved2
    assert row2['O_kg_total'] == pytest.approx(1.28e21, rel=1e-12)
    # Discrimination: the unwithheld total is 1.6e21, well clear of 1.28e21.
    assert abs(row2['O_kg_total'] - 1.6e21) > 1.0e20

    # The closure invariant skips a chemistry-owned total and still enforces
    # every other element, which is what keeps the check live where it applies.
    broken = _hf_row(M_planet=6.0e24, M_atm=0.0, M_vol_atm=0.0)
    broken['O_kg_total'] = 1.0e21
    broken['O_kg_atm'] = 0.0
    broken['O_kg_liquid'] = 0.0
    broken['O_kg_solid'] = 3.0e20  # deliberately does not close
    for element in ('H', 'C'):
        broken[f'{element}_kg_total'] = (
            broken[f'{element}_kg_atm'] + broken[f'{element}_kg_liquid']
        )
    assert_mass_conservation(broken, require_atm_le_planet=False, derived_elements=('O',))
    with pytest.raises(RuntimeError, match='closure failed for O'):
        assert_mass_conservation(broken, require_atm_le_planet=False)


def _closure_row(parts_over_total: float) -> dict:
    """Row whose sulfur reservoirs miss the carried total by a given fraction.

    Sulfur carries no trapped mass here, so any mismatch is the chemistry
    solver's residual rather than a trapping debit.
    """
    total = 8.729082e20
    row = _hf_row(M_planet=6.0e24, M_atm=0.0, M_vol_atm=0.0)
    for element in ('H', 'O', 'C'):
        row[f'{element}_kg_total'] = (
            row[f'{element}_kg_atm'] + row[f'{element}_kg_liquid'] + row[f'{element}_kg_solid']
        )
    row['S_kg_total'] = total
    row['S_kg_solid'] = 0.0
    row['S_kg_liquid'] = 8.590e17
    row['S_kg_atm'] = total * parts_over_total - row['S_kg_liquid']
    return row


def test_reservoir_closure_is_held_to_the_chemistry_solvers_own_tolerance():
    """The atmospheric and liquid reservoirs are outputs of a nonlinear
    chemistry solve that converges to ``solver_rtol``, while the total is carried
    forward by the escape chain, so they can agree no more tightly than the
    solver does. The three mismatches below are the ones a real coupled run
    produced: a trapping-free run at 1.49e-6, the same run with trapping at
    2.49e-5, and a genuine oxygen debit fault at 3.27e-4. Held to the solver's
    1e-4 the first two pass and the fault is still refused."""
    solver_rtol = 1.0e-4

    # A healthy run with trapping disabled, which the bare 1e-6 wrongly refused.
    healthy = _closure_row(1.0 - 1.49e-6)
    with pytest.raises(RuntimeError, match='closure failed for S'):
        assert_mass_conservation(healthy, require_atm_le_planet=False)
    assert_mass_conservation(healthy, require_atm_le_planet=False, closure_rtol=solver_rtol)

    # The same configuration with trapping on, still inside the solver's reach.
    trapped = _closure_row(1.0 + 2.49e-5)
    assert_mass_conservation(trapped, require_atm_le_planet=False, closure_rtol=solver_rtol)

    # Discrimination: a real bookkeeping fault three times the solver tolerance
    # is still caught, which is what keeps the relaxation from blinding the check.
    fault = _closure_row(1.0 + 3.27e-4)
    with pytest.raises(RuntimeError, match='closure failed for S'):
        assert_mass_conservation(fault, require_atm_le_planet=False, closure_rtol=solver_rtol)

    # The solver tolerance only ever loosens the closure: a tighter value leaves
    # atol_frac in charge, so the healthy row is refused exactly as before.
    with pytest.raises(RuntimeError, match='closure failed for S'):
        assert_mass_conservation(healthy, require_atm_le_planet=False, closure_rtol=1.0e-9)

    # The species-sum invariant keeps its own atol_frac: a stale M_vol_atm is
    # refused however loose the closure tolerance is.
    stale = _closure_row(1.0)
    stale['M_vol_atm'] = 1.0e20
    stale['H2O_kg_atm'] = 5.0e19
    with pytest.raises(RuntimeError, match='M_vol_atm bookkeeping'):
        assert_mass_conservation(stale, require_atm_le_planet=False, closure_rtol=1.0e-1)


def _reservoir_sums(hf_row: dict) -> dict[str, float]:
    """Per-element mass summed over the three reservoirs [kg]."""
    return {
        e: sum(hf_row[f'{e}_kg_{r}'] for r in ('atm', 'liquid', 'solid'))
        for e in ('H', 'O', 'C')
    }


@pytest.mark.physics_invariant
def test_a_front_resting_on_the_core_mantle_boundary_is_drained_not_bounded(caplog):
    """A mantle crystallising from the bottom up is still porous at its lowest
    node, so its front rests on the core-mantle boundary. The step integrates
    the drainage over that front instead of taking the no-drainage upper
    bound, measures the front from the boundary rather than from the lowest
    node, and conserves every element it moves from the melt into the solid."""
    # Melt fraction 0.30 at the lowest node rising to 1 at the surface. The
    # front spans nodes 0 to 11 and its top crosses 0.5 at 11.14 spacings.
    phi_stag = np.linspace(0.30, 1.0, _N_STAG)
    # The solid mantle grows by the 2e23 kg crystallised over the 1e4 yr step.
    hf_row = _hf_row(M_mantle_solid=2.4e24, gravity=9.8)
    before = _reservoir_sums(hf_row)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.trapping'):
        step = run_trapping(
            _drainage_config(),
            hf_row,
            _hf_all(M_mantle_solid=2.2e24),
            _aragog_interior(phi_stag),
        )

    # Percolation-limited: the matrix closes in about 1 kyr, the melt needs
    # about 40 kyr to percolate out of an 815 km front.
    assert step.branch == BRANCH_DARCY
    assert step.tau_d > 10.0 * step.tau_s
    assert step.guard_reason == ''
    assert step.n_front == 12
    assert not any('could not be integrated' in rec.getMessage() for rec in caplog.records)

    # Thickness from the boundary at 3500 km, not from the lowest node 35 km
    # above it, which would give 780 km instead of 815 km.
    dr = (_R_SURF - _R_CMB) / _N_STAG
    thickness = dr * (0.2 * 39.0 / 0.7 + 0.5)
    assert step.l_front == pytest.approx(thickness, rel=1e-9)
    # v_f dt / L with the new solid spread over the boundary sphere; the time
    # step cancels, leaving dM / (4 pi r_cmb^2 rho_s L).
    courant = 2.0e23 / (4.0 * np.pi * _R_CMB**2 * _RHO_SOLID * thickness)
    assert step.front_courant == pytest.approx(courant, rel=1e-9)
    assert 0.3 < step.front_courant < 0.5
    # The residence time is the step length over the Courant number [yr].
    assert step.t_res == pytest.approx(1.0e4 / courant, rel=1e-9)

    # The drained fraction is positive and well below the no-drainage bound
    # taken from the entry porosity, 0.471 as a mass fraction.
    bound = volume_to_mass_fraction(phi_stag[11], _RHO_MELT, _RHO_SOLID)
    assert 0.0 < step.f_tl < 0.9 * bound

    # Conservation: each element only moved between reservoirs, and the water
    # buried is the effective partition coefficient of the drained F_tl.
    after = _reservoir_sums(hf_row)
    for element, total in before.items():
        assert after[element] == pytest.approx(total, rel=1e-12)
    h2o = ((1.0 - step.f_tl) * 0.0017 + step.f_tl) * 1.0e-3 * _DM_RM
    assert hf_row['H2O_kg_solid'] == pytest.approx(h2o, rel=1e-12)


@pytest.mark.physics_invariant
def test_every_step_on_the_upper_bound_reports_its_cause_and_the_mass_it_buried(caplog):
    """A front the drainage integral cannot handle falls back to the
    no-drainage upper bound, which buries far more than drainage would. Every
    such step logs a warning naming the cause and the mass buried, the bound
    itself is the entry porosity converted to a mass fraction and capped at the
    critical melt fraction rfront_loc, and a step on which nothing crystallised
    still warns while burying nothing."""
    # A second porous layer near the surface splits the mush in two.
    phi_stag = np.linspace(0.30, 1.0, _N_STAG)
    phi_stag[30:33] = 0.45
    interior = _aragog_interior(phi_stag)
    hf_row = _hf_row(M_mantle_solid=2.4e24, gravity=9.8)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.trapping'):
        step = run_trapping(
            _drainage_config(), hf_row, _hf_all(M_mantle_solid=2.2e24), interior
        )

    assert step.branch == BRANCH_GUARD
    assert step.guard_reason == 'the mush is split into 2 separate layers'
    assert hf_row['trap_branch'] == pytest.approx(float(BRANCH_GUARD), rel=1e-12)
    bound = volume_to_mass_fraction(phi_stag[11], _RHO_MELT, _RHO_SOLID)
    assert step.f_tl == pytest.approx(bound, rel=1e-12)
    # At F_tl = 0.471 the water buried is 278 times the lattice-only term.
    lattice_only = 0.0017 * 1.0e-3 * _DM_RM
    h2o = ((1.0 - bound) * 0.0017 + bound) * 1.0e-3 * _DM_RM
    assert hf_row['H2O_kg_solid'] == pytest.approx(h2o, rel=1e-12)
    assert 250.0 < hf_row['H2O_kg_solid'] / lattice_only < 300.0
    guard_logs = [
        r.getMessage() for r in caplog.records if 'could not be integrated' in r.getMessage()
    ]
    assert len(guard_logs) == 1
    assert 'split into 2 separate layers' in guard_logs[0]
    assert f'{step.total_trapped:.3e} kg' in guard_logs[0]

    # The bound never exceeds the critical melt fraction. Densities implying
    # porosity 0.6 at the top node, a mass fraction of 0.574, would claim more
    # melt than the framework locks at the solver's transition; the step holds
    # rfront_loc = 0.5 instead.
    porous = phi_stag.copy()
    porous[11] = 0.6
    capped = run_trapping(
        _drainage_config(),
        _hf_row(M_mantle_solid=2.4e24, gravity=9.8),
        _hf_all(M_mantle_solid=2.2e24),
        _aragog_interior(phi_stag, porosity=porous),
    )
    assert capped.branch == BRANCH_GUARD
    assert capped.f_tl == pytest.approx(0.5, rel=1e-12)
    assert abs(capped.f_tl - volume_to_mass_fraction(0.6, _RHO_MELT, _RHO_SOLID)) > 0.05

    # Edge case: nothing crystallised over the step. The stalled solid is
    # named as a further cause, nothing is buried, and the warning still fires.
    caplog.clear()
    stalled = _hf_row(Phi_global=_PHI_PREV, M_mantle_solid=2.2e24, gravity=9.8)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.trapping'):
        idle = run_trapping(
            _drainage_config(), stalled, _hf_all(M_mantle_solid=2.2e24), interior
        )
    assert idle.branch == BRANCH_GUARD
    assert 'the solid mantle did not grow' in idle.guard_reason
    assert idle.total_trapped == pytest.approx(0.0, abs=1e-30)
    assert stalled['H2O_kg_solid'] == pytest.approx(0.0, abs=1e-30)
    assert any('0.000e+00 kg' in r.getMessage() for r in caplog.records)


def test_the_disaggregation_fraction_is_the_solver_transition_and_nothing_else():
    """phi_c is interior_energetics.rfront_loc, so the rheology, the reported
    front and every trapping bound share one number. The outgassing section
    carries no melt-fraction threshold of its own, and a configuration without
    the interior value fails loudly instead of falling back to the paper's
    0.3."""
    names = [f.name for f in attrs.fields(Outgas)]
    assert 'trap_mode' in names
    assert not [n for n in names if 'phi_c' in n or 'rfront' in n or 'disaggregat' in n]
    # The accessor follows the configured transition, including values away
    # from both the default 0.5 and the paper's 0.3; 0.05 is the low edge a
    # run might use near full crystallisation.
    for rfront_loc in (0.5, 0.42, 0.05):
        assert critical_melt_fraction(_config(rfront_loc=rfront_loc)) == pytest.approx(
            rfront_loc, rel=1e-12
        )
    # Error contract: no silent default when the interior section is absent.
    with pytest.raises(AttributeError):
        critical_melt_fraction(SimpleNamespace(outgas=SimpleNamespace()))


@pytest.mark.physics_invariant
def test_the_front_drains_under_the_interior_solvers_per_node_gravity(caplog):
    """Both drainage timescales scale as 1/g, and a front near the core-mantle
    boundary sits where gravity is well above its surface value. The front
    drains under the interior solver's per-node gravity averaged over the front
    nodes, then under the structure profile the solver was built from, and only
    without either under the surface gravity, with a warning."""
    phi_stag = np.linspace(0.30, 1.0, _N_STAG)  # front on nodes 0 to 11

    def matrix_time(interior, surface_g=8.0):
        step = run_trapping(
            _drainage_config(),
            _hf_row(M_mantle_solid=2.4e24, gravity=surface_g),
            _hf_all(M_mantle_solid=2.2e24),
            interior,
        )
        return step

    # tau_s = mu_s / (drho g L) [yr] with mu_s = 1e20 Pa s, drho = 400 kg/m3 and
    # the 815 km front of this profile.
    length = (_R_SURF - _R_CMB) / _N_STAG * (0.2 * 39.0 / 0.7 + 0.5)

    def expected_tau_s(g):
        return 1.0e20 / (400.0 * g * length) / 3.15576e7

    # The solver's own profile, 12 m/s2 at the base falling to 8 at the surface:
    # the front mean over nodes 0 to 11 is 12 - 4 * 5.5 / 39 = 11.436 m/s2.
    solver_g = np.linspace(12.0, 8.0, _N_STAG)
    g_front = 12.0 - 4.0 * 5.5 / 39.0
    deep = matrix_time(_aragog_interior(phi_stag, gravity=solver_g))
    assert deep.branch == BRANCH_DARCY
    assert deep.tau_s == pytest.approx(expected_tau_s(g_front), rel=1e-9)
    # Discrimination guard: the surface value would make tau_s 43% longer.
    assert deep.tau_s < 0.75 * expected_tau_s(8.0)

    # Invariant: both timescales scale as 1/g at a fixed front, so the uniform
    # 9.8 m/s2 run and the deep one agree on tau * g for each.
    uniform = matrix_time(_aragog_interior(phi_stag, gravity=9.8), surface_g=9.8)
    assert uniform.tau_s * 9.8 == pytest.approx(deep.tau_s * g_front, rel=1e-9)
    assert uniform.tau_d * 9.8 == pytest.approx(deep.tau_d * g_front, rel=1e-9)

    # Without the solver's values, the structure profile the solver was built
    # from is interpolated onto the nodes: 12 - 4 (k + 0.5) / 40 at node k, a
    # front mean of 11.4 m/s2. Also the error contract for a corrupt solver
    # profile, which is passed over rather than used.
    structure = (np.array([_R_CMB, _R_SURF]), np.array([12.0, 8.0]))
    for solver_values in (None, np.full(_N_STAG, np.nan)):
        fallback = matrix_time(
            _aragog_interior(phi_stag, gravity=solver_values, structure=structure)
        )
        assert fallback.tau_s == pytest.approx(expected_tau_s(11.4), rel=1e-9)

    # Edge case: no profile at all. The surface gravity stands in, loudly.
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.trapping'):
        bare = matrix_time(_aragog_interior(phi_stag, gravity=None))
    assert bare.tau_s == pytest.approx(expected_tau_s(8.0), rel=1e-9)
    assert any('no usable per-node gravity' in r.getMessage() for r in caplog.records)
