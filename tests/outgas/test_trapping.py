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
* the start condition, so the first step with no previous row traps nothing,
* the initialisation stage, whose steps do not advance the mantle, trapping
  nothing and recording no diagnostics,
* the trapped share of the solid reservoirs surviving a chemistry solve that
  writes them as zero, and adding to condensed graphite rather than competing
  with it,
* species totals rebuilt from each fresh partition rather than restored, so
  they follow the chemistry instead of freezing,
* desiccation keeping the trapped mass and emptying the totals with nothing
  left, so the closure holds on the desiccated row,
* escape and the desiccation gate both seeing only the reachable inventory,
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
from unittest.mock import patch

import attrs
import numpy as np
import pandas as pd
import pytest

from proteus.config._outgas import Outgas
from proteus.escape.wrapper import calc_new_elements, escapable_mass, reservoir_mass
from proteus.outgas.compaction import BRANCH_DARCY, BRANCH_GUARD, volume_to_mass_fraction
from proteus.outgas.trapping import (
    TrappingStep,
    critical_melt_fraction,
    crystallised_mass_from_phi,
    derived_total_elements,
    escapable_inventory,
    keep_only_trapped_mass,
    locked_solid_mass,
    restore_trapped_mass,
    run_trapping,
    trapped_mass,
    trapped_mass_withheld,
    withhold_trapped_mass,
)
from proteus.outgas.wrapper import check_desiccation
from proteus.utils.coupler import assert_mass_conservation
from proteus.utils.helper import eval_gas_mmw

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
        trap_mode='front',
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


@pytest.fixture
def fixed_front():
    """Front scheme held at F_tl = 0.02, the round value the bookkeeping pins use.

    The bracket and the reservoir bookkeeping are pinned with round numbers
    here; the front that sets F_tl is tested on its own further down, on
    interior-solver profiles.
    """

    def front(*_args, **_kwargs):
        return TrappingStep(
            mode='front', dm_rm=0.0, f_tl=0.02, melt_mass=0.0, branch=BRANCH_DARCY
        )

    with patch('proteus.outgas.trapping._drainage_fraction', side_effect=front) as mocked:
        yield mocked


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
    nothing and buries at the crystal partition coefficients alone. ``gravity`` is the solver's
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
    """Front trapping with the drainage integral over the resolved front."""
    config = _config(
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
    dm = crystallised_mass_from_phi(_M_MANTLE, _PHI_PREV, _PHI_NOW)
    assert dm == pytest.approx(_DM_RM, rel=1e-12)
    # Factor guard: differencing the solid mass instead would need
    # M_mantle_solid, and a regression that used (1 - Phi) rather than the
    # difference would land at 2.4e24 kg, an order of magnitude out.
    assert abs(dm - _M_MANTLE * (1.0 - _PHI_NOW)) > 1.0e24
    # Scale guard: ~1e23 kg, not ~1e21 (a per-cent/fraction mix-up).
    assert 1.0e23 < dm < 1.0e24

    # A remesh that grows the mantle by 10% at the same melt fractions scales
    # the increment by the same 10% and injects nothing else.
    dm_remeshed = crystallised_mass_from_phi(1.1 * _M_MANTLE, _PHI_PREV, _PHI_NOW)
    assert dm_remeshed == pytest.approx(1.1 * _DM_RM, rel=1e-12)

    # Edge case: an unchanged melt fraction crystallises nothing, which is what
    # the prevent-warming clamp asserts when it pins Phi_global.
    still = crystallised_mass_from_phi(_M_MANTLE, _PHI_NOW, _PHI_NOW)
    assert still == pytest.approx(0.0, abs=1.0e-6)

    # Remelting is the same increment with the sign reversed, which the step
    # applies as a release from the trapped reservoir.
    back = crystallised_mass_from_phi(_M_MANTLE, _PHI_NOW, _PHI_PREV)
    assert back == pytest.approx(-_DM_RM, rel=1e-12)

    # Error contract: a non-finite or absent mantle mass yields no increment
    # rather than propagating a NaN into every trapped mass downstream.
    for bad in (float('nan'), 0.0, -1.0):
        dm_bad = crystallised_mass_from_phi(bad, _PHI_PREV, _PHI_NOW)
        assert dm_bad == pytest.approx(0.0, abs=1.0e-6)


@pytest.mark.physics_invariant
def test_incompatible_species_traps_through_the_interstitial_melt_alone():
    """A species at D_Z = 0 takes no place in the crystal lattice, but the melt
    buried between the crystals carries it down regardless, so it still traps
    F_tl * C_Z * dM_RM. With F_tl = 0.02, C_Z = 2e-3 and dM_RM = 2e23 kg that
    is 8e18 kg of CO2, while H2O at D_Z = 0.0017 traps 4.3332e18 kg from a melt
    half as concentrated."""
    carbon, carbon_capped = trapped_mass(0.0, 0.02, 2.0e-3, _DM_RM, 3.6e21)
    assert carbon == pytest.approx(8.0e18, rel=1e-12)
    assert not carbon_capped
    # This is the whole point: a zero partition coefficient is not zero
    # trapping. A regression that short-circuited D_Z = 0 would give 0.0 here.
    assert carbon > 0.0
    # Scale guard: ~1e19 kg, not ~1e22 (melt mass mistaken for concentration).
    assert 1.0e18 < carbon < 1.0e19

    water, _ = trapped_mass(0.0017, 0.02, 1.0e-3, _DM_RM, 1.8e21)
    assert water == pytest.approx(4.3332e18, rel=1e-12)
    # Weight guard: dropping the (1 - F_tl) factor lands at 4.34e18 kg.
    assert abs(water - 0.0217 * 1.0e-3 * _DM_RM) > 1.0e15

    # The bracket is bounded below by F_tl and above by 1 for any physical D_Z,
    # so at equal concentration water can only ever outrun carbon.
    assert trapped_mass(0.0017, 0.02, 2.0e-3, _DM_RM, 3.6e21)[0] > carbon
    # A remelting increment moves the same flux: only its magnitude enters.
    assert trapped_mass(0.0017, 0.02, 1.0e-3, -_DM_RM, 1.8e21)[0] == pytest.approx(water)

    # Edge case: the supply cap binds when a long step asks for more than the
    # melt holds, so no step buries more of a species than exists.
    capped, was_capped = trapped_mass(0.0, 0.5, 1.0e-2, _DM_RM, 1.0e18)
    assert capped == pytest.approx(1.0e18, rel=1e-12)
    assert was_capped
    assert capped < 0.5 * 1.0e-2 * _DM_RM

    # Error contract: an unphysical partition coefficient or trapped fraction
    # is refused rather than moving mass on it.
    with pytest.raises(ValueError, match='D_Z'):
        trapped_mass(-0.1, 0.02, 1.0e-3, _DM_RM, 1.8e21)
    with pytest.raises(ValueError, match='F_tl'):
        trapped_mass(0.0017, 1.5, 1.0e-3, _DM_RM, 1.8e21)


@pytest.mark.physics_invariant
def test_trapping_step_moves_mass_between_reservoirs_and_conserves_each_element(fixed_front):
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
    # The same mass is recorded as trapping's own share of the solid, per
    # species and per element, which is what later solves leave alone.
    assert row['H2O_kg_trapped'] == pytest.approx(4.3332e18, rel=1e-12)
    assert row['H_kg_trapped'] == pytest.approx(row['H_kg_solid'], rel=1e-12)
    assert row['C_kg_trapped'] == pytest.approx(row['C_kg_solid'], rel=1e-12)

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
def test_melt_concentration_is_the_one_the_chemistry_last_solved_for(fixed_front):
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


def test_trapping_waits_for_a_previous_step_and_for_an_enabled_mode(fixed_front):
    """There is no dM_RM without a previous row, so the first step
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


@pytest.mark.physics_invariant
def test_the_initialisation_stage_buries_nothing_and_writes_no_diagnostics(fixed_front):
    """The initialisation iterations advance the clock but not the mantle, and
    the time is reset to zero after each, so a step flagged as one traps
    nothing and records nothing, even on inputs that would otherwise trap. The
    same inputs outside it bury [(1 - F_tl) D + F_tl] C dM_RM of water and
    record the drainage diagnostics at their front-scheme values."""
    held = _hf_row()
    assert run_trapping(_config(), held, _hf_all(), init_stage=True) is None
    assert held['H2O_kg_solid'] == pytest.approx(0.0, abs=0.0)
    assert held['H2O_kg_liquid'] == pytest.approx(1.8e21, rel=1e-12)
    assert 'trap_branch' not in held
    assert 'trap_n_exited' not in held

    # Outside the initialisation stage: (0.98 * 0.0017 + 0.02) * 1e-3 * 2e23.
    live = _hf_row()
    step = run_trapping(_config(), live, _hf_all(), init_stage=False)
    expected = (0.98 * 0.0017 + 0.02) * 1.0e-3 * _DM_RM
    assert live['H2O_kg_solid'] == pytest.approx(expected, rel=1e-9)
    # Dropping the interstitial-melt term would bury about 12 times less.
    assert 0.98 * 0.0017 * 1.0e-3 * _DM_RM < 0.1 * expected
    assert live['H2O_kg_solid'] + live['H2O_kg_liquid'] == pytest.approx(1.8e21, rel=1e-12)
    # The front scheme resolves no nodes and takes no sub-steps, and with no
    # node reaching the floor the share on the no-drainage bound is undefined.
    assert step.n_exited == 0
    assert live['trap_n_exited'] == pytest.approx(0.0, abs=0.0)
    assert live['trap_n_substeps'] == pytest.approx(0.0, abs=0.0)
    assert np.isnan(live['trap_frac_bound'])


@pytest.mark.physics_invariant
def test_trapped_mass_survives_a_chemistry_solve_that_rewrites_the_solid_columns():
    """CALLIOPE writes every _kg_solid field as a hard zero and atmodeller
    writes its condensed graphite there. The trapped share, recorded in
    _kg_trapped, is taken out of the solid columns for the solve and added back
    afterwards, so it survives a solve that zeroes them and adds to condensate
    instead of competing with it."""
    trapped = {'H2O': 4.3332e18, 'CO2': 8.0e18, 'H': 4.849e17, 'C': 2.1818e18}
    row = _hf_row()
    for name, mass in trapped.items():
        row[f'{name}_kg_solid'] = mass
        row[f'{name}_kg_trapped'] = mass

    with trapped_mass_withheld(row):
        # Inside the solve the backend holds only its own share of the solid.
        assert row['H2O_kg_solid'] == pytest.approx(0.0, abs=0.0)
        assert row['C_kg_solid'] == pytest.approx(0.0, abs=0.0)
        # Stand in for a solve that zeroes the species solids and condenses
        # 3e18 kg of carbon as graphite, as atmodeller writes it.
        for name in ('H2O', 'CO2', 'H'):
            row[f'{name}_kg_solid'] = 0.0
        row['C_kg_solid'] = 3.0e18

    assert row['H2O_kg_solid'] == pytest.approx(4.3332e18, rel=1e-12)
    assert row['CO2_kg_solid'] == pytest.approx(8.0e18, rel=1e-12)
    assert row['H_kg_solid'] == pytest.approx(4.849e17, rel=1e-12)
    # Condensed and trapped carbon add. Keeping the larger of the two, the
    # previous rule, would leave 3e18 kg and lose the trapped share.
    assert row['C_kg_solid'] == pytest.approx(3.0e18 + 2.1818e18, rel=1e-12)
    assert abs(row['C_kg_solid'] - 3.0e18) > 1.0e18
    # The record of what trapping owns is never touched by the solve.
    assert row['C_kg_trapped'] == pytest.approx(2.1818e18, rel=1e-12)

    # Error contract: a solve that raises still has the trapped mass put back,
    # so the row is never left carrying only the reachable inventory.
    with pytest.raises(RuntimeError, match='solver failed'):
        with trapped_mass_withheld(row):
            raise RuntimeError('solver failed')
    assert row['H2O_kg_solid'] == pytest.approx(4.3332e18, rel=1e-12)
    assert row['C_kg_solid'] == pytest.approx(5.1818e18, rel=1e-12)

    # Edge case: a row that never trapped anything leaves the backend's
    # graphite exactly as the backend wrote it, as before trapping existed.
    plain = _hf_row(C_kg_solid=3.0e18)
    with trapped_mass_withheld(plain):
        assert plain['C_kg_solid'] == pytest.approx(3.0e18, rel=1e-12)
        plain['C_kg_solid'] = 2.5e18
    assert plain['C_kg_solid'] == pytest.approx(2.5e18, rel=1e-12)


@pytest.mark.physics_invariant
def test_escape_cannot_reach_mass_locked_in_the_solid_mantle():
    """Escape debits the whole-planet total, so without the exclusion it would
    strip volatiles physically locked in the mantle. The mass it may draw on is
    the total less the solid reservoir, and the debit floors at the solid
    reservoir rather than at zero."""
    row = {f'{e}_kg_total': 0.0 for e in ('H', 'O', 'C', 'N', 'S')}
    row['H_kg_total'] = 1.0e20
    row['H_kg_solid'] = 4.0e19
    row['H_kg_trapped'] = 4.0e19
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

    # Condensate the chemistry owns is not locked: atmodeller's graphite in
    # C_kg_solid stays within reach of escape, as it was before trapping.
    row['C_kg_total'] = 5.0e18
    row['C_kg_solid'] = 5.0e18
    assert locked_solid_mass(row, 'C') == pytest.approx(0.0, abs=0.0)
    assert reservoir_mass(row, 'C', '_kg_total') == pytest.approx(5.0e18, rel=1e-12)

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
    locked['H_kg_trapped'] = 5.0e19

    assert escapable_inventory(locked, 'H') == pytest.approx(0.0, abs=1e-30)
    assert locked['H_kg_total'] > config.outgas.mass_thresh
    assert check_desiccation(config, locked)

    # The same inventory reachable rather than locked is not desiccated.
    reachable = dict(locked, H_kg_solid=0.0, H_kg_trapped=0.0, H_kg_atm=5.0e19)
    assert escapable_inventory(reachable, 'H') == pytest.approx(5.0e19, rel=1e-12)
    assert not check_desiccation(config, reachable)

    # Condensate is reachable too: a solid reservoir the chemistry owns does
    # not hold the planet in the desiccated state.
    condensed = dict(locked, H_kg_trapped=0.0)
    assert escapable_inventory(condensed, 'H') == pytest.approx(5.0e19, rel=1e-12)
    assert not check_desiccation(config, condensed)

    # Edge case: a partially locked inventory is judged on the remainder alone.
    partial = dict(locked, H_kg_total=5.0e19, H_kg_solid=4.99e19, H_kg_trapped=4.99e19)
    assert escapable_inventory(partial, 'H') == pytest.approx(1.0e17, rel=1e-9)
    assert not check_desiccation(config, partial)


@pytest.mark.physics_invariant
def test_reservoir_closure_invariant_catches_a_debit_without_its_credit(fixed_front):
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
    withheld from the element totals for the duration of the solve, so the
    chemistry shares out only what it can reach, and the full total is put back
    afterwards; without this the reservoirs over-count by exactly the solid. A
    species total is an output of the solve, so it is rebuilt from the fresh
    partition plus the trapped share rather than restored, and it follows the
    chemistry from one solve to the next instead of freezing."""
    mmw = eval_gas_mmw('H2O')
    row = _hf_row(H_kg_total=2.0e20, H_kg_solid=4.0e19, H2O_kg_total=1.8e21)
    row.update(H_kg_trapped=4.0e19, H2O_kg_solid=3.6e20, H2O_kg_trapped=3.6e20)

    def solve(h2o_atm: float, h2o_liquid: float) -> None:
        """Stand in for the solve: split the reachable hydrogen 1:3 between
        atmosphere and melt, write the water partition, and zero the solids."""
        row['H_kg_atm'] = 0.25 * row['H_kg_total']
        row['H_kg_liquid'] = 0.75 * row['H_kg_total']
        row['H_kg_solid'] = 0.0
        row.update(H2O_kg_atm=h2o_atm, H2O_kg_liquid=h2o_liquid, H2O_kg_solid=0.0)
        row['H2O_kg_total'] = h2o_atm + h2o_liquid
        row.update(H2O_mol_atm=h2o_atm / mmw, H2O_mol_liquid=h2o_liquid / mmw)
        row.update(H2O_mol_solid=0.0, H2O_mol_total=(h2o_atm + h2o_liquid) / mmw)

    withhold_trapped_mass(row)
    # The chemistry now sees the reachable hydrogen alone.
    assert row['H_kg_total'] == pytest.approx(1.6e20, rel=1e-12)
    assert abs(row['H_kg_total'] - 2.0e20) > 1.0e19
    solve(3.6e20, 1.08e21)
    restore_trapped_mass(row)
    assert row['H_kg_total'] == pytest.approx(2.0e20, rel=1e-12)

    # Closure, per element and per species: the reachable split plus the
    # trapped solid is the whole planet, in mass and in moles.
    parts = row['H_kg_atm'] + row['H_kg_liquid'] + row['H_kg_solid']
    assert parts == pytest.approx(row['H_kg_total'], rel=1e-12)
    assert row['H2O_kg_total'] == pytest.approx(3.6e20 + 1.08e21 + 3.6e20, rel=1e-12)
    assert row['H2O_mol_solid'] == pytest.approx(3.6e20 / mmw, rel=1e-12)
    mol_parts = row['H2O_mol_atm'] + row['H2O_mol_liquid'] + row['H2O_mol_solid']
    assert row['H2O_mol_total'] == pytest.approx(mol_parts, rel=1e-12)

    # A second solve turns some water into other species, so the reachable
    # water falls to 1.2e21 kg. The species total follows it to 1.56e21 kg;
    # restoring the total saved before the solve would freeze it at 1.8e21.
    withhold_trapped_mass(row)
    solve(4.0e20, 8.0e20)
    restore_trapped_mass(row)
    assert row['H2O_kg_total'] == pytest.approx(1.56e21, rel=1e-12)
    assert abs(row['H2O_kg_total'] - 1.8e21) > 1.0e20
    assert row['H2O_kg_solid'] == pytest.approx(3.6e20, rel=1e-12)

    # Edge case: with nothing trapped every total is untouched, so a run with
    # trapping disabled hands the chemistry exactly the inventory it had.
    plain = _hf_row(H_kg_total=2.0e20, H2O_kg_total=1.8e21)
    withhold_trapped_mass(plain)
    assert plain['H_kg_total'] == pytest.approx(2.0e20, rel=1e-12)
    restore_trapped_mass(plain)
    assert plain['H2O_kg_total'] == pytest.approx(1.8e21, rel=1e-12)

    # Error contract: an element whose total is not finite is left alone rather
    # than having a NaN propagated into the chemistry input, and a non-finite
    # trapped record counts as nothing trapped.
    broken = _hf_row(H_kg_total=float('nan'), H_kg_trapped=4.0e19)
    withhold_trapped_mass(broken)
    assert np.isnan(broken['H_kg_total'])
    stale = _hf_row(H_kg_total=2.0e20, H_kg_trapped=float('nan'))
    withhold_trapped_mass(stale)
    assert stale['H_kg_total'] == pytest.approx(2.0e20, rel=1e-12)


def test_oxygen_total_is_left_to_the_chemistry_when_the_buffer_owns_it():
    """Holding the oxygen fugacity at a fixed buffer offset requires the
    chemistry to move oxygen, so under ``fO2_source = 'user_constant'`` the
    oxygen total is an output of the solve rather than a conserved budget.
    Trapping must not withhold or debit it, and the per-element closure cannot
    be asserted for it; the trapped oxygen stays as a diagnostic in
    ``O_kg_trapped`` and ``O_kg_solid``. Under ``'from_O_budget'`` the wrapper
    restores the authoritative budget, so oxygen is conserved state and is
    treated like any other element."""
    buffered = SimpleNamespace(planet=SimpleNamespace(fO2_source='user_constant'))
    budgeted = SimpleNamespace(planet=SimpleNamespace(fO2_source='from_O_budget'))
    assert derived_total_elements(buffered) == ('O',)
    assert derived_total_elements(budgeted) == ()
    # Only oxygen is ever chemistry-owned; hydrogen and carbon are escape-owned
    # in both modes, so a regression widening this would be caught here.
    assert 'H' not in derived_total_elements(buffered)
    assert 'C' not in derived_total_elements(buffered)

    row = _hf_row(H_kg_total=2.0e20, H_kg_solid=4.0e19, O_kg_total=1.6e21, O_kg_solid=3.2e20)
    row.update(H_kg_trapped=4.0e19, O_kg_trapped=3.2e20)
    withhold_trapped_mass(row, derived_total_elements(buffered))
    # Hydrogen is withheld so the chemistry partitions only what it can reach.
    assert row['H_kg_total'] == pytest.approx(1.6e20, rel=1e-12)
    # The oxygen total is left exactly as it was, for the chemistry to
    # overwrite, while its solid column still holds only the backend's share.
    assert row['O_kg_total'] == pytest.approx(1.6e21, rel=1e-12)
    assert row['O_kg_solid'] == pytest.approx(0.0, abs=0.0)
    restore_trapped_mass(row, derived_total_elements(buffered))
    # Nothing is credited to the chemistry's total; the solid gets its share.
    assert row['O_kg_total'] == pytest.approx(1.6e21, rel=1e-12)
    assert row['O_kg_solid'] == pytest.approx(3.2e20, rel=1e-12)
    assert row['H_kg_total'] == pytest.approx(2.0e20, rel=1e-12)

    # With the budget authoritative, oxygen is withheld like everything else.
    row2 = _hf_row(O_kg_total=1.6e21, O_kg_solid=3.2e20, O_kg_trapped=3.2e20)
    withhold_trapped_mass(row2, derived_total_elements(budgeted))
    assert row2['O_kg_total'] == pytest.approx(1.28e21, rel=1e-12)
    # Discrimination: the unwithheld total is 1.6e21, well clear of 1.28e21.
    assert abs(row2['O_kg_total'] - 1.6e21) > 1.0e20
    restore_trapped_mass(row2, derived_total_elements(budgeted))
    assert row2['O_kg_total'] == pytest.approx(1.6e21, rel=1e-12)

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


def _desiccated_row() -> dict:
    """Row as desiccation leaves it: every outgassing reservoir zeroed.

    The escape-owned totals are not outgassing keys and survive the zeroing:
    hydrogen holds its trapped 4e19 kg plus a 5e15 kg remainder below the
    desiccation threshold, carbon a 1e15 kg remainder, and helium, which escape
    never floors, 1e14 kg. The oxygen total is an outgassing key and is zeroed.
    """
    row = {'M_planet': 6.0e24, 'M_atm': 0.0, 'M_vol_atm': 0.0}
    for element in ('H', 'O', 'C', 'N', 'S', 'He'):
        for reservoir in ('atm', 'liquid', 'solid', 'total'):
            row[f'{element}_kg_{reservoir}'] = 0.0
    row.update(H_kg_total=4.0e19 + 5.0e15, C_kg_total=1.0e15, He_kg_total=1.0e14)
    row.update(H_kg_trapped=4.0e19, O_kg_trapped=3.2e20, H2O_kg_trapped=3.6e20)
    return row


@pytest.mark.physics_invariant
def test_desiccation_keeps_the_trapped_mass_and_empties_the_rest():
    """Desiccation empties the atmosphere and the melt, not the solid mantle.
    On a row whose outgassing reservoirs have been zeroed, each trapped species
    and element gets its trapped mass back as its solid reservoir and its total,
    and each volatile or noble element that holds nothing has its total emptied,
    so the per-element closure holds on the desiccated row."""
    row = _desiccated_row()
    # The zeroed row fails the closure on the trapped hydrogen: the failure the
    # desiccated step would otherwise abort the run with.
    with pytest.raises(RuntimeError, match='closure failed for H'):
        assert_mass_conservation(row, require_atm_le_planet=False)

    keep_only_trapped_mass(row)
    assert row['H_kg_solid'] == pytest.approx(4.0e19, rel=1e-12)
    assert row['H_kg_total'] == pytest.approx(4.0e19, rel=1e-12)
    # Discrimination: the 5e15 kg remainder has nowhere left to live and goes.
    assert abs(row['H_kg_total'] - (4.0e19 + 5.0e15)) > 1.0e15
    assert row['O_kg_total'] == pytest.approx(3.2e20, rel=1e-12)
    assert row['H2O_kg_solid'] == pytest.approx(3.6e20, rel=1e-12)
    assert row['H2O_mol_total'] == pytest.approx(3.6e20 / eval_gas_mmw('H2O'), rel=1e-12)
    # Edge cases: an element that trapped nothing, and a noble gas, empty.
    assert row['C_kg_total'] == pytest.approx(0.0, abs=0.0)
    assert row['He_kg_total'] == pytest.approx(0.0, abs=0.0)
    assert_mass_conservation(row, require_atm_le_planet=False)

    # Under the oxygen buffer the chemistry's oxygen total excludes the solid:
    # it stays empty while the trapped oxygen is kept as the diagnostic solid.
    buffered = _desiccated_row()
    keep_only_trapped_mass(buffered, ('O',))
    assert buffered['O_kg_total'] == pytest.approx(0.0, abs=0.0)
    assert buffered['O_kg_solid'] == pytest.approx(3.2e20, rel=1e-12)
    assert_mass_conservation(buffered, require_atm_le_planet=False, derived_elements=('O',))

    # Error contract: a non-finite trapped record counts as nothing trapped.
    stale = _desiccated_row()
    stale['H_kg_trapped'] = float('nan')
    keep_only_trapped_mass(stale)
    assert stale['H_kg_total'] == pytest.approx(0.0, abs=0.0)
    assert stale['H_kg_solid'] == pytest.approx(0.0, abs=0.0)


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
    moves nothing and reports no guard."""
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

    # Edge case: nothing crystallised over the step. Nothing moves, the front
    # is not evaluated, and no guard is reported for a step with no flux.
    caplog.clear()
    stalled = _hf_row(Phi_global=_PHI_PREV, M_mantle_solid=2.2e24, gravity=9.8)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.trapping'):
        idle = run_trapping(
            _drainage_config(), stalled, _hf_all(M_mantle_solid=2.2e24), interior
        )
    assert idle.branch != BRANCH_GUARD
    assert idle.total_trapped == pytest.approx(0.0, abs=1e-30)
    assert stalled['H2O_kg_solid'] == pytest.approx(0.0, abs=1e-30)
    assert not any('could not be integrated' in r.getMessage() for r in caplog.records)


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


class _ShallowDenseMeltEOS:
    """Phase-boundary densities with the melt the denser phase below 5 GPa.

    The shallow phase-boundary tables of the Earth-analogue run read that way
    (3084 kg/m3 solid against 3456 kg/m3 melt at 1.9 GPa); deeper, the melt is
    the lighter phase, at the end-member densities of the other doubles.
    """

    def _lookup_at_phase_boundary(self, prop, pressure, phase):
        if prop != 'density':
            raise KeyError(prop)
        shallow = np.asarray(pressure, dtype=float) < 5.0e9
        if phase == 'solid':
            return np.where(shallow, 3084.0, _RHO_SOLID)
        return np.where(shallow, 3456.0, _RHO_MELT)


@pytest.mark.physics_invariant
def test_a_mush_reaching_the_dense_shallow_mantle_takes_the_no_drainage_bound(caplog):
    """Once no node is above the transition, the mush runs to the surface,
    where the shallow tables make its melt denser than the solid. The front
    must include those porous nodes and take the guard branch at the entry
    porosity of the top node, rather than stop at the last node with lighter
    melt and integrate the drainage of a front cut short there."""
    phi = np.linspace(0.02, 0.45, _N_STAG)
    pres = np.linspace(1.3e11, 1.0e5, _N_STAG)
    eos = _ShallowDenseMeltEOS()
    rho_s = eos._lookup_at_phase_boundary('density', pres, 'solid')
    rho_l = eos._lookup_at_phase_boundary('density', pres, 'melt')
    rho = 1.0 / (phi / rho_l + (1.0 - phi) / rho_s)
    shallow = pres < 5.0e9
    assert 1 < shallow.sum() < _N_STAG // 2  # a shallow band, not the whole column
    solver = SimpleNamespace(
        entropy_eos=eos,
        state=SimpleNamespace(phase_staggered=SimpleNamespace(_g=np.full(_N_STAG, 9.8))),
    )
    interior = SimpleNamespace(
        phi=phi,
        density=rho,
        pres=pres,
        radius=np.linspace(_R_CMB, _R_SURF, _N_STAG + 1),
        aragog_solver=solver,
    )
    row = _hf_row(M_mantle_solid=2.4e24, gravity=9.8)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.trapping'):
        step = run_trapping(_drainage_config(), row, _hf_all(M_mantle_solid=2.2e24), interior)

    assert step.branch == BRANCH_GUARD
    assert 'denser than the solid' in step.guard_reason
    assert 'reaches the surface node' in step.guard_reason
    # Every node is porous mush, the dense ones included, so the front spans
    # the whole column; cut short, it would have ended below the shallow band.
    assert step.n_front == _N_STAG
    # The bound is the entry porosity of the top node, its melt volume
    # fraction, converted to mass and capped at rfront_loc = 0.5.
    top_por = phi[-1] * rho[-1] / rho_l[-1]
    bound = volume_to_mass_fraction(top_por, float(np.mean(rho_l)), float(rho_s[0]))
    assert step.f_tl == pytest.approx(min(bound, 0.5), rel=1e-12)
    assert 0.3 < step.f_tl < 0.5
    assert any('denser than the solid' in r.getMessage() for r in caplog.records)


@pytest.mark.physics_invariant
def test_a_step_without_the_interior_profiles_buries_at_the_crystal_term_alone(caplog):
    """The front cannot be located without the interior solver's profiles. The
    crystals still take up D_Z of each species, so such a step buries at the
    partition coefficients alone, with F_tl = 0, on a branch of its own and
    with a warning, rather than inventing a trapped melt fraction."""
    from proteus.outgas.compaction import BRANCH_FALLBACK

    row = _hf_row(M_mantle_solid=2.4e24, gravity=9.8)
    bare = SimpleNamespace(phi=None, density=None, pres=None, radius=None)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.outgas.trapping'):
        step = run_trapping(_config(), row, _hf_all(M_mantle_solid=2.2e24), bare)

    assert step.branch == BRANCH_FALLBACK
    assert step.f_tl == pytest.approx(0.0, abs=0.0)
    # Water is buried at D_Z * C_Z * dM_RM = 0.0017 * 1e-3 * 2e23 kg, and CO2,
    # with no lattice incorporation, not at all.
    assert row['H2O_kg_trapped'] == pytest.approx(0.0017 * 1.0e-3 * _DM_RM, rel=1e-12)
    assert row.get('CO2_kg_trapped', 0.0) == pytest.approx(0.0, abs=0.0)
    # Discrimination: burying at the no-drainage bound instead would take the
    # interstitial melt too, some two orders of magnitude more water.
    assert row['H2O_kg_trapped'] < 0.01 * 0.5 * 1.0e-3 * _DM_RM
    assert any('interior profiles' in r.getMessage() for r in caplog.records)


def _remelting_row(**overrides) -> dict:
    """Row on which the melt fraction rises back from 0.40 to 0.45.

    The previous solve dissolved 1.6e21 kg of water into 4e24 * 0.40 kg of
    melt, a concentration of exactly 1e-3, and trapping holds 1e20 kg of it,
    split into its elements, with every element total closing.
    """
    from proteus.outgas.common import element_masses_from_species

    row = _hf_row(Phi_global=_PHI_PREV, H2O_kg_liquid=1.6e21, CO2_kg_liquid=0.0)
    row.update(H2O_kg_solid=1.0e20, H2O_kg_trapped=1.0e20, C_kg_liquid=0.0)
    for element, mass in element_masses_from_species({'H2O': 1.0e20}).items():
        row[f'{element}_kg_solid'] = mass
        row[f'{element}_kg_trapped'] = mass
    for element in ('H', 'O', 'C'):
        row[f'{element}_kg_total'] = sum(
            row[f'{element}_kg_{r}'] for r in ('atm', 'liquid', 'solid')
        )
    row.update(M_planet=6.0e24, M_atm=0.0, M_vol_atm=0.0)
    row.update(overrides)
    return row


@pytest.mark.physics_invariant
def test_remelting_releases_trapped_mass_by_the_same_flux_it_was_buried_with(fixed_front):
    """A step on which the mantle remelts moves the flux of a crystallising
    step, [(1 - F_tl) D_Z + F_tl] C_Z |dM_RM|, the other way: out of the
    trapped reservoir and back into the melt, capped at what trapping holds.
    A freeze followed by the matching remelt returns every reservoir to where
    it started."""
    row = _remelting_row()
    totals = {e: row[f'{e}_kg_total'] for e in ('H', 'O', 'C')}
    step = run_trapping(_config(), row, _hf_all(Phi_global=_PHI_NOW))

    # dM_RM = 4e24 * (0.40 - 0.45) = -2e23 kg, so the same 4.3332e18 kg of
    # water a freezing step at C_Z = 1e-3 buries comes back out.
    released = (0.98 * 0.0017 + 0.02) * 1.0e-3 * _DM_RM
    assert step.dm_rm == pytest.approx(-_DM_RM, rel=1e-12)
    assert step.trapped_kg['H2O'] == pytest.approx(-released, rel=1e-12)
    assert row['H2O_kg_trapped'] == pytest.approx(1.0e20 - released, rel=1e-12)
    assert row['H2O_kg_liquid'] == pytest.approx(1.6e21 + released, rel=1e-12)
    assert row['trap_kg_step'] < 0.0
    assert step.supply_capped == []
    # Discrimination: remelting used to release nothing at all.
    assert released > 1.0e18
    # Conservation: the element totals are untouched by moving mass back.
    for element, total in totals.items():
        parts = sum(row[f'{element}_kg_{r}'] for r in ('atm', 'liquid', 'solid'))
        assert parts == pytest.approx(total, rel=1e-12)
    assert_mass_conservation(row, require_atm_le_planet=False)

    # Cap: a small trapped inventory is released in full and no further.
    small = _remelting_row(H2O_kg_trapped=1.0e18, H2O_kg_solid=1.0e18)
    capped = run_trapping(_config(), small, _hf_all(Phi_global=_PHI_NOW))
    assert 'H2O' in capped.supply_capped
    assert small['H2O_kg_trapped'] == pytest.approx(0.0, abs=0.0)
    assert small['H2O_kg_liquid'] == pytest.approx(1.6e21 + 1.0e18, rel=1e-12)

    # Round trip: freeze from 0.45 to 0.40, then remelt back. With the melt
    # concentration of the second step above the first, the release is capped
    # at what the first step buried, and every reservoir returns to its start.
    start = _hf_row()
    for element in ('H', 'O', 'C'):
        start[f'{element}_kg_total'] = sum(
            start[f'{element}_kg_{r}'] for r in ('atm', 'liquid', 'solid')
        )
    frozen = dict(start)
    run_trapping(_config(), frozen, _hf_all())
    assert frozen['H2O_kg_trapped'] == pytest.approx(4.3332e18, rel=1e-12)
    thawed = dict(frozen, Phi_global=_PHI_PREV, Time=3.0e4)
    run_trapping(_config(), thawed, pd.DataFrame([dict(frozen)]))
    for key in ('H2O_kg_liquid', 'CO2_kg_liquid', 'H_kg_liquid', 'C_kg_liquid'):
        assert thawed[key] == pytest.approx(start[key], rel=1e-9)
    assert thawed['H2O_kg_trapped'] == pytest.approx(0.0, abs=1.0e3)

    # Edge case: nothing trapped, nothing to release.
    empty = _remelting_row(H2O_kg_trapped=0.0, H2O_kg_solid=0.0)
    idle = run_trapping(_config(), empty, _hf_all(Phi_global=_PHI_NOW))
    assert idle.total_trapped == pytest.approx(0.0, abs=0.0)
    assert empty['H2O_kg_liquid'] == pytest.approx(1.6e21, rel=1e-12)


@pytest.mark.physics_invariant
def test_the_supply_cap_holds_a_step_to_what_the_melt_contains(fixed_front):
    """A species whose crystals take up more than the melt concentration
    (D_Z > 1) can ask a step that freezes the whole melt for more than the
    melt holds. The step buries what there is, no more, and names the species,
    while a species well inside its supply is buried in full."""
    from proteus.outgas.common import element_masses_from_species

    # The whole melt is buried here, so the element liquids must be exactly
    # the species split rather than the rounded values of the shared fixture.
    row = _hf_row(Phi_global=0.0)
    split = element_masses_from_species({'H2O': 1.8e21, 'CO2': 3.6e21})
    for element in ('H', 'O', 'C'):
        row[f'{element}_kg_liquid'] = split[element]
        row[f'{element}_kg_total'] = split[element]
    row.update(M_planet=6.0e24, M_atm=0.0, M_vol_atm=0.0)
    step = run_trapping(_config(D_const_H2O=2.0), row, _hf_all())

    # dM_RM is the whole previous melt, 1.8e24 kg, so the bracket asks for
    # (0.98 * 2 + 0.02) * 1e-3 * 1.8e24 = 3.564e21 kg of the 1.8e21 kg of water.
    assert step.supply_capped == ['H2O']
    assert row['H2O_kg_trapped'] == pytest.approx(1.8e21, rel=1e-12)
    assert row['H2O_kg_liquid'] == pytest.approx(0.0, abs=0.0)
    # Carbon at D_Z = 0 asks for 0.02 * 2e-3 * 1.8e24 = 7.2e19 kg and gets it.
    assert row['CO2_kg_trapped'] == pytest.approx(7.2e19, rel=1e-12)
    assert 'CO2' not in step.supply_capped
    assert_mass_conservation(row, require_atm_le_planet=False)


@pytest.mark.physics_invariant
def test_the_front_speed_follows_the_crystallised_mass_not_the_solid_mass():
    """The front moves at the speed of the mass the step crystallises, the same
    increment the flux uses, so a step on which the solid mantle mass shrank
    while the melt fraction fell (a structure re-solve, or a released
    temperature clamp) is integrated like any other instead of being buried at
    the no-drainage bound."""
    phi_stag = np.linspace(0.30, 1.0, _N_STAG)
    grew = run_trapping(
        _drainage_config(),
        _hf_row(M_mantle_solid=2.4e24, gravity=9.8),
        _hf_all(M_mantle_solid=2.2e24),
        _aragog_interior(phi_stag),
    )
    shrank = run_trapping(
        _drainage_config(),
        _hf_row(M_mantle_solid=2.0e24, gravity=9.8),
        _hf_all(M_mantle_solid=2.2e24),
        _aragog_interior(phi_stag),
    )
    assert grew.branch == BRANCH_DARCY
    assert shrank.branch == BRANCH_DARCY
    assert shrank.v_front > 0.0
    assert shrank.v_front == pytest.approx(grew.v_front, rel=1e-12)
    assert shrank.f_tl == pytest.approx(grew.f_tl, rel=1e-12)
    # Discrimination: the no-drainage bound at this front is the entry
    # porosity as a mass fraction, far above the integrated value.
    bound = volume_to_mass_fraction(phi_stag[11], _RHO_MELT, _RHO_SOLID)
    assert shrank.f_tl < 0.9 * bound
