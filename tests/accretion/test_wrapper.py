"""Tests for the accretion wrapper and its initialisation contract.

This file targets accretion/wrapper.py (init_accretion). The wrapper is
what the main loop calls once at start-up, so what it must guarantee is
that a run with accretion disabled is untouched, that the configured
backend is the one consulted, and that impacts falling outside the
simulated interval are reported rather than dropped in silence.

See testing standards in docs/How-to/testing.md and
docs/Explanations/test_framework.md for required structure, speed, and
physics validity.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from proteus.accretion.common import TIMELINE_COLUMNS
from proteus.accretion.wrapper import init_accretion

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_ROWS = (
    (
        1.0e5,
        6.0e24,
        6.4e23,
        6.64e24,
        1.3e4,
        1.15e4,
        0.7,
        6.371e6,
        3.39e6,
        5510.0,
        3930.0,
        1.496e11,
        1.4e11,
        0.02,
        0.05,
        1,
        4,
    ),
    (
        5.0e5,
        6.64e24,
        1.0e23,
        6.74e24,
        1.2e4,
        1.1e4,
        0.3,
        6.4e6,
        2.0e6,
        5510.0,
        3930.0,
        1.4e11,
        1.35e11,
        0.03,
        0.02,
        1,
        7,
    ),
)


def _timeline_file(path):
    """Write a two-impact timeline at 1e5 and 5e5 yr."""
    lines = [','.join(TIMELINE_COLUMNS)]
    lines += [','.join(repr(v) for v in row) for row in _ROWS]
    path.write_text('\n'.join(lines) + '\n')
    return path


def _handler(
    module=None,
    timeline_path=None,
    time_offset=0.0,
    time_start=0.0,
    interior_module='dummy',
    temperature_mode='liquidus_super',
    output_dir=None,
    resume=False,
):
    """Build the minimal Proteus handler shape init_accretion reads."""
    return SimpleNamespace(
        config=SimpleNamespace(
            accretion=SimpleNamespace(
                module=module,
                time_offset=time_offset,
                timeline=SimpleNamespace(
                    timeline_path=None if timeline_path is None else str(timeline_path)
                ),
            ),
            interior_energetics=SimpleNamespace(module=interior_module),
            planet=SimpleNamespace(temperature_mode=temperature_mode),
            params=SimpleNamespace(resume=resume),
        ),
        directories={'output': str(output_dir) if output_dir is not None else '.'},
        hf_row={'Time': time_start},
    )


@pytest.mark.unit
def test_disabled_accretion_returns_no_impacts(tmp_path):
    """A run without accretion gets an empty schedule and reads no files.

    Every existing configuration has accretion off, so this path must stay
    a pure no-op: an empty list, and no attempt to touch a timeline. The
    file check matters because a stray read would make the disabled path
    fail on configs that never mention a timeline at all.
    """
    handler = _handler(module=None, timeline_path=tmp_path / 'never_written.csv')

    events = init_accretion(handler)

    assert events == []
    assert not (tmp_path / 'never_written.csv').exists()

    # The handler is not mutated on the disabled path.
    assert handler.hf_row == {'Time': 0.0}


@pytest.mark.unit
def test_enabled_backend_returns_the_scheduled_impacts(tmp_path):
    """The configured backend supplies the schedule the main loop consults.

    The returned list is what the timestep clamp and the impact handler
    read on every step, so it has to arrive complete and in time order,
    with the physical content of each record preserved.
    """
    handler = _handler(
        module='timeline',
        timeline_path=_timeline_file(tmp_path / 't.csv'),
        output_dir=tmp_path,
    )

    events = init_accretion(handler)

    assert len(events) == 2
    assert [e.time for e in events] == [1.0e5, 5.0e5]
    assert events[0].mass_delta == pytest.approx(6.4e23)

    # Chain continuity survives the wrapper, so the schedule describes one
    # growing body rather than a set of unrelated impacts.
    assert events[1].M_target_before == pytest.approx(events[0].M_merged_after)


@pytest.mark.unit
def test_impacts_before_the_run_starts_are_reported_and_excluded(tmp_path, caplog):
    """Impacts outside the simulated interval are announced, not swallowed.

    The configuration owns the planet's initial mass and orbit, so an
    impact landing before the run begins cannot be applied without
    contradicting it. Dropping it silently would understate the planet's
    accretion history with no trace in the log, so the count and the
    missing mass are reported and the offset is named as the fix.
    """
    path = _timeline_file(tmp_path / 't.csv')

    # Start the run after the first impact but before the second.
    handler = _handler(
        module='timeline', timeline_path=path, time_start=2.0e5, output_dir=tmp_path
    )

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        events = init_accretion(handler)

    assert [e.time for e in events] == [5.0e5]

    warning = '\n'.join(r.getMessage() for r in caplog.records)
    assert '1 impact' in warning
    assert 'time_offset' in warning
    # The mass that will not be accreted is quantified, so the size of the
    # omission is visible rather than merely its existence.
    assert '0.107' in warning  # 6.4e23 kg expressed in Earth masses

    # An impact landing exactly on the start time is already accounted for
    # by the initial condition and is excluded too.
    boundary = _handler(
        module='timeline', timeline_path=path, time_start=1.0e5, output_dir=tmp_path
    )
    assert [e.time for e in init_accretion(boundary)] == [5.0e5]

    # Shifting the timeline forward brings both impacts back into range,
    # which is the documented remedy.
    shifted = _handler(
        module='timeline',
        timeline_path=path,
        time_offset=3.0e5,
        time_start=2.0e5,
        output_dir=tmp_path,
    )
    assert len(init_accretion(shifted)) == 2


def _impact_event(**overrides):
    """Build one physically self-consistent impact record for the handler."""
    from proteus.accretion.common import ImpactEvent

    base = dict(
        time=1.0e5,
        M_target_before=6.0e24,
        M_impactor=6.4e23,
        M_merged_after=6.64e24,
        v_impact=1.30e4,
        v_esc=1.15e4,
        impact_parameter=0.7,
        R_target_before=6.371e6,
        R_impactor=3.39e6,
        rho_target=5510.0,
        rho_impactor=3930.0,
        a_before=1.496e11,
        a_after=1.4e11,
        e_before=0.02,
        e_after=0.05,
        id_target=1,
        id_impactor=4,
    )
    base.update(overrides)
    return ImpactEvent(**base)


def _impact_accretion(atmloss_module=None, atmloss_frac=0.0, impactor_volatiles=None, **ppmw):
    """Accretion sub-config: impactor volatiles and atmosphere loss (default off).

    The content mode defaults to 'ppmw' when per-element budgets are given and
    to 'dry' otherwise, so a test states only the physics it exercises.
    """
    if impactor_volatiles is None:
        impactor_volatiles = 'ppmw' if any(v > 0.0 for v in ppmw.values()) else 'dry'
    return SimpleNamespace(
        impactor_volatiles=impactor_volatiles,
        impactor_H_ppmw=ppmw.get('H', 0.0),
        impactor_C_ppmw=ppmw.get('C', 0.0),
        impactor_N_ppmw=ppmw.get('N', 0.0),
        impactor_S_ppmw=ppmw.get('S', 0.0),
        impactor_O_ppmw=ppmw.get('O', 0.0),
        atmloss_module=atmloss_module,
        atmloss_frac=atmloss_frac,
    )


def _impact_handler(
    mass_tot=1.0,
    semimajoraxis=0.5,
    eccentricity=0.1,
    tsurf_init=4000.0,
    crystallized=False,
    accretion=None,
):
    """Build the minimal handler shape apply_impact reads and mutates.

    The dummy interior is used so the mantle re-melt runs for real (it resets
    the temperature and the melt state) without needing a live solver.
    """
    from proteus.utils.constants import AU

    return SimpleNamespace(
        config=SimpleNamespace(
            planet=SimpleNamespace(mass_tot=mass_tot, tsurf_init=tsurf_init),
            orbit=SimpleNamespace(semimajoraxis=semimajoraxis, eccentricity=eccentricity),
            interior_energetics=SimpleNamespace(
                module='dummy',
                dummy=SimpleNamespace(mantle_tliq=2700.0, mantle_tsol=1700.0),
            ),
            interior_struct=SimpleNamespace(core_frac=0.55),
            accretion=accretion if accretion is not None else _impact_accretion(),
        ),
        hf_row={
            'semimajorax': semimajoraxis * AU,
            'eccentricity': eccentricity,
            'T_magma': 2000.0,  # cooled; the re-melt should reset it
            'M_int': mass_tot * 5.9736e24,
            'M_core': 0.3 * mass_tot * 5.9736e24,
            'R_int': 6.4e6,
            'R_core': 3.5e6,
        },
        hf_all=None,
        interior_o=SimpleNamespace(impact_reset=False),
        crystallized=crystallized,
        directories={'output': '/tmp/unused'},
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_impact_grows_the_planet_by_the_impactor_mass_and_re_solves(monkeypatch):
    """An impact adds the impactor mass and rebuilds the interior structure.

    The mass the planet gains is the impactor mass, the difference between
    the merged and target masses, not the merged mass itself, which is an
    order of magnitude larger here and is the plausible wrong reading. The
    structure is re-solved once against the new total mass so the radius and
    the core/mantle split follow it rather than staying frozen at the old
    mass.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import M_earth

    calls = []
    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure',
        lambda *a, **k: calls.append(a),
    )

    handler = _impact_handler(mass_tot=1.0)
    # Impactor is 0.5 Earth masses; merged mass is 6.5 (ten times larger).
    event = _impact_event(
        M_target_before=6.0 * M_earth,
        M_impactor=0.5 * M_earth,
        M_merged_after=6.5 * M_earth,
    )
    apply_impact(handler, event)

    assert handler.config.planet.mass_tot == pytest.approx(1.5, rel=1e-12)
    # Discrimination: adding the merged mass instead would land near 7.5,
    # five Earth masses away, far outside any tolerance.
    assert abs(handler.config.planet.mass_tot - (1.0 + 6.5)) > 1.0

    # The structure was re-solved exactly once, against the grown planet.
    assert len(calls) == 1

    # The mantle was re-melted to its molten initial temperature, above the
    # cooled 2000 K it started this step at, and fully molten.
    assert handler.hf_row['T_magma'] == pytest.approx(4000.0, rel=1e-12)
    assert handler.hf_row['T_magma'] > 2000.0
    assert handler.hf_row['Phi_global'] == pytest.approx(1.0, rel=1e-12)
    # The interior stepper is told the temperature jump is a deliberate reset.
    assert handler.interior_o.impact_reset is True


@pytest.mark.unit
def test_impact_on_a_crystallised_planet_reopens_outgassing(monkeypatch):
    """A re-melting impact clears the one-way solidification latch.

    Once the mantle solidifies the run latches into a frozen-mantle path with
    outgassing shut off. A giant impact that re-melts the mantle to a magma
    ocean must lift that latch, or the re-melted planet would keep being
    treated as a solid with its volatiles trapped for the rest of the run.
    """
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler(crystallized=True)
    assert handler.crystallized is True  # latched before the impact
    apply_impact(handler, _impact_event())

    # The impact re-melted the mantle, so the latch is lifted.
    assert handler.crystallized is False


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_impact_moves_the_orbit_in_both_the_config_and_the_row(monkeypatch):
    """The orbit change is applied as a jump to both the config and the row.

    Both elements move by the change the impact made rather than taking the
    followed body's absolute values, because the configuration owns the
    planet's orbit: a borrowed impact history moves it, it does not replace it.
    The semi-major axis takes the ratio and the eccentricity the difference,
    since eccentricity is dimensionless and routinely zero, which a ratio
    cannot express. Both the configuration, which pins the orbit when tides are
    off, and the running row, which the tidal evolution carries forward when
    tides are on, must be written, or the jump would be lost under one of the
    two orbit modes.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import AU

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler(semimajoraxis=0.5, eccentricity=0.1)
    # a_after / a_before = 1.4e11 / 1.4e11 scaled: choose a clean 1.2 ratio.
    # The followed body goes 0.02 -> 0.03, so the impact excites it by +0.01.
    event = _impact_event(a_before=1.0e11, a_after=1.2e11, e_before=0.02, e_after=0.03)
    ratio = 1.2

    apply_impact(handler, event)

    assert handler.config.orbit.semimajoraxis == pytest.approx(0.5 * ratio, rel=1e-12)
    assert handler.hf_row['semimajorax'] == pytest.approx(0.5 * AU * ratio, rel=1e-12)
    # Config (AU) and row (metres) describe the same orbit after the jump.
    assert handler.hf_row['semimajorax'] / AU == pytest.approx(
        handler.config.orbit.semimajoraxis, rel=1e-12
    )
    # The planet's own 0.1 is excited by the impact's +0.01, not replaced by
    # the followed body's 0.03.
    assert handler.config.orbit.eccentricity == pytest.approx(0.11, rel=1e-12)
    assert handler.hf_row['eccentricity'] == pytest.approx(0.11, rel=1e-12)
    # Discrimination: transplanting the absolute value would give 0.03, which
    # is nearly four times away from the correct 0.11.
    assert abs(0.03 - 0.11) > 0.5 * 0.11


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_a_grazing_head_on_impact_leaves_the_orbit_circular(monkeypatch):
    """A circularising impact damps the planet's own eccentricity, and stops at zero.

    An impact that circularises the followed body applies a negative change,
    which must reduce the planet's eccentricity rather than replace it. The
    result is clamped at zero, since a negative eccentricity has no meaning and
    would propagate into the separation and Hill-radius formulae as a sign
    error. The semi-major axis still moves by its ratio independently of it.
    """
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    # The followed body is circularised from 0.05 to 0, a change of -0.05,
    # which damps a planet at 0.2 to 0.15 rather than resetting it.
    handler = _impact_handler(semimajoraxis=1.0, eccentricity=0.2)
    event = _impact_event(a_before=1.0e11, a_after=1.0e11, e_before=0.05, e_after=0.0)
    apply_impact(handler, event)

    assert handler.config.orbit.eccentricity == pytest.approx(0.15, rel=1e-12)
    assert handler.hf_row['eccentricity'] == pytest.approx(0.15, rel=1e-12)
    # Equal before/after semi-major axis is a unit ratio, so the orbit size
    # is unchanged while the eccentricity is damped.
    assert handler.config.orbit.semimajoraxis == pytest.approx(1.0, rel=1e-12)

    # A change larger than the planet's own eccentricity clamps at zero rather
    # than going negative, which is the boundary the clamp exists for.
    floored = _impact_handler(semimajoraxis=1.0, eccentricity=0.01)
    apply_impact(
        floored, _impact_event(a_before=1.0e11, a_after=1.0e11, e_before=0.05, e_after=0.0)
    )
    assert floored.config.orbit.eccentricity == 0.0
    assert floored.hf_row['eccentricity'] == 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_impact_delivers_configured_volatiles_into_the_element_budgets(monkeypatch):
    """An opted-in impactor adds its volatile content to the planet budgets.

    Delivery is the impactor mass times the configured content in parts per
    million by weight, added to the whole-planet element inventory the
    outgassing step reads. Only the elements with a non-zero content are
    touched: a dry element leaves its budget, and a budget deferred to the
    chemistry step, untouched.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import M_earth

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    # Impactor delivers 1000 ppmw H and 500 ppmw S; C, N, O are dry.
    handler = _impact_handler(accretion=_impact_accretion(H=1000.0, S=500.0))
    handler.hf_row['H_kg_total'] = 2.0e20  # a pre-existing hydrogen budget
    handler.hf_row['S_kg_total'] = 1.0e20
    m_impactor = 0.5 * M_earth
    apply_impact(handler, _impact_event(M_impactor=m_impactor))

    # Hydrogen grew by exactly M_impactor * 1000e-6.
    expected_H = 2.0e20 + m_impactor * 1000.0 / 1.0e6
    assert handler.hf_row['H_kg_total'] == pytest.approx(expected_H, rel=1e-12)
    # Discrimination: forgetting the ppmw-to-fraction 1e6 would overshoot by a
    # million-fold, and delivering nothing would leave it at 2e20.
    assert handler.hf_row['H_kg_total'] > 2.0e20
    assert handler.hf_row['H_kg_total'] < 2.0e20 + m_impactor  # never the full impactor mass

    # Sulfur grew by its own configured amount.
    assert handler.hf_row['S_kg_total'] == pytest.approx(
        1.0e20 + m_impactor * 500.0 / 1.0e6, rel=1e-12
    )
    # A dry element that was never in the row is not created.
    assert 'O_kg_total' not in handler.hf_row


@pytest.mark.unit
def test_a_dry_impactor_delivers_no_volatiles(monkeypatch):
    """The default dry impactor leaves every element budget untouched.

    Delivery is opt-in per element and defaults to zero, so a run that sets no
    impactor content must not create or grow any element budget at an impact.
    """
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler()  # dry impactor (all ppmw zero)
    handler.hf_row['H_kg_total'] = 3.0e20
    apply_impact(handler, _impact_event())

    # The existing budget is unchanged and no new element key appears.
    assert handler.hf_row['H_kg_total'] == pytest.approx(3.0e20, rel=1e-12)
    assert not any(k.endswith('_kg_total') and k != 'H_kg_total' for k in handler.hf_row)


def _atm_state(hf_row, **kg):
    """Write an atmospheric composition: per-element atm and total budgets.

    Each keyword is an element symbol mapped to ``(kg_atm, kg_total)`` so a
    test can set up asymmetric atmospheric and dissolved reservoirs.
    """
    for e, (atm, total) in kg.items():
        hf_row[f'{e}_kg_atm'] = atm
        hf_row[f'{e}_kg_total'] = total


@pytest.mark.unit
def test_impact_atmosphere_loss_is_off_by_default(monkeypatch):
    """Without an atmosphere-loss module the impact leaves the atmosphere alone.

    Every existing accretion configuration predates impact atmosphere loss, so
    the default must be a strict no-op: no element budget moves and the
    escaped-mass ledger is untouched, even for a violent impact on a planet
    with a massive atmosphere.
    """
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler()  # atmloss_module=None
    _atm_state(handler.hf_row, H=(2.0e20, 5.0e20), N=(1.0e19, 4.0e19))
    handler.hf_row['esc_kg_cumulative'] = 7.0e18
    apply_impact(handler, _impact_event())

    assert handler.hf_row['H_kg_total'] == pytest.approx(5.0e20, rel=1e-12)
    assert handler.hf_row['N_kg_total'] == pytest.approx(4.0e19, rel=1e-12)
    assert handler.hf_row['esc_kg_cumulative'] == pytest.approx(7.0e18, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_impact_strips_the_atmosphere_in_proportion_to_its_composition(monkeypatch):
    """The stripped mass is drawn from the atmosphere, element by element.

    A constant 25% loss removes exactly a quarter of each element's
    ATMOSPHERIC reservoir from its whole-planet budget: the dissolved interior
    inventory is untouched, so an element that is mostly dissolved loses far
    less of its total than one that is mostly atmospheric. Partitioning by the
    total budgets instead would shift mass between the two, which the
    asymmetric reservoirs here are chosen to expose. The removed mass is
    booked into the escaped-mass ledger the desiccation gate audits.
    """
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler(
        accretion=_impact_accretion(atmloss_module='constant', atmloss_frac=0.25)
    )
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    # H is mostly atmospheric; N is mostly dissolved. A total-budget
    # partitioning would debit N nearly 4x more than the atmosphere holds.
    _atm_state(handler.hf_row, H=(4.0e20, 5.0e20), N=(1.0e19, 4.0e20))
    apply_impact(handler, _impact_event())

    # Each element loses a quarter of its ATMOSPHERIC mass from the total.
    assert handler.hf_row['H_kg_total'] == pytest.approx(5.0e20 - 0.25 * 4.0e20, rel=1e-9)
    assert handler.hf_row['N_kg_total'] == pytest.approx(4.0e20 - 0.25 * 1.0e19, rel=1e-9)
    # Discrimination: partitioning over the equal TOTAL budgets would debit
    # both elements identically (0.25 * 0.5 * (4e20 + 1e19) each ~ 5.1e19),
    # putting N at ~3.49e20, more than 5e18 away from the correct 3.975e20.
    assert abs(handler.hf_row['N_kg_total'] - 3.4875e20) > 4.0e18

    # The debit never exceeds what the atmosphere held.
    assert handler.hf_row['H_kg_total'] >= 5.0e20 - 4.0e20
    assert handler.hf_row['N_kg_total'] >= 4.0e20 - 1.0e19

    # The stripped mass is booked for the desiccation ledger.
    assert handler.hf_row['esc_kg_cumulative'] == pytest.approx(
        0.25 * (4.0e20 + 1.0e19), rel=1e-9
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_total_impact_loss_removes_the_atmosphere_but_not_the_interior(monkeypatch):
    """A loss fraction of one is the boundary: the atmosphere goes, no more.

    Full stripping removes each element's atmospheric reservoir exactly, so
    the dissolved inventory survives and no budget goes negative. Loss beyond
    the atmosphere is unphysical, and the ledger booking equals the
    atmosphere's whole mass.
    """
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler(
        accretion=_impact_accretion(atmloss_module='constant', atmloss_frac=1.0)
    )
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    _atm_state(handler.hf_row, H=(4.0e20, 5.0e20), C=(2.0e19, 9.0e19))
    apply_impact(handler, _impact_event())

    # The dissolved part survives complete atmospheric stripping.
    assert handler.hf_row['H_kg_total'] == pytest.approx(1.0e20, rel=1e-9)
    assert handler.hf_row['C_kg_total'] == pytest.approx(7.0e19, rel=1e-9)
    assert handler.hf_row['H_kg_total'] >= 0.0
    assert handler.hf_row['C_kg_total'] >= 0.0
    assert handler.hf_row['esc_kg_cumulative'] == pytest.approx(4.2e20, rel=1e-9)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_stripping_a_sub_threshold_atmosphere_leaves_the_dissolved_inventory(monkeypatch):
    """An atmosphere below the outgassing mass threshold is not strippable.

    On a magma-ocean planet most volatiles are dissolved and the atmosphere can
    sit below ``outgas.mass_thresh`` (1e16 kg by default) while the totals are
    orders of magnitude larger. The strip must leave every whole-planet budget
    and the escaped-mass ledger untouched in that regime: the failure mode this
    pins is the totals being overwritten with the tiny atmospheric masses,
    which deletes the dissolved inventory and books it as escaped.
    """
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler(
        accretion=_impact_accretion(atmloss_module='constant', atmloss_frac=0.5)
    )
    # Production default threshold; the atmosphere sits well below it while the
    # dissolved reservoirs dominate the totals.
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e16)
    _atm_state(handler.hf_row, H=(1.0e15, 5.0e20), C=(5.0e14, 2.0e20))
    handler.hf_row['esc_kg_cumulative'] = 0.0
    apply_impact(handler, _impact_event())

    # The dissolved inventory survives, exactly.
    assert handler.hf_row['H_kg_total'] == pytest.approx(5.0e20, rel=1e-12)
    assert handler.hf_row['C_kg_total'] == pytest.approx(2.0e20, rel=1e-12)
    # Nothing is booked as escaped: the corrupted path would book ~7e20 kg.
    assert handler.hf_row['esc_kg_cumulative'] == pytest.approx(0.0, abs=1.0)
    # Discrimination: the failure mode leaves the totals at the atmospheric
    # masses, five orders of magnitude below the correct values.
    assert handler.hf_row['H_kg_total'] > 1.0e18


@pytest.mark.unit
def test_stripping_with_no_atmosphere_at_all_is_a_clean_no_op(monkeypatch):
    """Loss enabled on an airless planet strips nothing and books nothing.

    An impact can land before any outgassing has produced an atmosphere. With
    the loss module active the strip must pass through without touching the
    budgets, creating atmospheric keys, or moving the escaped-mass ledger.
    """
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler(
        accretion=_impact_accretion(atmloss_module='constant', atmloss_frac=0.9)
    )
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    handler.hf_row['H_kg_total'] = 3.0e20  # dissolved only; no _kg_atm keys exist
    apply_impact(handler, _impact_event())

    assert handler.hf_row['H_kg_total'] == pytest.approx(3.0e20, rel=1e-12)
    assert float(handler.hf_row.get('esc_kg_cumulative', 0.0)) == pytest.approx(0.0, abs=1.0)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_impact_strips_oxygen_with_the_other_atmospheric_elements(monkeypatch):
    """Atmospheric oxygen is stripped in proportion, like every other element.

    Under whole-planet oxygen accounting the atmosphere carries O (in H2O,
    CO2, SO2), so an impact that removes atmosphere removes O with it. The
    strip must debit O_kg_total by the loss fraction times the atmospheric O,
    or the O ledger would keep mass the atmosphere no longer holds.
    """
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler(
        accretion=_impact_accretion(atmloss_module='constant', atmloss_frac=0.4)
    )
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    _atm_state(handler.hf_row, H=(1.0e20, 3.0e20), O=(8.0e20, 1.2e21))
    apply_impact(handler, _impact_event())

    assert handler.hf_row['O_kg_total'] == pytest.approx(1.2e21 - 0.4 * 8.0e20, rel=1e-9)
    assert handler.hf_row['H_kg_total'] == pytest.approx(3.0e20 - 0.4 * 1.0e20, rel=1e-9)
    # O dominates the atmosphere 8:1, so the ledger booking is mostly O; a
    # partitioning that skipped O would book 4e19 instead of 3.6e20.
    assert handler.hf_row['esc_kg_cumulative'] == pytest.approx(
        0.4 * (8.0e20 + 1.0e20), rel=1e-9
    )


def _history(rows):
    """Build a minimal helpfile history DataFrame for the formation lookup."""
    import pandas as pd

    return pd.DataFrame(rows)


def _converging_solve_structure():
    """Mock of solve_structure faithful to the root-finder's convergence state.

    The real solve moves R_int until the whole-planet mass matches the target:
    at convergence ``M_planet = mass_tot * M_earth`` and the interior carries
    what the volatile budgets do not, ``M_int = M_planet - M_ele``. The mock
    reproduces exactly that end state (with the budgets it finds, mirroring
    the config-driven recompute), so a test can check how apply_impact's mass
    ledger and budget updates CLOSE into M_planet, which a no-op mock hides.
    """
    from proteus.utils.constants import M_earth, element_list

    def _mock(dirs, config, hf_all, hf_row, outdir):
        m_target = config.planet.mass_tot * M_earth
        m_ele = sum(float(hf_row.get(f'{e}_kg_total', 0.0)) for e in element_list)
        hf_row['M_int'] = m_target - m_ele
        hf_row['M_ele'] = m_ele
        hf_row['M_planet'] = m_target

    return _mock


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_impact_mass_closure_counts_each_volatile_channel_once(monkeypatch):
    """The planet's mass closes to before + rock + delivered - stripped.

    The interior anchor (mass_tot) and the volatile budgets (M_ele) are the
    two halves of M_planet, so each impact channel must land in exactly one
    of them: the impactor's rock grows the anchor, its delivered volatiles
    and the target strip move the budgets. Booking a channel in both halves
    double-counts it: growing the anchor by the full merger mass while also
    crediting the delivered content would inflate M_planet by the delivery,
    and subtracting the strip from the anchor while also debiting the
    budgets would remove it twice.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import M_earth

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure',
        _converging_solve_structure(),
    )

    m_planet_0 = 6.0e24
    handler = _impact_handler(
        mass_tot=m_planet_0 / M_earth,
        accretion=_impact_accretion(
            impactor_volatiles='match_planet',
            atmloss_module='constant',
            atmloss_frac=0.5,
        ),
    )
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    handler.hf_all = _history([{'Time': 0.0, 'M_planet': m_planet_0, 'H_kg_total': 4.0e22}])
    # Half the hydrogen is atmospheric: the mirror loses half the impactor's
    # content and the constant strip removes half the target atmosphere.
    _atm_state(handler.hf_row, H=(2.0e22, 4.0e22))
    handler.hf_row['M_planet'] = m_planet_0

    m_imp = 0.5 * M_earth
    event = _impact_event(
        M_target_before=m_planet_0,
        M_impactor=m_imp,
        M_merged_after=m_planet_0 + m_imp,
    )
    apply_impact(handler, event)

    content = (4.0e22 / m_planet_0) * m_imp
    # Half the content is exposed by the mirror and half of that is lost
    # with the collision, so three quarters arrive.
    delivered = (1.0 - 0.5 * 0.5) * content
    stripped = 0.5 * 2.0e22
    rock = m_imp - content

    # The final whole-planet mass counts each channel exactly once.
    m_ele_after = sum(v for k, v in handler.hf_row.items() if k.endswith('_kg_total'))
    m_planet_after = handler.hf_row['M_int'] + m_ele_after
    expected = m_planet_0 + rock + delivered - stripped
    assert m_planet_after == pytest.approx(expected, rel=1e-9)

    # Discrimination: both double-counting failure modes sit far outside
    # tolerance. Growing the anchor by the full merger mass over-counts the
    # delivery (~1e21 kg); also subtracting the strip from the anchor
    # under-counts it by another 1e22 kg.
    assert abs(m_planet_after - (expected + delivered)) > 0.5 * delivered
    assert abs(m_planet_after - (expected - stripped)) > 0.5 * stripped

    # The anchor itself grew by the impactor's rock alone.
    assert handler.config.planet.mass_tot == pytest.approx(
        (m_planet_0 + rock) / M_earth, rel=1e-12
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_match_planet_impactor_carries_the_formation_composition(monkeypatch):
    """A planet-matching impactor is scaled from the FORMATION state, not today.

    Every embryo co-formed from the same disk material, so the impactor
    carries the planet's t=0 fractional abundances scaled to its own mass.
    The planet here has since lost 90% of its hydrogen to escape; using the
    live abundance instead of the formation one would deliver ten times less.
    The formation row is the settled end of the init epoch (the last row
    before one year), not the raw first row.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import M_earth

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    m_planet_0 = 6.0e24
    x_h0 = 4.0e22 / m_planet_0  # formation H fraction
    x_n0 = 2.0e21 / m_planet_0
    handler = _impact_handler(accretion=_impact_accretion(impactor_volatiles='match_planet'))
    # Init epoch: an unsettled first row, then the settled formation row the
    # lookup must select; both precede the 1 yr discriminator.
    handler.hf_all = _history(
        [
            {'Time': 0.0, 'M_planet': m_planet_0, 'H_kg_total': 1.0e21, 'N_kg_total': 1.0e19},
            {'Time': 0.0, 'M_planet': m_planet_0, 'H_kg_total': 4.0e22, 'N_kg_total': 2.0e21},
            {'Time': 5.0e2, 'M_planet': m_planet_0, 'H_kg_total': 4.0e21, 'N_kg_total': 2.0e21},
        ]
    )
    # The planet TODAY holds only 10% of its formation hydrogen.
    handler.hf_row['H_kg_total'] = 4.0e21
    handler.hf_row['N_kg_total'] = 2.0e21
    m_imp = 0.5 * M_earth
    apply_impact(handler, _impact_event(M_impactor=m_imp))

    # Delivery reflects the formation fractions (loss disabled: full content).
    assert handler.hf_row['H_kg_total'] == pytest.approx(4.0e21 + x_h0 * m_imp, rel=1e-9)
    assert handler.hf_row['N_kg_total'] == pytest.approx(2.0e21 + x_n0 * m_imp, rel=1e-9)
    # Discrimination 1: the LIVE H abundance would deliver 10x less, a 1.8e22
    # kg difference, far outside tolerance.
    x_h_live = 4.0e21 / m_planet_0
    assert abs(x_h0 * m_imp - x_h_live * m_imp) > 1.0e22
    # Discrimination 2: the unsettled first init row would deliver 40x less H
    # than the settled formation row the lookup must pick.
    assert x_h0 * m_imp > 40 * (1.0e21 / m_planet_0) * m_imp * 0.99


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_match_planet_partition_mirror_and_fallback(monkeypatch):
    """The impactor's loss split mirrors the planet, per element, with fallback.

    With loss active, each element's atmospheric (lost) fraction is the
    planet's own at impact time: hydrogen here is half atmospheric, so half
    the impactor's hydrogen is lost; nitrogen is fully dissolved, so all its
    nitrogen arrives. An element the planet no longer holds cannot be
    mirrored per-element and falls back to the planet's bulk atmospheric
    fraction instead.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import M_earth

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    m_planet_0 = 6.0e24
    handler = _impact_handler(
        accretion=_impact_accretion(
            impactor_volatiles='match_planet', atmloss_module='constant', atmloss_frac=0.5
        )
    )
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    handler.hf_all = _history(
        [
            {
                'Time': 0.0,
                'M_planet': m_planet_0,
                'H_kg_total': 4.0e22,
                'N_kg_total': 2.0e21,
                'C_kg_total': 1.0e21,
            }
        ]
    )
    # Today: H half atmospheric, N fully dissolved, C fully escaped (no
    # budget left to mirror). Bulk atm fraction = 2e21/6e21 = 1/3. The
    # half-strength collision also strips half the target atmosphere, which
    # the H expectation below accounts for.
    _atm_state(handler.hf_row, H=(2.0e21, 4.0e21), N=(0.0, 2.0e21))
    handler.hf_row['C_kg_total'] = 0.0
    m_imp = 0.5 * M_earth
    apply_impact(handler, _impact_event(M_impactor=m_imp))

    h_content = (4.0e22 / m_planet_0) * m_imp
    n_content = (2.0e21 / m_planet_0) * m_imp
    c_content = (1.0e21 / m_planet_0) * m_imp
    # H: the target strip removes half its atmospheric hydrogen (1e21 kg),
    # and the impactor's content, half exposed by the mirror, loses half of
    # that exposed part, delivering three quarters.
    assert handler.hf_row['H_kg_total'] == pytest.approx(
        4.0e21 - 0.5 * 2.0e21 + (1.0 - 0.5 * 0.5) * h_content, rel=1e-9
    )
    # N: fully dissolved on the planet, so the impactor's N all arrives.
    assert handler.hf_row['N_kg_total'] == pytest.approx(2.0e21 + n_content, rel=1e-9)
    # C: fallback to the bulk atm fraction (1/3 exposed, half of that lost).
    assert handler.hf_row['C_kg_total'] == pytest.approx(
        (1.0 - (1.0 / 3.0) * 0.5) * c_content, rel=1e-9
    )
    # Discrimination: losing the whole exposed part (the fully-lost
    # convention) would land the C budget at 2/3 of the content, a sixth of
    # the content away, resolvable at these magnitudes.
    assert abs(handler.hf_row['C_kg_total'] - (2.0 / 3.0) * c_content) > 0.1 * c_content


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_a_small_impactor_stripping_a_heavy_atmosphere_shrinks_the_planet(monkeypatch):
    """The whole-planet mass falls when losses beat accretion.

    A small dry impactor that blows off a much heavier atmosphere leaves the
    planet lighter than before: the interior anchor still grows by the
    accreted rock, but the stripped budgets pull the whole-planet mass below
    its pre-impact value.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import M_earth

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure',
        _converging_solve_structure(),
    )

    m_planet_0 = 6.0e24
    handler = _impact_handler(
        mass_tot=m_planet_0 / M_earth,
        accretion=_impact_accretion(atmloss_module='constant', atmloss_frac=1.0),
    )
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    # Atmosphere of 2e23 kg; the impactor adds only 6.4e21 kg of rock.
    _atm_state(handler.hf_row, H=(2.0e23, 5.0e23))
    event = _impact_event(
        M_target_before=m_planet_0, M_impactor=6.4e21, M_merged_after=6.0064e24
    )
    apply_impact(handler, event)

    # The dry impactor's whole mass is rock: the anchor grows by all of it.
    assert handler.config.planet.mass_tot == pytest.approx(
        (m_planet_0 + 6.4e21) / M_earth, rel=1e-9
    )
    # The whole-planet mass shrank: rock in, a far heavier atmosphere out.
    m_ele_after = sum(v for k, v in handler.hf_row.items() if k.endswith('_kg_total'))
    m_planet_after = handler.hf_row['M_int'] + m_ele_after
    assert m_planet_after == pytest.approx(m_planet_0 + 6.4e21 - 2.0e23, rel=1e-9)
    assert m_planet_after < m_planet_0  # the planet got lighter
    assert handler.hf_row['H_kg_total'] == pytest.approx(3.0e23, rel=1e-9)


@pytest.mark.unit
def test_match_planet_without_history_fails_loudly():
    """Planet-matching impactors need a usable formation state to scale from.

    With no helpfile history the impactor composition is undefined, and a
    formation row without a positive planet mass cannot normalise the
    fractions; both must refuse with an actionable error rather than deliver
    zeros in silence.
    """
    from proteus.accretion.wrapper import _impactor_volatile_content

    cfg = SimpleNamespace(
        accretion=_impact_accretion(impactor_volatiles='match_planet'),
        planet=SimpleNamespace(),
    )
    with pytest.raises(RuntimeError, match='formation composition'):
        _impactor_volatile_content(cfg, None, _impact_event())

    # A degenerate formation row (no positive planet mass) is refused too.
    broken = _history([{'Time': 0.0, 'M_planet': 0.0, 'H_kg_total': 1.0e21}])
    with pytest.raises(RuntimeError, match='M_planet'):
        _impactor_volatile_content(cfg, broken, _impact_event())


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_two_sequential_impacts_compose_their_consequences(monkeypatch):
    """Each impact conserves and delivers against the state it finds.

    A Morrigan timeline routinely carries several impacts. The second impact
    must act on the post-first-impact budgets: conservation brackets its own
    structure solve (proven against a rescaling solve both times) and the
    delivery adds its own impactor's content on top of the first's. With
    loss disabled the full content arrives and the planet grows by the full
    merger mass each time.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import M_earth

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure',
        _rescaling_solve_structure(1.2),
    )

    handler = _impact_handler(
        mass_tot=1.0,
        accretion=_impact_accretion(H=1000.0),  # ppmw mode, loss off
    )
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    _atm_state(handler.hf_row, H=(2.0e20, 6.0e20))
    m_imp = 0.2 * M_earth
    event = _impact_event(
        M_target_before=6.0 * M_earth,
        M_impactor=m_imp,
        M_merged_after=6.2 * M_earth,
    )

    apply_impact(handler, event)
    delivered = m_imp * 1000.0 / 1.0e6
    after_first = 6.0e20 + delivered
    assert handler.hf_row['H_kg_total'] == pytest.approx(after_first, rel=1e-9)

    # Second impact: the conservation bracket must defeat the rescaling solve
    # again, starting from the grown budget, and the delivery adds once more.
    apply_impact(handler, event)
    after_second = after_first + delivered
    assert handler.hf_row['H_kg_total'] == pytest.approx(after_second, rel=1e-9)
    # Discrimination: an unbracketed second solve would carry a 1.2x rescale
    # of after_first, over 1e20 kg above the correct composition.
    assert abs(handler.hf_row['H_kg_total'] - (1.2 * after_first + delivered)) > 1.0e20
    # The anchor grew by each impactor's rock (merger mass minus content);
    # the delivered volatiles reach the planet through the budgets instead.
    expected_mass = 1.0 + 2 * (event.mass_delta - delivered) / M_earth
    assert handler.config.planet.mass_tot == pytest.approx(expected_mass, rel=1e-12)
    assert float(handler.hf_row.get('esc_kg_cumulative', 0.0)) == pytest.approx(0.0, abs=1.0)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_impact_loss_composes_with_delivery_and_a_broken_provider_raises(monkeypatch):
    """With loss active, one collision fraction governs both bodies.

    One impact carries three volatile channels: the shock strips the loss
    fraction of the target's atmosphere, the impactor's atmospheric part
    (mirrored from the planet, here exactly one third) loses the same
    fraction, and everything else is delivered. The interior anchor grows
    by the impactor's rock alone. A loss module returning a fraction
    outside [0, 1] violates the partitioning contract and must raise rather
    than be clamped in silence.
    """
    from proteus.accretion.wrapper import _impact_loss_fraction, apply_impact
    from proteus.utils.constants import M_earth

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler(
        mass_tot=1.0,
        accretion=_impact_accretion(atmloss_module='constant', atmloss_frac=0.5, H=1000.0),
    )
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    # One third of the planet's hydrogen sits in the atmosphere: the mirror
    # then declares one third of the impactor's content atmospheric (lost)
    # and delivers the remaining two thirds.
    _atm_state(handler.hf_row, H=(2.0e20, 6.0e20))
    m_impactor = 0.5 * M_earth
    event = _impact_event(M_impactor=m_impactor)
    mass_delta = event.mass_delta
    apply_impact(handler, event)

    content = m_impactor * 1000.0 / 1.0e6
    stripped = 0.5 * 2.0e20
    # A third of the content is exposed by the mirror and half of that is
    # lost with the collision, so five sixths arrive.
    delivered = content * (1.0 - (1.0 / 3.0) * 0.5)
    expected = 6.0e20 - stripped + delivered
    assert handler.hf_row['H_kg_total'] == pytest.approx(expected, rel=1e-9)
    assert handler.hf_row['M_ele'] == pytest.approx(expected, rel=1e-9)
    # Discrimination: both neighbouring conventions sit far outside
    # tolerance, full delivery by half a sixth of the content (~5e20 kg)
    # and a fully-lost exposed part by a further sixth.
    assert abs(handler.hf_row['H_kg_total'] - (6.0e20 - stripped + content)) > 4.0e20
    assert (
        abs(handler.hf_row['H_kg_total'] - (6.0e20 - stripped + content * 2.0 / 3.0)) > 4.0e20
    )
    # Only the target's stripped mass enters the planet's escape ledger; the
    # impactor's lost volatiles never belonged to the planet's inventory.
    assert handler.hf_row['esc_kg_cumulative'] == pytest.approx(stripped, rel=1e-9)

    # The interior anchor grew by the impactor's rock alone; the delivered
    # and stripped volatiles reach the whole-planet mass through the budgets.
    expected_mass = 1.0 + (mass_delta - content) / M_earth
    assert handler.config.planet.mass_tot == pytest.approx(expected_mass, rel=1e-12)
    # Discrimination: growing the anchor by the full merger mass would put
    # the delivered content into the interior AND the budgets, resolvable
    # far above the tolerance.
    assert abs(handler.config.planet.mass_tot - (1.0 + mass_delta / M_earth)) > 1e-5

    # A provider outside the contract is rejected loudly.
    bad = _impact_handler(
        accretion=_impact_accretion(atmloss_module='constant', atmloss_frac=1.5)
    )
    with pytest.raises(ValueError, match=r'\[0, 1\]'):
        _impact_loss_fraction(bad.config, bad.hf_row, _impact_event())


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_zephyrus_loss_module_evaluates_the_kegerreis_law(monkeypatch):
    """The zephyrus module turns the impact record into the erosion fraction.

    For two identical Earth-like bodies colliding head-on at their mutual
    escape speed, Eqn. 1 of Kegerreis et al. (2020), ApJL 901, L31 collapses
    to X = 0.64 * 0.5**0.325 = 0.510911, so the dispatch is pinned against
    the published closed form through the real ZEPHYRUS implementation. The
    twin pin cannot see the target/impactor mapping (every ratio is
    symmetric there), so two asymmetric follow-up events pin the fraction
    on BOTH sides of the mass assignment to their absolute values: a
    dispatch that swapped the target and impactor masses would return
    0.526 where 0.267 is pinned and the reverse, failing both. (Radii
    cannot discriminate here: at equal densities the interacting mass and
    the mutual escape speed are both symmetric under a radius swap.)
    """
    import numpy as np

    pytest.importorskip('zephyrus.collision')
    from proteus.accretion.wrapper import _impact_loss_fraction

    m_e, r_e = 5.972e24, 6.371e6
    rho_e = m_e / (4.0 / 3.0 * np.pi * r_e**3)
    v_esc = np.sqrt(2.0 * 6.6743e-11 * 2.0 * m_e / (2.0 * r_e))
    cfg = SimpleNamespace(
        accretion=_impact_accretion(atmloss_module='zephyrus'),
    )
    twins = _impact_event(
        M_target_before=m_e,
        M_impactor=m_e,
        M_merged_after=2.0 * m_e,
        v_impact=v_esc,
        v_esc=v_esc,
        impact_parameter=0.0,
        R_target_before=r_e,
        R_impactor=r_e,
        rho_target=rho_e,
        rho_impactor=rho_e,
    )
    hf_row = {'M_planet': 6.3e24, 'H_kg_atm': 1.0e22}

    f = _impact_loss_fraction(cfg, hf_row, twins)
    assert f == pytest.approx(0.510911, rel=1e-4)
    assert 0.0 < f < 1.0

    # Asymmetric event: a half-radius impactor at one eighth the mass. The
    # mass-ratio term is the only tie-breaker, so pinning the fraction on
    # both sides of the mass assignment fixes the dispatch's mapping.
    r_i = 0.5 * r_e
    m_i = rho_e * 4.0 / 3.0 * np.pi * r_i**3
    asym = _impact_event(
        M_target_before=m_e,
        M_impactor=m_i,
        M_merged_after=m_e + m_i,
        v_impact=v_esc,
        impact_parameter=0.3,
        R_target_before=r_e,
        R_impactor=r_i,
        rho_target=rho_e,
        rho_impactor=rho_e,
    )
    f_asym = _impact_loss_fraction(cfg, hf_row, asym)
    swapped = _impact_event(
        M_target_before=m_i,
        M_impactor=m_e,
        M_merged_after=m_e + m_i,
        v_impact=v_esc,
        impact_parameter=0.3,
        R_target_before=r_e,
        R_impactor=r_i,
        rho_target=rho_e,
        rho_impactor=rho_e,
    )
    f_swapped = _impact_loss_fraction(cfg, hf_row, swapped)
    # Absolute pins on both sides of the mass assignment: a dispatch with
    # the target and impactor masses interchanged returns these two values
    # permuted, failing both pins, where a difference-only check would
    # survive the permutation unchanged.
    assert f_asym == pytest.approx(0.2675, rel=2e-3)
    assert f_swapped == pytest.approx(0.5258, rel=2e-3)
    assert f_asym < f_swapped  # the lighter impactor erodes less


@pytest.mark.unit
def test_zephyrus_loss_module_warns_outside_the_thin_atmosphere_regime(caplog):
    """A thick atmosphere triggers the fitted-domain warning, a thin one not.

    The erosion law is fitted for atmospheres of order 1 percent of the
    planet mass. The dispatch warns when the live atmosphere fraction is
    beyond a few percent, and stays quiet inside the regime, so a
    volatile-rich run cannot silently consume extrapolated fractions. The
    fraction is still returned in both cases.
    """
    import numpy as np

    pytest.importorskip('zephyrus.collision')
    from proteus.accretion.wrapper import _impact_loss_fraction

    m_e, r_e = 5.972e24, 6.371e6
    rho_e = m_e / (4.0 / 3.0 * np.pi * r_e**3)
    cfg = SimpleNamespace(accretion=_impact_accretion(atmloss_module='zephyrus'))
    event = _impact_event(
        M_target_before=m_e,
        M_impactor=m_e,
        M_merged_after=2.0 * m_e,
        v_impact=1.2e4,
        R_target_before=r_e,
        R_impactor=r_e,
        rho_target=rho_e,
        rho_impactor=rho_e,
    )

    # Just above the 3% threshold: the warning fires. Straddling the
    # boundary pins the cutoff itself, not merely the warning's existence.
    thick = {'M_planet': 6.0e24, 'H_kg_atm': 0.031 * 6.0e24}
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        f_thick = _impact_loss_fraction(cfg, thick, event)
    assert 0.0 <= f_thick <= 1.0
    assert 'thin-atmosphere regime' in '\n'.join(r.getMessage() for r in caplog.records)

    # Just below the threshold: no warning.
    caplog.clear()
    thin = {'M_planet': 6.0e24, 'H_kg_atm': 0.029 * 6.0e24}
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        f_thin = _impact_loss_fraction(cfg, thin, event)
    assert 0.0 <= f_thin <= 1.0
    assert 'thin-atmosphere regime' not in '\n'.join(r.getMessage() for r in caplog.records)


@pytest.mark.unit
def test_zephyrus_loss_module_without_the_law_fails_loudly(monkeypatch):
    """A fwl-zephyrus lacking the collision law is an actionable error.

    The zephyrus loss module needs zephyrus.collision; an installation
    predating it must produce an upgrade instruction at the first impact,
    not an AttributeError from deep inside the dispatch.
    """
    import sys

    from proteus.accretion.wrapper import _impact_loss_fraction

    cfg = SimpleNamespace(accretion=_impact_accretion(atmloss_module='zephyrus'))
    monkeypatch.setitem(sys.modules, 'zephyrus.collision', None)
    with pytest.raises(ImportError, match='fwl-zephyrus') as excinfo:
        _impact_loss_fraction(cfg, {'M_planet': 6.0e24}, _impact_event())

    # The message names the setting that asked for it, the module that is
    # absent, and the action that fixes it, so it can be acted on without
    # reading the dispatch.
    message = str(excinfo.value)
    assert 'atmloss_module' in message
    assert 'zephyrus.collision' in message
    assert 'upgrade' in message

    # With no loss module configured the same call is silent and loses
    # nothing, so the error is specific to the selected module rather than
    # raised on every impact.
    off = SimpleNamespace(accretion=_impact_accretion(atmloss_module=None))
    assert _impact_loss_fraction(off, {'M_planet': 6.0e24}, _impact_event()) == 0.0


def _rescaling_solve_structure(factor):
    """Mock of solve_structure that rescales the volatile budgets by ``factor``.

    The real structure solve calls calc_target_elemental_inventories, which for
    ppmw-mode budgets recomputes ``<e>_kg_total`` against the grown reservoir
    mass, so a mass-growth impact multiplies every volatile budget by roughly
    the mass-growth ratio and rewrites ``M_ele`` to match. This stand-in
    reproduces that mass-scaling so the conservation contract can be exercised
    without a live solver: a passing test must show the budgets are conserved
    against exactly this rescaling, not merely left untouched by a no-op mock.
    """

    def _mock(dirs, config, hf_all, hf_row, outdir):
        for key in list(hf_row):
            if key.endswith('_kg_total'):
                hf_row[key] *= factor
        hf_row['M_ele'] = sum(v for k, v in hf_row.items() if k.endswith('_kg_total'))

    return _mock


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_mass_growth_conserves_volatiles_a_dry_impactor_creates_none(monkeypatch):
    """Growing the planet with a dry impactor conserves the volatile budgets.

    The mass growth adds rock, not volatiles: a rock-dominated dry impactor
    cannot manufacture hydrogen. The structure re-solve rescales the ppmw
    budgets against the grown mass, so without conservation a dry impact would
    inflate H, C, N, S in lockstep with the added mass. The impact must leave
    every volatile budget at its pre-impact value.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import M_earth

    # A 0.5 Earth-mass impactor on a 1.0 Earth-mass planet grows the reservoir
    # by 1.5x, the factor by which the structure solve would rescale the ppmw
    # budgets. Dry impactor: no delivery.
    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure',
        _rescaling_solve_structure(1.5),
    )

    handler = _impact_handler(mass_tot=1.0)
    handler.hf_row['H_kg_total'] = 4.0e22
    handler.hf_row['C_kg_total'] = 1.0e21
    handler.hf_row['O_kg_total'] = 8.0e22
    event = _impact_event(
        M_target_before=6.0 * M_earth,
        M_impactor=0.5 * M_earth,
        M_merged_after=6.5 * M_earth,
    )
    apply_impact(handler, event)

    # Every volatile budget is conserved at its pre-impact value.
    assert handler.hf_row['H_kg_total'] == pytest.approx(4.0e22, rel=1e-12)
    assert handler.hf_row['C_kg_total'] == pytest.approx(1.0e21, rel=1e-12)
    assert handler.hf_row['O_kg_total'] == pytest.approx(8.0e22, rel=1e-12)
    # Discrimination: the mass-scaled (unconserved) value is 1.5x larger, a 50%
    # divergence far outside the 1e-12 tolerance. This is the value the row
    # would carry if the restore were absent.
    assert abs(handler.hf_row['H_kg_total'] - 4.0e22 * 1.5) > 1.0e22
    # M_ele reflects the conserved inventory, not the rescaled one.
    assert handler.hf_row['M_ele'] == pytest.approx(4.0e22 + 1.0e21 + 8.0e22, rel=1e-12)
    assert handler.hf_row['M_ele'] < 1.5 * (4.0e22 + 1.0e21 + 8.0e22)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_mass_growth_conserves_then_delivery_adds_only_the_delivered_mass(monkeypatch):
    """Under mass growth the budget is the conserved base plus the delivery.

    With a wet impactor the two mechanisms compose: the mass growth conserves
    the pre-impact inventory (it does not rescale it), and the delivery adds
    exactly the impactor mass times its ppmw content on top. The final budget
    must be base + delivered, never the mass-scaled base or the mass-scaled
    base plus the delivery.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import M_earth

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure',
        _rescaling_solve_structure(1.5),
    )

    handler = _impact_handler(mass_tot=1.0, accretion=_impact_accretion(H=1000.0))
    handler.hf_row['H_kg_total'] = 4.0e22
    m_impactor = 0.5 * M_earth
    event = _impact_event(
        M_target_before=6.0 * M_earth,
        M_impactor=m_impactor,
        M_merged_after=6.5 * M_earth,
    )
    apply_impact(handler, event)

    delivered = m_impactor * 1000.0 / 1.0e6
    expected = 4.0e22 + delivered  # conserved base + delivery
    assert handler.hf_row['H_kg_total'] == pytest.approx(expected, rel=1e-12)
    # Discrimination against the two wrong compositions: rescaled base (+50%)
    # and rescaled base plus delivery both exceed the correct value by the
    # 2.0e22 mass-scaling term, far outside tolerance.
    assert abs(handler.hf_row['H_kg_total'] - (4.0e22 * 1.5)) > 1.0e22
    assert abs(handler.hf_row['H_kg_total'] - (4.0e22 * 1.5 + delivered)) > 1.0e22
    assert handler.hf_row['M_ele'] == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_strip_never_reaches_the_dissolved_inventory(monkeypatch):
    """A partial strip removes atmosphere only, whatever the threshold does.

    The collision reaches the atmosphere, not the mantle, so an element's
    dissolved inventory must survive the impact even when the post-strip total
    lands below the outgassing mass threshold. The continuous-escape path
    treats an element that falls under that threshold as fully depleted and
    zeroes its whole-planet total, a reasonable convention for an element
    ground down over many steps but wrong for one collision: it would delete
    dissolved mass the impact never touched and book it as lost to space.

    The threshold here is set so exactly that trap is sprung: H holds 1.0e16 kg
    in the atmosphere and 0.2e16 kg dissolved, and stripping half the
    atmosphere leaves 0.7e16 kg, below the 1.0e16 kg threshold.
    """
    from proteus.accretion.wrapper import _target_strip_amounts

    config = SimpleNamespace(outgas=SimpleNamespace(mass_thresh=1.0e16))
    hf_row = {}
    _atm_state(hf_row, H=(1.0e16, 1.2e16))

    strip = _target_strip_amounts(config, hf_row, f_loss=0.5)

    # Exactly half the atmospheric mass, and not one kilogram of the 0.2e16 kg
    # that is dissolved in the mantle.
    assert strip['H'] == pytest.approx(0.5e16, rel=1e-12)
    assert strip['H'] < hf_row['H_kg_total']

    # Discrimination: routing this through the desiccation floor would remove
    # the whole 1.2e16 kg budget, which is 2.4x the correct debit.
    assert abs(1.2e16 - 0.5e16) > 0.5 * 0.5e16

    # The strip can never exceed the atmosphere it is drawn from, at any loss
    # fraction including a total one.
    total_loss = _target_strip_amounts(config, hf_row, f_loss=1.0)
    assert total_loss['H'] == pytest.approx(1.0e16, rel=1e-12)
    assert total_loss['H'] <= hf_row['H_kg_atm']


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_impact_leaves_the_planet_mass_consistent_with_its_parts(monkeypatch):
    """M_planet equals M_int + M_ele when apply_impact returns.

    Escape runs later in the same iteration and reads M_planet, so leaving it
    at the value the structure solve wrote, before the strip and the delivery
    changed the volatile budgets, would size that iteration's escape against a
    planet that does not exist. The structure solve is mocked to write a
    deliberately stale M_planet, so a handler that failed to refresh it would
    keep that value and fail here.
    """
    from proteus.accretion.wrapper import apply_impact

    handler = _impact_handler(
        accretion=_impact_accretion(impactor_volatiles='ppmw', H_ppmw=1000.0)
    )
    _atm_state(handler.hf_row, H=(2.0e20, 5.0e20))
    handler.hf_row['M_ele'] = 5.0e20
    handler.hf_row['M_planet'] = 0.0  # stale sentinel; must not survive

    def _solve(dirs, config, hf_all, hf_row, output):
        hf_row['M_int'] = config.planet.mass_tot * 5.9736e24
        # Write the inconsistent pair a real structure solve would leave.
        hf_row['M_ele'] = 9.9e21
        hf_row['M_planet'] = hf_row['M_int'] + 9.9e21

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', _solve, raising=False
    )
    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.remelt_mantle', lambda *a, **k: None, raising=False
    )

    apply_impact(handler, _impact_event())

    hf_row = handler.hf_row
    assert hf_row['M_planet'] == pytest.approx(hf_row['M_int'] + hf_row['M_ele'], rel=1e-12)
    # The stale value the solve wrote is gone, so the refresh genuinely ran.
    assert hf_row['M_ele'] != pytest.approx(9.9e21, rel=1e-9)


@pytest.mark.unit
def test_a_resumed_run_rebuilds_the_mass_and_orbit_the_impacts_moved():
    """Growth applied before a resume point is restored, not discarded.

    The configuration is the run's specification and is rebuilt from file on
    every start, so the mass and orbit that impacts moved live only in the
    helpfile. Without the restore a resumed run would solve the structure
    against the planet's original mass, throwing away every pre-resume impact,
    and would snap the orbit back to its configured value on the first step.

    The rock ledger is the discriminating input: restoring from M_planet
    instead would fold the volatile budgets into the rock anchor, which this
    row makes visible by carrying a volatile mass far larger than the rounding
    of the rock itself.
    """
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU, M_earth

    handler = SimpleNamespace(
        config=SimpleNamespace(
            accretion=SimpleNamespace(module='morrigan'),
            params=SimpleNamespace(resume=True),
            planet=SimpleNamespace(mass_tot=1.0),
            orbit=SimpleNamespace(semimajoraxis=1.0, eccentricity=0.0),
        ),
        hf_row={
            'M_accreted_rock': 0.5 * M_earth,
            'M_planet': 2.5 * M_earth,  # carries volatiles too; must NOT be used
            'semimajorax': 1.25 * AU,
            'eccentricity': 0.04,
        },
    )

    restore_accretion_state(handler)

    assert handler.config.planet.mass_tot == pytest.approx(1.5, rel=1e-12)
    assert handler.config.orbit.semimajoraxis == pytest.approx(1.25, rel=1e-12)
    assert handler.config.orbit.eccentricity == pytest.approx(0.04, rel=1e-12)

    # Discrimination: anchoring on M_planet would have given 2.5 M_earth, which
    # differs from the correct 1.5 by two thirds of the correct value.
    assert abs(2.5 - 1.5) > 0.5 * 1.5


@pytest.mark.unit
def test_the_accretion_restore_is_inert_outside_a_resume():
    """A fresh run, a disabled module, and an impact-free resume change nothing.

    The restore adds accreted rock on top of the configured mass, so running it
    when the configuration already describes the current planet would double
    the growth. It must therefore be a strict no-op unless the run is a resume
    that has actually accreted something.

    Turning the module off is deliberately NOT one of those conditions.
    Continuing a run whose impacts are finished by setting the module to none is
    a reasonable thing to do, and the planet must keep the mass it accreted: the
    ledger records what happened, whatever the module is set to now.
    """
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import M_earth

    def _handler_for(resume, module, accreted):
        return SimpleNamespace(
            config=SimpleNamespace(
                accretion=SimpleNamespace(module=module),
                params=SimpleNamespace(resume=resume),
                planet=SimpleNamespace(mass_tot=1.0),
                orbit=SimpleNamespace(semimajoraxis=1.0, eccentricity=0.0),
            ),
            hf_row={'M_accreted_rock': accreted, 'semimajorax': 9.9e11, 'eccentricity': 0.9},
        )

    for resume, module, accreted in (
        (False, 'morrigan', 0.5 * M_earth),  # fresh run
        (True, 'morrigan', 0.0),  # resumed before any impact landed
    ):
        handler = _handler_for(resume, module, accreted)
        restore_accretion_state(handler)
        assert handler.config.planet.mass_tot == pytest.approx(1.0, rel=1e-12)
        assert handler.config.orbit.semimajoraxis == pytest.approx(1.0, rel=1e-12)
        assert handler.config.orbit.eccentricity == pytest.approx(0.0, rel=1e-12)

    # Accretion switched off after the impacts finished: the growth survives,
    # because the ledger and not the module setting is what records it.
    switched_off = _handler_for(True, None, 0.5 * M_earth)
    restore_accretion_state(switched_off)
    assert switched_off.config.planet.mass_tot == pytest.approx(1.5, rel=1e-12)


@pytest.mark.unit
def test_a_resumed_run_replays_the_timeline_the_first_session_resolved(tmp_path):
    """The impact history is a property of the run, not of model determinism.

    Re-deriving the timeline on resume would reproduce the original history
    only if the dynamical model is bit-reproducible at a fixed seed, which
    PROTEUS cannot check. The first session therefore records what it resolved
    and a resume reads that file back. The recorded file is authoritative: this
    test makes the module raise if it is consulted at all on the resume, so a
    fallback to re-deriving would fail rather than pass by coincidence.
    """
    handler = _handler(
        module='timeline',
        timeline_path=_timeline_file(tmp_path / 't.csv'),
        output_dir=tmp_path,
    )
    first = init_accretion(handler)
    assert (tmp_path / 'impact_timeline.csv').exists()

    resumed = _handler(
        module='timeline',
        timeline_path=tmp_path / 'absent.csv',  # would raise if consulted
        output_dir=tmp_path,
        resume=True,
    )
    replayed = init_accretion(resumed)

    assert [e.time for e in replayed] == [e.time for e in first]
    assert [e.M_impactor for e in replayed] == pytest.approx(
        [e.M_impactor for e in first], rel=1e-12
    )


@pytest.mark.unit
def test_the_recorded_timeline_is_not_offset_a_second_time(tmp_path):
    """Times are written on the PROTEUS axis and read back without the offset.

    The recorded file already carries the configured offset, so re-applying it
    on resume would move every impact by that amount again. A non-zero offset
    makes the double application unmissable: it would double the shift.
    """
    offset = 3.0e5
    handler = _handler(
        module='timeline',
        timeline_path=_timeline_file(tmp_path / 't.csv'),
        time_offset=offset,
        output_dir=tmp_path,
    )
    first = init_accretion(handler)
    assert first[0].time == pytest.approx(1.0e5 + offset)

    resumed = _handler(
        module='timeline',
        timeline_path=tmp_path / 'absent.csv',
        time_offset=offset,
        output_dir=tmp_path,
        resume=True,
    )
    replayed = init_accretion(resumed)

    assert replayed[0].time == pytest.approx(1.0e5 + offset)
    # Discrimination: a second application would put it at 1.0e5 + 2 * offset.
    assert abs((1.0e5 + 2 * offset) - replayed[0].time) > 0.5 * offset


@pytest.mark.unit
def test_a_temperature_mode_without_a_molten_guarantee_is_flagged(tmp_path, caplog):
    """Only liquidus_super suppresses the re-melt advisory on Aragog.

    Each impact re-melts the mantle by re-applying the run's temperature-mode
    initial condition, and only liquidus_super is molten for any planet mass
    and melting curve. The modes that merely tend to be molten, and are often
    chosen for exactly that reason, must still draw the advisory: treating them
    as guarantees is what lets a run apply an impact that melts nothing and
    report it as a re-melt.
    """
    path = _timeline_file(tmp_path / 't.csv')

    for mode in ('adiabatic_from_cmb', 'accretion', 'isothermal'):
        caplog.clear()
        handler = _handler(
            module='timeline',
            timeline_path=path,
            output_dir=tmp_path,
            interior_module='aragog',
            temperature_mode=mode,
        )
        with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
            init_accretion(handler)
        assert 'not guaranteed' in caplog.text, f'{mode} must draw the advisory'
        assert mode in caplog.text

    # The one mode that does guarantee it stays quiet, so the advisory
    # discriminates rather than firing for everything.
    caplog.clear()
    handler = _handler(
        module='timeline',
        timeline_path=path,
        output_dir=tmp_path,
        interior_module='aragog',
        temperature_mode='liquidus_super',
    )
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        init_accretion(handler)
    assert 'not guaranteed' not in caplog.text

    # A scalar interior re-melts by resetting a temperature, so the advisory
    # about the entropy initial condition does not apply to it at all.
    caplog.clear()
    handler = _handler(
        module='timeline',
        timeline_path=path,
        output_dir=tmp_path,
        interior_module='dummy',
        temperature_mode='isothermal',
    )
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        init_accretion(handler)
    assert 'not guaranteed' not in caplog.text


@pytest.mark.unit
def test_a_resumed_run_does_not_advise_changing_the_time_offset(tmp_path, caplog):
    """Impacts before a resume point were applied, and are reported as such.

    The same filter serves opposite purposes on the two paths. On a fresh run an
    impact before the start cannot be applied and the offset is the fix. On a
    resume the identical impacts were already applied and their mass is restored
    from the ledger, so repeating the fresh-run advice would tell a user to
    bring them back and accrete them a second time.
    """
    path = _timeline_file(tmp_path / 't.csv')

    # Fresh run starting after the first impact: the advice is correct there.
    fresh = _handler(
        module='timeline', timeline_path=path, time_start=2.0e5, output_dir=tmp_path
    )
    with caplog.at_level(logging.INFO, logger='fwl.proteus.accretion.wrapper'):
        init_accretion(fresh)
    assert 'time_offset' in caplog.text
    assert 'will not be applied' in caplog.text

    # Resume past the first impact: same drop, opposite meaning.
    caplog.clear()
    resumed = _handler(
        module='timeline',
        timeline_path=path,
        time_start=2.0e5,
        output_dir=tmp_path,
        resume=True,
    )
    with caplog.at_level(logging.INFO, logger='fwl.proteus.accretion.wrapper'):
        events = init_accretion(resumed)

    assert 'time_offset' not in caplog.text
    assert 'already carrying' in caplog.text
    # The surviving schedule is the same either way; only the report differs.
    assert [e.time for e in events] == [5.0e5]


@pytest.mark.unit
def test_the_impact_eccentricity_is_clamped_to_a_bound_orbit(monkeypatch, caplog):
    """An impact cannot drive the planet onto an open orbit, and says when it tries.

    The applied change is a difference, so a large positive one on an already
    eccentric planet can ask for an eccentricity at or above unity, which the
    rest of the model cannot represent: the separation, periapsis and Hill radius
    all assume a closed orbit. The result is clamped, and the clamp reports
    itself, because absorbing it in silence is how a compounding drift in the
    applied change would hide for a whole run.
    """
    from proteus.accretion.wrapper import _ECC_MAX, apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler(semimajoraxis=1.0, eccentricity=0.9)
    # The followed body is excited from 0.01 to 0.8, a change of +0.79, which
    # would take a planet at 0.9 to 1.69.
    event = _impact_event(a_before=1.0e11, a_after=1.0e11, e_before=0.01, e_after=0.8)

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        apply_impact(handler, event)

    assert handler.config.orbit.eccentricity == pytest.approx(_ECC_MAX, rel=1e-12)
    assert handler.hf_row['eccentricity'] == pytest.approx(_ECC_MAX, rel=1e-12)
    assert 0.0 <= handler.config.orbit.eccentricity < 1.0
    assert 'clamped' in caplog.text

    # Discrimination: unclamped the orbit would be reported at 1.69, which is not
    # an orbit at all, and every quantity derived from it would be nonsense.
    assert 0.9 + 0.79 > 1.0

    # A change that stays inside the range passes through untouched and silent.
    caplog.clear()
    quiet = _impact_handler(semimajoraxis=1.0, eccentricity=0.1)
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        apply_impact(
            quiet, _impact_event(a_before=1.0e11, a_after=1.0e11, e_before=0.01, e_after=0.05)
        )
    assert quiet.config.orbit.eccentricity == pytest.approx(0.14, rel=1e-12)
    assert 'clamped' not in caplog.text


@pytest.mark.unit
def test_discard_preimpact_snapshot_drops_only_the_impact_steps_own_snapshot(tmp_path, caplog):
    """A step that both wrote a snapshot and landed an impact discards it.

    Physical scenario: the interior writes its snapshot while the step is
    solved, which is before the impacts falling in that step are applied at
    the end of it. On such a step the snapshot holds the mantle from before
    the re-melt while the helpfile row it shares a time with already carries
    the impact's mass, orbit and volatile budgets. Resuming from that pair
    would restore a mantle the impact had melted while treating the impact as
    already applied, silently losing the re-melt.

    Contract clause: the stale snapshot is removed so the resume walks back to
    the previous complete pair and applies the impact again in full.

    Verifies:
    - The impact step's snapshot is removed for the interior that writes one.
    - The previous step's snapshot survives, so the resume has a pair to land
      on rather than being left with none.
    - An interior that writes no snapshot leaves the directory untouched, so
      the discard cannot delete another writer's file.
    - A step with no snapshot on disk is a no-op rather than an error.
    - The last remaining snapshot is kept and reported, because removing it
      would leave the run with no interior state to resume from at all.
    """
    from proteus.accretion.wrapper import discard_preimpact_snapshot

    def _handler(module, time=300.0):
        return SimpleNamespace(
            config=SimpleNamespace(interior_energetics=SimpleNamespace(module=module)),
            directories={'output': str(tmp_path)},
            hf_row={'Time': time},
        )

    data = tmp_path / 'data'
    data.mkdir()
    (data / '300_int.nc').write_text('pre-remelt')
    (data / '200_int.nc').write_text('previous')

    discard_preimpact_snapshot(_handler('aragog'))
    assert not (data / '300_int.nc').exists(), (
        'the impact step kept its pre-remelt snapshot, so a resume would load '
        'a mantle the impact had already melted'
    )
    assert (data / '200_int.nc').read_text() == 'previous', (
        'the previous complete snapshot was removed too, leaving the resume '
        'with nothing to walk back to'
    )

    # The scalar interiors carry their state in the helpfile row, which is
    # already post-impact, so they must not have files removed under them.
    (data / '300_int.nc').write_text('not mine to delete')
    for module in ('dummy', 'boundary', 'spider'):
        discard_preimpact_snapshot(_handler(module))
        assert (data / '300_int.nc').read_text() == 'not mine to delete', (
            f"the '{module}' interior discarded a snapshot it does not write"
        )

    # A step that wrote no snapshot is the ordinary case, not an error.
    discard_preimpact_snapshot(_handler('aragog', time=999.0))

    # The last snapshot is kept: discarding it would leave nothing for the
    # resume to land on, so the inconsistency is reported instead of the run
    # being stripped of its only interior state.
    for stale in data.glob('*_int.nc'):
        stale.unlink()
    (data / '300_int.nc').write_text('only one left')

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        discard_preimpact_snapshot(_handler('aragog'))

    assert (data / '300_int.nc').exists(), (
        'the only interior snapshot was discarded, so the run has no state to '
        'resume from and no interior history at its endpoint'
    )
    assert 'only one' in caplog.text, (
        'the kept snapshot predates the re-melt, so staying silent would hide '
        'an inconsistent resume'
    )
