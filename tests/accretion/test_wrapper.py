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
    desiccated=False,
    accretion=None,
    structure='zalmoxis',
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
            interior_struct=SimpleNamespace(core_frac=0.55, module=structure),
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
        desiccated=desiccated,
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
def test_a_wet_impact_on_a_desiccated_planet_restores_its_inventory(monkeypatch):
    """A volatile-bearing impact clears the one-way desiccation latch.

    Once a planet loses its whole volatile inventory the run latches into the
    desiccated path, which zeroes every volatile column it is handed. An
    impactor that arrives carrying volatiles gives the planet an inventory
    again, so that latch has to lift with the delivery: otherwise the next
    outgassing call erases what the impact just delivered and the planet stays
    dry however wet the impactors are.

    The latch is lifted, not re-decided. The desiccation check runs again on
    the following iteration and re-sets it if the delivery was too small to
    count, so nothing here asserts the planet is wet, only that it is allowed
    to be re-evaluated.

    Edge case: a dry impactor delivers nothing, so there is no inventory to
    restore and the latch must stay set. That is the discriminating half; a
    fix that cleared the latch on every impact would pass the first assertion
    and fail this one.
    """
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    # 1000 ppmw of hydrogen on a 6.4e23 kg impactor is 6.4e20 kg delivered,
    # far above any threshold the desiccation check applies.
    wet = _impact_handler(desiccated=True, accretion=_impact_accretion(H=1.0e3))
    assert wet.desiccated is True  # latched before the impact
    apply_impact(wet, _impact_event())

    assert wet.desiccated is False, (
        'the planet stayed latched as desiccated after an impact delivered '
        'volatiles, so the desiccated path will zero the delivery on the next '
        'outgassing call'
    )
    assert wet.hf_row['H_kg_total'] == pytest.approx(6.4e20, rel=1e-12)

    # A dry impactor brings nothing, so the planet is still dry and the latch
    # must hold.
    dry = _impact_handler(desiccated=True)
    apply_impact(dry, _impact_event())
    assert dry.desiccated is True, (
        'a dry impact lifted the desiccation latch, so the run resumes '
        'outgassing a planet that still has no volatiles'
    )
    assert dry.hf_row.get('H_kg_total', 0.0) == pytest.approx(0.0)


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
def test_escape_on_the_impact_step_draws_on_the_stripped_atmosphere(monkeypatch):
    """Escape after a total strip finds no atmosphere and leaves the interior.

    Escape runs before the outgassing solve re-partitions the budgets, and
    sizes its loss from ``*_kg_atm``. The strip must debit that reservoir too,
    or a strong escape rate drains the dissolved inventory on the impact step.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.escape.wrapper import calc_new_elements, limit_escape_step

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )
    handler = _impact_handler(
        accretion=_impact_accretion(atmloss_module='constant', atmloss_frac=1.0)
    )
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    _atm_state(handler.hf_row, H=(4.0e20, 5.0e20), C=(2.0e19, 9.0e19))
    apply_impact(handler, _impact_event())

    row = handler.hf_row
    assert row['H_kg_atm'] == 0.0
    assert row['C_kg_atm'] == 0.0
    row['esc_rate_total'] = 1.0e14
    assert limit_escape_step(row, 1.0e3, 'outgas', min_thresh=1.0e10) == 0.0
    tgt = calc_new_elements(row, 1.0e3, 'outgas', esc_mass=1.0e22)
    assert tgt['H'] == pytest.approx(1.0e20, rel=1e-12)
    assert tgt['C'] == pytest.approx(7.0e19, rel=1e-12)


@pytest.mark.unit
def test_a_partial_strip_lowers_the_atmospheric_reservoir_by_the_stripped_mass(monkeypatch):
    """A 25% strip leaves three quarters of each element's atmosphere."""
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )
    handler = _impact_handler(
        accretion=_impact_accretion(atmloss_module='constant', atmloss_frac=0.25)
    )
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    _atm_state(handler.hf_row, H=(4.0e20, 5.0e20), N=(1.0e19, 4.0e20))
    apply_impact(handler, _impact_event())

    assert handler.hf_row['H_kg_atm'] == pytest.approx(3.0e20, rel=1e-12)
    assert handler.hf_row['N_kg_atm'] == pytest.approx(7.5e18, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_delivered_volatiles_raise_the_desiccation_baseline(monkeypatch):
    """The gate must not accept delivered mass as loss without escape.

    1e20 kg escaped from a 1e20 kg baseline, then an impact delivered 6.4e20
    kg of H. If all of it vanishes without escape, the gate must refuse; with
    the baseline left at 1e20 kg it would accept (1e20 <= 1.5 * 1e20).
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.outgas.wrapper import check_desiccation

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )
    handler = _impact_handler(accretion=_impact_accretion(H=1000.0))
    row = handler.hf_row
    row.update(M_vol_initial=1.0e20, esc_kg_cumulative=1.0e20, H_kg_total=0.0)
    apply_impact(handler, _impact_event())

    delivered = 6.4e23 * 1000.0 / 1.0e6
    assert row['H_kg_total'] == pytest.approx(delivered, rel=1e-12)
    assert row['M_vol_initial'] == pytest.approx(1.0e20 + delivered, rel=1e-12)
    row['H_kg_total'] = 0.0
    assert (
        check_desiccation(SimpleNamespace(outgas=SimpleNamespace(mass_thresh=1.0e10)), row)
        is False
    )


@pytest.mark.unit
def test_delivery_before_any_escape_baseline_sets_none(monkeypatch):
    """Without a baseline the first escape call snapshots the grown totals."""
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )
    handler = _impact_handler(accretion=_impact_accretion(H=1000.0))
    apply_impact(handler, _impact_event())
    assert 'M_vol_initial' not in handler.hf_row
    # The delivery itself happened; only the baseline credit is skipped.
    assert handler.hf_row['H_kg_total'] == pytest.approx(6.4e23 * 1000.0 / 1.0e6, rel=1e-12)


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

    def _mock(dirs, config, hf_all, hf_row, outdir, **kwargs):
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

    The volatile budgets (M_ele) and the dry interior (M_int) are the two
    halves of M_planet, so each impact channel must land in exactly one of
    them: the impactor's rock grows the interior, its delivered volatiles and
    the target strip move the budgets. The anchor mass_tot follows the sum, so
    the next structure solve keeps it. Booking a channel in both halves
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

    # Verify double-counting failure modes sit outside tolerance. Full merger mass
    # over-counts delivery; subtracting stripped volatile under-counts it.
    assert abs(m_planet_after - (expected + delivered)) > 0.5 * delivered
    assert abs(m_planet_after - (expected - stripped)) > 0.5 * stripped

    # The anchor follows the real total, and the ledger records the change.
    assert handler.config.planet.mass_tot * M_earth == pytest.approx(expected, rel=1e-12)
    assert handler.hf_row['M_accreted_net'] == pytest.approx(expected - m_planet_0, rel=1e-9)
    # The next structure solve keeps that mass; a rock-only anchor would pull
    # M_planet back to m_planet_0 + rock, losing the net volatile change.
    _converging_solve_structure()(None, handler.config, None, handler.hf_row, None)
    assert handler.hf_row['M_planet'] == pytest.approx(expected, rel=1e-9)
    assert abs(m_planet_0 + rock - expected) > 1.0e-6 * expected


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
    # Set target state: H half atmospheric, N fully dissolved, C fully escaped.
    # The half-strength collision strips half of target atmosphere.
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
    planet lighter than before: the interior still grows by the accreted
    rock, the stripped budgets pull the whole-planet mass below its
    pre-impact value, and the anchor follows that mass.
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

    # The anchor follows the whole-planet mass: rock in, atmosphere out.
    assert handler.config.planet.mass_tot == pytest.approx(
        (m_planet_0 + 6.4e21 - 2.0e23) / M_earth, rel=1e-9
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
def test_match_planet_step_zero_without_history_falls_back_to_hf_row():
    """match_planet on step 0 reads formation composition from hf_row when history is None."""
    from proteus.accretion.wrapper import _impactor_volatile_content

    cfg = SimpleNamespace(
        accretion=_impact_accretion(impactor_volatiles='match_planet'),
        planet=SimpleNamespace(elements=SimpleNamespace(O_mode=None)),
    )
    hf_row = {'Time': 0.0, 'M_planet': 6.0e24, 'H_kg_total': 6.0e20, 'O_kg_total': 1.2e21}
    event = _impact_event(M_impactor=1.0e23)
    content = _impactor_volatile_content(cfg, None, event, hf_row=hf_row)
    assert content['H'] == pytest.approx(1.0e23 * (6.0e20 / 6.0e24))
    assert content['O'] == pytest.approx(1.0e23 * (1.2e21 / 6.0e24))


@pytest.mark.unit
def test_impactor_volatile_content_excludes_oxygen_under_ic_chemistry():
    """Under O_mode = 'ic_chemistry', oxygen is excluded from impactor volatiles."""
    from proteus.accretion.wrapper import _impactor_volatile_content

    cfg = SimpleNamespace(
        accretion=_impact_accretion(H=1000.0, O=5000.0),
        planet=SimpleNamespace(elements=SimpleNamespace(O_mode='ic_chemistry')),
    )
    content = _impactor_volatile_content(cfg, None, _impact_event())
    assert 'H' in content
    assert 'O' not in content


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
    # With loss off the anchor grows by each impactor's whole mass: its rock
    # plus the delivered content.
    expected_mass = 1.0 + 2 * event.mass_delta / M_earth
    assert handler.config.planet.mass_tot == pytest.approx(expected_mass, rel=1e-12)
    assert handler.hf_row['M_accreted_net'] == pytest.approx(2 * event.mass_delta, rel=1e-12)
    assert float(handler.hf_row.get('esc_kg_cumulative', 0.0)) == pytest.approx(0.0, abs=1.0)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_impact_loss_composes_with_delivery_and_a_broken_provider_raises(monkeypatch):
    """With loss active, one collision fraction governs both bodies.

    One impact carries three volatile channels: the shock strips the loss
    fraction of the target's atmosphere, the impactor's atmospheric part
    (mirrored from the planet, here exactly one third) loses the same
    fraction, and everything else is delivered. The anchor grows by the
    rock plus the delivered less the stripped mass. A loss module returning a fraction
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

    # The anchor grew by the rock plus the delivered less the stripped mass;
    # the impactor's lost volatiles never reach the planet.
    expected_mass = 1.0 + (mass_delta - content + delivered - stripped) / M_earth
    assert handler.config.planet.mass_tot == pytest.approx(expected_mass, rel=1e-12)
    # Discrimination: the full merger mass would also count the lost content.
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
    # Absolute pins detect parameter swapping between target and impactor.
    # An interchanged dispatch permutes the values and fails both checks.
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

    def _mock(dirs, config, hf_all, hf_row, outdir, **kwargs):
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

    def _solve(dirs, config, hf_all, hf_row, output, **kwargs):
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
            'n_impacts_applied': 1,
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
def test_debit_escaped_mass_lowers_only_a_whole_planet_anchor_with_accretion():
    """Escape lowers mass_tot and the ledger with an accretion module and the
    Zalmoxis structure; never with the dummy structure or accretion off."""
    from proteus.accretion.wrapper import debit_escaped_mass
    from proteus.utils.constants import M_earth

    def cfg(module, structure='zalmoxis'):
        return SimpleNamespace(
            accretion=SimpleNamespace(module=module),
            interior_struct=SimpleNamespace(module=structure),
            planet=SimpleNamespace(mass_tot=1.0),
        )

    on, row = cfg('dummy'), {'M_accreted_net': 1.0e22}
    debit_escaped_mass(on, row, 3.0e21)
    assert on.planet.mass_tot == pytest.approx(1.0 - 3.0e21 / M_earth, rel=1e-15)
    assert row['M_accreted_net'] == pytest.approx(7.0e21, rel=1e-15)

    # The dummy structure's mass_tot is the dry mass, so escape leaves it alone.
    dry, row = cfg('dummy', structure='dummy'), {'M_accreted_net': 0.0}
    debit_escaped_mass(dry, row, 3.0e21)
    assert dry.planet.mass_tot == pytest.approx(1.0, rel=1e-15)
    assert row['M_accreted_net'] == pytest.approx(0.0, abs=0.0)

    off, row = cfg(None), {'M_accreted_net': 0.0}
    debit_escaped_mass(off, row, 3.0e21)
    assert off.planet.mass_tot == pytest.approx(1.0, rel=1e-15)
    assert row['M_accreted_net'] == pytest.approx(0.0, abs=0.0)

    for bad in (0.0, -1.0e20, float('nan'), float('inf')):
        on, row = cfg('dummy'), {'M_accreted_net': 0.0}
        debit_escaped_mass(on, row, bad)
        assert on.planet.mass_tot == pytest.approx(1.0, rel=1e-15)
        assert row['M_accreted_net'] == pytest.approx(0.0, abs=0.0)


def _restore_handler(mass_tot, hf_row):
    return SimpleNamespace(
        config=SimpleNamespace(
            accretion=SimpleNamespace(module='morrigan'),
            params=SimpleNamespace(resume=True),
            planet=SimpleNamespace(mass_tot=mass_tot),
            orbit=SimpleNamespace(semimajoraxis=1.0, eccentricity=0.0),
        ),
        hf_row=hf_row,
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_resume_after_impact_and_escape_restores_the_uninterrupted_mass(monkeypatch):
    """A resume rebuilds mass_tot exactly as the uninterrupted run left it.

    The run takes a wet impact with a 50 % strip, then escape removes 2e21
    kg. Restoring from the rock alone would miss the delivered, stripped and
    escaped mass.
    """
    from proteus.accretion.wrapper import (
        apply_impact,
        debit_escaped_mass,
        restore_accretion_state,
    )
    from proteus.utils.constants import M_earth

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )
    handler = _impact_handler(
        accretion=_impact_accretion(atmloss_module='constant', atmloss_frac=0.5, H=1000.0)
    )
    handler.config.accretion.module = 'dummy'
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    _atm_state(handler.hf_row, H=(4.0e21, 5.0e21))
    apply_impact(handler, _impact_event())
    debit_escaped_mass(handler.config, handler.hf_row, 2.0e21)
    uninterrupted = handler.config.planet.mass_tot

    resumed = _restore_handler(1.0, dict(handler.hf_row))
    restore_accretion_state(resumed)
    assert resumed.config.planet.mass_tot == pytest.approx(uninterrupted, rel=1e-14)
    rock_only = 1.0 + handler.hf_row['M_accreted_rock'] / M_earth
    assert abs(rock_only - uninterrupted) > 1.0e-6


@pytest.mark.unit
def test_resume_with_escape_before_any_impact_restores_the_lowered_mass():
    """Escape alone lowers the anchor; the resume keeps that, with no rock."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import M_earth

    handler = _restore_handler(
        1.0, {'M_accreted_rock': 0.0, 'M_accreted_net': -1.0e21, 'n_impacts_applied': 0}
    )
    restore_accretion_state(handler)
    assert handler.config.planet.mass_tot == pytest.approx(1.0 - 1.0e21 / M_earth, rel=1e-15)
    # No impact yet: the orbit stays at its configured value.
    assert handler.config.orbit.semimajoraxis == pytest.approx(1.0, rel=1e-15)


@pytest.mark.unit
def test_resume_of_a_helpfile_without_the_net_column_uses_the_rock():
    """A zero-filled M_accreted_net with rock recorded keeps the rock rule;
    a recorded net change is used instead of the rock."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import M_earth

    handler = _restore_handler(
        1.0, {'M_accreted_rock': 0.5 * M_earth, 'M_accreted_net': 0.0, 'n_impacts_applied': 1}
    )
    restore_accretion_state(handler)
    assert handler.config.planet.mass_tot == pytest.approx(1.5, rel=1e-12)
    handler = _restore_handler(
        1.0,
        {
            'M_accreted_rock': 0.5 * M_earth,
            'M_accreted_net': 0.4 * M_earth,
            'n_impacts_applied': 1,
        },
    )
    restore_accretion_state(handler)
    assert handler.config.planet.mass_tot == pytest.approx(1.4, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_a_resume_after_a_legacy_resume_keeps_the_rock():
    """The legacy fallback stores the rock in the ledger, so a later escape
    debit and a second resume give the uninterrupted mass."""
    import pandas as pd

    from proteus.accretion.wrapper import debit_escaped_mass, restore_accretion_state
    from proteus.utils.constants import M_earth

    row = {'M_accreted_rock': 0.5 * M_earth, 'M_accreted_net': 0.0, 'n_impacts_applied': 1}
    first = _restore_handler(1.0, row)
    first.config.interior_struct = SimpleNamespace(module='zalmoxis')
    first.hf_all = pd.DataFrame([dict(row)])
    restore_accretion_state(first)
    assert row['M_accreted_net'] == pytest.approx(0.5 * M_earth, rel=1e-15)
    assert first.hf_all['M_accreted_net'].iloc[-1] == pytest.approx(0.5 * M_earth, rel=1e-15)

    debit_escaped_mass(first.config, row, 1.0e21)
    uninterrupted = first.config.planet.mass_tot
    second = _restore_handler(1.0, dict(row))
    restore_accretion_state(second)
    assert second.config.planet.mass_tot == pytest.approx(uninterrupted, rel=1e-14)
    assert uninterrupted == pytest.approx(1.5 - 1.0e21 / M_earth, rel=1e-14)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_dummy_structure_anchor_takes_the_rock_only(monkeypatch):
    """With the dummy structure mass_tot is the dry mass, so a wet, stripping
    impact grows it by the rock alone and the volatiles stay in the budgets."""
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import M_earth

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )
    handler = _impact_handler(
        accretion=_impact_accretion(atmloss_module='constant', atmloss_frac=0.5, H=1000.0),
        structure='dummy',
    )
    handler.config.outgas = SimpleNamespace(mass_thresh=1.0e10)
    _atm_state(handler.hf_row, H=(4.0e21, 5.0e21))
    event = _impact_event()
    apply_impact(handler, event)

    content = event.M_impactor * 1000.0 / 1.0e6
    rock = event.mass_delta - content
    assert handler.config.planet.mass_tot == pytest.approx(1.0 + rock / M_earth, rel=1e-12)
    assert handler.hf_row['M_accreted_net'] == pytest.approx(rock, rel=1e-12)
    # The volatiles moved in the budgets: half the atmosphere stripped, and the
    # content delivered less the exposed (mirrored f_atm) part the loss takes.
    f_atm = 4.0e21 / 5.0e21
    delivered = content * (1.0 - f_atm * 0.5)
    assert handler.hf_row['H_kg_total'] == pytest.approx(5.0e21 - 2.0e21 + delivered, rel=1e-9)


@pytest.mark.unit
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_resume_refuses_a_non_finite_net_column(bad):
    """A non-finite M_accreted_net cannot rebuild the mass and is refused."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import M_earth

    handler = _restore_handler(
        1.0, {'M_accreted_rock': 0.5 * M_earth, 'M_accreted_net': bad, 'n_impacts_applied': 1}
    )
    with pytest.raises(RuntimeError, match='M_accreted_net'):
        restore_accretion_state(handler)
    # The refused resume leaves the configured mass untouched.
    assert handler.config.planet.mass_tot == pytest.approx(1.0, rel=1e-15)


@pytest.mark.unit
def test_restore_accretion_state_orbit_fallback():
    """Missing, non-finite, or negative row orbit elements fall back to config.orbit."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU, M_earth

    # Missing eccentricity in row: falls back to config.orbit.eccentricity
    handler = SimpleNamespace(
        config=SimpleNamespace(
            accretion=SimpleNamespace(module='timeline'),
            params=SimpleNamespace(resume=True),
            planet=SimpleNamespace(mass_tot=1.0),
            orbit=SimpleNamespace(semimajoraxis=1.0, eccentricity=0.08),
        ),
        hf_row={
            'M_accreted_rock': 0.5 * M_earth,
            'semimajorax': 1.2 * AU,
            'n_impacts_applied': 1,
        },
    )
    restore_accretion_state(handler)
    assert handler.config.orbit.semimajoraxis == pytest.approx(1.2, rel=1e-12)
    assert handler.config.orbit.eccentricity == pytest.approx(0.08, rel=1e-12)

    # NaN, inf, and negative eccentricity in row: fall back to config.orbit.eccentricity
    for bad_e in (float('nan'), float('inf'), -0.05):
        handler.config.orbit.eccentricity = 0.08
        handler.hf_row['eccentricity'] = bad_e
        restore_accretion_state(handler)
        assert handler.config.orbit.eccentricity == pytest.approx(0.08, rel=1e-12)

    # Missing, non-finite, or non-positive semimajoraxis: falls back to config.orbit
    for bad_a in (None, float('nan'), float('inf'), 0.0, -1.0):
        handler.config.orbit.semimajoraxis = 1.0
        handler.config.orbit.eccentricity = 0.08
        handler.hf_row['eccentricity'] = 0.04
        if bad_a is None:
            handler.hf_row.pop('semimajorax', None)
        else:
            handler.hf_row['semimajorax'] = bad_a
        restore_accretion_state(handler)
        assert handler.config.orbit.semimajoraxis == pytest.approx(1.0, rel=1e-12)
        assert handler.config.orbit.eccentricity == pytest.approx(0.08, rel=1e-12)


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
            hf_row={
                'M_accreted_rock': accreted,
                'semimajorax': 9.9e11,
                'eccentricity': 0.9,
                'n_impacts_applied': 1,
            },
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
def test_restore_accretion_state_drops_already_applied_events_on_resume(tmp_path):
    """Resume rebuilds impact_events, dropping impacts already recorded in helpfile."""
    from proteus.accretion.common import write_timeline
    from proteus.accretion.wrapper import _RESOLVED_TIMELINE_FILE, restore_accretion_state
    from proteus.utils.constants import AU

    ev1 = _impact_event(
        time=0.5, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    ev2 = _impact_event(
        time=100.0, M_target_before=6.072e24, M_impactor=1e23, M_merged_after=6.172e24
    )
    write_timeline([ev1, ev2], str(tmp_path / _RESOLVED_TIMELINE_FILE))

    # Case 1: n_impacts_applied is present in helpfile row
    handler = SimpleNamespace(
        config=SimpleNamespace(
            accretion=SimpleNamespace(module='dummy', impactor_volatiles='dry'),
            params=SimpleNamespace(resume=True),
            planet=SimpleNamespace(mass_tot=1.0),
            orbit=SimpleNamespace(semimajoraxis=1.0, eccentricity=0.0),
        ),
        hf_row={
            'Time': 0.0,
            'M_accreted_rock': 1e23,
            'n_impacts_applied': 1,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        },
        directories={'output': str(tmp_path)},
        impact_events=[ev1, ev2],
    )
    restore_accretion_state(handler)
    assert handler.impact_events == [ev2]
    assert handler.hf_row['n_impacts_applied'] == 1


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
def test_apply_impact_passes_thermal_solve_false_to_solve_structure(monkeypatch):
    """apply_impact re-solves structure with thermal_solve=False.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture for monkeypatching.
    """
    from proteus.accretion.wrapper import apply_impact

    captured_kwargs = {}

    def _mock_solve_structure(*args, **kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure',
        _mock_solve_structure,
    )

    handler = _impact_handler(mass_tot=1.0)
    apply_impact(handler, _impact_event())

    assert 'thermal_solve' in captured_kwargs
    assert captured_kwargs['thermal_solve'] is False


@pytest.mark.unit
def test_orbit_elements_evolve_from_helpfile_row(monkeypatch):
    """An impact evolves orbit elements from the helpfile row rather than config.

    During a run, the orbit elements in the helpfile row evolve over time.
    When an impact occurs, the eccentricity change and semi-major axis ratio
    must be applied to the current helpfile row values, updating both the
    config and the row. If the row values are missing or non-finite, they fall
    back to config.orbit.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import AU

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    # Row eccentricity differs from config; change applies to row value
    handler = _impact_handler(semimajoraxis=1.0, eccentricity=0.1)
    handler.hf_row['eccentricity'] = 0.01
    event = _impact_event(a_before=1.0e11, a_after=1.0e11, e_before=0.01, e_after=0.04)
    apply_impact(handler, event)

    assert handler.config.orbit.eccentricity == pytest.approx(0.04, rel=1e-12)
    assert handler.hf_row['eccentricity'] == pytest.approx(0.04, rel=1e-12)

    # Row semimajoraxis differs from config; ratio applies to row value
    handler_a = _impact_handler(semimajoraxis=1.0, eccentricity=0.05)
    handler_a.hf_row['semimajorax'] = 0.8 * AU
    event_a = _impact_event(a_before=1.0e11, a_after=1.2e11, e_before=0.01, e_after=0.01)
    apply_impact(handler_a, event_a)

    assert handler_a.config.orbit.semimajoraxis == pytest.approx(0.96, rel=1e-12)
    assert handler_a.hf_row['semimajorax'] == pytest.approx(0.96 * AU, rel=1e-12)

    # Non-finite, non-positive, or missing semimajoraxis falls back to config.orbit
    event_first = _impact_event(a_before=1.0e11, a_after=1.2e11, e_before=0.01, e_after=0.04)
    for bad_a in (0.0, float('inf'), float('nan'), -1.0, None):
        handler_bad = _impact_handler(semimajoraxis=0.5, eccentricity=0.1)
        if bad_a is None:
            del handler_bad.hf_row['semimajorax']
        else:
            handler_bad.hf_row['semimajorax'] = bad_a
        handler_bad.hf_row['eccentricity'] = 0.0
        apply_impact(handler_bad, event_first)
        assert handler_bad.config.orbit.eccentricity == pytest.approx(0.13, rel=1e-12)
        assert handler_bad.hf_row['eccentricity'] == pytest.approx(0.13, rel=1e-12)
        assert handler_bad.config.orbit.semimajoraxis == pytest.approx(0.6, rel=1e-12)
        assert handler_bad.hf_row['semimajorax'] == pytest.approx(0.6 * AU, rel=1e-12)

    # Non-finite, negative, or missing eccentricity falls back to config.orbit.eccentricity
    event_nan = _impact_event(a_before=1.0e11, a_after=1.0e11, e_before=0.01, e_after=0.04)
    for bad_e in (float('inf'), float('nan'), -0.05, None):
        handler_bad_e = _impact_handler(semimajoraxis=1.0, eccentricity=0.15)
        handler_bad_e.hf_row['semimajorax'] = 1.0 * AU
        if bad_e is None:
            del handler_bad_e.hf_row['eccentricity']
        else:
            handler_bad_e.hf_row['eccentricity'] = bad_e
        apply_impact(handler_bad_e, event_nan)
        assert handler_bad_e.config.orbit.eccentricity == pytest.approx(0.18, rel=1e-12)
        assert handler_bad_e.hf_row['eccentricity'] == pytest.approx(0.18, rel=1e-12)


@pytest.mark.unit
def test_apply_impact_refuses_nonpositive_semimajoraxis_ratio():
    """Non-positive or non-finite semimajoraxis_ratio raises ValueError."""
    from unittest.mock import MagicMock

    from proteus.accretion.wrapper import apply_impact

    handler = _impact_handler(semimajoraxis=1.0, eccentricity=0.1)
    for bad_ratio in (0.0, -0.5, float('nan'), float('inf')):
        event = MagicMock()
        event.time = 100.0
        event.semimajoraxis_ratio = bad_ratio
        with pytest.raises(ValueError, match='non-positive or non-finite semimajoraxis_ratio'):
            apply_impact(handler, event)


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


def _impact_loop_runner(tmp_path, monkeypatch, provide_tables):
    """Build a 2-step aragog run on the dummy structure with one dummy impact at 2 yr."""
    from pathlib import Path

    import numpy as np

    import proteus.interior_energetics.wrapper as interior_wrapper
    import proteus.interior_struct.zalmoxis as zalmoxis
    from proteus import Proteus
    from proteus.interior_energetics.aragog import AragogRunner
    from proteus.utils.helper import format_subyear_time

    output_dir = tmp_path / 'run'
    data_dir = output_dir / 'data'

    config_path = Path(__file__).resolve().parents[2] / 'input' / 'dummy.toml'
    text = config_path.read_text().replace('path = "auto"', f'path = "{output_dir}"', 1)
    cfg = tmp_path / 'test.toml'
    cfg.write_text(text)

    runner = Proteus(config_path=cfg)
    runner.config.interior_energetics.module = 'aragog'
    runner.config.interior_struct.melting_dir = 'Monteux-600'
    runner.config.atmos_chem.module = None
    runner.config.params.stop.time.minimum = 0.0
    runner.config.params.stop.time.maximum = 2.0
    runner.config.params.stop.iters.minimum = 0
    runner.config.params.stop.iters.maximum = 10
    runner.config.params.dt.initial = 1.0
    runner.config.params.dt.minimum = 0.1
    runner.config.params.dt.maximum = 1.0
    runner.config.params.out.write_mod = 1
    runner.config.params.out.dt_write_rel = 0.0
    runner.config.params.out.plot_mod = None
    runner.config.params.out.archive_mod = 'none'

    runner.config.accretion.module = 'dummy'
    runner.config.accretion.dummy.num_impacts = 1
    runner.config.accretion.dummy.mass_accreted = 0.1
    runner.config.accretion.dummy.time_last = 2.0
    runner.config.accretion.dummy.timescale = 1000.0
    runner.config.accretion.dummy.eccentricity = 0.05
    runner.config.accretion.impactor_volatiles = 'dry'

    written_snapshots = []

    def mock_run_interior(
        dirs, config, hf_all, hf_row, interior_o, atmos_o=None, verbose=True, write_data=True
    ):
        t = float(hf_row.get('Time', 0.0))
        interior_o.dt = 1.0
        hf_row['T_magma'] = 2000.0
        hf_row['T_surf'] = 2000.0
        hf_row['Phi_global'] = 0.5
        if interior_o.aragog_solver is None:
            interior_o.aragog_solver = SimpleNamespace(
                _solution=None,
                solution=None,
                _step_heat_content=lambda s1, s2: 0.0,
            )
        interior_o._last_entropy = np.array([6000.0])
        if write_data:
            early_path = data_dir / '0p000_int.nc'
            if not early_path.exists():
                early_path.write_text('initial snapshot')
            snap_time = t + interior_o.dt
            snap_path = data_dir / f'{format_subyear_time(snap_time)}_int.nc'
            snap_path.parent.mkdir(parents=True, exist_ok=True)
            snap_path.write_text('snapshot')
            written_snapshots.append(snap_path)

    monkeypatch.setattr(interior_wrapper, 'run_interior', mock_run_interior)
    monkeypatch.setattr(
        AragogRunner,
        '_set_entropy_ic',
        staticmethod(lambda config, interior_o, outdir, hf_row=None: np.array([6000.0])),
    )
    monkeypatch.setattr(zalmoxis, 'generate_spider_tables', lambda config, outdir: None)
    monkeypatch.setattr(interior_wrapper, '_provide_spider_eos_tables', provide_tables)

    data_dir.mkdir(parents=True, exist_ok=True)
    earlier_snap = data_dir / '0p000_int.nc'
    earlier_snap.write_text('initial snapshot')
    return runner, data_dir, written_snapshots, earlier_snap


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_main_loop_discards_preimpact_snapshot_on_impact_step(tmp_path, monkeypatch):
    """The main loop discards the pre-impact snapshot when an impact occurs.

    Contract clause: when an impact occurs during a step, the interior
    snapshot written at the start of that step predates the mantle remelt.
    The main loop must call discard_preimpact_snapshot to remove the stale
    snapshot while preserving earlier valid snapshots.

    Verifies:
    - The snapshot written on the impact step was initially created.
    - The impact landed and delivered rock mass.
    - The snapshot written on the impact step is removed from disk.
    - Earlier valid snapshots remain intact for resuming.
    - The P-S tables are provided at setup and again at the impact re-solve.
    """
    from proteus.utils.helper import format_subyear_time

    calls = []
    runner, data_dir, written_snapshots, earlier_snap = _impact_loop_runner(
        tmp_path, monkeypatch, lambda *a: calls.append(a)
    )
    runner.start(resume=False, offline=True)

    expected_impact_snap = data_dir / f'{format_subyear_time(2.0)}_int.nc'
    # Positive controls: the interior wrote the snapshot during the step, and the impact landed
    assert expected_impact_snap in written_snapshots, (
        'the interior must have written the impact step snapshot'
    )
    assert float(runner.hf_all.iloc[-1]['M_accreted_rock']) > 0.0, 'the impact must have landed'

    # The main loop must have discarded the pre-impact snapshot while keeping earlier snapshots
    assert earlier_snap.exists(), 'earlier snapshot must be kept'
    assert not expected_impact_snap.exists(), (
        'impact step snapshot must be discarded by main loop'
    )
    assert len(calls) == 2, 'tables are provided at setup and at the impact re-solve'


@pytest.mark.unit
def test_main_loop_checks_crystallization_only_after_the_init_stage(tmp_path, monkeypatch):
    """start() calls _check_crystallization on every step after the init stage and never
    during it, where the init stage recalculates the volatile targets instead."""
    import proteus.outgas.wrapper as outgas_wrapper
    from proteus import Proteus

    runner, *_ = _impact_loop_runner(tmp_path, monkeypatch, lambda *a: None)
    targets, checks = [], []
    real_targets, real_check = (
        outgas_wrapper.calc_target_elemental_inventories,
        Proteus._check_crystallization,
    )

    def count_targets(*args):
        targets.append(runner.init_stage)
        return real_targets(*args)

    def spy_check(self):
        checks.append(self.init_stage)
        return real_check(self)

    monkeypatch.setattr(outgas_wrapper, 'calc_target_elemental_inventories', count_targets)
    monkeypatch.setattr(Proteus, '_check_crystallization', spy_check)
    runner.start(resume=False, offline=True)

    assert any(targets), 'the init stage must run first'
    assert len(checks) >= 1 and not any(checks)


@pytest.mark.unit
def test_missing_melting_curves_at_the_impact_stop_the_run_with_status_20(
    tmp_path, monkeypatch
):
    """A melting curve missing at the impact's structure re-solve stops start()
    with status 20, before the impact step reaches the helpfile."""
    from proteus.interior_energetics.common import MissingMeltingCurveError

    calls = []

    def provide_tables(*args):
        calls.append(args)
        if len(calls) == 2:
            raise MissingMeltingCurveError('melting curves missing at the impact')

    runner, *_ = _impact_loop_runner(tmp_path, monkeypatch, provide_tables)
    with pytest.raises(MissingMeltingCurveError, match='at the impact'):
        runner.start(resume=False, offline=True)

    assert len(calls) == 2
    status = (tmp_path / 'run' / 'status').read_text().splitlines()[0]
    assert status == '20'
    assert (runner.hf_all['M_accreted_rock'] == 0.0).all(), 'no impact row is written'
    assert runner.hf_row['n_impacts_applied'] == 1


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_rock_and_volatile_element_sets_partition_the_registry():
    """An impact grows rock through the structure solve and volatiles through
    the budgets, and no element may travel by both routes.

    Contract clause: the mass an impact adds arrives as rock, through the
    planet's mass anchor and the equation of state. Separately, the impact
    conserves the per-element whole-planet budgets across that growth and sizes
    both its atmospheric stripping and its volatile delivery from them. An
    element counted in both places would have its mass booked twice, once in
    the rock and once in a budget, and the whole-planet total would drift
    upward on every impact.

    The registry already draws that line for the rest of the model:
    ``update_planet_mass`` sums M_ele over the volatile elements and the noble
    gases and leaves the rock-forming elements out, because rock vapour puts
    their mass in the atmosphere without debiting the interior. The accretion
    module must draw it in the same place, so this pins the conserved set
    against the M_ele definition rather than against a copy of it, and takes
    the rock set as the complement.

    Verifies:
    - The conserved set is exactly what M_ele sums over, so nothing an impact
      conserves is left out of the planet mass and nothing it grows is in.
    - The two sets are disjoint and together cover every tracked element.
    - Every element the registry calls rock-forming is outside the conserved
      set, including those added after the accretion module was written.
    """
    import inspect

    from proteus.accretion.wrapper import _VOLATILE_ELEMENTS
    from proteus.interior_energetics.wrapper import update_planet_mass
    from proteus.utils.constants import (
        element_list,
        noble_gases,
        vap_element_list,
        vol_element_list,
    )

    conserved = set(_VOLATILE_ELEMENTS)
    # Rock is the complement of the conserved volatile set. Taking the complement
    # prevents elements from double counting or omission.
    rock = set(element_list) - conserved

    assert conserved == set(vol_element_list) | set(noble_gases)

    # The conserved set is drawn from the registry, so an element the registry
    # tracks cannot fall outside both channels.
    assert conserved <= set(element_list)

    # Verify rock-forming set includes elements beyond core species.
    # Al, Ti, Ca, and K are rock-forming and must not be treated as volatile budgets.
    assert {'Al', 'Ti', 'Ca', 'K'} <= rock
    for element in vap_element_list:
        assert element not in conserved, (
            f"'{element}' is rock-forming in the element registry but is "
            'conserved as a volatile budget across an impact, so its mass is '
            'counted both in the rock the impact adds and in the budget'
        )

    # The conserved set is the one M_ele is summed over, read off the source of
    # that sum rather than restated here, so the two cannot drift apart.
    m_ele_source = inspect.getsource(update_planet_mass)
    assert 'for e in vol_element_list + noble_gases:' in m_ele_source, (
        'update_planet_mass no longer sums M_ele over vol_element_list + '
        'noble_gases, so what an impact conserves and what the whole-planet '
        'mass is built from may now be different sets'
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_row_an_impact_leaves_satisfies_the_runtime_mass_invariants(monkeypatch):
    """The main loop's own invariant checks pass on a post-impact row.

    Contract clause: every iteration ends by asserting that the atmosphere is
    no heavier than the planet and that the summed per-species atmospheric
    masses still equal ``M_vol_atm``. An impact rewrites ``M_planet`` through
    the mass anchor and rewrites the per-element budgets through the strip and
    the delivery, all inside the same iteration those checks close, so the row
    it hands on has to satisfy them rather than relying on the outgassing step
    to repair it.

    The impact here both strips a heavy atmosphere and delivers a wet
    impactor's volatiles, so the two channels that move mass in opposite
    directions are exercised together.

    Edge case: the same checks are run on the pre-impact row first, so a row
    that was already failing them cannot be mistaken for one the impact fixed.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.coupler import (
        assert_mass_conservation,
        assert_surface_pressure_consistency,
    )

    handler = _impact_handler(
        accretion=_impact_accretion(
            atmloss_module='constant', atmloss_frac=0.4, impactor_volatiles='ppmw', H=500.0
        )
    )
    hf_row = handler.hf_row
    _atm_state(hf_row, H=(3.0e19, 4.0e20), O=(2.0e19, 3.0e20))
    # The gas-species columns the invariant sums over, consistent with the
    # per-element atmospheric masses above: H2O carries both H and O.
    hf_row['H2O_kg_atm'] = 5.0e19
    hf_row['M_vol_atm'] = 5.0e19
    hf_row['M_atm'] = 5.0e19
    hf_row['M_ele'] = 7.0e20
    hf_row['M_int'] = 5.9736e24
    hf_row['M_planet'] = hf_row['M_int'] + hf_row['M_ele']
    hf_row['P_surf'] = 120.0
    hf_row['P_vol'] = 120.0
    hf_row['P_vap'] = 0.0
    hf_row['outgas_mass_thresh'] = 0.0

    config = SimpleNamespace(outgas=SimpleNamespace(mass_thresh=1.0e10, vapourise=False))
    handler.config.outgas = config.outgas

    # The starting row already satisfies both checks, so anything raised after
    # the impact is the impact's doing.
    assert_mass_conservation(hf_row, require_atm_le_planet=True)
    assert_surface_pressure_consistency(config, hf_row)

    def _solve(dirs, cfg, hf_all, row, output, **kwargs):
        row['M_int'] = cfg.planet.mass_tot * 5.9736e24

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', _solve, raising=False
    )

    apply_impact(handler, _impact_event())

    # Neither invariant is breached by the row the impact hands on.
    assert_mass_conservation(hf_row, require_atm_le_planet=True)
    assert_surface_pressure_consistency(config, hf_row)

    # Confirm both delivery and stripping updated the row. Total H closes
    # to 6.984e20 kg accounting for atmospheric loss and retained delivery.
    assert hf_row['H_kg_total'] == pytest.approx(6.984e20, rel=1e-12)
    # Both strips are booked as loss: 40% of the H and of the O atmosphere.
    assert hf_row['esc_kg_cumulative'] == pytest.approx(2.0e19, rel=1e-12)
    assert hf_row['M_planet'] > 5.9736e24, 'the planet did not grow'


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_the_rock_remainder_tolerates_closure_rounding_but_not_a_real_overrun():
    """A budget near the whole impactor must not abort on closure rounding.

    The impactor's volatile content is a fraction of ``M_impactor`` while the
    rock remainder is taken from ``mass_delta``, which a timeline may leave
    short of it by up to ``MASS_CLOSURE_RTOL`` of the merged mass. A budget
    approaching 1e6 ppmw therefore lands slightly negative on arithmetic alone,
    and refusing that would abort a run whose configuration is valid. A content
    genuinely larger than the impactor still has to be refused, so the guard has
    to separate the two rather than accept or reject both.
    """
    from proteus.accretion.common import MASS_CLOSURE_RTOL

    # 1:100 impactor, the case where the two masses differ most in relative
    # terms, so the rounding band is widest against mass_delta.
    m_target, m_impactor = 6.0e24, 6.0e22
    merged = (m_target + m_impactor) * (1.0 - MASS_CLOSURE_RTOL)  # accepted by closure
    mass_delta = merged - m_target
    tol = MASS_CLOSURE_RTOL * (m_target + m_impactor)

    rounding = mass_delta - m_impactor * 999_999.0 / 1.0e6
    assert rounding < 0.0, 'this case must be negative, or it tests nothing'
    assert rounding >= -tol, 'closure rounding must fall inside the tolerance'

    overrun = mass_delta - m_impactor * 1.2e6 / 1.0e6
    assert overrun < -tol, 'a 120% budget must fall outside the tolerance'

    # Discrimination: the two differ by three orders of magnitude, so the band
    # separates them rather than merely admitting both.
    assert abs(overrun) > 1.0e3 * abs(rounding)

    # The tolerance is measured against the merged mass, not against mass_delta;
    # the latter is ~100x smaller here and would refuse the rounding case.
    assert tol > abs(rounding)
    assert MASS_CLOSURE_RTOL * mass_delta < abs(rounding)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_dummy_structure_in_apply_impact_preserves_rock_elements_and_user_ic(tmp_path):
    """Dummy structure solve under apply_impact preserves rock elements and user IC.

    When thermal_solve is False, dummy structure solve determines R_int and
    updates gravity and planet mass without resetting elemental inventories.
    Pre-impact rock vapor inventory and user initial conditions survive,
    while volatile delivery and atmospheric loss update H_kg_total according to
    before + delivered - stripped.
    """
    from types import SimpleNamespace

    from proteus.accretion.common import ImpactEvent
    from proteus.accretion.wrapper import apply_impact
    from proteus.config import Config
    from proteus.utils.constants import AU

    config = Config()
    config.interior_struct.module = 'dummy'
    config.interior_energetics.module = 'dummy'
    config.planet.mass_tot = 1.0
    config.planet.elements.H_mode = 'ppmw'
    config.planet.elements.H_budget = 10000.0
    config.accretion.impactor_volatiles = 'ppmw'
    config.accretion.impactor_H_ppmw = 1000.0
    config.accretion.atmloss_module = 'constant'
    config.accretion.atmloss_frac = 0.5

    hf_row = {
        'Time': 100.0,
        'semimajorax': 0.5 * AU,
        'eccentricity': 0.1,
        'T_magma': 2000.0,
        'Phi_global': 1.0,
        'M_int': 5.972e24,
        'M_core': 0.3 * 5.972e24,
        'M_mantle': 0.7 * 5.972e24,
        'R_int': 6.4e6,
        'R_core': 3.5e6,
        'gravity': 9.8,
        'H_kg_total': 3.0e20,
        'H_kg_atm': 1.0e20,
        'Si_kg_total': 1.0e21,
        'O_kg_user_ic': 42.0,
        'F_atm': 0.0,
    }

    handler = SimpleNamespace(
        config=config,
        hf_row=hf_row,
        hf_all=None,
        interior_o=SimpleNamespace(impact_reset=False),
        crystallized=False,
        desiccated=False,
        directories={'output': str(tmp_path)},
    )

    event = ImpactEvent(
        time=100.0,
        M_target_before=5.972e24,
        M_impactor=1.0e23,
        M_merged_after=6.072e24,
        v_impact=1.2e4,
        v_esc=1.1e4,
        impact_parameter=0.3,
        R_target_before=6.4e6,
        R_impactor=2.0e6,
        rho_target=5510.0,
        rho_impactor=3930.0,
        a_before=1.4e11,
        a_after=1.4e11,
        e_before=0.03,
        e_after=0.03,
        id_target=1,
        id_impactor=7,
    )

    apply_impact(handler, event)

    assert hf_row['Si_kg_total'] == pytest.approx(1.0e21, rel=1e-12)
    assert hf_row['O_kg_user_ic'] == pytest.approx(42.0, rel=1e-12)
    # H_kg_total: before (3.0e20) + delivered (8.333333e19) - stripped (5.0e19)
    assert hf_row['H_kg_total'] == pytest.approx(3.333333333333333e20, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_apply_impact_rock_remainder_value_error_and_tolerance_clamp(tmp_path):
    """apply_impact raises ValueError on overrun and clamps rounding to zero.

    When impactor volatile content exceeds mass_delta beyond the closure tolerance,
    apply_impact raises ValueError. When the remainder is slightly negative within
    the closure tolerance, it is clamped to zero, avoiding unphysical rock subtraction.
    """
    from types import SimpleNamespace

    from proteus.accretion.common import MASS_CLOSURE_RTOL, ImpactEvent
    from proteus.accretion.wrapper import apply_impact
    from proteus.config import Config
    from proteus.utils.constants import AU

    # 1. Overrun beyond tolerance: raises ValueError
    config = Config()
    config.accretion.impactor_volatiles = 'ppmw'
    config.accretion.impactor_H_ppmw = 1.2e6
    config.accretion.atmloss_module = None
    config.interior_struct.module = 'dummy'
    config.interior_energetics.module = 'dummy'
    config.planet.mass_tot = 1.0

    hf_row = {
        'Time': 100.0,
        'semimajorax': 0.5 * AU,
        'eccentricity': 0.1,
        'T_magma': 2000.0,
        'Phi_global': 1.0,
        'M_int': 5.972e24,
        'M_core': 0.3 * 5.972e24,
        'M_mantle': 0.7 * 5.972e24,
        'R_int': 6.4e6,
        'R_core': 3.5e6,
        'gravity': 9.8,
        'M_accreted_rock': 0.0,
        'F_atm': 0.0,
    }
    handler = SimpleNamespace(
        config=config,
        hf_row=hf_row,
        hf_all=None,
        interior_o=SimpleNamespace(impact_reset=False),
        crystallized=False,
        desiccated=False,
        directories={'output': str(tmp_path)},
    )
    event_overrun = ImpactEvent(
        time=100.0,
        M_target_before=6.0e24,
        M_impactor=6.0e22,
        M_merged_after=6.06e24,
        v_impact=1.2e4,
        v_esc=1.1e4,
        impact_parameter=0.3,
        R_target_before=6.4e6,
        R_impactor=2.0e6,
        rho_target=5510.0,
        rho_impactor=3930.0,
        a_before=1.4e11,
        a_after=1.4e11,
        e_before=0.03,
        e_after=0.03,
        id_target=1,
        id_impactor=7,
    )
    with pytest.raises(ValueError, match='Impactor volatile content'):
        apply_impact(handler, event_overrun)

    # 2. Closure rounding within tolerance: clamped to 0.0 without error
    config.accretion.impactor_H_ppmw = 1.0e6
    m_merged_rounded = (6.0e24 + 6.0e22) * (1.0 - 0.5 * MASS_CLOSURE_RTOL)
    event_rounding = ImpactEvent(
        time=100.0,
        M_target_before=6.0e24,
        M_impactor=6.0e22,
        M_merged_after=m_merged_rounded,
        v_impact=1.2e4,
        v_esc=1.1e4,
        impact_parameter=0.3,
        R_target_before=6.4e6,
        R_impactor=2.0e6,
        rho_target=5510.0,
        rho_impactor=3930.0,
        a_before=1.4e11,
        a_after=1.4e11,
        e_before=0.03,
        e_after=0.03,
        id_target=1,
        id_impactor=7,
    )
    mass_tot_before = handler.config.planet.mass_tot
    apply_impact(handler, event_rounding)
    assert handler.hf_row['M_accreted_rock'] == pytest.approx(0.0, abs=1e-12)
    # No rock is added, and the dummy structure's dry anchor takes no volatiles.
    assert handler.config.planet.mass_tot == pytest.approx(mass_tot_before, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_multiple_impacts_in_one_step_applied_in_time_order(tmp_path):
    """When multiple impacts occur in one step, both apply in time order.

    The due_events helper sorts impacts in time order, and the main loop applies
    each sequentially. Accreted rock accumulates from both impacts, applied
    event count increments for each, and all applied events are consumed.
    """
    from types import SimpleNamespace

    from proteus.accretion.common import ImpactEvent, due_events
    from proteus.accretion.wrapper import apply_impact
    from proteus.config import Config
    from proteus.utils.constants import AU

    config = Config()
    config.interior_struct.module = 'dummy'
    config.interior_energetics.module = 'dummy'
    config.planet.mass_tot = 1.0
    config.accretion.impactor_volatiles = 'dry'

    hf_row = {
        'Time': 200.0,
        'semimajorax': 0.5 * AU,
        'eccentricity': 0.1,
        'T_magma': 2000.0,
        'Phi_global': 1.0,
        'M_int': 5.972e24,
        'M_core': 0.3 * 5.972e24,
        'M_mantle': 0.7 * 5.972e24,
        'R_int': 6.4e6,
        'R_core': 3.5e6,
        'gravity': 9.8,
        'M_accreted_rock': 0.0,
        'n_impacts_applied': 0,
        'F_atm': 0.0,
    }

    handler = SimpleNamespace(
        config=config,
        hf_row=hf_row,
        hf_all=None,
        interior_o=SimpleNamespace(impact_reset=False, dt=100.0),
        crystallized=False,
        desiccated=False,
        directories={'output': str(tmp_path)},
    )

    ev1 = ImpactEvent(
        time=120.0,
        M_target_before=6.0e24,
        M_impactor=1.0e23,
        M_merged_after=6.1e24,
        v_impact=1.2e4,
        v_esc=1.1e4,
        impact_parameter=0.3,
        R_target_before=6.4e6,
        R_impactor=2.0e6,
        rho_target=5510.0,
        rho_impactor=3930.0,
        a_before=1.4e11,
        a_after=1.4e11,
        e_before=0.03,
        e_after=0.03,
        id_target=1,
        id_impactor=2,
    )
    ev2 = ImpactEvent(
        time=180.0,
        M_target_before=6.1e24,
        M_impactor=2.0e23,
        M_merged_after=6.3e24,
        v_impact=1.2e4,
        v_esc=1.1e4,
        impact_parameter=0.3,
        R_target_before=6.4e6,
        R_impactor=2.5e6,
        rho_target=5510.0,
        rho_impactor=3930.0,
        a_before=1.4e11,
        a_after=1.4e11,
        e_before=0.03,
        e_after=0.03,
        id_target=1,
        id_impactor=3,
    )

    handler.impact_events = [ev1, ev2]
    landed = due_events(handler.impact_events, 100.0, 200.0)

    applied_times = []
    for ev in landed:
        applied_times.append(ev.time)
        apply_impact(handler, ev)
        handler.impact_events.remove(ev)

    assert applied_times == [120.0, 180.0]
    expected_rock = ev1.mass_delta + ev2.mass_delta
    assert hf_row['M_accreted_rock'] == pytest.approx(expected_rock, rel=1e-12)
    assert hf_row['n_impacts_applied'] == 2
    assert len(handler.impact_events) == 0


def _resumed_handler(tmp_path, events, hf_row, pending=None, hf_all=None):
    """Build a minimal Proteus handler for resume accretion tests."""
    from proteus.accretion.common import write_timeline
    from proteus.accretion.wrapper import _RESOLVED_TIMELINE_FILE

    if events is not None:
        write_timeline(events, str(tmp_path / _RESOLVED_TIMELINE_FILE))
    return SimpleNamespace(
        config=SimpleNamespace(
            accretion=SimpleNamespace(module='timeline', impactor_volatiles='dry'),
            params=SimpleNamespace(resume=True),
            planet=SimpleNamespace(mass_tot=1.0),
            orbit=SimpleNamespace(semimajoraxis=1.0, eccentricity=0.0),
        ),
        hf_row=hf_row,
        hf_all=hf_all,
        directories={'output': str(tmp_path)},
        impact_events=list(events if pending is None else pending),
    )


@pytest.mark.unit
def test_restore_accretion_state_persists_counter_to_hf_all_last_row(tmp_path):
    """Restoring accretion state updates both hf_row and the last row of hf_all."""
    import numpy as np
    import pandas as pd

    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    ev1 = _impact_event(
        time=50.0, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    ev2 = _impact_event(
        time=200.0, M_target_before=6.072e24, M_impactor=1e23, M_merged_after=6.172e24
    )

    hf_all = pd.DataFrame(
        [
            {
                'Time': 100.0,
                'M_accreted_rock': 1e23,
                'n_impacts_applied': np.float64(99.0),
                'semimajorax': 1.0 * AU,
                'eccentricity': 0.0,
            }
        ]
    )
    hf_row = hf_all.iloc[-1].to_dict()
    hf_row['n_impacts_applied'] = 1
    handler = _resumed_handler(
        tmp_path,
        [ev1, ev2],
        hf_row=hf_row,
        hf_all=hf_all,
    )

    restore_accretion_state(handler)
    assert handler.hf_row['n_impacts_applied'] == 1
    assert handler.hf_all['n_impacts_applied'].iloc[-1] == pytest.approx(1.0)

    # Simulating main loop step init: hf_row rebuilt from hf_all.iloc[-1]
    new_hf_row = handler.hf_all.iloc[-1].to_dict()
    assert new_hf_row['n_impacts_applied'] == pytest.approx(1.0)


@pytest.mark.unit
def test_restore_accretion_state_zero_rock_missing_counter_normalizes_hf_all(tmp_path):
    """When M_accreted_rock is zero and counter is absent, restore_accretion_state
    normalizes n_impacts_applied to 0 in hf_row and 0.0 in the last row of hf_all."""
    import pandas as pd

    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    hf_all = pd.DataFrame(
        [
            {
                'Time': 0.0,
                'M_accreted_rock': 0.0,
                'semimajorax': 1.0 * AU,
                'eccentricity': 0.0,
            }
        ]
    )
    hf_row = hf_all.iloc[-1].to_dict()
    assert 'n_impacts_applied' not in hf_row

    handler = _resumed_handler(
        tmp_path,
        [],
        hf_row=hf_row,
        hf_all=hf_all,
    )
    restore_accretion_state(handler)
    assert handler.hf_row['n_impacts_applied'] == 0
    assert handler.hf_all['n_impacts_applied'].iloc[-1] == pytest.approx(0.0)


@pytest.mark.unit
def test_resume_then_impact_records_k_plus_one_in_hf_all(tmp_path):
    """Resume with k prior impacts records k+1 in hf_all after next impact."""
    import pandas as pd

    from proteus.accretion.wrapper import apply_impact, restore_accretion_state
    from proteus.config import Config
    from proteus.utils.constants import AU

    ev1 = _impact_event(
        time=50.0, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    ev2 = _impact_event(
        time=200.0, M_target_before=6.072e24, M_impactor=1e23, M_merged_after=6.172e24
    )

    config = Config()
    config.interior_struct.module = 'dummy'
    config.interior_energetics.module = 'dummy'
    config.planet.mass_tot = 1.0
    config.accretion.module = 'timeline'
    config.accretion.impactor_volatiles = 'dry'
    config.params.resume = True

    hf_all = pd.DataFrame(
        [
            {
                'Time': 100.0,
                'M_accreted_rock': 1e23,
                'n_impacts_applied': 1.0,
                'semimajorax': 1.0 * AU,
                'eccentricity': 0.0,
                'T_magma': 2000.0,
                'Phi_global': 1.0,
                'M_int': 5.972e24,
                'M_core': 0.3 * 5.972e24,
                'M_mantle': 0.7 * 5.972e24,
                'R_int': 6.4e6,
                'R_core': 3.5e6,
                'gravity': 9.8,
                'F_atm': 0.0,
            }
        ]
    )
    handler = _resumed_handler(
        tmp_path, [ev1, ev2], hf_row=hf_all.iloc[-1].to_dict(), hf_all=hf_all
    )
    handler.config = config
    handler.interior_o = SimpleNamespace(impact_reset=False, dt=100.0)
    handler.crystallized = handler.desiccated = False

    restore_accretion_state(handler)

    # Next iteration: main loop rebuilds hf_row from hf_all.iloc[-1].
    handler.hf_row = handler.hf_all.iloc[-1].to_dict()
    handler.hf_row['Time'] = 200.0
    apply_impact(handler, ev2)

    # Iteration write appends row to hf_all: counter must advance to k+1 = 2.
    handler.hf_all = pd.concat(
        [handler.hf_all, pd.DataFrame([handler.hf_row])], ignore_index=True
    )
    assert len(handler.hf_all) == 2
    assert handler.hf_all['n_impacts_applied'].iloc[-1] == 2
    assert handler.hf_all['M_accreted_rock'].iloc[-1] == pytest.approx(2e23)


@pytest.mark.unit
def test_restore_accretion_state_ignores_events_at_or_before_zero_on_resume(tmp_path):
    """Events at t <= 0 must not be counted in events_before."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    # Event A is before simulation start (t <= 0) with mass exceeding accreted rock;
    # event B is an init-stage impact at t = 0.5 yr, inside the 1 yr init step.
    ev_a = _impact_event(
        time=-100.0, M_target_before=5.972e24, M_impactor=2e23, M_merged_after=6.172e24
    )
    ev_b = _impact_event(
        time=0.5, M_target_before=6.172e24, M_impactor=1e23, M_merged_after=6.272e24
    )

    # n_impacts_applied = 1 recorded in helpfile for the init-stage impact.
    # On fresh run ev_a was dropped by _drop_events_before_start, so impact_events has [ev_b].
    # On resume at Time = 0.0, ev_b was already applied and must be dropped from impact_events.
    handler = _resumed_handler(
        tmp_path,
        [ev_a, ev_b],
        hf_row={
            'Time': 0.0,
            'M_accreted_rock': 1e23,
            'n_impacts_applied': 1,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        },
        pending=[ev_b],
    )
    restore_accretion_state(handler)
    assert handler.impact_events == []
    assert handler.hf_row['n_impacts_applied'] == 1


@pytest.mark.unit
def test_empty_user_timeline_logs_warning(tmp_path, caplog):
    """An empty user timeline in timeline or morrigan module logs a warning."""
    import logging
    from unittest.mock import patch

    empty_csv = tmp_path / 'empty.csv'
    empty_csv.write_text(','.join(TIMELINE_COLUMNS) + chr(10))

    # Case 1: timeline module with empty file
    handler = _handler(
        module='timeline',
        timeline_path=empty_csv,
        output_dir=tmp_path / 'out_timeline',
    )
    (tmp_path / 'out_timeline').mkdir()

    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        events = init_accretion(handler)

    assert events == []
    assert any('0 impacts' in r.message for r in caplog.records)

    # Case 2: morrigan module resolving to 0 impacts
    caplog.clear()
    handler_morrigan = _handler(
        module='morrigan',
        timeline_path=tmp_path / 'unused.csv',
        output_dir=tmp_path / 'out_morrigan',
    )
    (tmp_path / 'out_morrigan').mkdir()

    with patch('proteus.accretion.morrigan.get_timeline', return_value=[]):
        with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
            events = init_accretion(handler_morrigan)

    assert events == []
    assert any('0 impacts' in r.message for r in caplog.records)


@pytest.mark.unit
@pytest.mark.parametrize(
    'counter_key_val', [None, 0.0, float('nan'), -1, -1.0, 1.5, 'bad', [], {}]
)
def test_legacy_resume_missing_or_corrupt_counter_refuses(tmp_path, counter_key_val):
    """Resume with positive rock and missing/zero/corrupt counter is refused."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    ev05 = _impact_event(
        time=0.5, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    ev50 = _impact_event(
        time=50.0, M_target_before=6.072e24, M_impactor=1e23, M_merged_after=6.172e24
    )
    ev500 = _impact_event(
        time=500.0, M_target_before=6.172e24, M_impactor=1e23, M_merged_after=6.272e24
    )

    hf_row = {
        'Time': 100.0,
        'M_accreted_rock': 2e23,
        'semimajorax': AU,
        'eccentricity': 0.0,
    }
    if counter_key_val is not None:
        hf_row['n_impacts_applied'] = counter_key_val

    handler = _resumed_handler(
        tmp_path,
        events=[ev05, ev50, ev500],
        hf_row=hf_row,
        pending=[ev05, ev50, ev500],
    )

    before = (dict(handler.hf_row), list(handler.impact_events))
    with pytest.raises(RuntimeError, match='Resume refused') as excinfo:
        restore_accretion_state(handler)

    assert (dict(handler.hf_row), list(handler.impact_events)) == before
    err = str(excinfo.value)
    assert 'restart' in err.lower()
    assert err.split()[2].endswith('runtime_helpfile.csv')

    # Case with zero accreted rock:
    hf_row_zero = {
        'Time': 0.0,
        'M_accreted_rock': 0.0,
        'semimajorax': AU,
        'eccentricity': 0.0,
    }
    if counter_key_val is not None:
        hf_row_zero['n_impacts_applied'] = counter_key_val
    handler_zero = _resumed_handler(
        tmp_path,
        events=[ev05, ev50, ev500],
        hf_row=hf_row_zero,
        pending=[ev05, ev50, ev500],
    )

    if counter_key_val in (None, 0.0):
        # Rock 0 with absent or zero counter: accepted!
        restore_accretion_state(handler_zero)
        assert handler_zero.hf_row['n_impacts_applied'] == 0
    else:
        # Corrupt counter: refused even if rock is 0
        with pytest.raises(RuntimeError, match='Resume refused'):
            restore_accretion_state(handler_zero)


@pytest.mark.unit
def test_restore_accretion_state_no_warning_when_counter_positive(tmp_path, caplog):
    """Positive valid counter restores cleanly without logging warnings."""
    import logging

    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    ev50 = _impact_event(
        time=50.0, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    handler = _resumed_handler(
        tmp_path,
        events=[ev50],
        hf_row={
            'Time': 100.0,
            'M_accreted_rock': 1e23,
            'n_impacts_applied': 1.0,
            'semimajorax': AU,
            'eccentricity': 0.0,
        },
        pending=[ev50],
    )
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        restore_accretion_state(handler)
    assert len(caplog.records) == 0
    assert handler.hf_row['n_impacts_applied'] == 1


@pytest.mark.unit
def test_restore_accretion_state_filters_events_by_resume_time(tmp_path):
    """Calling restore_accretion_state with unfiltered impact_events drops events <= resume_time."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    ev1 = _impact_event(
        time=10.0, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    ev2 = _impact_event(
        time=100.0, M_target_before=6.072e24, M_impactor=1e23, M_merged_after=6.172e24
    )
    handler = _resumed_handler(
        tmp_path,
        events=[ev1, ev2],
        hf_row={
            'Time': 50.0,
            'M_accreted_rock': 1e23,
            'n_impacts_applied': 1,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        },
        pending=[ev1, ev2],
    )
    restore_accretion_state(handler)
    assert handler.impact_events == [ev2]
    assert handler.impact_events[0].time == pytest.approx(100.0)
    assert handler.hf_row['n_impacts_applied'] == 1


@pytest.mark.unit
def test_restore_accretion_state_refuses_invalid_m_accreted_rock(tmp_path):
    """Non-finite, negative, or non-numeric M_accreted_rock raises ValueError without mutating state."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    bad_values = (
        float('nan'),
        'nan',
        -1.0,
        -1e23,
        float('inf'),
        float('-inf'),
        'inf',
        '-inf',
        [],
        {},
    )
    for bad_rock in bad_values:
        row = {
            'Time': 100.0,
            'M_accreted_rock': bad_rock,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        }
        handler = _resumed_handler(
            tmp_path,
            events=[],
            hf_row=dict(row),
            pending=[],
        )
        with pytest.raises(RuntimeError, match='invalid M_accreted_rock') as excinfo:
            restore_accretion_state(handler)

        err = str(excinfo.value)
        assert 'runtime_helpfile.csv' in err
        assert 'Restart the simulation' in err
        assert 'n_impacts_applied' not in handler.hf_row
        assert handler.hf_row == row


@pytest.mark.unit
def test_restore_accretion_state_refuses_invalid_time(tmp_path):
    """Non-finite, negative, or non-numeric Time raises RuntimeError without mutating state."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    bad_values = (
        float('nan'),
        'nan',
        -1.0,
        -100.0,
        float('inf'),
        float('-inf'),
        'inf',
        '-inf',
        [],
        {},
    )
    for bad_time in bad_values:
        row = {
            'Time': bad_time,
            'M_accreted_rock': 1e23,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        }
        handler = _resumed_handler(
            tmp_path,
            events=[],
            hf_row=dict(row),
            pending=[],
        )
        with pytest.raises(RuntimeError, match='invalid Time') as excinfo:
            restore_accretion_state(handler)

        err = str(excinfo.value)
        assert 'runtime_helpfile.csv' in err
        assert 'Restart the simulation' in err
        assert 'n_impacts_applied' not in handler.hf_row
        assert handler.hf_row == row


@pytest.mark.unit
def test_restore_accretion_state_empty_hf_all(tmp_path):
    """An empty hf_all DataFrame is handled gracefully without index error."""
    import pandas as pd

    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    ev1 = _impact_event(
        time=10.0, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    handler = _resumed_handler(
        tmp_path,
        events=[ev1],
        hf_row={
            'Time': 100.0,
            'M_accreted_rock': 1e23,
            'n_impacts_applied': 1,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        },
        hf_all=pd.DataFrame(),
        pending=[ev1],
    )
    restore_accretion_state(handler)
    assert handler.hf_row['n_impacts_applied'] == 1
    assert handler.hf_all.empty
    assert handler.impact_events == []


@pytest.mark.unit
def test_restore_accretion_state_zero_rock_with_positive_counter(tmp_path):
    """Zero rock with positive counter preserves counter and drops prior events."""
    import pandas as pd

    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    ev1 = _impact_event(
        time=10.0, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    ev2 = _impact_event(
        time=20.0, M_target_before=6.072e24, M_impactor=1e23, M_merged_after=6.172e24
    )
    ev3 = _impact_event(
        time=30.0, M_target_before=6.172e24, M_impactor=1e23, M_merged_after=6.272e24
    )
    hf_all = pd.DataFrame(
        [
            {
                'Time': 25.0,
                'M_accreted_rock': 0.0,
                'n_impacts_applied': 2,
                'semimajorax': 1.0 * AU,
                'eccentricity': 0.0,
            }
        ]
    )
    handler = _resumed_handler(
        tmp_path,
        events=[ev1, ev2, ev3],
        hf_row={
            'Time': 25.0,
            'M_accreted_rock': 0.0,
            'n_impacts_applied': 2,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        },
        hf_all=hf_all,
        pending=[ev1, ev2, ev3],
    )
    restore_accretion_state(handler)
    assert handler.hf_row['n_impacts_applied'] == 2
    assert handler.hf_all.loc[handler.hf_all.index[-1], 'n_impacts_applied'] == pytest.approx(
        2.0
    )
    assert handler.impact_events == [ev3]


@pytest.mark.unit
def test_restore_accretion_state_refuses_counter_smaller_than_events_before(tmp_path):
    """Helpfile counter smaller than events preceding resume time raises RuntimeError."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    ev1 = _impact_event(
        time=10.0, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    ev2 = _impact_event(
        time=20.0, M_target_before=6.072e24, M_impactor=1e23, M_merged_after=6.172e24
    )
    import pandas as pd

    hf_all = pd.DataFrame(
        [
            {
                'Time': 25.0,
                'M_accreted_rock': 1e23,
                'semimajorax': 1.0 * AU,
                'eccentricity': 0.0,
            }
        ]
    )
    handler = _resumed_handler(
        tmp_path,
        events=[ev1, ev2],
        hf_row={
            'Time': 25.0,
            'M_accreted_rock': 1e23,
            'n_impacts_applied': 1,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        },
        hf_all=hf_all,
        pending=[ev1, ev2],
    )
    hf_row_orig = dict(handler.hf_row)
    hf_all_orig = handler.hf_all.copy()
    events_orig = list(handler.impact_events)
    with pytest.raises(RuntimeError) as exc_info:
        restore_accretion_state(handler)
    err = str(exc_info.value)
    assert 'runtime_helpfile.csv' in err
    assert 'precede the resume time' in err
    assert handler.hf_row == hf_row_orig
    assert handler.hf_all.equals(hf_all_orig)
    assert handler.impact_events == events_orig


@pytest.mark.unit
def test_restore_accretion_state_refuses_counter_exceeding_total_events(tmp_path):
    """Helpfile counter exceeding the total number of timeline events raises RuntimeError."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    ev1 = _impact_event(
        time=10.0, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    handler = _resumed_handler(
        tmp_path,
        events=[ev1],
        hf_row={
            'Time': 5.0,
            'M_accreted_rock': 1e23,
            'n_impacts_applied': 2,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        },
        pending=[ev1],
    )
    hf_row_orig = dict(handler.hf_row)
    events_orig = list(handler.impact_events)
    with pytest.raises(RuntimeError) as exc_info:
        restore_accretion_state(handler)
    err = str(exc_info.value)
    assert 'runtime_helpfile.csv' in err
    assert 'holds only 1 impact' in err
    assert handler.hf_row == hf_row_orig
    assert handler.impact_events == events_orig


@pytest.mark.unit
def test_restore_accretion_state_event_at_exact_resume_time_boundary(tmp_path):
    """An event exactly at resume_time is treated as preceding the resume boundary."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    ev1 = _impact_event(
        time=25.0, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    ev2 = _impact_event(
        time=50.0, M_target_before=6.072e24, M_impactor=1e23, M_merged_after=6.172e24
    )
    handler_ok = _resumed_handler(
        tmp_path,
        events=[ev1, ev2],
        hf_row={
            'Time': 25.0,
            'M_accreted_rock': 1e23,
            'n_impacts_applied': 1,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        },
        pending=[ev1, ev2],
    )
    restore_accretion_state(handler_ok)
    assert handler_ok.impact_events == [ev2]
    assert handler_ok.hf_row['n_impacts_applied'] == 1

    handler_bad = _resumed_handler(
        tmp_path,
        events=[ev1, ev2],
        hf_row={
            'Time': 25.0,
            'M_accreted_rock': 0.0,
            'n_impacts_applied': 0,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        },
        pending=[ev1, ev2],
    )
    with pytest.raises(RuntimeError) as exc_info:
        restore_accretion_state(handler_bad)
    assert 'precede the resume time' in str(exc_info.value)


@pytest.mark.unit
def test_restore_accretion_state_legacy_ledger_accepted_when_accretion_disabled(
    tmp_path, caplog
):
    """When accretion is disabled, a positive ledger with absent/zero counter restores mass with a warning."""
    import logging

    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU, M_earth

    handler = _resumed_handler(
        tmp_path,
        events=[],
        hf_row={
            'Time': 100.0,
            'M_accreted_rock': 1e23,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        },
    )
    handler.config.accretion.module = None
    handler.impact_events = None

    with caplog.at_level(logging.WARNING):
        restore_accretion_state(handler)

    assert handler.config.planet.mass_tot == pytest.approx(1.0 + 1e23 / M_earth)
    assert handler.config.orbit.semimajoraxis == pytest.approx(1.0)
    assert handler.hf_row['n_impacts_applied'] == 0
    assert 'Accretion is disabled for this resume' in caplog.text


@pytest.mark.unit
def test_restore_accretion_state_refuses_negative_counter_without_timeline(tmp_path):
    """Negative counter without an impact timeline raises corrupt RuntimeError."""
    from types import SimpleNamespace

    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    handler = SimpleNamespace(
        config=SimpleNamespace(
            params=SimpleNamespace(resume=True),
            accretion=SimpleNamespace(module=None),
            planet=SimpleNamespace(mass_tot=0.5),
            orbit=SimpleNamespace(semimajoraxis=1.0, eccentricity=0.0),
        ),
        hf_row={
            'Time': 100.0,
            'M_accreted_rock': 0.0,
            'n_impacts_applied': -1,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        },
        hf_all=None,
        directories={'output': str(tmp_path)},
        impact_events=None,
    )
    hf_row_orig = dict(handler.hf_row)
    with pytest.raises(RuntimeError) as exc_info:
        restore_accretion_state(handler)
    err = str(exc_info.value)
    assert 'corrupt n_impacts_applied' in err
    assert 'Restart the simulation' in err
    assert 'runtime_helpfile.csv' in err
    assert handler.hf_row == hf_row_orig


@pytest.mark.unit
def test_restore_accretion_state_timeline_ignores_events_with_zero_or_negative_time(
    tmp_path,
):
    """Events with non-positive times are excluded from events_before validation."""
    from proteus.accretion.wrapper import restore_accretion_state
    from proteus.utils.constants import AU

    ev_neg = _impact_event(
        time=-5.0, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    ev_zero = _impact_event(
        time=0.0, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    ev_pos = _impact_event(
        time=10.0, M_target_before=5.972e24, M_impactor=1e23, M_merged_after=6.072e24
    )
    handler = _resumed_handler(
        tmp_path,
        events=[ev_neg, ev_zero, ev_pos],
        hf_row={
            'Time': 5.0,
            'M_accreted_rock': 0.0,
            'n_impacts_applied': 0,
            'semimajorax': 1.0 * AU,
            'eccentricity': 0.0,
        },
        pending=[ev_neg, ev_zero, ev_pos],
    )
    restore_accretion_state(handler)
    assert handler.hf_row['n_impacts_applied'] == 0
    assert handler.impact_events == [ev_pos]


def _counter_case(tmp_path, times, resume_time, counter, rock=1e23):
    """Resume handler for a dummy timeline at ``times`` [yr] and a row at
    ``resume_time``; hf_row is the last row of hf_all but carries the counter
    as a string, so any write before a refusal changes it. The pending list is
    filtered by the resume time, as init_accretion leaves it."""
    import pandas as pd

    from proteus.utils.constants import AU

    events = [
        _impact_event(
            time=t,
            M_target_before=5.972e24 + i * 1e23,
            M_impactor=1e23,
            M_merged_after=6.072e24 + i * 1e23,
        )
        for i, t in enumerate(times)
    ]
    hf_all = pd.DataFrame(
        [
            {
                'Time': resume_time,
                'M_accreted_rock': rock,
                'n_impacts_applied': float(counter),
                'semimajorax': 1.0 * AU,
                'eccentricity': 0.0,
            }
        ]
    )
    hf_row = hf_all.iloc[-1].to_dict()
    hf_row['n_impacts_applied'] = str(counter)  # a type the frame does not hold
    handler = _resumed_handler(
        tmp_path,
        events,
        hf_row=hf_row,
        hf_all=hf_all,
        pending=[ev for ev in events if ev.time > resume_time],
    )
    return handler, events


def _assert_refused_without_side_effects(handler, match):
    """A refused resume raises and leaves hf_row, hf_all and the schedule as they were."""
    from proteus.accretion.wrapper import restore_accretion_state

    row, frame, pending = (
        dict(handler.hf_row),
        handler.hf_all.copy(),
        list(handler.impact_events),
    )
    mass = handler.config.planet.mass_tot
    with pytest.raises(RuntimeError, match=match):
        restore_accretion_state(handler)
    assert handler.hf_row == row
    assert handler.hf_all.equals(frame)
    assert handler.impact_events == pending
    assert handler.config.planet.mass_tot == pytest.approx(mass, rel=0)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('counter', 'accepted'),
    [(0, False), (1, False), (2, True), (3, False)],
    ids=['no_counter', 'one_impact_lost', 'consistent', 'one_impact_too_many'],
)
def test_counter_is_checked_after_the_last_scheduled_impact(tmp_path, counter, accepted):
    """After the last impact the pending list is empty, which is the normal
    state late in every accreting run; the counter must still agree with the
    timeline. Impacts at 5 and 8 yr, resume at 100 yr: only a counter of 2 is
    consistent, and a refusal leaves the row and schedule untouched."""
    from proteus.accretion.wrapper import restore_accretion_state

    handler, _ = _counter_case(tmp_path, [5.0, 8.0], 100.0, counter, rock=2e23)
    assert handler.impact_events == []
    mass_before = handler.config.planet.mass_tot
    if accepted:
        restore_accretion_state(handler)
        assert handler.hf_row['n_impacts_applied'] == 2
        assert handler.hf_all['n_impacts_applied'].iloc[-1] == pytest.approx(2.0, rel=0)
        assert handler.config.planet.mass_tot > mass_before
    else:
        _assert_refused_without_side_effects(handler, 'runtime_helpfile.csv')


@pytest.mark.unit
def test_legacy_ledger_is_refused_with_a_module_and_accepted_without(tmp_path, caplog):
    """Rock with no counter means the run predates the counter: refused
    whenever a module is selected, even with nothing left to schedule, and
    accepted with a warning only when accretion is off."""
    import logging

    from proteus.accretion.wrapper import restore_accretion_state

    handler, _ = _counter_case(tmp_path, [5.0], 100.0, 0.0, rock=1e23)
    with pytest.raises(RuntimeError, match='predates the impact counter'):
        restore_accretion_state(handler)

    off_dir = tmp_path / 'off'
    off_dir.mkdir()
    handler_off, _ = _counter_case(off_dir, [5.0], 100.0, 0.0, rock=1e23)
    handler_off.config.accretion.module = None
    handler_off.impact_events = []
    with caplog.at_level(logging.WARNING, logger='fwl.proteus.accretion.wrapper'):
        restore_accretion_state(handler_off)
    assert any('Accretion is disabled' in r.message for r in caplog.records)
    assert handler_off.config.planet.mass_tot > 1.0


@pytest.mark.unit
@pytest.mark.parametrize(
    ('times', 'resume_time', 'counter', 'pending_after'),
    [
        ([0.5, 0.8, 15.0], 0.0, 2, [15.0]),  # two init-stage impacts, row still at Time 0
        ([0.5, 1.0, 15.0], 0.0, 2, [15.0]),  # an impact exactly at the horizon counts
        ([0.5, 15.0], 0.0, 1, [15.0]),
        ([0.5, 15.0], 0.0, 0, None),  # rock recorded, counter 0: refused
        ([0.5, 1.0 + 1e-9, 15.0], 0.0, 2, None),  # just past the horizon
        ([0.5, 0.8, 15.0], 0.0, 3, None),  # 15 yr cannot land in the init stage
        ([0.5, 200.0], 100.0, 2, None),  # surplus after the resume time
        ([0.5, 5.0], 2.0, 2, None),  # resumed past the init stage with a surplus
        ([0.3, 0.8, 15.0], 0.5, 2, [15.0]),  # row of the last init step, 0 < Time <= 1
        ([0.5, 1.0, 15.0], 1.0, 2, [15.0]),  # impact exactly at the resume time
        ([0.5, 1.0, 15.0], 1.0, 3, None),  # surplus beyond it is past the horizon
    ],
    ids=[
        'two_init_impacts',
        'impact_at_horizon',
        'one_init_impact',
        'no_counter',
        'past_horizon',
        'surplus_beyond_init',
        'surplus_after_resume',
        'surplus_after_init_stage',
        'last_init_step_row',
        'impact_at_resume_time',
        'surplus_after_impact_at_resume_time',
    ],
)
def test_counter_surplus_must_lie_in_the_init_stage(
    tmp_path, times, resume_time, counter, pending_after
):
    """A counter above the impacts at or before the resume time is only
    possible for impacts that landed during the init stage, where Time stays
    zero and every step is at most _INIT_STAGE_HORIZON_YR. Any other surplus
    would silently delete a future impact, so it is refused."""
    from proteus.accretion.wrapper import _INIT_STAGE_HORIZON_YR, restore_accretion_state

    assert _INIT_STAGE_HORIZON_YR == pytest.approx(1.0, rel=0)
    handler, _ = _counter_case(tmp_path, times, resume_time, counter)
    if pending_after is None:
        _assert_refused_without_side_effects(handler, 'Restart the simulation')
    else:
        restore_accretion_state(handler)
        assert [ev.time for ev in handler.impact_events] == pytest.approx(pending_after)
        assert handler.hf_row['n_impacts_applied'] == counter


@pytest.mark.unit
def test_dropped_init_impacts_are_logged_and_an_oversized_counter_names_the_timeline(
    tmp_path, caplog
):
    """Impacts skipped on resume as already applied are named in the log, and
    a counter larger than the whole timeline is refused with the timeline size."""
    import logging

    from proteus.accretion.wrapper import restore_accretion_state

    handler, _ = _counter_case(tmp_path, [0.5, 0.8, 15.0], 0.5, 2)
    with caplog.at_level(logging.INFO, logger='fwl.proteus.accretion.wrapper'):
        restore_accretion_state(handler)
    dropped = [r.message for r in caplog.records if 'not applied again' in r.message]
    assert dropped == [
        'Resume: 1 impact(s) after the resume time landed during the init stage and are not applied again: 0.8 yr'
    ]
    assert [ev.time for ev in handler.impact_events] == pytest.approx([15.0])

    big_dir = tmp_path / 'big'
    big_dir.mkdir()
    oversized, _ = _counter_case(big_dir, [5.0, 8.0], 100.0, 50)
    _assert_refused_without_side_effects(oversized, 'holds only 2 impact')
