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


def _handler(module=None, timeline_path=None, time_offset=0.0, time_start=0.0):
    """Build the minimal Proteus handler shape init_accretion reads."""
    return SimpleNamespace(
        config=SimpleNamespace(
            accretion=SimpleNamespace(
                module=module,
                time_offset=time_offset,
                dummy=SimpleNamespace(
                    timeline_path=None if timeline_path is None else str(timeline_path)
                ),
            )
        ),
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
    handler = _handler(module='dummy', timeline_path=_timeline_file(tmp_path / 't.csv'))

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
    handler = _handler(module='dummy', timeline_path=path, time_start=2.0e5)

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
    boundary = _handler(module='dummy', timeline_path=path, time_start=1.0e5)
    assert [e.time for e in init_accretion(boundary)] == [5.0e5]

    # Shifting the timeline forward brings both impacts back into range,
    # which is the documented remedy.
    shifted = _handler(module='dummy', timeline_path=path, time_offset=3.0e5, time_start=2.0e5)
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
        e_after=0.05,
        id_target=1,
        id_impactor=4,
    )
    base.update(overrides)
    return ImpactEvent(**base)


def _impact_handler(
    mass_tot=1.0, semimajoraxis=0.5, eccentricity=0.1, tsurf_init=4000.0, crystallized=False
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

    The semi-major axis moves by the impact's proportional change and the
    eccentricity takes its post-impact value. Both the configuration, which
    pins the orbit when tides are off, and the running row, which the tidal
    evolution carries forward when tides are on, must be written, or the
    jump would be lost under one of the two orbit modes.
    """
    from proteus.accretion.wrapper import apply_impact
    from proteus.utils.constants import AU

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler(semimajoraxis=0.5, eccentricity=0.1)
    # a_after / a_before = 1.4e11 / 1.4e11 scaled: choose a clean 1.2 ratio.
    event = _impact_event(a_before=1.0e11, a_after=1.2e11, e_after=0.03)
    ratio = 1.2

    apply_impact(handler, event)

    assert handler.config.orbit.semimajoraxis == pytest.approx(0.5 * ratio, rel=1e-12)
    assert handler.hf_row['semimajorax'] == pytest.approx(0.5 * AU * ratio, rel=1e-12)
    # Config (AU) and row (metres) describe the same orbit after the jump.
    assert handler.hf_row['semimajorax'] / AU == pytest.approx(
        handler.config.orbit.semimajoraxis, rel=1e-12
    )
    # Eccentricity takes the post-impact value in both places.
    assert handler.config.orbit.eccentricity == pytest.approx(0.03, rel=1e-12)
    assert handler.hf_row['eccentricity'] == pytest.approx(0.03, rel=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_a_grazing_head_on_impact_leaves_the_orbit_circular(monkeypatch):
    """A zero post-impact eccentricity is a valid boundary and is applied.

    The eccentricity is written directly, so the circular limit must come
    through as exactly zero rather than being clamped away, and the
    semi-major axis still moves by its ratio independently of it.
    """
    from proteus.accretion.wrapper import apply_impact

    monkeypatch.setattr(
        'proteus.interior_energetics.wrapper.solve_structure', lambda *a, **k: None
    )

    handler = _impact_handler(semimajoraxis=1.0, eccentricity=0.2)
    event = _impact_event(a_before=1.0e11, a_after=1.0e11, e_after=0.0)
    apply_impact(handler, event)

    assert handler.config.orbit.eccentricity == 0.0
    assert handler.hf_row['eccentricity'] == 0.0
    # Equal before/after semi-major axis is a unit ratio, so the orbit size
    # is unchanged while the eccentricity is reset.
    assert handler.config.orbit.semimajoraxis == pytest.approx(1.0, rel=1e-12)
