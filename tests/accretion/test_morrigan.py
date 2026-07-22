"""Tests for the Morrigan giant-impact module wrapper.

This file targets accretion/morrigan.py (require_morrigan, select_planet,
build_parameters, get_timeline). The wrapper turns a PROTEUS
configuration into a dynamical-model run and reduces the resulting system
to one body's impact history, so what it must guarantee is that the unit
conversions into the model are right, that the survivor selection picks
the body the configuration asks for, and that a missing package is
reported rather than crashed through.

See testing standards in docs/How-to/testing.md and
docs/Explanations/test_framework.md for required structure, speed, and
physics validity.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from proteus.accretion import morrigan as backend
from proteus.utils.constants import AU, M_earth

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _survivor(ident, mass_initial, a_initial, mass_final, a_final):
    """Build a survivor record in the shape the dynamical model returns."""
    return {
        'id': ident,
        'mass_initial': mass_initial * M_earth,
        'a_initial': a_initial * AU,
        'mass_final': mass_final * M_earth,
        'a_final': a_final * AU,
    }


# A system whose survivors differ in mass and orbit, so each selector has
# a distinct correct answer and no two selectors can be confused.
_SURVIVORS = [
    _survivor(0, 0.3, 0.10, 0.3, 0.10),
    _survivor(1, 0.8, 0.50, 2.4, 0.62),
    _survivor(2, 1.0, 1.00, 1.1, 0.95),
    _survivor(3, 0.5, 2.00, 1.6, 2.20),
]


def _config(selector='match_config', selector_value=None, mass_tot=1.0, semimajoraxis=1.0):
    """Build the minimal config shape the Morrigan wrapper reads."""
    return SimpleNamespace(
        accretion=SimpleNamespace(
            module='morrigan',
            time_offset=0.0,
            morrigan=SimpleNamespace(
                selector=selector,
                selector_value=selector_value,
                seed=7,
                num_planets=4,
                masses=[],
                mass_equal=0.5,
                eccentricity_init=0.01,
                inner_edge=0.1,
                spacing=10.0,
                density=5500.0,
                impact_angle=45.0,
                evolution_time=1.0,
                inner_cutoff=0.005,
            ),
        ),
        planet=SimpleNamespace(mass_tot=mass_tot),
        orbit=SimpleNamespace(semimajoraxis=semimajoraxis),
        star=SimpleNamespace(mass=1.0),
    )


@pytest.mark.unit
def test_missing_package_is_reported_with_an_install_hint(monkeypatch):
    """An unavailable dynamical model explains itself instead of crashing.

    The package is an optional dependency, so selecting the backend
    without it must produce an actionable message rather than a bare
    ModuleNotFoundError from an import deep in the call stack. A package
    that is present but too old to expose the entry point is the other
    realistic failure and must be distinguished from absence.
    """
    monkeypatch.setattr(backend, 'morrigan', None, raising=False)
    with pytest.raises(ImportError, match='requires the morrigan package'):
        backend.require_morrigan()

    # Installed but without the entry point: a different, specific message.
    monkeypatch.setattr(backend, 'morrigan', SimpleNamespace(), raising=False)
    with pytest.raises(ImportError, match='does not expose'):
        backend.require_morrigan()

    # Installed and complete: returned for use.
    complete = SimpleNamespace(run_system=lambda **kw: None)
    monkeypatch.setattr(backend, 'morrigan', complete, raising=False)
    assert backend.require_morrigan() is complete


@pytest.mark.unit
def test_each_selector_picks_its_own_body():
    """The four selectors resolve to four different survivors here.

    Selection decides whose impact history the whole run follows, so a
    selector wired to the wrong field would silently simulate a different
    planet. The system is built so the most massive body, the body nearest
    a target orbit, the body matching the configuration, and an explicitly
    named body are all distinct; any two selectors returning the same
    body would mean one of them is not reading what it claims to.
    """
    # Most massive at the end of the run: body 1 at 2.4 M_earth.
    chosen = backend.select_planet(_SURVIVORS, _config(selector='mass'))
    assert chosen['id'] == 1

    # Nearest a 2.2 AU target orbit: body 3, not the most massive one.
    chosen = backend.select_planet(
        _SURVIVORS, _config(selector='semimajoraxis', selector_value=2.2)
    )
    assert chosen['id'] == 3

    # Explicitly named body wins regardless of mass or orbit.
    chosen = backend.select_planet(_SURVIVORS, _config(selector='id', selector_value=0))
    assert chosen['id'] == 0

    # Closest to a 1 M_earth, 1 AU configured planet at the start of the
    # run: body 2, which matches both quantities exactly.
    chosen = backend.select_planet(
        _SURVIVORS, _config(selector='match_config', mass_tot=1.0, semimajoraxis=1.0)
    )
    assert chosen['id'] == 2


@pytest.mark.unit
def test_match_config_weighs_mass_and_orbit_comparably():
    """Matching compares relative offsets, so neither quantity dominates.

    Masses are around 1e24 kg and orbits around 1e11 m, so an absolute
    distance in SI would be decided by mass alone and the orbit would
    never matter. Holding the mass target fixed and moving only the orbit
    target must still change the answer; that is what discriminates a
    relative metric from an absolute one.
    """
    # Same 0.5 M_earth mass target, two different orbit targets.
    near = backend.select_planet(
        _SURVIVORS, _config(selector='match_config', mass_tot=0.5, semimajoraxis=2.0)
    )
    far = backend.select_planet(
        _SURVIVORS, _config(selector='match_config', mass_tot=0.5, semimajoraxis=0.1)
    )

    assert near['id'] == 3
    assert far['id'] == 0
    assert near['id'] != far['id']


@pytest.mark.unit
def test_selection_fails_loudly_on_an_impossible_request():
    """Selecting a body that is not there is an error, not an empty run.

    A run that left no survivors, or a named body that was consumed,
    would otherwise produce an empty impact history that looks exactly
    like a successful run with no impacts.
    """
    with pytest.raises(ValueError, match='no surviving bodies'):
        backend.select_planet([], _config(selector='mass'))

    with pytest.raises(ValueError, match='did not survive'):
        backend.select_planet(_SURVIVORS, _config(selector='id', selector_value=99))

    # The error names what is available, so the config can be fixed.
    with pytest.raises(ValueError, match=r'\[0, 1, 2, 3\]'):
        backend.select_planet(_SURVIVORS, _config(selector='id', selector_value=99))


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_parameters_are_converted_into_model_units():
    """Configuration units are converted once, on the way into the model.

    The configuration states masses in Earth masses and orbits in AU
    because that is what a user reasons in, while the dynamical model
    works in SI. Getting a conversion wrong would shift the whole system
    by 24 orders of magnitude in mass or 11 in length, so each converted
    quantity is pinned against its hand-computed SI value.
    """
    config = _config()
    config.accretion.morrigan.masses = [0.5, 1.5, 2.0]

    params = backend.build_parameters(config)

    assert params['masses'] == pytest.approx([0.5 * M_earth, 1.5 * M_earth, 2.0 * M_earth])
    assert params['inner_edge'] == pytest.approx(0.1 * AU)
    assert params['inner_cutoff'] == pytest.approx(0.005 * AU)

    # Masses must be far above the Earth-mass number they came from, which
    # is what an omitted conversion would leave behind.
    assert min(params['masses']) > 1.0e23

    # Dimensionless and already-SI quantities pass through untouched.
    assert params['spacing'] == pytest.approx(10.0)
    assert params['density'] == pytest.approx(5500.0)
    assert params['seed'] == 7

    # The host star comes from the star section, not the accretion one, so
    # the dynamical model and the rest of the run cannot disagree.
    config.star.mass = 0.4
    assert backend.build_parameters(config)['stellar_mass'] == pytest.approx(0.4)


@pytest.mark.unit
def test_equal_mass_system_expands_to_one_entry_per_embryo():
    """An empty mass list is the documented equal-mass initial condition.

    The alternative reading, passing an empty list straight through, would
    start a system with no bodies at all. The expansion must produce
    exactly num_planets entries, all at the configured value.
    """
    config = _config()
    config.accretion.morrigan.masses = []
    config.accretion.morrigan.num_planets = 6
    config.accretion.morrigan.mass_equal = 0.75

    params = backend.build_parameters(config)

    assert len(params['masses']) == 6
    assert params['masses'] == pytest.approx([0.75 * M_earth] * 6)

    # An explicit list is used verbatim and is not overwritten by
    # mass_equal, which is the opposite failure.
    config.accretion.morrigan.masses = [0.2, 0.4]
    config.accretion.morrigan.num_planets = 2
    assert backend.build_parameters(config)['masses'] == pytest.approx(
        [0.2 * M_earth, 0.4 * M_earth]
    )


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_generated_timeline_is_selected_ordered_and_validated(monkeypatch):
    """A model run is reduced to one body's validated, ordered history.

    Three things must hold together: only the selected body's impacts are
    kept, they are sorted in time even if the model reports them
    otherwise, and the same physical validation applied to a file-loaded
    timeline is applied here. Skipping validation on the generated path
    would let an inconsistent model result reach the main loop through a
    side door.
    """
    impacts = {
        1: [
            {
                'time': 5.0e5,
                'M_target_before': 6.64e24,
                'M_impactor': 1.0e23,
                'M_merged_after': 6.74e24,
                'v_impact': 1.2e4,
                'v_esc': 1.1e4,
                'impact_parameter': 0.3,
                'R_target_before': 6.4e6,
                'R_impactor': 2.0e6,
                'rho_target': 5510.0,
                'rho_impactor': 3930.0,
                'a_before': 1.4e11,
                'a_after': 1.35e11,
                'e_after': 0.02,
                'id_target': 1,
                'id_impactor': 7,
            },
            {
                'time': 1.0e5,
                'M_target_before': 6.0e24,
                'M_impactor': 6.4e23,
                'M_merged_after': 6.64e24,
                'v_impact': 1.3e4,
                'v_esc': 1.15e4,
                'impact_parameter': 0.7,
                'R_target_before': 6.371e6,
                'R_impactor': 3.39e6,
                'rho_target': 5510.0,
                'rho_impactor': 3930.0,
                'a_before': 1.496e11,
                'a_after': 1.4e11,
                'e_after': 0.05,
                'id_target': 1,
                'id_impactor': 4,
            },
        ],
        2: [],
    }
    fake = SimpleNamespace(
        run_system=lambda **kw: {'survivors': _SURVIVORS, 'impacts': impacts}
    )
    monkeypatch.setattr(backend, 'morrigan', fake, raising=False)

    config = _config(selector='mass')  # resolves to body 1
    events = backend.get_timeline(config)

    # Reported out of order, returned in order.
    assert [e.time for e in events] == [1.0e5, 5.0e5]
    assert events[0].id_impactor == 4

    # The chain is continuous, which is what validation enforces.
    assert events[1].M_target_before == pytest.approx(events[0].M_merged_after)

    # The offset is applied on this path too.
    config.accretion.time_offset = 1.0e6
    assert backend.get_timeline(config)[0].time == pytest.approx(1.0e5 + 1.0e6)

    # A physically inconsistent model result is rejected, not passed on.
    impacts[1][1]['M_merged_after'] = 9.9e24
    config.accretion.time_offset = 0.0
    with pytest.raises(ValueError, match='does not close'):
        backend.get_timeline(config)
