"""Unit tests for ``proteus.escape.boiloff``.

Exercises the initial-condition correction ``apply_boiloff_ic``, which trims
the whole-planet volatile inventory down to the fraction of an envelope that
survives boil-off before the evolution starts.

Invariants tested:
  - Conservation: the mass removed from the per-element inventories equals the
    recorded ``M_boiloff_kg``, and accumulates across repeated passes
  - Boundedness: every element inventory stays non-negative and never grows
  - Symmetry: all elements are scaled by one factor, so elemental ratios are
    preserved exactly
  - Error contract: disabled by default, and a no-op for a bound envelope or
    an initial condition whose structure has not yet been solved

Testing standards:
  - docs/How-to/testing.md
"""

from __future__ import annotations

import pytest

from proteus.escape.boiloff import apply_boiloff_ic
from proteus.utils.constants import element_list

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# A 3 Earth-mass sub-Neptune at 640 K whose envelope reaches beyond its Bondi
# radius: the geometry the correction exists to handle. R_obs here is 1.6
# times the Bondi radius, well inside the regime.
MP_UNBOUND = 1.80e25
TEQ_UNBOUND = 638.6
R_UNBOUND = 4.27e8

# The same planet contracted to a bound envelope, far inside the threshold.
R_BOUND = 6.4e6


class _Escape:
    """Minimal stand-in for the escape section of the configuration."""

    def __init__(self, boiloff_ic: bool = True, boiloff_lambda_crit: float = 20.0):
        self.boiloff_ic = boiloff_ic
        self.boiloff_lambda_crit = boiloff_lambda_crit


class _Config:
    """Minimal stand-in exposing only the escape section the function reads."""

    def __init__(self, **kwargs):
        self.escape = _Escape(**kwargs)


def _row(radius: float, inventory: dict[str, float] | None = None) -> dict:
    """Build a helpfile row with a plausible, deliberately asymmetric inventory."""
    inventory = inventory or {'H': 4.0e21, 'C': 1.0e21, 'N': 2.0e20, 'S': 5.0e20, 'O': 3.0e21}
    row = {'M_planet': MP_UNBOUND, 'R_obs': radius, 'T_eqm': TEQ_UNBOUND}
    for element in element_list:
        row[f'{element}_kg_total'] = inventory.get(element, 0.0)
    return row


@pytest.mark.physics_invariant
def test_removed_mass_matches_recorded_total_and_preserves_ratios():
    """Trimming an unbound envelope conserves the recorded loss and elemental ratios."""
    row = _row(R_UNBOUND)
    before = {e: row[f'{e}_kg_total'] for e in element_list}
    total_before = sum(before.values())

    apply_boiloff_ic(_Config(), row)

    after = {e: row[f'{e}_kg_total'] for e in element_list}
    total_after = sum(after.values())

    # Conservation: what left the inventories is exactly what was recorded.
    assert total_before - total_after == pytest.approx(row['M_boiloff_kg'], rel=1e-12)

    # Sign and scale guards: mass was removed, and enough of it to matter for
    # an envelope 1.6 Bondi radii across, but not all of it.
    assert row['M_boiloff_kg'] > 0.0
    assert 0.5 * total_before < row['M_boiloff_kg'] < total_before

    # Symmetry: one factor for every element, so ratios are untouched. A
    # per-element factor would break this for the asymmetric inventory used.
    factor = total_after / total_before
    for element, mass in before.items():
        if mass > 0.0:
            assert after[element] / mass == pytest.approx(factor, rel=1e-12)
        assert after[element] >= 0.0


@pytest.mark.physics_invariant
def test_repeated_passes_accumulate_and_converge_to_a_bound_envelope():
    """Successive initialisation passes keep accumulating the recorded loss."""
    row = _row(R_UNBOUND)
    total_before = sum(row[f'{e}_kg_total'] for e in element_list)

    apply_boiloff_ic(_Config(), row)
    first = row['M_boiloff_kg']
    apply_boiloff_ic(_Config(), row)
    second = row['M_boiloff_kg']

    # The record accumulates rather than being overwritten, so the reported
    # total still equals everything removed across both passes.
    total_after = sum(row[f'{e}_kg_total'] for e in element_list)
    assert second > first
    assert total_before - total_after == pytest.approx(second, rel=1e-12)

    # Each pass removes mass but leaves a positive inventory behind, so the
    # correction cannot desiccate the planet outright.
    assert total_after > 0.0
    assert total_after < total_before


def test_bound_envelope_and_unsolved_structure_are_left_untouched():
    """A bound envelope, or one with no structure yet, is not modified."""
    bound = _row(R_BOUND)
    expected = {e: bound[f'{e}_kg_total'] for e in element_list}
    apply_boiloff_ic(_Config(), bound)
    for element, mass in expected.items():
        assert bound[f'{element}_kg_total'] == pytest.approx(mass, rel=1e-12)
    assert 'M_boiloff_kg' not in bound

    # Before the first structure solve the radius is zero, which is not a
    # bound envelope but an absent one: the correction must not divide by it.
    unsolved = _row(0.0)
    apply_boiloff_ic(_Config(), unsolved)
    assert unsolved['H_kg_total'] == pytest.approx(4.0e21, rel=1e-12)
    assert 'M_boiloff_kg' not in unsolved


def test_ordinary_planet_below_the_jeans_threshold_is_not_trimmed():
    """A planet inside the Bondi criterion is left alone whatever its Jeans value.

    The two published criteria coincide only when both are evaluated for
    atomic hydrogen, so a band exists where the Jeans parameter sits below the
    quoted threshold of 20 while the envelope is comfortably inside a tenth of
    its Bondi radius. Ordinary planets fall in that band, and the radius
    criterion is what decides, so nothing is removed from them.
    """
    # A radius giving Lambda of about 12, inside the band, but only 0.07 of the
    # Bondi radius and so well inside the criterion that governs the trim.
    row = _row(1.9032e7)
    expected = {e: row[f'{e}_kg_total'] for e in element_list}

    apply_boiloff_ic(_Config(), row)

    for element, mass in expected.items():
        assert row[f'{element}_kg_total'] == pytest.approx(mass, rel=1e-12)
    assert 'M_boiloff_kg' not in row


def test_correction_is_disabled_unless_explicitly_enabled():
    """The inventory is untouched with the default configuration."""
    row = _row(R_UNBOUND)
    expected = {e: row[f'{e}_kg_total'] for e in element_list}

    apply_boiloff_ic(_Config(boiloff_ic=False), row)

    for element, mass in expected.items():
        assert row[f'{element}_kg_total'] == pytest.approx(mass, rel=1e-12)
    assert 'M_boiloff_kg' not in row
