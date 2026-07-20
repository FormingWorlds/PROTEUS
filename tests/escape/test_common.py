"""Unit tests for ``proteus.escape.common``.

Exercises the unfractionated escape-rate distribution function
``calc_unfract_fluxes``, which partitions a bulk escape rate across
elements proportional to their mass fractions in a chosen reservoir.

Invariants tested:
  - Conservation: sum of per-element rates equals the total escape rate
  - Positivity: per-element rates are non-negative when inputs are non-negative
  - Boundedness: no single element rate exceeds the total
  - Error contract: invalid reservoir string raises ValueError

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import pytest

from proteus.escape.common import calc_unfract_fluxes
from proteus.utils.constants import element_list

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_hf_row(
    *,
    reservoir: str = 'bulk',
    esc_rate_total: float = 1e6,
    element_masses: dict | None = None,
) -> dict:
    """Build a minimal hf_row dict for calc_unfract_fluxes.

    Parameters
    ----------
    reservoir : str
        'bulk' uses '_kg_total' keys; 'outgas' uses '_kg_atm' keys.
    esc_rate_total : float
        Total escape rate [kg/s].
    element_masses : dict or None
        Per-element masses [kg]. Defaults to an asymmetric mix
        dominated by H with trace C, N, S, O.
    """
    if element_masses is None:
        element_masses = {
            'H': 1.0e20,
            'O': 5.0e19,
            'C': 1.0e18,
            'N': 5.0e17,
            'S': 1.0e16,
            'Si': 0.0,
            'Mg': 0.0,
            'Fe': 0.0,
            'Na': 0.0,
        }

    key_suffix = '_kg_total' if reservoir == 'bulk' else '_kg_atm'
    hf_row: dict = {'esc_rate_total': esc_rate_total}
    for e in element_list:
        hf_row[e + key_suffix] = element_masses.get(e, 0.0)
    return hf_row


# -----------------------------------------------------------------------
# Conservation
# -----------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_unfract_fluxes_mass_conservation_bulk():
    """Sum of per-element escape rates equals the total escape rate.

    Uses the 'bulk' reservoir with an asymmetric composition
    (H-dominated with traces) so the test discriminates against
    regressions that omit one element from the sum. An equal-mass
    composition would mask a single-element skip.
    """
    hf_row = _make_hf_row(reservoir='bulk', esc_rate_total=1e6)
    calc_unfract_fluxes(hf_row, reservoir='bulk', min_thresh=1.0)

    rate_sum = sum(hf_row.get(f'esc_rate_{e}', 0.0) for e in element_list)
    # Conservation: sum equals total to within floating-point rounding
    assert rate_sum == pytest.approx(1e6, rel=1e-12)
    # Sign guard: total is positive, so the sum must be too
    assert rate_sum > 0


@pytest.mark.physics_invariant
def test_unfract_fluxes_mass_conservation_outgas():
    """Conservation holds for the 'outgas' (atmospheric) reservoir.

    The reservoir key switches from '_kg_total' to '_kg_atm'; the
    proportional arithmetic is the same, but this test catches a
    regression that hard-codes the wrong suffix.
    """
    hf_row = _make_hf_row(reservoir='outgas', esc_rate_total=5e5)
    calc_unfract_fluxes(hf_row, reservoir='outgas', min_thresh=1.0)

    rate_sum = sum(hf_row.get(f'esc_rate_{e}', 0.0) for e in element_list)
    assert rate_sum == pytest.approx(5e5, rel=1e-12)
    # Scale guard: order of magnitude is 5e5, not 5e2 or 5e8
    assert 1e5 < rate_sum < 1e6


# -----------------------------------------------------------------------
# Proportionality and boundedness
# -----------------------------------------------------------------------


@pytest.mark.physics_invariant
def test_unfract_fluxes_proportional_to_mass_fraction():
    """Per-element rate is proportional to the element's mass fraction.

    H holds ~65% of the total volatile mass in the default mix, so
    its escape rate should be ~65% of the total. A regression that
    distributed equally (1/N_elements) would give ~11% instead.
    """
    hf_row = _make_hf_row(reservoir='bulk', esc_rate_total=1e6)
    calc_unfract_fluxes(hf_row, reservoir='bulk', min_thresh=1.0)

    # H mass fraction: 1e20 / (1e20 + 5e19 + 1e18 + 5e17 + 1e16) ~ 0.659
    total_mass = 1e20 + 5e19 + 1e18 + 5e17 + 1e16
    expected_h_frac = 1e20 / total_mass
    actual_h_rate = hf_row['esc_rate_H']

    assert actual_h_rate == pytest.approx(expected_h_frac * 1e6, rel=1e-12)
    # Discrimination: equal distribution would give 1e6/9 ~ 1.11e5;
    # the correct H rate is ~6.59e5. The gap is > 5e5.
    assert abs(actual_h_rate - 1e6 / len(element_list)) > 5e5
    # Boundedness: no element rate exceeds the total
    for e in element_list:
        assert hf_row.get(f'esc_rate_{e}', 0.0) <= hf_row['esc_rate_total']


# -----------------------------------------------------------------------
# Edge cases
# -----------------------------------------------------------------------


def test_unfract_fluxes_below_threshold_does_not_set_rates():
    """When total volatile mass is below min_thresh, the function
    returns early without setting per-element rates.

    Edge case: desiccated planet with negligible volatiles.
    """
    hf_row = _make_hf_row(reservoir='bulk', esc_rate_total=1e6)
    # Set all masses to near-zero (below any reasonable threshold)
    for e in element_list:
        hf_row[e + '_kg_total'] = 1e-30

    calc_unfract_fluxes(hf_row, reservoir='bulk', min_thresh=1.0)

    # No esc_rate_<element> keys should be written
    for e in element_list:
        assert f'esc_rate_{e}' not in hf_row
    # esc_rate_total remains unchanged (not zeroed by early return)
    assert hf_row['esc_rate_total'] == pytest.approx(1e6, rel=1e-12)


def test_unfract_fluxes_zero_element_gets_zero_rate():
    """An element with zero mass in the reservoir gets zero escape rate,
    even when other elements have large masses.

    Edge case: Si, Mg, Fe, Na are zero in the default volatile mix.
    """
    hf_row = _make_hf_row(reservoir='bulk', esc_rate_total=1e6)
    calc_unfract_fluxes(hf_row, reservoir='bulk', min_thresh=1.0)

    # Zero-mass elements must have zero rates
    for e in ('Si', 'Mg', 'Fe', 'Na'):
        assert hf_row[f'esc_rate_{e}'] == pytest.approx(0.0, abs=1e-30)
    # Non-zero elements must have positive rates
    assert hf_row['esc_rate_H'] > 0
    assert hf_row['esc_rate_O'] > 0


# -----------------------------------------------------------------------
# Error contract
# -----------------------------------------------------------------------


def test_unfract_fluxes_invalid_reservoir_raises():
    """Invalid reservoir string raises ValueError.

    The function uses a match/case dispatch on reservoir; an
    unrecognised value must raise, not silently return.
    """
    hf_row = _make_hf_row(reservoir='bulk', esc_rate_total=1e6)
    with pytest.raises(ValueError, match='Invalid escape reservoir'):
        calc_unfract_fluxes(hf_row, reservoir='invalid', min_thresh=1.0)

    # Adjacent-valid: 'bulk' and 'outgas' must NOT raise
    hf_bulk = _make_hf_row(reservoir='bulk', esc_rate_total=1e3)
    calc_unfract_fluxes(hf_bulk, reservoir='bulk', min_thresh=1.0)
    assert 'esc_rate_H' in hf_bulk
