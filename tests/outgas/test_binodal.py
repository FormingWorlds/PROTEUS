"""Unit tests for the binodal H2 partitioning module
(``proteus.outgas.binodal``).

Exercises early-return guards, the Rogers+2025 sigma plumbing, and
the bookkeeping that recomputes partial pressures, VMRs, and the
atmospheric mean molecular weight after redistribution.

Anti-happy-path coverage:

- Each early-return guard (H2 not included, zero H2 mass, missing
  state variables) is tested individually so a regression that
  drops one of them is caught.
- Sigma is mocked at three discriminating values (0, 0.5, 1) to
  verify the linear partition ``H2_kg_liquid = sigma * H2_kg_total``
  and the H2_solid := 0 invariant.
- VMR closure is checked: the new ``H2_vmr`` plus the other
  species' VMRs must sum to 1.0 exactly within float precision.
- Adversarial inputs: zero ``M_mantle``, zero ``R_int``, and zero
  ``gravity`` must all trip the guard, leaving ``hf_row``
  unchanged.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import pytest

from proteus.outgas.binodal import apply_binodal_h2
from proteus.utils.constants import gas_list

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30), pytest.mark.physics_invariant]


def _make_config(h2_included: bool = True) -> Any:
    """Minimal Config stub exposing ``config.outgas.calliope.is_included``."""
    calliope = SimpleNamespace(
        is_included=lambda species: h2_included if species == 'H2' else False
    )
    outgas = SimpleNamespace(calliope=calliope)
    return cast(Any, SimpleNamespace(outgas=outgas))


def _make_hf_row(
    *,
    H2_kg_total: float = 1e18,
    H2_kg_atm: float = 1e18,
    H2_kg_liquid: float = 0.0,
    T_magma: float = 3000.0,
    P_surf_bar: float = 100.0,
    M_mantle: float = 4e24,
    R_int: float = 6.371e6,
    gravity: float = 9.81,
    extra: dict | None = None,
) -> dict:
    row = {
        'H2_kg_total': H2_kg_total,
        'H2_kg_atm': H2_kg_atm,
        'H2_kg_liquid': H2_kg_liquid,
        'T_magma': T_magma,
        'P_surf': P_surf_bar,
        'M_mantle': M_mantle,
        'R_int': R_int,
        'gravity': gravity,
        # Seed the other species so VMR closure can be checked.
        'H2O_bar': 100.0,
    }
    for s in gas_list:
        row.setdefault(s + '_bar', 0.0)
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Early-return guards
# ---------------------------------------------------------------------------


def test_returns_early_when_h2_not_included_in_calliope():
    """If CALLIOPE does not include H2 the function must return without
    touching ``hf_row``."""
    cfg = _make_config(h2_included=False)
    hf_row = _make_hf_row()
    snapshot = dict(hf_row)
    apply_binodal_h2(hf_row, cfg)
    assert hf_row == snapshot


def test_returns_early_for_zero_h2_total_mass():
    """Zero total H2 mass: no partitioning, no log noise."""
    cfg = _make_config()
    hf_row = _make_hf_row(H2_kg_total=0.0)
    snapshot = dict(hf_row)
    apply_binodal_h2(hf_row, cfg)
    assert hf_row == snapshot


@pytest.mark.parametrize(
    'override',
    [
        {'T_magma': 0.0},
        {'M_mantle': 0.0},
        {'R_int': 0.0},
        {'gravity': 0.0},
        {'T_magma': -10.0},  # negative is treated identically to zero
    ],
)
def test_returns_early_when_any_state_variable_is_non_physical(override):
    """Each adversarial-zero (or negative) state variable must trip the
    guard separately. A regression that dropped one of these checks
    would let the function continue into a divide-by-zero or NaN path.
    """
    cfg = _make_config()
    hf_row = _make_hf_row(**override)
    snapshot = dict(hf_row)
    apply_binodal_h2(hf_row, cfg)
    assert hf_row == snapshot


# ---------------------------------------------------------------------------
# Sigma plumbing: partition between dissolved and atmospheric reservoirs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'sigma,expected_liquid_frac,expected_atm_frac',
    [
        (0.0, 0.0, 1.0),  # Fully immiscible: everything in the atmosphere
        (0.5, 0.5, 0.5),  # Equal split
        (1.0, 1.0, 0.0),  # Fully miscible: everything dissolved
    ],
)
def test_sigma_partitions_h2_linearly(sigma, expected_liquid_frac, expected_atm_frac):
    """``H2_kg_liquid = sigma * H2_kg_total`` and
    ``H2_kg_atm = (1 - sigma) * H2_kg_total``: linear partition.
    """
    cfg = _make_config()
    hf_row = _make_hf_row(H2_kg_total=1e18)
    with patch('zalmoxis.binodal.rogers2025_suppression_weight', return_value=sigma):
        apply_binodal_h2(hf_row, cfg)
    assert hf_row['H2_kg_liquid'] == pytest.approx(expected_liquid_frac * 1e18, rel=1e-12)
    assert hf_row['H2_kg_atm'] == pytest.approx(expected_atm_frac * 1e18, rel=1e-12)


def test_solid_h2_reservoir_is_always_zero():
    """H2 does not partition into solid silicate; the solid reservoir
    must be zeroed regardless of sigma."""
    cfg = _make_config()
    hf_row = _make_hf_row(extra={'H2_kg_solid': 1e20})  # stale junk to be cleared
    with patch('zalmoxis.binodal.rogers2025_suppression_weight', return_value=0.3):
        apply_binodal_h2(hf_row, cfg)
    assert hf_row['H2_kg_solid'] == 0.0
    assert hf_row['H2_mol_solid'] == 0.0


@pytest.mark.reference_pinned
def test_h2_mass_is_conserved_per_rogers2025_partition():
    """Mass closure across the Rogers et al. (2025) binodal partition:
    for any miscibility weight ``sigma`` returned by the H2-MgSiO3
    suppression function, the post-partition reservoirs must satisfy
    ``H2_kg_atm + H2_kg_liquid + H2_kg_solid = H2_kg_total``. This
    is the conservation invariant the binodal model is required to
    preserve (Rogers et al. 2025, Section 3.2).
    """
    cfg = _make_config()
    hf_row = _make_hf_row(H2_kg_total=5e17)
    with patch('zalmoxis.binodal.rogers2025_suppression_weight', return_value=0.4):
        apply_binodal_h2(hf_row, cfg)
    total = hf_row['H2_kg_atm'] + hf_row['H2_kg_liquid'] + hf_row['H2_kg_solid']
    assert total == pytest.approx(5e17, rel=1e-12)
    # Sign guard: every reservoir is non-negative.
    assert hf_row['H2_kg_atm'] >= 0.0
    assert hf_row['H2_kg_liquid'] >= 0.0
    assert hf_row['H2_kg_solid'] == 0.0  # H2 has no solid silicate sink
    # Scale guard: each reservoir is bounded above by the total.
    assert hf_row['H2_kg_atm'] <= 5e17
    assert hf_row['H2_kg_liquid'] <= 5e17


# ---------------------------------------------------------------------------
# Bookkeeping: partial pressure, VMRs, MMW
# ---------------------------------------------------------------------------


def test_h2_partial_pressure_uses_g_over_area_in_pa_to_bar():
    """``P_H2 = m * g / (4 pi R^2)`` then ``/ 1e5`` to convert Pa to bar."""
    import math

    cfg = _make_config()
    hf_row = _make_hf_row(
        H2_kg_total=1e18,
        R_int=6.371e6,
        gravity=9.81,
        extra={'H2O_bar': 0.0},  # isolate H2 contribution
    )
    with patch('zalmoxis.binodal.rogers2025_suppression_weight', return_value=0.0):
        # sigma=0 → all H2 in the atmosphere
        apply_binodal_h2(hf_row, cfg)
    area = 4.0 * math.pi * 6.371e6**2
    expected_bar = 1e18 * 9.81 / area / 1e5
    assert hf_row['H2_bar'] == pytest.approx(expected_bar, rel=1e-12)


def test_vmr_closure_after_partition():
    """Volume mixing ratios must sum to 1.0 once ``P_surf`` is rebuilt."""
    cfg = _make_config()
    hf_row = _make_hf_row(H2_kg_total=1e18, extra={'H2O_bar': 50.0, 'CO2_bar': 25.0})
    with patch('zalmoxis.binodal.rogers2025_suppression_weight', return_value=0.5):
        apply_binodal_h2(hf_row, cfg)
    vmr_sum = sum(hf_row[s + '_vmr'] for s in gas_list)
    assert vmr_sum == pytest.approx(1.0, rel=1e-12)


def test_atmospheric_mmw_recomputed_when_h2_mass_changes():
    """Redistributing H2 (lighter than H2O) into the dissolved phase
    must increase the atmospheric MMW, not decrease it.
    """
    cfg = _make_config()
    hf_row = _make_hf_row(H2_kg_total=1e18, extra={'H2O_bar': 50.0})
    # sigma=0 → H2-dominated atmosphere → light MMW
    with patch('zalmoxis.binodal.rogers2025_suppression_weight', return_value=0.0):
        apply_binodal_h2(hf_row, cfg)
        mmw_h2_heavy = hf_row['atm_kg_per_mol']

    # Reset hf_row to the same starting state but with sigma=1
    hf_row2 = _make_hf_row(H2_kg_total=1e18, extra={'H2O_bar': 50.0})
    with patch('zalmoxis.binodal.rogers2025_suppression_weight', return_value=1.0):
        apply_binodal_h2(hf_row2, cfg)
        mmw_h2_dissolved = hf_row2['atm_kg_per_mol']

    # With H2 dissolved (sigma=1), only H2O remains in the atmosphere
    # → MMW should be heavier (closer to 18 g/mol vs 2 g/mol).
    assert mmw_h2_dissolved > mmw_h2_heavy
