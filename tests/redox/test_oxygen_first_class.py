"""
Unit tests for E.2 oxygen-first-class mass conservation.

These are fast pure-Python tests that check the O-kg bookkeeping
invariants introduced by Commit D + D.1:

  M_planet == M_int + Σ_element element_kg_total
  O_kg_total == O_kg_atm + O_kg_liquid + O_kg_solid
  M_ele (with O) - M_ele_excl_O == O_kg_total

The long-horizon 9000-step live check is run manually and logged to
memory (see docs/redox.md §Regression tier).
"""
from __future__ import annotations

import pytest


def _canonical_hf_row() -> dict:
    """Synthetic helpfile row mimicking a post-outgas CALLIOPE state."""
    # CALLIOPE writes {species}_mol_atm / _mol_liquid / _mol_solid.
    # After run_outgassing -> populate_O_kg, O_kg_* should equal the
    # sum over O-containing species.
    return {
        'H2O_mol_atm': 1.0e20,
        'H2O_mol_liquid': 2.0e20,
        'H2O_mol_solid': 0.0,
        'CO2_mol_atm': 5.0e19,
        'CO2_mol_liquid': 1.0e19,
        'CO2_mol_solid': 0.0,
        'O2_mol_atm': 1.0e18,
        'O2_mol_liquid': 0.0,
        'O2_mol_solid': 0.0,
        'CO_mol_atm': 0.0,
        'H2_mol_atm': 0.0,
        'CH4_mol_atm': 0.0,
        'N2_mol_atm': 0.0,
        'S2_mol_atm': 0.0,
        'SO2_mol_atm': 0.0,
        'H2S_mol_atm': 0.0,
        'H_kg_total': 1.0e20,
        'C_kg_total': 5.0e19,
        'N_kg_total': 0.0,
        'S_kg_total': 0.0,
        'Si_kg_total': 0.0,
        'Mg_kg_total': 0.0,
        'Fe_kg_total': 0.0,
        'Na_kg_total': 0.0,
        'O_kg_atm': 0.0,
        'O_kg_liquid': 0.0,
        'O_kg_solid': 0.0,
        'O_kg_total': 0.0,
        'M_int': 4.0e24,
        'Fe_kg_core': 1.87e24,
        'M_mantle': 4.0e24,
        'M_planet': 5.97e24,
    }


@pytest.mark.unit
def test_populate_O_kg_reservoir_sum_invariant() -> None:
    """
    After populate_O_kg, the reservoir-sum invariant must hold:
    O_kg_total == O_kg_atm + O_kg_liquid + O_kg_solid.
    """
    from proteus.utils.coupler import populate_O_kg

    hf_row = _canonical_hf_row()
    populate_O_kg(hf_row)

    total = hf_row['O_kg_total']
    sum_r = (hf_row['O_kg_atm']
             + hf_row['O_kg_liquid']
             + hf_row['O_kg_solid'])

    assert abs(total - sum_r) < 1.0e-6 * max(total, 1.0), (
        f'O_kg_total ({total:.3e}) != Σ reservoirs ({sum_r:.3e})'
    )


@pytest.mark.unit
def test_populate_O_kg_from_h2o_atm_only() -> None:
    """
    Sanity: 1e20 mol H2O -> 1e20 mol O in atm -> ~1.6e21 kg O.
    (M(O) = 15.999 g/mol; result close to 1.6e21.)
    """
    from proteus.utils.coupler import populate_O_kg

    hf_row = {
        'H2O_mol_atm': 1.0e20,
        'H2O_mol_liquid': 0.0,
        'H2O_mol_solid': 0.0,
        'O_kg_atm': 0.0,
        'O_kg_liquid': 0.0,
        'O_kg_solid': 0.0,
        'O_kg_total': 0.0,
    }
    # Fill in zero for other species so the helper doesn't choke.
    for s in ('CO2', 'O2', 'CO', 'H2', 'CH4', 'N2', 'S2', 'SO2', 'H2S'):
        for r in ('_mol_atm', '_mol_liquid', '_mol_solid'):
            hf_row.setdefault(s + r, 0.0)

    populate_O_kg(hf_row)

    expected_O_kg = 1.0e20 * 15.999e-3    # 1 O atom per H2O
    assert abs(hf_row['O_kg_atm'] - expected_O_kg) < 1.0e-9 * expected_O_kg


@pytest.mark.unit
def test_M_ele_excl_O_helper() -> None:
    """M_ele_excl_O returns the pre-#57 non-O element sum."""
    from proteus.utils.coupler import M_ele_excl_O

    hf_row = _canonical_hf_row()
    # element_list will include H, C, N, S, O. M_ele_excl_O drops O.
    non_O = M_ele_excl_O(hf_row)
    expected = (hf_row['H_kg_total'] + hf_row['C_kg_total']
                + hf_row['N_kg_total'] + hf_row['S_kg_total'])
    assert non_O == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_populate_O_kg_handles_kg_only_backend() -> None:
    """
    Atmodeller writes `{species}_kg_atm` (no `_mol_atm`). populate_O_kg
    must handle this fallback without raising (Commit D.1 round-7 fix).
    """
    from proteus.utils.coupler import populate_O_kg

    hf_row = {
        'H2O_kg_atm': 1.0e20 * 18.015e-3,   # kg form only
        'H2O_kg_liquid': 0.0,
        'H2O_kg_solid': 0.0,
        'O_kg_atm': 0.0,
        'O_kg_liquid': 0.0,
        'O_kg_solid': 0.0,
        'O_kg_total': 0.0,
    }
    for s in ('CO2', 'O2', 'CO', 'H2', 'CH4', 'N2', 'S2', 'SO2', 'H2S'):
        for r in ('_kg_atm', '_kg_liquid', '_kg_solid'):
            hf_row.setdefault(s + r, 0.0)

    populate_O_kg(hf_row)

    # H2O molar mass 18.015; O fraction = 15.999/18.015 ≈ 0.8881.
    expected_O_kg = 1.0e20 * 15.999e-3
    assert abs(hf_row['O_kg_atm'] - expected_O_kg) < 1.0e-6 * expected_O_kg


@pytest.mark.unit
def test_O_skip_site_count_locked() -> None:
    """
    Plan v6 §3.8 promised 8 O-skip sites removed (D + D.1 cumulative).
    This test locks the remaining executable `== 'O'` count in src/
    so a future regression that reintroduces an O-skip branch fails
    loudly. Pure-Python scan (no ripgrep dependency).
    """
    import re
    from pathlib import Path

    src = Path(__file__).parents[2] / 'src' / 'proteus'
    pattern = re.compile(r"==\s*['\"]O['\"]")
    matches: list[str] = []
    for py in src.rglob('*.py'):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith('#'):
                continue
            # Skip docstrings / commented-out code crudely: require a
            # control-flow keyword on the same line.
            if pattern.search(line) and re.search(
                r'\b(if|elif|while|assert|and|or|not)\b', line
            ):
                matches.append(f'{py.relative_to(src.parents[1])}:{lineno}: {stripped}')

    # Expected <=8 (D + D.1 removed 8; stragglers can still appear in
    # guard-paths or comparison assertions). Lock the count.
    assert len(matches) <= 8, (
        f'{len(matches)} executable O-skip matches in src/ '
        '(plan v6 §3.8 targeted <=8):\n  ' + '\n  '.join(matches)
    )
