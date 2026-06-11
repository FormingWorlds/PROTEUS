"""Unit tests for ``proteus.outgas.common``.

Exercises the ``expected_keys`` function, which constructs the list of
helpfile keys that the outgassing wrapper copies from the solver result
into hf_row.

Invariants tested:
  - Key completeness: all gas/element/reservoir combinations are present
  - Key uniqueness: no duplicates in the returned list
  - Structural: O2 and O follow distinct inclusion rules (issue #677)
  - Structural: element mass-ratio keys are symmetric-pair unique

Testing standards:
  - docs/How-to/test_infrastructure.md
  - docs/How-to/test_categorization.md
  - docs/How-to/test_building.md
"""

from __future__ import annotations

import pytest

from proteus.outgas.common import expected_keys
from proteus.utils.constants import element_list, gas_list

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# -----------------------------------------------------------------------
# Key completeness
# -----------------------------------------------------------------------


def test_expected_keys_contains_gas_bar_and_vmr():
    """Every gas in gas_list has both a '_bar' and '_vmr' key.

    These hold the surface partial pressure [bar] and volume mixing
    ratio for each gas species.
    """
    keys = expected_keys()
    for gas in gas_list:
        assert f'{gas}_bar' in keys, f'Missing {gas}_bar key'
        assert f'{gas}_vmr' in keys, f'Missing {gas}_vmr key'
    # Discrimination: gas_list has 15 species (11 vol + 4 vap),
    # so there should be at least 30 bar+vmr keys
    bar_keys = [k for k in keys if k.endswith('_bar')]
    vmr_keys = [k for k in keys if k.endswith('_vmr')]
    assert len(bar_keys) == len(gas_list)
    assert len(vmr_keys) == len(gas_list)


def test_expected_keys_contains_gas_reservoir_keys():
    """Every gas has '_kg_<reservoir>' and '_mol_<reservoir>' keys
    for all four reservoirs: atm, liquid, solid, total.

    This ensures the full mass/mole accounting is copied from the
    solver output.
    """
    keys = expected_keys()
    reservoirs = ('atm', 'liquid', 'solid', 'total')
    for gas in gas_list:
        for res in reservoirs:
            assert f'{gas}_kg_{res}' in keys, f'Missing {gas}_kg_{res}'
            assert f'{gas}_mol_{res}' in keys, f'Missing {gas}_mol_{res}'


def test_expected_keys_contains_element_reservoir_keys():
    """Element keys follow the issue #677 convention: atm, liquid,
    solid reservoirs for all elements, plus _kg_total only for O.

    Under the O-accounting fix, O_kg_total is written by the chemistry
    solver (from_O_budget restores it to the authoritative value after copy);
    other elements' _kg_total is owned by escape and must NOT be in
    this list.
    """
    keys = expected_keys()
    for e in element_list:
        for res in ('atm', 'liquid', 'solid'):
            assert f'{e}_kg_{res}' in keys, f'Missing {e}_kg_{res}'

    # O_kg_total IS in the list (issue #677: O is tracked by chemistry)
    assert 'O_kg_total' in keys
    # H_kg_total is NOT in the list (owned by escape)
    assert 'H_kg_total' not in keys
    assert 'C_kg_total' not in keys
    assert 'N_kg_total' not in keys


# -----------------------------------------------------------------------
# Key uniqueness
# -----------------------------------------------------------------------


def test_expected_keys_no_duplicates():
    """The returned key list has no duplicates.

    A duplicate would cause the copy loop to overwrite a value with
    itself (harmless but wasteful) and could mask a naming collision
    between gas and element keys.
    """
    keys = expected_keys()
    assert len(keys) == len(set(keys)), (
        f'Duplicate keys found: {[k for k in keys if keys.count(k) > 1]}'
    )
    assert len(keys) > 10  # structural: key set is non-trivial


# -----------------------------------------------------------------------
# Structural: scalar keys
# -----------------------------------------------------------------------


def test_expected_keys_contains_scalar_diagnostics():
    """The scalar diagnostic keys (P_surf, M_atm, atm_kg_per_mol,
    fO2_shift_IW_derived, O_res) are present.

    These are critical coupling variables between the outgassing
    solver and the main loop.
    """
    keys = expected_keys()
    for k in ('P_surf', 'M_atm', 'atm_kg_per_mol', 'fO2_shift_IW_derived', 'O_res'):
        assert k in keys, f'Missing scalar key: {k}'
    assert len(keys) > len(('P_surf', 'M_atm', 'atm_kg_per_mol'))  # more than just scalars


# -----------------------------------------------------------------------
# Structural: element mass-ratio keys
# -----------------------------------------------------------------------


def test_expected_keys_contains_element_ratio_keys():
    """Element mass-ratio keys (e.g. 'H/O_atm', 'C/N_atm') are
    present for all unordered pairs of distinct elements.

    The function generates one key per unordered pair (not both
    orderings), so the total count is C(n, 2) = n*(n-1)/2.
    """
    keys = expected_keys()
    n = len(element_list)
    expected_pair_count = n * (n - 1) // 2
    ratio_keys = [k for k in keys if k.endswith('_atm') and '/' in k]
    assert len(ratio_keys) == expected_pair_count, (
        f'Expected {expected_pair_count} ratio keys, got {len(ratio_keys)}'
    )
    # Each pair appears exactly once (not both A/B and B/A)
    pairs_seen = set()
    for k in ratio_keys:
        pair_str = k.replace('_atm', '')
        e1, e2 = pair_str.split('/')
        canonical = tuple(sorted([e1, e2]))
        assert canonical not in pairs_seen, f'Duplicate pair: {canonical}'
        pairs_seen.add(canonical)
