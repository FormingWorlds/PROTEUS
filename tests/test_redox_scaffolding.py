"""
Scaffolding tests for the redox subpackage (#57 Commit A).

This commit adds the hf_row schema + constants + helpers needed by
Commits B/C/D but does NOT yet wire the redox solver into the main
loop and does NOT yet remove the oxygen-skip branches. The latter is
Commit C. These tests pin the Commit-A contract: new constants and
keys exist with zero defaults; legacy CSVs migrate cleanly; the eleven
O-skip sites are still in place (so static-mode runs are unchanged).
"""
from __future__ import annotations

import os

import pytest

from proteus.utils.constants import (
    element_list,
    element_mmw,
    oxide_mmw,
)
from proteus.utils.coupler import (
    GetHelpfileKeys,
    M_ele_excl_O,
    ReadHelpfileFromCSV,
    ZeroHelpfileRow,
)


@pytest.mark.unit
def test_oxide_mmw_derivations():
    """Oxide molar masses must derive from element_mmw to sub-gram accuracy."""
    # FeO = Fe + O
    assert oxide_mmw['FeO'] == pytest.approx(
        element_mmw['Fe'] + element_mmw['O'], rel=1e-12
    )
    # Fe2O3 = 2 Fe + 3 O
    assert oxide_mmw['Fe2O3'] == pytest.approx(
        2 * element_mmw['Fe'] + 3 * element_mmw['O'], rel=1e-12
    )
    # SiO2 = Si + 2 O
    assert oxide_mmw['SiO2'] == pytest.approx(
        element_mmw['Si'] + 2 * element_mmw['O'], rel=1e-12
    )
    # MgO = Mg + O
    assert oxide_mmw['MgO'] == pytest.approx(
        element_mmw['Mg'] + element_mmw['O'], rel=1e-12
    )


@pytest.mark.unit
def test_oxide_mmw_known_values():
    """Spot-check vs. commonly tabulated values (kg/mol)."""
    # FeO ~ 71.844 g/mol per IUPAC 2019
    assert oxide_mmw['FeO'] == pytest.approx(71.844e-3, abs=5e-6)
    # Fe2O3 ~ 159.687 g/mol
    assert oxide_mmw['Fe2O3'] == pytest.approx(159.687e-3, abs=5e-6)
    # MgO ~ 40.304 g/mol
    assert oxide_mmw['MgO'] == pytest.approx(40.304e-3, abs=5e-6)
    # SiO2 ~ 60.083 g/mol
    assert oxide_mmw['SiO2'] == pytest.approx(60.083e-3, abs=5e-6)


@pytest.mark.unit
def test_oxide_mmw_covers_schaefer24_nine_oxides():
    """All nine Schaefer+24 Table 1 oxides must be present."""
    required = {'SiO2', 'TiO2', 'Al2O3', 'FeO', 'MgO', 'CaO',
                'Na2O', 'K2O', 'P2O5'}
    # Fe2O3 is also required for the Fe3+/Fe2+ split bookkeeping.
    required.add('Fe2O3')
    missing = required - set(oxide_mmw.keys())
    assert not missing, f'oxide_mmw missing entries: {missing}'


@pytest.mark.unit
def test_redox_hf_row_keys_present():
    """New redox-scaffolding keys must appear in the helpfile schema."""
    keys = set(GetHelpfileKeys())
    required = {
        'fO2_shift_IW',
        'fO2_dIW', 'fO2_dQFM', 'fO2_dNNO',
        'R_budget_atm', 'R_budget_mantle',
        'R_budget_core', 'R_budget_total',
        'R_escaped_cum',
        'redox_conservation_residual',
        'redox_solver_fallback_count',
        'redox_mode_active',
        'Fe3_frac', 'FeO_total_wt', 'Fe2O3_wt',
        'n_Fe3_melt', 'n_Fe2_melt',
        'n_Fe3_solid_total', 'n_Fe2_solid_total', 'n_Fe0_solid_total',
        'dm_Fe0_to_core_cum',
    }
    missing = required - keys
    assert not missing, f'GetHelpfileKeys is missing: {sorted(missing)}'


@pytest.mark.unit
def test_per_element_core_columns_present():
    """Every tracked element must have a _kg_core column."""
    keys = set(GetHelpfileKeys())
    for e in element_list:
        assert f'{e}_kg_core' in keys, (
            f'Missing per-element core column: {e}_kg_core'
        )


@pytest.mark.unit
def test_zero_helpfile_row_covers_new_keys():
    """ZeroHelpfileRow must produce zero values for every new redox key."""
    row = ZeroHelpfileRow()
    schema = GetHelpfileKeys()
    for k in schema:
        assert k in row, f'ZeroHelpfileRow missing {k}'
        assert row[k] == 0.0, f'ZeroHelpfileRow[{k}] should be 0.0'


@pytest.mark.unit
def test_M_ele_excl_O_excludes_oxygen():
    """M_ele_excl_O must return the pre-#57 M_ele semantics."""
    row = ZeroHelpfileRow()
    # Seed each non-O element with distinct nonzero mass.
    for i, e in enumerate(element_list):
        row[f'{e}_kg_total'] = (i + 1) * 1000.0
    total_excl_O = M_ele_excl_O(row)
    expected = sum(
        (i + 1) * 1000.0
        for i, e in enumerate(element_list)
        if e != 'O'
    )
    assert total_excl_O == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_M_ele_excl_O_nonzero_O_excluded():
    """Non-zero O mass must NOT be summed by the helper."""
    row = ZeroHelpfileRow()
    for e in element_list:
        row[f'{e}_kg_total'] = 0.0
    row['O_kg_total'] = 1.23e10  # large nonzero
    assert M_ele_excl_O(row) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_read_helpfile_csv_migrates_missing_columns(tmp_path):
    """
    Legacy CSVs written before #57 lack the new redox columns. On
    resume ReadHelpfileFromCSV must seed them with zero without
    raising.
    """
    # Construct a dataframe with only the pre-#57 keys.
    legacy_subset = [
        'Time', 'T_surf', 'T_magma', 'M_ele', 'Phi_global',
    ]
    # Write minimal legacy CSV.
    import pandas as pd
    df_legacy = pd.DataFrame(
        [{k: 0.0 for k in legacy_subset}],
        columns=legacy_subset,
    )
    fpath = tmp_path / 'runtime_helpfile.csv'
    df_legacy.to_csv(fpath, index=False, sep='\t', float_format='%.10e')

    # Read back via the migration path.
    df_migrated = ReadHelpfileFromCSV(str(tmp_path))

    # All schema keys must now be present.
    for key in GetHelpfileKeys():
        assert key in df_migrated.columns, f'Missing after migration: {key}'
    # New redox keys seeded to zero.
    assert df_migrated['R_budget_atm'].iloc[0] == 0.0
    assert df_migrated['Fe3_frac'].iloc[0] == 0.0
    assert df_migrated['fO2_shift_IW'].iloc[0] == 0.0


@pytest.mark.unit
def test_oxygen_skip_sites_countinvariant():
    """
    Lock the count of executable O-skip sites at 11 (as of
    tl/interior-refactor 2401e6d2). This test goes RED in Commit C when
    the redox module populates O_kg_total and the sites are removed.

    When Commit C lands, update this test to assert count == 0 and
    remove this docstring.
    """
    import re
    repo_root = _find_repo_root()
    src = os.path.join(repo_root, 'src', 'proteus')
    coupler_path = os.path.join(src, 'utils', 'coupler.py')

    pattern = re.compile(r"e == 'O'|e != 'O'")
    hits: list[str] = []
    for dirpath, _, filenames in os.walk(src):
        for fname in filenames:
            if not fname.endswith('.py'):
                continue
            fpath = os.path.join(dirpath, fname)
            with open(fpath, 'r') as fh:
                for i, line in enumerate(fh, 1):
                    if pattern.search(line):
                        # Exclude the M_ele_excl_O helper in coupler.py
                        # which deliberately contains `e != 'O'`.
                        if fpath == coupler_path:
                            continue
                        hits.append(f'{fpath}:{i}: {line.rstrip()}')

    assert len(hits) == 11, (
        f'Expected 11 O-skip sites pre-Commit-C, found {len(hits)}:\n'
        + '\n'.join(hits)
    )


def _find_repo_root():
    """Walk up from this test file to find the PROTEUS repo root."""
    path = os.path.abspath(__file__)
    while path != '/':
        if os.path.exists(os.path.join(path, 'pyproject.toml')):
            return path
        path = os.path.dirname(path)
    raise RuntimeError('Could not locate PROTEUS repo root')
