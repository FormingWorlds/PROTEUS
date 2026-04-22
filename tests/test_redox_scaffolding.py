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
    compute_O_kg_from_species,
    populate_O_kg,
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
def test_compute_O_kg_pure_H2O_atmosphere():
    """An atmosphere of pure H2O yields O_kg = n_H2O × MW_O."""
    from proteus.utils.constants import element_mmw
    row = ZeroHelpfileRow()
    row['H2O_mol_atm'] = 1.0e20
    O_kg = compute_O_kg_from_species(row, 'atm')
    assert O_kg == pytest.approx(1.0e20 * element_mmw['O'], rel=1e-12)


@pytest.mark.unit
def test_compute_O_kg_mixed_atmosphere():
    """Mixed species: H2O (1 O), CO2 (2 O), H2 (0 O)."""
    from proteus.utils.constants import element_mmw
    row = ZeroHelpfileRow()
    row['H2O_mol_atm'] = 1.0e20   # 1 × 1e20 mol O
    row['CO2_mol_atm'] = 2.0e19   # 2 × 2e19 mol O
    row['H2_mol_atm'] = 5.0e20    # 0 mol O
    O_kg = compute_O_kg_from_species(row, 'atm')
    expected_mol_O = 1.0e20 + 4.0e19
    assert O_kg == pytest.approx(
        expected_mol_O * element_mmw['O'], rel=1e-12,
    )


@pytest.mark.unit
def test_populate_O_kg_sets_all_reservoirs_and_total():
    """populate_O_kg writes atm, liquid, solid, and total."""
    row = ZeroHelpfileRow()
    row['H2O_mol_atm'] = 1.0e20
    row['CO2_mol_liquid'] = 1.0e19
    row['SiO2_mol_solid'] = 5.0e18
    populate_O_kg(row)
    assert row['O_kg_atm'] > 0
    assert row['O_kg_liquid'] > 0
    assert row['O_kg_solid'] > 0
    # O_kg_total must equal the sum of the three reservoirs.
    assert row['O_kg_total'] == pytest.approx(
        row['O_kg_atm'] + row['O_kg_liquid'] + row['O_kg_solid'],
        rel=1e-12,
    )


@pytest.mark.unit
def test_populate_O_kg_idempotent():
    """Calling populate_O_kg twice gives the same result."""
    row = ZeroHelpfileRow()
    row['H2O_mol_atm'] = 1.0e20
    row['CO2_mol_liquid'] = 1.0e19
    populate_O_kg(row)
    after_first = {
        k: row[k] for k in (
            'O_kg_atm', 'O_kg_liquid', 'O_kg_solid', 'O_kg_total',
        )
    }
    populate_O_kg(row)
    for k, v in after_first.items():
        assert row[k] == v, f'{k} changed on second call: {v} → {row[k]}'


@pytest.mark.unit
def test_oxygen_skip_sites_countinvariant():
    """
    Lock the count of executable O-skip sites at 8 after Commit D.

    Commit D made oxygen a first-class element (O_kg_total populated
    from species inventory by `populate_O_kg`). Three executable skip
    sites were removed:
      - `outgas/wrapper.py` M_ele update loop
      - `outgas/common.py` copy-keys filter
      - `interior_energetics/wrapper.py::update_planet_mass`

    Eight sites KEEP the O-skip intentionally. They are semantically
    correct: these loops compute "volatile mass" or go through the
    CALLIOPE API, where O-in-silicates (part of M_int, not a mobile
    element) should NOT be counted. Specifically:
      1. `outgas/wrapper.py` desiccation threshold check
      2. `outgas/wrapper.py` escape-balance cur_m_ele sum
      3. `outgas/calliope.py` construct_guess zero-check (CALLIOPE API)
      4. `outgas/calliope.py` target dict construction (CALLIOPE API)
      5. `interior_struct/zalmoxis.py` M_volatiles computation
      6. `escape/common.py` unfractionating flux ratios
      7. `escape/wrapper.py` M_vol_initial baseline
      8. `escape/wrapper.py` fractionating escape element loop

    Comment-only matches (no executable continue / filter) are
    excluded. The `M_ele_excl_O` helper in `coupler.py` is excluded
    as a deliberate helper function.
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
                    if not pattern.search(line):
                        continue
                    # Exclude the M_ele_excl_O helper in coupler.py
                    # which deliberately contains `e != 'O'`.
                    if fpath == coupler_path:
                        continue
                    # Exclude comment-only matches (no conditional /
                    # filter). Simple heuristic: if the match is
                    # preceded by '#' on the same line (i.e. the
                    # match is inside a comment), skip it.
                    match_pos = pattern.search(line).start()
                    comment_pos = line.find('#')
                    if 0 <= comment_pos < match_pos:
                        continue
                    hits.append(f'{fpath}:{i}: {line.rstrip()}')

    expected = 8
    assert len(hits) == expected, (
        f'Expected {expected} O-skip sites after Commit D (O-as-first-class '
        f'rework removed the 3 legacy skips), found {len(hits)}:\n'
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
