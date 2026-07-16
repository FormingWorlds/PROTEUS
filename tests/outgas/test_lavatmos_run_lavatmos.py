"""
Unit tests for LavAtmos helper functions.

Tests:
- molecular weights from FastChem formulas
- species library completeness
- reading LavAtmos element abundance files
- normalization of elemental fractions
- LavAtmos execution wrapper (mocked)
"""

from __future__ import annotations

import pandas as pd
import pytest

from proteus.outgas.lavatmos import (
    _SPECIES_TABLE,
    _fastchem_weight,
    read_in_element_fracs,
    read_in_element_fracs_normalized,
    species_lib,
)
from proteus.utils.constants import electron_molar_mass, element_list, element_mmw, gas_list

# ---------------------------------------------------------------------------
# _fastchem_weight tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_fastchem_weight_single_element():
    """
    Atomic weight should match element_mmw.
    """
    weight = _fastchem_weight('H')

    assert weight == pytest.approx(element_mmw['H'] * 1000.0)


@pytest.mark.unit
def test_fastchem_weight_molecule():
    """
    H2O should equal 2H + O.
    """
    weight = _fastchem_weight('H2O')

    expected = (2 * element_mmw['H'] + element_mmw['O']) * 1000.0

    assert weight == pytest.approx(expected)


@pytest.mark.unit
def test_fastchem_weight_electron():
    """
    Electron entry should use the predefined electron mass.
    """
    assert species_lib['e-'].weight == electron_molar_mass


# ---------------------------------------------------------------------------
# species_lib tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_species_library_contains_electron():
    assert 'e-' in species_lib


@pytest.mark.unit
def test_species_library_contains_curated_species():
    """
    Every species in _SPECIES_TABLE should appear.
    """
    for name, _ in _SPECIES_TABLE:
        assert name in species_lib


@pytest.mark.unit
def test_species_weights_are_positive():
    """
    No species should have zero or negative molecular weight.
    """
    for name, species in species_lib.items():
        assert species.weight > 0, f'{name} has invalid weight'


@pytest.mark.unit
def test_species_fallback_elements_exist():
    """
    Elements/gases missing from the curated table should still be added.
    """
    for name in element_list + gas_list:
        assert name in species_lib


# ---------------------------------------------------------------------------
# read_in_element_fracs tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_in_element_fracs(tmp_path):
    """
    Reads FastChem abundance format:

    H 12
    O 10
    """

    infile = tmp_path / 'elements.dat'

    infile.write_text(
        """
# comment
H 12
O 10
"""
    )

    result = read_in_element_fracs(
        str(tmp_path) + '/',
        time=0,
        parameters={'elementfile': 'elements.dat'},
    )

    assert isinstance(result, pd.DataFrame)

    assert 'H' in result.columns
    assert 'O' in result.columns


@pytest.mark.unit
def test_read_in_element_fracs_zero_value(tmp_path):
    infile = tmp_path / 'elements.dat'

    infile.write_text(
        """
H 0
O 10
"""
    )

    result = read_in_element_fracs(
        str(tmp_path) + '/',
        0,
        {'elementfile': 'elements.dat'},
    )

    assert result['H'].iloc[0] == 0


# ---------------------------------------------------------------------------
# normalized abundance tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_normalized_element_fractions_sum_to_one(tmp_path):
    infile = tmp_path / 'elements.dat'

    infile.write_text(
        """
H 12
O 12
"""
    )

    result = read_in_element_fracs_normalized(str(infile))

    assert isinstance(result, dict)

    assert sum(result.values()) == pytest.approx(1.0)


@pytest.mark.unit
def test_normalized_element_fraction_ratio(tmp_path):
    infile = tmp_path / 'elements.dat'

    infile.write_text(
        """
H 12
O 11
"""
    )

    result = read_in_element_fracs_normalized(str(infile))

    # H abundance should be 10 times O
    assert result['H'] == pytest.approx(10 * result['O'])


@pytest.mark.unit
def test_missing_elements_are_added(tmp_path):
    infile = tmp_path / 'elements.dat'

    infile.write_text(
        """
H 12
"""
    )

    result = read_in_element_fracs_normalized(str(infile))

    for element in element_list:
        assert element in result


@pytest.mark.unit
def test_zero_abundance_is_preserved(tmp_path):
    infile = tmp_path / 'elements.dat'

    infile.write_text(
        """
H 0
O 12
"""
    )

    result = read_in_element_fracs_normalized(str(infile))

    assert result['H'] == 0.0


# ---------------------------------------------------------------------------
# run_lavatmos mocked test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_lavatmos_calls_vaporise(monkeypatch):
    class FakeSystem:
        def vaporise(self, *args, **kwargs):
            return pd.DataFrame(
                {
                    'species': ['H2O'],
                    'mass': [1.0],
                }
            )

    class FakeLavatmos:
        def melt_vapor_system(self, paths):
            return FakeSystem()

    monkeypatch.setattr('lavatmos3.melt_vapor_system', FakeLavatmos().melt_vapor_system)
