"""
Unit tests for LavAtmos helper functions.

Tests:
- molecular weights from FastChem formulas
- species library completeness
- reading LavAtmos element abundance files
- normalization of elemental fractions
- LavAtmos execution wrapper (mocked)
- path preparation
- LavAtmos import
- Magma creation
- melt composition reading
- abundance conversion
- fO2 conversion
- vaporise arguments
- first iteration behaviour
- output naming
- output writing
- missing input failures
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from proteus.outgas.lavatmos import (
    _SPECIES_TABLE,
    _fastchem_weight,
    read_in_element_fracs,
    read_in_element_fracs_normalized,
    run_lavatmos,
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

    H 12.0
    O 10.0
    """

    infile = tmp_path / 'elements.dat'

    infile.write_text(
        """
# comment
H 12.0
O 10.0
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
H 0.0
O 10.0
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
H 12.0
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
H 12.0
O 11.0
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
H 12.0
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
H 0.0
O 12.0
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


def make_paths(tmp_path):
    lava = tmp_path / 'lava_comps'
    output = tmp_path / 'output'
    lavatmos = tmp_path / 'lavatmos'

    lava.mkdir()
    output.mkdir()
    lavatmos.mkdir()

    return SimpleNamespace(
        lava_comps=str(lava) + '/',
        output_dir=str(output) + '/',
        lavatmos_dir=str(lavatmos),
    )


def make_magma():
    return SimpleNamespace(
        melt_comp_name='basalt',
        T_surf=2000,
        P_volatile=100,
        melt_fraction=0.8,
        run_name='test_run',
    )


def create_melt_file(paths):
    fname = Path(paths.lava_comps) / 'basalt.csv'

    fname.write_text('SiO2,50\nMgO,30\nFeO,20\n')

    return fname


def install_fake_lavatmos(monkeypatch, output=None):
    fake_system = MagicMock()

    if output is None:
        output = pd.DataFrame(
            {
                'species': ['O2'],
                'fraction': [1.0],
            }
        )

    fake_system.vaporise.return_value = output

    fake_lavatmos = MagicMock()

    fake_lavatmos.melt_vapor_system.return_value = fake_system

    monkeypatch.setitem(
        sys.modules,
        'lavatmos3',
        fake_lavatmos,
    )

    return fake_system, fake_lavatmos


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


@pytest.mark.unit
def test_run_lavatmos_calls_paths_importer(
    tmp_path,
    monkeypatch,
):
    paths = make_paths(tmp_path)

    called = {}

    def fake_paths_importer(dirs):
        called['dirs'] = dirs
        return paths

    monkeypatch.setattr(
        'lavatmos.paths_importer',
        fake_paths_importer,
    )

    monkeypatch.setattr(
        'lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {'test': 1},
        'config',
        {'log10_fO2_vapourise': -5},
        {},
        True,
    )

    assert called['dirs'] == {'test': 1}


@pytest.mark.unit
def test_run_lavatmos_calls_set_magmaproperties(
    tmp_path,
    monkeypatch,
):
    paths = make_paths(tmp_path)

    captured = {}

    magma = make_magma()

    def fake_set(
        config,
        hf_row,
        volatile_fracs,
        dirs,
    ):
        captured['config'] = config
        captured['hf_row'] = hf_row
        captured['volatile_fracs'] = volatile_fracs
        captured['dirs'] = dirs

        return magma

    monkeypatch.setattr(
        'lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'lavatmos.set_magmaproperties',
        fake_set,
    )

    create_melt_file(paths)

    install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {'a': 1},
        'CONFIG',
        {'log10_fO2_vapourise': -4},
        {'H': 0.1},
        True,
    )

    assert captured['config'] == 'CONFIG'
    assert captured['hf_row']['log10_fO2_vapourise'] == -4
    assert captured['volatile_fracs'] == {'H': 0.1}
    assert captured['dirs'] == {'a': 1}


@pytest.mark.unit
def test_run_lavatmos_reads_melt_composition(
    tmp_path,
    monkeypatch,
):
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {},
        None,
        {'log10_fO2_vapourise': -5},
        {},
        True,
    )

    args, kwargs = fake_system.vaporise.call_args

    assert args[2] == {
        'SiO2': 50.0,
        'MgO': 30.0,
        'FeO': 20.0,
    }


@pytest.mark.unit
def test_run_lavatmos_sets_fixed_melt_pressure(
    tmp_path,
    monkeypatch,
):
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {},
        None,
        {'log10_fO2_vapourise': -5},
        {},
        True,
    )

    kwargs = fake_system.vaporise.call_args.kwargs

    assert kwargs['P_melt'] == pytest.approx(0.01)


@pytest.mark.unit
def test_run_lavatmos_converts_fO2_guess(
    tmp_path,
    monkeypatch,
):
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {},
        None,
        {'log10_fO2_vapourise': -3},
        {},
        True,
    )

    kwargs = fake_system.vaporise.call_args.kwargs

    assert kwargs['fO2_initial_guess'] == pytest.approx(1e-3)


@pytest.mark.unit
def test_run_lavatmos_first_iteration_disables_previous_fO2(
    tmp_path,
    monkeypatch,
):
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {},
        None,
        {'log10_fO2_vapourise': -5},
        {},
        True,
    )

    assert fake_system.vaporise.call_args.kwargs['fO2_tries_from_last'] is False


@pytest.mark.unit
def test_run_lavatmos_second_iteration_uses_previous_fO2(
    tmp_path,
    monkeypatch,
):
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {},
        None,
        {'log10_fO2_vapourise': -5},
        {},
        False,
    )

    assert fake_system.vaporise.call_args.kwargs['fO2_tries_from_last'] is True


@pytest.mark.unit
def test_run_lavatmos_sets_tolerance(
    tmp_path,
    monkeypatch,
):
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {},
        None,
        {'log10_fO2_vapourise': -5},
        {},
        True,
    )

    assert fake_system.vaporise.call_args.kwargs['xatol'] == pytest.approx(1e-5)


@pytest.mark.unit
def test_run_lavatmos_output_filename(
    tmp_path,
    monkeypatch,
):
    paths = make_paths(tmp_path)

    magma = make_magma()
    magma.run_name = 'special_case'

    monkeypatch.setattr(
        'lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'lavatmos.set_magmaproperties',
        lambda *args: magma,
    )

    create_melt_file(paths)

    install_fake_lavatmos(
        monkeypatch,
        pd.DataFrame({'x': [1]}),
    )

    run_lavatmos(
        {},
        None,
        {'log10_fO2_vapourise': -5},
        {},
        True,
    )

    assert (Path(paths.output_dir) / 'special_case.csv').exists()


@pytest.mark.unit
def test_run_lavatmos_output_contents(
    tmp_path,
    monkeypatch,
):
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    install_fake_lavatmos(
        monkeypatch,
        pd.DataFrame(
            {
                'species': ['O2'],
                'fraction': [0.5],
            }
        ),
    )

    run_lavatmos(
        {},
        None,
        {'log10_fO2_vapourise': -5},
        {},
        True,
    )

    result = pd.read_csv(Path(paths.output_dir) / 'test_run.csv')

    assert result['species'].iloc[0] == 'O2'
    assert result['fraction'].iloc[0] == pytest.approx(0.5)


@pytest.mark.unit
def test_run_lavatmos_missing_melt_file_raises(
    tmp_path,
    monkeypatch,
):
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    install_fake_lavatmos(monkeypatch)

    with pytest.raises(FileNotFoundError):
        run_lavatmos(
            {},
            None,
            {'log10_fO2_vapourise': -5},
            {},
            True,
        )
