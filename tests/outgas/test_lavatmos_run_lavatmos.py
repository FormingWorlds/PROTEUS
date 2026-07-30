"""
Unit tests for the LavAtmos rock-vapourisation helpers in
``src/proteus/outgas/lavatmos.py``.

Invariants and contract clauses exercised here:

- Molecular weights built from FastChem formula strings reproduce the IUPAC
  atomic-weight table (``proteus.utils.constants.element_mmw``) in g/mol,
  including the electron entry.
- Every species carried by the library has a strictly positive, physically
  bounded molar mass.
- Elemental number fractions read back from a LavAtmos abundance file close to
  unity (conservation), stay inside [0, 1], and honour the FastChem
  ``e_j = 10**(x_j - 12)`` logarithmic definition.
- The all-zero abundance file degrades to an all-zero fraction dictionary
  instead of dividing by zero.
- ``run_lavatmos`` passes surface state, melt composition, config-driven
  solver settings, and the fO2 warm-start flag through to the melt-vapour
  system, writes the diagnostic csv, and propagates a missing melt-composition
  file without running the solve.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from proteus.outgas.lavatmos import (
    _SPECIES_TABLE,
    _fastchem_weight,
    read_in_element_fracs_normalized,
    run_lavatmos,
    species_lib,
)
from proteus.utils.constants import electron_molar_mass, element_list, element_mmw, gas_list

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# ---------------------------------------------------------------------------
# _fastchem_weight tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_fastchem_weight_single_element():
    """A bare atomic symbol carries that element's molar mass, expressed in g/mol.

    Hydrogen is the lightest element in the table, so the kg-to-g conversion is
    the dominant failure mode here: 1.008 g/mol versus 1.008e-3 kg/mol.
    """
    weight = _fastchem_weight('H')

    assert weight == pytest.approx(element_mmw['H'] * 1000.0, rel=1e-12)
    # Sign guard: a molar mass is strictly positive.
    assert weight > 0.0
    # Scale guard: g/mol, so of order 1, not 1e-3 (kg/mol left unconverted) and
    # not 1e3 (conversion applied twice).
    assert 0.5 < weight < 5.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_fastchem_weight_molecule():
    """A multi-atom FastChem formula sums its stoichiometric atomic masses.

    Water is chosen because the two hydrogens make the stoichiometric
    coefficient visible: dropping the ``2`` shifts the answer by a full
    hydrogen mass, far outside tolerance.
    """
    weight = _fastchem_weight('H2O')

    expected = (2 * element_mmw['H'] + element_mmw['O']) * 1000.0

    assert weight == pytest.approx(expected, rel=1e-12)
    # Coefficient guard: treating H2O as one H plus one O lands at 17.007 g/mol.
    wrong_single_h = (element_mmw['H'] + element_mmw['O']) * 1000.0
    assert abs(weight - wrong_single_h) > 0.5
    # Sign guard.
    assert weight > 0.0
    # Scale guard: 18 g/mol, not 0.018 (kg/mol) and not 18000.
    assert 10.0 < weight < 30.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_fastchem_weight_electron():
    """The electron entry carries the electron molar mass, not a formula sum.

    ``mol_to_ele`` cannot decompose ``e-`` into elements, so the library
    injects the electron mass directly. The physical check is that it sits
    roughly 1800 times below the hydrogen atom.
    """
    weight = species_lib['e-'].weight

    assert weight == pytest.approx(electron_molar_mass, rel=1e-12)
    # Sign guard: the electron still has mass, so a zero or negative entry is a bug.
    assert weight > 0.0
    # Proton-to-electron mass ratio is ~1836; the hydrogen atom includes its
    # electron, so 1837 is expected. rel=1e-2 absorbs the rounded table values.
    assert element_mmw['H'] * 1000.0 / weight == pytest.approx(1837.0, rel=1e-2)


@pytest.mark.unit
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_molecular_weights_match_iupac_atomic_weight_table():
    """Molar masses reproduce the IUPAC atomic-weight table for a spread of species.

    Reference: the IUPAC atomic weights tabulated in
    ``proteus.utils.constants.element_mmw`` (source noted there as
    https://iupac.qmul.ac.uk/AtWt/), converted from kg/mol to g/mol. The cases
    span a light hydride (H2O), a rock-forming oxide (SiO2), a bare heavy metal
    (Fe), and the electron, so a single wrong path cannot satisfy all four.
    """
    # Independently summed from the table, in g/mol.
    ref_h2o = (2 * element_mmw['H'] + element_mmw['O']) * 1000.0  # 18.015
    ref_sio2 = (element_mmw['Si'] + 2 * element_mmw['O']) * 1000.0  # 60.083
    ref_fe = element_mmw['Fe'] * 1000.0  # 55.845

    assert species_lib['H2O'].weight == pytest.approx(ref_h2o, rel=1e-12)
    assert species_lib['SiO2'].weight == pytest.approx(ref_sio2, rel=1e-12)
    assert species_lib['Fe'].weight == pytest.approx(ref_fe, rel=1e-12)
    assert species_lib['e-'].weight == pytest.approx(electron_molar_mass, rel=1e-12)

    # Coefficient guard: reading SiO2 as SiO gives 44.084 g/mol, 16 g/mol away.
    wrong_sio = (element_mmw['Si'] + element_mmw['O']) * 1000.0
    assert abs(species_lib['SiO2'].weight - wrong_sio) > 1.0

    # Unit guard: skipping the kg-to-g conversion would leave 0.060 kg/mol, and
    # applying it twice would give 6.0e4. Pin the g/mol decade for all three
    # neutral species.
    for name in ('H2O', 'SiO2', 'Fe'):
        assert 1.0 < species_lib[name].weight < 1000.0

    # Sign guard, and the electron must not collapse onto the neutral scale:
    # forgetting the special case would make ``mol_to_ele('e-')`` fail or return
    # an element sum, never a sub-milligram-per-mole value.
    assert species_lib['e-'].weight > 0.0
    assert species_lib['e-'].weight < 1e-3

    # Ordering across the four cases: electron << H2O < Fe < SiO2.
    assert species_lib['e-'].weight < species_lib['H2O'].weight
    assert species_lib['H2O'].weight < species_lib['Fe'].weight
    assert species_lib['Fe'].weight < species_lib['SiO2'].weight


# ---------------------------------------------------------------------------
# species_lib tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_species_library_contains_electron():
    """The library exposes an electron entry keyed and named ``e-``.

    FastChem treats free electrons as a species, so the wrapper must be able to
    look one up by the same string FastChem uses.
    """
    assert 'e-' in species_lib

    electron = species_lib['e-']

    assert electron.name == 'e-'
    assert electron.fc_name == 'e-'


@pytest.mark.unit
def test_species_library_contains_curated_species():
    """Every curated table row becomes a library entry with its FastChem alias kept.

    The curated FastChem name is not derivable from the plain formula (element
    order and explicit ``1`` counts differ), so the alias must survive the
    build.
    """
    for name, _ in _SPECIES_TABLE:
        assert name in species_lib

    # The alias is preserved rather than regenerated from the plain formula.
    assert species_lib['H2O'].fc_name == 'H2O1'
    assert species_lib['SO2'].fc_name == 'O2S1'
    # The electron plus the fallback back-fill mean the library is never
    # smaller than the curated table.
    assert len(species_lib) > len(_SPECIES_TABLE)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_species_weights_are_positive():
    """Every library species has a strictly positive, physically bounded molar mass.

    A zero or negative weight would divide by zero in ``run_vapourisation``
    when converting atmospheric mass to moles, so the bound is checked for the
    whole library and pinned for three representatives.
    """
    # Pinned against sums over element_mmw, in g/mol.
    assert species_lib['CO2'].weight == pytest.approx(
        (element_mmw['C'] + 2 * element_mmw['O']) * 1000.0, rel=1e-12
    )
    assert species_lib['MgO'].weight == pytest.approx(
        (element_mmw['Mg'] + element_mmw['O']) * 1000.0, rel=1e-12
    )
    assert species_lib['Fe'].weight == pytest.approx(element_mmw['Fe'] * 1000.0, rel=1e-12)

    # Coefficient guard: CO instead of CO2 lands 16 g/mol lower.
    wrong_co = (element_mmw['C'] + element_mmw['O']) * 1000.0
    assert abs(species_lib['CO2'].weight - wrong_co) > 1.0

    for name, species in species_lib.items():
        # Sign guard across the whole library.
        assert species.weight > 0, f'{name} has invalid weight'
        # Scale guard: the heaviest entry in the library is xenon at 131 g/mol,
        # so anything above 300 g/mol signals a doubled conversion or a bad sum.
        assert species.weight < 300.0, f'{name} has implausible weight'


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_species_fallback_elements_exist():
    """Elements and gases missing from the curated table are still back-filled.

    The fallback assumes the plain name is a valid FastChem formula, which holds
    for single-symbol elements such as the noble gases; their weights must
    equal the tabulated atomic masses in g/mol.
    """
    for name in element_list + gas_list:
        assert name in species_lib

    for name in element_list:
        assert species_lib[name].weight == pytest.approx(element_mmw[name] * 1000.0, rel=1e-12)
        # Sign guard on each element mass.
        assert species_lib[name].weight > 0.0


# ---------------------------------------------------------------------------
# normalized abundance tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_normalized_element_fractions_sum_to_one(tmp_path):
    """Elemental number fractions close to unity for symmetric and skewed inputs.

    Closure is checked twice: an equal-abundance file where symmetry alone
    could hide a normalisation bug, and a decade-skewed file where it cannot.
    """
    symmetric = tmp_path / 'elements_symmetric.dat'
    symmetric.write_text('H 12.0\nO 12\n')

    result = read_in_element_fracs_normalized(str(symmetric))

    assert sum(result.values()) == pytest.approx(1.0, rel=1e-12)
    assert result['H'] == pytest.approx(0.5, rel=1e-12)

    # Skewed case: H one decade above O, so an unnormalised or mis-scaled
    # return cannot survive closure by symmetry.
    skewed = tmp_path / 'elements_skewed.dat'
    skewed.write_text('H 12.0\nO 11.0\n')

    skewed_result = read_in_element_fracs_normalized(str(skewed))

    assert sum(skewed_result.values()) == pytest.approx(1.0, rel=1e-12)
    # Boundedness: number fractions live in [0, 1].
    for element, frac in skewed_result.items():
        assert 0.0 <= frac <= 1.0, f'{element} fraction out of range'


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_normalized_element_fraction_ratio(tmp_path):
    """A one-dex abundance difference becomes a factor-ten number-fraction ratio.

    FastChem abundances are logarithmic (``e_j = 10**(x_j - 12)``), so 12.0
    versus 11.0 must map onto 10:1, not onto the 12:11 the raw column values
    would suggest.
    """
    infile = tmp_path / 'elements.dat'
    infile.write_text('H 12.0\nO 11.0\n')

    result = read_in_element_fracs_normalized(str(infile))

    assert result['H'] == pytest.approx(10.0 * result['O'], rel=1e-12)
    # Absolute pins: the only two non-zero elements share the 11 parts.
    assert result['H'] == pytest.approx(10.0 / 11.0, rel=1e-12)
    assert result['O'] == pytest.approx(1.0 / 11.0, rel=1e-12)
    # Log-versus-linear guard: reading the column linearly would give
    # 12/23 = 0.522 for H rather than 0.909.
    assert abs(result['H'] - 12.0 / 23.0) > 0.1
    # Sign guard: number fractions are positive.
    assert result['H'] > 0.0


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_missing_elements_are_added(tmp_path):
    """Elements absent from the LavAtmos file appear as exact zeros, keeping closure.

    Downstream code indexes every entry of ``element_list``, so a partial file
    (here hydrogen only) must be padded rather than raising a key error, and
    the padding must not perturb the normalisation.
    """
    infile = tmp_path / 'elements.dat'
    infile.write_text('H 12.0\n')

    result = read_in_element_fracs_normalized(str(infile))

    for element in element_list:
        assert element in result

    # Padding carries no abundance, so the single listed element takes it all.
    assert result['H'] == pytest.approx(1.0, rel=1e-12)
    assert sum(result.values()) == pytest.approx(1.0, rel=1e-12)
    for element in element_list:
        if element != 'H':
            assert result[element] == pytest.approx(0.0, abs=1e-30)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_zero_abundance_is_preserved(tmp_path):
    """A zero-abundance element stays exactly zero, and an all-zero file stays finite.

    The source special-cases zero so that ``10**(0 - 12)`` is never applied to
    a genuine absence, and guards the normalisation against a vanishing total.
    """
    infile = tmp_path / 'elements.dat'
    infile.write_text('H 0.0\nO 12.0\n')

    result = read_in_element_fracs_normalized(str(infile))

    assert result['H'] == pytest.approx(0.0, abs=1e-30)
    # The remaining element absorbs the whole budget, so closure still holds.
    assert result['O'] == pytest.approx(1.0, rel=1e-12)
    assert sum(result.values()) == pytest.approx(1.0, rel=1e-12)

    # Limit input: every abundance zero drives the total below the 1e-30 guard,
    # which must return finite zeros rather than dividing by zero.
    empty = tmp_path / 'elements_empty.dat'
    empty.write_text('H 0.0\nO 0.0\n')

    empty_result = read_in_element_fracs_normalized(str(empty))

    assert sum(empty_result.values()) == pytest.approx(0.0, abs=1e-30)
    for element, frac in empty_result.items():
        assert math.isfinite(frac), f'{element} fraction is not finite'


# ---------------------------------------------------------------------------
# run_lavatmos mocked test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_lavatmos_calls_vaporise(tmp_path, monkeypatch):
    """run_lavatmos drives the LavAtmos melt-vapour system's vaporise() step.

    On the first iteration the previous-fO2 warm start is disabled, so vaporise
    is called exactly once with fO2_tries_from_last=False.
    """
    paths = make_paths(tmp_path)
    monkeypatch.setattr('proteus.outgas.lavatmos.paths_importer', lambda dirs: paths)
    monkeypatch.setattr(
        'proteus.outgas.lavatmos.set_magmaproperties', lambda *args: make_magma()
    )
    create_melt_file(paths)
    fake_system, _ = install_fake_lavatmos(monkeypatch)

    run_lavatmos({'a': 1}, make_config(), {'fO2_vapourise_derived': -5}, {'H': 0.1}, True)

    fake_system.vaporise.assert_called_once()
    # First iteration must not reuse the previous solve's fO2 as a warm start.
    assert fake_system.vaporise.call_args.kwargs['fO2_tries_from_last'] is False


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


def make_config(p_melt=0.01, xatol=1e-5):
    """Config stub exposing the outgas.lavatmos fields run_lavatmos reads."""
    return SimpleNamespace(
        outgas=SimpleNamespace(lavatmos=SimpleNamespace(P_melt=p_melt, xatol=xatol))
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
    """The caller's directory dictionary is resolved once through paths_importer.

    LavAtmos writes its element abundances and FastChem output as a side effect
    into the run's output tree, so the directory dictionary must reach the path
    resolver unaltered, and only one path set may be built per solve.
    """
    paths = make_paths(tmp_path)

    called = {}
    calls = []

    def fake_paths_importer(dirs):
        called['dirs'] = dirs
        calls.append(dirs)
        return paths

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.paths_importer',
        fake_paths_importer,
    )

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, fake_lavatmos = install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {'test': 1},
        make_config(),
        {'fO2_vapourise_derived': -5},
        {},
        True,
    )

    assert called['dirs'] == {'test': 1}
    # One resolution per solve; the resolved set is what the solver is built on.
    assert len(calls) == 1
    fake_lavatmos.melt_vapor_system.assert_called_once_with(paths)
    fake_system.vaporise.assert_called_once()


@pytest.mark.unit
def test_run_lavatmos_calls_set_magmaproperties(
    tmp_path,
    monkeypatch,
):
    """Surface state reaches the magma-property builder without substitution.

    The magma properties fix the surface temperature and volatile pressure that
    the melt-vapour equilibrium is solved at, so config, helpfile row, volatile
    fractions, and directories must all arrive as the caller supplied them.
    """
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
        'proteus.outgas.lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.set_magmaproperties',
        fake_set,
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    config = make_config()

    run_lavatmos(
        {'a': 1},
        config,
        {'fO2_vapourise_derived': -4},
        {'H': 0.1},
        True,
    )

    assert captured['config'] is config
    assert captured['hf_row']['fO2_vapourise_derived'] == -4
    assert captured['volatile_fracs'] == {'H': 0.1}
    assert captured['dirs'] == {'a': 1}
    # The surface state from the builder is what the solve is handed: T_surf and
    # P_volatile are the first two positional arguments to vaporise.
    args = fake_system.vaporise.call_args.args
    assert args[0] == pytest.approx(magma.T_surf, rel=1e-12)
    assert args[1] == pytest.approx(magma.P_volatile, rel=1e-12)


@pytest.mark.unit
def test_run_lavatmos_reads_melt_composition(
    tmp_path,
    monkeypatch,
):
    """The melt composition csv is parsed into the oxide dictionary vaporise expects.

    Oxide weights are read as floats from a headerless two-column file; a
    mis-parse would either shift the values or leave them as strings, both of
    which break the melt-activity calculation.
    """
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {},
        make_config(),
        {'fO2_vapourise_derived': -5},
        {},
        True,
    )

    melt_comp = fake_system.vaporise.call_args.args[2]

    # Asymmetric composition, so a swapped or dropped row is visible.
    assert melt_comp == pytest.approx({'SiO2': 50.0, 'MgO': 30.0, 'FeO': 20.0}, rel=1e-12)
    # Every abundance is a float, not the raw string from the csv.
    for oxide, abund in melt_comp.items():
        assert isinstance(abund, float), f'{oxide} abundance was not coerced to float'


@pytest.mark.unit
def test_run_lavatmos_uses_config_melt_pressure(
    tmp_path,
    monkeypatch,
):
    """The melt-activity pressure handed to vaporise comes from config, not a
    hard-coded constant."""
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    # Non-default value so the assertion proves config drives P_melt rather
    # than matching the retired 0.01 bar constant.
    run_lavatmos(
        {},
        make_config(p_melt=0.05),
        {'fO2_vapourise_derived': -5},
        {},
        True,
    )

    kwargs = fake_system.vaporise.call_args.kwargs

    assert kwargs['P_melt'] == pytest.approx(0.05)
    # Discrimination guard: the old hard-coded default would give 0.01.
    assert kwargs['P_melt'] != pytest.approx(0.01)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_run_lavatmos_converts_fO2_guess(
    tmp_path,
    monkeypatch,
):
    """The stored log10 oxygen fugacity becomes a linear fugacity in bar.

    The helpfile carries log10(fO2/bar); the solver wants the fugacity itself,
    so a shift of -3 dex must become 1e-3 bar. A value of -3 is used rather
    than 0 because 10**0 equals exp(0) equals 1 and would not discriminate.
    """
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {},
        make_config(),
        {'fO2_vapourise_derived': -3},
        {},
        True,
    )

    kwargs = fake_system.vaporise.call_args.kwargs
    guess = kwargs['fO2_initial_guess']

    assert guess == pytest.approx(1e-3, rel=1e-12)
    # Sign guard on the exponent: a dropped minus sign lands at 1e3 bar.
    assert abs(guess - 1e3) > 1.0
    # Base guard: exp(-3) = 0.0498, three orders above the correct value.
    assert abs(guess - math.exp(-3.0)) > 1e-2
    # A fugacity is strictly positive and, at a reducing 3 dex below 1 bar,
    # well below unity.
    assert 0.0 < guess < 1.0


@pytest.mark.unit
def test_run_lavatmos_first_iteration_disables_previous_fO2(
    tmp_path,
    monkeypatch,
):
    """On the first iteration there is no previous solve to warm-start the fO2 from.

    Reusing a stale bracket before any solve has run would seed the root find
    with an undefined value, so the flag must be off while the explicit initial
    guess is still supplied.
    """
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {},
        make_config(),
        {'fO2_vapourise_derived': -5},
        {},
        True,
    )

    kwargs = fake_system.vaporise.call_args.kwargs

    assert kwargs['fO2_tries_from_last'] is False
    # The explicit guess is still handed over, so the solve is not left without
    # a starting point.
    assert kwargs['fO2_initial_guess'] == pytest.approx(1e-5, rel=1e-12)


@pytest.mark.unit
def test_run_lavatmos_second_iteration_uses_previous_fO2(
    tmp_path,
    monkeypatch,
):
    """After the first iteration the previous solve's fO2 is reused as a warm start.

    Continuing from the last converged fugacity is what keeps the coupled loop
    affordable, and it must not displace the explicit guess derived from the
    helpfile.
    """
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    run_lavatmos(
        {},
        make_config(),
        {'fO2_vapourise_derived': -5},
        {},
        False,
    )

    kwargs = fake_system.vaporise.call_args.kwargs

    assert kwargs['fO2_tries_from_last'] is True
    # The helpfile-derived guess is passed regardless of the warm-start flag.
    assert kwargs['fO2_initial_guess'] == pytest.approx(1e-5, rel=1e-12)


@pytest.mark.unit
def test_run_lavatmos_uses_config_tolerance(
    tmp_path,
    monkeypatch,
):
    """The fO2-solve tolerance handed to vaporise comes from config, not a
    hard-coded constant."""
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    fake_system, _ = install_fake_lavatmos(monkeypatch)

    # Non-default value so the assertion proves config drives xatol rather
    # than matching the retired 1e-5 constant.
    run_lavatmos(
        {},
        make_config(xatol=2e-6),
        {'fO2_vapourise_derived': -5},
        {},
        True,
    )

    assert fake_system.vaporise.call_args.kwargs['xatol'] == pytest.approx(2e-6)
    # Discrimination guard: the old hard-coded default would give 1e-5.
    assert fake_system.vaporise.call_args.kwargs['xatol'] != pytest.approx(1e-5)


@pytest.mark.unit
def test_run_lavatmos_output_filename(
    tmp_path,
    monkeypatch,
):
    """The diagnostic csv is named after the magma run name, inside the output tree.

    Several vapourisation runs share one output directory, so the run name and
    not a fixed filename must determine where the diagnostics land.
    """
    paths = make_paths(tmp_path)

    magma = make_magma()
    magma.run_name = 'special_case'

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.set_magmaproperties',
        lambda *args: magma,
    )

    create_melt_file(paths)

    install_fake_lavatmos(
        monkeypatch,
        pd.DataFrame({'x': [1]}),
    )

    run_lavatmos(
        {},
        make_config(),
        {'fO2_vapourise_derived': -5},
        {},
        True,
    )

    assert (Path(paths.output_dir) / 'special_case.csv').exists()
    # Nothing is written under the default run name, and exactly one csv appears.
    assert not (Path(paths.output_dir) / 'proteus_run.csv').exists()
    assert [p.name for p in Path(paths.output_dir).glob('*.csv')] == ['special_case.csv']


@pytest.mark.unit
def test_run_lavatmos_output_contents(
    tmp_path,
    monkeypatch,
):
    """The solver's vapour composition is written through to the diagnostic csv.

    The csv is the human-readable record of the melt-vapour solve, so the
    species labels and their fractions must round-trip unchanged.
    """
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    create_melt_file(paths)

    install_fake_lavatmos(
        monkeypatch,
        pd.DataFrame(
            {
                'species': ['O2', 'SiO'],
                'fraction': [0.3, 0.7],
            }
        ),
    )

    run_lavatmos(
        {},
        make_config(),
        {'fO2_vapourise_derived': -5},
        {},
        True,
    )

    result = pd.read_csv(Path(paths.output_dir) / 'test_run.csv')

    assert list(result['species']) == ['O2', 'SiO']
    # Asymmetric fractions, so a row swap on write would be visible.
    assert result['fraction'].iloc[0] == pytest.approx(0.3, rel=1e-12)
    assert result['fraction'].iloc[1] == pytest.approx(0.7, rel=1e-12)
    assert len(result) == 2


@pytest.mark.unit
def test_run_lavatmos_missing_melt_file_raises(
    tmp_path,
    monkeypatch,
):
    """A missing melt-composition file aborts before any melt-vapour solve is run.

    Silently continuing with an empty composition would hand the solver a melt
    with no oxides, so the read error must propagate and no diagnostic output
    may be produced.
    """
    paths = make_paths(tmp_path)

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.paths_importer',
        lambda dirs: paths,
    )

    monkeypatch.setattr(
        'proteus.outgas.lavatmos.set_magmaproperties',
        lambda *args: make_magma(),
    )

    # The composition csv is deliberately not created.
    fake_system, _ = install_fake_lavatmos(monkeypatch)

    with pytest.raises(FileNotFoundError):
        run_lavatmos(
            {},
            make_config(),
            {'fO2_vapourise_derived': -5},
            {},
            True,
        )

    # No side effect: the solve never ran and nothing was written.
    assert fake_system.vaporise.call_count == 0
    assert list(Path(paths.output_dir).glob('*.csv')) == []
