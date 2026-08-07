"""Unit tests for the helpfile reference generators in ``tools/``.

The modules under test parse the helpfile column schema out of
``GetHelpfileKeys`` (``_helpfile_schema``), statically attribute producers
and consumers across ``src/proteus`` (``_helpfile_scan``), and render the
joined matrix (``generate_output_reference``). These tests exercise:

* byte-exact agreement between the parsed-and-expanded schema and the
  executed ``GetHelpfileKeys()`` (names, count, and order),
* backend return-dict extraction including templated keys and the
  albedo-to-bond_albedo merge rename,
* complete producer coverage: every column attributed or listed unresolved,
* module-conditionality pins for backend-specific columns,
* the missing-function error contract and check-mode drift detection.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    """Load a tools/ script by path, registering it so sibling imports resolve."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_docgen = _load('_docgen')
_hs = _load('_helpfile_schema')
_scan = _load('_helpfile_scan')
_load('_config_schema')
_load('generate_module_map')  # imported lazily by the condition mapping
_gor = _load('generate_output_reference')


@pytest.fixture(scope='module')
def matrix():
    """Build the full matrix once; the scan is pure and read-only."""
    return _gor.build_matrix()


def test_parsed_schema_matches_executed_gethelpfilekeys():
    """The statically parsed and template-expanded key list equals the
    executed ``GetHelpfileKeys()`` in names, count, AND order; any coupler
    edit that the parser cannot follow fails here first. Isolated in one
    test so an import regression of the heavy coupler module fails one
    named test."""
    parsed = [r['name'] for r in _hs.parse_schema()]
    _docgen.import_proteus_config()
    import importlib

    coupler = importlib.import_module('proteus.utils.coupler')
    executed = list(coupler.GetHelpfileKeys())
    assert parsed == executed
    # The schema is large and expanded: the literal block alone cannot reach
    # this count, so a parser that silently skips the loops fails the floor.
    assert len(parsed) > 700
    assert len(set(parsed)) == len(parsed)


def test_backend_extraction_resolves_templates_and_renames():
    """AGNI's return dict yields both literal keys and the per-gas ocean
    template; the merge rename maps its 'albedo' onto the schema's
    bond_albedo instead of dropping it."""
    species = _scan._species_lists()
    agni_keys = _scan.extract_backend_keys('atmos_clim/agni.py', 'run_agni', species)
    assert 'F_olr' in agni_keys
    assert 'H2O_ocean' in agni_keys  # templated output[g + '_ocean'] expansion
    assert 'albedo' in agni_keys  # renamed at the merge boundary, not here
    renames = next(r for f, fn, r in _scan.MERGE_SITES if fn == 'run_agni')
    assert renames['albedo'] == 'bond_albedo'
    # JANUS produces no ocean columns; a shared extractor bug that copied
    # AGNI's template into every backend would fail this discrimination.
    janus_keys = _scan.extract_backend_keys('atmos_clim/janus.py', 'RunJANUS', species)
    assert 'H2O_ocean' not in janus_keys


def test_backend_extraction_missing_function_raises():
    """Error contract: a declared merge site whose function vanished raises
    ScanError naming the file, instead of yielding an empty key set that
    would silently blank the producer column."""
    species = _scan._species_lists()
    with pytest.raises(_scan.ScanError, match='no_such_backend'):
        _scan.extract_backend_keys('atmos_clim/agni.py', 'no_such_backend', species)


def test_every_column_attributed_or_listed_unresolved(matrix):
    """Completeness: each schema column either has at least one producer or
    appears in the unresolved section; on the current tree the unresolved
    set is empty and every column is attributed."""
    assert len(matrix['keys']) > 700
    unattributed = [k['name'] for k in matrix['keys'] if not k['producers']]
    assert unattributed == []
    assert matrix['unresolved_events'] == []
    # The rendered page mirrors that state explicitly rather than omitting
    # the section.
    page = _gor.render(matrix)
    assert 'Columns without a statically attributed producer' in page
    assert 'None; every column above' in page


def test_backend_specific_columns_carry_their_condition(matrix):
    """Conditionality pins: ocean coverage is AGNI-only, the per-step energy
    integrals are Aragog-only, and the rock-vapour fO2 column is
    LavAtmos-only. A file-condition regression would mislabel these as
    written under every configuration."""
    by_name = {k['name']: k for k in matrix['keys']}

    ocean = by_name['H2O_ocean']['producers']
    assert [p['condition'] for p in ocean] == ['atmos_clim.module = "agni"']

    step = by_name['step_dE_F_int_J']['producers']
    assert {p['condition'] for p in step} == {'interior_energetics.module = "aragog"'}

    vap = by_name['fO2_vapourise_derived']['producers']
    assert any(p['condition'] == 'outgas.vapourise = true' for p in vap)

    # Multi-producer column keeps one entry per (file, condition) pair.
    tsurf = by_name['T_surf']['producers']
    assert len(tsurf) == len({(p['file'], p['condition']) for p in tsurf})
    assert len(tsurf) >= 4  # atmosphere backends plus interior paths


def test_check_mode_detects_page_drift(matrix, tmp_path, monkeypatch, capsys):
    """A mutated committed page makes --check exit 1 naming the regeneration
    command; after --write the same check exits 0 (round-trip contract)."""
    page = tmp_path / 'output.md'
    page.write_text(
        '# Output\n\nprose\n\n'
        '<!-- BEGIN GENERATED: helpfile-matrix -->\nstale\n'
        '<!-- END GENERATED: helpfile-matrix -->\n'
    )
    json_path = tmp_path / 'output_schema.json'
    monkeypatch.setattr(_gor, 'PAGE', page)
    monkeypatch.setattr(_gor, 'JSON_PATH', json_path)

    monkeypatch.setattr(sys, 'argv', ['generate_output_reference.py', '--check'])
    assert _gor.main() == 1

    monkeypatch.setattr(sys, 'argv', ['generate_output_reference.py', '--write'])
    assert _gor.main() == 0
    assert 'prose' in page.read_text()  # hand-written text outside markers kept
    assert json_path.exists()

    monkeypatch.setattr(sys, 'argv', ['generate_output_reference.py', '--check'])
    assert _gor.main() == 0
    out = capsys.readouterr().out
    assert 'python tools/generate_output_reference.py' in out
