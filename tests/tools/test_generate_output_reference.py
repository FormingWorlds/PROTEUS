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

import ast
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
    ScanError naming both the missing symbol and the file, instead of
    yielding an empty key set that would silently blank the producer
    column; the same file still extracts through a valid function after
    the failure (no broken state left behind)."""
    species = _scan._species_lists()
    with pytest.raises(_scan.ScanError, match='no_such_backend'):
        _scan.extract_backend_keys('atmos_clim/agni.py', 'no_such_backend', species)
    with pytest.raises(_scan.ScanError, match='agni'):
        _scan.extract_backend_keys('atmos_clim/agni.py', 'no_such_backend', species)
    assert 'F_olr' in _scan.extract_backend_keys('atmos_clim/agni.py', 'run_agni', species)


def test_every_column_attributed_or_listed_unresolved(matrix):
    """Completeness: each schema column either has at least one producer or
    appears in the unresolved section; on the current tree every column is
    attributed and unresolved events are read sites with computed keys."""
    assert len(matrix['keys']) > 700
    unattributed = [k['name'] for k in matrix['keys'] if not k['producers']]
    assert unattributed == []
    expected_unresolved_reads = [
        'src/proteus/atmos_clim/agni.py::_validate_surface_state:dynamic key name',
        'src/proteus/escape/boreas.py::_set_boreas_params:template <g>_vmr_xuv',
        'src/proteus/escape/boreas.py::_set_boreas_params:template <g>_vmr_xuv',
        'src/proteus/escape/common.py::calc_unfract_fluxes:dynamic key e + key',
        "src/proteus/escape/wrapper.py::calc_new_elements:dynamic key f'{e}{key}'",
        "src/proteus/escape/wrapper.py::escapable_mass:dynamic key f'{e}{key}'",
        'src/proteus/observe/petitRADTRANS.py::_get_mix:dynamic key key',
        "src/proteus/outgas/atmodeller.py::_populate_volatile_element_reservoirs:dynamic key f'{sp}_kg_{r}'",
        'src/proteus/outgas/atmodeller.py::calc_surface_pressures_atmodeller:dynamic key key',
        'src/proteus/outgas/atmodeller.py::calc_surface_pressures_atmodeller:dynamic key key',
        'src/proteus/plot/cpl_global.py::plot_global:dynamic key k',
        'src/proteus/plot/cpl_orbit.py::_plot_orbit_snapshot:dynamic key ecc_col',
        'src/proteus/plot/cpl_orbit.py::_plot_orbit_snapshot:dynamic key sma_col',
    ]
    actual_reads = sorted(
        f'{e["file"]}::{e["function"]}:{e["reason"]}' for e in matrix['unresolved_events']
    )
    assert actual_reads == expected_unresolved_reads
    for event in matrix['unresolved_events']:
        assert event['kind'] == 'read'
        path = _gor.REPO_ROOT / event['file']
        assert path.is_file(), f'{event["file"]} does not exist'
        source = path.read_text()
        lines = source.splitlines()
        assert 1 <= event['line'] <= len(lines), (
            f'Line {event["line"]} out of range in {event["file"]}'
        )
        tree = ast.parse(source)
        nodes_at_line = [
            n for n in ast.walk(tree) if getattr(n, 'lineno', None) == event['line']
        ]
        assert len(nodes_at_line) > 0
        reason_text = event['reason']
        if reason_text.startswith('template <'):
            core = reason_text.removeprefix('template <')
            var_part, _, suffix_part = core.partition('>')
            assert any(
                var_part in ast.unparse(n) and suffix_part in ast.unparse(n)
                for n in nodes_at_line
            )
        elif reason_text.startswith('dynamic key '):
            expr_str = reason_text.removeprefix('dynamic key ')
            assert any(expr_str in ast.unparse(n) for n in nodes_at_line)
    # The rendered page mirrors that state explicitly.
    page = _gor.render(matrix)
    assert 'Reads with computed keys' in page
    assert 'A computed key is a helpfile column name constructed dynamically at runtime' in page
    assert 'these read sites are not attributed to specific columns' in page
    assert 'Writes with computed keys' not in page


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


def test_committed_page_and_json_are_current(matrix):
    """The committed output.md region and JSON byte-match a fresh render, and
    every column row carries a unit; a coupler edit cannot land without
    regeneration, and a unit-extraction regression surfaces here."""
    fresh_page = _docgen.normalize(
        _docgen.replace_between_markers(
            _gor.PAGE.read_text(), 'GENERATED: helpfile-matrix', _gor.render(matrix)
        )
    )
    assert _gor.PAGE.read_text() == fresh_page
    assert _gor.JSON_PATH.read_text() == _docgen.dump_json(matrix)
    unitless = [k['name'] for k in matrix['keys'] if not k['unit']]
    assert unitless == []


def test_declared_scan_tables_are_all_live(monkeypatch):
    """Staleness guard for the declared attribution tables: with the template
    overrides and suppression list emptied, the scan must report an
    unresolved event matching every table entry, proving each entry still
    corresponds to a live write site in the source. A stale entry (its code
    refactored away) would produce no event and fail here."""
    overrides = dict(_scan.TEMPLATE_OVERRIDES)
    suppressed = set(_scan.SUPPRESSED_DYNAMIC_WRITES)
    assert overrides and suppressed  # the guard itself must have teeth

    monkeypatch.setattr(_scan, 'TEMPLATE_OVERRIDES', {})
    monkeypatch.setattr(_scan, 'SUPPRESSED_DYNAMIC_WRITES', set())
    events = _scan.scan_tree()['unresolved']
    by_file: dict[str, list[str]] = {}
    for rel, _line, reason, *_ in events:
        by_file.setdefault(rel, []).append(reason)

    for rel, func, pattern in overrides:
        needle = pattern.replace('<?>', '<')  # events name the real variable
        prefix, _sep, suffix = needle.partition('<')
        matched = any(
            reason.startswith(f'template {prefix}') and reason.endswith(suffix)
            for r_rel, _line, reason, _kind, r_func in events
            if r_rel == rel and r_func == func
        )
        assert matched, (
            f'TEMPLATE_OVERRIDES entry ({rel}, {func}, {pattern}) matches no access site'
        )

    for rel, _function in suppressed:
        assert by_file.get(rel), f'SUPPRESSED_DYNAMIC_WRITES names {rel} but no event arises'

    # EXTRA_PRODUCERS: the ratio loop must still be a live dynamic write.
    for rel, _function, _pattern in _scan.EXTRA_PRODUCERS:
        assert by_file.get(rel), f'EXTRA_PRODUCERS names {rel} but no event arises'


@pytest.mark.parametrize(
    ('code', 'reason'),
    [
        (
            'def f(hf_row, k):\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row, element):\n    return hf_row.get(f"{element}_unknown")\n',
            'template <element>_unknown',
        ),
        (
            'def f(hf_row, k):\n    return hf_row[k]\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row):\n    k = "a"\n    k = dyn()\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row):\n    k = "a"\n    k += "b"\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row):\n    k = "a"\n    if (k := dyn()):\n        pass\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row):\n    k = "a"\n    k: int = 1\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row, it):\n    k = "a"\n    for k in it:\n        pass\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'async def f(hf_row, it):\n    k = "a"\n    async for k in it:\n        pass\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row, ctx):\n    k = "a"\n    with ctx as k:\n        pass\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'async def f(hf_row, ctx):\n    k = "a"\n    async with ctx as k:\n        pass\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row):\n    k = "a"\n    try:\n        pass\n    except Exception as k:\n        pass\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row, it):\n    k = "a"\n    _ = [hf_row.get(k) for k in it]\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row, val):\n    k = "a"\n    match val:\n        case k:\n            pass\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row, val):\n    k = "a"\n    match val:\n        case [*k]:\n            pass\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row, val):\n    k = "a"\n    match val:\n        case {1: k}:\n            pass\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row, pair):\n    k = "a"\n    k, *other = pair\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row):\n    k = "a"\n    del k\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
    ],
    ids=[
        'get_var',
        'get_template',
        'subscript_var',
        'assign',
        'augassign',
        'walrus',
        'annassign',
        'for',
        'async_for',
        'with',
        'async_with',
        'except',
        'comprehension',
        'match_as',
        'match_star',
        'match_mapping',
        'unpack_starred',
        'del',
    ],
)
def test_computed_key_reads_reported_as_unresolved(code, reason):
    """Variable and unexpanded template reads cannot be attributed statically
    and must be recorded as unresolved read events."""
    tree = ast.parse(code)
    visitor = _scan.HfRowVisitor('test_file.py', {})
    visitor.visit(tree)
    assert len(visitor.unresolved) == 1
    assert visitor.unresolved[0][1:3] == (reason, 'read')


def test_frame_subscripts_reported_consistently_with_row_subscripts():
    """Frame subscripts report dynamic column keys and ignore row filters and slices."""
    code_dyn = 'def f(hf, k):\n    return hf[k]\n'
    visitor = _scan.HfRowVisitor('test.py', {})
    visitor.visit(ast.parse(code_dyn))
    assert len(visitor.unresolved) == 1
    assert visitor.unresolved[0][1:3] == ('dynamic key k', 'read')

    code_mask = (
        'def f(hf):\n    _ = hf[hf["Time"] > 0]\n    _ = hf[~hf["flag"]]\n    _ = hf[1:5]\n'
    )
    visitor_mask = _scan.HfRowVisitor('test.py', {})
    visitor_mask.visit(ast.parse(code_mask))
    assert visitor_mask.unresolved == []


def test_template_overrides_are_valid_and_consumed():
    """Each template override names valid species lists and is consumed in the scan."""
    species = _scan._species_lists()

    # 1. Structural validity: 3-tuple keys, valid species domains
    for key, domains in _scan.TEMPLATE_OVERRIDES.items():
        assert len(key) == 3, f'Expected 3-tuple key, got {key}'
        rel, func, pattern = key
        assert pattern.count('<?>') == 1, f'Invalid pattern {pattern}'
        for d in domains:
            assert d in species, f'Domain {d} not in species lists for {key}'

    # 2. Consumption: every single declared override must be matched and consumed
    class TrackingDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.used = {k: 0 for k in self}

        def get(self, key, default=None):
            if key in self.used:
                self.used[key] += 1
            return super().get(key, default)

    t_dict = TrackingDict(_scan.TEMPLATE_OVERRIDES)
    orig_overrides = _scan.TEMPLATE_OVERRIDES
    try:
        _scan.TEMPLATE_OVERRIDES = t_dict
        _scan.scan_tree()
        unconsumed = [k for k, count in t_dict.used.items() if count == 0]
        assert unconsumed == [], f'Unconsumed TEMPLATE_OVERRIDES entries: {unconsumed}'
    finally:
        _scan.TEMPLATE_OVERRIDES = orig_overrides
