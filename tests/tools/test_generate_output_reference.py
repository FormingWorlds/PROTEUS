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
    by_name = {k['name']: k for k in matrix['keys']}
    assert by_name['H2O_vmr_xuv']['consumers'] == ['escape']
    assert by_name['H2O_vmr_xuv']['consumers_possible'] == ['escape']
    assert 'outgas' in by_name['H_kg_solid']['consumers']
    assert by_name['T_obs']['consumers'] == ['atmos_clim', 'escape']
    for event in matrix['unresolved_events']:
        assert 'line' not in event
        assert event['kind'] == 'read'
    for f, line, reason_text, kind, _func in _scan.scan_tree()['unresolved']:
        assert kind == 'read'
        path = _gor.REPO_ROOT / f'src/proteus/{f}'
        assert path.is_file(), f'{f} does not exist'
        source = path.read_text()
        lines = source.splitlines()
        assert 1 <= line <= len(lines), f'Line {line} out of range in {f}'
        tree = ast.parse(source)
        nodes_at_line = [n for n in ast.walk(tree) if getattr(n, 'lineno', None) == line]
        assert len(nodes_at_line) > 0
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


def test_render_computed_keys_sections():
    """Render includes own sections for read and write computed keys when present,
    omits empty sections, merges duplicate sites with a count, and formats
    reasons inside backticks."""
    matrix_empty = {'keys': [], 'unresolved_events': []}
    out_empty = _gor.render(matrix_empty)
    assert 'Reads with computed keys' not in out_empty
    assert 'Writes with computed keys' not in out_empty

    matrix_both = {
        'keys': [],
        'unresolved_events': [
            {
                'file': 'src/proteus/outgas/atmodeller.py',
                'kind': 'read',
                'reason': 'dynamic key key',
                'function': 'calc_surface_pressures_atmodeller',
            },
            {
                'file': 'src/proteus/outgas/atmodeller.py',
                'kind': 'read',
                'reason': 'dynamic key key',
                'function': 'calc_surface_pressures_atmodeller',
            },
            {
                'file': 'src/proteus/escape/wrapper.py',
                'kind': 'write',
                'reason': 'dynamic key f"{e}{key}"',
                'function': 'run_escape',
            },
        ],
    }
    out_both = _gor.render(matrix_both)
    assert '### Reads with computed keys' in out_both
    assert 'accessed column names' in out_both
    assert '### Writes with computed keys' in out_both
    assert 'modified column names' in out_both
    assert (
        '- `src/proteus/outgas/atmodeller.py::calc_surface_pressures_atmodeller` (2 sites): '
        '`dynamic key key` (touches <element>_kg_total)'
    ) in out_both
    assert (
        '- `src/proteus/escape/wrapper.py::run_escape`: `dynamic key f"{e}{key}"`'
    ) in out_both


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
        (
            'k = "a"\ndef g():\n    global k\n    k = dyn()\ndef f(hf_row):\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def outer():\n    k = "a"\n    def g():\n        nonlocal k\n        k = dyn()\n    def f(hf_row):\n        return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row, it):\n    k = "a"\n    _ = [(k := x) for x in it]\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row):\n    k = "a"\n    def k():\n        pass\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row):\n    k = "a"\n    class k:\n        pass\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'def f(hf_row):\n    k = "a"\n    import k.sub\n    return hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'NAMES = ("a", "b")\ndef f(hf_row, NAMES):\n    for k in NAMES:\n        hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'NAMES = ("a", "b")\ndef f(hf_row):\n    NAMES = dyn()\n    for k in NAMES:\n        hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'NAMES = ("a", "b")\nNAMES = ("c",)\ndef f(hf_row):\n    for k in NAMES:\n        hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'NAMES = ["a", "b"]\nNAMES.append("c")\ndef f(hf_row):\n    for k in NAMES:\n        hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'NAMES = ("a",)\nNAMES += ("b",)\ndef f(hf_row):\n    for k in NAMES:\n        hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'NAMES = ("a",)\nNAMES: tuple = ("b",)\ndef f(hf_row):\n    for k in NAMES:\n        hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'NAMES = ("a",)\ndef NAMES():\n    pass\ndef f(hf_row):\n    for k in NAMES:\n        hf_row.get(k)\n',
            'dynamic key k',
        ),
        (
            'NAMES = ("a",)\nimport NAMES\ndef f(hf_row):\n    for k in NAMES:\n        hf_row.get(k)\n',
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
        'global',
        'nonlocal',
        'walrus_in_comp',
        'def_stmt',
        'class_stmt',
        'dotted_import',
        'module_const_shadowed_by_param',
        'module_const_shadowed_by_local',
        'module_const_rebound',
        'module_const_mutated',
        'module_const_aug',
        'module_const_ann',
        'module_const_def',
        'module_const_import',
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

    # BinOp templated keys on frames: expanded inside loop, unresolved outside
    code_loop = 'def f(hf_all):\n    for e in element_list:\n        _ = hf_all[e + "_kg"]\n'
    visitor_loop = _scan.HfRowVisitor('test.py', {'element_list': ['H', 'O', 'C']})
    visitor_loop.visit(ast.parse(code_loop))
    assert visitor_loop.unresolved == []
    assert visitor_loop.reads == [('H_kg', False), ('O_kg', False), ('C_kg', False)]

    code_tmpl = 'def f(hf_all, e):\n    return hf_all[e + "_kg"]\n'
    visitor_tmpl = _scan.HfRowVisitor('test.py', {})
    visitor_tmpl.visit(ast.parse(code_tmpl))
    assert len(visitor_tmpl.unresolved) == 1
    assert visitor_tmpl.unresolved[0][1:3] == ('template <e>_kg', 'read')
    assert visitor_tmpl.reads == []


def test_template_overrides_are_valid_and_consumed():
    """Each template override names valid species lists and is consumed in the scan."""
    species = _scan._species_lists()

    # 1. Structural validity: 3-tuple keys, valid species domains
    for key, override in _scan.TEMPLATE_OVERRIDES.items():
        assert len(key) == 3, f'Expected 3-tuple key, got {key}'
        rel, func, pattern = key
        assert pattern.count('<?>') == 1, f'Invalid pattern {pattern}'
        for d in override.domains:
            assert d in species, f'Domain {d} not in species lists for {key}'

    atmod_solid = _scan.TEMPLATE_OVERRIDES[
        ('outgas/atmodeller.py', 'calc_surface_pressures_atmodeller', '<?>_kg_solid')
    ]
    assert 'vol_element_list' in atmod_solid.domains

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


def test_comprehension_element_transform():
    """Comprehension element transform returns unresolved rather than bare gas names."""
    code_transformed = (
        'X = tuple(g + "_vmr_xuv" for g in gas_list)\n'
        'def f(hf_row):\n'
        '    for k in X:\n'
        '        hf_row.get(k)\n'
    )
    visitor_transformed = _scan.HfRowVisitor('test.py', {'gas_list': ['H2O', 'CO2']})
    visitor_transformed.visit(ast.parse(code_transformed))
    assert visitor_transformed.reads == []
    assert len(visitor_transformed.unresolved) == 1
    assert visitor_transformed.unresolved[0][1:3] == ('dynamic key k', 'read')

    code_direct = (
        'def f(hf_row):\n'
        '    for k in [g + "_vmr_xuv" for g in gas_list]:\n'
        '        hf_row.get(k)\n'
    )
    visitor_direct = _scan.HfRowVisitor('test.py', {'gas_list': ['H2O', 'CO2']})
    visitor_direct.visit(ast.parse(code_direct))
    assert visitor_direct.reads == []
    assert len(visitor_direct.unresolved) == 1
    assert visitor_direct.unresolved[0][1:3] == ('dynamic key k', 'read')

    code_bare = (
        'X = tuple(g for g in gas_list)\n'
        'def f(hf_row):\n'
        '    for k in X:\n'
        '        hf_row.get(k)\n'
    )
    visitor_bare = _scan.HfRowVisitor('test.py', {'gas_list': ['H2O', 'CO2']})
    visitor_bare.visit(ast.parse(code_bare))
    assert visitor_bare.unresolved == []
    assert visitor_bare.reads == [('H2O', False), ('CO2', False)]


def test_loop_domain_shadowing_and_invalidation():
    """Rebinding loop variable clears domain, and inner scopes restore outer domain."""
    code_rebound = (
        'def f(hf_row):\n'
        '    for e in element_list:\n'
        '        e = "dyn"\n'
        '        hf_row.get(e + "_kg")\n'
    )
    visitor_rebound = _scan.HfRowVisitor('test.py', {'element_list': ['H', 'O', 'C']})
    visitor_rebound.visit(ast.parse(code_rebound))
    assert visitor_rebound.reads == []
    assert len(visitor_rebound.unresolved) == 1

    code_nested = (
        'def f(hf_row):\n'
        '    for e in element_list:\n'
        '        _ = [x for e in other]\n'
        '        hf_row.get(e + "_kg")\n'
    )
    visitor_nested = _scan.HfRowVisitor('test.py', {'element_list': ['H', 'O', 'C']})
    visitor_nested.visit(ast.parse(code_nested))
    assert visitor_nested.unresolved == []
    assert visitor_nested.reads == [('H_kg', False), ('O_kg', False), ('C_kg', False)]


def test_frame_store_subscript_not_recorded_as_read():
    """Store subscripts on frames (e.g. hf_all['X'] = 1) are recorded as writes."""
    code = 'def f(hf_all):\n    hf_all["X"] = 1\n'
    visitor = _scan.HfRowVisitor('test.py', {})
    visitor.visit(ast.parse(code))
    assert visitor.reads == []
    assert visitor.writes == [('X', 'f')]
    assert visitor.unresolved == []


def test_augassign_on_subscript_records_both_read_and_write():
    """AugAssign on row or frame subscript records both a read and a write."""
    code_row = 'def f(hf_row):\n    hf_row["X"] += 1\n'
    visitor_row = _scan.HfRowVisitor('test.py', {})
    visitor_row.visit(ast.parse(code_row))
    assert visitor_row.reads == [('X', False)]
    assert visitor_row.writes == [('X', 'f')]
    assert visitor_row.unresolved == []

    code_frame = 'def f(hf_all):\n    hf_all["Y"] += 2\n'
    visitor_frame = _scan.HfRowVisitor('test.py', {})
    visitor_frame.visit(ast.parse(code_frame))
    assert visitor_frame.reads == [('Y', False)]
    assert visitor_frame.writes == [('Y', 'f')]
    assert visitor_frame.unresolved == []


def test_domain_list_shadowed_by_param_or_local():
    """Parameters or locals shadowing domain list names prevent domain expansion."""
    code_param = (
        'def f(hf_row, gas_list):\n    for g in gas_list:\n        hf_row.get(g + "_bar")\n'
    )
    visitor_param = _scan.HfRowVisitor('test.py', {'gas_list': ['H2O', 'CO2']})
    visitor_param.visit(ast.parse(code_param))
    assert visitor_param.reads == []
    assert len(visitor_param.unresolved) == 1
    assert visitor_param.unresolved[0][1:3] == ('template <g>_bar', 'read')

    code_local = (
        'def f(hf_row):\n'
        '    gas_list = dyn()\n'
        '    for g in gas_list:\n'
        '        hf_row.get(g + "_bar")\n'
    )
    visitor_local = _scan.HfRowVisitor('test.py', {'gas_list': ['H2O', 'CO2']})
    visitor_local.visit(ast.parse(code_local))
    assert visitor_local.reads == []
    assert len(visitor_local.unresolved) == 1
    assert visitor_local.unresolved[0][1:3] == ('template <g>_bar', 'read')


def test_loop_variable_rebinding_clears_domain():
    """Rebinding loop variable via with/except/import/match/del clears loop domain."""
    for stmt in [
        'with ctx as e: pass',
        'try: pass\n        except Exception as e: pass',
        'import e',
        'match val:\n            case e: pass',
        'del e',
    ]:
        code = (
            f'def f(hf_row, ctx, val):\n'
            f'    for e in element_list:\n'
            f'        {stmt}\n'
            f'        hf_row.get(e + "_kg")\n'
        )
        visitor = _scan.HfRowVisitor('test.py', {'element_list': ['H', 'O']})
        visitor.visit(ast.parse(code))
        assert visitor.reads == [], f'Failed for {stmt}'
        assert len(visitor.unresolved) == 1, f'Failed for {stmt}'


def test_reads_inside_comprehension_if_clauses_recorded():
    """Reads inside comprehension if clauses are detected and attributed or unresolved."""
    code = 'def f(hf_row):\n    _ = [g for g in gas_list if hf_row.get(g + "_bar")]\n'
    visitor = _scan.HfRowVisitor('test.py', {'gas_list': ['H2O', 'CO2']})
    visitor.visit(ast.parse(code))
    assert visitor.reads == [('H2O_bar', False), ('CO2_bar', False)]
    assert visitor.unresolved == []


def test_global_in_later_function_clears_module_constant():
    """Global declaration in a later function invalidates constant in earlier function."""
    code = (
        'k = "a"\n'
        'def f(hf_row):\n'
        '    return hf_row.get(k)\n'
        'def g():\n'
        '    global k\n'
        '    k = dyn()\n'
    )
    visitor = _scan.HfRowVisitor('test.py', {})
    visitor.visit(ast.parse(code))
    assert visitor.reads == []
    assert len(visitor.unresolved) == 1
    assert visitor.unresolved[0][1:3] == ('dynamic key k', 'read')


def test_module_constant_defined_after_use_resolves():
    """Module-level constant defined after function resolves order-independently."""
    code = 'def f(hf_row):\n    return hf_row.get(k)\nk = "a"\n'
    visitor = _scan.HfRowVisitor('test.py', {})
    visitor.visit(ast.parse(code))
    assert visitor.reads == [('a', False)]
    assert visitor.unresolved == []


def test_scan_error_raised_for_unknown_override_domain(monkeypatch):
    """ScanError is raised when an override names an unknown domain."""
    monkeypatch.setitem(
        _scan.TEMPLATE_OVERRIDES,
        ('test.py', 'f', '<?>_unknown'),
        _scan.TemplateOverride(('nonexistent_domain',)),
    )
    code = 'def f(hf_row, x):\n    return hf_row[f"{x}_unknown"]\n'
    visitor = _scan.HfRowVisitor('test.py', {})
    with pytest.raises(_scan.ScanError, match='unknown domain name "nonexistent_domain"'):
        visitor.visit(ast.parse(code))


def test_key_lookup_from_enclosing_scope():
    """Key lookup resolves constant from enclosing function scope."""
    code = (
        'def outer():\n    k = "T_surf"\n    def inner(hf_row):\n        return hf_row.get(k)\n'
    )
    visitor = _scan.HfRowVisitor('test.py', {})
    visitor.visit(ast.parse(code))
    assert visitor.reads == [('T_surf', False)]
    assert visitor.unresolved == []


def test_scope_restore_in_comprehension():
    """Scope restore in comprehension preserves subsequent bindings and resolution."""
    code_a = (
        'def f(hf_row, it):\n'
        '    k = "a"\n'
        '    _ = [x for x in it]\n'
        '    k = dyn()\n'
        '    return hf_row.get(k)\n'
    )
    visitor_a = _scan.HfRowVisitor('test.py', {})
    visitor_a.visit(ast.parse(code_a))
    assert visitor_a.reads == []
    assert len(visitor_a.unresolved) == 1
    assert visitor_a.unresolved[0][1:3] == ('dynamic key k', 'read')

    code_c = (
        'def f(hf_row, it):\n    _ = [x for x in it]\n    k = "a"\n    return hf_row.get(k)\n'
    )
    visitor_c = _scan.HfRowVisitor('test.py', {})
    visitor_c.visit(ast.parse(code_c))
    assert visitor_c.reads == [('a', False)]
    assert visitor_c.unresolved == []


def test_template_overrides_possible_flag_and_separation():
    """TemplateOverride with possible=True records in consumers_possible."""
    matrix = _gor.build_matrix()
    by_name = {k['name']: k for k in matrix['keys']}
    assert by_name['T_obs']['consumers_possible'] == []

    # Render test: verify (possible) suffix appears only for possible consumers
    rendered = _gor.render(matrix)
    assert '| `H2O_vmr_xuv` | `1` | volume mixing ratio at XUV level |' in rendered
    assert 'escape (possible)' in rendered
