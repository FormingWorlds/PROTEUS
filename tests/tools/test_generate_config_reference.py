"""Unit tests for ``tools/generate_config_reference.py`` and ``tools/_config_schema.py``.

The modules under test extract the attrs Config schema (types, defaults,
choices, bounds, three description channels, cross-field constraints) and
render it into the reference pages plus a JSON sidecar. These tests exercise:

* the converter probe that maps the TOML string 'none' to Python None,
* recovery of fields with no source annotation (``planet.R_int_override``),
* enum and bound extraction across the validator shapes the schema uses,
* the missing-docstring error contract for cross-field validators,
* the two-way ``all_options.toml`` completeness check in both directions,
* PAGE_MAP coverage of every section the walk produces.

See ``docs/How-to/testing.md`` and ``docs/Explanations/test_framework.md``
for the test framework.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    """Load a tools/ script by path, resolving sibling imports via sys.modules.

    The ``tools/`` directory is not a package; registering each module under
    its plain name lets ``import _docgen`` inside the scripts resolve without
    putting the repo root on ``sys.path`` (which would shadow installed
    ecosystem packages).
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Registered for its sys.modules side effect only; the scripts import it by name.
_load('_docgen')
_cs = _load('_config_schema')
_gcr = _load('generate_config_reference')


@pytest.fixture(scope='module')
def schema():
    """Build the schema once per module; the walk is pure and read-only."""
    return _cs.build_schema()


def test_accepts_none_probe_discriminates_converters(schema):
    """The probe is true exactly for none_if_none-style fields: rot_period
    maps 'none' to None; zero_if_none (escape.dummy.rate) maps it to 0.0 and
    must NOT be reported as accepting none; plain floats have no converter."""
    by_path = {f['path']: f for f in schema['fields']}
    assert by_path['star.mors.rot_period']['accepts_none'] is True
    assert by_path['star.mors.rot_period']['type'] == 'float or none'
    # zero_if_none: 'none' means 0.0, not None; the display type stays bare.
    assert by_path['escape.dummy.rate']['accepts_none'] is False
    assert by_path['escape.dummy.rate']['type'] == 'float'
    assert by_path['planet.mass_tot']['accepts_none'] is False


def test_unannotated_fields_survive_with_declared_types(schema):
    """Fields with no source annotation (edge case: attrs reports type=None)
    must appear with the display type declared in ANNOTATION_OVERRIDES, not
    be silently dropped from the schema."""
    by_path = {f['path']: f for f in schema['fields']}
    for path in _cs.ANNOTATION_OVERRIDES:
        assert path in by_path, f'{path} missing from schema'
    assert by_path['planet.R_int_override']['type'] == 'float or none'
    assert by_path['planet.R_int_override']['bounds'] == [{'op': '>', 'value': 0}]
    # The override list must not rot: every entry still lacks an annotation.
    assert len(_cs.ANNOTATION_OVERRIDES) == 7


def test_enum_and_bound_extraction_pins_real_validator_sets(schema):
    """Choices come from in_() validators (tuple, list, and single shapes all
    occur in the schema) and bounds from ge/gt/le/lt, pinned against known
    source values rather than derived from the extraction itself."""
    by_path = {f['path']: f for f in schema['fields']}
    assert by_path['star.mors.tracks']['choices'] == ['spada', 'baraffe']
    assert set(by_path['escape.module']['choices']) == {None, 'dummy', 'zephyrus', 'boreas'}
    # fO2_source passes its validators as a LIST (wrapped by attrs into and_):
    # the in_ options must still be recovered through the wrapper.
    assert by_path['planet.fO2_source']['choices'] == [
        'user_constant',
        'from_O_budget',
        'from_mantle_redox',
    ]
    zeph = by_path['escape.zephyrus.efficiency']
    assert zeph['bounds'] == [{'op': '>=', 'value': 0}, {'op': '<=', 'value': 1}]


def test_cross_field_validators_all_documented(schema):
    """Every constraint carries a non-empty doc summary and at least one
    attachment path; the known inert hooks are excluded."""
    names = {c['validator'] for c in schema['constraints']}
    for skipped in _cs.SKIP_VALIDATORS:
        assert skipped not in names
    assert 'spada_zephyrus' in names
    for c in schema['constraints']:
        assert c['doc'].strip(), f'{c["validator"]} has an empty doc summary'
        assert c['attached_to'], f'{c["validator"]} is attached nowhere'


def test_missing_validator_docstring_is_schema_error():
    """Error contract: a cross-field validator without a docstring raises
    SchemaError naming the offender, so an undocumented rule cannot silently
    vanish from the generated constraint list."""
    validator_src = {
        ('_fake', 'undocumented_rule'): {'doc': None, 'cross': True, 'touches': ['a.b']},
    }
    with pytest.raises(_cs.SchemaError, match='undocumented_rule'):
        _cs._build_constraints(validator_src, {'undocumented_rule': ['a.b']})
    # A local (non-cross) undocumented validator is not an error: it guards a
    # single field and is not rendered into the constraints list.
    ok = _cs._build_constraints(
        {('_fake', 'local_rule'): {'doc': None, 'cross': False, 'touches': []}},
        {'local_rule': ['a.b']},
    )
    assert ok == []


def test_attributes_block_parser_handles_multiline_and_sections():
    """The numpydoc parser joins wrapped descriptions, tolerates both
    'name: type' and 'name : type', and stops at the next underlined
    section instead of swallowing it (edge case: trailing Notes block)."""
    doc = (
        'Class summary.\n\n'
        'Attributes\n----------\n'
        'alpha: float\n    First line\n    second line.\n'
        'beta : str\n    Beta doc.\n\n'
        'Notes\n-----\nNot an attribute.\n'
    )
    parsed = _cs.parse_attributes_block(doc)
    assert parsed['alpha'] == 'First line second line.'
    assert parsed['beta'] == 'Beta doc.'
    assert 'Notes' not in parsed and len(parsed) == 2
    # Error-contract adjacent edges: no docstring, and no Attributes block.
    assert _cs.parse_attributes_block(None) == {}
    assert _cs.parse_attributes_block('Just a summary.') == {}


def test_page_map_covers_every_section(schema):
    """Every TOML section the walk produces resolves to exactly one page;
    an unmapped section would make build_targets raise instead of silently
    dropping its table."""
    sections = {f['toml_section'] for f in schema['fields']}
    for section in sections:
        assert section.split('.')[0] in _gcr.PAGE_MAP, f'[{section}] unmapped'
    # The map points only at the seven real pages.
    pages = set(_gcr.PAGE_MAP.values())
    assert len(pages) == 7
    for page in pages:
        assert (_REPO_ROOT / 'docs' / 'Reference' / 'config' / page).is_file()


def test_check_all_options_catches_drift_both_directions(schema, monkeypatch, tmp_path):
    """Error contract: an unknown key in all_options.toml and a schema field
    absent from it are each reported; a fully matching file yields none."""
    lines = []
    tables: dict[str, list[str]] = {}
    for f in schema['fields']:
        tables.setdefault(f['toml_section'], []).append(f['path'].rsplit('.', 1)[-1])
    for section in sorted(tables, key=lambda s: (s != '', s)):
        if section:
            lines.append(f'[{section}]')
        lines.extend(f'{name} = 1' for name in tables[section])
    good = tmp_path / 'all_options.toml'
    good.write_text('\n'.join(lines) + '\n')
    monkeypatch.setattr(_gcr, 'ALL_OPTIONS', good)
    assert _gcr.check_all_options(schema) == []

    bad = tmp_path / 'drifted.toml'
    bad.write_text('\n'.join(lines) + '\nrogue_key = 1\n')
    monkeypatch.setattr(_gcr, 'ALL_OPTIONS', bad)
    problems = _gcr.check_all_options(schema)
    assert any('rogue_key' in p and 'does not exist' in p for p in problems)

    # Missing direction: drop the first line (a root-level schema field).
    partial = tmp_path / 'partial.toml'
    partial.write_text('\n'.join(lines[1:]) + '\n')
    monkeypatch.setattr(_gcr, 'ALL_OPTIONS', partial)
    problems = _gcr.check_all_options(schema)
    assert any('missing from input/all_options.toml' in p for p in problems)


def test_section_table_and_choices_dedup(schema):
    """The rendered table carries one row per field, folds bounds into the
    description, and does not repeat a hand-written trailing 'Choices:'
    sentence next to the validator-derived one."""
    fields = [f for f in schema['fields'] if f['toml_section'] == 'escape']
    table = _gcr._section_table(fields)
    assert table.count('\n| `') == len(fields)
    module_row = next(line for line in table.split('\n') if line.startswith('| `module`'))
    assert module_row.count('Choices:') == 1
    assert '`"zephyrus"`' in module_row  # authoritative validator-derived list
    rate_row = next(
        line for line in table.split('\n') if line.startswith('| `hill_clamp_frac`')
    )
    assert 'Must be > 0.' in rate_row


# Independently written, but CommonMark-faithful on backtick runs: a span
# delimited by N backticks may hold shorter runs, so the content is .+?, not
# [^`]*?, or a ``code with ` inside`` span would desynchronise the sweep.
_CODE_OR_LINK = re.compile(r'(?s:(`+).+?\1)|\[[^\]]*\]\([^)]*\)|\[[^\]]*\]\[[^\]]*\]')


def test_generated_blocks_leave_no_bare_square_bracket(schema):
    """Units keep their brackets but escaped, so markdown cannot read them as a
    shortcut reference, while code spans and real links stay verbatim."""
    p_top = next(f for f in schema['fields'] if f['path'] == 'atmos_clim.p_top')
    assert '\\[bar\\]' in _gcr._describe(p_top)
    linked = next(f for f in schema['fields'] if '](' in f['description'])
    assert '](' in _gcr._describe(linked)
    code_span = next(f for f in schema['fields'] if '``[' in f['description'])
    assert '``[' in _gcr._describe(code_span)  # a backslash would render literally

    sections = sorted({f['toml_section'] for f in schema['fields']})
    blocks = [
        _gcr._section_table([f for f in schema['fields'] if f['toml_section'] == s])
        for s in sections
    ]
    blocks.append(_gcr._constraints_block(schema['constraints']))
    offenders = []
    for block in blocks:
        for line in _CODE_OR_LINK.sub('', block).split('\n'):
            if re.search(r'(?<!\\)[\[\]]', line):
                offenders.append(line)
    assert offenders == []
    # The sweep must have something to check, or it passes vacuously.
    assert sum(block.count('\\[') for block in blocks) > 100


def test_describe_escapes_pipes_region_aware():
    """The cell renderer must escape a free-text pipe (table syntax) while a
    pipe inside a code span survives; a blind replace would backslash both."""
    field = {
        'description': 'Selector `H2|CO2` keyed on |X|.',
        'choices': [],
        'bounds': [{'op': '>', 'value': 0}],
    }
    cell = _gcr._describe(field)
    assert '`H2|CO2`' in cell  # a backslash inside the span renders literally
    assert '\\|X\\|' in cell
    assert 'Must be > 0.' in cell  # bounds still folded into the same cell


def test_json_sidecar_descriptions_stay_raw(schema):
    """The sidecar carries unrendered text: markdown escaping belongs to the
    page render, so no description or constraint doc may hold an escaped
    bracket or pipe, while the bracketed units are still present and bare."""
    for f in schema['fields']:
        assert '\\[' not in f['description'] and '\\]' not in f['description']
        assert '\\|' not in f['description']
    for c in schema['constraints']:
        assert '\\[' not in c['doc'] and '\\]' not in c['doc']
    # Non-vacuity: the raw text really does carry bare bracketed units.
    assert any('[bar]' in f['description'] for f in schema['fields'])
    assert any('[' in c['doc'] for c in schema['constraints'])


def test_every_real_link_survives_rendering(schema):
    """Every inline link written in a source docstring must reach the page
    verbatim; over-escaping would silently turn a working link into text."""
    link = re.compile(r'\[[^\]]*\]\([^()]*(?:\([^()]*\)[^()]*)*\)')
    found = 0
    for f in schema['fields']:
        for m in link.finditer(f['description']):
            found += 1
            assert m.group(0) in _gcr._describe(f), f'{f["path"]} lost {m.group(0)}'
    assert found >= 1  # the sweep is not vacuous: the schema has real links


def test_committed_pages_are_current_and_fully_described(schema):
    """The committed pages and JSON byte-match a fresh render, every field
    carries a description, and all_options.toml agrees with the schema; any
    of these failing means a source edit landed without regeneration."""
    targets = _gcr.build_targets(schema)
    assert len(targets) == 8  # seven pages plus the JSON sidecar
    for path, content in targets:
        assert path.read_text() == content, f'{path.name} is stale'
    assert _gcr.check_all_options(schema) == []
    undocumented = [f['path'] for f in schema['fields'] if f['doc_source'] == 'missing']
    assert undocumented == []
