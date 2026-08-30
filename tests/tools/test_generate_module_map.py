"""Unit tests for ``tools/generate_module_map.py``.

The module under test renders the dispatch map (module option, entry point,
role per physics area) and machine-verifies every declared row against the
source: entry symbols must exist, declared options must match the config
validators, and module-name comparisons across the tree must use real
options. These tests exercise:

* the comparison scanner across every syntactic form the wrappers use
  (equality, membership, is None, falsy tests, match statements, aliases),
* the verification pass on the committed tree (zero problems),
* the error contract for a broken declared entry and an option-set mismatch,
* freshness of the KNOWN_DEAD_ARMS allowlist,
* agreement between the rendered page region and the JSON sidecar.

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
_cs = _load('_config_schema')
_gmm = _load('generate_module_map')


@pytest.fixture(scope='module')
def schema():
    """Build the config schema once; the walk is pure and read-only."""
    return _cs.build_schema()


def test_scanner_recovers_every_comparison_form(tmp_path):
    """The scanner sees equality, membership, is-None, falsy, match, and
    local-alias comparisons; an unrelated attribute chain contributes
    nothing (edge case guarding against over-matching)."""
    src = tmp_path / 'wrapper.py'
    src.write_text(
        'def f(config):\n'
        "    if config.escape.module == 'dummy':\n"
        '        pass\n'
        "    if config.escape.module in ('zephyrus', 'boreas'):\n"
        '        pass\n'
        '    if config.escape.module is None:\n'
        '        pass\n'
        '    if not config.escape.module:\n'
        '        pass\n'
        '    module = config.escape.module\n'
        '    match module:\n'
        "        case 'matched':\n"
        '            pass\n'
        "    if config.other.module == 'unrelated':\n"
        '        pass\n'
    )
    found = _gmm.scan_compared_options(src, ('escape', 'module'))
    assert found == {None, 'dummy', 'zephyrus', 'boreas', 'matched'}
    # Error-contract adjacent edge: a file with no comparisons yields nothing.
    empty = tmp_path / 'empty.py'
    empty.write_text('x = 1\n')
    assert _gmm.scan_compared_options(empty, ('escape', 'module')) == set()


def test_verify_passes_on_committed_tree(schema):
    """The declared map, the config validators, and the wrappers agree on the
    committed tree; a non-empty result here means either real source drift or
    a stale declaration in DISPATCH_SITES."""
    problems = _gmm.verify(schema)
    assert problems == []
    # The map covers all ten dispatch sites, one per physics area.
    assert len(_gmm.DISPATCH_SITES) == 10
    assert len({site['config_path'] for site in _gmm.DISPATCH_SITES}) == 10


def test_verify_flags_broken_entry_and_option_drift(schema, monkeypatch):
    """Error contract: a declared symbol that does not exist and a declared
    option set that disagrees with the validator are each reported with the
    config path of the offending site."""
    import copy

    broken = copy.deepcopy(_gmm.DISPATCH_SITES)
    site = next(s for s in broken if s['config_path'] == 'escape.module')
    site['entries']['dummy'] = ('escape/wrapper.py', 'no_such_function', 'x')
    del site['entries']['boreas']
    monkeypatch.setattr(_gmm, 'DISPATCH_SITES', broken)
    problems = _gmm.verify(schema)
    assert any('no_such_function' in p for p in problems)
    assert any('escape.module' in p and 'do not match' in p for p in problems)


def test_known_dead_arms_allowlist_is_current():
    """Every allowlisted dead arm still exists in the source; a stale entry
    would quietly re-open the door it documents."""
    assert len(_gmm.KNOWN_DEAD_ARMS) == 1  # grows only with a documented reason
    for rel_file, dotted, value in _gmm.KNOWN_DEAD_ARMS:
        path = _REPO_ROOT / rel_file
        assert path.is_file(), f'{rel_file} vanished; prune KNOWN_DEAD_ARMS'
        found = _gmm.scan_compared_options(path, tuple(dotted.split('.')))
        assert value in found, f'{rel_file} no longer compares {dotted} to "{value}"'


def test_render_and_json_agree_row_for_row(schema):
    """The markdown table rows and the JSON option entries agree in count and
    content for every site; a renderer that drops an option would diverge."""
    page = _gmm.render()
    data = _gmm.build_json()
    assert len(data['sites']) == len(_gmm.DISPATCH_SITES)
    for site in data['sites']:
        assert f'`{site["config_path"]}`' in page
        for opt in site['options']:
            label = '`none`' if opt['option'] is None else f'`"{opt["option"]}"`'
            assert label in page
            if opt['entry'] is not None:
                assert opt['entry'].removeprefix('src/proteus/') in page
    # Spot-pin real backends so a mis-declared map cannot pass on shape alone.
    interior = next(
        s for s in data['sites'] if s['config_path'] == 'interior_energetics.module'
    )
    entries = {o['option']: o['entry'] for o in interior['options']}
    assert entries['spider'] == 'src/proteus/interior_energetics/spider.py:RunSPIDER'
    assert entries['aragog'] == 'src/proteus/interior_energetics/aragog.py:AragogRunner'


def test_committed_page_and_json_are_current(schema):
    """The committed module_map.md region and JSON byte-match a fresh render;
    a wrapper or validator edit cannot land without regeneration."""
    fresh_page = _docgen.normalize(
        _docgen.replace_between_markers(
            _gmm.PAGE.read_text(), 'GENERATED: module-map', _gmm.render()
        )
    )
    assert _gmm.PAGE.read_text() == fresh_page
    assert _gmm.JSON_PATH.read_text() == _docgen.dump_json(_gmm.build_json())
    # Every module-selection field in the schema is mapped or exempt; the
    # verify pass enforces it, so the committed tree must satisfy it too.
    assert _gmm.verify(schema) == []
