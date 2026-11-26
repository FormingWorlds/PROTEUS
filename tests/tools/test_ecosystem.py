"""Unit tests for the ecosystem graph and notification drafts in ``tools/ecosystem/``.

The drafts tag people in other repositories, so these tests pin the parts that decide
whether a change reaches a component and at which level: the call-site extraction, the
version and pin parsing, the diff matchers and the rules. No test uses the network;
GitHub answers come from a fake.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'tools' / 'ecosystem'))

import graph  # noqa: E402
import notify  # noqa: E402
import people_kb  # noqa: E402

HOME = 'FormingWorlds/PROTEUS'
SETTINGS = {'bots': ['dependabot'], 'tag_limit': 2}

PYPROJECT = """\
[project]
dependencies = [
    "attrs",
    # 26.11.14: zalmoxis.solver.main takes adiabat_seed (#133).
    "fwl-zalmoxis>=26.11.14",
]
# fwl-zalmoxis is a single-source PyPI package.

[tool.proteus.modules.agni]
# Radiative-convective atmosphere module.
url = "https://github.com/nichollsh/AGNI.git"
ref = "f4a16a8691a662dd2dd6011ce11218448ff05ce3"

[tool.proteus.modules.petsc]
url = "https://zenodo.org/records/15805756/files/petsc.zip?download=1"
"""


class FakeGitHub:
    """Answers ``get_json`` from a dict of REST paths (URL without the API root); an unknown path is a 404."""

    def __init__(self, answers: dict):
        self.answers = answers
        self.asked = []

    def get_json(self, path: str):
        path = path.removeprefix(graph.API + '/')
        self.asked.append(path)
        return self.answers.get(path)


def node(dist=None, julia=None, requires=()):
    return {
        'default_branch': 'main',
        'dist': dist,
        'julia': julia,
        'authors': [],
        'requires': list(requires),
    }


def test_code_references_finds_imports_uses_and_julia_calls():
    """Imports, later uses of the bound names and Julia calls all become call sites."""
    text = (
        'import zalmoxis.eos_export\n'
        'from aragog.core import build as b\n'
        'def f():\n'
        '    return b(1) + zalmoxis.eos_export.x\n'
        'jl.AGNI.atmosphere.setup_b(atmos)\n'
    )
    refs = set(graph.code_references(text))
    assert (2, 'aragog.core:build') in refs
    assert (4, 'aragog.core:build') in refs, 'the use through the alias b is a call site'
    assert (4, 'zalmoxis.eos_export') in refs
    assert (5, 'AGNI.atmosphere:setup!') in refs, 'juliacall writes setup! as setup_b'


def test_code_references_keeps_julia_calls_of_a_file_python_cannot_parse():
    """A syntax error drops the Python references but not the Julia ones."""
    refs = list(
        graph.code_references('def broken(:\n    jl.Obliqua.interior.get_drained_bulk(x)\n')
    )
    assert refs == [(2, 'Obliqua.interior:get_drained_bulk')]
    assert all(':' in key for _, key in refs)


def test_version_compares_calendar_versions_as_numbers():
    """26.11.9 is older than 26.11.15; a string comparison says the opposite."""
    assert graph.version('26.11.9') < graph.version('v26.11.15')
    assert '26.11.9' > '26.11.15', 'guard: the wrong comparison gives the wrong order'
    assert graph.version('26.10.05') == (26, 10, 5)
    assert graph.version('1.0.2rc1') is None
    assert graph.version('') is None


def test_module_pins_reads_github_urls_and_skips_other_hosts():
    """A ``.git`` suffix is dropped and a Zenodo archive is not a component."""
    pins = graph.module_pins(graph.tomllib.loads(PYPROJECT))
    assert pins == {'agni': ('nichollsh/AGNI', 'f4a16a8691a662dd2dd6011ce11218448ff05ce3')}
    assert 'petsc' not in pins


def test_pin_lines_finds_the_requirement_and_its_comment_not_a_later_mention():
    """The dependency line wins over a comment that names the same package."""
    first, start, end = graph.pin_lines(
        PYPROJECT, {'kind': 'runtime', 'package': 'fwl-zalmoxis'}
    )
    lines = PYPROJECT.splitlines()
    assert lines[start - 1].strip() == '"fwl-zalmoxis>=26.11.14",'
    assert (first, start, end) == (4, 5, 5)
    first, start, end = graph.pin_lines(PYPROJECT, {'kind': 'git pin', 'package': 'agni'})
    assert lines[end - 1].startswith('ref = ')
    assert lines[first - 1] == '[tool.proteus.modules.agni]'


def test_pin_lines_returns_none_for_an_unknown_pin():
    """A pin that the file does not hold gives no line range."""
    assert graph.pin_lines(PYPROJECT, {'kind': 'runtime', 'package': 'fwl-mors'}) is None
    assert graph.pin_lines(PYPROJECT, {'kind': 'git pin', 'package': 'spider'}) is None


def test_index_maps_distribution_and_julia_names_but_not_a_stripped_prefix():
    """``fwl-io`` imports as ``fwl_io``; the standard-library ``io`` is not fwl-io."""
    nodes = {
        'FormingWorlds/fwl-io': node(dist='fwl-io'),
        'FormingWorlds/JANUS': node(
            dist='fwl-janus', requires=[['runtime', 'fwl_io>=26.10.7']]
        ),
        'FormingWorlds/Erebus.jl': node(julia='Erebus', requires=[['julia', 'AGNI 1.13']]),
        'nichollsh/AGNI': node(julia='AGNI'),
    }
    index = graph.build_index(nodes)
    assert index['name']['fwl_io'] == 'FormingWorlds/fwl-io'
    assert 'io' not in index['name']
    assert index['name']['janus'] == 'FormingWorlds/JANUS'
    edges = {}
    graph.manifest_edges(nodes, index, edges)
    assert set(edges) == {
        ('FormingWorlds/JANUS', 'FormingWorlds/fwl-io'),
        ('FormingWorlds/Erebus.jl', 'nichollsh/AGNI'),
    }
    assert edges[('FormingWorlds/Erebus.jl', 'nichollsh/AGNI')]['how'][0]['spec'] == '1.13'


def test_definition_matches_a_removed_definition_line_only():
    """A call or a longer name with the same prefix is not a definition."""
    assert notify.definition('main').match('-def main(config, adiabat_seed=None):')
    assert notify.definition('setup!').match('-function setup!(atmos::Atmos_t)')
    assert not notify.definition('main').match('-    result = main(config)')
    assert not notify.definition('setup').match('-function setup!(atmos)')


def test_used_ignores_mentions_in_text():
    """A call, an attribute or an import counts; the name inside a message does not."""
    assert notify.used('AGNI').search('    jl.AGNI.atmosphere.setup_b(atmos)')
    assert notify.used('vulcan').search('+    from vulcan.paths import THERMO_DIR')
    assert not notify.used('AGNI').search("+    f'AGNI writes runtime.sf in {io_dir}'")
    assert not notify.used('main').search('+# the secant tail of a ramp that main starts')


def test_pin_package_reads_every_fact_key_form():
    """Runtime, extra and module-table keys all give the package name."""
    assert notify.pin_package('dependencies:fwl-io') == 'fwl-io'
    assert notify.pin_package('extra vulcan:fwl-vulcan') == 'fwl-vulcan'
    assert notify.pin_package('tool.proteus.modules.agni.ref') == 'agni'
    assert notify.pin_package('tool.proteus') == 'tool.proteus'


def window(gh, prs=(), releases=(), edges=(), nodes=None, kb=None, facts=()):
    data = {
        'window': {'since': '2026-11-12T00:00:00Z', 'until': '2026-11-19T00:00:00Z'},
        'prs': list(prs),
        'releases': list(releases),
        'facts': list(facts),
    }
    g = {'home': HOME, 'edges': list(edges), 'nodes': nodes or {}}
    return notify.Window(gh, data, g, kb or {}, SETTINGS, _ROOT)


def edge(dst, how=(), symbols=None, decisions=()):
    return {
        'from': HOME,
        'to': dst,
        'how': list(how),
        'symbols': symbols or {},
        'dispatch': [],
        'config': [],
        'decisions': list(decisions),
    }


def pr(repo, number, files, author='someone'):
    return {
        'ref': f'{repo.split("/")[1]}#{number}',
        'repo': repo,
        'number': number,
        'status': 'merged',
        'author': author,
        'author_kind': 'User',
        'files': files,
    }


def test_releases_flag_a_release_above_the_floor_and_only_inform_for_a_commit_pin():
    """A release newer than the floor asks for a check; one below it says nothing."""
    zal = edge(
        'FormingWorlds/Zalmoxis',
        [{'kind': 'runtime', 'package': 'fwl-zalmoxis', 'spec': '>=26.11.9'}],
    )
    agni = edge(
        'nichollsh/AGNI',
        [{'kind': 'git pin', 'package': 'agni', 'spec': 'f4a16a8691a6', 'behind': 3}],
    )
    rels = [
        {'repo': 'FormingWorlds/Zalmoxis', 'tag': '26.11.15', 'author': 'x', 'url': 'u1'},
        {'repo': 'FormingWorlds/Zalmoxis', 'tag': '26.11.8', 'author': 'x', 'url': 'u2'},
        {'repo': 'nichollsh/AGNI', 'tag': '1.13.3', 'author': 'x', 'url': 'u3'},
    ]
    w = window(FakeGitHub({}), releases=rels, edges=[zal, agni])
    notify.releases(w)
    levels = sorted((i['source'], i['level']) for i in w.items)
    assert levels == [
        ('[FormingWorlds/Zalmoxis 26.11.15](u1)', 'check'),
        ('[nichollsh/AGNI 1.13.3](u3)', 'fyi'),
    ]
    assert '3 commits behind' in next(i['text'] for i in w.items if i['level'] == 'fyi')


def upstream_window(patch, home_body=''):
    up = pr('FormingWorlds/Zalmoxis', 133, ['src/zalmoxis/solver.py'], author='tim')
    home = pr(HOME, 1088, ['src/proteus/interior_struct/zalmoxis.py'], author='tim')
    gh = FakeGitHub(
        {
            'repos/FormingWorlds/Zalmoxis/pulls/133/files?per_page=100&page=1': [
                {'filename': 'src/zalmoxis/solver.py', 'patch': patch}
            ],
            'repos/FormingWorlds/PROTEUS/pulls/1088/files?per_page=100&page=1': [
                {
                    'filename': 'src/proteus/interior_struct/zalmoxis.py',
                    'patch': '@@ -1 +1 @@ def other(\n+x = 1',
                }
            ],
            'repos/FormingWorlds/PROTEUS/pulls/1088': {'body': home_body},
        }
    )
    sites = {'zalmoxis.solver:main': ['src/proteus/interior_struct/zalmoxis.py:20']}
    nodes = {'FormingWorlds/Zalmoxis': node(dist='fwl-zalmoxis')}
    return window(
        gh, prs=[up, home], edges=[edge('FormingWorlds/Zalmoxis', symbols=sites)], nodes=nodes
    )


def test_upstream_code_levels_follow_the_kind_of_change():
    """A removed definition line is an action; a new definition only informs."""
    w = upstream_window('@@ -1,2 +1,2 @@\n-def main(config):\n+def main(config, seed=None):')
    notify.upstream_code(w)
    assert [i['level'] for i in w.items] == ['action']
    w = upstream_window('@@ -1 +1,2 @@\n+def main(config):\n+    pass')
    notify.upstream_code(w)
    assert [i['level'] for i in w.items] == ['fyi']
    assert w.items[0]['text'].startswith('adds `zalmoxis.solver.main`')


def test_upstream_code_lowers_the_level_when_a_proteus_pr_links_the_change():
    """A PROTEUS pull request that links the upstream one turns an action into a check."""
    patch = '@@ -1,2 +1,2 @@\n-def main(config):\n+def main(config, seed=None):'
    w = upstream_window(patch, home_body='Follows FormingWorlds/Zalmoxis#133.')
    notify.upstream_code(w)
    assert [i['level'] for i in w.items] == ['check']
    assert 'FormingWorlds/PROTEUS#1088' in w.items[0]['text']
    w = upstream_window(patch, home_body='Follows FormingWorlds/Zalmoxis#1330.')
    notify.upstream_code(w)
    assert [i['level'] for i in w.items] == ['action'], 'guard: #1330 is not #133'


def test_upstream_code_skips_a_file_change_that_does_not_name_the_symbol():
    """A changed file whose diff never names the called function gives no item."""
    w = upstream_window('@@ -40,2 +40,2 @@ def helper(\n-    a = 1\n+    a = 2')
    notify.upstream_code(w)
    assert w.items == []
    assert any('pulls/133/files' in p for p in w.gh.asked)


def test_render_puts_fyi_only_targets_in_the_digest_and_never_tags_them():
    """A component with only FYI items gets no draft issue and no tag."""
    w = window(FakeGitHub({}))
    w.add('FormingWorlds/aragog', 'fyi', 'FormingWorlds/PROTEUS#1087', 'sets the pin.', ['tim'])
    w.add(HOME, 'check', 'FormingWorlds/aragog@26.11.17', 'is released.', ['tim'])
    assert w.items[0]['people'] == [], 'an FYI item keeps no people to tag'
    text = notify.render(w, {'window': {'since': '2026-11-12', 'until': '2026-11-19'}})
    assert '## Draft issue in FormingWorlds/PROTEUS' in text
    assert '## Draft issue in FormingWorlds/aragog' not in text
    assert '- aragog: FormingWorlds/PROTEUS#1087 sets the pin.' in text
    assert text.count('@tim') == 1


def test_interfaces_lists_exact_proteus_files_per_kind_and_skips_other_edges():
    """Each PROTEUS edge gives one entry per kind with exact paths; a module edge gives none."""
    g = {
        'home': HOME,
        'edges': [
            edge(
                'nichollsh/AGNI',
                [
                    {'kind': 'git pin', 'package': 'agni', 'spec': 'f4a1'},
                    {'kind': 'script', 'file': 'tools/get_agni.sh'},
                ],
                symbols={
                    'AGNI.atmosphere:setup!': [
                        'src/proteus/atmos_clim/agni.py:400',
                        'src/proteus/atmos_clim/agni.py:410',
                    ]
                },
            ),
            {
                **edge(
                    'FormingWorlds/fwl-io',
                    [{'kind': 'runtime', 'package': 'fwl-io', 'spec': '>=1'}],
                ),
                'from': 'FormingWorlds/JANUS',
            },
        ],
    }
    g['edges'][0]['dispatch'] = ['src/proteus/atmos_clim/agni.py:run_agni']
    out = graph.interfaces(g, 'abc123')
    assert [(i['kind'], i['interface_files']) for i in out] == [
        ('import', ['src/proteus/atmos_clim/agni.py']),
        ('pin', ['pyproject.toml']),
        ('script', ['tools/get_agni.sh']),
        ('wrapper', ['src/proteus/atmos_clim/agni.py']),
    ]
    assert {i['built_from'] for i in out} == {'abc123'}
    assert all(i['from'] == HOME for i in out), 'the JANUS edge is not a PROTEUS interface'


def test_function_spans_attributes_a_method_and_its_decorator():
    """A decorated method spans from its decorator; a class body line is in no function."""
    text = 'class A:\n    x = 1\n\n    @staticmethod\n    def f():\n        return 2\n'
    spans = people_kb.function_spans(text)
    assert spans == [(4, 6, 'A.f')]
    assert not any(s[0] <= 2 <= s[1] for s in spans)


def test_module_interfaces_maps_names_to_module_files_and_lists_misses():
    """Python names map to their module file, a Julia call to <submodule>.jl in any folder, a miss stays unresolved."""
    tree = {
        'tree': [
            {'path': 'src/aragog/solver/__init__.py', 'type': 'blob'},
            {'path': 'src/aragog/core.py', 'type': 'blob'},
            {'path': 'src/aragog', 'type': 'tree'},
        ]
    }
    gh = FakeGitHub(
        {
            'repos/FormingWorlds/aragog/commits/main': {'sha': 'a1'},
            'repos/FormingWorlds/aragog/git/trees/a1?recursive=1': tree,
            'repos/nichollsh/AGNI/commits/main': {'sha': 'b2'},
            'repos/nichollsh/AGNI/git/trees/b2?recursive=1': {
                'tree': [
                    {'path': 'src/state/atmosphere.jl', 'type': 'blob'},
                    {'path': 'agni.jl', 'type': 'blob'},
                ]
            },
        }
    )
    g = {
        'home': HOME,
        'nodes': {
            'FormingWorlds/aragog': node(dist='fwl-aragog'),
            'nichollsh/AGNI': node(julia='AGNI'),
        },
        'edges': [
            edge(
                'FormingWorlds/aragog',
                symbols={
                    'aragog.solver:EntropySolver': ['a.py:1'],
                    'aragog.core:f': ['a.py:2'],
                    'aragog.mesh:g': ['a.py:3'],
                },
            ),
            edge('nichollsh/AGNI', symbols={'AGNI.atmosphere:setup!': ['b.py:1']}),
        ],
    }
    out = {(i['to'], i['kind']): i for i in graph.module_interfaces(gh, g)}
    aragog = out[('FormingWorlds/aragog', 'import')]
    assert aragog['module_files'] == ['src/aragog/core.py', 'src/aragog/solver/__init__.py']
    assert aragog['unresolved'] == ['aragog.mesh:g']
    assert aragog['built_from'] == 'a1'
    assert out[('nichollsh/AGNI', 'call')]['module_files'] == ['src/state/atmosphere.jl']


def test_symbols_leave_version_constants_to_the_release_rule():
    """A version constant changes with every release, so it is no call-site signal."""
    e = edge(
        'nichollsh/AGNI',
        symbols={
            'AGNI.consts:AGNI_VERSION': ['a.py:1'],
            'aragog:__version__': ['a.py:2'],
            'AGNI.solver:solve_energy!': ['a.py:3'],
            'AGNI': ['a.py:4'],
        },
    )
    found = {(module, name) for module, name, _, _ in notify.symbols(e, 'AGNI')}
    assert found == {('AGNI.solver', 'solve_energy!'), ('AGNI', None)}
    assert all('VERSION' not in (name or '').upper() for _, name in found)
