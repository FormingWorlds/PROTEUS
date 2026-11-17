"""Unit tests for the weekly ecosystem digest scripts in ``tools/digest/``.

The digest posts text that a model drafts from pull-request text, which is untrusted.
These tests pin the parts that keep a wrong or injected entry out of a post: the
cleaning of collected text, the mechanical facts, the fail-closed checks, the
mention rules, the posting decision and the settings validation. No test uses
the network; GitHub answers are recorded fixtures or fakes.
"""

from __future__ import annotations

import json
import re
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

pytest.importorskip('hypothesis')
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]

_ROOT = Path(__file__).resolve().parents[2]
_DATA = _ROOT / 'tests' / 'data' / 'digest'
sys.path.insert(0, str(_ROOT / 'tools' / 'digest'))

import check  # noqa: E402
import collect  # noqa: E402
import common  # noqa: E402
import draft  # noqa: E402
import facts  # noqa: E402
import post  # noqa: E402
import render  # noqa: E402

SINCE, UNTIL = '2026-09-01T00:00:00Z', '2026-09-30T00:00:00Z'


def _pr(ref='PROTEUS#849', status='merged', author='alice', kind='User', **extra):
    repo = 'FormingWorlds/' + ref.split('#')[0]
    return {
        'ref': ref,
        'repo': repo,
        'number': int(ref.split('#')[1]),
        'url': f'https://github.com/{repo}/pull/{ref.split("#")[1]}',
        'status': status,
        'title': 'Lower the default interior grain size to 1e-3 m',
        'body': 'Settling velocity scales with grain radius squared; 0.1 m was too large.',
        'author': author,
        'author_kind': kind,
        'created_at': '2026-09-05T10:00:00Z',
        'merged_at': '2026-09-06T21:07:16Z' if status == 'merged' else None,
        'size': 10,
        'labels': [],
        'files': [],
        'merge_oid': 'm1',
        'parent_oid': 'p1',
        **extra,
    }


def _release(ref='AGNI@1.13.0', author='bob'):
    return {
        'ref': ref,
        'repo': 'nichollsh/AGNI',
        'tag': ref.split('@')[1],
        'url': 'https://github.com/nichollsh/AGNI/releases/tag/1.13.0',
        'name': 'AGNI v1.13.0',
        'prerelease': False,
        'body': 'Handles unbound atmospheres.',
        'author': author,
        'author_kind': 'User',
        'published_at': '2026-09-29T09:19:58Z',
    }


def _fact(
    fid='F1', pr='PROTEUS#849', key='interior_energetics.grain_size', old='0.1', new='0.001'
):
    return {
        'id': fid,
        'pr': pr,
        'file': facts.CONFIG_SCHEMA,
        'kind': 'config',
        'change': 'default',
        'key': key,
        'old': old,
        'new': new,
    }


def _data(prs=None, releases=None, fact_list=None, previous=None):
    return {
        'window': {'since': SINCE, 'until': UNTIL, 'backdated': True, 'previous': previous},
        'repos': ['FormingWorlds/PROTEUS', 'nichollsh/AGNI'],
        'prs': prs if prs is not None else [_pr()],
        'releases': releases if releases is not None else [],
        'facts': fact_list if fact_list is not None else [_fact()],
        'known_keys': [
            'interior_energetics.grain_size',
            'interior_energetics.rfront_loc',
            'T_surf',
        ],
        'bodies_kept': 1,
        'bodies_cut': 0,
    }


def _entry(**over):
    entry = {
        'section': 'results',
        'title': 'Smaller default grain size',
        'text': 'The default grain size is now smaller, so settling is slower.',
        'prs': ['PROTEUS#849'],
        'releases': [],
        'facts': ['F1'],
        'logins': ['alice'],
    }
    entry.update(over)
    return entry


def _draft(*entries, empty=False):
    return {'entries': list(entries), 'empty': empty}


# common


def test_parse_time_date_and_zulu():
    """A bare date is 00:00 UTC and a Z time keeps its clock time."""
    assert common.parse_time('2026-09-01') == datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert common.iso(common.parse_time('2026-09-01T12:30:00Z')) == '2026-09-01T12:30:00Z'
    assert common.iso(common.parse_time('2026-09-01T14:30:00+02:00')) == '2026-09-01T12:30:00Z'


@pytest.mark.parametrize(
    ('login', 'kind', 'expected'),
    [
        ('alice', 'User', False),
        ('alice', None, False),
        ('dependabot', 'User', True),
        ('Dependabot', 'User', True),
        ('renovate[bot]', 'User', True),
        ('github-actions', 'Bot', True),
        ('acme', 'Organization', True),
        (None, 'User', True),
    ],
)
def test_is_bot(login, kind, expected):
    """Bots, organisations and deleted accounts are never mentioned; people are."""
    assert common.is_bot(login, kind, ['dependabot']) is expected


def test_repository_settings_are_valid_and_exclude_archived_and_boreas():
    """The committed settings load, list each repository once and leave BOREAS and the archived SOCRATES out."""
    loaded = common.load_settings()
    assert loaded['publish'] is False
    assert len(set(loaded['repos'])) == len(loaded['repos'])
    assert not any('BOREAS' in r or r == 'nichollsh/SOCRATES' for r in loaded['repos'])
    assert loaded['repos'][0] == 'FormingWorlds/PROTEUS'


_VALID = common.load_settings()
_BAD_VALUES = {
    'publish': [1, 'true', 0, None],
    'category': ['', '  ', 3, None],
    'model': ['', None, 5],
    'max_bodies': [-1, 501, 1.5, True, 'ten', float('nan'), float('inf')],
    'body_chars': [0, -5, 20001, 2.0, False, float('nan')],
    'bots': ['dependabot', [1], None],
    'repos': [[], ['PROTEUS'], ['a/b/c'], 'FormingWorlds/PROTEUS', [3]],
    'tracked': [[], {'a/b': 'file'}, {'a/b': [1]}],
}


@pytest.mark.parametrize(('key', 'bad'), [(k, v) for k, vs in _BAD_VALUES.items() for v in vs])
def test_settings_reject_wrong_types_and_ranges(key, bad):
    """Every setting rejects a wrong type, a bool for an int, NaN, inf and an out-of-range value."""
    with pytest.raises(common.DigestError, match=key):
        common.validate_settings({**_VALID, key: bad})


@pytest.mark.parametrize('key', sorted(_BAD_VALUES))
def test_settings_reject_a_missing_key(key):
    """A setting left out of the file stops the run."""
    with pytest.raises(common.DigestError, match=key):
        common.validate_settings({k: v for k, v in _VALID.items() if k != key})


@given(st.integers(min_value=0, max_value=500), st.integers(min_value=1, max_value=20000))
@settings(max_examples=50, deadline=None)
def test_settings_accept_the_valid_range(max_bodies, body_chars):
    """Every integer inside both ranges is accepted, including the limits."""
    out = common.validate_settings(
        {**_VALID, 'max_bodies': max_bodies, 'body_chars': body_chars}
    )
    assert (out['max_bodies'], out['body_chars']) == (max_bodies, body_chars)


class _Resp:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.body


def _http_error(code):
    import urllib.error

    return urllib.error.HTTPError('u', code, 'x', {'Retry-After': '0'}, None)


def test_github_retries_a_rate_limit_then_reads(monkeypatch):
    """A 403 is retried; a 404 gives None; a 500 stops the run."""
    calls = []

    def fake(req, timeout):
        calls.append(req.full_url)
        if len(calls) == 1:
            raise _http_error(403)
        return _Resp(b'text')

    monkeypatch.setattr(common.urllib.request, 'urlopen', fake)
    monkeypatch.setattr(common.time, 'sleep', lambda s: None)
    assert common.GitHub('t').raw_file('a/b', 'x y.toml', 'abc') == 'text'
    assert len(calls) == 2 and 'x%20y.toml?ref=abc' in calls[0]

    monkeypatch.setattr(
        common.urllib.request,
        'urlopen',
        lambda r, timeout: (_ for _ in ()).throw(_http_error(404)),
    )
    assert common.GitHub('t').raw_file('a/b', 'f', 'r') is None
    monkeypatch.setattr(
        common.urllib.request,
        'urlopen',
        lambda r, timeout: (_ for _ in ()).throw(_http_error(500)),
    )
    with pytest.raises(common.DigestError, match='500'):
        common.GitHub('t').raw_file('a/b', 'f', 'r')


def test_graphql_error_stops_the_run(monkeypatch):
    """A GraphQL answer with errors raises instead of returning partial data."""
    body = json.dumps({'data': None, 'errors': [{'message': 'boom'}]}).encode()
    monkeypatch.setattr(common.urllib.request, 'urlopen', lambda r, timeout: _Resp(body))
    with pytest.raises(common.DigestError, match='boom'):
        common.GitHub('t').graphql('{x}')


# collect


def test_clean_text_removes_links_html_mentions_and_cuts():
    """Link text stays; URLs, images, HTML, comments and every @ go; the result is cut."""
    raw = (
        '<!-- template -->Fixes [the bug](https://evil.example/x) for @alice. '
        '![img](https://i.png) <b>bold</b> see https://x.org/a and www.y.org; ping @org/team'
    )
    out = collect.clean_text(raw, 1000)
    assert out == 'Fixes the bug for alice. bold see and ping org/team'
    assert collect.clean_text(raw, 5) == out[:5]
    assert collect.clean_text(None, 10) == '' and collect.clean_text('abc', 0) == ''


@given(st.text(max_size=400), st.integers(min_value=0, max_value=300))
@settings(max_examples=200, deadline=None)
def test_clean_text_never_leaks_a_mention_link_or_tag(text, limit):
    """Whatever the input, the cleaned text has no @, no URL scheme, no tag and fits the limit."""
    out = collect.clean_text(text, limit)
    assert '@' not in out and '://' not in out and len(out) <= limit
    assert not re.search(r'<[^>]*>', out)


def test_resolve_window_rules():
    """Inputs make a back-dated window; else the last post's end; else 7 days."""
    now = datetime(2026, 11, 23, 22, 23, tzinfo=timezone.utc)
    got = collect.resolve_window('2026-09-01', '2026-09-30', None, now)
    assert got == {'since': SINCE, 'until': UNTIL, 'backdated': True}
    got = collect.resolve_window(None, None, {'until': '2026-11-16T22:30:00Z'}, now)
    assert got == {
        'since': '2026-11-16T22:30:00Z',
        'until': '2026-11-23T22:23:00Z',
        'backdated': False,
    }
    got = collect.resolve_window(None, None, None, now)
    assert got['since'] == '2026-11-16T22:23:00Z' and not got['backdated']
    got = collect.resolve_window(None, '2026-10-01', {'until': '2026-09-01T00:00:00Z'}, now)
    assert got['since'] == '2026-09-24T00:00:00Z' and got['backdated']
    with pytest.raises(common.DigestError, match='empty window'):
        collect.resolve_window('2026-09-30', '2026-09-30', None, now)


def _node(merged=None, created='2026-09-02T00:00:00Z', draft=False):
    return {
        'number': 7,
        'url': 'u',
        'title': 't',
        'body': 'b',
        'isDraft': draft,
        'state': 'MERGED' if merged else 'OPEN',
        'createdAt': created,
        'mergedAt': merged,
        'updatedAt': '2026-10-01T00:00:00Z',
        'additions': 3,
        'deletions': 4,
        'author': {'login': 'alice', '__typename': 'User'},
        'labels': {'nodes': []},
        'files': {'nodes': [{'path': 'a.py'}]},
        'mergeCommit': {'oid': 'm', 'parents': {'nodes': [{'oid': 'p'}]}} if merged else None,
    }


@pytest.mark.parametrize(
    ('node', 'status'),
    [
        (_node(merged=SINCE), 'merged'),
        (_node(merged='2026-09-29T23:59:59Z'), 'merged'),
        (_node(merged='2026-08-31T23:59:59Z'), None),
        (_node(merged=UNTIL), 'open'),
        (_node(merged='2026-10-13T00:00:00Z'), 'open'),
        (_node(), 'open'),
        (_node(draft=True), None),
        (_node(created=UNTIL), None),
    ],
)
def test_pr_record_window_edges(node, status):
    """Merged in [since, until) is merged; open at until and not draft is open; else dropped."""
    rec = collect.pr_record('FormingWorlds/PROTEUS', node, SINCE, UNTIL)
    assert (rec['status'] if rec else None) == status
    if status == 'merged':
        assert (rec['merge_oid'], rec['parent_oid'], rec['size']) == ('m', 'p', 7)
    if status == 'open':
        assert rec['merged_at'] is None and rec['merge_oid'] is None


def test_release_record_window_and_draft():
    """Only a published, non-draft release inside the window is kept."""
    node = {
        'tagName': '1.13.0',
        'name': None,
        'url': 'u',
        'publishedAt': '2026-09-29T09:19:58Z',
        'isDraft': False,
        'isPrerelease': False,
        'description': 'd',
        'author': {'login': 'bob', '__typename': 'User'},
    }
    rec = collect.release_record('nichollsh/AGNI', node, SINCE, UNTIL)
    assert (rec['ref'], rec['name']) == ('AGNI@1.13.0', '1.13.0')
    assert (
        collect.release_record('nichollsh/AGNI', {**node, 'isDraft': True}, SINCE, UNTIL)
        is None
    )
    assert (
        collect.release_record('nichollsh/AGNI', {**node, 'publishedAt': UNTIL}, SINCE, UNTIL)
        is None
    )
    assert (
        collect.release_record('nichollsh/AGNI', {**node, 'publishedAt': None}, SINCE, UNTIL)
        is None
    )


class _FakeGH:
    """Answers GraphQL queries from a list of pages and files from a dict."""

    def __init__(self, pages=(), files=None):
        self.pages, self.files, self.queries = list(pages), files or {}, []

    def graphql(self, query, **variables):
        self.queries.append(variables)
        return self.pages.pop(0)

    def raw_file(self, repo, path, ref):
        return self.files.get((repo, path, ref))


def _janus():
    page = json.loads((_DATA / 'janus_page.json').read_text())
    page['repository']['pullRequests']['pageInfo'] = {'hasNextPage': False, 'endCursor': None}
    return page


def test_collect_repo_reads_a_recorded_page():
    """The recorded JANUS page gives the merged PR of the window, the open PRs and the release."""
    window = {'since': '2026-11-01T00:00:00Z', 'until': '2026-11-17T12:00:00Z'}
    prs, rels = collect.collect_repo(_FakeGH([_janus()]), 'FormingWorlds/JANUS', window)
    assert [(p['ref'], p['status']) for p in prs] == [('JANUS#103', 'merged')]
    assert [r['ref'] for r in rels] == ['JANUS@26.11.05']
    later = {'since': '2026-11-01T00:00:00Z', 'until': '2026-11-18T00:00:00Z'}
    prs, _ = collect.collect_repo(_FakeGH([_janus()]), 'FormingWorlds/JANUS', later)
    assert {p['ref']: p['status'] for p in prs} == {
        'JANUS#105': 'open',
        'JANUS#104': 'open',
        'JANUS#103': 'merged',
    }


def test_collect_repo_pages_until_older_than_the_window():
    """A full page whose oldest PR is in the window asks for the next page; an older one stops."""
    first = _janus()
    first['repository']['pullRequests']['pageInfo'] = {'hasNextPage': True, 'endCursor': 'c1'}
    second = _janus()
    second['repository']['pullRequests']['nodes'] = []
    gh = _FakeGH([first, second])
    collect.collect_repo(
        gh,
        'FormingWorlds/JANUS',
        {'since': '2026-11-01T00:00:00Z', 'until': '2026-11-18T00:00:00Z'},
    )
    assert [q['cursor'] for q in gh.queries] == [None, 'c1']
    gh = _FakeGH([_janus()])
    gh.pages[0]['repository']['pullRequests']['pageInfo'] = {
        'hasNextPage': True,
        'endCursor': 'c1',
    }
    collect.collect_repo(
        gh,
        'FormingWorlds/JANUS',
        {'since': '2026-11-15T00:00:00Z', 'until': '2026-11-18T00:00:00Z'},
    )
    assert len(gh.queries) == 1


@pytest.mark.parametrize('flag', ['isPrivate', 'isArchived'])
def test_collect_repo_stops_on_a_private_or_archived_repository(flag):
    """A repository that is not public and active stops the run."""
    page = _janus()
    page['repository'][flag] = True
    with pytest.raises(common.DigestError, match='not a public, active'):
        collect.collect_repo(
            _FakeGH([page]), 'FormingWorlds/JANUS', {'since': SINCE, 'until': UNTIL}
        )
    with pytest.raises(common.DigestError):
        collect.collect_repo(
            _FakeGH([{'repository': None}]), 'a/b', {'since': SINCE, 'until': UNTIL}
        )


def test_add_bodies_keeps_the_largest_and_every_release():
    """Bodies stay for the largest PRs and for releases; the other PRs give titles only."""
    prs = [dict(_pr(f'PROTEUS#{n}', size=n), raw_body=f'body @x {n}') for n in (1, 5, 3)]
    rels = [dict(_release(), raw_body='rel https://u body')]
    out = collect.add_bodies(prs, rels, 2, 100)
    assert out == {'bodies_kept': 2, 'bodies_cut': 1}
    assert {p['ref']: p['body'] for p in prs} == {
        'PROTEUS#1': '',
        'PROTEUS#5': 'body x 5',
        'PROTEUS#3': 'body x 3',
    }
    assert rels[0]['body'] == 'rel body' and 'raw_body' not in rels[0]


def test_collect_refuses_two_repositories_with_one_name():
    """Two repositories with the same name would make references ambiguous."""
    with pytest.raises(common.DigestError, match='share a name'):
        collect.collect(_FakeGH(), {**_VALID, 'repos': ['a/SOCRATES', 'b/SOCRATES']}, {})


def _post(
    author='github-actions',
    kind='Bot',
    state='{"until":"2026-11-16T22:30:00Z","announced_open":["X#1"]}',
):
    return {
        'url': 'https://d/1',
        'createdAt': 'c',
        'body': f'text\n<!-- weekly-digest-state {state} -->',
        'author': {'login': author, '__typename': kind},
    }


def test_previous_state_trusts_only_the_workflow_bot():
    """A marker in a post by a person is skipped; the newest bot post gives the state."""
    cats = {
        'repository': {
            'id': 'R',
            'discussionCategories': {'nodes': [{'id': 'C1', 'slug': 'weekly-digest'}]},
        }
    }
    posts = {
        'repository': {
            'discussions': {
                'nodes': [
                    _post(
                        author='mallory', kind='User', state='{"until":"2020-01-01T00:00:00Z"}'
                    ),
                    _post(),
                ]
            }
        }
    }
    state = collect.previous_state(
        _FakeGH([cats, posts]), 'FormingWorlds/PROTEUS', 'weekly-digest'
    )
    assert state['until'] == '2026-11-16T22:30:00Z' and state['url'] == 'https://d/1'
    posts = {'repository': {'discussions': {'nodes': [_post(kind='User')]}}}
    assert (
        collect.previous_state(_FakeGH([cats, posts]), 'FormingWorlds/PROTEUS', 'weekly-digest')
        is None
    )
    none = {'repository': {'id': 'R', 'discussionCategories': {'nodes': []}}}
    assert (
        collect.previous_state(_FakeGH([none]), 'FormingWorlds/PROTEUS', 'weekly-digest')
        is None
    )


# facts


def _schema(**fields):
    return json.dumps({'fields': [{'path': k, **v} for k, v in fields.items()]})


def test_config_schema_diff_kinds():
    """Added, removed, default, choices, bounds and description changes each give one fact."""
    old = _schema(
        a={'default': '1'}, b={'default': '2', 'choices': ['x']}, gone={'default': '0'}
    )
    new = _schema(
        a={'default': '1', 'description': 'new'},
        b={'default': '3', 'choices': ['x', 'y']},
        c={'default': '5'},
    )
    got = [
        (f['change'], f['key'], f['old'], f['new'])
        for f in facts.diff_file(facts.CONFIG_SCHEMA, old, new)
    ]
    assert got == [
        ('description', 'a', None, 'new'),
        ('default', 'b', '2', '3'),
        ('choices', 'b', ['x'], ['x', 'y']),
        ('added', 'c', None, '5'),
        ('removed', 'gone', '0', None),
    ]
    assert facts.diff_file(facts.CONFIG_SCHEMA, old, old) == []


def test_september_schema_changes_are_recovered():
    """The recorded schemas of PROTEUS 849, 918, 848 and 858 give the facts of the September rows."""
    pairs = json.loads((_DATA / 'september_schema.json').read_text())

    def run(ref):
        return [
            (f['change'], f['key'], f['old'], f['new'])
            for f in facts.diff_file(
                facts.CONFIG_SCHEMA,
                json.dumps(pairs[ref]['old']),
                json.dumps(pairs[ref]['new']),
            )
        ]

    assert run('PROTEUS#849') == [('default', 'interior_energetics.grain_size', '0.1', '0.001')]
    assert ('default', 'interior_energetics.rfront_loc', '0.5', '0.4') in run('PROTEUS#918')
    assert run('PROTEUS#848') == [
        ('added', 'interior_energetics.aragog.separation_viscosity', None, '"mixture"')
    ]
    assert [c[:2] for c in run('PROTEUS#858')] == [
        ('description', 'interior_energetics.aragog.phi_step_cap')
    ]


def test_output_schema_diff_reads_columns_and_units():
    """A new helpfile column and a unit change are facts; producers are ignored."""
    old = json.dumps({'keys': [{'name': 'T_surf', 'unit': 'K', 'producers': [1]}]})
    new = json.dumps(
        {
            'keys': [
                {'name': 'T_surf', 'unit': 'C', 'producers': [2]},
                {'name': 'R_sat', 'unit': 'm'},
            ]
        }
    )
    got = {(f['change'], f['key']) for f in facts.diff_file(facts.OUTPUT_SCHEMA, old, new)}
    assert got == {('unit', 'T_surf'), ('added', 'R_sat')}


def test_pyproject_floors_extras_and_module_pins():
    """Runtime floors, extras and PROTEUS module pins are compared per package."""
    old = '[project]\ndependencies = ["fwl-zalmoxis>=26.07.17", "numpy"]\n[project.optional-dependencies]\nvulcan = ["fwl-vulcan>=1"]\n[tool.proteus.modules.agni]\nref = "aaa"\n'
    new = '[project]\ndependencies = ["fwl-zalmoxis>=26.09.21", "numpy", "scipy[all] >= 1.0 ; python_version>\'3\'"]\n[project.optional-dependencies]\nvulcan = ["fwl-vulcan>=2"]\n[tool.proteus.modules.agni]\nref = "bbb"\n'
    got = {
        (f['change'], f['key'], f['old'], f['new'])
        for f in facts.diff_file('pyproject.toml', old, new)
    }
    assert got == {
        ('value', 'dependencies:fwl-zalmoxis', '>=26.07.17', '>=26.09.21'),
        ('added', 'dependencies:scipy', None, ">= 1.0 ; python_version>'3'"),
        ('value', 'extra vulcan:fwl-vulcan', '>=1', '>=2'),
        ('value', 'tool.proteus.modules.agni.ref', 'aaa', 'bbb'),
    }


def test_julia_compat_and_default_toml():
    """Julia compat bounds and nested default TOML values are compared; a missing file is empty."""
    got = facts.diff_file(
        'Project.toml', '[compat]\nJulia = "1.11"\n', '[compat]\nJulia = "1.13"\n'
    )
    assert [(f['key'], f['old'], f['new']) for f in got] == [('compat:Julia', '1.11', '1.13')]
    got = facts.diff_file(
        'input/default.toml', '[a]\n[a.b]\nc = 1\n', '[a]\n[a.b]\nc = 2\nd = 3\n'
    )
    assert {(f['change'], f['key']) for f in got} == {('value', 'a.b.c'), ('added', 'a.b.d')}
    assert [
        f['change'] for f in facts.diff_file('Project.toml', None, '[compat]\nX = "1"\n')
    ] == ['added']
    assert facts.diff_file('pyproject.toml', None, None) == []


def test_long_values_are_cut():
    """A description longer than the limit is cut and marked."""
    long_text = 'x' * (facts.DESCRIPTION_CHARS + 10)
    got = facts.diff_file(
        facts.CONFIG_SCHEMA,
        _schema(a={'description': 'short'}),
        _schema(a={'description': long_text}),
    )
    assert (
        got[0]['new'].endswith(' [cut]') and len(got[0]['new']) == facts.DESCRIPTION_CHARS + 6
    )


def test_group_facts_merges_large_sets_only():
    """More than the limit of additions in one file become one fact with all keys; fewer stay."""
    many = [
        {
            'file': 'f',
            'kind': 'config',
            'change': 'added',
            'key': f'orbit.solver.k{i}',
            'old': None,
            'new': i,
        }
        for i in range(9)
    ]
    other = {'file': 'f', 'kind': 'config', 'change': 'default', 'key': 'z', 'old': 1, 'new': 2}
    out = facts.group_facts(many + [other], limit=8)
    grouped = [f for f in out if f.get('members')]
    assert len(out) == 2 and len(grouped) == 1
    assert grouped[0]['key'] == 'orbit.solver' and grouped[0]['members'] == [
        f['key'] for f in many
    ]
    assert grouped[0]['new'].startswith('9 keys added: orbit.solver.k0')
    assert facts.group_facts(many[:8], limit=8) == many[:8]
    mixed = [dict(m, key=k) for m, k in zip(many, ['a.x', 'b.y'] * 5)]
    assert facts.group_facts(mixed, limit=8)[0]['key'] == 'f'


def test_attach_facts_reads_only_merged_prs_that_touch_tracked_files():
    """Facts come from the parent and merge commit of merged PRs that change a tracked file."""
    touch = _pr(
        'PROTEUS#1', files=['pyproject.toml', 'src/x.py'], merge_oid='m', parent_oid='p'
    )
    skip_files = _pr('PROTEUS#2', files=['src/x.py'])
    skip_open = _pr('PROTEUS#3', status='open', files=['pyproject.toml'], parent_oid=None)
    gh = _FakeGH(
        files={
            (
                'FormingWorlds/PROTEUS',
                'pyproject.toml',
                'p',
            ): '[project]\ndependencies = ["a>=1"]\n',
            (
                'FormingWorlds/PROTEUS',
                'pyproject.toml',
                'm',
            ): '[project]\ndependencies = ["a>=2"]\n',
            ('FormingWorlds/PROTEUS', facts.CONFIG_SCHEMA, None): _schema(
                **{'a.grain_size': {}, 'plain': {}}
            ),
        }
    )
    data = {'prs': [touch, skip_files, skip_open]}
    facts.attach_facts(gh, data, {'tracked': {'FormingWorlds/PROTEUS': [facts.CONFIG_SCHEMA]}})
    assert [(f['id'], f['pr'], f['key']) for f in data['facts']] == [
        ('F1', 'PROTEUS#1', 'dependencies:a')
    ]
    assert data['known_keys'] == ['a.grain_size']


# check


def test_a_valid_draft_passes():
    """A draft whose every reference, login, number and key comes from the data has no failure."""
    assert check.check(_data(), _draft(_entry())) == []


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        ({'prs': ['PROTEUS#9999']}, 'PR PROTEUS#9999 is not in the collected set'),
        ({'releases': ['AGNI@9.9.9']}, 'release AGNI@9.9.9 is not in the collected set'),
        ({'facts': ['F1', 'F77']}, 'fact F77 is not in the collected set'),
        ({'logins': ['mallory']}, 'login mallory is not an author'),
        ({'prs': [], 'facts': []}, 'cites no pull request or release'),
        ({'facts': []}, 'a results entry cites no fact or release'),
        ({'text': 'See https://evil.example now.'}, 'text holds a URL'),
        ({'text': 'See www.evil.example now.'}, 'text holds a URL'),
        ({'text': 'Thanks to @alice.'}, 'text holds an @'),
        ({'text': 'A <img src=x> tag.'}, 'text holds HTML'),
        ({'text': 'A [link](x) here.'}, 'text holds a markdown link'),
        ({'text': 'Merged in PROTEUS#849.'}, 'text holds a # reference'),
        ({'text': 'Use `code` here.'}, 'text holds a backtick'),
        ({'title': 'Grain size @bob'}, 'text holds an @'),
        ({'text': 'The grain size is now 0.05 m.'}, 'number 0.05 is in no cited source'),
        ({'text': 'The rfront_loc moved too.'}, 'key rfront_loc is in no cited fact'),
        (
            {'text': 'interior_energetics.rfront_loc moved.'},
            'key interior_energetics.rfront_loc',
        ),
        ({'text': 'T_surf is lower.'}, 'key t_surf is in no cited fact'),
        ({'section': 'open'}, 'is merged, not allowed in section open'),
    ],
)
def test_check_fails_closed(change, message):
    """Each invented or unsafe item in a draft is a failure that names the cause."""
    fails = check.check(_data(), _draft(_entry(**change)))
    assert any(message in f for f in fails), fails


def test_check_accepts_numbers_and_keys_from_cited_sources():
    """A number written another way, a version after a v and a key of a cited fact pass."""
    data = _data(releases=[_release()])
    entry = _entry(
        text='The grain size is now 1e-3 instead of 0.10; grain_size and interior_energetics.grain_size changed with AGNI 1.13.0.',
        releases=['AGNI@1.13.0'],
        logins=['alice', 'bob'],
    )
    assert check.check(data, _draft(entry)) == []


def test_check_section_rules_for_open_prs_and_releases():
    """Open PRs belong in open only; a release cannot be in open; a fact needs its PR cited."""
    data = _data(
        prs=[_pr(), _pr('PROTEUS#900', status='open', author='carol')], releases=[_release()]
    )
    fails = check.check(
        data,
        _draft(_entry(section='capabilities', prs=['PROTEUS#900'], facts=[], logins=['carol'])),
    )
    assert any('PROTEUS#900 is open' in f for f in fails)
    fails = check.check(
        data,
        _draft(
            _entry(
                section='open',
                prs=['PROTEUS#900'],
                releases=['AGNI@1.13.0'],
                facts=[],
                logins=[],
            )
        ),
    )
    assert any('a release in section open' in f for f in fails)
    ok = _entry(
        section='open', prs=['PROTEUS#900'], facts=[], logins=['carol'], text='Needs review.'
    )
    assert check.check(data, _draft(ok)) == []
    fails = check.check(
        data, _draft(_entry(prs=['PROTEUS#900'], section='open', facts=['F1'], logins=[]))
    )
    assert any('fact F1 belongs to PROTEUS#849, not cited' in f for f in fails)


def test_check_grouped_fact_members_count_as_cited_keys():
    """A key listed in a grouped fact may be named in the text."""
    grouped = {
        **_fact('F2', key='orbit.solver', old=None, new='9 keys added'),
        'change': 'added',
        'members': ['orbit.solver.max_rel_de'],
    }
    data = _data(fact_list=[_fact(), grouped])
    data['known_keys'].append('orbit.solver.max_rel_de')
    entry = _entry(section='capabilities', facts=['F2'], text='New max_rel_de limit.')
    assert check.check(data, _draft(entry)) == []
    data['facts'][1]['members'] = []
    assert check.check(data, _draft(entry)) != []


def test_check_empty_flag_and_missing_entries():
    """The empty flag must match the window, and a non-empty window needs entries."""
    assert 'does not match' in check.check(_data(), _draft(empty=True))[0]
    assert 'no entries' in check.check(_data(), _draft())[0]
    quiet = _data(prs=[_pr(status='open')], fact_list=[])
    assert check.check(quiet, _draft(empty=True)) == []
    assert 'does not match' in check.check(quiet, _draft(_entry()))[0]


def test_check_main_exit_code(tmp_path, capsys):
    """The command exits 1 and lists the failure on a bad draft, 0 on a good one."""
    src, bad, good = tmp_path / 'c.json', tmp_path / 'bad.json', tmp_path / 'good.json'
    src.write_text(json.dumps(_data()))
    bad.write_text(json.dumps(_draft(_entry(prs=['PROTEUS#1']))))
    good.write_text(json.dumps(_draft(_entry())))
    assert check.main([str(src), str(bad)]) == 1
    assert 'FAIL' in capsys.readouterr().out
    assert check.main([str(src), str(good)]) == 0


@given(st.text(min_size=1, max_size=60))
@settings(max_examples=150, deadline=None)
def test_any_text_with_a_url_or_mention_fails(noise):
    """Text that holds a URL or an @ never passes, whatever surrounds it."""
    for bad in (f'{noise} https://x.y {noise}', f'{noise}@{noise}'):
        assert check.check(_data(), _draft(_entry(text=bad))) != []


# render


def test_tuesday_title_in_amsterdam_time():
    """Monday night UTC gives the next day; a Tuesday stays; a late Monday in summer is Tuesday."""
    assert (
        render.tuesday_title('2026-11-23T22:23:00Z')
        == 'PROTEUS weekly digest, 24 November 2026'
    )
    assert (
        render.tuesday_title('2026-11-24T10:00:00Z')
        == 'PROTEUS weekly digest, 24 November 2026'
    )
    assert render.tuesday_title('2026-06-22T22:23:00Z') == 'PROTEUS weekly digest, 23 June 2026'
    assert (
        render.tuesday_title('2026-11-25T00:00:00Z') == 'PROTEUS weekly digest, 1 December 2026'
    )


@pytest.mark.parametrize(
    ('fact', 'line'),
    [
        (_fact(), '`interior_energetics.grain_size`: `0.1` → `0.001`'),
        (
            {**_fact(), 'change': 'added', 'old': None, 'new': '"mixture"'},
            '`interior_energetics.grain_size` added, default `"mixture"`',
        ),
        (
            {**_fact(), 'change': 'added', 'old': None, 'new': None},
            '`interior_energetics.grain_size` added',
        ),
        ({**_fact(), 'change': 'removed'}, '`interior_energetics.grain_size` removed'),
        (
            {**_fact(), 'change': 'description'},
            '`interior_energetics.grain_size`: description changed',
        ),
        (
            {**_fact(), 'change': 'choices', 'old': 'a|b', 'new': 'a`c'},
            '`interior_energetics.grain_size` choices: `a b` → `a c`',
        ),
        (
            {**_fact(key=facts.CONFIG_SCHEMA), 'change': 'added', 'members': ['a', 'b']},
            '2 keys added in `config_schema.json`',
        ),
        (
            {**_fact(key='orbit.solver'), 'change': 'added', 'members': ['a', 'b', 'c']},
            '3 keys added in `orbit.solver`',
        ),
    ],
)
def test_fact_line(fact, line):
    """Each kind of fact is printed in words, with pipes and backticks removed from values."""
    assert render.fact_line(fact) == line


def test_facts_text_dedupes_and_caps():
    """Equal lines from two PRs print once; more than the cap end in a count."""
    same = [_fact('F1'), _fact('F2', pr='PROTEUS#2')]
    assert render.facts_text(same) == '`interior_energetics.grain_size`: `0.1` → `0.001`'
    many = [_fact(f'F{i}', key=f'k{i}') for i in range(render.MAX_FACTS + 2)]
    assert render.facts_text(many).endswith('; and 2 more')


def test_plain_replaces_dashes_pipes_and_breaks():
    """Model text loses em and en dashes, pipes and line breaks."""
    assert render.plain('A \u2014 B\u2013C | D\n\nE') == 'A, B, C / D E'


def _render(draft_doc, data=None, live=False, bots=('dependabot',)):
    return render.render_body(data or _data(), draft_doc, live, 'https://run', list(bots))


def test_render_results_table_and_footer():
    """A results entry is a table row with the setting change, date, link and author."""
    body, state = _render(_draft(_entry()))
    assert '**Dry run: not posted.' in body
    row = next(
        line for line in body.splitlines() if line.startswith('| Smaller default grain size')
    )
    assert '`interior_energetics.grain_size`: `0.1` → `0.001`.' in row
    assert row.endswith(
        '| 6 Sep | [PROTEUS#849](https://github.com/FormingWorlds/PROTEUS/pull/849), by `@alice` |'
    )
    assert (
        'between 1 and 29 September 2026 (UTC 2026-09-01T00:00:00Z to 2026-09-30T00:00:00Z)'
        in body
    )
    assert '[Workflow run](https://run)' in body
    assert body.rstrip().endswith(
        '<!-- weekly-digest-state {"until":"2026-09-30T00:00:00Z","announced_open":[]} -->'
    )
    assert state == {'until': UNTIL, 'announced_open': []}


def test_render_mentions_once_per_person_and_never_bots():
    """Live: the first mention is @login and later ones plain; bots never; dry: code spans."""
    data = _data(
        prs=[
            _pr(),
            _pr('PROTEUS#850', author='alice'),
            _pr('PROTEUS#851', author='dependabot', kind='Bot'),
        ],
        fact_list=[],
    )
    entries = [
        _entry(section='capabilities', facts=[], title='A'),
        _entry(section='install', facts=[], prs=['PROTEUS#850'], title='B'),
        _entry(
            section='install', facts=[], prs=['PROTEUS#851'], title='C', logins=['dependabot']
        ),
    ]
    live, _ = _render(_draft(*entries), data, live=True)
    assert (
        live.count('@alice') == 1
        and 'by alice' in live
        and '@dependabot' not in live
        and 'by dependabot' in live
    )
    assert 'Dry run' not in live
    dry, _ = _render(_draft(*entries), data)
    assert 'by `@alice`' in dry and dry.count('@alice') == 1


def test_render_open_pr_announced_before_is_not_mentioned_again():
    """An open PR from the previous post keeps its author unmentioned; a new open PR mentions."""
    prs = [
        _pr(),
        _pr('PROTEUS#900', status='open', author='carol'),
        _pr('PROTEUS#901', status='open', author='dan'),
        _pr('PROTEUS#902', status='open', author='erin'),
    ]
    previous = {'until': SINCE, 'announced_open': ['PROTEUS#900', 'PROTEUS#902', 'PROTEUS#5']}
    data = _data(prs=prs, previous=previous)
    entries = [
        _entry(),
        _entry(
            section='open',
            prs=['PROTEUS#900', 'PROTEUS#901'],
            facts=[],
            logins=['carol', 'dan'],
            title='O',
        ),
    ]
    body, state = _render(_draft(*entries), data, live=True)
    assert 'by carol, @dan' in body
    assert state['announced_open'] == ['PROTEUS#900', 'PROTEUS#901', 'PROTEUS#902']


def test_render_quiet_week():
    """An empty window renders the short note and no section."""
    body, _ = _render(_draft(empty=True), _data(prs=[], fact_list=[]))
    assert 'Quiet week: no pull request was merged' in body and '## ' not in body


def test_render_main_writes_both_files(tmp_path):
    """The command writes the readable markdown and the post JSON with the same body."""
    src, dr = tmp_path / 'c.json', tmp_path / 'd.json'
    src.write_text(json.dumps(_data()))
    dr.write_text(json.dumps(_draft(_entry())))
    assert (
        render.main([str(src), str(dr), str(tmp_path / 'out'), '--live', '--run-url', 'R']) == 0
    )
    doc = json.loads((tmp_path / 'out' / 'digest.json').read_text())
    assert doc['title'] == 'PROTEUS weekly digest, 6 October 2026'
    assert (tmp_path / 'out' / 'digest.md').read_text() == f'# {doc["title"]}\n\n{doc["body"]}'
    assert '@alice' in doc['body'] and '`@alice`' not in doc['body']


# draft


def test_model_input_has_no_urls_or_commit_ids():
    """The model sees references, text and facts, never URLs, commits or the window flags."""
    text = json.dumps(draft.model_input(_data(releases=[_release()])))
    assert 'https://' not in text and '"m1"' not in text and 'backdated' not in text
    assert '"ref": "PROTEUS#849"' in text and '"id": "F1"' in text


def test_prompt_puts_instruction_and_data_together():
    """The instruction and the JSON data go in one stdin text."""
    prompt = draft.build_prompt(_data())
    assert prompt.startswith((_ROOT / 'tools' / 'digest' / 'prompt.md').read_text())
    assert '```json\n{"window": {"since": "2026-09-01T00:00:00Z"' in prompt


def _fake_cli(tmp_path, outputs, rc=0):
    """A stand-in for the CLI that answers each call with the next output and logs its stdin."""
    state = tmp_path / 'calls'
    script = tmp_path / 'claude'
    answers = tmp_path / 'answers.json'
    answers.write_text(json.dumps(outputs))
    script.write_text(
        f'#!{sys.executable}\n'
        'import json, sys, pathlib\n'
        f'state = pathlib.Path({str(state)!r})\n'
        'n = len(state.read_text().splitlines()) if state.exists() else 0\n'
        'state.open("a").write(json.dumps(sys.argv[1:]) + "\\n")\n'
        f'pathlib.Path({str(tmp_path)!r}, f"stdin{{n}}").write_text(sys.stdin.read())\n'
        f'print(json.dumps(json.load(open({str(answers)!r}))[n]))\n'
        f'sys.exit({rc})\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script, state


def test_run_model_passes_no_tools_and_the_schema(tmp_path, monkeypatch):
    """The CLI gets no tools, the schema and the prompt on stdin; its structured output returns."""
    script, state = _fake_cli(
        tmp_path, [{'is_error': False, 'structured_output': {'entries': []}}]
    )
    monkeypatch.setenv('CLAUDE_BIN', str(script))
    assert draft.run_model('PROMPT', 'm') == {'entries': []}
    args = json.loads(state.read_text().splitlines()[0])
    assert args[args.index('--tools') + 1] == '' and args[args.index('--model') + 1] == 'm'
    assert json.loads(args[args.index('--json-schema') + 1])['required'] == ['entries']
    assert '-p' in args and not any(a == 'PROMPT' for a in args)
    assert (tmp_path / 'stdin0').read_text() == 'PROMPT'


@pytest.mark.parametrize(
    ('outputs', 'rc', 'message'),
    [
        ([{'is_error': True, 'structured_output': {'entries': []}}], 0, 'no structured output'),
        ([{'is_error': False, 'result': 'text'}], 0, 'no structured output'),
        ([{'is_error': False, 'structured_output': {'entries': []}}], 3, 'failed with code 3'),
    ],
)
def test_run_model_failures_stop_the_run(tmp_path, monkeypatch, outputs, rc, message):
    """An error result, a missing structured output or a failed call raises."""
    script, _ = _fake_cli(tmp_path, outputs, rc)
    monkeypatch.setenv('CLAUDE_BIN', str(script))
    with pytest.raises(common.DigestError, match=message):
        draft.run_model('p', 'm')


def test_draft_main_repairs_once_then_keeps_the_answer(tmp_path, monkeypatch):
    """A first answer that fails the checks gets one repair call with the failures; no third call."""
    bad = {'entries': [_entry(prs=['PROTEUS#9999'])]}
    good = {'entries': [_entry()]}
    script, state = _fake_cli(
        tmp_path, [{'structured_output': bad}, {'structured_output': good}, {}]
    )
    monkeypatch.setenv('CLAUDE_BIN', str(script))
    src, dst = tmp_path / 'c.json', tmp_path / 'd.json'
    src.write_text(json.dumps(_data()))
    assert draft.main([str(src), str(dst)]) == 0
    out = json.loads(dst.read_text())
    assert out['entries'] == good['entries'] and out['empty'] is False
    assert any('PROTEUS#9999' in f for f in out['first_failures'])
    assert len(state.read_text().splitlines()) == 2
    assert 'It failed these checks' in (tmp_path / 'stdin1').read_text()


def test_draft_main_skips_the_model_on_an_empty_window(tmp_path, monkeypatch):
    """A window with no merged PR and no release gives an empty draft without a model call."""
    monkeypatch.setenv('CLAUDE_BIN', str(tmp_path / 'missing'))
    src, dst = tmp_path / 'c.json', tmp_path / 'd.json'
    src.write_text(json.dumps(_data(prs=[_pr(status='open')], fact_list=[])))
    assert draft.main([str(src), str(dst)]) == 0
    assert json.loads(dst.read_text()) == {'entries': [], 'empty': True}


# post


@pytest.mark.parametrize(
    ('publish', 'event', 'post_input', 'backdated', 'live'),
    [
        (False, 'schedule', False, False, False),
        (True, 'schedule', False, False, True),
        (True, 'schedule', False, True, False),
        (True, 'workflow_dispatch', False, False, False),
        (True, 'workflow_dispatch', True, False, True),
        (True, 'workflow_dispatch', True, True, False),
        (False, 'workflow_dispatch', True, False, False),
        (True, 'push', False, False, False),
    ],
)
def test_decide(publish, event, post_input, backdated, live):
    """Only a switched-on, non-back-dated scheduled run or confirmed dispatch posts."""
    assert post.decide({'publish': publish}, event, post_input, backdated) is live


def test_decide_command_prints_the_output_line(capsys, monkeypatch):
    """The decide command prints a line for GITHUB_OUTPUT; the committed settings never post."""
    assert (
        post.main(
            ['decide', '--event', 'schedule', '--post-input', 'true', '--backdated', 'false']
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == 'live=false'
    monkeypatch.setattr(post, 'load_settings', lambda: {**_VALID, 'publish': True})
    post.main(
        ['decide', '--event', 'schedule', '--post-input', 'false', '--backdated', 'false']
    )
    assert capsys.readouterr().out.strip() == 'live=true'


def test_create_discussion_finds_the_category_by_slug():
    """The category id is looked up by slug; a missing slug stops before any mutation."""
    cats = {
        'repository': {
            'id': 'R',
            'discussionCategories': {'nodes': [{'id': 'C9', 'slug': 'weekly-digest'}]},
        }
    }
    made = {'createDiscussion': {'discussion': {'url': 'https://d/5'}}}
    gh = _FakeGH([cats, made])
    assert (
        post.create_discussion(gh, 'FormingWorlds/PROTEUS', 'weekly-digest', 'T', 'B')
        == 'https://d/5'
    )
    assert gh.queries[1] == {
        'retry': False,
        'repo': 'R',
        'category': 'C9',
        'title': 'T',
        'body': 'B',
    }
    gh = _FakeGH(
        [
            {
                'repository': {
                    'id': 'R',
                    'discussionCategories': {'nodes': [{'id': 'C1', 'slug': 'general'}]},
                }
            }
        ]
    )
    with pytest.raises(common.DigestError, match='no discussion category'):
        post.create_discussion(gh, 'FormingWorlds/PROTEUS', 'weekly-digest', 'T', 'B')
    assert len(gh.queries) == 1


def test_publish_posts_once_and_has_no_alert_command(tmp_path, monkeypatch):
    """Publish posts the rendered title and body; there is no Discord ping and no alert command."""
    (tmp_path / 'digest.json').write_text(json.dumps({'title': 'T', 'body': 'B'}))
    calls = []
    monkeypatch.setattr(post, 'GitHub', lambda: 'gh')
    monkeypatch.setattr(post, 'previous_state', lambda *a: None)
    monkeypatch.setattr(post, 'create_discussion', lambda *a: calls.append(a) or 'https://d/1')
    assert post.main(['publish', str(tmp_path)]) == 0
    assert calls == [('gh', 'FormingWorlds/PROTEUS', 'weekly-digest', 'T', 'B')]
    assert not hasattr(post, 'discord')
    with pytest.raises(SystemExit):
        post.main(['alert', 'failed'])


def _jobs():
    """The jobs of the workflow as text blocks, read without a YAML parser."""
    text = (_ROOT / '.github' / 'workflows' / 'weekly-digest.yml').read_text()
    parts = re.split(r'^  ([a-z]+):\n', text.split('\njobs:\n', 1)[1], flags=re.M)
    return text, dict(zip(parts[1::2], parts[2::2]))


def test_workflow_jobs_keep_the_model_token_away_from_write_tokens():
    """The draft job alone sees the model token and has no permissions; posting needs the live flag."""
    text, jobs = _jobs()
    assert set(jobs) == {'collect', 'draft', 'publish'}
    assert 'DISCORD' not in text and 'discord' not in text.lower()
    assert '\npermissions: {}\n' in text and 'shell: bash' in text
    assert '    permissions: {}\n' in jobs['draft']
    perms = re.search(r'    permissions:\n((?:      .*\n)+)', jobs['publish']).group(1)
    assert perms == '      contents: read\n      discussions: write\n'
    assert all('write' not in jobs[j] for j in ('collect', 'draft'))
    assert 'github.token' not in jobs['draft'] and 'GITHUB_TOKEN' not in jobs['draft']
    for name, job in jobs.items():
        assert ('CLAUDE_CODE_OAUTH_TOKEN' in job) is (name == 'draft'), name
    assert "if: steps.mode.outputs.live == 'true'" in jobs['publish']
    assert '@anthropic-ai/claude-code@2.1.318' in jobs['draft']


def test_workflow_uploads_the_evidence_of_a_failed_check():
    """The artifact with the check result is uploaded also when the check fails."""
    _, jobs = _jobs()
    upload = jobs['publish'][jobs['publish'].index('upload-artifact') :]
    assert 'if: always()' in upload.split('- name:')[0] and 'check.txt' in upload


def test_collect_joins_repositories_and_records_the_cut():
    """Collect reads each repository once and records how many bodies were kept."""
    settings_ = {
        **_VALID,
        'repos': ['FormingWorlds/JANUS', 'FormingWorlds/MORS'],
        'max_bodies': 1,
    }
    window = {'since': '2026-11-01T00:00:00Z', 'until': '2026-11-18T00:00:00Z'}
    mors = _janus()
    mors['repository']['pullRequests']['nodes'] = []
    mors['repository']['releases']['nodes'] = []
    out = collect.collect(_FakeGH([_janus(), mors]), settings_, window)
    assert [p['ref'] for p in out['prs']] == ['JANUS#105', 'JANUS#104', 'JANUS#103']
    assert (out['bodies_kept'], out['bodies_cut'], out['window']) == (1, 2, window)
    assert all('raw_body' not in p for p in out['prs'])


def test_collect_main_marks_a_dated_run_and_skips_the_state(tmp_path, monkeypatch):
    """With --since the previous post is not read and the window is back-dated."""
    seen = []
    monkeypatch.setattr(collect, 'GitHub', lambda: 'gh')
    monkeypatch.setattr(collect, 'previous_state', lambda *a: seen.append(a))
    monkeypatch.setattr(
        collect, 'collect', lambda gh, s, w: {'window': w, 'prs': [], 'releases': []}
    )
    out = tmp_path / 'c.json'
    assert (
        collect.main(['--out', str(out), '--since', '2026-09-01', '--until', '2026-09-30']) == 0
    )
    window = json.loads(out.read_text())['window']
    assert window == {'since': SINCE, 'until': UNTIL, 'backdated': True, 'previous': None}
    assert seen == []
    monkeypatch.setattr(collect, 'previous_state', lambda *a: {'until': '2026-11-10T00:00:00Z'})
    assert collect.main(['--out', str(out), '--until', '2026-11-17']) == 0
    window = json.loads(out.read_text())['window']
    assert (
        window['previous'] == {'until': '2026-11-10T00:00:00Z'}
        and window['since'] == '2026-11-10T00:00:00Z'
    )


def test_known_keys_cover_every_tracked_file_kind():
    """Config paths, helpfile columns and default-file keys with _ or . are known keys."""
    gh = _FakeGH(
        files={
            ('a/P', facts.OUTPUT_SCHEMA, None): json.dumps(
                {'keys': [{'name': 'T_surf'}, {'name': 'Time'}]}
            ),
            ('a/Z', 'input/default.toml', None): '[eos]\nmushy = 1\n',
        }
    )
    keys = facts.known_keys(
        gh, {'tracked': {'a/P': [facts.OUTPUT_SCHEMA], 'a/Z': ['input/default.toml']}}
    )
    assert keys == {'T_surf', 'eos.mushy'}


def test_facts_main_rewrites_the_file(tmp_path, monkeypatch):
    """The command adds facts and known keys to the collected file in place."""
    path = tmp_path / 'c.json'
    path.write_text(json.dumps({'prs': []}))
    monkeypatch.setattr(facts, 'GitHub', lambda: _FakeGH())
    monkeypatch.setattr(facts, 'load_settings', lambda: {'tracked': {}})
    assert facts.main([str(path)]) == 0
    assert json.loads(path.read_text()) == {'prs': [], 'facts': [], 'known_keys': []}


# markdown and autolink forms that must not reach a post


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        ({'text': '[x]: //evil.example/phish'}, 'text holds a markdown link'),
        ({'title': 'Grain size [details][x]'}, 'text holds a markdown link'),
        ({'text': 'See //evil.example/x for more.'}, 'text holds a URL'),
        ({'text': 'Write to mailto:x for more.'}, 'text holds a URL'),
        ({'text': 'This follows GH-12 and GH-13.'}, 'text holds a # reference'),
        ({'text': 'It reverts commit 1a2b3c4d5e6f.'}, 'text holds a commit id'),
        ({'text': 'Thanks &#x40;alice and &commat;bob.'}, 'text holds an HTML entity'),
        ({'text': 'Thanks ＠alice.'}, 'text holds an @'),
        (
            {'text': 'The grain size default is now -0.001 m.'},
            'number -0.001 is in no cited source',
        ),
        ({'text': 'Now 1,000,000 grains.'}, 'number 1,000,000 is in no cited source'),
        ({'text': 'Now 0.1⁻³ m.'}, 'text holds a superscript'),
        (
            {'text': 'The Interior_Energetics.RFRONT_LOC moved.'},
            'key interior_energetics.rfront_loc',
        ),
        ({'text': 'The energetics.rfront_loc moved.'}, 'key energetics.rfront_loc'),
        ({'section': 'bogus'}, 'unknown section bogus'),
    ],
)
def test_check_rejects_more_link_and_value_forms(change, message):
    """Reference links, autolinks, entities, look-alikes, signs, groups and partial keys fail."""
    fails = check.check(_data(), _draft(_entry(**change)))
    assert any(message in f for f in fails), fails


def test_check_rejects_a_string_value_of_a_cited_fact():
    """Text may not repeat the old string value of a cited change, so it cannot state the change backwards."""
    fact = _fact(key='interior_energetics.mixing_length', old='"nearest"', new='"constant"')
    data = _data(fact_list=[fact])
    data['known_keys'].append('interior_energetics.mixing_length')
    fails = check.check(data, _draft(_entry(text='The default mixing length is now nearest.')))
    assert any('old value "nearest" of fact F1' in f for f in fails), fails
    assert (
        check.check(data, _draft(_entry(text='The default mixing length is now constant.')))
        == []
    )
    assert (
        check.check(data, _draft(_entry(text='The default mixing length rule changed.'))) == []
    )


@pytest.mark.parametrize(
    'field', ['section', 'title', 'text', 'prs', 'releases', 'facts', 'logins']
)
def test_check_reports_a_missing_or_mistyped_field(field):
    """A malformed entry is a failure, not a crash, so the repair round can run."""
    entry = _entry()
    del entry[field]
    assert any(f'field {field}' in f for f in check.check(_data(), _draft(entry)))
    bad = _entry(**{field: 5})
    assert any(f'field {field}' in f for f in check.check(_data(), _draft(bad)))


def test_render_makes_model_text_inert():
    """Markdown in model text is escaped, so it cannot form a link, a heading or a table cell."""
    entry = _entry(
        section='capabilities',
        facts=[],
        title='Lid [details][x] | cell',
        text='# Big <b>news</b> *now* [x]: /evil and `x`',
    )
    body, _ = _render(_draft(entry))
    line = next(x for x in body.splitlines() if x.startswith('### '))
    assert line == '### Lid \\[details\\]\\[x\\] / cell'
    assert '\\# Big \\<b\\>news\\</b\\> \\*now\\* \\[x\\]: /evil and \\`x\\`' in body


def test_fact_values_cannot_break_a_row_or_spoof_the_state():
    """A carriage return or a state marker in a fact value is neutralised."""
    assert render.value('3000.0\r| injected') == '`3000.0   injected`'
    spoof = render.value('<!-- weekly-digest-state {"until":"2099-01-01"} -->')
    assert '<!--' not in spoof and '-->' not in spoof
    assert render.value('>=26.11.8') == '`>=26.11.8`' and render.value('<2') == '`<2`'


def test_state_marker_is_the_last_one_in_the_body():
    """Only the marker at the end of a post counts, so a marker in the content cannot steer the window."""
    body = 'x <!-- weekly-digest-state {"until":"2099-01-01T00:00:00Z"} --> y\n<!-- weekly-digest-state {"until":"2026-11-16T22:30:00Z"} -->\n'
    assert collect.parse_state(body) == {'until': '2026-11-16T22:30:00Z'}
    assert collect.parse_state('no marker') is None


def test_release_tag_cannot_form_a_link():
    """A release tag with brackets or a pipe is cleaned in the link text."""
    rel = dict(_release(), ref='AGNI@1.0](/x)|y')
    assert (
        render.links([rel])
        == '[AGNI 1.0\\]\\(/x\\)/y](https://github.com/nichollsh/AGNI/releases/tag/1.13.0)'
    )


def test_merged_pr_announced_as_open_before_is_mentioned():
    """The open-PR rule suppresses a mention only while the PR is still open."""
    previous = {'until': SINCE, 'announced_open': ['PROTEUS#849']}
    body, _ = _render(_draft(_entry()), _data(previous=previous), live=True)
    assert '@alice' in body


def test_collect_drops_prs_merged_into_another_branch():
    """A pull request merged into a feature branch is not a merge of the repository."""
    page = _janus()
    for node in page['repository']['pullRequests']['nodes']:
        node['baseRefName'] = 'feature'
    page['repository']['defaultBranchRef'] = {'name': 'main'}
    window = {'since': '2026-11-01T00:00:00Z', 'until': '2026-11-18T00:00:00Z'}
    prs, _ = collect.collect_repo(_FakeGH([page]), 'FormingWorlds/JANUS', window)
    assert prs == []


def test_github_retries_a_network_error_and_rejects_a_missing_graphql_answer(monkeypatch):
    """A reset connection is retried; a GraphQL 404 is a digest error, not a TypeError."""
    import urllib.error

    calls = []

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError('reset')
        return _Resp(b'ok')

    monkeypatch.setattr(common.urllib.request, 'urlopen', flaky)
    monkeypatch.setattr(common.time, 'sleep', lambda s: None)
    assert common.GitHub('t').raw_file('a/b', 'f', None) == 'ok'
    monkeypatch.setattr(
        common.urllib.request,
        'urlopen',
        lambda r, timeout: (_ for _ in ()).throw(_http_error(404)),
    )
    with pytest.raises(common.DigestError, match='GraphQL'):
        common.GitHub('t').graphql('{x}')


def test_known_keys_fail_when_a_tracked_file_is_missing():
    """A schema that cannot be read stops the run instead of turning the key check off."""
    with pytest.raises(common.DigestError, match='cannot read'):
        facts.known_keys(_FakeGH(), {'tracked': {'a/P': [facts.CONFIG_SCHEMA]}})


def test_publish_skips_a_window_that_is_already_posted(tmp_path, monkeypatch, capsys):
    """A rerun after a post finds the post of the same window and does not post again."""
    body = (
        'x\n<!-- weekly-digest-state {"until":"2026-11-23T22:23:00Z","announced_open":[]} -->\n'
    )
    (tmp_path / 'digest.json').write_text(json.dumps({'title': 'T', 'body': body}))
    monkeypatch.setattr(post, 'GitHub', lambda: 'gh')
    monkeypatch.setattr(post, 'previous_state', lambda *a: {'until': '2026-11-23T22:23:00Z'})
    monkeypatch.setattr(post, 'create_discussion', lambda *a: pytest.fail('posted twice'))
    post.main(['publish', str(tmp_path)])
    assert 'already posted' in capsys.readouterr().out


def test_model_runs_outside_the_repository(tmp_path, monkeypatch):
    """The CLI starts in an empty folder, so no project instruction file joins the prompt."""
    script, _ = _fake_cli(tmp_path, [{'structured_output': {'entries': []}}])
    log = tmp_path / 'cwd.txt'
    script.write_text(
        script.read_text().replace(
            'import json, sys, pathlib\n',
            f'import json, sys, pathlib, os\npathlib.Path({str(log)!r}).write_text(os.getcwd())\n',
        )
    )
    monkeypatch.setenv('CLAUDE_BIN', str(script))
    draft.run_model('p', 'm')
    cwd = Path(log.read_text())
    assert cwd != _ROOT and not (cwd / 'CLAUDE.md').exists() and _ROOT not in cwd.parents


def test_previous_state_skips_a_malformed_marker_of_a_person():
    """A broken marker in a post by anybody else neither crashes the run nor counts."""
    cats = {
        'repository': {
            'id': 'R',
            'discussionCategories': {'nodes': [{'id': 'C1', 'slug': 'weekly-digest'}]},
        }
    }
    bad = _post(author='mallory', kind='User', state='{oops}')
    posts = {'repository': {'discussions': {'nodes': [bad, _post()]}}}
    state = collect.previous_state(
        _FakeGH([cats, posts]), 'FormingWorlds/PROTEUS', 'weekly-digest'
    )
    assert state['until'] == '2026-11-16T22:30:00Z'
    assert collect.parse_state('<!-- weekly-digest-state {oops} -->') is None
    assert collect.parse_state('<!-- weekly-digest-state ["x"] -->') is None


def test_check_rejects_a_numeric_value_stated_backwards():
    """The old number of a cited change is no source, so the text cannot state it as the new value."""
    data = _data()
    data['prs'][0]['body'] = 'Settling velocity scales with grain radius squared.'
    fails = check.check(data, _draft(_entry(text='The default grain size is now 0.1 m.')))
    assert any('number 0.1 is in no cited source' in f for f in fails), fails
    fails = check.check(data, _draft(_entry(text='The default grain size is now −0.001 m.')))
    assert any('number -0.001 is in no cited source' in f for f in fails), fails


def test_check_accepts_subscripts_and_rejects_uppercase_commit_ids():
    """Chemical formulas with subscripts pass; a commit id in capitals still fails."""
    assert check.check(_data(), _draft(_entry(text='Outgassed CO₂ changed.'))) == []
    fails = check.check(_data(), _draft(_entry(text='It reverts 1A2B3C4D5E6F.')))
    assert any('commit id' in f for f in fails), fails


def test_mutation_is_sent_once_on_a_network_error(monkeypatch):
    """A create call is not retried, so a lost answer cannot post the digest twice."""
    calls = []

    def flaky(req, timeout):
        calls.append(1)
        raise TimeoutError('read timeout')

    monkeypatch.setattr(common.urllib.request, 'urlopen', flaky)
    monkeypatch.setattr(common.time, 'sleep', lambda s: None)
    with pytest.raises(common.DigestError):
        common.GitHub('t').graphql('mutation {x}', retry=False)
    assert len(calls) == 1


def test_render_escapes_dollar_and_release_url():
    """A dollar sign cannot start math, and a release URL cannot close its link early."""
    assert render.escape('costs $5 and $6') == 'costs \\$5 and \\$6'
    rel = dict(_release(), url='https://github.com/a/b/releases/tag/1.0)|x y')
    assert render.links([rel]).endswith('(https://github.com/a/b/releases/tag/1.0%29%7Cx%20y)')


@pytest.mark.parametrize(
    ('text', 'message'),
    [
        ('Runs take 1​000 steps.', 'an invisible character'),
        ('Also rfront​_loc changed.', 'an invisible character'),
        ('Shown as ‮1.0‬ now.', 'an invisible character'),
        ('Half is ½ now.', 'a special number character'),
        ('Now ‐0.001 m.', 'number -0.001 is in no cited source'),
        ('Now ˗0.001 m.', 'number -0.001 is in no cited source'),
        ('Runs are x100 faster.', 'number 100 is in no cited source'),
        ('Grain size now r0.5 m.', 'number 0.5 is in no cited source'),
    ],
)
def test_check_rejects_hidden_characters_dashes_and_prefixed_numbers(text, message):
    """Invisible characters, fraction characters, dash-like signs and letter-prefixed numbers fail."""
    fails = check.check(_data(), _draft(_entry(text=text)))
    assert any(message in f for f in fails), fails


def test_number_signs_match_in_text_and_source():
    """A Unicode minus in a source matches an ASCII minus in the text, and formulas give no number."""
    assert check.numbers('−0.8 and H2O and CO₂') == {'-0.8'}
    data = _data()
    data['prs'][0]['body'] = 'The offset is now −0.8 dex.'
    assert check.check(data, _draft(_entry(text='The offset is now -0.8 dex.'))) == []


def test_quiet_week_note_is_a_paragraph():
    """The quiet-week sentence is followed by a blank line, so the rule below cannot make it a heading."""
    body, _ = _render(_draft(empty=True), _data(prs=[], fact_list=[]))
    assert '2026-09-30T00:00:00Z).\n\n---' in body


def test_publish_skips_a_second_post_with_the_same_title(tmp_path, monkeypatch, capsys):
    """A full rerun collects a new short window; the same title as the newest post still stops it."""
    body = (
        'x\n<!-- weekly-digest-state {"until":"2026-11-23T22:30:00Z","announced_open":[]} -->\n'
    )
    (tmp_path / 'digest.json').write_text(
        json.dumps({'title': 'PROTEUS weekly digest, 24 November 2026', 'body': body})
    )
    monkeypatch.setattr(post, 'GitHub', lambda: 'gh')
    last = {'until': '2026-11-23T22:23:00Z', 'title': 'PROTEUS weekly digest, 24 November 2026'}
    monkeypatch.setattr(post, 'previous_state', lambda *a: last)
    monkeypatch.setattr(post, 'create_discussion', lambda *a: pytest.fail('posted twice'))
    post.main(['publish', str(tmp_path)])
    assert 'already posted' in capsys.readouterr().out


def test_http_protocol_errors_are_handled(monkeypatch, capsys):
    """A broken HTTP answer from GitHub is a digest error."""
    import http.client

    monkeypatch.setattr(
        common.urllib.request,
        'urlopen',
        lambda r, timeout: (_ for _ in ()).throw(http.client.IncompleteRead(b'')),
    )
    monkeypatch.setattr(common.time, 'sleep', lambda s: None)
    with pytest.raises(common.DigestError):
        common.GitHub('t').graphql('{x}')


@pytest.mark.parametrize('marker', ['{"until":1}', '{"x":1}', '{"until":"garbage"}', '{}'])
def test_state_without_a_valid_end_is_ignored(marker):
    """A marker whose window end is missing or not a time counts as no marker."""
    good = '<!-- weekly-digest-state {"until":"2026-11-16T22:30:00Z"} -->'
    assert collect.parse_state(f'{good}\n<!-- weekly-digest-state {marker} -->') is None
    assert collect.parse_state(f'<!-- weekly-digest-state {marker} -->\n{good}') == {
        'until': '2026-11-16T22:30:00Z'
    }


def test_previous_state_passes_over_a_broken_bot_marker():
    """A bot post with a broken marker is skipped and the next bot post gives the state."""
    cats = {
        'repository': {
            'id': 'R',
            'discussionCategories': {'nodes': [{'id': 'C1', 'slug': 'weekly-digest'}]},
        }
    }
    posts = {'repository': {'discussions': {'nodes': [_post(state='{oops}'), _post()]}}}
    state = collect.previous_state(
        _FakeGH([cats, posts]), 'FormingWorlds/PROTEUS', 'weekly-digest'
    )
    assert state['until'] == '2026-11-16T22:30:00Z'


def test_old_value_as_an_ordinary_word_passes():
    """The old value fails only in a sentence that states a new state without the new value."""
    fact = _fact(key='interior_energetics.mixing_length', old='"nearest"', new='"constant"')
    data = _data(fact_list=[fact])
    real = (
        'Aragog and SPIDER now default to a mixing length that is a fixed fraction of the mantle depth. '
        'The other form, based on the distance to the nearest boundary, let the core heat flux stay near zero.'
    )
    assert check.check(data, _draft(_entry(text=real))) == []
    assert check.check(data, _draft(_entry(text='It changed from nearest to constant.'))) == []
    assert (
        check.check(data, _draft(_entry(text='Defaults are now nearest; see the guide.'))) != []
    )


def test_numeric_old_value_in_a_now_sentence_fails():
    """A number that is the old value of a cited change fails in a sentence that says what it is now."""
    fact = _fact(key='interior_energetics.grain_size', old='100', new='200')
    data = _data(fact_list=[fact])
    data['prs'][0]['body'] = 'The cell count went from 100 to 200.'
    fails = check.check(data, _draft(_entry(text='The default is now 100 cells.')))
    assert any('old value "100" of fact F1' in f for f in fails), fails
    assert (
        check.check(data, _draft(_entry(text='The default rose from 100 to 200 cells.'))) == []
    )


def test_check_survivor_cases():
    """Close cases: a subscript minus, a near number, a non-string list item, a value change, a 3-letter value."""
    fails = check.check(_data(), _draft(_entry(text='Now ₋0.001 m.')))
    assert any('number -0.001 is in no cited source' in f for f in fails), fails
    fails = check.check(_data(), _draft(_entry(text='The grain size is now 0.105 m.')))
    assert any('number 0.105 is in no cited source' in f for f in fails), fails
    assert any('field prs' in f for f in check.check(_data(), _draft(_entry(prs=[849]))))
    fact = {**_fact(key='planet.mode', old='"off"', new='"auto"'), 'change': 'value'}
    fails = check.check(_data(fact_list=[fact]), _draft(_entry(text='The mode is now off.')))
    assert any('old value "off" of fact F1' in f for f in fails), fails


def test_previous_state_ignores_another_bot():
    """A marker in a post by a bot other than the workflow does not count."""
    cats = {
        'repository': {
            'id': 'R',
            'discussionCategories': {'nodes': [{'id': 'C1', 'slug': 'weekly-digest'}]},
        }
    }
    other = _post(author='dependabot', state='{"until":"2020-01-01T00:00:00Z"}')
    posts = {'repository': {'discussions': {'nodes': [other, _post()]}}}
    state = collect.previous_state(
        _FakeGH([cats, posts]), 'FormingWorlds/PROTEUS', 'weekly-digest'
    )
    assert state['until'] == '2026-11-16T22:30:00Z'


def test_fact_line_falsy_default_value_change_and_cap_edge():
    """A falsy new default is printed, a value change reads like a default, and 4 lines need no count."""
    added = {**_fact(), 'change': 'added', 'old': None, 'new': 0}
    assert render.fact_line(added) == '`interior_energetics.grain_size` added, default `0`'
    assert (
        render.fact_line({**_fact(), 'change': 'value'})
        == '`interior_energetics.grain_size`: `0.1` → `0.001`'
    )
    four = [_fact(f'F{i}', key=f'k{i}') for i in range(render.MAX_FACTS)]
    assert 'more' not in render.facts_text(four) and render.facts_text(four).count(';') == 3


def test_short_numeric_old_value_in_a_now_sentence_fails():
    """A one- or two-digit old value is checked like a longer one."""
    fact = _fact(key='interior_energetics.grain_size', old='10', new='20')
    data = _data(fact_list=[fact])
    data['prs'][0]['body'] = 'The count went from 10 to 20.'
    fails = check.check(data, _draft(_entry(text='The default is now 10 cells.')))
    assert any('old value "10" of fact F1' in f for f in fails), fails


def test_render_follows_the_highlights_layout():
    """Sections, lede, note line and table follow the September highlights page."""
    data = _data(prs=[_pr(), _pr('PROTEUS#900', status='open', author='carol')])
    entries = [
        _entry(
            section='open',
            prs=['PROTEUS#900'],
            facts=[],
            logins=['carol'],
            title='O',
            text='Needs review.',
        ),
        _entry(section='design', facts=[], title='D'),
        _entry(section='install', facts=[], title='I', action=True, text='Update Julia.'),
        _entry(section='install', facts=[], title='J', action=False, text='Faster install.'),
        _entry(section='data', facts=[], title='E'),
        _entry(section='capabilities', facts=[], title='C'),
        _entry(),
    ]
    body, _ = _render(_draft(*entries), data)
    heads = [x for x in body.splitlines() if x.startswith('## ')]
    assert heads == [
        '## Changes that affect your results',
        '## New capabilities',
        '## Installing and running',
        '## Data and licensing',
        '## Design choices to look at',
        '## In review, coming soon',
    ]
    paragraphs = body.split('\n\n')
    lede = 'Additions and changes from across the ecosystem that affect more than one project. This is a selection, not a complete change list.'
    assert paragraphs[1] == lede and paragraphs[2].startswith(
        'This digest is written automatically'
    )
    assert (
        'Runs started after these merges will not reproduce earlier runs with the same config. Compare old and new runs with care, and check whether any figures in progress need a rerun.'
        in body
    )
    assert '| Setting | Change | Merged | Links |\n| --- | --- | --- | --- |' in body
    assert (
        body.count('Action for everyone') == 1
        and '**Action for everyone:** Update Julia.' in body
    )


@pytest.mark.parametrize(
    ('since', 'until', 'days'),
    [
        ('2026-09-01T00:00:00Z', '2026-09-30T00:00:00Z', '1 and 29 September 2026'),
        ('2026-09-24T07:00:00Z', '2026-10-06T07:00:00Z', '24 September and 6 October 2026'),
        ('2025-12-30T00:00:00Z', '2026-01-02T00:00:00Z', '30 December 2025 and 1 January 2026'),
        ('2026-10-06T00:00:00Z', '2026-10-06T00:00:00Z', '6 October 2026'),
        ('2026-10-06T07:00:00Z', '2026-10-06T09:00:00Z', '6 October 2026'),
    ],
)
def test_window_dates(since, until, days):
    """The footer names the first and last day of the window in the form of the highlights page."""
    assert (
        render.window_dates({'since': since, 'until': until})
        == f'{days} (UTC {since} to {until})'
    )


def test_quiet_week_uses_readable_dates():
    """The quiet-week line names the days like the footer."""
    body, _ = _render(_draft(empty=True), _data(prs=[], fact_list=[]))
    assert 'between 1 and 29 September 2026 (UTC' in body.split('---')[0]


def test_old_draft_without_action_renders():
    """A draft made before the action field passes the check and renders with no marker."""
    entry = _entry(section='install', facts=[])
    assert 'action' not in entry
    assert check.check(_data(), _draft(entry)) == []
    body, _ = _render(_draft(entry))
    assert 'Action for everyone' not in body


@pytest.mark.parametrize('value', [None, 1, 0, 'true', [True]])
def test_check_action_must_be_boolean(value):
    """A non-boolean action marker is a shape failure."""
    assert any('field action' in f for f in check.check(_data(), _draft(_entry(action=value))))


@pytest.mark.parametrize(
    ('change', 'message'),
    [
        ({'action': True}, 'action is only for section install'),
        (
            {
                'section': 'open',
                'prs': ['PROTEUS#900'],
                'facts': [],
                'logins': ['carol'],
                'action': True,
            },
            'action is only for section install',
        ),
        (
            {
                'section': 'install',
                'facts': [],
                'action': True,
                'text': 'Action for everyone: update Julia.',
            },
            'text holds the action marker',
        ),
        (
            {'section': 'capabilities', 'facts': [], 'text': 'Action for everyone: reinstall.'},
            'text holds the action marker',
        ),
    ],
)
def test_check_action_rules(change, message):
    """The marker comes only from the field, and the field only in Installing and running."""
    data = _data(prs=[_pr(), _pr('PROTEUS#900', status='open', author='carol')])
    fails = check.check(data, _draft(_entry(**change)))
    assert any(message in f for f in fails), fails
    assert check.check(data, _draft(_entry(section='install', facts=[], action=True))) == []
    assert check.check(data, _draft(_entry(action=False))) == []


# guards, dedupe and text rules


def test_check_output_is_inert_in_the_log_and_the_summary(tmp_path, capsys):
    """Model text in a failure line is one JSON string; the summary is an escaped pre block."""
    src, bad = tmp_path / 'c.json', tmp_path / 'd.json'
    src.write_text(json.dumps(_data()))
    title = '[w](https://evil.example) </pre><b>x</b> @egpbos'
    bad.write_text(json.dumps(_draft(_entry(title=title, logins=['x\n::warning::injected']))))
    out, summary = tmp_path / 'check.txt', tmp_path / 'summary.md'
    summary.write_text('before\n')
    assert check.main([str(src), str(bad), '--out', str(out), '--summary', str(summary)]) == 1
    printed = capsys.readouterr().out
    assert printed == out.read_text()
    lines = printed.splitlines()
    assert all(line.startswith('FAIL "') for line in lines[:-1]), lines
    assert not any(line.startswith('::') for line in lines)
    text = summary.read_text()
    assert text.startswith('before\n<pre>\n') and text.endswith('</pre>\n')
    assert text.count('<') == 2 and '&lt;/pre&gt;' in text


def test_model_error_text_is_one_json_string(tmp_path, monkeypatch):
    """The CLI error tail cannot start a workflow command line in the log."""
    script = tmp_path / 'claude'
    script.write_text(
        f'#!{sys.executable}\nimport sys\nsys.stderr.write("a\\n::warning::x")\nsys.exit(2)\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv('CLAUDE_BIN', str(script))
    with pytest.raises(common.DigestError) as err:
        draft.run_model('p', 'm')
    assert '\n' not in str(err.value) and '"a\\n::warning::x"' in str(err.value)


def _publish(tmp_path, monkeypatch, last, until, run_id='9', entry=None, **window):
    """Render a digest for ``until`` and run publish against a newest post ``last``."""
    data = _data()
    data['window'].update(until=until, backdated=False, **window)
    src, dft, out = tmp_path / 'c.json', tmp_path / 'd.json', tmp_path / 'out'
    src.write_text(json.dumps(data))
    dft.write_text(json.dumps(_draft(entry or _entry())))
    render.main([str(src), str(dft), str(out), '--run-id', run_id])
    posted = []
    monkeypatch.setattr(post, 'load_settings', lambda: {**_VALID, 'publish': True})
    monkeypatch.setattr(post, 'GitHub', lambda: 'gh')
    monkeypatch.setattr(post, 'previous_state', lambda *a: last)
    monkeypatch.setattr(
        post, 'create_discussion', lambda *a: posted.append(a[3]) or 'https://d/2'
    )
    return post.main(['publish', str(src), str(dft), str(out)]), posted


_MON = '2026-11-23T22:23:00Z'


def test_publish_refuses_a_failed_check_a_backdated_window_or_publish_off(
    tmp_path, monkeypatch
):
    """The post command repeats the guards, so a broken shell step cannot post."""
    with pytest.raises(common.DigestError, match='check'):
        _publish(tmp_path, monkeypatch, None, _MON, entry=_entry(prs=['PROTEUS#1']))
    with pytest.raises(common.DigestError, match='back-dated'):
        _publish(tmp_path, monkeypatch, None, _MON, backdated=True)
    rc, posted = _publish(tmp_path, monkeypatch, None, _MON)
    assert rc == 0 and posted == ['PROTEUS weekly digest, 24 November 2026']
    monkeypatch.setattr(post, 'load_settings', lambda: _VALID)
    with pytest.raises(common.DigestError, match='publish'):
        post.main(['publish', *(str(tmp_path / f) for f in ('c.json', 'd.json', 'out'))])


@pytest.mark.parametrize(
    ('case', 'last', 'until', 'run_id', 'posts'),
    [
        # (a) a re-run of all jobs on a later day keeps the run id
        (
            'rerun',
            {'run_id': '7', 'until': _MON, 'title': 'x'},
            '2026-12-02T09:00:00Z',
            '7',
            False,
        ),
        # (b) a Wednesday recovery, then the Monday schedule
        (
            'recovery',
            {
                'run_id': '8',
                'until': '2026-11-25T10:00:00Z',
                'title': 'PROTEUS weekly digest, 24 November 2026',
            },
            '2026-11-30T22:23:00Z',
            '9',
            True,
        ),
        # (c) 2 dispatches on one day
        (
            'twice',
            {
                'run_id': '8',
                'until': '2026-11-25T10:00:00Z',
                'title': 'PROTEUS weekly digest, 24 November 2026',
            },
            '2026-11-25T15:00:00Z',
            '10',
            False,
        ),
        # (e) a Monday daytime dispatch, then the Monday schedule
        (
            'monday',
            {
                'run_id': '8',
                'until': '2026-11-23T10:00:00Z',
                'title': 'PROTEUS weekly digest, 24 November 2026',
            },
            _MON,
            '9',
            False,
        ),
    ],
)
def test_publish_dedupe_cases(tmp_path, monkeypatch, capsys, case, last, until, run_id, posts):
    """One post per meeting and per run; a skip says why in an annotation and exits 0."""
    rc, posted = _publish(tmp_path, monkeypatch, last, until, run_id)
    assert rc == 0 and bool(posted) is posts, case
    if not posts:
        assert '::notice' in capsys.readouterr().out


def test_monday_dispatch_window_continues_at_its_end():
    """After a skipped schedule run the next window starts at the end of the dispatch."""
    state = {'until': '2026-11-23T10:00:00Z'}
    now = datetime(2026, 11, 30, 22, 23, tzinfo=timezone.utc)
    assert collect.resolve_window(None, None, state, now)['since'] == state['until']


@pytest.mark.parametrize(
    ('until', 'day'),
    [
        ('2026-11-23T22:23:00Z', '24 November'),  # Monday 23:23 CET
        ('2026-06-29T22:23:00Z', '30 June'),  # Tuesday 00:23 CEST
        ('2026-11-25T09:00:00Z', '24 November'),  # Wednesday recovery
        ('2026-11-29T20:00:00Z', '24 November'),  # Sunday
        ('2026-11-30T10:00:00Z', '1 December'),  # Monday daytime
    ],
)
def test_title_is_the_meeting_tuesday(until, day):
    """The title is the Tuesday on or before the day after the window end, in Amsterdam time."""
    assert render.tuesday_title(until) == f'PROTEUS weekly digest, {day} {until[:4]}'


def test_state_marker_holds_the_run_id(tmp_path):
    """The run id goes into the state marker, so a re-run finds its own post."""
    src, dft, out = tmp_path / 'c.json', tmp_path / 'd.json', tmp_path / 'out'
    src.write_text(json.dumps(_data()))
    dft.write_text(json.dumps(_draft(_entry())))
    render.main([str(src), str(dft), str(out), '--run-id', '42'])
    body = json.loads((out / 'digest.json').read_text())['body']
    assert collect.parse_state(body)['run_id'] == '42'


def test_workflow_wiring_of_the_guards():
    """The check step fails the job, the decide step reads the window, and post repeats the guards."""
    text, jobs = _jobs()
    assert 'continue-on-error' not in text and '||' not in jobs['publish']
    assert 'needs: draft' in jobs['publish'] and 'needs: collect' in jobs['draft']
    assert (
        'run: python tools/digest/check.py collected.json draft.json --out check.txt '
        '--summary "$GITHUB_STEP_SUMMARY"\n' in jobs['publish']
    )
    assert (
        'post.py decide collected.json --event "$EVENT" --post-input "$POST"' in jobs['publish']
    )
    assert 'post.py publish collected.json draft.json out\n' in jobs['publish']
    assert '--run-id "$GITHUB_RUN_ID"' in jobs['publish']


def test_announced_open_pr_stays_until_merged_or_closed():
    """An announced pull request stays in the state while GitHub reports it open."""
    pages = [
        {'repository': {'pullRequest': {'state': 'OPEN'}}},
        {'repository': {'pullRequest': {'state': 'MERGED'}}},
        {'repository': {'pullRequest': {'state': 'CLOSED'}}},
    ]
    refs = ['PROTEUS#900', 'AGNI#5', 'PROTEUS#901']
    repos = ['FormingWorlds/PROTEUS', 'nichollsh/AGNI']
    gh = _FakeGH(pages)
    assert collect.still_open(gh, repos, refs) == ['PROTEUS#900']
    assert [q['number'] for q in gh.queries] == [900, 5, 901]
    data = _data(previous={'until': SINCE, 'announced_open': ['PROTEUS#900']})
    data['still_open'] = ['PROTEUS#900']
    body, state = _render(_draft(_entry()), data)
    assert state['announced_open'] == ['PROTEUS#900']
    data['still_open'] = []
    assert _render(_draft(_entry()), data)[1]['announced_open'] == []


@pytest.mark.parametrize(
    ('text', 'found'),
    [
        ('Runs are X100 faster.', {'100'}),
        ('Use V9.5 now.', {'9.5'}),
        ('Runs use mode_42 cells.', {'42'}),
        ('Water as H2O and CO2, and MgSiO3.', set()),
        ('Escape of H2O.', set()),
        ('v1.0.0 and x100', {'1.0.0', '100'}),
    ],
)
def test_numbers_split_a_capital_or_underscore_prefix_but_keep_formulas(text, found):
    """A digit run after a single capital or an underscore is a number; a formula is not."""
    assert check.numbers(text) == found


@pytest.mark.parametrize(
    'text',
    [
        'The grain size was lowered to 0.1 m.',
        'Grain size is set to 0.1 m.',
        'Grain size becomes 0.1 m.',
    ],
)
def test_old_value_stated_as_the_setting_fails(text):
    """The old value of a cited change fails where the sentence assigns it."""
    fails = check.check(_data(), _draft(_entry(title='Grain size', text=text)))
    assert any('old value "0.1" of fact F1' in f for f in fails), fails


def test_a_third_string_value_and_title_sentences():
    """A sentence that sets a string key to another value fails; the title is its own sentence."""
    fact = _fact(key='interior_energetics.mixing_length', old='"nearest"', new='"constant"')
    data = _data(fact_list=[fact])
    fails = check.check(data, _draft(_entry(text='The default mixing length is now linear.')))
    assert any('new value "constant" of fact F1' in f for f in fails), fails
    real = (
        'It uses a fixed fraction of the depth instead of the distance to the nearest boundary.'
    )
    assert check.check(data, _draft(_entry(text=real))) == []
    entry = _entry(title='Slower settling by default', text='Settling is slower with 0.1 m.')
    assert check.check(_data(), _draft(entry)) == []


def test_truncated_file_list_diffs_every_tracked_file():
    """A pull request with more files than the query returns has every tracked file compared."""
    node = _node(merged='2026-09-05T00:00:00Z')
    node['files'] = {'totalCount': 250, 'nodes': [{'path': 'src/a.py'}]}
    record = collect.pr_record('FormingWorlds/PROTEUS', node, SINCE, UNTIL)
    assert record['files_truncated'] is True
    old, new = _schema(a={'default': 1}), _schema(a={'default': 2})
    files = {
        ('FormingWorlds/PROTEUS', facts.CONFIG_SCHEMA, 'p1'): old,
        ('FormingWorlds/PROTEUS', facts.CONFIG_SCHEMA, 'm1'): new,
        ('FormingWorlds/PROTEUS', facts.CONFIG_SCHEMA, None): new,
    }
    data = {'prs': [_pr(files=['src/a.py'], files_truncated=True)]}
    settings_ = {'tracked': {'FormingWorlds/PROTEUS': [facts.CONFIG_SCHEMA]}}
    facts.attach_facts(_FakeGH(files=files), data, settings_)
    assert [f['key'] for f in data['facts']] == ['a']


_CATS1 = {
    'repository': {
        'id': 'R',
        'discussionCategories': {'nodes': [{'id': 'C1', 'slug': 'weekly-digest'}]},
    }
}


def test_previous_state_pages_until_a_bot_post():
    """Posts by people on the first page do not hide the newest bot post on a later page."""
    people = [_post(author='p', kind='User') for _ in range(3)]
    first = {
        'repository': {
            'discussions': {
                'nodes': people,
                'pageInfo': {'hasNextPage': True, 'endCursor': 'c1'},
            }
        }
    }
    second = {
        'repository': {
            'discussions': {
                'nodes': [_post()],
                'pageInfo': {'hasNextPage': False, 'endCursor': None},
            }
        }
    }
    gh = _FakeGH([_CATS1, first, second])
    state = collect.previous_state(gh, 'FormingWorlds/PROTEUS', 'weekly-digest')
    assert state['until'] == '2026-11-16T22:30:00Z' and gh.queries[2]['cursor'] == 'c1'


def test_create_discussion_refuses_an_author_the_next_run_would_not_trust():
    """The author of the new post is read back; a post the state reader would skip stops the run."""
    made = {
        'createDiscussion': {
            'discussion': {'url': 'https://d/5', 'author': {'login': 'x', '__typename': 'User'}}
        }
    }
    with pytest.raises(common.DigestError, match='https://d/5'):
        post.create_discussion(
            _FakeGH([_CATS1, made]), 'FormingWorlds/PROTEUS', 'weekly-digest', 'T', 'B'
        )


def test_check_enforces_the_schema_lengths():
    """A title or text longer than the schema allows fails the check."""
    fails = check.check(_data(), _draft(_entry(title='a' * 81, text='b ' * 201)))
    assert any('title is longer than 80' in f for f in fails), fails
    assert any('text is longer than 400' in f for f in fails), fails


def test_render_refuses_a_body_over_the_post_limit(tmp_path, monkeypatch):
    """A body longer than a discussion can hold stops the run before the post step."""
    monkeypatch.setattr(render, 'BODY_LIMIT', 100)
    src, dft = tmp_path / 'c.json', tmp_path / 'd.json'
    src.write_text(json.dumps(_data()))
    dft.write_text(json.dumps(_draft(_entry())))
    with pytest.raises(common.DigestError, match='longer than'):
        render.main([str(src), str(dft), str(tmp_path / 'out')])


def test_footer_without_a_cut_says_every_body_was_read():
    """With no pull request cut, the footer does not count the titles of 0 others."""
    body, _ = _render(_draft(_entry()))
    assert 'the other 0' not in body and 'read the text of every pull request' in body
