"""Link people to the components and call sites of the ecosystem graph.

Usage::

    python tools/ecosystem/people_kb.py OUT/graph.json --out OUT [--people-py PATH]

Writes ``OUT/people.json`` with, per component, its recent committers, the openers of its
open issues and the authors in its ``pyproject.toml``, and the openers of open PROTEUS
issues that name it. With ``--people-py`` (``people.py`` of the related-work review), it
also finds who last changed the PROTEUS functions that call each module, with that
script's ``people_for_lines``, its stable interface.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from graph import ROOT, load_settings

sys.path.insert(0, str(ROOT / 'tools' / 'digest'))
from common import API, GitHub, is_bot, iso  # noqa: E402


def load_people(path: Path):
    spec = importlib.util.spec_from_file_location('people', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def committers(
    gh: GitHub, repo: str, since: str, bots: list[str], pages: int = 3
) -> list[dict]:
    """Logins with their number of commits on the default branch since ``since``."""
    counts, last = Counter(), {}
    for page in range(1, pages + 1):
        got = (
            gh.get_json(f'{API}/repos/{repo}/commits?since={since}&per_page=100&page={page}')
            or []
        )
        for c in got:
            author = c.get('author') or {}
            if not is_bot(author.get('login'), author.get('type'), bots):
                counts[author['login']] += 1
                last.setdefault(author['login'], c['commit']['author']['date'][:10])
        if len(got) < 100:
            break
    return [{'login': k, 'commits': n, 'last': last[k]} for k, n in counts.most_common()]


def open_issues(gh: GitHub, repo: str, bots: list[str]) -> list[dict]:
    """Open issues (not pull requests) of a repository, newest 100."""
    got = gh.get_json(f'{API}/repos/{repo}/issues?state=open&per_page=100') or []
    return [
        {
            'ref': f'{repo}#{i["number"]}',
            'login': i['user']['login'],
            'text': f'{i["title"]} {i.get("body") or ""}',
        }
        for i in got
        if 'pull_request' not in i and not is_bot(i['user']['login'], i['user']['type'], bots)
    ]


def openers(issues: list[dict]) -> list[dict]:
    by = defaultdict(list)
    for issue in issues:
        by[issue['login']].append(issue['ref'])
    return [
        {'login': k, 'issues': v} for k, v in sorted(by.items(), key=lambda kv: -len(kv[1]))
    ]


def function_spans(text: str) -> list[tuple[int, int, str]]:
    """``(first, last, qualified name)`` of every function, decorators included."""
    spans = []

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f'{prefix}{child.name}'
                if not isinstance(child, ast.ClassDef):
                    first = min([child.lineno, *(d.lineno for d in child.decorator_list)])
                    spans.append((first, child.end_lineno, name))
                walk(child, f'{name}.')

    walk(ast.parse(text), '')
    return spans


def call_site_owners(people, clone: Path, home: str, edge: dict, cache: Path) -> dict:
    """``{file: {'sites': {line: function}, 'functions': {function: {login: lines}}}}``: who
    last changed the lines of the PROTEUS functions that call the module, from
    ``people.people_for_lines``. ``''`` holds call lines outside a function and ``~name``
    a commit author without a known GitHub login."""
    sites = defaultdict(set)
    for refs in edge['symbols'].values():
        for ref in refs:
            path, line = ref.rsplit(':', 1)
            sites[path].add(int(line))
    out = {}
    for path, lines in sorted(sites.items()):
        spans = function_spans((clone / path).read_text())
        where, blocks = {}, defaultdict(set)
        for line in lines:
            inside = [s for s in spans if s[0] <= line <= s[1]]
            span = min(inside, key=lambda s: s[1] - s[0]) if inside else None
            where[line] = span[2] if span else ''
            blocks[where[line]] |= set(range(span[0], span[1] + 1)) if span else {line}
        functions = {}
        for fn, block in blocks.items():
            found = people.people_for_lines(str(clone), home, 'HEAD', {path: block}, cache)
            functions[fn] = {
                p['login'] or '~' + sorted(p['names'])[0]: sum(p['lines'].values())
                for p in found.values()
            }
        out[path] = {'sites': where, 'functions': functions}
    return out


def build(gh: GitHub, graph: dict, settings: dict, clone: Path, people, cache: Path) -> dict:
    bots = settings['bots']
    since = iso(datetime.now(timezone.utc) - timedelta(days=30.44 * settings['months']))
    home = graph['home']
    home_issues = open_issues(gh, home, bots)
    repos = {}
    for repo, node in graph['nodes'].items():
        short = repo.split('/')[1].removesuffix('.jl')
        named = [
            i
            for i in home_issues
            if repo != home and re.search(rf'\b{re.escape(short)}\b', i['text'], re.I)
        ]
        repos[repo] = {
            'authors': node['authors'],
            'committers': committers(gh, repo, since, bots),
            'issue_openers': openers(open_issues(gh, repo, bots)),
            'home_issue_openers': openers(named),
        }
    sites = {}
    if people:
        for e in (e for e in graph['edges'] if e['from'] == home and e['symbols']):
            sites[f'{e["from"]} -> {e["to"]}'] = call_site_owners(people, clone, home, e, cache)
    return {'since': since, 'repos': repos, 'call_sites': sites}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('graph', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument(
        '--root', type=Path, default=ROOT, help='PROTEUS clone with full history'
    )
    parser.add_argument('--people-py', type=Path, help='people.py of the related-work review')
    args = parser.parse_args(argv)
    people = load_people(args.people_py) if args.people_py else None
    cache = args.out / 'people-cache'
    cache.mkdir(parents=True, exist_ok=True)
    graph = json.loads(args.graph.read_text())
    kb = build(GitHub(), graph, load_settings(), args.root, people, cache)
    (args.out / 'people.json').write_text(json.dumps(kb, indent=1))
    print(f'{len(kb["repos"])} components, {len(kb["call_sites"])} call-site maps')
    return 0


if __name__ == '__main__':
    sys.exit(main())
