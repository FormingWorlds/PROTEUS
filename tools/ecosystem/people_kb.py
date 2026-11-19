"""Link people to the components and call sites of the ecosystem graph.

Usage::

    python tools/ecosystem/people_kb.py OUT/graph.json --out OUT [--people-py PATH]

Writes ``OUT/people.json`` with, per component, its recent committers, the openers of its
open issues and the authors in its ``pyproject.toml``, and the openers of open PROTEUS
issues that name it. With ``--people-py`` (``people.py`` of the related-work review), it
also blames the PROTEUS functions that call each module and merges the commit emails by
GitHub login with that script's ``blame``, ``functions``, ``enclosing`` and ``by_login``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from graph import ROOT, load_settings

sys.path.insert(0, str(ROOT / 'tools' / 'digest'))
from common import GitHub, is_bot, iso  # noqa: E402


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
        got = gh.get_json(f'repos/{repo}/commits?since={since}&per_page=100&page={page}') or []
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
    got = gh.get_json(f'repos/{repo}/issues?state=open&per_page=100') or []
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


def call_site_owners(people, clone: Path, home: str, edge: dict, cache: Path) -> dict:
    """``{file: {'sites': {line: function}, 'functions': {function: {login: lines}}}}``: blame
    of the PROTEUS functions that call the module; ``''`` holds lines outside a function and
    ``~name`` a commit author without a known GitHub login."""
    sites = defaultdict(set)
    for refs in edge['symbols'].values():
        for ref in refs:
            path, line = ref.rsplit(':', 1)
            sites[path].add(int(line))
    out = {}
    for path, lines in sorted(sites.items()):
        spans = [s for s in people.functions((clone / path).read_text()) if not s[3]]
        where, owned = {}, {}
        for line in lines:
            span = people.enclosing(spans, line)
            where[line] = span[2] if span else ''
            owned.update(
                {n: where[line] for n in (range(span[0], span[1] + 1) if span else [line])}
            )
        records = defaultdict(people.record)
        for n, (email, name, sha) in people.blame(str(clone), 'HEAD', path, set(owned)).items():
            rec = records[email]
            rec['names'].add(name)
            rec['emails'].add(email)
            rec['sha'] = rec['sha'] or sha
            rec['function_lines'][owned[n]] += 1
        functions = defaultdict(dict)
        for p in people.by_login(home, records, cache).values():
            for fn, n in p['function_lines'].items():
                functions[fn][p['login'] or '~' + sorted(p['names'])[0]] = n
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
