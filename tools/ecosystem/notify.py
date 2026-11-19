"""Draft the cross-repository notifications of a digest window.

Usage::

    python tools/ecosystem/notify.py collected.json OUT/graph.json --out OUT [--people OUT/people.json]

Reads the merged pull requests, releases and facts of ``tools/digest/collect.py`` and
``facts.py``, and the graph of ``graph.py``. Writes ``OUT/notifications.json`` and
``OUT/notifications.md``. Each item goes to the component that a change reaches:

- ``action``: an upstream pull request changes the definition line of a name that PROTEUS
  calls, or removes a config key that a PROTEUS call site names.
- ``check``: a release newer than a floor; a changed requirement that a neighbour pins with
  an upper bound; an upstream changed line that names a called symbol; a changed upstream
  default that a PROTEUS call site names.
- ``fyi``: PROTEUS changed how it calls, configures or pins a module; a release of a module
  pinned by commit; a new upstream config key; a whole imported module changed.

A component with an action or check item gets one draft issue for the window, which tags
at most ``tag_limit`` people per item; FYI items become lines for the digest and tag nobody.
Nothing is posted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from graph import ROOT, load_settings, norm, version

sys.path.insert(0, str(ROOT / 'tools' / 'digest'))
from common import GitHub, is_bot  # noqa: E402

LEVELS = {'action': 'Action needed', 'check': 'Please check', 'fyi': 'For information'}
DEV_EXTRAS = {'extra dev', 'extra develop', 'extra docs', 'extra test', 'extra tests'}
REQ = re.compile(r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)')


def word(name: str) -> re.Pattern:
    return re.compile(rf'(?<![\w!]){re.escape(name)}(?![\w!])')


def definition(name: str) -> re.Pattern:
    kinds = r'(?:async\s+def|def|class|function|mutable\s+struct|struct)'
    return re.compile(rf'^[-+]\s*{kinds}\s+(?:\w+\.)*{re.escape(name)}(?![\w!])')


def used(name: str) -> re.Pattern:
    """A call, an attribute access or an import of ``name``, not a mention in text."""
    n = rf'(?<![\w!]){re.escape(name)}(?![\w!])'
    return re.compile(rf'{n}(?=\s*\(|\.[A-Za-z_])|\bimport\b.*{n}|{n}.*\bimport\b')


def changed(patch: str | None) -> list[str]:
    """Changed lines and hunk headers; a header names the function around the hunk."""
    return [
        x
        for x in (patch or '').splitlines()
        if x[:2] == '@@' or (x[:1] in '+-' and x[:3] not in ('+++', '---'))
    ]


class Window:
    """The collected data, the graph and the lookups that the rules share."""

    def __init__(
        self, gh: GitHub, data: dict, graph: dict, kb: dict, settings: dict, root: Path
    ):
        self.gh, self.graph, self.kb, self.root = gh, graph, kb, root
        self.home, self.settings = graph['home'], settings
        self.edges = {(e['from'], e['to']): e for e in graph['edges']}
        self.prs = {
            p['ref']: p
            for p in data['prs']
            if p['status'] == 'merged'
            and not is_bot(p['author'], p['author_kind'], settings['bots'])
        }
        self.releases = data['releases']
        self.facts = defaultdict(list)
        for fact in data.get('facts', []):
            if fact['pr'] in self.prs:
                self.facts[fact['pr']].append(fact)
        self.patches, self.bodies = {}, {}
        self.items = []

    def full(self, pr: dict) -> str:
        return f'{pr["repo"]}#{pr["number"]}'

    def files(self, pr: dict) -> dict:
        """``{path: patch or None}`` of a pull request; GitHub omits the patch of a large diff."""
        if pr['ref'] not in self.patches:
            out = {}
            for page in range(1, 31):
                got = (
                    self.gh.get_json(
                        f'repos/{pr["repo"]}/pulls/{pr["number"]}/files?per_page=100&page={page}'
                    )
                    or []
                )
                out |= {f['filename']: f.get('patch') for f in got}
                if len(got) < 100:
                    break
            self.patches[pr['ref']] = out
        return self.patches[pr['ref']]

    def add(self, target: str, level: str, source: str, text: str, people: list[str] = ()):
        self.items.append(
            {
                'target': target,
                'level': level,
                'source': source,
                'text': text,
                'people': list(people) if level != 'fyi' else [],
            }
        )


def symbols(edge: dict, julia: str | None):
    """``(module, name or None, sites, candidate files)`` of every name PROTEUS takes from a module."""
    for key, sites in edge['symbols'].items():
        module, _, name = key.partition(':')
        if name.startswith('__'):
            continue
        parts = module.split('.')
        if parts[0] == julia:
            files = [f'src/{parts[-1]}.jl']
        else:
            base = '/'.join(parts)
            files = [f'{base}.py', f'{base}/__init__.py'] + (
                [f'{base}/{name}.py'] if name else []
            )
        yield module, name or None, sites, files


def site_files(edge: dict) -> set[str]:
    return {s.rsplit(':', 1)[0] for sites in edge['symbols'].values() for s in sites} | {
        d.split(':')[0] for d in edge['dispatch']
    }


def owners(w: Window, edge: dict, sites, exclude: set[str]) -> list[str]:
    """People who wrote most lines of the PROTEUS functions that hold ``sites`` (``file:line``)."""
    by_file = w.kb.get('call_sites', {}).get(f'{edge["from"]} -> {edge["to"]}', {})
    functions = set()
    for site in sites:
        path, line = site.rsplit(':', 1)
        if (fn := by_file.get(path, {}).get('sites', {}).get(line)) is not None:
            functions.add((path, fn))
    count = Counter()
    for path, fn in functions:
        for login, n in by_file[path]['functions'][fn].items():
            if not login.startswith('~') and login.lower() not in exclude:
                count[login] += n
    return [login for login, _ in count.most_common(w.settings['tag_limit'])]


def pin_owner(w: Window, edge: dict, exclude: str) -> list[str]:
    """Author of the pull request that last changed the pin, unless that is ``exclude``."""
    last = next((d for d in edge['decisions'] if d['source'] == 'changed the pin'), None)
    if not last:
        return []
    repo, number = last['ref'].split('#')
    login = ((w.gh.get_json(f'repos/{repo}/pulls/{number}') or {}).get('user') or {}).get(
        'login'
    )
    return [login] if login and login.lower() != exclude else []


def maintainers(w: Window, repo: str, exclude: set[str]) -> list[str]:
    rows = w.kb.get('repos', {}).get(repo, {}).get('committers', [])
    return [r['login'] for r in rows if r['login'].lower() not in exclude][
        : w.settings['tag_limit']
    ]


def upstream_code(w: Window) -> None:
    """Upstream pull requests that change files from which PROTEUS imports or calls names."""
    for pr in (p for p in w.prs.values() if p['repo'] != w.home):
        edge = w.edges.get((w.home, pr['repo']))
        if not edge or not edge['symbols']:
            continue
        julia = w.graph['nodes'][pr['repo']].get('julia')
        hits = defaultdict(list)
        for module, name, sites, files in symbols(edge, julia):
            touched = [
                f for f in pr['files'] if any(f == c or f.endswith('/' + c) for c in files)
            ]
            if not touched:
                continue
            patches = w.files(pr)
            lines = [x for f in touched for x in changed(patches.get(f))]
            if name is None:
                kind = 'module'
            elif any(definition(name).match(x) and x[0] == '-' for x in lines):
                kind = 'definition'
            elif any(definition(name).match(x) for x in lines):
                kind = 'added'
            elif any(word(name).search(x) for x in lines) or any(
                patches.get(f) is None for f in touched
            ):
                kind = 'lines'
            else:
                continue
            hits[kind].append((name or module, f'{module}.{name}' if name else module, sites))
        for kind, found in hits.items():
            level = {'definition': 'action', 'lines': 'check'}.get(kind, 'fyi')
            names = ', '.join(f'`{n}`' for _, n, _ in found[:6]) + (
                f' and {len(found) - 6} more' if len(found) > 6 else ''
            )
            sites = sorted({s for _, _, ss in found for s in ss})
            verb = {
                'definition': 'changes the definition of',
                'added': 'adds',
                'lines': 'changes code in or naming',
                'module': 'changes the module',
            }[kind]
            per_file = Counter(s.rsplit(':', 1)[0] for s in sites)
            text = f'{verb} {names}, which PROTEUS uses in ' + ', '.join(
                f'`{f}` ({n} lines)' for f, n in sorted(per_file.items())
            )
            if handled := handled_by(w, pr, edge, sites, {n for n, _, _ in found}):
                level = 'fyi' if level == 'check' else 'check' if level == 'action' else level
                one = len(handled) == 1
                text += (
                    f'; {", ".join(handled)} {"changes" if one else "change"} the calling PROTEUS code'
                    f' or {"links" if one else "link"} this change'
                )
            w.add(
                w.home,
                level,
                w.full(pr),
                text + '.',
                owners(w, edge, sites, {pr['author'].lower()}),
            )


def handled_by(
    w: Window, upstream: dict, edge: dict, sites: list[str], names: set[str]
) -> list[str]:
    """PROTEUS pull requests of the window that link ``upstream`` in their body, change a call
    of ``names``, or change a function that holds one of ``sites``."""
    owner, repo = upstream['repo'].split('/')
    link = re.compile(
        rf'(?:{owner}/)?{repo}(?:/pull/|/issues/|#){upstream["number"]}(?!\d)', re.I
    )
    by_file = w.kb.get('call_sites', {}).get(f'{edge["from"]} -> {edge["to"]}', {})
    files, functions = set(), set()
    for site in sites:
        path, line = site.rsplit(':', 1)
        files.add(path)
        functions.add((by_file.get(path, {}).get('sites', {}).get(line) or '').split('.')[-1])
    out = []
    for pr in (p for p in w.prs.values() if p['repo'] == w.home):
        if pr['ref'] not in w.bodies:
            w.bodies[pr['ref']] = (
                w.gh.get_json(f'repos/{w.home}/pulls/{pr["number"]}') or {}
            ).get('body') or ''
        lines = [x for f in files & set(pr['files']) for x in changed(w.files(pr).get(f))]
        if (
            link.search(w.bodies[pr['ref']])
            or any(used(n).search(x) for n in names for x in lines if x[:2] != '@@')
            or any(
                re.search(rf'\bdef {re.escape(fn)}\(', x)
                for fn in functions - {''}
                for x in lines
                if x[:2] == '@@'
            )
        ):
            out.append(w.full(pr))
    return out


def upstream_config(w: Window) -> None:
    """Changed keys of an upstream default config file that a PROTEUS call site names."""
    for ref, facts in w.facts.items():
        pr = w.prs[ref]
        edge = w.edges.get((w.home, pr['repo']))
        for fact in (f for f in facts if edge and f['kind'] == 'config'):
            leaf = fact['key'].split('.')[-1]
            found = [
                p
                for p in sorted(site_files(edge))
                if (w.root / p).exists() and word(leaf).search((w.root / p).read_text())
            ]
            level = (
                {'removed': 'action', 'value': 'check'}.get(fact['change']) if found else None
            )
            level = 'fyi' if fact['change'] == 'added' else level
            if level:
                where = f', named in {", ".join(f"`{p}`" for p in found)}' if found else ''
                text = f'{fact["change"]} the key `{fact["key"]}` in `{fact["file"]}`{where}.'
                sites = [
                    x
                    for v in edge['symbols'].values()
                    for x in v
                    if x.rsplit(':', 1)[0] in found
                ]
                w.add(
                    w.home,
                    level,
                    w.full(pr),
                    text,
                    owners(w, edge, sites, {pr['author'].lower()}),
                )


def requirement(spec_list: list, package: str) -> list[tuple[str, str]]:
    out = []
    for group, spec in spec_list:
        if (m := REQ.match(spec)) and norm(m[1]) == norm(package) and group not in DEV_EXTRAS:
            out.append((group, spec[m.end() :].strip()))
    return out


def shared_requirements(w: Window) -> None:
    """A changed requirement on a package that a neighbouring component also requires."""
    for ref, facts in w.facts.items():
        pr = w.prs[ref]
        for fact in (f for f in facts if f['kind'] == 'dependency' and ':' in f['key']):
            package = fact['key'].split(':', 1)[1]
            for a, b in w.edges:
                other = b if a == pr['repo'] else a if b == pr['repo'] else None
                if not other or other not in w.graph['nodes']:
                    continue
                for group, spec in requirement(w.graph['nodes'][other]['requires'], package):
                    bounded = any(op in f'{spec} {fact["new"]}' for op in ('<', '==', '~='))
                    text = (
                        f'changes the requirement on `{package}` from `{fact["old"]}` to `{fact["new"]}`; '
                        f'{other.split("/")[1]} requires `{package} {spec}` ({group}).'
                    )
                    w.add(
                        other,
                        'check' if bounded else 'fyi',
                        w.full(pr),
                        text,
                        maintainers(w, other, {pr['author'].lower()}),
                    )


def releases(w: Window) -> None:
    """A release newer than the floor of a dependent, or of a module pinned by commit."""
    for rel in w.releases:
        new = version(rel['tag'])
        for edge in (e for e in w.graph['edges'] if e['to'] == rel['repo']):
            for how in edge['how']:
                spec = how.get('spec') or ''
                if how['kind'] == 'git pin':
                    behind = (
                        f', {how["behind"]} commits behind the default branch'
                        if how.get('behind')
                        else ''
                    )
                    text = (
                        f'is released; {edge["from"].split("/")[1]} pins `{spec[:9]}`{behind}.'
                    )
                    w.add(
                        edge['from'], 'fyi', f'[{rel["repo"]} {rel["tag"]}]({rel["url"]})', text
                    )
                elif '>=' in spec and how['kind'] not in DEV_EXTRAS:
                    floor = version(spec.split('>=')[1].split(',')[0])
                    if new and floor and new > floor:
                        text = f'is released; {edge["from"].split("/")[1]} requires `{how["package"]} {spec}`.'
                        people = pin_owner(w, edge, (rel['author'] or '').lower())
                        w.add(
                            edge['from'],
                            'check',
                            f'[{rel["repo"]} {rel["tag"]}]({rel["url"]})',
                            text,
                            people,
                        )


def downstream(w: Window) -> None:
    """PROTEUS pull requests that change how PROTEUS calls, configures or pins a module."""
    for ref, pr in ((r, p) for r, p in w.prs.items() if p['repo'] == w.home):
        facts = w.facts.get(ref, [])
        for edge in (e for e in w.graph['edges'] if e['from'] == w.home):
            sections = [c['section'] for c in edge['config']]
            config = [
                f['key']
                for f in facts
                if f['kind'] == 'config' and any(f['key'].startswith(s + '.') for s in sections)
            ]
            packages = {norm(h['package']) for h in edge['how'] if h.get('package')}
            pins = [
                f
                for f in facts
                if f['kind'] == 'dependency' and norm(pin_package(f['key'])) in packages
            ]
            calls = []
            for path in sorted(site_files(edge) & set(pr['files'])):
                names = {k.partition(':')[2] or k.split('.')[-1] for k in edge['symbols']} | {
                    edge['to'].split('/')[1].lower()
                }
                lines = changed(w.files(pr).get(path))
                if any(
                    used(n).search(x)
                    for n in names
                    if n and not n.startswith('__')
                    for x in lines
                ):
                    calls.append(path)
            parts = []
            if calls:
                parts.append('changes the calls in ' + ', '.join(f'`{c}`' for c in calls))
            if config:
                parts.append(f'changes {len(config)} config keys under `{sections[0]}`')
            if pins:
                parts.append('sets the pin to ' + ', '.join(f'`{f["new"]}`' for f in pins))
            if parts:
                w.add(edge['to'], 'fyi', w.full(pr), '; '.join(parts) + '.')


def pin_package(key: str) -> str:
    """Package name of a dependency fact key such as ``dependencies:fwl-io`` or
    ``tool.proteus.modules.agni.ref``."""
    if ':' in key:
        return key.split(':', 1)[1]
    parts = key.split('.')
    return parts[3] if key.startswith('tool.proteus.modules.') and len(parts) > 3 else key


def render(w: Window, data: dict) -> str:
    since, until = data['window']['since'][:10], data['window']['until'][:10]
    out = [
        '# 🕸️ Cross-repository notifications',
        '',
        f'Window {since} to {until}. Drafts only: nothing is posted and nobody is tagged.',
        '',
    ]
    by_target = defaultdict(list)
    for item in w.items:
        by_target[item['target']].append(item)
    fyi = []
    for target in sorted(by_target):
        items = by_target[target]
        if not any(i['level'] != 'fyi' for i in items):
            fyi += items
            continue
        out += [
            f'## Draft issue in {target}',
            '',
            f'**Title:** 🕸️ Changes in other components to check ({since} to {until})',
            '',
        ]
        body = [
            f'🕸️ Changes in components that {target.split("/")[1]} depends on, or that depend on it, '
            'landed this week. Each line names the change and where it reaches this repository.'
        ]
        for level in LEVELS:
            rows = [i for i in items if i['level'] == level]
            if rows:
                body += ['', f'**{LEVELS[level]}**', '']
                body += [
                    f'- {i["source"]} {i["text"]}' + ''.join(f' @{p}' for p in i['people'])
                    for i in rows
                ]
        out += ['> ' + x if x else '>' for x in body] + ['']
    out += ['## FYI lines for the weekly digest', '']
    out += [f'- {i["target"].split("/")[1]}: {i["source"]} {i["text"]}' for i in fyi] or [
        'None.'
    ]
    return '\n'.join(out) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('collected', type=Path)
    parser.add_argument('graph', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--people', type=Path)
    parser.add_argument('--root', type=Path, default=ROOT, help='PROTEUS checkout')
    args = parser.parse_args(argv)
    data = json.loads(args.collected.read_text())
    kb = json.loads(args.people.read_text()) if args.people else {}
    w = Window(
        GitHub(), data, json.loads(args.graph.read_text()), kb, load_settings(), args.root
    )
    for rule in (upstream_code, upstream_config, shared_requirements, releases, downstream):
        rule(w)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'notifications.json').write_text(
        json.dumps({'window': data['window'], 'items': w.items}, indent=1)
    )
    (args.out / 'notifications.md').write_text(render(w, data))
    print(Counter(i['level'] for i in w.items))
    return 0


if __name__ == '__main__':
    sys.exit(main())
