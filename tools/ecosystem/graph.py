"""Build the dependency and decision graph of the FWL ecosystem.

Usage::

    python tools/ecosystem/graph.py --out DIR

Writes ``DIR/graph.json``, ``DIR/graph.md``, ``DIR/interfaces.json`` (the PROTEUS files
that each edge of PROTEUS depends on, per kind) and ``DIR/module_interfaces.json`` (the
files of each module that PROTEUS imports from or calls). The public, active repositories of the
organisation and the extra repositories of ``ecosystem.toml`` are read through the GitHub
API: ``pyproject.toml`` and ``Project.toml`` give the package names and the dependency
edges. The PROTEUS checkout adds the module pins of ``[tool.proteus.modules]``, the
``tools/get_*.sh`` scripts, the call sites (Python imports and Julia ``jl.<Module>``
calls), the dispatch entries of ``module_map.json``, the config section of each module
and, from its history, the pull requests that set each pin.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'tools' / 'digest'))
from common import GitHub, iso  # noqa: E402

GITHUB_URL = re.compile(r'github\.com[/:]([\w.-]+)/([\w.-]+?)(?:\.git)?(?=[/"\'\s]|$)')
REQ = re.compile(r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*(.*)$')
JL_USING = re.compile(r"""seval\(\s*['"](?:using|import)\s+([A-Z]\w*)""")
JL_CALL = re.compile(r'\bjl\.([A-Z]\w*)\.([\w.]+)')
PR_REF = re.compile(r'\s*\(#(\d+)\)$')
SHA = re.compile(r'^[0-9a-f]{40}$')


def load_settings() -> dict:
    """Read ``ecosystem.toml`` and add the repositories of the weekly digest."""
    with open(HERE / 'ecosystem.toml', 'rb') as fh:
        settings = tomllib.load(fh)
    with open(ROOT / 'tools' / 'digest' / 'repos.toml', 'rb') as fh:
        digest = tomllib.load(fh)
    settings['digest_repos'], settings['bots'] = digest['repos'], digest['bots']
    return settings


def norm(name: str) -> str:
    """Normalised Python package name (PEP 503)."""
    return re.sub(r'[-_.]+', '-', name).lower()


def module_pins(pyproject: dict) -> dict:
    """``{name: (owner/repo, ref)}`` of the GitHub entries of ``[tool.proteus.modules]``."""
    out = {}
    for name, entry in pyproject.get('tool', {}).get('proteus', {}).get('modules', {}).items():
        if found := GITHUB_URL.search(entry.get('url', '')):
            out[name] = (f'{found[1]}/{found[2]}', entry.get('ref'))
    return out


def list_repos(gh: GitHub, settings: dict, pinned: list[str]) -> dict:
    """``{owner/repo: default branch}`` of the public, active repositories to read."""
    repos, page = {}, 1
    while True:
        got = gh.get_json(f'orgs/{settings["org"]}/repos?per_page=100&page={page}') or []
        repos |= {
            r['full_name']: r['default_branch']
            for r in got
            if not (r['archived'] or r['private'])
        }
        if len(got) < 100:
            break
        page += 1
    known = {r.lower() for r in repos}
    for name in sorted({*settings['extra_repos'], *settings['digest_repos'], *pinned}):
        if name.lower() not in known and (info := gh.get_json(f'repos/{name}')):
            if not (info['archived'] or info['private']):
                repos[info['full_name']] = info['default_branch']
    return repos


def read_node(gh: GitHub, repo: str, branch: str, unreadable: list) -> dict:
    """Package names, authors and requirements of one repository."""
    node = {
        'default_branch': branch,
        'dist': None,
        'julia': None,
        'authors': [],
        'requires': [],
    }
    for path in ('pyproject.toml', 'Project.toml'):
        text = gh.raw_file(repo, path, None)
        try:
            data = tomllib.loads(text) if text else None
        except tomllib.TOMLDecodeError as err:
            unreadable.append({'repo': repo, 'file': path, 'error': str(err)[:200]})
            continue
        if not data:
            continue
        if path == 'pyproject.toml':
            project = data.get('project', {})
            node['dist'] = project.get('name')
            node['authors'] = [a['name'] for a in project.get('authors', []) if a.get('name')]
            node['requires'] += [['runtime', s] for s in project.get('dependencies', [])]
            for extra, specs in project.get('optional-dependencies', {}).items():
                node['requires'] += [[f'extra {extra}', s] for s in specs]
        else:
            node['julia'] = data.get('name')
            compat = data.get('compat', {})
            node['requires'] += [
                ['julia', f'{n} {compat.get(n, "")}'] for n in data.get('deps', {})
            ]
    return node


def build_index(nodes: dict) -> dict:
    """Map names to repositories: ``pypi`` by normalised distribution name, ``name`` by
    lowercase repository, import or Julia package name. The first repository wins."""
    pypi, names = {}, {}
    for repo, node in sorted(nodes.items()):
        keys = [repo.split('/')[1].lower().removesuffix('.jl')]
        if node['dist']:
            pypi.setdefault(norm(node['dist']), repo)
            keys.append(node['dist'].replace('-', '_').lower())
        if node['julia']:
            keys.append(node['julia'].lower())
        for key in keys:
            names.setdefault(key, repo)
    return {'pypi': pypi, 'name': names, 'repo': {r.lower(): r for r in nodes}}


def edge(edges: dict, src: str, dst: str) -> dict:
    """The record of the edge ``src`` depends on ``dst``, created when absent."""
    return edges.setdefault(
        (src, dst),
        {
            'from': src,
            'to': dst,
            'how': [],
            'symbols': {},
            'dispatch': [],
            'config': [],
            'decisions': [],
        },
    )


def manifest_edges(nodes: dict, index: dict, edges: dict) -> None:
    """Edges from the Python and Julia requirements of every repository."""
    for repo, node in nodes.items():
        for group, spec in node['requires']:
            if not (found := REQ.match(spec)):
                continue
            name, rest = found[1], found[2].strip()
            dst = index['name' if group == 'julia' else 'pypi'].get(
                name.lower() if group == 'julia' else norm(name)
            )
            if dst and dst != repo:
                edge(edges, repo, dst)['how'].append(
                    {'kind': group, 'package': name, 'spec': rest}
                )


def code_references(text: str):
    """``(line, key)`` of each Python import, each use of an imported name, and each Julia
    call in a source file. The key is ``module:name`` or, for a whole module, ``module``."""
    try:
        nodes = list(ast.walk(ast.parse(text)))
    except SyntaxError:
        nodes = []
    bound = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for a in node.names:
                bound.setdefault(a.asname or a.name.split('.')[0], a.name)
                yield node.lineno, a.name
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for a in node.names:
                bound.setdefault(a.asname or a.name, f'{node.module}:{a.name}')
                yield node.lineno, f'{node.module}:{a.name}'
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in bound:
            yield node.lineno, bound[node.id]
    for n, line in enumerate(text.splitlines(), 1):
        yield from ((n, m[1]) for m in JL_USING.finditer(line))
        for m in JL_CALL.finditer(line):
            *mods, name = m[2].split('.')
            name = name[:-2] + '!' if name.endswith('_b') else name
            yield n, f'{".".join([m[1], *mods])}:{name}'


def proteus_edges(root: Path, home: str, index: dict, edges: dict) -> None:
    """Edges of PROTEUS from its module pins, install scripts, code and config schema."""
    pyproject = tomllib.loads((root / 'pyproject.toml').read_text())
    for name, (repo, ref) in module_pins(pyproject).items():
        if dst := index['repo'].get(repo.lower()):
            edge(edges, home, dst)['how'].append(
                {'kind': 'git pin', 'package': name, 'spec': ref}
            )
    for script in sorted((root / 'tools').glob('get_*.sh')):
        if dst := index['name'].get(script.stem[4:].lower()):
            edge(edges, home, dst)['how'].append(
                {'kind': 'script', 'file': f'tools/{script.name}'}
            )
    for path in sorted((root / 'src' / 'proteus').rglob('*.py')):
        rel = str(path.relative_to(root))
        for line, key in code_references(path.read_text()):
            dst = index['name'].get(key.split(':')[0].split('.')[0].lower())
            if dst and dst != home:
                sites = edge(edges, home, dst)['symbols'].setdefault(key, [])
                if f'{rel}:{line}' not in sites:
                    sites.append(f'{rel}:{line}')
    for site in json.loads((root / 'docs/Reference/module_map.json').read_text())['sites']:
        for opt in site['options']:
            if (
                opt['option']
                and opt['entry']
                and (dst := index['name'].get(opt['option'].lower()))
            ):
                edge(edges, home, dst)['dispatch'].append(opt['entry'])
    fields = json.loads((root / 'docs/Reference/config/config_schema.json').read_text())[
        'fields'
    ]
    paths = [f['path'] for f in fields]
    for field in (f for f in fields if f['path'].endswith('.module')):
        area = field['path'].removesuffix('.module')
        for choice in field.get('choices') or []:
            if choice and (dst := index['name'].get(choice.lower())):
                section = f'{area}.{choice}'
                keys = sum(p.startswith(section + '.') for p in paths)
                edge(edges, home, dst)['config'].append(
                    {
                        'selector': f'{field["path"]} = {choice}',
                        'section': section,
                        'keys': keys,
                    }
                )


def version(text: str | None) -> tuple | None:
    """Integer tuple of a version such as ``v26.11.5``; None when a part is not a number."""
    parts = (text or '').strip().lstrip('v').split('.')
    return tuple(int(p) for p in parts) if all(p.isdigit() for p in parts) else None


def pin_status(gh: GitHub, nodes: dict, edges: dict) -> None:
    """Add how far each git pin is behind its default branch and the latest release."""
    latest = {}
    for e in edges.values():
        dst = e['to']
        for how in e['how']:
            spec = how.get('spec') or ''
            if how['kind'] == 'git pin' and SHA.match(spec):
                branch = nodes[dst]['default_branch']
                found = gh.get_json(f'repos/{dst}/compare/{spec}...{branch}') or {}
                how['behind'] = found.get('ahead_by')
                how['status'] = found.get('status')
            elif how['kind'] != 'git pin' and '>=' in spec:
                if dst not in latest:
                    latest[dst] = (gh.get_json(f'repos/{dst}/releases/latest') or {}).get(
                        'tag_name'
                    )
                floor = version(spec.split('>=')[1].split(',')[0])
                how['latest'] = latest[dst]
                new = version(latest[dst])
                how['newer_release'] = bool(floor and new and new > floor)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(root), *args], capture_output=True, text=True, check=True
    ).stdout


def pin_lines(text: str, how: dict) -> tuple[int, int, int] | None:
    """``(first comment line, first pin line, last pin line)`` of a pin in ``pyproject.toml``."""
    lines = text.splitlines()
    if how['kind'] == 'git pin':
        header = f'[tool.proteus.modules.{how["package"]}]'
        start = next((i for i, x in enumerate(lines) if x.strip() == header), None)
        if start is None:
            return None
        end = next((i for i in range(start, len(lines)) if lines[i].startswith('ref')), start)
    else:
        pattern = re.compile(rf'^\s*"{re.escape(how["package"])}\b')
        start = next((i for i, x in enumerate(lines) if pattern.match(x)), None)
        if start is None:
            return None
        end = start
    first = start
    while first > 0 and lines[first - 1].lstrip().startswith('#'):
        first -= 1
    return first + 1, start + 1, end + 1


def pin_decisions(gh: GitHub, root: Path, home: str, edges: dict, months: int) -> None:
    """Attach to each PROTEUS pin the pull requests that changed it and the issues and pull
    requests that its comment names."""
    text = (root / 'pyproject.toml').read_text()
    since = iso(datetime.now(timezone.utc) - timedelta(days=30.44 * months))
    for e in (e for e in edges.values() if e['from'] == home):
        for how in e['how']:
            if how['kind'] not in ('git pin', 'runtime') and not how['kind'].startswith(
                'extra'
            ):
                continue
            if not (span := pin_lines(text, how)):
                continue
            first, start, end = span
            log = git(
                root,
                'log',
                f'--since={since}',
                '--format=COMMIT\t%H\t%as\t%s',
                '-L',
                f'{start},{end}:pyproject.toml',
            )
            for row in (r for r in log.splitlines() if r.startswith('COMMIT\t')):
                _, sha, date, subject = row.split('\t', 3)
                if ref := PR_REF.search(subject):
                    e['decisions'].append(
                        {
                            'ref': f'{home}#{ref[1]}',
                            'date': date,
                            'source': 'changed the pin',
                            'title': subject[: ref.start()],
                        }
                    )
            if first < start:
                comment = ' '.join(
                    x.strip().lstrip('#').strip()
                    for x in text.splitlines()[first - 1 : start - 1]
                )
                date = git(
                    root,
                    'log',
                    '-1',
                    '--format=%as',
                    '-L',
                    f'{first},{start - 1}:pyproject.toml',
                )
                for n in sorted(set(re.findall(r'#(\d+)', comment)), key=int):
                    if ref := resolve_ref(gh, int(n), [e['to'], home], date.split('\n')[0]):
                        e['decisions'].append(
                            {**ref, 'source': 'named in the pin comment', 'comment': comment}
                        )


def resolve_ref(gh: GitHub, number: int, repos: list[str], date: str) -> dict | None:
    """The issue or pull request ``#number`` of the repository where it was opened last
    before ``date`` (``YYYY-MM-DD``), the date the comment that names it was written."""
    found = []
    for repo in repos:
        item = gh.get_json(f'repos/{repo}/issues/{number}')
        if item and item['created_at'][:10] <= date:
            found.append((item['created_at'], repo, item['title']))
    if not found:
        return None
    created, repo, title = max(found)
    return {
        'ref': f'{repo}#{number}',
        'date': created[:10],
        'title': title,
        'ambiguous': len(found) > 1,
    }


def mermaid_id(repo: str) -> str:
    return re.sub(r'\W', '_', repo.split('/')[1])


def render(graph: dict) -> str:
    """Markdown with a Mermaid diagram and one table row per edge."""
    out = [
        '# FWL ecosystem graph',
        '',
        f'Generated {graph["generated"]} from {len(graph["nodes"])} repositories. '
        'An arrow points from a component to a component it depends on.',
        '',
        '```mermaid',
        'flowchart LR',
    ]
    for e in graph['edges']:
        kinds = sorted(
            {h['kind'].split(' ')[0] for h in e['how']} | ({'code'} if e['symbols'] else set())
        )
        label = ', '.join(kinds) or 'config'
        out.append(f'  {mermaid_id(e["from"])} -->|{label}| {mermaid_id(e["to"])}')
    out += [
        '```',
        '',
        '## Edges',
        '',
        '| Component | Depends on | Pin | Status | Call sites | Config section | Decisions |',
        '|---|---|---|---|---|---|---|',
    ]
    for e in graph['edges']:
        pins = '; '.join(
            f'{h["kind"]} {h.get("package", h.get("file", ""))} {h.get("spec", "")[:12]}'.strip()
            for h in e['how']
            if h['kind'] != 'script'
        )
        status = '; '.join(
            f'{h["behind"]} commits behind' if 'behind' in h else f'latest {h["latest"]}'
            for h in e['how']
            if h.get('behind') or h.get('newer_release')
        )
        files = sorted({s.split(':')[0] for sites in e['symbols'].values() for s in sites})
        config = ', '.join(f'`{c["section"]}` ({c["keys"]} keys)' for c in e['config'])
        decisions = '<br>'.join(
            f'{d["ref"].split("/")[1]} {d["title"][:60]}' for d in e['decisions'][:4]
        )
        out.append(
            f'| {e["from"]} | {e["to"]} | {pins} | {status} | {len(files)} files | {config} | {decisions} |'
        )
    return '\n'.join(out) + '\n'


def build(gh: GitHub, root: Path, settings: dict) -> dict:
    home = settings['home']
    pinned = [
        r for r, _ in module_pins(tomllib.loads((root / 'pyproject.toml').read_text())).values()
    ]
    unreadable = []
    nodes = {
        r: read_node(gh, r, b, unreadable)
        for r, b in sorted(list_repos(gh, settings, pinned).items())
    }
    index = build_index(nodes)
    edges = {}
    manifest_edges(nodes, index, edges)
    proteus_edges(root, home, index, edges)
    for d in settings.get('declared', []):
        edge(edges, d['from'], d['to'])['how'].append(
            {'kind': 'declared', 'source': d['source']}
        )
    pin_status(gh, nodes, edges)
    pin_decisions(gh, root, home, edges, settings['months'])
    linked = {r for e in edges for r in e}
    return {
        'generated': iso(datetime.now(timezone.utc)),
        'home': home,
        'nodes': {r: n for r, n in nodes.items() if r in linked},
        'unlinked': sorted(set(nodes) - linked),
        'edges': sorted(edges.values(), key=lambda e: (e['from'] != home, e['from'], e['to'])),
        'unreadable': unreadable,
    }


def interfaces(graph: dict, sha: str) -> list[dict]:
    """The PROTEUS files of each PROTEUS edge, per kind: ``pin`` (``pyproject.toml``),
    ``script`` (``tools/get_*.sh``), ``import`` (files that import or call the module) and
    ``wrapper`` (dispatch entry files), with the commit the graph was built from."""
    out = []
    for e in (e for e in graph['edges'] if e['from'] == graph['home']):
        files = defaultdict(set)
        for how in e['how']:
            if how['kind'] == 'script':
                files['script'].add(how['file'])
            elif how['kind'] in ('git pin', 'runtime') or how['kind'].startswith('extra'):
                files['pin'].add('pyproject.toml')
        files['import'] = {s.rsplit(':', 1)[0] for v in e['symbols'].values() for s in v}
        files['wrapper'] = {d.split(':')[0] for d in e['dispatch']}
        out += [
            {
                'from': e['from'],
                'to': e['to'],
                'kind': kind,
                'interface_files': sorted(paths),
                'built_from': sha,
            }
            for kind, paths in sorted(files.items())
            if paths
        ]
    return out


def module_files(module: str, name: str | None, julia: str | None) -> list[str]:
    """Path suffixes of the files that can define ``module`` or ``module.name``. A Julia
    submodule ``AGNI.atmosphere`` lives in a file ``atmosphere.jl`` in any folder; two files
    of that name both count."""
    parts = module.split('.')
    if parts[0] == julia:
        return [f'{parts[-1]}.jl']
    base = '/'.join(parts)
    return [f'{base}.py', f'{base}/__init__.py'] + ([f'{base}/{name}.py'] if name else [])


def module_interfaces(gh: GitHub, graph: dict) -> list[dict]:
    """Per PROTEUS edge, the files of the module repository at the head of its default branch
    that define what PROTEUS imports (``import``) or calls through juliacall (``call``).
    A name whose file is not found is listed in ``unresolved``."""
    out = []
    for e in (e for e in graph['edges'] if e['from'] == graph['home'] and e['symbols']):
        node = graph['nodes'][e['to']]
        head = gh.get_json(f'repos/{e["to"]}/commits/{node["default_branch"]}') or {}
        tree = gh.get_json(f'repos/{e["to"]}/git/trees/{head.get("sha")}?recursive=1') or {}
        paths = [t['path'] for t in tree.get('tree', []) if t['type'] == 'blob']
        found = defaultdict(lambda: {'files': set(), 'symbols': set(), 'unresolved': set()})
        for key in e['symbols']:
            module, _, name = key.partition(':')
            kind = 'call' if module.split('.')[0] == node['julia'] else 'import'
            suffixes = module_files(module, name or None, node['julia'])
            hits = {p for p in paths if any(p == c or p.endswith('/' + c) for c in suffixes)}
            found[kind]['files' if hits else 'unresolved'] |= hits or {key}
            found[kind]['symbols'].add(key)
        out += [
            {
                'from': e['from'],
                'to': e['to'],
                'kind': kind,
                'module_files': sorted(f['files']),
                'symbols': sorted(f['symbols']),
                'unresolved': sorted(f['unresolved']),
                'built_from': head.get('sha'),
            }
            for kind, f in sorted(found.items())
        ]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=ROOT, help='PROTEUS checkout')
    args = parser.parse_args(argv)
    graph = build(GitHub(), args.root, load_settings())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'graph.json').write_text(json.dumps(graph, indent=1))
    (args.out / 'graph.md').write_text(render(graph))
    sha = git(args.root, 'rev-parse', 'HEAD').strip()
    (args.out / 'interfaces.json').write_text(json.dumps(interfaces(graph, sha), indent=1))
    modules = module_interfaces(GitHub(), graph)
    (args.out / 'module_interfaces.json').write_text(json.dumps(modules, indent=1))
    print(
        f'{len(graph["nodes"])} components, {len(graph["edges"])} edges, {len(graph["unlinked"])} unlinked'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
