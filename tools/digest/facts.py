"""Mechanical facts of the digest: config, output and dependency changes per merged PR.

Usage::

    python tools/digest/facts.py collected.json [--summary FILE]

For every merged PR that touches a tracked file, the file before and after the
merge commit is compared; each change becomes a fact with an id ``F<n>``. The
model cites facts by id and never writes a setting, a value or a floor itself.
A file that does not parse is listed in ``facts_unreadable`` and gives no facts.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import tomllib

from common import TRACKED_ALWAYS, DigestError, GitHub, load_settings

CONFIG_SCHEMA = 'docs/Reference/config/config_schema.json'
OUTPUT_SCHEMA = 'docs/Reference/output_schema.json'
DESCRIPTION_CHARS = 600
GROUP_LIMIT = 8
CONFIG_FIELDS = ('default', 'choices', 'bounds', 'description')
UNREADABLE = (ValueError, KeyError, TypeError, AttributeError, RecursionError)
REQ_RE = re.compile(r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)$')


def _same(value: object) -> str:
    """A comparable form of a value, so that an unchanged nan is equal to itself."""
    return json.dumps(value, default=str, sort_keys=True)


def _changes(
    old: dict, new: dict, fields: tuple[str, ...]
) -> list[tuple[str, str, object, object]]:
    """Compare two ``{key: record}`` maps: added, removed and changed fields."""
    out = []
    for key in sorted(set(old) | set(new)):
        if key not in old:
            out.append(('added', key, None, new[key].get('default', new[key].get('value'))))
        elif key not in new:
            out.append(('removed', key, old[key].get('default', old[key].get('value')), None))
        else:
            for field in fields:
                if _same(old[key].get(field)) != _same(new[key].get(field)):
                    out.append((field, key, old[key].get(field), new[key].get(field)))
    return out


def config_schema_map(text: str | None) -> dict:
    """Map each config path of a PROTEUS ``config_schema.json`` to its record."""
    return {f['path']: f for f in json.loads(text)['fields']} if text else {}


def output_schema_map(text: str | None) -> dict:
    """Map each helpfile column of a PROTEUS ``output_schema.json`` to its record."""
    return {k['name']: k for k in json.loads(text)['keys']} if text else {}


def flatten(table: dict, prefix: str = '') -> dict:
    """Flatten a nested TOML table into ``{'a.b.c': {'value': v}}``."""
    out = {}
    for key, value in table.items():
        path = f'{prefix}{key}'
        if isinstance(value, dict):
            out.update(flatten(value, path + '.'))
        else:
            out[path] = {'value': value}
    return out


def toml_map(text: str | None) -> dict:
    """Flatten a TOML file; a missing file gives an empty map."""
    return flatten(tomllib.loads(text)) if text else {}


def requirement_map(specs: list[str], group: str) -> dict:
    """Map ``group:package`` to its version specifier for a list of requirement strings."""
    out = {}
    for spec in specs:
        found = REQ_RE.match(spec)
        if found:
            out[f'{group}:{found.group(1).lower()}'] = {'value': found.group(3).strip()}
    return out


def pyproject_map(text: str | None) -> dict:
    """Dependency floors of a ``pyproject.toml``: runtime, extras and PROTEUS module pins."""
    data = tomllib.loads(text or '')
    project = data.get('project', {})
    out = requirement_map(project.get('dependencies', []), 'dependencies')
    for extra, specs in project.get('optional-dependencies', {}).items():
        out.update(requirement_map(specs, f'extra {extra}'))
    modules = data.get('tool', {}).get('proteus', {}).get('modules', {})
    out.update(flatten(modules, 'tool.proteus.modules.'))
    return out


def project_toml_map(text: str | None) -> dict:
    """Julia ``[compat]`` entries of a ``Project.toml``."""
    compat = tomllib.loads(text or '').get('compat', {})
    return {f'compat:{k}': {'value': v} for k, v in compat.items()}


def _short(value: object) -> object:
    """A JSON value of at most ``DESCRIPTION_CHARS`` characters; a date becomes text."""
    if not isinstance(value, (str, int, float, bool, type(None))):
        text = json.dumps(value, default=str)
        value = json.loads(text) if len(text) <= DESCRIPTION_CHARS else text
    if isinstance(value, str) and len(value) > DESCRIPTION_CHARS:
        return value[:DESCRIPTION_CHARS] + ' [cut]'
    return value


def diff_file(path: str, old: str | None, new: str | None) -> list[dict]:
    """Return the changes of one tracked file as fact records without ids.

    Parameters
    ----------
    path : str
        Repository path; it selects the parser.
    old, new : str or None
        File text before and after the merge; None when the file is absent.
    """
    if path == CONFIG_SCHEMA:
        kind, parse, fields = 'config', config_schema_map, CONFIG_FIELDS
    elif path == OUTPUT_SCHEMA:
        kind, parse, fields = 'output column', output_schema_map, ('unit',)
    elif path.endswith('pyproject.toml'):
        kind, parse, fields = 'dependency', pyproject_map, ('value',)
    elif path.endswith('Project.toml'):
        kind, parse, fields = 'dependency', project_toml_map, ('value',)
    else:
        kind, parse, fields = 'config', toml_map, ('value',)
    found = _changes(parse(old), parse(new), fields)
    return [
        {'file': path, 'kind': kind, 'change': c, 'key': k, 'old': _short(o), 'new': _short(n)}
        for c, k, o, n in found
    ]


def group_facts(facts: list[dict], limit: int = GROUP_LIMIT) -> list[dict]:
    """Merge more than ``limit`` additions or removals of one file into one fact.

    The merged fact's key is the common dotted prefix of the keys (or the file),
    and its ``new`` value names the count and the keys.
    """
    out, groups = [], {}
    for fact in facts:
        if fact['change'] in ('added', 'removed'):
            groups.setdefault((fact['file'], fact['change']), []).append(fact)
        else:
            out.append(fact)
    for (path, change), members in groups.items():
        if len(members) <= limit:
            out += members
            continue
        keys = [m['key'] for m in members]
        parts = os.path.commonprefix([k.split('.') for k in keys])
        names = ', '.join(keys)
        out.append(
            {
                'file': path,
                'kind': members[0]['kind'],
                'change': change,
                'key': '.'.join(parts) or path,
                'old': None,
                'new': _short(f'{len(keys)} keys {change}: {names}'),
                'members': keys,
            }
        )
    return out


def attach_facts(gh: GitHub, data: dict, settings: dict) -> None:
    """Add ``facts``, ``facts_unreadable`` and ``known_keys`` to the collected data in place.

    A tracked file that does not parse is recorded with its PR and error and gives
    no facts, so one broken file cannot stop every later run.
    """
    facts, unreadable = [], []
    for pr in data['prs'] + data.get('commits', []):
        if pr.get('status', 'merged') != 'merged' or not pr['parent_oid']:
            continue
        for path in [*settings.get('tracked', {}).get(pr['repo'], []), *TRACKED_ALWAYS]:
            if path not in pr['files'] and not pr.get('files_truncated'):
                continue
            try:
                old = gh.raw_file(pr['repo'], path, pr['parent_oid'])
                found = diff_file(path, old, gh.raw_file(pr['repo'], path, pr['merge_oid']))
            except UNREADABLE as err:
                unreadable.append(_unreadable(pr['ref'], path, err))
                continue
            for fact in group_facts(found):
                facts.append({'id': f'F{len(facts) + 1}', 'pr': pr['ref'], **fact})
    data['facts'], data['facts_unreadable'] = facts, unreadable
    data['known_keys'] = sorted(known_keys(gh, settings, unreadable))


def _unreadable(where: str, path: str, err: Exception) -> dict:
    return {'pr': where, 'file': path, 'error': f'{type(err).__name__}: {err}'[:200]}


def known_keys(gh: GitHub, settings: dict, unreadable: list[dict]) -> set[str]:
    """Config paths, helpfile columns and default-file keys that hold ``_`` or ``.``.

    A tracked file on the default branch that does not parse is added to ``unreadable``.
    """
    keys = set()
    parse = {CONFIG_SCHEMA: config_schema_map, OUTPUT_SCHEMA: output_schema_map}
    for repo, paths in settings.get('tracked', {}).items():
        for path in paths:
            try:
                text = gh.raw_file(repo, path, None)
                if text is None:
                    raise DigestError(f'cannot read {path} of {repo} for the key check')
                keys |= set(parse.get(path, toml_map)(text))
            except UNREADABLE as err:
                unreadable.append(_unreadable(f'{repo} default branch', path, err))
    return {k for k in keys if '_' in k or '.' in k}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: add facts to a collected JSON file in place."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('collected')
    parser.add_argument('--summary')
    args = parser.parse_args(argv)
    with open(args.collected) as fh:
        data = json.load(fh)
    attach_facts(GitHub(), data, load_settings())
    with open(args.collected, 'w') as fh:
        json.dump(data, fh)
    lines = [f'{u["pr"]} {u["file"]}: {u["error"]}' for u in data['facts_unreadable']]
    for line in lines:
        print(f'Not read: {json.dumps(line)}')
    print(f'{len(data["facts"])} facts, {len(data["known_keys"])} known keys')
    if args.summary and lines:
        with open(args.summary, 'a') as fh:
            text = html.escape('\n'.join(['Files that do not parse, no facts:', *lines]))
            fh.write(f'<pre>\n{text}\n</pre>\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
