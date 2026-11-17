"""Mechanical facts of the digest: config, output and dependency changes per merged PR.

Usage::

    python tools/digest/facts.py collected.json

For every merged PR that touches a tracked file, the file before and after the
merge commit is compared; each change becomes a fact with an id ``F<n>``. The
model cites facts by id and never writes a setting, a value or a floor itself.
"""

from __future__ import annotations

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
REQ_RE = re.compile(r'^\s*([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?\s*(.*)$')


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
                if old[key].get(field) != new[key].get(field):
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
        kind = 'config'
        found = _changes(
            config_schema_map(old),
            config_schema_map(new),
            ('default', 'choices', 'bounds', 'description'),
        )
    elif path == OUTPUT_SCHEMA:
        kind = 'output column'
        found = _changes(output_schema_map(old), output_schema_map(new), ('unit',))
    elif path.endswith('pyproject.toml'):
        kind = 'dependency'
        found = _changes(pyproject_map(old), pyproject_map(new), ('value',))
    elif path.endswith('Project.toml'):
        kind = 'dependency'
        found = _changes(project_toml_map(old), project_toml_map(new), ('value',))
    else:
        kind = 'config'
        found = _changes(toml_map(old), toml_map(new), ('value',))
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
    """Add ``facts`` and ``known_keys`` to the collected data in place."""
    facts = []
    for pr in data['prs']:
        if pr['status'] != 'merged' or not pr['parent_oid']:
            continue
        for path in [*settings.get('tracked', {}).get(pr['repo'], []), *TRACKED_ALWAYS]:
            if path not in pr['files'] and not pr.get('files_truncated'):
                continue
            old = gh.raw_file(pr['repo'], path, pr['parent_oid'])
            new = gh.raw_file(pr['repo'], path, pr['merge_oid'])
            for fact in group_facts(diff_file(path, old, new)):
                facts.append({'id': f'F{len(facts) + 1}', 'pr': pr['ref'], **fact})
    data['facts'] = facts
    data['known_keys'] = sorted(known_keys(gh, settings))


def known_keys(gh: GitHub, settings: dict) -> set[str]:
    """Config paths, helpfile columns and default-file keys that hold ``_`` or ``.``."""
    keys = set()
    for repo, paths in settings.get('tracked', {}).items():
        for path in paths:
            text = gh.raw_file(repo, path, None)
            if text is None:
                raise DigestError(f'cannot read {path} of {repo} for the key check')
            if path == CONFIG_SCHEMA:
                keys |= set(config_schema_map(text))
            elif path == OUTPUT_SCHEMA:
                keys |= set(output_schema_map(text))
            else:
                keys |= set(toml_map(text))
    return {k for k in keys if '_' in k or '.' in k}


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: add facts to a collected JSON file in place."""
    path = (argv or sys.argv[1:])[0]
    with open(path) as fh:
        data = json.load(fh)
    attach_facts(GitHub(), data, load_settings())
    with open(path, 'w') as fh:
        json.dump(data, fh)
    print(f'{len(data["facts"])} facts, {len(data["known_keys"])} known keys')
    return 0


if __name__ == '__main__':
    sys.exit(main())
