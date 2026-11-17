"""Check the model draft against the collected data; any failure stops the run.

Usage::

    python tools/digest/check.py collected.json draft.json

Exit code 1 and one line per failure when a check fails. Every pull request,
release, fact and login in the draft must come from the collected data; the text
must hold no link, mention, HTML or reference; every number must appear in a
cited source, and every config key or helpfile column in a cited fact.
"""

from __future__ import annotations

import json
import math
import re
import sys

NUMBER_RE = re.compile(r'(?<![\w.])\d+(?:\.\d+)*(?:[eE][-+]?\d+)?')
TOKEN_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_.]*[A-Za-z0-9_]')
FORBIDDEN = {
    'a URL': re.compile(r'(?:https?|ftp)://|www\.', re.I),
    'an @': re.compile(r'@'),
    'HTML': re.compile(r'<\s*[A-Za-z/!]'),
    'a markdown link or image': re.compile(r'\]\s*\(|!\['),
    'a # reference': re.compile(r'#\s*\d'),
    'a backtick': re.compile(r'`'),
}


def numbers(text: str) -> set[str]:
    """Return the number tokens of a text, for example ``0.8``, ``1.13.0``, ``v1.0.0`` or ``1e-3``."""
    return set(NUMBER_RE.findall(re.sub(r'\bv(?=\d)', ' ', text)))


def number_in(token: str, sources: set[str]) -> bool:
    """True when ``token`` equals a source token as text or as a float."""
    if token in sources:
        return True
    try:
        value = float(token)
    except ValueError:
        return False
    for src in sources:
        try:
            if math.isclose(value, float(src), rel_tol=1e-9):
                return True
        except ValueError:
            continue
    return False


def leaves(keys: set[str]) -> set[str]:
    """Last segments of dotted keys that hold an underscore, for example ``grain_size``."""
    return {k.rsplit('.', 1)[-1] for k in keys if '_' in k.rsplit('.', 1)[-1]}


def check_entry(entry: dict, data: dict, index: dict) -> list[str]:
    """Return the failures of one draft entry."""
    name = f'entry "{entry.get("title", "")[:40]}"'
    fails = []
    prs = [index['prs'].get(r) for r in entry['prs']]
    rels = [index['releases'].get(r) for r in entry['releases']]
    facts = [index['facts'].get(f) for f in entry['facts']]
    for kind, refs, found in (
        ('PR', entry['prs'], prs),
        ('release', entry['releases'], rels),
        ('fact', entry['facts'], facts),
    ):
        fails += [
            f'{name}: {kind} {r} is not in the collected set'
            for r, x in zip(refs, found)
            if not x
        ]
    prs, rels, facts = [p for p in prs if p], [r for r in rels if r], [f for f in facts if f]
    if not prs and not rels:
        fails.append(f'{name}: cites no pull request or release')
    want_open = entry['section'] == 'open'
    fails += [
        f'{name}: {p["ref"]} is {p["status"]}, not allowed in section {entry["section"]}'
        for p in prs
        if (p['status'] == 'open') != want_open
    ]
    if want_open and rels:
        fails.append(f'{name}: a release in section open')
    if entry['section'] == 'results' and not facts and not rels:
        fails.append(f'{name}: a results entry cites no fact or release')
    cited = {p['ref'] for p in prs}
    fails += [
        f'{name}: fact {f["id"]} belongs to {f["pr"]}, not cited'
        for f in facts
        if f['pr'] not in cited
    ]
    authors = {x['author'] for x in prs + rels}
    fails += [
        f'{name}: login {x} is not an author of a cited item'
        for x in entry['logins']
        if x not in authors
    ]
    text = f'{entry["title"]} {entry["text"]}'
    fails += [f'{name}: text holds {what}' for what, rx in FORBIDDEN.items() if rx.search(text)]
    sources = set()
    for item in prs + rels:
        sources |= numbers(
            f'{item.get("title", "")} {item.get("name", "")} {item.get("body", "")}'
        )
    for fact in facts:
        sources |= numbers(f'{fact["key"]} {fact["old"]} {fact["new"]}')
    fails += [
        f'{name}: number {n} is in no cited source'
        for n in sorted(numbers(text))
        if not number_in(n, sources)
    ]
    fact_keys = {f['key'] for f in facts} | {k for f in facts for k in f.get('members', [])}
    allowed = fact_keys | leaves(fact_keys)
    known = index['keys'] | leaves(index['keys'])
    fails += [
        f'{name}: key {t} is in no cited fact'
        for t in sorted(set(TOKEN_RE.findall(text)))
        if t in known and t not in allowed
    ]
    return fails


def check(data: dict, draft: dict) -> list[str]:
    """Return every failure of the draft; an empty list means it may be rendered."""
    empty = not data['releases'] and not any(p['status'] == 'merged' for p in data['prs'])
    if draft.get('empty') != empty:
        return [
            f'draft empty flag {draft.get("empty")} does not match the window (empty={empty})'
        ]
    if not empty and not draft['entries']:
        return ['the model returned no entries for a window with merged pull requests']
    index = {
        'prs': {p['ref']: p for p in data['prs']},
        'releases': {r['ref']: r for r in data['releases']},
        'facts': {f['id']: f for f in data.get('facts', [])},
        'keys': set(data.get('known_keys', [])),
    }
    fails = []
    for entry in draft['entries']:
        fails += check_entry(entry, data, index)
    return fails


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    src, draft_path = argv or sys.argv[1:3]
    with open(src) as fh:
        data = json.load(fh)
    with open(draft_path) as fh:
        draft = json.load(fh)
    fails = check(data, draft)
    for line in fails:
        print(f'FAIL {line}')
    print(f'{len(fails)} failures in {len(draft["entries"])} entries')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
