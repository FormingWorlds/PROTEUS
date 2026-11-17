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
import unicodedata

from common import SECTIONS

NUMBER_RE = re.compile(
    r'(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)*(?:[eE][-+]?\d+)?)'
)
TOKEN_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_.]*[A-Za-z0-9_]')
FORBIDDEN = {
    'a URL': re.compile(r'(?:https?|ftp)://|www\.|//|mailto:|javascript:', re.I),
    'an @': re.compile('[@\uff20\ufe6b]'),
    'HTML': re.compile(r'<\s*[A-Za-z/!]'),
    'an HTML entity': re.compile(r'&#?[A-Za-z0-9]+;'),
    'a markdown link or image': re.compile(r'[\[\]]'),
    'a # reference': re.compile(r'#\s*\d|\bGH-\d', re.I),
    'a backtick': re.compile(r'`'),
    'a commit id': re.compile(r'\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b', re.I),
    'a superscript': re.compile('[\u00b2\u00b3\u00b9\u2070-\u207f]'),
}
MINUS = '\u2212\u02d7'
FIELDS = {
    'section': str,
    'title': str,
    'text': str,
    'prs': list,
    'releases': list,
    'facts': list,
    'logins': list,
}


def numbers(text: str) -> set[str]:
    """Return the number tokens of a text, for example ``-0.8``, ``1.13.0``, ``v1.0.0``, ``1,000`` or ``1e-3``."""
    text = unicodedata.normalize('NFKC', re.sub(r'\bv(?=\d)', ' ', text))
    text = ''.join('-' if c in MINUS or unicodedata.category(c) == 'Pd' else c for c in text)
    return set(NUMBER_RE.findall(re.sub(r'(?<![\w.])([a-z])(?=\d)', r'\1 ', text)))


def _float(token: str) -> float | None:
    try:
        return float(token.replace(',', ''))
    except ValueError:
        return None


def number_in(token: str, sources: set[str]) -> bool:
    """True when ``token`` equals a source token as text or as a float, sign included."""
    value = _float(token)
    if token in sources or value is None:
        return token in sources
    return any(
        v is not None and math.isclose(value, v, rel_tol=1e-9) for v in map(_float, sources)
    )


def key_forms(keys: set[str]) -> set[str]:
    """Lower-case keys, their dotted suffixes, and their last segments that hold an underscore."""
    out = set()
    for key in keys:
        parts = key.lower().split('.')
        out |= {
            '.'.join(parts[i:])
            for i in range(len(parts))
            if i < len(parts) - 1 or '_' in parts[-1]
        }
    return out


def is_empty(data: dict) -> bool:
    """True when the window has no merged pull request and no release."""
    return not data['releases'] and not any(p['status'] == 'merged' for p in data['prs'])


def shape_failures(entry: object) -> list[str]:
    """Missing or mistyped fields of one entry, and an unknown section."""
    if not isinstance(entry, dict):
        return ['an entry is not an object']
    name = f'entry "{str(entry.get("title", ""))[:40]}"'
    fails = [
        f'{name}: field {k} is missing or not a {t.__name__} of strings'
        for k, t in FIELDS.items()
        if not isinstance(entry.get(k), t)
        or (t is list and not all(isinstance(x, str) for x in entry[k]))
    ]
    if not fails and entry['section'] not in SECTIONS:
        fails.append(f'{name}: unknown section {entry["section"]}')
    return fails


def _has(word: str, text: str) -> bool:
    return bool(re.search(rf'(?<!\w){re.escape(word)}(?!\w)', text, re.I))


def fact_value_failures(name: str, text: str, facts: list[dict]) -> list[str]:
    """Sentences that state a new setting with the old string value of a cited change.

    A sentence fails when it holds the old value and "now" or "default" but not the
    new value; the old value as an ordinary word elsewhere passes.
    """
    fails = []
    sentences = re.split(r'[.;!?](?:\s|$)', text)
    for fact in facts:
        old, new = (str(v).strip('\'"') for v in (fact['old'], fact['new']))
        if (
            fact['change'] not in ('default', 'value')
            or not 3 <= len(old) <= 40
            or _float(old) is not None
        ):
            continue
        if any(
            _has(old, s) and not _has(new, s) and re.search(r'\b(now|default)', s, re.I)
            for s in sentences
        ):
            fails.append(f'{name}: text holds the old value "{old}" of fact {fact["id"]}')
    return fails


def check_entry(entry: dict, index: dict) -> list[str]:
    """Return the failures of one draft entry with a valid shape."""
    name = f'entry "{entry["title"][:40]}"'
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
    kinds = {unicodedata.category(c) for c in text if c not in '\t\n\r'}
    if kinds & {'Cf', 'Cc'}:
        fails.append(f'{name}: text holds an invisible character')
    if any(unicodedata.category(c) == 'No' and not '\u2080' <= c <= '\u2089' for c in text):
        fails.append(f'{name}: text holds a special number character')
    sources = set()
    for item in prs + rels:
        sources |= numbers(
            f'{item.get("title", "")} {item.get("name", "")} {item.get("body", "")}'
        )
    fails += [
        f'{name}: number {n} is in no cited source'
        for n in sorted(numbers(text))
        if not number_in(n, sources)
    ]
    allowed = key_forms(
        {f['key'] for f in facts} | {k for f in facts for k in f.get('members', [])}
    )
    for token in sorted({t.lower() for t in TOKEN_RE.findall(text)}):
        forms = key_forms({token}) | {token}
        if forms & index['keys'] and not forms & allowed:
            fails.append(f'{name}: key {token} is in no cited fact')
    return fails + fact_value_failures(name, text, facts)


def check(data: dict, draft: dict) -> list[str]:
    """Return every failure of the draft; an empty list means it may be rendered."""
    empty = is_empty(data)
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
        'keys': key_forms(set(data.get('known_keys', []))),
    }
    fails = []
    for entry in draft['entries']:
        fails += shape_failures(entry) or check_entry(entry, index)
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
