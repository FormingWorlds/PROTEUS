"""Check the model draft against the collected data; any failure stops the run.

Usage::

    python tools/digest/check.py collected.json draft.json [--out FILE] [--summary FILE]

Exit code 1 and one line per failure when a check fails. Each failure line holds
model text as one JSON string, so it cannot start a workflow command; the
summary copy is an HTML-escaped ``<pre>`` block, so it renders as plain text. Every pull request,
release, fact and login in the draft must come from the collected data; the text
must hold no link, mention, HTML or reference; every number must appear in a
cited source, and every config key or helpfile column in a cited fact.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
import unicodedata

from common import DIGEST_DIR, SECTIONS

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
    'the action marker': re.compile(r'action\W+for\W+everyone', re.I),
}
MINUS = '\u2212\u02d7'
ELEMENTS = set(
    'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As '
    'Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd '
    'Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am '
    'Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split()
)
PREFIX_RE = re.compile(r'(?<![\w.])[A-Za-z](?=\d)(?:[\w.]*\w)?')
ASSIGN_RE = re.compile(
    r'(?:\b(?:set|changed?|lower(?:s|ed)?|rais(?:es|ed)|reduc(?:es|ed)|increas(?:es|ed)|'
    r'revert(?:s|ed)?|switch(?:es|ed)?|defaults?|moved?)\s+to|\bbecomes?|\b(?:is|are)(?:\s+now)?)'
    r'\s+(?:the\s+)?$',
    re.I,
)
OLD_MARK_RE = re.compile(
    r'\b(?:from|instead\s+of|rather\s+than|previously|replac\w*|used\s+to)\b', re.I
)
CLAIM_RE = re.compile(
    r'(?:\b(?:is|are)\s+now|\bnow\s+(?:is|are|uses?)|\bdefaults?\s+to|\bset\s+to|\bbecomes?)'
    r'\s+[\w-]+\s*(?:[,:(]|$)',
    re.I,
)
_ITEM = json.loads((DIGEST_DIR / 'entry_schema.json').read_text())['properties']['entries']
LIMITS = {
    k: v['maxLength'] for k, v in _ITEM['items']['properties'].items() if 'maxLength' in v
}
FIELDS = {
    'section': str,
    'title': str,
    'text': str,
    'prs': list,
    'releases': list,
    'facts': list,
    'logins': list,
}


def _split_prefix(match: re.Match) -> str:
    word = match.group(0)
    symbols = re.findall(r'[A-Z][a-z]?', word)
    if re.fullmatch(r'(?:[A-Z][a-z]?\d*)+', word) and set(symbols) <= ELEMENTS:
        return word
    return f'{word[0]} {word[1:]}'


def prepare(text: str) -> str:
    """Normalise a text for number matching: NFKC, one minus sign, prefixes split from digits.

    A single letter or an underscore before digits is split off (``v1.0``, ``X100``,
    ``mode_42``); a chemical formula of element symbols (``H2O``) stays whole.
    """
    text = unicodedata.normalize('NFKC', text)
    text = ''.join('-' if c in MINUS or unicodedata.category(c) == 'Pd' else c for c in text)
    return re.sub(r'_(?=\d)', '_ ', PREFIX_RE.sub(_split_prefix, text))


def numbers(text: str) -> set[str]:
    """Return the number tokens of a text, for example ``-0.8``, ``1.13.0``, ``v1.0.0``, ``1,000`` or ``1e-3``."""
    return set(NUMBER_RE.findall(prepare(text)))


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
    if 'action' in entry and not isinstance(entry['action'], bool):
        fails.append(f'{name}: field action is not a boolean')
    if not fails and entry['section'] not in SECTIONS:
        fails.append(f'{name}: unknown section {entry["section"]}')
    if not fails and entry.get('action') and entry['section'] != 'install':
        fails.append(f'{name}: action is only for section install')
    return fails


def _find(value: str, sentence: str) -> int | None:
    """Position of ``value`` in a prepared sentence (numbers compared as numbers), or None."""
    if _float(value) is not None:
        found = (m for m in NUMBER_RE.finditer(sentence) if number_in(m.group(), {value}))
    else:
        found = re.finditer(rf'(?<!\w){re.escape(value)}(?!\w)', sentence, re.I)
    return next((m.start() for m in found), None)


def fact_value_failures(name: str, parts: list[str], facts: list[dict]) -> list[str]:
    """Sentences that give a cited setting a value other than its new value.

    For a default or value change, a sentence fails when it holds the old value but
    not the new one, with "now" or "default" in it or an assignment ("set to",
    "becomes", "is") just before the old value, and no "from", "instead of" or
    similar before it. For a string value, a sentence that names the key and ends a
    clause with one word after "is now", "defaults to" or similar, while it does not
    hold the new value, also fails ("the mixing length is now linear"). The
    title and the text are split into sentences separately.
    """
    fails = []
    sentences = [s for p in parts for s in re.split(r'[.;!?](?:\s|$)', prepare(p))]
    for fact in facts:
        if fact['change'] not in ('default', 'value'):
            continue
        old, new = (str(v).strip('\'"') for v in (fact['old'], fact['new']))
        key = re.escape(fact['key'].rsplit('.', 1)[-1]).replace('_', '[ _]')
        for s in sentences:
            at = _find(old, s) if _float(old) is not None or 3 <= len(old) <= 40 else None
            if (
                at is not None
                and _find(new, s) is None
                and not OLD_MARK_RE.search(s[:at])
                and (re.search(r'\b(now|default)', s, re.I) or ASSIGN_RE.search(s[:at]))
            ):
                fails.append(f'{name}: text holds the old value "{old}" of fact {fact["id"]}')
                break
            if (
                _float(new) is None
                and CLAIM_RE.search(s)
                and re.search(rf'(?<!\w){key}(?!\w)', s, re.I)
                and _find(new, s) is None
            ):
                fails.append(
                    f'{name}: text gives a value other than the new value "{new}" of fact {fact["id"]}'
                )
                break
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
    fails += [
        f'{name}: {k} is longer than {n} characters'
        for k, n in LIMITS.items()
        if len(entry[k]) > n
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
    return fails + fact_value_failures(name, [entry['title'], entry['text']], facts)


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
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('collected')
    parser.add_argument('draft')
    parser.add_argument('--out')
    parser.add_argument('--summary')
    args = parser.parse_args(argv)
    with open(args.collected) as fh:
        data = json.load(fh)
    with open(args.draft) as fh:
        draft = json.load(fh)
    fails = check(data, draft)
    report = ''.join(f'FAIL {json.dumps(line)}\n' for line in fails)
    report += f'{len(fails)} failures in {len(draft["entries"])} entries\n'
    print(report, end='')
    if args.out:
        with open(args.out, 'w') as fh:
            fh.write(report)
    if args.summary:
        with open(args.summary, 'a') as fh:
            fh.write(f'<pre>\n{html.escape(report)}</pre>\n')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
