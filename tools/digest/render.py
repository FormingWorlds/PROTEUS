"""Render the checked draft as the markdown of the digest post.

Usage::

    python tools/digest/render.py collected.json draft.json OUTDIR [--live] [--run-url URL] [--run-id ID]

Writes ``OUTDIR/digest.md`` (title and body, for reading) and ``OUTDIR/digest.json``
(``title``, ``body`` for the post). Settings, values, links, dates and author
names come from the collected data, never from the model text. Without ``--live``
the body starts with the dry-run line, which stops the post step.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
import urllib.parse
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from common import SECTIONS, DigestError, count, is_bot, load_settings, parse_time

MAX_FACTS = 4
MAX_DOCS = 5
GITHUB = 'https://github.com/'
VALUE_CHARS = 60
BODY_LIMIT = 65536
LINK_RE = re.compile(r'\[((?:\\.|[^\\\[\]])*)\]\(([^()\s]*)\)')
CODE_RE = re.compile(r'(?<!\\)`[^`\n]*`')
MARKER_RE = re.compile(r'\n<!-- weekly-digest-state \{[^\n<>]*\} -->\n*\Z')
MENTION_RE = re.compile(r'(?<![\w`])@([A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)(?![\w/-])')
LITERAL = {
    'a URL': re.compile(r'://|www\.|mailto:|xmpp:|javascript:', re.I),
    'an @': re.compile('[@\uff20\ufe6b]'),
    'a # reference': re.compile(r'(?<!#)#\d|\bGH-\d', re.I),
    'a commit id': re.compile(r'(?<![\w.-])[0-9a-f]{7,40}(?![\w-])', re.I),
}
RAW = {
    'HTML': re.compile(r'<\s*[A-Za-z/!?]'),
    'an HTML entity': re.compile(r'&#?[A-Za-z0-9]+;'),
    'a bracket': re.compile(r'[\[\]]'),
    'a backtick': re.compile('`'),
}


def tuesday_title(until: str) -> str:
    """Title with the Tuesday of the Monday-to-Sunday week of ``until`` in Amsterdam time."""
    day = parse_time(until).astimezone(ZoneInfo('Europe/Amsterdam')).date()
    day += timedelta(days=1 - day.weekday())
    return f'PROTEUS weekly digest, {day.day} {day:%B %Y}'


def window_dates(window: dict) -> str:
    """``1 and 29 September 2026 (UTC ...)``: first and last day of the window ``[since, until)``."""
    first = parse_time(window['since'])
    last = max(first, parse_time(window['until']) - timedelta(microseconds=1))
    if first.date() == last.date():
        days = f'{first.day} {first:%B %Y}'
    elif (first.year, first.month) == (last.year, last.month):
        days = f'{first.day} and {last.day} {last:%B %Y}'
    elif first.year == last.year:
        days = f'{first.day} {first:%B} and {last.day} {last:%B %Y}'
    else:
        days = f'{first.day} {first:%B %Y} and {last.day} {last:%B %Y}'
    return f'{days} (UTC {window["since"]} to {window["until"]})'


def short_date(stamp: str) -> str:
    """``2026-09-06T21:07:16Z`` as ``6 Sep``."""
    value = parse_time(stamp)
    return f'{value.day} {value:%b}'


def escape(text: str) -> str:
    """Backslash-escape the markdown characters of untrusted text, so it renders as text only."""
    text = re.sub(r'([\\`*_{}\[\]()<>#&!~$])', r'\\\1', text)
    text = re.sub(r'^(\d+)\.(?=\s|$)', r'\1\\.', text)
    return re.sub(r'^([-+=])', r'\\\1', text)


def plain(text: str) -> str:
    """Model text for a table cell or paragraph: one line, no dashes or pipes, markdown escaped."""
    text = re.sub(r'\s*[\u2013\u2014]\s*', ', ', text)
    return escape(re.sub(r'\s+', ' ', text.replace('|', '/')).strip())


def value(item: object) -> str:
    """A fact value as a short code span with no line break, pipe, backtick or comment delimiter."""
    if item is None:
        return '(none)'
    if not str(item).strip():
        return '(empty)'
    text = re.sub(r'<!--|--!?>|[`|\s]', ' ', str(item))[:VALUE_CHARS]
    return f'`{text}`' if text.strip() else '(not shown)'


def fact_line(fact: dict) -> str:
    """One fact in words, for example ``grain_size``: ``0.1`` to ``0.001``."""
    key, change = value(fact['key']), fact['change']
    if fact.get('members'):
        return (
            f'{len(fact["members"])} keys {change} in {value(fact["key"].rsplit("/", 1)[-1])}'
        )
    if change == 'added':
        return f'{key} added' + (
            f', default {value(fact["new"])}' if fact['new'] not in (None, '') else ''
        )
    if change == 'removed':
        return f'{key} removed'
    if change == 'description':
        return f'{key}: description changed'
    label = '' if change in ('default', 'value') else f' {change}'
    return f'{key}{label}: {value(fact["old"])} → {value(fact["new"])}'


def facts_text(facts: list[dict]) -> str:
    """At most ``MAX_FACTS`` fact lines joined by semicolons."""
    lines = list(dict.fromkeys(fact_line(f) for f in facts))
    if len(lines) > MAX_FACTS:
        lines = lines[:MAX_FACTS] + [f'and {len(lines) - MAX_FACTS} more']
    return '; '.join(lines)


def author_link(item: dict, bots: list[str]) -> str:
    """The author of a cited item: a link to the GitHub profile, or a code span for a bot."""
    login = item['author']
    if not login:
        return 'unknown'
    if is_bot(login, item['author_kind'], bots):
        return f'`{login}`'
    return f'[{link_text(login)}]({link_url(GITHUB + login)})'


def link_url(url: str) -> str:
    """The URL of a markdown link, quoted so it has no space or parenthesis."""
    return urllib.parse.quote(url, safe=':/#?=&%@+.,-_~')


def link_text(text: str) -> str:
    """Trimmed one-line link text with no ``@``, pipe or control character, markdown escaped."""
    text = ''.join(' ' if unicodedata.category(c) in ('Cc', 'Cf') else c for c in text)
    text = re.sub(r'\s+', ' ', text.replace('@', ' ').replace('|', '/')).strip()
    return re.sub(r'([\\\[\]()`<>*_])', r'\\\1', text)


def ref_text(item: dict) -> str:
    """Link text of a cited item: its ref, or for a commit the repository name and the 7-character id."""
    return (
        f'{item["repo"].split("/", 1)[1]} {item["oid"][:7]}' if 'oid' in item else item['ref']
    )


def links(items: list[dict]) -> str:
    """Markdown links to the cited pull requests, releases and commits."""
    return ', '.join(f'[{link_text(ref_text(i))}]({link_url(i["url"])})' for i in items)


def docs_links(items: list[dict]) -> str:
    """Links to the served docs pages of the cited pull requests, at most ``MAX_DOCS``."""
    pages = list({d['url']: d for i in items for d in i.get('docs', [])}.values())
    texts = [f'[{link_text(d["path"])}]({link_url(d["url"])})' for d in pages[:MAX_DOCS]]
    if len(pages) > MAX_DOCS:
        more = len(pages) - MAX_DOCS
        texts.append(f'and {more} more page' + ('s' if more > 1 else ''))
    return ', '.join(texts)


def scan_body(body: str, urls: list[str], logins: list[str]) -> list[str]:
    """Return the problems of a rendered body: a link or mention that the data does not allow.

    Code spans, the state marker and the links to ``urls`` are removed first. In
    the rest, a URL, ``@`` or ``#`` reference fails even when escaped, because
    GitHub links them in plain text; markup characters fail when not escaped.
    """
    allowed, problems = {link_url(u) for u in urls}, []

    def link(match: re.Match) -> str:
        if match.group(2) not in allowed:
            problems.append(f'a link to {match.group(2)}')
        return ' '

    def mention(match: re.Match) -> str:
        if match.group(1) not in logins:
            problems.append(f'a mention of {match.group(1)}')
        return ' '

    rest = LINK_RE.sub(link, CODE_RE.sub(' ', MARKER_RE.sub('\n', body)))
    literal = MENTION_RE.sub(mention, re.sub(r'\\([!-/:-@\[-`{-~])', r'\1', rest))
    raw = re.sub(r'\\.', ' ', rest)
    problems += [what for what, rx in LITERAL.items() if rx.search(literal)]
    return problems + [what for what, rx in RAW.items() if rx.search(raw)]


def authors(items: list[dict], bots: list[str]) -> str:
    """Authors of the cited items, each once, in order of first appearance."""
    first = {}
    for item in items:
        first.setdefault(item['author'], item)
    return ', '.join(author_link(item, bots) for item in first.values())


def allowed_urls(data: dict, run_url: str, bots: list[str]) -> list[str]:
    """Every URL the body may link: cited items, the run, repositories, profiles and docs pages."""
    items = data['prs'] + data['releases'] + data.get('commits', [])
    return (
        [i['url'] for i in items]
        + [run_url]
        + [GITHUB + r for r in data['repos']]
        + [
            GITHUB + i['author']
            for i in items
            if not is_bot(i['author'], i['author_kind'], bots)
        ]
        + [d['url'] for p in data['prs'] for d in p.get('docs', [])]
    )


def not_read(data: dict) -> str:
    """One sentence per repository on its commits of the window over ``max_commits``."""
    return ''.join(
        f' {count(n, "more commit")} of {r.split("/", 1)[1]} in the window '
        f'{"is" if n == 1 else "are"} not read.'
        for r, n in data.get('commits_left_out', {}).items()
    )


def summary(data: dict, run_url: str) -> str:
    """The window, the counts, the linked repositories and the run link that open the post."""
    merged = sum(p['status'] == 'merged' for p in data['prs'])
    opened = sum(p['status'] == 'open' for p in data['prs'])
    names = ', '.join(
        f'[{link_text(r.split("/", 1)[1])}]({link_url(GITHUB + r)})' for r in data['repos']
    )
    per = collections.Counter(c['repo'].split('/', 1)[1] for c in data.get('commits', []))
    also = ''.join(
        f' The digest also covers {count(n, "commit")} to the default branch of {name}, '
        'a repository without pull requests.'
        for name, n in per.items()
    )
    kept, cut = data['bodies_kept'], data['bodies_cut']
    other = (
        'the title of the other pull request' if cut == 1 else f'the titles of the other {cut}'
    )
    if not cut:
        read = 'the text of every pull request'
    elif not kept:
        read = (
            'only the title of the pull request'
            if cut == 1
            else f'only the titles of the {cut} pull requests'
        )
    elif kept == 1:
        read = f'the body of the largest pull request and only {other}'
    else:
        read = f'the bodies of the {kept} largest pull requests and only {other}'
    return (
        f'Collected between {window_dates(data["window"])} from '
        f'{count(merged, "merged pull request")}, {count(opened, "open pull request")} '
        f'and {count(len(data["releases"]), "release")} in {names}.{also}{not_read(data)} '
        f'The model read {read}. [Workflow run]({run_url}).'
    )


def render_body(
    data: dict, draft: dict, live: bool, run_url: str, bots: list[str], run_id: str = ''
) -> tuple[str, dict]:
    """Return the markdown body and the state marker of the next post."""
    window, home = data['window'], data['repos'][0]
    prs = {p['ref']: p for p in data['prs']}
    rels = {r['ref']: r for r in data['releases']}
    cms = {c['ref']: c for c in data.get('commits', [])}
    facts = {f['id']: f for f in data.get('facts', [])}
    lines = [
        summary(data, run_url),
        '',
        'Additions and changes from across the ecosystem that affect more than one project. '
        'This is a selection, not a complete change list.',
        '',
        f'This digest is written automatically by a workflow in [{link_text(home)}]({link_url(GITHUB + home)}) from '
        'merged pull requests, open pull requests, releases and the commits of repositories '
        'without pull requests; check the linked items before you act on it.',
        '',
    ]
    if not live:
        lines[:0] = ['**Dry run: not posted.**', '']
    if draft['empty']:
        left = not_read(data)
        lines.append(
            'Quiet week: no pull request was merged, no release was published and no commit '
            f'was {"kept" if left else "read"} between {window_dates(window)}.{left}'
        )
        lines.append('')
    for key, heading in SECTIONS.items():
        entries = [e for e in draft['entries'] if e['section'] == key]
        if not entries:
            continue
        lines += [f'## {heading}', '']
        if key == 'results':
            lines += [
                'Runs started after these merges will not reproduce earlier runs with the same '
                'config. Compare old and new runs with care, and check whether any figures in '
                'progress need a rerun.',
                '',
                '| Setting | Change | Merged | Links |',
                '| --- | --- | --- | --- |',
            ]
        for entry in entries:
            items = [prs[r] for r in entry['prs']] + [rels[r] for r in entry['releases']]
            items += [cms[r] for r in entry['commits']]
            cited = [facts[f] for f in entry['facts']]
            text = plain(entry['text'])
            detail = facts_text(cited)
            if key == 'results':
                when = min(
                    i.get('merged_at') or i.get('committed_at') or i['published_at']
                    for i in items
                )
                change = f'{text} {detail}.' if detail else text
                docs = f'; docs: {pages};' if (pages := docs_links(items)) else ','
                lines.append(
                    f'| {plain(entry["title"])} | {change} | {short_date(when)} | {links(items)}'
                    f'{docs} by {authors(items, bots)} |'
                )
                continue
            if entry.get('action'):
                text = f'**Action for everyone:** {text}'
            lines += [f'### {plain(entry["title"])}', '', text, '']
            if detail:
                lines += [f'Settings: {detail}.', '']
            if key == 'capabilities' and (pages := docs_links(items)):
                lines += [f'Docs: {pages}.', '']
            lines += [f'{links(items)}, by {authors(items, bots)}', '']
        lines.append('')
    state = {'until': window['until'], 'run_id': run_id}
    lines += [
        f'<!-- weekly-digest-state {json.dumps(state, separators=(",", ":"))} -->',
    ]
    return '\n'.join(lines).replace('\n\n\n', '\n\n') + '\n', state


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('collected')
    parser.add_argument('draft')
    parser.add_argument('outdir')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--run-url', default='')
    parser.add_argument('--run-id', default='')
    args = parser.parse_args(argv)
    data = json.loads(Path(args.collected).read_text())
    draft = json.loads(Path(args.draft).read_text())
    bots = load_settings()['bots']
    body, _ = render_body(data, draft, args.live, args.run_url, bots, args.run_id)
    if len(body) > BODY_LIMIT:
        raise DigestError(f'the body has {len(body)} characters, longer than {BODY_LIMIT}')
    title = tuesday_title(data['window']['until'])
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'digest.md').write_text(f'# {title}\n\n{body}')
    (out / 'digest.json').write_text(json.dumps({'title': title, 'body': body}))
    print(title)
    return 0


if __name__ == '__main__':
    sys.exit(main())
