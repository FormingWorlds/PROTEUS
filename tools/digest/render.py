"""Render the checked draft as the markdown of the digest post.

Usage::

    python tools/digest/render.py collected.json draft.json OUTDIR [--live] [--run-url URL]

Writes ``OUTDIR/digest.md`` (title and body, for reading) and ``OUTDIR/digest.json``
(``title``, ``body`` for the post). Settings, values, links, dates and mentions
come from the collected data, never from the model text. Without ``--live`` the
mentions are shown in code spans, so a dry run notifies nobody.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from common import is_bot, load_settings, parse_time

SECTIONS = {
    'results': 'Changes that affect your results',
    'capabilities': 'New capabilities',
    'install': 'Installing and running',
    'data': 'Data and licensing',
    'design': 'Design choices to look at',
    'open': 'Open for review',
}
MAX_FACTS = 4
VALUE_CHARS = 60


def tuesday_title(until: str) -> str:
    """Title with the date of the first Tuesday on or after ``until`` in Amsterdam time."""
    day = parse_time(until).astimezone(ZoneInfo('Europe/Amsterdam')).date()
    day += timedelta(days=(1 - day.weekday()) % 7)
    return f'PROTEUS weekly digest, {day.day} {day:%B %Y}'


def short_date(stamp: str) -> str:
    """``2026-09-06T21:07:16Z`` as ``6 Sep``."""
    value = parse_time(stamp)
    return f'{value.day} {value:%b}'


def plain(text: str) -> str:
    """Model text for a table cell or paragraph: dashes to commas, no pipes or line breaks."""
    text = re.sub(r'\s*[–—]\s*', ', ', text)
    return re.sub(r'\s+', ' ', text.replace('|', '/')).strip()


def value(item: object) -> str:
    """A fact value as a short code span."""
    text = re.sub(r'[`|\n]', ' ', str(item))[:VALUE_CHARS]
    return f'`{text}`'


def fact_line(fact: dict) -> str:
    """One fact in words, for example ``grain_size``: ``0.1`` to ``0.001``."""
    key, change = value(fact['key']), fact['change']
    if fact.get('members'):
        where = fact['key'] if '/' not in fact['key'] else fact['key'].rsplit('/', 1)[1]
        return f'{len(fact["members"])} keys {change} in {value(where)}'
    if change == 'added':
        return f'{key} added' + (
            f', default {value(fact["new"])}' if fact['new'] is not None else ''
        )
    if change == 'removed':
        return f'{key} removed'
    if change == 'description':
        return f'{key}: description changed'
    label = '' if change in ('default', 'value') else f' {change}'
    return f'{key}{label}: {value(fact["old"])} to {value(fact["new"])}'


def facts_text(facts: list[dict]) -> str:
    """At most ``MAX_FACTS`` fact lines joined by semicolons."""
    lines = list(dict.fromkeys(fact_line(f) for f in facts))
    if len(lines) > MAX_FACTS:
        lines = lines[:MAX_FACTS] + [f'and {len(lines) - MAX_FACTS} more']
    return '; '.join(lines)


class Mentions:
    """Mention each person once per post, never a bot, never in a dry run."""

    def __init__(self, live: bool, bots: list[str], announced: set[str]):
        self.live, self.bots, self.announced, self.done = live, bots, announced, set()

    def name(self, item: dict) -> str:
        """Render the author of a cited pull request or release."""
        login = item['author']
        if is_bot(login, item['author_kind'], self.bots):
            return login or 'unknown'
        if login in self.done or item['ref'] in self.announced:
            return login
        self.done.add(login)
        return f'@{login}' if self.live else f'`@{login}`'


def links(items: list[dict]) -> str:
    """Markdown links to the cited pull requests and releases."""
    return ', '.join(f'[{i["ref"].replace("@", " ")}]({i["url"]})' for i in items)


def authors(items: list[dict], mentions: Mentions) -> str:
    """Authors of the cited items, each once, in order of first appearance."""
    seen, names = set(), []
    for item in items:
        if item['author'] not in seen:
            seen.add(item['author'])
            names.append(mentions.name(item))
    return ', '.join(names)


def render_body(
    data: dict, draft: dict, live: bool, run_url: str, bots: list[str]
) -> tuple[str, dict]:
    """Return the markdown body and the state marker of the next post."""
    window, prev = data['window'], data['window'].get('previous') or {}
    announced_before = set(prev.get('announced_open', []))
    mentions = Mentions(live, bots, announced_before)
    prs = {p['ref']: p for p in data['prs']}
    rels = {r['ref']: r for r in data['releases']}
    facts = {f['id']: f for f in data.get('facts', [])}
    lines = [
        'This digest is written automatically by a workflow in FormingWorlds/PROTEUS from merged '
        'pull requests, open pull requests and releases. It is a selection; check the linked pull '
        'requests before you act on it.',
        '',
    ]
    if not live:
        lines[:0] = [
            '**Dry run: not posted. Mentions are shown in code spans and notify nobody.**',
            '',
        ]
    if draft['empty']:
        lines.append(
            f'Quiet week: no pull request was merged and no release was published between '
            f'{window["since"]} and {window["until"]} (UTC).'
        )
    announced_now = set()
    for key, heading in SECTIONS.items():
        entries = [e for e in draft['entries'] if e['section'] == key]
        if not entries:
            continue
        lines += [f'## {heading}', '']
        if key == 'results':
            lines += [
                'Runs started after these merges can give different results than earlier runs with '
                'the same config.',
                '',
                '| Setting | Change | Merged | Links | By |',
                '| --- | --- | --- | --- | --- |',
            ]
        for entry in entries:
            items = [prs[r] for r in entry['prs']] + [rels[r] for r in entry['releases']]
            cited = [facts[f] for f in entry['facts']]
            if key == 'open':
                announced_now |= {i['ref'] for i in items}
            text = plain(entry['text'])
            detail = facts_text(cited)
            if key == 'results':
                when = min(i.get('merged_at') or i['published_at'] for i in items)
                change = f'{text} {detail}.' if detail else text
                lines.append(
                    f'| {plain(entry["title"])} | {change} | {short_date(when)} | {links(items)} | '
                    f'{authors(items, mentions)} |'
                )
                continue
            lines += [f'### {plain(entry["title"])}', '', text, '']
            if detail:
                lines += [f'Settings: {detail}.', '']
            lines += [f'{links(items)}, by {authors(items, mentions)}', '']
        lines.append('')
    open_now = {r for r, p in prs.items() if p['status'] == 'open'}
    state = {
        'until': window['until'],
        'announced_open': sorted(announced_now | (announced_before & open_now)),
    }
    merged = sum(p['status'] == 'merged' for p in data['prs'])
    names = ', '.join(r.split('/', 1)[1] for r in data['repos'])
    lines += [
        '---',
        '',
        f'Collected between {window["since"]} and {window["until"]} (UTC) from {merged} merged '
        f'and {len(open_now)} open pull requests and {len(data["releases"])} releases in {names}. '
        f'The model read the bodies of the {data["bodies_kept"]} largest '
        f'pull requests and only the titles of the other {data["bodies_cut"]}. '
        f'[Workflow run]({run_url}).',
        '',
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
    args = parser.parse_args(argv)
    data = json.loads(Path(args.collected).read_text())
    draft = json.loads(Path(args.draft).read_text())
    body, _ = render_body(data, draft, args.live, args.run_url, load_settings()['bots'])
    title = tuesday_title(data['window']['until'])
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'digest.md').write_text(f'# {title}\n\n{body}')
    (out / 'digest.json').write_text(json.dumps({'title': title, 'body': body}))
    print(title)
    return 0


if __name__ == '__main__':
    sys.exit(main())
