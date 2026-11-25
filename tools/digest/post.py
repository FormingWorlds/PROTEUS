"""Decide whether a run is live, and deposit the live digest as the artifact ``digest-live``.

Usage::

    python tools/digest/post.py decide collected.json --event EVENT --post-input BOOL
    python tools/digest/post.py deposit collected.json draft.json OUTDIR --event EVENT --post-input BOOL --run-url URL --live-dir DIR --output FILE

``decide`` prints ``live=true`` or ``live=false`` for ``$GITHUB_OUTPUT``; it is false
off the default branch and in a rerun. ``deposit``
repeats every guard (posting switched on, a run on the default branch, a window that
is not back-dated, a scheduled run or a ticked post box, a draft that passes the
check, a body that is not a dry run, a body whose links all come from
``render.allowed_urls`` and that mentions nobody), then writes ``digest.json`` and
``digest.md`` to DIR, which the workflow uploads as ``digest-live``. It deposits
nothing, with a notice, in a rerun (``GITHUB_RUN_ATTEMPT`` above 1), and when the
newest live digest has the same run id or title, or a window end that is not earlier
than this one: one live digest per run and per meeting. It writes ``deposited=true`` or ``deposited=false`` to FILE.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from check import check
from collect import live_state, parse_state
from common import DigestError, GitHub, load_settings, parse_time
from render import allowed_urls, scan_body


def live_event(event: str, post_input: bool) -> bool:
    """True for a scheduled run or a manual run with the post box ticked."""
    return event == 'schedule' or post_input


def decide(
    settings: dict, event: str, post_input: bool, backdated: bool, ref: str, attempt: str
) -> bool:
    """True only for a first attempt on main of a scheduled or confirmed live run with posting on.

    A run off main or a rerun deposits nothing, so its body is rendered as a dry run.
    """
    return (
        settings['publish']
        and not backdated
        and live_event(event, post_input)
        and ref == 'refs/heads/main'
        and attempt == '1'
    )


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = argv or sys.argv[1:]
    settings = load_settings()
    data = json.loads(Path(args[1]).read_text())
    if args[0] == 'decide':
        opts = dict(zip(args[2::2], args[3::2]))
        live = decide(
            settings,
            opts['--event'],
            opts['--post-input'] == 'true',
            data['window']['backdated'],
            os.environ.get('GITHUB_REF', ''),
            os.environ.get('GITHUB_RUN_ATTEMPT', '1'),
        )
        print(f'live={str(live).lower()}')
        return 0
    if args[0] != 'deposit':
        raise SystemExit(f'unknown command {args[0]}')
    opts = dict(zip(args[4::2], args[5::2]))
    fails = check(data, json.loads(Path(args[2]).read_text()))
    post = json.loads((Path(args[3]) / 'digest.json').read_text())
    problems = scan_body(
        post['body'], allowed_urls(data, opts.get('--run-url', ''), settings['bots']), []
    )
    for problem in problems:
        print(f'Link problem: {json.dumps(problem)}')
    reasons = [
        reason
        for reason, bad in (
            ('posting is switched off', not settings['publish']),
            ('not on the default branch', os.environ.get('GITHUB_REF') != 'refs/heads/main'),
            ('the window is back-dated', data['window']['backdated']),
            ('not a live run', not live_event(opts['--event'], opts['--post-input'] == 'true')),
            (f'{len(fails)} check failures', fails),
            ('the body is a dry run', post['body'].startswith('**Dry run')),
            (f'{len(problems)} link problems', problems),
        )
        if bad
    ]
    if reasons:
        raise DigestError(f'not deposited: {", ".join(reasons)}')
    gh, home = GitHub(), settings['repos'][0]
    last = live_state(gh, home) or {}
    mine = {**(parse_state(post['body']) or {}), 'title': post['title']}
    same = [
        f'the same {k.replace("_", " ")}'
        for k in ('run_id', 'title')
        if mine.get(k) and last.get(k) == mine[k]
    ]
    rerun = os.environ.get('GITHUB_RUN_ATTEMPT', '1') != '1'
    if last.get('until') and parse_time(mine['until']) <= parse_time(last['until']):
        same.append('a window end that is not earlier')
    if same:
        print(
            f'::notice::Not deposited: the newest live digest of run {last.get("run_id")} has {", ".join(same)}'
        )
    if rerun:
        print('::notice::Not deposited: this is a rerun; start a new manual run instead')
    if not same and not rerun:
        doc = {
            'live': True,
            'run_id': mine.get('run_id', ''),
            'run_url': opts.get('--run-url', ''),
            'event': opts['--event'],
            'since': data['window']['since'],
            'until': data['window']['until'],
            'title': post['title'],
            'body': post['body'],
        }
        live = Path(opts['--live-dir'])
        live.mkdir(parents=True, exist_ok=True)
        (live / 'digest.json').write_text(json.dumps(doc))
        (live / 'digest.md').write_text(post['body'])
        print(f'Deposited {post["title"]} for the artifact digest-live')
    with open(opts['--output'], 'a') as fh:
        fh.write(f'deposited={str(not same and not rerun).lower()}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
