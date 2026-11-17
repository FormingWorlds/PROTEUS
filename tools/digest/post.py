"""Post the rendered digest as a discussion, then ping Discord when a webhook is set.

Usage::

    python tools/digest/post.py decide --event EVENT --post-input BOOL --backdated BOOL
    python tools/digest/post.py publish OUTDIR
    python tools/digest/post.py alert MESSAGE

``decide`` prints ``live=true`` or ``live=false`` for ``$GITHUB_OUTPUT``. ``publish``
creates the discussion in the digest category of the first repository in the
settings, unless the newest digest post covers the same window or has the same
title (a rerun),
and then sends the link to ``DISCORD_WEBHOOK_URL``. Without that secret the Discord
step is skipped, and a failed send is a warning, because the post stands. ``alert``
sends a failure message to Discord, or does nothing without the secret.
"""

from __future__ import annotations

import http.client
import json
import os
import sys
import urllib.request
from pathlib import Path

from collect import parse_state, previous_state
from common import DigestError, GitHub, find_category, load_settings

CREATE = """
mutation($repo: ID!, $category: ID!, $title: String!, $body: String!) {
  createDiscussion(input: {repositoryId: $repo, categoryId: $category, title: $title, body: $body}) {
    discussion { url }
  }
}
"""


def decide(settings: dict, event: str, post_input: bool, backdated: bool) -> bool:
    """True only when posting is switched on and the run is a scheduled or confirmed live run."""
    return bool(settings['publish']) and not backdated and (event == 'schedule' or post_input)


def create_discussion(gh: GitHub, home: str, category: str, title: str, body: str) -> str:
    """Create the discussion and return its URL.

    Raises
    ------
    DigestError
        When the category does not exist.
    """
    repo, cat = find_category(gh, home, category)
    if cat is None:
        raise DigestError(f'no discussion category with slug "{category}" in {home}')
    out = gh.graphql(CREATE, retry=False, repo=repo, category=cat, title=title, body=body)
    return out['createDiscussion']['discussion']['url']


def discord(message: str, webhook: str | None = None) -> bool:
    """Send a message to the Discord webhook; return False when none is set or the send fails."""
    webhook = webhook if webhook is not None else os.environ.get('DISCORD_WEBHOOK_URL', '')
    if not webhook:
        print('No DISCORD_WEBHOOK_URL: Discord step skipped.')
        return False
    payload = json.dumps({'content': message, 'allowed_mentions': {'parse': []}}).encode()
    headers = {'Content-Type': 'application/json', 'User-Agent': 'proteus-weekly-digest'}
    try:
        req = urllib.request.Request(webhook, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=30):
            return True
    except (OSError, ValueError, http.client.HTTPException) as err:
        print(f'Warning: Discord ping failed: {err}')
        return False


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    args = argv or sys.argv[1:]
    settings = load_settings()
    if args[0] == 'decide':
        opts = dict(zip(args[1::2], args[2::2]))
        live = decide(
            settings,
            opts['--event'],
            opts['--post-input'] == 'true',
            opts['--backdated'] == 'true',
        )
        print(f'live={str(live).lower()}')
    elif args[0] == 'publish':
        post = json.loads((Path(args[1]) / 'digest.json').read_text())
        gh, home = GitHub(), settings['repos'][0]
        last = previous_state(gh, home, settings['category']) or {}
        until = (parse_state(post['body']) or {}).get('until')
        if (until and last.get('until') == until) or last.get('title') == post['title']:
            print(f'This window is already posted: {last.get("url")}')
            return 0
        url = create_discussion(gh, home, settings['category'], post['title'], post['body'])
        print(f'Posted {url}')
        discord(f'{post["title"]}: {url}')
    elif args[0] == 'alert':
        discord(args[1])
    else:
        raise SystemExit(f'unknown command {args[0]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
