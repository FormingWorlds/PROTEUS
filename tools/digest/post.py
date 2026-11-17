"""Decide whether to post the rendered digest, and post it as a discussion.

Usage::

    python tools/digest/post.py decide collected.json --event EVENT --post-input BOOL
    python tools/digest/post.py publish collected.json draft.json OUTDIR

``decide`` prints ``live=true`` or ``live=false`` for ``$GITHUB_OUTPUT``. ``publish``
repeats the guards (posting switched on, a window that is not back-dated, a draft
that passes the check), then creates the discussion in the digest category of the
first repository in the settings. It skips with a notice when the newest digest
post has the same run id, window end or title: one post per run and per meeting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from check import check
from collect import parse_state, previous_state, trusted
from common import DigestError, GitHub, find_category, load_settings

CREATE = """
mutation($repo: ID!, $category: ID!, $title: String!, $body: String!) {
  createDiscussion(input: {repositoryId: $repo, categoryId: $category, title: $title, body: $body}) {
    discussion { url author { login __typename } }
  }
}
"""


def decide(settings: dict, event: str, post_input: bool, backdated: bool) -> bool:
    """True only when posting is switched on and the run is a scheduled or confirmed live run."""
    return settings['publish'] and not backdated and (event == 'schedule' or post_input)


def create_discussion(gh: GitHub, home: str, category: str, title: str, body: str) -> str:
    """Create the discussion and return its URL.

    Raises
    ------
    DigestError
        When the category does not exist, or when the new post has an author that
        the state reader does not trust, so the next run would not find it.
    """
    repo, cat = find_category(gh, home, category)
    if cat is None:
        raise DigestError(f'no discussion category with slug "{category}" in {home}')
    out = gh.graphql(CREATE, retry=False, repo=repo, category=cat, title=title, body=body)
    made = out['createDiscussion']['discussion']
    if not trusted(made['author']):
        raise DigestError(
            f'posted {made["url"]} as {made["author"]}, which the next run ignores'
        )
    return made['url']


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
        )
        print(f'live={str(live).lower()}')
        return 0
    if args[0] != 'publish':
        raise SystemExit(f'unknown command {args[0]}')
    fails = check(data, json.loads(Path(args[2]).read_text()))
    if not settings['publish'] or data['window']['backdated'] or fails:
        raise DigestError(
            f'not posted: publish={settings["publish"]}, back-dated={data["window"]["backdated"]}, '
            f'{len(fails)} check failures'
        )
    post = json.loads((Path(args[3]) / 'digest.json').read_text())
    gh, home = GitHub(), settings['repos'][0]
    last = previous_state(gh, home, settings['category']) or {}
    state = parse_state(post['body']) or {}
    pairs = (
        ('run_id', state.get('run_id')),
        ('until', state.get('until')),
        ('title', post['title']),
    )
    same = [k for k, v in pairs if v and last.get(k) == v]
    if same:
        print(
            f'::notice::Not posted: the newest digest {last.get("url")} has the same {", ".join(same)}'
        )
        return 0
    url = create_discussion(gh, home, settings['category'], post['title'], post['body'])
    print(f'Posted {url}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
