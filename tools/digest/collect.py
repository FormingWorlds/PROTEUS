"""Collect the merged and open pull requests and the releases of the digest window.

Usage::

    python tools/digest/collect.py --out collected.json [--since DATE] [--until DATE]

Without ``--since`` the window starts where the last digest post of
``github-actions`` ended (its state marker), or 7 days before ``--until``.
A given ``--since`` or ``--until`` marks the window as back-dated, which keeps
the run a dry run. Pull-request and release text is untrusted: links, images,
HTML and mentions are removed before anything reaches the model.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import unicodedata
from datetime import datetime, timedelta, timezone

from common import DigestError, GitHub, find_category, iso, load_settings, parse_time

MEMBERS = ('OWNER', 'MEMBER', 'COLLABORATOR', 'CONTRIBUTOR')
RAW_CHARS = 20000  # raw text read per title or body; the largest body_chars allowed
STATE_RE = re.compile(r'<!-- weekly-digest-state (\{.*?\}) -->', re.S)

PR_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    isPrivate isArchived defaultBranchRef { name }
    pullRequests(first: 50, after: $cursor, states: [OPEN, MERGED],
                 orderBy: {field: UPDATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number url title body isDraft state createdAt mergedAt updatedAt baseRefName authorAssociation
        additions deletions
        author { login __typename }
        labels(first: 10) { nodes { name } }
        files(first: 100) { totalCount nodes { path } }
        mergeCommit { oid parents(first: 1) { nodes { oid } } }
      }
    }
    releases(first: 20, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes { tagName name url publishedAt isDraft isPrerelease description
              author { login __typename } }
    }
  }
}
"""

POSTS_QUERY = """
query($owner: String!, $name: String!, $category: ID!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 50, after: $cursor, categoryId: $category,
                orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes { url title createdAt body author { login __typename } }
    }
  }
}
"""

PR_STATE_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) { pullRequest(number: $number) { state } }
}
"""


def dropped_char(c: str) -> bool:
    """Return True for a format (Cf) or control (Cc) character or one of U+E0000 to U+E007F."""
    return unicodedata.category(c) in ('Cf', 'Cc') or '\U000e0000' <= c <= '\U000e007f'


def clean_text(text: str | None, limit: int) -> str:
    """Remove links, images, HTML and mentions from untrusted text and cut it to ``limit``.

    Only the first ``RAW_CHARS`` characters are read. Format and control characters and
    U+E0000 to U+E007F go first; white space stays. An HTML comment ends
    only at ``-->`` and an unclosed one removes the rest. A tag starts with ``<`` and a
    letter, ``/``, ``!`` or ``?``; every other ``<`` or ``>`` goes, and the text around it
    stays, as in ``T < 300 K``. ``@`` goes, then every token with ``://`` or ``www.``.
    The result has no ``<``, ``>``, ``@``, ``://`` or ``www.``. Text inside markup, such
    as attribute values or script content, can stay.

    Parameters
    ----------
    text : str or None
        A pull-request or release body or title.
    limit : int
        Maximum number of characters kept; 0 keeps nothing.

    Returns
    -------
    str
        Plain text with markdown link text kept.
    """
    if not text or limit <= 0:
        return ''
    text = ''.join(c for c in text[:RAW_CHARS] if c.isspace() or not dropped_char(c))
    text = re.sub(r'<!--(?:-?>|.*?-->|.*)', ' ', text, flags=re.S)
    text = re.sub(r'!\[[^\]]*\]\([^)]*\)', ' ', text)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'<[A-Za-z/!?][^<>]*>', ' ', text)
    text = re.sub(r'[<>]', ' ', text).replace('@', '')
    text = re.sub(r'www\.\S*', ' ', ' '.join(w for w in text.split() if '://' not in w))
    return ' '.join(text.split())[:limit]


def short_name(repo: str) -> str:
    """Return the repository name without its owner, as used in references."""
    return repo.split('/', 1)[1]


def parse_state(body: str) -> dict | None:
    """Return the state marker at the end of a post body, or None.

    Only the last marker counts, so a marker that appears inside the content of a
    post (for example in a fact value) cannot replace the real one.
    """
    found = STATE_RE.findall(body or '')
    try:
        state = json.loads(found[-1]) if found else None
        parse_time(state['until'])
    except (json.JSONDecodeError, TypeError, KeyError, ValueError):
        return None
    return state


def trusted(author: dict | None) -> bool:
    """True for the account that the workflow posts as: ``github-actions``, a bot."""
    author = author or {}
    return author.get('login') == 'github-actions' and author.get('__typename') == 'Bot'


def previous_state(gh: GitHub, home: str, category: str) -> dict | None:
    """Return the state marker of the newest digest post written by ``github-actions``.

    A marker in a post by any other author is ignored, because anybody can write a
    discussion in the category. Pages are read until a trusted post with a valid
    marker is found.
    """
    _, cat = find_category(gh, home, category)
    if cat is None:
        return None
    owner, name = home.split('/')
    cursor = None
    while True:
        page = gh.graphql(POSTS_QUERY, owner=owner, name=name, category=cat, cursor=cursor)
        posts = page['repository']['discussions']
        for post in posts['nodes']:
            if trusted(post['author']) and (state := parse_state(post['body'])):
                return {**state, 'url': post['url'], 'title': post.get('title')}
        if not posts['pageInfo']['hasNextPage']:
            return None
        cursor = posts['pageInfo']['endCursor']


def still_open(gh: GitHub, repos: list[str], refs: list[str]) -> list[str]:
    """Return the announced pull requests (``name#number``) that GitHub reports open."""
    full = {short_name(r): r for r in repos}
    out = []
    for ref in refs:
        name, number = ref.split('#')
        if name not in full:
            print(
                f'Announced pull request {ref} leaves the state: its repository is not in the settings'
            )
            continue
        owner, repo = full[name].split('/')
        try:
            found = gh.graphql(PR_STATE_QUERY, owner=owner, name=repo, number=int(number))
        except DigestError as err:
            if not re.search(r'NOT_FOUND|Could not resolve', str(err)):
                out.append(ref)
            print(f'Announced pull request {ref} cannot be read: {err}')
            continue
        if ((found['repository'] or {}).get('pullRequest') or {}).get('state') == 'OPEN':
            out.append(ref)
    return out


def resolve_window(
    since: str | None, until: str | None, state: dict | None, now: datetime
) -> dict:
    """Decide the window ``[since, until)`` and whether it is back-dated."""
    end = parse_time(until) if until else now
    if since:
        start = parse_time(since)
    elif state and not until:
        start = parse_time(state['until'])
    else:
        start = end - timedelta(days=7)
    if start >= end:
        raise DigestError(f'empty window: {iso(start)} is not before {iso(end)}')
    return {'since': iso(start), 'until': iso(end), 'backdated': bool(since or until)}


def pr_record(repo: str, node: dict, since: str, until: str) -> dict | None:
    """Return a pull request as a record when it was merged in the window or open at its end."""
    merged = node['mergedAt']
    if merged and since <= merged < until:
        status = 'merged'
    elif node['createdAt'] < until and (not merged or merged >= until) and not node['isDraft']:
        status = 'open'
    else:
        return None
    author = node['author'] or {}
    commit = node['mergeCommit'] if status == 'merged' else None
    parents = (commit or {}).get('parents', {}).get('nodes', [])
    return {
        'ref': f'{short_name(repo)}#{node["number"]}',
        'repo': repo,
        'number': node['number'],
        'url': node['url'],
        'status': status,
        'title': node['title'],
        'raw_body': node['body'] or '',
        'author': author.get('login'),
        'author_kind': author.get('__typename'),
        'association': node['authorAssociation'],
        'created_at': node['createdAt'],
        'merged_at': merged if status == 'merged' else None,
        'size': node['additions'] + node['deletions'],
        'labels': [label['name'] for label in node['labels']['nodes']],
        'files': [f['path'] for f in node['files']['nodes']],
        'files_truncated': node['files'].get('totalCount', 0) > len(node['files']['nodes']),
        'merge_oid': commit['oid'] if commit else None,
        'parent_oid': parents[0]['oid'] if parents else None,
    }


def release_record(repo: str, node: dict, since: str, until: str) -> dict | None:
    """Return a published release of the window as a record."""
    published = node['publishedAt']
    if node['isDraft'] or not published or not since <= published < until:
        return None
    author = node['author'] or {}
    return {
        'ref': f'{short_name(repo)}@{node["tagName"]}',
        'repo': repo,
        'tag': node['tagName'],
        'url': node['url'],
        'name': node['name'] or node['tagName'],
        'prerelease': node['isPrerelease'],
        'raw_body': node['description'] or '',
        'author': author.get('login'),
        'author_kind': author.get('__typename'),
        'published_at': published,
    }


def collect_repo(gh: GitHub, repo: str, window: dict) -> tuple[list, list]:
    """Return the pull-request and release records of one repository.

    Raises
    ------
    DigestError
        When the repository is private, archived or missing.
    """
    owner, name = repo.split('/')
    since, until = window['since'], window['until']
    prs, releases, cursor = [], None, None
    while True:
        data = gh.graphql(PR_QUERY, owner=owner, name=name, cursor=cursor)['repository']
        if data is None or data['isPrivate'] or data['isArchived']:
            raise DigestError(f'{repo} is not a public, active repository')
        if releases is None:
            releases = [
                release_record(repo, n, since, until) for n in data['releases']['nodes']
            ]
        page, main = data['pullRequests'], (data.get('defaultBranchRef') or {}).get('name')
        prs += [
            pr_record(repo, n, since, until)
            for n in page['nodes']
            if n['updatedAt'] >= since and n.get('baseRefName', main) == main
        ]
        oldest = page['nodes'][-1]['updatedAt'] if page['nodes'] else ''
        if not page['pageInfo']['hasNextPage'] or oldest < since:
            break
        cursor = page['pageInfo']['endCursor']
    return [p for p in prs if p], [r for r in releases if r]


def add_bodies(prs: list, releases: list, max_bodies: int, body_chars: int) -> dict:
    """Clean every title, and keep cleaned bodies only for the ``max_bodies`` largest PRs.

    Returns
    -------
    dict
        ``{'bodies_kept': n, 'bodies_cut': m}`` for the post footer.
    """
    ranked = sorted(prs, key=lambda p: (-p['size'], p['ref']))
    keep = {p['ref'] for p in ranked[:max_bodies]}
    for item in prs + releases:
        item['title' if 'title' in item else 'name'] = clean_text(
            item.get('title', item.get('name')), 300
        )
        limit = body_chars if item['ref'] in keep or 'tag' in item else 0
        item['body'] = clean_text(item.pop('raw_body'), limit)
    return {'bodies_kept': len(keep), 'bodies_cut': len(prs) - len(keep)}


def collect(gh: GitHub, settings: dict, window: dict) -> dict:
    """Collect all repositories of the settings into one JSON-ready dict."""
    names = [short_name(r) for r in settings['repos']]
    if len(set(names)) != len(names):
        raise DigestError('two repositories share a name; references would be ambiguous')
    prs, releases = [], []
    for repo in settings['repos']:
        got_prs, got_releases = collect_repo(gh, repo, window)
        prs += got_prs
        releases += got_releases
    dropped = [p for p in prs if p['status'] == 'open' and p['association'] not in MEMBERS]
    prs = [p for p in prs if p not in dropped]
    cut = add_bodies(prs, releases, settings['max_bodies'], settings['body_chars'])
    return {
        'window': window,
        'repos': settings['repos'],
        'prs': prs,
        'releases': releases,
        'dropped_open': [
            {'ref': p['ref'], 'login': p['author'], 'association': p['association']}
            for p in dropped
        ],
        **cut,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', required=True)
    parser.add_argument('--since')
    parser.add_argument('--until')
    parser.add_argument('--summary')
    args = parser.parse_args(argv)
    settings = load_settings()
    gh = GitHub()
    home = settings['repos'][0]
    state = None if args.since else previous_state(gh, home, settings['category'])
    window = resolve_window(args.since, args.until, state, datetime.now(timezone.utc))
    window['previous'] = state
    data = collect(gh, settings, window)
    data['still_open'] = still_open(
        gh, settings['repos'], (state or {}).get('announced_open', [])
    )
    with open(args.out, 'w') as fh:
        json.dump(data, fh)
    print(f'{len(data["prs"])} PRs, {len(data["releases"])} releases in {window}')
    lines = [
        f'Open pull request left out: {d["ref"]} by {d["login"]} (association {d["association"]})'
        for d in data.get('dropped_open', [])
    ]
    if lines:
        print(*lines, sep='\n')
    if args.summary and lines:
        with open(args.summary, 'a') as fh:
            fh.write(f'<pre>\n{html.escape(chr(10).join(lines))}\n</pre>\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
