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
import collections
import html
import http.client
import json
import re
import sys
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from common import API, DigestError, GitHub, find_category, iso, load_settings, parse_time

MEMBERS = ('OWNER', 'MEMBER', 'COLLABORATOR', 'CONTRIBUTOR')
REST_FILES = 300  # files in the REST answer for one commit
RAW_CHARS = 20000  # raw text read per title or body; the largest body_chars allowed
LIVE_ARTIFACT = 'digest-live'
WORKFLOW = '.github/workflows/weekly-digest.yml'
ARTIFACT_PAGES = 10  # pages of 100 artifacts searched for a trusted live digest
FALLBACK_DAYS = 35
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
        files(first: 100) { totalCount nodes { path changeType } }
        mergeCommit { oid parents(first: 1) { nodes { oid } } }
      }
    }"""
RELEASES_FIELD = """
    releases(first: 20, orderBy: {field: CREATED_AT, direction: DESC}) {
      nodes { tagName name url publishedAt isDraft isPrerelease description
              author { login __typename } }
    }
  }
}
"""
PR_QUERY += RELEASES_FIELD
COMMIT_QUERY = (
    """
query($owner: String!, $name: String!, $since: GitTimestamp!, $until: GitTimestamp!, $max: Int!) {
  repository(owner: $owner, name: $name) {
    isPrivate isArchived
    defaultBranchRef { target { ... on Commit {
      history(since: $since, until: $until, first: $max) {
        totalCount
        nodes { oid url committedDate messageHeadline messageBody additions deletions
                parents(first: 2) { totalCount nodes { oid } }
                author { user { login __typename } } }
      } } } }"""
    + RELEASES_FIELD
)

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


def dropped_char(c: str) -> bool:
    """Return True for a format (Cf) or control (Cc) character or one of U+E0000 to U+E007F."""
    return unicodedata.category(c) in ('Cf', 'Cc') or '\U000e0000' <= c <= '\U000e007f'


def clean_text(text: str | None, limit: int) -> str:
    """Remove links, images, HTML and mentions from untrusted text and cut it to ``limit``.

    Only the first ``RAW_CHARS`` characters are read. Format and control characters and
    U+E0000 to U+E007F go first; white space stays. An HTML comment ends only at ``-->``
    and an unclosed one removes the rest. A tag starts with ``<`` and an ASCII letter,
    ``/``, ``!`` or ``?``; every other ``<`` or ``>`` goes, and the text around it stays,
    as in ``T < 300 K``. ``@`` goes, then every word that holds ``://`` or ``www.`` in
    any case. The result has no ``<``, ``>``, ``@``, ``://`` or ``www.``. Text inside
    markup, such as attribute values or script content, can stay.

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
    text = re.sub(r'<[A-Za-z/!?][^<>]*>|[<>]', ' ', text).replace('@', '')
    text = ' '.join(w for w in text.split() if '://' not in w and 'www.' not in w.lower())
    return text[:limit]


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


def trusted_run(run: dict | None, home: str) -> bool:
    """True for a successful scheduled or manual run of the digest workflow on main of ``home``."""
    run = run or {}
    return (
        run.get('path') == WORKFLOW
        and run.get('event') in ('schedule', 'workflow_dispatch')
        and run.get('head_branch') == 'main'
        and (run.get('head_repository') or {}).get('full_name') == home
        and run.get('conclusion') == 'success'
    )


def live_doc_state(doc: object) -> dict | None:
    """The window state of a ``digest.json`` of ``digest-live``: only a live one with a valid end."""
    if not isinstance(doc, dict) or doc.get('live') is not True:
        return None
    try:
        parse_time(doc['until'])
    except (KeyError, TypeError, ValueError):
        return None
    return {
        'until': doc['until'],
        'run_id': doc.get('run_id'),
        'title': doc.get('title'),
        'url': doc.get('run_url'),
    }


def live_state(gh: GitHub, home: str) -> dict | None:
    """Return the state of the trusted ``digest-live`` with the latest window end, or None.

    Anybody who can open a pull request can upload an artifact with that name, so each
    candidate counts only when its run passes ``trusted_run``. GitHub lists artifacts by
    id, not by time, so every page is read and the latest ``until`` wins.
    """
    best = None
    for page in range(1, ARTIFACT_PAGES + 1):
        url = f'{API}/repos/{home}/actions/artifacts?name={LIVE_ARTIFACT}&per_page=100&page={page}'
        artifacts = (gh.get_json(url) or {}).get('artifacts', [])
        if not artifacts:
            break
        for art in artifacts:
            if art.get('expired'):
                continue
            run = gh.get_json(f'{API}/repos/{home}/actions/runs/{art["workflow_run"]["id"]}')
            state = trusted_run(run, home) and live_doc_state(
                gh.artifact_json(art, 'digest.json')
            )
            if state and (
                best is None or parse_time(state['until']) > parse_time(best['until'])
            ):
                best = state
    return best


def window_state(gh: GitHub, home: str, category: str, now: datetime) -> dict | None:
    """The state that starts the window: the newest live digest, else the last post marker.

    The post marker counts only while it ends at most ``FALLBACK_DAYS`` days before ``now``.
    """
    if state := live_state(gh, home):
        return state
    state = previous_state(gh, home, category)
    if state and parse_time(state['until']) < now - timedelta(days=FALLBACK_DAYS):
        print(
            f'::notice::The last post ends at {state["until"]}, more than {FALLBACK_DAYS} days '
            'ago; the window is the last 7 days'
        )
        return None
    return state


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
        'deleted': [f['path'] for f in node['files']['nodes'] if f['changeType'] == 'DELETED'],
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


def commit_record(repo: str, node: dict, since: str, until: str) -> dict | None:
    """Return a commit as a record when it is no merge commit and its commit date is in the window.

    The commit date in UTC, not the author date, places a commit, so a commit rebased
    onto the default branch in the window counts in it.
    """
    committed = iso(parse_time(node['committedDate']))
    if node['parents']['totalCount'] > 1 or not since <= committed < until:
        return None
    user = (node['author'] or {}).get('user') or {}
    parents = node['parents']['nodes']
    return {
        'repo': repo,
        'oid': node['oid'],
        'url': node['url'],
        'title': node['messageHeadline'],
        'raw_body': node['messageBody'] or '',
        'author': user.get('login'),
        'author_kind': user.get('__typename'),
        'committed_at': committed,
        'size': node['additions'] + node['deletions'],
        'merge_oid': node['oid'],
        'parent_oid': parents[0]['oid'] if parents else None,
    }


def collect_commits(gh: GitHub, repo: str, window: dict, cap: int) -> tuple[list, list, int]:
    """Return the commit and release records of a repository without pull requests.

    The newest ``cap`` commits of the window to the default branch are read, and each
    kept commit gets its changed files from one REST call. Refs count up from the
    oldest commit date (``PALEOS:c1``); on one date they keep the reverse order of the
    history, a parent before its child. The third value is the number of commits over the cap.

    Raises
    ------
    DigestError
        When the repository is private, archived, missing or has no default branch.
    """
    owner, name = repo.split('/')
    since, until = window['since'], window['until']
    asked = {'owner': owner, 'name': name, 'since': since, 'until': until, 'max': cap}
    data = gh.graphql(COMMIT_QUERY, **asked)['repository']
    if data is None or data['isPrivate'] or data['isArchived'] or not data['defaultBranchRef']:
        raise DigestError(f'{repo} is not a public, active repository')
    history = data['defaultBranchRef']['target']['history']
    commits = sorted(
        (c for n in reversed(history['nodes']) if (c := commit_record(repo, n, since, until))),
        key=lambda c: c['committed_at'],
    )
    for i, c in enumerate(commits, 1):
        files = gh.commit_files(repo, c['oid'])
        c['ref'] = f'{short_name(repo)}:c{i}'
        if files is None:
            print(
                f'Commit {c["ref"]} {c["oid"]}: no file list from GitHub; facts read every tracked file'
            )
        c['files'] = [f['filename'] for f in files or []]
        c['deleted'] = [f['filename'] for f in files or [] if f['status'] == 'removed']
        c['files_truncated'] = files is None or len(files) >= REST_FILES
    releases = [release_record(repo, n, since, until) for n in data['releases']['nodes']]
    return commits, [r for r in releases if r], history['totalCount'] - len(history['nodes'])


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


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """Give a redirect back as its 3xx answer, so only a page served at its own URL counts."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


PAGE_OPENER = urllib.request.build_opener(NoRedirect)


def page_status(url: str) -> int | str:
    """HTTP status of one HEAD request to a docs page, or the error text; it never raises."""
    try:
        req = urllib.request.Request(
            url, method='HEAD', headers={'User-Agent': 'weekly-digest'}
        )
        with PAGE_OPENER.open(req, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as err:
        return err.code
    except (OSError, http.client.HTTPException, ValueError) as err:
        return f'{type(err).__name__}: {err}'


def docs_pages(files: list[str], base: str | None, status) -> list[dict]:
    """Return the served docs pages among the files of a pull request.

    A file ``docs/<path>.md`` maps to ``<base>/<path>.html``; it is kept only when
    ``status(url)`` gives 200, so a page that is deleted or not yet deployed is left out.
    A page name of other characters than ASCII letters, digits, ``_``, space, ``.``, ``/``
    and ``-``, or with no letter or digit, is left out unasked, so its link is visible text.

    Parameters
    ----------
    files : list of str
        Paths of the pull request.
    base : str or None
        Docs site of the repository; None gives no pages.
    status : callable
        Maps a URL to its HTTP status, or to an error text.

    Returns
    -------
    list of dict
        ``{'path': <path>, 'url': <url>}`` in file order.
    """
    pages = []
    for path in files if base else []:
        if not (path.startswith('docs/') and path.endswith('.md')):
            continue
        page = path[len('docs/') : -len('.md')]
        if not re.fullmatch(r'[\w ./-]*[^\W_][\w ./-]*', page, re.ASCII):
            print(f'Docs page left out: {path!r} has a character a link text cannot hold')
            continue
        url = f'{base}/{urllib.parse.quote(page)}.html'
        if (code := status(url)) == 200:
            pages.append({'path': page, 'url': url})
        else:
            print(f'Docs page left out: {url} answered {code}')
    return pages


def collect(gh: GitHub, settings: dict, window: dict) -> dict:
    """Collect all repositories of the settings into one JSON-ready dict."""
    names = [short_name(r) for r in settings['repos']]
    if len(set(names)) != len(names):
        raise DigestError('two repositories share a name; references would be ambiguous')
    prs, releases, commits, over = [], [], [], {}
    for repo in settings['repos']:
        if repo in settings['commits_only']:
            got, got_releases, n = collect_commits(gh, repo, window, settings['max_commits'])
            commits += got
            if n:
                over[repo] = n
        else:
            got_prs, got_releases = collect_repo(gh, repo, window)
            prs += got_prs
        releases += got_releases
    for c in commits:
        c['title'] = clean_text(c['title'], 300)
        c['body'] = clean_text(c.pop('raw_body'), settings['body_chars'])
    dropped = [p for p in prs if p['status'] == 'open' and p['association'] not in MEMBERS]
    prs = [p for p in prs if p not in dropped]
    cut = add_bodies(prs, releases, settings['max_bodies'], settings['body_chars'])
    down, answers, left = set(), {}, collections.defaultdict(set)
    for p in prs:
        base = settings['docs'].get(p['repo']) if p['status'] == 'merged' else None

        def status(url: str, base: str | None = base) -> int | str:
            """One request per URL, and none to a docs site after a network error on it."""
            if url not in answers:
                if base in down:
                    return 'not asked: the site failed before'
                answers[url] = page_status(url)
                if isinstance(answers[url], str):
                    down.add(base)
            return answers[url]

        files = [f for f in p['files'] if f not in p['deleted']]
        p['docs'] = docs_pages(files, base, status)
        if base:
            kept = {f'docs/{d["path"]}.md' for d in p['docs']}
            left[base] |= {
                f for f in files if f.startswith('docs/') and f.endswith('.md')
            } - kept
    return {
        'window': window,
        'repos': settings['repos'],
        'prs': prs,
        'releases': releases,
        'commits': commits,
        'commits_left_out': over,
        'dropped_open': [
            {'ref': p['ref'], 'login': p['author'], 'association': p['association']}
            for p in dropped
        ],
        'docs_left_out': {site: len(paths) for site, paths in left.items() if paths},
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
    now = datetime.now(timezone.utc)
    state = None if args.since else window_state(gh, home, settings['category'], now)
    window = resolve_window(args.since, args.until, state, now)
    data = collect(gh, settings, window)
    with open(args.out, 'w') as fh:
        json.dump(data, fh)
    print(f'{len(data["prs"])} PRs, {len(data["releases"])} releases in {window}')
    for c in data.get('commits', []):
        print(f'Commit {c["ref"]} {c["oid"]} {c["committed_at"]}')
    lines = (
        [
            f'Open pull request left out: {d["ref"]} by {d["login"]} (association {d["association"]})'
            for d in data.get('dropped_open', [])
        ]
        + [
            f'Docs pages left out of {site}: {n}'
            for site, n in data.get('docs_left_out', {}).items()
        ]
        + [
            f'Commits left out of {repo}: {n}'
            for repo, n in data.get('commits_left_out', {}).items()
        ]
    )
    if lines:
        print(*lines, sep='\n')
    if args.summary and lines:
        with open(args.summary, 'a') as fh:
            fh.write(f'<pre>\n{html.escape(chr(10).join(lines))}\n</pre>\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
