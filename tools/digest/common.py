"""Shared helpers of the weekly ecosystem digest: settings, GitHub API access, times.

The digest scripts use only the standard library, so the workflow needs no install
step beyond Python itself.
"""

from __future__ import annotations

import http.client
import json
import os
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DIGEST_DIR = Path(__file__).resolve().parent
API = 'https://api.github.com'
TRACKED_ALWAYS = ('pyproject.toml', 'Project.toml')
SECTIONS = {
    'results': 'Changes that affect your results',
    'capabilities': 'New capabilities',
    'install': 'Installing and running',
    'data': 'Data and licensing',
    'design': 'Design choices to look at',
    'open': 'In review, coming soon',
}
CATEGORY_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    id
    discussionCategories(first: 50) { nodes { id slug } }
  }
}
"""


class DigestError(RuntimeError):
    """A condition that stops the digest run before anything is posted."""


def load_settings(path: Path | None = None) -> dict:
    """Read ``repos.toml``.

    Parameters
    ----------
    path : Path, optional
        Settings file; the default is ``repos.toml`` next to this module.

    Returns
    -------
    dict
        The parsed settings.

    Raises
    ------
    DigestError
        When a setting is missing or has the wrong type or range.
    """
    with open(path or DIGEST_DIR / 'repos.toml', 'rb') as fh:
        return validate_settings(tomllib.load(fh))


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_settings(settings: dict) -> dict:
    """Check the types and ranges of the settings and return them unchanged."""
    rules = {
        'publish': lambda v: isinstance(v, bool),
        'category': lambda v: isinstance(v, str) and v.strip() != '',
        'model': lambda v: isinstance(v, str) and v.strip() != '',
        'max_bodies': lambda v: _is_int(v) and 0 <= v <= 500,
        'body_chars': lambda v: _is_int(v) and 1 <= v <= 20000,
        'bots': lambda v: isinstance(v, list) and all(isinstance(b, str) for b in v),
        'repos': lambda v: (
            isinstance(v, list)
            and len(v) > 0
            and all(isinstance(r, str) and r.count('/') == 1 for r in v)
        ),
        'tracked': lambda v: (
            isinstance(v, dict)
            and all(
                isinstance(p, list) and all(isinstance(x, str) for x in p) for p in v.values()
            )
        ),
    }
    for key, ok in rules.items():
        if key not in settings or not ok(settings[key]):
            raise DigestError(f'setting {key} is missing or invalid: {settings.get(key)!r}')
    return settings


def parse_time(text: str) -> datetime:
    """Parse an ISO date or date-time; a bare date means 00:00 UTC of that day.

    Parameters
    ----------
    text : str
        For example ``2026-09-01`` or ``2026-09-01T12:00:00Z``.

    Returns
    -------
    datetime
        A timezone-aware UTC time.
    """
    value = datetime.fromisoformat(text)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    """Format a UTC time as ``YYYY-MM-DDTHH:MM:SSZ``."""
    return value.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def is_bot(login: str | None, kind: str | None, bots: list[str]) -> bool:
    """Return True for an account that must never be mentioned.

    Parameters
    ----------
    login : str or None
        GitHub login; None for a deleted account.
    kind : str or None
        GraphQL ``__typename`` of the author (``User``, ``Bot``, ...).
    bots : list of str
        Logins treated as bots in any case.
    """
    if not login or kind not in (None, 'User'):
        return True
    return login.endswith('[bot]') or login.lower() in {b.lower() for b in bots}


class GitHub:
    """Minimal GitHub REST and GraphQL client with retries on rate limits."""

    def __init__(self, token: str | None = None, retries: int = 3):
        self.token = token if token is not None else os.environ.get('GITHUB_TOKEN', '')
        self.retries = retries

    def _request(
        self, url: str, data: bytes | None = None, accept: str | None = None, retry: bool = True
    ):
        headers = {'Accept': accept or 'application/vnd.github+json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        retries = self.retries if retry else 0
        for attempt in range(retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return resp.read()
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    return None
                if err.code not in (403, 429, 502, 503) or attempt == retries:
                    raise DigestError(f'GitHub API {err.code} for {url}') from err
                time.sleep(int(err.headers.get('Retry-After') or 20 * (attempt + 1)))
            except (OSError, http.client.HTTPException) as err:
                if attempt == retries:
                    raise DigestError(f'GitHub API unreachable for {url}: {err}') from err
                time.sleep(20 * (attempt + 1))

    def graphql(self, query: str, retry: bool = True, **variables) -> dict:
        """Run a GraphQL query and return its ``data``; raise on any error.

        A mutation passes ``retry=False``: a lost answer must not apply it twice.
        """
        body = json.dumps({'query': query, 'variables': variables}).encode()
        raw = self._request(f'{API}/graphql', data=body, retry=retry)
        out = json.loads(raw) if raw is not None else {'errors': 'not found'}
        if out.get('errors'):
            raise DigestError(f'GraphQL error: {out["errors"]}')
        return out['data']

    def raw_file(self, repo: str, path: str, ref: str | None) -> str | None:
        """Return a file's text at ``ref`` (None: the default branch), or None when it is absent."""
        query = f'?ref={ref}' if ref else ''
        raw = self._request(
            f'{API}/repos/{repo}/contents/{urllib.parse.quote(path)}{query}',
            accept='application/vnd.github.raw',
        )
        return None if raw is None else raw.decode()


def find_category(gh: GitHub, home: str, slug: str) -> tuple[str, str | None]:
    """Return the repository id and the id of its discussion category ``slug`` (None if absent)."""
    owner, name = home.split('/')
    repo = gh.graphql(CATEGORY_QUERY, owner=owner, name=name)['repository']
    ids = [c['id'] for c in repo['discussionCategories']['nodes'] if c['slug'] == slug]
    return repo['id'], (ids[0] if ids else None)
