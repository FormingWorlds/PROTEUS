"""Shared helpers of the weekly ecosystem digest: settings, GitHub API access, times.

The digest scripts use only the standard library, so the workflow needs no install
step beyond Python itself.
"""

from __future__ import annotations

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
    """
    with open(path or DIGEST_DIR / 'repos.toml', 'rb') as fh:
        return tomllib.load(fh)


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
    value = datetime.fromisoformat(text.replace('Z', '+00:00'))
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

    def _request(self, url: str, data: bytes | None = None, accept: str | None = None):
        headers = {'Accept': accept or 'application/vnd.github+json'}
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return resp.read()
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    return None
                if err.code in (403, 429, 502, 503) and attempt < self.retries:
                    time.sleep(int(err.headers.get('Retry-After') or 20 * (attempt + 1)))
                    continue
                raise DigestError(f'GitHub API {err.code} for {url}') from err
        raise DigestError(f'GitHub API gave up on {url}')

    def graphql(self, query: str, **variables) -> dict:
        """Run a GraphQL query and return its ``data``; raise on any error."""
        body = json.dumps({'query': query, 'variables': variables}).encode()
        out = json.loads(self._request(f'{API}/graphql', data=body))
        if out.get('errors'):
            raise DigestError(f'GraphQL error: {out["errors"]}')
        return out['data']

    def raw_file(self, repo: str, path: str, ref: str) -> str | None:
        """Return a file's text at ``ref``, or None when it does not exist there."""
        quoted = urllib.parse.quote(path)
        raw = self._request(
            f'{API}/repos/{repo}/contents/{quoted}?ref={ref}',
            accept='application/vnd.github.raw',
        )
        return None if raw is None else raw.decode()
