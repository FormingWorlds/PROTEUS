"""Snapshot the documentation badges as local static SVG files.

The documentation pages show status badges (continuous-integration results,
coverage, license, website) and test-count badges. By default these are fetched
from third-party services (shields.io, codecov.io) when the page loads. Those
services intermittently rate-limit or time out and return an HTML error instead
of an SVG, which the browser renders as a broken-image icon and which clears
only after several reloads. This script downloads each badge once at
documentation build time and writes it as a local SVG under the output
directory, so the published page serves every badge from its own origin with no
render-time dependency on an external service.

Workflow-status badges (unit, integration, docs) resolve the latest run
conclusion through the GitHub REST API and render it as a static shields.io
badge, so the label and style stay constant regardless of which upstream
service is reachable. Test-count badges reuse the collection logic in
``generate_test_badges`` to compute the counts and render them the same way.
The static ``/badge`` endpoint carries no GitHub-API round trip at fetch time
and is far more reliable than the dynamic workflow-status endpoint. Coverage,
license, and website badges are fetched from their own canonical sources.

If a badge cannot be refreshed and a previously cached SVG exists, that file is
kept (last-good fallback); otherwise a neutral placeholder SVG is written so the
page layout still holds. The script never fails the documentation build: a badge
that cannot be refreshed degrades to its last-good or placeholder form.

Usage
-----
    python tools/cache_badges.py --out docs/badges/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# tools/ is on sys.path[0] when this file is run directly, so the sibling
# module imports without packaging.
from generate_test_badges import _BADGES as _COUNT_BADGES
from generate_test_badges import count_tests

_REPO = 'FormingWorlds/PROTEUS'
_BRANCH = 'main'
_USER_AGENT = 'PROTEUS-docs-badge-cache'

# Map a GitHub Actions run conclusion to the badge message and colour. A run
# that is queued or in progress has a null conclusion and is shown as pending.
_CONCLUSION_STYLE: dict[str | None, tuple[str, str]] = {
    'success': ('passing', 'brightgreen'),
    'failure': ('failing', 'red'),
    'cancelled': ('cancelled', 'inactive'),
    'timed_out': ('timed out', 'red'),
    None: ('no status', 'lightgrey'),
}


def _latest_conclusion(workflow: str, timeout: float, retries: int) -> str | None:
    """Return the conclusion of the most recent run of a workflow on the branch.

    Queries the GitHub REST API for the latest run of ``workflow`` on
    ``_BRANCH``. Uses ``GITHUB_TOKEN`` from the environment if present (for the
    higher authenticated rate limit), otherwise queries unauthenticated.

    Parameters
    ----------
    workflow : str
        Workflow file name, for example ``ci-pr-checks.yml``.
    timeout : float
        Per-request timeout in seconds.
    retries : int
        Attempts before giving up.

    Returns
    -------
    str or None
        The run conclusion (for example ``'success'``), ``None`` if the latest
        run has no conclusion yet, or the sentinel ``'unknown'`` if the API
        could not be reached or reported no runs.
    """
    url = (
        f'https://api.github.com/repos/{_REPO}/actions/workflows/'
        f'{workflow}/runs?branch={_BRANCH}&per_page=1'
    )
    headers = {'User-Agent': _USER_AGENT, 'Accept': 'application/vnd.github+json'}
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f'Bearer {token}'
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            runs = data.get('workflow_runs', [])
            if not runs:
                return None
            return runs[0].get('conclusion')
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            if attempt < retries:
                time.sleep(1.0 * attempt)
    return 'unknown'


def _status_badge_url(workflow: str, label: str, timeout: float, retries: int) -> str | None:
    """Build a static shields.io URL for a workflow-status badge.

    Resolves the live run conclusion through the GitHub API, then renders it as
    a static badge so the label and style stay constant across deploys
    regardless of which upstream service is reachable.

    Parameters
    ----------
    workflow : str
        Workflow file name.
    label : str
        Left-hand label, for example ``'Unit Tests'``.
    timeout : float
        Per-request timeout in seconds.
    retries : int
        Attempts when querying the API.

    Returns
    -------
    str or None
        A static shields badge URL, or ``None`` if the conclusion could not be
        determined (so the caller keeps the last-good badge).
    """
    conclusion = _latest_conclusion(workflow, timeout, retries)
    if conclusion == 'unknown':
        return None
    message, color = _CONCLUSION_STYLE.get(conclusion, (conclusion, 'lightgrey'))
    return _static_shields_url(label, message, color)


def _static_shields_url(label: str, message: str, color: str = 'blue') -> str:
    """Build a static shields.io badge URL.

    The static ``/badge`` endpoint renders a fixed label and message with no
    upstream API call, so it is reliable to fetch at build time. shields.io
    escapes a literal dash as ``--``, a literal underscore as ``__``, and a
    space as ``_``.

    Parameters
    ----------
    label : str
        Left-hand label text.
    message : str
        Right-hand message text.
    color : str, optional
        Badge colour, by default ``'blue'``.

    Returns
    -------
    str
        The fully-escaped static badge URL.
    """

    def esc(text: str) -> str:
        return text.replace('-', '--').replace('_', '__').replace(' ', '_')

    return f'https://img.shields.io/badge/{esc(label)}-{esc(message)}-{color}'


def _looks_like_svg(body: bytes) -> bool:
    """Return True if the response body is plausibly an SVG document.

    Parameters
    ----------
    body : bytes
        Raw response body.

    Returns
    -------
    bool
        True if the leading bytes open an XML or SVG document. A rate-limit or
        timeout page is HTML and fails this test.
    """
    head = body[:512].lstrip().lower()
    return head.startswith(b'<?xml') or head.startswith(b'<svg') or b'<svg' in head


def _fetch_svg(
    urls: list[str], timeout: float, retries: int
) -> tuple[bytes | None, str | None]:
    """Fetch the first candidate URL that returns a valid SVG.

    Parameters
    ----------
    urls : list[str]
        Candidate source URLs, tried in order.
    timeout : float
        Per-request timeout in seconds.
    retries : int
        Attempts per URL before moving to the next candidate.

    Returns
    -------
    tuple of (bytes or None, str or None)
        The SVG bytes and the URL they came from, or ``(None, None)`` if every
        candidate failed.
    """
    for url in urls:
        for attempt in range(1, retries + 1):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': _USER_AGENT})
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = resp.read()
                    ctype = resp.headers.get('Content-Type', '')
                if _looks_like_svg(body) and ('svg' in ctype.lower() or _looks_like_svg(body)):
                    return body, url
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            if attempt < retries:
                time.sleep(1.0 * attempt)
    return None, None


def _placeholder(label: str) -> bytes:
    """Return a neutral grey SVG used when no source and no cache are available.

    Parameters
    ----------
    label : str
        Text rendered in the placeholder badge.

    Returns
    -------
    bytes
        A minimal flat badge SVG.
    """
    width = max(60, 8 * len(label) + 20)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20" '
        f'role="img" aria-label="{label}">'
        f'<rect width="{width}" height="20" rx="3" fill="#9f9f9f"/>'
        f'<text x="{width // 2}" y="14" fill="#fff" font-family="Verdana,sans-serif" '
        f'font-size="11" text-anchor="middle">{label}</text></svg>'
    )
    return svg.encode('utf-8')


def cache_badge(out_dir: Path, name: str, urls: list[str], timeout: float, retries: int) -> str:
    """Write one badge SVG, preferring a fresh download then the last-good file.

    Parameters
    ----------
    out_dir : Path
        Directory to write ``<name>.svg`` into.
    name : str
        Output file stem.
    urls : list[str]
        Candidate source URLs.
    timeout : float
        Per-request timeout in seconds.
    retries : int
        Attempts per URL.

    Returns
    -------
    str
        One of ``'ok'``, ``'kept-last-good'``, or ``'placeholder'``.
    """
    out_path = out_dir / f'{name}.svg'
    body, _ = _fetch_svg(urls, timeout, retries)
    if body is not None:
        out_path.write_bytes(body)
        return 'ok'
    if out_path.exists():
        return 'kept-last-good'
    out_path.write_bytes(_placeholder(name))
    return 'placeholder'


def _count_badges(timeout: float, retries: int) -> list[tuple[str, list[str]]]:
    """Build the test-count badge specs from live collection counts.

    Parameters
    ----------
    timeout : float
        Unused here; kept for signature symmetry with the status badges.
    retries : int
        Unused here; kept for signature symmetry with the status badges.

    Returns
    -------
    list of (str, list of str)
        ``(name, [url])`` pairs, one per count badge. A collection failure for
        one marker expression drops that badge to a placeholder rather than
        aborting the run.
    """
    specs: list[tuple[str, list[str]]] = []
    for name, label, expr in _COUNT_BADGES:
        try:
            count = count_tests(expr)
            url = _static_shields_url(label, str(count))
            specs.append((f'tests-{name}', [url]))
        except Exception as exc:  # noqa: BLE001 - badge refresh must never abort the build
            print(
                f'  warning: count for {label!r} failed ({exc}); using last-good',
                file=sys.stderr,
            )
            specs.append((f'tests-{name}', []))
    return specs


def main() -> int:
    """Entry point.

    Returns
    -------
    int
        Always 0. A badge that cannot be refreshed degrades gracefully.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--out',
        type=Path,
        default=Path('docs/badges'),
        help='Directory to write the cached badge SVGs into.',
    )
    parser.add_argument('--timeout', type=float, default=10.0, help='Per-request timeout [s].')
    parser.add_argument('--retries', type=int, default=3, help='Attempts per candidate URL.')
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    def status(name: str, workflow: str, label: str) -> tuple[str, list[str]]:
        url = _status_badge_url(workflow, label, args.timeout, args.retries)
        return (name, [url] if url else [])

    status_badges: list[tuple[str, list[str]]] = [
        status('unit', 'ci-pr-checks.yml', 'Unit Tests'),
        status('integration', 'ci-nightly.yml', 'Integration Tests'),
        status('docs', 'docs.yaml', 'Docs'),
        ('codecov', [f'https://codecov.io/gh/{_REPO}/branch/{_BRANCH}/graph/badge.svg']),
        ('license', ['https://img.shields.io/badge/License-Apache_2.0-blue.svg']),
        (
            'website',
            [
                'https://img.shields.io/website?url=https%3A%2F%2Fproteus-framework.org'
                '&label=Website&up_message=proteus-framework.org'
                '&down_message=proteus-framework.org'
            ],
        ),
    ]

    badges = status_badges + _count_badges(args.timeout, args.retries)

    for name, urls in badges:
        result = cache_badge(out_dir, name, urls, args.timeout, args.retries)
        print(f'{name}.svg: {result}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
