"""Unit tests for README badge validation.

Validates that badge URLs in README.md are well-formed, use HTTPS, and follow
expected patterns. This prevents badge regressions like the Codecov "unknown"
issue caused by a missing `/branch/main/` path segment.

Testing standards: docs/test_infrastructure.md, docs/test_categorization.md,
docs/test_building.md
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import pytest

from tests.helpers.helpers import PROTEUS_ROOT

README_PATH = PROTEUS_ROOT / 'README.md'

# Domains we expect badge images to come from
ALLOWED_IMG_DOMAINS = {'img.shields.io', 'codecov.io', 'raw.githubusercontent.com'}

# Badges we expect to find (substrings matched against <img src="..."> URLs)
EXPECTED_BADGES = [
    'Unit%20Tests',
    'Integration%20Tests',
    'docs.yaml',
    'License',
    'graph/badge.svg',
    'DOI',
]

# Workflow files referenced by badge URLs
BADGE_WORKFLOW_FILES = ['ci-pr-checks.yml', 'ci-nightly.yml', 'docs.yaml']


@pytest.fixture
def readme_content():
    """Read README.md content once for all tests."""
    return README_PATH.read_text()


@pytest.mark.unit
def test_readme_exists():
    """README.md must exist at the repository root."""
    assert README_PATH.is_file(), f'README.md not found at {README_PATH}'


@pytest.mark.unit
def test_all_badge_image_urls_are_valid_https(readme_content):
    """All badge <img src='...'> URLs must use HTTPS and point to allowed domains.

    Allowed domains: img.shields.io, codecov.io, raw.githubusercontent.com.
    This ensures badges load correctly and come from trusted sources.
    """
    img_urls = re.findall(r'<img\s+src="([^"]+)"', readme_content)
    assert len(img_urls) > 0, 'No <img> tags found in README'

    for url in img_urls:
        # Strip GitHub dark/light mode fragment identifiers before validation
        clean_url = url.split('#')[0]
        assert clean_url.startswith('https://'), f'Badge image URL is not HTTPS: {url}'

        domain = clean_url.split('https://')[1].split('/')[0]
        assert domain in ALLOWED_IMG_DOMAINS, (
            f"Badge image URL domain '{domain}' not in allowed list: {ALLOWED_IMG_DOMAINS}"
        )


@pytest.mark.unit
def test_all_badge_links_are_valid_https(readme_content):
    """All badge <a href='...'> URLs must use HTTPS.

    Badges link to GitHub Actions, documentation, and external services —
    all of which should be accessed over HTTPS.
    """
    href_urls = re.findall(r'<a\s+href="([^"]+)"', readme_content)
    assert len(href_urls) > 0, 'No <a> tags found in README'

    for url in href_urls:
        assert url.startswith('https://'), f'Badge link URL is not HTTPS: {url}'


@pytest.mark.unit
@pytest.mark.parametrize('badge_marker', EXPECTED_BADGES)
def test_expected_badges_present(readme_content, badge_marker):
    """All expected badges must be present in the README.

    We check for 6 badges: Unit Tests, Integration Tests, docs, License,
    Codecov, and DOI. Each is identified by a unique substring in its URL.
    """
    img_urls = re.findall(r'<img\s+src="([^"]+)"', readme_content)
    matching = [url for url in img_urls if badge_marker in url]
    assert len(matching) > 0, f"Expected badge with '{badge_marker}' not found in README"


@pytest.mark.unit
def test_codecov_badge_specifies_branch(readme_content):
    """Codecov badge URL must include /branch/main/ to avoid 'unknown' status.

    Without the branch specifier, Codecov returns 'unknown' for the badge.
    This was the root cause of the regression fixed in PR #638.
    """
    img_urls = re.findall(r'<img\s+src="([^"]+)"', readme_content)
    codecov_urls = [url for url in img_urls if urlparse(url).hostname == 'codecov.io']
    assert len(codecov_urls) == 1, (
        f'Expected exactly 1 Codecov badge, found {len(codecov_urls)}'
    )
    assert '/branch/main/' in codecov_urls[0], (
        f"Codecov badge URL missing '/branch/main/' specifier: {codecov_urls[0]}"
    )


@pytest.mark.unit
@pytest.mark.parametrize('workflow_file', BADGE_WORKFLOW_FILES)
def test_workflow_badges_reference_existing_workflows(workflow_file):
    """Workflow badge URLs must reference workflow files that actually exist.

    Checks that the workflow YAML files referenced in badge URLs
    (ci-pr-checks.yml, ci-nightly.yml, docs.yaml) exist in .github/workflows/.
    """
    workflow_path = PROTEUS_ROOT / '.github' / 'workflows' / workflow_file
    assert workflow_path.is_file(), (
        f'Workflow file referenced by badge does not exist: {workflow_path}'
    )
