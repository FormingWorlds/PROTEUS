#!/usr/bin/env python3
"""Regenerate the version badge table in docs/Reference/module_versions.md.

Reads version pins from pyproject.toml and writes shields.io badge markdown
into the doc page, replacing the content between the marker comments.

Usage:
    python tools/generate_version_badges.py
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / 'pyproject.toml'
TARGET = REPO_ROOT / 'docs' / 'Reference' / 'module_versions.md'

PYPI_MODULES = {
    'fwl-aragog': ('Interior thermal evolution', 'https://proteus-framework.org/aragog/', 'Docs'),
    'fwl-zalmoxis': ('Interior structure', 'https://proteus-framework.org/Zalmoxis/', 'Docs'),
    'fwl-calliope': ('Volatile outgassing', 'https://proteus-framework.org/CALLIOPE/', 'Docs'),
    'fwl-janus': ('1D convective atmosphere', 'https://proteus-framework.org/JANUS/', 'Docs'),
    'fwl-mors': ('Stellar evolution', 'https://proteus-framework.org/MORS/', 'Docs'),
    'fwl-zephyrus': ('Atmospheric escape', 'https://github.com/FormingWorlds/ZEPHYRUS', 'GitHub'),
    'fwl-vulcan': ('Atmospheric chemistry', 'https://github.com/FormingWorlds/VULCAN', 'GitHub'),
}

GIT_MODULES = {
    'agni': (
        'AGNI',
        'Radiative-convective atmosphere (Julia)',
        'https://github.com/nichollsh/AGNI',
        'https://www.h-nicholls.space/AGNI/',
        'Docs',
    ),
    'socrates': (
        'SOCRATES',
        'Spectral radiative transfer (Fortran)',
        'https://github.com/FormingWorlds/SOCRATES',
        'https://github.com/FormingWorlds/SOCRATES',
        'GitHub',
    ),
    'spider': (
        'SPIDER',
        'Interior evolution, T-S formulation (C)',
        'https://github.com/FormingWorlds/SPIDER',
        'https://proteus-framework.org/SPIDER/',
        'Docs',
    ),
}


def _badge(label: str, value: str, color: str, link: str) -> str:
    """Build a shields.io badge markdown string."""
    label_enc = quote(label, safe='')
    value_enc = quote(value, safe='')
    img = f'https://img.shields.io/badge/{label_enc}-{value_enc}-{color}'
    return f'[![{label}]({img})]({link})'


def main():
    cfg = tomllib.loads(PYPROJECT.read_text())
    deps = cfg['project']['dependencies']
    modules = cfg.get('tool', {}).get('proteus', {}).get('modules', {})

    # Parse PyPI deps
    pypi_rows = []
    for dep_str in deps:
        name = dep_str.split('>')[0].split('@')[0].split('=')[0].strip().lower().replace('_', '-')
        if name not in PYPI_MODULES:
            continue
        role, doc_url, doc_label = PYPI_MODULES[name]

        if '@' in dep_str and 'git+' in dep_str:
            # Git-pinned PyPI dep (e.g. CALLIOPE branch)
            branch = dep_str.rsplit('@', 1)[-1]
            badge_val = f'branch: {branch}'
            color = 'orange'
            badge_link = f'https://github.com/FormingWorlds/CALLIOPE/tree/{branch}'
        else:
            # Normal version bound
            match = re.search(r'>=([0-9.]+)', dep_str)
            if match:
                ver = match.group(1)
                badge_val = f'>={ver}'
                color = 'blue'
                # Link to PyPI release
                badge_link = f'https://pypi.org/project/{name}/{ver}/'
            else:
                badge_val = 'any'
                color = 'lightgrey'
                badge_link = f'https://pypi.org/project/{name}/'

        badge = _badge(name, badge_val, color, badge_link)
        pypi_rows.append(
            f'| {name} | {role} | {badge} | [{doc_label}]({doc_url}) |'
        )

    # Parse git modules
    git_rows = []
    for key, (display, role, repo_url, doc_url, doc_label) in GIT_MODULES.items():
        spec = modules.get(key, {})
        ref = spec.get('ref', 'n/a')
        short = ref[:8] if len(ref) > 8 else ref
        badge = _badge(display, short, 'green', f'{repo_url}/commit/{ref}')
        git_rows.append(
            f'| {display} | {role} | {badge} | [{doc_label}]({doc_url}) |'
        )

    # Build table sections
    pypi_table = '\n'.join([
        '| Module | Role | Pin | Docs |',
        '|--------|------|-----|------|',
        *pypi_rows,
    ])

    git_table = '\n'.join([
        '| Module | Role | Pin | Docs |',
        '|--------|------|-----|------|',
        *git_rows,
    ])

    print('PyPI packages:')
    print(pypi_table)
    print()
    print('Git modules:')
    print(git_table)
    print()
    print(f'Update {TARGET.relative_to(REPO_ROOT)} manually with the tables above.')
    print('(Automatic in-place replacement can be added as a follow-up.)')


if __name__ == '__main__':
    main()
