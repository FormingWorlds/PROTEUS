#!/usr/bin/env python3
"""Regenerate the version badge tables in docs/Reference/module_versions.md.

Reads version pins from pyproject.toml and replaces the content between
marker comments in the doc page with fresh shields.io badge markdown.

Usage:
    python tools/generate_version_badges.py

Run this after bumping any version in pyproject.toml.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / 'pyproject.toml'
TARGET = REPO_ROOT / 'docs' / 'Reference' / 'module_versions.md'

PYPI_META = {
    'fwl-aragog': ('Interior thermal evolution', 'https://proteus-framework.org/aragog/', 'Docs'),
    'fwl-zalmoxis': ('Interior structure', 'https://proteus-framework.org/Zalmoxis/', 'Docs'),
    'fwl-calliope': ('Volatile outgassing', 'https://proteus-framework.org/CALLIOPE/', 'Docs'),
    'fwl-janus': ('1D convective atmosphere', 'https://proteus-framework.org/JANUS/', 'Docs'),
    'fwl-mors': ('Stellar evolution', 'https://proteus-framework.org/MORS/', 'Docs'),
    'fwl-zephyrus': ('Atmospheric escape', 'https://github.com/FormingWorlds/ZEPHYRUS', 'GitHub'),
    'fwl-vulcan': ('Atmospheric chemistry', 'https://github.com/FormingWorlds/VULCAN', 'GitHub'),
}

GIT_META = {
    'agni': ('AGNI', 'Radiative-convective atmosphere (Julia)',
             'https://github.com/nichollsh/AGNI',
             'https://www.h-nicholls.space/AGNI/', 'Docs'),
    'socrates': ('SOCRATES', 'Spectral radiative transfer (Fortran)',
                 'https://github.com/FormingWorlds/SOCRATES',
                 'https://proteus-framework.org/SOCRATES/', 'Docs'),
    'spider': ('SPIDER', 'Interior evolution (C, requires PETSc)',
               'https://github.com/FormingWorlds/SPIDER',
               'https://proteus-framework.org/SPIDER/', 'Docs'),
}

OPTIONAL = [
    ('LovePy', 'Multi-phase tidal heating (Julia)',
     'main', 'lightgrey',
     'https://github.com/nichollsh/LovePy',
     'https://github.com/nichollsh/LovePy', 'GitHub'),
    ('atmodeller', 'Alternative outgassing backend',
     '>=1.0.0', 'blue',
     'https://pypi.org/project/atmodeller/',
     'https://github.com/djbower/atmodeller', 'GitHub'),
    ('Obliqua', 'Orbital evolution and tides (Julia)',
     None, None, None,
     'https://github.com/FormingWorlds/Obliqua', 'GitHub'),
    ('PLATON', 'Synthetic observations',
     None, None, None,
     'https://platon.readthedocs.io/', 'Docs'),
]


def _badge(label: str, value: str, color: str, link: str) -> str:
    label_enc = quote(label, safe='').replace('-', '--')
    value_enc = quote(value, safe='').replace('-', '--')
    img = f'https://img.shields.io/badge/{label_enc}-{value_enc}-{color}'
    return f'[![{label}]({img})]({link})'


def _build_pypi_table(deps: list[str]) -> str:
    rows = []
    for dep_str in deps:
        name = dep_str.split('>')[0].split('@')[0].split('=')[0].strip().lower().replace('_', '-')
        if name not in PYPI_META:
            continue
        role, doc_url, doc_label = PYPI_META[name]
        if '@' in dep_str and 'git+' in dep_str:
            branch = dep_str.rsplit('@', 1)[-1]
            badge = _badge(name, f'branch: {branch}', 'orange',
                           f'https://github.com/FormingWorlds/CALLIOPE/tree/{branch}')
        else:
            match = re.search(r'>=([0-9.]+)', dep_str)
            if match:
                ver = match.group(1)
                badge = _badge(name, f'>={ver}', 'blue',
                               f'https://pypi.org/project/{name}/{ver}/')
            else:
                badge = _badge(name, 'any', 'lightgrey',
                               f'https://pypi.org/project/{name}/')
        rows.append(f'| {name} | {role} | {badge} | [{doc_label}]({doc_url}) |')
    return '| Module | Role | Pin | Docs |\n|--------|------|-----|------|\n' + '\n'.join(rows)


def _build_git_table(modules: dict) -> str:
    rows = []
    for key, (display, role, repo_url, doc_url, doc_label) in GIT_META.items():
        spec = modules.get(key, {})
        ref = spec.get('ref', 'n/a')
        short = ref[:8] if len(ref) > 8 else ref
        badge = _badge(display, short, 'green', f'{repo_url}/commit/{ref}')
        rows.append(f'| {display} | {role} | {badge} | [{doc_label}]({doc_url}) |')
    return '| Module | Role | Pin | Docs |\n|--------|------|-----|------|\n' + '\n'.join(rows)


def _build_optional_table() -> str:
    rows = []
    for name, role, pin_val, color, pin_link, doc_url, doc_label in OPTIONAL:
        if pin_val and color and pin_link:
            badge = _badge(name, pin_val, color, pin_link)
        else:
            badge = 'n/a'
        rows.append(f'| {name} | {role} | {badge} | [{doc_label}]({doc_url}) |')
    return '| Module | Role | Pin | Docs |\n|--------|------|-----|------|\n' + '\n'.join(rows)


def _replace_between_markers(content: str, marker: str, replacement: str) -> str:
    pattern = f'(<!-- BEGIN {marker} -->)(.*?)(<!-- END {marker} -->)'
    return re.sub(pattern, rf'\1\n{replacement}\n\3', content, flags=re.DOTALL)


def main():
    cfg = tomllib.loads(PYPROJECT.read_text())
    deps = cfg['project']['dependencies']
    modules = cfg.get('tool', {}).get('proteus', {}).get('modules', {})

    pypi_table = _build_pypi_table(deps)
    git_table = _build_git_table(modules)
    optional_table = _build_optional_table()

    content = TARGET.read_text()
    content = _replace_between_markers(content, 'PYPI_TABLE', pypi_table)
    content = _replace_between_markers(content, 'GIT_TABLE', git_table)
    content = _replace_between_markers(content, 'OPTIONAL_TABLE', optional_table)
    TARGET.write_text(content)

    print(f'Updated {TARGET.relative_to(REPO_ROOT)}')
    print(f'  PyPI: {len(PYPI_META)} packages')
    print(f'  Git: {len(GIT_META)} modules')
    print(f'  Optional: {len(OPTIONAL)} modules')


if __name__ == '__main__':
    main()
