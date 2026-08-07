#!/usr/bin/env python3
"""Regenerate the version badge tables in docs/Reference/module_versions.md.

Reads version pins from pyproject.toml and replaces the content between
marker comments in the doc page with fresh shields.io badge markdown. Every
fwl-* dependency must be covered by a table (or listed with a reason in
INTENTIONALLY_ABSENT), so a newly added module cannot silently miss the page.

Usage:
    python tools/generate_version_badges.py            # regenerate in place
    python tools/generate_version_badges.py --check    # verify, write nothing

Run the regeneration after bumping any version in pyproject.toml.

Exit codes:
    0  committed page is current and every fwl-* dependency is covered
    1  page is stale, or an fwl-* dependency is missing from the tables
    2  structural error (marker pair missing from the page)
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from urllib.parse import quote

import _docgen
from _docgen import REPO_ROOT

PYPROJECT = REPO_ROOT / 'pyproject.toml'
TARGET = REPO_ROOT / 'docs' / 'Reference' / 'module_versions.md'
REGEN_CMD = 'python tools/generate_version_badges.py'

# fwl-* distributions deliberately absent from every table, with the reason.
INTENTIONALLY_ABSENT = {
    'fwl-io': 'infrastructure library for data download, not a physics module',
}

PYPI_META = {
    'fwl-aragog': (
        'Interior thermal evolution',
        'https://proteus-framework.org/aragog/',
        'Docs',
    ),
    'fwl-zalmoxis': ('Interior structure', 'https://proteus-framework.org/Zalmoxis/', 'Docs'),
    'fwl-calliope': ('Volatile outgassing', 'https://proteus-framework.org/CALLIOPE/', 'Docs'),
    'fwl-janus': ('1D convective atmosphere', 'https://proteus-framework.org/JANUS/', 'Docs'),
    'fwl-mors': ('Stellar evolution', 'https://proteus-framework.org/MORS/', 'Docs'),
    'fwl-zephyrus': (
        'Atmospheric escape',
        'https://github.com/FormingWorlds/ZEPHYRUS',
        'GitHub',
    ),
    # fwl-vulcan is an optional backend (see OPTIONAL below), not a mandatory
    # PyPI dependency, so it is intentionally absent from this table.
}

GIT_META = {
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
        'https://proteus-framework.org/SOCRATES/',
        'Docs',
    ),
    'spider': (
        'SPIDER',
        'Interior evolution (C, requires PETSc)',
        'https://github.com/FormingWorlds/SPIDER',
        'https://proteus-framework.org/SPIDER/',
        'Docs',
    ),
}

# Each entry: (label, role, pin_val, color, pin_link, doc_url, doc_label, extra).
# When `extra` is set to (extra_key, dist_name), the version badge is derived
# from pyproject's [project.optional-dependencies] so the pin lives in one
# place; pin_val/pin_link are then ignored. Otherwise pin_val/pin_link are
# used as-is (git refs, docs-only entries).
OPTIONAL = [
    (
        'LovePy',
        'Multi-phase tidal heating (Julia)',
        'main',
        'lightgrey',
        'https://github.com/nichollsh/LovePy',
        'https://github.com/nichollsh/LovePy',
        'GitHub',
        None,
    ),
    (
        'atmodeller',
        'Alternative outgassing backend (GPL-3.0)',
        None,
        'blue',
        None,
        'https://github.com/djbower/atmodeller',
        'GitHub',
        ('atmodeller', 'atmodeller'),
    ),
    (
        'VULCAN',
        'Atmospheric chemistry (GPL-3.0)',
        None,
        'blue',
        None,
        'https://github.com/FormingWorlds/VULCAN',
        'GitHub',
        ('vulcan', 'fwl-vulcan'),
    ),
    (
        'Obliqua',
        'Orbital evolution and tides (Julia)',
        None,
        None,
        None,
        'https://github.com/FormingWorlds/Obliqua',
        'GitHub',
        None,
    ),
]


def _badge(label: str, value: str, color: str, link: str) -> str:
    label_enc = quote(label, safe='').replace('-', '--')
    value_enc = quote(value, safe='').replace('-', '--')
    img = f'https://img.shields.io/badge/{label_enc}-{value_enc}-{color}'
    # attr_list adds target/rel so the badge link opens in a new tab, matching
    # every other badge link across the docs.
    return f'[![{label}]({img})]({link}){{target="_blank" rel="noopener"}}'


def _build_pypi_table(deps: list[str]) -> str:
    rows = []
    for dep_str in deps:
        name = (
            dep_str.split('>')[0].split('@')[0].split('=')[0].strip().lower().replace('_', '-')
        )
        if name not in PYPI_META:
            continue
        role, doc_url, doc_label = PYPI_META[name]
        if '@' in dep_str and 'git+' in dep_str:
            branch = dep_str.rsplit('@', 1)[-1]
            badge = _badge(
                name,
                f'branch: {branch}',
                'orange',
                f'https://github.com/FormingWorlds/CALLIOPE/tree/{branch}',
            )
        else:
            match = re.search(r'>=([0-9.]+)', dep_str)
            if match:
                ver = match.group(1)
                badge = _badge(
                    name, f'>={ver}', 'blue', f'https://pypi.org/project/{name}/{ver}/'
                )
            else:
                badge = _badge(name, 'any', 'lightgrey', f'https://pypi.org/project/{name}/')
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


def _optional_pin_from_extra(extras: dict, extra_key: str, dist_name: str):
    """Return (pin_label, pypi_link) derived from a pyproject extra.

    Reads the requirement for ``dist_name`` from
    ``[project.optional-dependencies].<extra_key>`` and turns its ``>=X``
    floor into a shields.io badge value and a PyPI release link, so the
    optional badge tracks the single pyproject pin. Returns ``(None, None)``
    if the extra or floor cannot be resolved.
    """
    for req in extras.get(extra_key, []):
        if req.split('>')[0].split('=')[0].split('[')[0].strip() != dist_name:
            continue
        match = re.search(r'>=([0-9.]+)', req)
        if match:
            ver = match.group(1)
            return f'>={ver}', f'https://pypi.org/project/{dist_name}/{ver}/'
        return 'any', f'https://pypi.org/project/{dist_name}/'
    return None, None


def _build_optional_table(extras: dict) -> str:
    rows = []
    for name, role, pin_val, color, pin_link, doc_url, doc_label, extra in OPTIONAL:
        if extra is not None:
            pin_val, pin_link = _optional_pin_from_extra(extras, *extra)
        if pin_val and color and pin_link:
            badge = _badge(name, pin_val, color, pin_link)
        else:
            badge = 'n/a'
        rows.append(f'| {name} | {role} | {badge} | [{doc_label}]({doc_url}) |')
    return '| Module | Role | Pin | Docs |\n|--------|------|-----|------|\n' + '\n'.join(rows)


def _dist_name(req: str) -> str:
    """Normalised distribution name of a requirement string."""
    return req.split('>')[0].split('@')[0].split('=')[0].split('[')[0].strip().lower()


def check_fwl_coverage(deps: list[str], extras: dict) -> list[str]:
    """Every fwl-* requirement must appear in a table or be excused."""
    optional_dists = {extra[1] for *_rest, extra in OPTIONAL if extra is not None}
    problems = []
    for req in deps:
        name = _dist_name(req).replace('_', '-')
        if (
            name.startswith('fwl-')
            and name not in PYPI_META
            and name not in INTENTIONALLY_ABSENT
        ):
            problems.append(f'dependency "{name}" is missing from PYPI_META')
    for reqs in extras.values():
        for req in reqs:
            name = _dist_name(req).replace('_', '-')
            if (
                name.startswith('fwl-')
                and name not in optional_dists
                and name not in INTENTIONALLY_ABSENT
            ):
                problems.append(f'optional dependency "{name}" is missing from OPTIONAL')
    for name in INTENTIONALLY_ABSENT:
        all_reqs = list(deps) + [r for reqs in extras.values() for r in reqs]
        if not any(_dist_name(r).replace('_', '-') == name for r in all_reqs):
            problems.append(f'INTENTIONALLY_ABSENT entry "{name}" matches no requirement')
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _docgen.add_check_write_cli(parser)
    args = parser.parse_args()

    cfg = tomllib.loads(PYPROJECT.read_text())
    deps = cfg['project']['dependencies']
    modules = cfg.get('tool', {}).get('proteus', {}).get('modules', {})
    extras = cfg['project'].get('optional-dependencies', {})

    rc = 0
    for problem in check_fwl_coverage(deps, extras):
        print(f'[x] {problem}')
        rc = 1

    try:
        content = TARGET.read_text()
        content = _docgen.replace_between_markers(
            content, 'PYPI_TABLE', _build_pypi_table(deps)
        )
        content = _docgen.replace_between_markers(
            content, 'GIT_TABLE', _build_git_table(modules)
        )
        content = _docgen.replace_between_markers(
            content, 'OPTIONAL_TABLE', _build_optional_table(extras)
        )
    except _docgen.DocgenError as exc:
        print(f'[x] {exc}', file=sys.stderr)
        return 2

    targets = [(TARGET, _docgen.normalize(content))]
    return max(rc, _docgen.run_generator(targets, check=args.check, regen_cmd=REGEN_CMD))


if __name__ == '__main__':
    sys.exit(main())
