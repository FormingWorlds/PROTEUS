"""Aggregate coverage XML reports into a per-tier / per-shard summary.

Replaces the inline Python step in ``.github/workflows/ci-nightly.yml``
that wrote ``coverage-by-type.json``. The aggregator job in the
parallel-matrix nightly downloads many coverage XML artifacts (one per
shard + per OS) and calls this script to produce a single summary JSON
that downstream tools and dashboards can read.

Output JSON shape (extended from the legacy shape; the ``total`` and
``threshold`` keys remain in place so existing readers keep working)::

    {
      "timestamp": "2026-05-22T17:18:34.854716+00:00",
      "threshold": 59.0,
      "total": {"percent": 84.06, "covered": 11712, "total": 13933},
      "per_tier": {
        "unit":        {"percent": ..., "covered": ..., "total": ...},
        "integration": {"percent": ..., "covered": ..., "total": ...},
        "slow":        {"percent": ..., "covered": ..., "total": ...}
      },
      "per_shard": {
        "slow-aragog-ubuntu-latest":  {"percent": ..., ...},
        "slow-aragog-macos-latest":   {"percent": ..., ...},
        ...
      }
    }

A per-tier entry is the union coverage of the XMLs supplied for that
tier (i.e. ``coverage combine`` semantics applied via XML merge: a line
is covered in the tier if any tier XML marks it as a hit). Per-shard
entries are individual XML stats (no merge), useful for spotting which
shard contributed which coverage.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import tomllib
import xml.etree.ElementTree as ET
from datetime import datetime, timezone


def _parse_xml(xml_path: str) -> dict | None:
    """Return ``{path: set_of_hit_line_nos}`` for an XML, or None if
    the file does not exist or is empty.
    """
    p = pathlib.Path(xml_path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        tree = ET.parse(p)
    except ET.ParseError:
        return None
    root = tree.getroot()
    out: dict[str, set[int]] = {}
    for pkg in root.iter('package'):
        for cls in pkg.iter('class'):
            filename = cls.attrib.get('filename', '')
            if not filename.startswith('src/proteus/'):
                continue
            hits = out.setdefault(filename, set())
            lines = cls.find('lines')
            if lines is None:
                continue
            for line in lines.iter('line'):
                if int(line.attrib.get('hits', '0')) > 0:
                    hits.add(int(line.attrib['number']))
            # Track existence even if no hits, so the file's denominator
            # is preserved.
    return out


def _xml_totals(xml_path: str) -> dict | None:
    """Return ``{percent, covered, total}`` summary for a single XML."""
    p = pathlib.Path(xml_path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        tree = ET.parse(p)
    except ET.ParseError:
        return None
    root = tree.getroot()
    covered = 0
    total = 0
    for pkg in root.iter('package'):
        for cls in pkg.iter('class'):
            if not cls.attrib.get('filename', '').startswith('src/proteus/'):
                continue
            lines = cls.find('lines')
            if lines is None:
                continue
            for line in lines.iter('line'):
                total += 1
                if int(line.attrib.get('hits', '0')) > 0:
                    covered += 1
    if total == 0:
        return None
    return {'percent': round(100 * covered / total, 2), 'covered': covered, 'total': total}


def _file_universe(xml_paths: list[str]) -> dict[str, int]:
    """Return ``{path: total_line_count}`` over the union of files seen
    in any of the supplied XMLs. Total is the max line count across
    XMLs to defend against partial reports.
    """
    universe: dict[str, int] = {}
    for xml_path in xml_paths:
        p = pathlib.Path(xml_path)
        if not p.exists() or p.stat().st_size == 0:
            continue
        try:
            tree = ET.parse(p)
        except ET.ParseError:
            continue
        for pkg in tree.iter('package'):
            for cls in pkg.iter('class'):
                filename = cls.attrib.get('filename', '')
                if not filename.startswith('src/proteus/'):
                    continue
                lines = cls.find('lines')
                if lines is None:
                    continue
                count = sum(1 for _ in lines.iter('line'))
                if count > universe.get(filename, 0):
                    universe[filename] = count
    return universe


def merge_tier(xml_paths: list[str]) -> dict | None:
    """Merge multiple XMLs by unioning per-file hit-line sets.

    Returns ``{percent, covered, total}`` for the unioned coverage, or
    None if no valid XMLs were supplied.
    """
    merged_hits: dict[str, set[int]] = {}
    for xml_path in xml_paths:
        parsed = _parse_xml(xml_path)
        if parsed is None:
            continue
        for f, hits in parsed.items():
            merged_hits.setdefault(f, set()).update(hits)
    if not merged_hits:
        return None
    universe = _file_universe(xml_paths)
    total = sum(universe.values())
    covered = sum(len(merged_hits.get(f, set())) for f in universe)
    if total == 0:
        return None
    return {'percent': round(100 * covered / total, 2), 'covered': covered, 'total': total}


def _read_threshold(pyproject: pathlib.Path) -> float:
    """Read the ``[tool.coverage.report].fail_under`` floor from
    pyproject.toml. Returns 59.0 as a documented historical default if
    pyproject.toml is missing or malformed.
    """
    try:
        py = tomllib.loads(pyproject.read_text())
        return float(py['tool']['coverage']['report']['fail_under'])
    except Exception:
        return 59.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--combined',
        required=True,
        help='Path to the combined coverage.xml (the canonical total).',
    )
    parser.add_argument(
        '--unit',
        nargs='*',
        default=[],
        help='Paths to unit-tier coverage XMLs (e.g. one per OS). Merged by union.',
    )
    parser.add_argument(
        '--integration',
        nargs='*',
        default=[],
        help='Paths to integration-tier coverage XMLs. Merged by union.',
    )
    parser.add_argument(
        '--slow',
        nargs='*',
        default=[],
        help='Paths to slow-tier shard coverage XMLs (e.g. coverage-slow-aragog-ubuntu-latest.xml). Merged for the slow per-tier; each is also reported individually under per_shard.',
    )
    parser.add_argument(
        '--pyproject',
        default='pyproject.toml',
        help='pyproject.toml path for the fail-under threshold.',
    )
    parser.add_argument(
        '--out',
        default='coverage-by-type.json',
        help='Output JSON path.',
    )
    args = parser.parse_args(argv)

    total = _xml_totals(args.combined) or {'percent': 0, 'covered': 0, 'total': 0}
    threshold = _read_threshold(pathlib.Path(args.pyproject))

    summary = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'threshold': threshold,
        'total': total,
        'per_tier': {},
        'per_shard': {},
    }
    if args.unit:
        merged = merge_tier(args.unit)
        if merged is not None:
            summary['per_tier']['unit'] = merged
    if args.integration:
        merged = merge_tier(args.integration)
        if merged is not None:
            summary['per_tier']['integration'] = merged
    if args.slow:
        merged = merge_tier(args.slow)
        if merged is not None:
            summary['per_tier']['slow'] = merged
        # Per-shard breakdown.
        for xml_path in args.slow:
            shard_name = pathlib.Path(xml_path).stem
            shard_totals = _xml_totals(xml_path)
            if shard_totals is not None:
                summary['per_shard'][shard_name] = shard_totals

    pathlib.Path(args.out).write_text(json.dumps(summary, indent=2) + '\n')
    pct = total['percent']
    cov = total['covered']
    tot = total['total']
    sys.stdout.write(f'Coverage: {pct:.2f}% ({cov}/{tot} lines)\n')
    if summary['per_tier']:
        for tier, vals in summary['per_tier'].items():
            sys.stdout.write(f'  {tier}: {vals["percent"]:.2f}%\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
