#!/usr/bin/env python3
"""Extract and classify CI failure context from JUnit XML and log artifacts.

This script parses artifacts from a failed GitHub Actions workflow run
and produces a structured failure summary for the self-healing AI agent.

Usage:
    python selfheal/extract_failures.py <artifacts_dir> <workflow_name>

Output (stdout): JSON with keys:
    - type: "test_failure" | "build_failure" | "infrastructure_failure"
    - workflow_name: Name of the failed workflow
    - summary: Human-readable one-line summary
    - failed_tests: List of {name, classname, message} dicts
    - stack_traces: Concatenated stack traces (truncated to MAX_CHARS)
    - log_errors: Extracted error lines from log files
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MAX_CHARS = 4000  # Keep prompt within token limits


def parse_junit_xml(xml_path: Path) -> list[dict]:
    """Parse a JUnit XML file and return failed test cases."""
    failures = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for testsuite in root.iter('testsuite'):
            for testcase in testsuite.findall('testcase'):
                failure = testcase.find('failure')
                error = testcase.find('error')
                if failure is not None or error is not None:
                    element = failure if failure is not None else error
                    failures.append(
                        {
                            'name': testcase.get('name', 'unknown'),
                            'classname': testcase.get('classname', ''),
                            'message': (element.get('message', '') or '')[:500],
                            'text': (element.text or '')[:1000],
                        }
                    )
    except (ET.ParseError, OSError) as e:
        print(f'Warning: Could not parse {xml_path}: {e}', file=sys.stderr)
    return failures


def extract_log_errors(log_path: Path) -> list[str]:
    """Extract error lines from a log file."""
    error_lines = []
    error_patterns = re.compile(
        r'(ERROR|FATAL|FAILED|Exception|Traceback|'
        r'FileNotFoundError|ModuleNotFoundError|ImportError|'
        r'CalledProcessError|make\[.\]: \*\*\*)',
        re.IGNORECASE,
    )
    try:
        text = log_path.read_text(errors='replace')
        for line in text.splitlines():
            if error_patterns.search(line):
                error_lines.append(line.strip()[:200])
    except OSError as e:
        print(f'Warning: Could not read {log_path}: {e}', file=sys.stderr)
    return error_lines


def classify_failure(
    failed_tests: list[dict],
    log_errors: list[str],
    workflow_name: str,
) -> str:
    """Classify the failure type based on available evidence."""
    if failed_tests:
        return 'test_failure'

    # Check for build/compilation failures
    build_patterns = re.compile(
        r'(make\[|compilation|gfortran|gcc|cmake|docker build|'
        r'SOCRATES|SPIDER|PETSc|AGNI)',
        re.IGNORECASE,
    )
    for line in log_errors:
        if build_patterns.search(line):
            return 'build_failure'

    # Check for infrastructure failures
    infra_patterns = re.compile(
        r'(download|network|timeout|disk space|permission denied|'
        r'Zenodo|OSF|connection|HTTP)',
        re.IGNORECASE,
    )
    for line in log_errors:
        if infra_patterns.search(line):
            return 'infrastructure_failure'

    # Docker workflow failures are typically build failures
    if 'Docker' in workflow_name:
        return 'build_failure'

    return 'test_failure'


def truncate(text: str, max_chars: int) -> str:
    """Truncate text to max_chars, adding truncation notice."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + '\n... [truncated]'


def main(artifacts_dir: str, workflow_name: str) -> None:
    """Main entry point."""
    artifacts = Path(artifacts_dir)

    # Collect all JUnit XML files
    all_failures: list[dict] = []
    for xml_file in sorted(artifacts.rglob('*.xml')):
        all_failures.extend(parse_junit_xml(xml_file))

    # Collect all log errors
    all_log_errors: list[str] = []
    for log_file in sorted(artifacts.rglob('*.txt')):
        all_log_errors.extend(extract_log_errors(log_file))
    for log_file in sorted(artifacts.rglob('*.log')):
        all_log_errors.extend(extract_log_errors(log_file))

    # Classify
    failure_type = classify_failure(all_failures, all_log_errors, workflow_name)

    # Build summary
    if all_failures:
        test_names = [f['classname'] + '::' + f['name'] for f in all_failures]
        summary = f'{len(all_failures)} test(s) failed: {", ".join(test_names[:5])}'
        if len(test_names) > 5:
            summary += f' ... and {len(test_names) - 5} more'
    elif all_log_errors:
        summary = f'{failure_type}: {all_log_errors[0][:200]}'
    else:
        summary = f'{workflow_name} failed (no detailed error info available)'

    # Build stack traces
    traces = []
    for f in all_failures:
        traces.append(f'--- {f["classname"]}::{f["name"]} ---')
        traces.append(f['message'])
        if f['text']:
            traces.append(f['text'])
    stack_traces = truncate('\n'.join(traces), MAX_CHARS)

    # Build log error summary
    log_error_text = truncate(
        '\n'.join(dict.fromkeys(all_log_errors)),  # deduplicate, preserve order
        MAX_CHARS,
    )

    result = {
        'type': failure_type,
        'workflow_name': workflow_name,
        'summary': summary[:500],
        'failed_tests': [
            {'name': f['name'], 'classname': f['classname'], 'message': f['message']}
            for f in all_failures[:20]  # Cap at 20 tests
        ],
        'stack_traces': stack_traces,
        'log_errors': log_error_text,
    }

    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <artifacts_dir> <workflow_name>', file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
