#!/usr/bin/env python3
"""Print the shell commands of the install page for one platform tab.

The tool reads ``docs/How-to/installation.md`` and extracts the fenced
``console`` blocks in page order. A block at column 0 belongs to every
platform. A block indented under a tab header ``=== "<name>"`` belongs to that
tab only. A block inside an admonition (``!!! note``, ``!!! tip``, ...) is an
optional step and is skipped; ``--list-skipped`` lists those blocks.

A substitution table (TOML) records the documented deviations between the page
and a job that runs it on a machine with no terminal. A substitution whose
``match`` text is not on the page is an error (exit code 2), so the job fails
when the page changes.

Exit codes: 0 success, 2 structural problem or stale substitution.

Examples
--------
Print the plan for Ubuntu as a Markdown table::

    python tools/docs_install_steps.py docs/How-to/installation.md \\
        --tab "Debian / Ubuntu" --format table

Write the runnable script::

    python tools/docs_install_steps.py docs/How-to/installation.md \\
        --tab "Debian / Ubuntu" --format script --timing timing.tsv > run.sh
"""

from __future__ import annotations

import argparse
import re
import shlex
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

EXIT_STRUCTURE = 2

_TAB_RE = re.compile(r'^(?P<indent> *)=== +"(?P<name>[^"]+)"\s*$')
_ADMONITION_RE = re.compile(r'^(?P<indent> *)(?:!!!|\?\?\?\+?) +(?P<kind>\w+)')
_FENCE_RE = re.compile(r'^(?P<indent> *)(?P<fence>`{3,})\s*(?P<info>\S*)')
_COMMENT_RE = re.compile(r'\s+#\s.*$')


class InstallStepsError(Exception):
    """A structural problem with the page or the substitution table."""


@dataclass(frozen=True)
class Step:
    """One shell command taken from the page.

    Attributes
    ----------
    line : int
        1-based page line of the command.
    command : str
        Command text as on the page, joined over backslash continuations.
    tab : str or None
        Tab the block sits under, or None for a block shared by all platforms.
    admonition : str or None
        Admonition kind the block sits in, or None for a required step.
    """

    line: int
    command: str
    tab: str | None
    admonition: str | None


@dataclass(frozen=True)
class PlanEntry:
    """A page command with the command the job runs for it.

    Attributes
    ----------
    step : Step
        The page command.
    job_command : str
        Command the job runs (equal to the page command when not substituted).
    reason : str
        Reason of the substitution, empty when the command runs as written.
    """

    step: Step
    job_command: str
    reason: str

    @property
    def substituted(self) -> bool:
        """bool: True when the job runs a command other than the page's."""
        return self.job_command != self.step.command


def _indent_of(line: str) -> int:
    """Return the number of leading spaces of ``line``."""
    return len(line) - len(line.lstrip(' '))


def _block_commands(body: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Split the body of a console block into (page line, command) pairs.

    Parameters
    ----------
    body : list of (int, str)
        Page line number and de-indented text of each line in the block.

    Returns
    -------
    list of (int, str)
        One entry per command. Blank lines and whole-line comments are dropped,
        a leading ``$ `` prompt is removed and backslash continuations are
        joined into the command that starts on the first line.
    """
    commands: list[tuple[int, str]] = []
    pending: list[str] = []
    start = 0
    for number, text in body:
        stripped = text.strip()
        if not pending and (not stripped or stripped.startswith('#')):
            continue
        if not pending:
            start = number
            if stripped.startswith('$ '):
                stripped = stripped[2:]
        if stripped.endswith('\\'):
            pending.append(stripped[:-1].rstrip())
            continue
        pending.append(stripped)
        commands.append((start, ' '.join(pending)))
        pending = []
    if pending:
        commands.append((start, ' '.join(pending)))
    return commands


def parse_page(text: str) -> list[Step]:
    """Extract every command of every ``console`` block, in page order.

    Parameters
    ----------
    text : str
        Content of the Markdown page.

    Returns
    -------
    list of Step
        Commands of all tabs, with their tab and admonition context. Filtering
        by tab is done by :func:`select_steps`.
    """
    lines = text.splitlines()
    # Each container is (indent, kind, name) with kind 'tab' or 'admonition'.
    containers: list[tuple[int, str, str]] = []
    steps: list[Step] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        indent = _indent_of(line)
        while containers and indent <= containers[-1][0]:
            containers.pop()
        tab_match = _TAB_RE.match(line)
        adm_match = _ADMONITION_RE.match(line)
        fence_match = _FENCE_RE.match(line)
        if tab_match:
            containers.append((indent, 'tab', tab_match.group('name')))
        elif adm_match:
            containers.append((indent, 'admonition', adm_match.group('kind')))
        elif fence_match:
            fence = fence_match.group('fence')
            body: list[tuple[int, str]] = []
            j = i + 1
            while j < len(lines):
                closing = lines[j].strip()
                if closing.startswith(fence) and set(closing) == {'`'}:
                    break
                body.append((j + 1, lines[j][indent:] if lines[j].strip() else ''))
                j += 1
            if fence_match.group('info') == 'console':
                tab = next((n for _, k, n in containers if k == 'tab'), None)
                adm = next((n for _, k, n in containers if k == 'admonition'), None)
                for number, command in _block_commands(body):
                    steps.append(Step(number, command, tab, adm))
            i = j
        i += 1
    return steps


def select_steps(steps: list[Step], tab: str | None) -> tuple[list[Step], list[Step]]:
    """Split steps into required and optional ones for one tab.

    Parameters
    ----------
    steps : list of Step
        Output of :func:`parse_page`.
    tab : str or None
        Tab name to select. None keeps only the blocks shared by all platforms.

    Returns
    -------
    required, optional : tuple of list of Step
        Steps to run in page order, and the steps inside admonitions.

    Raises
    ------
    InstallStepsError
        If ``tab`` names no tab on the page or the selection has no command.
    """
    known = sorted({s.tab for s in steps if s.tab is not None})
    if tab is not None and tab not in known:
        raise InstallStepsError(
            f'unknown tab {tab!r}; the page has: {", ".join(known) or "(none)"}'
        )
    chosen = [s for s in steps if s.tab is None or s.tab == tab]
    required = [s for s in chosen if s.admonition is None]
    optional = [s for s in chosen if s.admonition is not None]
    if not required:
        raise InstallStepsError('no console block found for the selection')
    return required, optional


def load_substitutions(path: Path) -> list[dict[str, str]]:
    """Read the substitution table.

    Parameters
    ----------
    path : pathlib.Path
        TOML file with an array of tables ``[[substitution]]``, each with the
        keys ``match``, ``replace`` and ``reason``.

    Returns
    -------
    list of dict
        The entries, validated for the three keys.

    Raises
    ------
    InstallStepsError
        If the file is unreadable or an entry lacks a key or repeats a match.
    """
    try:
        data = tomllib.loads(path.read_text(encoding='utf-8'))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise InstallStepsError(f'cannot read substitutions {path}: {exc}') from exc
    entries = data.get('substitution', [])
    seen: set[str] = set()
    for n, entry in enumerate(entries, start=1):
        for key in ('match', 'replace', 'reason'):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise InstallStepsError(f'substitution {n} in {path} has no {key!r}')
        if entry['match'] in seen:
            raise InstallStepsError(f'substitution {n} in {path} repeats a match')
        seen.add(entry['match'])
    return entries


def build_plan(steps: list[Step], substitutions: list[dict[str, str]]) -> list[PlanEntry]:
    """Apply the substitutions to the required steps.

    A substitution matches a step whose command text equals ``match``, either
    as on the page or without its trailing shell comment.

    Parameters
    ----------
    steps : list of Step
        Required steps in page order.
    substitutions : list of dict
        Entries from :func:`load_substitutions`.

    Returns
    -------
    list of PlanEntry
        One entry per step, in page order.

    Raises
    ------
    InstallStepsError
        If a substitution matches no step on the page.
    """
    plan: list[PlanEntry] = []
    used: set[str] = set()
    for step in steps:
        job, reason = step.command, ''
        for entry in substitutions:
            if entry['match'] in (step.command, _COMMENT_RE.sub('', step.command)):
                job, reason = entry['replace'], entry['reason']
                used.add(entry['match'])
                break
        plan.append(PlanEntry(step, job, reason))
    stale = [e['match'] for e in substitutions if e['match'] not in used]
    if stale:
        listed = '; '.join(repr(m) for m in stale)
        raise InstallStepsError(f'substitution matches no command on the page: {listed}')
    return plan


def _cell(text: str) -> str:
    """Escape ``text`` for a Markdown table cell."""
    return '`' + text.replace('|', '\\|').replace('`', "'") + '`'


def render_table(plan: list[PlanEntry]) -> str:
    """Render the plan as a Markdown table.

    A trailing shell comment of a page command is shown in its own note after
    the command, so the displayed command is the one that runs.

    Parameters
    ----------
    plan : list of PlanEntry
        Output of :func:`build_plan`.

    Returns
    -------
    str
        Markdown table with the columns page line, page command, job command
        and reason.
    """
    rows = ['| Page line | Page command | Job command | Reason |', '|---|---|---|---|']
    for entry in plan:
        page = _COMMENT_RE.sub('', entry.step.command)
        comment = entry.step.command[len(page) :].strip()
        page_cell = _cell(page) + (f' (page comment: {_cell(comment)})' if comment else '')
        job_cell = _cell(entry.job_command) if entry.substituted else 'as written'
        reason = entry.reason.replace('|', '\\|')
        rows.append(f'| {entry.step.line} | {page_cell} | {job_cell} | {reason} |')
    return '\n'.join(rows) + '\n'


def render_skipped(optional: list[Step]) -> str:
    """List the commands of the admonition blocks that the plan skips.

    Parameters
    ----------
    optional : list of Step
        Steps inside admonitions.

    Returns
    -------
    str
        One line per skipped command with page line and admonition kind.
    """
    if not optional:
        return 'No optional steps.\n'
    return ''.join(f'[page line {s.line}] ({s.admonition}) {s.command}\n' for s in optional)


def render_script(plan: list[PlanEntry], timing: str) -> str:
    """Render the plan as a bash script that runs in one shell session.

    Each command runs at the top level of the script, so ``cd``, ``source``
    and ``conda activate`` carry over to the next command. The script records
    wall time and exit code per command to ``timing`` (a TSV file) and stops
    at the first non-zero exit code. It handles the exit code itself, so it
    works with and without ``bash -e``.

    Parameters
    ----------
    plan : list of PlanEntry
        Output of :func:`build_plan`.
    timing : str
        Path of the TSV file to write.

    Returns
    -------
    str
        The bash script.
    """
    out = [
        '#!/usr/bin/env bash',
        '# Generated from the install page by tools/docs_install_steps.py',
        'set +e',
        f'__timing={shlex.quote(timing)}',
        'printf \'page_line\\twall_seconds\\texit_code\\tcommand\\n\' > "$__timing"',
        '__now() { printf "%s" "${EPOCHREALTIME:-$(date +%s)}"; }',
    ]
    for entry in plan:
        label = f'[page line {entry.step.line}] {entry.step.command}'
        out.append(f'printf "%s\\n" {shlex.quote("::group::" + label)}')
        if entry.substituted:
            deviation = (
                f'DEVIATION: page says {entry.step.command}; '
                f'job runs {entry.job_command}; reason: {entry.reason}'
            )
            out.append(f'printf "%s\\n" {shlex.quote(deviation)}')
        out += [
            '__t0=$(__now)',
            entry.job_command,
            '__rc=$?',
            '__t1=$(__now)',
            '__wall=$(awk -v a="$__t0" -v b="$__t1" \'BEGIN{printf "%.1f", b-a}\')',
            'printf "%s\\t%s\\t%s\\t%s\\n" '
            f'{entry.step.line} "$__wall" "$__rc" {shlex.quote(entry.step.command.replace(chr(9), " "))}'
            ' >> "$__timing"',
            'echo "::endgroup::"',
            f'if [ "$__rc" -ne 0 ]; then echo "::error::[page line {entry.step.line}] exit code $__rc"; exit "$__rc"; fi',
        ]
    return '\n'.join(out) + '\n'


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('page', type=Path, help='Markdown install page')
    parser.add_argument('--tab', help='platform tab name, for example "Debian / Ubuntu"')
    parser.add_argument(
        '--substitutions', type=Path, help='TOML table of documented deviations'
    )
    parser.add_argument('--format', choices=('table', 'script', 'commands'), default='commands')
    parser.add_argument(
        '--timing', default='install_timing.tsv', help='TSV path written by the script'
    )
    parser.add_argument(
        '--list-skipped', action='store_true', help='list the optional admonition steps'
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line tool.

    Parameters
    ----------
    argv : list of str, optional
        Arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code: 0 on success, 2 on a structural problem or stale substitution.
    """
    args = build_parser().parse_args(argv)
    try:
        text = args.page.read_text(encoding='utf-8')
    except OSError as exc:
        print(f'error: cannot read {args.page}: {exc}', file=sys.stderr)
        return EXIT_STRUCTURE
    try:
        required, optional = select_steps(parse_page(text), args.tab)
        subs = load_substitutions(args.substitutions) if args.substitutions else []
        plan = build_plan(required, subs)
    except InstallStepsError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return EXIT_STRUCTURE
    if args.list_skipped:
        sys.stdout.write(render_skipped(optional))
    elif args.format == 'table':
        sys.stdout.write(render_table(plan))
    elif args.format == 'script':
        sys.stdout.write(render_script(plan, args.timing))
    else:
        sys.stdout.write(''.join(f'{e.job_command}\n' for e in plan))
    return 0


if __name__ == '__main__':
    sys.exit(main())
