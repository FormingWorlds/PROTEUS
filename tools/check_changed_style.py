#!/usr/bin/env python3
"""Check the docstrings and comment blocks of the Python lines a branch changes.

The check compares the working tree with the merge base of ``HEAD`` and a base
ref (default ``origin/main``) and reports two kinds of finding:

* ruff ``D`` rules with the NumPy convention, for each function, method or
  class whose own lines the branch adds or changes (lines of a nested function
  or class count for that node only), and for the module docstring of a new
  file or of a file whose module docstring changes;
* full-line comment blocks longer than ``--max-comment-lines`` lines that
  contain an added or changed line. A block that starts at line 1 (a file
  header) is exempt.

Code the branch does not touch is not checked, so existing violations elsewhere
in a file do not fail the check. The same file is used by every repository of
the PROTEUS ecosystem; keep the copies identical.

Usage::

    python tools/check_changed_style.py [--base REF] [--max-comment-lines N] [PATH ...]

With paths, only those changed files are checked (pre-commit passes the files
it selects, so its ``exclude`` patterns apply).

Exit status: 0 when clean, 1 when there are findings, 2 on a git, ruff or file error.
Requires ``ruff`` in the running Python environment.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path

QUOTES = ('"""', "'''")
HUNK = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@')


class CheckError(Exception):
    """A git or ruff call failed, so the result is unknown."""


def run(cmd: list[str], cwd: Path) -> str:
    """Run a command and return its stdout.

    Parameters
    ----------
    cmd : list of str
        Command and arguments.
    cwd : Path
        Working directory.

    Returns
    -------
    str
        Standard output of the command.

    Raises
    ------
    CheckError
        If the command exits with a non-zero status.
    """
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        raise CheckError(f'{" ".join(cmd[:3])} failed: {res.stderr.strip()}')
    return res.stdout


def changed_spans(root: Path, base: str) -> dict[str, tuple[bool, list]]:
    """Map each changed Python file to its new-file flag and touched line spans.

    An added line ``n`` gives the span ``(n, n, [])``, and a deletion after line
    ``n`` gives ``(n, n + 1, [])``; the first span of each hunk carries the hunk's
    deleted lines, from which :func:`touches` decides which node or comment block
    a boundary change belongs to.

    Parameters
    ----------
    root : Path
        Repository root.
    base : str
        Ref whose merge base with ``HEAD`` is the comparison point.

    Returns
    -------
    dict
        Relative path to ``(is_new, spans)``.
    """
    merge_base = run(['git', 'merge-base', 'HEAD', base], root).strip()
    diff = run(
        [
            'git',
            '-c',
            'core.quotePath=false',
            'diff',
            '-U0',
            '--no-color',
            '--no-ext-diff',
            '--dst-prefix=b/',
            '-M',
            '--diff-filter=AMR',
            merge_base,
            '--',
            '*.py',
        ],
        root,
    )
    out: dict[str, tuple[bool, list]] = {}
    header, is_new, path, deleted = False, False, None, None
    for line in diff.splitlines():
        if line.startswith('diff --git '):
            header, deleted = True, None
        elif header and line.startswith('--- '):
            is_new = line == '--- /dev/null'
        elif header and line.startswith('+++ '):
            path = line[6:].rstrip('\t')
            out[path] = (is_new, [])
        elif (m := HUNK.match(line)) and path is not None:
            header, deleted = False, []
            start, count = int(m.group(1)), int(m.group(2) or 1)
            if count == 0:
                out[path][1].append((start, start + 1, deleted))
            else:
                out[path][1].append((start, start, deleted))
                out[path][1].extend((n, n, []) for n in range(start + 1, start + count))
        elif deleted is not None and line.startswith('-'):
            deleted.append(line[1:])
    return out


def node_spans(tree: ast.Module) -> list[tuple[int, int, int, list]]:
    """List every function and class with its lines, indent and children.

    Parameters
    ----------
    tree : ast.Module
        Parsed module.

    Returns
    -------
    list of tuple
        ``(first_line, last_line, indent, children)``, the first line including
        decorators, each child a ``(first_line, last_line, indent)`` tuple.
    """
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def span(n: ast.AST) -> tuple[int, int, int]:
        return (
            min([n.lineno] + [d.lineno for d in n.decorator_list]),
            n.end_lineno,
            n.col_offset,
        )

    return [
        (*span(n), [span(c) for c in ast.walk(n) if c is not n and isinstance(c, kinds)])
        for n in ast.walk(tree)
        if isinstance(n, kinds)
    ]


def touches(s: tuple, lo: int, hi: int, col: int | None = None) -> bool:
    """Return True when span ``s`` changes lines ``lo`` to ``hi``.

    A deletion just outside the lines counts when the deleted text belonged to
    them: a comment line next to a comment block (``col`` None), a decorator of
    a node at indent ``col``, or a body line deeper than ``col`` after it.
    """
    if lo <= s[0] and s[1] <= hi:
        return True
    if not s[2] or not lo - 1 <= s[0] <= hi:
        return False
    if col is None:
        return (s[2][-1] if s[0] < lo else s[2][0]).lstrip().startswith('#')
    text = [t for t in s[2] if t.strip()] or ['']
    edge = text[-1] if s[0] < lo else text[0]
    indent = len(edge) - len(edge.lstrip())
    if s[0] < lo:
        return indent == col and edge.lstrip().startswith('@')
    return indent > col


def docstring_findings(root: Path, files: dict[str, tuple[bool, list, ast.Module]]) -> list:
    """Return ruff ``D`` findings that belong to changed nodes.

    Parameters
    ----------
    root : Path
        Repository root.
    files : dict
        Relative path to ``(is_new, spans, tree)``.

    Returns
    -------
    list of tuple
        ``(path, line, message)`` per finding.
    """
    if not files:
        return []
    raw = run(
        [
            sys.executable,
            '-m',
            'ruff',
            'check',
            '--isolated',
            '--select',
            'D',
            '--config',
            "lint.pydocstyle.convention = 'numpy'",
            '--output-format',
            'json',
            '--no-cache',
            '--exit-zero',
            '--',
            *files,
        ],
        root,
    )
    nodes = {rel: node_spans(tree) for rel, (_, _, tree) in files.items()}
    found = []
    for d in json.loads(raw):
        rel = Path(d['filename']).resolve().relative_to(root.resolve()).as_posix()
        is_new, spans, tree = files[rel]
        row = d['location']['row']
        module_rule = d['code'] in ('D100', 'D104')
        holders = [] if module_rule else [n for n in nodes[rel] if n[0] <= row <= n[1]]
        if holders:
            lo, hi, col, kids = min(holders, key=lambda n: n[1] - n[0])
            changed = any(
                touches(s, lo, hi, col) and not any(touches(s, *k) for k in kids) for s in spans
            )
        else:
            doc = tree.body[0] if ast.get_docstring(tree) is not None else None
            top = tree.body[0].lineno if tree.body else 1
            changed = (
                is_new
                or (
                    doc is not None
                    and any(touches(s, doc.lineno, doc.end_lineno) for s in spans)
                )
                or any(s[0] <= top and any(q in t for t in s[2] for q in QUOTES) for s in spans)
            )
        if changed:
            found.append((rel, row, f'{d["code"]} {d["message"]}'))
    return found


def comment_findings(rel: str, source: str, spans: list, max_lines: int) -> list:
    """Return comment blocks longer than ``max_lines`` that the branch touches.

    Parameters
    ----------
    rel : str
        Relative path, for the report.
    source : str
        File content.
    spans : list of tuple
        Touched line spans from :func:`changed_spans`.
    max_lines : int
        Longest allowed block of consecutive full-line comments.

    Returns
    -------
    list of tuple
        ``(path, line, message)`` per block.
    """
    blocks: list[list[int]] = []
    for t in tokenize.generate_tokens(io.StringIO(source).readline):
        if t.type != tokenize.COMMENT or not t.line.lstrip().startswith('#'):
            continue
        if blocks and t.start[0] == blocks[-1][-1] + 1:
            blocks[-1].append(t.start[0])
        else:
            blocks.append([t.start[0]])
    return [
        (rel, b[0], f'CMT comment block of {len(b)} lines (max {max_lines})')
        for b in blocks
        if len(b) > max_lines and b[0] != 1 and any(touches(s, b[0], b[-1]) for s in spans)
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the check and print the findings.

    Parameters
    ----------
    argv : list of str, optional
        Command-line arguments; ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        0 when clean, 1 with findings, 2 on a git, ruff or file error.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--base', default='origin/main', help='base ref (default origin/main)')
    ap.add_argument('--max-comment-lines', type=int, default=4)
    ap.add_argument('paths', nargs='*', help='limit the check to these files')
    args = ap.parse_args(argv)
    try:
        root = Path(run(['git', 'rev-parse', '--show-toplevel'], Path.cwd()).strip())
        files, found = {}, []
        wanted = {Path(p).resolve().relative_to(root.resolve()).as_posix() for p in args.paths}
        for rel, (is_new, spans) in changed_spans(root, args.base).items():
            if args.paths and rel not in wanted:
                continue
            source = (root / rel).read_text(encoding='utf-8')
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                found.append((rel, e.lineno or 0, f'E999 syntax error: {e.msg}'))
                continue
            files[rel] = (is_new, spans, tree)
            found += comment_findings(rel, source, spans, args.max_comment_lines)
        found += docstring_findings(root, files)
    except (CheckError, OSError, ValueError, tokenize.TokenError) as e:
        print(f'check_changed_style: {e}', file=sys.stderr)
        return 2
    for rel, row, msg in sorted(found):
        print(f'{rel}:{row}: {msg}')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
