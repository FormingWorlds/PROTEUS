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

Exit status: 0 when clean, 1 when there are findings, 2 on a git or ruff error.
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


def changed_spans(root: Path, base: str) -> dict[str, tuple[bool, list[tuple[int, int]]]]:
    """Map each changed Python file to its new-file flag and touched line spans.

    An added line ``n`` gives the span ``(n, n)``. A deletion after line ``n``
    gives ``(n, n + 1)``, so it touches a node only when the deletion point lies
    inside the node.

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
            '-M',
            '--diff-filter=AMR',
            merge_base,
            '--',
            '*.py',
        ],
        root,
    )
    out: dict[str, tuple[bool, list[tuple[int, int]]]] = {}
    is_new, path = False, None
    for line in diff.splitlines():
        if line.startswith('--- '):
            is_new = line == '--- /dev/null'
        elif line.startswith('+++ '):
            path = line[6:].rstrip('\t')
            out[path] = (is_new, [])
        elif (m := HUNK.match(line)) and path is not None:
            start, count = int(m.group(1)), int(m.group(2) or 1)
            spans = out[path][1]
            if count == 0:
                spans.append((start, start + 1))
            else:
                spans.extend((n, n) for n in range(start, start + count))
    return out


def node_spans(tree: ast.Module) -> list[tuple[int, int, list[tuple[int, int]]]]:
    """List every function and class with its line span and its children's spans.

    Parameters
    ----------
    tree : ast.Module
        Parsed module.

    Returns
    -------
    list of tuple
        ``(first_line, last_line, child_spans)``; the first line includes decorators.
    """
    kinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

    def span(n: ast.AST) -> tuple[int, int]:
        return min([n.lineno] + [d.lineno for d in n.decorator_list]), n.end_lineno

    nodes = []
    for n in ast.walk(tree):
        if isinstance(n, kinds):
            kids = [span(c) for c in ast.walk(n) if c is not n and isinstance(c, kinds)]
            nodes.append((*span(n), kids))
    return nodes


def inside(s: tuple[int, int], lo: int, hi: int) -> bool:
    """Return True when span ``s`` lies within lines ``lo`` to ``hi``."""
    return lo <= s[0] and s[1] <= hi


def docstring_findings(
    root: Path, files: dict[str, tuple[bool, list, ast.Module]]
) -> list[str]:
    """Return ruff ``D`` findings that belong to changed nodes.

    Parameters
    ----------
    root : Path
        Repository root.
    files : dict
        Relative path to ``(is_new, spans, tree)``.

    Returns
    -------
    list of str
        One ``path:line: CODE message`` entry per finding.
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
    found = []
    for d in json.loads(raw):
        rel = Path(d['filename']).resolve().relative_to(root.resolve()).as_posix()
        is_new, spans, tree = files[rel]
        row = d['location']['row']
        holders = [n for n in node_spans(tree) if n[0] <= row <= n[1]]
        if holders:
            lo, hi, kids = min(holders, key=lambda n: n[1] - n[0])
            own = [
                s for s in spans if inside(s, lo, hi) and not any(inside(s, *k) for k in kids)
            ]
            changed = bool(own)
        else:
            doc = tree.body[0] if tree.body and isinstance(tree.body[0], ast.Expr) else None
            changed = is_new or (
                doc is not None and any(inside(s, doc.lineno, doc.end_lineno) for s in spans)
            )
        if changed:
            found.append(f'{rel}:{row}: {d["code"]} {d["message"]}')
    return found


def comment_findings(rel: str, source: str, spans: list, max_lines: int) -> list[str]:
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
    list of str
        One ``path:line: CMT message`` entry per block.
    """
    lines = source.splitlines()
    rows = sorted(
        t.start[0]
        for t in tokenize.generate_tokens(io.StringIO(source).readline)
        if t.type == tokenize.COMMENT and lines[t.start[0] - 1].lstrip().startswith('#')
    )
    blocks: list[list[int]] = []
    for r in rows:
        if blocks and r == blocks[-1][-1] + 1:
            blocks[-1].append(r)
        else:
            blocks.append([r])
    return [
        f'{rel}:{b[0]}: CMT comment block of {len(b)} lines (max {max_lines})'
        for b in blocks
        if len(b) > max_lines and b[0] != 1 and any(inside(s, b[0], b[-1]) for s in spans)
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
        0 when clean, 1 with findings, 2 on a git or ruff error.
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
                found.append(f'{rel}:{e.lineno}: E999 syntax error: {e.msg}')
                continue
            files[rel] = (is_new, spans, tree)
            found += comment_findings(rel, source, spans, args.max_comment_lines)
        found += docstring_findings(root, files)
    except CheckError as e:
        print(f'check_changed_style: {e}', file=sys.stderr)
        return 2
    for f in sorted(found):
        print(f)
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
