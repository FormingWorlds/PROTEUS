#!/usr/bin/env python3
"""Parse the helpfile column schema out of ``GetHelpfileKeys`` for the docs.

Reads ``src/proteus/utils/coupler.py`` as text and AST (never importing it,
so no heavy runtime dependencies load) and recovers every helpfile column
with its unit, description, and group heading. Loop-generated columns
(per-gas suffixes, per-element rates, element mass ratios) are recorded as
templates and expanded against the species lists in
``proteus.utils.constants``, which imports cleanly through the ``_docgen``
stub. The expansion is cross-checked against an executed
``GetHelpfileKeys()`` in the unit tests.

This module is imported by ``generate_output_reference.py``; no CLI.
"""

from __future__ import annotations

import ast
import importlib
import re

import _docgen

REPO_ROOT = _docgen.REPO_ROOT
COUPLER = REPO_ROOT / 'src' / 'proteus' / 'utils' / 'coupler.py'

# 'name',  # description [unit]   (list entries; name group used by the walk)
_ENTRY_RE = re.compile(r"'([^']+)'\s*,?\s*\)?\s*#\s*(.*?)\s*\[([^\]]*)\]\s*,?\s*$")
_ENTRY_NO_UNIT_RE = re.compile(r"'([^']+)'\s*,?\s*\)?\s*(?:#\s*(.*))?$")
# The unit is the LAST bracketed group in the trailing comment; text after it
# (e.g. "(do not use for conservation)") stays part of the description.
_COMMENT_RE = re.compile(r'#\s*(.*?)\s*$')
_UNIT_RE = re.compile(r'\[([^\][]+)\](?!.*\[)')


def _constants():
    """Species lists from the checkout, via the stub package import."""
    if 'proteus' not in importlib.sys.modules:
        _docgen.import_proteus_config()
    return importlib.import_module('proteus.utils.constants')


def _group_from_comment(line: str, current: str) -> str:
    """Group heading from a comment-only line, or the current group.

    Only the first line of a comment block can open a group, and only when
    its text (truncated at the first ':' or '.') is short enough to be a
    heading rather than prose.
    """
    text = line.strip().lstrip('#').strip()
    text = re.split(r'[:.]', text, maxsplit=1)[0].strip()
    if text and len(text) <= 40:
        return text[0].upper() + text[1:]
    return current


def _function_source() -> tuple[list[str], ast.FunctionDef]:
    source = COUPLER.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'GetHelpfileKeys':
            lines = source.split('\n')
            return lines, node
    raise _docgen.DocgenError('GetHelpfileKeys not found in utils/coupler.py')


def parse_schema() -> list[dict]:
    """One record per helpfile column, in schema order.

    Record shape: ``{name, unit, description, group, origin}`` where origin
    is 'literal' for named columns or the template pattern (e.g.
    '<gas>_kg_atm', 'esc_rate_<element>') for expanded ones.
    """
    lines, func = _function_source()
    constants = _constants()
    gas_list, element_list = constants.gas_list, constants.element_list

    records: list[dict] = []
    group = 'Model tracking'
    prev_was_comment = False

    def line_meta(lineno: int) -> tuple[str, str]:
        raw = lines[lineno - 1]
        comment = _COMMENT_RE.search(raw)
        if not comment:
            return '', ''
        text = comment.group(1).strip().rstrip(',').strip()
        unit_match = _UNIT_RE.search(text)
        if not unit_match:
            return text, ''
        description = (text[: unit_match.start()] + text[unit_match.end() :]).strip()
        return ' '.join(description.split()), unit_match.group(1).strip()

    def add(name: str, lineno: int, origin: str, expand_desc=None, unit_override=None) -> None:
        description, unit = line_meta(lineno)
        if expand_desc is not None:
            description = expand_desc(description)
        records.append(
            {
                'name': name,
                'unit': unit_override or unit or None,
                'description': description,
                'group': group,
                'origin': origin,
            }
        )

    # Walk the raw lines of the list literal for entries and group headers,
    # then the AST for the appends and loops that follow it.
    list_assign = next(n for n in func.body if isinstance(n, ast.Assign))
    for lineno in range(list_assign.lineno, list_assign.end_lineno + 1):
        raw = lines[lineno - 1].strip()
        if raw.startswith('#'):
            if not prev_was_comment:
                group = _group_from_comment(raw, group)
            prev_was_comment = True
            continue
        prev_was_comment = False
        match = _ENTRY_RE.search(raw) or _ENTRY_NO_UNIT_RE.search(raw)
        if raw.startswith("'") and match:
            add(match.group(1), lineno, 'literal')

    # Comment lines between statements are invisible to the AST; sweep the
    # remaining source range line-by-line, dispatching statements via AST
    # nodes indexed by line number.
    stmts = {n.lineno: n for n in func.body if n.lineno > list_assign.end_lineno}
    lineno = list_assign.end_lineno + 1
    prev_was_comment = False
    while lineno <= func.end_lineno:
        raw = lines[lineno - 1].strip()
        node = stmts.get(lineno)
        if node is None:
            if raw.startswith('#'):
                if not prev_was_comment:
                    group = _group_from_comment(raw, group)
                prev_was_comment = True
            elif raw:
                prev_was_comment = False
            lineno += 1
            continue
        prev_was_comment = False
        if isinstance(node, ast.Expr) and _append_arg(node) is not None:
            arg = _append_arg(node)
            if isinstance(arg, ast.Constant):
                add(arg.value, node.lineno, 'literal')
            else:
                raise _docgen.DocgenError(
                    f'coupler.py:{node.lineno}: top-level append with a non-literal key'
                )
        elif isinstance(node, ast.For):
            _expand_loop(node, add, gas_list, element_list)
        lineno = (node.end_lineno or lineno) + 1

    names = [r['name'] for r in records]
    if len(names) != len(set(names)):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise _docgen.DocgenError(f'duplicate helpfile keys parsed: {dupes}')
    return records


def _append_arg(node: ast.Expr):
    """The single argument of a ``keys.append(...)`` call, else None."""
    call = node.value
    if (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == 'append'
        and len(call.args) == 1
    ):
        return call.args[0]
    return None


def _template_parts(arg) -> tuple[str, str, str] | None:
    """(prefix, loopvar, suffix) of a concatenated or f-string key template."""
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Add):
        left, right = arg.left, arg.right
        if isinstance(left, ast.Name) and isinstance(right, ast.Constant):
            return '', left.id, right.value
        if isinstance(left, ast.Constant) and isinstance(right, ast.Name):
            return left.value, right.id, ''
    if isinstance(arg, ast.JoinedStr):
        prefix = suffix = ''
        var = None
        for part in arg.values:
            if isinstance(part, ast.Constant):
                if var is None:
                    prefix += part.value
                else:
                    suffix += part.value
            elif isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name):
                if var is not None:
                    return None  # two variables: not a simple template
                var = part.value.id
        if var is not None:
            return prefix, var, suffix
    return None


def _expand_loop(node: ast.For, add, gas_list: list, element_list: list) -> None:
    """Expand a ``for ... in gas_list/element_list`` append block."""
    iter_name = node.iter.id if isinstance(node.iter, ast.Name) else None
    if iter_name == 'gas_list':
        domain, label = list(gas_list), 'gas'
    elif iter_name == 'element_list':
        domain, label = list(element_list), 'element'
    else:
        raise _docgen.DocgenError(f'coupler.py:{node.lineno}: loop over unrecognised iterable')

    # The element-ratio block is a nested double loop with a dedup guard;
    # recognise it by shape and mirror the guard exactly. The append line
    # carries no unit comment; the ratio is a dimensionless mass ratio, as
    # the block comment above the loop states.
    inner = [n for n in node.body if isinstance(n, ast.For)]
    if inner:
        seen: set[tuple[str, str]] = set()
        for e1 in element_list:
            for e2 in element_list:
                if e1 == e2 or (e2, e1) in seen:
                    continue
                seen.add((e1, e2))
                add(
                    f'{e2}/{e1}_atm',
                    inner[0].body[-1].end_lineno,
                    '<e2>/<e1>_atm',
                    expand_desc=lambda _d, e1=e1, e2=e2: (
                        f'mass ratio of {e2} to {e1} in the atmosphere'
                    ),
                    unit_override='1',
                )
        return

    # An `if <var> in gas_list: continue` guard restricts the domain.
    for stmt in node.body:
        if isinstance(stmt, ast.If) and any(isinstance(s, ast.Continue) for s in stmt.body):
            domain = [d for d in domain if d not in gas_list]

    loopvar = node.target.id if isinstance(node.target, ast.Name) else None
    templates: list[tuple[str, str, int]] = []
    for stmt in node.body:
        if not isinstance(stmt, ast.Expr):
            continue
        arg = _append_arg(stmt)
        if arg is None:
            continue
        parts = _template_parts(arg)
        if parts is None or parts[1] != loopvar:
            raise _docgen.DocgenError(
                f'coupler.py:{stmt.lineno}: unrecognised append inside a species loop'
            )
        templates.append((parts[0], parts[2], stmt.lineno))
    # Species-outer expansion reproduces the source order exactly.
    for value in domain:
        for prefix, suffix, stmt_lineno in templates:
            pattern = f'{prefix}<{label}>{suffix}'
            add(f'{prefix}{value}{suffix}', stmt_lineno, pattern)
