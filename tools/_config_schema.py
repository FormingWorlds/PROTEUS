#!/usr/bin/env python3
"""Extract the PROTEUS configuration schema for the reference generators.

Walks the attrs Config tree from the checkout (via the stub import in
``_docgen``) and produces one record per leaf field: dotted TOML path, display
type, default, choices, bounds, whether the TOML string ``"none"`` maps to
Python None, and a description merged from three source channels (attrs
``metadata['doc']``, PEP 224 field docstrings, class-docstring numpydoc
Attributes blocks). Cross-field validator functions become constraint records
rendered from their docstrings.

This module is imported by ``generate_config_reference.py`` and
``generate_module_map.py``; it has no CLI of its own.

Raises:
    SchemaError   semantic schema problem, e.g. a cross-field validator with
                  no docstring (generators map this to exit code 1)
    DocgenError   unrecognised attrs internals or unreadable source
                  (generators map this to exit code 2)
"""

from __future__ import annotations

import ast
import inspect
import types
import typing

import _docgen

REPO_ROOT = _docgen.REPO_ROOT
CONFIG_DIR = REPO_ROOT / 'src' / 'proteus' / 'config'

# Fields with no source annotation (attrs reports type=None for them); their
# display types are declared here, taken from the class Attributes docstrings.
ANNOTATION_OVERRIDES = {
    'planet.R_int_override': 'float',
    'orbit.axial_period': 'float',
    'interior_struct.zalmoxis.ice_layer_eos': 'str',
    'interior_struct.core_density': 'float | str',
    'interior_struct.core_heatcap': 'float | str',
    'interior_struct.melting_dir': 'str',
    'interior_struct.eos_dir': 'str',
}

# Validator hooks that read config state but enforce nothing; excluded from
# the generated constraints so inert code is not advertised as a rule.
SKIP_VALIDATORS = {'planet_oxygen_mode_explicit', 'valid_escapeboreas'}


class SchemaError(Exception):
    """Semantic schema problem the source must fix (exit code 1)."""


# ---------------------------------------------------------------------------
# Source-level parsing: PEP 224 docstrings and validator functions
# ---------------------------------------------------------------------------


def _attr_chain(node: ast.Attribute) -> str | None:
    """Return the dotted chain of an attribute access rooted at ``instance``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name) and node.id == 'instance':
        return '.'.join(reversed(parts))
    return None


def _assigned_name(stmt: ast.stmt) -> str | None:
    """Field name of an annotated or plain assignment, else None."""
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id
    return None


def parse_source_docs() -> tuple[dict, dict]:
    """Parse the config source files once for docstrings the runtime hides.

    Returns ``(pep224, validators)``: PEP 224 field docstrings keyed by
    ``(class_name, field_name)``, and validator-function records keyed by
    ``(module_stem, function_name)`` with their docstring, whether they read
    cross-field state, and which ``instance`` attributes they touch.
    """
    pep224: dict[tuple[str, str], str] = {}
    validators: dict[tuple[str, str], dict] = {}
    for path in sorted(CONFIG_DIR.glob('*.py')):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for stmt, nxt in zip(node.body, node.body[1:]):
                    name = _assigned_name(stmt)
                    if (
                        name
                        and isinstance(nxt, ast.Expr)
                        and isinstance(nxt.value, ast.Constant)
                        and isinstance(nxt.value.value, str)
                    ):
                        pep224[(node.name, name)] = inspect.cleandoc(nxt.value.value)
            elif isinstance(node, ast.FunctionDef):
                if [a.arg for a in node.args.args] != ['instance', 'attribute', 'value']:
                    continue
                chains = set()
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Attribute):
                        chain = _attr_chain(sub)
                        if chain:
                            chains.add(chain)
                # Keep only maximal chains: 'mors.age_now' subsumes 'mors'.
                touches = sorted(
                    c for c in chains if not any(o.startswith(c + '.') for o in chains)
                )
                validators[(path.stem, node.name)] = {
                    'doc': ast.get_docstring(node),
                    'cross': bool(touches),
                    'touches': touches,
                }
    return pep224, validators


def parse_attributes_block(doc: str | None) -> dict[str, str]:
    """Parse a numpydoc ``Attributes`` block into ``{field: description}``.

    Line-based: an entry starts at ``name: type`` on the base indent; deeper
    lines continue its description; a dash underline or a dedent ends the
    block. Multi-line descriptions are joined with single spaces.
    """
    if not doc:
        return {}
    lines = inspect.cleandoc(doc).split('\n')
    out: dict[str, str] = {}
    idx = None
    for i in range(len(lines) - 1):
        if lines[i].strip() == 'Attributes' and set(lines[i + 1].strip()) == {'-'}:
            idx = i + 2
            break
    if idx is None:
        return {}
    current: str | None = None
    parts: list[str] = []
    for line in lines[idx:]:
        stripped = line.strip()
        if stripped and set(stripped) == {'-'}:
            # Underline of the next section header: drop that header line
            # (already absorbed as a description part) and stop.
            if parts:
                parts.pop()
            break
        entry = None
        if not line.startswith((' ', '\t')) and stripped:
            entry_match = stripped.split(':', 1)
            if len(entry_match) == 2 and entry_match[0].strip().isidentifier():
                entry = entry_match[0].strip()
        if entry:
            if current:
                out[current] = ' '.join(parts).strip()
            current, parts = entry, []
        elif current and stripped:
            parts.append(stripped)
    if current:
        out[current] = ' '.join(parts).strip()
    return out


# ---------------------------------------------------------------------------
# attrs introspection
# ---------------------------------------------------------------------------


def _iter_validators(validator):
    """Yield leaf validators, unwrapping tuples, and_(), and optional()."""
    if validator is None:
        return
    if isinstance(validator, tuple):
        for v in validator:
            yield from _iter_validators(v)
    elif hasattr(validator, '_validators'):  # _AndValidator
        for v in validator._validators:
            yield from _iter_validators(v)
    elif hasattr(validator, 'validator') and type(validator).__name__ == '_OptionalValidator':
        yield from _iter_validators(validator.validator)
    else:
        yield validator


def _accepts_none(field) -> bool:
    """True when the field's converter maps the TOML string 'none' to None."""
    conv = field.converter
    if conv is None:
        return False
    try:
        return conv('none') is None
    except Exception:
        return False


def _classify_validators(field, path: str) -> tuple[list, list[dict], list[str]]:
    """Split a field's validators into choices, bounds, and named functions.

    Raises DocgenError on a validator shape this module does not recognise,
    so a future attrs change fails loudly instead of emitting empty cells.
    """
    choices: list = []
    bounds: list[dict] = []
    named: list[str] = []
    for v in _iter_validators(field.validator):
        options = getattr(v, 'options', None)
        if options is not None:
            choices = list(options)
            continue
        bound = getattr(v, 'bound', None)
        if bound is not None:
            bounds.append({'op': v.compare_op, 'value': bound})
            continue
        if isinstance(v, types.FunctionType):
            named.append(v.__name__)
            continue
        raise _docgen.DocgenError(
            f'unrecognised validator {v!r} on field "{path}"; extend _classify_validators'
        )
    return choices, bounds, named


def _type_hints(cls) -> dict:
    """Resolved type hints for an attrs class; empty on failure."""
    try:
        return typing.get_type_hints(cls)
    except Exception:
        return {}


def _nested_class(field, hint):
    """The attrs class held by this field, from its factory or its hint."""
    factory = getattr(field.default, 'factory', None)
    if factory is not None and _has_attrs(factory):
        return factory
    if hint is not None:
        if isinstance(hint, type) and _has_attrs(hint):
            return hint
        for arg in typing.get_args(hint) or ():
            if isinstance(arg, type) and _has_attrs(arg):
                return arg
    return None


def _has_attrs(obj) -> bool:
    import attrs

    return isinstance(obj, type) and attrs.has(obj)


def _display_type(field, hint, path: str, accepts_none: bool) -> str:
    """Human-facing type string: 'float', 'str or none', 'int or none', ...

    Union members are reduced to what the user actually writes in TOML: the
    NoneType member and the cattrs-only ``str`` member (present so the string
    'none' can be structured before the converter runs) are dropped, and
    'or none' is appended when the converter maps 'none' to None.
    """
    if path in ANNOTATION_OVERRIDES:
        base_names = [ANNOTATION_OVERRIDES[path]]
        saw_none = False
    elif hint is not None:
        origin = typing.get_origin(hint)
        if origin in (typing.Union, types.UnionType):
            members = list(typing.get_args(hint))
            saw_none = type(None) in members
            members = [m for m in members if m is not type(None)]
            if accepts_none and str in members and len(members) > 1:
                members = [m for m in members if m is not str]
            base_names = [getattr(m, '__name__', str(m)) for m in members]
        else:
            base_names = [getattr(hint, '__name__', str(hint).replace('typing.', ''))]
            saw_none = False
    elif isinstance(field.type, str):
        parts = [p.strip() for p in field.type.split('|')]
        saw_none = 'None' in parts
        parts = [p for p in parts if p != 'None']
        if accepts_none and 'str' in parts and len(parts) > 1:
            parts = [p for p in parts if p != 'str']
        base_names = parts
    else:
        raise _docgen.DocgenError(f'field "{path}" has no recoverable type annotation')
    base = ' | '.join(base_names)
    if accepts_none or saw_none:
        base += ' or none'
    return base


def _default_repr(field) -> str:
    """TOML-flavoured default: strings quoted, bools lowercase, None as none."""
    import attrs

    default = field.default
    if default is attrs.NOTHING:
        return 'required'
    factory = getattr(default, 'factory', None)
    if factory is not None:
        default = factory()
    if field.converter is not None:
        # Converters are arbitrary callables and some reject their own field's
        # default (an accepts-none probe, a path that must exist). The raw
        # default is the right thing to display when that happens.
        try:
            default = field.converter(default)
        except Exception:
            pass
    if default is None:
        return 'none'
    if isinstance(default, bool):
        return 'true' if default else 'false'
    if isinstance(default, str):
        return f'"{default}"'
    return repr(default)


# ---------------------------------------------------------------------------
# Schema assembly
# ---------------------------------------------------------------------------


def build_schema() -> dict:
    """Walk the Config tree and return the full schema record.

    Returns ``{'fields': [...], 'constraints': [...]}`` with fields in
    definition order (stable across runs) and constraints sorted by name.
    """
    import attrs

    cfg = _docgen.import_proteus_config()
    pep224, validator_src = parse_source_docs()
    fields: list[dict] = []
    attached: dict[str, list[str]] = {}

    def visit(cls, prefix: str, seen: frozenset) -> None:
        # Best-effort: resolving string annotations needs every name in the
        # defining module's namespace. Where that fails, ``_type_hints``
        # falls back to the raw annotation text.
        try:
            attrs.resolve_types(cls)
        except Exception:
            pass
        hints = _type_hints(cls)
        attr_docs = parse_attributes_block(cls.__doc__)
        for field in attrs.fields(cls):
            path = f'{prefix}.{field.name}' if prefix else field.name
            nested = _nested_class(field, hints.get(field.name))
            if nested is not None and nested not in seen:
                for name in _classify_validators(field, path)[2]:
                    attached.setdefault(name, []).append(path)
                visit(nested, path, seen | {nested})
                continue
            accepts_none = _accepts_none(field)
            choices, bounds, named = _classify_validators(field, path)
            for name in named:
                attached.setdefault(name, []).append(path)
            description, source = _description(field, cls, pep224, attr_docs)
            fields.append(
                {
                    'path': path,
                    'toml_section': path.rsplit('.', 1)[0] if '.' in path else '',
                    'class': cls.__name__,
                    'type': _display_type(field, hints.get(field.name), path, accepts_none),
                    'accepts_none': accepts_none,
                    'default': _default_repr(field),
                    'choices': choices or None,
                    'bounds': bounds or None,
                    'description': description,
                    'doc_source': source,
                }
            )

    visit(cfg.Config, '', frozenset({cfg.Config}))
    constraints = _build_constraints(validator_src, attached)
    return {'fields': fields, 'constraints': constraints}


def _description(field, cls, pep224: dict, attr_docs: dict) -> tuple[str, str]:
    """Merge the three description channels; priority metadata > PEP 224 > Attributes."""
    meta = field.metadata.get('doc') if field.metadata else None
    if meta:
        return ' '.join(str(meta).split()), 'metadata'
    pep = pep224.get((cls.__name__, field.name))
    if pep:
        return ' '.join(pep.split()), 'field_docstring'
    attr = attr_docs.get(field.name)
    if attr:
        return attr, 'attributes'
    return '', 'missing'


def _build_constraints(validator_src: dict, attached: dict[str, list[str]]) -> list[dict]:
    """Constraint records for every attached cross-field validator.

    A cross-field validator without a docstring is a SchemaError: the
    generated docs render constraint prose from those docstrings, so an
    undocumented rule would silently vanish from the reference.
    """
    constraints: list[dict] = []
    missing: list[str] = []
    for (stem, name), info in sorted(validator_src.items()):
        if name in SKIP_VALIDATORS or not info['cross'] or name not in attached:
            continue
        if not info['doc']:
            missing.append(f'{stem}.py:{name}')
            continue
        summary = ' '.join(info['doc'].split('\n\n')[0].split())
        constraints.append(
            {
                'validator': name,
                'module': f'{stem}.py',
                'attached_to': sorted(set(attached[name])),
                'touches': info['touches'],
                'doc': summary,
            }
        )
    if missing:
        raise SchemaError(
            'cross-field validators need a one-line docstring for the generated '
            'constraint list: ' + ', '.join(missing)
        )
    return constraints
