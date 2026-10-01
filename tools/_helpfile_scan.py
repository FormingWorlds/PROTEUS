#!/usr/bin/env python3
"""Static producer/consumer scan of helpfile columns across ``src/proteus``.

Walks every source file for reads and writes of ``hf_row`` (and the frame
aliases the plotting code uses) and attributes each helpfile column to the
code that writes it. Literal subscripts, templated subscripts inside species
loops, tuple-literal loops, backend return dicts merged by the wrappers
(declared in ``MERGE_SITES``), and the registry-driven CALLIOPE merge are all
resolved statically; anything else is recorded as an unresolved event with
its file and line, never silently dropped.

This module is imported by ``generate_output_reference.py``; no CLI.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from collections import Counter
from dataclasses import dataclass

import _docgen

REPO_ROOT = _docgen.REPO_ROOT
SRC = REPO_ROOT / 'src' / 'proteus'

# Names that hold the live helpfile row (writes count as production) and
# frame aliases that only ever read columns (plots, termination checks).
# ``new_row`` is the row dict inside ExtendHelpfile and its helpers.
ROW_NAMES = {'hf_row', 'new_row'}
FRAME_NAMES = {'hf_all', 'hf', 'hf_crop', 'hf_early', 'row'}

# Species-list names resolvable to a domain of key expansions.
DOMAIN_LISTS = {
    'gas_list',
    'element_list',
    'vol_list',
    'noble_gases',
    'vap_list',
    'vol_gas_list',
    'vol_element_list',
    'vap_element_list',
    'VOLATILE_EOS_MAP',
}

# Backend functions whose returned dict the area wrapper merges into hf_row.
# Key renames applied at the merge boundary are declared per site.
MERGE_SITES = [
    ('atmos_clim/agni.py', 'run_agni', {'albedo': 'bond_albedo'}),
    ('atmos_clim/janus.py', 'RunJANUS', {'albedo': 'bond_albedo'}),
    ('atmos_clim/dummy.py', 'RunDummyAtm', {'albedo': 'bond_albedo'}),
    ('interior_energetics/spider.py', 'ReadSPIDER', {}),
    ('interior_energetics/aragog.py', '_build_helpfile_output', {}),
    ('interior_energetics/aragog_jax.py', '_extract_output', {}),
    ('interior_energetics/boundary.py', 'run_solver', {}),
    ('interior_energetics/dummy.py', 'run_dummy_int', {}),
]


@dataclass(frozen=True)
class TemplateOverride:
    """Declared domain expansion and possibility tag for a templated access site."""

    domains: tuple[str, ...]
    possible: bool = False


_ELEMENTS = TemplateOverride(('element_list',))
_GASES = TemplateOverride(('gas_list',))
_BOTH = TemplateOverride(('gas_list', 'element_list'))
_VAPS = TemplateOverride(('vap_list',))
_GASES_POSSIBLE = TemplateOverride(('gas_list',), possible=True)
_GASES_VOL = TemplateOverride(('gas_list', 'vol_element_list'))

# Templated access sites whose loop domain cannot be recovered statically.
# Values name domain lists; downstream code trims them against the schema.
TEMPLATE_OVERRIDES: dict[tuple[str, str, str], TemplateOverride] = {
    ('accretion/wrapper.py', '_partition_impactor_content', '<?>_kg_atm'): _ELEMENTS,
    ('accretion/wrapper.py', '_partition_impactor_content', '<?>_kg_total'): _ELEMENTS,
    ('accretion/wrapper.py', '_apply_volatile_consequences', '<?>_kg_total'): _ELEMENTS,
    ('accretion/wrapper.py', '_restore_volatile_budgets', '<?>_kg_total'): _ELEMENTS,
    ('escape/wrapper.py', 'calc_new_elements', '<?>_kg_total'): _ELEMENTS,
    ('escape/wrapper.py', 'run_escape', '<?>_kg_total'): _ELEMENTS,
    ('escape/common.py', 'calc_unfract_fluxes', 'esc_rate_<?>'): _ELEMENTS,
    ('escape/boreas.py', '_set_boreas_params', '<?>_vmr_xuv'): _GASES_POSSIBLE,
    ('outgas/calliope.py', 'calc_target_masses', '<?>_kg_total'): _ELEMENTS,
    ('outgas/atmodeller.py', '_populate_volatile_element_reservoirs', '<?>_kg_atm'): _ELEMENTS,
    (
        'outgas/atmodeller.py',
        '_populate_volatile_element_reservoirs',
        '<?>_kg_liquid',
    ): _ELEMENTS,
    (
        'outgas/atmodeller.py',
        '_populate_volatile_element_reservoirs',
        '<?>_kg_solid',
    ): _ELEMENTS,
    ('outgas/atmodeller.py', '_total_volatile_oxygen_kg', '<?>_kg_atm'): _GASES_POSSIBLE,
    ('outgas/atmodeller.py', '_total_volatile_oxygen_kg', '<?>_kg_liquid'): _GASES_POSSIBLE,
    ('outgas/atmodeller.py', 'calc_surface_pressures_atmodeller', '<?>_bar'): _GASES,
    ('outgas/atmodeller.py', 'calc_surface_pressures_atmodeller', '<?>_vmr'): _GASES,
    ('outgas/atmodeller.py', 'calc_surface_pressures_atmodeller', '<?>_kg_atm'): _GASES,
    ('outgas/atmodeller.py', 'calc_surface_pressures_atmodeller', '<?>_kg_liquid'): _GASES,
    ('outgas/atmodeller.py', 'calc_surface_pressures_atmodeller', '<?>_kg_solid'): _GASES_VOL,
    ('outgas/atmodeller.py', 'calc_surface_pressures_atmodeller', '<?>_kg_total'): _GASES,
    ('outgas/atmodeller.py', 'calc_surface_pressures_atmodeller', '<?>_mol_atm'): _GASES,
    ('outgas/atmodeller.py', 'calc_surface_pressures_atmodeller', '<?>_mol_liquid'): _GASES,
    ('outgas/atmodeller.py', 'calc_surface_pressures_atmodeller', '<?>_mol_solid'): _GASES,
    ('outgas/atmodeller.py', 'calc_surface_pressures_atmodeller', '<?>_mol_total'): _GASES,
    ('outgas/dummy.py', 'calc_surface_pressures_dummy', '<?>_bar'): _GASES,
    ('outgas/dummy.py', 'calc_surface_pressures_dummy', '<?>_kg_atm'): _BOTH,
    ('outgas/dummy.py', 'calc_surface_pressures_dummy', '<?>_kg_liquid'): _BOTH,
    ('outgas/dummy.py', 'calc_surface_pressures_dummy', '<?>_kg_solid'): _BOTH,
    ('outgas/dummy.py', 'calc_surface_pressures_dummy', '<?>_kg_total'): _BOTH,
    ('outgas/dummy.py', 'calc_surface_pressures_dummy', '<?>_mol_atm'): _GASES,
    ('outgas/dummy.py', 'calc_surface_pressures_dummy', '<?>_mol_liquid'): _GASES,
    ('outgas/dummy.py', 'calc_surface_pressures_dummy', '<?>_mol_solid'): _GASES,
    ('outgas/dummy.py', 'calc_surface_pressures_dummy', '<?>_mol_total'): _GASES,
    ('outgas/lavatmos.py', 'run_vapourisation', '<?>_bar'): _VAPS,
    ('outgas/lavatmos.py', 'run_vapourisation', '<?>_vmr'): _VAPS,
    ('outgas/lavatmos.py', 'run_vapourisation', '<?>_kg_atm'): _ELEMENTS,
}

# Dynamic-key writes that are not producers: save/restore of overridden
# values, carry-forward of previously converged values, and the wrapper
# merge loops whose sources are declared in MERGE_SITES.
SUPPRESSED_DYNAMIC_WRITES = {
    ('proteus.py', 'start'),
    ('atmos_clim/wrapper.py', 'carry_converged_levels'),
    ('atmos_clim/wrapper.py', 'run_atmosphere'),
    ('interior_energetics/wrapper.py', 'run_interior'),
    # The mass-ratio loop assembles its key in a local; EXTRA_PRODUCERS
    # declares the full expansion for it.
    ('outgas/wrapper.py', 'run_outgassing'),
    # hf_row.update(saved) restores of pre-call snapshots.
    ('interior_energetics/wrapper.py', '_solve_structure_with_adiabat_or_rollback'),
    ('interior_energetics/wrapper.py', 'update_structure_from_interior'),
    # Impact re-melt rewrites melt-state columns run_dummy_int already produces.
    ('interior_energetics/wrapper.py', '_remelt_scalar_backend'),
    # hf_row.clear(); hf_row.update(snapshot) restores the pre-substep state
    # on a rejected adaptive step; shared by evolve_orbit_star (orbit.py) and
    # evolve_orbit_satellite (satellite.py).
    ('orbit/common.py', 'run_adaptive_orbit_substeps'),
}

# Producers that assemble their key through a local variable the visitor
# cannot follow: run_outgassing derives every element mass-ratio column.
EXTRA_PRODUCERS = [
    ('outgas/wrapper.py', 'run_outgassing', '<e2>/<e1>_atm'),
]


class ScanError(_docgen.DocgenError):
    """The scan hit a shape it cannot attribute.

    A subclass of DocgenError so the generator CLIs map it to exit code 2
    (structural error) without a separate handler.
    """


def _species_lists() -> dict[str, list]:
    if 'proteus' not in importlib.sys.modules:
        _docgen.import_proteus_config()
    constants = importlib.import_module('proteus.utils.constants')
    return {name: list(getattr(constants, name)) for name in DOMAIN_LISTS}


def expected_registry_keys() -> list[str]:
    """The outgassing copy-registry, executed from the checkout.

    Loaded from its file directly: importing ``proteus.outgas`` would execute
    the package ``__init__``, which pulls the outgassing backends (calliope
    and friends) that the docs-freshness environment does not install.
    """
    if 'proteus' not in importlib.sys.modules:
        _docgen.import_proteus_config()
    spec = importlib.util.spec_from_file_location(
        '_proteus_outgas_common', SRC / 'outgas' / 'common.py'
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.expected_keys())


def _row_name(node) -> str | None:
    """The helpfile-row or frame alias a subscript targets, if any."""
    value = node.value
    if isinstance(value, ast.Name):
        name = value.id
    elif isinstance(value, ast.Attribute):
        name = value.attr
    else:
        return None
    return name if name in ROW_NAMES | FRAME_NAMES else None


def _template_of(key_node) -> tuple[str, str, str] | None:
    """(prefix, varname, suffix) for a one-variable templated key."""
    if isinstance(key_node, ast.BinOp) and isinstance(key_node.op, ast.Add):
        left, right = key_node.left, key_node.right
        if isinstance(left, ast.Name) and isinstance(right, ast.Constant):
            return '', left.id, right.value
        if isinstance(left, ast.Constant) and isinstance(right, ast.Name):
            return left.value, right.id, ''
    if isinstance(key_node, ast.JoinedStr):
        prefix = suffix = ''
        var = None
        for part in key_node.values:
            if isinstance(part, ast.Constant):
                if var is None:
                    prefix += part.value
                else:
                    suffix += part.value
            elif isinstance(part, ast.FormattedValue) and isinstance(part.value, ast.Name):
                if var is not None:
                    return None
                var = part.value.id
        if var is not None:
            return prefix, var, suffix
    return None


def _selects_frame_rows(node: ast.Subscript) -> bool:
    """Whether a frame subscript selects rows (mask, slice, arithmetic), not a column."""
    if _row_name(node) not in FRAME_NAMES:
        return False
    sl = node.slice
    if isinstance(sl, (ast.Compare, ast.Slice, ast.UnaryOp, ast.BoolOp)):
        return True
    return isinstance(sl, ast.BinOp) and not isinstance(sl.op, ast.Add)


def _target_name(target) -> str | None:
    """The key-carrying variable name in a loop or comprehension target."""
    if isinstance(target, ast.Name):
        return target.id
    if (
        isinstance(target, (ast.Tuple, ast.List))
        and target.elts
        and isinstance(target.elts[0], ast.Name)
    ):
        return target.elts[0].id
    return None


def _is_bare_target(elt: ast.AST, target: ast.AST) -> bool:
    """Whether the element expression is the unmodified loop variable."""
    return isinstance(elt, ast.Name) and isinstance(target, ast.Name) and elt.id == target.id


def _collect_bound_names(target: ast.AST) -> list[str]:
    """Collect all variable names bound by an assignment target."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Starred):
        return _collect_bound_names(target.value)
    if isinstance(target, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in target.elts:
            names.extend(_collect_bound_names(elt))
        return names
    return []


_MUTATORS = ('append', 'extend', 'insert', 'update', 'add', 'remove', 'pop', 'clear')
_COMPS = (ast.GeneratorExp, ast.ListComp, ast.SetComp)


class HfRowVisitor(ast.NodeVisitor):
    """Collect helpfile reads/writes in one file, tracking loop domains."""

    def __init__(self, rel_file: str, species: dict[str, list]):
        self.rel_file = rel_file
        self.species = species
        self.func_stack: list[str] = []
        self.loop_domains: dict[str, str] = {}  # loop var -> domain-list name
        self.module_constants: dict[str, list[str]] = {}
        self.scopes: list[dict[str, ast.AST | None]] = [{}]
        self.writes: list[tuple[str, str]] = []  # (key, function)
        self.reads: list[tuple[str, bool]] = []  # (key, is_possible)
        self.unresolved: list[tuple[int, str, str, str]] = []  # (lineno, reason, kind, func)

    @property
    def local_vars(self) -> dict[str, ast.AST | None]:
        return self.scopes[-1]

    # -- context tracking ---------------------------------------------------

    def visit_Module(self, node):
        assigned: Counter[str] = Counter()
        mutated: set[str] = set()
        candidates: dict[str, list[str]] = {}
        module_const_strings: dict[str, ast.Constant] = {}

        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                target_name = stmt.targets[0].id
                dom = self._domain_of_iter(stmt.value)
                if dom:
                    candidates[target_name] = self._expand_domain(dom)
                elif isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                    module_const_strings[target_name] = stmt.value

        for sub in ast.walk(node):
            if sub is node:
                continue
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                        mutated.add(t.value.id)
                    assigned.update(_collect_bound_names(t))
            elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                if isinstance(sub.target, ast.Subscript) and isinstance(
                    sub.target.value, ast.Name
                ):
                    mutated.add(sub.target.value.id)
                assigned.update(_collect_bound_names(sub.target))
            elif isinstance(sub, (ast.NamedExpr, ast.For, ast.AsyncFor)):
                assigned.update(_collect_bound_names(sub.target))
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                assigned[sub.name] += 1
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                assigned.update(alias.asname or alias.name.split('.')[0] for alias in sub.names)
            elif isinstance(sub, ast.withitem) and sub.optional_vars is not None:
                assigned.update(_collect_bound_names(sub.optional_vars))
            elif isinstance(sub, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)) and sub.name:
                assigned[sub.name] += 1
            elif isinstance(sub, ast.MatchMapping) and sub.rest:
                assigned[sub.rest] += 1
            elif isinstance(sub, (ast.Global, ast.Nonlocal)):
                mutated.update(sub.names)
            elif isinstance(sub, ast.Delete):
                for t in sub.targets:
                    mutated.update(_collect_bound_names(t))
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
                        mutated.add(t.value.id)
            elif isinstance(sub, ast.Call):
                func = sub.func
                if (
                    isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Name)
                    and func.attr in _MUTATORS
                ):
                    mutated.add(func.value.id)

        for name, vals in candidates.items():
            if assigned[name] == 1 and name not in mutated:
                self.module_constants[name] = vals

        for name, const_node in module_const_strings.items():
            if assigned[name] == 1 and name not in mutated:
                self.scopes[0][name] = const_node
            else:
                self.scopes[0][name] = None

        self._module_scanned = True
        self.generic_visit(node)

    def _invalidate_args(self, args: ast.arguments) -> None:
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
            if arg:
                self.local_vars[arg.arg] = None

    def _visit_scope(self, node):
        new_scope: dict[str, ast.AST | None] = {}
        self.scopes.append(new_scope)
        self._invalidate_args(node.args)
        self.generic_visit(node)
        self.scopes.pop()

    visit_Lambda = _visit_scope

    def _visit_func(self, node):
        self._invalidate(node.name)
        self.func_stack.append(node.name)
        self._visit_scope(node)
        self.func_stack.pop()

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def visit_ClassDef(self, node):
        self._invalidate(node.name)
        self.generic_visit(node)

    def _bind(self, name: str, value: ast.AST | None) -> None:
        """Track ``name`` as a constant string only on its first binding in scope."""
        is_str = isinstance(value, ast.Constant) and isinstance(value.value, str)
        if len(self.scopes) == 1 and getattr(self, '_module_scanned', False):
            if name not in self.scopes[0]:
                self.scopes[0][name] = None
            self.loop_domains.pop(name, None)
            return
        self.local_vars[name] = value if is_str and name not in self.local_vars else None
        self.loop_domains.pop(name, None)

    def _invalidate(self, target: ast.AST | str | None) -> None:
        """Clear any constant binding or loop domain for target variables."""
        for name in [target] if isinstance(target, str) else _collect_bound_names(target):
            self.local_vars[name] = None
            self.loop_domains.pop(name, None)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._bind(target.id, node.value)
            else:
                self._invalidate(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            self._bind(node.target.id, node.value)
        else:
            self._invalidate(node.target)
        self.generic_visit(node)

    def _visit_target(self, node):
        self._invalidate(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._invalidate(node.target)
        target = node.target
        if (
            isinstance(target, ast.Subscript)
            and _row_name(target) is not None
            and not _selects_frame_rows(target)
        ):
            self._record(target.slice, node.lineno, is_write=False, is_get=True)
            self._record(target.slice, node.lineno, is_write=True, is_get=False)
            self.visit(target.slice)
            self.visit(node.value)
            return
        self.generic_visit(node)

    visit_NamedExpr = _visit_target
    visit_comprehension = _visit_target

    def visit_withitem(self, node):
        self._invalidate(node.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self._invalidate(node.name)
        self.generic_visit(node)

    def visit_Delete(self, node):
        for target in node.targets:
            self._invalidate(target)
        self.generic_visit(node)

    def visit_Global(self, node):
        for name in node.names:
            self.scopes[0][name] = None
            self.module_constants.pop(name, None)
            self._invalidate(name)

    def visit_Nonlocal(self, node):
        for name in node.names:
            for s in reversed(self.scopes[:-1]):
                if name in s:
                    s[name] = None
                    break
            self._invalidate(name)

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname or alias.name.split('.')[0]
            self._invalidate(name)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            target = alias.asname or alias.name
            if (
                node.module
                and node.module.endswith(('constants', 'proteus.utils.constants'))
                and alias.name in DOMAIN_LISTS
                and not alias.asname
            ):
                self.local_vars.pop(target, None)
                continue
            self._invalidate(target)

    def _visit_match_capture(self, node):
        self._invalidate(getattr(node, 'name', None) or getattr(node, 'rest', None))
        self.generic_visit(node)

    visit_MatchAs = visit_MatchStar = visit_MatchMapping = _visit_match_capture

    def _visit_comp(self, node):
        old_locals = dict(self.local_vars)
        old_domains = dict(self.loop_domains)
        for gen in node.generators:
            self._invalidate(gen.target)
            self.visit(gen.iter)
            domain = self._domain_of_iter(gen.iter)
            target_name = _target_name(gen.target)
            if domain and target_name:
                self.loop_domains[target_name] = domain
            for if_expr in gen.ifs:
                self.visit(if_expr)
        if isinstance(node, ast.DictComp):
            self.visit(node.key)
            self.visit(node.value)
        else:
            self.visit(node.elt)
        self.local_vars.clear()
        self.local_vars.update(old_locals)
        self.loop_domains = old_domains

        # Walrus operator bindings escape comprehension into enclosing scope
        for sub in ast.walk(node):
            if isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name):
                self._invalidate(sub.target.id)

    visit_ListComp = _visit_comp
    visit_SetComp = _visit_comp
    visit_GeneratorExp = _visit_comp
    visit_DictComp = _visit_comp

    def _visit_for(self, node):
        old_domains = dict(self.loop_domains)
        self._invalidate(node.target)
        target_name = _target_name(node.target)
        domain = self._domain_of_iter(node.iter)
        if domain and target_name:
            self.loop_domains[target_name] = domain
        self.generic_visit(node)
        self.loop_domains = old_domains

    visit_For = _visit_for
    visit_AsyncFor = _visit_for

    def _domain_of_iter(self, iter_node) -> str | None:
        if isinstance(iter_node, ast.Name):
            if iter_node.id in DOMAIN_LISTS:
                if len(self.scopes) > 1 and iter_node.id in self.local_vars:
                    return None
                return iter_node.id
            if iter_node.id in self.module_constants:
                if iter_node.id in self.local_vars:
                    return None
                return 'literal:' + ','.join(self.module_constants[iter_node.id])
        if isinstance(iter_node, (ast.Tuple, ast.List)) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in iter_node.elts
        ):
            return 'literal:' + ','.join(e.value for e in iter_node.elts)
        if isinstance(iter_node, ast.Dict) and all(
            isinstance(k, ast.Constant) and isinstance(k.value, str) for k in iter_node.keys
        ):
            return 'literal:' + ','.join(k.value for k in iter_node.keys)
        if isinstance(iter_node, ast.Call):
            func = iter_node.func
            if isinstance(func, ast.Name) and func.id == 'expected_keys':
                return 'registry:expected_keys'
            if isinstance(func, ast.Attribute) and func.attr in ('keys', 'items'):
                inner_domain = self._domain_of_iter(func.value)
                if inner_domain:
                    return inner_domain
            if (
                isinstance(func, ast.Name)
                and func.id in ('tuple', 'list', 'set')
                and iter_node.args
            ):
                arg = iter_node.args[0]
                if isinstance(arg, _COMPS):
                    return self._domain_of_iter(arg)
        if isinstance(iter_node, _COMPS):
            gen = iter_node.generators[0]
            if not _is_bare_target(iter_node.elt, gen.target):
                return None
            return self._domain_of_iter(gen.iter)
        if isinstance(iter_node, ast.BinOp) and isinstance(iter_node.op, ast.Add):
            left = self._domain_of_iter(iter_node.left)
            right = self._domain_of_iter(iter_node.right)
            if left and right:
                vals = self._expand_domain(left) + self._expand_domain(right)
                return 'literal:' + ','.join(vals)
        return None

    # -- subscripts ---------------------------------------------------------

    def visit_Subscript(self, node):
        name = _row_name(node)
        if name is None:
            self.generic_visit(node)
            return
        if _selects_frame_rows(node):
            self.generic_visit(node)
            return
        is_write = (
            isinstance(node.ctx, (ast.Store, ast.AugStore)) and name in ROW_NAMES | FRAME_NAMES
        )
        is_get = isinstance(node.ctx, ast.Load) and name in ROW_NAMES | FRAME_NAMES
        if not is_write and not is_get:
            self.generic_visit(node)
            return
        self._record(node.slice, node.lineno, is_write, is_get=is_get)
        self.generic_visit(node)

    def visit_Call(self, node):
        # hf_row.get('key', ...) reads a column; hf_row.update(...) writes an
        # unknowable key set and must never pass silently.
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(
            func.value, (ast.Name, ast.Attribute)
        ):
            owner = func.value.id if isinstance(func.value, ast.Name) else func.value.attr
            if owner in ROW_NAMES | FRAME_NAMES:
                if func.attr == 'get' and node.args:
                    self._record(node.args[0], node.lineno, is_write=False, is_get=True)
                elif func.attr == 'update' and owner in ROW_NAMES and not self._suppressed():
                    caller = self.func_stack[-1] if self.func_stack else '<module>'
                    self.unresolved.append(
                        (node.lineno, f'{owner}.update(...) bulk write', 'write', caller)
                    )
        self.generic_visit(node)

    def _suppressed(self) -> bool:
        """Whether any enclosing function is a declared non-producer site."""
        return any((self.rel_file, fn) in SUPPRESSED_DYNAMIC_WRITES for fn in self.func_stack)

    def _record(self, key_node, lineno: int, is_write: bool, is_get: bool = False) -> None:
        func = self.func_stack[-1] if self.func_stack else '<module>'
        keys, is_possible = self._resolve_keys(key_node, lineno, func, is_write, is_get)
        for key in keys:
            if is_write:
                self.writes.append((key, func))
            else:
                self.reads.append((key, is_possible))

    def _resolve_keys(
        self, key_node, lineno: int, func: str, is_write: bool, is_get: bool = False
    ) -> tuple[list[str], bool]:
        if is_get and isinstance(key_node, ast.Name):
            for s in reversed(self.scopes):
                if key_node.id in s:
                    val = s[key_node.id]
                    if isinstance(val, ast.Constant):
                        key_node = val
                    break
        if isinstance(key_node, ast.Constant):
            return ([key_node.value] if isinstance(key_node.value, str) else []), False
        template = _template_of(key_node)
        if template is not None:
            prefix, var, suffix = template
            domain = self.loop_domains.get(var)
            if domain is not None:
                return [f'{prefix}{v}{suffix}' for v in self._expand_domain(domain)], False
            override = TEMPLATE_OVERRIDES.get((self.rel_file, func, f'{prefix}<?>{suffix}'))
            if override is not None:
                values = set()
                for item in override.domains:
                    if item not in self.species:
                        raise ScanError(
                            f'{self.rel_file}::{func}: unknown domain name "{item}" '
                            f'in TEMPLATE_OVERRIDES'
                        )
                    values.update(self.species[item])
                return [f'{prefix}{v}{suffix}' for v in sorted(values)], override.possible
            reason = f'template {prefix}<{var}>{suffix}'
        elif isinstance(key_node, ast.Name) and key_node.id in self.loop_domains:
            return self._expand_domain(self.loop_domains[key_node.id]), False
        else:
            reason = f'dynamic key {ast.unparse(key_node)}'

        if (is_write or is_get) and not self._suppressed():
            self.unresolved.append((lineno, reason, 'write' if is_write else 'read', func))
        return [], False

    def _expand_domain(self, domain: str) -> list[str]:
        if domain.startswith('literal:'):
            return domain.removeprefix('literal:').split(',')
        if domain.startswith('registry:'):
            return expected_registry_keys()
        return self.species[domain]


# ---------------------------------------------------------------------------
# Backend return-dict extraction for the declared merge sites
# ---------------------------------------------------------------------------


def extract_backend_keys(rel_file: str, function: str, species: dict[str, list]) -> set[str]:
    """String keys a backend function stores in or returns as dicts.

    Over-approximates by collecting every dict-literal key, dict-subscript
    store, and dict-literal-driven loop assignment in the function; callers
    intersect the result with the helpfile schema.
    """
    path = SRC / rel_file
    tree = ast.parse(path.read_text())
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            func_node = node
            break
    if func_node is None:
        raise ScanError(f'{rel_file}: function "{function}" not found')

    keys: set[str] = set()
    dict_literals: dict[str, list[str]] = {}
    loop_domains: dict[str, str] = {}

    for node in ast.walk(func_node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Dict):
                literal_keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
                dict_literals[target.id] = literal_keys

    def domain_values(name: str) -> list[str] | None:
        if name in DOMAIN_LISTS:
            return species[name]
        if name in dict_literals:
            return dict_literals[name]
        return None

    for node in ast.walk(func_node):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            iter_node = node.iter
            iter_name = None
            if isinstance(iter_node, ast.Name):
                iter_name = iter_node.id
            elif (
                isinstance(iter_node, ast.Call)
                and isinstance(iter_node.func, ast.Attribute)
                and iter_node.func.attr in ('keys', 'items')
                and isinstance(iter_node.func.value, ast.Name)
            ):
                iter_name = iter_node.func.value.id
            if iter_name and domain_values(iter_name) is not None:
                loop_domains[node.target.id] = iter_name
        elif isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}

    for node in ast.walk(func_node):
        if not (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        key_node = node.slice
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            keys.add(key_node.value)
        elif isinstance(key_node, ast.Name) and key_node.id in loop_domains:
            keys |= set(domain_values(loop_domains[key_node.id]) or [])
        else:
            template = _template_of(key_node)
            if template is not None:
                prefix, var, suffix = template
                if var in loop_domains:
                    values = domain_values(loop_domains[var]) or []
                    keys |= {f'{prefix}{v}{suffix}' for v in values}

    keys |= {k for lits in dict_literals.values() for k in lits}
    return keys


# ---------------------------------------------------------------------------
# Whole-tree scan
# ---------------------------------------------------------------------------


def scan_tree() -> dict:
    """Scan src/proteus and return writes, reads, and unresolved events.

    Returns ``{'writes': [(rel_file, function, key)], 'reads':
    [(rel_file, key, is_possible)], 'unresolved': [(rel_file, lineno, reason, kind, func)]}``.
    """
    species = _species_lists()
    merge_functions = {(f, fn) for f, fn, _renames in MERGE_SITES}
    writes: list[tuple[str, str, str]] = []
    reads: list[tuple[str, str, bool]] = []
    unresolved: list[tuple[str, int, str, str, str]] = []

    for path in sorted(SRC.rglob('*.py')):
        rel = str(path.relative_to(SRC))
        visitor = HfRowVisitor(rel, species)
        visitor.visit(ast.parse(path.read_text()))
        for key, func in visitor.writes:
            writes.append((rel, func, key))
        for key, possible in visitor.reads:
            reads.append((rel, key, possible))
        for lineno, reason, kind, func in visitor.unresolved:
            unresolved.append((rel, lineno, reason, kind, func))

    for rel_file, function, renames in MERGE_SITES:
        for key in extract_backend_keys(rel_file, function, species):
            writes.append((rel_file, function, renames.get(key, key)))
    for key in expected_registry_keys():
        writes.append(('outgas/calliope.py', 'calc_surface_pressures', key))
    for rel_file, function, _pattern in EXTRA_PRODUCERS:
        for e1 in species['element_list']:
            for e2 in species['element_list']:
                if e1 != e2:
                    writes.append((rel_file, function, f'{e2}/{e1}_atm'))

    return {
        'writes': writes,
        'reads': reads,
        'unresolved': unresolved,
        'merge_functions': merge_functions,
    }
