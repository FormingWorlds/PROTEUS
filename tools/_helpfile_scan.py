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

# Templated access sites whose loop domain cannot be recovered statically.
# Values name domain lists; downstream code trims them against the schema.
TEMPLATE_OVERRIDES: dict[tuple[str, str], tuple[str, ...]] = {
    ('accretion/wrapper.py', '<?>_kg_total'): ('element_list',),
    ('accretion/wrapper.py', '<?>_kg_atm'): ('element_list',),
    ('atmos_chem/dummy.py', '<?>_vmr'): (
        'H2O',
        'CO2',
        'N2',
        'H2',
        'CO',
        'CH4',
        'SO2',
        'S2',
        'H2S',
        'NH3',
        'O2',
    ),
    ('escape/wrapper.py', '<?>_kg_total'): ('element_list',),
    ('escape/common.py', 'esc_rate_<?>'): ('element_list',),
    ('interior_energetics/wrapper.py', '<?>_kg_total'): ('vol_element_list', 'noble_gases'),
    ('interior_energetics/wrapper.py', '<?>_kg_liquid'): ('H2O', 'H2'),
    ('interior_struct/zalmoxis.py', '<?>_kg_liquid'): ('H2O', 'H2'),
    ('interior_struct/zalmoxis.py', '<?>_kg_solid'): ('H2O', 'H2'),
    ('outgas/calliope.py', '<?>_kg_total'): ('element_list',),
    ('outgas/atmodeller.py', '<?>_bar'): ('gas_list',),
    ('outgas/atmodeller.py', '<?>_vmr'): ('gas_list',),
    ('outgas/atmodeller.py', '<?>_kg_atm'): ('gas_list', 'element_list'),
    ('outgas/atmodeller.py', '<?>_kg_liquid'): ('gas_list', 'element_list'),
    ('outgas/atmodeller.py', '<?>_kg_solid'): ('gas_list', 'element_list'),
    ('outgas/atmodeller.py', '<?>_kg_total'): ('gas_list', 'element_list'),
    ('outgas/atmodeller.py', '<?>_mol_atm'): ('gas_list',),
    ('outgas/atmodeller.py', '<?>_mol_liquid'): ('gas_list',),
    ('outgas/atmodeller.py', '<?>_mol_solid'): ('gas_list',),
    ('outgas/atmodeller.py', '<?>_mol_total'): ('gas_list',),
    ('outgas/binodal.py', '<?>_bar'): ('gas_list',),
    ('outgas/dummy.py', '<?>_bar'): ('gas_list',),
    ('outgas/dummy.py', '<?>_vmr'): ('gas_list',),
    ('outgas/dummy.py', '<?>_mol_atm'): ('gas_list',),
    ('outgas/dummy.py', '<?>_mol_liquid'): ('gas_list',),
    ('outgas/dummy.py', '<?>_mol_solid'): ('gas_list',),
    ('outgas/dummy.py', '<?>_mol_total'): ('gas_list',),
    ('outgas/dummy.py', '<?>_kg_atm'): ('gas_list', 'element_list'),
    ('outgas/dummy.py', '<?>_kg_liquid'): ('gas_list', 'element_list'),
    ('outgas/dummy.py', '<?>_kg_solid'): ('gas_list', 'element_list'),
    ('outgas/dummy.py', '<?>_kg_total'): ('gas_list', 'element_list'),
    ('outgas/lavatmos.py', '<?>_kg_atm'): ('gas_list', 'element_list'),
    ('outgas/lavatmos.py', '<?>_bar'): ('vap_list',),
    ('outgas/lavatmos.py', '<?>_vmr'): ('vap_list',),
    ('outgas/wrapper.py', '<?>_kg_total'): ('element_list',),
    ('outgas/wrapper.py', '<?>_kg_atm'): ('gas_list', 'element_list'),
    ('utils/coupler.py', '<?>_kg_atm'): ('vol_gas_list',),
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


class HfRowVisitor(ast.NodeVisitor):
    """Collect helpfile reads/writes in one file, tracking loop domains."""

    def __init__(self, rel_file: str, species: dict[str, list]):
        self.rel_file = rel_file
        self.species = species
        self.func_stack: list[str] = []
        self.loop_domains: dict[str, str] = {}  # loop var -> domain-list name
        self.local_vars: dict[str, ast.AST] = {}
        self.writes: list[tuple[str, str]] = []  # (key, function)
        self.reads: list[str] = []
        self.unresolved: list[tuple[int, str, str]] = []  # (lineno, reason, kind)

    # -- context tracking ---------------------------------------------------

    def _visit_func(self, node):
        old_locals = dict(self.local_vars)
        self.func_stack.append(node.name)
        a = node.args
        for arg in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg):
            if arg:
                self.local_vars[arg.arg] = None
        self.generic_visit(node)
        self.func_stack.pop()
        self.local_vars = old_locals

    visit_FunctionDef = _visit_func
    visit_AsyncFunctionDef = _visit_func

    def _bind(self, name: str, value: ast.AST | None) -> None:
        """Track ``name`` as a constant string only on its first binding in scope."""
        is_str = isinstance(value, ast.Constant) and isinstance(value.value, str)
        self.local_vars[name] = value if is_str and name not in self.local_vars else None

    def _invalidate(self, target: ast.AST | str | None) -> None:
        """Clear any constant binding for target variables."""
        if target is None:
            return
        if isinstance(target, str):
            self.local_vars[target] = None
        elif isinstance(target, ast.Name):
            self.local_vars[target.id] = None
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._invalidate(elt)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._bind(target.id, node.value)
            elif isinstance(target, (ast.Tuple, ast.List)):
                self._invalidate(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name):
            self._bind(node.target.id, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self._invalidate(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(self, node):
        self._invalidate(node.target)
        self.generic_visit(node)

    def visit_withitem(self, node):
        self._invalidate(node.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self._invalidate(node.name)
        self.generic_visit(node)

    def visit_comprehension(self, node):
        self._invalidate(node.target)
        self.generic_visit(node)

    def visit_For(self, node):
        added = []
        target_names = []
        if isinstance(node.target, ast.Name):
            target_names = [node.target.id]
        elif isinstance(node.target, ast.Tuple):
            target_names = [e.id for e in node.target.elts if isinstance(e, ast.Name)]
        for name in target_names:
            self.local_vars[name] = None
        domain = self._domain_of_iter(node.iter)
        if domain and target_names:
            # For tuple targets (e.g. `for e, mass in x.items()`) the first
            # name carries the key.
            self.loop_domains[target_names[0]] = domain
            added.append(target_names[0])
        self.generic_visit(node)
        for name in added:
            self.loop_domains.pop(name, None)

    def _domain_of_iter(self, iter_node) -> str | None:
        if isinstance(iter_node, ast.Name) and iter_node.id in DOMAIN_LISTS:
            return iter_node.id
        if isinstance(iter_node, (ast.Tuple, ast.List)) and all(
            isinstance(e, ast.Constant) and isinstance(e.value, str) for e in iter_node.elts
        ):
            return 'literal:' + ','.join(e.value for e in iter_node.elts)
        if isinstance(iter_node, ast.Call):
            func = iter_node.func
            if isinstance(func, ast.Name) and func.id == 'expected_keys':
                return 'registry:expected_keys'
            if isinstance(func, ast.Attribute) and func.attr in ('keys', 'items'):
                inner = func.value
                if isinstance(inner, ast.Name) and inner.id in DOMAIN_LISTS:
                    return inner.id
        return None

    # -- subscripts ---------------------------------------------------------

    def visit_Subscript(self, node):
        name = _row_name(node)
        if name is None:
            self.generic_visit(node)
            return
        is_write = isinstance(node.ctx, (ast.Store, ast.AugStore)) and name in ROW_NAMES
        is_get = isinstance(node.ctx, ast.Load) and name in ROW_NAMES
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
                    self.unresolved.append(
                        (node.lineno, f'{owner}.update(...) bulk write', 'write')
                    )
        self.generic_visit(node)

    def _suppressed(self) -> bool:
        """Whether any enclosing function is a declared non-producer site."""
        return any((self.rel_file, fn) in SUPPRESSED_DYNAMIC_WRITES for fn in self.func_stack)

    def _record(self, key_node, lineno: int, is_write: bool, is_get: bool = False) -> None:
        func = self.func_stack[-1] if self.func_stack else '<module>'
        keys = self._resolve_keys(key_node, lineno, is_write, is_get)
        for key in keys:
            if is_write:
                self.writes.append((key, func))
            else:
                self.reads.append(key)

    def _resolve_keys(
        self, key_node, lineno: int, is_write: bool, is_get: bool = False
    ) -> list[str]:
        if (
            is_get
            and isinstance(key_node, ast.Name)
            and isinstance(self.local_vars.get(key_node.id), ast.Constant)
        ):
            key_node = self.local_vars[key_node.id]
        if isinstance(key_node, ast.Constant):
            return [key_node.value] if isinstance(key_node.value, str) else []
        template = _template_of(key_node)
        reason = None
        if template is not None:
            prefix, var, suffix = template
            domain = self.loop_domains.get(var)
            if domain is not None:
                return [f'{prefix}{v}{suffix}' for v in self._expand_domain(domain)]
            override = TEMPLATE_OVERRIDES.get((self.rel_file, f'{prefix}<?>{suffix}'))
            if override is not None:
                values = {
                    v
                    for item in override
                    for v in (self.species[item] if item in self.species else [item])
                }
                return [f'{prefix}{v}{suffix}' for v in sorted(values)]
            reason = f'template {prefix}<{var}>{suffix}'
        elif isinstance(key_node, ast.Name) and key_node.id in self.loop_domains:
            return self._expand_domain(self.loop_domains[key_node.id])
        else:
            reason = f'dynamic key {ast.unparse(key_node)}'

        if (is_write or is_get) and not self._suppressed():
            self.unresolved.append((lineno, reason, 'write' if is_write else 'read'))
        return []

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
    [(rel_file, key)], 'unresolved': [(rel_file, lineno, reason, kind)]}``.
    """
    species = _species_lists()
    merge_functions = {(f, fn) for f, fn, _renames in MERGE_SITES}
    writes: list[tuple[str, str, str]] = []
    reads: list[tuple[str, str]] = []
    unresolved: list[tuple[str, int, str, str]] = []

    for path in sorted(SRC.rglob('*.py')):
        rel = str(path.relative_to(SRC))
        visitor = HfRowVisitor(rel, species)
        visitor.visit(ast.parse(path.read_text()))
        for key, func in visitor.writes:
            writes.append((rel, func, key))
        for key in visitor.reads:
            reads.append((rel, key))
        for lineno, reason, kind in visitor.unresolved:
            unresolved.append((rel, lineno, reason, kind))

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
