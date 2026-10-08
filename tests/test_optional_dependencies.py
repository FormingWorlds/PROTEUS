"""Guards on the optional `inference` extra.

The Bayesian-optimisation stack (torch, botorch, gpytorch) is installed only
by ``pip install "fwl-proteus[inference]"``. Three properties keep an install
without it working, and each is checked here:

* no module outside ``src/proteus/inference/`` imports one of the three at
  module scope, so ``import proteus`` and every command other than
  ``proteus infer`` load without them,
* the distribution list the CLI reports on matches the extra declared in
  ``pyproject.toml``,
* the test-quality linter requires an ``importorskip`` guard for all three,
  so a new test that imports them cannot silently break collection on an
  install without the extra.

The checks read source, so they run whether or not the extra is installed.

References:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

from proteus import cli

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / 'src' / 'proteus'
INFERENCE_ROOT = SRC_ROOT / 'inference'


def _module_level_imports(path: Path) -> set[str]:
    """Return the root names imported at module scope in ``path``.

    Imports nested inside functions, classes or ``try`` blocks are excluded:
    they run on call, not at import, so they cannot break ``import proteus``.
    """
    roots: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split('.', 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split('.', 1)[0])
    return roots


def _read_inference_extra() -> set[str]:
    """Return the distribution names declared in the `inference` extra."""
    cfg = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())
    extras = cfg['project']['optional-dependencies']
    requirements = extras['inference']
    return {
        req.split('>')[0].split('=')[0].split('[')[0].split(';')[0].strip()
        for req in requirements
    }


@pytest.mark.unit
def test_no_core_module_imports_the_inference_stack():
    """Only ``src/proteus/inference/`` may import the optimisation stack at
    module scope. A core module that imports torch breaks ``import proteus``
    for every user who installed without the extra, and CI would not notice
    because CI installs it.
    """
    stack = set(cli.INFERENCE_DISTRIBUTIONS)
    offenders = {}
    scanned = 0
    for path in sorted(SRC_ROOT.rglob('*.py')):
        if INFERENCE_ROOT in path.parents:
            continue
        scanned += 1
        hits = _module_level_imports(path) & stack
        if hits:
            offenders[str(path.relative_to(REPO_ROOT))] = sorted(hits)

    assert offenders == {}, f'core modules importing the inference stack: {offenders}'
    # Discrimination: an empty scan would satisfy the assertion above without
    # checking anything. The package is far larger than this floor.
    assert scanned > 50
    # The stack also arrives transitively: a core module importing anything
    # from proteus.inference at module scope pulls torch with it, so that is
    # the same defect wearing a different name.
    reexporters = {
        str(path.relative_to(REPO_ROOT))
        for path in sorted(SRC_ROOT.rglob('*.py'))
        if INFERENCE_ROOT not in path.parents
        and any(
            (
                isinstance(node, ast.ImportFrom)
                and (node.module or '').startswith('proteus.inference')
            )
            or (
                isinstance(node, ast.Import)
                and any(a.name.startswith('proteus.inference') for a in node.names)
            )
            for node in ast.parse(path.read_text()).body
        )
    }
    assert reexporters == set(), f'core modules importing proteus.inference: {reexporters}'
    # The scan reaches modules that do import the stack, so a path filter that
    # silently excluded everything would fail here.
    inference_hits = {
        path.name: sorted(_module_level_imports(path) & stack)
        for path in sorted(INFERENCE_ROOT.glob('*.py'))
    }
    assert 'torch' in inference_hits['BO.py']
    assert 'botorch' in inference_hits['BO.py']


@pytest.mark.unit
def test_cli_distribution_list_matches_the_declared_extra():
    """The names the CLI can report as missing are exactly the ones the extra
    installs. Drift in either direction is a silent regression: a package
    added to the extra but not here gets a raw traceback instead of the
    install hint, and a name dropped from the extra leaves dead advice.
    """
    declared = _read_inference_extra()

    assert declared == set(cli.INFERENCE_DISTRIBUTIONS)
    # Pin the membership as well as the equality, so a rename on both sides at
    # once still has to be deliberate.
    assert declared == {'torch', 'botorch', 'gpytorch'}
    # The extra must not leak into the core dependency list.
    cfg = tomllib.loads((REPO_ROOT / 'pyproject.toml').read_text())
    core = {
        req.split('>')[0].split('=')[0].split('[')[0].split(';')[0].strip()
        for req in cfg['project']['dependencies']
    }
    assert core & declared == set()


@pytest.mark.unit
def test_linter_requires_importorskip_for_the_inference_stack():
    """``tools/check_test_quality.py`` treats all three distributions as
    optional. Without that, a new test importing them at module scope passes
    the linter and then fails collection on an install without the extra,
    which is the recurring trap the rule exists to prevent.
    """
    spec = importlib.util.spec_from_file_location(
        'check_test_quality', REPO_ROOT / 'tools' / 'check_test_quality.py'
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules['check_test_quality'] = module
    try:
        spec.loader.exec_module(module)

        assert set(cli.INFERENCE_DISTRIBUTIONS) <= module.OPTIONAL_DEPS
        # The rule is enforced, not just declared: a module-top import with no
        # guard is reported, and the same import behind a guard is not.
        unguarded = ast.parse('import torch\n')
        guarded = ast.parse("import pytest\npytest.importorskip('torch')\nimport torch\n")
        assert 'torch' in module._missing_importorskip(unguarded)
        assert 'torch' not in module._missing_importorskip(guarded)
    finally:
        del sys.modules['check_test_quality']


@pytest.mark.unit
def test_missing_module_error_names_the_root_distribution():
    """The CLI reads ``ModuleNotFoundError.name`` and takes its first dotted
    component as the distribution. This pins that contract against the real
    import system rather than assuming it: a failed top-level import names the
    package itself, and a failed submodule import names the dotted path whose
    root is that package.
    """
    with pytest.raises(ModuleNotFoundError) as absent:
        importlib.import_module('proteus_inference_stack_absent_xyz')
    assert absent.value.name == 'proteus_inference_stack_absent_xyz'

    with pytest.raises(ModuleNotFoundError) as submodule:
        importlib.import_module('proteus.no_such_submodule_xyz')
    assert submodule.value.name == 'proteus.no_such_submodule_xyz'
    assert submodule.value.name.split('.')[0] == 'proteus'
