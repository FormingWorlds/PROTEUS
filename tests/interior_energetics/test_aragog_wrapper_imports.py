"""Guard against PROTEUS-Aragog public-API drift.

The PROTEUS Aragog wrapper at ``proteus.interior_energetics.aragog``
imports a fixed set of symbols from ``aragog.parser`` (the legacy
dataclass-based config layer that is also the canonical hydrated
config object the solver consumes). When Aragog renames or removes
any of those symbols, the wrapper raises ``ImportError`` at module
load and every downstream wrapper-touching test fails with the same
confusing module-load traceback.

This smoke test imports the wrapper module directly. Any drift on
the Aragog side surfaces here in under 100 ms, on the PR that
introduces it, instead of being hidden inside the per-test fixture
chain of a longer integration check.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_aragog_wrapper_module_imports():
    """``AragogRunner`` must load without any ImportError.

    Discriminator: a removed or renamed symbol on the
    ``aragog.parser`` import block (e.g. ``_EnergyParameters``,
    ``_PhaseMixedParameters``) raises here before any test body
    runs. A pure import test catches that without paying for the
    fixture chain of the heavier wrapper tests.
    """
    from proteus.interior_energetics.aragog import AragogRunner

    # The PROTEUS main loop calls into AragogRunner at three sites:
    # the constructor (which dispatches into ``setup_or_update_solver``
    # → ``setup_solver`` for the initial build), ``runner.run_solver``
    # for the per-iteration step, and ``AragogRunner._write_output_ncdf``
    # called as a static method from ``proteus.proteus`` for the
    # NetCDF snapshot. A wrapper-side rename of any of these would
    # let the import succeed but silently break the coupling at the
    # next dispatch; assert all four are present.
    assert callable(AragogRunner.__init__), 'AragogRunner.__init__ missing or non-callable'
    assert callable(AragogRunner.setup_solver), (
        'AragogRunner.setup_solver missing or non-callable'
    )
    assert callable(AragogRunner.run_solver), 'AragogRunner.run_solver missing or non-callable'
    assert callable(AragogRunner._write_output_ncdf), (
        'AragogRunner._write_output_ncdf missing or non-callable'
    )


def test_aragog_parser_symbols_used_by_wrapper_exist():
    """The Aragog parser symbols the wrapper imports must all be present.

    Anti-happy-path: covers the case where the wrapper reorganises
    its own imports but still depends on a now-missing symbol via a
    nested ``from aragog.parser import ...`` (e.g. inside a method
    body or an ``if TYPE_CHECKING:`` block). A simple
    ``importlib.import_module`` of the wrapper would not catch those.
    """
    from aragog import parser as aragog_parser

    required = (
        'Parameters',
        '_BoundaryConditionsParameters',
        '_EnergyParameters',
        '_InitialConditionParameters',
        '_MeshParameters',
        '_PhaseMixedParameters',
        '_PhaseParameters',
        '_Radionuclide',
        '_SolverParameters',
    )
    missing = [name for name in required if not hasattr(aragog_parser, name)]
    assert not missing, (
        f'aragog.parser is missing wrapper-required symbols: {missing}. '
        'Coordinate the rename/removal with '
        'src/proteus/interior_energetics/aragog.py.'
    )
    # Discrimination: each required symbol must be class-like (callable),
    # not just present as a None placeholder. A regression that stubbed
    # the names without bodies would still pass the hasattr check above
    # but fail this stricter constructor-availability gate.
    non_callable = [name for name in required if not callable(getattr(aragog_parser, name))]
    assert not non_callable, (
        f'aragog.parser symbols are present but not callable: {non_callable}'
    )


def test_aragog_solver_public_classes_exist():
    """``aragog.solver`` must still expose the wrapper-required classes.

    Edge case: a rename like ``EntropySolver`` -> ``Solver`` (or
    vice versa) on the Aragog side would silently break the
    coupling without showing up in unit tests of either side in
    isolation.
    """
    from aragog import solver as aragog_solver

    required = ('EntropySolver', 'SolverOutput')
    missing = [name for name in required if not hasattr(aragog_solver, name)]
    assert not missing, f'aragog.solver is missing wrapper-required classes: {missing}'
    # Discrimination: required names must be class-like (callable). A
    # regression that aliased the names to bare instances or None would
    # still pass hasattr above but break the wrapper's `EntropySolver(...)`
    # construction call.
    non_callable = [name for name in required if not callable(getattr(aragog_solver, name))]
    assert not non_callable, (
        f'aragog.solver symbols are present but not callable: {non_callable}'
    )
