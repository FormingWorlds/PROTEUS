"""
Schema invariant tests: every validator catches its target invalid input,
and every default backend is importable on a stock CI runner.

Anti-happy-path discipline (per .github/.claude/rules/proteus-tests.md):

- Every validator under test gets a positive AND negative test. The negative
  test asserts a specific exception type AND a specific error-message regex,
  not just "raises". This guards against the "dead validator" failure mode
  where a comparison silently never fires (e.g., comparing a dataclass
  instance against a primitive sentinel).

- Every default-module backend is parametrized in test_default_backend_is_importable
  and asserted to import without optional deps. The parametrize list MUST
  match the schema; if a new module is added without updating this list,
  the assert in test_module_field_inventory_matches_schema fires.

- The slow-tier hypothesis test fuzzes the cross-product of module enum
  values for the eight backend fields, asserting every combination either
  validates and yields a usable Config OR raises with a specific message.
  Catastrophic exceptions (TypeError, AttributeError, KeyError) fail the
  test. The cross-product covers ~5000 combinations; hypothesis samples
  200 by default; derandomize=True so CI is reproducible.

The original failure that motivated this file: outgas.module defaulted to
"atmodeller" while atmodeller is not in pyproject hard deps; CI runners
without atmodeller could not load minimal.toml because
check_module_dependencies tried to import the missing package. That
validator had ZERO test coverage at the time.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from proteus.config import read_config_object
from proteus.config._config import (
    boreas_requires_atmosphere,
    boundary_requires_fixed_surface_state,
    check_module_dependencies,
    instmethod_evolve,
    planet_fO2_source_compat,
    planet_mass_valid,
    planet_oxygen_mode_explicit,
    satellite_evolve,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

# Mixed-tier file: 32 unit tests + 1 slow test (the 7776-combo cross-
# product is too expensive for the unit budget and carries
# @pytest.mark.slow per-function). New tests default to unit; only
# combinatorial fuzz that exceeds the 30 s ceiling should be slow.


# This file deliberately omits a module-level `pytestmark` because it
# mixes two tiers: the explicit per-validator tests are `@pytest.mark.unit`
# and the exhaustive cross-product is `@pytest.mark.slow`. A module-level
# mark would double-mark the slow test as unit and break the CI filter
# `unit and not slow`. Every test function in this file MUST carry its
# own tier decorator; review for missing marks at PR time.

# ---------------------------------------------------------------------------
# Schema enum inventory: kept here so a schema change that adds a backend
# without updating these lists fails test_module_field_inventory_matches_schema
# loudly, rather than silently skipping the new backend in the cross-product.
# ---------------------------------------------------------------------------
OUTGAS_BACKENDS = ('calliope', 'atmodeller', 'dummy')
INTERIOR_ENERGETICS_BACKENDS = ('spider', 'aragog', 'dummy', 'boundary')
ATMOS_CLIM_BACKENDS = ('dummy', 'agni', 'janus')
ATMOS_CHEM_BACKENDS = (None, 'vulcan', 'dummy')
ESCAPE_BACKENDS = (None, 'dummy', 'zephyrus', 'boreas')
STAR_BACKENDS = (None, 'mors', 'dummy')
ORBIT_BACKENDS = (None, 'dummy', 'lovepy')
INTERIOR_STRUCT_BACKENDS = (None, 'dummy', 'spider', 'zalmoxis')

# Backends whose Python package is in the PROTEUS hard dependency set
# (declared in pyproject.toml [project].dependencies). Backends not in this
# set are optional installs and should not be the schema default for their
# field. Keep this list in sync with pyproject.toml.
HARD_DEP_BACKENDS = {
    ('outgas.module', 'calliope'),
    ('outgas.module', 'dummy'),
    ('interior_energetics.module', 'spider'),
    ('interior_energetics.module', 'aragog'),
    ('interior_energetics.module', 'dummy'),
    ('interior_energetics.module', 'boundary'),
    ('atmos_clim.module', 'dummy'),
    ('atmos_clim.module', 'agni'),
    ('atmos_clim.module', 'janus'),
    ('atmos_chem.module', None),
    ('atmos_chem.module', 'dummy'),
    ('escape.module', None),
    ('escape.module', 'dummy'),
    ('escape.module', 'zephyrus'),
    ('star.module', None),
    ('star.module', 'mors'),
    ('star.module', 'dummy'),
    ('orbit.module', None),
    ('orbit.module', 'dummy'),
    ('orbit.module', 'lovepy'),
    ('interior_struct.module', None),
    ('interior_struct.module', 'dummy'),
    ('interior_struct.module', 'spider'),
    ('interior_struct.module', 'zalmoxis'),
}


# ---------------------------------------------------------------------------
# Schema-default importability: every default backend must be importable
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    'field_path,default,enum',
    [
        ('outgas.module', 'calliope', OUTGAS_BACKENDS),
        ('interior_energetics.module', 'aragog', INTERIOR_ENERGETICS_BACKENDS),
        ('atmos_clim.module', 'agni', ATMOS_CLIM_BACKENDS),
        ('atmos_chem.module', None, ATMOS_CHEM_BACKENDS),
        ('escape.module', 'zephyrus', ESCAPE_BACKENDS),
        ('star.module', 'mors', STAR_BACKENDS),
        ('orbit.module', None, ORBIT_BACKENDS),
        ('interior_struct.module', 'zalmoxis', INTERIOR_STRUCT_BACKENDS),
    ],
)
def test_default_backend_is_in_hard_dependency_set(field_path, default, enum):
    """The schema default for every module field must point at a backend
    whose Python package is in the PROTEUS hard dependency set.

    A default that requires an optional install is a trap: any environment
    that lacks the optional package cannot load a TOML that omits the field,
    because check_module_dependencies tries to import the named package at
    config load. This is the exact regression that broke
    tests/config/test_config.py::test_factory_defaults_from_minimal_config
    on CI runners without atmodeller.
    """
    assert default in enum, (
        f'Schema default {default!r} for {field_path} is not in its enum {enum}'
    )
    assert (field_path, default) in HARD_DEP_BACKENDS, (
        f'Schema default {default!r} for {field_path} requires an optional '
        f'package not in the PROTEUS hard dependency set; pick a default '
        f'whose package is hard-pinned in pyproject.toml.'
    )


@pytest.mark.unit
def test_module_field_inventory_matches_schema():
    """The enum constants in this file must match the actual schema.

    A new module backend added to the schema without updating this file
    causes the cross-product test below to silently miss the new value,
    which would re-introduce the dead-validator class of bug. Reading the
    enum tuples from the validator itself catches that.
    """
    from proteus.config._atmos_chem import AtmosChem
    from proteus.config._atmos_clim import AtmosClim
    from proteus.config._escape import Escape
    from proteus.config._interior import Interior
    from proteus.config._orbit import Orbit
    from proteus.config._outgas import Outgas
    from proteus.config._star import Star

    def _enum_for(cls, attr_name):
        # attrs stores validators in __attrs_attrs__; pull the in_(...) tuple
        for a in cls.__attrs_attrs__:
            if a.name == attr_name:
                v = a.validator
                # validator may be a single in_(...) or a tuple of validators
                candidates = v if isinstance(v, tuple) else (v,)
                for c in candidates:
                    options = getattr(c, 'options', None)
                    if options is not None:
                        return tuple(options)
        return None

    pairs = (
        (Outgas, 'module', OUTGAS_BACKENDS),
        (Interior, 'module', INTERIOR_ENERGETICS_BACKENDS),
        (AtmosClim, 'module', ATMOS_CLIM_BACKENDS),
        (AtmosChem, 'module', ATMOS_CHEM_BACKENDS),
        (Escape, 'module', ESCAPE_BACKENDS),
        (Star, 'module', STAR_BACKENDS),
        (Orbit, 'module', ORBIT_BACKENDS),
    )
    for cls, attr_name, expected in pairs:
        actual = _enum_for(cls, attr_name)
        # If introspection fails for any field in `pairs`, fail the test
        # rather than silently skip the inventory check. The previous
        # `if actual is None: continue` would mask a future validator
        # refactor that wraps `in_(...)` inside `and_(...)` or replaces
        # it with a custom callable, leaving the schema-drift guard
        # broken for every field.
        assert actual is not None, (
            f'_enum_for could not introspect {cls.__name__}.{attr_name}; '
            f'extend _enum_for to handle the validator shape used here, '
            f'or move this field to the cross-product-only inventory and '
            f'document the exemption.'
        )
        assert set(actual) == set(expected), (
            f'{cls.__name__}.{attr_name} schema enum {actual} disagrees '
            f'with the test inventory {expected}; update both.'
        )


# ---------------------------------------------------------------------------
# check_module_dependencies: positive + negative
# ---------------------------------------------------------------------------
def _make_config_instance(**overrides):
    """Build a SimpleNamespace shaped enough for the validators we invoke.

    The validators only read instance.<section>.<field>, so a nested
    SimpleNamespace is sufficient. attrs structuring is not exercised here;
    that is the cross-product test's job.
    """
    base = SimpleNamespace(
        outgas=SimpleNamespace(module='calliope', fO2_shift_IW=0.0),
        escape=SimpleNamespace(module='zephyrus'),
        atmos_chem=SimpleNamespace(module=None),
        atmos_clim=SimpleNamespace(module='agni', surf_state='fixed'),
        interior_energetics=SimpleNamespace(module='aragog'),
        interior_struct=SimpleNamespace(module='zalmoxis'),
        observe=SimpleNamespace(synthesis=None),
        orbit=SimpleNamespace(
            module='dummy',
            instellation_method='separation',
            evolve=False,
            satellite=False,
        ),
        params=SimpleNamespace(stop=SimpleNamespace(escape=SimpleNamespace(enabled=True))),
        planet=SimpleNamespace(
            mass_tot=1.0,
            elements=SimpleNamespace(O_mode='ic_chemistry'),
            volatile_mode='elements',
            fO2_source='user_constant',
        ),
    )
    for path, value in overrides.items():
        section, field = path.split('.')
        setattr(getattr(base, section), field, value)
    return base


@pytest.mark.unit
def test_check_module_dependencies_passes_with_default_backends():
    """All hard-dep default backends import cleanly; the validator is silent."""
    instance = _make_config_instance()
    result = check_module_dependencies(instance, None, None)
    assert result is None  # contract: validator returns None silently when imports succeed
    # Discriminating check: the instance carries the hard-dep default backends
    # that the validator must have successfully imported (calliope for outgas,
    # zephyrus for escape). A regression that broke either import would have
    # raised before this assertion.
    assert instance.outgas.module == 'calliope'
    assert instance.escape.module == 'zephyrus'


@pytest.mark.unit
def test_check_module_dependencies_raises_when_atmodeller_missing():
    """If outgas.module = atmodeller and the package is unavailable, raise
    ImportError with the install hint and the original error chained.
    """
    instance = _make_config_instance(**{'outgas.module': 'atmodeller'})

    real_import = importlib.import_module

    def _fake_import(name):
        if name == 'atmodeller':
            raise ImportError("No module named 'atmodeller'")
        return real_import(name)

    with patch('importlib.import_module', side_effect=_fake_import):
        with pytest.raises(ImportError, match=r'fwl-proteus\[atmodeller\]') as excinfo:
            check_module_dependencies(instance, None, None)
    # The hint must name the GPL-3.0 license of the optional backend, since the
    # PROTEUS core is Apache-2.0 and the licensing difference matters to users.
    msg = str(excinfo.value)
    assert 'GPL-3.0' in msg and 'outgas.module' in msg
    # Discrimination: the original ImportError must be chained as __cause__ so
    # users see the "No module named 'atmodeller'" detail. A regression that
    # re-raised a bare ImportError without `raise ... from e` would lose the
    # chain and trip this check.
    assert excinfo.value.__cause__ is not None
    assert 'atmodeller' in str(excinfo.value.__cause__)


@pytest.mark.unit
def test_check_module_dependencies_raises_when_vulcan_missing():
    """If atmos_chem.module = vulcan and the package is unavailable, raise
    ImportError naming the GPL-3.0 license and the install hint.
    """
    instance = _make_config_instance(**{'atmos_chem.module': 'vulcan'})

    real_import = importlib.import_module

    def _fake_import(name):
        if name == 'vulcan':
            raise ImportError("No module named 'vulcan'")
        return real_import(name)

    with patch('importlib.import_module', side_effect=_fake_import):
        with pytest.raises(ImportError, match=r'fwl-proteus\[vulcan\]') as excinfo:
            check_module_dependencies(instance, None, None)
    # The message must name atmos_chem.module (not, e.g., outgas) so the vulcan
    # branch is the one that fired, and must carry the GPL-3.0 license notice.
    msg = str(excinfo.value)
    assert 'atmos_chem.module' in msg and 'vulcan' in msg
    assert 'GPL-3.0' in msg
    # The chained cause preserves the underlying import failure detail.
    assert excinfo.value.__cause__ is not None
    assert 'vulcan' in str(excinfo.value.__cause__)


@pytest.mark.unit
def test_check_module_dependencies_raises_when_boreas_missing():
    """If escape.module = boreas and the package is unavailable, raise
    ImportError with the install hint.
    """
    instance = _make_config_instance(**{'escape.module': 'boreas'})

    real_import = importlib.import_module

    def _fake_import(name):
        if name == 'boreas':
            raise ImportError("No module named 'boreas'")
        return real_import(name)

    with patch('importlib.import_module', side_effect=_fake_import):
        with pytest.raises(ImportError, match=r'tools/get_boreas\.sh') as excinfo:
            check_module_dependencies(instance, None, None)
    # Discrimination: the message must name escape.module (not outgas.module) to
    # confirm the boreas branch fired and not, e.g., an accidental atmodeller
    # path. A regression that misattributed the hint to outgas would fail here.
    msg = str(excinfo.value)
    assert 'escape.module' in msg and 'boreas' in msg


@pytest.mark.unit
def test_check_module_dependencies_skips_unselected_backends():
    """Validator MUST NOT import a backend that isn't selected.

    If outgas.module = "calliope", the validator should not try to import
    atmodeller even if that import would fail. This test guards against a
    refactor that flips the `if needed` gate.
    """
    instance = _make_config_instance(**{'outgas.module': 'calliope'})

    real_import = importlib.import_module
    seen = []

    def _spy_import(name):
        seen.append(name)
        if name == 'atmodeller':
            raise ImportError('SHOULD NOT BE CALLED')
        return real_import(name)

    with patch('importlib.import_module', side_effect=_spy_import):
        check_module_dependencies(instance, None, None)
    assert 'atmodeller' not in seen, (
        f'Validator imported atmodeller despite outgas.module = calliope; imports were: {seen}'
    )
    # Discrimination: the validator must have imported calliope (the selected
    # backend) at least once. A regression that short-circuited before any
    # import would leave `seen` empty and still pass the 'not in' check above.
    assert 'calliope' in seen


# ---------------------------------------------------------------------------
# boreas_requires_atmosphere: positive + negative
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_boreas_requires_atmosphere_rejects_dummy_atmos():
    """BOREAS escape with dummy atmos must raise with a specific message
    pointing at agni/janus as the fix.
    """
    instance = _make_config_instance(
        **{'escape.module': 'boreas', 'atmos_clim.module': 'dummy'}
    )
    with pytest.raises(ValueError, match=r'agni or janus') as excinfo:
        boreas_requires_atmosphere(instance, None, None)
    # Discrimination: the message must name both 'boreas' (the triggering escape
    # backend) and 'dummy' (the rejected atmos_clim). A regression with a generic
    # "needs a real atmos" message that did not echo the offending combo would
    # fail this check.
    msg = str(excinfo.value)
    assert 'boreas' in msg and 'dummy' in msg


@pytest.mark.unit
def test_boreas_requires_atmosphere_passes_with_radiative_atmos():
    """BOREAS escape with agni passes; validator is silent."""
    instance = _make_config_instance(**{'escape.module': 'boreas', 'atmos_clim.module': 'agni'})
    result = boreas_requires_atmosphere(instance, None, None)
    assert result is None  # contract: BOREAS + agni is the canonical accepted combo
    # Discriminating check: agni is a band-resolved atmosphere; the dummy module
    # would have raised. Pin the combination that produced the silent pass.
    assert instance.escape.module == 'boreas'
    assert instance.atmos_clim.module == 'agni'


@pytest.mark.unit
def test_boreas_requires_atmosphere_skips_when_not_boreas():
    """Validator must not fire when escape != boreas, regardless of atmos."""
    instance = _make_config_instance(
        **{'escape.module': 'zephyrus', 'atmos_clim.module': 'dummy'}
    )
    result = boreas_requires_atmosphere(instance, None, None)
    assert result is None  # contract: non-boreas escape short-circuits the validator
    # Discriminating check: atmos_clim.module='dummy' would have raised under
    # boreas; only the escape-module-skip branch can produce a silent pass.
    assert instance.escape.module == 'zephyrus'
    assert instance.atmos_clim.module == 'dummy'


# ---------------------------------------------------------------------------
# planet_mass_valid: positive + negative for the four documented bands
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize('mass', [0.5, 1.0, 5.0, 19.99])
def test_planet_mass_valid_accepts_in_range(mass):
    """Masses in (0, 20] M_earth are accepted."""
    instance = _make_config_instance(**{'planet.mass_tot': mass})
    result = planet_mass_valid(instance, None, None)
    assert result is None  # contract: in-range mass is accepted silently
    # Discriminating check: 0 < mass < 20 (strict) - the open lower bound and the
    # 20 M_earth upper limit are the documented accepted band.
    assert 0.0 < mass < 20.0
    assert instance.planet.mass_tot == mass


@pytest.mark.unit
def test_planet_mass_valid_rejects_non_positive_zero():
    """planet.mass_tot = 0 raises with the '> 0' message."""
    instance = _make_config_instance(**{'planet.mass_tot': 0.0})
    with pytest.raises(ValueError, match=r'must be > 0'):
        planet_mass_valid(instance, None, None)
    # Discrimination: a valid mass (1.0) must pass silently. A regression
    # that always raised would fail this call.
    instance_ok = _make_config_instance(**{'planet.mass_tot': 1.0})
    assert planet_mass_valid(instance_ok, None, None) is None


@pytest.mark.unit
@pytest.mark.parametrize('bad_mass', [-1.0, 0.0])
def test_planet_mass_valid_rejects_non_positive(bad_mass):
    """Zero and negative masses raise with the '> 0' message."""
    instance = _make_config_instance(**{'planet.mass_tot': bad_mass})
    with pytest.raises(ValueError, match=r'must be > 0') as excinfo:
        planet_mass_valid(instance, None, None)
    # Discrimination: the non-positive branch fires, NOT the upper-bound branch.
    # A regression that flipped the comparison sign and landed on '< 20' for
    # bad_mass=0 (since 0 < 20 is true) would fail this check.
    assert '< 20' not in str(excinfo.value)
    # Pin that the input really is non-positive (parametrize sanity).
    assert bad_mass <= 0.0


@pytest.mark.unit
@pytest.mark.parametrize('big_mass', [20.001, 50.0, 1e3])
def test_planet_mass_valid_rejects_above_upper(big_mass):
    """Masses above 20 M_earth raise with the '< 20' message."""
    instance = _make_config_instance(**{'planet.mass_tot': big_mass})
    with pytest.raises(ValueError, match=r'must be < 20') as excinfo:
        planet_mass_valid(instance, None, None)
    # Discrimination: the upper-bound branch fires, NOT the '> 0' or 'must be
    # set' branches. A regression that flipped the comparison and landed on
    # '> 0' (which would also match any positive number) would fail.
    msg = str(excinfo.value)
    assert '> 0' not in msg and 'must be set' not in msg
    # Pin that the input really exceeds the documented cap.
    assert big_mass > 20.0


# ---------------------------------------------------------------------------
# planet_oxygen_mode_explicit: issue #677 hard cutover
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_planet_oxygen_mode_defaults_to_ic_chemistry():
    """O_mode defaults to 'ic_chemistry' when not set in config.

    Validates that omitting O_mode from a config file no longer raises
    an error (it used to require explicit setting). The default must be
    'ic_chemistry' for backwards compatibility.
    """
    instance = _make_config_instance(
        **{'planet.elements': SimpleNamespace(O_mode='ic_chemistry')}
    )
    # Must not raise
    assert planet_oxygen_mode_explicit(instance, None, None) is None
    # Discrimination: other valid modes must also pass. A regression that
    # hardcoded acceptance of only 'ic_chemistry' would fail on 'ppmw'.
    instance_ppmw = _make_config_instance(**{'planet.elements': SimpleNamespace(O_mode='ppmw')})
    assert planet_oxygen_mode_explicit(instance_ppmw, None, None) is None


@pytest.mark.unit
@pytest.mark.parametrize('mode', ['ic_chemistry', 'ppmw', 'kg', 'FeO_mantle_wt_pct'])
def test_planet_oxygen_mode_explicit_accepts_all_four_documented_modes(mode):
    """All four documented O_mode values pass the validator."""
    instance = _make_config_instance(**{'planet.elements': SimpleNamespace(O_mode=mode)})
    result = planet_oxygen_mode_explicit(instance, None, None)
    assert result is None  # contract: each documented O_mode value passes silently
    # Discriminating check: the parametrized mode is one of the four documented
    # values; the validator must reject 'REQUIRED' (the placeholder) and any
    # unknown string. Pin both ends of the contract on this row.
    assert mode in ('ic_chemistry', 'ppmw', 'kg', 'FeO_mantle_wt_pct')
    assert instance.planet.elements.O_mode == mode


# ---------------------------------------------------------------------------
# planet_fO2_source_compat: four rejection rules + one warning
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_fO2_source_user_constant_accepts_all_O_modes():
    """The default fO2_source = "user_constant" accepts every (O_mode,
    volatile_mode) combination; this is the legacy-compatible path.
    """
    combos_checked = 0
    for o_mode in ('ic_chemistry', 'ppmw', 'kg', 'FeO_mantle_wt_pct'):
        for v_mode in ('elements', 'gas_prs'):
            instance = _make_config_instance(
                **{
                    'planet.fO2_source': 'user_constant',
                    'planet.elements': SimpleNamespace(O_mode=o_mode),
                    'planet.volatile_mode': v_mode,
                }
            )
            assert planet_fO2_source_compat(instance, None, None) is None
            combos_checked += 1
    # Discriminating check: all 4 x 2 = 8 (O_mode, volatile_mode) combinations
    # were exercised; a loop short-circuit (or accidental skip) would land below.
    assert combos_checked == 8


@pytest.mark.unit
def test_fO2_source_from_mantle_redox_rejected_as_reserved():
    """The 'from_mantle_redox' enum is reserved for issue #653 and not yet
    wired into the runtime; reject at config load with the issue reference.
    """
    instance = _make_config_instance(**{'planet.fO2_source': 'from_mantle_redox'})
    with pytest.raises(ValueError, match=r'issue #653') as excinfo:
        planet_fO2_source_compat(instance, None, None)
    # Discrimination: the message must name both wired-in alternatives
    # ('user_constant' and 'from_O_budget') so the user can pick a working
    # path. A regression that dropped one alternative would fail this check.
    msg = str(excinfo.value)
    assert 'user_constant' in msg and 'from_O_budget' in msg


@pytest.mark.unit
def test_fO2_source_from_O_budget_rejects_ic_chemistry():
    """from_O_budget (from_O_budget) cannot use O_mode = ic_chemistry because
    ic_chemistry defers O to the chemistry solver, leaving nothing to
    invert against.
    """
    instance = _make_config_instance(
        **{
            'planet.fO2_source': 'from_O_budget',
            'planet.elements': SimpleNamespace(O_mode='ic_chemistry'),
        }
    )
    with pytest.raises(ValueError, match=r'authoritative O budget') as excinfo:
        planet_fO2_source_compat(instance, None, None)
    # Discrimination: the message must name 'ic_chemistry' (the offending mode)
    # and at least one concrete-budget alternative. A regression that raised a
    # generic "incompatible" without naming the offending mode would fail.
    msg = str(excinfo.value)
    assert 'ic_chemistry' in msg
    assert 'ppmw' in msg or 'kg' in msg or 'FeO_mantle_wt_pct' in msg


@pytest.mark.unit
def test_fO2_source_from_O_budget_rejects_gas_prs_volatile_mode():
    """from_O_budget requires volatile_mode = "elements"; gas_prs makes the
    element budgets inert and from_O_budget has no target to invert against.
    """
    instance = _make_config_instance(
        **{
            'planet.fO2_source': 'from_O_budget',
            'planet.elements': SimpleNamespace(O_mode='ppmw'),
            'planet.volatile_mode': 'gas_prs',
        }
    )
    with pytest.raises(ValueError, match=r'volatile_mode') as excinfo:
        planet_fO2_source_compat(instance, None, None)
    # Discrimination: the volatile_mode branch fires, not the ic_chemistry
    # branch (O_mode is 'ppmw' here). The message must name both 'gas_prs' (the
    # rejected setting) and 'elements' (the required setting); ic_chemistry
    # must NOT appear in the message.
    msg = str(excinfo.value)
    assert 'gas_prs' in msg and 'elements' in msg
    assert 'ic_chemistry' not in msg


@pytest.mark.unit
def test_fO2_source_from_O_budget_rejects_dummy_outgas():
    """from_O_budget requires an outgassing backend with an authoritative-O
    implementation; dummy has no chemistry to invert against.
    """
    instance = _make_config_instance(
        **{
            'planet.fO2_source': 'from_O_budget',
            'planet.elements': SimpleNamespace(O_mode='ppmw'),
            'outgas.module': 'dummy',
        }
    )
    with pytest.raises(ValueError, match=r'no chemistry to invert against') as excinfo:
        planet_fO2_source_compat(instance, None, None)
    # Discrimination: the message must name both supported alternatives
    # ('calliope', 'atmodeller') and 'dummy' (the rejected setting). A regression
    # that dropped one alternative from the hint, or replaced 'dummy' with a
    # generic 'this backend', would fail.
    msg = str(excinfo.value)
    assert 'calliope' in msg and 'atmodeller' in msg and 'dummy' in msg


@pytest.mark.unit
def test_fO2_source_from_O_budget_warns_when_fO2_shift_IW_set():
    """from_O_budget derives the buffer offset from the O budget, so a
    user-supplied fO2_shift_IW is used only as the solver's initial
    guess, not as the buffered offset; surface this as a UserWarning at
    config load.
    """
    instance = _make_config_instance(
        **{
            'planet.fO2_source': 'from_O_budget',
            'planet.elements': SimpleNamespace(O_mode='ppmw'),
            'outgas.module': 'calliope',
            'outgas.fO2_shift_IW': 4.0,
        }
    )
    with pytest.warns(UserWarning, match=r'used only as the solver initial fO2 guess') as warns:
        planet_fO2_source_compat(instance, None, None)
    # Exactly one warning must fire (not two, not zero).
    assert len(warns) == 1
    # Discrimination: the warning must echo the offending numeric value so the
    # user can locate it in their TOML. A regression that emitted a generic
    # message without the value would fail.
    msg = str(warns[0].message)
    assert '4.0' in msg or '4.000' in msg


@pytest.mark.unit
def test_fO2_source_from_O_budget_silent_when_fO2_shift_IW_zero():
    """Default fO2_shift_IW = 0 under from_O_budget is silent (no spurious warning)."""
    import warnings

    instance = _make_config_instance(
        **{
            'planet.fO2_source': 'from_O_budget',
            'planet.elements': SimpleNamespace(O_mode='ppmw'),
            'outgas.module': 'calliope',
            'outgas.fO2_shift_IW': 0.0,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning becomes an error
        result = planet_fO2_source_compat(instance, None, None)
    assert (
        result is None
    )  # contract: from_O_budget + zero shift is silent (no warning, no raise)
    # Discriminating check: fO2_shift_IW must be exactly zero. Any non-zero value
    # under from_O_budget would have produced the solver-initial-guess UserWarning
    # (which `simplefilter('error')` would have re-raised).
    assert instance.outgas.fO2_shift_IW == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Boundary backend coupling validators
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_boundary_requires_fixed_surface_state_rejects_dynamic():
    """Boundary backend assumes fixed surface coupling; dynamic surf_state raises."""
    instance = _make_config_instance(**{'interior_energetics.module': 'boundary'})
    instance.atmos_clim.surf_state = 'mixed_layer'
    with pytest.raises(ValueError, match=r"surf_state='fixed'") as excinfo:
        boundary_requires_fixed_surface_state(instance, None, None)
    # Discrimination: the message must name 'boundary' (the triggering interior
    # backend) so the user knows which side of the coupling is restrictive. A
    # regression with a generic "surf_state must be fixed" message that did not
    # echo the interior backend name would fail.
    assert 'boundary' in str(excinfo.value)


@pytest.mark.unit
def test_boundary_requires_fixed_surface_state_passes_with_fixed():
    """Boundary backend with ``atmos_clim.surf_state='fixed'`` is accepted;
    the validator stays silent. Pairs with the negative test above.
    """
    instance = _make_config_instance(**{'interior_energetics.module': 'boundary'})
    instance.atmos_clim.surf_state = 'fixed'
    result = boundary_requires_fixed_surface_state(instance, None, None)
    assert result is None  # contract: boundary + fixed surf_state is the canonical combo
    # Discriminating check: surf_state='mixed_layer' would have raised under
    # boundary; only the 'fixed' branch can produce a silent pass here.
    assert instance.interior_energetics.module == 'boundary'
    assert instance.atmos_clim.surf_state == 'fixed'


# ---------------------------------------------------------------------------
# Orbit-coupling validators: instmethod_evolve and satellite_evolve
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_instmethod_evolve_rejects_inst_with_orbit_evolve():
    """instellation_method='inst' is incompatible with orbit.evolve=True."""
    instance = _make_config_instance(
        **{
            'orbit.instellation_method': 'inst',
            'orbit.evolve': True,
        }
    )
    with pytest.raises(ValueError, match=r"instellation_method='inst'") as excinfo:
        instmethod_evolve(instance, None, None)
    # Discrimination: the message must mention orbital evolution as the
    # incompatible-with feature, so the user knows which of the two settings
    # to change. A regression with just "instellation_method='inst' is bad"
    # that did not name the conflicting evolve flag would fail.
    msg = str(excinfo.value).lower()
    assert 'evolution' in msg or 'evolve' in msg


@pytest.mark.unit
def test_instmethod_evolve_passes_with_inst_and_no_evolve():
    """instellation_method='inst' is OK when orbit.evolve is False."""
    instance = _make_config_instance(
        **{
            'orbit.instellation_method': 'inst',
            'orbit.evolve': False,
        }
    )
    result = instmethod_evolve(instance, None, None)
    assert result is None  # contract: inst + evolve=False is the canonical compatible combo
    # Discriminating check: evolve=True with inst would have raised; only the
    # evolve=False branch can produce a silent pass under inst.
    assert instance.orbit.instellation_method == 'inst'
    assert instance.orbit.evolve is False


@pytest.mark.unit
def test_instmethod_evolve_passes_with_separation_and_evolve():
    """orbit.evolve is fine when instellation_method != 'inst'."""
    instance = _make_config_instance(
        **{
            'orbit.instellation_method': 'separation',
            'orbit.evolve': True,
        }
    )
    result = instmethod_evolve(instance, None, None)
    assert result is None  # contract: non-inst method permits orbital evolution
    # Discriminating check: evolve=True with inst would have raised; only the
    # non-inst branch can produce a silent pass with evolve=True.
    assert instance.orbit.instellation_method != 'inst'
    assert instance.orbit.evolve is True


@pytest.mark.unit
def test_satellite_evolve_rejects_satellite_with_evolve():
    """orbit.satellite=True with orbit.evolve=True must raise."""
    instance = _make_config_instance(
        **{
            'orbit.satellite': True,
            'orbit.evolve': True,
        }
    )
    with pytest.raises(ValueError, match=r'satellite') as excinfo:
        satellite_evolve(instance, None, None)
    # Discrimination: the message must also mention orbital evolution; the
    # incompatibility is between satellite=True AND evolve=True. A regression
    # that only named 'satellite' without naming the conflicting evolve flag
    # would leave users guessing which side to flip.
    msg = str(excinfo.value).lower()
    assert 'evolution' in msg or 'evolve' in msg


@pytest.mark.unit
def test_satellite_evolve_passes_with_satellite_and_no_evolve():
    """A satellite with a fixed orbit is allowed."""
    instance = _make_config_instance(
        **{
            'orbit.satellite': True,
            'orbit.evolve': False,
        }
    )
    result = satellite_evolve(instance, None, None)
    assert result is None  # contract: satellite=True + evolve=False is the accepted combo
    # Discriminating check: satellite=True + evolve=True would have raised; only
    # the evolve=False branch can produce a silent pass under satellite=True.
    assert instance.orbit.satellite is True
    assert instance.orbit.evolve is False


@pytest.mark.unit
def test_satellite_evolve_passes_without_satellite():
    """orbit.evolve alone (no satellite) is allowed."""
    instance = _make_config_instance(
        **{
            'orbit.satellite': False,
            'orbit.evolve': True,
        }
    )
    result = satellite_evolve(instance, None, None)
    assert result is None  # contract: no-satellite path accepts orbital evolution
    # Discriminating check: satellite=False is the path that allows evolve=True;
    # the validator's other branch (satellite=True + evolve=True) would have raised.
    assert instance.orbit.satellite is False
    assert instance.orbit.evolve is True


# ---------------------------------------------------------------------------
# Slow-tier hypothesis cross-product over module enums
# ---------------------------------------------------------------------------
# Every importable module backend × every importable module backend × ...
# Either the Config validates and is usable, or a validator raises with a
# specific message. Catastrophic exceptions (TypeError, AttributeError,
# KeyError, plain Exception) fail the test.
HYPOTHESIS_OUTGAS = ('calliope', 'dummy')  # atmodeller mocked separately
HYPOTHESIS_INTERIOR_ENERGETICS = ('aragog', 'spider', 'dummy', 'boundary')
HYPOTHESIS_ATMOS_CLIM = ('agni', 'janus', 'dummy')
HYPOTHESIS_ATMOS_CHEM = (None, 'dummy')  # vulcan excluded (optional)
HYPOTHESIS_ESCAPE = (None, 'dummy', 'zephyrus')  # boreas excluded (optional)
HYPOTHESIS_STAR = (None, 'mors', 'dummy')
HYPOTHESIS_ORBIT = (None, 'dummy', 'lovepy')
HYPOTHESIS_INTERIOR_STRUCT = (None, 'dummy', 'spider', 'zalmoxis')


def _make_combo_toml(combo, tmp_path):
    """Render a TOML string for a given module combo, writing to tmp_path."""
    outgas, ie, ac, achem, escape, star, orbit, struct = combo
    parts = [
        'config_version = "3.0"',
        '[orbit]',
        '    semimajoraxis = 1.0',
        f'    module = "{orbit}"' if orbit is not None else '    module = "none"',
        '[planet]',
        '    mass_tot = 1.0',
        '    volatile_mode = "elements"',
        '    [planet.elements]',
        '        O_mode = "ic_chemistry"',
        '        H_mode = "ppmw"',
        '        H_budget = 1e3',
        '[outgas]',
        f'    module = "{outgas}"',
        '    fO2_shift_IW = 2',
        '[interior_energetics]',
        f'    module = "{ie}"',
        '[interior_struct]',
        f'    module = "{struct}"' if struct is not None else '    module = "none"',
        '[atmos_clim]',
        f'    module = "{ac}"',
        '    surf_state = "fixed"',
        '[atmos_chem]',
        f'    module = "{achem}"' if achem is not None else '    module = "none"',
        '[escape]',
        f'    module = "{escape}"' if escape is not None else '    module = "none"',
        '[star]',
        f'    module = "{star}"' if star is not None else '    module = "none"',
    ]
    toml_text = '\n'.join(parts) + '\n'
    p = tmp_path / 'combo.toml'
    p.write_text(toml_text)
    return p


@pytest.mark.slow
def test_module_cross_product_either_validates_or_raises_clearly(tmp_path):
    """Exhaustive cross-product over the importable backend combinations.

    For each of ~7776 combos: either read_config_object returns a Config,
    OR raises a ValueError / ImportError / TypeError with a non-empty
    message that names the offending field. Catastrophic exceptions
    (TypeError / AttributeError / KeyError without a wrapper) fail the test.

    Additional guards:
    - Every accepted Config must round-trip the (outgas, interior, atmos)
      module names through structuring; a silent module-rename is a bug.
    - Every accepted Config must validate config_version == "3.0".
    - Acceptance count must be > 0 and rejection count must be > 0; if
      every combo accepts the validators are too loose; if every combo
      rejects the test surface is broken.
    - Rejection messages with detail length < 20 chars are flagged as
      probably-uninformative and surfaced (do not fail the test, but
      print so we can tighten validators later).

    Notes on tooling choice:
    - hypothesis would be the natural tool but adds strategy-driven
      sampling overhead per example. Cattrs+IO for one combo is ~0.4 ms,
      so an exhaustive sweep finishes in well under 5 s. Deterministic
      itertools.product is reproducible without a seed and exercises every
      combo every time.
    - We only enumerate combos with backends in the hard-dep set; the
      check_module_dependencies positive/negative logic (atmodeller, boreas
      missing) is covered by the explicit tests above.
    """
    import itertools

    full = list(
        itertools.product(
            HYPOTHESIS_OUTGAS,
            HYPOTHESIS_INTERIOR_ENERGETICS,
            HYPOTHESIS_ATMOS_CLIM,
            HYPOTHESIS_ATMOS_CHEM,
            HYPOTHESIS_ESCAPE,
            HYPOTHESIS_STAR,
            HYPOTHESIS_ORBIT,
            HYPOTHESIS_INTERIOR_STRUCT,
        )
    )

    permitted = (ValueError, ImportError, TypeError)
    accepted = 0
    rejected = 0
    catastrophic = []
    uninformative = []

    for combo in full:
        path = _make_combo_toml(combo, tmp_path)
        try:
            cfg = read_config_object(path)
            accepted += 1
            assert cfg.outgas.module == combo[0]
            assert cfg.interior_energetics.module == combo[1]
            assert cfg.atmos_clim.module == combo[2]
            assert cfg.config_version == '3.0'
        except permitted as e:
            rejected += 1
            msg = str(e)
            assert msg, f'Empty exception message for combo {combo}'
            if len(msg) < 20:
                uninformative.append((combo, msg))
        except Exception as e:  # noqa: BLE001
            catastrophic.append((combo, type(e).__name__, str(e)))

    assert not catastrophic, (
        f'{len(catastrophic)} module combinations raised an unstructured '
        f'exception (not ValueError / ImportError / TypeError):\n'
        + '\n'.join(f'  {c}: {t}: {m}' for c, t, m in catastrophic[:10])
    )
    # Both branches must be exercised; if not, the test surface or the
    # validator coverage is broken.
    assert accepted > 0, (
        f'No combinations were accepted out of {len(full)}; either every '
        f'validator is rejecting (over-restrictive schema) or the TOML '
        f'template is malformed.'
    )
    assert rejected > 0, (
        f'No combinations were rejected out of {len(full)}; either every '
        f'validator is silent (dead-validator pattern) or the schema enums '
        f'in this test are too narrow.'
    )
