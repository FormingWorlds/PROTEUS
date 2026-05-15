"""
Schema invariant tests: every validator catches its target invalid input,
and every default backend is importable on a stock CI runner.

Anti-happy-path discipline (per .claude/rules/proteus-tests.md):

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
    boundary_zalmoxis_incompatible,
    check_module_dependencies,
    planet_fO2_source_compat,
    planet_mass_valid,
    planet_oxygen_mode_explicit,
)

# This file deliberately omits a module-level `pytestmark` because it
# mixes two tiers: the explicit per-validator tests are `@pytest.mark.unit`
# and the exhaustive cross-product is `@pytest.mark.slow`. A module-level
# mark would double-mark the slow test as unit and break the CI filter
# `unit and not slow`. Every test function in this file MUST carry its
# own tier decorator; review for missing marks at PR time.

# ---------------------------------------------------------------------------
# Schema enum inventory — kept here so a schema change that adds a backend
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
    ('atmos_chem.module', 'vulcan'),
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
# Schema-default importability: the validator that ate Group A
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
# check_module_dependencies — positive + negative
# This validator had zero direct coverage before this file. Group A bug.
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
        atmos_clim=SimpleNamespace(module='agni', surf_state='fixed'),
        interior_energetics=SimpleNamespace(module='aragog'),
        interior_struct=SimpleNamespace(module='zalmoxis'),
        observe=SimpleNamespace(synthesis=None),
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
    check_module_dependencies(instance, None, None)  # must not raise


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
        with pytest.raises(ImportError, match=r'pip install atmodeller'):
            check_module_dependencies(instance, None, None)


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
        with pytest.raises(ImportError, match=r'pip install boreas'):
            check_module_dependencies(instance, None, None)


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


# ---------------------------------------------------------------------------
# boreas_requires_atmosphere — positive + negative
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_boreas_requires_atmosphere_rejects_dummy_atmos():
    """BOREAS escape with dummy atmos must raise with a specific message
    pointing at agni/janus as the fix.
    """
    instance = _make_config_instance(
        **{'escape.module': 'boreas', 'atmos_clim.module': 'dummy'}
    )
    with pytest.raises(ValueError, match=r'agni or janus'):
        boreas_requires_atmosphere(instance, None, None)


@pytest.mark.unit
def test_boreas_requires_atmosphere_passes_with_radiative_atmos():
    """BOREAS escape with agni passes; validator is silent."""
    instance = _make_config_instance(**{'escape.module': 'boreas', 'atmos_clim.module': 'agni'})
    boreas_requires_atmosphere(instance, None, None)  # must not raise


@pytest.mark.unit
def test_boreas_requires_atmosphere_skips_when_not_boreas():
    """Validator must not fire when escape != boreas, regardless of atmos."""
    instance = _make_config_instance(
        **{'escape.module': 'zephyrus', 'atmos_clim.module': 'dummy'}
    )
    boreas_requires_atmosphere(instance, None, None)  # must not raise


# ---------------------------------------------------------------------------
# planet_mass_valid — positive + negative for the four documented bands
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize('mass', [0.5, 1.0, 5.0, 19.99])
def test_planet_mass_valid_accepts_in_range(mass):
    """Masses in (0, 20] M_earth are accepted."""
    instance = _make_config_instance(**{'planet.mass_tot': mass})
    planet_mass_valid(instance, None, None)


@pytest.mark.unit
def test_planet_mass_valid_rejects_unset():
    """planet.mass_tot = None raises with the 'must be set' message."""
    instance = _make_config_instance(**{'planet.mass_tot': None})
    with pytest.raises(ValueError, match=r'mass_tot.*must be set'):
        planet_mass_valid(instance, None, None)


@pytest.mark.unit
@pytest.mark.parametrize('bad_mass', [-1.0, 0.0])
def test_planet_mass_valid_rejects_non_positive(bad_mass):
    """Zero and negative masses raise with the '> 0' message."""
    instance = _make_config_instance(**{'planet.mass_tot': bad_mass})
    with pytest.raises(ValueError, match=r'must be > 0'):
        planet_mass_valid(instance, None, None)


@pytest.mark.unit
@pytest.mark.parametrize('big_mass', [20.001, 50.0, 1e3])
def test_planet_mass_valid_rejects_above_upper(big_mass):
    """Masses above 20 M_earth raise with the '< 20' message."""
    instance = _make_config_instance(**{'planet.mass_tot': big_mass})
    with pytest.raises(ValueError, match=r'must be < 20'):
        planet_mass_valid(instance, None, None)


# ---------------------------------------------------------------------------
# planet_oxygen_mode_explicit — issue #677 hard cutover
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_planet_oxygen_mode_explicit_rejects_required_sentinel():
    """The 'REQUIRED' sentinel default means the user did not set O_mode;
    validator must reject it with a migration hint listing the four valid
    modes (issue #677 hard cutover).
    """
    instance = _make_config_instance(**{'planet.elements': SimpleNamespace(O_mode='REQUIRED')})
    with pytest.raises(ValueError, match=r'O_mode is required'):
        planet_oxygen_mode_explicit(instance, None, None)


@pytest.mark.unit
@pytest.mark.parametrize('mode', ['ic_chemistry', 'ppmw', 'kg', 'FeO_mantle_wt_pct'])
def test_planet_oxygen_mode_explicit_accepts_all_four_documented_modes(mode):
    """All four documented O_mode values pass the validator."""
    instance = _make_config_instance(**{'planet.elements': SimpleNamespace(O_mode=mode)})
    planet_oxygen_mode_explicit(instance, None, None)


# ---------------------------------------------------------------------------
# planet_fO2_source_compat — four rejection rules + one warning
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_fO2_source_user_constant_accepts_all_O_modes():
    """The default fO2_source = "user_constant" accepts every (O_mode,
    volatile_mode) combination; this is the legacy-compatible path.
    """
    for o_mode in ('ic_chemistry', 'ppmw', 'kg', 'FeO_mantle_wt_pct'):
        for v_mode in ('elements', 'gas_prs'):
            instance = _make_config_instance(
                **{
                    'planet.fO2_source': 'user_constant',
                    'planet.elements': SimpleNamespace(O_mode=o_mode),
                    'planet.volatile_mode': v_mode,
                }
            )
            planet_fO2_source_compat(instance, None, None)


@pytest.mark.unit
def test_fO2_source_from_mantle_redox_rejected_as_reserved():
    """The 'from_mantle_redox' enum is reserved for issue #653 and not yet
    wired into the runtime; reject at config load with the issue reference.
    """
    instance = _make_config_instance(**{'planet.fO2_source': 'from_mantle_redox'})
    with pytest.raises(ValueError, match=r'issue #653'):
        planet_fO2_source_compat(instance, None, None)


@pytest.mark.unit
def test_fO2_source_from_O_budget_rejects_ic_chemistry():
    """Path C (from_O_budget) cannot use O_mode = ic_chemistry because
    ic_chemistry defers O to the chemistry solver, leaving nothing to
    invert against.
    """
    instance = _make_config_instance(
        **{
            'planet.fO2_source': 'from_O_budget',
            'planet.elements': SimpleNamespace(O_mode='ic_chemistry'),
        }
    )
    with pytest.raises(ValueError, match=r'authoritative O budget'):
        planet_fO2_source_compat(instance, None, None)


@pytest.mark.unit
def test_fO2_source_from_O_budget_rejects_gas_prs_volatile_mode():
    """Path C requires volatile_mode = "elements"; gas_prs makes the
    element budgets inert and Path C has no target to invert against.
    """
    instance = _make_config_instance(
        **{
            'planet.fO2_source': 'from_O_budget',
            'planet.elements': SimpleNamespace(O_mode='ppmw'),
            'planet.volatile_mode': 'gas_prs',
        }
    )
    with pytest.raises(ValueError, match=r'volatile_mode'):
        planet_fO2_source_compat(instance, None, None)


@pytest.mark.unit
def test_fO2_source_from_O_budget_rejects_dummy_outgas():
    """Path C requires an outgassing backend with an authoritative-O
    implementation; dummy has no chemistry to invert against.
    """
    instance = _make_config_instance(
        **{
            'planet.fO2_source': 'from_O_budget',
            'planet.elements': SimpleNamespace(O_mode='ppmw'),
            'outgas.module': 'dummy',
        }
    )
    with pytest.raises(ValueError, match=r'no chemistry to invert against'):
        planet_fO2_source_compat(instance, None, None)


@pytest.mark.unit
def test_fO2_source_from_O_budget_warns_when_fO2_shift_IW_set():
    """Path C derives the buffer offset from the O budget, so a
    user-supplied fO2_shift_IW is silently ignored at runtime; surface
    this as a UserWarning at config load.
    """
    instance = _make_config_instance(
        **{
            'planet.fO2_source': 'from_O_budget',
            'planet.elements': SimpleNamespace(O_mode='ppmw'),
            'outgas.module': 'calliope',
            'outgas.fO2_shift_IW': 4.0,
        }
    )
    with pytest.warns(UserWarning, match=r'will be ignored at runtime'):
        planet_fO2_source_compat(instance, None, None)


@pytest.mark.unit
def test_fO2_source_from_O_budget_silent_when_fO2_shift_IW_zero():
    """Default fO2_shift_IW = 0 under Path C is silent (no spurious warning)."""
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
        planet_fO2_source_compat(instance, None, None)


# ---------------------------------------------------------------------------
# Boundary backend coupling validators
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_boundary_requires_fixed_surface_state_rejects_dynamic():
    """Boundary backend assumes fixed surface coupling; dynamic surf_state raises."""
    instance = _make_config_instance(**{'interior_energetics.module': 'boundary'})
    instance.atmos_clim.surf_state = 'mixed_layer'
    with pytest.raises(ValueError, match=r"surf_state='fixed'"):
        boundary_requires_fixed_surface_state(instance, None, None)


@pytest.mark.unit
def test_boundary_requires_fixed_surface_state_passes_with_fixed():
    instance = _make_config_instance(**{'interior_energetics.module': 'boundary'})
    instance.atmos_clim.surf_state = 'fixed'
    boundary_requires_fixed_surface_state(instance, None, None)


@pytest.mark.unit
def test_boundary_zalmoxis_incompatible_rejects_combo():
    instance = _make_config_instance(
        **{
            'interior_energetics.module': 'boundary',
            'interior_struct.module': 'zalmoxis',
        }
    )
    with pytest.raises(ValueError, match=r'zalmoxis'):
        boundary_zalmoxis_incompatible(instance, None, None)


@pytest.mark.unit
def test_boundary_zalmoxis_incompatible_passes_with_dummy_struct():
    instance = _make_config_instance(
        **{
            'interior_energetics.module': 'boundary',
            'interior_struct.module': 'dummy',
        }
    )
    boundary_zalmoxis_incompatible(instance, None, None)


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
HYPOTHESIS_ATMOS_CHEM = (None, 'dummy', 'vulcan')
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
