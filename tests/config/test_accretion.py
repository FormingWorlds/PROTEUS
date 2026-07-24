"""Tests for the giant-impact accretion config section.

This file targets _accretion.py (Accretion, Morrigan, AccretionDummy
parameters) and the ``[accretion]`` block of the reference configuration.
It exercises the module-selection contract, the conditional validators
that only bind when their backend is selected, the embryo-mass and
selector guards, and the impactor delivery budgets.

See testing standards in docs/How-to/testing.md and
docs/Explanations/test_framework.md for required structure, speed, and
physics validity.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_accretion_defaults_leave_the_module_disabled():
    """An unconfigured accretion section is inert and delivers nothing.

    The default must be off, because every existing config predates this
    section and must keep running unchanged. The discriminating check is
    that module resolves to the None singleton rather than the string
    'none': the main-loop dispatch tests identity against None, so a
    surviving 'none' string would silently select a non-existent backend.
    """
    from proteus.config._accretion import Accretion

    a = Accretion()

    assert a.module is None
    assert a.module != 'none'
    assert a.delivers_volatiles is False

    # Every impactor budget starts empty, so a bare accretion section
    # cannot move the volatile inventory even once impacts are enabled.
    for element in ('H', 'C', 'N', 'S', 'O'):
        assert getattr(a, f'impactor_{element}_ppmw') == pytest.approx(0.0)

    # Sub-configs exist even when unused, so downstream attribute access
    # never needs a None check before reading a backend parameter.
    assert a.dummy.timeline_path is None
    assert a.morrigan.selector == 'match_config'


@pytest.mark.unit
def test_module_validator_admits_only_registered_backends():
    """Module selection accepts the two backends and rejects anything else.

    'kimura' and 'formation_model' are the names this model was known by
    before, so they are the realistic typo cases and must fail loudly
    rather than fall through to a silent no-op.
    """
    from proteus.config._accretion import Accretion, AccretionDummy

    assert Accretion(module='morrigan').module == 'morrigan'
    assert (
        Accretion(module='dummy', dummy=AccretionDummy(timeline_path='x.csv')).module == 'dummy'
    )
    assert Accretion(module='none').module is None

    for bad in ('kimura', 'formation_model', 'Morrigan', ''):
        with pytest.raises(ValueError):
            Accretion(module=bad)


@pytest.mark.unit
def test_dummy_backend_requires_a_timeline_path():
    """The timeline-replay backend cannot run without a timeline to replay.

    Selecting the dummy backend with no path is a configuration error, not
    a quiet no-op, because the user asked for impacts and would otherwise
    get a run with none. The edge case in the other direction matters just
    as much: the same missing path must be accepted while the module is
    off, or every existing config would start failing validation.
    """
    from proteus.config._accretion import Accretion, AccretionDummy

    with pytest.raises(ValueError, match='timeline_path'):
        Accretion(module='dummy')

    supplied = Accretion(module='dummy', dummy=AccretionDummy(timeline_path='/tmp/t.csv'))
    assert supplied.dummy.timeline_path == '/tmp/t.csv'

    # Inert while the backend is unselected, including the explicit
    # 'none' sentinel that the reference TOML ships.
    assert Accretion(module='none').dummy.timeline_path is None
    assert Accretion().dummy.timeline_path is None


@pytest.mark.unit
def test_embryo_mass_list_must_match_the_embryo_count():
    """A per-embryo mass list is only meaningful at the declared length.

    A short or long list would silently truncate or pad the system, which
    changes the dynamics without any diagnostic. Lengths are checked
    against num_planets, an empty list is the documented "all equal"
    case, and non-positive masses are rejected because an embryo of zero
    or negative mass has no physical radius.
    """
    from proteus.config._accretion import Accretion, Morrigan

    # Empty list is the equal-mass case and stays legal.
    assert Accretion(module='morrigan', morrigan=Morrigan(num_planets=4)).morrigan.masses == []

    # Exact-length list is accepted and preserved in order.
    exact = Morrigan(num_planets=3, masses=[0.4, 1.2, 0.8])
    assert Accretion(module='morrigan', morrigan=exact).morrigan.masses == [0.4, 1.2, 0.8]

    # Too short and too long both fail, so the guard is not one-sided.
    for bad_masses in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0]):
        with pytest.raises(ValueError, match='num_planets'):
            Accretion(module='morrigan', morrigan=Morrigan(num_planets=3, masses=bad_masses))

    # Zero and negative embryo masses are unphysical.
    for bad_mass in (0.0, -1.0):
        with pytest.raises(ValueError, match='must be > 0'):
            Accretion(
                module='morrigan', morrigan=Morrigan(num_planets=2, masses=[1.0, bad_mass])
            )

    # The whole check is inert while the backend is unselected.
    off = Accretion(module='none', morrigan=Morrigan(num_planets=3, masses=[1.0]))
    assert off.morrigan.masses == [1.0]


@pytest.mark.unit
def test_targeted_selectors_require_a_selector_value():
    """Selectors that aim at a target need that target supplied.

    'semimajoraxis' and 'id' are meaningless without a value, so they must
    raise. 'mass' and 'match_config' derive their target from the run
    itself and must not, which is the discriminating half: a validator
    that demanded a value unconditionally would break the default
    configuration.
    """
    from proteus.config._accretion import Accretion, Morrigan

    for targeted in ('semimajoraxis', 'id'):
        with pytest.raises(ValueError, match='selector_value'):
            Accretion(module='morrigan', morrigan=Morrigan(selector=targeted))

    for targeted, value in (('semimajoraxis', 1.0), ('id', 3)):
        cfg = Accretion(
            module='morrigan', morrigan=Morrigan(selector=targeted, selector_value=value)
        )
        assert cfg.morrigan.selector_value == value

    for untargeted in ('mass', 'match_config'):
        cfg = Accretion(module='morrigan', morrigan=Morrigan(selector=untargeted))
        assert cfg.morrigan.selector == untargeted
        assert cfg.morrigan.selector_value is None

    with pytest.raises(ValueError):
        Morrigan(selector='instellation')


@pytest.mark.unit
def test_impactor_composition_drives_the_delivery_flag():
    """Delivery is on when any single element carries a positive budget.

    The flag decides whether the impact handler touches the element
    inventory at all, so it must respond to each element independently. A
    flag wired to only one element would look correct in any test that set
    hydrogen, which is why every element is checked in isolation here.
    """
    from proteus.config._accretion import Accretion

    assert Accretion().delivers_volatiles is False

    for element in ('H', 'C', 'N', 'S', 'O'):
        cfg = Accretion(**{f'impactor_{element}_ppmw': 250.0})
        assert cfg.delivers_volatiles is True, f'{element} budget ignored'
        assert getattr(cfg, f'impactor_{element}_ppmw') == pytest.approx(250.0)

    # A budget of exactly zero is the documented dry-impactor case and
    # must not switch delivery on.
    assert Accretion(impactor_H_ppmw=0.0).delivers_volatiles is False

    # Negative budgets would remove volatiles at an impact, which is the
    # escape module's job, not delivery's.
    with pytest.raises(ValueError):
        Accretion(impactor_C_ppmw=-1.0)


@pytest.mark.unit
def test_atmloss_config_bounds_and_module_selection_bind_at_load():
    """The impact atmosphere-loss options are validated at construction.

    The loss fraction only means anything on [0, 1]: partitioning a negative
    or beyond-total loss over the atmosphere is undefined, so both must be
    rejected when the config is built, not discovered mid-run at the first
    impact. The module selector accepts only the registered choices and the
    'none' string resolves to the None singleton, which is what the runtime
    dispatch tests identity against.
    """
    from proteus.config._accretion import Accretion

    # Defaults: loss disabled, fraction zero.
    a = Accretion()
    assert a.atmloss_module is None
    assert a.atmloss_module != 'none'
    assert a.atmloss_frac == pytest.approx(0.0)

    # The registered module and the full open interval load cleanly.
    assert Accretion(atmloss_module='constant', atmloss_frac=0.35).atmloss_frac == (
        pytest.approx(0.35)
    )
    # Both boundary values are legal: no loss, and complete stripping.
    assert Accretion(atmloss_frac=0.0).atmloss_frac == pytest.approx(0.0)
    assert Accretion(atmloss_frac=1.0).atmloss_frac == pytest.approx(1.0)

    # Out-of-bounds fractions are rejected at load.
    with pytest.raises(ValueError):
        Accretion(atmloss_frac=1.5)
    with pytest.raises(ValueError):
        Accretion(atmloss_frac=-0.1)

    # Unregistered loss modules are rejected at load; 'zephyrus' is the
    # realistic future name and must fail until the law actually exists.
    with pytest.raises(ValueError):
        Accretion(atmloss_module='zephyrus')
    with pytest.raises(ValueError):
        Accretion(atmloss_module='kegerreis')


@pytest.mark.unit
def test_reference_config_declares_the_accretion_section():
    """The shipped reference config parses and agrees with the schema.

    A key present in the TOML but absent from the schema is a silent
    orphan: the user sets it, nothing happens, nothing warns. This checks
    the section is orphan-free and that the documented values in the file
    match the attrs defaults, so the two layers cannot drift apart.
    """
    import tomllib

    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object
    from proteus.config._accretion import Accretion, Morrigan
    from proteus.config.orphans import check_config_orphan_free

    all_options = PROTEUS_ROOT / 'input' / 'all_options.toml'
    with open(all_options, 'rb') as f:
        raw = tomllib.load(f)

    assert 'accretion' in raw
    assert check_config_orphan_free(raw) is True

    # The reference file ships the section disabled.
    cfg = read_config_object(all_options)
    assert cfg.accretion.module is None
    assert cfg.accretion.delivers_volatiles is False

    # Documented values match the schema defaults, in both directions:
    # editing one layer without the other now fails here.
    defaults = Accretion()
    for element in ('H', 'C', 'N', 'S', 'O'):
        key = f'impactor_{element}_ppmw'
        assert raw['accretion'][key] == pytest.approx(getattr(defaults, key))

    morrigan_defaults = Morrigan()
    for key in ('seed', 'num_planets', 'selector'):
        assert raw['accretion']['morrigan'][key] == getattr(morrigan_defaults, key)
    for key in ('inner_edge', 'spacing', 'density', 'impact_angle', 'evolution_time'):
        assert raw['accretion']['morrigan'][key] == pytest.approx(
            getattr(morrigan_defaults, key)
        )


def _compat_instance(accretion_module, interior_module, temperature_mode='liquidus_super'):
    """Duck-typed config instance the interior-compatibility validator reads."""
    from types import SimpleNamespace

    return SimpleNamespace(
        accretion=SimpleNamespace(module=accretion_module),
        interior_energetics=SimpleNamespace(module=interior_module),
        planet=SimpleNamespace(temperature_mode=temperature_mode),
    )


@pytest.mark.unit
def test_accretion_on_spider_is_refused_at_config_load():
    """An accretion run on the SPIDER interior is rejected before it starts.

    A giant impact re-melts the mantle and SPIDER has no validated re-melt
    path, so the combination must fail at configuration load rather than many
    hours into a run at the first impact. The supported interiors are accepted,
    and a run without accretion is never blocked on this ground.
    """
    from proteus.config._config import check_accretion_interior_compatibility

    # The unsupported combination is refused, and the message names the fix.
    with pytest.raises(ValueError, match='SPIDER has no supported re-melt path'):
        check_accretion_interior_compatibility(
            _compat_instance('morrigan', 'spider'), None, None
        )
    with pytest.raises(ValueError, match='spider'):
        check_accretion_interior_compatibility(_compat_instance('dummy', 'spider'), None, None)

    # The supported interiors pass, and so does any run without accretion.
    for interior in ('aragog', 'dummy'):
        check_accretion_interior_compatibility(
            _compat_instance('morrigan', interior), None, None
        )
    # No accretion: SPIDER is fine, the check does not fire.
    check_accretion_interior_compatibility(_compat_instance(None, 'spider'), None, None)
