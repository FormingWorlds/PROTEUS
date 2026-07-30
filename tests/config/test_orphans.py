"""Tests for src/proteus/config/orphans.py.

Exercises the orphan-key detection functions that guard against TOML keys
being silently discarded by the cattrs-based config parser.
"""

from __future__ import annotations

import attrs
import pytest

from proteus.config.orphans import (
    _extract_attrs_class,
    find_key_problems,
    format_orphan_message,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@attrs.define
class _Leaf:
    """Innermost class of the synthetic schema used below."""

    value: int = 0


@attrs.define
class _Unresolvable:
    """Synthetic schema whose annotation names a class that does not exist.

    Used to drive the branch where `typing.get_type_hints` raises. A schema
    that cannot be introspected must still let the walk report the names it can
    compare, rather than aborting and letting every key through unchecked.
    """

    field: 'NoSuchClassAnywhere' = None  # noqa: F821


@attrs.define
class _Holder:
    """Synthetic schema pairing a container of tables with a single table.

    The real Config declares no container of nested classes, so the two shapes
    can only be compared against a schema written for the purpose. Defined at
    module scope because the walk resolves annotations through
    `typing.get_type_hints`, which cannot see names local to a function.
    """

    many: list[_Leaf] = attrs.field(factory=list)
    one: _Leaf = attrs.field(factory=_Leaf)


_MINIMAL_VALID = {
    'config_version': '3.0',
    'star': {'module': 'dummy', 'dummy': {'radius': 1.0}},
    'planet': {
        'mass_tot': 1.0,
        'volatile_mode': 'elements',
        'temperature_mode': 'adiabatic_from_cmb',
        'elements': {
            'O_mode': 'ic_chemistry',
            'O_budget': 0.0,
            'H_mode': 'oceans',
            'H_budget': 1.0,
            'C_mode': 'C/H',
            'C_budget': 0.0,
            'N_mode': 'N/H',
            'N_budget': 0.0,
            'S_mode': 'S/H',
            'S_budget': 0.0,
        },
    },
    'orbit': {'module': 'dummy', 'semimajoraxis': 0.5, 'eccentricity': 0.0},
    'interior_struct': {'module': 'dummy'},
    'interior_energetics': {'module': 'dummy'},
    'outgas': {'module': 'dummy'},
    'atmos_clim': {'module': 'dummy'},
    'escape': {'module': 'dummy'},
}


# ---------------------------------------------------------------------------
# _extract_attrs_class
# ---------------------------------------------------------------------------


def test_orphans_extract_attrs_class_direct_attrs_class():
    """Direct attrs class is returned unchanged."""
    from proteus.config._planet import Planet

    result = _extract_attrs_class(Planet)
    assert result is Planet
    # Discrimination: a plain scalar type must not be returned.
    assert _extract_attrs_class(str) is None


def test_orphans_extract_attrs_class_union_with_attrs():
    """Attrs class is extracted from a union hint such as ``Mors | None``."""
    from proteus.config._star import Mors

    union_hint = Mors | None
    result = _extract_attrs_class(union_hint)
    assert result is Mors
    # Discrimination: a union of two non-attrs types must return None so that
    # the caller does not attempt a recursive attrs walk on a plain scalar.
    assert _extract_attrs_class(str | None) is None


def test_orphans_extract_attrs_class_returns_none_for_non_attrs_hints():
    """Scalars and non-attrs containers yield no class, so no attrs walk follows.

    Two branches reach the same answer by different routes: a bare scalar has no
    ``__args__`` and exits early, while a parameterised container does enter the
    ``__args__`` loop but holds no attrs member. Both must fall through to None,
    or ``find_key_problems`` would recurse into a value the schema never
    declares as a sub-config and report its keys as orphans.
    """
    from proteus.config._planet import Planet

    for hint in (float, bool, int, str):
        assert _extract_attrs_class(hint) is None, f'Expected None for {hint}'

    # Container hints reach the __args__ loop but hold no attrs member.
    for hint in (list[float], dict[str, float]):
        assert _extract_attrs_class(hint) is None, f'Expected None for {hint}'

    # Discrimination: a real attrs class is still returned, so the None results
    # above come from the scalar and container branches and not from a
    # regression that returns None unconditionally.
    assert _extract_attrs_class(Planet) is Planet


# ---------------------------------------------------------------------------
# find_key_problems, unknown names
# ---------------------------------------------------------------------------


def test_orphans_find_key_problems_empty_dict_is_clean():
    """An empty raw dict has no orphans; the result must be an empty list."""
    from proteus.config._config import Config

    result = find_key_problems({}, Config)[0]
    assert result == []
    # Discrimination: a non-empty invalid dict must NOT produce an empty list;
    # this verifies the clean path is really empty and not a degenerate always-pass.
    dirty = {'GHOST': 1}
    assert find_key_problems(dirty, Config)[0] != []


def test_orphans_find_key_problems_valid_top_level_keys():
    """Known top-level Config fields do not appear in the orphan list.

    The paired negative adds the same unknown key to that very dict and expects
    it back as the whole list.
    """
    from proteus.config._config import Config

    data = {'star': {}, 'planet': {}, 'orbit': {}, 'config_version': '3.0'}
    result = find_key_problems(data, Config)[0]
    assert result == []

    # Discrimination: an unknown key alongside the very same valid keys is
    # still reported, so the empty list above reflects the keys being known
    # rather than a scan that never inspects a top-level dict.
    # ..._single_top_level_orphan covers an unknown key on its own and asserts
    # membership; this pins exact-list equality against a mixed dict, so a scan
    # that reported a known sibling alongside the orphan would fail here.
    mixed = dict(data, NOT_A_FIELD=1)
    assert find_key_problems(mixed, Config)[0] == ['NOT_A_FIELD']


def test_orphans_find_key_problems_single_top_level_orphan():
    """A single unknown top-level key is returned with its bare name."""
    from proteus.config._config import Config

    data = {'UNKNOWN_TOP': 'bad'}
    result = find_key_problems(data, Config)[0]
    assert result == ['UNKNOWN_TOP']

    # Discrimination: the known key 'star' must not appear in orphans.
    data_mixed = {'star': {}, 'UNKNOWN_TOP': 'bad'}
    mixed_result = find_key_problems(data_mixed, Config)[0]
    assert 'star' not in mixed_result
    assert 'UNKNOWN_TOP' in mixed_result


def test_orphans_find_key_problems_nested_orphan_in_planet():
    """A typo inside the planet section is reported with its dotted path."""
    from proteus.config._config import Config

    data = {'planet': {'mass_tot': 1.0, 'typo_field': 99}}
    result = find_key_problems(data, Config)[0]
    assert result == ['planet.typo_field']

    # Discrimination: the valid key 'mass_tot' must not appear in orphans.
    assert 'planet.mass_tot' not in result


def test_orphans_find_key_problems_deeply_nested_orphan():
    """Orphan keys inside doubly-nested sub-configs are reported with full path."""
    from proteus.config._config import Config

    data = {'star': {'mors': {'age_now': 4.5, 'bad_star_key': 'oops'}}}
    result = find_key_problems(data, Config)[0]
    assert result == ['star.mors.bad_star_key']

    # Orphans at two different depths in a single walk: each dotted path must
    # carry the prefix of the level it was found at. A recursion that reset or
    # dropped the prefix would report a bare 'bad_star_key' here and mislead
    # the user into looking for the typo one section too high.
    two_depths = {
        'star': {
            'module': 'mors',
            'NOT_A_STAR_FIELD': 1,
            'mors': {'age_now': 4.5, 'bad_star_key': 'oops'},
        }
    }
    assert sorted(find_key_problems(two_depths, Config)[0]) == [
        'star.NOT_A_STAR_FIELD',
        'star.mors.bad_star_key',
    ]


def test_orphans_find_key_problems_multiple_orphans_all_reported():
    """All orphan keys are collected; none is silently dropped."""
    from proteus.config._config import Config

    data = {
        'GHOST_A': 1,
        'planet': {'mass_tot': 1.0, 'typo_b': 2},
        'star': {'extra_c': 3},
    }
    result = find_key_problems(data, Config)[0]
    assert set(result) == {'GHOST_A', 'planet.typo_b', 'star.extra_c'}
    # Discrimination: exactly three orphans; valid keys must not inflate the count.
    assert len(result) == 3


def test_orphans_find_key_problems_valid_nested_dict_does_not_raise():
    """A known nested dict with all valid keys produces an empty orphan list.

    The paired negative misspells one of those same nested keys and expects the
    dotted path back.
    """
    from proteus.config._config import Config

    data = {
        'star': {'module': 'dummy'},
        'planet': {'mass_tot': 1.0},
    }
    result = find_key_problems(data, Config)[0]
    assert result == []

    # Discrimination: the walk does descend into these same nested dicts, so a
    # typo one level down is caught. Without this, the empty list above would
    # also be produced by a scan that never recursed past the top level.
    # ..._nested_orphan_in_planet covers a planet-level typo from a standing
    # start; this one mutates the dict just asserted clean, so the two results
    # come from the same input and the empty list cannot be a shape artefact.
    data['planet']['mass_totl'] = 1.0
    assert find_key_problems(data, Config)[0] == ['planet.mass_totl']


def test_orphans_find_key_problems_non_attrs_type_skips_recursion():
    """A known field with a plain scalar type does not trigger dict recursion."""
    from proteus.config._config import Config

    # config_version is a str field; a dict value for it is unusual but the
    # validator must not recurse into it (it has no attrs type).
    data = {'config_version': {'nested': 'should_not_recurse'}}
    result = find_key_problems(data, Config)[0]

    # 'config_version' itself is a known field, so it should not appear.
    assert 'config_version' not in result

    # The nested key 'nested' should not appear because we do not recurse
    # into non-attrs-typed fields (since config_version is a str field)
    assert 'config_version.nested' not in result


# ---------------------------------------------------------------------------
# find_key_problems, nested paths
# ---------------------------------------------------------------------------


def test_find_key_problems_names_reports_nested_paths_and_none_for_a_clean_config():
    """Unknown keys come back as dotted paths; a schema-conforming config yields none.

    The nested case is the one that matters: a misspelling two levels down in
    ``planet.elements`` costs the user the same silently-applied default as one
    at the top level, but only recursion into the nested class can see it.
    """
    from copy import deepcopy

    dirty = deepcopy(_MINIMAL_VALID)
    # Plural typo two levels deep, and an unknown top-level section.
    dirty['planet']['elements']['H_budgets'] = 1.0
    dirty['atmosphere'] = {'module': 'agni'}

    orphans = find_key_problems(dirty)[0]
    assert set(orphans) == {'planet.elements.H_budgets', 'atmosphere'}

    # The correctly spelled sibling sits right next to the typo and must not be
    # swept up with it, otherwise every config with a typo would look wholly
    # unrecognised.
    assert 'planet.elements.H_budget' not in orphans

    # An unknown section is reported once, at the section, rather than once per
    # key inside it: there is no schema below an unrecognised branch to compare
    # against.
    assert 'atmosphere.module' not in orphans

    # Edge case: the same config without the two injected keys is clean, so the
    # result above is driven by the injection rather than by the fixture.
    assert find_key_problems(_MINIMAL_VALID)[0] == []


# ---------------------------------------------------------------------------
# find_key_problems
# ---------------------------------------------------------------------------


def test_find_key_problems_catches_a_section_that_is_not_a_table():
    """A section given as anything but a table is reported, at any depth.

    Writing ``[[planet]]`` instead of ``[planet]`` declares an array of tables.
    The name still matches a schema field, so a check that only looks at names
    sees nothing wrong, while structuring discards the section whole and every
    parameter inside it silently reverts to its default. That is the same
    outcome as a misspelling and is reported the same way.
    """
    from copy import deepcopy

    # A discriminating value: it differs from the schema default, so a section
    # that is dropped rather than read is visible in the result.
    good = deepcopy(_MINIMAL_VALID)
    good['planet']['mass_tot'] = 3.7
    assert find_key_problems(good) == ([], [])

    # Array of tables at the top level.
    aot = deepcopy(good)
    aot['planet'] = [good['planet']]
    assert find_key_problems(aot) == ([], ['planet'])

    # Array of tables one level down, where only recursion can see it.
    nested = deepcopy(good)
    nested['planet']['elements'] = [good['planet']['elements']]
    assert find_key_problems(nested) == ([], ['planet.elements'])

    # Edge cases: a scalar and a string in place of a table are the same fault.
    for value in (5, 'earth'):
        scalar = deepcopy(good)
        scalar['planet'] = value
        assert find_key_problems(scalar) == ([], ['planet'])

    # The two kinds are reported separately rather than one masking the other.
    both = deepcopy(good)
    both['planet'] = [good['planet']]
    both['star']['typo_field'] = 1
    assert find_key_problems(both) == (['star.typo_field'], ['planet'])


def test_find_key_problems_allows_repeated_tables_for_a_container_field():
    """A field holding several nested classes accepts a list rather than a table.

    The schema declares no such field today, so this is checked against a
    synthetic one. It matters because the rule that refuses a non-table is
    otherwise applied to every nested class: adding a field typed as a list of
    them later would start refusing configurations that are correct, which is a
    worse failure than the one the refusal exists to prevent.
    """
    # A list against the container field is the intended shape, so neither list
    # reports it.
    assert find_key_problems({'many': [{'value': 1}]}, _Holder) == ([], [])

    # Discriminator: the same list against the single-table field beside it is
    # still refused, so the carve-out is driven by the field's type and is not
    # a blanket exemption for lists.
    assert find_key_problems({'one': [{'value': 1}]}, _Holder) == ([], ['one'])

    # A table remains valid for the single-table field, and an unknown name
    # inside it is still found, so recursion is unaffected.
    assert find_key_problems({'one': {'value': 1}}, _Holder) == ([], [])
    assert find_key_problems({'one': {'nope': 1}}, _Holder) == (['one.nope'], [])


def test_find_key_problems_catches_a_mistyped_optional_section():
    """An optional section typed as a union is held to the same shape rule.

    Module sections such as ``star.mors`` are declared ``Mors | None``, so the
    class is reached through the union branch rather than directly. Without it
    the most common optional sections in the schema would accept ``[[mors]]``
    and silently default the whole block.
    """
    from copy import deepcopy

    from proteus.config._star import Mors
    from proteus.config.orphans import _expects_single_table

    assert _expects_single_table(Mors | None) is True
    # Discrimination: a union carrying no attrs member must not demand a table,
    # or ordinary optional scalars would be refused.
    assert _expects_single_table(float | None) is False

    aot = deepcopy(_MINIMAL_VALID)
    aot['star']['mors'] = [{'age_now': 4.5}]
    assert find_key_problems(aot) == ([], ['star.mors'])

    # Edge case: the same section as a table is accepted, and an unknown name
    # inside it is still reported, so recursion through the union is intact.
    good = deepcopy(_MINIMAL_VALID)
    good['star']['mors'] = {'age_now': 4.5}
    assert find_key_problems(good) == ([], [])
    good['star']['mors']['bad_key'] = 1
    assert find_key_problems(good) == (['star.mors.bad_key'], [])


def test_find_key_problems_handles_a_schema_it_cannot_introspect():
    """A class that is not attrs, or whose hints fail to resolve, is survivable.

    Neither case should raise. The unintrospectable class contributes nothing,
    and the class whose annotations cannot be resolved still has its own field
    names compared, so an unknown key beside the broken one is reported rather
    than the whole section being waved through.
    """
    # Not an attrs class at all: nothing to compare against, nothing reported.
    assert find_key_problems({'anything': 1}, str) == ([], [])

    # Annotations that cannot be resolved: the walk logs and carries on using
    # the field names, so the known field is accepted and the unknown one is
    # still named.
    orphans, mistyped = find_key_problems({'field': {}, 'not_a_field': 1}, _Unresolvable)
    assert orphans == ['not_a_field']
    assert mistyped == []


def test_find_key_problems_leaves_a_mistyped_section_out_of_the_orphan_list():
    """A misdeclared section is not reported as an unrecognised key.

    ``planet`` is a name the schema declares, so calling it unrecognised would
    send the user looking for a spelling mistake that is not there. The two
    faults need different advice and are kept apart.
    """
    from copy import deepcopy

    aot = deepcopy(_MINIMAL_VALID)
    aot['planet'] = [_MINIMAL_VALID['planet']]

    orphans, mistyped = find_key_problems(aot)
    assert mistyped == ['planet']
    assert orphans == []
    # The name-only helper agrees, so the split is a property of the walk and
    # not of one caller's interpretation.
    assert find_key_problems(aot)[0] == []


# ---------------------------------------------------------------------------
# format_orphan_message
# ---------------------------------------------------------------------------


def test_format_orphan_message_quotes_every_key_and_names_the_file():
    """The rejection message lists all unrecognised keys, the file, and the reference.

    This message is the only guidance the user gets when a load is refused, so
    it has to name every key rather than just the first one, and say where the
    valid names are written down.
    """
    msg = format_orphan_message(['planet.mass_total', 'atmosphere'], '/runs/case.toml')

    # Both keys, not just the head of the list.
    assert '"planet.mass_total"' in msg
    assert '"atmosphere"' in msg
    assert '/runs/case.toml' in msg
    assert 'all_options.toml' in msg

    # Limit input: one key yields exactly one quoted name. A regression that
    # padded the list with a stray empty entry would quote four times here.
    single = format_orphan_message(['params.dt.maxium'], 'case.toml')
    assert single.count('"') == 2
    assert '"params.dt.maxium"' in single

    # The wording agrees with the count. Telling someone who mistyped one key
    # that "these keys are not part of the schema" reads as though the file has
    # more wrong with it than it does.
    assert 'Unrecognised configuration key in' in single
    assert 'This key is not part' in single
    assert 'these keys' not in single.lower()

    # The many-key message keeps the plural, so the singular above is chosen
    # from the count rather than applied to every message.
    assert 'Unrecognised configuration keys in' in msg
    assert 'These keys are not part' in msg


def test_format_orphan_message_reports_mistyped_sections_in_their_own_block():
    """A misdeclared section gets its own advice, not the spelling advice.

    The remedy differs: an unrecognised key wants a spelling check, a section
    written as an array of tables wants a bracket removed. Folding them into
    one block would give whichever user is in the minority the wrong
    instruction.
    """
    both = format_orphan_message(['planet.mass_total'], '/runs/case.toml', ['star'])

    assert '"planet.mass_total"' in both
    assert '"star"' in both
    # The unrecognised key comes first: it is the more common mistake, and the
    # section advice is useless to someone who has neither.
    assert both.index('Unrecognised') < both.index('Misdeclared')
    assert '[[name]]' in both
    # The reference line is printed once, not once per block.
    assert both.count('all_options.toml') == 1

    # Only sections: the spelling block is absent rather than empty, so nobody
    # is told to check a spelling when no name was misspelled.
    sections_only = format_orphan_message([], '/runs/case.toml', ['star'])
    assert 'Misdeclared configuration section in' in sections_only
    assert 'Unrecognised' not in sections_only

    # Only keys: symmetrically, no bracket advice appears.
    keys_only = format_orphan_message(['planet.mass_total'], '/runs/case.toml')
    assert 'Misdeclared' not in keys_only
    assert '[[name]]' not in keys_only


# ---------------------------------------------------------------------------
# Integration with read_config_object
# ---------------------------------------------------------------------------


def test_orphans_detected_in_raw_dict_from_toml_file(tmp_path):
    """An orphan key injected into a TOML file is found by name."""
    import tomllib

    from helpers import PROTEUS_ROOT

    dummy_path = PROTEUS_ROOT / 'input' / 'dummy.toml'
    base_content = dummy_path.read_text()

    # Inject an orphan key into the [planet] section.
    bad_content = base_content.replace(
        '[planet]', '[planet]\n    TYPO_FIELD = "this_key_does_not_exist"'
    )
    cfg_path = tmp_path / 'bad_config.toml'
    cfg_path.write_text(bad_content)

    with open(cfg_path, 'rb') as f:
        raw = tomllib.load(f)

    assert find_key_problems(raw)[0] == ['planet.TYPO_FIELD']
    # Discrimination: the original dummy.toml must be orphan-free. A regression
    # that always reported a key would fail this second check.
    with open(dummy_path, 'rb') as f:
        clean_raw = tomllib.load(f)
    assert find_key_problems(clean_raw)[0] == []


def test_orphans_detected_at_top_level_in_toml_file(tmp_path):
    """An unknown top-level key in a TOML file is found by name."""
    import tomllib

    from helpers import PROTEUS_ROOT

    dummy_path = PROTEUS_ROOT / 'input' / 'dummy.toml'
    base_content = dummy_path.read_text()

    # Prepend an unknown top-level key.
    bad_content = 'EXTRA_SECTION = "not_a_real_config_key"\n' + base_content
    cfg_path = tmp_path / 'top_level_orphan.toml'
    cfg_path.write_text(bad_content)

    # Read the config and check for orphans.
    with open(cfg_path, 'rb') as f:
        raw = tomllib.load(f)

    # The orphan key is detected at the TOP level.
    orphans = find_key_problems(raw)[0]
    assert orphans == ['EXTRA_SECTION']

    # Discrimination: the same walk over the schema root reports the identical
    # key, so the public helper is not silently widening the search.
    from proteus.config._config import Config

    assert find_key_problems(raw, Config)[0] == orphans


# ---------------------------------------------------------------------------
# Proteus.__init__ integration
# ---------------------------------------------------------------------------


def test_orphans_proteus_init_raises_on_dirty_config(tmp_path):
    """The runner names the unknown key and writes status 20 before refusing.

    The dummy.toml is used as the valid base; one orphan key is injected into
    the [planet] section. Proteus resolves the output directory from the config,
    writes the status file, and then raises before completing initialisation.
    The key has to appear in the exception itself: at this point in startup no
    log handler is attached, so anything reported only through the logger is
    lost and the user is left with a run that stopped for no stated reason.
    """
    import tomllib

    import tomlkit
    from helpers import PROTEUS_ROOT

    from proteus import Proteus

    dummy_path = PROTEUS_ROOT / 'input' / 'dummy.toml'

    # Build a dirty config: copy dummy.toml and inject an unknown key.
    with open(dummy_path, 'rb') as f:
        raw = tomllib.load(f)
    raw['planet']['TYPO_FIELD'] = 'bad_value'

    # Point the output path to a known location so we can inspect the status file.
    out_path = str(tmp_path / 'run_output')
    raw['params']['out']['path'] = out_path

    dirty_path = tmp_path / 'dirty.toml'
    with open(dirty_path, 'w') as f:
        tomlkit.dump(raw, f)

    with pytest.raises(ValueError) as excinfo:
        Proteus(config_path=dirty_path)

    # The offending key is carried by the exception, not only by a log line.
    assert 'planet.TYPO_FIELD' in str(excinfo.value)

    # Status file must exist (this checks for regression where file isn't written)
    status_file = tmp_path / 'run_output' / 'status'
    assert status_file.exists(), 'Status file must be written before the raise'

    # Status file must be code 20
    content = status_file.read_text()
    assert content.startswith('20'), f'Expected status 20, got: {content!r}'


def test_orphans_proteus_init_names_the_unknown_key_before_a_bad_value(tmp_path):
    """The runner leads with an unknown key but keeps the other complaint too.

    A misspelling is usually what leaves a value wrong, so reporting the value
    first sends the user after a symptom. Dropping the value complaint is no
    better: it can name a missing package or an unreadable path that the key
    alone does not explain, so both belong in the message with the key first.

    A file that fails to structure has no status file written for it, unlike a
    file whose only fault is the key. The output directory is named inside the
    configuration, so a file that cannot be structured has nowhere to record
    anything; the message is the whole of the report. That limit is pinned here
    so it stays a deliberate one.
    """
    import tomllib

    import tomlkit
    from helpers import PROTEUS_ROOT

    from proteus import Proteus

    with open(PROTEUS_ROOT / 'input' / 'dummy.toml', 'rb') as f:
        raw = tomllib.load(f)
    raw['params']['out']['path'] = str(tmp_path / 'run_output')
    raw['planet']['mass_tot'] = -5.0  # rejected: the planet mass must be positive
    raw['planet']['mass_total'] = 2.5  # unknown key

    both = tmp_path / 'both.toml'
    with open(both, 'w') as f:
        tomlkit.dump(raw, f)

    with pytest.raises(ValueError) as excinfo:
        Proteus(config_path=both)
    message = str(excinfo.value)
    assert 'planet.mass_total' in message
    # The value complaint survives alongside it, and comes second.
    assert 'must be > 0' in message
    assert message.index('planet.mass_total') < message.index('must be > 0')

    # Nothing was recorded on disk: structuring failed before the output
    # directory could be resolved, so the refusal lives only in the message.
    # The sibling test above, where the key is the only fault, does get a
    # status file, and that contrast is the point.
    assert not (tmp_path / 'run_output' / 'status').exists()

    # Edge case: with the unknown key removed the negative mass is still the
    # only thing reported, so the key does not crowd out an unrelated failure
    # and does not attach itself to a file that has none.
    del raw['planet']['mass_total']
    value_only = tmp_path / 'value_only.toml'
    with open(value_only, 'w') as f:
        tomlkit.dump(raw, f)
    with pytest.raises(ValueError) as value_info:
        Proteus(config_path=value_only)
    assert 'must be > 0' in str(value_info.value)
    # Singular substring, so neither the one-key nor the many-key heading slips
    # past this check.
    assert 'Unrecognised configuration key' not in str(value_info.value)


def test_orphans_proteus_init_refuses_a_section_written_as_an_array_of_tables(tmp_path):
    """The runner refuses ``[[planet]]`` and records it, like any other refusal.

    The runner checks the raw file itself rather than letting the strict loader
    do it, so the loader being correct says nothing about this path, which is
    the one an actual run takes. Left unrefused the section is discarded whole
    and every parameter inside it reverts to its default.
    """
    from helpers import PROTEUS_ROOT

    from proteus import Proteus

    source = (PROTEUS_ROOT / 'input' / 'dummy.toml').read_text()
    # A value distinct from the schema default, so a dropped section would be
    # visible rather than having to be inferred.
    source = source.replace('mass_tot      = 1.0', 'mass_tot      = 3.7')
    assert 'mass_tot      = 3.7' in source, 'fixture no longer matches dummy.toml'
    assert source.count('path = "auto"') == 1, 'fixture no longer matches dummy.toml'
    source = source.replace('path = "auto"', f'path = "{tmp_path / "run_output"}"')

    array = tmp_path / 'array_of_tables.toml'
    array.write_text(source.replace('\n[planet]\n', '\n[[planet]]\n'))

    with pytest.raises(ValueError) as excinfo:
        Proteus(config_path=array)
    assert 'planet' in str(excinfo.value)
    assert 'Misdeclared' in str(excinfo.value)

    # Recorded on disk as well, so a run stopped this way is distinguishable
    # from one that died without reaching the check.
    status_file = tmp_path / 'run_output' / 'status'
    assert status_file.exists(), 'Status file must be written before the raise'
    assert status_file.read_text().strip().startswith('20')

    # Discriminator: the same file as a plain table is accepted and carries the
    # value, so the refusal is caused by the brackets rather than by this
    # fixture being unloadable.
    single = tmp_path / 'single_table.toml'
    single.write_text(source)
    assert Proteus(config_path=single).config.planet.mass_tot == pytest.approx(3.7)


def test_orphans_proteus_init_keeps_the_key_when_the_directories_fail(tmp_path, monkeypatch):
    """A pending unknown key survives a failure to resolve the output directory.

    Resolving the directories needs a configured environment and can fail on
    its own, most often because FWL_DATA is unset. Someone installing for the
    first time can easily have both that and a typo, and reporting only the
    environment leaves the typo to be discovered on the next attempt.
    """
    import tomllib

    import tomlkit
    from helpers import PROTEUS_ROOT

    import proteus.utils.coupler as coupler
    from proteus import Proteus

    def _no_directories(_config):
        raise OSError('The FWL_DATA environment variable has not been set.')

    monkeypatch.setattr(coupler, 'set_directories', _no_directories)

    with open(PROTEUS_ROOT / 'input' / 'dummy.toml', 'rb') as f:
        raw = tomllib.load(f)
    raw['params']['out']['path'] = str(tmp_path / 'run_output')
    raw['planet']['mass_total'] = 2.5  # unknown key, the only fault in the file

    typo = tmp_path / 'typo.toml'
    with open(typo, 'w') as f:
        tomlkit.dump(raw, f)

    with pytest.raises(ValueError) as excinfo:
        Proteus(config_path=typo)
    message = str(excinfo.value)
    # Both complaints reach the user, with the key first: the environment is
    # the easier of the two to notice without being told.
    assert 'planet.mass_total' in message
    assert 'FWL_DATA' in message
    assert message.index('planet.mass_total') < message.index('FWL_DATA')

    # Edge case and discriminator: the same directory failure on a file with no
    # unknown key propagates untouched, so the wrapping is driven by the pending
    # key rather than applied to every startup failure.
    del raw['planet']['mass_total']
    clean = tmp_path / 'clean.toml'
    with open(clean, 'w') as f:
        tomlkit.dump(raw, f)
    with pytest.raises(OSError) as os_info:
        Proteus(config_path=clean)
    assert 'FWL_DATA' in str(os_info.value)
    assert 'Unrecognised configuration key' not in str(os_info.value)


def test_orphans_proteus_init_succeeds_on_clean_config(tmp_path):
    """Proteus.__init__ completes without error when the config has no orphan keys.

    This is the discrimination twin of the dirty-config test: it verifies that
    the orphan gate does not fire on a valid config, confirming the check is
    conditional rather than always-raising.
    """

    # import toml libs
    import tomllib

    import tomlkit
    from helpers import PROTEUS_ROOT

    from proteus import Proteus
    from proteus.config import Config

    # read dummy config
    dummy_path = PROTEUS_ROOT / 'input' / 'dummy.toml'
    with open(dummy_path, 'rb') as f:
        raw = tomllib.load(f)

    # update output path to tmpdir
    out_path = str(tmp_path / 'run_output')
    raw['params']['out']['path'] = out_path

    # write clean config to tmpdir again
    clean_path = tmp_path / 'clean.toml'
    with open(clean_path, 'w') as f:
        tomlkit.dump(raw, f)

    # try to set up proteus runner (should not raise)
    runner = Proteus(config_path=clean_path)
    assert isinstance(runner.config, Config)

    # Discrimination: the output directory must have been resolved
    assert runner.directories is not None


def test_all_options_phase_boundary_margin_declared_and_resolves():
    """The phase-boundary entropy margin lives in both the TOML and the schema.

    A key present in the reference TOML but absent from the schema would be a
    silent orphan: the user could set it with intent and get no effect and no
    warning. This pins that ``phase_boundary_entropy_margin`` is declared in
    both layers, sits at the documented 200.0 default in the reference TOML,
    keeps ``input/all_options.toml`` orphan-free, and resolves to 200.0 through
    the parser. The omitted-key bit-identity claim is exercised separately by
    ``test_omitted_phase_boundary_margin_resolves_to_default``.
    """
    import tomllib

    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object

    all_options = PROTEUS_ROOT / 'input' / 'all_options.toml'
    with open(all_options, 'rb') as f:
        raw = tomllib.load(f)

    # Present in the reference TOML at the documented default.
    toml_value = raw['interior_energetics']['aragog']['phase_boundary_entropy_margin']
    assert toml_value == pytest.approx(200.0)

    # The whole reference file stays orphan-free, so the new key has a schema
    # home and is not one of the silently-discarded orphans.
    assert find_key_problems(raw)[0] == []

    # End-to-end resolution through the parser yields the same 200.0. Pinning
    # the exact band is itself the discriminator: it rejects a 0.0 step-cap-
    # style sentinel that would silently disable the near-boundary max_step
    # tightening this knob exists to control.
    cfg = read_config_object(all_options)
    resolved = cfg.interior_energetics.aragog.phase_boundary_entropy_margin
    assert resolved == pytest.approx(200.0)


def test_omitted_phase_boundary_margin_resolves_to_default(tmp_path):
    """A config that omits the margin key resolves to Aragog's 200.0 default.

    The release-gate guarantee is that surfacing the knob does not move any
    converged result: a TOML with no ``phase_boundary_entropy_margin`` must
    reach the solver with the same 200.0 band that was previously hard-coded
    inside Aragog. This deletes the key from the reference config, re-parses,
    and asserts the schema default fills in exactly, so the omitted-key path is
    bit-identical to prior behaviour rather than merely orphan-free.
    """
    import tomllib

    import tomlkit
    from helpers import PROTEUS_ROOT

    from proteus.config import read_config_object

    all_options = PROTEUS_ROOT / 'input' / 'all_options.toml'
    with open(all_options, 'rb') as f:
        raw = tomllib.load(f)

    # Drop the key so the parser must fall back to the attrs default.
    del raw['interior_energetics']['aragog']['phase_boundary_entropy_margin']
    assert 'phase_boundary_entropy_margin' not in raw['interior_energetics']['aragog']

    stripped = tmp_path / 'no_margin.toml'
    with open(stripped, 'w') as f:
        tomlkit.dump(raw, f)

    cfg = read_config_object(stripped)
    resolved = cfg.interior_energetics.aragog.phase_boundary_entropy_margin
    # Exact 200.0 default, not the 0.0 sentinel the step caps use: an omitted
    # band must not silently disable the near-boundary max_step tightening.
    assert resolved == pytest.approx(200.0)
