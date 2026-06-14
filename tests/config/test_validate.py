"""Tests for src/proteus/config/validate.py.

Exercises the orphan-key detection functions that guard against TOML keys
being silently discarded by the cattrs-based config parser.
"""

from __future__ import annotations

import pytest

from proteus.config import read_config_object
from proteus.config.validate import (
    _collect_orphan_keys,
    _extract_attrs_class,
    check_for_unknown_keys,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def test_cfgvalid_extract_attrs_class_direct_attrs_class():
    """Direct attrs class is returned unchanged."""
    from proteus.config._planet import Planet

    result = _extract_attrs_class(Planet)
    assert result is Planet
    # Discrimination: a plain scalar type must not be returned.
    assert _extract_attrs_class(str) is None


def test_cfgvalid_extract_attrs_class_union_with_attrs():
    """Attrs class is extracted from a union hint such as ``Mors | None``."""
    from proteus.config._star import Mors

    union_hint = Mors | None
    result = _extract_attrs_class(union_hint)
    assert result is Mors
    # Discrimination: a union of two non-attrs types must return None so that
    # the caller does not attempt a recursive attrs walk on a plain scalar.
    assert _extract_attrs_class(str | None) is None


def test_cfgvalid_extract_attrs_class_plain_scalar_returns_none():
    """Plain scalars (float, bool, str) return None; no attrs walk should follow."""
    for hint in (float, bool, int, str):
        assert _extract_attrs_class(hint) is None, f'Expected None for {hint}'


# ---------------------------------------------------------------------------
# _collect_orphan_keys
# ---------------------------------------------------------------------------


def test_cfgvalid_collect_orphan_keys_empty_dict_is_clean():
    """An empty raw dict has no orphans; the result must be an empty list."""
    from proteus.config._config import Config

    result = _collect_orphan_keys({}, Config)
    assert result == []
    # Discrimination: a non-empty invalid dict must NOT produce an empty list;
    # this verifies the clean path is really empty and not a degenerate always-pass.
    dirty = {'GHOST': 1}
    assert _collect_orphan_keys(dirty, Config) != []


def test_cfgvalid_collect_orphan_keys_valid_top_level_keys():
    """Known top-level Config fields do not appear in the orphan list."""
    from proteus.config._config import Config

    data = {'star': {}, 'planet': {}, 'orbit': {}, 'config_version': '3.0'}
    result = _collect_orphan_keys(data, Config)
    assert result == []


def test_cfgvalid_collect_orphan_keys_single_top_level_orphan():
    """A single unknown top-level key is returned with its bare name."""
    from proteus.config._config import Config

    data = {'UNKNOWN_TOP': 'bad'}
    result = _collect_orphan_keys(data, Config)
    assert result == ['UNKNOWN_TOP']
    # Discrimination: the known key 'star' must not appear in orphans.
    data_mixed = {'star': {}, 'UNKNOWN_TOP': 'bad'}
    mixed_result = _collect_orphan_keys(data_mixed, Config)
    assert 'star' not in mixed_result
    assert 'UNKNOWN_TOP' in mixed_result


def test_cfgvalid_collect_orphan_keys_nested_orphan_in_planet():
    """A typo inside the planet section is reported with its dotted path."""
    from proteus.config._config import Config

    data = {'planet': {'mass_tot': 1.0, 'typo_field': 99}}
    result = _collect_orphan_keys(data, Config)
    assert result == ['planet.typo_field']
    # Discrimination: the valid key 'mass_tot' must not appear in orphans.
    assert 'planet.mass_tot' not in result


def test_cfgvalid_collect_orphan_keys_deeply_nested_orphan():
    """Orphan keys inside doubly-nested sub-configs are reported with full path."""
    from proteus.config._config import Config

    data = {'star': {'mors': {'age_now': 4.5, 'bad_star_key': 'oops'}}}
    result = _collect_orphan_keys(data, Config)
    assert result == ['star.mors.bad_star_key']
    # Discrimination: a key at the wrong nesting level is not reported at the
    # parent level, only at its actual location.
    assert 'star.bad_star_key' not in result
    assert 'bad_star_key' not in result


def test_cfgvalid_collect_orphan_keys_multiple_orphans_all_reported():
    """All orphan keys are collected; none is silently dropped."""
    from proteus.config._config import Config

    data = {
        'GHOST_A': 1,
        'planet': {'mass_tot': 1.0, 'typo_b': 2},
        'star': {'extra_c': 3},
    }
    result = _collect_orphan_keys(data, Config)
    assert set(result) == {'GHOST_A', 'planet.typo_b', 'star.extra_c'}
    # Discrimination: exactly three orphans; valid keys must not inflate the count.
    assert len(result) == 3


def test_cfgvalid_collect_orphan_keys_valid_nested_dict_does_not_raise():
    """A known nested dict with all valid keys produces an empty orphan list."""
    from proteus.config._config import Config

    data = {
        'star': {'module': 'dummy'},
        'planet': {'mass_tot': 1.0},
    }
    result = _collect_orphan_keys(data, Config)
    assert result == []


def test_cfgvalid_collect_orphan_keys_non_attrs_type_skips_recursion():
    """A known field with a plain scalar type does not trigger dict recursion."""
    from proteus.config._config import Config

    # config_version is a str field; a dict value for it is unusual but the
    # validator must not recurse into it (it has no attrs type).
    data = {'config_version': {'nested': 'should_not_recurse'}}
    result = _collect_orphan_keys(data, Config)
    # 'config_version' itself is a known field, so it should not appear.
    assert 'config_version' not in result
    # The nested key 'nested' should not appear because we do not recurse
    # into non-attrs-typed fields.
    assert 'config_version.nested' not in result


# ---------------------------------------------------------------------------
# check_for_unknown_keys
# ---------------------------------------------------------------------------


def test_cfgvalid_check_for_unknown_keys_clean_config_passes():
    """A config with no orphans returns without raising."""
    result = check_for_unknown_keys(_MINIMAL_VALID)
    assert result is None


def test_cfgvalid_check_for_unknown_keys_orphan_raises_value_error():
    """Any orphan key causes ValueError with the dotted path in the message."""
    dirty = {**_MINIMAL_VALID, 'planet': {**_MINIMAL_VALID['planet'], 'TYPO': 1}}
    with pytest.raises(ValueError, match='planet.TYPO'):
        check_for_unknown_keys(dirty)

    # Discrimination: a config without the orphan must not raise. A regression
    # that always raised would fail this second call.
    check_for_unknown_keys(_MINIMAL_VALID)


def test_cfgvalid_check_for_unknown_keys_message_names_all_orphans():
    """The error message lists every orphan key, not just the first one."""
    dirty = {**_MINIMAL_VALID, 'GHOST_A': 1, 'GHOST_B': 2}
    with pytest.raises(ValueError, match='GHOST_A') as exc_info:
        check_for_unknown_keys(dirty)
    assert 'GHOST_B' in str(exc_info.value)
    # Discrimination: the count in the message must be 2 (not 0 or 1).
    assert '2 unrecognised' in str(exc_info.value)


def test_cfgvalid_check_for_unknown_keys_message_references_all_options():
    """The error message points users to input/all_options.toml for guidance."""
    with pytest.raises(ValueError, match='all_options.toml'):
        check_for_unknown_keys({**_MINIMAL_VALID, 'BAD': 'x'})


def test_cfgvalid_check_for_unknown_keys_updates_status_when_outdir_provided(tmp_path):
    """Status file is written with code 20 when outdir is supplied, on error."""
    dirty = {**_MINIMAL_VALID, 'ORPHAN': 42}

    with pytest.raises(ValueError):
        check_for_unknown_keys(dirty, outdir=str(tmp_path))

    status_file = tmp_path / 'status'
    assert status_file.exists(), 'Status file must be created on orphan-key error'
    content = status_file.read_text()
    # Status code 20 is the generic configuration-error sentinel.
    assert content.startswith('20'), f'Expected status 20, got: {content!r}'
    # Discrimination: the file must contain an error description as well, not
    # just the numeric code. The two-line format is mandated by UpdateStatusfile.
    lines = content.splitlines()
    assert len(lines) >= 2, 'Status file must have at least two lines (code + description)'


def test_cfgvalid_check_for_unknown_keys_no_status_without_outdir(tmp_path):
    """Status file is not written when outdir is None (early config-load path)."""
    dirty = {**_MINIMAL_VALID, 'ORPHAN': 42}

    with pytest.raises(ValueError):
        check_for_unknown_keys(dirty, outdir=None)

    # No status file should have been created anywhere under tmp_path because
    # no outdir was supplied. (We verify via the absence of any 'status' file.)
    status_file = tmp_path / 'status'
    assert not status_file.exists()


# ---------------------------------------------------------------------------
# Integration with read_config_object
# ---------------------------------------------------------------------------


def test_cfgvalid_read_config_object_raises_on_orphan_key(tmp_path):
    """read_config_object propagates orphan-key errors from the raw TOML."""
    from helpers import PROTEUS_ROOT

    dummy_path = PROTEUS_ROOT / 'input' / 'dummy.toml'
    base_content = dummy_path.read_text()

    # Inject an orphan key into the [planet] section.
    bad_content = base_content.replace(
        '[planet]', '[planet]\n    TYPO_FIELD = "this_key_does_not_exist"'
    )
    cfg_path = tmp_path / 'bad_config.toml'
    cfg_path.write_text(bad_content)

    with pytest.raises(ValueError, match='planet.TYPO_FIELD'):
        read_config_object(cfg_path)

    # Discrimination: the original dummy.toml must parse cleanly. A regression
    # that always raised would fail this second call.
    from proteus.config import Config

    assert isinstance(read_config_object(dummy_path), Config)


def test_cfgvalid_read_config_object_raises_on_top_level_orphan_key(tmp_path):
    """A completely unknown top-level TOML section is detected and rejected."""
    from helpers import PROTEUS_ROOT

    dummy_path = PROTEUS_ROOT / 'input' / 'dummy.toml'
    base_content = dummy_path.read_text()

    # Prepend an unknown top-level key before any section.
    bad_content = 'EXTRA_SECTION = "not_a_real_config_key"\n' + base_content
    cfg_path = tmp_path / 'top_level_orphan.toml'
    cfg_path.write_text(bad_content)

    with pytest.raises(ValueError, match='EXTRA_SECTION'):
        read_config_object(cfg_path)
