"""Tests for src/proteus/config/orphans.py.

Exercises the orphan-key detection functions that guard against TOML keys
being silently discarded by the cattrs-based config parser.
"""

from __future__ import annotations

import pytest

from proteus.config.orphans import (
    _collect_orphan_keys,
    _extract_attrs_class,
    check_config_orphan_free,
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


def test_orphans_extract_attrs_class_plain_scalar_returns_none():
    """Plain scalars (float, bool, str) return None; no attrs walk should follow."""
    for hint in (float, bool, int, str):
        assert _extract_attrs_class(hint) is None, f'Expected None for {hint}'


# ---------------------------------------------------------------------------
# _collect_orphan_keys
# ---------------------------------------------------------------------------


def test_orphans_collect_orphan_keys_empty_dict_is_clean():
    """An empty raw dict has no orphans; the result must be an empty list."""
    from proteus.config._config import Config

    result = _collect_orphan_keys({}, Config)
    assert result == []
    # Discrimination: a non-empty invalid dict must NOT produce an empty list;
    # this verifies the clean path is really empty and not a degenerate always-pass.
    dirty = {'GHOST': 1}
    assert _collect_orphan_keys(dirty, Config) != []


def test_orphans_collect_orphan_keys_valid_top_level_keys():
    """Known top-level Config fields do not appear in the orphan list."""
    from proteus.config._config import Config

    data = {'star': {}, 'planet': {}, 'orbit': {}, 'config_version': '3.0'}
    result = _collect_orphan_keys(data, Config)
    assert result == []


def test_orphans_collect_orphan_keys_single_top_level_orphan():
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


def test_orphans_collect_orphan_keys_nested_orphan_in_planet():
    """A typo inside the planet section is reported with its dotted path."""
    from proteus.config._config import Config

    data = {'planet': {'mass_tot': 1.0, 'typo_field': 99}}
    result = _collect_orphan_keys(data, Config)
    assert result == ['planet.typo_field']

    # Discrimination: the valid key 'mass_tot' must not appear in orphans.
    assert 'planet.mass_tot' not in result


def test_orphans_collect_orphan_keys_deeply_nested_orphan():
    """Orphan keys inside doubly-nested sub-configs are reported with full path."""
    from proteus.config._config import Config

    data = {'star': {'mors': {'age_now': 4.5, 'bad_star_key': 'oops'}}}
    result = _collect_orphan_keys(data, Config)
    assert result == ['star.mors.bad_star_key']


def test_orphans_collect_orphan_keys_multiple_orphans_all_reported():
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


def test_orphans_collect_orphan_keys_valid_nested_dict_does_not_raise():
    """A known nested dict with all valid keys produces an empty orphan list."""
    from proteus.config._config import Config

    data = {
        'star': {'module': 'dummy'},
        'planet': {'mass_tot': 1.0},
    }
    result = _collect_orphan_keys(data, Config)
    assert result == []


def test_orphans_collect_orphan_keys_non_attrs_type_skips_recursion():
    """A known field with a plain scalar type does not trigger dict recursion."""
    from proteus.config._config import Config

    # config_version is a str field; a dict value for it is unusual but the
    # validator must not recurse into it (it has no attrs type).
    data = {'config_version': {'nested': 'should_not_recurse'}}
    result = _collect_orphan_keys(data, Config)

    # 'config_version' itself is a known field, so it should not appear.
    assert 'config_version' not in result

    # The nested key 'nested' should not appear because we do not recurse
    # into non-attrs-typed fields (since config_version is a str field)
    assert 'config_version.nested' not in result


# ---------------------------------------------------------------------------
# check_config_orphan_free
# ---------------------------------------------------------------------------


def test_orphans_check_config_orphan_free_clean_config_passes():
    """A config with no orphans returns True without raising."""
    result = check_config_orphan_free(_MINIMAL_VALID)
    assert result is True


# ---------------------------------------------------------------------------
# Integration with read_config_object
# ---------------------------------------------------------------------------


def test_orphans_detected_in_raw_dict_from_toml_file(tmp_path):
    """check_config_orphan_free detects orphan keys injected into a TOML file."""
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

    result = check_config_orphan_free(raw)
    assert result is False
    # Discrimination: the original dummy.toml must be orphan-free. A regression
    # that always returned False would fail this second check.
    with open(dummy_path, 'rb') as f:
        clean_raw = tomllib.load(f)
    assert check_config_orphan_free(clean_raw) is True


def test_orphans_detected_at_top_level_in_toml_file(tmp_path):
    """check_config_orphan_free detects an unknown top-level key in a TOML file."""
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
    result = check_config_orphan_free(raw)
    assert result is False

    # Discrimination: confirm the orphan is specifically EXTRA_SECTION
    from proteus.config._config import Config

    orphans = _collect_orphan_keys(raw, Config)
    assert 'EXTRA_SECTION' in orphans


# ---------------------------------------------------------------------------
# Proteus.__init__ integration
# ---------------------------------------------------------------------------


def test_orphans_proteus_init_raises_on_dirty_config(tmp_path):
    """Proteus.__init__ raises RuntimeError and writes status 20 for a dirty config.

    The dummy.toml is used as the valid base; one orphan key is injected into
    the [planet] section. Proteus resolves the output directory from the config,
    writes the status file, and then raises before completing initialisation.
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

    with pytest.raises(RuntimeError, match='Unknown configuration keys'):
        Proteus(config_path=dirty_path)

    # Status file must exist (this checks for regression where file isn't written)
    status_file = tmp_path / 'run_output' / 'status'
    assert status_file.exists(), 'Status file must be written before the raise'

    # Status file must be code 20
    content = status_file.read_text()
    assert content.startswith('20'), f'Expected status 20, got: {content!r}'


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
    assert check_config_orphan_free(raw) is True

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
