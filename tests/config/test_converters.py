"""Unit tests for config converter utilities.

Tests cover type conversion utilities used during TOML config parsing,
ensuring proper handling of special values (none/None), nested dictionaries,
and string transformations.
"""

from __future__ import annotations

import pytest

from proteus.config._converters import (
    dict_replace_none,
    lowercase,
    none_if_none,
    zero_if_none,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


class TestNoneIfNone:
    """Test none_if_none converter for TOML 'none' string → Python None."""

    @pytest.mark.unit
    def test_converts_none_string_to_none(self):
        """'none' string converts to None literal for optional config values."""
        assert none_if_none('none') is None
        # Discriminating check: the converter is strictly case-sensitive; a
        # regression that called `.lower()` first would have collapsed 'None'
        # to None too and corrupted any module name a user spelled in CamelCase.
        assert none_if_none('None') == 'None'

    @pytest.mark.unit
    def test_preserves_non_none_strings(self):
        """Non-'none' strings pass through unchanged for valid config values."""
        assert none_if_none('spider') == 'spider'
        assert none_if_none('mors') == 'mors'
        assert none_if_none('janus') == 'janus'

    @pytest.mark.unit
    def test_case_sensitive_none_detection(self):
        """Only lowercase 'none' converts; 'None', 'NONE' are preserved."""
        assert none_if_none('None') == 'None'
        assert none_if_none('NONE') == 'NONE'
        assert none_if_none('NoNe') == 'NoNe'

    @pytest.mark.unit
    def test_empty_string_preserved(self):
        """Empty string is distinct from 'none' and passes through."""
        assert none_if_none('') == ''
        # Discrimination: a regression that treated any falsy string as
        # 'none' would return None here. Pin the identity: the return is
        # the empty string itself, not None. `is None` would also catch a
        # regression that returned None for any non-'none' input.
        assert none_if_none('') is not None

    @pytest.mark.unit
    def test_whitespace_strings_preserved(self):
        """Whitespace variations are not treated as 'none'."""
        assert none_if_none(' none') == ' none'
        assert none_if_none('none ') == 'none '
        assert none_if_none(' none ') == ' none '


class TestZeroIfNone:
    """Test zero_if_none converter for numeric config defaults."""

    @pytest.mark.unit
    def test_converts_none_string_to_zero(self):
        """'none' string converts to 0.0 for optional numeric parameters."""
        assert zero_if_none('none') == pytest.approx(0.0, abs=1e-12)
        # Discrimination: case-sensitive on lowercase 'none' only. A
        # regression that called `.lower()` on the input first would
        # collapse 'None' to 0.0 too and silently coerce any string a
        # downstream consumer expected to round-trip as a label.
        assert zero_if_none('None') == 'None'

    @pytest.mark.unit
    def test_preserves_numeric_strings(self):
        """Numeric strings pass through for conversion by caller."""
        assert zero_if_none('42') == '42'
        assert zero_if_none('3.14') == '3.14'
        assert zero_if_none('-273.15') == '-273.15'

    @pytest.mark.unit
    def test_preserves_non_none_strings(self):
        """Non-'none' strings pass through unchanged."""
        assert zero_if_none('auto') == 'auto'
        assert zero_if_none('default') == 'default'

    @pytest.mark.unit
    def test_case_sensitive_none_detection(self):
        """Only lowercase 'none' converts to zero; variants preserved."""
        assert zero_if_none('None') == 'None'
        assert zero_if_none('NONE') == 'NONE'


class TestDictReplaceNone:
    """Test dict_replace_none for serializing configs to TOML."""

    @pytest.mark.unit
    def test_replaces_none_values_with_string(self):
        """None values convert to 'none' strings for TOML serialization."""
        data = {'star_module': None, 'atmos_module': 'janus'}
        result = dict_replace_none(data)
        assert result['star_module'] == 'none'
        assert result['atmos_module'] == 'janus'

    @pytest.mark.unit
    def test_preserves_non_none_values(self):
        """Non-None values pass through unchanged."""
        data = {
            'temperature': 300.0,
            'pressure': 1e5,
            'name': 'Earth',
            'active': True,
        }
        result = dict_replace_none(data)
        assert result == data
        # Discrimination: the converter must return a NEW dict, not the
        # same object. The input mapping is part of the user's config and
        # later mutation of `result` (e.g. TOML serialization) must not
        # write back through to `data`. A regression that returned `data`
        # itself would fail this identity check.
        assert result is not data

    @pytest.mark.unit
    def test_handles_nested_dictionaries(self):
        """Recursively processes nested config sections."""
        data = {
            'star': {'module': None, 'mass': 1.0},
            'planet': {'radius': None, 'name': 'test'},
        }
        result = dict_replace_none(data)
        assert result['star']['module'] == 'none'
        assert result['star']['mass'] == pytest.approx(1.0, rel=1e-12)
        assert result['planet']['radius'] == 'none'
        assert result['planet']['name'] == 'test'

    @pytest.mark.unit
    def test_handles_empty_dictionary(self):
        """Empty dict returns empty dict."""
        result = dict_replace_none({})
        assert result == {}
        # Discrimination: the converter must return a fresh dict (or at
        # least a value that compares equal to {} and is itself a dict),
        # not None or some scalar sentinel. A regression that returned the
        # input unchanged via `return d or None` would fail this type
        # check on the empty-dict input.
        assert isinstance(result, dict)

    @pytest.mark.unit
    def test_handles_all_none_values(self):
        """Dict with all None values converts all to 'none' strings."""
        data = {'a': None, 'b': None, 'c': None}
        result = dict_replace_none(data)
        assert result == {'a': 'none', 'b': 'none', 'c': 'none'}
        # Discrimination: the input dict must not be mutated in place. A
        # regression that operated on `data` directly would leave the
        # original mapping with 'none' strings instead of the True None
        # objects, breaking any caller that retained a reference.
        assert data == {'a': None, 'b': None, 'c': None}

    @pytest.mark.unit
    def test_deeply_nested_dictionaries(self):
        """Handles multiple levels of nesting correctly."""
        data = {
            'level1': {
                'level2': {
                    'level3': {'value': None, 'name': 'test'},
                    'other': 42,
                },
                'direct': None,
            }
        }
        result = dict_replace_none(data)
        assert result['level1']['level2']['level3']['value'] == 'none'
        assert result['level1']['level2']['level3']['name'] == 'test'
        assert result['level1']['level2']['other'] == 42
        assert result['level1']['direct'] == 'none'

    @pytest.mark.unit
    def test_mixed_types_preserved(self):
        """Non-dict, non-None values of various types preserved."""
        data = {
            'int_val': 42,
            'float_val': 3.14,
            'str_val': 'proteus',
            'bool_val': True,
            'list_val': [1, 2, 3],
            'none_val': None,
        }
        result = dict_replace_none(data)
        assert result['int_val'] == 42
        assert result['float_val'] == pytest.approx(3.14, rel=1e-12)
        assert result['str_val'] == 'proteus'
        assert result['bool_val'] is True
        assert result['list_val'] == [1, 2, 3]
        assert result['none_val'] == 'none'


class TestLowercase:
    """Test lowercase converter for case-insensitive config parsing."""

    @pytest.mark.unit
    def test_converts_uppercase_to_lowercase(self):
        """Uppercase strings convert to lowercase for standardization."""
        assert lowercase('SPIDER') == 'spider'
        assert lowercase('JANUS') == 'janus'

    @pytest.mark.unit
    def test_converts_mixed_case_to_lowercase(self):
        """Mixed-case strings convert to lowercase."""
        assert lowercase('ProTeus') == 'proteus'
        assert lowercase('AgNi') == 'agni'

    @pytest.mark.unit
    def test_preserves_already_lowercase(self):
        """Already-lowercase strings remain unchanged."""
        assert lowercase('mors') == 'mors'
        assert lowercase('calliope') == 'calliope'

    @pytest.mark.unit
    def test_empty_string_handling(self):
        """Empty string converts to empty string (idempotent on the
        boundary case)."""
        result = lowercase('')
        assert result == ''
        # Discrimination: the converter must be idempotent (lowercase of
        # an already-lowercase output is the same output). A regression
        # that returned a non-string sentinel (e.g. `None` or a stripped
        # value) on the empty input would fail this second pass.
        assert lowercase(result) == ''

    @pytest.mark.unit
    def test_preserves_numbers_and_special_chars(self):
        """Non-alphabetic characters pass through unchanged."""
        assert lowercase('Module123') == 'module123'
        assert lowercase('test_value') == 'test_value'
        assert lowercase('CONFIG-FILE.TOML') == 'config-file.toml'
