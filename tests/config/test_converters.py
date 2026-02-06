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


class TestNoneIfNone:
    """Test none_if_none converter for TOML 'none' string → Python None."""

    @pytest.mark.unit
    def test_converts_none_string_to_none(self):
        """'none' string converts to None literal for optional config values."""
        assert none_if_none('none') is None

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
        assert zero_if_none('none') == 0.0

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

    @pytest.mark.unit
    def test_handles_nested_dictionaries(self):
        """Recursively processes nested config sections."""
        data = {
            'star': {'module': None, 'mass': 1.0},
            'planet': {'radius': None, 'name': 'test'},
        }
        result = dict_replace_none(data)
        assert result['star']['module'] == 'none'
        assert result['star']['mass'] == 1.0
        assert result['planet']['radius'] == 'none'
        assert result['planet']['name'] == 'test'

    @pytest.mark.unit
    def test_handles_empty_dictionary(self):
        """Empty dict returns empty dict."""
        assert dict_replace_none({}) == {}

    @pytest.mark.unit
    def test_handles_all_none_values(self):
        """Dict with all None values converts all to 'none' strings."""
        data = {'a': None, 'b': None, 'c': None}
        result = dict_replace_none(data)
        assert result == {'a': 'none', 'b': 'none', 'c': 'none'}

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
        assert result['float_val'] == 3.14
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
        """Empty string converts to empty string."""
        assert lowercase('') == ''

    @pytest.mark.unit
    def test_preserves_numbers_and_special_chars(self):
        """Non-alphabetic characters pass through unchanged."""
        assert lowercase('Module123') == 'module123'
        assert lowercase('test_value') == 'test_value'
        assert lowercase('CONFIG-FILE.TOML') == 'config-file.toml'
