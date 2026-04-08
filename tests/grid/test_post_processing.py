"""
Unit tests for proteus.grid.post_processing helper functions.

Tests the pure-Python helper functions that require no running grid,
no file I/O, and no compiled binaries. Each test is fast (<100 ms).

See also:
    docs/How-to/test_infrastructure.md
    docs/How-to/test_categorization.md
    docs/How-to/test_building.md
"""

from __future__ import annotations

import pandas as pd
import pytest

from proteus.grid.post_processing import (
    clean_series,
    flatten_input_parameters,
    get_grid_name,
    get_label,
    get_log_scale,
    get_scale,
    group_output_by_parameter,
    latex,
    load_ecdf_plot_settings,
    validate_output_variables,
)

# ---------------------------------------------------------
# get_grid_name
# ---------------------------------------------------------


@pytest.mark.unit
def test_get_grid_name_returns_directory_name(tmp_path):
    """get_grid_name should return the last component of a valid directory path."""
    grid_dir = tmp_path / 'my_test_grid'
    grid_dir.mkdir()
    assert get_grid_name(grid_dir) == 'my_test_grid'


@pytest.mark.unit
def test_get_grid_name_raises_for_nonexistent_path(tmp_path):
    """get_grid_name should raise ValueError if the path is not a directory."""
    missing = tmp_path / 'does_not_exist'
    with pytest.raises(ValueError, match='not a valid directory'):
        get_grid_name(missing)


@pytest.mark.unit
def test_get_grid_name_accepts_string(tmp_path):
    """get_grid_name should accept a plain string as well as a Path."""
    grid_dir = tmp_path / 'str_grid'
    grid_dir.mkdir()
    assert get_grid_name(str(grid_dir)) == 'str_grid'


# ---------------------------------------------------------
# get_label
# ---------------------------------------------------------


@pytest.mark.unit
def test_get_label_known_quantity():
    """Preset quantities should return their human-readable label."""
    label = get_label('T_surf')
    assert 'surf' in label.lower() or 'T' in label


@pytest.mark.unit
def test_get_label_unknown_quantity_returns_last_segment():
    """Unknown dotted path should return only the last segment."""
    label = get_label('some.deeply.nested.param')
    assert label == 'param'


@pytest.mark.unit
def test_get_label_simple_unknown():
    """Unknown non-dotted key should be returned unchanged."""
    label = get_label('totally_unknown_var')
    assert label == 'totally_unknown_var'


# ---------------------------------------------------------
# get_scale
# ---------------------------------------------------------


@pytest.mark.unit
def test_get_scale_known_quantity():
    """Preset quantities should return a non-default (or explicitly 1.0) scale."""
    scale = get_scale('Phi_global')
    # Phi_global is stored as fraction but plotted as %, so scale should be 100
    assert scale == pytest.approx(100.0, rel=1e-5)


@pytest.mark.unit
def test_get_scale_unknown_returns_one():
    """Unknown quantities should fall back to scale factor of 1.0."""
    assert get_scale('not_a_real_quantity') == pytest.approx(1.0, rel=1e-5)


# ---------------------------------------------------------
# get_log_scale
# ---------------------------------------------------------


@pytest.mark.unit
def test_get_log_scale_known_log_quantity():
    """esc_rate_total should use log scale."""
    # esc_rate_total is in preset_log_scales as True (escape rates span many orders)
    from proteus.utils.plot import _preset_log_scales

    if 'esc_rate_total' in _preset_log_scales:
        assert get_log_scale('esc_rate_total') == _preset_log_scales['esc_rate_total']


@pytest.mark.unit
def test_get_log_scale_unknown_returns_false():
    """Unknown quantities default to linear scale (False)."""
    assert get_log_scale('some_unknown_output') is False


# ---------------------------------------------------------
# latex
# ---------------------------------------------------------


@pytest.mark.unit
def test_latex_wraps_backslash_label():
    """Labels containing a backslash should be wrapped in dollar signs."""
    assert latex('\\rm surf') == '$\\rm surf$'


@pytest.mark.unit
def test_latex_plain_label_unchanged():
    """Labels without backslash should not be wrapped."""
    assert latex('T_surf') == 'T_surf'


@pytest.mark.unit
def test_latex_empty_string():
    """Empty string has no backslash, so it should be returned as-is."""
    assert latex('') == ''


# ---------------------------------------------------------
# clean_series
# ---------------------------------------------------------


@pytest.mark.unit
def test_clean_series_removes_nan():
    """NaN values should be dropped from the series."""
    s = pd.Series([1.0, float('nan'), 3.0])
    result = clean_series(s, log_scale=False)
    assert len(result) == 2
    assert not result.isna().any()


@pytest.mark.unit
def test_clean_series_removes_inf():
    """Infinite values should be replaced by NaN and then dropped."""
    s = pd.Series([1.0, float('inf'), -float('inf'), 2.0])
    result = clean_series(s, log_scale=False)
    assert len(result) == 2


@pytest.mark.unit
def test_clean_series_log_scale_removes_nonpositive():
    """With log_scale=True, zero and negative values should be dropped."""
    s = pd.Series([-1.0, 0.0, 0.5, 2.0])
    result = clean_series(s, log_scale=True)
    assert (result > 0).all()
    assert len(result) == 2


@pytest.mark.unit
def test_clean_series_linear_keeps_zeros():
    """With log_scale=False, zero values should be retained."""
    s = pd.Series([0.0, 1.0, 2.0])
    result = clean_series(s, log_scale=False)
    assert 0.0 in result.values
    assert len(result) == 3


# ---------------------------------------------------------
# validate_output_variables
# ---------------------------------------------------------


@pytest.mark.unit
def test_validate_output_variables_all_present():
    """All requested outputs that exist in the DataFrame should be returned."""
    df = pd.DataFrame({'Time': [1], 'T_surf': [300], 'P_surf': [1e5]})
    valid = validate_output_variables(df, ['T_surf', 'P_surf'])
    assert valid == ['T_surf', 'P_surf']


@pytest.mark.unit
def test_validate_output_variables_missing_excluded(capsys):
    """Missing output variables should be excluded and a warning printed."""
    df = pd.DataFrame({'Time': [1], 'T_surf': [300]})
    valid = validate_output_variables(df, ['T_surf', 'not_a_column'])
    assert valid == ['T_surf']
    captured = capsys.readouterr()
    assert 'WARNING' in captured.out or 'not_a_column' in captured.out


@pytest.mark.unit
def test_validate_output_variables_empty_request():
    """Empty request list should return empty list."""
    df = pd.DataFrame({'Time': [1], 'T_surf': [300]})
    valid = validate_output_variables(df, [])
    assert valid == []


# ---------------------------------------------------------
# flatten_input_parameters
# ---------------------------------------------------------


@pytest.mark.unit
def test_flatten_input_parameters_simple():
    """Flat dict with leaf blocks should be returned unchanged."""
    d = {'escape': {'efficiency': {'label': 'eff', 'log': True}}}
    result = flatten_input_parameters(d)
    assert 'escape.efficiency' in result
    assert result['escape.efficiency']['label'] == 'eff'


@pytest.mark.unit
def test_flatten_input_parameters_skips_colormap():
    """The 'colormap' key should be skipped at any nesting level."""
    d = {'colormap': 'viridis', 'orbit': {'sma': {'label': 'a'}}}
    result = flatten_input_parameters(d)
    assert 'colormap' not in result
    assert 'orbit.sma' in result


@pytest.mark.unit
def test_flatten_input_parameters_deeply_nested():
    """Multi-level nesting should produce correct dot-separated keys."""
    d = {'a': {'b': {'c': {'label': 'deep'}}}}
    result = flatten_input_parameters(d)
    assert 'a.b.c' in result


# ---------------------------------------------------------
# load_ecdf_plot_settings
# ---------------------------------------------------------


@pytest.mark.unit
def test_load_ecdf_plot_settings_basic():
    """Should return three items: param_settings, output_settings, plot_format."""
    cfg = {
        'colormap': 'viridis',
        'output_variables': ['T_surf', 'P_surf'],
        'plot_format': 'png',
    }
    tested_params = {'struct.mass_tot': [0.7, 1.0]}
    param_settings, output_settings, plot_format = load_ecdf_plot_settings(cfg, tested_params)

    assert 'struct.mass_tot' in param_settings
    assert 'T_surf' in output_settings
    assert 'P_surf' in output_settings
    assert plot_format == 'png'


@pytest.mark.unit
def test_load_ecdf_plot_settings_respects_colormap():
    """The colormap from config should be applied to param_settings."""
    import matplotlib.cm as cm

    cfg = {
        'colormap': 'plasma',
        'output_variables': ['T_surf'],
        'plot_format': 'pdf',
    }
    tested_params = {'orbit.semimajoraxis': [0.1, 1.0]}
    param_settings, _, _ = load_ecdf_plot_settings(cfg, tested_params)
    # The colormap object stored should correspond to 'plasma'
    assert param_settings['orbit.semimajoraxis']['colormap'] is cm.plasma


@pytest.mark.unit
def test_load_ecdf_plot_settings_raises_for_empty_params():
    """Should raise ValueError when no tested parameters are provided."""
    cfg = {'colormap': 'viridis', 'output_variables': ['T_surf'], 'plot_format': 'png'}
    with pytest.raises(ValueError, match='No tested parameters'):
        load_ecdf_plot_settings(cfg, {})


# ---------------------------------------------------------
# group_output_by_parameter
# ---------------------------------------------------------


@pytest.mark.unit
def test_group_output_by_parameter_basic():
    """Should group output values correctly by input parameter."""
    df = pd.DataFrame(
        {
            'struct.mass_tot': [0.7, 0.7, 1.0, 1.0],
            'T_surf': [500.0, 600.0, 700.0, 800.0],
        }
    )
    result = group_output_by_parameter(df, ['struct.mass_tot'], ['T_surf'])
    key = 'T_surf_per_struct.mass_tot'
    assert key in result
    assert 0.7 in result[key]
    assert 1.0 in result[key]
    assert len(result[key][0.7]) == 2


@pytest.mark.unit
def test_group_output_by_parameter_applies_scale():
    """Scale factor from _preset_scales should be applied to output values."""
    # Phi_global has a scale of 100 (fraction → percentage)
    df = pd.DataFrame(
        {
            'struct.corefrac': [0.35, 0.35],
            'Phi_global': [0.5, 0.8],
        }
    )
    result = group_output_by_parameter(df, ['struct.corefrac'], ['Phi_global'])
    key = 'Phi_global_per_struct.corefrac'
    values = list(result[key][0.35])
    # Values should be scaled by 100
    assert pytest.approx(values, rel=1e-5) == [50.0, 80.0]
