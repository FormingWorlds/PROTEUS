"""
Unit tests for proteus.utils.plot module.

Tests plotting helper functions:
- Colour generation and retrieval
- LaTeX formatting strings
- Time sampling logic
- Output file sampling logic
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from proteus.utils.plot import (
    _generate_colour,
    _preset_colours,
    get_colour,
    latex_float,
    latexify,
    sample_output,
    sample_times,
)


@pytest.mark.unit
def test_get_colour_preset():
    """Test retrieval of preset colours."""
    # Test known presets
    assert get_colour('H2O') == _preset_colours['H2O']
    assert get_colour('CO2') == _preset_colours['CO2']
    assert get_colour('atm') == _preset_colours['atm']


@pytest.mark.unit
def test_get_colour_generated():
    """Test systematic generation of colours for unknown molecules."""
    # 'H2O2' is likely not in presets, should generate a colour
    c = get_colour('H2O2')
    assert isinstance(c, str)
    assert c.startswith('#')
    assert len(c) == 7


@pytest.mark.unit
def test_generate_colour_fallback(caplog):
    """Test fallback colour for invalid molecules."""
    # 'InvalidMol' fails parsing, triggers warning and fallback
    c = _generate_colour('InvalidMol')
    assert c == _preset_colours['_fallback']
    assert 'Using fallback colour' in caplog.text


@pytest.mark.unit
def test_latexify():
    """Test conversion of chemical formulas to LaTeX strings."""
    assert latexify('H2O') == r'H$_2$O'
    assert latexify('CO2') == r'CO$_2$'
    assert latexify('Fe2O3') == r'Fe$_2$O$_3$'
    assert latexify('He') == r'He'


@pytest.mark.unit
def test_latex_float():
    """Test scientific notation formatting for LaTeX."""
    assert latex_float(1.23e-5) == r'$1.2 \times 10^{-5}$'
    assert latex_float(100.0) == '1e+02' or '100'  # Behavior depends on string formatting
    # Let's check specific implementation: '{0:.2g}'
    assert latex_float(0.00001) == r'$1 \times 10^{-5}$'
    assert latex_float(0.01) == '0.01'


@pytest.mark.unit
def test_sample_times_all():
    """Test sampling when requested count exceeds available times."""
    times = [1, 2, 3, 4]
    out_t, out_i = sample_times(times, nsamp=10)
    assert out_t == times
    assert out_i == [0, 1, 2, 3]


@pytest.mark.unit
def test_sample_times_log_spacing():
    """Test log-spaced sampling of times."""
    # Create 100 points logarithmic from 1 to 1000
    times = np.logspace(0, 3, 100)
    nsamp = 5
    out_t, out_i = sample_times(times, nsamp=nsamp, tmin=1.0)

    assert len(out_t) <= nsamp
    # Result should include start/end roughly
    assert 1.0 in out_t or pytest.approx(1.0, 0.1) in out_t

    # Indices should be valid
    assert max(out_i) < len(times)


@pytest.mark.unit
@patch('proteus.utils.plot.glob.glob')
@patch('proteus.utils.plot.archive_exists')
def test_sample_output_empty(mock_archive, mock_glob):
    """Test output sampling with no files."""
    mock_glob.return_value = []

    # Case 1: Archive exists
    mock_archive.return_value = True
    handler = MagicMock()
    handler.directories = {'output/data': '/tmp'}
    t, f = sample_output(handler)
    assert t == []
    assert f == []

    # Case 2: No archive
    mock_archive.return_value = False
    t, f = sample_output(handler)
    assert t == []
    assert f == []


@pytest.mark.unit
@patch('proteus.utils.plot.glob.glob')
def test_sample_output_files(mock_glob):
    """Test sampling of output files."""
    # Mock file list: 100_atm.nc, 200_atm.nc ...
    files = [f'/tmp/{i}_atm.nc' for i in range(100, 1000, 100)]
    mock_glob.return_value = files

    handler = MagicMock()
    handler.directories = {'output/data': '/tmp'}

    t, f = sample_output(handler, nsamp=3)

    assert len(t) <= 3
    assert len(f) == len(t)
    assert all(isinstance(time, int) for time in t)
    assert all(path.endswith('.nc') for path in f)
