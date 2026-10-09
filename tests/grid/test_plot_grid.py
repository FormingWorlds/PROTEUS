"""Unit tests for ``proteus.grid.plot_grid``.

Covers the ECDF/status plotting helpers without a display:
- the empirical CDF rises monotonically from 0 to 100 percent
- clean_series drops non-positive values only on a log axis (and warns)
- numeric vs string parameter colouring picks the right keying
- the status and ECDF figures are written to disk in the requested format

Testing standards:
  - docs/How-to/testing.md
  - docs/Explanations/test_framework.md
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')  # headless backend for file-only tests

from proteus.grid.plot_grid import (  # noqa: E402
    WONG_PALETTE,
    _value_colours,
    clean_series,
    ecdf,
    plot_ecdf_grid,
    plot_status_summary,
)
from proteus.utils.plot import get_colourmap  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_ecdf_monotonic_from_zero_to_hundred_percent():
    """The ECDF y-values rise monotonically and reach 100 percent at the top.

    Uses distinct, unordered inputs so a sort bug or an off-by-one normalisation
    (dividing by n-1 or n+1) lands away from 100.
    """
    x, y = ecdf(np.array([30.0, 10.0, 20.0]))
    assert np.all(np.diff(y) >= 0)
    assert y[-1] == pytest.approx(100.0)
    assert y[0] == pytest.approx(0.0)
    # x is returned sorted (with a leading duplicate for the steps-post start).
    assert list(x) == [10.0, 10.0, 20.0, 30.0]


def test_ecdf_empty_input_returns_empty_arrays():
    """An empty series yields empty x and y rather than raising."""
    x, y = ecdf(np.array([]))
    assert len(x) == 0
    assert len(y) == 0


def test_clean_series_drops_nonpositive_only_on_log_axis():
    """On a log axis, non-positive and inf values are removed; on a linear axis
    zeros and negatives are kept (only NaN/inf dropped).
    """
    s = pd.Series([1.0, 0.0, -2.0, np.inf, 10.0])
    log_cleaned = clean_series(s, log_axis=True)
    lin_cleaned = clean_series(s, log_axis=False)
    assert sorted(log_cleaned.tolist()) == [1.0, 10.0]
    # Linear axis keeps 0 and -2, drops only the inf.
    assert sorted(lin_cleaned.tolist()) == [-2.0, 0.0, 1.0, 10.0]


def test_clean_series_warns_when_dropping_on_log_axis(caplog):
    """A log-axis drop of non-positive values is logged with the count dropped."""
    s = pd.Series([1.0, -1.0, -3.0])
    with caplog.at_level('WARNING'):
        cleaned = clean_series(s, log_axis=True)
    assert cleaned.tolist() == [1.0]
    assert any('non-positive' in rec.message for rec in caplog.records)


def test_value_colours_numeric_builds_norm_and_colourbar_flag():
    """Numeric parameter values are flagged numeric and get a continuous norm
    spanning their range, so a colourbar (not a legend) is used.
    """
    cmap = get_colourmap('viridis')
    colour_of, is_numeric, norm = _value_colours([0.5, 1.0, 2.0], log_norm=False, cmap=cmap)
    assert is_numeric
    assert norm is not None
    assert norm(0.5) == pytest.approx(0.0)
    assert norm(2.0) == pytest.approx(1.0)


def test_value_colours_string_uses_discrete_legend_mapping():
    """String parameter values are flagged non-numeric with no norm, and each
    distinct value maps to its own colour.
    """
    cmap = get_colourmap('viridis')
    colour_of, is_numeric, norm = _value_colours(['janus', 'agni'], log_norm=False, cmap=cmap)
    assert not is_numeric
    assert norm is None
    assert colour_of('agni') != colour_of('janus')


def test_plot_status_summary_writes_file(tmp_path):
    """The status summary figure is written with the requested extension, and a
    missing 'status' column is a clear error.
    """
    df = pd.DataFrame(
        {'status': ['Completed (solidified)', 'Error (died)', 'Completed (solidified)']}
    )
    out = plot_status_summary(df, tmp_path, 'demo', plot_format='png')
    assert out.exists()
    assert out.suffix == '.png'
    with pytest.raises(ValueError):
        plot_status_summary(pd.DataFrame({'x': [1]}), tmp_path, 'demo')


def test_plot_ecdf_grid_writes_file(tmp_path):
    """plot_ecdf_grid writes one figure covering the tested parameters and
    outputs present in the data.
    """
    df = pd.DataFrame(
        {
            'planet.mass_tot': [0.5, 0.5, 1.0, 1.0],
            'P_surf': [1.0, 2.0, 3.0, 4.0],
            'T_surf': [300.0, 400.0, 500.0, 600.0],
        }
    )
    tested = {'planet.mass_tot': [0.5, 1.0]}
    out = plot_ecdf_grid(
        df, tested, ['P_surf', 'T_surf'], tmp_path, 'demo', colourmap='viridis'
    )
    assert out.exists()
    assert out.name == 'ecdf_grid_plot.png'


def test_wong_palette_has_eight_distinct_colours():
    """The Wong palette used for status bars is the full eight colour-blind-safe set."""
    assert len(WONG_PALETTE) == 8
    assert len(set(WONG_PALETTE)) == 8
