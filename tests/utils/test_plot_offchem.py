"""Tests for proteus.utils.plot_offchem readers.

These regressions cover the flat ``offchem/vulcan*.pkl`` layout that
replaced the historic ``offchem/<year>/output.vul`` tree. Both online
(``vulcan_<year>.pkl`` per snapshot) and offline (single
``vulcan.pkl``) modes must be supported, and the read-time
``read_const`` path must fail loudly because the per-year
``vulcan_cfg.py`` dump is no longer produced.
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from proteus.utils.plot_offchem import offchem_read_year

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _make_fake_vul_payload():
    """Return a minimal pickle-ready VULCAN snapshot.

    Two species at three layers; pressures and temperatures
    distinguishable from any default values so a wrong-array bug would
    show up immediately.
    """
    return {
        'variable': {
            'species': ['H2O', 'CO2'],
            'ymix': np.array(
                [
                    [0.40, 0.10],
                    [0.30, 0.20],
                    [0.20, 0.30],
                ]
            ),
        },
        'atm': {
            # pco is in barye (CGS) per VULCAN convention; reader divides
            # by 1e6 to convert to bar.
            'pco': np.array([1.0e8, 1.0e6, 1.0e4]),
            'Tco': np.array([1850.0, 950.0, 300.0]),
        },
    }


def _write_pickle(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(payload, f)


def test_offchem_read_year_online_per_snapshot_file(tmp_path):
    """Online layout: ``offchem/vulcan_<year>.pkl`` resolved by year."""
    payload = _make_fake_vul_payload()
    _write_pickle(str(tmp_path / 'offchem' / 'vulcan_1000.pkl'), payload)

    out = offchem_read_year(str(tmp_path) + '/', 1000)

    assert out['year'] == 1000
    # Species columns: pull col index 0 (H2O) and col index 1 (CO2).
    # If the reader swapped axes, mx_H2O would be 0.10 not 0.40 at layer 0.
    np.testing.assert_allclose(out['mx_H2O'], [0.40, 0.30, 0.20])
    np.testing.assert_allclose(out['mx_CO2'], [0.10, 0.20, 0.30])
    # 1e8 barye = 1e2 bar. Catches a wrong /1e5 or /1e7 divisor.
    np.testing.assert_allclose(out['pressure'], [100.0, 1.0, 0.01])
    # Distinct values at all layers so an off-by-one indexing bug would
    # be visible.
    np.testing.assert_allclose(out['temperature'], [1850.0, 950.0, 300.0])


def test_offchem_read_year_offline_falls_back_to_fixed_name(tmp_path):
    """Offline layout: single ``offchem/vulcan.pkl`` shared across years."""
    payload = _make_fake_vul_payload()
    _write_pickle(str(tmp_path / 'offchem' / 'vulcan.pkl'), payload)

    # Year arg is irrelevant for the offline payload, but the function
    # still tags the snapshot with whatever year the caller asked for.
    out = offchem_read_year(str(tmp_path) + '/', 42)

    assert out['year'] == 42
    # No vulcan_42.pkl exists, so the reader must hit the fallback.
    np.testing.assert_allclose(out['pressure'], [100.0, 1.0, 0.01])


def test_offchem_read_year_prefers_per_snapshot_over_offline(tmp_path):
    """When both files exist, the per-year file wins."""
    payload_online = _make_fake_vul_payload()
    payload_offline = _make_fake_vul_payload()
    # Distinguish offline payload by a wildly different temperature so
    # the assertion is unambiguous about which file was read.
    payload_offline['atm']['Tco'] = np.array([99.0, 99.0, 99.0])
    _write_pickle(str(tmp_path / 'offchem' / 'vulcan_2000.pkl'), payload_online)
    _write_pickle(str(tmp_path / 'offchem' / 'vulcan.pkl'), payload_offline)

    out = offchem_read_year(str(tmp_path) + '/', 2000)
    # Must be the online payload's temperatures, not the offline 99 K.
    np.testing.assert_allclose(out['temperature'], [1850.0, 950.0, 300.0])
    # Discrimination: pin that the offline payload was NOT silently mixed
    # in. The offline file's distinctive 99 K marker must not appear in
    # the returned snapshot.
    assert 99.0 not in out['temperature']


def test_offchem_read_year_raises_when_no_file(tmp_path):
    """Missing snapshot raises FileNotFoundError, not a vague pickle error."""
    # No file at all: empty offchem dir.
    os.makedirs(str(tmp_path / 'offchem'))
    with pytest.raises(FileNotFoundError, match='No VULCAN snapshot found'):
        offchem_read_year(str(tmp_path) + '/', 100)
    # Discrimination: confirm the offchem directory genuinely is empty, so
    # the FileNotFoundError above can only have come from the missing-
    # snapshot guard, not from a different I/O on another path.
    assert not list((tmp_path / 'offchem').iterdir())


def test_offchem_read_year_read_const_unsupported(tmp_path):
    """``read_const=True`` is the deprecated vulcan_cfg.py path."""
    # Provide a valid payload so the only thing that can fail is the
    # read_const guard itself.
    payload = _make_fake_vul_payload()
    _write_pickle(str(tmp_path / 'offchem' / 'vulcan_500.pkl'), payload)
    with pytest.raises(NotImplementedError, match='read_const'):
        offchem_read_year(str(tmp_path) + '/', 500, read_const=True)
    # Discrimination: the same call with read_const=False on the same
    # payload must succeed. Without this counter-case, a regression that
    # raised NotImplementedError unconditionally would still pass the
    # check above.
    out = offchem_read_year(str(tmp_path) + '/', 500, read_const=False)
    assert out['year'] == 500


def test_offchem_read_year_clip_applied_to_extremes(tmp_path):
    """Mixing-ratio clip must squash values outside the requested band."""
    payload = _make_fake_vul_payload()
    # Inject a 0.0 value (below default 1e-30 floor) and a 1.5 value
    # (above default 1.0 ceiling). These are physically impossible
    # mixing ratios; the clip must catch both.
    payload['variable']['ymix'] = np.array(
        [
            [0.0, 1.5],
            [1.0e-40, 2.0],
            [0.5, 0.5],
        ]
    )
    _write_pickle(str(tmp_path / 'offchem' / 'vulcan_300.pkl'), payload)

    out = offchem_read_year(
        str(tmp_path) + '/',
        300,
        mx_clip_min=1.0e-30,
        mx_clip_max=1.0,
    )
    # Floor: 0.0 and 1e-40 both clipped up to 1e-30.
    np.testing.assert_allclose(out['mx_H2O'][:2], [1.0e-30, 1.0e-30])
    # Ceiling: 1.5 and 2.0 clipped down to 1.0.
    np.testing.assert_allclose(out['mx_CO2'][:2], [1.0, 1.0])
    # In-band value preserved.
    assert out['mx_H2O'][2] == pytest.approx(0.5)
    assert out['mx_CO2'][2] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# offchem_read_grid and offchem_slice_grid: grid-level aggregators.
# ---------------------------------------------------------------------------


def _build_grid_case(case_dir, case_options, snapshot_years):
    """Lay down a single case_NN folder with a TOML config and one
    vulcan_<year>.pkl per snapshot year. The pickles use the minimal
    payload from _make_fake_vul_payload so the reader produces a
    consistent dict shape per year.
    """
    os.makedirs(case_dir, exist_ok=True)
    offchem_dir = os.path.join(case_dir, 'offchem')
    os.makedirs(offchem_dir, exist_ok=True)
    for y in snapshot_years:
        _write_pickle(os.path.join(offchem_dir, f'vulcan_{y}.pkl'), _make_fake_vul_payload())
    # Minimal init_coupler.toml that read_config can parse and whose
    # keys offchem_slice_grid will be able to filter on.
    toml_text = (
        '\n'.join(['[params.case]'] + [f'{k} = {v!r}' for k, v in case_options.items()]) + '\n'
    )
    with open(os.path.join(case_dir, 'init_coupler.toml'), 'w') as fh:
        fh.write(toml_text)


def test_offchem_read_grid_raises_when_folder_is_empty(tmp_path):
    """offchem_read_grid raises an Exception when no case_* subfolders
    are present in the target directory.

    Edge: limit-input case for a freshly-scaffolded grid that has not
    yet produced any case output. Discriminating: pin the exception
    AND the exception message so a regression that returned empty
    arrays silently would fail the pytest.raises block.
    """
    from proteus.utils.plot_offchem import offchem_read_grid

    with pytest.raises(Exception, match='no grid points were found'):
        offchem_read_grid(str(tmp_path))
    assert tmp_path.is_dir()  # directory not deleted by the function


def test_offchem_read_grid_aggregates_year_data_across_cases(tmp_path, monkeypatch):
    """A grid with two case folders, each carrying two snapshot years,
    must produce 2x2 years/data arrays and a length-2 opts array.

    Discriminating: pin the shape AND the per-year content. A
    regression that mis-ordered the years (sorted vs unsorted) or
    that mixed cases would fail the per-case year list comparison.
    """
    import numpy as np

    from proteus.utils.plot_offchem import offchem_read_grid

    # Two cases with the same two snapshot years.
    _build_grid_case(
        str(tmp_path / 'case_00'),
        case_options={'name': 'alpha'},
        snapshot_years=[100, 50],  # intentionally unsorted on disk
    )
    _build_grid_case(
        str(tmp_path / 'case_01'),
        case_options={'name': 'beta'},
        snapshot_years=[200, 25],
    )
    # offchem_read_grid calls read_config which expects the full
    # PROTEUS config structure. Stub it to return our minimal dict
    # tree so the test does not depend on a full Config schema.

    def _fake_read_config(path):
        # Return a (config, _) tuple where the first element is a
        # plain dict (so dict(options[0]) works in the source).
        if 'case_00' in path:
            return ({'name': 'alpha'}, None)
        return ({'name': 'beta'}, None)

    monkeypatch.setattr('proteus.utils.plot_offchem.read_config', _fake_read_config)

    years, opts, data = offchem_read_grid(str(tmp_path))
    assert years.shape == (2, 2)
    assert opts.shape == (2,)
    assert data.shape == (2, 2)
    # Years must be sorted per case despite the unsorted on-disk order.
    np.testing.assert_array_equal(np.sort(years, axis=1), years)
    # Discrimination: both case names must be present in opts (the
    # glob ordering is not guaranteed, so we do not pin per-index).
    # A regression that read the same case twice, or that dropped a
    # case silently, would fail the set equality below.
    names = {row['name'] for row in opts}
    assert names == {'alpha', 'beta'}
    # Per-case year sum: the case named 'alpha' has years [50, 100]
    # (sum 150) and 'beta' has [25, 200] (sum 225). Pair each opts
    # row with its years row and verify the per-case totals match.
    # This catches a regression that joined the wrong case folder to
    # the wrong opts dict.
    expected_per_case = {'alpha': 50 + 100, 'beta': 25 + 200}
    for i, row in enumerate(opts):
        assert int(years[i].sum()) == expected_per_case[row['name']]


def test_offchem_slice_grid_keeps_only_matching_grid_points():
    """offchem_slice_grid returns a subset of the grid where every
    point's options dict matches every key/value in cvar_filter.

    Discriminating: a regression that flipped the match logic (kept
    non-matching points or excluded matching ones) would fail the
    cardinality assertion AND the per-row name check.
    """
    import numpy as np

    from proteus.utils.plot_offchem import offchem_slice_grid

    years = np.array([[100, 200], [150, 250], [300, 400]], dtype=int)
    opts = np.array([{'name': 'alpha'}, {'name': 'beta'}, {'name': 'alpha'}], dtype=dict)
    data = np.array(
        [
            [{'mx_H2O': 0.1}, {'mx_H2O': 0.2}],
            [{'mx_H2O': 0.5}, {'mx_H2O': 0.6}],
            [{'mx_H2O': 0.7}, {'mx_H2O': 0.8}],
        ],
        dtype=dict,
    )
    s_years, s_opts, s_data = offchem_slice_grid(
        years, opts, data, cvar_filter={'name': 'alpha'}
    )
    assert s_opts.shape == (2,)
    assert s_years.shape == (2, 2)
    assert s_data.shape == (2, 2)
    for row in s_opts:
        assert row['name'] == 'alpha'
    # Discrimination: a regression returning the inverse subset
    # (beta only, 1 row) would fail the shape check above.


def test_offchem_slice_grid_warns_when_filter_excludes_every_point(capsys):
    """An overly-strict filter that excludes every grid point must
    print a warning and return empty arrays.

    Edge: limit-input case. Discriminating: pin both the warning
    appearance in stdout and the zero-length result.
    """
    import numpy as np

    from proteus.utils.plot_offchem import offchem_slice_grid

    years = np.array([[100], [200]], dtype=int)
    opts = np.array([{'name': 'alpha'}, {'name': 'beta'}], dtype=dict)
    data = np.array([[{'x': 1}], [{'x': 2}]], dtype=dict)
    s_years, s_opts, s_data = offchem_slice_grid(
        years, opts, data, cvar_filter={'name': 'no-such-value'}
    )
    captured = capsys.readouterr()
    assert 'No grid points left after slicing' in captured.out
    assert s_opts.shape == (0,)


def test_offchem_slice_grid_warns_on_filter_key_not_in_options(capsys):
    """A filter key absent from a grid point's options must trigger
    a warning. The point itself is then included (the source's
    `continue` skips the exclusion check for unknown keys).

    Discriminating: pin both the warning content (the missing key
    name) AND the inclusion behaviour. A regression that excluded on
    missing keys would land at zero matches.
    """
    import numpy as np

    from proteus.utils.plot_offchem import offchem_slice_grid

    years = np.array([[100]], dtype=int)
    opts = np.array([{'name': 'alpha'}], dtype=dict)
    data = np.array([[{'x': 1}]], dtype=dict)
    s_years, s_opts, s_data = offchem_slice_grid(
        years, opts, data, cvar_filter={'unknown_key': 'whatever'}
    )
    captured = capsys.readouterr()
    assert "'unknown_key'" in captured.out
    assert 'is not present in OPTIONS' in captured.out
    # The single grid point survives because the unknown filter key
    # is treated as a no-op exclusion check.
    assert s_opts.shape == (1,)
