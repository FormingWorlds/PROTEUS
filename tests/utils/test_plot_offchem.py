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

pytestmark = pytest.mark.unit


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


def test_offchem_read_year_raises_when_no_file(tmp_path):
    """Missing snapshot raises FileNotFoundError, not a vague pickle error."""
    # No file at all: empty offchem dir.
    os.makedirs(str(tmp_path / 'offchem'))
    with pytest.raises(FileNotFoundError, match='No VULCAN snapshot found'):
        offchem_read_year(str(tmp_path) + '/', 100)


def test_offchem_read_year_read_const_unsupported(tmp_path):
    """``read_const=True`` is the deprecated vulcan_cfg.py path."""
    # Provide a valid payload so the only thing that can fail is the
    # read_const guard itself.
    payload = _make_fake_vul_payload()
    _write_pickle(str(tmp_path / 'offchem' / 'vulcan_500.pkl'), payload)
    with pytest.raises(NotImplementedError, match='read_const'):
        offchem_read_year(str(tmp_path) + '/', 500, read_const=True)


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
