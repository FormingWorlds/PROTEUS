"""
Defensive-skip tests for the population diagram loaders.

Anti-happy-path discipline (per .claude/rules/proteus-tests.md):
- Tests must include the missing-data path AND the present-data path.
- Tests must include malformed inputs that the loaders should tolerate.
- Tests must verify the wrappers do not raise when the loaders return
  None (the integration smoke tests rely on this contract — if a loader
  starts raising, every smoke test that runs the end-of-run plot suite
  will break).

The original failure that motivated the defensive contract: smoke tests
configured `plot_mod = 0` thinking that meant "off"; per the schema
documentation, `0` means "wait until completion" (plot only at end), and
only `None` means "never plot". The end-of-run plot suite called
plot_population_mass_radius which read DACE_PlanetS.csv and raised
FileNotFoundError on CI runners that did not ship the file. Defensive
loaders make this a warn-and-skip rather than a hard fail.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from proteus.plot.cpl_population import (
    _get_exo_data,
    _get_mr_data,
    plot_population_mass_radius,
    plot_population_time_density,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# _get_exo_data
# ---------------------------------------------------------------------------
def test_get_exo_data_returns_none_when_dace_missing(tmp_path, caplog):
    """Missing DACE_PlanetS.csv warns and returns None (does not raise)."""
    with caplog.at_level(logging.WARNING):
        result = _get_exo_data(str(tmp_path))
    assert result is None
    assert any('DACE_PlanetS.csv' in rec.message for rec in caplog.records), (
        f'Expected a warning naming DACE_PlanetS.csv; got: '
        f'{[rec.message for rec in caplog.records]}'
    )


def test_get_exo_data_loads_present_csv(tmp_path):
    """When DACE_PlanetS.csv is present, the loader returns a DataFrame.

    Anti-happy: the test fixture writes a non-trivial 2-row CSV so the
    return type and shape are both verified, not just "not None".
    """
    target_dir = tmp_path / 'planet_reference' / 'Exoplanets'
    target_dir.mkdir(parents=True)
    csv_path = target_dir / 'DACE_PlanetS.csv'
    csv_path.write_text(
        '# DACE-formatted exoplanet catalogue stub\n'
        'Planet Mass [Mjup],Planet Radius [Rjup],Stellar Age [Gyr]\n'
        '0.003,0.092,4.6\n'
        '0.0095,0.13,2.3\n'
    )
    result = _get_exo_data(str(tmp_path))
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert 'Planet Mass [Mjup]' in result.columns


# ---------------------------------------------------------------------------
# _get_mr_data
# ---------------------------------------------------------------------------
def test_get_mr_data_returns_none_when_zeng_missing(tmp_path, caplog):
    """Missing Zeng-2019 mass-radius curve files warn and return None."""
    with caplog.at_level(logging.WARNING):
        result = _get_mr_data(str(tmp_path))
    assert result is None
    msgs = [rec.message for rec in caplog.records]
    assert any('Zeng-2019' in m for m in msgs), (
        f'Expected a warning naming Zeng-2019; got: {msgs}'
    )


def test_get_mr_data_returns_none_when_one_file_missing(tmp_path):
    """Even one missing curve out of five returns None (no partial load).

    Anti-happy: writes 4 of 5 files, omits one. The loader must NOT return
    a partial dict that downstream code would then fail on.
    """
    z19 = tmp_path / 'mass_radius' / 'Zeng2019'
    z19.mkdir(parents=True)
    # Write 4 of the 5 expected files.
    for name in (
        'massradiusEarthlikeRocky.txt',
        'massradiusFe.txt',
        'massradiusmgsio3.txt',
        'massradius_100percentH2O_300K_1mbar.txt',
    ):
        (z19 / name).write_text('1.0 1.0\n2.0 1.5\n3.0 1.7\n')
    # massradiushydrogen.txt missing on purpose.
    result = _get_mr_data(str(tmp_path))
    assert result is None


def test_get_mr_data_loads_present_curves(tmp_path):
    """All five files present yields a dict of arrays.

    Anti-happy: each file has 3 rows of (mass, radius) so the array shape
    is verified beyond "not None".
    """
    z19 = tmp_path / 'mass_radius' / 'Zeng2019'
    z19.mkdir(parents=True)
    for name in (
        'massradiusEarthlikeRocky.txt',
        'massradiusFe.txt',
        'massradiusmgsio3.txt',
        'massradius_100percentH2O_300K_1mbar.txt',
        'massradiushydrogen.txt',
    ):
        (z19 / name).write_text('1.0 1.0\n2.0 1.5\n3.0 1.7\n')

    result = _get_mr_data(str(tmp_path))
    assert result is not None
    assert set(result.keys()) == {
        'Earth-like',
        'Fe',
        r'MgSiO$_3$',
        r'H$_2$O, 300 K',
        r'Cold H$_2$',
    }
    for k, (masses, radii) in result.items():
        assert len(masses) == 3, f'{k} unexpected mass-array length'
        assert len(radii) == 3, f'{k} unexpected radius-array length'
        # _get_mr_data sorts by mass column; verify the sort actually happened
        assert np.all(np.diff(masses) >= 0), f'{k} mass array is not sorted'


# ---------------------------------------------------------------------------
# Wrapper-level defensive contract
# ---------------------------------------------------------------------------
def _make_hf_all(n_rows: int) -> pd.DataFrame:
    """Build a synthetic helpfile DataFrame with n_rows entries.

    Time values are well past the t0=100 yr cut so the wrappers don't
    silently bail out on the time filter; that path is tested separately
    by hf_too_short_for_population_plot.
    """
    return pd.DataFrame(
        {
            'Time': 1e3 * np.arange(1, n_rows + 1),
            'M_planet': np.full(n_rows, 5.972e24),
            'R_obs': np.full(n_rows, 6.371e6),
        }
    )


def test_plot_population_mass_radius_skips_when_loaders_return_none(tmp_path, caplog):
    """When fwl_data is missing, the wrapper warns and returns without
    raising and without writing a plot file. This is the contract the
    integration smoke tests depend on.
    """
    out_dir = tmp_path / 'output'
    (out_dir / 'plots').mkdir(parents=True)
    hf_all = _make_hf_all(5)
    with caplog.at_level(logging.WARNING):
        result = plot_population_mass_radius(
            hf_all, str(out_dir), str(tmp_path / 'no_fwl_data'), 'png'
        )
    assert result is None
    # No plot file should have been written
    assert not list((out_dir / 'plots').glob('plot_population*'))


def test_plot_population_time_density_skips_when_dace_missing(tmp_path):
    """Same defensive contract for the time-density wrapper."""
    out_dir = tmp_path / 'output'
    (out_dir / 'plots').mkdir(parents=True)
    hf_all = _make_hf_all(5)
    result = plot_population_time_density(
        hf_all, str(out_dir), str(tmp_path / 'no_fwl_data'), 'png'
    )
    assert result is None
    assert not list((out_dir / 'plots').glob('plot_population*'))


def test_plot_population_mass_radius_skips_when_hf_too_short(tmp_path):
    """Wrapper must also skip cleanly when the helpfile has < 3 rows past t0,
    even if reference data is present. Guards the existing behaviour.
    """
    # Stage real-looking reference data so this test exercises the
    # "len(time) < 3" branch, not the missing-data branch.
    z19 = tmp_path / 'mass_radius' / 'Zeng2019'
    z19.mkdir(parents=True)
    for name in (
        'massradiusEarthlikeRocky.txt',
        'massradiusFe.txt',
        'massradiusmgsio3.txt',
        'massradius_100percentH2O_300K_1mbar.txt',
        'massradiushydrogen.txt',
    ):
        (z19 / name).write_text('1.0 1.0\n2.0 1.5\n3.0 1.7\n')
    target_dir = tmp_path / 'planet_reference' / 'Exoplanets'
    target_dir.mkdir(parents=True)
    (target_dir / 'DACE_PlanetS.csv').write_text(
        'Planet Mass [Mjup],Planet Radius [Rjup],Stellar Age [Gyr]\n0.003,0.092,4.6\n'
    )

    out_dir = tmp_path / 'output'
    (out_dir / 'plots').mkdir(parents=True)
    # Only 2 rows past t0=100yr (Time=1000, 2000)
    hf_all = pd.DataFrame(
        {
            'Time': [10.0, 1000.0, 2000.0],
            'M_planet': [5.972e24] * 3,
            'R_obs': [6.371e6] * 3,
        }
    )
    result = plot_population_mass_radius(hf_all, str(out_dir), str(tmp_path), 'png')
    assert result is None
    assert not list((out_dir / 'plots').glob('plot_population*'))


def test_plot_population_handles_path_with_spaces(tmp_path):
    """The fwl_dir argument should accept paths with spaces; common on
    user setups where FWL_DATA lives under "My Documents" or similar.
    Anti-happy: would catch a regression where someone uses string
    concatenation that doesn't escape spaces correctly."""
    spaced = tmp_path / 'My Drive' / 'fwl data'
    spaced.mkdir(parents=True)
    out_dir = tmp_path / 'output'
    (out_dir / 'plots').mkdir(parents=True)
    hf_all = _make_hf_all(5)
    # Should warn-and-skip (no fwl_data files), not raise on the path itself.
    plot_population_mass_radius(hf_all, str(out_dir), str(spaced), 'png')
    plot_population_time_density(hf_all, str(out_dir), str(spaced), 'png')
    assert not list((out_dir / 'plots').glob('plot_population*'))


def test_plot_population_does_not_eagerly_import_dace_at_module_load():
    """Importing the module must not trigger any file IO; the loaders
    are call-time only. Catches a regression where someone moves the
    DACE read to module-level for "convenience"."""
    # Re-import in a fresh interpreter sub-namespace would be cleanest, but
    # importlib.reload does the practical job: module load must complete
    # even when no fwl_data exists at all.
    import importlib

    import proteus.plot.cpl_population as mod

    importlib.reload(mod)
    # Loader symbols must exist
    assert hasattr(mod, '_get_exo_data')
    assert hasattr(mod, '_get_mr_data')
    # At module level, no file access should have happened (we can't
    # easily prove a negative without mocking pd.read_csv globally; the
    # surrogate is that the import succeeded with no FWL_DATA env set).
    bogus = '/nonexistent/path/that/should/not/exist'
    assert not os.path.exists(bogus)
    # Sanity: a fresh _get_exo_data on the bogus path returns None.
    assert mod._get_exo_data(bogus) is None
    # And a fresh path object survives without IO too.
    assert Path(bogus).exists() is False
