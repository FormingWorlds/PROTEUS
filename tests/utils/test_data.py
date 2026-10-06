"""
Unit tests for proteus.utils.data module.

This module validates the data management utilities, ensuring reliable access to
external physics data (spectral files, lookup tables). It covers:
- fwl-io dataset fetches (with mocking to prevent real network calls).
- Configuration mapping for remote resources.

See also:
- docs/How-to/testing.md
- docs/Explanations/test_framework.md
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from fwl_io import DownloadError

from proteus.utils.data import (
    GetFWLData,
    download_spectral_file,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
@patch('proteus.utils.data.FWL_DATA_DIR', __file__)
def test_GetFWLData_returns_absolute_path():
    """
    GetFWLData returns an absolute path to the FWL data directory.

    Physical scenario: callers need a single canonical path for data;
    absolute path avoids ambiguity with cwd.
    """
    result = GetFWLData()
    assert result.is_absolute()
    assert 'test_data' in str(result) or 'utils' in str(result)


@pytest.mark.unit
def test_download_spectral_file_errors():
    """Test input validation for spectral file download helper."""
    with pytest.raises(Exception, match='Must provide name'):
        download_spectral_file('', '256')

    with pytest.raises(Exception, match='Must provide number of bands'):
        download_spectral_file('Dayspring', '')


@pytest.mark.unit
@patch('proteus.utils.data.download_spectral_file')
def test_download_spectral_files_dispatch(mock_single):
    """Group/band dispatch of the plural spectral downloader.

    The bare call must fetch every registered folder, a group-only call
    every band count of that group, and a full selection exactly one
    file; band-only and unknown-group selections are rejected.
    """
    from proteus.utils.data import SPECTRAL_FILE_FOLDERS, download_spectral_files

    # Bare call: the whole registry, in registry order.
    download_spectral_files()
    assert mock_single.call_count == len(SPECTRAL_FILE_FOLDERS)
    # Registry sanity: more than one group, so the all-vs-group counts
    # below discriminate a broken filter from a working one.
    groups = {f.split('/')[0] for f in SPECTRAL_FILE_FOLDERS}
    assert len(groups) > 1

    # Group-only call: exactly the Dayspring band counts (4), not 1, not all.
    mock_single.reset_mock()
    download_spectral_files('Dayspring')
    dayspring = [f for f in SPECTRAL_FILE_FOLDERS if f.startswith('Dayspring/')]
    assert mock_single.call_count == len(dayspring)
    assert mock_single.call_count < len(SPECTRAL_FILE_FOLDERS)
    called_groups = {c.args[0] for c in mock_single.call_args_list}
    assert called_groups == {'Dayspring'}

    # Full selection: one file, args forwarded verbatim.
    mock_single.reset_mock()
    download_spectral_files('Honeyside', '4096')
    mock_single.assert_called_once_with('Honeyside', '4096')

    # Error contract: band-only selection is ambiguous.
    with pytest.raises(ValueError, match='band count alone'):
        download_spectral_files(None, '256')

    # Error contract: unknown group names the known ones.
    with pytest.raises(ValueError, match='Unknown spectral file group'):
        download_spectral_files('NoSuchGroup')

    # Error contract: known group with an unknown band count lists the
    # available band counts instead of deferring to a source-map error.
    with pytest.raises(ValueError, match='Unknown band count'):
        download_spectral_files('Dayspring', '999')

    # Error contract: unknown group with explicit bands is still a
    # group error, not a band error.
    with pytest.raises(ValueError, match='Unknown spectral file group'):
        download_spectral_files('NoSuchGroup', '48')


@pytest.mark.unit
def test_spectral_folder_registry_matches_manifest():
    """Every spectral folder is a distinct manifest dataset.

    Each SPECTRAL_FILE_FOLDERS entry must resolve to its own manifest table, so
    the bare `proteus get spectral` cannot fail on an entry with no source.
    """
    from proteus.data import _dataset, spectral_file_key
    from proteus.utils.data import SPECTRAL_FILE_FOLDERS

    keys = [spectral_file_key(*folder.split('/')) for folder in SPECTRAL_FILE_FOLDERS]
    assert len(set(keys)) == len(keys)
    for key in keys:
        assert _dataset(key).zenodo.startswith('10.5281/zenodo.')


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_spectral_file_call(mock_fetch):
    """
    Test spectral file download wrapper.

    Ensures that the convenience function fetches the manifest dataset for the
    requested group and band count, and rejects an unlisted pair.
    """
    download_spectral_file('Oak', '318')

    mock_fetch.assert_called_once_with('atmos_clim.spectral_files.oak.318')

    with pytest.raises(ValueError, match='No data source mapping found for folder: Oak/16'):
        download_spectral_file('Oak', '16')
    mock_fetch.assert_called_once()


def _phoenix_zip(
    path, lte_name='LTE_T05800_logg4.50_FeH-0.0_alpha+0.0_phoenixMedRes_R05000.txt'
):
    """Write a small zip that holds one PHOENIX spectrum file."""
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as zf:
        if lte_name:
            zf.writestr(lte_name, 'dummy spectrum')
        else:
            zf.writestr('README.txt', 'no spectra here')
    return path


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
def test_download_phoenix_fetches_one_zip_and_unpacks(mock_fetch_file, tmp_path, monkeypatch):
    """``download_phoenix`` asks fwl-io for the single zip of the requested
    grid, unpacks it into the versioned dataset directory and keeps the zip.
    """
    from proteus.data import STELLAR_SPECTRA_PHOENIX, dataset_dir
    from proteus.utils.data import download_phoenix

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    base_dir = dataset_dir(STELLAR_SPECTRA_PHOENIX, data_root=tmp_path)
    zip_name = 'FeH-0.0_alpha+0.0_phoenixMedRes_R05000.zip'
    zip_path = _phoenix_zip(base_dir / zip_name)
    mock_fetch_file.return_value = zip_path

    assert download_phoenix(alpha=0.0, FeH=0.0) is True

    mock_fetch_file.assert_called_once_with(
        STELLAR_SPECTRA_PHOENIX, zip_name, data_root=tmp_path
    )
    grid_dir = base_dir / 'FeH-0.0_alpha+0.0'
    assert any(grid_dir.glob('LTE_T*_phoenixMedRes_R05000.txt'))
    assert zip_path.is_file()
    assert not list(base_dir.glob('*.partial'))
    assert base_dir == tmp_path / 'star' / 'spectra' / 'phoenix' / 'r17674612'


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
def test_download_phoenix_skips_fetch_when_grid_is_unpacked(
    mock_fetch_file, tmp_path, monkeypatch
):
    """An unpacked grid is found on disk and nothing is fetched."""
    from proteus.data import STELLAR_SPECTRA_PHOENIX, dataset_dir
    from proteus.utils.data import download_phoenix

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    grid_dir = dataset_dir(STELLAR_SPECTRA_PHOENIX, data_root=tmp_path) / 'FeH-0.0_alpha+0.0'
    grid_dir.mkdir(parents=True)
    (grid_dir / 'LTE_T05800_logg4.50_FeH-0.0_alpha+0.0_phoenixMedRes_R05000.txt').write_text(
        'x'
    )

    assert download_phoenix(alpha=0.0, FeH=0.0) is True
    mock_fetch_file.assert_not_called()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
def test_download_phoenix_force_replaces_stale_grid(mock_fetch_file, tmp_path, monkeypatch):
    """``force=True`` removes the unpacked grid and the archive, fetches again,
    and leaves no file of the old grid behind.
    """
    from proteus.data import STELLAR_SPECTRA_PHOENIX, dataset_dir
    from proteus.utils.data import download_phoenix

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    base_dir = dataset_dir(STELLAR_SPECTRA_PHOENIX, data_root=tmp_path)
    zip_name = 'FeH-0.0_alpha+0.0_phoenixMedRes_R05000.zip'
    grid_dir = base_dir / 'FeH-0.0_alpha+0.0'
    grid_dir.mkdir(parents=True)
    stale = grid_dir / 'LTE_T99999_stale_phoenixMedRes_R05000.txt'
    stale.write_text('old')

    def fetch_again(key, name, data_root=None):
        assert not (base_dir / name).exists(), 'the old archive was not removed first'
        return _phoenix_zip(base_dir / name)

    mock_fetch_file.side_effect = fetch_again

    assert download_phoenix(alpha=0.0, FeH=0.0, force=True) is True
    mock_fetch_file.assert_called_once()
    assert not stale.exists()
    assert (
        grid_dir / 'LTE_T05800_logg4.50_FeH-0.0_alpha+0.0_phoenixMedRes_R05000.txt'
    ).is_file()
    assert (base_dir / zip_name).is_file()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
def test_download_phoenix_alpha_grid_uses_its_own_zip(mock_fetch_file, tmp_path, monkeypatch):
    """A grid with nonzero alpha and metallicity is a different zip, named with
    explicit signs, and the alpha=0 grid of the same metallicity is another one.
    """
    from proteus.data import STELLAR_SPECTRA_PHOENIX
    from proteus.utils.data import download_phoenix

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    mock_fetch_file.side_effect = lambda key, name, data_root=None: _phoenix_zip(
        tmp_path / name, lte_name=None
    )

    # The stub zips hold no spectra, so the call fails after the fetch; the
    # requested archive name is what this test pins.
    assert download_phoenix(alpha=0.2, FeH=-0.5) is False
    assert download_phoenix(alpha=0.0, FeH=-0.5) is False
    names = [c.args[1] for c in mock_fetch_file.call_args_list]
    assert names == [
        'FeH-0.5_alpha+0.2_phoenixMedRes_R05000.zip',
        'FeH-0.5_alpha+0.0_phoenixMedRes_R05000.zip',
    ]
    assert all(c.args[0] == STELLAR_SPECTRA_PHOENIX for c in mock_fetch_file.call_args_list)


@pytest.mark.unit
@pytest.mark.parametrize('error', [KeyError('no such file'), RuntimeError('mirror down')])
@patch('proteus.data.fetch_dataset_file')
def test_download_phoenix_returns_false_if_fetch_fails(
    mock_fetch_file, error, tmp_path, monkeypatch
):
    """A missing registry name or an exhausted mirror gives False, not a raise."""
    from proteus.utils.data import download_phoenix

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    mock_fetch_file.side_effect = error

    assert download_phoenix(alpha=0.0, FeH=0.0) is False
    mock_fetch_file.assert_called_once()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
def test_download_phoenix_rejects_archive_without_spectra(
    mock_fetch_file, tmp_path, monkeypatch
):
    """A zip that unpacks without any LTE file gives False and leaves no
    grid directory that a later call could mistake for a complete grid.
    """
    from proteus.data import STELLAR_SPECTRA_PHOENIX, dataset_dir
    from proteus.utils.data import download_phoenix

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    base_dir = dataset_dir(STELLAR_SPECTRA_PHOENIX, data_root=tmp_path)
    mock_fetch_file.return_value = _phoenix_zip(
        base_dir / 'FeH-0.0_alpha+0.0_phoenixMedRes_R05000.zip', lte_name=None
    )

    assert download_phoenix(alpha=0.0, FeH=0.0) is False
    assert not (base_dir / 'FeH-0.0_alpha+0.0').exists()
    assert not list(base_dir.glob('*.partial'))


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
def test_download_phoenix_rejects_corrupt_archive(mock_fetch_file, tmp_path, monkeypatch):
    """A file that is not a zip gives False and leaves no partial directory."""
    from proteus.data import STELLAR_SPECTRA_PHOENIX, dataset_dir
    from proteus.utils.data import download_phoenix

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    base_dir = dataset_dir(STELLAR_SPECTRA_PHOENIX, data_root=tmp_path)
    bad = base_dir / 'FeH-0.0_alpha+0.0_phoenixMedRes_R05000.zip'
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b'not a zip')
    mock_fetch_file.return_value = bad

    assert download_phoenix(alpha=0.0, FeH=0.0) is False
    assert not (base_dir / 'FeH-0.0_alpha+0.0').exists()
    assert not list(base_dir.glob('*.partial'))


@pytest.mark.unit
def test_phoenix_registry_covers_every_reachable_grid():
    """Every (FeH, alpha) pair that ``phoenix_to_grid`` can return names a zip
    in the dataset registry, so the runtime alpha override never asks for a
    file the record does not hold.
    """
    import logging

    import numpy as np

    from proteus.data import STELLAR_SPECTRA_PHOENIX, _dataset
    from proteus.utils.phoenix_helper import phoenix_param, phoenix_to_grid

    names = set(_dataset(STELLAR_SPECTRA_PHOENIX).registry())
    assert len(names) == 52

    logging.disable(logging.CRITICAL)
    try:
        pairs = set()
        for feh in np.arange(-4.5, 1.6, 0.1):
            for alpha in np.arange(-0.4, 1.5, 0.1):
                for teff in (None, 2500, 3000, 4000, 5800, 7500, 9000):
                    grid = phoenix_to_grid(
                        FeH=feh, alpha=alpha, Teff=teff, logg=4.5 if teff else None
                    )
                    pairs.add((grid['FeH'], grid['alpha']))
    finally:
        logging.disable(logging.NOTSET)

    assert len(pairs) > 40
    for feh, alpha in sorted(pairs):
        name = (
            f'FeH{phoenix_param(feh, kind="FeH")}_alpha{phoenix_param(alpha, kind="alpha")}'
            '_phoenixMedRes_R05000.zip'
        )
        assert name in names, name


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_muscles_default_download_all(mock_fetch):
    """If stars is None, download_muscles fetches the whole MUSCLES dataset."""
    from proteus.data import STELLAR_SPECTRA_MUSCLES
    from proteus.utils.data import download_muscles

    ok = download_muscles(stars=None, force=False)

    assert ok is True
    mock_fetch.assert_called_once_with(STELLAR_SPECTRA_MUSCLES)


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
def test_download_muscles_single_star(mock_fetch_file):
    """If stars is a string, download_muscles fetches that one registry file."""
    from proteus.data import STELLAR_SPECTRA_MUSCLES
    from proteus.utils.data import download_muscles

    ok = download_muscles(stars='trappist-1', force=False)

    assert ok is True
    mock_fetch_file.assert_called_once_with(STELLAR_SPECTRA_MUSCLES, 'trappist-1.txt')


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
def test_download_muscles_multiple_stars(mock_fetch_file):
    """A star absent from the registry fails the call but not the other stars."""
    from proteus.data import STELLAR_SPECTRA_MUSCLES
    from proteus.utils.data import download_muscles

    mock_fetch_file.side_effect = [Path('a'), KeyError('starB.txt')]

    ok = download_muscles(stars=['starA', 'starB'], force=False)

    assert ok is False
    assert mock_fetch_file.call_args_list == [
        call(STELLAR_SPECTRA_MUSCLES, 'starA.txt'),
        call(STELLAR_SPECTRA_MUSCLES, 'starB.txt'),
    ]


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_muscles_reports_fetch_failure(mock_fetch):
    """A fetch error is reported as a failed download, not raised."""
    from proteus.utils.data import download_muscles

    mock_fetch.side_effect = RuntimeError('mirror down')

    assert download_muscles(stars=None) is False
    mock_fetch.assert_called_once()


@pytest.mark.unit
def test_download_muscles_force_removes_the_requested_file_first(tmp_path, monkeypatch):
    """``force`` deletes the file before the fetch, so a present file is fetched again."""
    from proteus.data import STELLAR_SPECTRA_MUSCLES, dataset_dir
    from proteus.utils.data import download_muscles

    monkeypatch.setenv('FWL_DATA', str(tmp_path))
    star_file = dataset_dir(STELLAR_SPECTRA_MUSCLES, data_root=tmp_path) / 'gj876.txt'
    star_file.parent.mkdir(parents=True, exist_ok=True)
    star_file.write_text('stale')
    seen = {}

    def fake_fetch(key, name, data_root=None):
        seen['present'] = star_file.exists()
        return star_file

    monkeypatch.setattr('proteus.data.fetch_dataset_file', fake_fetch)
    monkeypatch.setattr('proteus.data._data_root', lambda: tmp_path)

    assert download_muscles(stars='gj876', force=True) is True
    assert seen == {'present': False}


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_interior_lookuptables(mock_fetch, tmp_path, monkeypatch):
    """The always-needed Wolf and Bower melting curves are fetched through fwl-io."""
    from proteus.data import MELTING_WOLF_BOWER_2018
    from proteus.utils.data import download_interior_lookuptables

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)

    download_interior_lookuptables(clean=False)

    mock_fetch.assert_called_once_with(MELTING_WOLF_BOWER_2018, data_root=tmp_path)
    assert mock_fetch.call_args.args[0] == 'interior.melting_curves.wolf_bower_2018'


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_interior_lookuptables_clean(mock_fetch, tmp_path, monkeypatch):
    """``clean=True`` removes the fetched dataset directory, then fetches again."""
    from proteus.data import MELTING_WOLF_BOWER_2018, dataset_dir
    from proteus.utils.data import download_interior_lookuptables

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    folder = dataset_dir(MELTING_WOLF_BOWER_2018, data_root=tmp_path)
    folder.mkdir(parents=True)
    stale = folder / 'stale.dat'
    stale.write_text('old')

    download_interior_lookuptables(clean=True)

    assert not stale.exists()
    mock_fetch.assert_called_once_with(MELTING_WOLF_BOWER_2018, data_root=tmp_path)


_LEGACY_CURVES = ('solidus.dat', 'liquidus.dat')


@pytest.mark.unit
@pytest.mark.parametrize(
    'name, attr',
    [
        ('Monteux+600', 'MELTING_MONTEUX_PLUS600'),
        ('Monteux-600', 'MELTING_MONTEUX_MINUS600'),
        ('Wolf_Bower+2018', 'MELTING_WOLF_BOWER_2018'),
    ],
)
@patch('proteus.data.fetch_dataset')
def test_download_melting_curves_fetches_known_name(
    mock_fetch, name, attr, tmp_path, monkeypatch
):
    """Each of the three named curves is fetched from its own manifest dataset."""
    import proteus.data as data_mod
    from proteus.utils.data import download_melting_curves

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    config = MagicMock()
    config.interior_struct.melting_dir = name

    download_melting_curves(config, clean=False)

    mock_fetch.assert_called_once_with(getattr(data_mod, attr), data_root=tmp_path)
    assert not (tmp_path / 'interior_lookup_tables').exists()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_melting_curves_served_name_fetches_despite_a_local_dir(
    mock_fetch, tmp_path, monkeypatch
):
    """A served name is fetched although a local directory holds both P-T files."""
    from proteus.data import MELTING_WOLF_BOWER_2018
    from proteus.utils.data import download_melting_curves

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Wolf_Bower+2018'
    local.mkdir(parents=True)
    for name in ('solidus_P-T.dat', 'liquidus_P-T.dat'):
        (local / name).write_text('dummy\n')
    config = MagicMock()
    config.interior_struct.melting_dir = 'Wolf_Bower+2018'

    download_melting_curves(config, clean=False)

    mock_fetch.assert_called_once_with(MELTING_WOLF_BOWER_2018, data_root=tmp_path)
    assert (local / 'solidus_P-T.dat').read_text() == 'dummy\n'


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_melting_curves_incomplete_local_dir_fetches(
    mock_fetch, tmp_path, monkeypatch
):
    """A local directory with only one of the two P-T files does not count as present."""
    from proteus.data import MELTING_WOLF_BOWER_2018
    from proteus.utils.data import download_melting_curves

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Wolf_Bower+2018'
    local.mkdir(parents=True)
    (local / 'solidus_P-T.dat').write_text('dummy\n')
    config = MagicMock()
    config.interior_struct.melting_dir = 'Wolf_Bower+2018'

    download_melting_curves(config, clean=False)

    mock_fetch.assert_called_once_with(MELTING_WOLF_BOWER_2018, data_root=tmp_path)
    assert sorted(p.name for p in local.iterdir()) == ['solidus_P-T.dat']


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_melting_curves_unknown_name_raises(mock_fetch, tmp_path, monkeypatch):
    """A name that no dataset serves and no local directory holds is an error."""
    from proteus.utils.data import download_melting_curves

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    config = MagicMock()
    config.interior_struct.melting_dir = 'UnknownCurve'

    with pytest.raises(ValueError, match='No dataset serves melting_dir'):
        download_melting_curves(config)
    mock_fetch.assert_not_called()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
@pytest.mark.parametrize('names', [('solidus_P-T.dat', 'liquidus_P-T.dat'), _LEGACY_CURVES])
def test_download_melting_curves_unknown_name_local_dir_ok(
    mock_fetch, tmp_path, monkeypatch, names
):
    """A free-form local directory with either pair of names is accepted without a
    fetch, and clean=True keeps it."""
    from proteus.utils.data import download_melting_curves

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'MyCurve'
    local.mkdir(parents=True)
    for name in names:
        (local / name).write_text('dummy\n')
    config = MagicMock()
    config.interior_struct.melting_dir = 'MyCurve'

    download_melting_curves(config, clean=True)

    mock_fetch.assert_not_called()
    assert (local / names[1]).read_text() == 'dummy\n'


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_melting_curves_clean_removes_the_served_dataset_and_local_dir(
    mock_fetch, tmp_path, monkeypatch
):
    """``clean=True`` on a served name removes the fetched dataset and the local directory."""
    from proteus.data import MELTING_WOLF_BOWER_2018, dataset_dir
    from proteus.utils.data import download_melting_curves

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    fetched = dataset_dir(MELTING_WOLF_BOWER_2018, data_root=tmp_path)
    fetched.mkdir(parents=True)
    stale = fetched / 'stale.dat'
    stale.write_text('old')
    local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Wolf_Bower+2018'
    local.mkdir(parents=True)
    keep = local / 'keep.dat'
    keep.write_text('mine')
    config = MagicMock()
    config.interior_struct.melting_dir = 'Wolf_Bower+2018'

    download_melting_curves(config, clean=True)

    assert not stale.exists()
    assert not local.exists()
    mock_fetch.assert_called_once_with(MELTING_WOLF_BOWER_2018, data_root=tmp_path)


@pytest.mark.unit
def test_resolve_melting_curve_files_manifest_layout(tmp_path):
    """A known name without a local directory resolves to the versioned dataset files."""
    from proteus.utils.data import resolve_melting_curve_files

    expected = {
        'Monteux+600': ('monteux_plus_600', 'r15728091'),
        'Monteux-600': ('monteux_minus_600', 'r15728138'),
        'Wolf_Bower+2018': ('wolf_bower_2018', 'r15728072'),
    }
    for name, (folder, rec) in expected.items():
        solidus, liquidus = resolve_melting_curve_files(name, data_root=tmp_path)
        base = tmp_path / 'interior' / 'melting_curves' / folder / rec
        assert solidus == base / 'solidus.dat'
        assert liquidus == base / 'liquidus.dat'


@pytest.mark.unit
def test_resolve_melting_curve_files_prefers_the_served_dataset(tmp_path):
    """For a served name the fetched dataset wins over a local directory, which is
    used only while the dataset is not on disk."""
    from proteus.data import MELTING_WOLF_BOWER_2018, dataset_dir
    from proteus.utils.data import resolve_melting_curve_files

    local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Wolf_Bower+2018'
    local.mkdir(parents=True)
    (local / 'solidus_P-T.dat').write_text('s')
    (local / 'liquidus_P-T.dat').write_text('l')
    assert resolve_melting_curve_files('Wolf_Bower+2018', data_root=tmp_path) == (
        local / 'solidus_P-T.dat',
        local / 'liquidus_P-T.dat',
    )

    folder = dataset_dir(MELTING_WOLF_BOWER_2018, data_root=tmp_path)
    folder.mkdir(parents=True)
    for name in _LEGACY_CURVES:
        (folder / name).write_text('x')
    assert resolve_melting_curve_files('Wolf_Bower+2018', data_root=tmp_path) == (
        folder / 'solidus.dat',
        folder / 'liquidus.dat',
    )


@pytest.mark.unit
def test_resolve_melting_curve_files_partial_local_falls_back(tmp_path):
    """One local file alone does not shadow the manifest dataset."""
    from proteus.utils.data import resolve_melting_curve_files

    local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Monteux+600'
    local.mkdir(parents=True)
    (local / 'liquidus_P-T.dat').write_text('l')

    solidus, liquidus = resolve_melting_curve_files('Monteux+600', data_root=tmp_path)

    assert solidus.parent.parent.name == 'monteux_plus_600'
    assert solidus.name == 'solidus.dat' and liquidus.name == 'liquidus.dat'


@pytest.mark.unit
def test_resolve_melting_curve_files_unknown_name_gives_local_paths(tmp_path):
    """An unknown name resolves to the local P-T paths, which the caller reports."""
    from proteus.utils.data import resolve_melting_curve_files

    solidus, liquidus = resolve_melting_curve_files('Free_Form', data_root=tmp_path)

    local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Free_Form'
    assert (solidus, liquidus) == (local / 'solidus_P-T.dat', local / 'liquidus_P-T.dat')
    assert not tmp_path.joinpath('interior_struct').exists()


@pytest.mark.unit
def test_resolve_melting_curve_files_legacy_names_only_for_an_unserved_name(tmp_path):
    """solidus.dat and liquidus.dat in a local directory count for a name the manifest
    does not serve, and not for a served name."""
    from proteus.utils.data import resolve_melting_curve_files

    for name in ('Free_Form', 'Monteux+600'):
        local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / name
        local.mkdir(parents=True)
        for curve in _LEGACY_CURVES:
            (local / curve).write_text('x')
    free = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Free_Form'
    assert resolve_melting_curve_files('Free_Form', data_root=tmp_path) == (
        free / 'solidus.dat',
        free / 'liquidus.dat',
    )
    solidus, _ = resolve_melting_curve_files('Monteux+600', data_root=tmp_path)
    assert solidus.parent.parent.name == 'monteux_plus_600'


@pytest.mark.unit
def test_resolve_lookup_table_dir_layout(tmp_path):
    """The lookup tables resolve to the versioned dataset directory of record 19473625."""
    from proteus.utils.data import resolve_lookup_table_dir

    folder = resolve_lookup_table_dir(data_root=tmp_path)

    assert (
        folder
        == tmp_path
        / 'interior'
        / 'eos'
        / 'dk09_1tpa_elec_free'
        / 'mgsio3_wolf_bower_2018_1tpa'
        / 'r19473625'
    )
    assert resolve_lookup_table_dir(data_root=str(tmp_path)) == folder


@pytest.mark.unit
def test_find_lookup_table_dir(tmp_path):
    """The directory is returned only once it holds every SPIDER table file.

    fwl-io fetches the files in sorted order, so an interrupted fetch leaves the
    last ones missing; the check must reject that state, not only an empty one.
    """
    from proteus.interior_energetics.common import (
        _SPIDER_EOS_MELTING_CURVES,
        _SPIDER_EOS_PHASE_FILES,
    )
    from proteus.utils.data import find_lookup_table_dir, resolve_lookup_table_dir

    assert find_lookup_table_dir(data_root=tmp_path) is None

    folder = resolve_lookup_table_dir(data_root=tmp_path)
    folder.mkdir(parents=True, exist_ok=True)
    names = sorted(_SPIDER_EOS_PHASE_FILES + _SPIDER_EOS_MELTING_CURVES)
    for name in names[:-1]:
        (folder / name).write_text('x')
    assert find_lookup_table_dir(data_root=tmp_path) is None

    (folder / names[-1]).write_text('x')
    assert find_lookup_table_dir(data_root=tmp_path) == folder


@pytest.mark.unit
def test_find_lookup_table_dir_unwritable_root_gives_none():
    """A data root that cannot be created gives None, not an error."""
    from proteus.utils.data import find_lookup_table_dir

    with patch('proteus.data.dataset_dir', side_effect=OSError('read-only')) as mock_dir:
        folder = find_lookup_table_dir(data_root='/nonexistent')

    assert folder is None
    mock_dir.assert_called_once()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_stellar_spectra_default(mock_fetch):
    """Default stellar spectra download fetches Named, solar and MUSCLES through fwl-io."""
    from proteus.data import (
        STELLAR_SPECTRA_MUSCLES,
        STELLAR_SPECTRA_NAMED,
        STELLAR_SPECTRA_SOLAR,
    )
    from proteus.utils.data import download_stellar_spectra

    download_stellar_spectra()

    fetched = [c.args[0] for c in mock_fetch.call_args_list]
    assert sorted(fetched) == sorted(
        [STELLAR_SPECTRA_NAMED, STELLAR_SPECTRA_SOLAR, STELLAR_SPECTRA_MUSCLES]
    )
    # Each collection goes to the default data root, with no other argument.
    assert all(c.args == (c.args[0],) and c.kwargs == {} for c in mock_fetch.call_args_list)


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_stellar_spectra_phoenix_is_not_a_bulk_collection(mock_fetch):
    """PHOENIX is fetched one grid at a time through download_phoenix, so the
    bulk collection downloader rejects it instead of pulling the 9 GB record.
    """
    from proteus.utils.data import download_stellar_spectra

    with pytest.raises(ValueError, match="choose from \\['Named', 'solar', 'MUSCLES'\\]"):
        download_stellar_spectra(folders=('PHOENIX',))
    mock_fetch.assert_not_called()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_optional_dataset_failure_does_not_abort_the_run(mock_fetch, caplog):
    """A mirror failure on decorative data is reported, not raised.

    fwl-io reports an exhausted mirror with DownloadError, a RuntimeError that
    the start-of-run guard does not catch, so an escaping error would abort
    every simulation over an overlay that only decorates a plot.
    """
    from fwl_io import DownloadError

    from proteus.utils.data import download_exoplanet_data

    mock_fetch.side_effect = DownloadError('could not obtain from any mirror')

    with caplog.at_level(logging.ERROR):
        result = download_exoplanet_data()

    assert result is False
    # Discrimination: the fetch really ran, so the survival is not a
    # short-circuit that skipped the download altogether.
    mock_fetch.assert_called_once()
    assert any('exoplanet data' in rec.getMessage() for rec in caplog.records), (
        f'expected an error naming the dataset; got {[r.getMessage() for r in caplog.records]}'
    )


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_optional_dataset_offline_failure_does_not_abort_the_run(mock_fetch):
    """An offline data tree skips the overlay instead of failing the run.

    OfflineDataError is also a RuntimeError, so it escapes the same guard; a
    user running without network must still be able to start a simulation.
    """
    from fwl_io import OfflineDataError

    from proteus.utils.data import download_massradius_data

    mock_fetch.side_effect = OfflineDataError('offline mode is active')

    assert download_massradius_data() is False
    mock_fetch.assert_called_once()


@pytest.mark.unit
def test_reference_data_failure_still_reaches_the_interior_data(monkeypatch):
    """A failed overlay fetch does not block the interior tables fetched after it.

    The reference overlays are downloaded before the interior lookup tables and
    melting curves a run actually needs, so an abort there would deprive the
    solver of its data.
    """
    from fwl_io import DownloadError

    from proteus.utils import data as data_mod

    def _boom(key, data_root=None):
        raise DownloadError('could not obtain from any mirror')

    reached = {'interior': False, 'melting': False}
    monkeypatch.setattr('proteus.data.fetch_dataset', _boom)
    monkeypatch.setattr(
        data_mod,
        'download_interior_lookuptables',
        lambda **kw: reached.__setitem__('interior', True),
    )
    monkeypatch.setattr(
        data_mod,
        'download_melting_curves',
        lambda *a, **kw: reached.__setitem__('melting', True),
    )
    for name in (
        'download_stellar_spectra',
        'download_stellar_tracks',
        'download_spectral_file',
        'download_surface_albedos',
        'download_scattering',
        'download_zalmoxis_eos_for_config',
        'download_eos_dynamic',
    ):
        monkeypatch.setattr(data_mod, name, lambda *a, **kw: None)

    cfg = SimpleNamespace(
        star=SimpleNamespace(module='dummy'),
        atmos_clim=SimpleNamespace(module='dummy', aerosols_enabled=False),
        interior_energetics=SimpleNamespace(module='aragog'),
        interior_struct=SimpleNamespace(module='dummy', eos_dir=None),
    )
    data_mod._get_sufficient(cfg)

    assert reached['interior'], 'interior lookup tables must still be fetched'
    assert reached['melting'], 'melting curves must still be fetched'


@pytest.mark.unit
def test_required_dataset_failure_does_not_stop_later_fetches(monkeypatch, tmp_path):
    """An unreachable mirror for one dataset leaves the other datasets to be fetched.

    fwl-io reports a failed fetch with RuntimeError subclasses, which an OSError
    guard does not catch; each step and each stellar collection is guarded on
    its own, so a failing named-star record does not stop the solar spectra.
    """
    from fwl_io import DownloadError

    from proteus.config import read_config_object
    from proteus.utils import data as data_mod

    attempted = []

    def _fetch(key, *args, data_root=None):
        attempted.append(key)
        if not key.startswith('interior.melting_curves.'):
            raise DownloadError('could not obtain from any mirror')
        return []

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path)
    monkeypatch.setattr('proteus.data.fetch_dataset', _fetch)
    monkeypatch.setattr('proteus.data.fetch_dataset_file', _fetch)
    monkeypatch.setattr(data_mod, 'download_stellar_tracks', lambda *a, **kw: None)
    cfg = read_config_object(
        str(Path(__file__).resolve().parents[2] / 'input' / 'minimal.toml')
    )
    cfg.star.mors.spectrum_source = None

    data_mod.download_sufficient_data(cfg)

    assert attempted[:3] == ['star.spectra.named', 'star.spectra.solar', 'star.spectra.muscles']
    assert 'atmos_clim.spectral_files.honeyside.48' in attempted
    assert 'interior.melting_curves.wolf_bower_2018' in attempted
    # A failed file inside the EOS step does not stop the next one.
    assert {
        'interior.eos.paleos_iron',
        'interior.eos.paleos_mgsio3_unified',
        'interior.eos.paleos_mgsio3',
    } <= set(attempted)


@pytest.mark.unit
@pytest.mark.parametrize(
    ('melting_dir', 'failing_key'),
    [
        (None, 'interior.melting_curves.wolf_bower_2018'),
        ('Monteux-600', 'interior.melting_curves.monteux_minus_600'),
    ],
)
@pytest.mark.parametrize('error', [DownloadError, PermissionError])
def test_melting_curve_fetch_failure_is_reported_and_later_steps_run(
    monkeypatch, tmp_path, melting_dir, failing_key, error
):
    """A failed melting-curve fetch is logged and the EOS fetch still runs.

    A configured curve that is still missing stops the run where it is read
    (_provide_spider_eos_tables), so the fetch step needs no special case;
    an OSError such as a read-only data root is handled like a mirror error.
    """
    from proteus.config import read_config_object
    from proteus.utils import data as data_mod

    attempted = []

    def _fetch(key, *args, data_root=None):
        attempted.append(key)
        if key == failing_key:
            raise error('could not obtain from any mirror')
        return []

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path)
    monkeypatch.setattr('proteus.data.fetch_dataset', _fetch)
    monkeypatch.setattr('proteus.data.fetch_dataset_file', _fetch)
    monkeypatch.setattr(data_mod, 'download_stellar_tracks', lambda *a, **kw: None)
    cfg = read_config_object(
        str(Path(__file__).resolve().parents[2] / 'input' / 'minimal.toml')
    )
    cfg.interior_struct.melting_dir = melting_dir

    data_mod.download_sufficient_data(cfg)

    assert failing_key in attempted
    assert 'interior.eos.paleos_iron' in attempted


@pytest.mark.unit
def test_fetch_errors_on_an_old_fwl_io_ask_for_the_upgrade(monkeypatch):
    """An fwl-io without the fetch error classes is reported as too old.

    Importing the classes at module level would fail on import of
    proteus.utils.data, before any message could name the required version.
    """
    import sys

    from proteus.data import FWL_IO_FLOOR
    from proteus.utils.data import _fetch_errors

    assert DownloadError in _fetch_errors()
    monkeypatch.setitem(sys.modules, 'fwl_io.archive', None)

    with pytest.raises(RuntimeError, match=f'upgrade to fwl-io>={FWL_IO_FLOOR}') as raised:
        _fetch_errors()

    assert isinstance(raised.value.__cause__, ImportError)


@pytest.mark.unit
def test_attempt_on_an_old_fwl_io_does_not_run_the_step(monkeypatch):
    """The fetch error classes are resolved before the step runs.

    Resolving them inside the except clause would replace an unrelated error of
    the step with the upgrade message, hiding the real cause.
    """
    from proteus.utils import data as data_mod

    def _stale():
        raise RuntimeError('upgrade to fwl-io>=26.10.6')

    calls = []
    monkeypatch.setattr(data_mod, '_fetch_errors', _stale)

    with pytest.raises(RuntimeError, match='upgrade to fwl-io'):
        data_mod._attempt('test data', lambda: calls.append(1))

    assert calls == []


@pytest.mark.unit
@pytest.mark.parametrize(
    'error',
    [
        'DownloadError',
        'OfflineDataError',
        'ArchiveError',
        'MissingDataRootError',
        'OSError',
    ],
)
def test_attempt_reports_fetch_errors_and_passes_other_errors(error, caplog):
    """Every fetch error is logged and reported, a programming error still raises."""
    import fwl_io
    from fwl_io.archive import ArchiveError

    from proteus.utils.data import _attempt

    errors = {
        'DownloadError': fwl_io.DownloadError,
        'OfflineDataError': fwl_io.OfflineDataError,
        'ArchiveError': ArchiveError,
        'MissingDataRootError': fwl_io.MissingDataRootError,
        'OSError': OSError,
    }

    def _fail():
        raise errors[error]('mirror down')

    with caplog.at_level('WARNING'):
        assert _attempt('test data', _fail) is False
    assert 'test data: mirror down' in caplog.text
    # Discrimination: an error that is not a fetch failure is not swallowed,
    # including a plain RuntimeError such as the stale-fwl-io message.
    with pytest.raises(TypeError):
        _attempt('test data', lambda: (_ for _ in ()).throw(TypeError('bug')))
    with pytest.raises(RuntimeError, match='stale'):
        _attempt('test data', lambda: (_ for _ in ()).throw(RuntimeError('stale')))


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_exoplanet_data(mock_fetch):
    """The exoplanet catalogue is fetched through fwl-io."""
    from proteus.data import EXOPLANET_REFERENCE
    from proteus.utils.data import download_exoplanet_data

    download_exoplanet_data()

    mock_fetch.assert_called_once_with(EXOPLANET_REFERENCE)
    # Discrimination: the key must be the catalogue, not the mass-radius
    # dataset declared beside it in the same manifest.
    assert mock_fetch.call_args.args[0] == 'observe.exoplanet_reference'


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_massradius_data(mock_fetch):
    """The mass-radius relations are fetched through fwl-io."""
    from proteus.data import MASS_RADIUS_ZENG_2019
    from proteus.utils.data import download_massradius_data

    download_massradius_data()

    mock_fetch.assert_called_once_with(MASS_RADIUS_ZENG_2019)
    # Discrimination: the key must be the mass-radius dataset, not the
    # catalogue declared beside it in the same manifest.
    assert mock_fetch.call_args.args[0] == 'observe.mass_radius.zeng_2019'


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_surface_albedos(mock_fetch):
    """The surface albedos are fetched through fwl-io."""
    from proteus.data import SURFACE_ALBEDOS_HAMMOND_2024
    from proteus.utils.data import download_surface_albedos

    download_surface_albedos()

    mock_fetch.assert_called_once_with(SURFACE_ALBEDOS_HAMMOND_2024)
    assert mock_fetch.call_args.args[0] == 'atmos_clim.surface_albedos.hammond_2024'


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_scattering_fetches_the_manifest_dataset(mock_fetch):
    """The .mon scattering tables are fetched through fwl-io from the PROTEUS manifest."""
    from proteus.data import SCATTERING
    from proteus.utils.data import download_scattering

    download_scattering()

    mock_fetch.assert_called_once_with(SCATTERING)
    assert mock_fetch.call_args.args[0] == 'atmos_clim.scattering.socrates_aerosols'


@pytest.mark.unit
@patch('proteus.data.fetch_dataset', side_effect=OSError('mirror unreachable'))
def test_download_surface_albedos_propagates_fetch_failure(mock_fetch):
    """AGNI needs the albedo files, so a failed fetch raises instead of being logged."""
    from proteus.utils.data import download_surface_albedos

    with pytest.raises(OSError, match='mirror unreachable'):
        download_surface_albedos()
    mock_fetch.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.FWL_DATA_DIR')
def test_GetFWLData(mock_fwl_data_dir, tmp_path):
    """Test FWL data directory getter."""
    from proteus.utils.data import GetFWLData

    # Patch FWL_DATA_DIR to return tmp_path
    with patch('proteus.utils.data.FWL_DATA_DIR', tmp_path):
        result = GetFWLData()
        assert result == tmp_path.absolute()
        # Discrimination: a regression returning a relative path would
        # still equal tmp_path.absolute() only on the cwd-matches edge
        # case; pin the absolute-path property explicitly.
        assert result.is_absolute()


@pytest.mark.unit
def test_get_Seager_EOS_exists(tmp_path):
    """Test get_Seager_EOS when EOS folder already exists."""
    from proteus.utils.data import get_Seager_EOS

    # Create EOS folder
    eos_folder = _seed_seager(tmp_path)

    # Patch FWL_DATA_DIR at module level
    with patch('proteus.utils.data.FWL_DATA_DIR', tmp_path):
        iron_silicate, water = get_Seager_EOS()

    # Check structure of returned dictionaries
    # iron_silicate has 'mantle' and 'core' keys
    assert 'mantle' in iron_silicate
    assert 'core' in iron_silicate
    # water dict has 'core', 'mantle', and 'ice_layer' keys (Seager 2007 water planet structure)
    assert 'core' in water
    assert 'mantle' in water
    assert 'ice_layer' in water

    # Check file paths (returned as str, not Path, for Zalmoxis compatibility)
    assert iron_silicate['mantle']['eos_file'] == str(eos_folder / 'eos_seager07_silicate.txt')
    assert iron_silicate['core']['eos_file'] == str(eos_folder / 'eos_seager07_iron.txt')
    assert water['ice_layer']['eos_file'] == str(eos_folder / 'eos_seager07_water.txt')


@pytest.mark.unit
@patch('proteus.utils.data.download_Seager_EOS')
def test_get_Seager_EOS_not_exists(mock_download, tmp_path):
    """Test get_Seager_EOS when EOS folder doesn't exist."""
    from proteus.data import EOS_SEAGER_2007, dataset_dir
    from proteus.utils.data import get_Seager_EOS

    # Precondition: the EOS folder really is absent before the call so
    # the missing-folder code path is what gets exercised.
    assert not (
        dataset_dir(EOS_SEAGER_2007, data_root=tmp_path) / 'eos_seager07_iron.txt'
    ).exists()

    # Patch FWL_DATA_DIR and call function
    with patch('proteus.utils.data.FWL_DATA_DIR', tmp_path):
        get_Seager_EOS()

    # Should call download
    mock_download.assert_called_once()


@pytest.mark.unit
def test_download_Seager_EOS(monkeypatch, tmp_path):
    """The Seager EOS is fetched through fwl-io under the manifest key."""
    import proteus.data as data_pkg
    import proteus.utils.data as data_mod
    from proteus.utils.data import download_Seager_EOS

    calls = []
    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    monkeypatch.setattr(
        data_pkg, 'fetch_dataset', lambda key, data_root=None: calls.append((key, data_root))
    )

    download_Seager_EOS()

    assert calls == [('interior_struct.eos.seager_2007', tmp_path)]
    assert data_pkg.EOS_SEAGER_2007 == calls[0][0]


@pytest.mark.unit
@pytest.mark.parametrize('folder', ['UnknownFolder', 'scattering'])
@patch('proteus.data.fetch_dataset')
def test_download_stellar_spectra_rejects_other_collections(mock_fetch, folder):
    """A name other than Named, solar or MUSCLES is rejected before any fetch."""
    from proteus.utils.data import download_stellar_spectra

    with pytest.raises(ValueError, match=f"Unknown stellar spectra collection.*'{folder}'"):
        download_stellar_spectra(folders=('solar', folder))
    mock_fetch.assert_not_called()


@pytest.mark.unit
def test_download_Seager_EOS_failure_raises(monkeypatch, tmp_path):
    """A failed Seager fetch raises: Zalmoxis cannot run without the tables."""
    import proteus.data as data_pkg
    import proteus.utils.data as data_mod
    from proteus.utils.data import download_Seager_EOS

    attempts = []

    def boom(key, data_root=None):
        attempts.append(key)
        raise OSError('no network')

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    monkeypatch.setattr(data_pkg, 'fetch_dataset', boom)

    with pytest.raises(OSError, match='no network'):
        download_Seager_EOS()
    assert attempts == ['interior_struct.eos.seager_2007']


# =============================================================================
# get_petsc / get_spider wrapper tests
# =============================================================================


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.os.path.isdir')
@patch('proteus.utils.data._none_dirs')
def test_get_petsc_passes_workpath_as_first_arg(mock_dirs, mock_isdir, mock_run, tmp_path):
    """``get_petsc()`` calls ``get_petsc.sh`` with ``workpath`` as ``$1``.

    The shell script receives the full petsc path as its first positional
    argument so it can install into the correct location.
    """
    from proteus.utils.data import get_petsc

    mock_dirs.return_value = {
        'proteus': str(tmp_path),
        'tools': str(tmp_path / 'tools'),
    }
    mock_isdir.return_value = False  # petsc dir does not exist yet

    # Create a dummy log target dir
    (tmp_path / 'tools').mkdir(exist_ok=True)

    get_petsc()

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0].endswith('get_petsc.sh')
    # Second element is the workpath passed as $1
    assert 'petsc' in cmd[1]


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.os.path.isdir')
@patch('proteus.utils.data._none_dirs')
def test_get_petsc_skips_when_dir_exists(mock_dirs, mock_isdir, mock_run, tmp_path):
    """``get_petsc()`` returns early if the petsc directory already exists."""
    from proteus.utils.data import get_petsc

    mock_dirs.return_value = {
        'proteus': str(tmp_path),
        'tools': str(tmp_path / 'tools'),
    }
    mock_isdir.return_value = True  # petsc dir already exists

    get_petsc()

    mock_run.assert_not_called()
    # Discrimination: the early-return must consult the directory check;
    # a regression that skipped the isdir guard but also bypassed sp.run
    # by another path would pass assert_not_called for the wrong reason.
    mock_isdir.assert_called()


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.os.path.isdir')
@patch('proteus.utils.data._none_dirs')
def test_get_petsc_logs_output_to_file(mock_dirs, mock_isdir, mock_run, tmp_path):
    """``get_petsc()`` redirects stdout/stderr to a log file.

    Verifies that ``sp.run`` is called with file handles for stdout and
    stderr (not None/PIPE), indicating output is captured to a log file.
    """
    from proteus.utils.data import get_petsc

    mock_dirs.return_value = {
        'proteus': str(tmp_path),
        'tools': str(tmp_path / 'tools'),
    }
    mock_isdir.return_value = False
    (tmp_path / 'tools').mkdir(exist_ok=True)

    get_petsc()

    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args[1]
    # stdout and stderr should be file handles (not None)
    assert call_kwargs.get('stdout') is not None
    assert call_kwargs.get('stderr') is not None
    assert call_kwargs.get('check') is True


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.os.path.isdir')
@patch('proteus.utils.data._none_dirs')
def test_get_spider_passes_workpath_as_first_arg(mock_dirs, mock_isdir, mock_run, tmp_path):
    """``get_spider()`` calls ``get_spider.sh`` with ``workpath`` as ``$1``."""
    from proteus.utils.data import get_spider

    mock_dirs.return_value = {
        'proteus': str(tmp_path),
        'tools': str(tmp_path / 'tools'),
    }
    (tmp_path / 'tools').mkdir(exist_ok=True)

    # isdir returns True for petsc (skip get_petsc), False for SPIDER
    def isdir_side_effect(path):
        return 'petsc' in path

    mock_isdir.side_effect = isdir_side_effect

    get_spider()

    # sp.run should be called once (for get_spider.sh only; get_petsc skipped)
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0].endswith('get_spider.sh')
    assert 'SPIDER' in cmd[1]


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.os.path.isdir')
@patch('proteus.utils.data._none_dirs')
def test_get_spider_skips_when_dir_exists(mock_dirs, mock_isdir, mock_run, tmp_path):
    """``get_spider()`` returns early if the SPIDER directory already exists."""
    from proteus.utils.data import get_spider

    mock_dirs.return_value = {
        'proteus': str(tmp_path),
        'tools': str(tmp_path / 'tools'),
    }

    # Both petsc and SPIDER dirs exist
    mock_isdir.return_value = True

    get_spider()

    # sp.run should never be called (both dirs exist)
    mock_run.assert_not_called()
    # Discrimination: the early-return must have consulted the isdir
    # check (otherwise assert_not_called could pass on a regression that
    # bypassed both the isdir guard and the sp.run call by some other
    # short-circuit).
    mock_isdir.assert_called()


@pytest.mark.unit
@patch('proteus.utils.data.get_petsc')
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.os.path.isdir')
@patch('proteus.utils.data._none_dirs')
def test_get_spider_calls_get_petsc_first(
    mock_dirs, mock_isdir, mock_run, mock_get_petsc, tmp_path
):
    """``get_spider()`` invokes ``get_petsc()`` before installing SPIDER.

    PETSc is a build dependency of SPIDER, so it must be set up first.
    """
    from proteus.utils.data import get_spider

    mock_dirs.return_value = {
        'proteus': str(tmp_path),
        'tools': str(tmp_path / 'tools'),
    }
    mock_isdir.return_value = False
    (tmp_path / 'tools').mkdir(exist_ok=True)

    get_spider()

    mock_get_petsc.assert_called_once()
    # Discrimination: sp.run runs exactly once here (the SPIDER install
    # script), because get_petsc is itself mocked. A regression that
    # double-dispatched the script or skipped the install entirely would
    # break this pin.
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0].endswith('get_spider.sh')


# ============================================================================
# test _get_sufficient: Zalmoxis EOS branches
# ============================================================================


@pytest.mark.unit
@patch('proteus.utils.data.download_zalmoxis_eos')
@patch('proteus.utils.data.download_eos_dynamic')
@patch('proteus.utils.data.download_melting_curves')
@patch('proteus.utils.data.download_interior_lookuptables')
@patch('proteus.utils.data.download_massradius_data')
@patch('proteus.utils.data.download_surface_albedos')
@patch('proteus.utils.data.download_exoplanet_data')
@patch('proteus.utils.data.download_stellar_spectra')
@patch('proteus.utils.data.download_spectral_file')
@patch('proteus.utils.data.download_phoenix')
def test_get_sufficient_zalmoxis_wolf_bower(
    _m_ph,
    _m_sp,
    _m_st,
    _m_ex,
    _m_sa,
    _m_mr,
    _m_il,
    _m_mc,
    mock_dyn,
    mock_zalmoxis_eos,
):
    """_get_sufficient calls download_zalmoxis_eos for Zalmoxis WolfBower2018."""
    from unittest.mock import MagicMock

    from proteus.utils.data import _get_sufficient

    config = MagicMock()
    config.interior_energetics.module = 'spider'
    config.interior_energetics.eos_dir = 'WolfBower2018_MgSiO3'
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.zalmoxis.mantle_eos = 'WolfBower2018:MgSiO3'
    config.interior_struct.zalmoxis.core_eos = 'Seager2007:iron'
    config.interior_struct.zalmoxis.ice_layer_eos = ''

    _get_sufficient(config, clean=False)

    # Zalmoxis EOS download called with the full EOS identifiers
    mock_zalmoxis_eos.assert_called_once_with(
        mantle_eos='WolfBower2018:MgSiO3',
        core_eos='Seager2007:iron',
        ice_layer_eos='',
        volatile_eos='',
        anchor_pair=False,
    )
    # SPIDER dynamic EOS still downloaded separately
    mock_dyn.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.download_zalmoxis_eos_for_config')
@patch('proteus.utils.data.download_eos_dynamic')
@patch('proteus.utils.data.download_melting_curves')
@patch('proteus.utils.data.download_interior_lookuptables')
@patch('proteus.utils.data.download_massradius_data')
@patch('proteus.utils.data.download_surface_albedos')
@patch('proteus.utils.data.download_exoplanet_data')
@patch('proteus.utils.data.download_stellar_spectra')
@patch('proteus.utils.data.download_spectral_file')
@patch('proteus.utils.data.download_phoenix')
def test_get_sufficient_dummy_structure_fetches_ps_tables_without_eos_dir(
    _m_ph, _m_sp, _m_st, _m_ex, _m_sa, _m_mr, _m_il, _m_mc, mock_dyn, _m_zal
):
    """The dummy structure always reads the P-S set, so it is fetched without eos_dir."""
    from unittest.mock import MagicMock

    from proteus.utils.data import _get_sufficient

    config = MagicMock()
    config.interior_energetics.module = 'aragog'
    config.interior_struct.module = 'dummy'
    config.interior_struct.eos_dir = None

    _get_sufficient(config, clean=False)
    assert mock_dyn.call_count == 1

    # Discrimination: Zalmoxis with a PALEOS mantle generates its own tables.
    mock_dyn.reset_mock()
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS:MgSiO3'
    _get_sufficient(config, clean=False)
    assert mock_dyn.call_count == 0

    # A mixture with a Wolf and Bower MgSiO3 component reads the fetched set.
    config.interior_struct.zalmoxis.mantle_eos = 'WolfBower2018:MgSiO3:0.9+PALEOS:H2O:0.1'
    _get_sufficient(config, clean=False)
    assert mock_dyn.call_count == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ('energetics', 'struct', 'eos_dir', 'mantle_eos', 'expected'),
    [
        ('aragog', 'dummy', None, 'PALEOS:MgSiO3', True),
        ('spider', 'dummy', None, 'PALEOS:MgSiO3', True),
        ('spider', 'spider', 'WolfBower2018_MgSiO3', 'PALEOS:MgSiO3', True),
        ('aragog', 'zalmoxis', None, 'PALEOS:MgSiO3', False),
        ('aragog', 'zalmoxis', None, 'PALEOS-2phase:MgSiO3', False),
        ('aragog', 'zalmoxis', None, 'WolfBower2018:MgSiO3', True),
        ('spider', 'zalmoxis', None, 'PALEOS:MgSiO3:0.9+PALEOS:H2O:0.1', False),
        ('spider', 'zalmoxis', None, 'WolfBower2018:MgSiO3:0.9+PALEOS:H2O:0.1', True),
        ('aragog', 'zalmoxis', None, 'PALEOS:MgSiO3:1.0', False),
        ('aragog', 'zalmoxis', 'WolfBower2018_MgSiO3', 'PALEOS:MgSiO3', True),
        ('dummy', 'dummy', None, 'PALEOS:MgSiO3', False),
    ],
)
def test_needs_spider_ps_tables(energetics, struct, eos_dir, mantle_eos, expected):
    """The P-S lookup set is needed exactly when no PALEOS table set is generated."""
    from types import SimpleNamespace

    from proteus.utils.data import needs_spider_ps_tables

    config = SimpleNamespace(
        interior_energetics=SimpleNamespace(module=energetics),
        interior_struct=SimpleNamespace(
            module=struct, eos_dir=eos_dir, zalmoxis=SimpleNamespace(mantle_eos=mantle_eos)
        ),
    )
    assert needs_spider_ps_tables(config) is expected
    # A set eos_dir asks for the set whenever SPIDER or Aragog run.
    config.interior_struct.eos_dir = 'WolfBower2018_MgSiO3'
    assert needs_spider_ps_tables(config) is (energetics in ('spider', 'aragog'))
    assert needs_spider_ps_tables({'fake': 'config'}) is False


@pytest.mark.unit
@patch('proteus.utils.data.download_zalmoxis_eos')
@patch('proteus.utils.data.download_eos_dynamic')
@patch('proteus.utils.data.download_melting_curves')
@patch('proteus.utils.data.download_interior_lookuptables')
@patch('proteus.utils.data.download_massradius_data')
@patch('proteus.utils.data.download_surface_albedos')
@patch('proteus.utils.data.download_exoplanet_data')
@patch('proteus.utils.data.download_stellar_spectra')
@patch('proteus.utils.data.download_spectral_file')
@patch('proteus.utils.data.download_phoenix')
def test_get_sufficient_zalmoxis_seager_only(
    _m_ph,
    _m_sp,
    _m_st,
    _m_ex,
    _m_sa,
    _m_mr,
    _m_il,
    _m_mc,
    mock_dyn,
    mock_zalmoxis_eos,
):
    """_get_sufficient calls download_zalmoxis_eos for Seager-only config."""
    from unittest.mock import MagicMock

    from proteus.utils.data import _get_sufficient

    config = MagicMock()
    config.interior_energetics.module = 'dummy'  # no spider/aragog
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.zalmoxis.mantle_eos = 'Seager2007:MgSiO3'
    config.interior_struct.zalmoxis.core_eos = 'Seager2007:iron'
    config.interior_struct.zalmoxis.ice_layer_eos = ''

    _get_sufficient(config, clean=False)

    mock_zalmoxis_eos.assert_called_once_with(
        mantle_eos='Seager2007:MgSiO3',
        core_eos='Seager2007:iron',
        ice_layer_eos='',
        volatile_eos='',
        anchor_pair=False,
    )
    mock_dyn.assert_not_called()


@pytest.mark.unit
@patch('proteus.utils.data.download_zalmoxis_eos')
@patch('proteus.utils.data.download_eos_dynamic')
@patch('proteus.utils.data.download_melting_curves')
@patch('proteus.utils.data.download_interior_lookuptables')
@patch('proteus.utils.data.download_massradius_data')
@patch('proteus.utils.data.download_surface_albedos')
@patch('proteus.utils.data.download_exoplanet_data')
@patch('proteus.utils.data.download_stellar_spectra')
@patch('proteus.utils.data.download_spectral_file')
@patch('proteus.utils.data.download_phoenix')
def test_get_sufficient_zalmoxis_paleos(
    _m_ph,
    _m_sp,
    _m_st,
    _m_ex,
    _m_sa,
    _m_mr,
    _m_il,
    _m_mc,
    mock_dyn,
    mock_zalmoxis_eos,
):
    """_get_sufficient calls download_zalmoxis_eos for PALEOS config."""
    from unittest.mock import MagicMock

    from proteus.utils.data import _get_sufficient

    config = MagicMock()
    config.interior_energetics.module = 'dummy'
    config.interior_struct.module = 'zalmoxis'
    config.interior_struct.zalmoxis.mantle_eos = 'PALEOS:MgSiO3'
    config.interior_struct.zalmoxis.core_eos = 'PALEOS:iron'
    config.interior_struct.zalmoxis.ice_layer_eos = 'PALEOS:H2O'

    _get_sufficient(config, clean=False)

    mock_zalmoxis_eos.assert_called_once_with(
        mantle_eos='PALEOS:MgSiO3',
        core_eos='PALEOS:iron',
        ice_layer_eos='PALEOS:H2O',
        volatile_eos='',
        anchor_pair=False,
    )
    mock_dyn.assert_not_called()


# ============================================================================
# test download_zalmoxis_eos dispatch
# ============================================================================


_UNIFIED_IRON = 'paleos_iron_eos_table_pt.dat'
_UNIFIED_MGSIO3 = 'paleos_mgsio3_eos_table_pt.dat'
_UNIFIED_WATER = 'paleos_water_eos_table_pt.dat'
_PAIR = (
    'paleos_mgsio3_tables_pt_proteus_liquid.dat',
    'paleos_mgsio3_tables_pt_proteus_solid.dat',
)


def _fetched_datasets(mock_fetch):
    """Dataset keys passed to a mocked ``fetch_dataset``, in call order."""
    return [c.args[0] for c in mock_fetch.call_args_list]


def _fetched_files(mock_file):
    """``(dataset key, file name)`` pairs passed to a mocked ``fetch_dataset_file``."""
    return [(c.args[0], c.args[1]) for c in mock_file.call_args_list]


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_seager(mock_static, mock_fetch, mock_file):
    """download_zalmoxis_eos for Seager2007 calls download_eos_static only."""
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('Seager2007:MgSiO3', core_eos='Seager2007:iron')
    download_zalmoxis_eos('Seager2007:MgSiO3', with_core=False)

    assert mock_static.call_count == 2
    mock_fetch.assert_not_called()
    mock_file.assert_not_called()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_wolfbower(mock_static, mock_fetch, mock_file):
    """WolfBower2018 fetches the Seager set and the five Wolf and Bower tables it reads,
    heat capacities included."""
    from proteus.data import EOS_WOLF_BOWER_2018
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('WolfBower2018:MgSiO3', core_eos='Seager2007:iron')

    mock_static.assert_called_once()
    # The shared record holds 8 tables; a whole-dataset fetch would pull all of them.
    mock_fetch.assert_not_called()
    assert _fetched_files(mock_file) == [
        (EOS_WOLF_BOWER_2018, 'density_melt.dat'),
        (EOS_WOLF_BOWER_2018, 'adiabat_temp_grad_melt.dat'),
        (EOS_WOLF_BOWER_2018, 'heat_capacity_melt.dat'),
        (EOS_WOLF_BOWER_2018, 'density_solid.dat'),
        (EOS_WOLF_BOWER_2018, 'heat_capacity_solid.dat'),
    ]


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_rtpress(mock_static, mock_fetch, mock_file):
    """RTPress100TPa fetches its three melt tables plus the Wolf and Bower solid density and
    heat capacity only."""
    from proteus.data import EOS_RTPRESS_100TPA, EOS_WOLF_BOWER_2018
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('RTPress100TPa:MgSiO3', core_eos='Seager2007:iron')

    mock_static.assert_called_once()
    mock_fetch.assert_not_called()
    # The RTPress registry entry reads its solid tables from the Wolf and
    # Bower dataset; the rest of that dataset must not be pulled for those files.
    assert _fetched_files(mock_file) == [
        (EOS_WOLF_BOWER_2018, 'density_solid.dat'),
        (EOS_WOLF_BOWER_2018, 'heat_capacity_solid.dat'),
        (EOS_RTPRESS_100TPA, 'density_melt.dat'),
        (EOS_RTPRESS_100TPA, 'adiabat_temp_grad_melt.dat'),
        (EOS_RTPRESS_100TPA, 'heat_capacity_melt.dat'),
    ]


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_rtpress_with_wolfbower_fetches_solid_once(
    mock_static, mock_fetch, mock_file
):
    """When Wolf and Bower is selected too, its tables cover the solid file once."""
    from proteus.data import EOS_RTPRESS_100TPA, EOS_WOLF_BOWER_2018
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos(
        'WolfBower2018:MgSiO3+RTPress100TPa:MgSiO3', core_eos='Seager2007:iron'
    )

    fetched = _fetched_files(mock_file)
    assert fetched.count((EOS_WOLF_BOWER_2018, 'density_solid.dat')) == 1
    assert {name for key, name in fetched if key == EOS_RTPRESS_100TPA} == {
        'density_melt.dat',
        'adiabat_temp_grad_melt.dat',
        'heat_capacity_melt.dat',
    }
    mock_fetch.assert_not_called()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_paleos_2phase(mock_static, mock_fetch, mock_file):
    """PALEOS-2phase:MgSiO3 fetches the standard-resolution pair only, not the unified table."""
    from proteus.data import EOS_PALEOS_MGSIO3
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('PALEOS-2phase:MgSiO3', core_eos='Seager2007:iron')

    # The shared record also holds the ~1.3 GB highres pair; it must stay unfetched.
    mock_fetch.assert_not_called()
    assert _fetched_files(mock_file) == [
        (EOS_PALEOS_MGSIO3, 'paleos_mgsio3_tables_pt_proteus_liquid.dat'),
        (EOS_PALEOS_MGSIO3, 'paleos_mgsio3_tables_pt_proteus_solid.dat'),
    ]


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_paleos_2phase_highres(mock_static, mock_fetch, mock_file):
    """PALEOS-2phase:MgSiO3-highres fetches the high-resolution pair only."""
    from proteus.data import EOS_PALEOS_MGSIO3
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('PALEOS-2phase:MgSiO3-highres', core_eos='Seager2007:iron')

    mock_fetch.assert_not_called()
    assert _fetched_files(mock_file) == [
        (EOS_PALEOS_MGSIO3, 'paleos_mgsio3_tables_pt_proteus_liquid_highres.dat'),
        (EOS_PALEOS_MGSIO3, 'paleos_mgsio3_tables_pt_proteus_solid_highres.dat'),
    ]


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_paleos_unified(mock_static, mock_fetch, mock_file):
    """PALEOS unified fetches each material's table from its own dataset, plus the
    MgSiO3 2-phase pair the mantle energetics read."""
    from proteus.data import (
        EOS_PALEOS_H2O,
        EOS_PALEOS_IRON,
        EOS_PALEOS_MGSIO3,
        EOS_PALEOS_MGSIO3_UNIFIED,
    )
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('PALEOS:MgSiO3', core_eos='PALEOS:iron', ice_layer_eos='PALEOS:H2O')

    # Seager not needed (no Seager component, core_eos is set)
    mock_static.assert_not_called()
    # A whole-dataset fetch would pull the full 2.29 GB record.
    mock_fetch.assert_not_called()
    assert sorted(_fetched_files(mock_file)) == sorted(
        [
            (EOS_PALEOS_IRON, _UNIFIED_IRON),
            (EOS_PALEOS_MGSIO3_UNIFIED, _UNIFIED_MGSIO3),
            (EOS_PALEOS_H2O, _UNIFIED_WATER),
            *((EOS_PALEOS_MGSIO3, name) for name in _PAIR),
        ]
    )


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_one_failed_file_does_not_stop_the_rest(
    mock_static, mock_fetch, caplog
):
    """An unreachable core table is logged and raised after every other table of the
    set was still attempted, so one fetch attempt gets all the reachable files."""
    from fwl_io import DownloadError

    from proteus.data import EOS_PALEOS_IRON, EOS_PALEOS_MGSIO3, EOS_PALEOS_MGSIO3_UNIFIED
    from proteus.utils.data import download_zalmoxis_eos

    attempted = []

    def _fetch(key, name, data_root=None):
        attempted.append((key, name))
        if key == EOS_PALEOS_IRON:
            raise DownloadError('could not obtain from any mirror')

    with patch('proteus.data.fetch_dataset_file', side_effect=_fetch):
        with caplog.at_level('WARNING'), pytest.raises(DownloadError):
            download_zalmoxis_eos('PALEOS:MgSiO3', core_eos='PALEOS:iron')

    assert attempted[0] == (EOS_PALEOS_IRON, _UNIFIED_IRON)
    assert sorted(attempted[1:]) == sorted(
        [
            (EOS_PALEOS_MGSIO3_UNIFIED, _UNIFIED_MGSIO3),
            *((EOS_PALEOS_MGSIO3, name) for name in _PAIR),
        ]
    )
    assert f'Could not fetch {_UNIFIED_IRON}' in caplog.text


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
def test_download_zalmoxis_eos_seager_failure_does_not_stop_the_pair(
    mock_fetch, mock_file, caplog
):
    """A 2-phase mantle whose Seager fallback set cannot be fetched still gets its pair
    and core table attempted, then the Seager error is raised."""
    from fwl_io import DownloadError

    from proteus.data import EOS_PALEOS_IRON, EOS_PALEOS_MGSIO3
    from proteus.utils.data import download_zalmoxis_eos

    with (
        patch(
            'proteus.utils.data.download_eos_static',
            side_effect=DownloadError('seager mirror down'),
        ),
        caplog.at_level('WARNING'),
        pytest.raises(DownloadError, match='seager mirror down'),
    ):
        download_zalmoxis_eos('PALEOS-2phase:MgSiO3', core_eos='PALEOS:iron')

    assert sorted(_fetched_files(mock_file)) == sorted(
        [(EOS_PALEOS_IRON, _UNIFIED_IRON), *((EOS_PALEOS_MGSIO3, name) for name in _PAIR)]
    )
    assert 'Could not fetch the Seager 2007 tables' in caplog.text


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_water_mantle_fetches_the_mgsio3_pair(
    mock_static, mock_fetch, mock_file
):
    """A PALEOS water mantle also fetches the MgSiO3 2-phase pair its energetics read,
    but not the MgSiO3 unified table and not the iron table, which it never reads."""
    from proteus.data import EOS_PALEOS_H2O, EOS_PALEOS_MGSIO3
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('PALEOS:H2O', core_eos='Seager2007:iron')

    mock_static.assert_called_once()
    mock_fetch.assert_not_called()
    assert sorted(_fetched_files(mock_file)) == sorted(
        [
            (EOS_PALEOS_H2O, _UNIFIED_WATER),
            *((EOS_PALEOS_MGSIO3, name) for name in _PAIR),
        ]
    )
    names = {name for _, name in _fetched_files(mock_file)}
    assert _UNIFIED_IRON not in names and _UNIFIED_MGSIO3 not in names


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_fetches_the_volatile_tables(mock_static, mock_fetch, mock_file):
    """The dissolved-volatile components add the water table and the Chabrier record."""
    from proteus.data import EOS_CHABRIER_2021, EOS_PALEOS_H2O
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos(
        'PALEOS:MgSiO3', core_eos='PALEOS:iron', volatile_eos='PALEOS:H2O+Chabrier:H'
    )
    assert (EOS_PALEOS_H2O, _UNIFIED_WATER) in _fetched_files(mock_file)
    assert [c.args[0] for c in mock_fetch.call_args_list] == [EOS_CHABRIER_2021]


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_strips_spaces_in_a_mixture(mock_static, mock_fetch, mock_file):
    """A spaced mixture fetches the table of every component."""
    from proteus.data import EOS_PALEOS_H2O, EOS_PALEOS_MGSIO3_UNIFIED
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos(' PALEOS:MgSiO3:0.9 + PALEOS:H2O:0.1 ', core_eos='PALEOS:iron')
    fetched = _fetched_files(mock_file)
    assert (EOS_PALEOS_H2O, _UNIFIED_WATER) in fetched
    assert (EOS_PALEOS_MGSIO3_UNIFIED, _UNIFIED_MGSIO3) in fetched


@pytest.mark.unit
@pytest.mark.parametrize(
    'dry, volatiles, mode, pair',
    [
        (True, '', 'adiabatic', False),
        (False, 'PALEOS:H2O+Chabrier:H', 'adiabatic', False),
        (True, '', 'liquidus_super', True),
        (True, '', 'adiabatic_from_cmb', True),
    ],
)
def test_download_zalmoxis_eos_for_config_adds_the_volatiles_of_a_wet_mantle(
    monkeypatch, dry, volatiles, mode, pair
):
    """With dry_mantle = false the fetch includes the dissolved-volatile EOS, and with a
    CMB or liquidus anchor the 2-phase pair it reads."""
    from types import SimpleNamespace

    from proteus.utils import data as dmod

    calls = []
    monkeypatch.setattr(dmod, 'download_zalmoxis_eos', lambda **kw: calls.append(kw))
    zconf = SimpleNamespace(
        mantle_eos='PALEOS:MgSiO3', core_eos='PALEOS:iron', ice_layer_eos=None, dry_mantle=dry
    )
    dmod.download_zalmoxis_eos_for_config(
        SimpleNamespace(
            interior_struct=SimpleNamespace(module='zalmoxis', zalmoxis=zconf),
            planet=SimpleNamespace(temperature_mode=mode),
        )
    )
    assert calls == [
        dict(
            mantle_eos='PALEOS:MgSiO3',
            core_eos='PALEOS:iron',
            ice_layer_eos='',
            volatile_eos=volatiles,
            anchor_pair=pair,
        )
    ]


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_fetches_the_anchor_pair_for_a_wolf_bower_mantle(
    mock_static, mock_fetch, mock_file, caplog
):
    """A Wolf and Bower mantle fetches the MgSiO3 2-phase pair only for an anchored
    temperature mode, and its component raises no unknown-family warning."""
    from proteus.data import EOS_PALEOS_MGSIO3
    from proteus.utils.data import download_zalmoxis_eos

    with caplog.at_level('WARNING', logger='fwl.proteus.utils.data'):
        download_zalmoxis_eos('WolfBower2018:MgSiO3', core_eos='PALEOS:iron')
    assert EOS_PALEOS_MGSIO3 not in {d for d, _ in _fetched_files(mock_file)}
    assert 'no handler for component' not in caplog.text
    download_zalmoxis_eos('WolfBower2018:MgSiO3', core_eos='PALEOS:iron', anchor_pair=True)
    assert EOS_PALEOS_MGSIO3 in {d for d, _ in _fetched_files(mock_file)}


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_paleos_unified_only_selected_tables(
    mock_static, mock_fetch, mock_file
):
    """Only the selected tables are fetched, not the whole record: no water table
    and no highres pair."""
    from proteus.data import EOS_PALEOS_IRON, EOS_PALEOS_MGSIO3, EOS_PALEOS_MGSIO3_UNIFIED
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('PALEOS:MgSiO3', core_eos='PALEOS:iron')

    assert sorted(_fetched_files(mock_file)) == sorted(
        [
            (EOS_PALEOS_IRON, _UNIFIED_IRON),
            (EOS_PALEOS_MGSIO3_UNIFIED, _UNIFIED_MGSIO3),
            *((EOS_PALEOS_MGSIO3, name) for name in _PAIR),
        ]
    )
    mock_fetch.assert_not_called()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_paleos_2phase_fetches_seager_fallback(
    mock_static, mock_fetch, mock_file
):
    """PALEOS-2phase with a PALEOS core still fetches the Seager static set.

    The registry entry for every multi-layer mantle family references the
    Seager iron table as its core fallback, and the start-of-run existence
    check requires every referenced file, so the fetch must cover the
    fallback even though neither the mantle nor the core names Seager. A
    fetch that skips it leaves a fresh install failing the existence check
    on its first run of the Earth tutorial config (PALEOS-2phase:MgSiO3
    mantle, PALEOS:iron core). The 2-phase tables and the unified iron
    table must download alongside.
    """
    from proteus.data import EOS_PALEOS_IRON, EOS_PALEOS_MGSIO3
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('PALEOS-2phase:MgSiO3', core_eos='PALEOS:iron')

    mock_static.assert_called_once()
    # The standard-resolution selection must not pull the ~1.3 GB highres pair.
    assert _fetched_files(mock_file) == [
        (EOS_PALEOS_IRON, _UNIFIED_IRON),
        (EOS_PALEOS_MGSIO3, 'paleos_mgsio3_tables_pt_proteus_liquid.dat'),
        (EOS_PALEOS_MGSIO3, 'paleos_mgsio3_tables_pt_proteus_solid.dat'),
    ]
    mock_fetch.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize(
    'mantle_component',
    [
        'WolfBower2018:MgSiO3',
        'RTPress100TPa:MgSiO3',
        'PALEOS-2phase:MgSiO3',
        'PALEOS-API-2phase:MgSiO3',
    ],
)
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_seager_fallback_every_family(
    mock_static, mock_fetch, mock_file, mantle_component
):
    """Every family with a Seager core fallback fetches the static set.

    Each family in SEAGER_FALLBACK_FAMILIES pairs here with a non-Seager
    core, so the fetch can only come from the fallback branch: dropping any
    family from the tuple fails its case. The Seager-core variants of the
    older tests cannot catch that regression because their explicit
    Seager2007 component triggers the direct branch instead. The PALEOS
    iron core must download its own table alongside, and the Chabrier
    fetch must stay untouched.
    """
    from proteus.data import EOS_CHABRIER_2021, EOS_PALEOS_IRON
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos(mantle_component, core_eos='PALEOS:iron')

    mock_static.assert_called_once()
    assert (EOS_PALEOS_IRON, _UNIFIED_IRON) in _fetched_files(mock_file)
    assert EOS_CHABRIER_2021 not in _fetched_datasets(mock_fetch)


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_api_2phase_fetches_only_seager(
    mock_static, mock_fetch, mock_file
):
    """PALEOS-API-2phase tabulates live but still needs the Seager fallback.

    The API-backed 2-phase mantle generates its own tables on demand, so no
    dataset fetch may fire for it, yet its registry entry carries the
    Seager iron core fallback whose file the existence check requires. The
    limit case of the fallback rule: the static set is the only download.
    """
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('PALEOS-API-2phase:MgSiO3', core_eos='PALEOS-API:iron')

    mock_static.assert_called_once()
    mock_fetch.assert_not_called()
    mock_file.assert_not_called()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_multi_component(mock_static, mock_fetch, mock_file):
    """download_zalmoxis_eos handles multi-component EOS strings."""
    from proteus.data import (
        EOS_CHABRIER_2021,
        EOS_PALEOS_H2O,
        EOS_PALEOS_MGSIO3,
        EOS_PALEOS_MGSIO3_UNIFIED,
    )
    from proteus.utils.data import download_zalmoxis_eos

    # Composite mantle with PALEOS + Chabrier
    download_zalmoxis_eos(
        'PALEOS:MgSiO3:0.98+Chabrier:H:0.01+PALEOS:H2O:0.01',
        core_eos='Seager2007:iron',
    )

    mock_static.assert_called_once()
    assert _fetched_datasets(mock_fetch) == [EOS_CHABRIER_2021]
    assert sorted(_fetched_files(mock_file)) == sorted(
        [
            (EOS_PALEOS_MGSIO3_UNIFIED, _UNIFIED_MGSIO3),
            (EOS_PALEOS_H2O, _UNIFIED_WATER),
            *((EOS_PALEOS_MGSIO3, name) for name in _PAIR),
        ]
    )


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_unknown_component_warns(
    mock_static, mock_fetch, mock_file, caplog
):
    """An unknown EOS family triggers a warning but no exception."""
    from proteus.utils.data import download_zalmoxis_eos

    with caplog.at_level('WARNING'):
        download_zalmoxis_eos('UnknownFamily:Foo', core_eos='Seager2007:iron')

    warnings = [r for r in caplog.records if 'no handler' in r.getMessage()]
    # Discrimination: at least one warning was emitted for the unknown
    # component, and nothing was fetched for it.
    assert len(warnings) >= 1
    assert any('UnknownFamily' in w.getMessage() for w in warnings)
    mock_fetch.assert_not_called()
    mock_file.assert_not_called()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_paleos_api_no_download(mock_static, mock_fetch, mock_file):
    """PALEOS-API:* and PALEOS-API-2phase:* are valid but trigger no dataset fetch."""
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('PALEOS-API:MgSiO3+PALEOS-API-2phase:H2O', core_eos='Seager2007:iron')

    # Seager static (always for core fallback) is the only download.
    mock_static.assert_called_once()
    mock_fetch.assert_not_called()
    mock_file.assert_not_called()


@pytest.mark.unit
def test_seager_fallback_families_match_registry():
    """SEAGER_FALLBACK_FAMILIES mirrors the registry's Seager fallbacks.

    The tuple in proteus.utils.data is a hand-maintained mirror of which
    multi-layer registry entries reference a Seager table; a family added
    to the registry without a tuple entry fails a fresh install's
    existence check on its first run, so the two must never drift. Parses
    the registry source (importing it would start the zalmoxis Julia
    bridge, far too heavy for a unit test) and collects every registry
    key whose entry dict maps a layer role to a ``_seager*`` table
    variable, then requires exact agreement with the tuple. The known
    2-phase family anchors the walk: if the registry moves away from a
    dict literal, the anchor fails loudly and this guard needs a rewrite
    rather than silently passing on an empty set.
    """
    import ast

    import proteus.utils.data as data_module
    from proteus.utils.data import SEAGER_FALLBACK_FAMILIES

    registry_path = Path(data_module.__file__).parents[1] / 'interior_struct' / 'zalmoxis.py'
    src = registry_path.read_text()
    expected = set()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if ':' not in key.value or not isinstance(value, ast.Dict):
                continue
            refs_seager = any(
                isinstance(sub, ast.Name) and sub.id.startswith('_seager')
                for sub in value.values
            )
            if refs_seager:
                expected.add(key.value.split(':')[0] + ':')
    expected -= {'Seager2007:'}

    assert 'PALEOS-2phase:' in expected
    assert expected == set(SEAGER_FALLBACK_FAMILIES)


# ============================================================================
# test download_eos_dynamic / download_eos_static
# ============================================================================


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_eos_dynamic_fetches_lookup_dataset(mock_fetch, tmp_path, monkeypatch):
    """download_eos_dynamic fetches the Wolf and Bower P-S lookup dataset through fwl-io."""
    from proteus.data import LOOKUP_WOLF_BOWER_2018_1TPA, dataset_dir
    from proteus.utils.data import download_eos_dynamic

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    target_dir = dataset_dir(LOOKUP_WOLF_BOWER_2018_1TPA, data_root=tmp_path)
    mock_fetch.return_value = [target_dir / 'temperature_melt.dat']

    download_eos_dynamic('WolfBower2018_MgSiO3')

    mock_fetch.assert_called_once_with(LOOKUP_WOLF_BOWER_2018_1TPA, data_root=tmp_path)
    assert mock_fetch.call_args.args[0] == (
        'interior.eos.dk09_1tpa_elec_free.mgsio3_wolf_bower_2018_1tpa'
    )


@pytest.mark.unit
@patch('proteus.utils.data.download_Seager_EOS')
def test_download_eos_static_delegates(mock_seager):
    """download_eos_static delegates to download_Seager_EOS."""
    from proteus.utils.data import download_eos_static

    download_eos_static()
    mock_seager.assert_called_once()
    # Discrimination: confirm delegation passes no positional/keyword args
    # (the static helper is expected to call the upstream Seager downloader
    # with its own defaults); a regression that forwarded a stray arg would
    # break this pin.
    assert mock_seager.call_args == ((), {})


# ============================================================================
# AGNI spectral_file dispatch: skip group/bands download when user-provided
# ============================================================================


@pytest.mark.unit
def test_get_sufficient_agni_skips_group_band_lookup_when_spectral_file_set(monkeypatch):
    """When AGNI receives a user-supplied spectral_file (either a custom
    path or 'greygas'), _get_sufficient must NOT call
    get_spfile_name_and_bands or queue a second download_spectral_file
    call. The Honeyside post-processing file is always downloaded.
    """
    from types import SimpleNamespace

    import proteus.atmos_clim.common as atmos_common
    import proteus.utils.data as data_mod

    spectral_calls = []
    lookup_calls = []

    monkeypatch.setattr(
        data_mod,
        'download_spectral_file',
        lambda group, bands: spectral_calls.append((group, bands)),
    )
    monkeypatch.setattr(data_mod, 'download_stellar_spectra', lambda *args, **kwargs: None)
    monkeypatch.setattr(data_mod, 'download_stellar_tracks', lambda *args, **kwargs: None)
    monkeypatch.setattr(data_mod, 'download_surface_albedos', lambda: None)
    monkeypatch.setattr(data_mod, 'download_scattering', lambda: None)
    monkeypatch.setattr(data_mod, 'download_exoplanet_data', lambda: None)
    monkeypatch.setattr(data_mod, 'download_massradius_data', lambda: None)
    monkeypatch.setattr(
        data_mod, 'download_interior_lookuptables', lambda *args, **kwargs: None
    )
    monkeypatch.setattr(data_mod, 'download_melting_curves', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        atmos_common,
        'get_spfile_name_and_bands',
        lambda *_args: lookup_calls.append(1),
    )

    config = SimpleNamespace(
        star=SimpleNamespace(module='dummy'),
        atmos_clim=SimpleNamespace(
            module='agni',
            aerosols_enabled=False,
            agni=SimpleNamespace(spectral_file='/tmp/custom.spc'),
        ),
        interior_energetics=SimpleNamespace(module='dummy'),
        interior_struct=SimpleNamespace(module='dummy'),
    )

    data_mod._get_sufficient(config, clean=False)

    # Only the Honeyside high-res post-processing file gets downloaded.
    assert spectral_calls == [('Honeyside', '4096')]
    # The group/bands lookup was bypassed.
    assert lookup_calls == []


@pytest.mark.unit
def test_get_sufficient_agni_downloads_group_and_bands_when_no_spectral_file(monkeypatch):
    """When spectral_file is None (the default), _get_sufficient must
    resolve group/bands via get_spfile_name_and_bands and queue the
    matching spectral-file download."""
    from types import SimpleNamespace

    import proteus.atmos_clim.common as atmos_common
    import proteus.utils.data as data_mod

    spectral_calls = []

    monkeypatch.setattr(
        data_mod,
        'download_spectral_file',
        lambda group, bands: spectral_calls.append((group, bands)),
    )
    monkeypatch.setattr(data_mod, 'download_stellar_spectra', lambda *args, **kwargs: None)
    monkeypatch.setattr(data_mod, 'download_stellar_tracks', lambda *args, **kwargs: None)
    monkeypatch.setattr(data_mod, 'download_surface_albedos', lambda: None)
    monkeypatch.setattr(data_mod, 'download_scattering', lambda: None)
    monkeypatch.setattr(data_mod, 'download_exoplanet_data', lambda: None)
    monkeypatch.setattr(data_mod, 'download_massradius_data', lambda: None)
    monkeypatch.setattr(
        data_mod, 'download_interior_lookuptables', lambda *args, **kwargs: None
    )
    monkeypatch.setattr(data_mod, 'download_melting_curves', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        atmos_common,
        'get_spfile_name_and_bands',
        lambda *_args: ('Frostflow', '256'),
    )

    config = SimpleNamespace(
        star=SimpleNamespace(module='dummy'),
        atmos_clim=SimpleNamespace(
            module='agni',
            aerosols_enabled=False,
            agni=SimpleNamespace(spectral_file=None),
        ),
        interior_energetics=SimpleNamespace(module='dummy'),
        interior_struct=SimpleNamespace(module='dummy'),
    )

    data_mod._get_sufficient(config, clean=False)

    # Honeyside post-processing file + the resolved group/bands file.
    assert spectral_calls == [('Honeyside', '4096'), ('Frostflow', '256')]
    # Discrimination: confirm the resolved group/bands entry came second
    # (the Honeyside post-processing download always runs first); a
    # regression that swapped the call order or replayed Honeyside twice
    # would still produce a 2-element list but break this pin.
    assert spectral_calls[-1] == ('Frostflow', '256')


@pytest.mark.unit
def test_get_sufficient_janus_always_downloads_group_and_bands(monkeypatch):
    """JANUS does not have a custom-spectral-file branch (it predates the
    grey-gas feature). The new conditional must NOT bypass the lookup
    for module='janus' even if the test config happens to expose a
    spectral_file attribute somewhere."""
    from types import SimpleNamespace

    import proteus.atmos_clim.common as atmos_common
    import proteus.utils.data as data_mod

    spectral_calls = []
    lookup_calls = []

    monkeypatch.setattr(
        data_mod,
        'download_spectral_file',
        lambda group, bands: spectral_calls.append((group, bands)),
    )
    monkeypatch.setattr(data_mod, 'download_stellar_spectra', lambda *args, **kwargs: None)
    monkeypatch.setattr(data_mod, 'download_stellar_tracks', lambda *args, **kwargs: None)
    monkeypatch.setattr(data_mod, 'download_surface_albedos', lambda: None)
    monkeypatch.setattr(data_mod, 'download_scattering', lambda: None)
    monkeypatch.setattr(data_mod, 'download_exoplanet_data', lambda: None)
    monkeypatch.setattr(data_mod, 'download_massradius_data', lambda: None)
    monkeypatch.setattr(
        data_mod, 'download_interior_lookuptables', lambda *args, **kwargs: None
    )
    monkeypatch.setattr(data_mod, 'download_melting_curves', lambda *args, **kwargs: None)

    def _record_lookup(*_args):
        lookup_calls.append(1)
        return ('Frostflow', '256')

    monkeypatch.setattr(atmos_common, 'get_spfile_name_and_bands', _record_lookup)

    config = SimpleNamespace(
        star=SimpleNamespace(module='dummy'),
        atmos_clim=SimpleNamespace(module='janus', aerosols_enabled=False),
        interior_energetics=SimpleNamespace(module='dummy'),
        interior_struct=SimpleNamespace(module='dummy'),
    )

    data_mod._get_sufficient(config, clean=False)

    assert ('Frostflow', '256') in spectral_calls
    assert lookup_calls == [1]

    # A group no manifest declares is read from the local tree, not downloaded,
    # since download_spectral_file refuses it with a ValueError.
    monkeypatch.setattr(atmos_common, 'get_spfile_name_and_bands', lambda *_: ('MyGroup', '48'))
    spectral_calls.clear()
    data_mod._get_sufficient(config, clean=False)
    assert spectral_calls == [('Honeyside', '4096')]


# ============================================================================
# download_melting_curves additional coverage
# ============================================================================


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
def test_download_melting_curves_none_dir_is_noop(mock_getfwl, tmp_path):
    """When melting_dir is None, the function returns early and fetches nothing."""
    from unittest.mock import MagicMock

    from proteus.utils.data import download_melting_curves

    mock_getfwl.return_value = tmp_path
    config = MagicMock()
    config.interior_struct.melting_dir = None

    with patch('proteus.data.fetch_dataset') as mock_fetch:
        download_melting_curves(config, clean=False)

    mock_fetch.assert_not_called()
    # The early return precedes the directory probe as well.
    mock_getfwl.assert_not_called()


# ============================================================================
# download_eos_dynamic manifest validation coverage
# ============================================================================


_LOOKUP_EXPECTED_FILES = [
    'temperature_melt.dat',
    'temperature_solid.dat',
    'density_melt.dat',
    'density_solid.dat',
    'heat_capacity_melt.dat',
    'heat_capacity_solid.dat',
    'adiabat_temp_grad_melt.dat',
    'adiabat_temp_grad_solid.dat',
    'thermal_exp_melt.dat',
    'thermal_exp_solid.dat',
    'solidus_P-S.dat',
    'liquidus_P-S.dat',
]


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_eos_dynamic_manifest_complete_no_warning(
    mock_fetch, tmp_path, monkeypatch, caplog
):
    """When all 12 expected files are present, no missing-file warning fires."""
    from proteus.data import LOOKUP_WOLF_BOWER_2018_1TPA, dataset_dir
    from proteus.utils.data import download_eos_dynamic

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    target_dir = dataset_dir(LOOKUP_WOLF_BOWER_2018_1TPA, data_root=tmp_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    for fname in _LOOKUP_EXPECTED_FILES:
        (target_dir / fname).write_text('data')
    mock_fetch.return_value = [target_dir / fname for fname in _LOOKUP_EXPECTED_FILES]

    with caplog.at_level('WARNING'):
        download_eos_dynamic('WolfBower2018_MgSiO3')

    assert [r for r in caplog.records if 'missing' in r.getMessage().lower()] == []
    mock_fetch.assert_called_once()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_eos_dynamic_manifest_incomplete_warns(
    mock_fetch, tmp_path, monkeypatch, caplog
):
    """When files are missing, the warning names the count of missing files."""
    from proteus.data import LOOKUP_WOLF_BOWER_2018_1TPA, dataset_dir
    from proteus.utils.data import download_eos_dynamic

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    target_dir = dataset_dir(LOOKUP_WOLF_BOWER_2018_1TPA, data_root=tmp_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / 'density_melt.dat').write_text('data')
    (target_dir / 'density_solid.dat').write_text('data')
    mock_fetch.return_value = [
        target_dir / 'density_melt.dat',
        target_dir / 'density_solid.dat',
    ]

    with caplog.at_level('WARNING'):
        download_eos_dynamic('WolfBower2018_MgSiO3')

    messages = ' '.join(r.getMessage() for r in caplog.records)
    assert 'missing 10 of 12' in messages
    mock_fetch.assert_called_once()


# ============================================================================
# download_stellar_tracks coverage
# ============================================================================


def _fake_mors(monkeypatch, dirs, download=lambda track: None):
    """Install a stand-in ``mors.data`` whose accessors return ``dirs`` and log their use."""
    import sys
    import types

    used = []
    fake_data = types.ModuleType('mors.data')
    fake_data.DownloadEvolutionTracks = download
    fake_data.baraffe_data_dir = lambda: used.append('Baraffe') or dirs['Baraffe']
    fake_data.spada_data_dir = lambda: used.append('Spada') or dirs['Spada']
    fake_mors = types.ModuleType('mors')
    fake_mors.data = fake_data
    monkeypatch.setitem(sys.modules, 'mors', fake_mors)
    monkeypatch.setitem(sys.modules, 'mors.data', fake_data)
    return used


@pytest.mark.unit
@pytest.mark.parametrize('track', ['Baraffe', 'Spada'])
def test_download_stellar_tracks_checks_the_track_directory(tmp_path, monkeypatch, track):
    """MORS fetches the named set, and the result is checked in that set's directory."""
    import proteus.utils.data as data_mod

    dirs = {name: tmp_path / name for name in ('Baraffe', 'Spada')}
    dirs[track].mkdir()
    (dirs[track] / 'track.dat').write_text('track')
    fetched = []
    used = _fake_mors(monkeypatch, dirs, download=fetched.append)

    data_mod.download_stellar_tracks(track)

    assert fetched == [track]
    # Discrimination: only the requested set's directory is consulted.
    assert used == [track]


@pytest.mark.unit
@pytest.mark.parametrize('track', ['', 'spada', 'Phoenix'])
def test_download_stellar_tracks_rejects_an_unknown_set_before_fetching(
    tmp_path, monkeypatch, track
):
    """A name other than Spada or Baraffe raises before MORS downloads anything."""
    import proteus.utils.data as data_mod

    fetched = []
    used = _fake_mors(monkeypatch, {'Baraffe': tmp_path, 'Spada': tmp_path}, fetched.append)

    with pytest.raises(ValueError, match='Unknown stellar track set'):
        data_mod.download_stellar_tracks(track)
    assert fetched == []
    assert used == []


@pytest.mark.unit
@pytest.mark.parametrize('present', [False, True])
def test_download_stellar_tracks_refuses_a_missing_or_empty_directory(
    tmp_path, monkeypatch, present
):
    """A fetch that leaves the track directory missing or empty raises FileNotFoundError."""
    import proteus.utils.data as data_mod

    dirs = {name: tmp_path / name for name in ('Baraffe', 'Spada')}
    if present:
        dirs['Spada'].mkdir()
    _fake_mors(monkeypatch, dirs)

    with pytest.raises(FileNotFoundError, match='empty or missing') as raised:
        data_mod.download_stellar_tracks('Spada')
    assert str(dirs['Spada']) in str(raised.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    'error', [ConnectionError('mirror down'), RuntimeError('upgrade to fwl-io>=26.10.6')]
)
def test_download_stellar_tracks_passes_mors_errors_unchanged(tmp_path, monkeypatch, error):
    """A failed download or another MORS error reaches the caller as MORS raised it."""
    import proteus.utils.data as data_mod

    def fail(track):
        raise error

    used = _fake_mors(monkeypatch, {'Baraffe': tmp_path, 'Spada': tmp_path}, download=fail)

    with pytest.raises(type(error)) as raised:
        data_mod.download_stellar_tracks('Spada')
    assert raised.value is error
    assert used == []


# ============================================================================
# download_sufficient_data orchestration
# ============================================================================


@pytest.mark.unit
def test_download_sufficient_data_offline_skips_download(monkeypatch):
    """When config.params.offline is True, _get_sufficient is NOT invoked."""
    from unittest.mock import MagicMock

    import proteus.utils.data as data_mod
    from proteus.utils.data import download_sufficient_data

    called = []
    monkeypatch.setattr(data_mod, '_get_sufficient', lambda *a, **k: called.append('x'))

    config = MagicMock()
    config.params.offline = True

    download_sufficient_data(config, clean=False)

    assert called == []
    # Discrimination: the function returned cleanly; no exception raised
    # and the orchestration helper never ran.
    assert config.params.offline is True


@pytest.mark.unit
def test_download_sufficient_data_oserror_is_caught(monkeypatch, caplog):
    """When _get_sufficient raises OSError, it is logged and swallowed (not propagated)."""
    from unittest.mock import MagicMock

    import proteus.utils.data as data_mod
    from proteus.utils.data import download_sufficient_data

    call_count = {'n': 0}

    def raising(*a, **k):
        call_count['n'] += 1
        raise OSError('network gone')

    monkeypatch.setattr(data_mod, '_get_sufficient', raising)

    config = MagicMock()
    config.params.offline = False

    with caplog.at_level('WARNING'):
        # Should NOT raise
        download_sufficient_data(config, clean=False)
    # Discrimination: _get_sufficient was invoked exactly once (so the
    # offline branch did NOT short-circuit). The OSError was caught and
    # logged as a warning, not re-raised.
    assert call_count['n'] == 1
    messages = ' '.join(r.getMessage() for r in caplog.records)
    assert 'network gone' in messages or 'Problem when downloading' in messages


@pytest.mark.unit
def test_download_sufficient_data_non_oserror_propagates(monkeypatch):
    """Errors other than OSError are NOT caught and should propagate."""
    from unittest.mock import MagicMock

    import proteus.utils.data as data_mod
    from proteus.utils.data import download_sufficient_data

    call_count = {'n': 0}

    def raising(*a, **k):
        call_count['n'] += 1
        raise RuntimeError('something else')

    monkeypatch.setattr(data_mod, '_get_sufficient', raising)

    config = MagicMock()
    config.params.offline = False

    with pytest.raises(RuntimeError, match='something else'):
        download_sufficient_data(config, clean=False)
    # Discrimination: _get_sufficient was invoked once; RuntimeError
    # propagated rather than being caught like OSError would.
    assert call_count['n'] == 1


# ============================================================================
# _none_dirs / get_socrates default-dirs path
# ============================================================================


@pytest.mark.unit
def test_none_dirs_returns_proteus_and_tools_paths(monkeypatch):
    """_none_dirs returns a dict with 'proteus' and 'tools' keys."""
    import os

    import proteus.utils.data as data_mod

    fake_proteus_root = '/fake/proteus/root'

    import proteus.utils.helper as helper_mod

    monkeypatch.setattr(helper_mod, 'get_proteus_dir', lambda: fake_proteus_root)

    dirs = data_mod._none_dirs()

    assert dirs['proteus'] == fake_proteus_root
    # Discrimination: tools must be a path under proteus, not a sibling.
    assert dirs['tools'] == os.path.join(fake_proteus_root, 'tools')
    assert dirs['tools'].startswith(fake_proteus_root)


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
def test_get_socrates_skips_when_workpath_exists(mock_run, tmp_path):
    """get_socrates returns early when the socrates workpath already exists.

    The lowercase directory name matters: it matches install.sh, the CI
    action, and RAD_DIR, and on case-sensitive filesystems an uppercase
    directory would not satisfy the check.
    """
    from proteus.utils.data import get_socrates

    workpath = tmp_path / 'socrates'
    workpath.mkdir()

    dirs = {'proteus': str(tmp_path), 'tools': str(tmp_path / 'tools')}
    get_socrates(dirs=dirs)

    # Discrimination: no subprocess should have run (the dir exists),
    # and the function did not delete the existing workpath either.
    mock_run.assert_not_called()
    assert workpath.is_dir()


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
def test_get_socrates_runs_setup_script_when_missing(mock_run, tmp_path, monkeypatch):
    """get_socrates invokes get_socrates.sh and sets RAD_DIR when workpath is missing."""
    import os

    from proteus.utils.data import get_socrates

    dirs = {'proteus': str(tmp_path), 'tools': str(tmp_path / 'tools')}
    (tmp_path / 'tools').mkdir()

    # Make sure RAD_DIR is unset to start
    monkeypatch.delenv('RAD_DIR', raising=False)

    get_socrates(dirs=dirs)

    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    # Discrimination: the second arg is the workpath. Lowercase matches
    # install.sh, the CI action, and RAD_DIR; an uppercase path would
    # create a second checkout on case-sensitive filesystems.
    assert cmd[0].endswith('get_socrates.sh')
    assert cmd[1] == os.path.abspath(os.path.join(str(tmp_path), 'socrates'))
    # RAD_DIR was set by the function
    assert os.environ.get('RAD_DIR', '').endswith('socrates')


# ============================================================================
# load_melting_curve / get_zalmoxis_melting_curves
# ============================================================================


@pytest.mark.unit
def test_load_melting_curve_returns_interpolator(tmp_path):
    """load_melting_curve returns a callable interpolator that linearly maps P->T."""
    import numpy as np

    from proteus.utils.data import load_melting_curve

    melt_file = tmp_path / 'melt.dat'
    # 4 anchor points: linear T(P): T = 1000 + P / 1e6
    melt_file.write_text(
        '# header line\n1.0e9   2000.0\n2.0e9   3000.0\n3.0e9   4000.0\n4.0e9   5000.0\n'
    )

    f = load_melting_curve(melt_file)
    assert f is not None

    # Linear interpolation between (1e9, 2000) and (2e9, 3000) at P=1.5e9
    # should give T=2500.
    val = float(f(1.5e9))
    assert val == pytest.approx(2500.0, rel=1e-12)
    # Discrimination: a discrete (nearest) interpolation would have
    # returned 2000 or 3000; our 2500 confirms LINEAR mode.
    assert 2400.0 < val < 2600.0
    # Out-of-bounds returns NaN (fill_value=np.nan, bounds_error=False)
    assert np.isnan(float(f(1e6)))


@pytest.mark.unit
def test_load_melting_curve_returns_none_on_bad_file(tmp_path, caplog):
    """load_melting_curve returns None when the file cannot be parsed."""
    import logging

    from proteus.utils.data import load_melting_curve

    missing = tmp_path / 'no_such_file.dat'
    with caplog.at_level(logging.ERROR, logger='fwl'):
        result = load_melting_curve(missing)
    assert result is None
    assert 'Error loading melting curve data' in caplog.text


@pytest.mark.unit
def test_get_zalmoxis_melting_curves_none_dir_returns_none(monkeypatch, tmp_path):
    """When melting_dir is None, the helper returns None and does not consult the disk."""
    from unittest.mock import MagicMock

    import proteus.utils.data as data_mod
    from proteus.utils.data import get_zalmoxis_melting_curves

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)

    # Create the directory that the function would otherwise consult;
    # this lets us verify the early return does not touch disk.
    curves_dir = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'X'
    curves_dir.mkdir(parents=True, exist_ok=True)
    sentinel = curves_dir / 'sentinel.dat'
    sentinel.write_text('untouched')

    config = MagicMock()
    config.interior_struct.melting_dir = None

    result = get_zalmoxis_melting_curves(config)
    # Discrimination: result is the early-return sentinel and the
    # filesystem under FWL_DATA_DIR was not modified.
    assert result is None
    assert sentinel.read_text() == 'untouched'


@pytest.mark.unit
def test_get_zalmoxis_melting_curves_missing_dir_raises(monkeypatch, tmp_path):
    """When a free-form melting curve directory is absent, FileNotFoundError names the file."""
    import proteus.utils.data as data_mod
    from proteus.utils.data import get_zalmoxis_melting_curves

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)

    config = MagicMock()
    config.interior_struct.melting_dir = 'NonexistentCurve'

    with pytest.raises(
        FileNotFoundError, match='Melting curve file not found.*proteus get interiordata'
    ) as exc:
        get_zalmoxis_melting_curves(config)
    assert '`fwl-io relocate`' in str(exc.value)
    missing = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'NonexistentCurve'
    assert not missing.exists()


@pytest.mark.unit
def test_get_zalmoxis_melting_curves_reads_manifest_dataset(monkeypatch, tmp_path):
    """A known name without a local directory is read from its versioned dataset."""
    import proteus.utils.data as data_mod
    from proteus.data import MELTING_MONTEUX_PLUS600, dataset_dir
    from proteus.utils.data import get_zalmoxis_melting_curves

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    folder = dataset_dir(MELTING_MONTEUX_PLUS600, data_root=tmp_path)
    folder.mkdir(parents=True)
    (folder / 'solidus.dat').write_text('1e9 2000\n2e9 3000\n')
    (folder / 'liquidus.dat').write_text('1e9 2500\n2e9 3500\n')

    config = MagicMock()
    config.interior_struct.melting_dir = 'Monteux+600'

    sol, liq = get_zalmoxis_melting_curves(config)

    assert float(sol(1.5e9)) == pytest.approx(2500.0, rel=1e-12)
    assert float(liq(1.5e9)) == pytest.approx(3000.0, rel=1e-12)


@pytest.mark.unit
def test_get_zalmoxis_melting_curves_manifest_beats_local_dir(monkeypatch, tmp_path):
    """When both exist for a served name, the dataset files are read, not the local ones."""
    import proteus.utils.data as data_mod
    from proteus.data import MELTING_MONTEUX_PLUS600, dataset_dir
    from proteus.utils.data import get_zalmoxis_melting_curves

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    folder = dataset_dir(MELTING_MONTEUX_PLUS600, data_root=tmp_path)
    folder.mkdir(parents=True)
    (folder / 'solidus.dat').write_text('1e9 2000\n2e9 3000\n')
    (folder / 'liquidus.dat').write_text('1e9 2500\n2e9 3500\n')
    local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Monteux+600'
    local.mkdir(parents=True)
    (local / 'solidus_P-T.dat').write_text('1e9 1000\n2e9 1000\n')
    (local / 'liquidus_P-T.dat').write_text('1e9 1000\n2e9 1000\n')

    config = MagicMock()
    config.interior_struct.melting_dir = 'Monteux+600'

    sol, liq = get_zalmoxis_melting_curves(config)

    assert float(sol(1.5e9)) == pytest.approx(2500.0, rel=1e-12)
    assert float(liq(1.5e9)) == pytest.approx(3000.0, rel=1e-12)


@pytest.mark.unit
def test_get_zalmoxis_melting_curves_returns_two_interpolators(monkeypatch, tmp_path):
    """When the directory and files exist, the helper returns (solidus, liquidus) callables."""
    from unittest.mock import MagicMock

    import proteus.utils.data as data_mod
    from proteus.utils.data import get_zalmoxis_melting_curves

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)

    curves_dir = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Wolf_Bower+2018'
    curves_dir.mkdir(parents=True, exist_ok=True)

    (curves_dir / 'solidus_P-T.dat').write_text('1e9 2000\n2e9 3000\n')
    (curves_dir / 'liquidus_P-T.dat').write_text('1e9 2500\n2e9 3500\n')

    config = MagicMock()
    config.interior_struct.melting_dir = 'Wolf_Bower+2018'

    sol, liq = get_zalmoxis_melting_curves(config)
    assert sol is not None and liq is not None

    # Mid-point T values for the two curves at P=1.5e9
    s_val = float(sol(1.5e9))
    l_val = float(liq(1.5e9))
    # Discrimination: solidus should be cooler than liquidus at the same
    # pressure, otherwise something has gone wrong with file ordering.
    assert s_val == pytest.approx(2500.0, rel=1e-12)
    assert l_val == pytest.approx(3000.0, rel=1e-12)
    assert s_val < l_val


# ============================================================================
# get_zalmoxis_EOS branches
# ============================================================================


_SEAGER_FILES = (
    'eos_seager07_iron.txt',
    'eos_seager07_silicate.txt',
    'eos_seager07_water.txt',
)


def _seed_seager(root):
    """Create the Seager EOS files in their versioned dataset directory under ``root``."""
    from proteus.data import EOS_SEAGER_2007, dataset_dir

    folder = dataset_dir(EOS_SEAGER_2007, data_root=root)
    folder.mkdir(parents=True, exist_ok=True)
    for fname in _SEAGER_FILES:
        (folder / fname).write_text('eos')
    return folder


@pytest.mark.unit
def test_get_zalmoxis_EOS_seager_versioned_dir_used(monkeypatch, tmp_path):
    """The Seager tables are read from their versioned fwl-io dataset directory."""
    import proteus.utils.data as data_mod
    from proteus.utils.data import get_zalmoxis_EOS

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    seager = _seed_seager(tmp_path)

    iron_silicate, _, water, _ = get_zalmoxis_EOS()

    assert Path(iron_silicate['core']['eos_file']) == seager / 'eos_seager07_iron.txt'
    assert Path(iron_silicate['mantle']['eos_file']) == seager / 'eos_seager07_silicate.txt'
    assert Path(water['ice_layer']['eos_file']) == seager / 'eos_seager07_water.txt'
    assert seager.parts[-4:-1] == ('interior_struct', 'eos', 'seager_2007')
    assert seager.name.startswith('r')


@pytest.mark.unit
def test_get_zalmoxis_EOS_seager_missing_fetches(monkeypatch, tmp_path):
    """Absent Seager files trigger one fetch through download_eos_static."""
    import proteus.utils.data as data_mod
    from proteus.data import EOS_SEAGER_2007, dataset_dir
    from proteus.utils.data import get_zalmoxis_EOS

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    calls = []
    monkeypatch.setattr(data_mod, 'download_eos_static', lambda: calls.append(1))

    eos = get_zalmoxis_EOS()

    assert calls == [1]
    seager = dataset_dir(EOS_SEAGER_2007, data_root=tmp_path)
    assert eos[0]['core']['eos_file'] == str(seager / 'eos_seager07_iron.txt')


@pytest.mark.unit
def test_get_zalmoxis_EOS_wb_versioned_dir_with_optional_tables(monkeypatch, tmp_path):
    """Wolf and Bower paths point into the versioned dataset; optional tables are listed."""
    import proteus.utils.data as data_mod
    from proteus.data import EOS_WOLF_BOWER_2018, dataset_dir
    from proteus.utils.data import get_zalmoxis_EOS

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    _seed_seager(tmp_path)
    wb = dataset_dir(EOS_WOLF_BOWER_2018, data_root=tmp_path)
    wb.mkdir(parents=True)
    (wb / 'heat_capacity_melt.dat').write_text('cp')
    (wb / 'heat_capacity_solid.dat').write_text('cp')
    (wb / 'adiabat_temp_grad_melt.dat').write_text('grad')

    _, iron_Tdep, _, _ = get_zalmoxis_EOS()

    assert 'cp_file' in iron_Tdep['melted_mantle']
    assert 'adiabat_grad_file' in iron_Tdep['melted_mantle']
    assert 'cp_file' in iron_Tdep['solid_mantle']
    assert Path(iron_Tdep['melted_mantle']['eos_file']) == wb / 'density_melt.dat'
    assert Path(iron_Tdep['solid_mantle']['eos_file']) == wb / 'density_solid.dat'


@pytest.mark.unit
def test_get_zalmoxis_EOS_wb_without_optional_tables_omits_keys(monkeypatch, tmp_path):
    """Without the optional Cp and gradient tables the dict omits their keys."""
    import proteus.utils.data as data_mod
    from proteus.data import EOS_WOLF_BOWER_2018, dataset_dir
    from proteus.utils.data import get_zalmoxis_EOS

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    _seed_seager(tmp_path)
    dataset_dir(EOS_WOLF_BOWER_2018, data_root=tmp_path).mkdir(parents=True)

    _, iron_Tdep, _, _ = get_zalmoxis_EOS()

    assert 'cp_file' not in iron_Tdep['melted_mantle']
    assert 'adiabat_grad_file' not in iron_Tdep['melted_mantle']
    assert 'cp_file' not in iron_Tdep['solid_mantle']


@pytest.mark.unit
def test_get_zalmoxis_EOS_rt_versioned_folder_used(monkeypatch, tmp_path, caplog):
    """The RTPress tables are read from their versioned dataset directory."""
    import proteus.utils.data as data_mod
    from proteus.data import EOS_RTPRESS_100TPA, dataset_dir
    from proteus.utils.data import get_zalmoxis_EOS

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    _seed_seager(tmp_path)
    rt = dataset_dir(EOS_RTPRESS_100TPA, data_root=tmp_path)
    rt.mkdir(parents=True)
    (rt / 'heat_capacity_melt.dat').write_text('cp')
    (rt / 'adiabat_temp_grad_melt.dat').write_text('grad')

    with caplog.at_level('WARNING'):
        _, _, _, iron_rt = get_zalmoxis_EOS()

    assert Path(iron_rt['melted_mantle']['eos_file']) == rt / 'density_melt.dat'
    assert Path(iron_rt['melted_mantle']['cp_file']) == rt / 'heat_capacity_melt.dat'
    assert (
        Path(iron_rt['melted_mantle']['adiabat_grad_file']) == rt / 'adiabat_temp_grad_melt.dat'
    )
    assert [r for r in caplog.records if 'EOS folder not found' in r.getMessage()] == []


@pytest.mark.unit
def test_get_zalmoxis_EOS_rt_missing_cp_warns(monkeypatch, tmp_path, caplog):
    """When the RTPress Cp table is missing, a warning fires and the dict omits cp_file."""
    import proteus.utils.data as data_mod
    from proteus.data import EOS_RTPRESS_100TPA, dataset_dir
    from proteus.utils.data import get_zalmoxis_EOS

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    _seed_seager(tmp_path)
    # The versioned folder exists but holds no heat_capacity_melt.dat.
    dataset_dir(EOS_RTPRESS_100TPA, data_root=tmp_path).mkdir(parents=True)

    with caplog.at_level('WARNING'):
        _, _, _, iron_rt = get_zalmoxis_EOS()

    cp_warnings = [r for r in caplog.records if 'Cp table not found' in r.getMessage()]
    folder_warnings = [r for r in caplog.records if 'EOS folder not found' in r.getMessage()]
    assert len(cp_warnings) >= 1
    assert folder_warnings == []
    assert 'cp_file' not in iron_rt['melted_mantle']


@pytest.mark.unit
def test_get_zalmoxis_EOS_rt_folder_missing_warns(monkeypatch, tmp_path, caplog):
    """When the versioned RTPress folder is absent, a folder-not-found warning fires."""
    import proteus.utils.data as data_mod
    from proteus.data import EOS_RTPRESS_100TPA, dataset_dir
    from proteus.utils.data import get_zalmoxis_EOS

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    _seed_seager(tmp_path)

    with caplog.at_level('WARNING'):
        _, _, _, iron_rt = get_zalmoxis_EOS()

    folder_warnings = [
        r for r in caplog.records if 'RTPress100TPa EOS folder not found' in r.getMessage()
    ]
    cp_warnings = [r for r in caplog.records if 'Cp table not found' in r.getMessage()]
    assert len(folder_warnings) >= 1
    assert len(cp_warnings) >= 1
    # The path still resolves inside the versioned dataset directory.
    rt = dataset_dir(EOS_RTPRESS_100TPA, data_root=tmp_path)
    assert Path(iron_rt['melted_mantle']['eos_file']) == rt / 'density_melt.dat'


# ============================================================================
# _get_sufficient: MORS star module branch + scattering branch
# ============================================================================


@pytest.mark.unit
def test_get_sufficient_mors_solar_spectrum_only(monkeypatch):
    """When star.mors.spectrum_source='solar', MUSCLES is skipped."""
    from types import SimpleNamespace

    import proteus.utils.data as data_mod

    spectra_calls = []
    monkeypatch.setattr(
        data_mod,
        'download_stellar_spectra',
        lambda *args, **kwargs: spectra_calls.append(kwargs.get('folders', args)),
    )
    monkeypatch.setattr(data_mod, 'download_stellar_tracks', lambda *args, **kwargs: None)
    monkeypatch.setattr(data_mod, 'download_spectral_file', lambda *args, **kwargs: None)
    monkeypatch.setattr(data_mod, 'download_surface_albedos', lambda: None)
    monkeypatch.setattr(data_mod, 'download_scattering', lambda: None)
    monkeypatch.setattr(data_mod, 'download_exoplanet_data', lambda: None)
    monkeypatch.setattr(data_mod, 'download_massradius_data', lambda: None)
    monkeypatch.setattr(data_mod, 'download_interior_lookuptables', lambda *a, **k: None)
    monkeypatch.setattr(data_mod, 'download_melting_curves', lambda *a, **k: None)

    config = SimpleNamespace(
        star=SimpleNamespace(
            module='mors',
            mors=SimpleNamespace(spectrum_source='solar', tracks='spada'),
        ),
        atmos_clim=SimpleNamespace(module='dummy', aerosols_enabled=False),
        interior_energetics=SimpleNamespace(module='dummy'),
        interior_struct=SimpleNamespace(module='dummy'),
    )

    data_mod._get_sufficient(config)

    # One call per collection, so a failing collection does not stop the others.
    assert spectra_calls == [('Named',), ('solar',)]
    folders = [name for call in spectra_calls for name in call]
    # Discrimination: solar source should request 'Named' and 'solar'
    # but NOT 'MUSCLES'.
    assert 'Named' in folders
    assert 'solar' in folders
    assert 'MUSCLES' not in folders


@pytest.mark.unit
def test_get_sufficient_mors_muscles_spectrum_only(monkeypatch):
    """When star.mors.spectrum_source='muscles', solar is skipped."""
    from types import SimpleNamespace

    import proteus.utils.data as data_mod

    spectra_calls = []
    tracks_calls = []
    monkeypatch.setattr(
        data_mod,
        'download_stellar_spectra',
        lambda *args, **kwargs: spectra_calls.append(kwargs.get('folders', args)),
    )
    monkeypatch.setattr(
        data_mod, 'download_stellar_tracks', lambda name: tracks_calls.append(name)
    )
    monkeypatch.setattr(data_mod, 'download_spectral_file', lambda *args, **kwargs: None)
    monkeypatch.setattr(data_mod, 'download_surface_albedos', lambda: None)
    monkeypatch.setattr(data_mod, 'download_scattering', lambda: None)
    monkeypatch.setattr(data_mod, 'download_exoplanet_data', lambda: None)
    monkeypatch.setattr(data_mod, 'download_massradius_data', lambda: None)
    monkeypatch.setattr(data_mod, 'download_interior_lookuptables', lambda *a, **k: None)
    monkeypatch.setattr(data_mod, 'download_melting_curves', lambda *a, **k: None)

    config = SimpleNamespace(
        star=SimpleNamespace(
            module='mors',
            mors=SimpleNamespace(spectrum_source='muscles', tracks='baraffe'),
        ),
        atmos_clim=SimpleNamespace(module='dummy', aerosols_enabled=False),
        interior_energetics=SimpleNamespace(module='dummy'),
        interior_struct=SimpleNamespace(module='dummy'),
    )

    data_mod._get_sufficient(config)

    folders = [name for call in spectra_calls for name in call]
    assert len(spectra_calls) == 2
    assert 'Named' in folders
    assert 'MUSCLES' in folders
    assert 'solar' not in folders
    # Discrimination: tracks='baraffe' must dispatch Baraffe, not Spada.
    assert tracks_calls == ['Baraffe']


@pytest.mark.unit
def test_get_sufficient_agni_aerosols_downloads_scattering(monkeypatch):
    """When aerosols_enabled is True, download_scattering is invoked."""
    from types import SimpleNamespace

    import proteus.atmos_clim.common as atmos_common
    import proteus.utils.data as data_mod

    scattering_calls = []
    monkeypatch.setattr(data_mod, 'download_scattering', lambda: scattering_calls.append('x'))
    monkeypatch.setattr(data_mod, 'download_stellar_spectra', lambda *a, **k: None)
    monkeypatch.setattr(data_mod, 'download_stellar_tracks', lambda *a, **k: None)
    monkeypatch.setattr(data_mod, 'download_spectral_file', lambda *a, **k: None)
    monkeypatch.setattr(data_mod, 'download_surface_albedos', lambda: None)
    monkeypatch.setattr(data_mod, 'download_exoplanet_data', lambda: None)
    monkeypatch.setattr(data_mod, 'download_massradius_data', lambda: None)
    monkeypatch.setattr(data_mod, 'download_interior_lookuptables', lambda *a, **k: None)
    monkeypatch.setattr(data_mod, 'download_melting_curves', lambda *a, **k: None)
    monkeypatch.setattr(
        atmos_common, 'get_spfile_name_and_bands', lambda *a: ('Frostflow', '256')
    )

    config = SimpleNamespace(
        star=SimpleNamespace(module='dummy'),
        atmos_clim=SimpleNamespace(
            module='agni',
            aerosols_enabled=True,
            agni=SimpleNamespace(spectral_file=None),
        ),
        interior_energetics=SimpleNamespace(module='dummy'),
        interior_struct=SimpleNamespace(module='dummy'),
    )

    surface_calls = []
    monkeypatch.setattr(data_mod, 'download_surface_albedos', lambda: surface_calls.append('s'))

    data_mod._get_sufficient(config)

    # Discrimination: download_scattering fired exactly once, and
    # download_surface_albedos also fired (both are AGNI-only paths).
    assert scattering_calls == ['x']
    assert surface_calls == ['s']


# ============================================================================
# get_socrates default-dirs path
# ============================================================================


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
def test_get_socrates_uses_none_dirs_when_not_given(mock_run, tmp_path, monkeypatch):
    """When dirs is None, get_socrates derives them from _none_dirs."""
    import proteus.utils.data as data_mod
    from proteus.utils.data import get_socrates

    fake_dirs = {'proteus': str(tmp_path), 'tools': str(tmp_path / 'tools')}
    (tmp_path / 'tools').mkdir()

    monkeypatch.setattr(data_mod, '_none_dirs', lambda: fake_dirs)

    get_socrates(dirs=None)

    # Discrimination: subprocess invoked, and the cmd's tool path
    # corresponds to the fake_dirs['tools'] entry.
    mock_run.assert_called_once()
    cmd = mock_run.call_args[0][0]
    assert cmd[0].startswith(str(tmp_path / 'tools'))


@pytest.mark.unit
@pytest.mark.parametrize(
    ('energetics', 'mantle', 'files'),
    [
        ('aragog', 'PALEOS:MgSiO3', {'paleos_mgsio3_eos_table_pt.dat'}),
        (
            'spider',
            'PALEOS:MgSiO3:0.9+PALEOS:H2O:0.1',
            {'paleos_mgsio3_eos_table_pt.dat', 'paleos_water_eos_table_pt.dat'},
        ),
        ('aragog', 'PALEOS-2phase:MgSiO3', set()),
        ('aragog', 'WolfBower2018:MgSiO3', None),
        ('aragog', 'PALEOS:H2O:0.1+WolfBower2018:MgSiO3:0.9', None),
        ('dummy', 'PALEOS:MgSiO3', None),
    ],
)
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_for_config_fetches_the_paleos_mantle_of_a_dummy_structure(
    mock_static, mock_fetch, mock_file, energetics, mantle, files
):
    """Under the dummy structure, SPIDER and Aragog with a PALEOS mantle EOS (a mixture
    follows its MgSiO3 component) fetch every mantle table and the 2-phase pair, and no
    Seager core; other mantles and the dummy energetics fetch nothing."""
    from types import SimpleNamespace

    from proteus.utils.data import _PALEOS_2PHASE_FILES, download_zalmoxis_eos_for_config

    download_zalmoxis_eos_for_config(
        SimpleNamespace(
            interior_struct=SimpleNamespace(
                module='dummy', zalmoxis=SimpleNamespace(mantle_eos=mantle)
            ),
            interior_energetics=SimpleNamespace(module=energetics),
        )
    )
    mock_static.assert_not_called()
    mock_fetch.assert_not_called()
    fetched = {name for _, name in _fetched_files(mock_file)}
    assert fetched == (set() if files is None else files | set(_PALEOS_2PHASE_FILES))
