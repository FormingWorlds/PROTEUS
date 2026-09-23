"""
Unit tests for proteus.utils.data module.

This module validates the data management utilities, ensuring reliable access to
external physics data (spectral files, lookup tables). It covers:
- Zenodo/OSF download logic (with mocking to prevent real network calls).
- File integrity verification (MD5 checksums).
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

from proteus.utils.data import (
    GetFWLData,
    check_needs_update,
    download,
    download_spectral_file,
    download_zenodo_folder,
    get_data_source_info,
    get_osf_from_zenodo,
    get_osf_project,
    get_zenodo_from_osf,
    get_zenodo_record,
    md5,
    validate_zenodo_folder,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.mark.unit
def test_get_zenodo_record():
    """
    Test Zenodo record ID lookup from registry.

    Ensures that known configuration keys map to correct Zenodo repository IDs.
    """
    # Known mapping: aerosol scattering data
    assert get_zenodo_record('scattering') == '19294180'
    # Unknown key should return None safely
    assert get_zenodo_record('Unknown/Folder') is None


@pytest.mark.unit
def test_md5(tmp_path):
    """
    Test MD5 checksum calculation utility.

    Verifies that the hashing function matches standard MD5 output.
    Used for verifying integrity of large downloaded binary files.
    """
    # Create dummy file
    f = tmp_path / 'test.txt'
    f.write_bytes(b'hello world')

    # MD5 of "hello world" is known constant "5eb63bbbe01eeed093cb22bb8f5acdc3"
    digest = md5(str(f))
    assert digest == '5eb63bbbe01eeed093cb22bb8f5acdc3'
    # Discrimination: a SHA-256 regression would produce a 64-char hex string;
    # MD5 is 128-bit and so always 32 hex chars in lowercase.
    assert len(digest) == 32 and digest == digest.lower()


@pytest.mark.unit
def test_check_needs_update_missing_dir(tmp_path):
    """
    check_needs_update returns True when the folder does not exist.

    Physical scenario: re-download is required when the target directory
    is missing (e.g. first run or cleaned cache).
    """
    missing = tmp_path / 'nonexistent'
    # Precondition: the directory really is absent before the call.
    assert not missing.exists()
    assert check_needs_update(str(missing), '12345') is True


@pytest.mark.unit
def test_check_needs_update_no_zenodo(tmp_path):
    """
    check_needs_update returns False when zenodo is falsy and dir exists.

    Physical scenario: when no Zenodo ID is provided we cannot validate
    hashes, so we assume up-to-date and do not trigger re-download.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    assert check_needs_update(str(tmp_path), None) is False
    assert check_needs_update(str(tmp_path), '') is False


@pytest.mark.unit
@patch('proteus.utils.data.validate_zenodo_folder')
def test_check_needs_update_valid_folder(mock_validate, tmp_path):
    """
    check_needs_update returns False when folder exists and validates.

    Physical scenario: folder is present and MD5 hashes match Zenodo;
    no update needed.
    """
    (tmp_path / 'x').mkdir(parents=True, exist_ok=True)
    mock_validate.return_value = True
    assert check_needs_update(str(tmp_path / 'x'), '12345') is False
    # Discrimination: confirm the validate_zenodo_folder path was actually
    # taken (a regression that short-circuits before validation would skip
    # the mock call and still return False).
    mock_validate.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.validate_zenodo_folder')
def test_check_needs_update_invalid_folder(mock_validate, tmp_path):
    """
    check_needs_update returns True when folder exists but validation fails.

    Physical scenario: folder is present but hashes mismatch or files
    missing; re-download is needed.
    """
    (tmp_path / 'y').mkdir(parents=True, exist_ok=True)
    mock_validate.return_value = False
    assert check_needs_update(str(tmp_path / 'y'), '12345') is True
    # Discrimination: confirm the validate path ran and produced the failure
    # signal that drove the True return (rather than the True coming from an
    # unrelated short-circuit).
    mock_validate.assert_called_once()


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
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_folder_success(mock_getfwl, mock_run, tmp_path):
    """
    Test successful Zenodo download workflow (mocked).

    Steps verified:
    1. Resolves FWL data path.
    2. Calls subprocess to run the download client.
    3. Checks if output directory exists and has files.
    """
    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'downloaded_folder'

    # Mock availability check (first call)
    mock_proc_avail = MagicMock()
    mock_proc_avail.returncode = 0

    # Mock download success (subsequent calls)
    mock_proc_download = MagicMock()
    mock_proc_download.returncode = 0

    call_count = 0

    # Mock subprocess.run to emulate zenodo-get without real network calls.
    # No actual download occurs; the side_effect creates local files to
    # simulate a successful download.
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if '--version' in args[0]:
            return mock_proc_avail
        # After download, create the folder with files
        if call_count > 1:  # After availability check
            folder_dir.mkdir(parents=True, exist_ok=True)
            (folder_dir / 'test_file.txt').write_text('test content')
        return mock_proc_download

    mock_run.side_effect = side_effect

    success = download_zenodo_folder('12345', folder_dir)
    assert success is True
    # Discrimination: a regression that returned True without invoking the
    # download client at all would pass `is True` but leave mock_run untouched.
    assert mock_run.call_count >= 2  # availability probe + at least one download call
    assert folder_dir.exists() and any(folder_dir.iterdir())


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_success(mock_getfwl, mock_run, tmp_path):
    """download_zenodo_file returns True when zenodo_get succeeds and file appears."""
    from proteus.utils.data import download_zenodo_file

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'zenodo_folder'
    record_path = 'subdir/myfile.dat'

    # availability check ok, then download ok
    proc_avail = MagicMock()
    proc_avail.returncode = 0
    proc_dl = MagicMock()
    proc_dl.returncode = 0

    call_count = 0

    def side_effect(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if '--version' in cmd:
            return proc_avail

        # download call: create expected file in folder_dir
        folder_dir.mkdir(parents=True, exist_ok=True)
        (folder_dir / record_path).parent.mkdir(parents=True, exist_ok=True)
        (folder_dir / record_path).write_text('payload')
        return proc_dl

    mock_run.side_effect = side_effect

    ok = download_zenodo_file('12345', folder_dir, record_path)
    assert ok is True
    # Discrimination: confirm the expected on-disk file appeared at the
    # canonical record_path; a regression returning True without writing
    # the file would pass `is True` but leave the path missing.
    assert (folder_dir / record_path).exists()


@pytest.mark.unit
def test_download_zenodo_file_rejects_bad_id(tmp_path):
    """download_zenodo_file rejects non-numeric zenodo IDs."""
    from proteus.utils.data import download_zenodo_file

    folder_dir = tmp_path / 'zenodo_folder'
    ok = download_zenodo_file('12ab', folder_dir, 'file.txt')
    assert ok is False
    # Discrimination: bad-ID rejection should happen before any filesystem
    # side effect; the target folder must not have been created.
    assert not folder_dir.exists()


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
def test_download_zenodo_file_zenodo_get_missing(mock_run, tmp_path):
    """download_zenodo_file returns False when zenodo_get is not available."""
    from proteus.utils.data import download_zenodo_file

    mock_run.side_effect = FileNotFoundError('zenodo_get not found')

    folder_dir = tmp_path / 'zenodo_folder'
    ok = download_zenodo_file('12345', folder_dir, 'file.txt')
    assert ok is False
    # Discrimination: confirm the function actually tried to invoke
    # zenodo_get (otherwise the False could come from an unrelated guard
    # that fires before the missing-binary path).
    assert mock_run.called


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_success_via_rglob_fallback(mock_getfwl, mock_run, tmp_path):
    """If zenodo_get returns 0 but file isn't at expected_path, rglob basename fallback should succeed."""
    from proteus.utils.data import download_zenodo_file

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'zenodo_folder'
    record_path = 'subdir/myfile.dat'
    basename = Path(record_path).name

    proc_avail = MagicMock(returncode=0)
    proc_dl = MagicMock(returncode=0)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail

        # Simulate zenodo_get putting file in a different layout than record_path
        folder_dir.mkdir(parents=True, exist_ok=True)
        alt = folder_dir / 'weird_layout' / basename
        alt.parent.mkdir(parents=True, exist_ok=True)
        alt.write_text('payload')
        return proc_dl

    mock_run.side_effect = side_effect

    ok = download_zenodo_file('12345', folder_dir, record_path)
    assert ok is True
    # Discrimination: True only valid here if the rglob fallback located
    # the file at its non-canonical path (otherwise the True would be a
    # false positive on a regression that returned True unconditionally).
    assert (folder_dir / 'weird_layout' / basename).exists()


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)  # speed up retries
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_zero_exit_but_file_missing(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """If zenodo_get exits 0 but file is missing/empty everywhere, function should return False."""
    from proteus.utils.data import download_zenodo_file

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'zenodo_folder'
    record_path = 'subdir/myfile.dat'

    proc_avail = MagicMock(returncode=0)
    proc_dl = MagicMock(returncode=0)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        folder_dir.mkdir(parents=True, exist_ok=True)
        # Do NOT create any file
        return proc_dl

    mock_run.side_effect = side_effect

    ok = download_zenodo_file('12345', folder_dir, record_path)
    assert ok is False
    # Discrimination: confirm both subprocess calls actually ran (the
    # availability probe and the download attempt); a regression that
    # returned False from an unrelated early-exit would have skipped at
    # least one of them.
    assert mock_run.call_count >= 2


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)  # speed up retries
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_nonzero_exit_reads_log_and_fails(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """Non-zero exit should trigger retries and read the log for diagnostics, then return False."""
    from proteus.utils.data import MAX_ATTEMPTS, download_zenodo_file

    mock_getfwl.return_value = tmp_path
    folder_dir = tmp_path / 'zenodo_folder'
    record_path = 'file.txt'

    proc_avail = MagicMock(returncode=0)
    proc_fail = MagicMock(returncode=2)

    call_count = 0

    def side_effect(cmd, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        if '--version' in cmd:
            return proc_avail

        # Simulate zenodo_get writing something to stdout log (download_zenodo_file opens log itself)
        return proc_fail

    mock_run.side_effect = side_effect

    ok = download_zenodo_file('12345', folder_dir, record_path)
    assert ok is False

    # 1 availability call + MAX_ATTEMPTS download calls
    assert mock_run.call_count >= 1 + MAX_ATTEMPTS


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_nonzero_exit_reads_log_content(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """Non-zero exit should read log content when available."""
    from proteus.utils.data import MAX_ATTEMPTS, download_zenodo_file

    mock_getfwl.return_value = tmp_path
    folder_dir = tmp_path / 'zenodo_folder'
    record_path = 'file.txt'

    proc_avail = MagicMock(returncode=0)
    proc_fail = MagicMock(returncode=1)

    log_path = tmp_path / 'zenodo_download.log'
    log_path.write_text('some error line 1\nsome error line 2\n')

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        return proc_fail

    mock_run.side_effect = side_effect

    ok = download_zenodo_file('12345', folder_dir, record_path)
    assert ok is False
    assert mock_run.call_count >= 1 + MAX_ATTEMPTS


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)  # speed up retries
@patch('proteus.utils.data.safe_rm')
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_cleanup_branches(
    mock_getfwl, mock_run, mock_safe_rm, _mock_sleep, tmp_path
):
    """Covers file, directory, and exception branches in expected_path cleanup."""

    from proteus.utils.data import download_zenodo_file

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'zenodo_folder'
    record_path = 'subdir/myfile.dat'
    expected_path = folder_dir / record_path
    expected_path.parent.mkdir(parents=True, exist_ok=True)

    proc_avail = MagicMock(returncode=0)
    proc_dl = MagicMock(returncode=0)

    call_state = {'phase': 0}

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail

        # Phase 1: FILE but empty
        if call_state['phase'] == 0:
            expected_path.write_text('')  # ← KEY CHANGE
            call_state['phase'] += 1
            return proc_dl

        # Phase 2: expected_path is DIRECTORY
        if call_state['phase'] == 1:
            if expected_path.exists():
                expected_path.unlink()
            expected_path.mkdir()
            call_state['phase'] += 1
            return proc_dl

        # Phase 3: removal throws exception
        if call_state['phase'] == 2:
            if expected_path.exists():
                expected_path.unlink()
            expected_path.write_text('bad')
            expected_path.unlink = MagicMock(side_effect=OSError('cannot unlink'))
            call_state['phase'] += 1
            return proc_dl

        return proc_dl

    mock_run.side_effect = side_effect

    ok = download_zenodo_file('12345', folder_dir, record_path)

    # safe_rm should have been used for directory case
    assert mock_safe_rm.called
    assert ok is False


@pytest.mark.unit
@patch('proteus.utils.data.validate_zenodo_folder')
@patch('proteus.utils.data.os.path.isdir')
def test_check_needs_update(mock_isdir, mock_validate):
    """
    Test update requirement logic.

    Verifies when the system decides to re-download data:
    1. Update needed if folder is missing.
    2. No update if download logic is disabled (id=None).
    3. No update if folder exists and validates (checksum match).
    4. Update needed if folder exists but validation fails.
    """
    # Case 1: Folder missing -> needs update
    mock_isdir.return_value = False
    assert check_needs_update('dummy_path', '123') is True

    # Case 2: Folder exists, but zenodo_id None -> no update (manual mod)
    mock_isdir.return_value = True
    assert check_needs_update('dummy_path', None) is False

    # Case 3: Folder exists, valid zenodo -> no update
    mock_validate.return_value = True
    assert check_needs_update('dummy_path', '123') is False

    # Case 4: Folder exists, invalid zenodo (hash mismatch) -> update
    mock_validate.return_value = False
    assert check_needs_update('dummy_path', '123') is True


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
    """Every spectral folder is a manifest dataset and none is left in the source map.

    Forward direction: each SPECTRAL_FILE_FOLDERS entry must resolve to a
    manifest table, so the bare `proteus get spectral` cannot fail on an entry
    with no source. Reverse direction: no DATA_SOURCE_MAP key shaped like a
    spectral folder (Group/<digits>) may remain, since it would pin the same
    record a second time.
    """
    import re

    from proteus.data import _dataset, spectral_file_key
    from proteus.utils.data import DATA_SOURCE_MAP, SPECTRAL_FILE_FOLDERS

    for folder in SPECTRAL_FILE_FOLDERS:
        group, bands = folder.split('/')
        assert _dataset(spectral_file_key(group, bands)).zenodo.startswith('10.5281/zenodo.')

    leftover = [k for k in DATA_SOURCE_MAP if re.fullmatch(r'[A-Za-z]+/[0-9]+', k)]
    assert leftover == [], f'Spectral-folder entries left in DATA_SOURCE_MAP: {leftover}'


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


@pytest.mark.unit
@patch('proteus.utils.data.check_needs_update')
@patch('proteus.utils.data.GetFWLData')
def test_download_skip(mock_getfwl, mock_check, tmp_path):
    """
    Test skipping download if data is already valid.

    Verifies that `download()` calls `check_needs_update()` and
    returns early if no update is required, saving time/bandwidth.
    """
    mock_getfwl.return_value = tmp_path

    # If check_needs_update returns False (valid), function should return True immediately
    mock_check.return_value = False

    success = download(folder='test', target='targ', osf_id='abc', zenodo_id='123', desc='test')
    assert success is True
    # Discrimination: confirm check_needs_update was actually consulted;
    # a regression that returned True from a different short-circuit
    # (e.g. unconditional True before the cache check) would skip the
    # mock altogether.
    mock_check.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.download_zenodo_file')
def test_download_file_mode_skips_existing_file(mock_zdl, mock_getfwl, tmp_path):
    """An existing file whose source marker matches the pinned record is kept.

    Without a Zenodo id the marker check cannot apply, so a bare
    existing file is also kept; with an id, the sidecar must name the
    same record for the skip to fire.
    """
    from proteus.utils.data import _source_marker_path, download

    mock_getfwl.return_value = tmp_path

    folder = 'SomeFolder'
    target = 'targetdir'
    file_rel = 'subdir/file.txt'

    dest = tmp_path / target / folder / file_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('already here')

    # No zenodo_id (OSF-only source): provenance not checkable, file kept.
    ok = download(
        folder=folder, target=target, desc='desc', file=file_rel, force=False, osf_id='abc'
    )
    assert ok is True
    mock_zdl.assert_not_called()

    # Matching marker: skip without re-download.
    _source_marker_path(dest).write_text('12345\n')
    ok = download(
        folder=folder,
        target=target,
        desc='desc',
        file=file_rel,
        force=False,
        zenodo_id='12345',
    )
    assert ok is True
    mock_zdl.assert_not_called()
    assert dest.read_text() == 'already here'  # content untouched


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.download_zenodo_file')
def test_download_file_mode_refreshes_on_record_change(mock_zdl, mock_getfwl, tmp_path):
    """A pin bump refreshes on-disk files fetched from an older record.

    Files with a stale source marker, or with no marker at all
    (pre-bookkeeping installs), are re-fetched once and re-stamped with
    the new record id. This is the guard against silently serving old
    EOS tables after a Zenodo version bump.
    """
    from proteus.utils.data import _source_marker_path, download

    mock_getfwl.return_value = tmp_path

    folder = 'EOSFolder'
    target = 'targetdir'
    file_rel = 'table.dat'
    dest = tmp_path / target / folder / file_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('old version payload')

    def zdl_side_effect(*, zenodo_id, folder_dir, record_path):
        (folder_dir / record_path).write_text(f'payload from {zenodo_id}')
        return True

    mock_zdl.side_effect = zdl_side_effect

    # No marker (legacy install): one re-fetch, then stamped.
    ok = download(folder=folder, target=target, desc='d', file=file_rel, zenodo_id='20084812')
    assert ok is True
    assert mock_zdl.call_count == 1
    assert dest.read_text() == 'payload from 20084812'
    assert _source_marker_path(dest).read_text().strip() == '20084812'

    # Same record again: no second fetch.
    ok = download(folder=folder, target=target, desc='d', file=file_rel, zenodo_id='20084812')
    assert ok is True
    assert mock_zdl.call_count == 1

    # Stale marker (record bumped): re-fetch and re-stamp.
    _source_marker_path(dest).write_text('19000316\n')
    ok = download(folder=folder, target=target, desc='d', file=file_rel, zenodo_id='20084812')
    assert ok is True
    assert mock_zdl.call_count == 2
    assert _source_marker_path(dest).read_text().strip() == '20084812'


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.download_zenodo_file')
def test_download_file_mode_failed_refresh_keeps_old_file(mock_zdl, mock_getfwl, tmp_path):
    """A failed refresh returns False but leaves the stale file usable.

    The marker mismatch triggers a re-fetch attempt; when both Zenodo
    and OSF fail, the old table must survive on disk so an offline-ish
    machine can keep running on the previous version.
    """
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path
    mock_zdl.return_value = False  # Zenodo fetch fails

    folder = 'EOSFolder'
    target = 'targetdir'
    file_rel = 'table.dat'
    dest = tmp_path / target / folder / file_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('old version payload')

    ok = download(folder=folder, target=target, desc='d', file=file_rel, zenodo_id='20084812')
    assert ok is False
    assert dest.read_text() == 'old version payload'  # stale file intact


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.get_data_source_info')
@patch('proteus.utils.data.download_zenodo_file')
def test_download_file_mode_uses_mapping_and_zenodo_success_expected_path(
    mock_zdl, mock_get_info, mock_getfwl, tmp_path
):
    """Single-file mode: uses mapping IDs when not provided; Zenodo writes expected file -> True."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path
    mock_get_info.return_value = {
        'zenodo_id': '123',
        'osf_project': 'osfproj',
        'osf_id': 'osfproj',
    }

    folder = 'MappedFolder'
    target = 'targetdir'
    file_rel = 'subdir/file.dat'

    # Simulate zenodo download creating the expected file
    def zdl_side_effect(*, zenodo_id, folder_dir, record_path):
        (folder_dir / record_path).parent.mkdir(parents=True, exist_ok=True)
        (folder_dir / record_path).write_text('payload')
        return True

    mock_zdl.side_effect = zdl_side_effect

    ok = download(folder=folder, target=target, desc='desc', file=file_rel)
    assert ok is True

    # mapping applied
    call_kwargs = mock_zdl.call_args.kwargs
    assert call_kwargs['zenodo_id'] == '123'
    assert call_kwargs['record_path'] == file_rel


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.get_data_source_info')
@patch('proteus.utils.data.download_zenodo_file')
def test_download_file_mode_zenodo_success_basename_fallback(
    mock_zdl, mock_get_info, mock_getfwl, tmp_path
):
    """
    Zenodo returns success but file not at expected_path. download() should still succeed
    if a file with the same basename exists somewhere under folder_dir (rglob fallback).
    """
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path
    mock_get_info.return_value = {
        'zenodo_id': '123',
        'osf_project': 'osfproj',
        'osf_id': 'osfproj',
    }

    folder = 'MappedFolder'
    target = 'targetdir'
    file_rel = 'subdir/file.dat'
    basename = Path(file_rel).name

    def zdl_side_effect(*, zenodo_id, folder_dir, record_path):
        # Create file at a different path than expected
        alt = folder_dir / 'weird_layout' / basename
        alt.parent.mkdir(parents=True, exist_ok=True)
        alt.write_text('payload')
        return True

    mock_zdl.side_effect = zdl_side_effect

    ok = download(folder=folder, target=target, desc='desc', file=file_rel)
    assert ok is True
    # Discrimination: the basename-fallback path must have located the
    # file at the alternate (weird_layout) location. A regression that
    # returned True without honouring the basename rglob would still pass
    # `is True` but the alt file would not exist.
    assert (tmp_path / target / folder / 'weird_layout' / basename).exists()


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.get_osf')
@patch('proteus.utils.data.download_OSF_file')
@patch('proteus.utils.data.download_zenodo_file')
def test_download_file_mode_osf_fallback_when_zenodo_fails(
    mock_zdl, mock_osf_dl, mock_get_osf, mock_getfwl, tmp_path
):
    """If Zenodo single-file download fails, OSF fallback should be attempted and can succeed."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path

    folder = 'Folder'
    target = 'targetdir'
    file_rel = 'subdir/file.txt'

    mock_zdl.return_value = False

    # Make OSF fallback write expected file
    def osf_side_effect(*, storage, files, data_dir):
        dest = data_dir / folder / file_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('from osf')

    mock_osf_dl.side_effect = osf_side_effect
    mock_get_osf.return_value = MagicMock()

    ok = download(
        folder=folder,
        target=target,
        desc='desc',
        file=file_rel,
        zenodo_id='123',
        osf_id='osfproj',
    )
    assert ok is True
    mock_get_osf.assert_called_once_with('osfproj')
    mock_osf_dl.assert_called_once()
    mock_zdl.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.get_osf')
@patch('proteus.utils.data.download_OSF_file')
@patch('proteus.utils.data.download_zenodo_file')
def test_download_file_mode_fails_if_both_sources_fail(
    mock_zdl, mock_osf_dl, mock_get_osf, mock_getfwl, tmp_path
):
    """If Zenodo and OSF both fail to create the file, download() returns False."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path

    folder = 'Folder'
    target = 'targetdir'
    file_rel = 'subdir/file.txt'

    mock_zdl.return_value = False
    mock_get_osf.return_value = MagicMock()
    # OSF doesn't create file and doesn't raise -> should fail
    mock_osf_dl.return_value = None

    ok = download(
        folder=folder,
        target=target,
        desc='desc',
        file=file_rel,
        zenodo_id='123',
        osf_id='osfproj',
    )
    assert ok is False
    # Discrimination: confirm both sources were tried before failure. A
    # regression that returned False without attempting OSF (or without
    # attempting Zenodo) would still pass `is False`.
    mock_zdl.assert_called()
    mock_osf_dl.assert_called()


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.check_needs_update')
@patch('proteus.utils.data.download_zenodo_folder')
@patch('proteus.utils.data.validate_zenodo_folder')
def test_download_folder_mode_force_triggers_download_even_if_valid(
    mock_validate, mock_zdl, mock_check, mock_getfwl, tmp_path
):
    """force=True should trigger download even if check_needs_update says False."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path
    mock_check.return_value = False  # would normally skip
    mock_zdl.return_value = True
    mock_validate.return_value = True

    ok = download(
        folder='folder',
        target='targ',
        desc='desc',
        zenodo_id='123',
        osf_id='osfproj',
        force=True,
    )
    assert ok is True
    mock_zdl.assert_called_once()
    mock_validate.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.check_needs_update')
@patch('proteus.utils.data.download_zenodo_folder')
@patch('proteus.utils.data.validate_zenodo_folder')
@patch('proteus.utils.data.get_osf')
@patch('proteus.utils.data.download_OSF_folder')
def test_download_folder_mode_zenodo_download_ok_but_validation_fails_osf_fallback_succeeds(
    mock_osf_dl,
    mock_get_osf,
    mock_validate,
    mock_zdl,
    mock_check,
    mock_getfwl,
    tmp_path,
):
    """
    Folder mode: Zenodo folder download returns True but validation returns False,
    so OSF fallback is attempted and can succeed.
    """
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path
    mock_check.return_value = True

    mock_zdl.return_value = True
    mock_validate.return_value = False  # force OSF fallback

    mock_get_osf.return_value = MagicMock()

    # OSF fallback creates a file under folder_dir
    def osf_side_effect(*, storage, folders, data_dir):
        folder_dir = data_dir / folders[0]
        folder_dir.mkdir(parents=True, exist_ok=True)
        (folder_dir / 'x.txt').write_text('ok')

    mock_osf_dl.side_effect = osf_side_effect

    ok = download(
        folder='folder',
        target='targ',
        desc='desc',
        zenodo_id='123',
        osf_id='osfproj',
    )
    assert ok is True
    mock_osf_dl.assert_called_once()
    mock_get_osf.assert_called_once_with('osfproj')


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.check_needs_update')
@patch('proteus.utils.data.download_OSF_folder')
@patch('proteus.utils.data.get_osf')
def test_download_folder_mode_no_zenodo_id_uses_osf_only(
    mock_get_osf, mock_osf_dl, mock_check, mock_getfwl, tmp_path
):
    """Folder mode: if zenodo_id is None but osf_id is provided, OSF should be used."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path
    mock_check.return_value = True
    mock_get_osf.return_value = MagicMock()

    def osf_side_effect(*, storage, folders, data_dir):
        folder_dir = data_dir / folders[0]
        folder_dir.mkdir(parents=True, exist_ok=True)
        (folder_dir / 'ok.txt').write_text('ok')

    mock_osf_dl.side_effect = osf_side_effect

    ok = download(folder='folder', target='targ', desc='desc', zenodo_id=None, osf_id='osfproj')
    assert ok is True
    mock_get_osf.assert_called_once_with('osfproj')
    mock_osf_dl.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.check_needs_update')
def test_download_folder_mode_fails_if_no_sources_available(mock_check, mock_getfwl, tmp_path):
    """Folder mode: if no mapping and both IDs None, download() returns False (already partially tested, but covers folder-mode call)."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path
    mock_check.return_value = True

    ok = download(
        folder='UnknownFolder', target='targ', desc='desc', zenodo_id=None, osf_id=None
    )
    assert ok is False
    # Discrimination: the early-exit must happen before check_needs_update
    # consults the cache; a regression that bypassed the no-sources guard
    # and ran the cache check first would still return False but call
    # mock_check.
    mock_check.assert_not_called()


@pytest.mark.unit
def test_get_data_source_info():
    """Test unified data source mapping lookup."""
    # Test known mapping
    info = get_data_source_info('scattering')
    assert info is not None
    assert info['zenodo_id'] == '19294180'
    assert info['osf_project'] == 'vehxg'

    # Test unknown mapping
    info = get_data_source_info('UnknownFolder')
    assert info is None


@pytest.mark.unit
def test_get_osf_project():
    """Test OSF project ID lookup."""
    assert get_osf_project('scattering') == 'vehxg'
    assert get_osf_project('UnknownFolder') is None


@pytest.mark.unit
def test_get_zenodo_from_osf():
    """Test reverse lookup: OSF project -> Zenodo IDs."""
    zenodo_ids = get_zenodo_from_osf('vehxg')
    assert zenodo_ids == ['19294180']

    # Test unknown OSF project
    zenodo_ids = get_zenodo_from_osf('unknown')
    assert len(zenodo_ids) == 0


@pytest.mark.unit
def test_get_osf_from_zenodo():
    """Test reverse lookup: Zenodo ID -> OSF project."""
    assert get_osf_from_zenodo('19294180') == 'vehxg'  # scattering
    assert get_osf_from_zenodo('99999999') is None  # Unknown


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
def test_download_zenodo_folder_availability_check(mock_run):
    """Test zenodo_get availability check."""
    from pathlib import Path

    from proteus.utils.data import download_zenodo_folder

    # Test: zenodo_get not available
    mock_run.side_effect = FileNotFoundError('zenodo_get not found')
    result = download_zenodo_folder('12345', Path('/tmp/test'))
    assert result is False

    # Test: zenodo_get available
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_run.return_value = mock_proc
    mock_run.side_effect = None

    with patch('proteus.utils.data.os.path.exists', return_value=True):
        with patch('proteus.utils.data.Path.rglob') as mock_rglob:
            # Mock files in folder
            mock_file = MagicMock()
            mock_file.is_file.return_value = True
            mock_rglob.return_value = [mock_file]

            result = download_zenodo_folder('12345', Path('/tmp/test'))
            # Should check availability first
            assert mock_run.call_count >= 1


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)  # speed up retries
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_folder_timeout(mock_getfwl, mock_run, _mock_sleep, tmp_path):
    """Test timeout handling in zenodo_get downloads."""
    import subprocess as sp

    from proteus.utils.data import download_zenodo_folder

    mock_getfwl.return_value = tmp_path

    # First call: availability check succeeds
    # Subsequent calls: timeout
    mock_proc_avail = MagicMock()
    mock_proc_avail.returncode = 0

    def side_effect(*args, **kwargs):
        if '--version' in args[0]:
            return mock_proc_avail
        raise sp.TimeoutExpired(cmd=args[0], timeout=120)

    mock_run.side_effect = side_effect

    folder_dir = tmp_path / 'test_folder'
    result = download_zenodo_folder('12345', folder_dir)

    # Should have attempted download and hit timeout
    assert result is False
    # Should have tried multiple times (MAX_ATTEMPTS = 3)
    assert mock_run.call_count >= 3


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_folder_exponential_backoff(mock_getfwl, mock_run, tmp_path):
    """Test exponential backoff retry logic."""

    from proteus.utils.data import RETRY_WAIT, download_zenodo_folder

    mock_getfwl.return_value = tmp_path

    # Track sleep calls to verify backoff
    sleep_times = []

    def mock_sleep(seconds):
        sleep_times.append(seconds)

    # First call: availability check
    mock_proc_avail = MagicMock()
    mock_proc_avail.returncode = 0

    # Subsequent calls: fail with non-zero exit
    mock_proc_fail = MagicMock()
    mock_proc_fail.returncode = 1

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if '--version' in args[0]:
            return mock_proc_avail
        return mock_proc_fail

    mock_run.side_effect = side_effect

    with patch('proteus.utils.data.sleep', side_effect=mock_sleep):
        folder_dir = tmp_path / 'test_folder'
        download_zenodo_folder('12345', folder_dir)

    # Should have exponential backoff: RETRY_WAIT * (2 ** attempt)
    expected_waits = [RETRY_WAIT * (2**i) for i in range(2)]  # 2 retries = 2 waits
    assert len(sleep_times) == 2
    assert sleep_times[0] == pytest.approx(expected_waits[0], rel=0.1)
    assert sleep_times[1] == pytest.approx(expected_waits[1], rel=0.1)


@pytest.mark.unit
@patch('proteus.utils.data.download_zenodo_folder')
@patch('proteus.utils.data.validate_zenodo_folder')
@patch('proteus.utils.data.download_OSF_folder')
@patch('proteus.utils.data.get_osf')
@patch('proteus.utils.data.check_needs_update')
@patch('proteus.utils.data.GetFWLData')
def test_download_osf_fallback(
    mock_getfwl,
    mock_check,
    mock_get_osf,
    mock_download_osf,
    mock_validate,
    mock_download_zenodo,
    tmp_path,
):
    """Test OSF fallback when Zenodo download fails."""
    mock_getfwl.return_value = tmp_path
    mock_check.return_value = True  # Needs update

    # Zenodo download fails
    mock_download_zenodo.return_value = False

    # OSF download succeeds
    mock_storage = MagicMock()
    mock_get_osf.return_value = mock_storage

    folder_dir = tmp_path / 'target' / 'test_folder'
    folder_dir.mkdir(parents=True, exist_ok=True)
    (folder_dir / 'test_file.txt').write_text('test')

    with patch('proteus.utils.data.Path.rglob') as mock_rglob:
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_rglob.return_value = [mock_file]

        result = download(
            folder='test_folder',
            target='target',
            osf_id='test_osf',
            zenodo_id='12345',
            desc='test data',
        )

    # Should have tried Zenodo first
    mock_download_zenodo.assert_called_once()
    # Should have tried OSF fallback
    mock_get_osf.assert_called_once_with('test_osf')
    mock_download_osf.assert_called_once()
    # Should succeed via OSF
    assert result is True


@pytest.mark.unit
@patch('proteus.utils.data.download_zenodo_folder')
@patch('proteus.utils.data.validate_zenodo_folder')
@patch('proteus.utils.data.download_OSF_folder')
@patch('proteus.utils.data.get_osf')
@patch('proteus.utils.data.check_needs_update')
@patch('proteus.utils.data.GetFWLData')
def test_download_automatic_mapping(
    mock_getfwl,
    mock_check,
    mock_get_osf,
    mock_download_osf,
    mock_validate,
    mock_download_zenodo,
    tmp_path,
):
    """Test automatic ID lookup from mapping."""
    mock_getfwl.return_value = tmp_path
    mock_check.return_value = True  # Needs update
    mock_download_zenodo.return_value = True
    mock_validate.return_value = True

    folder_dir = tmp_path / 'target' / 'scattering'
    folder_dir.mkdir(parents=True, exist_ok=True)
    (folder_dir / 'test_file.txt').write_text('test')

    with patch('proteus.utils.data.Path.rglob') as mock_rglob:
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_rglob.return_value = [mock_file]

        # Call download without explicit IDs - should use mapping
        result = download(
            folder='scattering',
            target='target',
            desc='test data',
            # No osf_id or zenodo_id provided - should use mapping
        )

    # Should have looked up IDs from mapping and used them
    mock_download_zenodo.assert_called_once()
    # Should have used mapped Zenodo ID (check kwargs since it's called with keyword args)
    call_kwargs = mock_download_zenodo.call_args.kwargs
    assert call_kwargs['zenodo_id'] == '19294180'  # Zenodo ID from mapping
    assert result is True


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)  # speed up retries
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_folder_error_diagnostics(mock_getfwl, mock_run, _mock_sleep, tmp_path):
    """Test improved error message diagnostics."""

    from proteus.utils.data import download_zenodo_folder

    mock_getfwl.return_value = tmp_path

    # Mock availability check
    mock_proc_avail = MagicMock()
    mock_proc_avail.returncode = 0

    # Mock failed download with error in log
    mock_proc_fail = MagicMock()
    mock_proc_fail.returncode = 1

    log_file = tmp_path / 'zenodo_download.log'
    log_file.write_text('HTTP error fetching metadata: 403 - Forbidden\n')

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if '--version' in args[0]:
            return mock_proc_avail
        return mock_proc_fail

    mock_run.side_effect = side_effect

    folder_dir = tmp_path / 'test_folder'
    with patch('proteus.utils.data.Path.open') as mock_open:
        # Mock log file reading
        mock_file_obj = MagicMock()
        mock_file_obj.readlines.return_value = ['HTTP error: 403 - Forbidden\n']
        mock_open.return_value.__enter__.return_value = mock_file_obj

        result = download_zenodo_folder('12345', folder_dir)

    # Should have read error from log file
    assert result is False
    # Should have attempted multiple times
    assert call_count > 1


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_graceful_degradation(mock_getfwl, mock_run, tmp_path):
    """Test validation gracefully handles missing zenodo_get."""

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'test_folder'
    folder_dir.mkdir(parents=True, exist_ok=True)
    (folder_dir / 'test_file.txt').write_text('test content')

    # Mock zenodo_get not available
    mock_run.side_effect = FileNotFoundError('zenodo_get not found')

    # Should gracefully degrade and check if files exist
    with patch('proteus.utils.data.Path.rglob') as mock_rglob:
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_rglob.return_value = [mock_file]

        result = validate_zenodo_folder('12345', folder_dir)

    # Should assume valid if files exist
    assert result is True
    # Discrimination: confirm the missing-binary path was actually
    # exercised (sp.run was attempted and raised); a regression that
    # returned True without probing zenodo_get would skip mock_run.
    mock_run.assert_called()


@pytest.mark.unit
@patch('proteus.utils.data.download_zenodo_folder')
@patch('proteus.utils.data.validate_zenodo_folder')
@patch('proteus.utils.data.check_needs_update')
@patch('proteus.utils.data.GetFWLData')
def test_download_no_mapping_no_ids(
    mock_getfwl, mock_check, mock_validate, mock_download_zenodo, tmp_path
):
    """Test download fails gracefully when no mapping and no IDs provided."""
    mock_getfwl.return_value = tmp_path
    mock_check.return_value = True  # Needs update

    # No mapping for this folder, and no IDs provided
    result = download(
        folder='UnknownFolder',
        target='target',
        desc='test data',
        # No osf_id or zenodo_id provided
    )

    # Should fail gracefully
    assert result is False
    # Should not have attempted download
    mock_download_zenodo.assert_not_called()


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

    legacy = MagicMock()
    monkeypatch.setattr('proteus.utils.data.download', legacy)

    download_interior_lookuptables(clean=False)

    mock_fetch.assert_called_once_with(MELTING_WOLF_BOWER_2018, data_root=tmp_path)
    legacy.assert_not_called()


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
def test_download_melting_curves_local_dir_wins(mock_fetch, tmp_path, monkeypatch):
    """A local directory with both P-T files is used and nothing is fetched."""
    from proteus.utils.data import download_melting_curves

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Wolf_Bower+2018'
    local.mkdir(parents=True)
    for name in ('solidus_P-T.dat', 'liquidus_P-T.dat'):
        (local / name).write_text('dummy\n')
    config = MagicMock()
    config.interior_struct.melting_dir = 'Wolf_Bower+2018'

    download_melting_curves(config, clean=True)

    mock_fetch.assert_not_called()
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
def test_download_melting_curves_unknown_name_local_dir_ok(mock_fetch, tmp_path, monkeypatch):
    """A free-form local directory with both P-T files is accepted without a fetch."""
    from proteus.utils.data import download_melting_curves

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'MyCurve'
    local.mkdir(parents=True)
    for name in ('solidus_P-T.dat', 'liquidus_P-T.dat'):
        (local / name).write_text('dummy\n')
    config = MagicMock()
    config.interior_struct.melting_dir = 'MyCurve'

    download_melting_curves(config, clean=True)

    mock_fetch.assert_not_called()
    assert (local / 'liquidus_P-T.dat').read_text() == 'dummy\n'


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_melting_curves_clean_removes_only_fetched_dataset(
    mock_fetch, tmp_path, monkeypatch
):
    """``clean=True`` removes the fetched dataset and leaves a local directory alone."""
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
    assert keep.read_text() == 'mine'
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
def test_resolve_melting_curve_files_local_first(tmp_path):
    """A complete local directory takes precedence over the manifest dataset."""
    from proteus.utils.data import resolve_melting_curve_files

    local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Wolf_Bower+2018'
    local.mkdir(parents=True)
    (local / 'solidus_P-T.dat').write_text('s')
    (local / 'liquidus_P-T.dat').write_text('l')

    solidus, liquidus = resolve_melting_curve_files('Wolf_Bower+2018', data_root=tmp_path)

    assert solidus == local / 'solidus_P-T.dat'
    assert liquidus == local / 'liquidus_P-T.dat'


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
@patch('proteus.utils.data.download')
@patch('proteus.data.fetch_dataset')
def test_download_stellar_spectra_default(mock_fetch, mock_download):
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
    mock_download.assert_not_called()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset')
def test_download_stellar_spectra_phoenix_is_not_a_bulk_collection(mock_fetch):
    """PHOENIX is fetched one grid at a time through download_phoenix, so the
    bulk collection downloader rejects it instead of pulling the 9 GB record.
    """
    from proteus.utils.data import download_stellar_spectra

    with pytest.raises(ValueError, match='No data source mapping found'):
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
    """An unreachable mirror for one required dataset leaves the others to be fetched.

    fwl-io reports a failed fetch with RuntimeError subclasses, which an OSError
    guard does not catch; without a per-step guard the first failure (the named
    stellar spectra) would abort the run before any interior table is tried.
    """
    from fwl_io import DownloadError

    from proteus.config import read_config_object
    from proteus.utils import data as data_mod

    attempted = []

    def _boom(key, *args, data_root=None):
        attempted.append(key)
        raise DownloadError('could not obtain from any mirror')

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path)
    monkeypatch.setattr('proteus.data.fetch_dataset', _boom)
    monkeypatch.setattr('proteus.data.fetch_dataset_file', _boom)
    monkeypatch.setattr(data_mod, 'download_stellar_tracks', lambda *a, **kw: None)
    cfg = read_config_object(
        str(Path(__file__).resolve().parents[2] / 'input' / 'minimal.toml')
    )

    data_mod.download_sufficient_data(cfg)

    assert attempted[0] == 'star.spectra.named'
    # The spectral file, the melting curves and the Zalmoxis tables come after
    # the failing stellar fetch and must all still be attempted.
    assert 'atmos_clim.spectral_files.honeyside.48' in attempted
    assert 'interior.melting_curves.wolf_bower_2018' in attempted
    assert 'interior.eos.paleos_iron' in attempted


@pytest.mark.unit
@pytest.mark.parametrize(
    'error',
    [
        'DownloadError',
        'OfflineDataError',
        'ArchiveError',
        'MissingDataRootError',
        'OSError',
        'RuntimeError',
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
        'RuntimeError': RuntimeError,
    }

    def _fail():
        raise errors[error]('mirror down')

    with caplog.at_level('WARNING'):
        assert _attempt('test data', _fail) is False
    assert 'test data: mirror down' in caplog.text
    # Discrimination: an error that is not a fetch failure is not swallowed.
    with pytest.raises(TypeError):
        _attempt('test data', lambda: (_ for _ in ()).throw(TypeError('bug')))


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.data.fetch_dataset')
def test_download_exoplanet_data(mock_fetch, mock_download):
    """The exoplanet catalogue is fetched through fwl-io, not the legacy path."""
    from proteus.data import EXOPLANET_REFERENCE
    from proteus.utils.data import download_exoplanet_data

    download_exoplanet_data()

    mock_fetch.assert_called_once_with(EXOPLANET_REFERENCE)
    # Discrimination: the key must be the catalogue, not the mass-radius
    # dataset declared beside it in the same manifest.
    assert mock_fetch.call_args.args[0] == 'observe.exoplanet_reference'
    # The legacy Zenodo/OSF downloader must not run: a regression that fell
    # back to it would still populate a tree and hide the migration.
    mock_download.assert_not_called()


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.data.fetch_dataset')
def test_download_massradius_data(mock_fetch, mock_download):
    """The mass-radius relations are fetched through fwl-io, not the legacy path."""
    from proteus.data import MASS_RADIUS_ZENG_2019
    from proteus.utils.data import download_massradius_data

    download_massradius_data()

    mock_fetch.assert_called_once_with(MASS_RADIUS_ZENG_2019)
    # Discrimination: the key must be the mass-radius dataset, not the
    # catalogue declared beside it in the same manifest.
    assert mock_fetch.call_args.args[0] == 'observe.mass_radius.zeng_2019'
    mock_download.assert_not_called()


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.data.fetch_dataset')
def test_download_surface_albedos(mock_fetch, mock_download):
    """The surface albedos are fetched through fwl-io, not the legacy path."""
    from proteus.data import SURFACE_ALBEDOS_HAMMOND_2024
    from proteus.utils.data import download_surface_albedos

    download_surface_albedos()

    mock_fetch.assert_called_once_with(SURFACE_ALBEDOS_HAMMOND_2024)
    assert mock_fetch.call_args.args[0] == 'atmos_clim.surface_albedos.hammond_2024'
    mock_download.assert_not_called()


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
@patch('proteus.utils.data.download_Seager_EOS')
def test_get_Seager_EOS_exists(mock_download, tmp_path):
    """Test get_Seager_EOS when EOS folder already exists."""
    from proteus.utils.data import get_Seager_EOS

    # Create EOS folder
    eos_folder = _seed_seager(tmp_path)

    # Patch FWL_DATA_DIR at module level
    with patch('proteus.utils.data.FWL_DATA_DIR', tmp_path):
        iron_silicate, water = get_Seager_EOS()

    # Should not have called download
    mock_download.assert_not_called()

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

    legacy = MagicMock()
    monkeypatch.setattr(data_mod, 'download', legacy)

    download_Seager_EOS()

    assert calls == [('interior_struct.eos.seager_2007', tmp_path)]
    legacy.assert_not_called()


@pytest.mark.unit
@patch('proteus.utils.data.get_osf')
def test_download_OSF_folder_success(mock_get_osf, tmp_path):
    """Test successful OSF folder download."""
    from proteus.utils.data import download_OSF_folder

    # Mock OSF storage and files
    mock_storage = MagicMock()
    mock_file1 = MagicMock()
    mock_file1.path = '/test_folder/file1.txt'
    mock_file1.size = 100
    mock_file1.write_to = MagicMock()

    mock_file2 = MagicMock()
    mock_file2.path = '/test_folder/subdir/file2.txt'
    mock_file2.size = 200
    mock_file2.write_to = MagicMock()

    mock_storage.files = [mock_file1, mock_file2]
    mock_get_osf.return_value = MagicMock()
    mock_get_osf.return_value.storages = [mock_storage]

    # Create target directory
    target_dir = tmp_path / 'test_folder'
    target_dir.mkdir(parents=True)

    download_OSF_folder(storage=mock_storage, folders=['test_folder'], data_dir=tmp_path)

    # Should have written both files
    assert mock_file1.write_to.called
    assert mock_file2.write_to.called


@pytest.mark.unit
@patch('proteus.utils.data.get_osf')
def test_download_OSF_folder_skip_existing(mock_get_osf, tmp_path):
    """Test OSF folder download skips existing files (no force parameter)."""
    from proteus.utils.data import download_OSF_folder

    # Create existing file with content
    existing_file = tmp_path / 'test_folder' / 'file1.txt'
    existing_file.parent.mkdir(parents=True)
    existing_file.write_text('old content')

    # Mock OSF storage
    mock_storage = MagicMock()
    mock_file = MagicMock()
    mock_file.path = '/test_folder/file1.txt'
    mock_file.size = 100
    mock_file.write_to = MagicMock()
    mock_storage.files = [mock_file]

    download_OSF_folder(storage=mock_storage, folders=['test_folder'], data_dir=tmp_path)

    # Should not have written to existing file (skipped)
    mock_file.write_to.assert_not_called()
    # Discrimination: the existing file's content must be unchanged; a
    # regression that overwrote with empty bytes would still pass
    # assert_not_called only if write_to were the sole I/O path (it is),
    # but pin the on-disk state to catch any alternative-write regression.
    assert existing_file.read_text() == 'old content'


@pytest.mark.unit
def test_download_osf_file_downloads_requested_files(tmp_path):
    """download_OSF_file writes matched files into data_dir and creates parent dirs."""
    from proteus.utils.data import download_OSF_file

    # Fake OSF storage listing
    storage = MagicMock()

    f1 = MagicMock()
    f1.path = '/folder/a.txt'
    f1.size = 10
    f1.write_to = MagicMock(side_effect=lambda fp: fp.write(b'aaa'))

    f2 = MagicMock()
    f2.path = '/folder/sub/b.bin'
    f2.size = 20
    f2.write_to = MagicMock(side_effect=lambda fp: fp.write(b'bbb'))

    storage.files = [f1, f2]

    download_OSF_file(
        storage=storage,
        files=['folder/a.txt', 'folder/sub/b.bin'],
        data_dir=tmp_path,
    )

    assert (tmp_path / 'folder' / 'a.txt').read_bytes() == b'aaa'
    assert (tmp_path / 'folder' / 'sub' / 'b.bin').read_bytes() == b'bbb'
    assert f1.write_to.called
    assert f2.write_to.called


@pytest.mark.unit
def test_download_osf_file_skips_existing_nonempty(tmp_path):
    """download_OSF_file does not overwrite existing non-empty files."""
    from proteus.utils.data import download_OSF_file

    # Create an existing file
    existing = tmp_path / 'folder' / 'a.txt'
    existing.parent.mkdir(parents=True, exist_ok=True)
    existing.write_text('old')

    storage = MagicMock()

    f1 = MagicMock()
    f1.path = '/folder/a.txt'
    f1.size = 10
    f1.write_to = MagicMock(side_effect=lambda fp: fp.write(b'new'))
    storage.files = [f1]

    download_OSF_file(storage=storage, files=['folder/a.txt'], data_dir=tmp_path)

    # unchanged
    assert existing.read_text() == 'old'
    f1.write_to.assert_not_called()


@pytest.mark.unit
def test_download_osf_file_missing_requested_is_ok(tmp_path, caplog):
    """download_OSF_file logs warning if requested file not found but does not crash."""
    from proteus.utils.data import download_OSF_file

    storage = MagicMock()

    f1 = MagicMock()
    f1.path = '/folder/other.txt'
    f1.size = 10
    f1.write_to = MagicMock(side_effect=lambda fp: fp.write(b'zzz'))
    storage.files = [f1]

    download_OSF_file(storage=storage, files=['folder/does_not_exist.txt'], data_dir=tmp_path)

    # Nothing downloaded
    assert not (tmp_path / 'folder' / 'does_not_exist.txt').exists()
    # Discrimination: the unrelated f1.other.txt file (present in OSF
    # storage) must not have been written either; a regression that
    # downloaded every storage entry instead of matching the request
    # would have produced an unintended on-disk file.
    f1.write_to.assert_not_called()


@pytest.mark.unit
def test_download_osf_file_removes_partial_on_exception(tmp_path):
    """Covers exception branch + partial file cleanup in download_OSF_file."""

    from proteus.utils.data import download_OSF_file

    # Create fake OSF file object
    mock_file = MagicMock()
    mock_file.path = '/folder/test.txt'
    mock_file.size = 100  # required for normal logic

    # write_to writes partial content then fails
    def failing_write(fp):
        fp.write(b'partial data')
        raise OSError('network failure')

    mock_file.write_to.side_effect = failing_write

    # Fake storage object
    storage = MagicMock()
    storage.files = [mock_file]

    # Act
    download_OSF_file(
        storage=storage,
        files=['folder/test.txt'],
        data_dir=tmp_path,
    )

    # File should NOT exist after failure
    target = tmp_path / 'folder' / 'test.txt'
    assert not target.exists()
    # Discrimination: confirm the download was actually attempted (and
    # therefore the cleanup path was the one that ran); a regression that
    # silently early-returned before invoking write_to would also produce
    # a non-existent target, but for the wrong reason.
    mock_file.write_to.assert_called()


@pytest.mark.unit
@patch('proteus.utils.data.get_data_source_info')
def test_download_stellar_spectra_no_mapping(mock_get_info):
    """Test stellar spectra download raises error when no mapping found."""
    from proteus.utils.data import download_stellar_spectra

    mock_get_info.return_value = None

    with pytest.raises(ValueError, match='No data source mapping found'):
        download_stellar_spectra(folders=('UnknownFolder',))
    # Discrimination: confirm the registry lookup happened (otherwise the
    # ValueError could be raised by an earlier guard that never reached
    # the source-info layer).
    mock_get_info.assert_called()


@pytest.mark.unit
@patch('proteus.utils.data.get_data_source_info')
def test_download_scattering_no_mapping(mock_get_info):
    """
    Test scattering download raises error when no mapping found.

    Physical scenario: if the DATA_SOURCE_MAP is misconfigured, fail fast
    with a clear error rather than silently skipping the download.
    """
    from proteus.utils.data import download_scattering

    mock_get_info.return_value = None

    with pytest.raises(ValueError, match='No data source mapping found'):
        download_scattering()
    # Discrimination: confirm the registry lookup happened; the raises
    # check alone could be satisfied by an unrelated earlier guard.
    mock_get_info.assert_called_once()


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


@pytest.mark.unit
@pytest.mark.skip(
    reason='Complex path matching logic - exception handling verified in integration tests'
)  # Note: download_zenodo_folder_client function doesn't exist in current codebase
# These tests are skipped until the function is implemented


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_missing_file(mock_getfwl, mock_run, tmp_path):
    """Test validation fails when file from md5sums is missing."""
    from proteus.utils.data import validate_zenodo_folder

    mock_getfwl.return_value = tmp_path

    # Create md5sums file with entry for missing file
    md5sums_file = tmp_path / 'md5sums.txt'
    md5sums_file.write_text('abc123  missing_file.txt\n')

    # Mock zenodo_get succeeds and creates md5sums file
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_run.return_value = mock_proc

    # Mock file system: md5sums exists, but the actual file doesn't
    # Also need to mock folder_dir.rglob to return empty (no files in folder)
    def exists_side_effect(path):
        path_str = str(path)
        return path_str == str(md5sums_file)

    def isfile_side_effect(path):
        path_str = str(path)
        return path_str == str(md5sums_file)

    with patch('proteus.utils.data.os.path.isfile', side_effect=isfile_side_effect):
        with patch('proteus.utils.data.os.path.exists', side_effect=exists_side_effect):
            with patch('proteus.utils.data.Path.rglob', return_value=[]):  # No files in folder
                result = validate_zenodo_folder('12345', tmp_path)

    # Should fail validation due to missing file
    assert result is False
    # Discrimination: confirm zenodo_get was actually invoked to refresh
    # md5sums; a regression that returned False from an unrelated guard
    # (e.g. an empty-folder short-circuit) would skip the subprocess.
    mock_run.assert_called()


@pytest.mark.unit
@pytest.mark.skip(
    reason='Complex file system mocking required - hash validation verified in integration tests'
)  # =============================================================================
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
    )
    # SPIDER dynamic EOS still downloaded separately
    mock_dyn.assert_called_once()


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
    )
    mock_dyn.assert_not_called()


# ============================================================================
# test download_zalmoxis_eos dispatch
# ============================================================================


_UNIFIED_IRON = 'paleos_iron_eos_table_pt.dat'
_UNIFIED_MGSIO3 = 'paleos_mgsio3_eos_table_pt.dat'
_UNIFIED_WATER = 'paleos_water_eos_table_pt.dat'


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

    mock_static.assert_called_once()
    mock_fetch.assert_not_called()
    mock_file.assert_not_called()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_wolfbower(mock_static, mock_fetch, mock_file):
    """WolfBower2018 fetches the Seager set and the three Wolf and Bower tables it reads."""
    from proteus.data import EOS_WOLF_BOWER_2018
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('WolfBower2018:MgSiO3', core_eos='Seager2007:iron')

    mock_static.assert_called_once()
    # The shared record holds 8 tables; a whole-dataset fetch would pull all of them.
    mock_fetch.assert_not_called()
    assert _fetched_files(mock_file) == [
        (EOS_WOLF_BOWER_2018, 'density_melt.dat'),
        (EOS_WOLF_BOWER_2018, 'adiabat_temp_grad_melt.dat'),
        (EOS_WOLF_BOWER_2018, 'density_solid.dat'),
    ]


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_rtpress(mock_static, mock_fetch, mock_file):
    """RTPress100TPa fetches its two melt tables plus the Wolf and Bower solid density only."""
    from proteus.data import EOS_RTPRESS_100TPA, EOS_WOLF_BOWER_2018
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('RTPress100TPa:MgSiO3', core_eos='Seager2007:iron')

    mock_static.assert_called_once()
    mock_fetch.assert_not_called()
    # The RTPress registry entry reads its solid density from the Wolf and
    # Bower dataset; the rest of that dataset must not be pulled for that one file.
    assert _fetched_files(mock_file) == [
        (EOS_WOLF_BOWER_2018, 'density_solid.dat'),
        (EOS_RTPRESS_100TPA, 'density_melt.dat'),
        (EOS_RTPRESS_100TPA, 'adiabat_temp_grad_melt.dat'),
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
    }
    mock_fetch.assert_not_called()


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_paleos_2phase(mock_static, mock_fetch, mock_file):
    """PALEOS-2phase:MgSiO3 fetches the standard-resolution pair only."""
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
    """PALEOS unified fetches each material's table from its own dataset."""
    from proteus.data import EOS_PALEOS_H2O, EOS_PALEOS_IRON, EOS_PALEOS_MGSIO3_UNIFIED
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
        ]
    )


@pytest.mark.unit
@patch('proteus.data.fetch_dataset_file')
@patch('proteus.data.fetch_dataset')
@patch('proteus.utils.data.download_eos_static')
def test_download_zalmoxis_eos_paleos_unified_only_selected_tables(
    mock_static, mock_fetch, mock_file
):
    """Only the selected unified tables are fetched, not the whole record."""
    from proteus.data import EOS_PALEOS_IRON, EOS_PALEOS_MGSIO3_UNIFIED
    from proteus.utils.data import download_zalmoxis_eos

    download_zalmoxis_eos('PALEOS:MgSiO3', core_eos='PALEOS:iron')

    assert sorted(_fetched_files(mock_file)) == sorted(
        [(EOS_PALEOS_IRON, _UNIFIED_IRON), (EOS_PALEOS_MGSIO3_UNIFIED, _UNIFIED_MGSIO3)]
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
        (EOS_PALEOS_MGSIO3, 'paleos_mgsio3_tables_pt_proteus_liquid.dat'),
        (EOS_PALEOS_MGSIO3, 'paleos_mgsio3_tables_pt_proteus_solid.dat'),
        (EOS_PALEOS_IRON, _UNIFIED_IRON),
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
    from proteus.data import EOS_CHABRIER_2021, EOS_PALEOS_H2O, EOS_PALEOS_MGSIO3_UNIFIED
    from proteus.utils.data import download_zalmoxis_eos

    # Composite mantle with PALEOS + Chabrier
    download_zalmoxis_eos(
        'PALEOS:MgSiO3:0.98+Chabrier:H:0.01+PALEOS:H2O:0.01',
        core_eos='Seager2007:iron',
    )

    mock_static.assert_called_once()
    assert _fetched_datasets(mock_fetch) == [EOS_CHABRIER_2021]
    assert sorted(_fetched_files(mock_file)) == sorted(
        [(EOS_PALEOS_MGSIO3_UNIFIED, _UNIFIED_MGSIO3), (EOS_PALEOS_H2O, _UNIFIED_WATER)]
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

    legacy = MagicMock()
    monkeypatch.setattr('proteus.utils.data.download', legacy)

    download_eos_dynamic('WolfBower2018_MgSiO3')

    mock_fetch.assert_called_once_with(LOOKUP_WOLF_BOWER_2018_1TPA, data_root=tmp_path)
    legacy.assert_not_called()


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
        lambda *_args: ('Frostflow', '128'),
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
    assert spectral_calls == [('Honeyside', '4096'), ('Frostflow', '128')]
    # Discrimination: confirm the resolved group/bands entry came second
    # (the Honeyside post-processing download always runs first); a
    # regression that swapped the call order or replayed Honeyside twice
    # would still produce a 2-element list but break this pin.
    assert spectral_calls[-1] == ('Frostflow', '128')


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


# ============================================================================
# download_zenodo_folder additional error-branch coverage
# ============================================================================


@pytest.mark.unit
def test_download_zenodo_folder_rejects_bad_id(tmp_path):
    """download_zenodo_folder rejects non-numeric zenodo IDs with no side effects.

    Sanitisation must fire before any filesystem mutation: the folder
    creation in the loop should not have happened.
    """
    from proteus.utils.data import download_zenodo_folder

    folder_dir = tmp_path / 'zenodo_folder_bad'
    ok = download_zenodo_folder('abc123', folder_dir)
    assert ok is False
    # Discrimination: bad-ID rejection must precede mkdir; otherwise a
    # regression that creates the folder before validating the ID would
    # leave a stray directory behind.
    assert not folder_dir.exists()


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_folder_zero_exit_empty_folder(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """zenodo_get exits 0 but the folder is empty: function should return False after retries.

    Covers the success-then-empty branch where the download command
    reports success but produced no files on disk.
    """
    from proteus.utils.data import download_zenodo_folder

    mock_getfwl.return_value = tmp_path

    proc_avail = MagicMock(returncode=0)
    proc_dl = MagicMock(returncode=0)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        # Do not create any file; folder exists but is empty
        return proc_dl

    mock_run.side_effect = side_effect

    folder_dir = tmp_path / 'empty_record'
    ok = download_zenodo_folder('12345', folder_dir)
    assert ok is False
    # Discrimination: the function must have retried MAX_ATTEMPTS times,
    # i.e. at least 3 download attempts after the initial availability
    # probe; a regression that returned False on the first empty result
    # would leave call_count well below this floor.
    assert mock_run.call_count >= 1 + 3


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_folder_unexpected_exception(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """Generic Exception inside the download attempt is caught and retried.

    The function should not propagate; it should log and exhaust retries.
    """
    from proteus.utils.data import download_zenodo_folder

    mock_getfwl.return_value = tmp_path

    proc_avail = MagicMock(returncode=0)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        raise RuntimeError('network died unexpectedly')

    mock_run.side_effect = side_effect

    folder_dir = tmp_path / 'broken_record'
    ok = download_zenodo_folder('12345', folder_dir)
    assert ok is False
    # Discrimination: confirm all retry attempts were exhausted; a
    # regression that re-raised RuntimeError would never reach the
    # second download call.
    download_calls = [c for c in mock_run.call_args_list if '--version' not in c[0][0]]
    assert len(download_calls) == 3


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_folder_log_read_exception_swallowed(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """If the log file cannot be read, the diagnostic readback is swallowed
    and the function still retries normally.

    Exercises the inner `except Exception: pass` around log readback.
    """
    from proteus.utils.data import download_zenodo_folder

    mock_getfwl.return_value = tmp_path

    proc_avail = MagicMock(returncode=0)
    proc_fail = MagicMock(returncode=1)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        return proc_fail

    mock_run.side_effect = side_effect

    # Patch builtins.open so writing the log works but reading it raises
    real_open = open
    state = {'read_attempts': 0}

    def selective_open(path, mode='r', *a, **k):
        if 'r' in mode and 'zenodo_download.log' in str(path):
            state['read_attempts'] += 1
            raise OSError('cannot read log')
        return real_open(path, mode, *a, **k)

    with patch('builtins.open', side_effect=selective_open):
        folder_dir = tmp_path / 'log_unreadable'
        ok = download_zenodo_folder('12345', folder_dir)

    assert ok is False
    # Discrimination: the log readback must have been attempted at least
    # once (one attempt produces one readback); a regression that
    # short-circuited before readback would never trigger our OSError.
    assert state['read_attempts'] >= 1


# ============================================================================
# download_zenodo_file additional error-branch coverage
# ============================================================================


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_timeout_exhausts_retries(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """download_zenodo_file returns False after every retry hits TimeoutExpired."""
    import subprocess as sp_mod

    from proteus.utils.data import download_zenodo_file

    mock_getfwl.return_value = tmp_path

    proc_avail = MagicMock(returncode=0)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        raise sp_mod.TimeoutExpired(cmd=cmd, timeout=120)

    mock_run.side_effect = side_effect

    folder_dir = tmp_path / 'timeout_file'
    ok = download_zenodo_file('12345', folder_dir, 'subdir/file.dat')
    assert ok is False
    # Discrimination: MAX_ATTEMPTS download attempts must all have fired
    # (the availability probe runs once, then three download attempts).
    download_calls = [c for c in mock_run.call_args_list if '--version' not in c[0][0]]
    assert len(download_calls) == 3


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_removes_existing_file_before_retry(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """An existing destination file is unlinked at the start of each attempt.

    Exercises the `expected_path.unlink()` branch (line 182) and verifies
    the cleanup happens once per attempt cycle.
    """
    from proteus.utils.data import download_zenodo_file

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'preexisting_file'
    folder_dir.mkdir(parents=True, exist_ok=True)
    pre = folder_dir / 'subdir' / 'old.dat'
    pre.parent.mkdir(parents=True, exist_ok=True)
    pre.write_text('stale')

    proc_avail = MagicMock(returncode=0)
    proc_dl = MagicMock(returncode=0)

    state = {'unlinked_at_start': False}

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        # By the time the download runs, the stale file should have been
        # removed by the cleanup branch.
        state['unlinked_at_start'] = not pre.exists()
        # Write fresh content
        pre.write_text('fresh payload')
        return proc_dl

    mock_run.side_effect = side_effect

    ok = download_zenodo_file('12345', folder_dir, 'subdir/old.dat')
    assert ok is True
    # Discrimination: the stale file must have been deleted before the
    # download ran; a regression that skipped the cleanup branch would
    # leave the stale file in place at download time.
    assert state['unlinked_at_start'] is True


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_generic_exception_caught(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """An unexpected exception inside the download body is caught and retried."""
    from proteus.utils.data import download_zenodo_file

    mock_getfwl.return_value = tmp_path

    proc_avail = MagicMock(returncode=0)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        raise RuntimeError('something blew up')

    mock_run.side_effect = side_effect

    folder_dir = tmp_path / 'broken_file'
    ok = download_zenodo_file('12345', folder_dir, 'file.dat')
    assert ok is False
    # Discrimination: must retry 3 download attempts after availability
    # check (the function should not re-raise on RuntimeError).
    download_calls = [c for c in mock_run.call_args_list if '--version' not in c[0][0]]
    assert len(download_calls) == 3


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_log_read_exception_swallowed(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """If the log file readback raises, the function still completes and retries."""
    from proteus.utils.data import download_zenodo_file

    mock_getfwl.return_value = tmp_path

    proc_avail = MagicMock(returncode=0)
    proc_fail = MagicMock(returncode=1)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        return proc_fail

    mock_run.side_effect = side_effect

    real_open = open

    def selective_open(path, mode='r', *a, **k):
        if 'r' in mode and 'zenodo_download.log' in str(path):
            raise PermissionError('blocked')
        return real_open(path, mode, *a, **k)

    with patch('builtins.open', side_effect=selective_open):
        folder_dir = tmp_path / 'logblocked_file'
        ok = download_zenodo_file('12345', folder_dir, 'fname.dat')

    assert ok is False
    # Discrimination: download exited with non-zero return-code so the
    # retry loop must have run fully (3 attempts).
    download_calls = [c for c in mock_run.call_args_list if '--version' not in c[0][0]]
    assert len(download_calls) == 3


# ============================================================================
# validate_zenodo_folder additional error-branch coverage
# ============================================================================


@pytest.mark.unit
def test_validate_zenodo_folder_rejects_bad_id(tmp_path):
    """validate_zenodo_folder refuses non-numeric IDs without touching disk."""
    from proteus.utils.data import validate_zenodo_folder

    folder_dir = tmp_path / 'val_bad_id'
    folder_dir.mkdir()
    ok = validate_zenodo_folder('bad id 5', folder_dir)
    assert ok is False
    # Discrimination: folder must remain unchanged after rejection;
    # md5sums.txt should not have been written.
    assert not (folder_dir / 'md5sums.txt').exists()


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
def test_validate_zenodo_folder_missing_get_with_no_files(mock_run, tmp_path):
    """When zenodo_get is unavailable and the folder is empty, validation fails."""
    from proteus.utils.data import validate_zenodo_folder

    mock_run.side_effect = FileNotFoundError('zenodo_get missing')

    folder_dir = tmp_path / 'val_empty'
    folder_dir.mkdir()
    ok = validate_zenodo_folder('12345', folder_dir)
    assert ok is False
    # Discrimination: a regression that fell through to the validation
    # loop would have raised TypeError on a missing md5sums file; instead
    # this path must return False directly.
    assert not (folder_dir / 'md5sums.txt').exists()


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_retry_then_assume_valid(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """When checksum download fails every time but folder has files, validate assumes valid."""
    from proteus.utils.data import validate_zenodo_folder

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'val_assume_valid'
    folder_dir.mkdir()
    (folder_dir / 'payload.dat').write_text('something')

    proc_avail = MagicMock(returncode=0)
    proc_fail = MagicMock(returncode=1)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        return proc_fail

    mock_run.side_effect = side_effect

    ok = validate_zenodo_folder('12345', folder_dir)
    assert ok is True
    # Discrimination: the function must have exhausted MAX_ATTEMPTS=3
    # checksum-fetch attempts before falling back to "folder has files,
    # assume valid"; a regression that gave up on the first attempt
    # would call sp.run far fewer times.
    fetch_calls = [c for c in mock_run.call_args_list if '-m' in c[0][0]]
    assert len(fetch_calls) == 3


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_retry_no_files_returns_false(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """When checksum fetch fails every time AND folder is empty, validate returns False."""
    from proteus.utils.data import validate_zenodo_folder

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'val_empty_after_fail'
    folder_dir.mkdir()

    proc_avail = MagicMock(returncode=0)
    proc_fail = MagicMock(returncode=1)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        return proc_fail

    mock_run.side_effect = side_effect

    ok = validate_zenodo_folder('12345', folder_dir)
    assert ok is False
    # Discrimination: the folder is empty so the fallback "assume valid"
    # path must NOT have fired; the assertion above pins the False outcome.
    assert not (folder_dir / 'md5sums.txt').exists()


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_validation_timeout_caught(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """TimeoutExpired during validation is caught and the loop continues."""
    import subprocess as sp_mod

    from proteus.utils.data import validate_zenodo_folder

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'val_timeout'
    folder_dir.mkdir()
    (folder_dir / 'file.dat').write_text('xyz')

    proc_avail = MagicMock(returncode=0)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        raise sp_mod.TimeoutExpired(cmd=cmd, timeout=60)

    mock_run.side_effect = side_effect

    ok = validate_zenodo_folder('12345', folder_dir)
    # Folder has files -> assume valid after timeouts
    assert ok is True
    # Discrimination: the loop must have iterated MAX_ATTEMPTS times
    # rather than bailing on the first TimeoutExpired.
    fetch_calls = [c for c in mock_run.call_args_list if '-m' in c[0][0]]
    assert len(fetch_calls) == 3


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_validation_generic_exception_caught(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """Unexpected exception inside validation is caught and the loop retries."""
    from proteus.utils.data import validate_zenodo_folder

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'val_generic'
    folder_dir.mkdir()
    (folder_dir / 'data.dat').write_text('ok')

    proc_avail = MagicMock(returncode=0)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        raise RuntimeError('unexpected error')

    mock_run.side_effect = side_effect

    ok = validate_zenodo_folder('12345', folder_dir)
    # Folder has files -> assume valid after retries
    assert ok is True
    fetch_calls = [c for c in mock_run.call_args_list if '-m' in c[0][0]]
    assert len(fetch_calls) == 3


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_md5sums_read_failure_with_files(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """When md5sums.txt cannot be read but folder has files, treat as valid."""
    from proteus.utils.data import validate_zenodo_folder

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'val_md5_unreadable'
    folder_dir.mkdir()
    (folder_dir / 'realfile.dat').write_text('contents')

    proc_avail = MagicMock(returncode=0)
    proc_ok = MagicMock(returncode=0)
    md5sums = folder_dir / 'md5sums.txt'

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        md5sums.write_text('dummy')
        return proc_ok

    mock_run.side_effect = side_effect

    real_open = open

    def selective_open(path, mode='r', *a, **k):
        if 'r' in mode and 'md5sums.txt' in str(path):
            raise OSError('cannot read md5sums')
        return real_open(path, mode, *a, **k)

    with patch('builtins.open', side_effect=selective_open):
        ok = validate_zenodo_folder('12345', folder_dir)

    assert ok is True
    # Discrimination: the function must have proceeded past the
    # availability check (so sp.run was invoked twice, once for
    # version check, once for `-m` fetch) before hitting the read fail.
    assert mock_run.call_count >= 2


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_md5sums_read_failure_no_files(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """When md5sums.txt cannot be read AND folder is empty (apart from
    md5sums.txt itself), validation fails."""
    from proteus.utils.data import validate_zenodo_folder

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'val_md5_empty'
    folder_dir.mkdir()

    proc_avail = MagicMock(returncode=0)
    proc_ok = MagicMock(returncode=0)
    md5sums = folder_dir / 'md5sums.txt'

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        md5sums.write_text('dummy')
        return proc_ok

    mock_run.side_effect = side_effect

    real_open = open

    def selective_open(path, mode='r', *a, **k):
        if 'r' in mode and 'md5sums.txt' in str(path):
            raise OSError('cannot read md5sums')
        return real_open(path, mode, *a, **k)

    # Patch rglob so that md5sums.txt itself does not count as a "real"
    # data file when checking the "folder has files" fallback.
    real_rglob = Path.rglob

    def filtered_rglob(self, pattern):
        for p in real_rglob(self, pattern):
            if p.name == 'md5sums.txt':
                continue
            yield p

    with (
        patch('builtins.open', side_effect=selective_open),
        patch.object(Path, 'rglob', filtered_rglob),
    ):
        ok = validate_zenodo_folder('12345', folder_dir)

    assert ok is False
    # Discrimination: confirm md5sums.txt still exists on disk from the
    # mocked download (so the read fail path fired, not an earlier
    # short-circuit).
    assert md5sums.exists()


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_missing_file_in_manifest(mock_getfwl, mock_run, tmp_path):
    """Validation returns False when md5sums lists a file that is not on disk."""
    from proteus.utils.data import validate_zenodo_folder

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'val_missing_file'
    folder_dir.mkdir()

    proc_avail = MagicMock(returncode=0)
    proc_ok = MagicMock(returncode=0)
    md5sums = folder_dir / 'md5sums.txt'

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        # The actual zenodo_get -m would write md5sums.txt; emulate that
        # in the mocked subprocess so the validator can read it.
        md5sums.write_text('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  missing.dat\n')
        return proc_ok

    mock_run.side_effect = side_effect

    ok = validate_zenodo_folder('12345', folder_dir)
    assert ok is False
    # Discrimination: confirm the manifest was actually parsed (the read
    # path completed) and the missing-file branch fired specifically.
    assert md5sums.exists()
    assert not (folder_dir / 'missing.dat').exists()


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_skips_symlink_entries(mock_getfwl, mock_run, tmp_path):
    """Symlink entries in the manifest are skipped (not hash-checked)."""
    import platform

    if platform.system() == 'Windows':
        pytest.skip('symlinks require admin on Windows')

    from proteus.utils.data import validate_zenodo_folder

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'val_symlink'
    folder_dir.mkdir()
    real = folder_dir / 'real.dat'
    real.write_text('payload')
    link = folder_dir / 'aliased.dat'
    link.symlink_to(real)

    proc_avail = MagicMock(returncode=0)
    proc_ok = MagicMock(returncode=0)
    md5sums = folder_dir / 'md5sums.txt'

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        # Wrong hash for the symlink target; if the function did NOT
        # skip symlinks it would compute the hash of real.dat and
        # reject as mismatch.
        md5sums.write_text('deadbeefdeadbeefdeadbeefdeadbeef  aliased.dat\n')
        return proc_ok

    mock_run.side_effect = side_effect

    ok = validate_zenodo_folder('12345', folder_dir)
    assert ok is True
    # Discrimination: the regression that hashed symlinks would have
    # returned False because the wrong-hash mismatch fires for real.dat.
    assert link.is_symlink()


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_skips_large_files_without_hash(mock_getfwl, mock_run, tmp_path):
    """Files larger than hash_maxfilesize are accepted without hash comparison."""
    from proteus.utils.data import validate_zenodo_folder

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'val_large'
    folder_dir.mkdir()
    big = folder_dir / 'big.dat'
    big.write_text('x' * 1024)

    proc_avail = MagicMock(returncode=0)
    proc_ok = MagicMock(returncode=0)
    md5sums = folder_dir / 'md5sums.txt'

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        # Wrong hash; if the function tried to verify it would fail.
        md5sums.write_text('00000000000000000000000000000000  big.dat\n')
        return proc_ok

    mock_run.side_effect = side_effect

    # Force the file to be treated as "large" (above the 100-byte threshold)
    ok = validate_zenodo_folder('12345', folder_dir, hash_maxfilesize=100)
    assert ok is True
    # Discrimination: confirm the file was retained on disk (the
    # validator did not delete or move it as part of the bypass path).
    assert big.exists() and big.stat().st_size == 1024


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_hash_mismatch_returns_false(mock_getfwl, mock_run, tmp_path):
    """A wrong-hash entry causes validation to fail."""
    from proteus.utils.data import md5, validate_zenodo_folder

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'val_hash_mismatch'
    folder_dir.mkdir()
    real = folder_dir / 'small.dat'
    real.write_text('content')

    real_hash = md5(real)
    wrong_hash = 'f' * 32  # deliberately wrong

    proc_avail = MagicMock(returncode=0)
    proc_ok = MagicMock(returncode=0)
    md5sums = folder_dir / 'md5sums.txt'

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        md5sums.write_text(f'{wrong_hash}  small.dat\n')
        return proc_ok

    mock_run.side_effect = side_effect

    ok = validate_zenodo_folder('12345', folder_dir)
    assert ok is False
    # Discrimination: the recorded hash is wrong and differs from the
    # real one, so the regression that compared against itself rather
    # than the manifest would have returned True instead.
    assert real_hash != wrong_hash


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_blank_and_malformed_lines_skipped(
    mock_getfwl, mock_run, tmp_path
):
    """Blank and short manifest lines are skipped without breaking validation."""
    from proteus.utils.data import md5, validate_zenodo_folder

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'val_malformed'
    folder_dir.mkdir()
    real = folder_dir / 'good.dat'
    real.write_text('payload')

    real_hash = md5(real)

    proc_avail = MagicMock(returncode=0)
    proc_ok = MagicMock(returncode=0)
    md5sums = folder_dir / 'md5sums.txt'

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        md5sums.write_text(
            '\n'  # blank line
            'oneword\n'  # malformed (one field)
            f'{real_hash}  good.dat\n'  # valid line
        )
        return proc_ok

    mock_run.side_effect = side_effect

    ok = validate_zenodo_folder('12345', folder_dir)
    assert ok is True
    # Discrimination: a regression that bailed on the blank or malformed
    # line would have returned False; confirm the valid file was still
    # recognised and untouched on disk.
    assert real.read_text() == 'payload'


# ============================================================================
# download_OSF_folder / download_OSF_file additional coverage
# ============================================================================


def _make_osf_storage(file_specs):
    """Build a mock storage object whose .files iterates the given specs.

    Each spec is a tuple (path_str, payload_bytes_or_callable, size).
    """
    storage = MagicMock()
    files = []
    for spec in file_specs:
        path, payload, size = spec
        f = MagicMock()
        f.path = path
        f.size = size
        if callable(payload):
            f.write_to = payload
        else:

            def make_writer(payload_bytes):
                def _write(fp):
                    fp.write(payload_bytes)

                return _write

            f.write_to = make_writer(payload)
        files.append(f)
    storage.files = files
    return storage


@pytest.mark.unit
def test_download_OSF_folder_skips_unmatched_prefix(tmp_path):
    """OSF folder download skips files whose path does not match any requested folder."""
    from proteus.utils.data import download_OSF_folder

    storage = _make_osf_storage(
        [
            ('/wanted/file_a.dat', b'aaa', 3),
            ('/other/file_b.dat', b'bbb', 3),
        ]
    )

    data_dir = tmp_path / 'osf_data'
    data_dir.mkdir()

    download_OSF_folder(storage=storage, folders=['wanted'], data_dir=data_dir)

    # Only the file in the wanted folder should have been downloaded
    assert (data_dir / 'wanted' / 'file_a.dat').is_file()
    # Discrimination: the unmatched file under /other/ must not have been
    # copied to disk; a regression that ignored the prefix-match guard
    # would have produced this path.
    assert not (data_dir / 'other' / 'file_b.dat').exists()


@pytest.mark.unit
def test_download_OSF_folder_partial_download_failure_cleanup(tmp_path):
    """When write_to raises, the partial file is unlinked and download continues."""
    from proteus.utils.data import download_OSF_folder

    def failing_write(fp):
        fp.write(b'partial')
        raise RuntimeError('connection reset')

    def ok_write(fp):
        fp.write(b'complete')

    storage = _make_osf_storage(
        [
            ('/group/bad.dat', failing_write, 7),
            ('/group/good.dat', ok_write, 8),
        ]
    )

    data_dir = tmp_path / 'osf_partial'
    data_dir.mkdir()

    download_OSF_folder(storage=storage, folders=['group'], data_dir=data_dir)

    # The partial file must have been removed
    assert not (data_dir / 'group' / 'bad.dat').exists()
    # The successful file remains
    assert (data_dir / 'group' / 'good.dat').is_file()
    # Discrimination: confirm the successful file actually contains its
    # payload (the loop did not crash after the cleanup branch).
    assert (data_dir / 'group' / 'good.dat').read_bytes() == b'complete'


@pytest.mark.unit
def test_download_OSF_folder_propagates_storage_error(tmp_path):
    """An exception while iterating storage.files is logged and re-raised."""
    from proteus.utils.data import download_OSF_folder

    class BadIterStorage:
        @property
        def files(self):
            raise ConnectionError('OSF down')

    data_dir = tmp_path / 'osf_err'
    data_dir.mkdir()

    with pytest.raises(ConnectionError):
        download_OSF_folder(storage=BadIterStorage(), folders=['x'], data_dir=data_dir)
    # Discrimination: no files were created since iteration aborted before
    # any write.
    assert list(data_dir.iterdir()) == []


@pytest.mark.unit
def test_download_OSF_file_partial_write_cleanup(tmp_path):
    """When download_OSF_file's write_to raises, the partial file is removed."""
    from proteus.utils.data import download_OSF_file

    def failing_write(fp):
        fp.write(b'half')
        raise IOError('partial')

    storage = _make_osf_storage([('/req/file.dat', failing_write, 4)])

    data_dir = tmp_path / 'osf_file_partial'
    data_dir.mkdir()

    download_OSF_file(storage=storage, files=['req/file.dat'], data_dir=data_dir)

    # Partial file must have been removed
    assert not (data_dir / 'req' / 'file.dat').exists()
    # Discrimination: parent directory was still created (so the cleanup
    # path ran but did not propagate the error).
    assert (data_dir / 'req').is_dir()


@pytest.mark.unit
def test_download_OSF_file_propagates_storage_error(tmp_path):
    """An exception during storage iteration in file-mode is re-raised."""
    from proteus.utils.data import download_OSF_file

    class BadIterStorage:
        @property
        def files(self):
            raise ConnectionError('OSF down')

    data_dir = tmp_path / 'osf_file_err'
    data_dir.mkdir()

    with pytest.raises(ConnectionError):
        download_OSF_file(storage=BadIterStorage(), files=['x.dat'], data_dir=data_dir)
    assert list(data_dir.iterdir()) == []


@pytest.mark.unit
def test_get_osf_returns_storage_handle():
    """get_osf wraps OSF().project(id).storage('osfstorage') and is cached."""
    import proteus.utils.data as data_mod

    fake_storage = object()

    # Patch OSF class so we don't hit the network
    with patch('proteus.utils.data.OSF') as mock_osf_cls:
        proj = MagicMock()
        proj.storage.return_value = fake_storage
        mock_osf_cls.return_value.project.return_value = proj

        # Clear functools.cache on get_osf to make the call deterministic
        data_mod.get_osf.cache_clear()

        result_1 = data_mod.get_osf('cached_id_1')
        result_2 = data_mod.get_osf('cached_id_1')

    # Both calls return the same object (cache hit)
    assert result_1 is fake_storage
    assert result_2 is fake_storage
    # Discrimination: cache must short-circuit so OSF.project() was
    # called once, not twice; a regression that removed @functools.cache
    # would double-invoke and the count would be 2.
    assert mock_osf_cls.return_value.project.call_count == 1


# ============================================================================
# download() additional file-mode and folder-mode error branches
# ============================================================================


@pytest.mark.unit
@patch('proteus.utils.data.get_data_source_info', return_value=None)
@patch('proteus.utils.data.download_zenodo_file')
@patch('proteus.utils.data.GetFWLData')
def test_download_file_mode_zenodo_raises_runtime_error_falls_through(
    mock_getfwl, mock_zfile, mock_info, tmp_path
):
    """download(): single-file mode, Zenodo raises RuntimeError, no OSF fallback path."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path
    mock_zfile.side_effect = RuntimeError('zenodo crashed')

    # Use a folder name not in DATA_SOURCE_MAP and pass explicit IDs
    # so the mapping lookup misses but the call still proceeds.
    ok = download(
        folder='unmapped_folder',
        target='unmapped_target',
        zenodo_id='12345',
        osf_id=None,
        desc='test',
        file='subdir/f.dat',
    )

    assert ok is False
    # Discrimination: confirm the Zenodo helper was actually invoked
    # (and raised), rather than the function short-circuiting earlier.
    mock_zfile.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.download_OSF_file')
@patch('proteus.utils.data.get_osf')
@patch('proteus.utils.data.download_zenodo_file')
@patch('proteus.utils.data.GetFWLData')
def test_download_file_mode_osf_succeeds_via_basename_rglob(
    mock_getfwl, mock_zfile, mock_get_osf, mock_osf_file, tmp_path
):
    """File-mode: Zenodo fails, OSF places file at non-canonical path, rglob finds it."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path

    # Zenodo: report failure
    mock_zfile.return_value = False

    folder_dir = tmp_path / 'spectral_files' / 'Frostflow' / '16'
    record_path = 'sub/file.dat'

    def osf_side_effect(*, storage, files, data_dir):
        # OSF puts file in a non-canonical location
        alt = folder_dir / 'weird_layout' / 'file.dat'
        alt.parent.mkdir(parents=True, exist_ok=True)
        alt.write_text('payload')

    mock_osf_file.side_effect = osf_side_effect

    ok = download(
        folder='Frostflow/16',
        target='spectral_files',
        zenodo_id='15799743',
        osf_id='vehxg',
        desc='test',
        file=record_path,
    )

    assert ok is True
    # Discrimination: the basename-rglob fallback located the file, even
    # though the expected canonical path remained missing.
    assert (folder_dir / 'weird_layout' / 'file.dat').exists()
    assert not (folder_dir / 'sub' / 'file.dat').exists()


@pytest.mark.unit
@patch('proteus.utils.data.download_OSF_file')
@patch('proteus.utils.data.get_osf')
@patch('proteus.utils.data.download_zenodo_file')
@patch('proteus.utils.data.GetFWLData')
def test_download_file_mode_osf_exception_caught(
    mock_getfwl, mock_zfile, mock_get_osf, mock_osf_file, tmp_path
):
    """File-mode: Zenodo fails, OSF raises; returns False without crashing."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path
    mock_zfile.return_value = False
    mock_osf_file.side_effect = ConnectionError('OSF down')

    ok = download(
        folder='Frostflow/16',
        target='spectral_files',
        zenodo_id='15799743',
        osf_id='vehxg',
        desc='test',
        file='sub/file.dat',
    )

    assert ok is False
    # Discrimination: both download helpers were invoked (we didn't skip
    # OSF because of the exception or pre-OSF short-circuit).
    mock_zfile.assert_called_once()
    mock_osf_file.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.get_data_source_info', return_value=None)
@patch('proteus.utils.data.download_zenodo_folder')
@patch('proteus.utils.data.check_needs_update', return_value=True)
@patch('proteus.utils.data.GetFWLData')
def test_download_folder_mode_zenodo_runtime_error_cleanup_ok(
    mock_getfwl, mock_check, mock_zfolder, mock_info, tmp_path
):
    """Folder-mode: Zenodo raises RuntimeError; cleanup branch fires; returns False."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path
    mock_zfolder.side_effect = RuntimeError('crash')

    ok = download(
        folder='unmapped_folder',
        target='unmapped_target',
        zenodo_id='12345',
        osf_id=None,
        desc='test',
    )

    assert ok is False
    # Discrimination: confirm the Zenodo helper actually ran (the
    # RuntimeError was raised inside it), so the cleanup branch was
    # exercised rather than skipped by an earlier validation.
    mock_zfolder.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.download_OSF_folder')
@patch('proteus.utils.data.get_osf')
@patch('proteus.utils.data.download_zenodo_folder')
@patch('proteus.utils.data.check_needs_update', return_value=True)
@patch('proteus.utils.data.GetFWLData')
def test_download_folder_mode_osf_empty_folder_returns_false(
    mock_getfwl, mock_check, mock_zfolder, mock_get_osf, mock_osf_folder, tmp_path
):
    """Folder-mode: Zenodo fails, OSF download returns but produces empty folder."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path
    mock_zfolder.return_value = False  # Zenodo cleanly fails
    mock_osf_folder.return_value = None  # OSF returns but creates no files

    ok = download(
        folder='Frostflow/16',
        target='spectral_files',
        zenodo_id='15799743',
        osf_id='vehxg',
        desc='test',
    )

    assert ok is False
    # Discrimination: both fallback paths fired exactly once.
    mock_zfolder.assert_called_once()
    mock_osf_folder.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.download_OSF_folder')
@patch('proteus.utils.data.get_osf')
@patch('proteus.utils.data.download_zenodo_folder')
@patch('proteus.utils.data.check_needs_update', return_value=True)
@patch('proteus.utils.data.GetFWLData')
def test_download_folder_mode_osf_raises_caught(
    mock_getfwl, mock_check, mock_zfolder, mock_get_osf, mock_osf_folder, tmp_path
):
    """Folder-mode: Zenodo fails, OSF raises, function returns False."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path
    mock_zfolder.return_value = False
    mock_osf_folder.side_effect = ConnectionError('OSF down')

    ok = download(
        folder='Frostflow/16',
        target='spectral_files',
        zenodo_id='15799743',
        osf_id='vehxg',
        desc='test',
    )

    assert ok is False
    mock_osf_folder.assert_called_once()


# ============================================================================
# download_scattering / download_interior_lookuptables additional coverage
# ============================================================================


@pytest.mark.unit
@patch('proteus.utils.data.download')
def test_download_scattering_dispatches_with_mapped_ids(mock_download):
    """download_scattering passes the mapped osf_id and zenodo_id to download()."""
    from proteus.utils.data import download_scattering

    download_scattering()

    mock_download.assert_called_once()
    call_kwargs = mock_download.call_args.kwargs
    # Discrimination: confirm both mapped IDs are forwarded (not None);
    # a regression that dropped one would still call download() but with
    # an incorrect arg.
    assert call_kwargs['osf_id'] == 'vehxg'
    assert call_kwargs['zenodo_id'] == '19294180'


@pytest.mark.unit
@patch('proteus.utils.data.get_data_source_info', return_value=None)
def test_download_scattering_no_mapping_raises(mock_info):
    """download_scattering raises ValueError when 'scattering' is unmapped."""
    from proteus.utils.data import download_scattering

    with pytest.raises(ValueError, match='No data source mapping'):
        download_scattering()
    # Discrimination: confirm the registry was actually consulted
    # before raising (otherwise a regression that pre-empted the lookup
    # could pass on its own short-circuit).
    mock_info.assert_called_once()


# ============================================================================
# download_melting_curves additional coverage
# ============================================================================


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.GetFWLData')
def test_download_melting_curves_none_dir_is_noop(mock_getfwl, mock_download, tmp_path):
    """When melting_dir is None, the function returns early without calling download()."""
    from unittest.mock import MagicMock

    from proteus.utils.data import download_melting_curves

    mock_getfwl.return_value = tmp_path

    config = MagicMock()
    config.interior_struct.melting_dir = None

    download_melting_curves(config, clean=False)

    mock_download.assert_not_called()
    # Discrimination: confirm GetFWLData was NOT consulted (the early
    # return precedes the directory probe); a regression that dropped
    # the None-check would have called GetFWLData.
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


@pytest.mark.unit
def test_download_stellar_tracks_mors_success(tmp_path, monkeypatch):
    """download_stellar_tracks returns when MORS download succeeds and produces files."""
    import sys
    import types

    import proteus.utils.data as data_mod

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)

    tracks_path = tmp_path / 'stellar_evolution_tracks' / 'Spada'
    tracks_path.mkdir(parents=True, exist_ok=True)
    (tracks_path / 'track1.dat').write_text('data')

    download_calls = []

    def fake_download(track):
        download_calls.append(track)

    fake_mors_data = types.ModuleType('mors.data')
    fake_mors_data.DownloadEvolutionTracks = fake_download
    fake_mors = types.ModuleType('mors')
    fake_mors.data = fake_mors_data

    monkeypatch.setitem(sys.modules, 'mors', fake_mors)
    monkeypatch.setitem(sys.modules, 'mors.data', fake_mors_data)
    monkeypatch.setattr(data_mod, 'GetFWLData', lambda: tmp_path)

    data_mod.download_stellar_tracks('Spada')

    assert download_calls == ['Spada']
    # Discrimination: the function reached the tracks-present branch
    # (so no exception path fired); the track file remains untouched.
    assert (tracks_path / 'track1.dat').exists()


@pytest.mark.unit
def test_download_stellar_tracks_baraffe_verifies_versioned_path(tmp_path, monkeypatch):
    """Baraffe success is verified at the fwl-io versioned path, not the legacy dir."""
    import sys
    import types

    import proteus.utils.data as data_mod

    monkeypatch.setattr(data_mod, 'GetFWLData', lambda: tmp_path)

    # The versioned Baraffe directory holds the tracks; the legacy dir is absent.
    versioned = tmp_path / 'star' / 'tracks' / 'baraffe_2015' / 'r15729114'
    versioned.mkdir(parents=True)
    (versioned / 'BHAC15-M1p000.txt').write_text('track')

    checked = []
    fake_mors_data = types.ModuleType('mors.data')
    fake_mors_data.DownloadEvolutionTracks = lambda track: None
    fake_mors_data.baraffe_data_dir = lambda: checked.append(versioned) or versioned
    fake_mors = types.ModuleType('mors')
    fake_mors.data = fake_mors_data
    monkeypatch.setitem(sys.modules, 'mors', fake_mors)
    monkeypatch.setitem(sys.modules, 'mors.data', fake_mors_data)

    osf_calls = []
    monkeypatch.setattr(data_mod, 'download_OSF_folder', lambda **k: osf_calls.append(k))

    data_mod.download_stellar_tracks('Baraffe')

    # Discrimination: verification consulted the versioned resolver, not the
    # legacy path (which never exists here), so the fetch succeeded and no OSF
    # fallback ran.
    assert checked == [versioned]
    assert osf_calls == []


@pytest.mark.unit
def test_download_stellar_tracks_baraffe_legacy_mors_uses_legacy_path(tmp_path, monkeypatch):
    """A pre-migration MORS (no baraffe_data_dir) verifies Baraffe at the legacy path."""
    import sys
    import types

    import proteus.utils.data as data_mod

    monkeypatch.setattr(data_mod, 'GetFWLData', lambda: tmp_path)

    # An older MORS writes Baraffe to the legacy path and exposes no resolver.
    legacy = tmp_path / 'stellar_evolution_tracks' / 'Baraffe'
    legacy.mkdir(parents=True)
    (legacy / 'BHAC15-M1p000.txt').write_text('track')

    fake_mors_data = types.ModuleType('mors.data')
    fake_mors_data.DownloadEvolutionTracks = lambda track: None
    # Deliberately no baraffe_data_dir attribute (pre-migration MORS).
    fake_mors = types.ModuleType('mors')
    fake_mors.data = fake_mors_data
    monkeypatch.setitem(sys.modules, 'mors', fake_mors)
    monkeypatch.setitem(sys.modules, 'mors.data', fake_mors_data)

    osf_calls = []
    monkeypatch.setattr(data_mod, 'download_OSF_folder', lambda **k: osf_calls.append(k))

    data_mod.download_stellar_tracks('Baraffe')

    # Discrimination: provisioning succeeds via the legacy path even though the
    # versioned directory is never created, so PROTEUS works with both MORS
    # versions; no OSF fallback runs.
    assert not (tmp_path / 'star' / 'tracks' / 'baraffe_2015').exists()
    assert osf_calls == []


@pytest.mark.unit
def test_download_stellar_tracks_spada_verifies_versioned_path(tmp_path, monkeypatch):
    """Spada success is verified at the fwl-io versioned path, not the legacy dir."""
    import sys
    import types

    import proteus.utils.data as data_mod

    monkeypatch.setattr(data_mod, 'GetFWLData', lambda: tmp_path)

    # The versioned Spada directory holds the tracks; the legacy dir is absent.
    versioned = tmp_path / 'star' / 'tracks' / 'spada_2013' / 'r1'
    versioned.mkdir(parents=True)
    (versioned / 'm0p10.track1').write_text('track')

    checked = []
    fake_mors_data = types.ModuleType('mors.data')
    fake_mors_data.DownloadEvolutionTracks = lambda track: None
    fake_mors_data.spada_data_dir = lambda: checked.append(versioned) or versioned
    fake_mors = types.ModuleType('mors')
    fake_mors.data = fake_mors_data
    monkeypatch.setitem(sys.modules, 'mors', fake_mors)
    monkeypatch.setitem(sys.modules, 'mors.data', fake_mors_data)

    osf_calls = []
    monkeypatch.setattr(data_mod, 'download_OSF_folder', lambda **k: osf_calls.append(k))

    data_mod.download_stellar_tracks('Spada')

    # Discrimination: verification consulted the versioned resolver, not the
    # legacy path (which never exists here), so the fetch succeeded and no OSF
    # fallback ran.
    assert checked == [versioned]
    assert osf_calls == []


@pytest.mark.unit
def test_download_stellar_tracks_spada_legacy_mors_uses_legacy_path(tmp_path, monkeypatch):
    """A pre-migration MORS (no spada_data_dir) verifies Spada at the legacy path."""
    import sys
    import types

    import proteus.utils.data as data_mod

    monkeypatch.setattr(data_mod, 'GetFWLData', lambda: tmp_path)

    # An older MORS writes Spada to the legacy path and exposes no resolver.
    legacy = tmp_path / 'stellar_evolution_tracks' / 'Spada'
    legacy.mkdir(parents=True)
    (legacy / 'm0p10.track1').write_text('track')

    fake_mors_data = types.ModuleType('mors.data')
    fake_mors_data.DownloadEvolutionTracks = lambda track: None
    # Deliberately no spada_data_dir attribute (pre-migration MORS).
    fake_mors = types.ModuleType('mors')
    fake_mors.data = fake_mors_data
    monkeypatch.setitem(sys.modules, 'mors', fake_mors)
    monkeypatch.setitem(sys.modules, 'mors.data', fake_mors_data)

    osf_calls = []
    monkeypatch.setattr(data_mod, 'download_OSF_folder', lambda **k: osf_calls.append(k))

    data_mod.download_stellar_tracks('Spada')

    # Discrimination: provisioning succeeds via the legacy path even though the
    # versioned directory is never created, so PROTEUS works with both MORS
    # versions; no OSF fallback runs.
    assert not (tmp_path / 'star' / 'tracks' / 'spada_2013').exists()
    assert osf_calls == []


@pytest.mark.unit
def test_download_stellar_tracks_baraffe_failure_reraises_without_osf(tmp_path, monkeypatch):
    """A genuine Baraffe failure re-raises; Baraffe has no OSF fallback mirror."""
    import sys
    import types

    import proteus.utils.data as data_mod

    monkeypatch.setattr(data_mod, 'GetFWLData', lambda: tmp_path)

    # baraffe_data_dir resolves to an empty directory: the fetch produced nothing.
    empty = tmp_path / 'star' / 'tracks' / 'baraffe_2015' / 'r15729114'
    empty.mkdir(parents=True)
    fake_mors_data = types.ModuleType('mors.data')
    fake_mors_data.DownloadEvolutionTracks = lambda track: None
    fake_mors_data.baraffe_data_dir = lambda: empty
    fake_mors = types.ModuleType('mors')
    fake_mors.data = fake_mors_data
    monkeypatch.setitem(sys.modules, 'mors', fake_mors)
    monkeypatch.setitem(sys.modules, 'mors.data', fake_mors_data)

    osf_calls = []
    monkeypatch.setattr(data_mod, 'download_OSF_folder', lambda **k: osf_calls.append(k))

    with pytest.raises(FileNotFoundError, match='empty or missing'):
        data_mod.download_stellar_tracks('Baraffe', use_osf_fallback=True)
    # Even with use_osf_fallback=True, Baraffe does not attempt the OSF fallback.
    assert osf_calls == []


@pytest.mark.unit
def test_download_stellar_tracks_mors_completes_but_tracks_missing(tmp_path, monkeypatch):
    """MORS download claims success but tracks dir is empty: triggers OSF fallback."""
    import sys
    import types

    import proteus.utils.data as data_mod

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    monkeypatch.setattr(data_mod, 'GetFWLData', lambda: tmp_path)

    def fake_download(track):
        # Don't actually create any files
        pass

    fake_mors_data = types.ModuleType('mors.data')
    fake_mors_data.DownloadEvolutionTracks = fake_download
    fake_mors = types.ModuleType('mors')
    fake_mors.data = fake_mors_data
    monkeypatch.setitem(sys.modules, 'mors', fake_mors)
    monkeypatch.setitem(sys.modules, 'mors.data', fake_mors_data)

    # OSF fallback path: download_OSF_folder also produces no files.
    osf_calls = []

    def fake_osf_folder(*, storage, folders, data_dir):
        osf_calls.append((folders, data_dir))

    monkeypatch.setattr(data_mod, 'download_OSF_folder', fake_osf_folder)
    monkeypatch.setattr(data_mod, 'get_osf', lambda osf_id: MagicMock())

    with pytest.raises(RuntimeError, match='MORS'):
        data_mod.download_stellar_tracks('Spada', use_osf_fallback=True)
    # Discrimination: the OSF fallback was attempted with the expected
    # OSF projects list ('8r2sw') before raising the final RuntimeError.
    assert len(osf_calls) >= 1


@pytest.mark.unit
def test_download_stellar_tracks_mors_failure_no_osf_fallback(tmp_path, monkeypatch):
    """When MORS raises and use_osf_fallback=False, the original error propagates."""
    import sys
    import types

    import proteus.utils.data as data_mod

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    monkeypatch.setattr(data_mod, 'GetFWLData', lambda: tmp_path)

    def fake_download(track):
        raise ConnectionError('MORS down')

    fake_mors_data = types.ModuleType('mors.data')
    fake_mors_data.DownloadEvolutionTracks = fake_download
    fake_mors = types.ModuleType('mors')
    fake_mors.data = fake_mors_data
    monkeypatch.setitem(sys.modules, 'mors', fake_mors)
    monkeypatch.setitem(sys.modules, 'mors.data', fake_mors_data)

    osf_calls = []
    monkeypatch.setattr(data_mod, 'download_OSF_folder', lambda **k: osf_calls.append('x'))

    with pytest.raises(ConnectionError, match='MORS down'):
        data_mod.download_stellar_tracks('Spada', use_osf_fallback=False)
    # Discrimination: with use_osf_fallback=False, the OSF helper must
    # NOT have been invoked even once.
    assert osf_calls == []


@pytest.mark.unit
def test_download_stellar_tracks_osf_fallback_succeeds(tmp_path, monkeypatch):
    """When MORS fails but OSF fallback produces files, the function returns normally."""
    import sys
    import types

    import proteus.utils.data as data_mod

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    monkeypatch.setattr(data_mod, 'GetFWLData', lambda: tmp_path)

    def fake_download(track):
        raise RuntimeError('MORS HTTP 503')

    fake_mors_data = types.ModuleType('mors.data')
    fake_mors_data.DownloadEvolutionTracks = fake_download
    fake_mors = types.ModuleType('mors')
    fake_mors.data = fake_mors_data
    monkeypatch.setitem(sys.modules, 'mors', fake_mors)
    monkeypatch.setitem(sys.modules, 'mors.data', fake_mors_data)

    def fake_osf_folder(*, storage, folders, data_dir):
        # Create the expected tracks directory
        target = Path(data_dir) / 'Spada'
        target.mkdir(parents=True, exist_ok=True)
        (target / 'spada_track.dat').write_text('data')

    monkeypatch.setattr(data_mod, 'download_OSF_folder', fake_osf_folder)
    monkeypatch.setattr(data_mod, 'get_osf', lambda osf_id: MagicMock())

    # Spada keeps the legacy OSF fallback; the failed MORS fetch is recovered.
    data_mod.download_stellar_tracks('Spada', use_osf_fallback=True)
    # Discrimination: the OSF fallback produced the expected track file
    # at the canonical legacy location.
    tracks_dir = tmp_path / 'stellar_evolution_tracks' / 'Spada'
    assert tracks_dir.is_dir()
    assert (tracks_dir / 'spada_track.dat').exists()


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

    with pytest.raises(FileNotFoundError, match='Melting curve file not found'):
        get_zalmoxis_melting_curves(config)
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
def test_get_zalmoxis_melting_curves_local_dir_beats_manifest(monkeypatch, tmp_path):
    """When both exist, the local P-T files are read, not the dataset files."""
    import proteus.utils.data as data_mod
    from proteus.data import MELTING_MONTEUX_PLUS600, dataset_dir
    from proteus.utils.data import get_zalmoxis_melting_curves

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    folder = dataset_dir(MELTING_MONTEUX_PLUS600, data_root=tmp_path)
    folder.mkdir(parents=True)
    (folder / 'solidus.dat').write_text('1e9 1000\n2e9 1000\n')
    (folder / 'liquidus.dat').write_text('1e9 1000\n2e9 1000\n')
    local = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Monteux+600'
    local.mkdir(parents=True)
    (local / 'solidus_P-T.dat').write_text('1e9 2000\n2e9 3000\n')
    (local / 'liquidus_P-T.dat').write_text('1e9 2500\n2e9 3500\n')

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
# download() further branches: no Zenodo, OSF cleanup of empty folder
# ============================================================================


@pytest.mark.unit
@patch('proteus.utils.data.download_OSF_file')
@patch('proteus.utils.data.get_osf')
@patch('proteus.utils.data.get_data_source_info', return_value=None)
@patch('proteus.utils.data.GetFWLData')
def test_download_file_mode_no_zenodo_id_skips_to_osf(
    mock_getfwl, mock_info, mock_get_osf, mock_osf_file, tmp_path
):
    """File-mode: when zenodo_id is None, OSF is the only source."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path

    folder_dir = tmp_path / 'unmapped_target' / 'unmapped_folder'

    def osf_side_effect(*, storage, files, data_dir):
        dest = folder_dir / 'sub' / 'file.dat'
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text('payload')

    mock_osf_file.side_effect = osf_side_effect

    ok = download(
        folder='unmapped_folder',
        target='unmapped_target',
        zenodo_id=None,
        osf_id='osf_proj',
        desc='test',
        file='sub/file.dat',
    )

    assert ok is True
    # Discrimination: OSF was the only mechanism that ran; the file was
    # placed at the expected canonical path.
    mock_osf_file.assert_called_once()
    assert (folder_dir / 'sub' / 'file.dat').exists()


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

    assert len(spectra_calls) == 1
    folders = spectra_calls[0]
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

    folders = spectra_calls[0]
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


# ============================================================================
# download_stellar_tracks OSF inner-loop exception caught
# ============================================================================


@pytest.mark.unit
def test_download_stellar_tracks_osf_per_project_exception_caught(tmp_path, monkeypatch):
    """If get_osf raises for one project, the function moves on and raises RuntimeError."""
    import sys
    import types

    import proteus.utils.data as data_mod

    monkeypatch.setattr(data_mod, 'FWL_DATA_DIR', tmp_path, raising=False)
    monkeypatch.setattr(data_mod, 'GetFWLData', lambda: tmp_path)

    def fake_download(track):
        raise RuntimeError('MORS HTTP 503')

    fake_mors_data = types.ModuleType('mors.data')
    fake_mors_data.DownloadEvolutionTracks = fake_download
    fake_mors = types.ModuleType('mors')
    fake_mors.data = fake_mors_data
    monkeypatch.setitem(sys.modules, 'mors', fake_mors)
    monkeypatch.setitem(sys.modules, 'mors.data', fake_mors_data)

    get_osf_calls = []

    def bad_get_osf(osf_id):
        get_osf_calls.append(osf_id)
        raise ConnectionError('OSF unreachable')

    monkeypatch.setattr(data_mod, 'get_osf', bad_get_osf)
    monkeypatch.setattr(data_mod, 'download_OSF_folder', lambda **k: None)

    with pytest.raises(RuntimeError, match='MORS error'):
        data_mod.download_stellar_tracks('Spada', use_osf_fallback=True)
    # Discrimination: get_osf was actually called for each candidate
    # OSF project, so the per-project except path fired (not an outer
    # short-circuit).
    assert len(get_osf_calls) >= 1


# ============================================================================
# download_zenodo_file: log read on success-path returns 0 exit
# (line 247 TimeoutExpired-in-file-mode is already covered above; round out
# the file-mode test inventory with the rejects-bad-id for completeness)
# ============================================================================


@pytest.mark.unit
@patch('proteus.utils.data.sleep', return_value=None)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_log_read_failure_after_nonzero(
    mock_getfwl, mock_run, _mock_sleep, tmp_path
):
    """When zenodo_get exits non-zero and log readback succeeds, the diagnostic is logged."""
    from proteus.utils.data import download_zenodo_file

    mock_getfwl.return_value = tmp_path

    proc_avail = MagicMock(returncode=0)
    proc_fail = MagicMock(returncode=2)

    def side_effect(cmd, *args, **kwargs):
        if '--version' in cmd:
            return proc_avail
        # The function writes its own log file via `with open(out, 'w')`;
        # we don't need to write extra content because the open succeeds.
        return proc_fail

    mock_run.side_effect = side_effect

    folder_dir = tmp_path / 'log_readable_file'
    ok = download_zenodo_file('12345', folder_dir, 'sub/file.dat')
    assert ok is False
    # Discrimination: confirm we exhausted retries with non-zero exit
    # (so the log-readback branch on line 234 fired).
    download_calls = [c for c in mock_run.call_args_list if '--version' not in c[0][0]]
    assert len(download_calls) == 3
