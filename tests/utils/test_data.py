"""
Unit tests for proteus.utils.data module.

This module validates the data management utilities, ensuring reliable access to
external physics data (spectral files, lookup tables). It covers:
- Zenodo/OSF download logic (with mocking to prevent real network calls).
- File integrity verification (MD5 checksums).
- Configuration mapping for remote resources.

See also:
- docs/test_infrastructure.md
"""

from __future__ import annotations

from pathlib import Path
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


@pytest.mark.unit
def test_get_zenodo_record():
    """
    Test Zenodo record ID lookup from registry.

    Ensures that known configuration keys map to correct Zenodo repository IDs.
    """
    # Known mapping: Frostflow 16 band table
    assert get_zenodo_record('Frostflow/16') == '15799743'
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
    assert md5(str(f)) == '5eb63bbbe01eeed093cb22bb8f5acdc3'


@pytest.mark.unit
def test_check_needs_update_missing_dir(tmp_path):
    """
    check_needs_update returns True when the folder does not exist.

    Physical scenario: re-download is required when the target directory
    is missing (e.g. first run or cleaned cache).
    """
    missing = tmp_path / 'nonexistent'
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
    # No actual download occurs — the side_effect creates local files to
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


@pytest.mark.unit
def test_download_zenodo_file_rejects_bad_id(tmp_path):
    """download_zenodo_file rejects non-numeric zenodo IDs."""
    from proteus.utils.data import download_zenodo_file

    folder_dir = tmp_path / 'zenodo_folder'
    ok = download_zenodo_file('12ab', folder_dir, 'file.txt')
    assert ok is False


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
def test_download_zenodo_file_zenodo_get_missing(mock_run, tmp_path):
    """download_zenodo_file returns False when zenodo_get is not available."""
    from proteus.utils.data import download_zenodo_file

    mock_run.side_effect = FileNotFoundError('zenodo_get not found')

    folder_dir = tmp_path / 'zenodo_folder'
    ok = download_zenodo_file('12345', folder_dir, 'file.txt')
    assert ok is False


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


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_zero_exit_but_file_missing(mock_getfwl, mock_run, tmp_path):
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
@patch('proteus.utils.data.safe_rm')
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_file_cleanup_branches(mock_getfwl, mock_run, mock_safe_rm, tmp_path):
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
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.get_data_source_info')
def test_download_spectral_file_call(mock_get_info, mock_download):
    """
    Test spectral file download wrapper.

    Ensures that the convenience function correctly interprets the
    name/bands arguments to construct the target folder path and
    resolves the correct Zenodo and OSF IDs from mapping.
    """
    # Mock mapping lookup
    mock_get_info.return_value = {
        'zenodo_id': '99999',
        'osf_project': 'test_osf',
        'osf_id': 'test_osf',
    }

    download_spectral_file('TestName', '123')

    mock_download.assert_called_once()
    args, kwargs = mock_download.call_args
    assert kwargs['folder'] == 'TestName/123'  # Folder structure
    assert kwargs['zenodo_id'] == '99999'  # Resolved ID from mapping
    assert kwargs['osf_id'] == 'test_osf'  # OSF ID from mapping
    assert kwargs['target'] == 'spectral_files'  # Target subdir


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


@pytest.mark.unit
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.download_zenodo_file')
def test_download_file_mode_skips_existing_file(mock_zdl, mock_getfwl, tmp_path):
    """If file exists and non-empty and force=False, download() returns True without calling Zenodo/OSF."""
    from proteus.utils.data import download

    mock_getfwl.return_value = tmp_path

    folder = 'SomeFolder'
    target = 'targetdir'
    file_rel = 'subdir/file.txt'

    dest = tmp_path / target / folder / file_rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text('already here')

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


@pytest.mark.unit
def test_get_data_source_info():
    """Test unified data source mapping lookup."""
    # Test known mapping
    info = get_data_source_info('Zeng2019')
    assert info is not None
    assert info['zenodo_id'] == '15727899'
    assert info['osf_project'] == 'xge8t'

    # Test unknown mapping
    info = get_data_source_info('UnknownFolder')
    assert info is None


@pytest.mark.unit
def test_get_osf_project():
    """Test OSF project ID lookup."""
    assert get_osf_project('Zeng2019') == 'xge8t'
    assert get_osf_project('Named') == '8r2sw'
    assert get_osf_project('UnknownFolder') is None


@pytest.mark.unit
def test_get_zenodo_from_osf():
    """Test reverse lookup: OSF project -> Zenodo IDs."""
    zenodo_ids = get_zenodo_from_osf('phsxf')
    assert len(zenodo_ids) > 0
    assert '17417017' in zenodo_ids  # Should include ARAGOG data

    # Test unknown OSF project
    zenodo_ids = get_zenodo_from_osf('unknown')
    assert len(zenodo_ids) == 0


@pytest.mark.unit
def test_get_osf_from_zenodo():
    """Test reverse lookup: Zenodo ID -> OSF project."""
    assert get_osf_from_zenodo('15727899') == 'xge8t'  # Zeng2019
    assert get_osf_from_zenodo('15721440') == '8r2sw'  # Named
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
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_folder_timeout(mock_getfwl, mock_run, tmp_path):
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

    folder_dir = tmp_path / 'target' / 'Zeng2019'
    folder_dir.mkdir(parents=True, exist_ok=True)
    (folder_dir / 'test_file.txt').write_text('test')

    with patch('proteus.utils.data.Path.rglob') as mock_rglob:
        mock_file = MagicMock()
        mock_file.is_file.return_value = True
        mock_rglob.return_value = [mock_file]

        # Call download without explicit IDs - should use mapping
        result = download(
            folder='Zeng2019',
            target='target',
            desc='test data',
            # No osf_id or zenodo_id provided - should use mapping
        )

    # Should have looked up IDs from mapping and used them
    mock_download_zenodo.assert_called_once()
    # Should have used mapped Zenodo ID (check kwargs since it's called with keyword args)
    call_kwargs = mock_download_zenodo.call_args.kwargs
    assert call_kwargs['zenodo_id'] == '15727899'  # Zenodo ID from mapping
    assert result is True


@pytest.mark.unit
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_download_zenodo_folder_error_diagnostics(mock_getfwl, mock_run, tmp_path):
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


@pytest.mark.unit
@patch('proteus.utils.data.zipfile.ZipFile')
@patch('proteus.utils.data.download')
def test_download_phoenix(mock_download, mock_zipfile, tmp_path, monkeypatch):
    """Test PHOENIX stellar spectra download wrapper (single-file + unzip + cleanup)."""

    from proteus.utils.data import download_phoenix
    from proteus.utils.phoenix_helper import phoenix_param

    # Arrange inputs
    FeH = 0.0
    alpha = 0.0
    feh_str = phoenix_param(FeH, kind='FeH')
    alpha_str = phoenix_param(alpha, kind='alpha')
    zip_name = f'FeH{feh_str}_alpha{alpha_str}_phoenixMedRes_R05000.zip'

    base_dir = tmp_path / 'stellar_spectra' / 'PHOENIX'
    base_dir.mkdir(parents=True, exist_ok=True)

    # Make GetFWLData() return tmp_path
    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)

    # Our function expects the zip to exist at base_dir/zip_name after download.
    # The production code will check zip_path.is_file(), so create it.
    zip_path = base_dir / zip_name
    zip_path.write_bytes(b'dummy zip bytes')

    # Simulate extracted LTE file presence by patching Path.glob used in:
    # any(grid_dir.glob("LTE_T*_phoenixMedRes_R05000.txt"))
    grid_dir = base_dir / f'FeH{feh_str}_alpha{alpha_str}'
    grid_dir.mkdir(parents=True, exist_ok=True)
    lte_file = grid_dir / 'LTE_T05800_logg4.50_FeH-0.0_alpha+0.0_phoenixMedRes_R05000.txt'
    lte_file.write_text('dummy spectrum')

    # Mock ZipFile context manager so no real unzip is attempted
    zf = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = zf

    # Make download() report success
    mock_download.return_value = True

    # Also create a marker to verify it gets removed
    marker = base_dir / f'.extracted_{zip_path.stem}'
    marker.write_text('marker')

    # Act
    ok = download_phoenix(alpha=alpha, FeH=FeH, force=False)

    # Assert: download() called with new single-file mode
    mock_download.assert_called_once_with(
        folder='PHOENIX',
        target='stellar_spectra',
        desc='PHOENIX stellar spectra (alpha=+0.0, [Fe/H]=+0.0)',
        force=False,
        file=zip_name,
    )

    assert ok is True

    assert not zip_path.exists()


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.zipfile.ZipFile')
def test_download_phoenix_unzips_and_cleans_up(
    mock_zipfile, mock_download, tmp_path, monkeypatch
):
    """Covers the unzip path: ZipFile/extractall called, LTE file appears, zip removed."""

    from proteus.utils.data import download_phoenix
    from proteus.utils.phoenix_helper import phoenix_param

    # Arrange inputs
    FeH = 0.0
    alpha = 0.0
    feh_str = phoenix_param(FeH, kind='FeH')
    alpha_str = phoenix_param(alpha, kind='alpha')
    zip_name = f'FeH{feh_str}_alpha{alpha_str}_phoenixMedRes_R05000.zip'

    # Make GetFWLData() return tmp_path
    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)

    base_dir = tmp_path / 'stellar_spectra' / 'PHOENIX'
    base_dir.mkdir(parents=True, exist_ok=True)

    # Production expects zip at base_dir/zip_name after download()
    zip_path = base_dir / zip_name
    zip_path.write_bytes(b'dummy zip bytes')

    # grid_dir where extraction happens
    grid_dir = base_dir / f'FeH{feh_str}_alpha{alpha_str}'

    # IMPORTANT: do NOT pre-create LTE file here; we want the unzip path.

    # Mock download() report success
    mock_download.return_value = True

    # Mock ZipFile so no real unzip occurs; simulate extraction by creating LTE file
    zf = MagicMock()

    def extractall_side_effect(dest):
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'LTE_T02300_logg0.00_FeH-0.0_alpha+0.0_phoenixMedRes_R05000.txt').write_text(
            'dummy spectrum'
        )

    zf.extractall.side_effect = extractall_side_effect
    mock_zipfile.return_value.__enter__.return_value = zf

    # Act
    ok = download_phoenix(alpha=alpha, FeH=FeH, force=True)

    # Assert
    assert ok is True

    # unzip was actually attempted
    mock_zipfile.assert_called_once_with(zip_path, 'r')
    zf.extractall.assert_called_once_with(grid_dir)

    # LTE file now exists after "extraction"
    assert any(grid_dir.glob('LTE_T*_phoenixMedRes_R05000.txt'))

    # zip removed after successful unpack
    assert not zip_path.exists()


@pytest.mark.unit
@patch('proteus.utils.data.safe_rm')
@patch('proteus.utils.data.zipfile.ZipFile')
@patch('proteus.utils.data.download')
def test_download_phoenix_force_removes_existing_grid_dir(
    mock_download, mock_zipfile, mock_safe_rm, tmp_path, monkeypatch
):
    from proteus.utils.data import download_phoenix
    from proteus.utils.phoenix_helper import phoenix_param

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    mock_download.return_value = True

    FeH = 0.0
    alpha = 0.0
    feh_str = phoenix_param(FeH, kind='FeH')
    alpha_str = phoenix_param(alpha, kind='alpha')
    zip_name = f'FeH{feh_str}_alpha{alpha_str}_phoenixMedRes_R05000.zip'

    base_dir = tmp_path / 'stellar_spectra' / 'PHOENIX'
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / zip_name).write_bytes(b'zip')

    grid_dir = base_dir / f'FeH{feh_str}_alpha{alpha_str}'
    grid_dir.mkdir(parents=True, exist_ok=True)

    # Make extraction succeed by pre-creating LTE file after "extractall"
    def extractall_side_effect(_dest):
        dest = Path(_dest)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'LTE_T02300_logg0.00_FeH-0.0_alpha+0.0_phoenixMedRes_R05000.txt').write_text(
            'ok'
        )

    zf = MagicMock()
    zf.extractall.side_effect = extractall_side_effect
    mock_zipfile.return_value.__enter__.return_value = zf

    ok = download_phoenix(alpha=alpha, FeH=FeH, force=True)
    assert ok is True
    mock_safe_rm.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.download')
def test_download_phoenix_returns_false_if_download_fails(mock_download, tmp_path, monkeypatch):
    from proteus.utils.data import download_phoenix

    monkeypatch.setattr('proteus.utils.data.GetFWLData', lambda: tmp_path)
    mock_download.return_value = False

    ok = download_phoenix(alpha=0.0, FeH=0.0, force=False)
    assert ok is False


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.get_data_source_info')
def test_download_muscles_default_download_all(mock_get_info, mock_download):
    """If stars is None, download_muscles should download the whole MUSCLES catalogue (folder mode)."""
    from proteus.utils.data import download_muscles

    mock_get_info.return_value = {
        'zenodo_id': 'ZEN',
        'osf_project': 'OSFPROJ',
        'osf_id': 'OSFPROJ',
    }
    mock_download.return_value = True

    ok = download_muscles(stars=None, force=False)

    assert ok is True
    mock_get_info.assert_called_once_with('MUSCLES')
    mock_download.assert_called_once_with(
        folder='MUSCLES',
        target='stellar_spectra',
        osf_id='OSFPROJ',
        zenodo_id='ZEN',
        desc='MUSCLES stellar spectra catalogue',
        force=False,
    )


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.get_data_source_info')
def test_download_muscles_single_star(mock_get_info, mock_download):
    """If stars is a string, download_muscles should call download() once in single-file mode."""
    from proteus.utils.data import download_muscles

    mock_get_info.return_value = {
        'zenodo_id': 'ZEN',
        'osf_project': 'OSFPROJ',
        'osf_id': 'OSFPROJ',
    }
    mock_download.return_value = True

    ok = download_muscles(stars='trappist-1', force=True)

    assert ok is True
    mock_get_info.assert_called_once_with('MUSCLES')
    mock_download.assert_called_once_with(
        folder='MUSCLES',
        target='stellar_spectra',
        osf_id='OSFPROJ',
        zenodo_id='ZEN',
        desc='MUSCLES stellar spectrum (trappist-1)',
        force=True,
        file='trappist-1.txt',
    )


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.get_data_source_info')
def test_download_muscles_multiple_stars(mock_get_info, mock_download):
    """If stars is a list, download_muscles should call download() once per star and return AND of results."""
    from proteus.utils.data import download_muscles

    mock_get_info.return_value = {
        'zenodo_id': 'ZEN',
        'osf_project': 'OSFPROJ',
        'osf_id': 'OSFPROJ',
    }

    # First star succeeds, second fails -> overall False
    mock_download.side_effect = [True, False]

    ok = download_muscles(stars=['starA', 'starB'], force=False)

    assert ok is False
    mock_get_info.assert_called_once_with('MUSCLES')
    assert mock_download.call_count == 2

    # Check both calls precisely (order matters)
    expected_calls = [
        call(
            folder='MUSCLES',
            target='stellar_spectra',
            osf_id='OSFPROJ',
            zenodo_id='ZEN',
            desc='MUSCLES stellar spectrum (starA)',
            force=False,
            file='starA.txt',
        ),
        call(
            folder='MUSCLES',
            target='stellar_spectra',
            osf_id='OSFPROJ',
            zenodo_id='ZEN',
            desc='MUSCLES stellar spectrum (starB)',
            force=False,
            file='starB.txt',
        ),
    ]
    assert mock_download.call_args_list == expected_calls


@pytest.mark.unit
@patch('proteus.utils.data.get_data_source_info')
def test_download_muscles_no_mapping_raises(mock_get_info):
    """download_muscles should raise if MUSCLES is not in the mapping."""
    from proteus.utils.data import download_muscles

    mock_get_info.return_value = None

    with pytest.raises(ValueError):
        download_muscles(stars=None)


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.safe_rm')
def test_download_interior_lookuptables(mock_rm, mock_getfwl, mock_download, tmp_path):
    """Test interior lookup tables download."""
    from proteus.utils.data import ARAGOG_BASIC, download_interior_lookuptables

    mock_getfwl.return_value = tmp_path

    download_interior_lookuptables(clean=False)

    # Should download each directory in ARAGOG_BASIC
    assert mock_download.call_count == len(ARAGOG_BASIC)
    for dir_name in ARAGOG_BASIC:
        # Check that download was called with correct folder
        calls = [
            call for call in mock_download.call_args_list if call.kwargs['folder'] == dir_name
        ]
        assert len(calls) == 1


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.safe_rm')
def test_download_interior_lookuptables_clean(mock_rm, mock_getfwl, mock_download, tmp_path):
    """Test interior lookup tables download with clean=True."""
    from proteus.utils.data import ARAGOG_BASIC, download_interior_lookuptables

    mock_getfwl.return_value = tmp_path

    download_interior_lookuptables(clean=True)

    # Should have called safe_rm for each directory
    assert mock_rm.call_count == len(ARAGOG_BASIC)


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.safe_rm')
def test_download_melting_curves(mock_rm, mock_getfwl, mock_download, tmp_path):
    """Test melting curves download."""
    from unittest.mock import MagicMock

    from proteus.config import Config
    from proteus.utils.data import download_melting_curves

    mock_getfwl.return_value = tmp_path

    # Create mock config with melting_dir
    mock_config = MagicMock(spec=Config)
    mock_config.interior.melting_dir = 'Wolf_Bower+2018'

    download_melting_curves(mock_config, clean=False)

    mock_download.assert_called_once()
    call_kwargs = mock_download.call_args.kwargs
    assert call_kwargs['folder'] == 'Melting_curves/Wolf_Bower+2018'
    assert call_kwargs['desc'] == 'Melting curve data: Melting_curves/Wolf_Bower+2018'


@pytest.mark.unit
@patch('proteus.utils.data.download')
def test_download_stellar_spectra_default(mock_download):
    """Test stellar spectra download with default folders."""
    from proteus.utils.data import download_stellar_spectra

    download_stellar_spectra()

    # Should download Named, solar, and MUSCLES
    assert mock_download.call_count == 3
    folders = [call.kwargs['folder'] for call in mock_download.call_args_list]
    assert 'Named' in folders
    assert 'solar' in folders
    assert 'MUSCLES' in folders


@pytest.mark.unit
@patch('proteus.utils.data.download')
def test_download_stellar_spectra_custom(mock_download):
    """Test stellar spectra download with custom folders."""
    from proteus.utils.data import download_stellar_spectra

    download_stellar_spectra(folders=('Named', 'PHOENIX'))

    assert mock_download.call_count == 2
    folders = [call.kwargs['folder'] for call in mock_download.call_args_list]
    assert 'Named' in folders
    assert 'PHOENIX' in folders


@pytest.mark.unit
@patch('proteus.utils.data.download')
def test_download_exoplanet_data(mock_download):
    """Test exoplanet data download."""
    from proteus.utils.data import download_exoplanet_data

    download_exoplanet_data()

    mock_download.assert_called_once()
    call_kwargs = mock_download.call_args.kwargs
    assert call_kwargs['folder'] == 'Exoplanets'
    assert call_kwargs['target'] == 'planet_reference'
    assert call_kwargs['desc'] == 'exoplanet data'


@pytest.mark.unit
@patch('proteus.utils.data.download')
def test_download_massradius_data(mock_download):
    """Test mass-radius data download."""
    from proteus.utils.data import download_massradius_data

    download_massradius_data()

    mock_download.assert_called_once()
    call_kwargs = mock_download.call_args.kwargs
    assert call_kwargs['folder'] == 'Zeng2019'
    assert call_kwargs['target'] == 'mass_radius'
    assert call_kwargs['desc'] == 'mass radius data'


@pytest.mark.unit
@patch('proteus.utils.data.download')
def test_download_surface_albedos(mock_download):
    """Test surface albedos download."""
    from proteus.utils.data import download_surface_albedos

    download_surface_albedos()

    mock_download.assert_called_once()
    call_kwargs = mock_download.call_args.kwargs
    assert call_kwargs['folder'] == 'Hammond24'
    assert call_kwargs['target'] == 'surface_albedos'
    assert call_kwargs['desc'] == 'surface reflectance data'


@pytest.mark.unit
@patch('proteus.utils.data.FWL_DATA_DIR')
def test_GetFWLData(mock_fwl_data_dir, tmp_path):
    """Test FWL data directory getter."""
    from proteus.utils.data import GetFWLData

    # Patch FWL_DATA_DIR to return tmp_path
    with patch('proteus.utils.data.FWL_DATA_DIR', tmp_path):
        result = GetFWLData()
        assert result == tmp_path.absolute()


@pytest.mark.unit
@patch('proteus.utils.data.download_Seager_EOS')
def test_get_Seager_EOS_exists(mock_download, tmp_path):
    """Test get_Seager_EOS when EOS folder already exists."""
    from proteus.utils.data import get_Seager_EOS

    # Create EOS folder
    eos_folder = tmp_path / 'EOS_material_properties' / 'EOS_Seager2007'
    eos_folder.mkdir(parents=True, exist_ok=True)

    # Create required files
    (eos_folder / 'eos_seager07_silicate.txt').write_text('test')
    (eos_folder / 'eos_seager07_iron.txt').write_text('test')
    (eos_folder / 'eos_seager07_water.txt').write_text('test')

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
    from proteus.utils.data import get_Seager_EOS

    # EOS folder doesn't exist (not created, so download will be triggered)

    # Patch FWL_DATA_DIR and call function
    with patch('proteus.utils.data.FWL_DATA_DIR', tmp_path):
        get_Seager_EOS()

    # Should call download
    mock_download.assert_called_once()


@pytest.mark.unit
@patch('proteus.utils.data.download')
def test_download_Seager_EOS(mock_download):
    """Test Seager EOS download."""
    from proteus.utils.data import download_Seager_EOS

    download_Seager_EOS()

    mock_download.assert_called_once()
    call_kwargs = mock_download.call_args.kwargs
    assert call_kwargs['folder'] == 'EOS_Seager2007'
    assert call_kwargs['target'] == 'EOS_material_properties'
    assert call_kwargs['desc'] == 'EOS Seager2007 material files'


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


@pytest.mark.unit
@patch('proteus.utils.data.get_data_source_info')
def test_download_stellar_spectra_no_mapping(mock_get_info):
    """Test stellar spectra download raises error when no mapping found."""
    from proteus.utils.data import download_stellar_spectra

    mock_get_info.return_value = None

    with pytest.raises(ValueError, match='No data source mapping found'):
        download_stellar_spectra(folders=('UnknownFolder',))


@pytest.mark.unit
@patch('proteus.utils.data.get_data_source_info')
def test_download_melting_curves_no_mapping(mock_get_info):
    """Test melting curves download raises error when no mapping found."""
    from unittest.mock import MagicMock

    from proteus.config import Config
    from proteus.utils.data import download_melting_curves

    mock_config = MagicMock(spec=Config)
    mock_config.interior.melting_dir = 'UnknownCurve'
    mock_get_info.return_value = None

    with pytest.raises(ValueError, match='No data source mapping found'):
        download_melting_curves(mock_config)


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.safe_rm')
def test_download_melting_curves_canonical_copy(mock_rm, mock_getfwl, mock_download, tmp_path):
    """Test canonical P-T copy from legacy Zenodo names (solidus.dat → solidus_P-T.dat)."""
    from proteus.utils.data import download_melting_curves

    mock_getfwl.return_value = tmp_path

    # Create legacy files that Zenodo would download
    mc_dir = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Wolf_Bower+2018'
    mc_dir.mkdir(parents=True)
    (mc_dir / 'solidus.dat').write_text('solidus data')
    (mc_dir / 'liquidus.dat').write_text('liquidus data')

    mock_config = MagicMock()
    mock_config.interior.melting_dir = 'Wolf_Bower+2018'

    download_melting_curves(mock_config, clean=False)

    # Canonical copies should be created
    assert (mc_dir / 'solidus_P-T.dat').exists()
    assert (mc_dir / 'liquidus_P-T.dat').exists()
    assert (mc_dir / 'solidus_P-T.dat').read_text() == 'solidus data'


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.GetFWLData')
@patch('proteus.utils.data.safe_rm')
def test_download_melting_curves_canonical_skip_existing(
    mock_rm, mock_getfwl, mock_download, tmp_path
):
    """Canonical copy is skipped when P-T file already exists."""
    from proteus.utils.data import download_melting_curves

    mock_getfwl.return_value = tmp_path

    mc_dir = tmp_path / 'interior_lookup_tables' / 'Melting_curves' / 'Wolf_Bower+2018'
    mc_dir.mkdir(parents=True)
    (mc_dir / 'solidus.dat').write_text('old solidus')
    (mc_dir / 'solidus_P-T.dat').write_text('existing canonical')
    (mc_dir / 'liquidus.dat').write_text('old liquidus')

    mock_config = MagicMock()
    mock_config.interior.melting_dir = 'Wolf_Bower+2018'

    download_melting_curves(mock_config, clean=False)

    # Should NOT overwrite existing canonical file
    assert (mc_dir / 'solidus_P-T.dat').read_text() == 'existing canonical'


@pytest.mark.unit
@patch('proteus.utils.data.get_data_source_info')
def test_download_exoplanet_data_no_mapping(mock_get_info):
    """Test exoplanet data download raises error when no mapping found."""
    from proteus.utils.data import download_exoplanet_data

    mock_get_info.return_value = None

    with pytest.raises(ValueError, match='No data source mapping found'):
        download_exoplanet_data()


@pytest.mark.unit
@patch('proteus.utils.data.get_data_source_info')
def test_download_surface_albedos_no_mapping(mock_get_info):
    """Test surface albedos download raises error when no mapping found."""
    from proteus.utils.data import download_surface_albedos

    mock_get_info.return_value = None

    with pytest.raises(ValueError, match='No data source mapping found'):
        download_surface_albedos()


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


@pytest.mark.unit
@patch('proteus.utils.data.get_data_source_info')
def test_download_massradius_data_no_mapping(mock_get_info):
    """Test mass-radius data download raises error when no mapping found."""
    from proteus.utils.data import download_massradius_data

    mock_get_info.return_value = None

    with pytest.raises(ValueError, match='No data source mapping found'):
        download_massradius_data()


@pytest.mark.unit
@patch('proteus.utils.data.get_data_source_info')
def test_download_Seager_EOS_no_mapping(mock_get_info):
    """Test Seager EOS download raises error when no mapping found."""
    from proteus.utils.data import download_Seager_EOS

    mock_get_info.return_value = None

    with pytest.raises(ValueError, match='No data source mapping found'):
        download_Seager_EOS()


@pytest.mark.unit
@pytest.mark.skip(
    reason='Complex path matching logic - exception handling verified in integration tests'
)
@patch('proteus.utils.data.get_osf')
def test_download_OSF_folder_exception_handling(mock_get_osf, tmp_path):
    """Test OSF folder download handles exceptions gracefully.

    Note: Skipped due to complex path matching logic in download_OSF_folder.
    Exception handling is verified in integration tests with real OSF downloads.
    """
    pass


# Note: download_zenodo_folder_client function doesn't exist in current codebase
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


@pytest.mark.unit
@pytest.mark.skip(
    reason='Complex file system mocking required - hash validation verified in integration tests'
)
@patch('proteus.utils.data.sp.run')
@patch('proteus.utils.data.GetFWLData')
def test_validate_zenodo_folder_hash_mismatch(mock_getfwl, mock_run, tmp_path):
    """Test validation fails when file hash doesn't match.

    Note: Skipped due to complex file system mocking required for os.path operations.
    Hash validation is verified in integration tests with real Zenodo downloads.
    """
    pass


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


# ============================================================================
# test _get_sufficient — Zalmoxis EOS branches
# ============================================================================


@pytest.mark.unit
@patch('proteus.utils.data.download_eos_dynamic')
@patch('proteus.utils.data.download_eos_static')
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
    mock_static,
    mock_dyn,
):
    """_get_sufficient downloads Zalmoxis static + WolfBower2018 dynamic EOS."""
    from unittest.mock import MagicMock

    from proteus.utils.data import _get_sufficient

    config = MagicMock()
    config.interior.module = 'spider'
    config.interior.eos_dir = 'WolfBower2018_MgSiO3'
    config.struct.module = 'zalmoxis'
    config.struct.zalmoxis.mantle_eos = 'WolfBower2018_MgSiO3'

    _get_sufficient(config, clean=False)

    mock_static.assert_called_once()
    # Dynamic called twice: once for interior.eos_dir, once for Zalmoxis
    assert mock_dyn.call_count == 2


@pytest.mark.unit
@patch('proteus.utils.data.download_eos_dynamic')
@patch('proteus.utils.data.download_eos_static')
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
    mock_static,
    mock_dyn,
):
    """_get_sufficient skips dynamic EOS for Seager-only Zalmoxis config."""
    from unittest.mock import MagicMock

    from proteus.utils.data import _get_sufficient

    config = MagicMock()
    config.interior.module = 'dummy'  # no spider/aragog → skip first dynamic
    config.struct.module = 'zalmoxis'
    config.struct.zalmoxis.mantle_eos = 'Seager2007:silicate'

    _get_sufficient(config, clean=False)

    mock_static.assert_called_once()
    mock_dyn.assert_not_called()


# ============================================================================
# test download_eos_dynamic / download_eos_static
# ============================================================================


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.get_data_source_info')
def test_download_eos_dynamic_calls_download(mock_info, mock_dl):
    """download_eos_dynamic calls download with the legacy folder path."""
    from proteus.utils.data import download_eos_dynamic

    mock_info.return_value = {'osf_project': 'abc123', 'zenodo_id': '999'}
    download_eos_dynamic('WolfBower2018_MgSiO3')

    mock_dl.assert_called_once()
    call_kwargs = mock_dl.call_args
    assert 'interior_lookup_tables' in str(call_kwargs)


@pytest.mark.unit
@patch('proteus.utils.data.download')
@patch('proteus.utils.data.get_data_source_info')
def test_download_eos_dynamic_no_mapping(mock_info, mock_dl):
    """download_eos_dynamic returns early when no source mapping found."""
    from proteus.utils.data import download_eos_dynamic

    mock_info.return_value = None
    download_eos_dynamic('UnknownEOS')

    mock_dl.assert_not_called()


@pytest.mark.unit
@patch('proteus.utils.data.download_Seager_EOS')
def test_download_eos_static_delegates(mock_seager):
    """download_eos_static delegates to download_Seager_EOS."""
    from proteus.utils.data import download_eos_static

    download_eos_static()
    mock_seager.assert_called_once()
