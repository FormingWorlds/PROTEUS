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

from unittest.mock import MagicMock, patch

import pytest

from proteus.utils.data import (
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
