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
    get_zenodo_record,
    md5,
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
    3. Checks if output directory exists.
    """
    mock_getfwl.return_value = str(tmp_path)

    # Mock subprocess success (exit code 0)
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_run.return_value = mock_proc

    folder_dir = tmp_path / 'downloaded_folder'

    # download_zenodo_folder checks existence of folder after run
    # so we must mock os.path.exists to simulate success
    with patch('proteus.utils.data.os.path.exists') as mock_exists:
        mock_exists.return_value = True
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
@patch('proteus.utils.data.get_zenodo_record')
def test_download_spectral_file_call(mock_get_record, mock_download):
    """
    Test spectral file download wrapper.

    Ensures that the convenience function correctly interprets the
    name/bands arguments to construct the target folder path and
    resolves the correct Zenodo ID.
    """
    mock_get_record.return_value = '99999'

    download_spectral_file('TestName', '123')

    mock_download.assert_called_once()
    args, kwargs = mock_download.call_args
    assert kwargs['folder'] == 'TestName/123'  # Folder structure
    assert kwargs['zenodo_id'] == '99999'  # Resolved ID
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
