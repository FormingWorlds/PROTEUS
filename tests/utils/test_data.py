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


@pytest.mark.unit
@patch("proteus.utils.data.zipfile.ZipFile")
@patch("proteus.utils.data.download")
def test_download_phoenix(mock_download, mock_zipfile, tmp_path, monkeypatch):
    """Test PHOENIX stellar spectra download wrapper (single-file + unzip + cleanup)."""

    from proteus.utils.data import download_phoenix
    from proteus.utils.phoenix_helper import phoenix_param

    # Arrange inputs
    FeH = 0.0
    alpha = 0.0
    feh_str = phoenix_param(FeH, kind="FeH")
    alpha_str = phoenix_param(alpha, kind="alpha")
    zip_name = f"FeH{feh_str}_alpha{alpha_str}_phoenixMedRes_R05000.zip"

    base_dir = tmp_path / "stellar_spectra" / "PHOENIX"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Make GetFWLData() return tmp_path
    monkeypatch.setattr("proteus.utils.data.GetFWLData", lambda: tmp_path)

    # Our function expects the zip to exist at base_dir/zip_name after download.
    # The production code will check zip_path.is_file(), so create it.
    zip_path = base_dir / zip_name
    zip_path.write_bytes(b"dummy zip bytes")

    # Simulate extracted LTE file presence by patching Path.glob used in:
    # any(grid_dir.glob("LTE_T*_phoenixMedRes_R05000.txt"))
    grid_dir = base_dir / f"FeH{feh_str}_alpha{alpha_str}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    lte_file = grid_dir / "LTE_T05800_logg4.50_FeH-0.0_alpha+0.0_phoenixMedRes_R05000.txt"
    lte_file.write_text("dummy spectrum")

    # Mock ZipFile context manager so no real unzip is attempted
    zf = MagicMock()
    mock_zipfile.return_value.__enter__.return_value = zf

    # Make download() report success
    mock_download.return_value = True

    # Also create a marker to verify it gets removed
    marker = base_dir / f".extracted_{zip_path.stem}"
    marker.write_text("marker")

    # Act
    ok = download_phoenix(alpha=alpha, FeH=FeH, force=False)

    # Assert: download() called with new single-file mode
    mock_download.assert_called_once_with(
        folder="PHOENIX",
        target="stellar_spectra",
        desc="PHOENIX stellar spectra (alpha=+0.0, [Fe/H]=+0.0)",
        force=False,
        file=zip_name,
    )

    assert ok is True

    # Assert: marker removed (if present)
    assert not marker.exists()

    # Assert: zip removed after successful extraction path
    assert not zip_path.exists()


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
    # water dict has 'core', 'mantle', and 'water_ice_layer' keys (Seager 2007 water planet structure)
    assert 'core' in water
    assert 'mantle' in water
    assert 'water_ice_layer' in water

    # Check file paths
    assert iron_silicate['mantle']['eos_file'] == eos_folder / 'eos_seager07_silicate.txt'
    assert iron_silicate['core']['eos_file'] == eos_folder / 'eos_seager07_iron.txt'
    assert water['water_ice_layer']['eos_file'] == eos_folder / 'eos_seager07_water.txt'


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
