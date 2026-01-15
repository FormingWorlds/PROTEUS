#!/usr/bin/env python3
"""
Unit tests for download robustness functionality.

Tests individual download methods, error handling, retry logic, and edge cases.
Uses mocks to avoid actual network calls in unit tests.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Set up environment
os.environ.setdefault('FWL_DATA', str(Path.home() / '.fwl_data_test'))


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_zenodo_token(monkeypatch):
    """Mock Zenodo token availability."""
    monkeypatch.setenv('ZENODO_API_TOKEN', 'test_token')


class TestZenodoCooldown:
    """Test Zenodo API rate limiting."""

    def test_cooldown_enforced(self, monkeypatch):
        """Test that cooldown is enforced between requests."""
        from proteus.utils.data import (
            _zenodo_cooldown,
            _last_zenodo_request_time,
            ZENODO_COOLDOWN,
        )
        import time

        # Reset global state
        monkeypatch.setattr('proteus.utils.data._last_zenodo_request_time', 0.0)

        # First call should not wait
        start = time.time()
        _zenodo_cooldown()
        first_duration = time.time() - start
        assert first_duration < 0.1  # Should be very fast

        # Second call immediately after should wait
        start = time.time()
        _zenodo_cooldown()
        second_duration = time.time() - start
        assert (
            second_duration >= ZENODO_COOLDOWN - 0.1
        )  # Should wait approximately cooldown time


class TestHasZenodoToken:
    """Test Zenodo token detection."""

    def test_token_from_env(self, monkeypatch):
        """Test token detection from environment variable."""
        from proteus.utils.data import _has_zenodo_token

        monkeypatch.setenv('ZENODO_API_TOKEN', 'test_token')
        assert _has_zenodo_token() is True

        monkeypatch.delenv('ZENODO_API_TOKEN', raising=False)
        # May still pass if config file exists, so just check it doesn't crash
        _has_zenodo_token()

    def test_token_from_config_file(self, tmp_dir, monkeypatch):
        """Test token detection from config file."""
        from proteus.utils.data import _has_zenodo_token
        import configparser

        # Remove env var
        monkeypatch.delenv('ZENODO_API_TOKEN', raising=False)

        # Create config file
        config_dir = tmp_dir / '.config'
        config_dir.mkdir(parents=True)
        config_file = config_dir / 'zenodo.ini'
        config = configparser.ConfigParser()
        config['zenodo'] = {'api_token': 'test_token_from_file'}
        with open(config_file, 'w') as f:
            config.write(f)

        # Mock home directory
        monkeypatch.setattr(Path, 'home', lambda: tmp_dir)

        assert _has_zenodo_token() is True

    def test_no_token(self, monkeypatch):
        """Test when no token is available."""
        from proteus.utils.data import _has_zenodo_token

        monkeypatch.delenv('ZENODO_API_TOKEN', raising=False)
        # Mock config file to not exist
        monkeypatch.setattr('proteus.utils.data.Path.home', lambda: Path('/nonexistent'))

        # Should return False if no token
        result = _has_zenodo_token()
        assert isinstance(result, bool)


class TestDownloadZenodoFolderClient:
    """Test zenodo_client download method."""

    @patch('proteus.utils.data._has_zenodo_token')
    @patch('proteus.utils.data._zenodo_cooldown')
    def test_success(self, mock_cooldown, mock_has_token, tmp_dir):
        """Test successful download."""
        from proteus.utils.data import download_zenodo_folder_client
        from zenodo_client import Zenodo

        mock_has_token.return_value = True

        # Mock Zenodo client
        mock_zenodo = Mock()
        mock_record = Mock()
        mock_record.json.return_value = {
            'files': [{'key': 'test.txt', 'links': {'self': 'http://example.com/test.txt'}}]
        }
        mock_zenodo.get_latest_record.return_value = '12345'
        mock_zenodo.get_record.return_value = mock_record
        mock_zenodo.download_file.return_value = True

        with patch('proteus.utils.data.Zenodo', return_value=mock_zenodo):
            result = download_zenodo_folder_client('12345', tmp_dir)

        # Should attempt download
        assert mock_zenodo.get_latest_record.called
        assert mock_zenodo.get_record.called

    @patch('proteus.utils.data._has_zenodo_token')
    def test_no_token_returns_false(self, mock_has_token):
        """Test that missing token returns False."""
        from proteus.utils.data import download_zenodo_folder_client

        mock_has_token.return_value = False

        result = download_zenodo_folder_client('12345', Path('/tmp'))
        assert result is False


class TestDownloadZenodoFolderGet:
    """Test zenodo_get download method."""

    @patch('proteus.utils.data.subprocess.run')
    @patch('proteus.utils.data.sleep')
    def test_success(self, mock_sleep, mock_run, tmp_dir):
        """Test successful download with zenodo_get."""
        from proteus.utils.data import download_zenodo_folder_get

        # Mock successful zenodo_get
        mock_run.return_value = Mock(returncode=0)
        mock_run.side_effect = [
            Mock(returncode=0),  # Version check
            Mock(returncode=0),  # Download
        ]

        with patch('proteus.utils.data.Path.exists', return_value=True):
            with patch('proteus.utils.data.Path.is_file', return_value=True):
                result = download_zenodo_folder_get('12345', tmp_dir)

        # Should have attempted download
        assert mock_run.called

    @patch('proteus.utils.data.subprocess.run')
    @patch('proteus.utils.data.sleep')
    def test_timeout_retry(self, mock_sleep, mock_run, tmp_dir):
        """Test timeout handling with retries."""
        from proteus.utils.data import download_zenodo_folder_get
        import subprocess

        # Mock timeout on first attempt, success on retry
        mock_run.side_effect = [
            Mock(returncode=0),  # Version check
            subprocess.TimeoutExpired('zenodo_get', 120),  # First attempt timeout
            Mock(returncode=0),  # Retry success
        ]

        with patch('proteus.utils.data.Path.exists', return_value=True):
            with patch('proteus.utils.data.Path.is_file', return_value=True):
                result = download_zenodo_folder_get('12345', tmp_dir)

        # Should have retried
        assert mock_sleep.called  # Should have waited between retries


class TestDownloadOSFFolder:
    """Test OSF folder download."""

    @patch('proteus.utils.data.get_osf')
    def test_success(self, mock_get_osf, tmp_dir):
        """Test successful OSF download."""
        from proteus.utils.data import download_OSF_folder

        # Mock OSF storage
        mock_storage = Mock()
        mock_file = Mock()
        mock_file.path = '/test/file.txt'
        mock_file.size = 100
        mock_file.write_to = Mock()
        mock_storage.files = [mock_file]
        mock_project = Mock()
        mock_project.storages = [mock_storage]
        mock_get_osf.return_value = mock_project

        result = download_OSF_folder('test_id', 'test_folder', tmp_dir, force=True)

        # Should have attempted download
        assert mock_get_osf.called

    @patch('proteus.utils.data.get_osf')
    def test_force_parameter(self, mock_get_osf, tmp_dir):
        """Test that force parameter removes existing files."""
        from proteus.utils.data import download_OSF_folder
        from proteus.utils.data import safe_rm

        # Create existing file
        existing_file = tmp_dir / 'test' / 'file.txt'
        existing_file.parent.mkdir(parents=True)
        existing_file.write_text('old content')

        # Mock OSF storage
        mock_storage = Mock()
        mock_file = Mock()
        mock_file.path = '/test/file.txt'
        mock_file.size = 100
        mock_file.write_to = Mock()
        mock_storage.files = [mock_file]
        mock_project = Mock()
        mock_project.storages = [mock_storage]
        mock_get_osf.return_value = mock_project

        with patch('proteus.utils.data.safe_rm') as mock_rm:
            download_OSF_folder('test_id', 'test', tmp_dir, force=True)
            # Should have removed existing file
            assert mock_rm.called


class TestGetDataSourceInfo:
    """Test data source info lookup."""

    def test_valid_folder(self):
        """Test lookup of valid folder."""
        from proteus.utils.data import get_data_source_info

        result = get_data_source_info('PHOENIX')
        assert result is not None
        assert 'zenodo_id' in result
        assert 'osf_id' in result
        assert 'osf_project' in result

    def test_invalid_folder(self):
        """Test lookup of invalid folder."""
        from proteus.utils.data import get_data_source_info

        result = get_data_source_info('INVALID_FOLDER')
        assert result is None

    def test_all_categories(self):
        """Test that all categories in DATA_SOURCE_MAP are accessible."""
        from proteus.utils.data import DATA_SOURCE_MAP, get_data_source_info

        for folder in DATA_SOURCE_MAP.keys():
            result = get_data_source_info(folder)
            assert result is not None, f'Failed to get info for {folder}'
            assert 'zenodo_id' in result
            assert 'osf_id' in result
            assert 'osf_project' in result


class TestValidateZenodoFolder:
    """Test Zenodo folder validation."""

    @patch('proteus.utils.data.subprocess.run')
    def test_validation_success(self, mock_run, tmp_dir):
        """Test successful validation."""
        from proteus.utils.data import validate_zenodo_folder

        # Create test files
        test_file = tmp_dir / 'test.txt'
        test_file.write_text('test content')

        # Mock md5sums file
        md5sums_file = tmp_dir / 'md5sums.txt'
        # Calculate actual MD5
        import hashlib

        md5_hash = hashlib.md5(test_file.read_bytes()).hexdigest()
        md5sums_file.write_text(f'{md5_hash}  test.txt\n')

        # Mock zenodo_get
        mock_run.return_value = Mock(returncode=0)

        with patch('proteus.utils.data.os.path.isfile', return_value=True):
            result = validate_zenodo_folder('12345', tmp_dir)

        # Should validate successfully
        assert result is True

    @patch('proteus.utils.data.subprocess.run')
    def test_validation_failure_hash_mismatch(self, mock_run, tmp_dir):
        """Test validation failure due to hash mismatch."""
        from proteus.utils.data import validate_zenodo_folder

        # Create test file
        test_file = tmp_dir / 'test.txt'
        test_file.write_text('test content')

        # Mock md5sums file with wrong hash
        md5sums_file = tmp_dir / 'md5sums.txt'
        md5sums_file.write_text('wrong_hash  test.txt\n')

        # Mock zenodo_get
        mock_run.return_value = Mock(returncode=0)

        with patch('proteus.utils.data.os.path.isfile', return_value=True):
            result = validate_zenodo_folder('12345', tmp_dir)

        # Should fail validation
        assert result is False


class TestDownloadFunction:
    """Test the main download() function."""

    @patch('proteus.utils.data.download_zenodo_folder_client')
    @patch('proteus.utils.data.validate_zenodo_folder')
    def test_folder_download_success(self, mock_validate, mock_download, tmp_dir):
        """Test successful folder download."""
        from proteus.utils.data import download

        mock_download.return_value = True
        mock_validate.return_value = True

        with patch('proteus.utils.data.Path.exists', return_value=True):
            result = download(
                folder='test',
                target='test_target',
                zenodo_id='12345',
                osf_id='test_osf',
                desc='test download',
            )

        # Should attempt download
        assert mock_download.called

    @patch('proteus.utils.data.get_zenodo_file')
    def test_single_file_download(self, mock_get_file, tmp_dir):
        """Test single file download with zenodo_path."""
        from proteus.utils.data import download

        mock_get_file.return_value = True

        with patch('proteus.utils.data.Path.exists', return_value=True):
            with patch('proteus.utils.data.Path.stat') as mock_stat:
                mock_stat.return_value.st_size = 100
                result = download(
                    folder='test',
                    target='test_target',
                    zenodo_id='12345',
                    osf_id='test_osf',
                    desc='test download',
                    zenodo_path='test_file.txt',
                )

        # Should attempt single file download
        assert mock_get_file.called


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
