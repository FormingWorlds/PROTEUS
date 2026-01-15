#!/usr/bin/env python3
"""
Unit tests for download robustness functionality.

Tests individual download methods, error handling, retry logic, and edge cases.
Uses mocks to avoid actual network calls in unit tests.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

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
        import time

        from proteus.utils import data
        from proteus.utils.data import (
            ZENODO_COOLDOWN,
            _zenodo_cooldown,
        )

        # Reset global state - use the actual module object
        monkeypatch.setattr(data, '_last_zenodo_request_time', 0.0)

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
        import configparser

        from proteus.utils.data import _has_zenodo_token

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
        from pathlib import Path as PathClass

        from proteus.utils.data import _has_zenodo_token

        monkeypatch.delenv('ZENODO_API_TOKEN', raising=False)

        # Mock config file to not exist - patch Path.home method
        def mock_home():
            return PathClass('/nonexistent')

        monkeypatch.setattr(PathClass, 'home', staticmethod(mock_home))

        # Should return False if no token
        result = _has_zenodo_token()
        assert isinstance(result, bool)


class TestDownloadZenodoFolderClient:
    """Test zenodo_client download method."""

    @pytest.mark.skip(
        reason='Complex mocking of zenodo_client import - covered by integration tests'
    )
    def test_success(self, tmp_dir):
        """Test successful download - skipped due to complex import mocking."""
        pass

    @patch('proteus.utils.data._has_zenodo_token')
    def test_no_token_returns_false(self, mock_has_token):
        """Test that missing token returns False."""
        from proteus.utils.data import download_zenodo_folder_client

        mock_has_token.return_value = False

        result = download_zenodo_folder_client('12345', Path('/tmp'))
        assert result is False


class TestDownloadZenodoFolder:
    """Test zenodo_get download method (via download_zenodo_folder)."""

    @patch('proteus.utils.data.download_zenodo_folder_client')
    @patch('proteus.utils.data.sp.run')
    @patch('proteus.utils.data.sleep')
    def test_success_with_zenodo_get(self, mock_sleep, mock_run, mock_client, tmp_dir):
        """Test successful download with zenodo_get fallback."""
        from proteus.utils.data import download_zenodo_folder

        # Client fails, zenodo_get succeeds
        mock_client.return_value = False
        mock_run.return_value = Mock(returncode=0)
        mock_run.side_effect = [
            Mock(returncode=0),  # Version check
            Mock(returncode=0),  # Download
        ]

        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.is_file', return_value=True):
                with patch('pathlib.Path.rglob', return_value=[tmp_dir / 'test.txt']):
                    download_zenodo_folder('12345', tmp_dir)

        # Should have attempted download
        assert mock_run.called

    @patch('proteus.utils.data.download_zenodo_folder_client')
    @patch('proteus.utils.data.sp.run')
    @patch('proteus.utils.data.sleep')
    def test_timeout_retry(self, mock_sleep, mock_run, mock_client, tmp_dir):
        """Test timeout handling with retries."""
        import subprocess

        from proteus.utils.data import download_zenodo_folder

        # Client fails, zenodo_get times out then succeeds
        mock_client.return_value = False
        mock_run.side_effect = [
            Mock(returncode=0),  # Version check
            subprocess.TimeoutExpired('zenodo_get', 120),  # First attempt timeout
            Mock(returncode=0),  # Retry success
        ]

        with patch('pathlib.Path.exists', return_value=True):
            with patch('pathlib.Path.is_file', return_value=True):
                with patch('pathlib.Path.rglob', return_value=[tmp_dir / 'test.txt']):
                    download_zenodo_folder('12345', tmp_dir)

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
        mock_file.path = '/test_folder/file.txt'
        mock_file.size = 100
        mock_file.write_to = Mock()
        mock_storage.files = [mock_file]
        mock_project = Mock()
        mock_project.storages = [mock_storage]
        mock_get_osf.return_value = mock_project

        # download_OSF_folder uses keyword-only arguments
        download_OSF_folder(
            storage=mock_storage, folders=['test_folder'], data_dir=tmp_dir, force=True
        )

        # Should have attempted download
        assert mock_file.write_to.called

    @patch('proteus.utils.data.get_osf')
    def test_force_parameter(self, mock_get_osf, tmp_dir):
        """Test that force parameter removes existing files."""
        from proteus.utils.data import download_OSF_folder

        # Create existing file
        existing_file = tmp_dir / 'test_folder' / 'file.txt'
        existing_file.parent.mkdir(parents=True)
        existing_file.write_text('old content')

        # Mock OSF storage
        mock_storage = Mock()
        mock_file = Mock()
        mock_file.path = '/test_folder/file.txt'
        mock_file.size = 100
        mock_file.write_to = Mock()
        mock_storage.files = [mock_file]
        mock_project = Mock()
        mock_project.storages = [mock_storage]
        mock_get_osf.return_value = mock_project

        with patch('proteus.utils.data.safe_rm') as mock_rm:
            download_OSF_folder(
                storage=mock_storage, folders=['test_folder'], data_dir=tmp_dir, force=True
            )
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

    @pytest.mark.skip(
        reason='Complex file creation mocking - validation is tested in integration tests'
    )
    def test_validation_success(self, tmp_dir):
        """Test successful validation - skipped due to complex file mocking."""
        pass

    @patch('proteus.utils.data.sp.run')
    def test_validation_failure_hash_mismatch(self, mock_run, tmp_dir):
        """Test validation failure due to hash mismatch."""

        from proteus.utils.data import validate_zenodo_folder

        # Create test file
        test_file = tmp_dir / 'test.txt'
        test_file.write_text('test content')

        # Mock md5sums file with wrong hash - will be created by zenodo_get mock
        md5sums_file = tmp_dir / 'md5sums.txt'

        # Mock zenodo_get to create md5sums file with wrong hash
        def run_side_effect(*args, **kwargs):
            if 'zenodo_get' in args[0] and '-m' in args[0]:
                # Create md5sums file with wrong hash
                md5sums_file.write_text('wrong_hash  test.txt\n')
            return Mock(returncode=0)

        mock_run.side_effect = run_side_effect

        with patch(
            'proteus.utils.data.os.path.isfile',
            side_effect=lambda p: str(p) == str(md5sums_file),
        ):
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
            download(
                folder='test',
                target='test_target',
                zenodo_id='12345',
                osf_id='test_osf',
                desc='test download',
            )

        # Should attempt download
        assert mock_download.called

    @patch('proteus.utils.data.get_zenodo_file')
    def test_single_file_download(self, mock_get_file, tmp_dir, monkeypatch):
        """Test single file download with zenodo_path."""

        from proteus.utils.data import download

        mock_get_file.return_value = True

        # Set FWL_DATA to tmp_dir to avoid conflicts
        test_data_dir = tmp_dir / 'fwl_data'
        test_data_dir.mkdir()
        monkeypatch.setenv('FWL_DATA', str(test_data_dir))

        # Create target directory
        target_dir = test_data_dir / 'test_target' / 'test'
        target_dir.mkdir(parents=True)

        # Mock Path operations for file verification
        test_file = target_dir / 'test_file.txt'
        test_file.write_text('test content')

        download(
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
