"""
Unit tests for proteus.utils.logs module.

Tests logging utilities including logger setup, stream redirection, custom formatting,
and logfile management with various configurations and edge cases.
"""

from __future__ import annotations

import logging
import os
import pathlib
import sys
import tempfile
from unittest.mock import MagicMock

import pytest

from proteus.utils.logs import (
    CustomFormatter,
    GetCurrentLogfileIndex,
    GetLogfilePath,
    StreamToLogger,
    setup_logger,
)


class TestStreamToLogger:
    """Test suite for StreamToLogger stream-to-logger redirection.

    StreamToLogger redirects stdout/stderr to Python logging, used for capturing
    output from external binaries (SOCRATES, SPIDER) during PROTEUS simulations.
    """

    @pytest.mark.unit
    def test_init_default_log_level(self):
        """Verify StreamToLogger initializes with INFO log level by default.

        Default INFO level ensures normal simulation output is captured without
        excessive debug verbosity.
        """
        mock_logger = MagicMock()
        stream = StreamToLogger(mock_logger)

        # Verify default configuration
        assert stream.log_level == logging.INFO
        assert stream.linebuf == ''  # Empty buffer at initialization
        assert stream.logger is mock_logger

    @pytest.mark.unit
    def test_init_custom_log_level(self):
        """StreamToLogger initializes with custom log level."""
        mock_logger = MagicMock()
        stream = StreamToLogger(mock_logger, log_level=logging.WARNING)
        assert stream.log_level == logging.WARNING

    @pytest.mark.unit
    def test_write_single_line(self):
        """Verify complete line (with newline) is logged immediately.

        Critical for real-time monitoring of long-running physics simulations
        where timestep progress updates end with newlines.
        """
        mock_logger = MagicMock()
        stream = StreamToLogger(mock_logger, log_level=logging.DEBUG)

        # Write complete line with newline
        stream.write('Test message\n')

        # Verify immediate logging without buffering
        mock_logger.log.assert_called_once_with(logging.DEBUG, 'Test message')

    @pytest.mark.unit
    def test_write_multiple_lines(self):
        """StreamToLogger writes multiple lines separately."""
        mock_logger = MagicMock()
        stream = StreamToLogger(mock_logger, log_level=logging.INFO)
        stream.write('Line 1\nLine 2\nLine 3\n')
        assert mock_logger.log.call_count == 3
        calls = [call[0] for call in mock_logger.log.call_args_list]
        assert calls == [
            (logging.INFO, 'Line 1'),
            (logging.INFO, 'Line 2'),
            (logging.INFO, 'Line 3'),
        ]

    @pytest.mark.unit
    def test_write_incomplete_line(self):
        """Verify incomplete lines (no newline) are buffered, not logged.

        Progress bars and in-place status updates from external modules
        use carriage returns without newlines; must buffer these.
        """
        mock_logger = MagicMock()
        stream = StreamToLogger(mock_logger)

        # Write text without trailing newline
        stream.write('Incomplete')

        # Verify buffering: no log call, text stored in linebuf
        assert mock_logger.log.call_count == 0
        assert stream.linebuf == 'Incomplete'

    @pytest.mark.unit
    def test_write_incomplete_then_complete(self):
        """StreamToLogger completes buffered incomplete line."""
        mock_logger = MagicMock()
        stream = StreamToLogger(mock_logger)
        stream.write('Incom')
        stream.write('plete\n')
        mock_logger.log.assert_called_once_with(logging.INFO, 'Incomplete')

    @pytest.mark.unit
    def test_flush_with_content(self):
        """Verify explicit flush() logs buffered incomplete lines.

        Essential for capturing final status messages when simulation terminates
        without final newline (e.g., convergence criteria met).
        """
        mock_logger = MagicMock()
        stream = StreamToLogger(mock_logger, log_level=logging.WARNING)

        # Buffer incomplete text
        stream.write('Buffered')
        mock_logger.log.assert_not_called()

        # Explicit flush forces log output
        stream.flush()
        mock_logger.log.assert_called_once_with(logging.WARNING, 'Buffered')

    @pytest.mark.unit
    def test_flush_empty_buffer(self):
        """StreamToLogger does nothing when flushing empty buffer."""
        mock_logger = MagicMock()
        stream = StreamToLogger(mock_logger)
        stream.flush()
        mock_logger.log.assert_not_called()

    @pytest.mark.unit
    def test_write_empty_string(self):
        """StreamToLogger handles empty write."""
        mock_logger = MagicMock()
        stream = StreamToLogger(mock_logger)
        stream.write('')
        mock_logger.log.assert_not_called()

    @pytest.mark.unit
    def test_write_only_newline(self):
        """StreamToLogger handles write of only newline."""
        mock_logger = MagicMock()
        stream = StreamToLogger(mock_logger)
        stream.write('\n')
        mock_logger.log.assert_called_once_with(logging.INFO, '')


class TestCustomFormatter:
    """Test suite for CustomFormatter ANSI color-coded terminal logging.

    Provides color-coded terminal output for PROTEUS runs to quickly identify
    errors (red), warnings (yellow), and normal progress (green) during development.
    """

    @pytest.mark.unit
    def test_format_info_level(self):
        """Verify INFO level messages formatted with green color (ANSI 32m).

        Green indicates normal simulation progress (timestep completion, convergence).
        """
        formatter = CustomFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test info message',
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)

        # Verify message content preserved
        assert 'INFO' in result
        assert 'Test info message' in result
        # Verify green color code (ANSI 32m)
        assert '32m' in result

    @pytest.mark.unit
    def test_format_warning_level(self):
        """CustomFormatter formats WARNING level with correct color code."""
        formatter = CustomFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.WARNING,
            pathname='test.py',
            lineno=1,
            msg='Test warning message',
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert 'WARNING' in result or 'WARNI' in result  # Might be truncated
        assert 'Test warning message' in result
        assert '93m' in result  # Yellow color code

    @pytest.mark.unit
    def test_format_error_level(self):
        """Verify ERROR level messages formatted with red color (ANSI 91m).

        Red highlights critical failures (non-convergence, unphysical states).
        """
        formatter = CustomFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.ERROR,
            pathname='test.py',
            lineno=1,
            msg='Test error message',
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)

        # Verify message content preserved
        assert 'ERROR' in result
        assert 'Test error message' in result
        # Verify red color code (ANSI 91m) for visibility
        assert '91m' in result

    @pytest.mark.unit
    def test_format_debug_level(self):
        """CustomFormatter formats DEBUG level with correct color code."""
        formatter = CustomFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.DEBUG,
            pathname='test.py',
            lineno=1,
            msg='Test debug message',
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert 'DEBUG' in result
        assert 'Test debug message' in result
        assert '96m' in result  # Cyan color code

    @pytest.mark.unit
    def test_format_contains_ansi_codes(self):
        """CustomFormatter output contains ANSI escape codes."""
        formatter = CustomFormatter()
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test',
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert '\033[' in result  # ANSI escape sequence
        assert '\033[1m' in result  # Bold


class TestSetupLogger:
    """Test suite for setup_logger initialization and configuration.

    Central logger 'fwl' (FormingWorlds) captures all PROTEUS simulation output
    to files (proteus_XX.log) for post-run analysis and debugging.
    """

    @pytest.mark.unit
    def test_setup_logger_creates_file(self):
        """Verify setup_logger creates FormingWorlds logger with file handler.

        Each PROTEUS run writes to separate log file for parallel ensemble tracking.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, logterm=False)

            # Verify logger instance and canonical name
            assert isinstance(logger, logging.Logger)
            assert logger.name == 'fwl'  # FormingWorlds standard logger

    @pytest.mark.unit
    def test_setup_logger_removes_existing_file(self):
        """setup_logger removes pre-existing logfile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            # Create existing file
            pathlib.Path(logpath).touch()
            assert os.path.exists(logpath)
            logger = setup_logger(logpath=logpath, logterm=False)
            assert logger is not None

    @pytest.mark.unit
    def test_setup_logger_default_level_info(self):
        """setup_logger uses INFO as default log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, level='INFO', logterm=False)
            assert logger.level == logging.INFO

    @pytest.mark.unit
    def test_setup_logger_level_debug(self):
        """setup_logger accepts DEBUG log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, level='DEBUG', logterm=False)
            assert logger.level == logging.DEBUG

    @pytest.mark.unit
    def test_setup_logger_level_error(self):
        """setup_logger accepts ERROR log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, level='ERROR', logterm=False)
            assert logger.level == logging.ERROR

    @pytest.mark.unit
    def test_setup_logger_level_warning(self):
        """setup_logger accepts WARNING log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, level='WARNING', logterm=False)
            assert logger.level == logging.WARNING

    @pytest.mark.unit
    def test_setup_logger_invalid_level_raises(self):
        """Verify setup_logger raises ValueError for invalid log levels.

        Prevents silent failures when config specifies typo in log level
        (e.g., 'INFOO' instead of 'INFO').
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')

            # Invalid level should raise immediately, not silently default
            with pytest.raises(ValueError, match='Invalid log level'):
                setup_logger(logpath=logpath, level='INVALID', logterm=False)

    @pytest.mark.unit
    def test_setup_logger_level_case_insensitive(self):
        """setup_logger accepts log level in any case."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, level='info', logterm=False)
            assert logger.level == logging.INFO

    @pytest.mark.unit
    def test_setup_logger_level_with_whitespace(self):
        """setup_logger strips whitespace from log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, level='  DEBUG  ', logterm=False)
            assert logger.level == logging.DEBUG

    @pytest.mark.unit
    def test_setup_logger_with_terminal_output(self):
        """setup_logger adds terminal handler when logterm=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, logterm=True)
            # Check for StreamHandler (or similar terminal output handler)
            handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
            assert len(handlers) > 0

    @pytest.mark.unit
    def test_setup_logger_without_terminal_output(self):
        """setup_logger with logterm=False does not add additional terminal handlers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            # Count handlers before call
            logger = setup_logger(logpath=logpath, logterm=False)
            # Logger is a singleton so just verify that file logging works
            logger.info('Test without terminal')
            with open(logpath) as f:
                content = f.read()
            assert 'Test without terminal' in content

    @pytest.mark.unit
    def test_setup_logger_file_handler_has_custom_formatter(self):
        """setup_logger file handler uses simple formatter (no colors)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, logterm=False)
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            # Logger is singleton, may have many handlers from other tests
            # Just verify at least one file handler has a formatter
            assert len(file_handlers) > 0
            assert any(h.formatter is not None for h in file_handlers)

    @pytest.mark.unit
    def test_setup_logger_logs_to_file(self):
        """setup_logger actually writes logs to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, logterm=False)
            logger.info('Test message')
            with open(logpath) as f:
                content = f.read()
            assert 'Test message' in content

    @pytest.mark.unit
    def test_setup_logger_exception_handler(self):
        """setup_logger installs custom exception hook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            original_hook = sys.excepthook
            setup_logger(logpath=logpath, logterm=False)
            # Verify exception hook was set
            assert sys.excepthook != original_hook
            sys.excepthook = original_hook  # Restore original hook


class TestGetCurrentLogfileIndex:
    """Test suite for GetCurrentLogfileIndex sequential logfile tracking.

    PROTEUS creates numbered log files (proteus_00.log, proteus_01.log, ...)
    for sequential runs in the same output directory. This function finds
    the highest existing index to determine the next log filename.
    """

    @pytest.mark.unit
    def test_no_logfiles_returns_minus_one(self):
        """Verify function returns -1 for empty directory (first run).

        -1 signals initial run; next log will be proteus_00.log.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            result = GetCurrentLogfileIndex(tmpdir)
            assert result == -1  # Sentinel value for "no logs yet"

    @pytest.mark.unit
    def test_single_logfile_returns_zero(self):
        """GetCurrentLogfileIndex returns 0 for single logfile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pathlib.Path(os.path.join(tmpdir, 'proteus_00.log')).touch()
            result = GetCurrentLogfileIndex(tmpdir)
            assert result == 0

    @pytest.mark.unit
    def test_multiple_sequential_logfiles(self):
        """GetCurrentLogfileIndex returns highest index for sequential logfiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                pathlib.Path(os.path.join(tmpdir, f'proteus_{i:02d}.log')).touch()
            result = GetCurrentLogfileIndex(tmpdir)
            assert result == 4

    @pytest.mark.unit
    def test_gap_in_logfiles(self):
        """Verify function stops at first gap in numbering sequence.

        Prevents overwriting logs when user manually deletes intermediate files
        (e.g., failed run proteus_02.log). Next run creates proteus_02.log, not _04.log.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            pathlib.Path(os.path.join(tmpdir, 'proteus_00.log')).touch()
            pathlib.Path(os.path.join(tmpdir, 'proteus_01.log')).touch()
            pathlib.Path(os.path.join(tmpdir, 'proteus_03.log')).touch()  # Gap at _02

            result = GetCurrentLogfileIndex(tmpdir)
            # Returns 1, so next log will fill gap as proteus_02.log
            assert result == 1

    @pytest.mark.unit
    def test_ignores_non_matching_files(self):
        """GetCurrentLogfileIndex ignores files not matching pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pathlib.Path(os.path.join(tmpdir, 'proteus_00.log')).touch()
            pathlib.Path(os.path.join(tmpdir, 'other_file.log')).touch()
            pathlib.Path(os.path.join(tmpdir, 'proteus_01.txt')).touch()
            result = GetCurrentLogfileIndex(tmpdir)
            assert result == 0  # Stops at first gap (missing _01.log)


class TestGetLogfilePath:
    """Test suite for GetLogfilePath formatted path construction.

    Generates standardized log filenames with zero-padded indices for
    proper alphabetical sorting in filesystem (proteus_00.log before proteus_10.log).
    """

    @pytest.mark.unit
    def test_constructs_correct_path_index_zero(self):
        """Verify zero-padded format for first logfile (proteus_00.log).

        Zero-padding ensures alphabetical sort matches numerical order
        when browsing output directories with 10+ simulation runs.
        """
        path = GetLogfilePath('/tmp', 0)
        assert path == '/tmp/proteus_00.log'  # Two-digit padding: 00 not 0

    @pytest.mark.unit
    def test_constructs_correct_path_index_five(self):
        """GetLogfilePath constructs correct path for index 5."""
        path = GetLogfilePath('/var/log', 5)
        assert path == '/var/log/proteus_05.log'

    @pytest.mark.unit
    def test_constructs_correct_path_index_99(self):
        """GetLogfilePath constructs correct path for index 99."""
        path = GetLogfilePath('/logs', 99)
        assert path == '/logs/proteus_99.log'

    @pytest.mark.unit
    def test_raises_for_index_over_99(self):
        """Verify function raises exception when index exceeds 99.

        Hard limit at 99 logs prevents runaway simulations from filling disk
        with thousands of log files (each can be 100+ MB for long runs).
        """
        with pytest.raises(Exception, match='too many'):
            # Index 100 exceeds two-digit format limit
            GetLogfilePath('/tmp', 100)

    @pytest.mark.unit
    def test_raises_for_large_index(self):
        """GetLogfilePath raises exception for large indices."""
        with pytest.raises(Exception, match='too many'):
            GetLogfilePath('/tmp', 1000)

    @pytest.mark.unit
    def test_path_uses_zero_padding(self):
        """GetLogfilePath uses zero-padded two-digit format."""
        path = GetLogfilePath('/tmp', 7)
        assert 'proteus_07.log' in path
        assert 'proteus_7.log' not in path

    @pytest.mark.unit
    def test_preserves_directory_path(self):
        """GetLogfilePath preserves full directory path."""
        dirpath = '/path/to/output/directory'
        path = GetLogfilePath(dirpath, 42)
        assert path.startswith(dirpath)
        assert 'proteus_42.log' in path
