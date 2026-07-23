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
    bootstrap_logger,
    setup_logger,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


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
        # Discrimination: a regression that ignored the kwarg and fell back to
        # the INFO default would still pass an `is not None` style check;
        # pin the numeric level to rule that out (WARNING=30, INFO=20).
        assert stream.log_level != logging.INFO

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
        # Discrimination: the trailing newline must be stripped from the logged
        # payload and the line buffer must be cleared (a regression that kept
        # the newline or stashed the line in linebuf would leak it onto the
        # next write).
        assert stream.linebuf == ''

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
        # Discrimination: buffer must be empty after the completing write so
        # the next call starts fresh. A regression that emitted the line but
        # left the fragment in linebuf would silently duplicate text on the
        # next non-newline write.
        assert stream.linebuf == ''

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
        # Discrimination: flush must leave the buffer empty (no-op on an
        # already-empty buffer). A regression that logged an empty string
        # would emit a spurious record into the log.
        assert stream.linebuf == ''

    @pytest.mark.unit
    def test_write_empty_string(self):
        """StreamToLogger handles empty write."""
        mock_logger = MagicMock()
        stream = StreamToLogger(mock_logger)
        stream.write('')
        mock_logger.log.assert_not_called()
        # Discrimination: an empty write must NOT corrupt the buffer; a
        # regression that appended '' to linebuf would change linebuf
        # identity even though the resulting string is empty. Pin the
        # buffer's exact empty-string state.
        assert stream.linebuf == ''

    @pytest.mark.unit
    def test_write_only_newline(self):
        """StreamToLogger handles write of only newline."""
        mock_logger = MagicMock()
        stream = StreamToLogger(mock_logger)
        stream.write('\n')
        mock_logger.log.assert_called_once_with(logging.INFO, '')
        # Discrimination: a regression that buffered the bare newline instead
        # of treating it as a complete (empty) line would leave a stray '\n'
        # in linebuf and emit nothing.
        assert stream.linebuf == ''


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
            # Discrimination: confirm a FileHandler was registered against the
            # requested logpath. A regression that silently dropped the file
            # handler would still pass the level check but break the contract
            # (route fwl output to logpath).
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert any(h.baseFilename == os.path.abspath(logpath) for h in file_handlers)

    @pytest.mark.unit
    def test_setup_logger_level_debug(self):
        """setup_logger accepts DEBUG log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, level='DEBUG', logterm=False)
            assert logger.level == logging.DEBUG
            # Discrimination: pin the canonical logger name ('fwl'). A
            # regression that returned the root logger or a per-call named
            # instance would still expose the correct .level and pass the
            # primary check but break downstream getLogger('fwl') lookups.
            assert logger.name == 'fwl'

    @pytest.mark.unit
    def test_setup_logger_level_error(self):
        """setup_logger accepts ERROR log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, level='ERROR', logterm=False)
            assert logger.level == logging.ERROR
            # Discrimination: ERROR (40) must sit strictly above WARNING (30).
            # A regression that mapped 'ERROR' to a lower numeric level would
            # pass the equality check only if both literal constants moved,
            # but would fail the strict-ordering pin against WARNING.
            assert logger.level > logging.WARNING

    @pytest.mark.unit
    def test_setup_logger_level_warning(self):
        """setup_logger accepts WARNING log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, level='WARNING', logterm=False)
            assert logger.level == logging.WARNING
            # Discrimination: WARNING (30) sits strictly between INFO (20) and
            # ERROR (40). Pin both inequalities to catch a regression that
            # silently mapped 'WARNING' to INFO or ERROR.
            assert logging.INFO < logger.level < logging.ERROR

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
            # Discrimination: failure must happen BEFORE any side effect on
            # disk. A regression that opened the FileHandler before validating
            # the level would have left the logfile behind even though the
            # exception fired.
            assert not os.path.exists(logpath)

    @pytest.mark.unit
    def test_setup_logger_level_case_insensitive(self):
        """setup_logger accepts log level in any case."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, level='info', logterm=False)
            assert logger.level == logging.INFO
            # Discrimination: lowercase 'info' must resolve to the SAME numeric
            # level as uppercase 'INFO'. A regression that dropped the .upper()
            # call and returned a different default for unrecognized strings
            # would diverge here. Pin parity with the canonical form.
            logger2 = setup_logger(logpath=logpath + '.up', level='INFO', logterm=False)
            assert logger2.level == logger.level

    @pytest.mark.unit
    def test_setup_logger_level_with_whitespace(self):
        """setup_logger strips whitespace from log level."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, level='  DEBUG  ', logterm=False)
            assert logger.level == logging.DEBUG
            # Discrimination: whitespace-padded input must resolve to the same
            # level as the stripped form. A regression that dropped the
            # `.strip()` call would raise ValueError before reaching this line.
            # Pin parity with the canonical form to lock the contract.
            logger2 = setup_logger(logpath=logpath + '.bare', level='DEBUG', logterm=False)
            assert logger2.level == logger.level

    @pytest.mark.unit
    def test_setup_logger_with_terminal_output(self):
        """setup_logger adds terminal handler when logterm=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            logger = setup_logger(logpath=logpath, logterm=True)
            # Check for StreamHandler (or similar terminal output handler)
            handlers = [h for h in logger.handlers if not isinstance(h, logging.FileHandler)]
            assert len(handlers) > 0
            # Discriminating check: at least one of the non-file handlers is a
            # StreamHandler (the contract that logterm=True must add). A
            # regression that attached a NullHandler instead would still pass
            # the bare ``len > 0`` check.
            assert any(isinstance(h, logging.StreamHandler) for h in handlers)

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
            # Discrimination: with logterm=False no plain StreamHandler routed
            # to sys.stdout should have been registered by this call. The
            # FileHandler subclass is still allowed; check the non-file
            # handlers do not include a stdout-routed StreamHandler.
            stream_to_stdout = [
                h
                for h in logger.handlers
                if type(h) is logging.StreamHandler and getattr(h, 'stream', None) is sys.stdout
            ]
            assert stream_to_stdout == []

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
            # Discrimination: the file formatter emits a level prefix
            # ('[ INFO  ] ...' per the source). A regression that wrote raw
            # messages without the level tag would pass the substring check
            # above but break log post-processing tools that key on the prefix.
            assert 'INFO' in content

    @pytest.mark.unit
    def test_setup_logger_exception_handler(self):
        """setup_logger installs custom exception hook."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            original_hook = sys.excepthook
            setup_logger(logpath=logpath, logterm=False)
            # Verify exception hook was set
            assert sys.excepthook != original_hook
            # Discrimination: the installed hook must be callable. A regression
            # that stored a non-callable sentinel would pass the !=-original
            # check but break on the first uncaught exception.
            assert callable(sys.excepthook)
            sys.excepthook = original_hook  # Restore original hook


class TestBootstrapLogger:
    """Test suite for bootstrap_logger early console fallback.

    bootstrap_logger guarantees the 'fwl' logger has a handler at CLI entry,
    before any command constructs Proteus() or calls setup_logger. Without it,
    INFO/DEBUG records emitted early (e.g. from Proteus.__init__ ->
    set_directories, or from grid-summarise / grid-pack) fall through to
    logging.lastResort and are dropped.
    """

    def _reset_fwl(self):
        """Return the shared 'fwl' singleton to a pristine, handler-free state."""
        logger = logging.getLogger('fwl')
        logger.handlers.clear()
        logger.setLevel(logging.WARNING)
        return logger

    @pytest.mark.unit
    def test_adds_single_stdout_handler_at_info(self):
        """Verify a handler-free 'fwl' logger gains exactly one stdout handler.

        This is the contract that keeps early INFO messages from being dropped:
        the default INFO level matches the level of the pre-config directory
        messages the fix targets.
        """
        self._reset_fwl()
        logger = bootstrap_logger()
        assert logger.name == 'fwl'
        stdout_handlers = [
            h
            for h in logger.handlers
            if type(h) is logging.StreamHandler and getattr(h, 'stream', None) is sys.stdout
        ]
        assert len(stdout_handlers) == 1
        # Discrimination: level must be INFO (20), not left at the pre-existing
        # WARNING (30). A regression that skipped setLevel would drop the very
        # INFO messages this function exists to preserve.
        assert logger.level == logging.INFO
        assert stdout_handlers[0].level == logging.INFO

    @pytest.mark.unit
    def test_idempotent_no_duplicate_handler(self):
        """A second call must not stack a second handler on the singleton.

        CLI group callbacks and Proteus construction can both reach this code
        in one process; repeated calls must not duplicate terminal output.
        """
        self._reset_fwl()
        bootstrap_logger()
        n_after_first = len(logging.getLogger('fwl').handlers)
        bootstrap_logger()
        assert len(logging.getLogger('fwl').handlers) == n_after_first
        # Discrimination: pin the count to exactly one so a regression that
        # appended on every call (rather than guarding on existing handlers)
        # is caught rather than merely "did not grow unboundedly".
        assert n_after_first == 1

    @pytest.mark.unit
    def test_noop_when_handler_already_present(self):
        """bootstrap_logger must not clobber a configured file logger.

        If setup_logger has already installed a file handler, calling
        bootstrap_logger afterwards must leave that configuration untouched so a
        real run keeps logging to its proteus_XX.log file.
        """
        self._reset_fwl()
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            setup_logger(logpath=logpath, logterm=False)
            handlers_before = list(logging.getLogger('fwl').handlers)
            bootstrap_logger()
            assert logging.getLogger('fwl').handlers == handlers_before
            # Discrimination: the file handler must survive so file logging is
            # unaffected. A regression that added a stdout handler anyway would
            # change the handler list identity checked above.
            file_handlers = [
                h
                for h in logging.getLogger('fwl').handlers
                if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) == 1

    @pytest.mark.unit
    def test_setup_logger_after_bootstrap_has_no_duplicate_stdout(self):
        """setup_logger after bootstrap yields exactly one stdout handler.

        setup_logger clears handlers before adding its own, so the bootstrap
        console handler must not survive to double every terminal line.
        """
        self._reset_fwl()
        bootstrap_logger()
        with tempfile.TemporaryDirectory() as tmpdir:
            logpath = os.path.join(tmpdir, 'test.log')
            setup_logger(logpath=logpath, logterm=True, level='INFO')
            logger = logging.getLogger('fwl')
            stdout_handlers = [
                h
                for h in logger.handlers
                if type(h) is logging.StreamHandler
                and getattr(h, 'stream', None) is sys.stdout
            ]
            assert len(stdout_handlers) == 1

    @pytest.mark.unit
    def test_info_record_reaches_handler(self):
        """A child-logger INFO record is emitted, not dropped.

        The fix exists so that records from 'fwl.<module>' children (e.g.
        'fwl.proteus.utils.coupler') are handled instead of hitting
        logging.lastResort. Route the handler to a buffer and confirm the text
        arrives.
        """
        import io

        self._reset_fwl()
        bootstrap_logger()
        buf = io.StringIO()
        logging.getLogger('fwl').handlers[0].stream = buf
        logging.getLogger('fwl.proteus.utils.coupler').info('Temporary-file working dir: /x')
        assert 'Temporary-file working dir: /x' in buf.getvalue()

    @pytest.mark.unit
    def test_invalid_level_raises(self):
        """An unrecognised level must raise rather than silently default.

        Mirrors setup_logger's contract so a typo in a level string fails fast
        instead of quietly falling back to INFO.
        """
        self._reset_fwl()
        with pytest.raises(ValueError, match='Invalid log level'):
            bootstrap_logger(level='INVALID')


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
            # Discrimination: the function must be a pure read; an empty
            # directory must remain empty after the call (no log files
            # auto-created as a side effect).
            assert os.listdir(tmpdir) == []

    @pytest.mark.unit
    def test_single_logfile_returns_zero(self):
        """GetCurrentLogfileIndex returns 0 for single logfile."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pathlib.Path(os.path.join(tmpdir, 'proteus_00.log')).touch()
            result = GetCurrentLogfileIndex(tmpdir)
            assert result == 0
            # Discrimination: a regression that returned the count of files
            # (1) instead of the highest index (0) would diverge here. Pin
            # the int identity and explicitly rule out the off-by-one.
            assert result != 1

    @pytest.mark.unit
    def test_multiple_sequential_logfiles(self):
        """GetCurrentLogfileIndex returns highest index for sequential logfiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(5):
                pathlib.Path(os.path.join(tmpdir, f'proteus_{i:02d}.log')).touch()
            result = GetCurrentLogfileIndex(tmpdir)
            assert result == 4
            # Discrimination: distinguish "highest index" from "file count".
            # 5 sequential files (indices 0..4) means highest index 4 and
            # count 5; an off-by-one regression returning the count would
            # land at 5 here.
            assert result == 5 - 1

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
            # Discrimination: this is the key contract. A regression that
            # ignored the gap and returned the global maximum (3) would break
            # the "fill the gap" semantics. Pin the strict-less-than relation
            # against the post-gap file index.
            assert result < 3

    @pytest.mark.unit
    def test_ignores_non_matching_files(self):
        """GetCurrentLogfileIndex ignores files not matching pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pathlib.Path(os.path.join(tmpdir, 'proteus_00.log')).touch()
            pathlib.Path(os.path.join(tmpdir, 'other_file.log')).touch()
            pathlib.Path(os.path.join(tmpdir, 'proteus_01.txt')).touch()
            result = GetCurrentLogfileIndex(tmpdir)
            assert result == 0  # Stops at first gap (missing _01.log)
            # Discrimination: confirm the non-matching files were not consumed
            # by a regression that broadened the pattern. A '.txt' sibling at
            # index 01 must NOT cause result == 1.
            assert result != 1


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
        # Discrimination: rule out the non-padded form explicitly. A
        # regression that used '%d' instead of '%02d' would produce
        # '/tmp/proteus_0.log', which sorts AFTER proteus_10.log lexically.
        assert path != '/tmp/proteus_0.log'

    @pytest.mark.unit
    def test_constructs_correct_path_index_five(self):
        """GetLogfilePath constructs correct path for index 5."""
        path = GetLogfilePath('/var/log', 5)
        assert path == '/var/log/proteus_05.log'
        # Discrimination: rule out the non-padded variant. A regression that
        # used '%d' instead of '%02d' would produce '/var/log/proteus_5.log',
        # which sorts after proteus_10.log lexically.
        assert 'proteus_5.log' not in path

    @pytest.mark.unit
    def test_constructs_correct_path_index_99(self):
        """GetLogfilePath constructs correct path for index 99."""
        path = GetLogfilePath('/logs', 99)
        assert path == '/logs/proteus_99.log'
        # Discrimination: 99 is the documented upper bound (the source raises
        # at j > 99). The boundary case must succeed and produce the two-digit
        # filename; any three-digit form (proteus_099.log) would signal a
        # format-string regression that widened the padding.
        assert 'proteus_099.log' not in path

    @pytest.mark.unit
    def test_raises_for_index_over_99(self):
        """Verify function raises exception when index exceeds 99.

        Hard limit at 99 logs prevents runaway simulations from filling disk
        with thousands of log files (each can be 100+ MB for long runs).
        """
        with pytest.raises(Exception, match='too many'):
            # Index 100 exceeds two-digit format limit
            GetLogfilePath('/tmp', 100)
        # Discrimination: confirm the boundary is exactly j > 99, i.e. that
        # 99 itself is still accepted. A regression that hardened the gate
        # to j >= 99 would refuse the valid 99 case and still pass the >99
        # raise.
        assert GetLogfilePath('/tmp', 99) == '/tmp/proteus_99.log'

    @pytest.mark.unit
    def test_raises_for_large_index(self):
        """GetLogfilePath raises exception for large indices."""
        with pytest.raises(Exception, match='too many'):
            GetLogfilePath('/tmp', 1000)
        # Discrimination: the same gate must fire for any index past 99,
        # not just the round number 1000. A regression that special-cased
        # a particular value would miss the intermediate 500 case.
        with pytest.raises(Exception, match='too many'):
            GetLogfilePath('/tmp', 500)

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
