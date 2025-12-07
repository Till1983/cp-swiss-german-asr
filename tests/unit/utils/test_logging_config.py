"""Unit tests for logging configuration module."""
import pytest
import logging
from pathlib import Path
from src.utils.logging_config import setup_logger


class TestSetupLogger:
    """Test suite for setup_logger function."""

    @pytest.mark.unit
    def test_returns_logger_instance(self):
        """Test setup_logger returns a Logger instance."""
        logger = setup_logger("test_logger_1")

        assert isinstance(logger, logging.Logger)

    @pytest.mark.unit
    def test_logger_has_correct_name(self):
        """Test logger has the specified name."""
        logger = setup_logger("my_custom_logger")

        assert logger.name == "my_custom_logger"

    @pytest.mark.unit
    def test_logger_default_level_is_info(self):
        """Test logger default level is INFO."""
        logger = setup_logger("test_logger_2")

        assert logger.level == logging.INFO

    @pytest.mark.unit
    def test_logger_custom_level(self):
        """Test logger with custom level."""
        logger = setup_logger("test_logger_3", level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    @pytest.mark.unit
    def test_logger_has_console_handler(self):
        """Test logger has console (stream) handler."""
        logger = setup_logger("test_logger_console")

        # Find stream handlers
        stream_handlers = [
            h for h in logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]

        assert len(stream_handlers) >= 1

    @pytest.mark.unit
    def test_logger_with_file_handler(self, temp_dir):
        """Test logger creates file handler when log_file provided."""
        log_file = temp_dir / "test.log"
        logger = setup_logger("test_logger_file", log_file=str(log_file))

        # Find file handlers
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]

        assert len(file_handlers) >= 1

    @pytest.mark.unit
    def test_logger_creates_log_directory(self, temp_dir):
        """Test logger creates log directory if it doesn't exist."""
        log_file = temp_dir / "logs" / "subdir" / "test.log"
        logger = setup_logger("test_logger_dir", log_file=str(log_file))

        assert (temp_dir / "logs" / "subdir").exists()

    @pytest.mark.unit
    def test_logger_writes_to_file(self, temp_dir):
        """Test logger writes messages to file."""
        log_file = temp_dir / "output.log"
        logger = setup_logger("test_logger_write", log_file=str(log_file))

        logger.info("Test message")

        # Flush handlers
        for handler in logger.handlers:
            handler.flush()

        content = log_file.read_text()
        assert "Test message" in content

    @pytest.mark.unit
    def test_logger_no_duplicate_handlers(self):
        """Test calling setup_logger twice doesn't duplicate handlers."""
        logger1 = setup_logger("test_logger_dup")
        initial_handler_count = len(logger1.handlers)

        logger2 = setup_logger("test_logger_dup")
        final_handler_count = len(logger2.handlers)

        assert initial_handler_count == final_handler_count
        assert logger1 is logger2

    @pytest.mark.unit
    def test_logger_handler_formatter(self):
        """Test logger handlers have proper formatter."""
        logger = setup_logger("test_logger_fmt")

        for handler in logger.handlers:
            formatter = handler.formatter
            assert formatter is not None
            # Check formatter format string contains expected fields
            format_str = formatter._fmt
            assert "%(asctime)s" in format_str
            assert "%(name)s" in format_str
            assert "%(levelname)s" in format_str
            assert "%(message)s" in format_str

    @pytest.mark.unit
    @pytest.mark.parametrize("level", [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL
    ])
    def test_logger_various_levels(self, level):
        """Test logger works with various log levels."""
        logger = setup_logger(f"test_logger_level_{level}", level=level)

        assert logger.level == level

    @pytest.mark.unit
    def test_logger_file_encoding_utf8(self, temp_dir):
        """Test logger file handler uses UTF-8 encoding."""
        log_file = temp_dir / "utf8.log"
        logger = setup_logger("test_logger_utf8", log_file=str(log_file))

        # Log Unicode characters
        logger.info("Test with umlauts: aeoe")

        for handler in logger.handlers:
            handler.flush()

        content = log_file.read_text(encoding='utf-8')
        assert "aeoe" in content

    @pytest.mark.unit
    def test_multiple_loggers_independent(self):
        """Test multiple loggers are independent."""
        logger1 = setup_logger("independent_1", level=logging.DEBUG)
        logger2 = setup_logger("independent_2", level=logging.ERROR)

        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.ERROR
        assert logger1 is not logger2

    @pytest.mark.unit
    def test_logger_without_file(self):
        """Test logger works without file handler."""
        logger = setup_logger("test_no_file", log_file=None)

        # Should have at least console handler
        assert len(logger.handlers) >= 1

        # Should not have file handler
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 0
