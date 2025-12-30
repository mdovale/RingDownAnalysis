"""
Example logging configuration for ringdownanalysis package.

This demonstrates how to configure logging for production use or debugging.
"""

import logging
import sys
from pathlib import Path


# Example 1: Basic console logging (INFO level)
def setup_basic_logging():
    """Set up basic console logging at INFO level."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Example 2: Detailed console logging (DEBUG level)
def setup_debug_logging():
    """Set up detailed console logging at DEBUG level."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Example 3: Structured logging to file (production)
def setup_file_logging(log_file="ringdown_analysis.log"):
    """Set up file logging with structured format."""
    log_path = Path(log_file)

    # Create file handler
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)

    # Structured format: event + key fields
    # Note: extra fields (like 'event') are accessible via %(event)s in the formatter
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            # Get event from extra, default to message if not present
            event = getattr(record, "event", record.getMessage())
            # Build message with key fields
            msg_parts = [f"event={event}"]
            # Add other common extra fields
            for key in ["filepath", "n_files", "n_samples", "error_type"]:
                if hasattr(record, key):
                    msg_parts.append(f"{key}={getattr(record, key)}")
            record.msg = " | ".join(msg_parts)
            record.args = ()
            return super().format(record)

    file_formatter = StructuredFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Get package logger
    package_logger = logging.getLogger("ringdownanalysis")
    package_logger.setLevel(logging.INFO)
    package_logger.addHandler(file_handler)

    # Also log to console at WARNING level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter(
        "%(levelname)s - %(name)s - %(message)s",
    )
    console_handler.setFormatter(console_formatter)
    package_logger.addHandler(console_handler)


# Example 4: JSON logging for structured log aggregation
def setup_json_logging(log_file="ringdown_analysis.jsonl"):
    """
    Set up JSON logging for structured log aggregation.

    Note: Requires json module (standard library) or a JSON logging library
    for production use. This is a simplified example.
    """
    import json
    from datetime import datetime

    class JSONFormatter(logging.Formatter):
        def format(self, record):
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "event": getattr(record, "event", "unknown"),
                "message": record.getMessage(),
            }
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in [
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                    "event",
                ]:
                    log_entry[key] = value
            return json.dumps(log_entry)

    log_path = Path(log_file)
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(JSONFormatter())

    package_logger = logging.getLogger("ringdownanalysis")
    package_logger.setLevel(logging.INFO)
    package_logger.addHandler(file_handler)


# Example 5: Production-ready configuration
def setup_production_logging(
    log_dir="logs",
    log_level=logging.INFO,
    console_level=logging.WARNING,
):
    """
    Set up production-ready logging configuration.

    Parameters:
    -----------
    log_dir : str
        Directory for log files
    log_level : int
        Logging level for file handler
    console_level : int
        Logging level for console handler (typically higher to reduce noise)
    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # File handler with rotation
    from logging.handlers import RotatingFileHandler

    file_handler = RotatingFileHandler(
        log_path / "ringdown_analysis.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setLevel(log_level)

    # Structured formatter that handles extra fields
    class StructuredFormatter(logging.Formatter):
        def format(self, record):
            event = getattr(record, "event", record.getMessage())
            msg_parts = [f"event={event}"]
            # Add other common extra fields
            for key in [
                "filepath",
                "n_files",
                "n_samples",
                "error_type",
                "f_nls",
                "f_dft",
                "tau_est",
            ]:
                if hasattr(record, key):
                    msg_parts.append(f"{key}={getattr(record, key)}")
            record.msg = " | ".join(msg_parts)
            record.args = ()
            return super().format(record)

    file_formatter = StructuredFormatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        "%(levelname)s - %(name)s - %(message)s",
    )
    console_handler.setFormatter(console_formatter)

    # Configure package logger
    package_logger = logging.getLogger("ringdownanalysis")
    package_logger.setLevel(logging.DEBUG)  # Capture all, filter via handlers
    package_logger.addHandler(file_handler)
    package_logger.addHandler(console_handler)
    package_logger.propagate = False  # Prevent duplicate logs


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        example = sys.argv[1]
    else:
        example = "1"

    print("=" * 70)
    print(f"Example {example}: Logging Configuration Demo")
    print("=" * 70)
    print()

    if example == "1":
        print("Setting up basic console logging (INFO level)...")
        setup_basic_logging()
        print("✓ Logging configured")
        print()
        print("Now importing ringdownanalysis - you should see logs when using it:")
        print()
        print("✓ Package imported")
        print()
        print("To see logging in action, try:")
        print("  analyzer = RingDownAnalyzer()")
        print("  result = analyzer.analyze_file('data/your_file.csv')")

    elif example == "2":
        print("Setting up debug logging (DEBUG level)...")
        setup_debug_logging()
        print("✓ Logging configured")
        print()
        print("This will show detailed debug information including:")
        print("  - Optimization iterations")
        print("  - Parameter values")
        print("  - Detailed diagnostic information")
        print()
        print("✓ Package imported")

    elif example == "3":
        print("Setting up file logging...")
        setup_file_logging("demo_ringdown_analysis.log")
        print("✓ Logging configured - logs will be written to 'demo_ringdown_analysis.log'")
        print("  Console will only show WARNING and ERROR messages")
        print()
        print("✓ Package imported")
        print("Try using the package - logs will be written to the file")

    elif example == "4":
        print("Setting up JSON logging...")
        setup_json_logging("demo_ringdown_analysis.jsonl")
        print("✓ Logging configured - logs will be written to 'demo_ringdown_analysis.jsonl'")
        print("  Each line is a JSON object for easy parsing")
        print()
        print("✓ Package imported")

    elif example == "5":
        print("Setting up production logging...")
        setup_production_logging("demo_logs", logging.INFO, logging.WARNING)
        print("✓ Logging configured")
        print("  - File logs: 'demo_logs/ringdown_analysis.log' (with rotation)")
        print("  - Console: WARNING and ERROR only")
        print()
        print("✓ Package imported")

    else:
        print("Available examples:")
        print("  1 - Basic console logging (INFO level)")
        print("  2 - Debug logging (DEBUG level)")
        print("  3 - File logging")
        print("  4 - JSON logging")
        print("  5 - Production logging")
        print()
        print("Usage: python logging_config_example.py [1-5]")
        print()
        print("Example: python logging_config_example.py 1")

    print()
    print("=" * 70)
