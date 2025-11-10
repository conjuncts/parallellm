from colorama import Fore, Style, init
import logging
import sys
from typing import Optional

from parallellm.logging.dash_logger import DashboardLogger

# Initialize colorama for colored output
init()


# Create a custom formatter with colors
class ColoredFormatter(logging.Formatter):
    _status_colors = {
        logging.INFO: Fore.CYAN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.DEBUG: Fore.LIGHTBLACK_EX,
    }

    def format(self, record):
        # Apply colors based on log level
        color = self._status_colors.get(record.levelno, "")
        record.levelname = f"{color}[pllm {record.levelname}]{Style.RESET_ALL}"

        return super().format(record)


class DashboardAwareHandler(logging.StreamHandler):
    """
    A logging handler that coordinates with the dashboard logger when active.
    When the dashboard is active, it prepends \r\033[K to clear the current line
    before outputting log messages.
    """

    def __init__(self, stream=None):
        super().__init__(stream)
        self._dash_logger: Optional[DashboardLogger] = None

    def set_dash_logger(self, dash_logger: DashboardLogger):
        """Set the dashboard logger to coordinate with."""
        self._dash_logger = dash_logger

    def emit(self, record):
        """Emit a log record, coordinating with dashboard if active."""
        try:
            msg = self.format(record)

            # Check if dashboard logger is active and displaying
            if self._dash_logger.display and self._dash_logger._console_written:
                # Dashboard is active, use coordinated approach
                # Clear current line and print message
                self.stream.write(f"\r\033[K{msg}\n")
                self.stream.flush()

            else:
                # No dashboard or dashboard not active, use regular output
                self.stream.write(msg + self.terminator)
                self.stream.flush()

        except Exception:
            self.handleError(record)


# Configure logging to output to console with colors - only for parallellm loggers
parallellm_log_handler = DashboardAwareHandler()
parallellm_log_handler.setFormatter(ColoredFormatter("%(levelname)s %(message)s"))
