from colorama import Fore, Style, init
import logging

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


# Configure logging to output to console with colors - only for parallellm loggers
parallellm_log_handler = logging.StreamHandler()
parallellm_log_handler.setFormatter(ColoredFormatter("%(levelname)s %(message)s"))
