from colorama import Fore, Style, init
import logging

# Initialize colorama for colored output
init()


# Create a custom formatter with colors
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # Apply colors based on log level
        if record.levelno == logging.INFO:
            record.levelname = f"{Fore.BLUE}pllm {record.levelname}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING:
            record.levelname = f"{Fore.YELLOW}pllm {record.levelname}{Style.RESET_ALL}"
        elif record.levelno == logging.ERROR:
            record.levelname = f"{Fore.RED}pllm {record.levelname}{Style.RESET_ALL}"
        elif record.levelno == logging.DEBUG:
            record.levelname = f"{Fore.MAGENTA}pllm {record.levelname}{Style.RESET_ALL}"

        return super().format(record)


# Configure logging to output to console with colors - only for parallellm loggers
parallellm_log_handler = logging.StreamHandler()
parallellm_log_handler.setFormatter(ColoredFormatter("[%(levelname)s] %(message)s"))
