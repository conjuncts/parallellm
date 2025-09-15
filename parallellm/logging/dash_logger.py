import sys
import threading
from collections import OrderedDict
from typing import Dict, Optional, Set
from colorama import Fore, Style, init
from dataclasses import dataclass
from enum import Enum

# Initialize colorama for colored output
init()


class HashStatus(Enum):
    """Enum for different hash statuses"""

    CACHED = "C"  # cached
    SENT = "↗"  # sent to provider
    RECEIVED = "↘"  # received from provider
    STORED = "✓"  # stored in datastore


@dataclass
class HashEntry:
    """Data class to represent a hash entry"""

    hash_id: str
    status: HashStatus
    full_hash: str


class DashboardLogger:
    """
    Sophisticated hash logger that displays top k=10 hashes with dynamic console updates.

    Features:
    - Shows the top k=10 hash entries
    - Allows updating hash statuses
    - Option to enable/disable console output
    - Console rewrites itself to minimize spam
    - Status changes update existing entries instead of adding new ones
    - Carriage return based rewriting that allows standard print() calls
    """

    def __init__(self, k: int = 10, display: bool = True):
        """
        Initialize the DashboardLogger.

        Args:
            k: Maximum number of hashes to display (default 10)
            display: Whether to display console output (default True)
        """
        self.k = k
        self.display = display
        self._lock = threading.Lock()

        # OrderedDict to maintain insertion order and allow efficient updates
        self._hashes: OrderedDict[str, HashEntry] = OrderedDict()

        # Track if we've written to console before
        self._console_written = False

        # Colors for different statuses
        self._status_colors = {
            HashStatus.CACHED: Fore.GREEN,
            HashStatus.SENT: Fore.CYAN,
            HashStatus.RECEIVED: Fore.GREEN,
            HashStatus.STORED: Fore.GREEN,
        }

    def update_hash(self, full_hash: str, status: HashStatus):
        """
        Update or add a hash with the given status.

        Args:
            full_hash: The full hash string
            status: The status of the hash
        """
        with self._lock:
            hash_id = full_hash[:8]  # Use first 8 characters as display ID

            if hash_id in self._hashes:
                # Update existing entry
                self._hashes[hash_id].status = status
            else:
                # Add new entry
                entry = HashEntry(hash_id=hash_id, status=status, full_hash=full_hash)
                self._hashes[hash_id] = entry

                # Keep only the most recent k entries
                while len(self._hashes) > self.k:
                    # Remove the oldest entry (first item in OrderedDict)
                    self._hashes.popitem(last=False)

            # Move the updated/added entry to the end (most recent)
            self._hashes.move_to_end(hash_id)

            if self.display:
                self._update_console()

    def _update_console(self):
        """Update the console display with current hash statuses"""
        if not self.display:
            return

        # Build the display line with grey [DASH] prefix
        prefix = f"{Fore.LIGHTBLACK_EX}[pllm DASH]{Style.RESET_ALL} "
        status_parts = []
        for entry in self._hashes.values():
            color = self._status_colors.get(entry.status, Fore.WHITE)
            status_parts.append(
                f"{color}{entry.status.value} {entry.hash_id}{Style.RESET_ALL}"
            )

        display_line = prefix + " ".join(status_parts)

        if self._console_written:
            # Move cursor to beginning of line and clear the entire line, then print new content
            # Use ANSI escape sequence to clear the entire line
            sys.stdout.write(f"\r\033[K{display_line}")
        else:
            # First time writing - just print normally
            sys.stdout.write(display_line)
            self._console_written = True

        sys.stdout.flush()

    def set_display(self, display: bool):
        """Enable or disable console display"""
        with self._lock:
            self.display = display
            if display:
                self._update_console()
            elif self._console_written:
                # Clear the line if disabling display using ANSI escape sequence
                sys.stdout.write(f"\r\033[K")
                sys.stdout.flush()
                self._console_written = False

    def clear(self):
        """Clear all hash entries"""
        with self._lock:
            self._hashes.clear()
            if self._console_written:
                # Clear the console line using ANSI escape sequence
                sys.stdout.write(f"\r\033[K")
                sys.stdout.flush()
                self._console_written = False

    def get_status(self, full_hash: str) -> Optional[HashStatus]:
        """
        Get the current status of a hash.

        Args:
            full_hash: The full hash string

        Returns:
            The current status of the hash, or None if not found
        """
        with self._lock:
            hash_id = full_hash[:8]
            entry = self._hashes.get(hash_id)
            return entry.status if entry else None

    def get_all_hashes(self) -> Dict[str, HashStatus]:
        """
        Get all current hashes and their statuses.

        Returns:
            Dictionary mapping full hashes to their statuses
        """
        with self._lock:
            return {entry.full_hash: entry.status for entry in self._hashes.values()}

    def finalize_line(self):
        """
        Finalize the current console line by moving to the next line.
        Call this when you want to ensure subsequent print() calls appear on new lines.
        """
        if self._console_written:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self._console_written = False
