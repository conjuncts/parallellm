import os
import json
import pickle
import time
import atexit
import re
import hashlib
from pathlib import Path
from typing import Optional

from parallellm.types import WorkingMetadata


class FileManager:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata_file = self.directory / "metadata.json"
        self.lock_file = self.directory / ".filemanager.lock"

        # Create directory if it doesn't exist
        self.directory.mkdir(parents=True, exist_ok=True)

        # Create lock file
        self._create_lock()

        # Register cleanup on exit
        atexit.register(self._cleanup)

        # Load existing metadata
        self.metadata = self._load_metadata()
        if self.metadata is None:
            self.metadata = {}
        self.metadata.setdefault(
            "agents",
            {
                "default-agent": {
                    "latest_checkpoint": None,
                    "checkpoint_counter": 0,
                }
            },
        )

        # session_counter: default to 0
        self.metadata["session_counter"] = self.metadata.get("session_counter", -1) + 1

    def _sanitize(
        self, user_input: Optional[str], *, default="default", add_hash=True
    ) -> str:
        """
        Sanitize user input to be safe for use as directory name.
        Uses format: <first_64_chars>-<8_letter_hash> for non-None checkpoints.
        Returns "default" if checkpoint is None (non-checkpointed mode).

        :param checkpoint: The checkpoint name to sanitize
        :returns: Sanitized checkpoint name ("default" for None)
        """
        if user_input is None:
            return default

        if not isinstance(user_input, str):
            user_input = str(user_input)

        cleaned = "".join(
            [c if (c.isalnum() or c in " _-") else "_" for c in user_input]
        )

        # Replace multiple spaces/underscores with single ones
        cleaned = re.sub(r"[_\s]+", "_", cleaned)

        # Remove leading/trailing whitespace, underscores, and dots
        cleaned = cleaned.strip().strip("_")

        # Take first 64 characters
        if len(cleaned) > 64:
            cleaned = cleaned[:64].rstrip("_.")

        if not cleaned:
            cleaned = "checkpoint"

        # Return format: chk-<first_64_chars>-<8_letter_hash>
        if not add_hash:
            return cleaned

        checkpoint_hash = hashlib.sha256(user_input.encode("utf-8")).hexdigest()[:8]
        return f"{cleaned}-{checkpoint_hash}"

    def _create_lock(self):
        """Create lock file with current process ID"""
        with open(self.lock_file, "w") as f:
            f.write(str(os.getpid()))

    def _cleanup(self):
        """Cleanup method called on object destruction"""
        if self.lock_file.exists():
            try:
                self.lock_file.unlink()
            except (FileNotFoundError, PermissionError):
                pass  # Lock was already removed or can't be removed

    # def __del__(self):
    #     """Destructor to ensure cleanup"""
    #     self._cleanup()

    def _load_metadata(self) -> Optional[WorkingMetadata]:
        """Load metadata from JSON file"""
        try:
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _save_metadata(self, metadata: WorkingMetadata):
        """Save metadata to JSON file"""
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    def save_userdata(self, key: str, value, overwrite=False):
        """
        Internally persist data across checkpoints

        :param checkpoint: The checkpoint name, or None for default checkpoint
        :param key: The data key
        :param value: The data value to save
        :param overwrite: Whether to overwrite existing data
        """

        # Create checkpoint directory
        checkpoint_dir = self.directory / "userdata"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save data using pickle for complex objects
        fname = self._sanitize(key)
        data_file = checkpoint_dir / f"{fname}.pkl"

        if data_file.exists() and not overwrite:
            return

        with open(data_file, "wb") as f:
            pickle.dump(value, f)

    def load_userdata(self, key: str):
        """
        Internally load data across checkpoints

        :param key: The data key to load
        :returns: The loaded data
        :raises FileNotFoundError: If the data file is not found
        """
        checkpoint_dir = self.directory / "userdata"

        fname = self._sanitize(key)
        data_file = checkpoint_dir / f"{fname}.pkl"

        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file, "rb") as f:
            return pickle.load(f)

    def allocate_datastore(self) -> Path:
        """
        Get the base datastore directory.

        :returns: Path to the datastore directory
        """
        datastore_dir = self.directory / "datastore"
        datastore_dir.mkdir(parents=True, exist_ok=True)
        return datastore_dir

    def persist(self):
        """
        Write all pending metadata changes to disk
        """
        if self.metadata is not None:
            self._save_metadata(self.metadata)

    def log_checkpoint_event(
        self,
        event_type: str,
        agent_name: Optional[str],
        checkpoint_name: Optional[str],
        seq_id: Optional[int],
    ):
        """
        Log checkpoint state change events to a TSV file

        :param event_type: Type of event ('enter', 'exit', 'switch')
        :param checkpoint_name: Name of the checkpoint
        :param seq_id: Sequence ID at the time of the event
        :param additional_info: Any additional information to log
        """
        log_dir = self.directory / "logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / "checkpoint_events.tsv"

        # Write header if file doesn't exist
        if not log_file.exists():
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("session_id\tevent_type\tagent_name\tcheckpoint\tseq_id\n")

        session_id = self.metadata.get("session_counter", "unknown")
        checkpoint_display = (
            checkpoint_name if checkpoint_name is not None else "anonymous"
        )

        # Use tab-separated values
        log_entry = f"{session_id}\t{event_type}\t{agent_name}\t{checkpoint_display}\t{seq_id}\n"

        # Append to log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

    def is_locked(self):
        """
        Check if another FileManager instance has locked this directory
        """
        if not self.lock_file.exists():
            return False

        try:
            with open(self.lock_file, "r") as f:
                lock_pid = int(f.read().strip())
                # Check if the process is still running (Windows-specific)
                try:
                    os.kill(lock_pid, 0)
                    return True  # Process exists
                except OSError:
                    return False  # Process doesn't exist
        except (ValueError, FileNotFoundError):
            return False
