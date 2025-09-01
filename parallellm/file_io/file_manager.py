import os
import json
import pickle
import time
import atexit
from pathlib import Path

class FileManager:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.metadata_file = self.directory / "metadata.json"
        self.lock_file = self.directory / ".filemanager.lock"
        
        # In-memory metadata that will be written on persist()
        self._pending_metadata = None
        
        # Create directory if it doesn't exist
        self.directory.mkdir(parents=True, exist_ok=True)
        
        # Create lock file
        self._create_lock()
        
        # Register cleanup on exit
        atexit.register(self._cleanup)
        
        # Load existing metadata
        self._pending_metadata = self._load_metadata()
        if self._pending_metadata is None:
            self._pending_metadata = {"current_stage": None}

    def _create_lock(self):
        """Create lock file with current process ID"""
        with open(self.lock_file, 'w') as f:
            f.write(str(os.getpid()))

    def _cleanup(self):
        """Cleanup method called on object destruction"""
        if self.lock_file.exists():
            try:
                self.lock_file.unlink()
            except (FileNotFoundError, PermissionError):
                pass  # Lock was already removed or can't be removed    def __del__(self):
        """Destructor to ensure cleanup"""
        self._cleanup()

    def _load_metadata(self):
        """Load metadata from JSON file"""
        try:
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def _save_metadata(self, metadata):
        """Save metadata to JSON file"""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def current_stage(self):
        """
        Get the current stage from in-memory metadata
        """
        return self._pending_metadata.get("current_stage")

    def set_current_stage(self, stage):
        """
        Set the current stage in memory (will be written on persist())
        """
        self._pending_metadata["current_stage"] = stage

    def save_data(self, stage, key, value, overwrite=False):
        """
        Internally persist data across stages
        """
        # Create stage directory
        stage_dir = self.directory / str(stage)
        stage_dir.mkdir(exist_ok=True)
        
        # Save data using pickle for complex objects
        data_file = stage_dir / f"{key}.pkl"

        if data_file.exists() and not overwrite:
            return

        with open(data_file, 'wb') as f:
            pickle.dump(value, f)

    def load_data(self, stage, key):
        """
        Internally load data across stages
        """
        # Construct expected file path
        stage_dir = self.directory / str(stage)
        data_file = stage_dir / f"{key}.pkl"
        
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        with open(data_file, 'rb') as f:
            return pickle.load(f)

    def persist(self):
        """
        Write all pending metadata changes to disk
        """
        if self._pending_metadata is not None:
            self._save_metadata(self._pending_metadata)

    def is_locked(self):
        """
        Check if another FileManager instance has locked this directory
        """
        if not self.lock_file.exists():
            return False
        
        try:
            with open(self.lock_file, 'r') as f:
                lock_pid = int(f.read().strip())
                # Check if the process is still running (Windows-specific)
                try:
                    os.kill(lock_pid, 0)
                    return True  # Process exists
                except OSError:
                    return False  # Process doesn't exist
        except (ValueError, FileNotFoundError):
            return False
