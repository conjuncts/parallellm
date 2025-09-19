from typing import Optional, List, Dict, Any
from parallellm.core.backend import BaseBackend
from parallellm.core.datastore.sqlite import SQLiteDataStore
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import DashboardLogger, HashStatus
from parallellm.provider.guess import guess_schema


class SyncBackend(BaseBackend):
    """
    A synchronous backend that executes operations directly without threading or event loops.
    This backend is simpler and more straightforward for synchronous workflows.
    """

    def __init__(self, fm: FileManager, dash_logger: Optional[DashboardLogger] = None):
        self._fm = fm
        self._ds = SQLiteDataStore(self._fm)
        self._dash_logger = dash_logger

        # Store results directly instead of managing async tasks
        self._pending_results: Dict[str, Any] = {}

    def submit_sync_call(
        self, checkpoint: str, doc_hash: str, seq_id: int, sync_function, *args, **kwargs
    ):
        """Submit a synchronous function call and store the result immediately"""
        try:
            if self._dash_logger is not None:
                self._dash_logger.update_hash(doc_hash, HashStatus.SENT)
            result = sync_function(*args, **kwargs)
            if self._dash_logger is not None:
                self._dash_logger.update_hash(doc_hash, HashStatus.RECEIVED)

            # Process and store the result
            resp_text, resp_id, resp_metadata = guess_schema(result)
            self._ds.store(checkpoint, doc_hash, int(seq_id), resp_text, resp_id)
            self._ds.store_metadata(
                checkpoint, doc_hash, int(seq_id), resp_id, resp_metadata
            )

            # Store in pending results for immediate retrieval
            key = f"{checkpoint}:{doc_hash}:{seq_id}"
            self._pending_results[key] = resp_text

            return resp_text, resp_id, resp_metadata

        except Exception as e:
            # Store the exception for later retrieval
            key = f"{checkpoint}:{doc_hash}:{seq_id}"
            self._pending_results[key] = e
            raise

    def _poll_changes(self, checkpoint: str, until_doc_hash: str, until_seq_id: int):
        """
        Synchronous version - no polling needed since operations complete immediately
        """
        # In synchronous mode, all operations complete immediately
        # So there's nothing to poll for
        pass

    def retrieve(self, checkpoint: str, doc_hash: str, seq_id: int) -> Optional[str]:
        """Synchronous retrieve that checks pending results first, then datastore"""
        # Check if we have a pending result
        key = f"{checkpoint}:{doc_hash}:{seq_id}"
        if key in self._pending_results:
            result = self._pending_results[key]
            if isinstance(result, Exception):
                raise result
            return result

        # Fall back to datastore
        return self._ds.retrieve(checkpoint, doc_hash, seq_id)

    def persist(self):
        """Persist any remaining data"""
        # SQLite commits immediately, but we can clear pending results
        self._pending_results.clear()

    def close(self):
        """Clean up resources"""
        if hasattr(self._ds, "close"):
            self._ds.close()
        self._pending_results.clear()

    def __del__(self):
        """Clean up resources when the SyncBackend is destroyed"""
        try:
            self.close()
        except Exception:
            # Ignore errors during cleanup in destructor
            pass
