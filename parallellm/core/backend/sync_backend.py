from typing import Optional, List, Dict, Any, TYPE_CHECKING
from parallellm.core.backend import BaseBackend
from parallellm.core.datastore.sqlite import SQLiteDatastore
from parallellm.core.response import ReadyLLMResponse
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import DashboardLogger, HashStatus
from parallellm.provider.schemas import guess_schema
from parallellm.types import CallIdentifier

if TYPE_CHECKING:
    from parallellm.provider.base import SyncProvider


class SyncBackend(BaseBackend):
    """
    A synchronous backend that executes operations directly without threading or event loops.
    This backend is simpler and more straightforward for synchronous workflows.
    """

    def __init__(
        self,
        fm: FileManager,
        dash_logger: Optional[DashboardLogger] = None,
        *,
        datastore_cls=None,
    ):
        self._fm = fm

        if datastore_cls is None:
            self._ds = SQLiteDatastore(self._fm)
        else:
            self._ds = datastore_cls(self._fm)
        self._dash_logger = dash_logger

        # Store results directly instead of managing async tasks
        self._pending_results: Dict[str, Any] = {}

    def submit_query(
        self,
        provider: "SyncProvider",
        instructions,
        documents,
        *,
        call_id: CallIdentifier,
        llm,
        _hoist_images=None,
        **kwargs,
    ):
        """
        New control flow: Backend calls provider to get callable, then executes it.
        This inverts control from provider calling backend.
        """
        # Get the callable from the provider
        sync_function = provider.prepare_sync_call(
            instructions,
            documents,
            llm=llm,
            _hoist_images=_hoist_images,
            **kwargs,
        )

        # Execute the call synchronously and store the result
        resp_text, _, _ = self.submit_sync_call(
            call_id, provider=provider, sync_function=sync_function
        )

        # Return a ready response since the operation completed immediately
        return ReadyLLMResponse(call_id=call_id, value=resp_text)

    def submit_sync_call(
        self, call_id: CallIdentifier, sync_function, provider=None, *args, **kwargs
    ):
        """Submit a synchronous function call and store the result immediately"""
        checkpoint = call_id["checkpoint"] or ""
        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]

        try:
            if self._dash_logger is not None:
                self._dash_logger.update_hash(doc_hash, HashStatus.SENT)

            # The below function typically calls the LLM
            result = sync_function(*args, **kwargs)
            if self._dash_logger is not None:
                self._dash_logger.update_hash(doc_hash, HashStatus.RECEIVED)

            # Process and store the result - use provider.parse_response if available
            if provider is not None:
                resp_text, resp_id, resp_metadata = provider.parse_response(result)
            else:
                # Fallback to guess_schema for backward compatibility
                resp_text, resp_id, resp_metadata = guess_schema(
                    result, provider_type=call_id.get("provider_type", None)
                )

            self._ds.store(call_id, resp_text, resp_id, metadata=resp_metadata)

            # Store in pending results for immediate retrieval
            key = f"{checkpoint}:{doc_hash}:{seq_id}"
            self._pending_results[key] = resp_text

            return resp_text, resp_id, resp_metadata

        except Exception as e:
            # Store the exception for later retrieval
            key = f"{checkpoint}:{doc_hash}:{seq_id}"
            self._pending_results[key] = e
            raise

    def _poll_changes(self, call_id: CallIdentifier):
        """
        Synchronous version - no polling needed since operations complete immediately
        """
        # In synchronous mode, all operations complete immediately
        # So there's nothing to poll for
        pass

    def retrieve(self, call_id: CallIdentifier) -> Optional[str]:
        """Synchronous retrieve that checks pending results first, then datastore"""
        # Check if we have a pending result
        key = f"{call_id['checkpoint']}:{call_id['doc_hash']}:{call_id['seq_id']}"
        if key in self._pending_results:
            result = self._pending_results[key]
            if isinstance(result, Exception):
                raise result
            return result

        # Fall back to datastore
        return self._ds.retrieve(call_id)

    def persist(self):
        """Persist any remaining data and datastore"""
        # SQLite commits immediately, but we can clear pending results
        self._ds.persist()

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
