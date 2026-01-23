import time
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from parallellm.core.backend import BaseBackend
from parallellm.core.throttler import Throttler
from parallellm.core.datastore.sqlite import SQLiteDatastore
from parallellm.core.response import ReadyLLMResponse
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import (
    DashboardLogger,
    HashStatus,
    PrimitiveDashboardLogger,
)
from parallellm.types import (
    CallIdentifier,
    CommonQueryParameters,
    ParsedResponse,
    CommonQueryParameters,
    to_serial_id,
)

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
        dashlog: DashboardLogger = PrimitiveDashboardLogger(),
        *,
        datastore_cls=None,
        rewrite_cache: bool = False,
        throttler=None,
    ):
        """
        Initialize the SyncBackend.

        :param fm: FileManager for data persistence
        :param dashlog: Optional dashboard logger for monitoring
        :param datastore_cls: Custom datastore class (defaults to SQLiteDatastore)
        :param rewrite_cache: Whether to overwrite existing cache entries
        :param throttler: Throttler instance for rate limiting (default: None)
        """
        self._fm = fm

        if datastore_cls is None:
            self._ds = SQLiteDatastore(self._fm)
        else:
            self._ds = datastore_cls(self._fm)
        self.dashlog = dashlog
        self._rewrite_cache = rewrite_cache

        if throttler is not None:
            self._throttler = throttler
        else:
            # No rate limiting by default
            self._throttler = Throttler(
                max_requests_per_window=None,
                window_seconds=None,
            )

        # Store results directly instead of managing async tasks
        self._pending_results: Dict[str, Any] = {}

    def _get_datastore(self):
        return self._ds

    def _apply_throttling(self) -> None:
        """Apply throttling by waiting if necessary"""
        delay = self._throttler.calculate_delay()
        if delay > 0:
            time.sleep(delay)
            # After waiting, record the actual submission time
            if self._throttler.is_enabled():
                self._throttler.record_request()

    def submit_query(
        self,
        provider: "SyncProvider",
        params: CommonQueryParameters,
        *,
        call_id: CallIdentifier,
        **kwargs,
    ):
        """
        New control flow: Backend calls provider to get callable, then executes it.
        This inverts control from provider calling backend.
        """

        """Submit a synchronous function call and store the result immediately"""
        doc_hash = call_id["doc_hash"]
        seq_id = call_id["seq_id"]

        try:
            # Apply throttling before making the request
            self._apply_throttling()

            self.dashlog.update_hash(doc_hash, HashStatus.SENT)

            # The below function typically calls the LLM
            result = provider.prepare_sync_call(
                params,
                **kwargs,
            )
            self.dashlog.update_hash(doc_hash, HashStatus.RECEIVED)

            parsed = provider.parse_response(result)

            self._ds.store(call_id, parsed, upsert=self._rewrite_cache)

            # Store in pending results for immediate retrieval
            # key = to_serial_id(call_id, add_sess=False)
            # self._pending_results[key] = parsed.text

            return ReadyLLMResponse(call_id=call_id, pr=parsed)

        except Exception as e:
            # Store the exception for later retrieval
            # key = to_serial_id(call_id, add_sess=False)
            # self._pending_results[key] = e
            raise

    def _poll_changes(self, call_id: CallIdentifier):
        """
        Synchronous version - no polling needed since operations complete immediately
        """
        # In synchronous mode, all operations complete immediately
        # So there's nothing to poll for
        pass

    def retrieve(
        self, call_id: CallIdentifier, metadata=False
    ) -> Optional[ParsedResponse]:
        """
        Synchronous retrieve that checks pending results first, then datastore.

        The required fields from call_id are:
        - agent_name, doc_hash, seq_id
        """
        # Check if we have a pending result
        # Within one session, doc_hash should not be necessary
        # key = to_serial_id(call_id, add_sess=False)
        # if key in self._pending_results:
        #     result = self._pending_results[key]
        #     if isinstance(result, Exception):
        #         raise result
        #     # If we have a pending result (which is just text), wrap it in ParsedResponse
        #     return ParsedResponse(text=result, response_id=None, metadata=None)

        # Fall back to datastore
        return self._ds.retrieve(call_id, metadata=metadata)

    def persist(self):
        """Persist any remaining data and datastore"""
        # Let datastore cleanup
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
