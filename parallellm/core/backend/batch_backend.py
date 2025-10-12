import json
from pathlib import Path
from typing import Literal, Optional, Union
from parallellm.core.backend import BaseBackend
from parallellm.core.backend.sync_backend import SyncBackend
from parallellm.core.datastore.sqlite import SQLiteDatastore
from parallellm.core.identity import LLMIdentity
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import DashboardLogger
from parallellm.provider.base import BatchProvider
from parallellm.types import (
    BatchIdentifier,
    BatchResult,
    BatchStatus,
    CallIdentifier,
    CohortIdentifier,
)


class BatchBackend(BaseBackend):
    """
    The batch backend is a bit different: it defers sending out requests
    (if the value is needed, NotAvailable() is emitted),
    then upon exit, a batch is created.
    """

    def __init__(
        self,
        fm: FileManager,
        dash_logger: Optional[DashboardLogger] = None,
        *,
        datastore_cls=None,
        session_id,
    ):
        self._fm = fm
        if datastore_cls is None:
            self._ds = SQLiteDatastore(fm)
        else:
            self._ds = datastore_cls(fm)
        self._dash_logger = dash_logger

        self._batch_buffer: list[tuple[CallIdentifier, dict]] = []
        self._provider = None
        self.session_id = session_id

        self.batch_group_counter = 0

    def _poll_changes(self, call_id: CallIdentifier):
        """
        For batch, not our responsibility
        """

    def retrieve(self, call_id: CallIdentifier) -> Optional[str]:
        """Synchronous retrieve that checks pending results first, then datastore"""
        # Fall back to datastore
        return self._ds.retrieve(call_id)

    def close(self):
        """Clean up resources"""
        if hasattr(self._ds, "close"):
            self._ds.close()

    def __del__(self):
        """Clean up resources when the SyncBackend is destroyed"""
        try:
            self.close()
        except Exception:
            # Ignore errors during cleanup in destructor
            pass

    def bookkeep_call(
        self,
        call_id: CallIdentifier,
        llm: LLMIdentity,
        stuff: dict,
        *,
        auto_assign_id: Optional[str] = None,
    ):
        """
        Saves a call so that it may be executed as a batch later

        :param call_id: The call identifier
        :param stuff: Arbitrary data needed to make the call later. Should be a dict.

        :param auto_assign_id: If not None, then `stuff` will be auto-assigned a custom_id, which
        will be f"{checkpoint_name}-{session_id}-{seq_id}", which is guaranteed to be unique unless a major
        error has occurred. It will be placed in stuff[auto_assign_id].
        """
        if auto_assign_id is not None:
            agent_name = call_id["agent_name"]
            checkpoint = call_id["checkpoint"] or ""
            seq_id = call_id["seq_id"]
            custom_id = f"{agent_name}-{checkpoint}-{self.session_id}-{seq_id}"
            stuff[auto_assign_id] = custom_id

        self._batch_buffer.append((call_id, llm.model_name, stuff))

    def _register_provider(self, provider: BatchProvider):
        """Register a provider"""
        if self._provider is not None:
            raise ValueError(
                f"BatchBackend already has provider of type {self._provider.provider_type}"
                f"; cannot register additional {provider.provider_type} provider."
            )
        self._provider = provider

    def execute_batch(
        self, *, max_batch_size=1000, partition_by_model_name=True
    ) -> CohortIdentifier:
        """Execute the batch of calls"""

        # 1. Split based on model_name
        # Many batch APIs (like OpenAI's) require the same model across the entire batch
        batches = []
        if partition_by_model_name:
            mn2b = {}
            for call_id, model_name, stuff in self._batch_buffer:
                if model_name not in mn2b:
                    mn2b[model_name] = []
                mn2b[model_name].append((call_id, stuff))
            batches = list(mn2b.values())
        else:
            batches = [(call_id, stuff) for call_id, _, stuff in self._batch_buffer]

        # 2. Split into groups of max_batch_size
        _bdr = []
        for batch in batches:
            if len(batch) <= max_batch_size:
                _bdr.append(batch)
            else:
                for i in range(0, len(batch), max_batch_size):
                    _bdr.append(batch[i : i + max_batch_size])
        batches = _bdr

        # 3. Submit each batch
        batch_ids = []
        for batch in batches:
            # Here, you would typically call the provider's batch method
            call_ids, _, stuff = zip(*batch)
            batch_id = self._provider.submit_batch_to_provider(call_ids, list(stuff))
            self._ds.store_pending_batch(batch_id)
            batch_ids.append(batch_id)

        cohort_id = CohortIdentifier(batch_ids=batch_ids, session_id=self.session_id)
        # Clear the batch buffer after execution
        self._batch_buffer.clear()
        return cohort_id

    def persist(self):
        """Persist any remaining data and datastore"""
        self._ds.persist()

        # Now is the time to execute the batch
        if self._provider is not None and self._batch_buffer:
            self.execute_batch()

    def persist_batch_to_jsonl(self, stuff: list[dict], *, preferred_name=None) -> Path:
        """
        Helper function to persist batch inputs to disk.
        Coordinates with the FileManager to get a suitable location.
        `stuff` is JSON-serialized.

        Because many SDKs want the entire batch to be sent over as a file.
        """
        if preferred_name is None:
            preferred_name = f"batch_{self.session_id}_{self.batch_group_counter}.jsonl"
            self.batch_group_counter += 1

        # Remnants of same-session_id files should not be possible, since
        # session_id should be unique per run.
        path = self._fm.allocate_batch_in() / preferred_name

        with open(path, "w", encoding="utf-8") as f:
            for item in stuff:
                f.write(json.dumps(item) + "\n")

        return path

    def persist_to_zip(self, stuff: Union[str, list[dict]], fpath: Path) -> None:
        """
        Helper function to persist anything to a zip file.
        """
        import zipfile

        with zipfile.ZipFile(fpath, "w") as zf:
            intended_name = fpath.stem
            if isinstance(stuff, str):
                zf.writestr(intended_name, stuff)
            else:
                # write a jsonl
                with zf.open(intended_name + ".jsonl", "w") as f:
                    for item in stuff:
                        line = json.dumps(item) + "\n"
                        f.write(line.encode("utf-8"))

    # TODO: Also need to store the batch_uuid
    # In that datastore, we need to store
    # (checkpoint, doc_hash, session_id, seq_id, batch_uuid)
    # so probably the easiest way is to add a possibly NULL column (batch_uuid)
    # to the existing SQL database.

    # But that requires response (TEXT) to be not null.
    # Hence, we will need to create a new table "unready_batch_responses"
    # That contains (agent_name, checkpoint?, seq_id, session_id, doc_hash, provider_type, batch_uuid)

    def download_batch_from_provider(
        self, batch_uuid: str, *, save_to_disk: Literal[None, "jsonl.zip"] = "jsonl.zip"
    ) -> BatchResult:
        """
        Given a batch UUID, download the results.

        This is a helper function that calls the provider's download_batch_results method.

        This has the inverted (correct) control flow - the rest of the codebase
        will eventually be rectified to have this control flow.

        :param batch_uuid: The UUID of the batch to download.
        :param save_to_disk: If "jsonl.zip", saves the results to a zip file containing a jsonl.
            If None, does not save to disk.
        """
        if self._provider is None:
            raise ValueError("No provider registered with BatchBackend")

        batch_result = self._provider.download_batch_from_provider(batch_uuid)

        if save_to_disk == "jsonl.zip":
            # Save to disk
            if batch_result.status == "ready" or batch_result.status == "error":
                ending = ".jsonl.zip"
                if batch_result.status == "error":
                    ending = "_err.jsonl.zip"
                fpath = self._fm.allocate_batch_out() / f"{batch_uuid}{ending}"
                self.persist_to_zip(
                    batch_result.raw_output, fpath=fpath
                )

        if batch_result.status == "ready":
            self._ds.store_ready_batch(batch_result)
        return batch_result

    def try_download_all_batches(
        self,
    ):
        """
        Try to download all batches
        """
        pending_batches = self._ds.get_all_pending_batch_uuids()
        for batch_uuid in pending_batches:
            try:
                batch_result = self.download_batch_from_provider(
                    batch_uuid, save_to_disk="jsonl.zip"
                )
                if batch_result.status == "ready":
                    print(f"Batch {batch_uuid} completed and stored.")
                elif batch_result.status == "error":
                    print(f"Batch {batch_uuid} completed with errors and stored.")
                else:
                    print(f"Batch {batch_uuid} is still pending.")
            except Exception as e:
                print(f"Error downloading batch {batch_uuid}: {e}")

class DebugBatchBackend(BatchBackend):
    """
    Mimics the BatchBackend behavior.
    Except having to wait for the batch to complete, upon persist(),
    values are actually generated synchronously, so that the batch always "finishes" immediately.
    Useful for testing.
    """
