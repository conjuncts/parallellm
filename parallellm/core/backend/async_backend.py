import asyncio
import logging
import types
import threading
import atexit
from typing import Optional
from parallellm.core.backend import BaseBackend
from parallellm.core.datastore.sqlite import SQLiteDataStore
from parallellm.file_io.file_manager import FileManager
from parallellm.provider.guess import guess_schema


class AsyncBackend(BaseBackend):
    """
    A backend is a data store, but also a way to poll.
    This backend owns its own event loop running in a separate thread.

    The DataStore
    """

    def __init__(self, fm: FileManager):
        self._fm = fm

        self.tasks: list[asyncio.Task] = []
        self._loop: asyncio.BaseEventLoop = None
        self._loop_thread = None
        self._shutdown_event = threading.Event()
        self._loop_ready_event = threading.Event()

        # Start the event loop in a separate thread
        # self._ds = SQLiteDataStore(self._fm)
        self._async_ds: Optional[SQLiteDataStore] = None
        self._start_event_loop()

        # Register cleanup to run on program exit
        atexit.register(self.shutdown)

    def _start_event_loop(self):
        """Start the event loop in a separate thread"""

        def run_event_loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # Initialize the datastore now that the loop is running
            self._async_ds = SQLiteDataStore(self._fm)

            # Signal that the loop is ready
            self._loop_ready_event.set()

            try:
                # Keep the loop running until shutdown
                self._loop.run_until_complete(self._wait_for_shutdown())
            finally:
                # Clean up any remaining tasks
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                self._loop.close()

        self._loop_thread = threading.Thread(target=run_event_loop, daemon=True)
        self._loop_thread.start()

        # Wait for the loop to be ready (blocks until event is set)
        self._loop_ready_event.wait()

    async def _wait_for_shutdown(self):
        """Keep the event loop running until shutdown is requested"""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(0.1)

        # Clean up the datastore from the same thread that created it
        if hasattr(self, "_async_ds"):
            self._async_ds.close()
            del self._async_ds

    def _run_coroutine(self, coro):
        """Run a coroutine in the backend's event loop and return the result"""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("AsyncBackend event loop is not running")

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        res = future.result()
        return res

    def shutdown(self):
        """Shutdown the event loop and cleanup"""
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)

    async def _cleanup_datastore(self):
        """Clean up the datastore from the async thread"""
        if hasattr(self, "_async_ds"):
            self._async_ds.close()
            del self._async_ds

    def cleanup_datastore_sync(self):
        """Synchronously trigger datastore cleanup in the async thread"""
        if self._loop is not None and not self._loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(
                self._cleanup_datastore(), self._loop
            )
            try:
                future.result(timeout=5.0)
            except Exception as e:
                print(f"Warning: Failed to cleanup datastore: {e}")

    def submit_coro(
        self, stage: str, doc_hash: str, seq_id: int, coro: types.CoroutineType
    ):
        """Submit a coroutine to be executed in the backend's event loop"""
        if self._loop is None or self._loop.is_closed():
            raise RuntimeError("AsyncBackend event loop is not running")

        # Create the task in the backend's event loop
        future = asyncio.run_coroutine_threadsafe(
            self._create_and_store_task(stage, doc_hash, seq_id, coro), self._loop
        )
        # Don't wait for the result, just submit it
        return future

    async def _create_and_store_task(
        self, stage: str, doc_hash: str, seq_id: int, coro: types.CoroutineType
    ):
        """Helper to create and store a task in the event loop"""

        # Wrap the coro to include metadata
        async def wrapped_coro():
            result = await coro
            return result, {"stage": stage, "doc_hash": doc_hash, "seq_id": seq_id}

        task = asyncio.create_task(wrapped_coro())
        self.tasks.append(task)
        return task

    async def _poll_changes(
        self, until_stage: str, until_doc_hash: str, until_seq_id: int = None
    ):
        """
        A chance to poll for changes and update the data store
        """
        # collect as results come in
        for coro in asyncio.as_completed(self.tasks):
            result, metadata = await coro

            stage = metadata["stage"]
            doc_hash = metadata["doc_hash"]
            seq_id = metadata["seq_id"]

            resp_text, resp_id, resp_metadata = guess_schema(result)
            self._async_ds.store(stage, doc_hash, int(seq_id), resp_text, resp_id)
            self._async_ds.store_metadata(
                stage, doc_hash, int(seq_id), resp_id, resp_metadata
            )
            # done_tasks.append(coro)

            # Stop if we reached the target
            if (
                until_stage == stage
                and until_doc_hash == doc_hash
                and (until_seq_id is None or until_seq_id == int(seq_id))
            ):
                break

        # for done_task in done_tasks:
        # self.tasks.remove(done_task)
        self.tasks = [t for t in self.tasks if not t.done()]
        pass

    async def aretrieve(
        self, stage: str, doc_hash: str, seq_id: int = None
    ) -> Optional[str]:
        await self._poll_changes(stage, doc_hash)
        return self._async_ds.retrieve(stage, doc_hash, seq_id)

    def persist(self, timeout=30.0):
        """Synchronous persist that uses the backend's event loop"""
        # SQLite commits immediately

        # but we DO want to wait for all pending tasks to complete
        if self._loop is not None and not self._loop.is_closed():
            future = asyncio.run_coroutine_threadsafe(
                self._poll_changes(None, None), self._loop
            )
            try:
                future.result(timeout=timeout)
            except Exception as e:
                print(f"Warning: Failed to wait for pending tasks: {e}")

    def retrieve(self, stage: str, doc_hash: str, seq_id: int = None) -> Optional[str]:
        """Synchronous retrieve that uses the backend's event loop"""
        return self._run_coroutine(self.aretrieve(stage, doc_hash, seq_id))

    def __del__(self):
        """Clean up resources when the AsyncBackend is destroyed"""
        try:
            # Try to clean up the datastore from the async thread first
            self.cleanup_datastore_sync()
            # Then shutdown the event loop
            self.shutdown()
        except Exception:
            # Ignore errors during cleanup in destructor
            pass
