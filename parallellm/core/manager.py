import logging
from typing import List, Optional, Union
from colorama import Fore, Style, init
from parallellm.core.backend import BaseBackend
from parallellm.core.cast.fix_docs import cast_documents
from parallellm.core.exception import GotoCheckpoint, NotAvailable, WrongCheckpoint
from parallellm.core.hash import compute_hash
from parallellm.core.response import (
    LLMIdentity,
    LLMDocument,
    LLMResponse,
    PendingLLMResponse,
    ReadyLLMResponse,
)
from parallellm.provider.base import BaseProvider
from parallellm.file_io.file_manager import FileManager
from parallellm.logging.dash_logger import DashboardLogger, HashStatus
from parallellm.types import CallIdentifier


# New context manager class for BatchManager
class ParalleLLMContext:
    """Context manager for BatchManager lifecycle (default context)"""

    def __init__(self, batch_manager: "BatchManager"):
        self._bm = batch_manager

    def __enter__(self):
        # Any setup logic can go here if needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Delegate to BatchManager's error handling logic

        if self._bm:
            # Exit their checkpoint if inside one
            self._bm._exit_checkpoint()

        if exc_type in (NotAvailable, WrongCheckpoint, GotoCheckpoint):
            return True
        return False

    def print(self, *args, **kwargs):
        """
        Print to console above the dashboard output.
        This ensures proper display ordering when the dashboard is active.
        """
        # Use the dashboard logger's coordinated print method
        print(*args, **kwargs)


class StatusDashboard(ParalleLLMContext):
    """Context manager for the hash status dashboard"""

    def __init__(self, batch_manager: "BatchManager", log_k: int):
        super().__init__(batch_manager)
        self._was_displaying = False
        self._bm._dash_logger.k = log_k

    @property
    def _dash_logger(self):
        return self._bm._dash_logger

    def __enter__(self):
        # Store current display state and enable display
        self._was_displaying = self._dash_logger.display
        self._dash_logger.set_display(True)
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Print final status of dashboard
        self._dash_logger._update_console()

        # Finalize the line and restore original display state
        self._dash_logger.finalize_line()
        self._dash_logger.set_display(self._was_displaying)
        print()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def print(self, *args, **kwargs):
        """
        Print to console above the dashboard output.
        This ensures proper display ordering when the dashboard is active.
        """
        # Use the dashboard logger's coordinated print method
        self._dash_logger.coordinated_print(*args, **kwargs)


class BatchManager:
    def __init__(
        self,
        file_manager: FileManager,
        backend: BaseBackend,
        provider: BaseProvider,
        *,
        logger,
        dash_logger,
    ):
        """
        channel: str, optional
            Parallellm is typically used as a singleton.
            If you want to have multiple instances,
            specify a channel name.
        dash_logger_k: int, optional
            Number of hashes to display in the hash logger (default 10)
        """
        self._backend = backend
        self._fm = file_manager
        self._provider = provider
        self._logger = logger
        self._anonymous_counter = 0  # Anonymous mode counter, always starts from 0
        self._checkpoint_counter = None  # Initialized when entering checkpoint

        # Initialize the hash logger with display disabled by default
        self._dash_logger: DashboardLogger = dash_logger

        self.active_checkpoint: Optional[str] = None
        """
        Active checkpoint: if we are inside a checkpoint context.
        Either a checkpoint or None (if anonymous).
        """

    def default(self):
        """
        Begins a 'default' checkpoint.
        Usage: `with pllm.default(): ...`
        """
        return ParalleLLMContext(self)

    def checkpoint(self):
        """
        Begins a context where checkpoints can be used.
        Usage: `with pllm.managed(): ...`
        """
        return ParalleLLMContext(self)

    def dashboard(self, log_k: int = 10):
        """
        Create a context manager for the status dashboard.

        :param log_k: Number of hashes to display in the dashboard (default 10)
        :returns: StatusDashboard context manager with print() method for clean console output
        """
        # TODO implement log_k
        return StatusDashboard(self, log_k=log_k)

    @property
    def latest_checkpoint(self):
        return self.metadata["latest_checkpoint"]

    @property
    def metadata(self):
        return self._fm.metadata

    @metadata.setter
    def metadata(self, value):
        self._fm.metadata = value

    def when_checkpoint(self, checkpoint_name):
        # Always allow enter if no checkpoints yet
        if self.latest_checkpoint is None:
            self.metadata["latest_checkpoint"] = checkpoint_name

        # But generally, only enter if it is the latest
        elif checkpoint_name != self.latest_checkpoint:
            raise WrongCheckpoint()

        # Set it as active and initialize local checkpoint counter
        self.active_checkpoint = checkpoint_name
        # Initialize local counter from persisted metadata (or 0 if first time)
        self._checkpoint_counter = self.metadata.get("checkpoint_counter", 0)

        # Log checkpoint entry to file
        # self._fm.log_checkpoint_event(
        #     "enter", checkpoint_name, self._checkpoint_counter
        # )

        self._logger.info(
            f"Entered checkpoint {Fore.CYAN}{checkpoint_name}{Style.RESET_ALL}"
        )

    def goto_checkpoint(self, checkpoint_name):
        """
        Nothing after this statement will be executed.
        """
        # Log the current seq_id before switching
        current_seq_id = None
        if self.active_checkpoint is not None:
            current_seq_id = self._checkpoint_counter
        else:
            current_seq_id = self._anonymous_counter

        # Revise metadata
        self.metadata["latest_checkpoint"] = checkpoint_name
        self.metadata["checkpoint_counter"] = self._checkpoint_counter

        # Logging
        self._fm.log_checkpoint_event("switch", checkpoint_name, current_seq_id)
        self._logger.info(
            f"Switched to checkpoint {Fore.CYAN}{checkpoint_name}{Style.RESET_ALL}"
        )
        raise GotoCheckpoint(checkpoint_name)

    def _exit_checkpoint(self):
        """
        Exit checkpoint context (called by context manager)
        """
        if self.active_checkpoint is not None:
            # Log the final seq_id before exiting
            final_seq_id = (
                self._checkpoint_counter if self._checkpoint_counter is not None else 0
            )

            # Log checkpoint exit to file
            # self._fm.log_checkpoint_event("exit", self.active_checkpoint, final_seq_id)

            # Do NOT persist local checkpoint counter
            # (local checkpoint counter should only persist upon a goto)
            self.active_checkpoint = None
            self._checkpoint_counter = None

    def save_userdata(self, key, value):
        """
        The intended way to let data persist across checkpoints
        """
        return self._fm.save_userdata(key, value)

    def load_userdata(self, key):
        """
        The intended way to let data persist across checkpoints
        """
        data = self._fm.load_userdata(key)

        # If the loaded data is an LLMResponse, inject the backend
        if isinstance(data, PendingLLMResponse):
            data._backend = self._backend
        elif isinstance(data, ReadyLLMResponse):
            data.value = self._backend.retrieve(data.call_id)

        return data

    def persist(self):
        """
        Ensure that everything is properly saved
        """
        self._backend.persist()

        self._fm.persist()

    def set_hash_display(self, display: bool):
        """
        Enable or disable hash logger console display

        Args:
            display: Whether to show hash logger output
        """
        self._dash_logger.set_display(display)

    def update_hash_status(self, hash_value: str, status: HashStatus):
        """
        Update the status of a hash in the logger

        Args:
            hash_value: The hash value to update
            status: New status - one of 'C' (cached), '↗' (sent), '↘' (received), '✓' (stored)
        """
        if self._dash_logger is not None:
            self._dash_logger.update_hash(hash_value, status)

    def finalize_hash_display(self):
        """
        Finalize the hash display line and move to next line.
        Useful when you want to ensure subsequent print() calls appear on new lines.
        """
        self._dash_logger.finalize_line()

    def ask_llm(
        self,
        documents: Union[LLMDocument, List[LLMDocument]],
        *additional_documents: LLMDocument,
        instructions: Optional[str] = None,
        llm: Union[LLMIdentity, str, None] = None,
        _hoist_images=None,
        **kwargs,
    ) -> LLMResponse:
        """
        Ask the LLM a question

        :param documents: Documents to use, such as the prompt.
            Can be strings or images.
        :param instructions: The system prompt to use.
        :param llm: The identity of the LLM to use.
            Can be helpful multi-agent or multi-model scenarios.
        :param _hoist_images: Gemini recommends that images be hoisted to the front of the message.
            Set to True/False to explicitly enforce/disable.
        :returns: A LLMResponse. The value is **lazy loaded**: for best efficiency,
            it should not be resolved until you actually need it.
        """

        if isinstance(llm, str):
            llm = LLMIdentity(llm)

        # Use dual counter system based on checkpoint mode
        if self.active_checkpoint is not None:
            # In checkpoint mode: use and increment local checkpoint counter
            seq_id = self._checkpoint_counter
            self._checkpoint_counter += 1
        else:
            # In anonymous mode: use and increment anonymous counter
            seq_id = self._anonymous_counter
            self._anonymous_counter += 1

        documents = cast_documents(documents, list(additional_documents))

        # Cache using datastore
        hashed = compute_hash(instructions, documents)

        call_id: CallIdentifier = {
            "checkpoint": self.active_checkpoint,
            "doc_hash": hashed,
            "seq_id": seq_id,
            "session_id": self._fm.metadata["session_counter"],
        }
        cached = self._backend.retrieve(call_id)
        if cached is not None:
            self.update_hash_status(hashed, HashStatus.CACHED)
            return ReadyLLMResponse(
                call_id=call_id,
                value=cached,
            )

        # Not cached, submit to provider
        self.update_hash_status(hashed, HashStatus.SENT)
        return self._provider.submit_query_to_provider(
            instructions,
            documents,
            call_id=call_id,
            llm=llm,
            _hoist_images=_hoist_images,
            **kwargs,
        )

    def save_to_file(
        self,
        responses: List[LLMResponse],
        fname: str,
        *,
        format: str = "batch-openai",
    ):
        """
        Save a list of responses to a file.
        Creates an openai batch file.
        """
        raise NotImplementedError
